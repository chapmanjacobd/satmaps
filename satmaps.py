#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import sys
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import mgrs
import numpy as np
from osgeo import gdal

import tiler

# Setup GDAL exceptions
gdal.UseExceptions()

AUTO_SCALE_MAX_PERCENTILE = 99.6
# --- State Management ---


def save_state(
    state_file: str,
    unique_id: str,
    completed_subtiles: Set[str],
    processed_tifs: List[str],
    args: argparse.Namespace,
) -> None:
    """Save the current processing state to a JSON file."""
    state = {
        "unique_id": unique_id,
        "completed_subtiles": list(completed_subtiles),
        "processed_tifs": processed_tifs,
        "args": vars(args),
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def load_state(state_file: str) -> Optional[Dict]:
    """Load the processing state from the JSON file if it exists."""
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load state file: {e}")
    return None


def delete_state(state_file: str) -> None:
    """Delete the state file upon successful completion."""
    if os.path.exists(state_file):
        os.remove(state_file)


# --- Discovery Layer (S3/CDSE Utils) ---


def setup_gdal_cdse() -> None:
    """Configure GDAL for CDSE S3 access."""
    gdal.SetConfigOption("AWS_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
    gdal.SetConfigOption("AWS_HTTPS", "YES")
    gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")
    gdal.SetConfigOption("AWS_PROFILE", "cdse")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")


# Global cache for S3 folder listings to avoid per-tile discovery overhead
S3_FOLDER_CACHE: Dict[str, set] = {}


def list_mosaic_folders_for_tile(
    mgrs_tile: str, date_paths: List[str], cache_dir: str
) -> List[Tuple[str, str]]:
    """Find all S3 folders for a specific MGRS sub-tile across multiple dates using the pre-populated cache."""
    mgrs_id, x, y = mgrs_tile.split("_")
    found = []

    for date_path in date_paths:
        quarter = "Q3"
        if "07/01" in date_path:
            quarter = "Q3"
        elif "10/01" in date_path:
            quarter = "Q4"
        elif "04/01" in date_path:
            quarter = "Q2"
        elif "01/01" in date_path:
            quarter = "Q1"

        year = date_path.split("/")[0]
        folder = f"Sentinel-2_mosaic_{year}_{quarter}_{mgrs_id}_{x}_{y}"

        # 1. Check local cache first
        local_b04 = os.path.join(
            cache_dir, date_path.replace("/", "-"), f"{mgrs_id}_{x}_{y}_B04.tif"
        )
        if os.path.exists(local_b04):
            found.append((folder, date_path))
            continue

        # 2. Check pre-populated S3 cache
        if date_path in S3_FOLDER_CACHE:
            if folder in S3_FOLDER_CACHE[date_path]:
                found.append((folder, date_path))
        else:
            # Fallback for non-global runs: check S3 directly for this specific folder
            s3_path = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}"
            try:
                if gdal.ReadDir(s3_path):
                    found.append((folder, date_path))
            except:
                pass

    return found


def get_tile_paths(
    folder_name: str,
    date_path: str,
    cache_dir: Optional[str] = None,
    download: bool = False,
) -> Dict[str, str]:
    """Construct local or S3 paths for RGB bands. Only downloads if 'download' is True."""
    cache_prefix = "_".join(folder_name.split("_")[4:])
    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder_name}"
    bands = {"B04": "red", "B03": "green", "B02": "blue"}
    paths = {}

    date_cache_dir = (
        os.path.join(cache_dir, date_path.replace("/", "-")) if cache_dir else None
    )
    for band_id, color_name in bands.items():
        local_path = (
            os.path.join(date_cache_dir, f"{cache_prefix}_{band_id}.tif")
            if date_cache_dir
            else None
        )

        # Use local if it exists
        if local_path and os.path.exists(local_path):
            paths[color_name] = local_path
            continue

        # Download if requested and local_path is valid
        if download and local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_path = f"{base_s3}/{band_id}.tif"
            print(f"Downloading {s3_path} to {local_path}...")
            src_ds = gdal.Open(s3_path)
            if src_ds is None:
                raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName("GTiff").CreateCopy(
                local_path, src_ds, callback=gdal.TermProgress_nocb
            )
            paths[color_name] = local_path
            continue

        # Fallback to streaming
        paths[color_name] = f"{base_s3}/{band_id}.tif"

    return paths


# --- Processing Layer (NumPy Manipulation) ---


def check_land_gebco(mgrs_tile: str, gebco_src: str) -> bool:
    """Check if an MGRS tile contains land according to GEBCO (sampling)."""
    m = mgrs.MGRS()
    try:
        clat, clon = m.toLatLon(mgrs_tile)
    except:
        return True

    # 100km is roughly 1 degree. We'll sample a grid around the center.
    ds = gdal.Open(gebco_src)
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    # Sample a 1.2 degree box (13x13 points)
    has_land = False
    for dlat in np.linspace(-0.6, 0.6, 13):
        for dlon in np.linspace(-0.6, 0.6, 13):
            px, py = gdal.ApplyGeoTransform(inv_gt, clon + dlon, clat + dlat)
            px, py = int(px), int(py)
            if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                val = ds.GetRasterBand(1).ReadAsArray(px, py, 1, 1)[0, 0]
                if val > 0.001:
                    has_land = True
                    break
        if has_land:
            break
    return has_land


def process_single_tile(
    mgrs_subtile: str,
    date_paths: List[str],
    args: argparse.Namespace,
    unique_id: str,
    gebco_src: Optional[str] = None,
) -> Optional[str]:
    """Process a single MGRS sub-tile: fetch dates, average, tone-map, and warp to Web Mercator."""
    folders = list_mosaic_folders_for_tile(mgrs_subtile, date_paths, args.cache)
    if not folders:
        return None

    if args.download:
        for folder_name, date_path in folders:
            get_tile_paths(folder_name, date_path, args.cache, download=True)
        return None

    # 1. Get Metadata and GEBCO Mask
    first_folder, first_date = folders[0]
    paths = get_tile_paths(first_folder, first_date, args.cache, download=False)
    ds_init = gdal.Open(paths["red"])
    projection = ds_init.GetProjection()
    geotransform = ds_init.GetGeoTransform()
    width, height = ds_init.RasterXSize, ds_init.RasterYSize
    ds_init = None

    ocean_mask = None
    alpha_weight = None
    # Transition constants
    LAND_STAY = -42.0  # 100% Sentinel-2 above this depth
    OCEAN_FADE = -50.0  # 0% Sentinel-2 below this depth

    if gebco_src:
        try:
            gebco_ds = gdal.Open(gebco_src)
            # Create in-memory dataset matching the Sentinel-2 grid
            mem_driver = gdal.GetDriverByName("MEM")
            gebco_mem_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Float32)
            gebco_mem_ds.SetProjection(projection)
            gebco_mem_ds.SetGeoTransform(geotransform)

            # Warp GEBCO into the Sentinel-2 grid
            gdal.Warp(
                gebco_mem_ds, gebco_ds, options=gdal.WarpOptions(resampleAlg="bilinear")
            )
            gebco_data = gebco_mem_ds.GetRasterBand(1).ReadAsArray()

            # Create a smooth alpha weight based on depth
            alpha_weight = np.clip(
                (gebco_data - OCEAN_FADE) / (LAND_STAY - OCEAN_FADE), 0.0, 1.0
            )

            # Optimization: If the whole tile is deep ocean (below OCEAN_FADE), skip it
            if np.all(gebco_data <= OCEAN_FADE):
                return None
        except Exception as e:
            print(f"Warning: Could not apply GEBCO mask to {mgrs_subtile}: {e}")

    print(f"Processing tile {mgrs_subtile} across {len(folders)} date(s)...")

    # 2. Load all dates into memory (with block-level skipping to save bandwidth)
    date_stacks = []
    for folder_name, date_path in folders:
        paths = get_tile_paths(folder_name, date_path, args.cache, download=False)
        bands_data = [
            np.full((height, width), np.nan, dtype=np.float32) for _ in range(3)
        ]

        # Open datasets once per folder
        dss = [gdal.Open(paths[color]) for color in ["red", "green", "blue"]]
        bands = [ds.GetRasterBand(1) for ds in dss]

        # Use large blocks to skip ocean regions efficiently while maintaining throughput
        block_size = 2048
        for yoff in range(0, height, block_size):
            bh = min(block_size, height - yoff)
            for xoff in range(0, width, block_size):
                bw = min(block_size, width - xoff)

                # If this block is 100% deep ocean according to GEBCO, skip the S3 request
                if gebco_src and np.all(
                    gebco_data[yoff : yoff + bh, xoff : xoff + bw] <= OCEAN_FADE
                ):
                    continue

                for i in range(3):
                    block_arr = (
                        bands[i].ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                    )
                    # Use NaN for nodata -32768
                    block_arr[block_arr == -32768] = np.nan
                    bands_data[i][yoff : yoff + bh, xoff : xoff + bw] = block_arr

        # Close datasets for this date
        for ds in dss:
            ds = None
        date_stacks.append(np.stack(bands_data))  # (3, H, W)

    if not date_stacks:
        return None

    # 3. Average dates (ignoring NaNs)
    full_stack = np.stack(date_stacks)  # (D, 3, H, W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        averaged = np.nanmean(full_stack, axis=0)

    # 4. Final Masking (clean up ocean pixels within land blocks)
    if alpha_weight is not None:
        # Hard cut only at the very end of the fade to prevent NaN pollution in blend
        deep_ocean_mask = gebco_data <= OCEAN_FADE
        for i in range(3):
            averaged[i][deep_ocean_mask] = np.nan

    # 3. Tone Mapping
    # Determine scaling (either hardcoded or via percentile)
    if args.stats_min is not None:
        s_min = args.stats_min
    else:
        s_min = 0.0

    if args.stats_max is not None:
        s_max = args.stats_max
    else:
        # 9000 is a "safe" universal default for Sentinel-2 (90% reflectance).
        # It prevents clipping in bright regions like sand, snow, or clouds.
        s_max = 9000.0

    normalized = np.clip((averaged - s_min) / (s_max - s_min), 0.0, 1.0)
    normalized[np.isnan(normalized)] = 0.0

    # Apply tone mapping (soft-knee)
    if args.tonemap:
        toned = tiler.apply_soft_knee_numpy(
            normalized,
            shadow_break=args.sb,
            highlight_break=args.hb,
            shadow_slope=args.ss,
            mid_slope=args.ms,
            highlight_slope=args.hs,
            exposure=args.exposure,
        )
    else:
        toned = np.clip(normalized * args.exposure, 0.0, 1.0)

    # Apply final grading (preview correction)
    if args.grade:
        toned = tiler.apply_preview_correction_numpy(
            toned,
            saturation=args.sat,
            darken_break=args.db,
            low_slope=args.ls,
            gamma=args.gamma,
        )

    # 4. Pixel-level Blending (actually blend the colors, not just transparency)
    if alpha_weight is not None:
        # Generate the same ocean color ramp used for the background
        mako_colors = (
            np.array([c[1:] for c in tiler.MAKO_RAMP], dtype=np.float32) / 255.0
        )
        if args.ocean_tonemap:
            mako_arr = mako_colors.T.reshape(3, -1, 1)
            toned_mako = tiler.apply_soft_knee_numpy(
                mako_arr,
                shadow_break=args.osb,
                highlight_break=args.ohb,
                shadow_slope=args.oss,
                mid_slope=args.oms,
                highlight_slope=args.ohs,
                exposure=args.ocean_exposure,
            )
            mako_colors = toned_mako.reshape(3, -1).T
        else:
            mako_colors = np.clip(mako_colors * args.ocean_exposure, 0.0, 1.0)

        if args.ocean_grade:
            mako_arr = mako_colors.T.reshape(3, -1, 1)
            graded_mako = tiler.apply_preview_correction_numpy(
                mako_arr,
                saturation=args.osat,
                darken_break=args.odb,
                low_slope=args.ols,
                gamma=args.ocean_gamma,
            )
            mako_colors = graded_mako.reshape(3, -1).T

        ocean_rgb = tiler.colorize_depth_numpy(
            gebco_data, mako_colors, args.ocean_depth_min, args.ocean_depth_max
        )

        # Blend: Toned * alpha + Ocean * (1 - alpha)
        # Note: alpha_weight is (H,W), toned is (3,H,W), ocean_rgb is (3,H,W)
        toned = toned * alpha_weight + ocean_rgb * (1.0 - alpha_weight)

    # 5. Save to temporary Byte GeoTIFF
    temp_utm_path = f".temp/processed_{mgrs_subtile}_{unique_id}_utm.tif"
    temp_3857_path = f".temp/processed_{mgrs_subtile}_{unique_id}_3857.tif"

    driver = gdal.GetDriverByName("GTiff")

    # Calculate alpha mask from finite values (non-NaN)
    # Since we blended with the ocean, the land tile is now opaque where we have depth data
    finite_mask = np.isfinite(averaged[0]).astype(np.float32)
    combined_alpha = (finite_mask * 255).astype(np.uint8)

    byte_arr = np.nan_to_num(toned * 255, nan=0).astype(np.uint8)

    ds_out = driver.Create(
        temp_utm_path,
        byte_arr.shape[2],
        byte_arr.shape[1],
        4,
        gdal.GDT_Byte,
        options=["COMPRESS=ZSTD", "TILED=YES"],
    )
    ds_out.SetProjection(projection)
    ds_out.SetGeoTransform(geotransform)
    for i in range(3):
        out_band = ds_out.GetRasterBand(i + 1)
        out_band.WriteArray(byte_arr[i])
        out_band.SetColorInterpretation(
            getattr(gdal, f"GCI_{['RedBand', 'GreenBand', 'BlueBand'][i]}")
        )

    alpha_band = ds_out.GetRasterBand(4)
    alpha_band.WriteArray(combined_alpha)
    alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)

    ds_out.FlushCache()
    ds_out = None

    # 5. Warp to Web Mercator
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        resampleAlg=args.resample_alg,
        # Using 4th band as alpha for transparency, no need for srcNodata/dstNodata
        creationOptions=["COMPRESS=ZSTD", "ZSTD_LEVEL=5", "PREDICTOR=2", "TILED=YES"],
    )
    gdal.Warp(temp_3857_path, temp_utm_path, options=warp_options)
    os.remove(temp_utm_path)

    return temp_3857_path


def calculate_estimates(args: argparse.Namespace) -> None:
    """Calculate and print estimations for the given command."""
    date_paths = [d.strip() for d in args.date.split(",")]
    num_dates = len(date_paths)

    land_only_file = "HLS.land.tiles.txt"
    land_set = None
    if os.path.exists(land_only_file):
        with open(land_only_file, "r") as f:
            land_set = {line.strip() for line in f if line.strip()}

    if args.bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.bbox.split(","))
        except ValueError:
            print(f"Error: Invalid bbox format: {args.bbox}")
            sys.exit(1)

        m_converter = mgrs.MGRS()
        discovered_mgrs = set()
        lon_steps = int((max_lon - min_lon) / 0.1) + 2
        lat_steps = int((max_lat - min_lat) / 0.1) + 2
        for i in range(lon_steps):
            lon = min(min_lon + i * 0.1, max_lon)
            for j in range(lat_steps):
                lat = min(min_lat + j * 0.1, max_lat)
                try:
                    mgrs_res = m_converter.toMGRS(lat, lon, MGRSPrecision=0)
                    mgrs_str = (
                        mgrs_res.decode("utf-8")
                        if isinstance(mgrs_res, bytes)
                        else mgrs_res
                    )
                    discovered_mgrs.add(mgrs_str)
                except Exception:
                    continue

        if land_set is not None:
            mgrs_bases = [m for m in discovered_mgrs if m in land_set]
        else:
            mgrs_bases = list(discovered_mgrs)
    elif args.all_tiles:
        if land_set is not None:
            mgrs_bases = list(land_set)
        else:
            print("Error: --global requires HLS.land.tiles.txt to be present.")
            sys.exit(1)
    else:
        mgrs_bases = [m.strip() for m in args.mgrs.split(",") if m.strip()]

    num_mgrs = len(mgrs_bases)
    num_subtiles = num_mgrs * 4
    total_tile_dates = num_subtiles * num_dates

    # Rough estimations based on 5000x5000 float32 subtiles
    # Network: ~200MB per band * 3 bands = 600MB per tile-date
    network_gb = (total_tile_dates * 600) / 1024

    # RAM: ~8GB per process for NumPy stack/mean operations
    ram_gb = args.parallel * 8

    # Time: ~15s per tile-date processing + ~30s per subtile for warping/tiling
    total_seconds = (total_tile_dates * 15 / args.parallel) + (
        num_subtiles * 30 / args.parallel
    )
    total_seconds += num_subtiles * 2

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    # Disk Space Peak (Optimized):
    # 1. Cached raw bands: 60MB * total_tile_dates (if using --cache)
    # 2. Processed intermediate TIFs: ~50MB * num_subtiles (ZSTD Predictor=2)
    # 3. MBTiles chunks: Now deleted during merge, peak is small (~100MB)
    disk_peak_gb = (total_tile_dates * 60 + num_subtiles * 50 + 100) / 1024

    # Disk Space End:
    # Final PMTiles: ~40MB * num_subtiles (very rough)
    disk_end_gb = (num_subtiles * 40) / 1024

    print(f"--- Processing Estimates ---")
    print(f"MGRS Tiles (100km): {num_mgrs}")
    print(f"MGRS Sub-tiles (4x): {num_subtiles}")
    print(f"Date(s):            {num_dates} ({', '.join(date_paths)})")
    print(f"Total tile-dates:    {total_tile_dates} (3 bands each)")
    print(f"---------------------------")
    print(f"Estimated Time:       {hours}h {minutes}m")
    print(f"Estimated RAM Usage:  {ram_gb:.1f} GB (peak)")
    print(f"Estimated Disk Peak:  {disk_peak_gb:.2f} GB")
    print(f"Estimated Disk End:   {disk_end_gb:.2f} GB")
    print(f"Estimated Network:    {network_gb:.2f} GB")
    print(f"---------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE (NumPy Refactor)."
    )
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID")
    parser.add_argument(
        "--global",
        dest="all_tiles",
        action="store_true",
        help="Process all available tiles",
    )
    parser.add_argument(
        "--date",
        default="2025/07/01,2025/01/01",
        help="Mosaic date(s), comma-separated",
    )
    parser.add_argument(
        "--output", "-o", default="output.pmtiles", help="Output PMTiles filename"
    )
    parser.add_argument(
        "--format",
        choices=["webp", "jpg", "png", "png8"],
        default="webp",
        help="Tile format",
    )
    parser.add_argument("--quality", type=int, default=74, help="Quality (0-100)")
    parser.add_argument(
        "--resample-alg",
        default="lanczos",
        choices=["bilinear", "average", "gauss", "lanczos"],
    )
    parser.add_argument(
        "--chunk-zoom",
        type=int,
        default=4,
        help="Zoom level to chunk the processing at",
    )
    parser.add_argument(
        "--parallel", type=int, default=2, help="Number of parallel processes"
    )
    parser.add_argument("--stats-min", type=float, help="Hardcoded source min")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source max")

    # Tone Mapping
    parser.add_argument("--exposure", type=float, default=tiler.DEFAULT_EXPOSURE)
    parser.add_argument(
        "--sb", "--shadow-break", type=float, default=tiler.SOFT_KNEE_SHADOW_BREAK
    )
    parser.add_argument(
        "--hb", "--highlight-break", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_BREAK
    )
    parser.add_argument(
        "--ss", "--shadow-slope", type=float, default=tiler.SOFT_KNEE_SHADOW_SLOPE
    )
    parser.add_argument(
        "--ms", "--mid-slope", type=float, default=tiler.SOFT_KNEE_MID_SLOPE
    )
    parser.add_argument(
        "--hs", "--highlight-slope", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_SLOPE
    )

    # Grading
    parser.add_argument("--gamma", type=float, default=tiler.DEFAULT_GAMMA)
    parser.add_argument(
        "--sat", "--saturation", type=float, default=tiler.PREVIEW_SATURATION
    )
    parser.add_argument(
        "--db", "--black-break", type=float, default=tiler.PREVIEW_DARKEN_BREAK
    )
    parser.add_argument(
        "--ls", "--black-slope", type=float, default=tiler.PREVIEW_DARKEN_LOW_SLOPE
    )

    # Ocean-specific Tone Mapping
    parser.add_argument("--ocean-exposure", type=float, default=tiler.DEFAULT_EXPOSURE)
    parser.add_argument(
        "--osb",
        "--ocean-shadow-break",
        type=float,
        default=tiler.SOFT_KNEE_SHADOW_BREAK,
    )
    parser.add_argument(
        "--ohb",
        "--ocean-highlight-break",
        type=float,
        default=tiler.SOFT_KNEE_HIGHLIGHT_BREAK,
    )
    parser.add_argument(
        "--oss",
        "--ocean-shadow-slope",
        type=float,
        default=tiler.SOFT_KNEE_SHADOW_SLOPE,
    )
    parser.add_argument(
        "--oms", "--ocean-mid-slope", type=float, default=tiler.SOFT_KNEE_MID_SLOPE
    )
    parser.add_argument(
        "--ohs",
        "--ocean-highlight-slope",
        type=float,
        default=tiler.SOFT_KNEE_HIGHLIGHT_SLOPE,
    )

    # Ocean-specific Grading
    parser.add_argument("--ocean-gamma", type=float, default=tiler.DEFAULT_GAMMA)
    parser.add_argument(
        "--osat", "--ocean-saturation", type=float, default=tiler.PREVIEW_SATURATION
    )
    parser.add_argument(
        "--odb", "--ocean-black-break", type=float, default=tiler.PREVIEW_DARKEN_BREAK
    )
    parser.add_argument(
        "--ols",
        "--ocean-black-slope",
        type=float,
        default=tiler.PREVIEW_DARKEN_LOW_SLOPE,
    )

    # Ocean-specific Depth Range
    parser.add_argument(
        "--ocean-depth-min",
        type=float,
        default=-11000,
        help="Depth value for 0.0 in the color ramp",
    )
    parser.add_argument(
        "--ocean-depth-max",
        type=float,
        default=0,
        help="Depth value for 1.0 in the color ramp",
    )

    parser.add_argument(
        "--tonemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land tone mapping",
    )
    parser.add_argument(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land final grading",
    )
    parser.add_argument(
        "--ocean-tonemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ocean tone mapping",
    )
    parser.add_argument(
        "--ocean-grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ocean final grading",
    )
    parser.add_argument(
        "--land",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land tile processing",
    )

    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download S3 tiles to local cache and exit",
    )
    parser.add_argument(
        "--bbox", help="WGS84 bounding box as min_lon,min_lat,max_lon,max_lat"
    )
    parser.add_argument(
        "--vrt", action="store_true", help="Write final VRT and skip MBTiles"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print estimated time, RAM, disk space, and network size then exit",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="Resume from a previous run if a state file exists",
    )
    args = parser.parse_args()

    if args.estimate:
        calculate_estimates(args)
        return

    setup_gdal_cdse()
    os.makedirs(".temp", exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    state_file = f".temp/state_{unique_id}.json"
    completed_subtiles = set()
    processed_tifs = []

    if args.resume:
        # If resume is a string, it's a specific state file
        resume_path = (
            args.resume
            if isinstance(args.resume, str) and os.path.exists(args.resume)
            else None
        )

        # If not, look for the most recent state file in .temp
        if not resume_path:
            states = glob.glob(".temp/state_*.json")
            if states:
                resume_path = max(states, key=os.path.getmtime)

        if resume_path:
            state = load_state(resume_path)
            if state:
                state_file = resume_path
                unique_id = state["unique_id"]
                completed_subtiles = set(state["completed_subtiles"])
                processed_tifs = [
                    f for f in state["processed_tifs"] if os.path.exists(f)
                ]
                print(
                    f"Resuming from state file: {resume_path} (unique_id: {unique_id})"
                )
                print(
                    f"Already completed {len(completed_subtiles)} sub-tiles, {len(processed_tifs)} TIFs found."
                )

    date_paths = [d.strip() for d in args.date.split(",")]

    # 0. GEBCO Discovery (for both filtering and background)
    gebco_zip = "gebco_2025_sub_ice_topo_geotiff.zip"
    gebco_vrt_source = None
    if os.path.exists(gebco_zip):
        print("Opening GEBCO bathymetry zip...")
        gebco_vsi = f"/vsizip/{gebco_zip}"
        files_in_zip = gdal.ReadDir(gebco_vsi)
        tif_files = [f for f in files_in_zip if f.lower().endswith(".tif")]
        if tif_files:
            gebco_vrt_source = f".temp/gebco_source_{unique_id}.vrt"
            # Recreate GEBCO VRT if it doesn't exist (it has unique_id in name)
            if not os.path.exists(gebco_vrt_source):
                tif_paths = [f"{gebco_vsi}/{f}" for f in tif_files]
                gdal.BuildVRT(gebco_vrt_source, tif_paths)

    # 1. Tile Discovery - Pre-populate S3 cache only for global runs
    if args.all_tiles:
        print("Populating S3 folder cache...")
        for dp in date_paths:
            s3_base = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{dp}"
            dirs = gdal.ReadDir(s3_base)
            if dirs:
                S3_FOLDER_CACHE[dp] = set(dirs)
            else:
                print(f"Warning: Could not list folders for {dp}")

    # 2. Sub-tile Expansion
    land_only_file = "HLS.land.tiles.txt"
    land_set = None
    if os.path.exists(land_only_file):
        with open(land_only_file, "r") as f:
            land_set = {line.strip() for line in f if line.strip()}

    if args.bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.bbox.split(","))
        except ValueError:
            print(f"Error: Invalid bbox format: {args.bbox}")
            sys.exit(1)

        m_converter = mgrs.MGRS()
        discovered_mgrs = set()
        # Sample coordinates within bbox every 0.1 degree (robust for 100km MGRS squares)
        lon_steps = int((max_lon - min_lon) / 0.1) + 2
        lat_steps = int((max_lat - min_lat) / 0.1) + 2
        for i in range(lon_steps):
            lon = min(min_lon + i * 0.1, max_lon)
            for j in range(lat_steps):
                lat = min(min_lat + j * 0.1, max_lat)
                try:
                    # Convert to MGRS string, e.g., '4QFJ' (precision 0 gives 100km square ID)
                    mgrs_res = m_converter.toMGRS(lat, lon, MGRSPrecision=0)
                    mgrs_str = (
                        mgrs_res.decode("utf-8")
                        if isinstance(mgrs_res, bytes)
                        else mgrs_res
                    )
                    discovered_mgrs.add(mgrs_str)
                except Exception:
                    continue

        discovered_mgrs = list(discovered_mgrs)

        # Filter tiles using land_set AND GEBCO
        mgrs_bases = []
        for m in discovered_mgrs:
            is_land = False
            if land_set and m in land_set:
                is_land = True
            elif gebco_vrt_source:
                if check_land_gebco(m, gebco_vrt_source):
                    is_land = True
            elif land_set is None:
                is_land = True

            if is_land:
                mgrs_bases.append(m)

        print(
            f"Discovered {len(discovered_mgrs)} MGRS tiles in bbox, {len(mgrs_bases)} kept after GEBCO/land filtering."
        )

    elif args.all_tiles:
        if land_set is not None:
            mgrs_bases = list(land_set)
        else:
            print("Error: --global requires HLS.land.tiles.txt to be present.")
            sys.exit(1)
    else:
        mgrs_bases = [m.strip() for m in args.mgrs.split(",") if m.strip()]

    subtiles = [f"{m}_{x}_{y}" for m in mgrs_bases for x in [0, 1] for y in [0, 1]]

    # 2. Parallel Tile Processing
    if args.land:
        subtiles_to_process = [st for st in subtiles if st not in completed_subtiles]
        if subtiles_to_process:
            print(f"Starting processing for {len(subtiles_to_process)} sub-tiles...")
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_st = {
                    executor.submit(
                        process_single_tile,
                        st,
                        date_paths,
                        args,
                        unique_id,
                        gebco_vrt_source,
                    ): st
                    for st in subtiles_to_process
                }
                for future in as_completed(future_to_st):
                    st = future_to_st[future]
                    res = future.result()
                    completed_subtiles.add(st)
                    if res:
                        processed_tifs.append(res)
                    save_state(
                        state_file, unique_id, completed_subtiles, processed_tifs, args
                    )
        else:
            print("All sub-tiles already processed.")
    else:
        print("Skipping land tile processing (--no-land).")

    if args.download:
        print("Download complete.")
        return

    if not processed_tifs and not args.bbox:
        print("Error: No data processed (no land tiles found and no bbox provided).")
        sys.exit(1)

    # 3. Bathymetry Background (Ocean)
    # If --bbox is provided, we fill non-land with GEBCO bathymetry
    if args.bbox and gebco_vrt_source:
        gebco_3857 = f".temp/gebco_3857_{unique_id}.vrt"
        if os.path.exists(gebco_3857):
            print("Bathymetry background already exists.")
        else:
            print("Generating bathymetry background...")
            gebco_src = gebco_vrt_source

            # Create color file based on 'Mako' color ramp
            color_file = f".temp/gebco_colors_{unique_id}.txt"

            # Mako-inspired stops (fraction 0.0 to 1.0)
            mako_ramp = tiler.MAKO_RAMP

            if args.ocean_tonemap:
                # Apply ocean tone mapping (soft-knee)
                # Extract RGB colors (0-255) and normalize to 0-1
                mako_colors = (
                    np.array([c[1:] for c in mako_ramp], dtype=np.float32) / 255.0
                )
                mako_arr = mako_colors.T.reshape(3, -1, 1)
                toned_mako = tiler.apply_soft_knee_numpy(
                    mako_arr,
                    shadow_break=args.osb,
                    highlight_break=args.ohb,
                    shadow_slope=args.oss,
                    mid_slope=args.oms,
                    highlight_slope=args.ohs,
                    exposure=args.ocean_exposure,
                )
                # Convert back for potential grading or final use
                mako_colors = toned_mako.reshape(3, -1).T
            else:
                # Just apply exposure if tonemap is off
                mako_colors = (
                    np.array([c[1:] for c in mako_ramp], dtype=np.float32) / 255.0
                )
                mako_colors = np.clip(mako_colors * args.ocean_exposure, 0.0, 1.0)

            if args.ocean_grade:
                # Apply ocean grading
                mako_arr = mako_colors.T.reshape(3, -1, 1)
                graded_mako = tiler.apply_preview_correction_numpy(
                    mako_arr,
                    saturation=args.osat,
                    darken_break=args.odb,
                    low_slope=args.ols,
                    gamma=args.ocean_gamma,
                )
                mako_colors = graded_mako.reshape(3, -1).T
            if args.ocean_tonemap or args.ocean_grade:
                # Convert back to uint8 and rebuild the ramp
                graded_uint8 = (mako_colors * 255).astype(np.uint8)
                mako_ramp = [
                    (mako_ramp[i][0], *graded_uint8[i]) for i in range(len(mako_ramp))
                ]

            # Map depth range to 0.0-1.0
            depth_min = args.ocean_depth_min
            depth_max = args.ocean_depth_max

            with open(color_file, "w") as f:
                for frac, r, g, b in mako_ramp:
                    val = depth_min + frac * (depth_max - depth_min)
                    f.write(f"{val} {r} {g} {b} 255\n")
                # Explicitly make land transparent (above max depth)
                f.write(f"{depth_max + 0.01} 0 0 0 0\n")
                f.write("10000 0 0 0 0\n")
                f.write("nv 0 0 0 0\n")

            # Colorize using DEMProcessing (format=VRT to avoid intermediate write)
            colored_vrt = f".temp/gebco_colored_{unique_id}.vrt"
            gdal.DEMProcessing(
                colored_vrt,
                gebco_src,
                "color-relief",
                colorFilename=color_file,
                format="VRT",
                addAlpha=True,
            )

            # Warp to EPSG:3857 and crop to bbox
            min_lon, min_lat, max_lon, max_lat = map(float, args.bbox.split(","))
            # Using -te minx miny maxx maxy in dstSRS
            warp_opts = gdal.WarpOptions(
                format="VRT",
                dstSRS="EPSG:3857",
                outputBounds=(min_lon, min_lat, max_lon, max_lat),
                outputBoundsSRS="EPSG:4326",
            )
            gdal.Warp(gebco_3857, colored_vrt, options=warp_opts)

        # Prepend to the processed list to make it the bottom layer if not already there
        if gebco_3857 not in processed_tifs:
            processed_tifs.insert(0, gebco_3857)

    # 4. Build Master VRT (Flat, over Web Mercator Byte TIFFs)
    master_vrt = f".temp/master_{unique_id}.vrt"
    gdal.BuildVRT(master_vrt, processed_tifs, resolution="highest")

    if args.vrt:
        print(f"Success! Master VRT: {master_vrt}")
        return

    # 4. Tiling (Directly from the Byte VRT)
    temp_mbtiles = f".temp/tiles_{unique_id}.mbtiles"
    tiling_opts = {
        "format": args.format,
        "quality": args.quality,
        "resample_alg": args.resample_alg,
        "chunk_zoom": args.chunk_zoom,
        "processes": args.parallel,
        "blocksize": args.blocksize,
        "name": "Sentinel-2 Mosaic",
        "description": "Copernicus Sentinel data",
        "unique_id": unique_id,
    }

    print("Generating MBTiles...")
    # Modified run_tiling will be simpler now
    artifacts = tiler.run_tiling_simplified(master_vrt, temp_mbtiles, tiling_opts)

    print("Converting to PMTiles...")
    subprocess.run(["pmtiles", "convert", temp_mbtiles, args.output], check=True)
    print(f"Success! {args.output}")

    # Cleanup
    for f in processed_tifs + [master_vrt, temp_mbtiles] + artifacts.cleanup_paths:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    delete_state(state_file)


if __name__ == "__main__":
    main()
