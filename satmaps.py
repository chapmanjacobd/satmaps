#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import mgrs
import numpy as np
from scipy.ndimage import distance_transform_edt
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
            except Exception:
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
    except Exception:
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
def fill_nan_nearest(
    arr: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Fill invalid pixels in a (C, H, W) array with the nearest valid pixel."""
    if valid_mask is None:
        valid_mask = np.all(np.isfinite(arr), axis=0)

    if valid_mask.shape != arr.shape[1:]:
        raise ValueError("valid_mask must match the spatial shape of arr")

    if np.all(valid_mask) or not np.any(valid_mask):
        return arr

    indices = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True
    )

    filled = np.array(arr, copy=True)
    invalid_mask = ~valid_mask
    for i in range(arr.shape[0]):
        nearest_band = arr[i][tuple(indices)]
        filled[i][invalid_mask] = nearest_band[invalid_mask]

    return filled


def mosaic_date_stacks(date_stacks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Average only complete RGB observations and track where any date was valid."""
    if not date_stacks:
        raise ValueError("date_stacks must not be empty")

    first_stack = date_stacks[0]
    summed = np.zeros_like(first_stack, dtype=np.float32)
    valid_counts = np.zeros(first_stack.shape[1:], dtype=np.uint16)
    valid_source_mask = np.zeros(first_stack.shape[1:], dtype=bool)

    for date_stack in date_stacks:
        complete_rgb_mask = np.all(np.isfinite(date_stack), axis=0)
        if not np.any(complete_rgb_mask):
            continue

        valid_source_mask |= complete_rgb_mask
        valid_counts[complete_rgb_mask] += 1
        for band_index in range(date_stack.shape[0]):
            np.add(
                summed[band_index],
                date_stack[band_index],
                out=summed[band_index],
                where=complete_rgb_mask,
            )

    averaged = np.full_like(first_stack, np.nan, dtype=np.float32)
    for band_index in range(first_stack.shape[0]):
        np.divide(
            summed[band_index],
            valid_counts,
            out=averaged[band_index],
            where=valid_source_mask,
        )

    return averaged, valid_source_mask


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

    gebco_band = None
    gebco_mem_ds = None
    fill_allowed_mask = None
    # Transition constants
    LAND_STAY = -42.0  # 100% Sentinel-2 above this depth
    OCEAN_FADE = -50.0  # 0% Sentinel-2 below this depth

    date_band_sets = []
    try:
        if gebco_src:
            try:
                gebco_ds = gdal.Open(gebco_src)
                if gebco_ds is None:
                    raise RuntimeError(f"Could not open GEBCO source: {gebco_src}")
                try:
                    # Create in-memory dataset matching the Sentinel-2 grid
                    mem_driver = gdal.GetDriverByName("MEM")
                    gebco_mem_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Float32)
                    gebco_mem_ds.SetProjection(projection)
                    gebco_mem_ds.SetGeoTransform(geotransform)

                    # Warp GEBCO into the Sentinel-2 grid
                    gdal.Warp(
                        gebco_mem_ds, gebco_ds, options=gdal.WarpOptions(resampleAlg="bilinear")
                    )
                finally:
                    gebco_ds = None
                gebco_band = gebco_mem_ds.GetRasterBand(1)
                fill_allowed_mask = np.zeros((height, width), dtype=bool)
            except Exception as e:
                print(f"Warning: Could not apply GEBCO mask to {mgrs_subtile}: {e}")

        print(f"Processing tile {mgrs_subtile} across {len(folders)} date(s)...")

        # 2. Average dates block-by-block so only one working window is resident at a time.
        averaged = np.full((3, height, width), np.nan, dtype=np.float32)
        source_valid_mask = np.zeros((height, width), dtype=bool)
        block_size = 1024
        found_non_ocean_pixels = gebco_band is None
        for folder_name, date_path in folders:
            paths = get_tile_paths(folder_name, date_path, args.cache, download=False)
            dss = [gdal.Open(paths[color]) for color in ["red", "green", "blue"]]
            date_band_sets.append(([ds.GetRasterBand(1) for ds in dss], dss))

        for yoff in range(0, height, block_size):
            bh = min(block_size, height - yoff)
            for xoff in range(0, width, block_size):
                bw = min(block_size, width - xoff)

                if gebco_band is not None:
                    gebco_block = gebco_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                    assert fill_allowed_mask is not None
                    fill_allowed_block = gebco_block > OCEAN_FADE
                    fill_allowed_mask[yoff : yoff + bh, xoff : xoff + bw] = fill_allowed_block
                    if not np.any(fill_allowed_block):
                        continue
                    found_non_ocean_pixels = True

                summed = np.zeros((3, bh, bw), dtype=np.float32)
                valid_counts = np.zeros((bh, bw), dtype=np.uint16)
                block_valid_sources = np.zeros((bh, bw), dtype=bool)

                for bands, _ in date_band_sets:
                    rgb_block = np.empty((3, bh, bw), dtype=np.float32)
                    for band_index, band in enumerate(bands):
                        block_arr = band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                        block_arr[block_arr == -32768] = np.nan
                        rgb_block[band_index] = block_arr

                    complete_rgb_mask = np.all(np.isfinite(rgb_block), axis=0)
                    if not np.any(complete_rgb_mask):
                        continue

                    block_valid_sources |= complete_rgb_mask
                    valid_counts[complete_rgb_mask] += 1
                    for band_index in range(rgb_block.shape[0]):
                        np.add(
                            summed[band_index],
                            rgb_block[band_index],
                            out=summed[band_index],
                            where=complete_rgb_mask,
                        )

                if not np.any(block_valid_sources):
                    continue

                averaged_block = np.full((3, bh, bw), np.nan, dtype=np.float32)
                for band_index in range(averaged_block.shape[0]):
                    np.divide(
                        summed[band_index],
                        valid_counts,
                        out=averaged_block[band_index],
                        where=block_valid_sources,
                    )
                averaged[:, yoff : yoff + bh, xoff : xoff + bw] = averaged_block
                source_valid_mask[yoff : yoff + bh, xoff : xoff + bw] = block_valid_sources

        if not found_non_ocean_pixels:
            return None

        # 3. Fill land/coastal gaps from the nearest valid source pixel while leaving
        # pure ocean transparent so the prebuilt ocean background can show through.
        fill_mask = ~source_valid_mask
        if fill_allowed_mask is not None:
            fill_mask &= fill_allowed_mask
        if np.any(fill_mask):
            averaged = fill_nan_nearest(averaged, valid_mask=source_valid_mask)

        # 4. Tone mapping is windowed to avoid creating more full-size NumPy copies.
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

        # 5. Save to temporary Byte GeoTIFF while converting each window independently.
        temp_utm_path = f".temp/processed_{mgrs_subtile}_{unique_id}_utm.tif"
        temp_3857_path = f".temp/processed_{mgrs_subtile}_{unique_id}_3857.tif"

        driver = gdal.GetDriverByName("GTiff")

        ds_out = driver.Create(
            temp_utm_path,
            width,
            height,
            4,
            gdal.GDT_Byte,
            options=["COMPRESS=ZSTD", "TILED=YES"],
        )
        ds_out.SetProjection(projection)
        ds_out.SetGeoTransform(geotransform)
        color_bands = [ds_out.GetRasterBand(i + 1) for i in range(3)]
        for band_index, color_name in enumerate(("RedBand", "GreenBand", "BlueBand")):
            color_bands[band_index].SetColorInterpretation(getattr(gdal, f"GCI_{color_name}"))

        alpha_band = ds_out.GetRasterBand(4)
        alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)

        scale = s_max - s_min
        if scale <= 0.0:
            raise ValueError("stats_max must be greater than stats_min")

        for yoff in range(0, height, block_size):
            bh = min(block_size, height - yoff)
            for xoff in range(0, width, block_size):
                bw = min(block_size, width - xoff)
                averaged_block = averaged[:, yoff : yoff + bh, xoff : xoff + bw]
                normalized = np.clip((averaged_block - s_min) / scale, 0.0, 1.0)
                normalized[np.isnan(normalized)] = 0.0

                if args.tonemap:
                    toned_block = tiler.apply_soft_knee_numpy(
                        normalized,
                        shadow_break=args.sb,
                        highlight_break=args.hb,
                        shadow_slope=args.ss,
                        mid_slope=args.ms,
                        highlight_slope=args.hs,
                        exposure=args.exposure,
                    )
                else:
                    toned_block = np.clip(normalized * args.exposure, 0.0, 1.0)

                if args.grade:
                    toned_block = tiler.apply_preview_correction_numpy(
                        toned_block,
                        saturation=args.sat,
                        darken_break=args.db,
                        low_slope=args.ls,
                        gamma=args.gamma,
                    )

                byte_block = np.nan_to_num(toned_block * 255.0, nan=0.0).astype(np.uint8)
                for band_index, out_band in enumerate(color_bands):
                    out_band.WriteArray(byte_block[band_index], xoff=xoff, yoff=yoff)

                if gebco_band is not None:
                    gebco_block = gebco_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                    alpha_block = np.clip(
                        (gebco_block - OCEAN_FADE) / (LAND_STAY - OCEAN_FADE), 0.0, 1.0
                    )
                    combined_alpha = np.clip(alpha_block * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    combined_alpha = source_valid_mask[yoff : yoff + bh, xoff : xoff + bw].astype(
                        np.uint8
                    ) * 255

                alpha_band.WriteArray(combined_alpha, xoff=xoff, yoff=yoff)

        ds_out.FlushCache()
        ds_out = None

        # 6. Warp to Web Mercator
        warp_options = gdal.WarpOptions(
            format="GTiff",
            dstSRS="EPSG:3857",
            resampleAlg=args.resample_alg,
            # Standard RGB GeoTIFF creation options
            creationOptions=["COMPRESS=ZSTD", "ZSTD_LEVEL=5", "PREDICTOR=2", "TILED=YES"],
        )
        gdal.Warp(temp_3857_path, temp_utm_path, options=warp_options)
        os.remove(temp_utm_path)

        return temp_3857_path
    finally:
        for bands, dss in date_band_sets:
            bands.clear()
            dss.clear()
        date_band_sets.clear()
        gebco_band = None
        gebco_mem_ds = None
        fill_allowed_mask = None


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

    print("--- Processing Estimates ---")
    print(f"MGRS Tiles (100km): {num_mgrs}")
    print(f"MGRS Sub-tiles (4x): {num_subtiles}")
    print(f"Date(s):            {num_dates} ({', '.join(date_paths)})")
    print(f"Total tile-dates:    {total_tile_dates} (3 bands each)")
    print("---------------------------")
    print(f"Estimated Time:       {hours}h {minutes}m")
    print(f"Estimated RAM Usage:  {ram_gb:.1f} GB (peak)")
    print(f"Estimated Disk Peak:  {disk_peak_gb:.2f} GB")
    print(f"Estimated Disk End:   {disk_end_gb:.2f} GB")
    print(f"Estimated Network:    {network_gb:.2f} GB")
    print("---------------------------")


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
        "--land",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land tile processing",
    )

    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument(
        "--ocean-background",
        default="ocean.tif",
        help="Standalone ocean background GeoTIFF to use under bbox renders",
    )
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

    # 0. GEBCO Discovery (for filtering and coastal blending)
    gebco_zip = "gebco_2025_sub_ice_topo_geotiff.zip"
    gebco_vrt_source = None
    if os.path.exists(gebco_zip):
        print("Opening GEBCO bathymetry zip...")
        gebco_vsi = f"/vsizip/{gebco_zip}"
        files_in_zip = gdal.ReadDir(gebco_vsi)
        tif_files = [f for f in files_in_zip if f.lower().endswith(".tif")]
        if tif_files:
            gebco_vrt_source = f".temp/gebco_source_{unique_id}.vrt"
            # Always recreate GEBCO VRT (it's fast and ensures consistency)
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

    # 3. Optional standalone ocean background
    if os.path.exists(args.ocean_background):
        if args.ocean_background not in processed_tifs:
            processed_tifs.insert(0, args.ocean_background)
    elif args.bbox:
        print(f"Warning: Ocean background not found, skipping: {args.ocean_background}")

    if not processed_tifs:
        print("Error: No data processed.")
        sys.exit(1)

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
