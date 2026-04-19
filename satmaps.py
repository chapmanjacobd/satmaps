#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from osgeo import gdal

import tiler

# Setup GDAL exceptions
gdal.UseExceptions()

@dataclass
class ProcessingOptions:
    """Container for tile generation settings."""
    format: str
    quality: int
    resample_alg: str
    chunk_zoom: int
    processes: int
    blocksize: int
    name: str
    description: str
    vrt: bool = False
    step5: bool = False
    stats_min: Optional[float] = None
    stats_max: Optional[float] = None


AUTO_SCALE_MAX_PERCENTILE = 99.6
HISTOGRAM_MIN = 0.0
HISTOGRAM_MAX = 40000.0
HISTOGRAM_BUCKETS = 4000

# --- Discovery Layer (S3/CDSE Utils) ---

def setup_gdal_cdse() -> None:
    """Configure GDAL for CDSE S3 access."""
    gdal.SetConfigOption('AWS_S3_ENDPOINT', 'eodata.dataspace.copernicus.eu')
    gdal.SetConfigOption('AWS_HTTPS', 'YES')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
    gdal.SetConfigOption('AWS_PROFILE', 'cdse')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('GDAL_CACHEMAX', '1024')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
    # Enable VRT expressions
    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')

def get_quarter(date_path: str) -> str:
    """Determine quarter (Q1-Q4) from a date path string."""
    if "07/01" in date_path: return "Q3"
    if "10/01" in date_path: return "Q4"
    if "04/01" in date_path: return "Q2"
    if "01/01" in date_path: return "Q1"
    return "Q3"

def list_mosaic_folders(date_path: str, mgrs_filter: Optional[str] = None, cache_dir: Optional[str] = None) -> List[str]:
    """Retrieve list of S3 folders for a given date, optionally filtered by one or more MGRS tiles."""
    quarter = get_quarter(date_path)
    year = date_path.split('/')[0]

    if mgrs_filter:
        mgrs_list = [m.strip() for m in mgrs_filter.split(',')]
        all_found_folders = []
        
        for mgrs in mgrs_list:
            # Optimization: Check local cache first
            if cache_dir and os.path.exists(os.path.join(cache_dir, f"{mgrs}_0_0_B04.tif")):
                for x, y in [(0,0), (1,0), (0,1), (1,1)]:
                    if os.path.exists(os.path.join(cache_dir, f"{mgrs}_{x}_{y}_B04.tif")):
                        all_found_folders.append(f"Sentinel-2_mosaic_{year}_{quarter}_{mgrs}_{x}_{y}")
                continue

            # Standard grid indices for Sentinel-2 mosaic sub-tiles
            potential_folders = [f"Sentinel-2_mosaic_{year}_{quarter}_{mgrs}_{x}_{y}" for x, y in [(0,0), (1,0), (0,1), (1,1)]]
            for folder in potential_folders:
                s3_path = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}/B04.tif"
                cmd = ["aws", "--endpoint-url", "https://eodata.dataspace.copernicus.eu", "--profile", "cdse", "s3", "ls", s3_path]
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    all_found_folders.append(folder)

        if not all_found_folders:
            raise RuntimeError(f"No folders found for MGRS {mgrs_filter} at {date_path}")
        return sorted(all_found_folders)

    # Full bucket listing for global mode
    s3_prefix = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/"
    cmd = ["aws", "--endpoint-url", "https://eodata.dataspace.copernicus.eu", "--profile", "cdse", "s3", "ls", s3_prefix]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    folders = []
    for line in result.stdout.splitlines():
        if "PRE Sentinel-2_mosaic" in line:
            folder = line.strip().split("PRE ")[1].strip().rstrip('/')
            folders.append(folder)
    return sorted(folders)

def get_tile_paths(folder_name: str, date_path: str, cache_dir: Optional[str] = None) -> Tuple[Dict[str, str], int, int]:
    """Construct local or S3 paths for RGB bands of a specific folder. Returns (paths, cached_count, downloaded_count)"""
    if folder_name.startswith("Sentinel-2_mosaic_"):
        # Extract MGRS_X_Y suffix for clean cache naming
        cache_prefix = "_".join(folder_name.split('_')[4:])
    else:
        cache_prefix = folder_name

    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder_name}"
    bands = {'B04': 'red', 'B03': 'green', 'B02': 'blue'}
    paths = {}
    cached = 0
    downloaded = 0

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for band_id, color_name in bands.items():
            local_path = os.path.join(cache_dir, f"{cache_prefix}_{band_id}.tif")
            if os.path.exists(local_path):
                paths[color_name] = local_path
                cached += 1
                continue

            s3_path = f"{base_s3}/{band_id}.tif"
            print(f"Downloading {s3_path} to {local_path}...")
            src_ds = gdal.Open(s3_path)
            if src_ds is None: raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName('GTiff').CreateCopy(local_path, src_ds, callback=gdal.TermProgress_nocb)
            paths[color_name] = local_path
            downloaded += 1
    else:
        for band_id, color_name in bands.items():
            paths[color_name] = f"{base_s3}/{band_id}.tif"
            downloaded += 1

    return paths, cached, downloaded

# --- Processing Layer (GDAL Manipulation) ---

def create_rgb_vrt(band_paths: Dict[str, str], output_vrt: str) -> None:
    """Create a virtual RGB stack from separate R, G, B files."""
    ordered_bands = [band_paths['red'], band_paths['green'], band_paths['blue']]
    gdal.BuildVRT(output_vrt, ordered_bands, separate=True, srcNodata=-32768, VRTNodata=-32768)
    tiler.apply_rgb_color_interpretation_to_vrt(output_vrt)


def prepare_folder_source(
    folder_name: str,
    date_path: str,
    cache_dir: Optional[str],
    unique_id: str,
    download_only: bool,
) -> Tuple[str, Optional[str], int, int, Optional[str]]:
    """Prepare one folder's source paths and optional RGB VRT."""
    try:
        paths, cached, downloaded = get_tile_paths(folder_name, date_path, cache_dir)
        tile_vrt = None
        if not download_only:
            tile_vrt = tiler.make_step_vrt_path(1, f"tile_{folder_name}", unique_id)
            create_rgb_vrt(paths, tile_vrt)
        return folder_name, tile_vrt, cached, downloaded, None
    except Exception as exc:
        return folder_name, None, 0, 0, str(exc)


def prepare_group_sources(
    folders: List[str],
    date_path: str,
    cache_dir: Optional[str],
    unique_id: str,
    download_only: bool,
    max_workers: int,
) -> Tuple[List[str], int, int]:
    """Prepare a group of folders, parallelizing per-folder I/O when helpful."""
    if not folders:
        return [], 0, 0

    worker_count = max(1, min(max_workers, len(folders)))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda folder: prepare_folder_source(folder, date_path, cache_dir, unique_id, download_only),
                    folders,
                )
            )
    else:
        results = [
            prepare_folder_source(folder, date_path, cache_dir, unique_id, download_only)
            for folder in folders
        ]

    tile_vrts: List[str] = []
    total_cached = 0
    total_downloaded = 0
    for folder_name, tile_vrt, cached, downloaded, error in results:
        if error is not None:
            print(f"Warning: Skipping {folder_name}: {error}")
            continue
        total_cached += cached
        total_downloaded += downloaded
        if tile_vrt is not None:
            tile_vrts.append(tile_vrt)

    return tile_vrts, total_cached, total_downloaded

def merge_date_vrts(output_vrt: str, input_vrts: List[str]) -> None:
    """Merge multiple date VRTs using a mean pixel function to handle overlaps."""
    gdal.BuildVRT(output_vrt, input_vrts, srcNodata=-32768, VRTNodata=-32768)

    tree = ET.parse(output_vrt)
    root = tree.getroot()
    nodata_val = "-32768"
    num_bands = len(input_vrts)
    
    # Pixel function expression for averaging valid pixels
    counts = " + ".join([f"(B{i+1} != {nodata_val})" for i in range(num_bands)])
    sums = " + ".join([f"(B{i+1} != {nodata_val} ? B{i+1} : 0)" for i in range(num_bands)])
    expr = f"( ({counts}) > 0 ) ? ( ({sums}) ) / ({counts}) : {nodata_val}"

    for band in root.findall("VRTRasterBand"):
        band.set("subClass", "VRTDerivedRasterBand")
        band_index = int(band.get("band", "0"))
        if 1 <= band_index <= 3:
            color_interp = band.find("ColorInterp")
            if color_interp is None:
                color_interp = ET.Element("ColorInterp")
                band.insert(0, color_interp)
            color_interp.text = tiler.RGB_COLOR_INTERPRETATIONS[band_index - 1]
        ET.SubElement(band, "PixelFunctionType").text = "expression"
        ET.SubElement(band, "PixelFunctionArguments").set("expression", expr)
        ET.SubElement(band, "NoDataValue").text = nodata_val

    tree.write(output_vrt, encoding="UTF-8", xml_declaration=True)

def histogram_bucket_index(value: float, histogram_min: float, histogram_max: float, buckets: int) -> Optional[int]:
    """Map a histogram value to its bucket index."""
    if value < histogram_min or value > histogram_max:
        return None
    if value == histogram_max:
        return buckets - 1
    bucket_width = (histogram_max - histogram_min) / buckets
    return int((value - histogram_min) / bucket_width)


def calculate_scaling_params(
    dataset: gdal.Dataset,
    shared: bool = True,
    max_percentile: float = AUTO_SCALE_MAX_PERCENTILE,
) -> List[List[float]]:
    """Determine min/max values for 8-bit scaling using a 0..max_percentile stretch."""
    band_stats = []
    bucket_width = (HISTOGRAM_MAX - HISTOGRAM_MIN) / HISTOGRAM_BUCKETS
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        hist = list(
            band.GetHistogram(
                min=HISTOGRAM_MIN,
                max=HISTOGRAM_MAX,
                buckets=HISTOGRAM_BUCKETS,
                approx_ok=True,
            )
        )
        nodata_value = band.GetNoDataValue()
        if nodata_value is not None:
            nodata_bucket = histogram_bucket_index(nodata_value, HISTOGRAM_MIN, HISTOGRAM_MAX, HISTOGRAM_BUCKETS)
            if nodata_bucket is not None:
                hist[nodata_bucket] = 0
        total_pixels = sum(hist)

        if total_pixels == 0:
            band_stats.append((0.0, 4000.0))
            continue

        high_thresh = total_pixels * (max_percentile / 100.0)

        accum, high_val = 0, 4000.0
        for idx, count in enumerate(hist):
            accum += count
            if accum >= high_thresh:
                high_val = HISTOGRAM_MIN + (idx * bucket_width)
                break
        band_stats.append((0.0, high_val))

    if shared:
        global_min = min(s[0] for s in band_stats)
        global_max = max(s[1] for s in band_stats)
        return [[global_min, global_max, 0, 255]] * dataset.RasterCount

    return [[s[0], s[1], 0, 255] for s in band_stats]

def run_warp(input_vrt: str, output_path: str, materialize_geotiff: bool = False) -> None:
    """Reproject to Web Mercator (EPSG:3857) for tiling."""
    creation_options = None
    output_format = "VRT"
    if materialize_geotiff:
        output_format = "GTiff"
        creation_options = [
            "COMPRESS=ZSTD",
            "ZSTD_LEVEL=3",
            "PREDICTOR=2",
            "TILED=YES",
        ]

    warp_options = gdal.WarpOptions(
        format=output_format,
        creationOptions=creation_options,
        dstSRS="EPSG:3857",
        resampleAlg="cubic",
        srcNodata=-32768,
        dstNodata=0,
        callback=gdal.TermProgress_nocb,
    )
    gdal.Warp(output_path, input_vrt, options=warp_options)

# --- Execution Layer (High-level Workflows) ---

def process_date(
    date: str, args: argparse.Namespace, unique_id: str, land_tiles: Optional[Set[str]]
) -> Tuple[List[str], List[str]]:
    """Discover folders and create individual tile VRTs for a specific date."""
    date_cache = os.path.join(args.cache, date.replace("/", "-")) if args.cache else None
    
    # Use mgrs_filter if it's not a global run OR if specific tiles are provided
    mgrs_filter = args.mgrs if not args.all_tiles or (args.mgrs and args.mgrs != "31TDF") else None
    folders = list_mosaic_folders(date, mgrs_filter=mgrs_filter, cache_dir=date_cache)
    
    if args.all_tiles and land_tiles:
        folders = [f for f in folders if f.split('_')[4] in land_tiles]
        print(f"Filtered to {len(folders)} land tiles for {date}.")

    if args.download_only and not args.cache:
        print("Error: --download-only requires --cache")
        sys.exit(1)

    tile_vrts: List[str] = []
    total_cached = 0
    total_downloaded = 0
    # Always group VRTs to avoid "too many open files"
    group_size, group_vrts = 50, []
    for i in range(0, len(folders), group_size):
        chunk = folders[i:i + group_size]
        chunk_vrt = tiler.make_step_vrt_path(2, f"group_{date.replace('/', '-')}_{i // group_size}", unique_id)
        chunk_tiles, chunk_cached, chunk_downloaded = prepare_group_sources(
            chunk,
            date,
            date_cache,
            unique_id,
            args.download_only,
            args.processes,
        )
        total_cached += chunk_cached
        total_downloaded += chunk_downloaded
        tile_vrts.extend(chunk_tiles)
        
        if not args.download_only and chunk_tiles:
            gdal.BuildVRT(chunk_vrt, chunk_tiles, srcNodata=-32768, VRTNodata=-32768)
            tiler.apply_rgb_color_interpretation_to_vrt(chunk_vrt)
            group_vrts.append(chunk_vrt)
    
    print(f"Date {date} summary: {total_cached} bands cached, {total_downloaded} bands downloaded.")
    
    if not args.download_only and group_vrts:
        date_vrt = tiler.make_step_vrt_path(3, f"mosaic_{date.replace('/', '-')}", unique_id)
        gdal.BuildVRT(date_vrt, group_vrts, srcNodata=-32768, VRTNodata=-32768)
        tiler.apply_rgb_color_interpretation_to_vrt(date_vrt)
        return [date_vrt], tile_vrts + group_vrts

    return [], tile_vrts

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE.")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01,2025/01/01", help="Mosaic date(s), comma-separated")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg", "png", "png8"], default="webp", help="Tile format")
    parser.add_argument("--quality", type=int, default=74, help="Quality (0-100)")
    parser.add_argument("--resample-alg", default="lanczos", choices=["bilinear", "average", "gauss", "lanczos"])
    parser.add_argument("--chunk-zoom", type=int, default=4, help="Zoom level to chunk the processing at (for resumability)")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes for tiling")
    parser.add_argument("--stats-min", type=float, help="Hardcoded source min for step 6; provide with --stats-max")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source max for step 6; provide with --stats-min")
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument("--download-only", action="store_true", help="Download only")
    parser.add_argument("--land-only", help="Path to land tile list txt")
    parser.add_argument("--vrt", action="store_true", help="Write the final inspection VRT in .temp and skip MBTiles/PMTiles output")
    parser.add_argument("--step5", action="store_true", help="Materialize step 5 as a compressed tiled GeoTIFF instead of a VRT")
    args = parser.parse_args()

    if (args.stats_min is None) != (args.stats_max is None):
        parser.error("--stats-min and --stats-max must be provided together")

    opts = ProcessingOptions(
        format=args.format, quality=args.quality, resample_alg=args.resample_alg,
        chunk_zoom=args.chunk_zoom, processes=args.processes,
        blocksize=args.blocksize, name="Sentinel-2 Mosaic", description="Copernicus Sentinel data",
        vrt=args.vrt, step5=args.step5,
        stats_min=args.stats_min, stats_max=args.stats_max
    )

    setup_gdal_cdse()
    os.makedirs(".temp", exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    temp_vrt = tiler.make_step_vrt_path(4, "master", unique_id)
    temp_warped_vrt = (
        f".temp/step5_warped_3857_{unique_id}.tif"
        if args.step5
        else tiler.make_step_vrt_path(5, "warped_3857", unique_id)
    )
    temp_mbtiles = f".temp/tiles_{unique_id}.mbtiles"
    
    land_tiles = None
    if args.land_only and os.path.exists(args.land_only):
        with open(args.land_only, 'r') as land_file:
            land_tiles = {line.strip() for line in land_file if line.strip()}

    cleanup_list: List[str] = []
    tiling_artifacts: Optional[tiler.TilingArtifacts] = None
    try:
        all_date_vrts: List[str] = []
        for d in [date.strip() for date in args.date.split(",")]:
            date_vrts, tile_vrts = process_date(d, args, unique_id, land_tiles)
            all_date_vrts.extend(date_vrts)
            cleanup_list.extend(tile_vrts)
        
        if args.download_only: return

        if not all_date_vrts:
            print("Error: No data found.")
            sys.exit(1)

        print("Building master VRT...")
        if len(all_date_vrts) > 1: merge_date_vrts(temp_vrt, all_date_vrts)
        else:
            gdal.BuildVRT(temp_vrt, all_date_vrts, srcNodata=-32768, VRTNodata=-32768)
            tiler.apply_rgb_color_interpretation_to_vrt(temp_vrt)
        cleanup_list.extend(all_date_vrts)

        print("Reprojecting...")
        run_warp(temp_vrt, temp_warped_vrt, materialize_geotiff=args.step5)
        if not args.step5:
            tiler.apply_rgb_color_interpretation_to_vrt(temp_warped_vrt)

        # Scaling setup
        ds = gdal.Open(temp_warped_vrt)
        if ds is None:
            raise RuntimeError(f"Could not open warped VRT {temp_warped_vrt}")
        if opts.stats_min is not None and opts.stats_max is not None:
            scale_params = [[opts.stats_min, opts.stats_max, 0, 255]] * ds.RasterCount
        else:
            scale_params = calculate_scaling_params(ds)
        ds = None

        stage_label = "step 6 inspection VRT" if args.vrt else f"MBTiles ({args.format}) via chunking"
        print(f"Generating {stage_label}...")
        tiling_artifacts = tiler.run_tiling(
            temp_warped_vrt,
            temp_mbtiles,
            args.format,
            scale_params,
            vars(opts) | {"unique_id": unique_id},
        )

        if args.vrt:
            print(f"Success! {tiling_artifacts.final_vrt}")
            return

        print("Converting to PMTiles...")
        subprocess.run(["pmtiles", "convert", temp_mbtiles, args.output], check=True)
        print(f"Success! {args.output}")

    finally:
        cleanup_candidates = [temp_vrt, temp_warped_vrt, temp_mbtiles, *cleanup_list]
        if tiling_artifacts is not None:
            cleanup_candidates.extend(tiling_artifacts.cleanup_paths)
        preserved_paths = {
            path for path in cleanup_candidates
            if args.vrt and (path.endswith(".vrt") or path.endswith(".tif"))
        }
        for f in cleanup_candidates:
            if f in preserved_paths:
                continue
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
