#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from osgeo import gdal

from tiler import run_gdal2tiles

# Setup GDAL exceptions
gdal.UseExceptions()

@dataclass
class ProcessingOptions:
    """Container for tile generation settings."""
    format: str
    quality: int
    resample_alg: str
    exponent: float
    blocksize: int
    name: str
    description: str
    stats_min: Optional[float] = None
    stats_max: Optional[float] = None

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

def get_tile_paths(folder_name: str, date_path: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Construct local or S3 paths for RGB bands of a specific folder."""
    if folder_name.startswith("Sentinel-2_mosaic_"):
        # Extract MGRS_X_Y suffix for clean cache naming
        cache_prefix = "_".join(folder_name.split('_')[4:])
    else:
        cache_prefix = folder_name

    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder_name}"
    bands = {'B04': 'red', 'B03': 'green', 'B02': 'blue'}
    paths = {}

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for band_id, color_name in bands.items():
            local_path = os.path.join(cache_dir, f"{cache_prefix}_{band_id}.tif")
            if os.path.exists(local_path):
                paths[color_name] = local_path
                continue

            s3_path = f"{base_s3}/{band_id}.tif"
            print(f"Downloading {s3_path} to {local_path}...")
            src_ds = gdal.Open(s3_path)
            if src_ds is None: raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName('GTiff').CreateCopy(local_path, src_ds, callback=gdal.TermProgress_nocb)
            paths[color_name] = local_path
    else:
        for band_id, color_name in bands.items():
            paths[color_name] = f"{base_s3}/{band_id}.tif"

    return paths

# --- Processing Layer (GDAL Manipulation) ---

def create_rgb_vrt(band_paths: Dict[str, str], output_vrt: str) -> None:
    """Create a virtual RGB stack from separate R, G, B files."""
    ordered_bands = [band_paths['red'], band_paths['green'], band_paths['blue']]
    gdal.BuildVRT(output_vrt, ordered_bands, separate=True, srcNodata=-32768, VRTNodata=-32768)

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
        ET.SubElement(band, "PixelFunctionType").text = "expression"
        ET.SubElement(band, "PixelFunctionArguments").set("expression", expr)
        ET.SubElement(band, "NoDataValue").text = nodata_val

    tree.write(output_vrt, encoding="UTF-8", xml_declaration=True)

def calculate_scaling_params(dataset: gdal.Dataset, shared: bool = True) -> List[List[float]]:
    """Determine min/max values for 8-bit scaling using percentiles (2% and 98%)."""
    band_stats = []
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        hist = band.GetHistogram(min=0.0, max=40000.0, buckets=4000, approx_ok=True)
        total_pixels = sum(hist)

        if total_pixels == 0:
            band_stats.append((0.0, 4000.0))
            continue

        low_thresh = total_pixels * 0.02
        high_thresh = total_pixels * 0.98
        
        accum, low_val, high_val = 0, 0.0, 4000.0
        for idx, count in enumerate(hist):
            accum += count
            if low_val == 0.0 and accum >= low_thresh:
                low_val = idx * 10.0
            if accum >= high_thresh:
                high_val = idx * 10.0
                break
        band_stats.append((low_val, high_val))

    if shared:
        global_min = min(s[0] for s in band_stats)
        global_max = max(s[1] for s in band_stats)
        return [[global_min, global_max, 0, 255]] * dataset.RasterCount

    return [[s[0], s[1], 0, 255] for s in band_stats]

def run_warp(input_vrt: str, output_vrt: str) -> None:
    """Reproject to Web Mercator (EPSG:3857) for tiling."""
    warp_options = gdal.WarpOptions(
        format="VRT", dstSRS="EPSG:3857", resampleAlg="cubic",
        srcNodata=-32768, dstNodata=0, callback=gdal.TermProgress_nocb
    )
    gdal.Warp(output_vrt, input_vrt, options=warp_options)

def run_translate_to_mbtiles(input_vrt: str, output_file: str, opts: ProcessingOptions, scale_params: List[List[float]]) -> None:
    """Convert VRT to MBTiles using native GDAL Translate."""
    driver_map = {"jpg": "JPEG", "png8": "PNG", "png": "PNG", "webp": "WEBP"}
    tile_driver = driver_map.get(opts.format.lower(), opts.format.upper())

    exponents = [opts.exponent] * 3 if opts.exponent != 0 else None
    
    translate_options = gdal.TranslateOptions(
        format="MBTiles", outputType=gdal.GDT_Byte, scaleParams=scale_params,
        exponents=exponents, callback=gdal.TermProgress_nocb,
        metadataOptions=[f"format={opts.format.lower()}", f"name={opts.name}", f"description={opts.description}"],
        creationOptions=[
            f"NAME={opts.name}", f"DESCRIPTION={opts.description}", "TYPE=baselayer",
            f"TILE_FORMAT={tile_driver}", f"QUALITY={opts.quality}",
            f"RESAMPLING={opts.resample_alg if opts.resample_alg != 'gauss' else 'bilinear'}",
            f"BLOCKSIZE={opts.blocksize}", "ZOOM_LEVEL_STRATEGY=UPPER"
        ]
    )
    
    if opts.format == "webp":
        gdal.SetConfigOption('WEBP_LEVEL', str(opts.quality))

    gdal.Translate(output_file, input_vrt, options=translate_options)

# --- Execution Layer (High-level Workflows) ---

def process_date(date: str, args: argparse.Namespace, unique_id: str, land_tiles: Optional[Set[str]], use_tiler: bool) -> List[str]:
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

    tile_vrts = []
    if use_tiler:
        # Group VRTs to avoid "too many open files"
        group_size, group_vrts = 50, []
        for i in range(0, len(folders), group_size):
            chunk = folders[i:i + group_size]
            chunk_vrt = f".temp/group_{date.replace('/', '-')}_{i // group_size}_{unique_id}.vrt"
            chunk_tiles = []
            for folder in chunk:
                try:
                    paths = get_tile_paths(folder, date, date_cache)
                    if args.download_only: continue
                    t_vrt = f".temp/tile_{folder}_{unique_id}.vrt"
                    create_rgb_vrt(paths, t_vrt)
                    chunk_tiles.append(t_vrt)
                    tile_vrts.append(t_vrt)
                except Exception as e: print(f"Warning: Skipping {folder}: {e}")
            
            if not args.download_only and chunk_tiles:
                gdal.BuildVRT(chunk_vrt, chunk_tiles, srcNodata=-32768, VRTNodata=-32768)
                group_vrts.append(chunk_vrt)
        
        if not args.download_only and group_vrts:
            date_vrt = f".temp/mosaic_{date.replace('/', '-')}_{unique_id}.vrt"
            gdal.BuildVRT(date_vrt, group_vrts, srcNodata=-32768, VRTNodata=-32768)
            return [date_vrt], tile_vrts + group_vrts
    else:
        date_tiles = []
        for folder in folders:
            try:
                paths = get_tile_paths(folder, date, date_cache)
                if args.download_only: continue
                t_vrt = f".temp/tile_{folder}_{unique_id}.vrt"
                create_rgb_vrt(paths, t_vrt)
                date_tiles.append(t_vrt)
                tile_vrts.append(t_vrt)
            except Exception as e: print(f"Warning: Skipping {folder}: {e}")
        
        if not args.download_only and date_tiles:
            date_vrt = f".temp/mosaic_{date.replace('/', '-')}_{unique_id}.vrt"
            gdal.BuildVRT(date_vrt, date_tiles, srcNodata=-32768, VRTNodata=-32768)
            return [date_vrt], tile_vrts

    return [], tile_vrts

def main():
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE.")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01,2025/01/01", help="Mosaic date(s), comma-separated")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg", "png", "png8"], default="webp", help="Tile format")
    parser.add_argument("--quality", type=int, default=74, help="Quality (0-100)")
    parser.add_argument("--resample-alg", default="lanczos", choices=["bilinear", "average", "gauss", "lanczos"])
    parser.add_argument("--exponent", type=float, default=0.4, help="Power law scaling exponent")
    parser.add_argument("--stats-min", type=float, default=0, help="Hardcoded source min")
    parser.add_argument("--stats-max", type=float, default=9000, help="Hardcoded source max")
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument("--download-only", action="store_true", help="Download only")
    parser.add_argument("--land-only", help="Path to land tile list txt")
    args = parser.parse_args()

    opts = ProcessingOptions(
        format=args.format, quality=args.quality, resample_alg=args.resample_alg,
        exponent=args.exponent,
        blocksize=args.blocksize, name="Sentinel-2 Mosaic", description="Copernicus Sentinel data",
        stats_min=args.stats_min, stats_max=args.stats_max
    )

    setup_gdal_cdse()
    os.makedirs(".temp", exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    temp_vrt, temp_warped_vrt, temp_mbtiles = f".temp/master_{unique_id}.vrt", f".temp/warped_{unique_id}.vrt", f".temp/tiles_{unique_id}.mbtiles"
    
    land_tiles = None
    if args.land_only and os.path.exists(args.land_only):
        with open(args.land_only, 'r') as f:
            land_tiles = {line.strip() for line in f if line.strip()}

    use_tiler = args.all_tiles or "," in args.mgrs

    cleanup_list = []
    try:
        all_date_vrts = []
        for d in [date.strip() for date in args.date.split(",")]:
            date_vrts, tile_vrts = process_date(d, args, unique_id, land_tiles, use_tiler)
            all_date_vrts.extend(date_vrts)
            cleanup_list.extend(tile_vrts)
        
        if args.download_only: return

        if not all_date_vrts:
            print("Error: No data found.")
            sys.exit(1)

        print("Building master VRT...")
        if len(all_date_vrts) > 1: merge_date_vrts(temp_vrt, all_date_vrts)
        else: gdal.BuildVRT(temp_vrt, all_date_vrts, srcNodata=-32768, VRTNodata=-32768)
        cleanup_list.extend(all_date_vrts)

        print("Reprojecting...")
        run_warp(temp_vrt, temp_warped_vrt)

        # Scaling setup
        if opts.stats_min is not None and opts.stats_max is not None:
            ds = gdal.Open(temp_warped_vrt)
            scale_params = [[opts.stats_min, opts.stats_max, 0, 255]] * ds.RasterCount
        else:
            ds = gdal.Open(temp_warped_vrt)
            scale_params = calculate_scaling_params(ds)
        ds = None

        print(f"Generating MBTiles ({args.format})...")
        if use_tiler:
            run_gdal2tiles(temp_warped_vrt, temp_mbtiles, args.format, scale_params, [opts.exponent]*3 if opts.exponent!=0 else None, args.resample_alg, vars(args) | {"name": opts.name, "description": opts.description})
        else:
            run_translate_to_mbtiles(temp_warped_vrt, temp_mbtiles, opts, scale_params)
            print("Building overviews...")
            gdaladdo_cmd = ["gdaladdo", "-r", args.resample_alg, "--config", "GDAL_NUM_THREADS", "ALL_CPUS", temp_mbtiles]
            subprocess.run(gdaladdo_cmd, check=True)

        print("Converting to PMTiles...")
        subprocess.run(["pmtiles", "convert", temp_mbtiles, args.output], check=True)
        print(f"Success! {args.output}")

    finally:
        for f in [temp_vrt, temp_warped_vrt, temp_mbtiles] + cleanup_list:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
