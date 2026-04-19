#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from osgeo import gdal

import tiler

# Setup GDAL exceptions
gdal.UseExceptions()

AUTO_SCALE_MAX_PERCENTILE = 99.6

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

# Global cache for S3 folder listings to avoid per-tile discovery overhead
S3_FOLDER_CACHE: Dict[str, set] = {}

def list_mosaic_folders_for_tile(mgrs_tile: str, date_paths: List[str], cache_dir: str) -> List[Tuple[str, str]]:
    """Find all S3 folders for a specific MGRS sub-tile across multiple dates using the pre-populated cache."""
    mgrs_id, x, y = mgrs_tile.split('_')
    found = []
    
    for date_path in date_paths:
        quarter = "Q3" 
        if "07/01" in date_path: quarter = "Q3"
        elif "10/01" in date_path: quarter = "Q4"
        elif "04/01" in date_path: quarter = "Q2"
        elif "01/01" in date_path: quarter = "Q1"
        
        year = date_path.split('/')[0]
        folder = f"Sentinel-2_mosaic_{year}_{quarter}_{mgrs_id}_{x}_{y}"
        
        # 1. Check local cache first
        local_b04 = os.path.join(cache_dir, date_path.replace("/", "-"), f"{mgrs_id}_{x}_{y}_B04.tif")
        if os.path.exists(local_b04):
            found.append((folder, date_path))
            continue

        # 2. Check pre-populated S3 cache
        if date_path in S3_FOLDER_CACHE and folder in S3_FOLDER_CACHE[date_path]:
            found.append((folder, date_path))
            
    return found

def get_tile_paths(folder_name: str, date_path: str, cache_dir: Optional[str] = None, download: bool = False) -> Dict[str, str]:
    """Construct local or S3 paths for RGB bands. Only downloads if 'download' is True."""
    cache_prefix = "_".join(folder_name.split('_')[4:])
    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder_name}"
    bands = {'B04': 'red', 'B03': 'green', 'B02': 'blue'}
    paths = {}

    date_cache_dir = os.path.join(cache_dir, date_path.replace("/", "-")) if cache_dir else None
    for band_id, color_name in bands.items():
        local_path = os.path.join(date_cache_dir, f"{cache_prefix}_{band_id}.tif") if date_cache_dir else None
        
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
            if src_ds is None: raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName('GTiff').CreateCopy(local_path, src_ds, callback=gdal.TermProgress_nocb)
            paths[color_name] = local_path
            continue
            
        # Fallback to streaming
        paths[color_name] = f"{base_s3}/{band_id}.tif"

    return paths

# --- Processing Layer (NumPy Manipulation) ---

def process_single_tile(
    mgrs_subtile: str,
    date_paths: List[str],
    args: argparse.Namespace,
    unique_id: str
) -> Optional[str]:
    """Process a single MGRS sub-tile: fetch dates, average, tone-map, and warp to Web Mercator."""
    folders = list_mosaic_folders_for_tile(mgrs_subtile, date_paths, args.cache)
    if not folders:
        return None

    if args.download:
        for folder_name, date_path in folders:
            get_tile_paths(folder_name, date_path, args.cache, download=True)
        return None

    print(f"Processing tile {mgrs_subtile} across {len(folders)} date(s)...")
    
    # 1. Load all dates into memory
    date_stacks = []
    projection = None
    geotransform = None
    
    for folder_name, date_path in folders:
        # Default to streaming (download=False) since --download exits early
        paths = get_tile_paths(folder_name, date_path, args.cache, download=False)
        bands_data = []
        for color in ['red', 'green', 'blue']:
            ds = gdal.Open(paths[color])
            if projection is None:
                projection = ds.GetProjection()
                geotransform = ds.GetGeoTransform()
            arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            # Use NaN for nodata -32768
            arr[arr == -32768] = np.nan
            bands_data.append(arr)
        date_stacks.append(np.stack(bands_data)) # (3, H, W)

    if not date_stacks:
        return None

    # 2. Average dates (ignoring NaNs)
    full_stack = np.stack(date_stacks) # (D, 3, H, W)
    averaged = np.nanmean(full_stack, axis=0)
    
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
    
    # Apply soft-knee tone mapping
    if not args.no_soft_knee:
        toned = tiler.apply_soft_knee_numpy(normalized, 
                                           shadow_break=args.sb, 
                                           highlight_break=args.hb,
                                           shadow_slope=args.ss,
                                           mid_slope=args.ms,
                                           highlight_slope=args.hs,
                                           exposure=args.exposure)
    else:
        toned = np.clip(normalized * args.exposure, 0.0, 1.0)
    
    # Apply final grading (preview correction)
    if not args.no_grading:
        toned = tiler.apply_preview_correction_numpy(toned,
                                                    saturation=args.sat,
                                                    darken_break=args.db,
                                                    low_slope=args.ls,
                                                    gamma=args.gamma)
    
    # 4. Save to temporary Byte GeoTIFF
    temp_utm_path = f".temp/processed_{mgrs_subtile}_{unique_id}_utm.tif"
    temp_3857_path = f".temp/processed_{mgrs_subtile}_{unique_id}_3857.tif"
    
    driver = gdal.GetDriverByName("GTiff")
    byte_arr = (toned * 255).astype(np.uint8)
    
    ds_out = driver.Create(temp_utm_path, byte_arr.shape[2], byte_arr.shape[1], 3, gdal.GDT_Byte, 
                           options=["COMPRESS=ZSTD", "TILED=YES"])
    ds_out.SetProjection(projection)
    ds_out.SetGeoTransform(geotransform)
    for i in range(3):
        out_band = ds_out.GetRasterBand(i + 1)
        out_band.WriteArray(byte_arr[i])
        out_band.SetNoDataValue(0)
        out_band.SetColorInterpretation(getattr(gdal, f"GCI_{['RedBand', 'GreenBand', 'BlueBand'][i]}"))
    ds_out.FlushCache()
    ds_out = None

    # 5. Warp to Web Mercator
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        resampleAlg=args.resample_alg,
        srcNodata=0,
        dstNodata=0,
        creationOptions=["COMPRESS=ZSTD", "TILED=YES"]
    )
    gdal.Warp(temp_3857_path, temp_utm_path, options=warp_options)
    os.remove(temp_utm_path)
    
    return temp_3857_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE (NumPy Refactor).")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01,2025/01/01", help="Mosaic date(s), comma-separated")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg", "png", "png8"], default="webp", help="Tile format")
    parser.add_argument("--quality", type=int, default=74, help="Quality (0-100)")
    parser.add_argument("--resample-alg", default="lanczos", choices=["bilinear", "average", "gauss", "lanczos"])
    parser.add_argument("--chunk-zoom", type=int, default=4, help="Zoom level to chunk the processing at")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--stats-min", type=float, help="Hardcoded source min")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source max")
    
    # Tone Mapping
    parser.add_argument("--exposure", type=float, default=tiler.DEFAULT_EXPOSURE)
    parser.add_argument("--sb", "--shadow-break", type=float, default=tiler.SOFT_KNEE_SHADOW_BREAK)
    parser.add_argument("--hb", "--highlight-break", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_BREAK)
    parser.add_argument("--ss", "--shadow-slope", type=float, default=tiler.SOFT_KNEE_SHADOW_SLOPE)
    parser.add_argument("--ms", "--mid-slope", type=float, default=tiler.SOFT_KNEE_MID_SLOPE)
    parser.add_argument("--hs", "--highlight-slope", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_SLOPE)
    
    # Grading
    parser.add_argument("--gamma", type=float, default=tiler.DEFAULT_GAMMA)
    parser.add_argument("--sat", "--saturation", type=float, default=tiler.PREVIEW_SATURATION)
    parser.add_argument("--db", "--black-break", type=float, default=tiler.PREVIEW_DARKEN_BREAK)
    parser.add_argument("--ls", "--black-slope", type=float, default=tiler.PREVIEW_DARKEN_LOW_SLOPE)

    parser.add_argument("--no-soft-knee", action="store_true", help="Disable soft-knee tone mapping")
    parser.add_argument("--no-grading", action="store_true", help="Disable final grading")

    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument("--download", action="store_true", help="Download S3 tiles to local cache and exit")
    parser.add_argument("--land-only", help="Path to land tile list txt")
    parser.add_argument("--vrt", action="store_true", help="Write final VRT and skip MBTiles")
    args = parser.parse_args()

    setup_gdal_cdse()
    os.makedirs(".temp", exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    date_paths = [d.strip() for d in args.date.split(",")]

    # 1. Tile Discovery - Pre-populate S3 cache with a single listing per date
    print("Populating S3 folder cache...")
    for dp in date_paths:
        s3_base = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{dp}"
        dirs = gdal.ReadDir(s3_base)
        if dirs:
            S3_FOLDER_CACHE[dp] = set(dirs)
        else:
            print(f"Warning: Could not list folders for {dp}")

    # 2. Sub-tile Expansion
    if args.all_tiles and args.land_only:
        with open(args.land_only, 'r') as f:
            mgrs_bases = [line.strip() for line in f if line.strip()]
        subtiles = [f"{m}_{x}_{y}" for m in mgrs_bases for x in [0, 1] for y in [0, 1]]
    else:
        subtiles = [f"{args.mgrs}_{x}_{y}" for x in [0, 1] for y in [0, 1]]

    # 2. Parallel Tile Processing
    processed_tifs = []
    print(f"Starting processing for {len(subtiles)} sub-tiles...")
    
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        futures = [executor.submit(process_single_tile, st, date_paths, args, unique_id) for st in subtiles]
        for future in futures:
            res = future.result()
            if res: processed_tifs.append(res)

    if args.download:
        print("Download complete.")
        return

    if not processed_tifs:
        print("Error: No data processed.")
        sys.exit(1)

    # 3. Build Master VRT (Flat, over Web Mercator Byte TIFFs)
    master_vrt = f".temp/master_{unique_id}.vrt"
    gdal.BuildVRT(master_vrt, processed_tifs)

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
        "processes": args.processes,
        "blocksize": args.blocksize,
        "name": "Sentinel-2 Mosaic",
        "description": "Copernicus Sentinel data",
        "unique_id": unique_id
    }
    
    print("Generating MBTiles...")
    # Modified run_tiling will be simpler now
    artifacts = tiler.run_tiling_simplified(master_vrt, temp_mbtiles, tiling_opts)

    print("Converting to PMTiles...")
    subprocess.run(["pmtiles", "convert", temp_mbtiles, args.output], check=True)
    print(f"Success! {args.output}")

    # Cleanup
    for f in processed_tifs + [master_vrt, temp_mbtiles] + artifacts.cleanup_paths:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
