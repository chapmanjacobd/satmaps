#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import uuid
from osgeo import gdal

from typing import Dict, List, Optional

# Setup GDAL exceptions
gdal.UseExceptions()

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

def get_tile_paths(mgrs_tile: Optional[str], date_path: str = "2025/07/01", cache_dir: Optional[str] = None, folder_name: Optional[str] = None) -> Dict[str, str]:
    """Construct S3 paths for a given MGRS tile or specific folder."""
    if folder_name:
        folder = folder_name
        # Extract a shorter name for caching if folder_name is the full Sentinel-2_mosaic...
        if folder.startswith("Sentinel-2_mosaic_"):
            # Sentinel-2_mosaic_2025_Q3_31TDF_0_0 -> 31TDF_0_0
            parts = folder.split('_')
            cache_prefix = "_".join(parts[4:])
        else:
            cache_prefix = folder
    else:
        # Fallback for when list_all_mosaic_folders isn't used (though it should be)
        q = get_quarter(date_path)
        folder = f"Sentinel-2_mosaic_2025_{q}_{mgrs_tile}_0_0"
        cache_prefix = f"{mgrs_tile}_0_0"

    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}"

    bands = {'B04': 'red', 'B03': 'green', 'B02': 'blue'}
    paths = {}

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for b, name in bands.items():
            local_path = os.path.join(cache_dir, f"{cache_prefix}_{b}.tif")
            if os.path.exists(local_path):
                print(f"Warning: Local file {local_path} already exists, skipping download.")
                paths[name] = local_path
                continue

            s3_path = f"{base_s3}/{b}.tif"
            print(f"Downloading {s3_path} to {local_path}...")
            try:
                src_ds = gdal.Open(s3_path)
                if src_ds is None:
                    raise RuntimeError(f"Could not open {s3_path}")
                gdal.GetDriverByName('GTiff').CreateCopy(local_path, src_ds, callback=gdal.TermProgress_nocb)
                paths[name] = local_path
            except Exception as e:
                print(f"Error downloading {s3_path}: {e}")
                raise
    else:
        for b, name in bands.items():
            paths[name] = f"{base_s3}/{b}.tif"

    return paths

def get_quarter(date_path: str) -> str:
    """Determine quarter from date_path."""
    if "07/01" in date_path: return "Q3"
    if "10/01" in date_path: return "Q4"
    if "04/01" in date_path: return "Q2"
    if "01/01" in date_path: return "Q1"
    return "Q3"

def create_rgb_vrt(paths: Dict[str, str], output_vrt: str) -> None:
    """Create a virtual RGB stack from individual band paths."""
    # Stack bands in order: Red (B04), Green (B03), Blue (B02)
    band_list = [paths['red'], paths['green'], paths['blue']]
    gdal.BuildVRT(output_vrt, band_list, separate=True)

def list_all_mosaic_folders(date_path: str = "2025/07/01", mgrs_filter: Optional[str] = None, cache_dir: Optional[str] = None) -> List[str]:
    q = get_quarter(date_path)
    
    if mgrs_filter:
        # User requested optimization: skip remote check if MGRS_0_0 exists locally
        if cache_dir:
            if os.path.exists(os.path.join(cache_dir, f"{mgrs_filter}_0_0_B04.tif")):
                found_local = []
                for x, y in [(0,0), (1,0), (0,1), (1,1)]:
                    if os.path.exists(os.path.join(cache_dir, f"{mgrs_filter}_{x}_{y}_B04.tif")):
                        found_local.append(f"Sentinel-2_mosaic_2025_{q}_{mgrs_filter}_{x}_{y}")
                return sorted(found_local)

        # Optimized path: instead of listing the whole S3 bucket, check for expected sub-tiles
        # Grid segment indices are usually 0_0, 1_0, 0_1, 1_1
        potential_folders = []
        for x, y in [(0,0), (1,0), (0,1), (1,1)]:
            potential_folders.append(f"Sentinel-2_mosaic_2025_{q}_{mgrs_filter}_{x}_{y}")
        
        found_folders = []
        for folder in potential_folders:
            # Check if B04.tif exists as a proxy for folder existence
            s3_path = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}/B04.tif"
            cmd = ["aws", "--endpoint-url", "https://eodata.dataspace.copernicus.eu", "--profile", "cdse", "s3", "ls", s3_path]
            # Use run instead of check=True to handle non-zero exits (file not found)
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                found_folders.append(folder)
        
        if not found_folders:
            raise RuntimeError(f"Error: No folders found for MGRS {mgrs_filter} at {date_path}")
            
        return sorted(found_folders)
    
    # Global mode still needs the full list
    s3_prefix = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/"
    cmd = ["aws", "--endpoint-url", "https://eodata.dataspace.copernicus.eu", "--profile", "cdse", "s3", "ls", s3_prefix]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    folders = []
    for line in result.stdout.splitlines():
        if "PRE Sentinel-2_mosaic" in line:
            folder = line.strip().split("PRE ")[1].strip()
            if folder.endswith('/'): folder = folder[:-1]
            folders.append(folder)
                
    return sorted(folders)

def get_percentiles(ds: gdal.Dataset, low: float = 2.0, high: float = 98.0) -> List[List[float]]:
    """Calculate the low and high percentiles for each band in the dataset."""
    scale_params = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        # Use a 1000-bucket histogram for range 0-10000 (Sentinel-2 typical reflectance range)
        # approx_ok=True for speed
        hist = band.GetHistogram(min=0.0, max=10000.0, buckets=1000, approx_ok=True)
        total = sum(hist)
        
        if total == 0:
            scale_params.append([0, 4000, 0, 255])
            continue
            
        low_threshold = total * (low / 100.0)
        high_threshold = total * (high / 100.0)
        
        accum = 0
        low_val = 0.0
        high_val = 4000.0
        
        for idx, count in enumerate(hist):
            accum += count
            if low_val == 0.0 and accum >= low_threshold:
                low_val = idx * 10.0 # (idx / 1000) * 10000
            if accum >= high_threshold:
                high_val = idx * 10.0
                break
        
        # Ensure some contrast even if the range is tight
        if high_val <= low_val:
            high_val = low_val + 100.0
            
        scale_params.append([low_val, high_val, 0, 255])
    
    return scale_params

def main():
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE.")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID (default: 31TDF for Barcelona. Other interesting ones: 60HTB, 07VEK, 50HQJ, 49QGF, 45RVL, 32TQK, 12SUD, 40RCN, 28QCH)")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01", help="Mosaic date path (default: 2025/07/01)")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg", "png", "png8"], default="webp", help="Tile format (default: webp)")
    parser.add_argument("--quality", type=int, default=74, help="WebP/JPEG quality (default: 74)")
    parser.add_argument("--resample-alg", default="bilinear", choices=["bilinear", "average", "gauss", "lanczos"], help="Resampling method (default: bilinear)")
    parser.add_argument("--exponent", type=float, default=0.6, help="Exponent for power law scaling (default: 0.6)")
    parser.add_argument("--blocksize", type=int, default=512, help="MBTiles block size (default: 256)")
    parser.add_argument("--minzoom", type=int, default=0)
    parser.add_argument("--maxzoom", type=int, default=14)
    parser.add_argument("--cache", help="Local directory to cache downloaded TIFs")
    parser.add_argument("--download-only", action="store_true", help="Only download files to cache, don't process")

    args = parser.parse_args()

    setup_gdal_cdse()

    # Metadata
    tileset_name = "Internet in a Box Maps - Sentinel-2"
    tileset_desc = f"Contains modified Copernicus Sentinel data {args.date[:4]}"

    # Unique temp files to avoid race conditions in parallel mode
    unique_id = uuid.uuid4().hex[:8]
    temp_vrt = f"temp_{unique_id}.vrt"
    temp_mbtiles = f"temp_{unique_id}.mbtiles"
    temp_warped_vrt = f"temp_warped_{unique_id}.vrt"

    # Use cubic for reprojection as it's better for that step, regardless of the downsampling algorithm chosen
    warp_resample = "cubic"

    tile_vrts = []
    try:
        if args.all_tiles:
            print(f"Listing all folders for global mosaic ({args.date})...")
            all_folders = list_all_mosaic_folders(args.date, cache_dir=args.cache)
            print(f"Found {len(all_folders)} folders.")

            if args.download_only and not args.cache:
                print("Error: --download-only requires --cache")
                sys.exit(1)

            # Use Grouped VRTs to avoid too many open files
            group_size = 50
            group_vrts = []
            for i in range(0, len(all_folders), group_size):
                chunk = all_folders[i:i + group_size]
                chunk_vrt_name = f"group_{i // group_size}_{unique_id}.vrt"

                print(f"[{i+1}/{len(all_folders)}] Handling group {i // group_size}...")
                chunk_tile_vrts = []
                for folder in chunk:
                    try:
                        paths = get_tile_paths(None, args.date, args.cache, folder_name=folder)
                        if args.download_only: continue

                        tile_vrt = f"tile_{folder}_{unique_id}.vrt"
                        create_rgb_vrt(paths, tile_vrt)
                        chunk_tile_vrts.append(tile_vrt)
                        tile_vrts.append(tile_vrt)
                    except Exception as e:
                        print(f"Warning: Could not process folder {folder}: {e}")

                if args.download_only: continue

                if chunk_tile_vrts:
                    print(f"Building Group VRT {chunk_vrt_name}...")
                    gdal.BuildVRT(chunk_vrt_name, chunk_tile_vrts)
                    group_vrts.append(chunk_vrt_name)

            if args.download_only:
                print("Download complete.")
                return

            print("Building master VRT from groups...")
            gdal.BuildVRT(temp_vrt, group_vrts)
            # Clean up group VRTs
            for f in group_vrts:
                if os.path.exists(f): os.remove(f)
        else:
            print(f"Processing tile: {args.mgrs} (Date: {args.date})")
            try:
                folders = list_all_mosaic_folders(args.date, mgrs_filter=args.mgrs, cache_dir=args.cache)
            except RuntimeError as e:
                print(e)
                sys.exit(1)

            for folder in folders:
                try:
                    paths = get_tile_paths(None, args.date, args.cache, folder_name=folder)
                    if args.download_only: continue
                    
                    tile_vrt = f"tile_{folder}_{unique_id}.vrt"
                    create_rgb_vrt(paths, tile_vrt)
                    tile_vrts.append(tile_vrt)
                except Exception as e:
                    print(f"Warning: Could not process folder {folder}: {e}")

            if args.download_only:
                print(f"Download of {args.mgrs} complete.")
                return
            
            if not tile_vrts:
                print(f"Error: No data found for {args.mgrs}")
                sys.exit(1)

            gdal.BuildVRT(temp_vrt, tile_vrts)

        print("Step 1: Reprojecting to Web Mercator (VRT)...")
        warp_options = gdal.WarpOptions(
            format="VRT",
            dstSRS="EPSG:3857",
            resampleAlg=warp_resample,
            callback=gdal.TermProgress_nocb
        )
        gdal.Warp(temp_warped_vrt, temp_vrt, options=warp_options)

        print(f"Step 2: Generating MBTiles ({args.format}) with zoom {args.minzoom}-{args.maxzoom}...")
        
        tile_format = args.format.upper()
        if tile_format == "JPG":
            tile_format = "JPEG"
        if tile_format == "PNG8":
            tile_format = "PNG"

        # Handle resampling for Translate: gauss is not supported there
        translate_resample = args.resample_alg
        if args.resample_alg == "gauss":
            translate_resample = "bilinear"

        # Calculate percentiles (2nd and 98th) for dynamic scaling
        ds_for_stats = gdal.Open(temp_warped_vrt)
        scale_params = get_percentiles(ds_for_stats)
        ds_for_stats = None

        translate_options = gdal.TranslateOptions(
            format="MBTiles",
            outputType=gdal.GDT_Byte,
            scaleParams=scale_params,
            exponent=args.exponent,
            callback=gdal.TermProgress_nocb,
            metadataOptions=[
                f"format={tile_format.lower()}",
                f"name={tileset_name}",
                f"description={tileset_desc}"
            ],
            creationOptions=[
                f"NAME={tileset_name}",
                f"DESCRIPTION={tileset_desc}",
                "TYPE=baselayer",
                f"TILE_FORMAT={tile_format}",
                f"QUALITY={args.quality}",
                f"MINZOOM={args.minzoom}",
                f"MAXZOOM={args.maxzoom}",
                f"RESAMPLING={translate_resample}",
                f"BLOCKSIZE={args.blocksize}",
                "ZOOM_LEVEL_STRATEGY=LOWER"
            ]
        )

        if args.format == "webp":
            gdal.SetConfigOption('WEBP_LEVEL', str(args.quality))

        gdal.Translate(temp_mbtiles, temp_warped_vrt, options=translate_options)

        print("Step 2.5: Building overviews...")
        gdaladdo_cmd = [
            "gdaladdo",
            "-r", args.resample_alg,
            "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
            "--config", "GDAL_TIFF_OVR_BLOCKSIZE", str(args.blocksize)
        ]

        if args.format == "webp":
            gdaladdo_cmd.extend([
                "--config", "COMPRESS_OVERVIEW", "WEBP",
                "--config", "WEBP_LEVEL_OVERVIEW", str(args.quality)
            ])
        elif args.format == "jpg":
            gdaladdo_cmd.extend([
                "--config", "COMPRESS_OVERVIEW", "JPEG",
                "--config", "JPEG_QUALITY_OVERVIEW", str(args.quality),
                "--config", "PHOTOMETRIC_OVERVIEW", "YCBCR",
                "--config", "INTERLEAVE_OVERVIEW", "PIXEL"
            ])
        elif args.format in ["png", "png8"]:
            gdaladdo_cmd.extend([
                "--config", "COMPRESS_OVERVIEW", "PNG"
            ])

        gdaladdo_cmd.append(temp_mbtiles)
        subprocess.run(gdaladdo_cmd, check=True)

        print("Step 3: Converting MBTiles to PMTiles...")
        pmtiles_cmd = ["pmtiles", "convert", temp_mbtiles, args.output]  # --no-deduplication
        subprocess.run(pmtiles_cmd, check=True)

        print(f"Success! Created {args.output}")

    finally:
        for f in [temp_vrt, temp_warped_vrt, temp_mbtiles] + tile_vrts:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()
