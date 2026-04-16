#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from osgeo import gdal

# Setup GDAL exceptions
gdal.UseExceptions()

def setup_gdal_cdse():
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

def get_tile_paths(mgrs_tile, date_path="2025/07/01", cache_dir=None):
    """Construct S3 paths for a given MGRS tile."""
    # Determine quarter from date_path
    if "07/01" in date_path:
        q = "Q3"
    elif "10/01" in date_path:
        q = "Q4"
    elif "04/01" in date_path:
        q = "Q2"
    elif "01/01" in date_path:
        q = "Q1"
    else:
        q = "Q3"

    folder = f"Sentinel-2_mosaic_2025_{q}_{mgrs_tile}_0_0"
    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}"
    
    bands = {'B04': 'red', 'B03': 'green', 'B02': 'blue'}
    paths = {}
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for b, name in bands.items():
            local_path = os.path.join(cache_dir, f"{mgrs_tile}_{b}.tif")
            if not os.path.exists(local_path):
                s3_path = f"{base_s3}/{b}.tif"
                print(f"Downloading {s3_path} to {local_path}...")
                src_ds = gdal.Open(s3_path)
                gdal.GetDriverByName('GTiff').CreateCopy(local_path, src_ds, callback=gdal.TermProgress_nocb)
            paths[name] = local_path
    else:
        for b, name in bands.items():
            paths[name] = f"{base_s3}/{b}.tif"
            
    return paths

def create_rgb_vrt(paths, output_vrt):
    # Sentinel-2 values are typically 0-10000.
    vrt_options = gdal.BuildVRTOptions(
        separate=True,
        callback=gdal.TermProgress_nocb
    )
    vrt = gdal.BuildVRT(output_vrt, [paths['red'], paths['green'], paths['blue']], options=vrt_options)
    if vrt is None:
        raise RuntimeError(f"Failed to create VRT: {output_vrt}")
    vrt.FlushCache()
    return output_vrt

def list_all_mgrs_tiles(date_path="2025/07/01"):
    """List all MGRS tiles available for the given date."""
    s3_prefix = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/"
    cmd = ["aws", "--endpoint-url", "https://eodata.dataspace.copernicus.eu", "--profile", "cdse", "s3", "ls", s3_prefix]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    tiles = []
    for line in result.stdout.splitlines():
        if "PRE Sentinel-2_mosaic" in line:
            parts = line.strip().split('_')
            # Format: Sentinel-2_mosaic_2025_Q3_31TDF_0_0/
            if len(parts) >= 4:
                mgrs = parts[4] if parts[3].startswith('Q') else parts[3]
                # Adjusting for different naming conventions if they exist
                if mgrs.endswith('/'): mgrs = mgrs[:-1]
                tiles.append(mgrs)
    return sorted(list(set(tiles)))

def main():
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE.")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID (default: 31TDF for Barcelona)")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01", help="Mosaic date path (default: 2025/07/01)")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg"], default="webp", help="Tile format (default: webp)")
    parser.add_argument("--quality", type=int, default=74, help="WebP/JPEG quality (default: 74)")
    parser.add_argument("--minzoom", type=int, default=0)
    parser.add_argument("--maxzoom", type=int, default=9)
    parser.add_argument("--cache", help="Local directory to cache downloaded TIFs")
    parser.add_argument("--download-only", action="store_true", help="Only download files to cache, don't process")

    args = parser.parse_args()

    setup_gdal_cdse()

    temp_vrt = "temp.vrt"
    temp_mbtiles = "temp.mbtiles"
    temp_warped_vrt = "temp_warped.vrt"

    try:
        if args.all_tiles:
            print(f"Listing all tiles for global mosaic ({args.date})...")
            tiles = list_all_mgrs_tiles(args.date)
            print(f"Found {len(tiles)} tiles.")

            if args.download_only and not args.cache:
                print("Error: --download-only requires --cache")
                sys.exit(1)

            tile_vrts = []
            for i, mgrs in enumerate(tiles):
                print(f"[{i+1}/{len(tiles)}] Handling tile {mgrs}...")
                try:
                    paths = get_tile_paths(mgrs, args.date, args.cache)
                    if args.download_only:
                        continue
                    
                    tile_vrt = f"tile_{mgrs}.vrt"
                    create_rgb_vrt(paths, tile_vrt)
                    tile_vrts.append(tile_vrt)
                except Exception as e:
                    print(f"Warning: Could not process tile {mgrs}: {e}")

            if args.download_only:
                print("Download complete.")
                return

            print("Building global VRT...")
            gdal.BuildVRT(temp_vrt, tile_vrts)
            # Clean up tile VRTs
            for f in tile_vrts:
                if os.path.exists(f):
                    os.remove(f)
        else:
            print(f"Processing tile: {args.mgrs} (Date: {args.date})")
            paths = get_tile_paths(args.mgrs, args.date, args.cache)
            if args.download_only:
                print(f"Download of {args.mgrs} complete.")
                return
            create_rgb_vrt(paths, temp_vrt)

        print("Step 1: Reprojecting to Web Mercator (VRT)...")
        warp_options = gdal.WarpOptions(
            format="VRT",
            dstSRS="EPSG:3857",
            resampleAlg="bilinear",
            srcNodata=0,
            dstNodata=0,
            callback=gdal.TermProgress_nocb
        )
        gdal.Warp(temp_warped_vrt, temp_vrt, options=warp_options)

        print(f"Step 2: Generating MBTiles ({args.format}) with zoom {args.minzoom}-{args.maxzoom}...")
        translate_options = gdal.TranslateOptions(
            format="MBTiles",
            outputType=gdal.GDT_Byte,
            scaleParams=[[0, 4000, 0, 255]],
            callback=gdal.TermProgress_nocb,
            creationOptions=[
                f"TILE_FORMAT={args.format.upper()}",
                f"QUALITY={args.quality}",
                f"MINZOOM={args.minzoom}",
                f"MAXZOOM={args.maxzoom}",
                "RESAMPLING=BILINEAR",
            ]
        )
        
        if args.format == "webp":
            gdal.SetConfigOption('WEBP_LEVEL', str(args.quality))

        gdal.Translate(temp_mbtiles, temp_warped_vrt, options=translate_options)

        print("Step 3: Converting MBTiles to PMTiles...")
        pmtiles_cmd = ["pmtiles", "convert", "--no-deduplication", temp_mbtiles, args.output]
        subprocess.run(pmtiles_cmd, check=True)

        print(f"Success! Created {args.output}")

    finally:
        for f in [temp_vrt, temp_warped_vrt, temp_mbtiles]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()
