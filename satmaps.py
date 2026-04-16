#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from osgeo import gdal

gdal.UseExceptions()


def setup_gdal_cdse():
    gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'NO')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', 'eodata.dataspace.copernicus.eu')
    gdal.SetConfigOption('AWS_HTTPS', 'YES')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
    gdal.SetConfigOption('AWS_PROFILE', 'cdse')
    gdal.SetConfigOption('GDAL_CACHEMAX', '512')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '3')


def get_tile_paths(mgrs_tile, date_path="2025/07/01"):
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
        q = "Q3" # Default

    base = (
        f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/Sentinel-2_mosaic_2025_{q}_{mgrs_tile}_0_0"
    )
    return {'red': f"{base}/B04.tif", 'green': f"{base}/B03.tif", 'blue': f"{base}/B02.tif"}


def create_rgb_vrt(paths, output_vrt):
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(output_vrt, [paths['red'], paths['green'], paths['blue']], options=vrt_options)
    if vrt is None:
        raise RuntimeError(f"Failed to create VRT: {output_vrt}")
    vrt.FlushCache()
    return output_vrt


def list_all_mgrs_tiles(date_path="2025/07/01"):
    s3_prefix = f"s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/"
    cmd = [
        "aws",
        "--endpoint-url",
        "https://eodata.dataspace.copernicus.eu",
        "--profile",
        "cdse",
        "s3",
        "ls",
        s3_prefix,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    tiles = []
    for line in result.stdout.splitlines():
        if "PRE Sentinel-2_mosaic" in line:
            parts = line.strip().split('_')
            if len(parts) >= 4:
                mgrs = parts[3]
                tiles.append(mgrs)
    return tiles


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

    args = parser.parse_args()

    setup_gdal_cdse()

    temp_vrt = "temp.vrt"
    temp_mbtiles = "temp.mbtiles"

    try:
        if args.all_tiles:
            print(f"Listing all tiles for global mosaic ({args.date})...")
            tiles = list_all_mgrs_tiles(args.date)
            print(f"Found {len(tiles)} tiles.")

            tile_vrts = []
            for i, mgrs in enumerate(tiles):
                paths = get_tile_paths(mgrs, args.date)
                tile_vrt = f"tile_{mgrs}.vrt"
                try:
                    create_rgb_vrt(paths, tile_vrt)
                    tile_vrts.append(tile_vrt)
                except Exception as e:
                    print(f"Warning: Could not process tile {mgrs}: {e}")
                
                if (i + 1) % 50 == 0:
                    print(f"Prepared {i+1}/{len(tiles)} tile VRTs...")

            print("Building global VRT...")
            gdal.BuildVRT(temp_vrt, tile_vrts)
            # Clean up tile VRTs
            for f in tile_vrts:
                if os.path.exists(f):
                    os.remove(f)
        else:
            print(f"Processing tile: {args.mgrs} (Date: {args.date})")
            paths = get_tile_paths(args.mgrs, args.date)
            create_rgb_vrt(paths, temp_vrt)

        print(f"Generating MBTiles ({args.format})...")
        translate_options = [
            "-of",
            "MBTiles",
            "-ot",
            "Byte",
            # Sentinel-2 values are typically 0-10000. Scale to 0-255 for RGB.
            "-scale",
            "0",
            "4000",
            "0",
            "255",
            "-co",
            f"TILE_FORMAT={args.format.upper()}",
            "-co",
            f"QUALITY={args.quality}",
            "-co",
            f"MINZOOM={args.minzoom}",
            "-co",
            f"MAXZOOM={args.maxzoom}",
        ]

        if args.format == "webp":
            gdal.SetConfigOption('WEBP_LEVEL', str(args.quality))

        gdal.Translate(temp_mbtiles, temp_vrt, options=translate_options)

        print("Converting MBTiles to PMTiles...")
        pmtiles_cmd = ["pmtiles", "convert", temp_mbtiles, args.output]
        subprocess.run(pmtiles_cmd, check=True)

        print(f"Done! Created {args.output}")

    finally:
        for f in [temp_vrt, temp_mbtiles]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
