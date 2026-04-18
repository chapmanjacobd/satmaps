#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import uuid
import sqlite3
import xml.etree.ElementTree as ET
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
    gdal.BuildVRT(output_vrt, band_list, separate=True, srcNodata=-32768, VRTNodata=-32768)

def merge_vrts(output_vrt: str, input_vrts: List[str]) -> None:
    # First build a standard VRT
    gdal.BuildVRT(output_vrt, input_vrts, srcNodata=-32768, VRTNodata=-32768)

    # Now modify it to use a pixel function by editing the XML
    tree = ET.parse(output_vrt)
    root = tree.getroot()

    # Sentinel-2 mosaics in this project use -32768 as NoData
    nodata_val = "-32768"
    num_bands = len(input_vrts)
    counts = " + ".join([f"(B{i+1} != {nodata_val})" for i in range(num_bands)])
    sums = " + ".join([f"(B{i+1} != {nodata_val} ? B{i+1} : 0)" for i in range(num_bands)])
    expr = f"( ({counts}) > 0 ) ? ( ({sums}) ) / ({counts}) : {nodata_val}"

    for band in root.findall("VRTRasterBand"):
        band.set("subClass", "VRTDerivedRasterBand")
        
        pix_func = ET.SubElement(band, "PixelFunctionType")
        pix_func.text = "expression"
        
        pix_args = ET.SubElement(band, "PixelFunctionArguments")
        pix_args.set("expression", expr)
        
        nodata = ET.SubElement(band, "NoDataValue")
        nodata.text = nodata_val

    tree.write(output_vrt, encoding="UTF-8", xml_declaration=True)

def run_warp(input_vrt: str, output_vrt: str) -> None:
    """Reproject to Web Mercator using GDAL Warp."""
    warp_options = gdal.WarpOptions(
        format="VRT",
        dstSRS="EPSG:3857",
        resampleAlg="cubic",
        callback=gdal.TermProgress_nocb, srcNodata=-32768, dstNodata=0)
    gdal.Warp(output_vrt, input_vrt, options=warp_options)

def run_translate(input_vrt: str, output_file: str, format: str, scale_params: List[List[float]], exponents: Optional[List[float]], resample_alg: str, options: Dict[str, str], projWin: Optional[List[float]] = None) -> None:
    """Convert to final format (MBTiles/PMTiles) using GDAL Translate."""
    tile_format = format.upper()
    if tile_format == "JPG":
        tile_format = "JPEG"
    if tile_format == "PNG8":
        tile_format = "PNG"

    if resample_alg == "gauss":
        resample_alg = "bilinear"

    translate_options = gdal.TranslateOptions(
        format="MBTiles",
        outputType=gdal.GDT_Byte,
        scaleParams=scale_params,
        exponents=exponents,
        projWin=projWin,
        noData=0,
        callback=gdal.TermProgress_nocb,
        metadataOptions=[
            f"format={tile_format.lower()}",
            f"name={options['name']}",
            f"description={options['description']}"
        ],
        creationOptions=[
            f"NAME={options['name']}",
            f"DESCRIPTION={options['description']}",
            "TYPE=baselayer",
            f"TILE_FORMAT={tile_format}",
            f"QUALITY={options['quality']}",
            f"MINZOOM={options['minzoom']}",
            f"MAXZOOM={options['maxzoom']}",
            f"RESAMPLING={resample_alg}",
            f"BLOCKSIZE={options['blocksize']}",
            "ZOOM_LEVEL_STRATEGY=UPPER"
        ]
    )
    
    if format == "webp":
        gdal.SetConfigOption('WEBP_LEVEL', str(options['quality']))

    gdal.Translate(output_file, input_vrt, options=translate_options)

def run_gdal2tiles(input_vrt: str, output_mbtiles: str, format: str, scale_params: List[List[float]], exponents: Optional[List[float]], resample_alg: str, options: Dict[str, str]) -> None:
    """Convert to final format (MBTiles) using gdal2tiles.py with a scaled VRT."""
    tile_format = format.upper()
    if tile_format == "JPG": tile_format = "JPEG"
    if tile_format == "PNG8": tile_format = "PNG"
    
    # 1. Translate to 8-bit scaled VRT
    scaled_vrt = input_vrt.replace(".vrt", "_scaled.vrt")
    print(f"Step 2a: Generating 8-bit scaled VRT ({scaled_vrt})...")
    translate_options = gdal.TranslateOptions(
        format="VRT",
        outputType=gdal.GDT_Byte,
        scaleParams=scale_params,
        exponents=exponents,
        noData=0
    )
    gdal.Translate(scaled_vrt, input_vrt, options=translate_options)

    # 2. Run gdal2tiles.py
    print(f"Step 2b: Running gdal2tiles.py with tile format {tile_format}...")
    tiles_dir = ".temp/raw_tiles_" + os.path.basename(output_mbtiles).replace(".mbtiles", "")
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    
    cmd = [
        "gdal2tiles.py",
        "--resume",
        "-z", f"{options['minzoom']}-{options['maxzoom']}",
        f"--processes={num_cpus}",
        "-r", resample_alg if resample_alg != "gauss" else "bilinear",
        "-a", "0",
        "--tiledriver", tile_format,
        scaled_vrt,
        tiles_dir
    ]
    if format == "webp":
        cmd.extend(["--webp-quality", str(options['quality'])])
    elif format == "jpg":
        cmd.extend(["--jpeg-quality", str(options['quality'])])

    env = os.environ.copy()
    env['AWS_S3_ENDPOINT'] = 'eodata.dataspace.copernicus.eu'
    env['AWS_HTTPS'] = 'YES'
    env['AWS_VIRTUAL_HOSTING'] = 'FALSE'
    env['AWS_PROFILE'] = 'cdse'
    env['VSI_CACHE'] = 'TRUE'
    env['GDAL_CACHEMAX'] = '1024'
    env['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    env['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
    env['GDAL_HTTP_MAX_RETRY'] = '5'

    subprocess.run(cmd, env=env, check=True)

    # 3. Pack into MBTiles
    print("Step 2c: Packing tiles into MBTiles...")
    if os.path.exists(output_mbtiles):
        os.remove(output_mbtiles)

    conn = sqlite3.connect(output_mbtiles)
    
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE metadata (name text, value text);")
    cursor.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);")
    
    metadata = [
        ("name", options['name']),
        ("description", options['description']),
        ("format", tile_format.lower()),
        ("type", "baselayer"),
        ("version", "1.1"),
        ("minzoom", str(options['minzoom'])),
        ("maxzoom", str(options['maxzoom'])),
    ]
    cursor.executemany("INSERT INTO metadata VALUES (?, ?)", metadata)
    
    count = 0
    # Walk directory. Note: gdal2tiles.py produces TMS tiles by default (no --xyz flag used)
    # The structure is tiles_dir/z/x/y.ext
    for root, dirs, files in os.walk(tiles_dir):
        for file in files:
            if file.endswith(f".{tile_format.lower()}") or file.endswith(format):
                path = os.path.join(root, file)
                parts = os.path.relpath(path, tiles_dir).split(os.sep)
                if len(parts) == 3:
                    z, x, y_ext = parts
                    y = os.path.splitext(y_ext)[0]
                    with open(path, "rb") as f:
                        blob = f.read()
                    cursor.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (int(z), int(x), int(y), blob))
                    count += 1
                    if count % 10000 == 0:
                        print(f"Packed {count} tiles...")
                        conn.commit()
    conn.commit()
    cursor.execute("CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);")
    conn.close()
    
    print(f"Total packed: {count} tiles. Cleaning up...")
    if os.path.exists(scaled_vrt):
        os.remove(scaled_vrt)
    # Keep tiles_dir if user wants to see it, or clean it up? The plan said clean it up.
    # But rm -rf in python:
    import shutil
    if os.path.exists(tiles_dir):
        shutil.rmtree(tiles_dir)

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

def get_percentiles(ds: gdal.Dataset, low: float = 2.0, high: float = 98.0, shared: bool = True) -> List[List[float]]:
    """
    Calculate the low and high percentiles for each band.
    If shared=True, the same min/max is used for all bands to preserve color balance.
    """
    band_stats = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        # Summed values can exceed 10000, so we increase the histogram range
        hist = band.GetHistogram(min=0.0, max=40000.0, buckets=4000, approx_ok=True)
        total = sum(hist)

        if total == 0:
            band_stats.append((0.0, 4000.0))
            continue

        low_threshold = total * (low / 100.0)
        high_threshold = total * (high / 100.0)

        accum = 0
        l_val = 0.0
        h_val = 4000.0

        for idx, count in enumerate(hist):
            accum += count
            if l_val == 0.0 and accum >= low_threshold:
                l_val = idx * 10.0
            if accum >= high_threshold:
                h_val = idx * 10.0
                break
        band_stats.append((l_val, h_val))

    if shared:
        # Use the global min (lowest of lows) and global max (highest of highs)
        global_low = min(s[0] for s in band_stats)
        global_high = max(s[1] for s in band_stats)
        print('min/max values', global_low, global_high)
        return [[global_low, global_high, 0, 255]] * ds.RasterCount

    return [[s[0], s[1], 0, 255] for s in band_stats]

def main():
    parser = argparse.ArgumentParser(description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE.")
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID (default: 31TDF for Barcelona. Other interesting ones: 60HTB, 07VEK, 50HQJ, 49QGF, 45RVL, 12SUD, 40RCN, 28QCH)")
    parser.add_argument("--global", dest="all_tiles", action="store_true", help="Process all available tiles")
    parser.add_argument("--date", default="2025/07/01", help="Mosaic date path (default: 2025/07/01). Can be a comma-separated list of dates.")
    parser.add_argument("--output", "-o", default="output.pmtiles", help="Output PMTiles filename")
    parser.add_argument("--format", choices=["webp", "jpg", "png", "png8"], default="webp", help="Tile format")
    parser.add_argument("--quality", type=int, default=74, help="WebP/JPEG quality")
    parser.add_argument("--resample-alg", default="lanczos", choices=["bilinear", "average", "gauss", "lanczos"], help="Downsampling method")
    parser.add_argument("--exponent", type=float, default=0.75, help="Exponent for power law scaling")
    parser.add_argument("--stats-min", type=float, help="Hardcoded source minimum value for scaling (bypasses band stats)")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source maximum value for scaling (bypasses band stats)")
    parser.add_argument("--blocksize", type=int, default=512, help="MBTiles block size")
    parser.add_argument("--minzoom", type=int, default=0)
    parser.add_argument("--maxzoom", type=int, default=14)
    parser.add_argument("--cache", help="Local directory to cache downloaded TIFs. If multiple dates are used, subdirectories for each date will be created.")
    parser.add_argument("--download-only", action="store_true", help="Only download files to cache, don't process")
    parser.add_argument("--land-only", help="Path to HLS.land.tiles.txt. If provided, tiles NOT in this list will be skipped (only in --global mode).")

    args = parser.parse_args()

    setup_gdal_cdse()

    dates = [d.strip() for d in args.date.split(",")]

    # Load land tiles if provided
    land_tiles = None
    if args.land_only:
        if os.path.exists(args.land_only):
            with open(args.land_only, 'r') as f:
                land_tiles = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(land_tiles)} land tiles from {args.land_only}")
        else:
            print(f"Warning: Land tile list {args.land_only} not found.")

    # Metadata
    tileset_name = "Internet in a Box Maps - Sentinel-2"
    tileset_desc = f"Contains modified Copernicus Sentinel data {', '.join(dates)}"

    # Unique temp files to avoid race conditions in parallel mode
    os.makedirs(".temp", exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    temp_vrt = f".temp/combined_{unique_id}.vrt"
    temp_mbtiles = f".temp/tiles_{unique_id}.mbtiles"
    temp_warped_vrt = f".temp/warped_{unique_id}.vrt"

    tile_vrts = []
    try:
        all_date_vrts = []
        for date in dates:
            date_tile_vrts = []
            date_cache = os.path.join(args.cache, date.replace("/", "-")) if args.cache else None

            if args.all_tiles:
                print(f"Listing all folders for global mosaic ({date})...")
                all_folders = list_all_mosaic_folders(date, cache_dir=date_cache)

                if land_tiles:
                    # Filter folders by land tiles
                    # Sentinel-2_mosaic_2025_Q3_31TDF_0_0 -> 31TDF
                    original_len = len(all_folders)
                    all_folders = [f for f in all_folders if f.split('_')[4] in land_tiles]
                    print(f"Filtered {original_len} -> {len(all_folders)} tiles using land mask for {date}.")
                else:
                    print(f"Found {len(all_folders)} folders for {date}.")

                if args.download_only and not args.cache:
                    print("Error: --download-only requires --cache")
                    sys.exit(1)

                # Use Grouped VRTs to avoid too many open files
                group_size = 50
                group_vrts = []
                for i in range(0, len(all_folders), group_size):
                    chunk = all_folders[i:i + group_size]
                    chunk_vrt_name = f".temp/group_{date.replace('/', '-')}_{i // group_size}_{unique_id}.vrt"

                    print(f"[{date}] [{i+1}/{len(all_folders)}] Handling group {i // group_size}...")
                    chunk_tile_vrts = []
                    for folder in chunk:
                        try:
                            paths = get_tile_paths(None, date, date_cache, folder_name=folder)
                            if args.download_only: continue

                            tile_vrt = f".temp/tile_{folder}_{unique_id}.vrt"
                            create_rgb_vrt(paths, tile_vrt)
                            chunk_tile_vrts.append(tile_vrt)
                            tile_vrts.append(tile_vrt)
                        except Exception as e:
                            print(f"Warning: Could not process folder {folder}: {e}")

                    if args.download_only: continue

                    if chunk_tile_vrts:
                        print(f"Building Group VRT {chunk_vrt_name}...")
                        gdal.BuildVRT(chunk_vrt_name, chunk_tile_vrts, srcNodata=-32768, VRTNodata=-32768)
                        group_vrts.append(chunk_vrt_name)

                if args.download_only:
                    continue

                if group_vrts:
                    date_vrt = f".temp/mosaic_{date.replace('/', '-')}_{unique_id}.vrt"
                    print(f"Building mosaic VRT {date_vrt} from groups...")
                    gdal.BuildVRT(date_vrt, group_vrts, srcNodata=-32768, VRTNodata=-32768)
                    all_date_vrts.append(date_vrt)
                    # Clean up group VRTs
                    for f in group_vrts:
                        if os.path.exists(f): os.remove(f)
            else:
                print(f"Processing tile: {args.mgrs} (Date: {date})")
                try:
                    folders = list_all_mosaic_folders(date, mgrs_filter=args.mgrs, cache_dir=date_cache)
                except RuntimeError as e:
                    print(e)
                    continue

                for folder in folders:
                    try:
                        paths = get_tile_paths(None, date, date_cache, folder_name=folder)
                        if args.download_only: continue

                        tile_vrt = f".temp/tile_{folder}_{unique_id}.vrt"
                        create_rgb_vrt(paths, tile_vrt)
                        date_tile_vrts.append(tile_vrt)
                        tile_vrts.append(tile_vrt)
                    except Exception as e:
                        print(f"Warning: Could not process folder {folder}: {e}")

                if args.download_only:
                    continue

                if date_tile_vrts:
                    date_vrt = f".temp/mosaic_{date.replace('/', '-')}_{unique_id}.vrt"
                    gdal.BuildVRT(date_vrt, date_tile_vrts, srcNodata=-32768, VRTNodata=-32768)
                    all_date_vrts.append(date_vrt)

        if args.download_only:
            print("Download complete.")
            return

        if not all_date_vrts:
            print("Error: No data found for any of the specified dates.")
            sys.exit(1)

        print("Building master VRT from all dates...")
        if len(all_date_vrts) > 1:
            merge_vrts(temp_vrt, all_date_vrts)
        else:
            gdal.BuildVRT(temp_vrt, all_date_vrts, srcNodata=-32768, VRTNodata=-32768)

        # Add all_date_vrts to cleanup list
        tile_vrts.extend(all_date_vrts)

        print("Step 1: Reprojecting to Web Mercator (VRT)...")
        run_warp(temp_vrt, temp_warped_vrt)

        print(f"Step 2: Generating MBTiles ({args.format}) with zoom {args.minzoom}-{args.maxzoom}...")

        # Calculate or use hardcoded scaling parameters
        if args.stats_min is not None and args.stats_max is not None:
            print(f"Using hardcoded scaling: {args.stats_min} to {args.stats_max}")
            ds_temp = gdal.Open(temp_warped_vrt)
            scale_params = [[args.stats_min, args.stats_max, 0, 255]] * ds_temp.RasterCount
            ds_temp = None
        else:
            # Calculate percentiles (2nd and 98th) for dynamic scaling
            ds_for_stats = gdal.Open(temp_warped_vrt)
            # Use shared=True to preserve color balance (fix "strange colors")
            scale_params = get_percentiles(ds_for_stats, shared=True)
            ds_for_stats = None

        # Exponent 0 means "don't apply"
        exponents = [args.exponent, args.exponent, args.exponent] if args.exponent != 0 else None

        options_dict = {
            "name": tileset_name,
            "description": tileset_desc,
            "quality": args.quality,
            "minzoom": args.minzoom,
            "maxzoom": args.maxzoom,
            "blocksize": args.blocksize
        }

        if args.all_tiles:
            run_gdal2tiles(
                temp_warped_vrt,
                temp_mbtiles,
                args.format,
                scale_params,
                exponents,
                args.resample_alg,
                options_dict
            )
        else:
            run_translate(
                temp_warped_vrt,
                temp_mbtiles,
                args.format,
                scale_params,
                exponents,
                args.resample_alg,
                options_dict
            )

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
