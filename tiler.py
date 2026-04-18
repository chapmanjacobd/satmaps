import os
import subprocess
import sqlite3
import shutil
import multiprocessing
from osgeo import gdal
from typing import Dict, List, Optional

def run_gdal2tiles(input_vrt: str, output_mbtiles: str, tile_format: str, scale_params: List[List[float]], exponents: Optional[List[float]], resample_alg: str, options: Dict) -> None:
    """Convert to final format (MBTiles) using gdal2tiles.py with a scaled VRT."""
    # Standardize format naming
    format_map = {"JPG": "JPEG", "PNG8": "PNG"}
    tile_driver = format_map.get(tile_format.upper(), tile_format.upper())
    
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
    print(f"Step 2b: Running gdal2tiles.py with tile format {tile_driver}...")
    tiles_dir = ".temp/raw_tiles_" + os.path.basename(output_mbtiles).replace(".mbtiles", "")
    num_cpus = multiprocessing.cpu_count()
    
    cmd = [
        "gdal2tiles.py",
        "--resume",
        "-z", f"{options['minzoom']}-{options['maxzoom']}",
        f"--processes={num_cpus}",
        "-r", resample_alg if resample_alg != "gauss" else "bilinear",
        "-a", "0",
        "--tiledriver", tile_driver,
        scaled_vrt,
        tiles_dir
    ]
    
    if tile_format.lower() == "webp":
        cmd.extend(["--webp-quality", str(options['quality'])])
    elif tile_format.lower() in ["jpg", "jpeg"]:
        cmd.extend(["--jpeg-quality", str(options['quality'])])

    # S3 Environment configuration for gdal2tiles subprocess
    env = os.environ.copy()
    env.update({
        'AWS_S3_ENDPOINT': 'eodata.dataspace.copernicus.eu',
        'AWS_HTTPS': 'YES',
        'AWS_VIRTUAL_HOSTING': 'FALSE',
        'AWS_PROFILE': 'cdse',
        'VSI_CACHE': 'TRUE',
        'GDAL_CACHEMAX': '1024',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
        'GDAL_HTTP_MAX_RETRY': '5'
    })

    subprocess.run(cmd, env=env, check=True)

    # 3. Pack into MBTiles
    print("Step 2c: Packing tiles into MBTiles...")
    pack_tiles_to_mbtiles(tiles_dir, output_mbtiles, tile_driver, options)
    
    print("Cleaning up temporary tile directory...")
    if os.path.exists(scaled_vrt):
        os.remove(scaled_vrt)
    if os.path.exists(tiles_dir):
        shutil.rmtree(tiles_dir)

def pack_tiles_to_mbtiles(tiles_dir: str, output_mbtiles: str, tile_format: str, options: Dict) -> None:
    """Pack a directory of TMS tiles into an MBTiles SQLite database."""
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
    # Walk directory structure: tiles_dir/z/x/y.ext
    for root, _, files in os.walk(tiles_dir):
        for file in files:
            if file.lower().endswith(tuple(['.webp', '.jpg', '.jpeg', '.png'])):
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
    print(f"Total packed: {count} tiles.")
