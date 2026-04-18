import os
import subprocess
import sqlite3
import shutil
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from osgeo import gdal
from typing import Dict, List, Tuple

gdal.UseExceptions()

def get_web_mercator_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Calculate Web Mercator (EPSG:3857) bounds for a given XYZ tile."""
    n = 2.0 ** z
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    # Latitude calculation (Slippy map tilenames)
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    
    def lonlat_to_3857(lon, lat):
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * 20037508.34 / 180
        return x, y
        
    x1, y1 = lonlat_to_3857(lon1, lat1)
    x2, y2 = lonlat_to_3857(lon2, lat2)
    # Return (min_x, max_y, max_x, min_y) which is projWin format: [ulx, uly, lrx, lry]
    return min(x1, x2), max(y1, y2), max(x1, x2), min(y1, y2)

def meters_to_tile(x_meters: float, y_meters: float, z: int) -> Tuple[int, int]:
    """Convert Web Mercator meters to tile coordinates at zoom z."""
    res = 20037508.34 * 2 / (2**z)
    tx = int((x_meters + 20037508.34) / res)
    ty = int((20037508.34 - y_meters) / res)
    return tx, ty

def create_color_corrected_vrt(input_vrt: str, output_vrt: str, 
                               scale_params: List[List[float]],
                               brightness: float, contrast: float, saturation: float) -> None:
    """Create a VRT that scales 16-bit to 8-bit and applies color corrections."""
    ds = gdal.Open(input_vrt)
    width, height = ds.RasterXSize, ds.RasterYSize
    
    vrt_content = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f'  <SRS>{ds.GetProjection()}</SRS>',
        f'  <GeoTransform>{", ".join(map(str, ds.GetGeoTransform()))}</GeoTransform>'
    ]
    
    for b_idx in range(1, 4):
        s_min, s_max, d_min, d_max = scale_params[b_idx-1]
        
        # Coefficients for luminance (CCIR 601)
        # L = 0.299*R + 0.587*G + 0.114*B
        # Normalize each band based on its scale_params
        def norm_expr(idx):
            sm, sx = scale_params[idx-1][0], scale_params[idx-1][1]
            return f"( (B{idx} - {sm}) / ({sx - sm}) )"
        
        luma = f"( 0.299*{norm_expr(1)} + 0.587*{norm_expr(2)} + 0.114*{norm_expr(3)} )"
        norm = norm_expr(b_idx)
        
        sat_expr = f"( {luma} + {saturation} * ({norm} - {luma}) )"
        bright_expr = f"( {sat_expr} * {brightness} )"
        cont_expr = f"( ({bright_expr} - 0.5) * {contrast} + 0.5 )"
        final_expr = f"min(255, max(0, {cont_expr} * 255))"
        
        vrt_content.extend([
            f'  <VRTRasterBand dataType="Byte" band="{b_idx}" subClass="VRTDerivedRasterBand">',
            '    <PixelFunctionType>expression</PixelFunctionType>',
            f'    <PixelFunctionArguments expression="{final_expr}" />',
            '    <NoDataValue>0</NoDataValue>'
        ])
        
        # Add all 3 bands as sources to this band so B1, B2, B3 are available
        for src_idx in range(1, 4):
            vrt_content.extend([
                '    <SimpleSource>',
                f'      <SourceFilename relativeToVRT="0">{os.path.abspath(input_vrt)}</SourceFilename>',
                f'      <SourceBand>{src_idx}</SourceBand>',
                f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
                f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
                '    </SimpleSource>'
            ])
        vrt_content.append('  </VRTRasterBand>')
        
    vrt_content.append('</VRTDataset>')
    
    with open(output_vrt, 'w') as f:
        f.write("\n".join(vrt_content))

def process_chunk(args: Tuple) -> str:
    """Worker function for parallel gdal.Translate."""
    input_vrt, chunk_file, format, options, proj_win = args
    try:
        gdal.UseExceptions()
        
        translate_options = gdal.TranslateOptions(
            format="MBTiles",
            outputType=gdal.GDT_Byte,
            projWin=proj_win,
            noData=0,
            metadataOptions=[
                f"format={format.lower()}",
                f"name={options['name']}",
                f"description={options['description']}"
            ],
            creationOptions=[
                f"NAME={options['name']}",
                f"DESCRIPTION={options['description']}",
                "TYPE=baselayer",
                f"TILE_FORMAT={'JPEG' if format.lower() == 'jpg' else format.upper()}",
                f"QUALITY={options['quality']}",
                f"RESAMPLING={options['resample_alg'] if options['resample_alg'] != 'gauss' else 'bilinear'}",
                f"BLOCKSIZE={options.get('blocksize', 512)}",
                "ZOOM_LEVEL_STRATEGY=UPPER"
            ]
        )
        
        if format.lower() == "webp":
            gdal.SetConfigOption('WEBP_LEVEL', str(options['quality']))

        gdal.Translate(chunk_file, input_vrt, options=translate_options)
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        return ""

def merge_mbtiles(output_mbtiles: str, input_mbtiles: List[str]) -> None:
    """Merge multiple MBTiles chunks into a single file."""
    if not input_mbtiles:
        return
    
    print(f"Merging {len(input_mbtiles)} chunks into {output_mbtiles}...")
    
    # Sort chunks to ensure consistent merging if needed
    input_mbtiles = sorted([f for f in input_mbtiles if f and os.path.exists(f)])
    if not input_mbtiles:
        return

    shutil.copyfile(input_mbtiles[0], output_mbtiles)
    
    conn = sqlite3.connect(output_mbtiles)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    
    for i, db_path in enumerate(input_mbtiles[1:]):
        cursor.execute(f"ATTACH DATABASE '{db_path}' AS db")
        try:
            cursor.execute("INSERT OR IGNORE INTO map SELECT * FROM db.map")
            cursor.execute("INSERT OR IGNORE INTO images SELECT * FROM db.images")
        except sqlite3.OperationalError:
            cursor.execute("INSERT OR IGNORE INTO tiles SELECT * FROM db.tiles")
        cursor.execute("DETACH DATABASE db")
        if i % 10 == 0:
            conn.commit()
    
    conn.commit()
    conn.close()

def run_tiling(input_vrt: str, output_mbtiles: str, tile_format: str, 
               scale_params: List[List[float]], options: Dict) -> None:
    """Main entry point for tiling. Handles chunking, parallel processing, and merging."""
    
    unique_id = options.get('unique_id', 'tiles')
    chunk_zoom = options.get('chunk_zoom', 4)
    
    # 1. Create color-corrected and scaled VRT
    color_vrt = f".temp/color_corrected_{unique_id}.vrt"
    run_tiling.color_vrt = color_vrt # Store for cleanup
    
    print(f"Applying color correction (B:{options['brightness']} C:{options['contrast']} S:{options['saturation']})...")
    create_color_corrected_vrt(
        input_vrt, color_vrt, scale_params,
        options['brightness'], options['contrast'], options['saturation']
    )
    
    # 2. Determine chunks
    ds = gdal.Open(color_vrt)
    gt = ds.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx = minx + gt[1] * ds.RasterXSize
    miny = maxy + gt[5] * ds.RasterYSize
    ds = None
    
    tx_min, ty_max = meters_to_tile(minx, miny, chunk_zoom)
    tx_max, ty_min = meters_to_tile(maxx, maxy, chunk_zoom)
    
    max_t = (2 ** chunk_zoom) - 1
    tx_min, tx_max = max(0, min(tx_min, max_t)), max(0, min(tx_max, max_t))
    ty_min, ty_max = max(0, min(ty_min, max_t)), max(0, min(ty_max, max_t))
    
    tasks = []
    chunk_files = []
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            chunk_file = f".temp/chunk_{chunk_zoom}_{tx}_{ty}_{unique_id}.mbtiles"
            chunk_files.append(chunk_file)
            
            if os.path.exists(chunk_file):
                continue
                
            ulx, uly, lrx, lry = get_web_mercator_bounds(chunk_zoom, tx, ty)
            tasks.append((color_vrt, chunk_file, tile_format, options, [ulx, uly, lrx, lry]))

    # 3. Parallel Execution
    if tasks:
        num_workers = options.get('processes', 1)
        print(f"Processing {len(tasks)} chunk(s) at zoom {chunk_zoom} with {num_workers} worker(s)...")
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk, tasks))
        else:
            for task in tasks:
                process_chunk(task)
            
    # 4. Merge Chunks
    merge_mbtiles(output_mbtiles, chunk_files)
    
    # 5. Build Overviews
    print("Building overviews...")
    gdaladdo_cmd = [
        "gdaladdo", "-r", options['resample_alg'], 
        "--config", "GDAL_NUM_THREADS", "ALL_CPUS", 
        output_mbtiles
    ]
    subprocess.run(gdaladdo_cmd, check=True)
    
    # Cleanup chunk files only if merge was successful
    if os.path.exists(output_mbtiles):
        for f in chunk_files:
            if f and os.path.exists(f):
                os.remove(f)
        if os.path.exists(color_vrt):
            os.remove(color_vrt)
