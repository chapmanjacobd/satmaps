from array import array
import os
import subprocess
import sqlite3
import shutil
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from osgeo import gdal
from typing import Any, Dict, List, Optional, Tuple

gdal.UseExceptions()

WEB_MERCATOR_LIMIT = 20037508.342789244
ProjWin = Tuple[float, float, float, float]
ChunkTask = Tuple[str, str, str, Dict[str, Any], ProjWin]
RGB_COLOR_INTERPRETATIONS = ("Red", "Green", "Blue")
SOFT_KNEE_SHADOW_BREAK = 0.3
SOFT_KNEE_HIGHLIGHT_BREAK = 0.75
SOFT_KNEE_SHADOW_SLOPE = 1.4
SOFT_KNEE_MID_SLOPE = 0.9
SOFT_KNEE_HIGHLIGHT_SLOPE = 0.5


@dataclass(frozen=True)
class TilingArtifacts:
    """Paths created during tiling so the caller can preserve or clean them up."""

    final_vrt: str
    cleanup_paths: List[str]


def ternary_expr(condition: str, true_value: str, false_value: str) -> str:
    """Build a parenthesized ternary expression for GDAL's expression dialect."""
    return f"(({condition}) ? ({true_value}) : ({false_value}))"


def clamp_expr(expression: str, lower: str, upper: str) -> str:
    """Clamp an expression to an inclusive range using nested ternaries."""
    return ternary_expr(
        f"({expression}) < {lower}",
        lower,
        ternary_expr(f"({expression}) > {upper}", upper, expression),
    )


def expression_literal(value: float) -> str:
    """Format a numeric literal for GDAL expression strings."""
    return repr(float(value))


def nodata_match_expr(band_ref: str, nodata_value: float) -> str:
    """Return an expression that matches a band's explicit nodata value."""
    if math.isnan(nodata_value):
        return f"({band_ref} != {band_ref})"
    return f"({band_ref} == {expression_literal(nodata_value)})"


def explicit_nodata_expr(dataset: gdal.Dataset, band_count: int = 3) -> Optional[str]:
    """Return an expression that masks only pixels matching explicit band nodata values."""
    nodata_matches: List[str] = []
    for band_index in range(1, band_count + 1):
        nodata_value = dataset.GetRasterBand(band_index).GetNoDataValue()
        if nodata_value is None:
            return None
        nodata_matches.append(nodata_match_expr(f"B{band_index}", nodata_value))

    return " && ".join(nodata_matches)


def soft_knee_tone_curve_expr(expression: str) -> str:
    """Lift shadows and roll off highlights with a monotonic piecewise-linear curve."""
    clamped = clamp_expr(expression, "0.0", "1.0")
    shadow_break = expression_literal(SOFT_KNEE_SHADOW_BREAK)
    highlight_break = expression_literal(SOFT_KNEE_HIGHLIGHT_BREAK)
    shadow_slope = expression_literal(SOFT_KNEE_SHADOW_SLOPE)
    mid_slope = expression_literal(SOFT_KNEE_MID_SLOPE)
    highlight_slope = expression_literal(SOFT_KNEE_HIGHLIGHT_SLOPE)
    shadow_output = expression_literal(SOFT_KNEE_SHADOW_BREAK * SOFT_KNEE_SHADOW_SLOPE)
    highlight_output = expression_literal(
        (SOFT_KNEE_SHADOW_BREAK * SOFT_KNEE_SHADOW_SLOPE)
        + ((SOFT_KNEE_HIGHLIGHT_BREAK - SOFT_KNEE_SHADOW_BREAK) * SOFT_KNEE_MID_SLOPE)
    )

    shadow_expr = f"({clamped} * {shadow_slope})"
    mid_expr = f"({shadow_output} + (({clamped} - {shadow_break}) * {mid_slope}))"
    highlight_expr = f"({highlight_output} + (({clamped} - {highlight_break}) * {highlight_slope}))"
    return clamp_expr(
        ternary_expr(
            f"({clamped}) < {shadow_break}",
            shadow_expr,
            ternary_expr(f"({clamped}) < {highlight_break}", mid_expr, highlight_expr),
        ),
        "0.0",
        "1.0",
    )


def vrt_dataset_preamble(dataset: gdal.Dataset) -> List[str]:
    """Return the opening XML for a VRT mirroring an existing dataset."""
    return [
        f'<VRTDataset rasterXSize="{dataset.RasterXSize}" rasterYSize="{dataset.RasterYSize}">',
        f'  <SRS>{dataset.GetProjection()}</SRS>',
        f'  <GeoTransform>{", ".join(map(str, dataset.GetGeoTransform()))}</GeoTransform>',
    ]


def apply_rgb_color_interpretation_to_vrt(vrt_path: str) -> None:
    """Set RGB color interpretation on the first three bands of a VRT."""
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    for band_index, color_name in enumerate(RGB_COLOR_INTERPRETATIONS, start=1):
        band = root.find(f"./VRTRasterBand[@band='{band_index}']")
        if band is None:
            continue

        color_interp = band.find("ColorInterp")
        if color_interp is None:
            color_interp = ET.Element("ColorInterp")
            band.insert(0, color_interp)
        color_interp.text = color_name

    tree.write(vrt_path, encoding="UTF-8", xml_declaration=True)


def make_step_vrt_path(step: int, label: str, unique_id: str) -> str:
    """Return a temporary VRT path with an explicit pipeline step prefix."""
    return f".temp/step{step}_{label}_{unique_id}.vrt"


def make_color_corrected_float_vrt_path(output_vrt: str) -> str:
    """Return the companion Float32 VRT path backing a Byte step-6 VRT."""
    return f"{os.path.splitext(output_vrt)[0]}_float.vrt"


def make_color_corrected_byte_tif_path(output_vrt: str) -> str:
    """Return the companion Byte GeoTIFF path backing a step-6 inspection VRT."""
    return f"{os.path.splitext(output_vrt)[0]}_byte.tif"

def get_web_mercator_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Calculate Web Mercator (EPSG:3857) bounds for a given XYZ tile."""
    n = 2.0 ** z
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    # Latitude calculation (Slippy map tilenames)
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    
    def lonlat_to_3857(lon: float, lat: float) -> Tuple[float, float]:
        x_meters = lon * WEB_MERCATOR_LIMIT / 180
        y_meters = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * WEB_MERCATOR_LIMIT / 180
        return x_meters, y_meters
        
    x1, y1 = lonlat_to_3857(lon1, lat1)
    x2, y2 = lonlat_to_3857(lon2, lat2)
    # Return (min_x, max_y, max_x, min_y) which is projWin format: [ulx, uly, lrx, lry]
    return min(x1, x2), max(y1, y2), max(x1, x2), min(y1, y2)

def meters_to_tile(x_meters: float, y_meters: float, z: int) -> Tuple[int, int]:
    """Convert Web Mercator meters to tile coordinates at zoom z."""
    res = WEB_MERCATOR_LIMIT * 2 / (2**z)
    tx = math.floor((x_meters + WEB_MERCATOR_LIMIT) / res)
    ty = math.floor((WEB_MERCATOR_LIMIT - y_meters) / res)
    return tx, ty

def get_dataset_bounds(dataset: gdal.Dataset) -> ProjWin:
    """Return dataset bounds as a projWin-style tuple."""
    gt = dataset.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx = minx + gt[1] * dataset.RasterXSize
    miny = maxy + gt[5] * dataset.RasterYSize
    return minx, maxy, maxx, miny

def get_chunk_tile_range(bounds: ProjWin, zoom: int) -> Tuple[int, int, int, int]:
    """Return inclusive XYZ tile indices covering raster bounds at the chunk zoom."""
    minx, maxy, maxx, miny = bounds
    max_t = (2 ** zoom) - 1

    tx_min, ty_min = meters_to_tile(minx, maxy, zoom)
    tx_max, ty_max = meters_to_tile(
        math.nextafter(maxx, float("-inf")),
        math.nextafter(miny, float("inf")),
        zoom,
    )

    return (
        max(0, min(tx_min, max_t)),
        max(0, min(ty_min, max_t)),
        max(0, min(tx_max, max_t)),
        max(0, min(ty_max, max_t)),
    )

def intersect_proj_win(proj_win: ProjWin, dataset_bounds: ProjWin) -> Optional[ProjWin]:
    """Clamp a projWin to the dataset extent, returning None for empty intersections."""
    ulx = max(proj_win[0], dataset_bounds[0])
    uly = min(proj_win[1], dataset_bounds[1])
    lrx = min(proj_win[2], dataset_bounds[2])
    lry = max(proj_win[3], dataset_bounds[3])

    if ulx >= lrx or lry >= uly:
        return None

    return ulx, uly, lrx, lry


def proj_win_to_src_win(dataset: gdal.Dataset, proj_win: ProjWin) -> Tuple[int, int, int, int]:
    """Convert a georeferenced projWin into a clipped pixel srcWin for a north-up raster."""
    inv_gt = gdal.InvGeoTransform(dataset.GetGeoTransform())
    px_ul, py_ul = gdal.ApplyGeoTransform(inv_gt, proj_win[0], proj_win[1])
    px_lr, py_lr = gdal.ApplyGeoTransform(inv_gt, proj_win[2], proj_win[3])

    xoff = max(0, math.floor(min(px_ul, px_lr)))
    yoff = max(0, math.floor(min(py_ul, py_lr)))
    xend = min(dataset.RasterXSize, math.ceil(max(px_ul, px_lr)))
    yend = min(dataset.RasterYSize, math.ceil(max(py_ul, py_lr)))

    return xoff, yoff, max(0, xend - xoff), max(0, yend - yoff)

def write_color_corrected_float_vrt(
    input_vrt: str,
    output_vrt: str,
    scale_params: List[List[float]],
) -> None:
    """Create a Float32 VRT that normalizes source data and applies a soft-knee tone curve."""
    ds = gdal.Open(input_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open input VRT {input_vrt}")
    width, height = ds.RasterXSize, ds.RasterYSize
    nodata_expr = explicit_nodata_expr(ds)
    
    vrt_content = vrt_dataset_preamble(ds)
    
    for b_idx in range(1, 4):
        def norm_expr(idx: int) -> str:
            sm, sx = scale_params[idx-1][0], scale_params[idx-1][1]
            band_range = sx - sm
            if band_range <= 0:
                band_range = 1.0
            sm_value = expression_literal(sm)
            band_range_value = expression_literal(band_range)
            raw = f"((B{idx} - {sm_value}) / {band_range_value})"
            return clamp_expr(raw, "0.0", "1.0")

        toned_expr = soft_knee_tone_curve_expr(norm_expr(b_idx))
        scaled_expr = f"({toned_expr} * 255.0)"
        clamped_expr = clamp_expr(scaled_expr, "0.0", "255.0")
        final_expr = clamped_expr if nodata_expr is None else ternary_expr(nodata_expr, "0.0", clamped_expr)
        escaped_expr = escape(final_expr, {'"': '&quot;'})
        
        vrt_content.extend([
            f'  <VRTRasterBand dataType="Float32" band="{b_idx}" subClass="VRTDerivedRasterBand">',
            f'    <ColorInterp>{RGB_COLOR_INTERPRETATIONS[b_idx - 1]}</ColorInterp>',
            '    <PixelFunctionType>expression</PixelFunctionType>',
            f'    <PixelFunctionArguments expression="{escaped_expr}" />',
            '    <NoDataValue>0</NoDataValue>'
        ])
        
        # Add all 3 bands as sources to this band so B1, B2, B3 are available
        for src_idx in range(1, 4):
            source_band = ds.GetRasterBand(src_idx)
            block_x, block_y = source_band.GetBlockSize()
            source_type = gdal.GetDataTypeName(source_band.DataType)
            vrt_content.extend([
                '    <SimpleSource>',
                f'      <SourceFilename relativeToVRT="0">{os.path.abspath(input_vrt)}</SourceFilename>',
                f'      <SourceBand>{src_idx}</SourceBand>',
                (
                    f'      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" '
                    f'DataType="{source_type}" BlockXSize="{block_x}" BlockYSize="{block_y}" />'
                ),
                f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
                f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
                '    </SimpleSource>'
            ])
        vrt_content.append('  </VRTRasterBand>')
        
    vrt_content.append('</VRTDataset>')
    
    with open(output_vrt, 'w') as f:
        f.write("\n".join(vrt_content))


def create_color_corrected_vrt(
    input_vrt: str,
    output_vrt: str,
    scale_params: List[List[float]],
) -> None:
    """Create the display-ready Byte step-6 VRT from Float32 and Byte companions."""
    float_output_vrt = make_color_corrected_float_vrt_path(output_vrt)
    byte_output_tif = make_color_corrected_byte_tif_path(output_vrt)
    write_color_corrected_float_vrt(input_vrt, float_output_vrt, scale_params)
    materialize_byte_geotiff(float_output_vrt, byte_output_tif)
    gdal.Translate(output_vrt, byte_output_tif, options=gdal.TranslateOptions(format="VRT"))
    apply_rgb_color_interpretation_to_vrt(output_vrt)


def materialize_byte_geotiff(input_vrt: str, output_tif: str, block_size: int = 512) -> None:
    """Materialize a Float32 RGB VRT into a Byte GeoTIFF without loading the full raster at once."""
    ds = gdal.Open(input_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open input VRT {input_vrt}")

    driver = gdal.GetDriverByName("GTiff")
    byte_ds = driver.Create(
        output_tif,
        ds.RasterXSize,
        ds.RasterYSize,
        ds.RasterCount,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=ZSTD", "ZSTD_LEVEL=1"],
    )
    if byte_ds is None:
        raise RuntimeError(f"Could not create Byte GeoTIFF {output_tif}")

    byte_ds.SetProjection(ds.GetProjection())
    byte_ds.SetGeoTransform(ds.GetGeoTransform())

    try:
        for band_index in range(1, ds.RasterCount + 1):
            src_band = ds.GetRasterBand(band_index)
            dst_band = byte_ds.GetRasterBand(band_index)
            dst_band.SetColorInterpretation(src_band.GetColorInterpretation())
            dst_band.SetNoDataValue(0)

            for yoff in range(0, ds.RasterYSize, block_size):
                ysize = min(block_size, ds.RasterYSize - yoff)
                for xoff in range(0, ds.RasterXSize, block_size):
                    xsize = min(block_size, ds.RasterXSize - xoff)
                    float_bytes = src_band.ReadRaster(xoff, yoff, xsize, ysize, buf_type=gdal.GDT_Float32)
                    if float_bytes is None:
                        raise RuntimeError(
                            f"Could not read Float32 block x={xoff} y={yoff} size={xsize}x{ysize} from {input_vrt}"
                        )

                    float_values = array("f")
                    float_values.frombytes(float_bytes)
                    byte_values = bytes(
                        0 if value <= 0.0 else 255 if value >= 255.0 else int(value + 0.5)
                        for value in float_values
                    )
                    dst_band.WriteRaster(xoff, yoff, xsize, ysize, byte_values, buf_type=gdal.GDT_Byte)
        byte_ds.FlushCache()
    finally:
        byte_ds = None
        ds = None


def create_byte_conversion_vrt(input_vrt: str, output_vrt: str) -> None:
    """Create a Byte VRT that casts an already-clamped RGB source into 8-bit bands."""
    ds = gdal.Open(input_vrt)
    width, height = ds.RasterXSize, ds.RasterYSize

    vrt_content = vrt_dataset_preamble(ds)

    for band_index in range(1, ds.RasterCount + 1):
        vrt_content.extend([
            f'  <VRTRasterBand dataType="Byte" band="{band_index}">',
            f'    <ColorInterp>{RGB_COLOR_INTERPRETATIONS[band_index - 1]}</ColorInterp>',
            '    <NoDataValue>0</NoDataValue>',
            '    <SimpleSource>',
            f'      <SourceFilename relativeToVRT="0">{os.path.abspath(input_vrt)}</SourceFilename>',
            f'      <SourceBand>{band_index}</SourceBand>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
            f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />',
            '    </SimpleSource>',
            '  </VRTRasterBand>',
        ])

    vrt_content.append('</VRTDataset>')

    with open(output_vrt, 'w') as f:
        f.write("\n".join(vrt_content))

def process_chunk(args: ChunkTask) -> str:
    """Worker function for parallel gdal.Translate."""
    input_vrt, chunk_file, format, options, proj_win = args
    ds = None
    temp_chunk_raster = chunk_file.replace(".mbtiles", ".tif")
    try:
        gdal.UseExceptions()
        
        # Open VRT to check intersection
        ds = gdal.Open(input_vrt)
        if ds is None:
            return ""

        clipped_proj_win = intersect_proj_win(proj_win, get_dataset_bounds(ds))
        if clipped_proj_win is None:
            return ""

        src_win = proj_win_to_src_win(ds, clipped_proj_win)
        if src_win[2] <= 0 or src_win[3] <= 0:
            return ""

        temp_ds = gdal.Translate(
            temp_chunk_raster,
            input_vrt,
            options=gdal.TranslateOptions(
                format="GTiff",
                srcWin=src_win,
            ),
        )
        temp_ds.FlushCache()
        temp_ds = None

        translate_options = gdal.TranslateOptions(
            format="MBTiles",
            outputType=gdal.GDT_Byte,
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

        gdal.Translate(chunk_file, temp_chunk_raster, options=translate_options)
        return chunk_file
    except Exception as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        return ""
    finally:
        ds = None
        if os.path.exists(temp_chunk_raster):
            os.remove(temp_chunk_raster)

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
    conn.execute("PRAGMA busy_timeout = 30000") # 30 seconds
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    
    for i, db_path in enumerate(input_mbtiles[1:]):
        alias = f"chunk_{i}"

        # Retry logic for ATTACH in case of locks
        attached = False
        for retry in range(5):
            try:
                cursor.execute(f"ATTACH DATABASE '{db_path}' AS {alias}")
                attached = True
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"Failed to attach {db_path}: {e}") from e
        
        if not attached:
            raise RuntimeError(f"Failed to attach {db_path} after retries due to database locks")

        try:
            # Check which tables exist in db
            cursor.execute(f"SELECT name FROM {alias}.sqlite_master WHERE type='table' AND name='map'")
            has_map = cursor.fetchone()
            
            if has_map:
                cursor.execute(f"INSERT OR IGNORE INTO map SELECT * FROM {alias}.map")
                cursor.execute(f"INSERT OR IGNORE INTO images SELECT * FROM {alias}.images")
            else:
                cursor.execute(f"INSERT OR IGNORE INTO tiles SELECT * FROM {alias}.tiles")
            conn.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Error merging {db_path}: {e}") from e
        finally:
            if attached:
                try:
                    cursor.execute(f"DETACH DATABASE {alias}")
                except sqlite3.OperationalError as e:
                    raise RuntimeError(f"Failed to detach {db_path}: {e}") from e
    
    conn.commit()
    conn.close()

def run_tiling(input_vrt: str, output_mbtiles: str, tile_format: str,
               scale_params: List[List[float]], options: Dict[str, Any]) -> TilingArtifacts:
    """Main entry point for tiling. Handles staging, chunking, and merging."""
    unique_id = options.get('unique_id', 'tiles')
    chunk_zoom = options.get('chunk_zoom', 4)
    vrt_only = options.get('vrt', False)

    # 6. Create the display-ready Byte VRT used for QGIS inspection.
    color_vrt = make_step_vrt_path(6, "color_corrected", unique_id)
    color_float_vrt = make_color_corrected_float_vrt_path(color_vrt)
    color_byte_tif = make_color_corrected_byte_tif_path(color_vrt)
    print("Applying step 6 soft-knee tone mapping...")
    create_color_corrected_vrt(input_vrt, color_vrt, scale_params)

    if vrt_only:
        return TilingArtifacts(final_vrt=color_vrt, cleanup_paths=[color_float_vrt, color_byte_tif])

    # 7. Materialize the final byte VRT used for packaging.
    byte_vrt = make_step_vrt_path(7, "byte_conversion", unique_id)
    create_byte_conversion_vrt(color_vrt, byte_vrt)

    # 8. Determine chunks from the final Byte VRT.
    ds = gdal.Open(byte_vrt)
    bounds = get_dataset_bounds(ds)
    ds = None

    tx_min, ty_min, tx_max, ty_max = get_chunk_tile_range(bounds, chunk_zoom)

    tasks: List[ChunkTask] = []
    chunk_files: List[str] = []
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            chunk_file = f".temp/chunk_{chunk_zoom}_{tx}_{ty}_{unique_id}.mbtiles"
            chunk_files.append(chunk_file)

            if os.path.exists(chunk_file):
                continue

            ulx, uly, lrx, lry = get_web_mercator_bounds(chunk_zoom, tx, ty)
            tasks.append((byte_vrt, chunk_file, tile_format, options, (ulx, uly, lrx, lry)))

    # 9. Parallel execution.
    if tasks:
        num_workers = options.get('processes', 1)
        print(f"Processing {len(tasks)} chunk(s) at zoom {chunk_zoom} with {num_workers} worker(s)...")
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk, tasks))
        else:
            for task in tasks:
                process_chunk(task)

    # 10. Merge chunks.
    merge_mbtiles(output_mbtiles, chunk_files)

    # 11. Build overviews.
    print("Building overviews...")
    gdaladdo_cmd = [
        "gdaladdo", "-r", options['resample_alg'], 
        "--config", "GDAL_NUM_THREADS", "ALL_CPUS", 
        output_mbtiles
    ]
    subprocess.run(gdaladdo_cmd, check=True)

    return TilingArtifacts(final_vrt=byte_vrt, cleanup_paths=[color_float_vrt, color_byte_tif, color_vrt, byte_vrt, *chunk_files])
