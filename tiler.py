import math
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

WEB_MERCATOR_LIMIT = 20037508.342789244
ProjWin = Tuple[float, float, float, float]
ChunkTask = Tuple[str, str, str, Dict[str, Any], ProjWin]
RGB_COLOR_INTERPRETATIONS = ("Red", "Green", "Blue")

# Tone-mapping constants
SOFT_KNEE_SHADOW_BREAK = 0.3
SOFT_KNEE_HIGHLIGHT_BREAK = 0.75
SOFT_KNEE_SHADOW_SLOPE = 1.4
SOFT_KNEE_MID_SLOPE = 0.9
SOFT_KNEE_HIGHLIGHT_SLOPE = 0.5

# Preview constants
PREVIEW_SATURATION = 0.9
PREVIEW_DARKEN_BREAK = 0.7
PREVIEW_DARKEN_LOW_SLOPE = 0.7

# New default constants
DEFAULT_EXPOSURE = 1.0
DEFAULT_GAMMA = 1.0

MAKO_RAMP = [
    (0.00, 0, 0, 0),      # Pure Black Abyss
    (0.05, 2, 1, 3),      # Near Black
    (0.10, 4, 2, 6),      # Deep Dark
    (0.15, 49, 33, 64),
    (0.20, 56, 42, 84),
    (0.25, 62, 53, 107),
    (0.30, 65, 64, 130),
    (0.35, 62, 79, 148),
    (0.40, 57, 93, 156),
    (0.45, 54, 108, 160),
    (0.50, 53, 122, 162),
    (0.55, 52, 137, 166),
    (0.60, 52, 153, 170),
    (0.65, 55, 166, 172),
    (0.70, 63, 181, 173),
    (0.75, 75, 194, 173),
    (0.80, 101, 208, 173),
    (0.85, 136, 217, 177),
    (0.90, 156, 213, 184),
    (0.95, 176, 208, 194),
    (1.00, 168, 195, 182),
]

LUMA_RED = 0.2126
LUMA_GREEN = 0.7152
LUMA_BLUE = 0.0722


def colorize_depth_numpy(
    depths: np.ndarray, ramp_colors: np.ndarray, depth_min: float, depth_max: float
) -> np.ndarray:
    """
    Colorize a depth map using a provided color ramp.
    depths: (H, W) or (1, H, W) float32
    ramp_colors: (N, 3) float32 [0, 1]
    depth_min/max: range for the ramp
    Returns: (3, H, W) float32 [0, 1]
    """
    if depths.ndim == 3:
        depths = depths[0]

    frac = np.clip((depths - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    ramp_fracs = np.linspace(0.0, 1.0, len(ramp_colors))

    r = np.interp(frac, ramp_fracs, ramp_colors[:, 0])
    g = np.interp(frac, ramp_fracs, ramp_colors[:, 1])
    b = np.interp(frac, ramp_fracs, ramp_colors[:, 2])

    return np.stack([r, g, b])


def apply_soft_knee_numpy(
    arr: np.ndarray,
    shadow_break: float = SOFT_KNEE_SHADOW_BREAK,
    highlight_break: float = SOFT_KNEE_HIGHLIGHT_BREAK,
    shadow_slope: float = SOFT_KNEE_SHADOW_SLOPE,
    mid_slope: float = SOFT_KNEE_MID_SLOPE,
    highlight_slope: float = SOFT_KNEE_HIGHLIGHT_SLOPE,
    exposure: float = DEFAULT_EXPOSURE,
) -> np.ndarray:
    """Apply soft-knee tone mapping using NumPy."""
    arr = np.clip(arr * exposure, 0.0, 1.0)

    shadow_output = shadow_break * shadow_slope
    highlight_output = shadow_output + (highlight_break - shadow_break) * mid_slope

    out = np.zeros_like(arr)

    # Shadow region
    mask_shadow = arr < shadow_break
    out[mask_shadow] = arr[mask_shadow] * shadow_slope

    # Mid region
    mask_mid = (arr >= shadow_break) & (arr < highlight_break)
    out[mask_mid] = shadow_output + (arr[mask_mid] - shadow_break) * mid_slope

    # Highlight region
    mask_highlight = arr >= highlight_break
    out[mask_highlight] = (
        highlight_output + (arr[mask_highlight] - highlight_break) * highlight_slope
    )

    return cast(np.ndarray, np.clip(out, 0.0, 1.0))


def apply_preview_correction_numpy(
    rgb_arr: np.ndarray,
    saturation: float = PREVIEW_SATURATION,
    darken_break: float = PREVIEW_DARKEN_BREAK,
    low_slope: float = PREVIEW_DARKEN_LOW_SLOPE,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Apply mild desaturation and preview darkening using NumPy. Expects (C, H, W) float32 [0,1]."""
    # 1. Gamma
    if gamma != 1.0:
        rgb_arr = np.power(np.clip(rgb_arr, 0.0, 1.0), 1.0 / gamma)

    # 2. Calculate luminance
    luma = LUMA_RED * rgb_arr[0] + LUMA_GREEN * rgb_arr[1] + LUMA_BLUE * rgb_arr[2]

    # 3. Desaturate
    out = luma + (rgb_arr - luma) * saturation

    # 4. Darken curve
    high_slope = (1.0 - (darken_break * low_slope)) / (1.0 - darken_break)
    break_output = darken_break * low_slope

    mask_low = out < darken_break
    out[mask_low] = out[mask_low] * low_slope
    out[~mask_low] = break_output + (out[~mask_low] - darken_break) * high_slope

    return cast(np.ndarray, np.clip(out, 0.0, 1.0))


@dataclass(frozen=True)
class TilingArtifacts:
    """Paths created during tiling so the caller can preserve or clean them up."""

    final_vrt: str
    cleanup_paths: List[str]


def get_web_mercator_bounds(
    z: int, x: int, y: int
) -> Tuple[float, float, float, float]:
    """Calculate Web Mercator (EPSG:3857) bounds for a given XYZ tile."""
    n = 2.0**z
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    # Latitude calculation (Slippy map tilenames)
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    def lonlat_to_3857(lon: float, lat: float) -> Tuple[float, float]:
        x_meters = lon * WEB_MERCATOR_LIMIT / 180
        y_meters = (
            math.log(math.tan((90 + lat) * math.pi / 360))
            / (math.pi / 180)
            * WEB_MERCATOR_LIMIT
            / 180
        )
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
    max_t = (2**zoom) - 1

    # We use a small epsilon to avoid including the next tile if the boundary
    # just barely touches it due to floating point precision.
    eps = 1e-8
    tx_min, ty_min = meters_to_tile(minx + eps, maxy - eps, zoom)
    tx_max, ty_max = meters_to_tile(maxx - eps, miny + eps, zoom)

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


def proj_win_to_src_win(
    dataset: gdal.Dataset, proj_win: ProjWin
) -> Tuple[int, int, int, int]:
    """Convert a georeferenced projWin into a clipped pixel srcWin for a north-up raster."""
    inv_gt = gdal.InvGeoTransform(dataset.GetGeoTransform())
    px_ul, py_ul = gdal.ApplyGeoTransform(inv_gt, proj_win[0], proj_win[1])
    px_lr, py_lr = gdal.ApplyGeoTransform(inv_gt, proj_win[2], proj_win[3])

    xoff = max(0, math.floor(min(px_ul, px_lr)))
    yoff = max(0, math.floor(min(py_ul, py_lr)))
    xend = min(dataset.RasterXSize, math.ceil(max(px_ul, px_lr)))
    yend = min(dataset.RasterYSize, math.ceil(max(py_ul, py_lr)))

    return xoff, yoff, max(0, xend - xoff), max(0, yend - yoff)


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
                f"description={options['description']}",
            ],
            creationOptions=[
                f"NAME={options['name']}",
                f"DESCRIPTION={options['description']}",
                "TYPE=baselayer",
                f"TILE_FORMAT={'JPEG' if format.lower() == 'jpg' else format.upper()}",
                f"QUALITY={options['quality']}",
                f"RESAMPLING={options['resample_alg'] if options['resample_alg'] != 'gauss' else 'bilinear'}",
                f"BLOCKSIZE={options.get('blocksize', 512)}",
                "ZOOM_LEVEL_STRATEGY=UPPER",
            ],
        )

        if format.lower() == "webp":
            gdal.SetConfigOption("WEBP_LEVEL", str(options["quality"]))

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

    # Sort chunks to ensure consistent merging
    input_mbtiles = sorted([f for f in input_mbtiles if f and os.path.exists(f)])
    if not input_mbtiles:
        return

    shutil.copyfile(input_mbtiles[0], output_mbtiles)

    conn = sqlite3.connect(output_mbtiles)
    conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")

    # Check if the output database has the 'map' table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='map'")
    output_has_map = cursor.fetchone() is not None

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
            raise RuntimeError(
                f"Failed to attach {db_path} after retries due to database locks"
            )

        try:
            # Check which tables exist in the source chunk
            cursor.execute(
                f"SELECT name FROM {alias}.sqlite_master WHERE type='table' AND name='map'"
            )
            source_has_map = cursor.fetchone() is not None

            if output_has_map and source_has_map:
                cursor.execute(f"INSERT OR IGNORE INTO map SELECT * FROM {alias}.map")
                cursor.execute(
                    f"INSERT OR IGNORE INTO images SELECT * FROM {alias}.images"
                )
            else:
                # Fallback to inserting into the 'tiles' view/table
                # Note: This might not work if 'tiles' is a view without an INSTEAD OF trigger,
                # but most chunks should have the same schema.
                cursor.execute(
                    f"INSERT OR IGNORE INTO tiles SELECT * FROM {alias}.tiles"
                )
            conn.commit()
        except sqlite3.OperationalError as e:
            # If inserting into tiles failed and we have no map table, it's a real error
            print(f"Warning: Error merging {db_path}: {e}")
        finally:
            if attached:
                try:
                    cursor.execute(f"DETACH DATABASE {alias}")
                except sqlite3.OperationalError:
                    pass
                try:
                    os.remove(db_path)
                except OSError:
                    pass

    conn.commit()
    conn.close()


def set_metadata_value(cursor: sqlite3.Cursor, name: str, value: str) -> None:
    """Replace a metadata entry in MBTiles, which does not enforce unique keys."""
    cursor.execute("DELETE FROM metadata WHERE name = ?", (name,))
    cursor.execute(
        "INSERT INTO metadata (name, value) VALUES (?, ?)",
        (name, value),
    )


def finalize_mbtiles_metadata(mbtiles_path: str) -> None:
    """Update MBTiles metadata with actual zoom levels and bounds."""
    conn = sqlite3.connect(mbtiles_path)
    cursor = conn.cursor()

    cursor.execute("SELECT min(zoom_level), max(zoom_level) FROM tiles")
    minz, maxz = cursor.fetchone()
    if minz is not None:
        set_metadata_value(cursor, "minzoom", str(minz))
        set_metadata_value(cursor, "maxzoom", str(maxz))

        cursor.execute(
            "SELECT min(tile_column), max(tile_column), min(tile_row), max(tile_row) FROM tiles WHERE zoom_level = ?",
            (maxz,),
        )
        min_c, max_c, min_r, max_r = cursor.fetchone()

        if min_c is not None:
            def tms_to_lonlat(z: int, x: int, y_tms: int) -> Tuple[float, float]:
                n = 2.0**z
                y_xyz = int(n - 1 - y_tms)
                lon = x / n * 360.0 - 180.0
                lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_xyz / n)))
                lat = math.degrees(lat_rad)
                return lon, lat

            w, n_lat = tms_to_lonlat(maxz, min_c, max_r)
            e, s_lat = tms_to_lonlat(maxz, max_c + 1, min_r - 1)
            bounds_str = f"{w},{s_lat},{e},{n_lat}"
            set_metadata_value(cursor, "bounds", bounds_str)
            set_metadata_value(
                cursor,
                "center",
                f"{(w + e) / 2},{(s_lat + n_lat) / 2},{maxz}",
            )

    conn.commit()
    conn.close()


def run_tiling_simplified(
    input_vrt: str, output_mbtiles: str, options: Dict[str, Any]
) -> TilingArtifacts:
    """Simplified tiling from a pre-processed Byte VRT."""
    unique_id = options.get("unique_id", "tiles")
    chunk_zoom = options.get("chunk_zoom", 4)
    tile_format = options.get("format", "webp")

    # Determine chunks from the Byte VRT.
    ds = gdal.Open(input_vrt)
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
            tasks.append(
                (input_vrt, chunk_file, tile_format, options, (ulx, uly, lrx, lry))
            )

    # Parallel execution.
    if tasks:
        num_workers = options.get("processes", 1)
        print(
            f"Processing {len(tasks)} chunk(s) at zoom {chunk_zoom} with {num_workers} worker(s)..."
        )
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk, tasks))
        else:
            for task in tasks:
                process_chunk(task)

    # Merge chunks.
    merge_mbtiles(output_mbtiles, chunk_files)

    # Refresh metadata before building overviews so GDAL sees the merged extent,
    # not just the bounds from the first copied chunk.
    finalize_mbtiles_metadata(output_mbtiles)

    # Build overviews (all levels from maxzoom down to 0)
    print("Building overviews...")
    gdaladdo_cmd = [
        "gdaladdo",
        "-r",
        options["resample_alg"] if options["resample_alg"] != "gauss" else "bilinear",
        "--config",
        "GDAL_NUM_THREADS",
        "ALL_CPUS",
        output_mbtiles,
    ]
    subprocess.run(gdaladdo_cmd, check=True)

    # Finalize metadata again after gdaladdo adds lower zoom tiles.
    finalize_mbtiles_metadata(output_mbtiles)

    return TilingArtifacts(final_vrt=input_vrt, cleanup_paths=chunk_files)
