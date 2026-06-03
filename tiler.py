import math
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
from common import build_staged_path, file_has_content, publish_staged_path, remove_if_exists
from osgeo import gdal
from PIL import Image

gdal.UseExceptions()

WEB_MERCATOR_LIMIT = 20037508.342789244
WEB_MERCATOR_MAX_LAT = 85.0511287798066
TEBounds = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) - for Warp/BuildVRT
ChunkTask = Tuple[str, str, str, Dict[str, Any], TEBounds]

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
PREVIEW_DARKEN_MID_SLOPE = 1.0

# New default constants
DEFAULT_EXPOSURE = 1.0
DEFAULT_GAMMA = 1.0
DEFAULT_SHOULDER = 1.0
TERRARIUM_OFFSET = 32768.0
TERRARIUM_MAX_VALUE = 65535.99609375

MAKO_RAMP = [
    (0.00, 0, 8, 37),
    (0.05, 5, 16, 46),
    (0.10, 10, 24, 55),
    (0.15, 25, 39, 76),
    (0.20, 31, 49, 88),
    (0.25, 37, 60, 101),
    (0.30, 42, 71, 114),
    (0.35, 46, 82, 127),
    (0.40, 49, 93, 136),
    (0.45, 53, 105, 145),
    (0.50, 58, 116, 154),
    (0.55, 62, 127, 163),
    (0.60, 68, 142, 175),
    (0.65, 77, 158, 189),
    (0.70, 86, 173, 202),
    (0.75, 95, 186, 214),
    (0.80, 104, 196, 224),
    (0.85, 113, 205, 233),
    (0.90, 118, 210, 241),
    (0.95, 121, 214, 247),
    (1.00, 121, 217, 251),
]

LUMA_RED = 0.2126
LUMA_GREEN = 0.7152
LUMA_BLUE = 0.0722
DEFAULT_MEMORY_RESERVE_BYTES = 1 << 30


def get_available_memory_bytes() -> int:
    """Return the current available system memory in bytes when it can be detected."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return max(0, int(parts[1]) * 1024)
                    break
    except OSError:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0

    if page_size <= 0 or available_pages <= 0:
        return 0
    return page_size * available_pages


def compute_in_memory_pixel_limit(
    bytes_per_pixel: int,
    *,
    usage_fraction: float,
    fallback_pixels: int,
    reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
    min_pixels: int = 1,
    max_pixels: int | None = None,
) -> int:
    """Convert available-memory headroom into a runtime pixel budget."""
    if bytes_per_pixel <= 0:
        raise ValueError("bytes_per_pixel must be positive")
    if not 0.0 < usage_fraction <= 1.0:
        raise ValueError("usage_fraction must be between 0 and 1")
    if fallback_pixels <= 0:
        raise ValueError("fallback_pixels must be positive")
    if min_pixels <= 0:
        raise ValueError("min_pixels must be positive")
    if max_pixels is not None and max_pixels < min_pixels:
        raise ValueError("max_pixels must be at least min_pixels")

    available_bytes = get_available_memory_bytes()
    if available_bytes <= 0:
        pixel_limit = fallback_pixels
    else:
        usable_bytes = available_bytes if available_bytes <= reserve_bytes else available_bytes - reserve_bytes
        budget_bytes = max(int(usable_bytes * usage_fraction), bytes_per_pixel * min_pixels)
        pixel_limit = max(min_pixels, budget_bytes // bytes_per_pixel)

    if max_pixels is not None:
        pixel_limit = min(pixel_limit, max_pixels)
    return max(min_pixels, pixel_limit)


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

    r = np.interp(frac, ramp_fracs, ramp_colors[:, 0]).astype(np.float32, copy=False)
    g = np.interp(frac, ramp_fracs, ramp_colors[:, 1]).astype(np.float32, copy=False)
    b = np.interp(frac, ramp_fracs, ramp_colors[:, 2]).astype(np.float32, copy=False)

    return np.stack([r, g, b]).astype(np.float32, copy=False)


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

    return apply_piecewise_linear_curve_numpy(
        arr,
        shadow_break=shadow_break,
        highlight_break=highlight_break,
        shadow_slope=shadow_slope,
        mid_slope=mid_slope,
        highlight_slope=highlight_slope,
    )


def apply_piecewise_linear_curve_numpy(
    arr: np.ndarray,
    *,
    shadow_break: float,
    highlight_break: float,
    shadow_slope: float,
    mid_slope: float,
    highlight_slope: float,
) -> np.ndarray:
    """Apply a monotonic three-segment linear curve using NumPy."""
    if shadow_break < 0.0 or highlight_break > 1.0 or shadow_break > highlight_break:
        raise ValueError("piecewise breaks must satisfy 0 <= shadow_break <= highlight_break <= 1")
    if shadow_slope < 0.0 or mid_slope < 0.0 or highlight_slope < 0.0:
        raise ValueError("piecewise slopes must be non-negative")

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


def derive_piecewise_high_slope(
    shadow_break: float,
    highlight_break: float,
    shadow_slope: float,
    mid_slope: float,
) -> float:
    """Resolve a highlight slope that keeps the curve anchored at x=1 -> y=1."""
    if shadow_break < 0.0 or highlight_break > 1.0 or shadow_break > highlight_break:
        raise ValueError("piecewise breaks must satisfy 0 <= shadow_break <= highlight_break <= 1")
    if shadow_slope < 0.0 or mid_slope < 0.0:
        raise ValueError("piecewise slopes must be non-negative")
    if highlight_break >= 1.0:
        return 1.0

    shadow_output = shadow_break * shadow_slope
    highlight_output = shadow_output + (highlight_break - shadow_break) * mid_slope
    derived_slope = (1.0 - highlight_output) / (1.0 - highlight_break)
    if derived_slope < 0.0:
        raise ValueError(
            "piecewise grading settings require an explicit highlight slope because the derived slope would be negative"
        )
    return derived_slope


def apply_highlight_shoulder_numpy(
    arr: np.ndarray,
    *,
    shoulder: float,
    start: float,
) -> np.ndarray:
    """Apply a mirrored gamma curve only to the upper tonal range."""
    clipped = np.clip(arr, 0.0, 1.0)
    clipped_start = float(np.clip(start, 0.0, 1.0))
    if shoulder == 1.0 or clipped_start >= 1.0:
        return cast(np.ndarray, clipped)

    normalized = np.clip((clipped - clipped_start) / (1.0 - clipped_start), 0.0, 1.0)
    adjusted = 1.0 - np.power(1.0 - normalized, shoulder)
    out = np.where(
        clipped > clipped_start,
        clipped_start + adjusted * (1.0 - clipped_start),
        clipped,
    )
    return cast(np.ndarray, np.clip(out, 0.0, 1.0))


def apply_preview_correction_numpy(
    rgb_arr: np.ndarray,
    saturation: float = PREVIEW_SATURATION,
    darken_break: float = PREVIEW_DARKEN_BREAK,
    low_slope: float = PREVIEW_DARKEN_LOW_SLOPE,
    gamma: float = DEFAULT_GAMMA,
    shoulder: float = DEFAULT_SHOULDER,
    highlight_break: float | None = None,
    mid_slope: float = PREVIEW_DARKEN_MID_SLOPE,
    high_slope: float | None = None,
) -> np.ndarray:
    """Apply mild desaturation and preview darkening using NumPy. Expects (C, H, W) float32 [0,1]."""
    resolved_highlight_break = darken_break if highlight_break is None else highlight_break

    # 1. Gamma
    if gamma != 1.0:
        rgb_arr = np.power(np.clip(rgb_arr, 0.0, 1.0), 1.0 / gamma)

    # 2. Shoulder
    if shoulder != 1.0:
        rgb_arr = apply_highlight_shoulder_numpy(
            rgb_arr,
            shoulder=shoulder,
            start=max(0.5, resolved_highlight_break),
        )

    # 3. Calculate luminance
    luma = LUMA_RED * rgb_arr[0] + LUMA_GREEN * rgb_arr[1] + LUMA_BLUE * rgb_arr[2]

    # 4. Desaturate
    out = luma + (rgb_arr - luma) * saturation

    # 5. Grading curve
    resolved_high_slope = (
        derive_piecewise_high_slope(
            darken_break,
            resolved_highlight_break,
            low_slope,
            mid_slope,
        )
        if high_slope is None
        else high_slope
    )

    return apply_piecewise_linear_curve_numpy(
        out,
        shadow_break=darken_break,
        highlight_break=resolved_highlight_break,
        shadow_slope=low_slope,
        mid_slope=mid_slope,
        highlight_slope=resolved_high_slope,
    )


@dataclass(frozen=True)
class TilingArtifacts:
    """Paths created during tiling so the caller can preserve or clean them up."""

    final_vrt: str
    cleanup_paths: List[str]


def encode_terrarium_numpy(
    elevations: np.ndarray, nodata_value: Optional[float] = None
) -> np.ndarray:
    """Encode elevations in meters into Terrarium RGB bytes."""
    values = np.asarray(elevations, dtype=np.float32)
    valid_mask = np.isfinite(values)
    if nodata_value is not None:
        valid_mask &= ~np.isclose(values, nodata_value)

    shifted = np.zeros_like(values, dtype=np.float32)
    shifted[valid_mask] = np.clip(
        values[valid_mask] + TERRARIUM_OFFSET,
        0.0,
        TERRARIUM_MAX_VALUE,
    )

    integer_part = np.floor(shifted).astype(np.int64, copy=False)
    red = (integer_part // 256).astype(np.uint8, copy=False)
    green = (integer_part % 256).astype(np.uint8, copy=False)
    blue = np.floor((shifted - integer_part) * 256.0).astype(np.uint8, copy=False)
    return np.stack((red, green, blue))


def write_terrarium_geotiff(source_dataset: gdal.Dataset, output_path: str) -> str:
    """Materialize a Terrarium RGB GeoTIFF from a single-band elevation dataset."""
    if source_dataset.RasterCount < 1:
        raise RuntimeError("Terrarium encoding requires at least one raster band")

    source_band = source_dataset.GetRasterBand(1)
    nodata_value = source_band.GetNoDataValue()
    block_width, block_height = source_band.GetBlockSize()
    if block_width <= 0:
        block_width = 512
    if block_height <= 0:
        block_height = 512

    driver = gdal.GetDriverByName("GTiff")
    terrarium_ds = driver.Create(
        output_path,
        source_dataset.RasterXSize,
        source_dataset.RasterYSize,
        3,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ],
    )
    if terrarium_ds is None:
        raise RuntimeError(f"Could not create Terrarium GeoTIFF: {output_path}")

    terrarium_ds.SetProjection(source_dataset.GetProjection())
    terrarium_ds.SetGeoTransform(source_dataset.GetGeoTransform())

    for band_index, color_name in enumerate(("RedBand", "GreenBand", "BlueBand"), start=1):
        terrarium_band = terrarium_ds.GetRasterBand(band_index)
        terrarium_band.SetColorInterpretation(getattr(gdal, f"GCI_{color_name}"))

    terrarium_bands = [terrarium_ds.GetRasterBand(index) for index in range(1, 4)]
    for yoff in range(0, source_dataset.RasterYSize, block_height):
        bh = min(block_height, source_dataset.RasterYSize - yoff)
        for xoff in range(0, source_dataset.RasterXSize, block_width):
            bw = min(block_width, source_dataset.RasterXSize - xoff)
            elevations = source_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
            rgb = encode_terrarium_numpy(elevations, nodata_value)
            for band_index, terrarium_band in enumerate(terrarium_bands):
                terrarium_band.WriteArray(rgb[band_index], xoff=xoff, yoff=yoff)

    terrarium_ds.FlushCache()
    terrarium_ds = None
    return output_path


def lonlat_to_3857(lon: float, lat: float) -> Tuple[float, float]:
    """Convert lon/lat degrees to Web Mercator meters."""
    clamped_lat = min(max(lat, -WEB_MERCATOR_MAX_LAT), WEB_MERCATOR_MAX_LAT)
    x_meters = lon * WEB_MERCATOR_LIMIT / 180
    y_meters = (
        math.log(math.tan((90 + clamped_lat) * math.pi / 360))
        / (math.pi / 180)
        * WEB_MERCATOR_LIMIT
        / 180
    )
    return x_meters, y_meters


def lonlat_bbox_to_mercator_bounds(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> TEBounds:
    """Convert a WGS84 bbox into (minx, miny, maxx, maxy) Web Mercator bounds."""
    west_x, south_y = lonlat_to_3857(min_lon, min_lat)
    east_x, north_y = lonlat_to_3857(max_lon, max_lat)
    return min(west_x, east_x), min(south_y, north_y), max(west_x, east_x), max(
        south_y, north_y
    )


def parse_bbox_string(bbox: str) -> TEBounds:
    """Parse a min_lon,min_lat,max_lon,max_lat string into floats."""
    values = tuple(float(value) for value in bbox.split(","))
    if len(values) != 4:
        raise ValueError("bbox must contain four comma-separated values")
    return values


def web_mercator_pixel_size(zoom: int) -> float:
    """Return the meters-per-pixel value for a Web Mercator XYZ zoom level."""
    return float((WEB_MERCATOR_LIMIT * 2) / (256 * (2**zoom)))


def web_mercator_pixel_size_for_tile_size(zoom: int, tile_size: int) -> float:
    """Return the meters-per-pixel needed for a nominal XYZ zoom and raster tile size."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    return float((WEB_MERCATOR_LIMIT * 2) / (tile_size * (2**zoom)))


def snap_bounds_to_pixel_grid(bounds: TEBounds, pixel_size: float) -> TEBounds:
    """Expand bounds (minx, miny, maxx, maxy) outward to the global Web Mercator pixel grid."""
    if pixel_size <= 0.0:
        raise ValueError("pixel_size must be positive")

    minx, miny, maxx, maxy = bounds
    snapped_minx = -WEB_MERCATOR_LIMIT + math.floor(
        (minx + WEB_MERCATOR_LIMIT) / pixel_size
    ) * pixel_size
    snapped_maxx = -WEB_MERCATOR_LIMIT + math.ceil(
        (maxx + WEB_MERCATOR_LIMIT) / pixel_size
    ) * pixel_size
    snapped_maxy = -WEB_MERCATOR_LIMIT + math.ceil(
        (maxy + WEB_MERCATOR_LIMIT) / pixel_size
    ) * pixel_size
    snapped_miny = -WEB_MERCATOR_LIMIT + math.floor(
        (miny + WEB_MERCATOR_LIMIT) / pixel_size
    ) * pixel_size
    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy


def get_web_mercator_bounds(
    z: int, x: int, y: int
) -> TEBounds:
    """Calculate Web Mercator (EPSG:3857) bounds for a given XYZ tile."""
    n = 2.0**z
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    # Latitude calculation (Slippy map tilenames)
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    x1, y1 = lonlat_to_3857(lon1, lat1)
    x2, y2 = lonlat_to_3857(lon2, lat2)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def meters_to_tile(x_meters: float, y_meters: float, z: int) -> Tuple[int, int]:
    """Convert Web Mercator meters to tile coordinates at zoom z."""
    res = WEB_MERCATOR_LIMIT * 2 / (2**z)
    tx = math.floor((x_meters + WEB_MERCATOR_LIMIT) / res)
    ty = math.floor((WEB_MERCATOR_LIMIT - y_meters) / res)
    return tx, ty


def get_dataset_bounds(dataset: gdal.Dataset) -> TEBounds:
    """Return dataset bounds as (minx, miny, maxx, maxy)."""
    gt = dataset.GetGeoTransform()
    # North-up assumed: gt[1] is xRes (>0), gt[5] is yRes (<0)
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * dataset.RasterXSize
    miny = maxy + gt[5] * dataset.RasterYSize
    return minx, miny, maxx, maxy


def get_chunk_tile_range(bounds: TEBounds, zoom: int) -> Tuple[int, int, int, int]:
    """Return inclusive XYZ tile indices covering raster bounds at the chunk zoom."""
    minx, miny, maxx, maxy = bounds
    max_t = (2**zoom) - 1

    # Nudge the sample point inward by a tiny fraction of a tile so exact
    # south/east boundaries stay in the touched tile even at Web Mercator
    # meter magnitudes where 1e-8 is smaller than floating-point resolution.
    res = WEB_MERCATOR_LIMIT * 2 / (2**zoom)
    eps = max(1e-8, res * 1e-9)
    tx_min, ty_min = meters_to_tile(minx + eps, maxy - eps, zoom)
    tx_max, ty_max = meters_to_tile(maxx - eps, miny + eps, zoom)

    return (
        max(0, min(tx_min, max_t)),
        max(0, min(ty_min, max_t)),
        max(0, min(tx_max, max_t)),
        max(0, min(ty_max, max_t)),
    )


def intersect_te_bounds(bounds_a: TEBounds, bounds_b: TEBounds) -> Optional[TEBounds]:
    """Calculate the intersection of two (minx, miny, maxx, maxy) bounds."""
    minx = max(bounds_a[0], bounds_b[0])
    miny = max(bounds_a[1], bounds_b[1])
    maxx = min(bounds_a[2], bounds_b[2])
    maxy = min(bounds_a[3], bounds_b[3])

    if minx >= maxx or miny >= maxy:
        return None

    return minx, miny, maxx, maxy


def build_tile_tree_tile_path(root_dir: str, z: int, x: int, y: int) -> str:
    """Return the deterministic z/x/y.webp path for one cached tile."""
    return os.path.join(root_dir, str(z), str(x), f"{y}.webp")


def save_webp_image(
    image: Image.Image,
    output_path: str,
    quality: int,
    *,
    lossless: bool = False,
) -> str:
    """Atomically write a PIL image as WebP."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    staged_output_path = build_staged_path(output_path)
    remove_if_exists(staged_output_path)
    save_kwargs: Dict[str, Any] = {"format": "WEBP"}
    if lossless:
        save_kwargs["lossless"] = True
        save_kwargs["quality"] = 100
    else:
        save_kwargs["quality"] = quality
    image.save(staged_output_path, **save_kwargs)
    return publish_staged_path(staged_output_path, output_path)


def render_dataset_tile(
    dataset: gdal.Dataset,
    bounds: TEBounds,
    tile_size: int,
    resample_alg: str,
) -> np.ndarray:
    """Read one Web Mercator XYZ tile from a dataset into a byte array."""
    def normalize_tile_array(tile_array: np.ndarray) -> np.ndarray:
        output = np.asarray(tile_array, dtype=np.uint8)
        if output.ndim == 2:
            output = output[np.newaxis, :, :]
        if output.shape[0] < 3:
            raise RuntimeError("Rendered tile must contain at least RGB bands")
        if output.shape[0] > 4:
            output = output[:4]
        return output

    def blank_tile(band_count: int) -> np.ndarray:
        return np.zeros((band_count, tile_size, tile_size), dtype=np.uint8)

    def overlap_to_tile_window(
        full_bounds: TEBounds,
        overlap_bounds: TEBounds,
    ) -> Tuple[int, int, int, int]:
        minx, miny, maxx, maxy = full_bounds
        overlap_minx, overlap_miny, overlap_maxx, overlap_maxy = overlap_bounds
        tile_width = maxx - minx
        tile_height = maxy - miny
        if tile_width <= 0.0 or tile_height <= 0.0:
            raise ValueError("Tile bounds must have positive width and height")

        xoff = int(round((overlap_minx - minx) * tile_size / tile_width))
        xend = int(round((overlap_maxx - minx) * tile_size / tile_width))
        yoff = int(round((maxy - overlap_maxy) * tile_size / tile_height))
        yend = int(round((maxy - overlap_miny) * tile_size / tile_height))

        xoff = max(0, min(tile_size, xoff))
        xend = max(0, min(tile_size, xend))
        yoff = max(0, min(tile_size, yoff))
        yend = max(0, min(tile_size, yend))
        return xoff, yoff, max(0, xend - xoff), max(0, yend - yoff)

    minx, miny, maxx, maxy = bounds
    overlap_bounds = intersect_te_bounds(bounds, get_dataset_bounds(dataset))
    default_band_count = min(max(dataset.RasterCount, 3), 4)
    if overlap_bounds is None:
        return blank_tile(default_band_count)

    dest_xoff = 0
    dest_yoff = 0
    dest_width = tile_size
    dest_height = tile_size
    if overlap_bounds != bounds:
        dest_xoff, dest_yoff, dest_width, dest_height = overlap_to_tile_window(
            bounds,
            overlap_bounds,
        )
        if dest_width <= 0 or dest_height <= 0:
            return blank_tile(default_band_count)
        minx, miny, maxx, maxy = overlap_bounds

    tile_ds = gdal.Translate(
        "",
        dataset,
        options=gdal.TranslateOptions(
            format="MEM",
            projWin=[minx, maxy, maxx, miny],
            width=dest_width,
            height=dest_height,
            resampleAlg=resample_alg if resample_alg != "gauss" else "bilinear",
        ),
    )
    if tile_ds is None:
        raise RuntimeError("Could not render tile into memory")
    tile_array = tile_ds.ReadAsArray()
    tile_ds = None
    if tile_array is None:
        raise RuntimeError("Could not read rendered tile data")
    output = normalize_tile_array(tile_array)
    if overlap_bounds == bounds:
        return output

    padded_output = blank_tile(output.shape[0])
    padded_output[
        :,
        dest_yoff : dest_yoff + output.shape[1],
        dest_xoff : dest_xoff + output.shape[2],
    ] = output
    return padded_output


def tile_array_to_image(tile_array: np.ndarray) -> Optional[Image.Image]:
    """Convert a rendered RGB(A) tile array into a PIL image, skipping empty alpha tiles."""
    if tile_array.shape[0] == 4 and not np.any(tile_array[3]):
        return None
    if tile_array.shape[0] == 4 and np.all(tile_array[3] == 255):
        tile_array = tile_array[:3]
    mode = "RGBA" if tile_array.shape[0] == 4 else "RGB"
    return Image.fromarray(np.moveaxis(tile_array, 0, -1), mode=mode)


def iter_dataset_tile_relpaths(dataset: gdal.Dataset, zoom: int) -> Iterator[str]:
    """Yield max-zoom z/x/y tile paths for a dataset's covered tile range."""
    bounds = get_dataset_bounds(dataset)
    tx_min, ty_min, tx_max, ty_max = get_chunk_tile_range(bounds, zoom)
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            yield os.path.join(str(zoom), str(tx), f"{ty}.webp")


def iter_dataset_webp_tile_images(
    dataset: gdal.Dataset,
    zoom: int,
    tile_size: int,
    resample_alg: str,
) -> Iterator[tuple[str, Image.Image]]:
    """Yield max-zoom z/x/y tile images for a dataset without buffering the full tile tree."""
    for relative_path in iter_dataset_tile_relpaths(dataset, zoom):
        _parsed_zoom, tx, ty_filename = os.path.normpath(relative_path).split(os.sep)
        tile_array = render_dataset_tile(
            dataset,
            get_web_mercator_bounds(zoom, int(tx), int(os.path.splitext(ty_filename)[0])),
            tile_size,
            resample_alg,
        )
        image = tile_array_to_image(tile_array)
        if image is None:
            continue
        yield relative_path, image


def render_raster_to_webp_tile_images(
    input_raster: str | gdal.Dataset,
    zoom: int,
    tile_size: int,
    resample_alg: str,
) -> Dict[str, Image.Image]:
    """Render a raster into max-zoom z/x/y tile images kept in memory."""
    close_dataset = False
    if isinstance(input_raster, str):
        dataset = gdal.Open(input_raster)
        close_dataset = True
        if dataset is None:
            raise RuntimeError(f"Could not open raster for tile rendering: {input_raster}")
    else:
        dataset = input_raster

    try:
        return dict(iter_dataset_webp_tile_images(dataset, zoom, tile_size, resample_alg))
    finally:
        if close_dataset:
            dataset = None


def export_raster_to_webp_tree(
    input_raster: str,
    output_dir: str,
    zoom: int,
    tile_size: int,
    quality: int,
    resample_alg: str,
    *,
    lossless: bool = False,
) -> List[str]:
    """Render a Web Mercator raster into max-zoom z/x/y.webp tiles."""
    dataset = gdal.Open(input_raster)
    if dataset is None:
        raise RuntimeError(f"Could not open raster for tile rendering: {input_raster}")

    written_tiles: List[str] = []
    try:
        for relative_path, image in iter_dataset_webp_tile_images(
            dataset,
            zoom,
            tile_size,
            resample_alg,
        ):
            try:
                output_path = os.path.join(output_dir, relative_path)
                save_webp_image(image, output_path, quality, lossless=lossless)
                written_tiles.append(relative_path)
            finally:
                image.close()
        written_tiles.sort()
        return written_tiles
    finally:
        dataset = None


def iter_raster_tile_relpaths(input_raster: str, zoom: int) -> List[str]:
    """Return all max-zoom z/x/y.webp paths touched by a raster."""
    dataset = gdal.Open(input_raster)
    if dataset is None:
        raise RuntimeError(f"Could not open raster for tile enumeration: {input_raster}")

    try:
        bounds = get_dataset_bounds(dataset)
        tx_min, ty_min, tx_max, ty_max = get_chunk_tile_range(bounds, zoom)
        relpaths: List[str] = []
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                relpaths.append(os.path.join(str(zoom), str(tx), f"{ty}.webp"))
        return relpaths
    finally:
        dataset = None


def compose_webp_tile_from_rasters(
    input_rasters: Sequence[str],
    output_path: str,
    zoom: int,
    tx: int,
    ty: int,
    tile_size: int,
    quality: int,
    resample_alg: str,
) -> bool:
    """Render and composite one final WebP tile entirely in memory."""
    tile_bounds = get_web_mercator_bounds(zoom, tx, ty)
    composed_rgba: Optional[Image.Image] = None

    for input_raster in input_rasters:
        dataset = gdal.Open(input_raster)
        if dataset is None:
            raise RuntimeError(f"Could not open raster for tile compositing: {input_raster}")
        try:
            tile_array = render_dataset_tile(dataset, tile_bounds, tile_size, resample_alg)
        finally:
            dataset = None

        image = tile_array_to_image(tile_array)
        if image is None:
            continue

        source_rgba = image.convert("RGBA")
        if composed_rgba is None:
            composed_rgba = source_rgba
        else:
            composed_rgba = Image.alpha_composite(composed_rgba, source_rgba)

    if composed_rgba is None:
        return False

    if composed_rgba.getchannel("A").getextrema() == (255, 255):
        final_image: Image.Image = composed_rgba.convert("RGB")
    else:
        final_image = composed_rgba
    save_webp_image(final_image, output_path, quality, lossless=False)
    return True


def stage_raster_to_webp_tree_commit(
    input_raster: str,
    output_dir: str,
    staging_dir: str,
    zoom: int,
    tile_size: int,
    quality: int,
    resample_alg: str,
    *,
    source_under_existing: bool = False,
) -> List[str]:
    """Render a raster into staged final WebP tiles composed against the current output tree."""
    dataset = gdal.Open(input_raster)
    if dataset is None:
        raise RuntimeError(f"Could not open raster for staged tile commit: {input_raster}")

    try:
        bounds = get_dataset_bounds(dataset)
        tx_min, ty_min, tx_max, ty_max = get_chunk_tile_range(bounds, zoom)
        staged_tiles: List[str] = []
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile_array = render_dataset_tile(
                    dataset,
                    get_web_mercator_bounds(zoom, tx, ty),
                    tile_size,
                    resample_alg,
                )
                image = tile_array_to_image(tile_array)
                if image is None:
                    continue

                relative_path = os.path.join(str(zoom), str(tx), f"{ty}.webp")
                destination_path = os.path.join(output_dir, relative_path)
                staged_output_path = os.path.join(staging_dir, relative_path)

                final_image: Image.Image = image
                if file_has_content(destination_path):
                    with Image.open(destination_path) as destination_image:
                        destination_rgba = destination_image.convert("RGBA")
                    source_rgba = image.convert("RGBA")
                    if source_under_existing:
                        composed = Image.alpha_composite(source_rgba, destination_rgba)
                    else:
                        composed = Image.alpha_composite(destination_rgba, source_rgba)
                    if composed.getchannel("A").getextrema() == (255, 255):
                        final_image = composed.convert("RGB")
                    else:
                        final_image = composed

                save_webp_image(final_image, staged_output_path, quality, lossless=False)
                staged_tiles.append(relative_path)
        return staged_tiles
    finally:
        dataset = None


def publish_staged_webp_tree_commit(
    staging_dir: str,
    output_dir: str,
    tile_relpaths: Sequence[str],
) -> int:
    """Publish a staged final-tile commit into the live output tree."""
    published_count = 0
    for relative_path in tile_relpaths:
        staged_path = os.path.join(staging_dir, relative_path)
        destination_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        if file_has_content(staged_path):
            publish_staged_path(staged_path, destination_path)
        elif not file_has_content(destination_path):
            raise RuntimeError(
                f"Missing staged and destination tile during publish recovery: {relative_path}"
            )
        published_count += 1
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    return published_count


def iter_tile_tree_paths(root_dir: str) -> Iterator[str]:
    """Yield z/x/y.webp paths relative to a tile tree root in deterministic order."""
    if not os.path.isdir(root_dir):
        return
    for current_root, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".webp"):
                absolute_path = os.path.join(current_root, filename)
                yield os.path.relpath(absolute_path, root_dir)


def merge_webp_trees(
    source_dirs: List[str],
    output_dir: str,
    quality: int,
) -> int:
    """Compose multiple contributor tile trees into one final z/x/y.webp tree."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    merged_count = 0
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for relative_path in iter_tile_tree_paths(source_dir):
            source_path = os.path.join(source_dir, relative_path)
            destination_path = os.path.join(output_dir, relative_path)
            with Image.open(source_path) as source_image:
                source_rgba = source_image.convert("RGBA")
            if file_has_content(destination_path):
                with Image.open(destination_path) as destination_image:
                    destination_rgba = destination_image.convert("RGBA")
                composed = Image.alpha_composite(destination_rgba, source_rgba)
            else:
                composed = source_rgba
            if composed.getchannel("A").getextrema() == (255, 255):
                final_image: Image.Image = composed.convert("RGB")
            else:
                final_image = composed
            save_webp_image(final_image, destination_path, quality, lossless=False)
            merged_count += 1
    return merged_count


def initialize_mbtiles(cursor: sqlite3.Cursor) -> None:
    """Create the basic MBTiles schema used by the custom tile-tree ingester."""
    cursor.executescript(
        """
        CREATE TABLE metadata (name TEXT, value TEXT);
        CREATE TABLE tiles (
            zoom_level INTEGER NOT NULL,
            tile_column INTEGER NOT NULL,
            tile_row INTEGER NOT NULL,
            tile_data BLOB NOT NULL
        );
        CREATE UNIQUE INDEX tile_index
        ON tiles (zoom_level, tile_column, tile_row);
        """
    )


def build_mbtiles_from_webp_tree(
    input_dir: str,
    output_mbtiles: str,
    *,
    name: str,
    description: str,
    maxzoom: int,
    bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
) -> int:
    """Build a flat MBTiles database by copying cached WebP tile bytes as-is."""
    staged_output_path = build_staged_path(output_mbtiles)
    remove_if_exists(staged_output_path)
    conn = sqlite3.connect(staged_output_path)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        initialize_mbtiles(cursor)
        metadata_values = {
            "name": name,
            "description": description,
            "type": "baselayer",
            "version": "1",
            "format": "webp",
            "minzoom": str(maxzoom),
            "maxzoom": str(maxzoom),
        }
        if bounds_wgs84 is not None:
            metadata_values["bounds"] = ",".join(str(value) for value in bounds_wgs84)
        for metadata_name, metadata_value in metadata_values.items():
            cursor.execute(
                "INSERT INTO metadata (name, value) VALUES (?, ?)",
                (metadata_name, metadata_value),
            )

        inserted_tiles = 0
        for relative_path in iter_tile_tree_paths(input_dir):
            parts = relative_path.split(os.sep)
            if len(parts) != 3:
                continue
            z_value, x_value, filename = parts
            y_stem, extension = os.path.splitext(filename)
            if extension.lower() != ".webp":
                continue
            zoom_level = int(z_value)
            tile_column = int(x_value)
            tile_y_xyz = int(y_stem)
            tile_row = (1 << zoom_level) - 1 - tile_y_xyz
            with open(os.path.join(input_dir, relative_path), "rb") as tile_file:
                tile_bytes = tile_file.read()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data)
                VALUES (?, ?, ?, ?)
                """,
                (zoom_level, tile_column, tile_row, tile_bytes),
            )
            inserted_tiles += 1

        conn.commit()
    finally:
        conn.close()
    finalize_mbtiles_metadata(staged_output_path)
    publish_staged_path(staged_output_path, output_mbtiles)
    return inserted_tiles


def build_mbtiles_overviews(mbtiles_path: str, resample_alg: str) -> None:
    """Populate lower zoom levels for an MBTiles archive using gdaladdo."""
    print("Building overviews...")
    gdaladdo_cmd = [
        "gdaladdo",
        "-r",
        resample_alg if resample_alg != "gauss" else "bilinear",
        "--config",
        "GDAL_NUM_THREADS",
        "ALL_CPUS",
        mbtiles_path,
    ]
    subprocess.run(gdaladdo_cmd, check=True)


def te_to_src_win(
    dataset: gdal.Dataset, te_bounds: TEBounds
) -> Tuple[int, int, int, int]:
    """Convert (minx, miny, maxx, maxy) into a clipped pixel srcWin for a north-up raster."""
    inv_gt = gdal.InvGeoTransform(dataset.GetGeoTransform())
    px_minx, py_miny = gdal.ApplyGeoTransform(inv_gt, te_bounds[0], te_bounds[1])
    px_maxx, py_maxy = gdal.ApplyGeoTransform(inv_gt, te_bounds[2], te_bounds[3])

    # For north-up images, py_maxy will be smaller than py_miny in pixel space
    xoff = max(0, math.floor(min(px_minx, px_maxx)))
    yoff = max(0, math.floor(min(py_miny, py_maxy)))
    xend = min(dataset.RasterXSize, math.ceil(max(px_minx, px_maxx)))
    yend = min(dataset.RasterYSize, math.ceil(max(py_miny, py_maxy)))

    return xoff, yoff, max(0, xend - xoff), max(0, yend - yoff)

def has_alpha_band(dataset: gdal.Dataset) -> bool:
    """Return whether the dataset advertises an explicit alpha band."""
    return any(
        dataset.GetRasterBand(index).GetColorInterpretation() == gdal.GCI_AlphaBand
        for index in range(1, dataset.RasterCount + 1)
    )


def process_chunk(args: ChunkTask) -> str:
    """Worker function for parallel gdal.Translate."""
    input_vrt, chunk_file, format, options, te_bounds = args
    ds = None
    temp_chunk_raster = chunk_file.replace(".mbtiles", ".tif")
    temp_chunk_rgb = chunk_file.replace(".mbtiles", "_rgb.tif")
    staged_chunk_file = build_staged_path(chunk_file)
    try:
        gdal.UseExceptions()

        # Open VRT to check intersection
        ds = gdal.Open(input_vrt)
        if ds is None:
            return ""

        dataset_bounds = get_dataset_bounds(ds)
        clipped_te_bounds = intersect_te_bounds(te_bounds, dataset_bounds)
        if clipped_te_bounds is None:
            return ""

        src_win = te_to_src_win(ds, clipped_te_bounds)
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

        chunk_source = temp_chunk_raster
        chunk_ds = gdal.Open(temp_chunk_raster)
        if chunk_ds is None:
            return ""
        if options.get("elevation_encoding") == "terrarium":
            if format.lower() != "png":
                raise RuntimeError("Terrarium DEM tiles must use PNG output")
            remove_if_exists(temp_chunk_rgb)
            write_terrarium_geotiff(chunk_ds, temp_chunk_rgb)
            chunk_source = temp_chunk_rgb
        elif has_alpha_band(chunk_ds):
            remove_if_exists(temp_chunk_rgb)
            rgb_ds = gdal.Translate(
                temp_chunk_rgb,
                temp_chunk_raster,
                options=gdal.TranslateOptions(format="GTiff", bandList=[1, 2, 3]),
            )
            if rgb_ds is None:
                return ""
            rgb_ds.FlushCache()
            rgb_ds = None
            chunk_source = temp_chunk_rgb
        chunk_ds = None

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
                "ZOOM_LEVEL_STRATEGY=LOWER",
            ],
        )

        if format.lower() == "webp":
            gdal.SetThreadLocalConfigOption("WEBP_LEVEL", str(options["quality"]))

        remove_if_exists(staged_chunk_file)
        gdal.Translate(staged_chunk_file, chunk_source, options=translate_options)
        publish_staged_path(staged_chunk_file, chunk_file)
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
        if os.path.exists(temp_chunk_rgb):
            os.remove(temp_chunk_rgb)
        remove_if_exists(staged_chunk_file)


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
    chunk_prefix = os.path.splitext(os.path.basename(output_mbtiles))[0] or "tiles"
    chunk_zoom = options.get("chunk_zoom", 4)
    tile_format = options.get("format", "webp")

    # Determine chunks from the Byte VRT.
    ds = gdal.Open(input_vrt)
    dataset_bounds = get_dataset_bounds(ds)
    ds = None

    requested_bounds = cast(TEBounds, options.get("chunk_bounds", dataset_bounds))
    bounds = intersect_te_bounds(requested_bounds, dataset_bounds)
    if bounds is None:
        raise RuntimeError("Requested chunk bounds do not intersect the input raster")

    tx_min, ty_min, tx_max, ty_max = get_chunk_tile_range(bounds, chunk_zoom)

    tasks: List[ChunkTask] = []
    chunk_files: List[str] = []
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            chunk_file = f".temp/{chunk_prefix}_chunk_{chunk_zoom}_{tx}_{ty}.mbtiles"
            chunk_files.append(chunk_file)

            if os.path.exists(chunk_file):
                continue

            te_bounds = get_web_mercator_bounds(chunk_zoom, tx, ty)
            tasks.append(
                (input_vrt, chunk_file, tile_format, options, te_bounds)
            )

    # Parallel execution.
    if tasks:
        num_workers = options.get("processes", 1)
        print(
            f"Processing {len(tasks)} chunk(s) at zoom {chunk_zoom} with {num_workers} worker(s)..."
        )
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk, tasks))
        else:
            for task in tasks:
                process_chunk(task)

    # Merge chunks.
    staged_output_mbtiles = build_staged_path(output_mbtiles)
    remove_if_exists(staged_output_mbtiles)
    merge_mbtiles(staged_output_mbtiles, chunk_files)

    # Refresh metadata before building overviews so GDAL sees the merged extent,
    # not just the bounds from the first copied chunk.
    finalize_mbtiles_metadata(staged_output_mbtiles)

    # Build overviews (all levels from maxzoom down to 0)
    build_mbtiles_overviews(staged_output_mbtiles, options["resample_alg"])

    # Finalize metadata again after gdaladdo adds lower zoom tiles.
    finalize_mbtiles_metadata(staged_output_mbtiles)
    publish_staged_path(staged_output_mbtiles, output_mbtiles)

    return TilingArtifacts(final_vrt=input_vrt, cleanup_paths=chunk_files)
