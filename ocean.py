#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Sequence, TypeVar
from xml.sax.saxutils import escape

import numpy as np
from common import (
    LiveProgressLine,
    build_output_namespace_dir,
    build_output_namespace,
    build_staged_path,
    ensure_directory,
    prepare_staged_path,
    file_has_content,
    format_eta,
    per_worker_warp_threads,
    print_settings_diff_warning,
    publish_staged_path,
    read_settings_file,
    remove_if_exists,
    write_settings_file,
    warp_thread_budget,
    warp_thread_options,
)
from osgeo import gdal
from scipy.ndimage import label

from tiler import (
    MAKO_RAMP,
    PREVIEW_DARKEN_MID_SLOPE,
    apply_preview_correction_numpy,
    apply_soft_knee_numpy,
    compute_in_memory_pixel_limit,
    colorize_depth_numpy,
    is_identity_value,
    lonlat_bbox_to_mercator_bounds,
    parse_bbox_string,
    snap_bounds_to_pixel_grid,
    web_mercator_pixel_size,
    web_mercator_pixel_size_for_tile_size,
    DEFAULT_BLACK_POINT,
    DEFAULT_VIBRANCE,
    DEFAULT_WHITE_POINT,
)

gdal.UseExceptions()

DEFAULT_GEBCO_ZIP = "gebco_2025_sub_ice_topo_geotiff.zip"
DEFAULT_OUTPUT = "ocean.tif"
GEBCO_OCEAN_NODATA = -32767.0
WEB_MERCATOR_WORLD_BOUNDS = (
    -20037508.342789244,
    -20037508.342789244,
    20037508.342789244,
    20037508.342789244,
)
DEFAULT_MAX_ZOOM = 13
SUPPORTED_MAX_ZOOMS = tuple(range(4, 21))
GTIFF_CREATION_OPTIONS = (
    "BIGTIFF=YES",
    "TILED=YES",
    "COMPRESS=ZSTD",
    "ZSTD_LEVEL=5",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)
OCEAN_DEFAULT_EXPOSURE = 1.0
OCEAN_DEFAULT_SHADOW_BREAK = 0.3
OCEAN_DEFAULT_HIGHLIGHT_BREAK = 0.75
OCEAN_DEFAULT_SHADOW_SLOPE = 1.4
OCEAN_DEFAULT_MID_SLOPE = 0.9
OCEAN_DEFAULT_HIGHLIGHT_SLOPE = 0.5
OCEAN_DEFAULT_GAMMA = 1.4
OCEAN_DEFAULT_SHOULDER = 1.0
OCEAN_DEFAULT_SATURATION = 1.0
OCEAN_DEFAULT_BLACK_BREAK = 0.35
OCEAN_DEFAULT_BLACK_SLOPE = 0.25
OCEAN_FADE_DEPTH = -50.0
SMALL_OCEAN_MAX_AREA_SQ_M = 1_500_000.0
MAX_COMPONENT_CLEANUP_PIXELS = 120_000_000
DEFAULT_MAX_IN_MEMORY_ALPHA_PIXELS = 12_000_000
DEFAULT_MAX_IN_MEMORY_COLOR_PIXELS = 4_000_000
DEFAULT_MAX_IN_MEMORY_SIEVE_MASK_PIXELS = 512_000_000
DEFAULT_OCEAN_CHUNK_SIZE = 8192
DEFAULT_MAX_FINAL_TRANSLATE_PIXELS = 512_000_000
TEMPORARY_TILED_GTIFF_OPTIONS = (
    "BIGTIFF=IF_SAFER",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)

T = TypeVar("T")


def build_step_progress_callback(
    progress_line: LiveProgressLine,
    step_label: str,
    started_at: float,
) -> Callable[[float], None]:
    """Return a callback that renders step progress with ETA."""

    def update(progress_fraction: float) -> None:
        bounded_fraction = min(max(progress_fraction, 0.0), 1.0)
        elapsed = time.perf_counter() - started_at
        remaining = None
        if bounded_fraction > 0.0 and elapsed > 0.0:
            remaining = elapsed * (1.0 - bounded_fraction) / bounded_fraction
        progress_line.update(
            f"{step_label} {bounded_fraction * 100:3.0f}%; "
            f"{format_eta(remaining, elapsed_seconds=elapsed)}"
        )

    return update


def run_step_with_eta(
    step_label: str,
    operation: Callable[[Callable[[float], None]], T],
) -> T:
    """Run a long step with a live ETA line."""
    progress_line = LiveProgressLine()
    progress = build_step_progress_callback(progress_line, step_label, time.perf_counter())
    progress(0.0)
    try:
        result = operation(progress)
    except Exception:
        progress_line.finish()
        raise
    progress(1.0)
    progress_line.finish()
    return result


def build_gdal_progress_callback(
    progress: Callable[[float], None] | None,
) -> Callable[[float, str, object], int] | None:
    """Adapt a simple 0..1 progress callback to GDAL's callback signature."""
    if progress is None:
        return None

    def callback(complete: float, _message: str, _data: object) -> int:
        progress(complete)
        return 1

    return callback


@dataclass(frozen=True)
class OceanBackgroundArtifacts:
    source_vrt: str
    masked_vrt: str
    warped_vrt: str
    alpha_vrt: str
    alpha_tif: str
    hillshade_tif: str
    color_tif: str
    rgba_vrt: str
    output_tif: str


@dataclass(frozen=True)
class OceanStyleOptions:
    tonemap: bool = False
    grade: bool = True
    exposure: float = OCEAN_DEFAULT_EXPOSURE
    shadow_break: float = OCEAN_DEFAULT_SHADOW_BREAK
    highlight_break: float = OCEAN_DEFAULT_HIGHLIGHT_BREAK
    shadow_slope: float = OCEAN_DEFAULT_SHADOW_SLOPE
    mid_slope: float = OCEAN_DEFAULT_MID_SLOPE
    highlight_slope: float = OCEAN_DEFAULT_HIGHLIGHT_SLOPE
    gamma: float = OCEAN_DEFAULT_GAMMA
    shoulder: float = OCEAN_DEFAULT_SHOULDER
    saturation: float = OCEAN_DEFAULT_SATURATION
    vibrance: float = DEFAULT_VIBRANCE
    black_point: float = DEFAULT_BLACK_POINT
    white_point: float = DEFAULT_WHITE_POINT
    black_break: float = OCEAN_DEFAULT_BLACK_BREAK
    black_slope: float = OCEAN_DEFAULT_BLACK_SLOPE
    grade_high_break: float | None = None
    grade_mid_slope: float = PREVIEW_DARKEN_MID_SLOPE
    grade_high_slope: float | None = None
    depth_min: float = -11000.0
    depth_max: float = 0.0


@dataclass(frozen=True)
class OceanChunkPlan:
    row: int
    col: int
    xoff: int
    yoff: int
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    expanded_bounds: tuple[float, float, float, float]
    core_src_win: tuple[int, int, int, int]


@dataclass(frozen=True)
class OceanBuildPlan:
    bounds: tuple[float, float, float, float]
    pixel_size: float
    zoom: int
    width: int
    height: int
    total_pixels: int
    chunk_size: int
    halo_pixels: int
    chunks: tuple[OceanChunkPlan, ...]


@dataclass(frozen=True)
class OceanChunkArtifacts:
    depth_tif: str
    alpha_vrt: str
    alpha_tif: str
    hillshade_tif: str
    color_tif: str
    rgba_vrt: str
    expanded_rgba_tif: str
    rgba_tif: str
    cleanup_mask_tif: Path


def max_in_memory_alpha_pixels() -> int:
    """Return the live pixel budget for full alpha-mask generation in memory."""
    return compute_in_memory_pixel_limit(
        32,
        usage_fraction=0.15,
        fallback_pixels=DEFAULT_MAX_IN_MEMORY_ALPHA_PIXELS,
        max_pixels=96_000_000,
    )


def max_in_memory_color_pixels() -> int:
    """Return the live pixel budget for full ocean RGB shading in memory."""
    return compute_in_memory_pixel_limit(
        40,
        usage_fraction=0.15,
        fallback_pixels=DEFAULT_MAX_IN_MEMORY_COLOR_PIXELS,
        max_pixels=64_000_000,
    )


def max_in_memory_sieve_mask_pixels() -> int:
    """Return the live pixel budget for keeping the sieve land mask in memory."""
    return compute_in_memory_pixel_limit(
        2,
        usage_fraction=0.2,
        fallback_pixels=DEFAULT_MAX_IN_MEMORY_SIEVE_MASK_PIXELS,
        max_pixels=1_500_000_000,
    )


def choose_ocean_parallel_workers(parallel: int | None, chunk_count: int) -> int:
    """Return the worker count used for chunked ocean processing."""
    if chunk_count <= 0:
        return 1
    if parallel is None:
        return max(1, min(chunk_count, os.cpu_count() or 1))
    if parallel <= 0:
        raise ValueError("parallel must be positive when provided")
    return min(parallel, chunk_count)


def compute_ocean_chunk_halo_pixels(pixel_size: float) -> int:
    """Return a halo wide enough to minimize chunk-edge cleanup and hillshade seams."""
    if pixel_size <= 0.0:
        raise ValueError("pixel_size must be positive")
    component_radius_pixels = int(np.ceil(np.sqrt(SMALL_OCEAN_MAX_AREA_SQ_M) / pixel_size))
    return max(1, min(512, component_radius_pixels + 2))


def build_ocean_output_plan(
    bbox: tuple[float, float, float, float] | None,
    *,
    max_zoom: int,
    chunk_size: int = DEFAULT_OCEAN_CHUNK_SIZE,
) -> OceanBuildPlan:
    """Plan the aligned final 3857 output grid and its independently processable chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if bbox is None:
        bounds = WEB_MERCATOR_WORLD_BOUNDS
        pixel_size = web_mercator_pixel_size(max_zoom)
        zoom = max_zoom
    else:
        bounds, pixel_size, zoom = snapped_tile_grid_for_bbox(bbox, max_zoom)

    width = max(1, int(round((bounds[2] - bounds[0]) / pixel_size)))
    height = max(1, int(round((bounds[3] - bounds[1]) / pixel_size)))
    halo_pixels = compute_ocean_chunk_halo_pixels(pixel_size)
    total_pixels = width * height

    chunks: list[OceanChunkPlan] = []
    for row, yoff in enumerate(range(0, height, chunk_size)):
        chunk_height = min(chunk_size, height - yoff)
        for col, xoff in enumerate(range(0, width, chunk_size)):
            chunk_width = min(chunk_size, width - xoff)

            minx = bounds[0] + xoff * pixel_size
            maxx = minx + chunk_width * pixel_size
            maxy = bounds[3] - yoff * pixel_size
            miny = maxy - chunk_height * pixel_size

            expanded_xoff = max(0, xoff - halo_pixels)
            expanded_yoff = max(0, yoff - halo_pixels)
            expanded_xend = min(width, xoff + chunk_width + halo_pixels)
            expanded_yend = min(height, yoff + chunk_height + halo_pixels)
            expanded_width = expanded_xend - expanded_xoff
            expanded_height = expanded_yend - expanded_yoff

            expanded_minx = bounds[0] + expanded_xoff * pixel_size
            expanded_maxx = expanded_minx + expanded_width * pixel_size
            expanded_maxy = bounds[3] - expanded_yoff * pixel_size
            expanded_miny = expanded_maxy - expanded_height * pixel_size

            chunks.append(
                OceanChunkPlan(
                    row=row,
                    col=col,
                    xoff=xoff,
                    yoff=yoff,
                    width=chunk_width,
                    height=chunk_height,
                    bounds=(minx, miny, maxx, maxy),
                    expanded_bounds=(expanded_minx, expanded_miny, expanded_maxx, expanded_maxy),
                    core_src_win=(
                        xoff - expanded_xoff,
                        yoff - expanded_yoff,
                        chunk_width,
                        chunk_height,
                    ),
                )
            )

    return OceanBuildPlan(
        bounds=bounds,
        pixel_size=pixel_size,
        zoom=zoom,
        width=width,
        height=height,
        total_pixels=total_pixels,
        chunk_size=chunk_size,
        halo_pixels=halo_pixels,
        chunks=tuple(chunks),
    )


def describe_ocean_output_plan(plan: OceanBuildPlan, *, vrt: bool, destination: str) -> None:
    """Print a compact summary of the planned final grid and merge shape."""
    print(
        "Ocean target grid: "
        f"{plan.width}x{plan.height} px "
        f"({plan.total_pixels:,} pixels) at ~{plan.pixel_size:.2f} m/px; "
        f"{len(plan.chunks):,} chunk(s) of up to {plan.chunk_size}px with {plan.halo_pixels}px halo."
    )
    if not vrt and plan.total_pixels > DEFAULT_MAX_FINAL_TRANSLATE_PIXELS:
        print(
            "Warning: Final GeoTIFF translation may still be very slow for this grid size. "
            f"Consider rerunning with --vrt or a smaller bbox before writing {destination}."
        )


def build_gebco_source_vrt(gebco_zip: str, output_vrt: str) -> str:
    """Build a source VRT from the GEBCO zip archive."""
    if not os.path.exists(gebco_zip):
        raise FileNotFoundError(f"GEBCO zip not found: {gebco_zip}")

    gebco_vsi = f"/vsizip/{gebco_zip}"
    files_in_zip = gdal.ReadDir(gebco_vsi) or []
    tif_paths = [
        f"{gebco_vsi}/{name}" for name in files_in_zip if name.lower().endswith(".tif")
    ]
    if not tif_paths:
        raise RuntimeError(f"No GeoTIFF files found in GEBCO zip: {gebco_zip}")

    staged_output_vrt = prepare_staged_path(output_vrt)
    gdal.BuildVRT(staged_output_vrt, tif_paths)
    publish_staged_path(staged_output_vrt, output_vrt)
    return output_vrt


def create_gebco_ocean_vrt(source_vrt: str, output_vrt: str) -> str:
    """Mask positive GEBCO values so only ocean elevations remain."""
    ds = gdal.Open(source_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open GEBCO source VRT: {source_vrt}")

    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = GEBCO_OCEAN_NODATA

    geotransform = ",".join(str(value) for value in ds.GetGeoTransform())
    projection = escape(ds.GetProjection())
    source_filename = escape(os.path.abspath(source_vrt))
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None

    staged_output_vrt = prepare_staged_path(output_vrt)
    with open(staged_output_vrt, "w") as f:
        f.write(
            f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
    <NoDataValue>{nodata_value}</NoDataValue>
    <PixelFunctionType>expression</PixelFunctionType>
    <PixelFunctionArguments dialect="muparser" expression="B1 &gt; 0.001 ? {nodata_value} : B1"/>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    publish_staged_path(staged_output_vrt, output_vrt)
    return output_vrt


def create_alpha_vrt(
    source_vrt: str,
    output_vrt: str,
    alpha_tif: str | None = None,
    cleanup_path: Path | None = None,
) -> str:
    """Create an explicit alpha mask VRT from depth thresholds and land-preferred cleanup."""
    ds = gdal.Open(source_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open source VRT for alpha generation: {source_vrt}")

    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = GEBCO_OCEAN_NODATA

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    alpha_tif = alpha_tif or str(Path(output_vrt).with_suffix(".tif"))
    staged_alpha_tif = prepare_staged_path(alpha_tif)
    driver = gdal.GetDriverByName("GTiff")
    alpha_ds = driver.Create(staged_alpha_tif, xsize, ysize, 1, gdal.GDT_Byte, options=list(GTIFF_CREATION_OPTIONS))
    if alpha_ds is None:
        raise RuntimeError(f"Could not create alpha TIFF: {alpha_tif}")
    alpha_ds.SetProjection(ds.GetProjection())
    alpha_ds.SetGeoTransform(ds.GetGeoTransform())
    alpha_band = alpha_ds.GetRasterBand(1)

    if xsize * ysize <= max_in_memory_alpha_pixels():
        depths = band.ReadAsArray().astype(np.float32)
        ocean_mask = build_ocean_threshold_mask(depths, nodata_value)
        cleaned_mask = remove_small_enclosed_ocean_regions(
            ocean_mask,
            ds.GetGeoTransform(),
            SMALL_OCEAN_MAX_AREA_SQ_M,
        )
        alpha_band.WriteArray(cleaned_mask.astype(np.uint8) * 255)
        alpha_band.FlushCache()
    else:
        block_width, block_height = band.GetBlockSize()
        if block_width <= 0:
            block_width = 512
        if block_height <= 0:
            block_height = 512

        for yoff in range(0, ysize, block_height):
            bh = min(block_height, ysize - yoff)
            for xoff in range(0, xsize, block_width):
                bw = min(block_width, xsize - xoff)
                depths = band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                ocean_mask = build_ocean_threshold_mask(depths, nodata_value)
                alpha_band.WriteArray(ocean_mask.astype(np.uint8) * 255, xoff=xoff, yoff=yoff)

        alpha_band.FlushCache()
        if xsize * ysize <= MAX_COMPONENT_CLEANUP_PIXELS:
            ocean_mask = alpha_band.ReadAsArray().astype(bool)
            cleaned_mask = remove_small_enclosed_ocean_regions(
                ocean_mask,
                ds.GetGeoTransform(),
                SMALL_OCEAN_MAX_AREA_SQ_M,
            )
            alpha_band.WriteArray(cleaned_mask.astype(np.uint8) * 255)
            alpha_band.FlushCache()
        else:
            cleanup_mask_path = cleanup_path or Path(alpha_tif).with_suffix(".land_mask.tif")
            remove_small_ocean_regions_sieve(
                alpha_ds,
                alpha_band,
                ds.GetGeoTransform(),
                SMALL_OCEAN_MAX_AREA_SQ_M,
                cleanup_mask_path,
            )
    alpha_ds = None
    ds = None
    publish_staged_path(staged_alpha_tif, alpha_tif)
    staged_output_vrt = prepare_staged_path(output_vrt)
    gdal.BuildVRT(staged_output_vrt, [alpha_tif])
    publish_staged_path(staged_output_vrt, output_vrt)
    return output_vrt


def build_ocean_threshold_mask(depths: np.ndarray, nodata_value: float) -> np.ndarray:
    """Return the boolean mask for ocean depths deep enough to render as ocean."""
    return (depths > nodata_value + 0.1) & (depths < OCEAN_FADE_DEPTH)


def remove_small_enclosed_ocean_regions(
    ocean_mask: np.ndarray,
    geotransform: Sequence[float],
    max_area_sq_m: float,
) -> np.ndarray:
    """Remove enclosed ocean components smaller than the configured area threshold."""
    if ocean_mask.ndim != 2:
        raise ValueError("ocean_mask must be 2D")
    if ocean_mask.size == 0 or max_area_sq_m <= 0.0:
        return ocean_mask

    pixel_area_sq_m = abs(geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4])
    if pixel_area_sq_m <= 0.0:
        raise ValueError("geotransform must define a positive pixel area")

    labels, component_count = label(ocean_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return ocean_mask

    component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    edge_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    enclosed_small_labels = [
        label_id
        for label_id in range(1, component_count + 1)
        if label_id not in edge_labels and component_sizes[label_id] * pixel_area_sq_m < max_area_sq_m
    ]
    if not enclosed_small_labels:
        return ocean_mask

    cleaned_mask = ocean_mask.copy()
    cleaned_mask[np.isin(labels, enclosed_small_labels)] = False
    return cleaned_mask


def remove_small_ocean_regions_sieve(
    alpha_dataset: gdal.Dataset,
    alpha_band: gdal.Band,
    geotransform: Sequence[float],
    max_area_sq_m: float,
    cleanup_mask_path: Path,
) -> None:
    """Remove small ocean regions with GDAL sieve while preserving original land pixels."""
    pixel_area_sq_m = abs(geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4])
    if pixel_area_sq_m <= 0.0:
        raise ValueError("geotransform must define a positive pixel area")

    sieve_threshold_pixels = max(1, int(np.ceil(max_area_sq_m / pixel_area_sq_m)))
    land_mask_ds, staged_cleanup_mask = create_sieve_cleanup_mask_dataset(
        alpha_dataset,
        cleanup_mask_path,
    )

    land_mask_ds.SetProjection(alpha_dataset.GetProjection())
    land_mask_ds.SetGeoTransform(alpha_dataset.GetGeoTransform())
    land_mask_band = land_mask_ds.GetRasterBand(1)

    block_width, block_height = alpha_band.GetBlockSize()
    if block_width <= 0:
        block_width = 512
    if block_height <= 0:
        block_height = 512

    for yoff in range(0, alpha_dataset.RasterYSize, block_height):
        bh = min(block_height, alpha_dataset.RasterYSize - yoff)
        for xoff in range(0, alpha_dataset.RasterXSize, block_width):
            bw = min(block_width, alpha_dataset.RasterXSize - xoff)
            alpha_block = alpha_band.ReadAsArray(xoff, yoff, bw, bh)
            land_mask_band.WriteArray((alpha_block == 0).astype(np.uint8) * 255, xoff=xoff, yoff=yoff)

    land_mask_band.FlushCache()
    gdal.SieveFilter(alpha_band, None, alpha_band, sieve_threshold_pixels, 8)

    for yoff in range(0, alpha_dataset.RasterYSize, block_height):
        bh = min(block_height, alpha_dataset.RasterYSize - yoff)
        for xoff in range(0, alpha_dataset.RasterXSize, block_width):
            bw = min(block_width, alpha_dataset.RasterXSize - xoff)
            cleaned_block = alpha_band.ReadAsArray(xoff, yoff, bw, bh)
            land_block = land_mask_band.ReadAsArray(xoff, yoff, bw, bh)
            cleaned_block[land_block != 0] = 0
            alpha_band.WriteArray(cleaned_block, xoff=xoff, yoff=yoff)

    alpha_band.FlushCache()
    land_mask_ds = None
    if staged_cleanup_mask is not None:
        remove_if_exists(staged_cleanup_mask)


def create_sieve_cleanup_mask_dataset(
    alpha_dataset: gdal.Dataset,
    cleanup_mask_path: Path,
) -> tuple[gdal.Dataset, str | None]:
    """Create the temporary land-mask dataset used to restore original land after sieving."""
    pixel_count = alpha_dataset.RasterXSize * alpha_dataset.RasterYSize
    if pixel_count <= max_in_memory_sieve_mask_pixels():
        mem_driver = gdal.GetDriverByName("MEM")
        if mem_driver is not None:
            land_mask_ds = mem_driver.Create(
                "",
                alpha_dataset.RasterXSize,
                alpha_dataset.RasterYSize,
                1,
                gdal.GDT_Byte,
            )
            if land_mask_ds is not None:
                return land_mask_ds, None

    mask_driver = gdal.GetDriverByName("GTiff")
    if mask_driver is None:
        raise RuntimeError("Could not load GTiff driver for ocean cleanup")

    staged_cleanup_mask = build_staged_path(str(cleanup_mask_path))
    remove_if_exists(staged_cleanup_mask)
    land_mask_ds = mask_driver.Create(
        staged_cleanup_mask,
        alpha_dataset.RasterXSize,
        alpha_dataset.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=list(TEMPORARY_TILED_GTIFF_OPTIONS),
    )
    if land_mask_ds is None:
        raise RuntimeError(f"Could not create temporary cleanup mask: {cleanup_mask_path}")
    return land_mask_ds, staged_cleanup_mask


def build_ocean_ramp_colors(style: OceanStyleOptions) -> np.ndarray:
    """Return the styled MAKO depth ramp as float32 RGB triples in [0, 1]."""
    mako_colors = np.array([c[1:] for c in MAKO_RAMP], dtype=np.float32) / 255.0
    mako_arr = mako_colors.T.reshape(3, -1, 1)

    if style.tonemap:
        toned_mako = apply_soft_knee_numpy(
            mako_arr,
            shadow_break=style.shadow_break,
            highlight_break=style.highlight_break,
            shadow_slope=style.shadow_slope,
            mid_slope=style.mid_slope,
            highlight_slope=style.highlight_slope,
            exposure=style.exposure,
        )
        mako_colors = toned_mako.reshape(3, -1).T
    else:
        mako_colors = (
            mako_colors
            if is_identity_value(style.exposure)
            else np.clip(mako_colors * style.exposure, 0.0, 1.0)
        )

    if style.grade:
        graded_mako = apply_preview_correction_numpy(
            mako_colors.T.reshape(3, -1, 1),
            saturation=style.saturation,
            vibrance=style.vibrance,
            black_point=style.black_point,
            white_point=style.white_point,
            darken_break=style.black_break,
            low_slope=style.black_slope,
            gamma=style.gamma,
            shoulder=style.shoulder,
            highlight_break=style.grade_high_break,
            mid_slope=style.grade_mid_slope,
            high_slope=style.grade_high_slope,
        )
        mako_colors = graded_mako.reshape(3, -1).T

    return np.asarray(mako_colors, dtype=np.float32)


def colorize_ocean_depths(depths: np.ndarray, style: OceanStyleOptions) -> np.ndarray:
    """Colorize GEBCO depth values using the shared ocean styling pipeline."""
    return colorize_depth_numpy(
        depths,
        build_ocean_ramp_colors(style),
        style.depth_min,
        style.depth_max,
    )


def snapped_tile_grid_for_bbox(
    bbox: tuple[float, float, float, float],
    max_zoom: int = DEFAULT_MAX_ZOOM,
    *,
    tile_size: int = 256,
) -> tuple[tuple[float, float, float, float], float, int]:
    """Snap a bbox outward to the target Web Mercator tile pixel grid."""
    pixel_size = web_mercator_pixel_size_for_tile_size(max_zoom, tile_size)
    zoom = max_zoom
    mercator_bounds = lonlat_bbox_to_mercator_bounds(*bbox)
    snapped_bounds = snap_bounds_to_pixel_grid(mercator_bounds, pixel_size)
    return snapped_bounds, pixel_size, zoom


def create_hillshade_tif(
    input_vrt: str,
    output_tif: str,
    z_factor: float,
) -> str:
    """Create the ocean hillshade GeoTIFF with GDAL's in-process DEM pipeline."""
    staged_output_tif = prepare_staged_path(output_tif)
    options = gdal.DEMProcessingOptions(
        format="GTiff",
        creationOptions=list(GTIFF_CREATION_OPTIONS),
        multiDirectional=True,
        zFactor=z_factor,
    )
    hillshade_ds = gdal.DEMProcessing(staged_output_tif, input_vrt, "hillshade", options=options)
    if hillshade_ds is None:
        raise RuntimeError(f"Could not create hillshade TIFF: {output_tif}")
    hillshade_ds = None
    publish_staged_path(staged_output_tif, output_tif)
    return output_tif


def create_ocean_rgb_tif(
    depth_vrt: str,
    hillshade_tif: str,
    output_tif: str,
    style: OceanStyleOptions,
) -> str:
    """Colorize warped GEBCO depths and modulate them with hillshade."""
    depth_ds = gdal.Open(depth_vrt)
    if depth_ds is None:
        raise RuntimeError(f"Could not open ocean depth VRT: {depth_vrt}")

    hillshade_ds = gdal.Open(hillshade_tif)
    if hillshade_ds is None:
        raise RuntimeError(f"Could not open hillshade TIFF: {hillshade_tif}")

    depth_band = depth_ds.GetRasterBand(1)
    hillshade_band = hillshade_ds.GetRasterBand(1)
    block_width, block_height = hillshade_band.GetBlockSize()
    if block_width <= 0:
        block_width = 512
    if block_height <= 0:
        block_height = 512
    ramp_colors = build_ocean_ramp_colors(style)

    driver = gdal.GetDriverByName("GTiff")
    staged_output_tif = prepare_staged_path(output_tif)
    color_ds = driver.Create(
        staged_output_tif,
        hillshade_ds.RasterXSize,
        hillshade_ds.RasterYSize,
        3,
        gdal.GDT_Byte,
        options=list(GTIFF_CREATION_OPTIONS),
    )
    if color_ds is None:
        raise RuntimeError(f"Could not create colorized ocean TIFF: {output_tif}")

    color_ds.SetProjection(hillshade_ds.GetProjection())
    color_ds.SetGeoTransform(hillshade_ds.GetGeoTransform())

    for band_index, color_name in enumerate(("RedBand", "GreenBand", "BlueBand"), start=1):
        band = color_ds.GetRasterBand(band_index)
        band.SetColorInterpretation(getattr(gdal, f"GCI_{color_name}"))

    color_bands = [color_ds.GetRasterBand(index) for index in range(1, 4)]
    if hillshade_ds.RasterXSize * hillshade_ds.RasterYSize <= max_in_memory_color_pixels():
        depths = depth_band.ReadAsArray().astype(np.float32)
        hillshade = hillshade_band.ReadAsArray().astype(np.float32)
        byte_arr = shade_ocean_rgb(depths, hillshade, ramp_colors, style)
        for band_index, band in enumerate(color_bands):
            band.WriteArray(byte_arr[band_index])
    else:
        for yoff in range(0, hillshade_ds.RasterYSize, block_height):
            bh = min(block_height, hillshade_ds.RasterYSize - yoff)
            for xoff in range(0, hillshade_ds.RasterXSize, block_width):
                bw = min(block_width, hillshade_ds.RasterXSize - xoff)
                depths = depth_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                hillshade = hillshade_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
                byte_arr = shade_ocean_rgb(depths, hillshade, ramp_colors, style)
                for band_index, band in enumerate(color_bands):
                    band.WriteArray(byte_arr[band_index], xoff=xoff, yoff=yoff)

    color_ds.FlushCache()
    color_ds = None
    depth_ds = None
    hillshade_ds = None
    publish_staged_path(staged_output_tif, output_tif)
    return output_tif


def shade_ocean_rgb(
    depths: np.ndarray,
    hillshade: np.ndarray,
    ramp_colors: np.ndarray,
    style: OceanStyleOptions,
) -> np.ndarray:
    """Return a shaded RGB byte array for one ocean raster block."""
    rgb = colorize_depth_numpy(
        depths,
        ramp_colors,
        style.depth_min,
        style.depth_max,
    )
    shade = (0.35 + 0.65 * np.clip(hillshade / 255.0, 0.0, 1.0)).astype(np.float32, copy=False)
    shaded_rgb = np.clip(rgb * shade[np.newaxis, :, :], 0.0, 1.0)
    return (shaded_rgb * 255.0).astype(np.uint8)


def create_rgb_with_alpha_vrt(
    rgb_tif: str,
    alpha_vrt: str,
    output_vrt: str,
    alpha_tif: str | None = None,
) -> str:
    """Attach an explicit alpha band to an RGB GeoTIFF via VRT."""
    rgb_ds = gdal.Open(rgb_tif)
    if rgb_ds is None:
        raise RuntimeError(f"Could not open RGB TIFF: {rgb_tif}")

    alpha_tif = alpha_tif or str(Path(alpha_vrt).with_suffix(".tif"))
    alpha_ds = gdal.Open(alpha_tif)
    if alpha_ds is None:
        raise RuntimeError(f"Could not open alpha TIFF: {alpha_tif}")

    xsize = rgb_ds.RasterXSize
    ysize = rgb_ds.RasterYSize
    geotransform = ",".join(str(value) for value in rgb_ds.GetGeoTransform())
    projection = escape(rgb_ds.GetProjection())
    rgb_filename = escape(os.path.abspath(rgb_tif))
    alpha_filename = escape(os.path.abspath(alpha_tif))
    rgb_ds = None
    alpha_ds = None

    staged_output_vrt = prepare_staged_path(output_vrt)
    with open(staged_output_vrt, "w") as f:
        f.write(
            f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <ColorInterp>Red</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="2">
    <ColorInterp>Green</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="3">
    <ColorInterp>Blue</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>3</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="4">
    <ColorInterp>Alpha</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{alpha_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    publish_staged_path(staged_output_vrt, output_vrt)
    return output_vrt

def translate_rgba_vrt(
    rgba_vrt: str,
    destination: str,
    *,
    progress: Callable[[float], None] | None = None,
) -> str:
    """Materialize the RGBA VRT as a tiled GeoTIFF."""
    options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=list(GTIFF_CREATION_OPTIONS),
    )
    staged_destination = prepare_staged_path(destination)
    translated = gdal.Translate(
        staged_destination,
        rgba_vrt,
        options=options,
        callback=build_gdal_progress_callback(progress),
    )
    if translated is None:
        raise RuntimeError(f"Could not materialize RGBA GeoTIFF: {destination}")
    translated = None
    publish_staged_path(staged_destination, destination)
    return destination


def crop_raster_to_src_win(
    source_raster: str,
    destination: str,
    src_win: tuple[int, int, int, int],
) -> str:
    """Crop a raster window into a tiled GeoTIFF while preserving grid alignment."""
    staged_destination = prepare_staged_path(destination)
    translated = gdal.Translate(
        staged_destination,
        source_raster,
        options=gdal.TranslateOptions(
            format="GTiff",
            srcWin=src_win,
            creationOptions=list(GTIFF_CREATION_OPTIONS),
        ),
    )
    if translated is None:
        raise RuntimeError(f"Could not crop raster window into {destination}")
    translated = None
    publish_staged_path(staged_destination, destination)
    return destination


def write_rgba_vrt(rgba_vrt: str, destination: str) -> str:
    """Persist the final RGBA VRT to a caller-visible output path."""
    staged_destination = prepare_staged_path(destination)
    copyfile(rgba_vrt, staged_destination)
    publish_staged_path(staged_destination, destination)
    return destination


def build_ocean_chunk_artifacts(
    temp_dir: str,
    stem: str,
    unique_id: str,
    chunk: OceanChunkPlan,
) -> OceanChunkArtifacts:
    """Return deterministic intermediate paths for one chunked ocean output."""
    chunk_stem = f"{stem}_chunk_{chunk.row:04d}_{chunk.col:04d}"
    return OceanChunkArtifacts(
        depth_tif=os.path.join(temp_dir, f"{chunk_stem}_depth.tif"),
        alpha_vrt=os.path.join(temp_dir, f"{chunk_stem}_alpha.vrt"),
        alpha_tif=os.path.join(temp_dir, f"{chunk_stem}_alpha.tif"),
        hillshade_tif=os.path.join(temp_dir, f"{chunk_stem}_hillshade.tif"),
        color_tif=os.path.join(temp_dir, f"{chunk_stem}_color.tif"),
        rgba_vrt=os.path.join(temp_dir, f"{chunk_stem}_rgba.vrt"),
        expanded_rgba_tif=os.path.join(temp_dir, f"{chunk_stem}_rgba_expanded.tif"),
        rgba_tif=os.path.join(temp_dir, f"{chunk_stem}_rgba.tif"),
        cleanup_mask_tif=Path(os.path.join(temp_dir, f"{chunk_stem}_alpha_land_mask.tif")),
    )


def build_ocean_run_settings(
    destination: str,
    bbox: tuple[float, float, float, float] | None,
    *,
    max_zoom: int,
    chunk_size: int,
    resample_alg: str,
    hillshade_z: float,
    style: OceanStyleOptions,
) -> dict[str, Any]:
    """Return the resumable ocean-processing settings for one destination."""
    return {
        "destination": os.path.abspath(destination),
        "bbox": bbox,
        "max_zoom": max_zoom,
        "chunk_size": chunk_size,
        "resample_alg": resample_alg,
        "hillshade_z": hillshade_z,
        "style": asdict(style),
    }


def build_ocean_run_metadata_path(temp_dir: str, unique_id: str) -> str:
    """Return the persistent settings sidecar for one ocean output namespace."""
    return os.path.join(temp_dir, "run.json")


def recover_ocean_chunk_outputs(
    temp_dir: str,
    stem: str,
    unique_id: str,
    chunks: Sequence[OceanChunkPlan],
) -> tuple[set[tuple[int, int]], list[str]]:
    """Return reusable chunk RGBA outputs and drop stale zero-byte leftovers."""
    recovered_keys: set[tuple[int, int]] = set()
    recovered_paths: list[str] = []
    for chunk in chunks:
        rgba_path = build_ocean_chunk_artifacts(temp_dir, stem, unique_id, chunk).rgba_tif
        if os.path.exists(rgba_path) and not file_has_content(rgba_path):
            remove_if_exists(rgba_path)
        if file_has_content(rgba_path):
            recovered_keys.add((chunk.row, chunk.col))
            recovered_paths.append(rgba_path)
    return recovered_keys, recovered_paths


def materialize_warped_ocean_chunk(
    source_vrt: str,
    destination: str,
    *,
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    resample_alg: str,
) -> str:
    """Warp one aligned 3857 ocean chunk directly into a reusable GeoTIFF."""
    staged_destination = prepare_staged_path(destination)
    warped = gdal.Warp(
        staged_destination,
        source_vrt,
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS="EPSG:3857",
            outputBounds=bounds,
            xRes=pixel_size,
            yRes=pixel_size,
            resampleAlg=resample_alg,
            outputType=gdal.GDT_Float32,
            multithread=True,
            warpOptions=warp_thread_options(),
            creationOptions=list(GTIFF_CREATION_OPTIONS),
        ),
    )
    if warped is None:
        raise RuntimeError(f"Could not materialize warped ocean chunk: {destination}")
    warped = None
    publish_staged_path(staged_destination, destination)
    return destination


def build_merged_vrt(
    output_vrt: str,
    source_rasters: Sequence[str],
    *,
    progress: Callable[[float], None] | None = None,
) -> str:
    """Build a VRT that merges aligned raster chunks in deterministic order."""
    if not source_rasters:
        raise ValueError("source_rasters must not be empty")
    staged_output_vrt = prepare_staged_path(output_vrt)
    merged = gdal.BuildVRT(
        staged_output_vrt,
        sorted(source_rasters),
        callback=build_gdal_progress_callback(progress),
    )
    if merged is None:
        raise RuntimeError(f"Could not build merged VRT: {output_vrt}")
    merged = None
    publish_staged_path(staged_output_vrt, output_vrt)
    return output_vrt


def process_ocean_chunk(
    *,
    masked_vrt: str,
    temp_dir: str,
    stem: str,
    unique_id: str,
    chunk: OceanChunkPlan,
    pixel_size: float,
    resample_alg: str,
    hillshade_z: float,
    style: OceanStyleOptions,
) -> str:
    """Generate one final RGBA chunk from the masked GEBCO source."""
    artifacts = build_ocean_chunk_artifacts(temp_dir, stem, unique_id, chunk)

    materialize_warped_ocean_chunk(
        masked_vrt,
        artifacts.depth_tif,
        bounds=chunk.expanded_bounds,
        pixel_size=pixel_size,
        resample_alg=resample_alg,
    )
    create_alpha_vrt(
        artifacts.depth_tif,
        artifacts.alpha_vrt,
        alpha_tif=artifacts.alpha_tif,
        cleanup_path=artifacts.cleanup_mask_tif,
    )
    create_hillshade_tif(artifacts.depth_tif, artifacts.hillshade_tif, hillshade_z)
    create_ocean_rgb_tif(artifacts.depth_tif, artifacts.hillshade_tif, artifacts.color_tif, style)
    create_rgb_with_alpha_vrt(
        artifacts.color_tif,
        artifacts.alpha_vrt,
        artifacts.rgba_vrt,
        alpha_tif=artifacts.alpha_tif,
    )
    translate_rgba_vrt(artifacts.rgba_vrt, artifacts.expanded_rgba_tif)

    if chunk.core_src_win[0] == 0 and chunk.core_src_win[1] == 0:
        expanded_ds = gdal.Open(artifacts.expanded_rgba_tif)
        if expanded_ds is None:
            raise RuntimeError(f"Could not reopen expanded RGBA chunk: {artifacts.expanded_rgba_tif}")
        whole_chunk = (
            chunk.core_src_win[2] == expanded_ds.RasterXSize
            and chunk.core_src_win[3] == expanded_ds.RasterYSize
        )
        expanded_ds = None
    else:
        whole_chunk = False

    if whole_chunk:
        os.replace(artifacts.expanded_rgba_tif, artifacts.rgba_tif)
    else:
        crop_raster_to_src_win(
            artifacts.expanded_rgba_tif,
            artifacts.rgba_tif,
            chunk.core_src_win,
        )
        remove_if_exists(artifacts.expanded_rgba_tif)

    for path in (
        artifacts.depth_tif,
        artifacts.alpha_vrt,
        artifacts.alpha_tif,
        artifacts.hillshade_tif,
        artifacts.color_tif,
        artifacts.rgba_vrt,
    ):
        remove_if_exists(path)

    return artifacts.rgba_tif


def generate_ocean_background(
    gebco_zip: str,
    destination: str,
    bbox: tuple[float, float, float, float] | None = None,
    temp_dir: str = ".temp",
    resample_alg: str = "cubicspline",
    hillshade_z: float = 5.0,
    style: OceanStyleOptions | None = None,
    vrt: bool = False,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    parallel: int | None = None,
    chunk_size: int = DEFAULT_OCEAN_CHUNK_SIZE,
) -> OceanBackgroundArtifacts:
    """Generate a standalone RGBA ocean background output."""
    if not hasattr(gdal, "DEMProcessing"):
        raise RuntimeError("GDAL DEMProcessing is required to generate the ocean background")

    ensure_directory(temp_dir)
    stem = Path(destination).stem or "ocean"
    run_mode = "bbox" if bbox is not None else "global"
    print(f"Ocean build: {run_mode} run at z{max_zoom} -> {destination}")

    if style is None:
        style = OceanStyleOptions()

    unique_id = build_output_namespace(destination, default_stem="ocean")
    output_temp_dir = ensure_directory(build_output_namespace_dir(temp_dir, unique_id))
    ocean_run_settings = build_ocean_run_settings(
        destination,
        bbox,
        max_zoom=max_zoom,
        chunk_size=chunk_size,
        resample_alg=resample_alg,
        hillshade_z=hillshade_z,
        style=style,
    )
    metadata_path = build_ocean_run_metadata_path(output_temp_dir, unique_id)
    previous_ocean_run_settings = read_settings_file(
        metadata_path,
        description="ocean run metadata",
    )
    if previous_ocean_run_settings is not None:
        print_settings_diff_warning(
            "Ocean run",
            previous_ocean_run_settings,
            ocean_run_settings,
        )
    write_settings_file(metadata_path, ocean_run_settings)
    source_vrt = os.path.join(output_temp_dir, f"{stem}_source.vrt")
    masked_vrt = os.path.join(output_temp_dir, f"{stem}_masked.vrt")
    warped_vrt = os.path.join(output_temp_dir, f"{stem}_depth_chunks.vrt")
    alpha_vrt = os.path.join(output_temp_dir, f"{stem}_alpha.vrt")
    alpha_tif = os.path.join(output_temp_dir, f"{stem}_alpha_chunks.vrt")
    hillshade_tif = os.path.join(output_temp_dir, f"{stem}_hillshade_chunks.vrt")
    color_tif = os.path.join(output_temp_dir, f"{stem}_color_chunks.vrt")
    rgba_vrt = os.path.join(output_temp_dir, f"{stem}_rgba.vrt")

    print("[1/6] Building GEBCO source VRT...")
    build_gebco_source_vrt(gebco_zip, source_vrt)
    print("[2/6] Masking land from GEBCO source...")
    create_gebco_ocean_vrt(source_vrt, masked_vrt)

    print("[3/6] Planning aligned Web Mercator chunks...")
    plan = build_ocean_output_plan(bbox, max_zoom=max_zoom, chunk_size=chunk_size)
    describe_ocean_output_plan(plan, vrt=vrt, destination=destination)

    print("[4/6] Processing ocean chunks...")
    chunk_workers = choose_ocean_parallel_workers(parallel, len(plan.chunks))
    print(f"Processing {len(plan.chunks):,} chunk(s) with {chunk_workers} worker(s)...")

    recovered_chunk_keys, recovered_chunk_paths = recover_ocean_chunk_outputs(
        output_temp_dir, stem, unique_id, plan.chunks
    )
    if recovered_chunk_paths:
        print(f"Reusing {len(recovered_chunk_paths):,} existing ocean chunk(s).")
    remaining_chunks = [
        chunk for chunk in plan.chunks if (chunk.row, chunk.col) not in recovered_chunk_keys
    ]

    chunk_rgba_paths: list[str] = list(recovered_chunk_paths)
    chunk_progress_line = LiveProgressLine()
    chunk_progress_started_at = time.perf_counter()
    total_chunks = len(plan.chunks)
    def update_chunk_progress(completed_chunks: int) -> None:
        remaining = None
        elapsed = time.perf_counter() - chunk_progress_started_at
        if completed_chunks > 0 and elapsed > 0.0:
            remaining = elapsed * (total_chunks - completed_chunks) / completed_chunks
        percent = round((completed_chunks / total_chunks) * 100) if total_chunks > 0 else 0
        chunk_progress_line.update(
            f"Ocean chunk progress: {completed_chunks:,}/{total_chunks:,} ({percent}%); "
            f"{format_eta(remaining, elapsed_seconds=elapsed)}"
        )

    update_chunk_progress(len(chunk_rgba_paths))
    try:
        if chunk_workers == 1:
            completed = len(chunk_rgba_paths)
            for chunk in remaining_chunks:
                chunk_rgba_paths.append(
                    process_ocean_chunk(
                        masked_vrt=masked_vrt,
                        temp_dir=output_temp_dir,
                        stem=stem,
                        unique_id=unique_id,
                        chunk=chunk,
                        pixel_size=plan.pixel_size,
                        resample_alg=resample_alg,
                        hillshade_z=hillshade_z,
                        style=style,
                    )
                )
                completed += 1
                update_chunk_progress(completed)
        else:
            with warp_thread_budget(
                per_worker_warp_threads(chunk_workers)
            ), ThreadPoolExecutor(max_workers=chunk_workers) as executor:
                future_to_chunk = {
                    executor.submit(
                        process_ocean_chunk,
                        masked_vrt=masked_vrt,
                        temp_dir=output_temp_dir,
                        stem=stem,
                        unique_id=unique_id,
                        chunk=chunk,
                        pixel_size=plan.pixel_size,
                        resample_alg=resample_alg,
                        hillshade_z=hillshade_z,
                        style=style,
                    ): chunk
                    for chunk in remaining_chunks
                }
                completed = len(chunk_rgba_paths)
                for future in as_completed(future_to_chunk):
                    chunk_rgba_paths.append(future.result())
                    completed += 1
                    update_chunk_progress(completed)
    except Exception:
        chunk_progress_line.finish()
        raise
    chunk_progress_line.finish()

    run_step_with_eta(
        "[5/6] Building merged RGBA VRT...",
        lambda progress: build_merged_vrt(rgba_vrt, chunk_rgba_paths, progress=progress),
    )
    if vrt:
        translate_path = str(Path(destination).with_suffix(".vrt"))
        print("[6/6] Writing final RGBA VRT...")
        write_rgba_vrt(rgba_vrt, translate_path)
        destination = translate_path
    else:
        run_step_with_eta(
            "[6/6] Translating final RGBA GeoTIFF...",
            lambda progress: translate_rgba_vrt(rgba_vrt, destination, progress=progress),
        )
        for path in chunk_rgba_paths:
            remove_if_exists(path)

    print(f"Ocean build complete: {destination}")

    return OceanBackgroundArtifacts(
        source_vrt=source_vrt,
        masked_vrt=masked_vrt,
        warped_vrt=warped_vrt,
        alpha_vrt=alpha_vrt,
        alpha_tif=alpha_tif,
        hillshade_tif=hillshade_tif,
        color_tif=color_tif,
        rgba_vrt=rgba_vrt,
        output_tif=destination,
    )


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    return parse_bbox_string(bbox)


def build_ocean_style_from_args(args: argparse.Namespace) -> OceanStyleOptions:
    """Build style options from parsed CLI arguments."""
    return OceanStyleOptions(
        tonemap=args.tonemap,
        grade=args.grade,
        exposure=args.exposure,
        gamma=args.gamma,
        shoulder=args.shoulder,
        saturation=args.sat,
        vibrance=args.vibrance,
        black_point=args.black_point,
        white_point=args.white_point,
        black_break=args.db,
        black_slope=args.ls,
        grade_high_break=args.ghb,
        grade_mid_slope=args.gms,
        grade_high_slope=args.ghs,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )


def build_ocean_argument_parser() -> argparse.ArgumentParser:
    """Build the ocean CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone GEBCO ocean hillshade GeoTIFF. "
            f"Defaults to Web Mercator zoom {DEFAULT_MAX_ZOOM} "
            f"(~{web_mercator_pixel_size(DEFAULT_MAX_ZOOM):.2f} m/px at the equator)."
        )
    )
    add = parser.add_argument
    add("gebco_zip", nargs="?", default=DEFAULT_GEBCO_ZIP, help="GEBCO zip archive")
    add("destination", nargs="?", default=DEFAULT_OUTPUT, help="Output GeoTIFF path")
    add(
        "--bbox",
        help=(
            "Optional WGS84 bbox as min_lon,min_lat,max_lon,max_lat. "
            f"When omitted, exports the full masked source raster in EPSG:3857 at "
            f"Web Mercator zoom {DEFAULT_MAX_ZOOM}."
        ),
    )
    add(
        "--max-zoom",
        type=int,
        choices=list(SUPPORTED_MAX_ZOOMS),
        default=DEFAULT_MAX_ZOOM,
        help="Target Web Mercator zoom used for output resolution",
    )
    add("--temp-dir", default=".temp", help="Directory for intermediary files")
    add(
        "--resample-alg",
        choices=["cubicspline", "lanczos"],
        default="cubicspline",
        help="Resampling algorithm for the GEBCO upscale into EPSG:3857",
    )
    add(
        "--hillshade-z",
        type=float,
        default=5.0,
        help="Vertical exaggeration passed to gdaldem hillshade",
    )
    add(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ocean final grading before colorization",
    )
    add(
        "--exposure",
        type=float,
        default=OCEAN_DEFAULT_EXPOSURE,
        help="Global brightness multiplier",
    )
    add(
        "--tonemap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    add("--gamma", type=float, default=OCEAN_DEFAULT_GAMMA)
    add(
        "--shoulder",
        type=float,
        default=OCEAN_DEFAULT_SHOULDER,
        help="Highlight shaping curve; values above 1 lift the top end",
    )
    add("--sat", "--saturation", type=float, default=OCEAN_DEFAULT_SATURATION)
    add("--vibrance", type=float, default=DEFAULT_VIBRANCE)
    add("--black-point", type=float, default=DEFAULT_BLACK_POINT)
    add("--white-point", type=float, default=DEFAULT_WHITE_POINT)
    add("--db", "--black-break", "--grade-low-break", type=float, default=OCEAN_DEFAULT_BLACK_BREAK)
    add("--ls", "--black-slope", "--grade-low-slope", type=float, default=OCEAN_DEFAULT_BLACK_SLOPE)
    add(
        "--ghb",
        "--grade-highlight-break",
        type=float,
        help="Upper breakpoint for the final grading curve; defaults to the low break",
    )
    add(
        "--gms",
        "--grade-mid-slope",
        type=float,
        default=PREVIEW_DARKEN_MID_SLOPE,
    )
    add(
        "--ghs",
        "--grade-highlight-slope",
        type=float,
        help="Highlight slope for the final grading curve; defaults to an anchored derived slope",
    )
    add(
        "--depth-min",
        type=float,
        default=-11000.0,
        help="Depth value mapped to the start of the ocean color ramp",
    )
    add(
        "--depth-max",
        type=float,
        default=0.0,
        help="Depth value mapped to the end of the ocean color ramp",
    )
    add(
        "--vrt",
        action="store_true",
        help="Write the final styled RGBA VRT instead of translating it to a GeoTIFF",
    )
    add(
        "--parallel",
        type=int,
        help="Number of parallel chunk workers for ocean processing",
        default=40,
    )
    add(
        "--chunk-size",
        type=int,
        default=DEFAULT_OCEAN_CHUNK_SIZE,
        help="Chunk edge length in output pixels for chunked ocean processing",
    )
    return parser


def main() -> None:
    args = build_ocean_argument_parser().parse_args()

    artifacts = generate_ocean_background(
        gebco_zip=args.gebco_zip,
        destination=args.destination,
        bbox=parse_bbox(args.bbox) if args.bbox else None,
        temp_dir=args.temp_dir,
        resample_alg=args.resample_alg,
        hillshade_z=args.hillshade_z,
        style=build_ocean_style_from_args(args),
        vrt=args.vrt,
        max_zoom=args.max_zoom,
        parallel=args.parallel,
        chunk_size=args.chunk_size,
    )
    print(artifacts.output_tif)


if __name__ == "__main__":
    main()
