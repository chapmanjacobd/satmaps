#!/usr/bin/python3
import argparse
import json
import math
import os
import sys
import time
import uuid
import warnings
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import mgrs
from mgrs import core as mgrs_core
import numpy as np
import ocean
from common import file_has_content, write_text_file
from osgeo import gdal, ogr, osr

import tiler

gdal.UseExceptions()

OCEAN_MASK_ALPHA_THRESHOLD = 254.5
OCEAN_MASK_SCAN_PROCESS_BLOCKS = 4
LAND_MGRS_HEADER_PREFIX = "# satmaps-land-mgrs "

BBox = Tuple[float, float, float, float]
SrcWin = Tuple[int, int, int, int]
Envelope = Tuple[float, float, float, float]


class _NoopProgressLine:
    def print(self, _text: str) -> None:
        return

    def finish(self) -> None:
        return


def _build_wgs84_srs() -> osr.SpatialReference:
    """Return a WGS84 SRS configured for traditional GIS axis ordering."""
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    if hasattr(wgs84_srs, "SetAxisMappingStrategy"):
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return wgs84_srs


def _clone_target_srs(target_srs: osr.SpatialReference) -> osr.SpatialReference:
    """Clone a target SRS and normalize axis ordering to match GDAL raster usage."""
    cloned_srs = target_srs.Clone()
    if hasattr(cloned_srs, "SetAxisMappingStrategy"):
        cloned_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return cloned_srs


def build_bbox_geometry(
    bbox: BBox,
    target_srs: Optional[osr.SpatialReference] = None,
) -> ogr.Geometry:
    """Build a bbox polygon, reprojecting it from WGS84 when a target SRS is supplied."""
    min_lon, min_lat, max_lon, max_lat = bbox
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(min_lon, min_lat)
    ring.AddPoint(max_lon, min_lat)
    ring.AddPoint(max_lon, max_lat)
    ring.AddPoint(min_lon, max_lat)
    ring.AddPoint(min_lon, min_lat)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    if target_srs is None:
        return polygon

    dataset_srs = _clone_target_srs(target_srs)
    wgs84_srs = _build_wgs84_srs()
    if not (dataset_srs.IsSameGeogCS(wgs84_srs) and dataset_srs.IsGeographic()):
        polygon.Transform(osr.CoordinateTransformation(wgs84_srs, dataset_srs))

    return polygon


def build_mgrs_tile_geometry(
    mgrs_tile: str,
    target_srs: Optional[osr.SpatialReference] = None,
) -> Optional[ogr.Geometry]:
    """Build a 100 km MGRS tile polygon geometry."""
    m_converter = mgrs.MGRS()
    corner_codes = (
        mgrs_tile,
        f"{mgrs_tile}9999900000",
        f"{mgrs_tile}9999999999",
        f"{mgrs_tile}0000099999",
        mgrs_tile,
    )
    ring = ogr.Geometry(ogr.wkbLinearRing)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for code in corner_codes:
                lat, lon = m_converter.toLatLon(code)
                ring.AddPoint(float(lon), float(lat))
    except mgrs_core.MGRSError:
        return None

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    if target_srs is not None:
        dataset_srs = _clone_target_srs(target_srs)
        wgs84_srs = _build_wgs84_srs()
        if not (dataset_srs.IsSameGeogCS(wgs84_srs) and dataset_srs.IsGeographic()):
            polygon.Transform(osr.CoordinateTransformation(wgs84_srs, dataset_srs))

    return polygon


def build_candidate_scan_envelopes(
    dataset: gdal.Dataset,
    candidate_tiles: Set[str],
    bbox: Optional[BBox] = None,
) -> List[Envelope]:
    """Build candidate tile envelopes in dataset coordinates for block prefiltering."""
    dataset_srs = dataset.GetSpatialRef()
    bbox_geometry = build_bbox_geometry(bbox, dataset_srs) if bbox is not None else None
    envelopes: List[Envelope] = []

    for mgrs_tile in sorted(candidate_tiles):
        tile_geometry = build_mgrs_tile_geometry(mgrs_tile, dataset_srs)
        if tile_geometry is None:
            continue
        if bbox_geometry is not None:
            if not tile_geometry.Intersects(bbox_geometry):
                continue
            tile_geometry = tile_geometry.Intersection(bbox_geometry)
            if tile_geometry is None or tile_geometry.IsEmpty():
                continue
        min_x, max_x, min_y, max_y = tile_geometry.GetEnvelope()
        if min_x >= max_x or min_y >= max_y:
            continue
        envelopes.append((min_x, max_x, min_y, max_y))

    return envelopes


def source_window_envelope(
    geotransform: Sequence[float],
    src_win: SrcWin,
) -> Envelope:
    """Return the dataset-space envelope for a source pixel window."""
    xoff, yoff, width, height = src_win
    pixel_corners = (
        (float(xoff), float(yoff)),
        (float(xoff + width), float(yoff)),
        (float(xoff + width), float(yoff + height)),
        (float(xoff), float(yoff + height)),
    )
    world_corners = [
        (
            geotransform[0] + pixel_x * geotransform[1] + pixel_y * geotransform[2],
            geotransform[3] + pixel_x * geotransform[4] + pixel_y * geotransform[5],
        )
        for pixel_x, pixel_y in pixel_corners
    ]
    xs = [point[0] for point in world_corners]
    ys = [point[1] for point in world_corners]
    return min(xs), max(xs), min(ys), max(ys)


def envelopes_overlap(
    first: Envelope,
    second: Envelope,
) -> bool:
    """Return True when two (min_x, max_x, min_y, max_y) envelopes overlap."""
    return not (
        first[1] <= second[0]
        or first[0] >= second[1]
        or first[3] <= second[2]
        or first[2] >= second[3]
    )


def get_ocean_mask_band_index(dataset: gdal.Dataset) -> Optional[int]:
    """Return the explicit alpha band used as the ocean-mask handoff."""
    for band_index in range(1, dataset.RasterCount + 1):
        if dataset.GetRasterBand(band_index).GetColorInterpretation() == gdal.GCI_AlphaBand:
            return band_index

    return None


def build_wgs84_transform(dataset: gdal.Dataset) -> Optional[osr.CoordinateTransformation]:
    """Build a dataset-to-lon/lat transform when the raster is not already in WGS84."""
    dataset_srs = dataset.GetSpatialRef()
    if dataset_srs is None:
        return None

    dataset_srs = _clone_target_srs(dataset_srs)
    wgs84_srs = _build_wgs84_srs()

    if dataset_srs.IsSameGeogCS(wgs84_srs) and dataset_srs.IsGeographic():
        return None

    return osr.CoordinateTransformation(dataset_srs, wgs84_srs)


def get_bbox_scan_window(
    dataset: gdal.Dataset,
    bbox: Optional[BBox],
) -> Optional[SrcWin]:
    """Return the source window covering the requested bbox before block iteration starts."""
    if bbox is None:
        return 0, 0, dataset.RasterXSize, dataset.RasterYSize

    dataset_srs = dataset.GetSpatialRef()
    bbox_geometry = build_bbox_geometry(bbox, dataset_srs)
    min_x, max_x, min_y, max_y = bbox_geometry.GetEnvelope()
    bbox_bounds = (min_x, min_y, max_x, max_y)
    src_win = tiler.te_to_src_win(dataset, bbox_bounds)
    if src_win[2] <= 0 or src_win[3] <= 0:
        return None

    return src_win


def resolve_ocean_mask_scan_source(ocean_mask_src: str) -> Tuple[str, Optional[str]]:
    """Resolve a scan source path, building a temporary /vsimem VRT for GEBCO zip inputs."""
    if not ocean_mask_src.lower().endswith(".zip"):
        return ocean_mask_src, None
    if not os.path.exists(ocean_mask_src):
        return ocean_mask_src, None

    gebco_vsi = f"/vsizip/{ocean_mask_src}"
    files_in_zip = gdal.ReadDir(gebco_vsi) or []
    tiff_paths = sorted(
        f"{gebco_vsi}/{name}" for name in files_in_zip if name.lower().endswith(".tif")
    )
    if not tiff_paths:
        raise RuntimeError(f"No GeoTIFF files found in GEBCO zip: {ocean_mask_src}")

    vrt_path = f"/vsimem/land_mgrs_{uuid.uuid4().hex}.vrt"
    vrt = gdal.BuildVRT(vrt_path, tiff_paths)
    if vrt is None:
        raise RuntimeError(f"Could not build GEBCO source VRT from {ocean_mask_src}")
    vrt = None
    return vrt_path, vrt_path


def build_discovery_fill_allowed_mask(
    data: np.ndarray,
    *,
    nodata: Optional[float],
    depth_mode: bool,
) -> np.ndarray:
    """Build the fill-allowed mask for either alpha-mask or depth-threshold scanning."""
    if depth_mode:
        fill_allowed_mask = np.isfinite(data)
        if nodata is not None:
            fill_allowed_mask &= data > nodata + 0.1
        fill_allowed_mask &= data >= ocean.OCEAN_FADE_DEPTH
        return np.asarray(fill_allowed_mask, dtype=bool)

    fill_allowed_mask = data < OCEAN_MASK_ALPHA_THRESHOLD
    if nodata is not None:
        fill_allowed_mask &= data != nodata
    return np.asarray(fill_allowed_mask, dtype=bool)


def process_ocean_mask_window(
    data: np.ndarray,
    xoff: int,
    yoff: int,
    scan_window: SrcWin,
    geotransform: Sequence[float],
    nodata: Optional[float],
    to_wgs84: Optional[osr.CoordinateTransformation],
    mgrs_converter: mgrs.MGRS,
    bbox: Optional[BBox],
    candidate_tiles: Optional[Set[str]],
    *,
    depth_mode: bool,
) -> Set[str]:
    """Classify one in-memory scan window into intersecting MGRS tiles."""
    scan_xoff, scan_yoff, scan_width, scan_height = scan_window
    width = data.shape[1]
    height = data.shape[0]

    if candidate_tiles is None:
        clipped_xoff = max(xoff, scan_xoff)
        clipped_yoff = max(yoff, scan_yoff)
        clipped_max_x = min(xoff + width, scan_xoff + scan_width)
        clipped_max_y = min(yoff + height, scan_yoff + scan_height)
        data = data[
            clipped_yoff - yoff : clipped_max_y - yoff,
            clipped_xoff - xoff : clipped_max_x - xoff,
        ]
        xoff = clipped_xoff
        yoff = clipped_yoff
        width = clipped_max_x - clipped_xoff
        height = clipped_max_y - clipped_yoff
        if width <= 0 or height <= 0:
            return set()
    elif width <= 0 or height <= 0:
        return set()

    fill_allowed_mask = build_discovery_fill_allowed_mask(
        data,
        nodata=nodata,
        depth_mode=depth_mode,
    )
    if not np.any(fill_allowed_mask):
        return set()

    rows, cols = np.nonzero(fill_allowed_mask)
    pixel_x = xoff + cols.astype(np.float64) + 0.5
    pixel_y = yoff + rows.astype(np.float64) + 0.5
    xs = (
        geotransform[0]
        + pixel_x * geotransform[1]
        + pixel_y * geotransform[2]
    )
    ys = (
        geotransform[3]
        + pixel_x * geotransform[4]
        + pixel_y * geotransform[5]
    )

    if to_wgs84 is None:
        lons = xs
        lats = ys
    else:
        transformed_points = to_wgs84.TransformPoints(
            list(zip(xs.tolist(), ys.tolist(), strict=False))
        )
        lons = np.array([point[0] for point in transformed_points], dtype=np.float64)
        lats = np.array([point[1] for point in transformed_points], dtype=np.float64)

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        in_bbox = (
            (lons >= min_lon)
            & (lons <= max_lon)
            & (lats >= min_lat)
            & (lats <= max_lat)
        )
        if not np.any(in_bbox):
            return set()
        lons = lons[in_bbox]
        lats = lats[in_bbox]

    mgrs_tiles: Set[str] = set()
    for lon, lat in zip(lons, lats, strict=False):
        try:
            tile = mgrs_converter.toMGRS(float(lat), float(lon), MGRSPrecision=0)
            tile_str = tile.decode() if isinstance(tile, bytes) else tile
            if candidate_tiles is not None and tile_str not in candidate_tiles:
                continue
            mgrs_tiles.add(tile_str)
        except mgrs_core.MGRSError:
            continue
    return mgrs_tiles


def discover_mgrs_tiles_from_ocean_mask(
    ocean_mask_src: str,
    bbox: Optional[BBox] = None,
    candidate_mgrs_tiles: Optional[Set[str]] = None,
    *,
    progress_factory: Optional[Callable[[], Any]] = None,
    update_count_progress_fn: Optional[Callable[..., None]] = None,
) -> Set[str]:
    """Discover MGRS tiles that contain land or shallow water from alpha-mask or GEBCO depth sources."""
    cleanup_path: Optional[str] = None
    try:
        scan_source, cleanup_path = resolve_ocean_mask_scan_source(ocean_mask_src)
    except RuntimeError as exc:
        print(f"Warning: Could not prepare ocean mask source {ocean_mask_src}: {exc}")
        return set()

    try:
        ds = gdal.Open(scan_source)
    except RuntimeError as exc:
        print(f"Warning: Could not open ocean mask source {ocean_mask_src}: {exc}")
        ds = None
    if ds is None:
        if cleanup_path is not None:
            gdal.Unlink(cleanup_path)
        return set()

    band_index = get_ocean_mask_band_index(ds)
    depth_mode = band_index is None
    if band_index is None:
        if ds.RasterCount <= 0:
            ds = None
            if cleanup_path is not None:
                gdal.Unlink(cleanup_path)
            return set()
        band_index = 1

    band = ds.GetRasterBand(band_index)
    mgrs_tiles: Set[str] = set()
    block_x, block_y = band.GetBlockSize()
    row_block_height = max(block_y, 1)
    process_block_width = max(block_x, 1)
    nodata = band.GetNoDataValue()
    to_wgs84 = build_wgs84_transform(ds)
    geotransform = ds.GetGeoTransform()
    mgrs_converter = mgrs.MGRS()
    scan_window = get_bbox_scan_window(ds, bbox)
    if scan_window is None:
        ds = None
        if cleanup_path is not None:
            gdal.Unlink(cleanup_path)
        return set()
    scan_xoff, scan_yoff, scan_width, scan_height = scan_window

    candidate_tiles = set(candidate_mgrs_tiles) if candidate_mgrs_tiles else None
    total_row_blocks = math.ceil(scan_height / row_block_height) if scan_height > 0 else 0
    total_covered_blocks = (
        math.ceil(scan_width / process_block_width) * total_row_blocks
        if scan_width > 0
        else 0
    )

    completed_row_blocks = 0
    progress_line = progress_factory() if progress_factory is not None else _NoopProgressLine()
    started_at = time.perf_counter()

    if candidate_tiles is not None:
        candidate_scan_envelopes = build_candidate_scan_envelopes(
            ds,
            candidate_tiles,
            bbox=bbox,
        )
        candidate_row_blocks: List[Tuple[int, int]] = []
        for block_yoff in range(scan_yoff, scan_yoff + scan_height, row_block_height):
            block_height = min(row_block_height, ds.RasterYSize - block_yoff)
            block_src_win = (scan_xoff, block_yoff, scan_width, block_height)
            block_envelope = source_window_envelope(geotransform, block_src_win)
            if any(
                envelopes_overlap(block_envelope, candidate_envelope)
                for candidate_envelope in candidate_scan_envelopes
            ):
                candidate_row_blocks.append((block_yoff, block_height))

        print(
            "Ocean mask scan window: "
            f"{scan_width}x{scan_height} px across {len(candidate_row_blocks)} candidate row blocks "
            f"via {len(candidate_row_blocks)} targeted sequential reads."
        )
        process_width = max(process_block_width * OCEAN_MASK_SCAN_PROCESS_BLOCKS, process_block_width)
        for completed_candidates, (block_yoff, block_height) in enumerate(
            candidate_row_blocks,
            start=1,
        ):
            read_data = band.ReadAsArray(
                scan_xoff,
                block_yoff,
                scan_width,
                block_height,
            )
            if read_data is not None:
                read_data = np.asarray(read_data)
                for process_offset in range(0, scan_width, process_width):
                    process_chunk_width = min(process_width, scan_width - process_offset)
                    found_tiles = process_ocean_mask_window(
                        read_data[:, process_offset : process_offset + process_chunk_width],
                        scan_xoff + process_offset,
                        block_yoff,
                        scan_window,
                        geotransform,
                        nodata,
                        to_wgs84,
                        mgrs_converter,
                        bbox,
                        candidate_tiles,
                        depth_mode=depth_mode,
                    )
                    mgrs_tiles.update(found_tiles)
            if update_count_progress_fn is not None:
                update_count_progress_fn(
                    progress_line,
                    "Ocean mask scan progress:",
                    completed_candidates,
                    len(candidate_row_blocks),
                    started_at,
                    f"candidate row blocks {completed_candidates}/{len(candidate_row_blocks)}; {len(mgrs_tiles)} tiles found so far.",
                )
    else:
        print(
            "Ocean mask scan window: "
            f"{scan_width}x{scan_height} px across {total_row_blocks} row blocks "
            f"covering {total_covered_blocks} blocks via {total_row_blocks} sequential reads."
        )
        process_width = max(process_block_width * OCEAN_MASK_SCAN_PROCESS_BLOCKS, process_block_width)
        for block_yoff in range(scan_yoff, scan_yoff + scan_height, row_block_height):
            block_height = min(row_block_height, ds.RasterYSize - block_yoff)
            read_data = band.ReadAsArray(
                scan_xoff,
                block_yoff,
                scan_width,
                block_height,
            )
            if read_data is not None:
                read_data = np.asarray(read_data)
                for process_offset in range(0, scan_width, process_width):
                    process_chunk_width = min(process_width, scan_width - process_offset)
                    found_tiles = process_ocean_mask_window(
                        read_data[:, process_offset : process_offset + process_chunk_width],
                        scan_xoff + process_offset,
                        block_yoff,
                        scan_window,
                        geotransform,
                        nodata,
                        to_wgs84,
                        mgrs_converter,
                        bbox,
                        None,
                        depth_mode=depth_mode,
                    )
                    mgrs_tiles.update(found_tiles)
            completed_row_blocks += 1
            if update_count_progress_fn is not None:
                update_count_progress_fn(
                    progress_line,
                    "Ocean mask scan progress:",
                    completed_row_blocks,
                    total_row_blocks,
                    started_at,
                    f"row blocks {completed_row_blocks}/{total_row_blocks}; {len(mgrs_tiles)} tiles found so far.",
                )
    progress_line.finish()
    if candidate_tiles is not None:
        mgrs_tiles.intersection_update(candidate_tiles)
    ds = None
    if cleanup_path is not None:
        gdal.Unlink(cleanup_path)
    return mgrs_tiles


def build_land_mgrs_list_path() -> str:
    """Return the cached land-MGRS list path used to skip repeat ocean scans."""
    return "land_mgrs.list"


def build_land_mgrs_cache_metadata(
    bbox: Optional[BBox],
    ocean_mask_source: Optional[str],
) -> dict[str, object]:
    """Return informational metadata written alongside cached land-MGRS tiles."""
    if ocean_mask_source is None:
        normalized_source: Optional[str] = None
    elif os.path.basename(ocean_mask_source) == ocean.DEFAULT_GEBCO_ZIP:
        normalized_source = ocean.DEFAULT_GEBCO_ZIP
    else:
        normalized_source = os.path.abspath(ocean_mask_source)
    return {
        "bbox": list(bbox) if bbox is not None else None,
        "ocean_mask_source": normalized_source,
    }


def _read_saved_land_mgrs_list(
    land_mgrs_list_path: str,
) -> tuple[Optional[dict[str, object]], Set[str]]:
    """Read cached land-MGRS tiles plus any optional metadata header."""
    with open(land_mgrs_list_path, "r") as file_obj:
        lines = [line.strip() for line in file_obj]
    if not lines:
        return None, set()

    metadata: Optional[dict[str, object]] = None
    data_start_index = 0
    if lines[0].startswith(LAND_MGRS_HEADER_PREFIX):
        data_start_index = 1
        try:
            parsed_metadata = json.loads(lines[0][len(LAND_MGRS_HEADER_PREFIX) :])
        except json.JSONDecodeError as exc:
            print(
                f"Warning: Could not parse land MGRS list metadata {land_mgrs_list_path}: {exc}; "
                "ignoring header."
            )
        else:
            if isinstance(parsed_metadata, dict):
                metadata = parsed_metadata

    land_mgrs = {
        line
        for line in lines[data_start_index:]
        if line and not line.startswith("#")
    }
    return metadata, land_mgrs


def _extract_cached_bbox(metadata: Optional[dict[str, object]]) -> Optional[list[object]]:
    """Return the cached bbox marker when present."""
    if metadata is None:
        return None
    cached_bbox = metadata.get("bbox")
    return cached_bbox if isinstance(cached_bbox, list) else None


def _print_land_mgrs_reuse_message(
    land_mgrs_list_path: str,
    metadata: Optional[dict[str, object]],
) -> None:
    """Print a cache reuse message, including the original source when available."""
    if metadata is not None:
        generated_from = metadata.get("ocean_mask_source")
        if isinstance(generated_from, str) and generated_from:
            print(
                f"Reusing land MGRS list from {land_mgrs_list_path} "
                f"(generated from {generated_from})."
            )
            return
    print(f"Reusing land MGRS list from {land_mgrs_list_path}.")


def load_saved_land_mgrs_list(
    land_mgrs_list_path: Optional[str],
    *,
    bbox: Optional[BBox],
    ocean_mask_source: Optional[str],
) -> Optional[Set[str]]:
    """Load a cached land-MGRS list, treating bbox scope as the only strict cache key."""
    del ocean_mask_source
    if land_mgrs_list_path is None or not file_has_content(land_mgrs_list_path):
        return None

    try:
        metadata, land_mgrs = _read_saved_land_mgrs_list(land_mgrs_list_path)
    except OSError as exc:
        print(f"Warning: Could not read land MGRS list {land_mgrs_list_path}: {exc}")
        return None

    if bbox is None:
        if _extract_cached_bbox(metadata) is not None:
            return None
    elif _extract_cached_bbox(metadata) != list(bbox):
        return None

    _print_land_mgrs_reuse_message(land_mgrs_list_path, metadata)
    return land_mgrs


def _load_global_land_mgrs_list(
    land_mgrs_list_path: Optional[str],
) -> Optional[Set[str]]:
    """Load the shared non-bbox land-MGRS cache, ignoring source metadata."""
    return load_saved_land_mgrs_list(
        land_mgrs_list_path,
        bbox=None,
        ocean_mask_source=None,
    )


def save_land_mgrs_list(
    land_mgrs_list_path: Optional[str],
    land_mgrs: Set[str],
    *,
    bbox: Optional[BBox],
    ocean_mask_source: Optional[str],
) -> None:
    """Persist the scanned land-MGRS list so later runs can skip the ocean scan."""
    if land_mgrs_list_path is None:
        return

    metadata = build_land_mgrs_cache_metadata(bbox, ocean_mask_source)
    payload_lines = [f"{LAND_MGRS_HEADER_PREFIX}{json.dumps(metadata, sort_keys=True)}"]
    payload_lines.extend(sorted(land_mgrs))
    write_text_file(land_mgrs_list_path, "\n".join(payload_lines) + "\n")
    print(f"Saved land MGRS list to {land_mgrs_list_path}.")


def resolve_land_mgrs_source() -> Optional[str]:
    """Resolve the GEBCO source used for land/shallow-water MGRS discovery."""
    if os.path.exists(ocean.DEFAULT_GEBCO_ZIP):
        return ocean.DEFAULT_GEBCO_ZIP
    return None


def generate_land_mgrs_list(
    gebco_zip: str,
    destination: str,
    *,
    bbox: Optional[BBox] = None,
    force_refresh: bool = False,
) -> str:
    """Generate or reuse a cached land-MGRS list from GEBCO."""
    if not os.path.exists(gebco_zip):
        raise FileNotFoundError(f"GEBCO zip not found: {gebco_zip}")

    if not force_refresh:
        cached_land_mgrs = load_saved_land_mgrs_list(
            destination,
            bbox=bbox,
            ocean_mask_source=gebco_zip,
        )
        if cached_land_mgrs is not None:
            return destination

    if force_refresh:
        print(f"Force regenerating land MGRS list at {destination}...")
    if bbox is None:
        print("Scanning GEBCO for land tiles...")
    else:
        print("Scanning GEBCO for land tiles within bbox...")

    land_mgrs = discover_mgrs_tiles_from_ocean_mask(gebco_zip, bbox=bbox)
    save_land_mgrs_list(
        destination,
        land_mgrs,
        bbox=bbox,
        ocean_mask_source=gebco_zip,
    )
    return destination


def _extract_s3_mgrs_tiles(
    s3_folder_cache: Mapping[str, Iterable[str]],
) -> Set[str]:
    """Extract unique MGRS tile IDs from the populated S3 folder cache."""
    s3_mgrs_set: Set[str] = set()
    for folders in s3_folder_cache.values():
        for folder in folders:
            parts = folder.split("_")
            if len(parts) >= 5:
                s3_mgrs_set.add(parts[4])
    return s3_mgrs_set


def discover_mgrs_bases(
    bbox: Optional[BBox],
    land_mgrs_source: Optional[str],
    land_mgrs_list_path: Optional[str],
    *,
    force_refresh: bool = False,
    s3_folder_cache: Mapping[str, Iterable[str]],
    discover_mgrs_tiles_in_bbox_fn: Callable[[float, float, float, float], List[str]],
    discover_mgrs_tiles_from_ocean_mask_fn: Callable[[str, Optional[BBox], Optional[Set[str]]], Set[str]],
) -> List[str]:
    """Resolve the requested MGRS tile list from bbox or the default all-tiles flow."""
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_candidates = set(discover_mgrs_tiles_in_bbox_fn(min_lon, min_lat, max_lon, max_lat))
        land_mgrs = None
        if not force_refresh:
            land_mgrs = load_saved_land_mgrs_list(
                land_mgrs_list_path,
                bbox=bbox,
                ocean_mask_source=land_mgrs_source,
            )
            if land_mgrs is None:
                global_land_mgrs = _load_global_land_mgrs_list(land_mgrs_list_path)
                if global_land_mgrs is not None:
                    land_mgrs = global_land_mgrs.intersection(bbox_candidates)
        if land_mgrs is None and land_mgrs_source:
            if force_refresh:
                print(f"Force regenerating land MGRS list at {land_mgrs_list_path}...")
            print("Scanning GEBCO for land tiles within bbox...")
            land_mgrs = discover_mgrs_tiles_from_ocean_mask_fn(
                land_mgrs_source,
                bbox,
                bbox_candidates,
            )
            save_land_mgrs_list(
                land_mgrs_list_path,
                land_mgrs,
                bbox=bbox,
                ocean_mask_source=land_mgrs_source,
            )
        if land_mgrs is not None:
            mgrs_bases = sorted(list(land_mgrs.intersection(bbox_candidates)))
            print(f"Discovered {len(mgrs_bases)} MGRS tiles with land/shallow water inside bbox.")
        else:
            mgrs_bases = sorted(list(bbox_candidates))
            print(f"Discovered {len(mgrs_bases)} MGRS tiles in bbox.")
        return mgrs_bases

    s3_mgrs_set = _extract_s3_mgrs_tiles(s3_folder_cache)
    if not s3_mgrs_set and not (land_mgrs_source and force_refresh):
        print("Error: non-bbox discovery found no tiles in the S3 cache.")
        sys.exit(1)

    land_mgrs = None
    if not force_refresh:
        land_mgrs = _load_global_land_mgrs_list(land_mgrs_list_path)
    if land_mgrs is None and land_mgrs_source:
        if force_refresh:
            print(f"Force regenerating land MGRS list at {land_mgrs_list_path}...")
        print("Scanning GEBCO for land tiles...")
        land_mgrs = discover_mgrs_tiles_from_ocean_mask_fn(
            land_mgrs_source,
            None,
            s3_mgrs_set if s3_mgrs_set else None,
        )
        save_land_mgrs_list(
            land_mgrs_list_path,
            land_mgrs,
            bbox=None,
            ocean_mask_source=land_mgrs_source,
        )

    if land_mgrs is not None:
        if s3_mgrs_set:
            mgrs_bases = sorted(list(land_mgrs.intersection(s3_mgrs_set)))
            print(
                f"All-tiles mode: {len(mgrs_bases)} MGRS tiles found with land/shallow water after S3 intersection."
            )
        else:
            mgrs_bases = sorted(list(land_mgrs))
            print(
                f"All-tiles mode: {len(mgrs_bases)} MGRS tiles found with land/shallow water from land_mgrs.list."
            )
    else:
        mgrs_bases = sorted(list(s3_mgrs_set))
        print(f"All-tiles mode: {len(mgrs_bases)} MGRS tiles found from S3 cache (no ocean mask provided).")

    return mgrs_bases


def add_land_mgrs_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register land-MGRS cache management CLI flags."""
    add = parser.add_argument
    add(
        "--refresh-land-mgrs-list",
        action="store_true",
        help=f"Force regenerate {build_land_mgrs_list_path()} from {ocean.DEFAULT_GEBCO_ZIP} and exit",
    )


def handle_land_mgrs_refresh(
    args: argparse.Namespace,
    requested_bbox: Optional[BBox],
    *,
    land_mgrs_source: Optional[str],
    land_mgrs_list_path: Optional[str],
) -> bool:
    """Handle --refresh-land-mgrs-list and return True when the process should exit."""
    if not args.refresh_land_mgrs_list:
        return False
    if land_mgrs_source is None:
        print(
            f"Error: --refresh-land-mgrs-list requires {ocean.DEFAULT_GEBCO_ZIP} in the current directory."
        )
        sys.exit(1)

    destination = land_mgrs_list_path or build_land_mgrs_list_path()
    generate_land_mgrs_list(
        land_mgrs_source,
        destination,
        bbox=requested_bbox,
        force_refresh=True,
    )
    return True


def build_land_mgrs_argument_parser() -> argparse.ArgumentParser:
    """Build the land-mgrs CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the reusable land_mgrs.list cache from GEBCO so satmaps can skip "
            "repeat land-tile discovery scans."
        )
    )
    add = parser.add_argument
    add(
        "--bbox",
        help=(
            "Optional WGS84 bbox as min_lon,min_lat,max_lon,max_lat. "
            "When omitted, writes the global land/shallow-water cache."
        ),
    )
    add(
        "--refresh",
        action="store_true",
        help="Force regeneration even when an existing matching land_mgrs.list is present",
    )
    add(
        "gebco_zip",
        nargs="?",
        default=ocean.DEFAULT_GEBCO_ZIP,
        help="GEBCO zip archive",
    )
    add(
        "destination",
        nargs="?",
        default=build_land_mgrs_list_path(),
        help="Output land_mgrs.list path",
    )
    return parser


def main() -> None:
    args = build_land_mgrs_argument_parser().parse_args()

    try:
        output_path = generate_land_mgrs_list(
            args.gebco_zip,
            args.destination,
            bbox=ocean.parse_bbox(args.bbox) if args.bbox else None,
            force_refresh=args.refresh,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(output_path)


if __name__ == "__main__":
    main()
