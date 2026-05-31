#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, cast

import mgrs
from mgrs import core as mgrs_core
import numpy as np
from common import (
    LiveProgressLine,
    build_staged_path,
    file_has_content,
    format_eta,
    publish_staged_path,
    remove_if_exists,
)
import land_mgrs as land_mgrs_module
import ocean
from scipy.ndimage import binary_dilation, distance_transform_edt
from osgeo import gdal, ogr, osr

import tiler

# Setup GDAL exceptions
gdal.UseExceptions()

DATE_PATH_QUARTERS = (
    ("07/01", "Q3"),
    ("10/01", "Q4"),
    ("04/01", "Q2"),
    ("01/01", "Q1"),
)
RGB_BANDS = (("B04", "red"), ("B03", "green"), ("B02", "blue"))
SUBTILE_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
SENTINEL_NODATA = -32768
PROCESS_SLAB_HEIGHT = 24
OCEAN_MASK_ALPHA_THRESHOLD = 254.5
OCEAN_MASK_SCAN_PROCESS_BLOCKS = 4
DEFAULT_PREFETCH_IF_LAND = 100.0
DEFAULT_MAX_IN_MEMORY_WRITE_PIXELS = 4_000_000
MGRS_TILE_SIZE_METERS = 100_000.0
LOCAL_SEASON_TRANSITION_TILES = 3.0
LOCAL_SEASON_TRANSITION_HALF_LATITUDE_DEGREES = (
    (LOCAL_SEASON_TRANSITION_TILES * MGRS_TILE_SIZE_METERS) / 2.0
) / 111_320.0
@dataclass(frozen=True)
class LandWorkUnit:
    unit_id: str
    source_subtiles: tuple[str, ...]


@dataclass(frozen=True)
class LandProcessingPlan:
    mgrs_bases: tuple[str, ...]
    work_units: tuple[LandWorkUnit, ...]


def max_in_memory_write_pixels() -> int:
    """Return the live pixel budget for whole-window RGB(A) writes."""
    return tiler.compute_in_memory_pixel_limit(
        48,
        usage_fraction=0.2,
        fallback_pixels=DEFAULT_MAX_IN_MEMORY_WRITE_PIXELS,
        max_pixels=64_000_000,
    )


def temp_basename_from_output(output_path: str) -> str:
    """Return a stable temp-file stem derived from the requested output path."""
    stem = os.path.splitext(os.path.basename(output_path))[0]
    return stem or "satmaps"


def build_state_file_path(unique_id: str) -> str:
    """Return the lightweight JSON resume state path."""
    return f".temp/state_{unique_id}.json"


def build_master_vrt_path(output_path: str, unique_id: str) -> str:
    """Return the run-scoped lightweight master VRT path."""
    return f".temp/master_{temp_basename_from_output(output_path)}_{unique_id}.vrt"


def build_temp_mbtiles_path(output_path: str) -> str:
    """Return the deterministic heavyweight MBTiles path."""
    return f".temp/{temp_basename_from_output(output_path)}.mbtiles"


def build_tile_cache_root(output_path: str, unique_id: str) -> str:
    """Return the run-scoped root directory for cached max-zoom WebP tiles."""
    return f".temp/{temp_basename_from_output(output_path)}_{unique_id}_tilecache"


def build_contributor_tile_cache_dir(output_path: str, unique_id: str, contributor_id: str) -> str:
    """Return the contributor-scoped cache directory for one ocean or land source."""
    return os.path.join(build_tile_cache_root(output_path, unique_id), "contributors", contributor_id)


def build_contributor_complete_marker(
    output_path: str, unique_id: str, contributor_id: str
) -> str:
    """Return the completion marker path for one cached tile contributor."""
    return os.path.join(
        build_tile_cache_root(output_path, unique_id),
        "markers",
        f"{contributor_id}.json",
    )


def build_final_tile_cache_dir(output_path: str, unique_id: str) -> str:
    """Return the final merged max-zoom WebP tree path."""
    return os.path.join(build_tile_cache_root(output_path, unique_id), "final")


def build_tile_cache_commit_root(output_path: str, unique_id: str) -> str:
    """Return the root directory for staged final-tile commit artifacts."""
    return os.path.join(build_tile_cache_root(output_path, unique_id), "commits")


def build_tile_cache_commit_stage_dir(
    output_path: str, unique_id: str, contributor_id: str
) -> str:
    """Return the staging directory for one contributor commit."""
    return os.path.join(build_tile_cache_commit_root(output_path, unique_id), contributor_id)


def build_tile_cache_commit_manifest_path(
    output_path: str, unique_id: str, contributor_id: str
) -> str:
    """Return the manifest path for one contributor commit."""
    return os.path.join(
        build_tile_cache_commit_root(output_path, unique_id),
        f"{contributor_id}.json",
    )


def build_land_mgrs_list_path() -> str:
    """Return the cached land-MGRS list path used to skip repeat ocean scans."""
    return land_mgrs_module.build_land_mgrs_list_path()


def format_progress(current: int, total: int) -> str:
    """Return a short human-readable progress string."""
    if total <= 0:
        return "0/0 (0%)"
    bounded_current = min(max(current, 0), total)
    percent = round((bounded_current / total) * 100)
    return f"{bounded_current}/{total} ({percent}%)"


def update_count_progress(
    progress_line: LiveProgressLine,
    label: str,
    current: int,
    total: int,
    started_at: float,
    detail: str,
    completed_before_start: int = 0,
) -> None:
    """Update a live progress line for count-based work."""
    remaining = None
    now = time.perf_counter()
    elapsed = now - started_at
    eta_total = max(total - completed_before_start, 0)
    eta_current = min(max(current - completed_before_start, 0), eta_total)
    if eta_current > 0 and eta_total > 0 and elapsed > 0.0:
        remaining = elapsed * (eta_total - eta_current) / eta_current
    progress_line.update(
        f"{label} {format_progress(current, total)}; "
        f"{format_eta(remaining, elapsed_seconds=elapsed)}; {detail}"
    )


def build_gdal_progress_callback(
    progress_line: LiveProgressLine,
    step_label: str,
    started_at: float,
) -> Callable[[float, str, object], int]:
    """Adapt GDAL progress updates to the live progress renderer."""
    def callback(complete: float, _message: str, _data: object) -> int:
        bounded_complete = min(max(complete, 0.0), 1.0)
        remaining = None
        elapsed = time.perf_counter() - started_at
        if bounded_complete > 0.0 and elapsed > 0.0:
            remaining = elapsed * (1.0 - bounded_complete) / bounded_complete
        progress_line.update(
            f"{step_label} {bounded_complete * 100:3.0f}%; "
            f"{format_eta(remaining, elapsed_seconds=elapsed)}"
        )
        return 1

    return callback


@dataclass(frozen=True)
class PackagedPMTiles:
    temp_mbtiles: str
    tiling_artifacts: tiler.TilingArtifacts


def convert_raster_to_pmtiles(
    input_raster: str,
    output_path: str,
    *,
    tile_format: str,
    quality: int,
    resample_alg: str,
    chunk_zoom: int,
    parallel: int,
    blocksize: int,
    name: str,
    description: str,
    requested_bbox: Optional[Tuple[float, float, float, float]] = None,
    tiling_options: Optional[Dict[str, object]] = None,
    cleanup_input_paths: Optional[Sequence[str]] = None,
) -> PackagedPMTiles:
    """Tile a Web Mercator raster into MBTiles and convert the result to PMTiles."""
    temp_mbtiles = build_temp_mbtiles_path(output_path)
    run_options: Dict[str, object] = {
        "format": tile_format,
        "quality": quality,
        "resample_alg": resample_alg,
        "chunk_zoom": chunk_zoom,
        "processes": parallel,
        "blocksize": blocksize,
        "name": name,
        "description": description,
    }
    if requested_bbox is not None:
        run_options["chunk_bounds"] = tiler.lonlat_bbox_to_mercator_bounds(*requested_bbox)
    if tiling_options:
        run_options.update(tiling_options)

    print("Generating MBTiles...")
    tiling_artifacts = tiler.run_tiling_simplified(input_raster, temp_mbtiles, run_options)
    if cleanup_input_paths:
        cleanup_temporary_files(cleanup_input_paths)

    print("Converting to PMTiles...")
    staged_output_path = build_staged_path(output_path)
    remove_if_exists(staged_output_path)
    subprocess.run(["pmtiles", "convert", temp_mbtiles, staged_output_path], check=True)
    publish_staged_path(staged_output_path, output_path)
    print(f"Success! {output_path}")
    return PackagedPMTiles(
        temp_mbtiles=temp_mbtiles,
        tiling_artifacts=tiling_artifacts,
    )


def convert_tile_tree_to_pmtiles(
    input_tile_tree: str,
    output_path: str,
    *,
    resample_alg: str,
    max_zoom: int,
    name: str,
    description: str,
    requested_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """Build MBTiles from a final max-zoom WebP tree, then convert it to PMTiles."""
    temp_mbtiles = build_temp_mbtiles_path(output_path)
    print("Generating MBTiles...")
    tiler.build_mbtiles_from_webp_tree(
        input_tile_tree,
        temp_mbtiles,
        name=name,
        description=description,
        maxzoom=max_zoom,
        bounds_wgs84=requested_bbox,
    )
    tiler.build_mbtiles_overviews(temp_mbtiles, resample_alg)
    tiler.finalize_mbtiles_metadata(temp_mbtiles)

    print("Converting to PMTiles...")
    staged_output_path = build_staged_path(output_path)
    remove_if_exists(staged_output_path)
    subprocess.run(["pmtiles", "convert", temp_mbtiles, staged_output_path], check=True)
    publish_staged_path(staged_output_path, output_path)
    print(f"Success! {output_path}")
    return temp_mbtiles


def cleanup_temporary_files(paths: Sequence[str]) -> None:
    """Best-effort cleanup for heavyweight temporary files."""
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f"Warning: Could not remove temporary file {path}: {exc}")


def build_prepared_ocean_path(output_path: str) -> str:
    """Return the deterministic heavyweight bbox-clipped ocean TIFF path."""
    return f".temp/{temp_basename_from_output(output_path)}_ocean_bbox.tif"


def write_tile_cache_marker(
    marker_path: str,
    contributor_id: str,
    tile_relpaths: Sequence[str],
) -> None:
    """Persist a contributor completion marker for resume and introspection."""
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    temp_marker_path = f"{marker_path}.tmp"
    with open(temp_marker_path, "w") as marker_file:
        json.dump(
            {
                "contributor_id": contributor_id,
                "tile_count": len(tile_relpaths),
                "tiles": list(tile_relpaths),
            },
            marker_file,
            indent=2,
        )
    os.replace(temp_marker_path, marker_path)


def write_tile_cache_commit_manifest(
    manifest_path: str,
    contributor_id: str,
    tile_relpaths: Sequence[str],
) -> None:
    """Persist the staged publish manifest for one final-tile commit."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    temp_manifest_path = f"{manifest_path}.tmp"
    with open(temp_manifest_path, "w") as manifest_file:
        json.dump(
            {
                "contributor_id": contributor_id,
                "tile_count": len(tile_relpaths),
                "tiles": list(tile_relpaths),
            },
            manifest_file,
            indent=2,
        )
    os.replace(temp_manifest_path, manifest_path)


def read_tile_cache_commit_manifest(manifest_path: str) -> Tuple[str, List[str]]:
    """Load the staged publish manifest for one final-tile commit."""
    with open(manifest_path) as manifest_file:
        payload = json.load(manifest_file)
    contributor_id = cast(str, payload["contributor_id"])
    tile_relpaths = [cast(str, tile_path) for tile_path in payload.get("tiles", [])]
    return contributor_id, tile_relpaths

def build_processed_tile_paths(mgrs_subtile: str, output_path: str) -> Tuple[str, str]:
    """Return the deterministic heavyweight TIFF paths for one processed subtile."""
    stem = temp_basename_from_output(output_path)
    return (
        f".temp/{stem}_{mgrs_subtile}_utm.tif",
        f".temp/{stem}_{mgrs_subtile}_3857.tif",
    )


def build_work_unit_output_path(work_unit: LandWorkUnit, output_path: str) -> str:
    """Return the deterministic heavyweight TIFF path for one processed work unit."""
    stem = temp_basename_from_output(output_path)
    return f".temp/{stem}_{work_unit.unit_id}_3857.tif"


def build_tile_prefetch_cache_dir(mgrs_subtile: str, output_path: str) -> str:
    """Return the tile-scoped cache directory used for one worker's prefetched inputs."""
    stem = temp_basename_from_output(output_path)
    return f".temp/{stem}_{mgrs_subtile}_prefetch"


def parse_prefetch_if_land(value: str) -> float:
    """Parse a land-percentage threshold for conditional RGB prefetching."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid prefetch-if-land percentage: {value}"
        ) from exc
    if parsed < 0.0 or parsed > 100.0:
        raise argparse.ArgumentTypeError(
            f"prefetch-if-land percentage must be between 0 and 100: {value}"
        )
    return parsed


def describe_land_processing_plan(plan: LandProcessingPlan, num_dates: int) -> str:
    """Return the summary line printed before land processing starts."""
    return (
        f"Expanded {len(plan.mgrs_bases)} MGRS tiles into {len(plan.work_units)} sub-tiles "
        f"across {num_dates} date(s)."
    )


def discover_available_subtiles_from_s3_cache(
    mgrs_bases: List[str],
) -> Optional[Set[str]]:
    """Return the subtiles present in the populated S3 folder cache."""
    if not S3_FOLDER_CACHE:
        return None

    requested_mgrs = set(mgrs_bases)
    available_subtiles: Set[str] = set()
    for folders in S3_FOLDER_CACHE.values():
        for folder in folders:
            parts = folder.split("_")
            if len(parts) < 7 or parts[4] not in requested_mgrs:
                continue
            available_subtiles.add(f"{parts[4]}_{parts[5]}_{parts[6]}")
    return available_subtiles


def plan_subtile_work_units(
    mgrs_bases: List[str],
    available_subtiles: Optional[Set[str]] = None,
) -> tuple[LandWorkUnit, ...]:
    """Return source-subtile work units, filtered to known S3 folders when available."""
    planned_subtiles = expand_subtiles(mgrs_bases)
    if available_subtiles is not None:
        planned_subtiles = [
            subtile for subtile in planned_subtiles if subtile in available_subtiles
        ]

    return tuple(
        LandWorkUnit(
            unit_id=subtile,
            source_subtiles=(subtile,),
        )
        for subtile in planned_subtiles
    )

def build_land_run_token(
    args: argparse.Namespace,
    date_paths: List[str],
    requested_bbox: Optional[Tuple[float, float, float, float]],
    gebco_vrt_source: Optional[str],
) -> str:
    """Return a stable token so identical land-processing runs reuse temp outputs."""
    tonemap_enabled = getattr(args, "tonemap", False)
    payload = json.dumps(
        {
            "output": os.path.abspath(args.output),
            "date_paths": date_paths,
            "bbox": requested_bbox,
            "max_zoom": args.max_zoom,
            "resample_alg": args.resample_alg,
            "stats_min": args.stats_min,
            "stats_max": args.stats_max,
            "tonemap": tonemap_enabled,
            "grade": args.grade,
            "exposure": args.exposure,
            "sb": getattr(args, "sb", tiler.SOFT_KNEE_SHADOW_BREAK) if tonemap_enabled else None,
            "hb": getattr(args, "hb", tiler.SOFT_KNEE_HIGHLIGHT_BREAK) if tonemap_enabled else None,
            "ss": getattr(args, "ss", tiler.SOFT_KNEE_SHADOW_SLOPE) if tonemap_enabled else None,
            "ms": getattr(args, "ms", tiler.SOFT_KNEE_MID_SLOPE) if tonemap_enabled else None,
            "hs": getattr(args, "hs", tiler.SOFT_KNEE_HIGHLIGHT_SLOPE) if tonemap_enabled else None,
            "gamma": args.gamma,
            "shoulder": getattr(args, "shoulder", tiler.DEFAULT_SHOULDER),
            "sat": args.sat,
            "db": args.db,
            "ls": args.ls,
            "ghb": args.ghb,
            "gms": args.gms,
            "ghs": args.ghs,
            "winter": getattr(args, "winter", False),
            "ocean_mask_source": os.path.abspath(gebco_vrt_source) if gebco_vrt_source else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class TileGrid:
    projection: str
    geotransform: Tuple[float, float, float, float, float, float]
    width: int
    height: int


@dataclass
class OceanMaskWarp:
    alpha_band: gdal.Band
    dataset: gdal.Dataset


@dataclass(frozen=True)
class ProcessingWindow:
    xoff: int
    yoff: int
    width: int
    height: int


@dataclass(frozen=True)
class OceanMaskSlab:
    xoff: int
    yoff: int
    width: int
    height: int
    alpha_block: np.ndarray
    coverage_block: np.ndarray
    fill_allowed_block: np.ndarray


# --- Discovery Layer (S3/CDSE Utils) ---


def setup_gdal_cdse() -> None:
    """Configure GDAL for CDSE S3 access."""
    gdal.SetConfigOption("AWS_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
    gdal.SetConfigOption("AWS_HTTPS", "YES")
    gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")
    gdal.SetConfigOption("AWS_PROFILE", "cdse")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")


# Global cache for S3 folder listings to avoid per-tile discovery overhead
S3_FOLDER_CACHE: Dict[str, set[str]] = {}


def list_mosaic_folders_for_tile(
    mgrs_tile: str, date_paths: List[str], cache_dir: str
) -> List[Tuple[str, str]]:
    """Find all S3 folders for a specific MGRS sub-tile across multiple dates using the pre-populated cache."""
    mgrs_id, x, y = mgrs_tile.split("_")
    found = []

    for date_path in date_paths:
        quarter = "Q3"
        for marker, candidate_quarter in DATE_PATH_QUARTERS:
            if marker in date_path:
                quarter = candidate_quarter
                break
        year = date_path.split("/")[0]
        folder = f"Sentinel-2_mosaic_{year}_{quarter}_{mgrs_id}_{x}_{y}"

        # 1. Check local cache first
        local_b04 = os.path.join(
            cache_dir, date_path.replace("/", "-"), f"{mgrs_id}_{x}_{y}_B04.tif"
        )
        if os.path.exists(local_b04):
            found.append((folder, date_path))
            continue

        # 2. Check pre-populated S3 cache
        if date_path in S3_FOLDER_CACHE:
            if folder in S3_FOLDER_CACHE[date_path]:
                found.append((folder, date_path))
        else:
            # Fallback when the shared S3 cache was not pre-populated: check this folder directly.
            s3_path = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder}"
            try:
                if gdal.ReadDir(s3_path):
                    found.append((folder, date_path))
            except RuntimeError as exc:
                print(f"Warning: Could not inspect remote folder {s3_path}: {exc}")

    return found


def get_tile_paths(
    folder_name: str,
    date_path: str,
    cache_dir: Optional[str] = None,
    download: bool = False,
    *,
    quiet: bool = False,
) -> Dict[str, str]:
    """Construct local or S3 paths for RGB bands. Only downloads if 'download' is True."""
    cache_prefix = "_".join(folder_name.split("_")[4:])
    base_s3 = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}/{folder_name}"
    paths = {}

    date_cache_dir = (
        os.path.join(cache_dir, date_path.replace("/", "-")) if cache_dir else None
    )
    for band_id, color_name in RGB_BANDS:
        local_path = (
            os.path.join(date_cache_dir, f"{cache_prefix}_{band_id}.tif")
            if date_cache_dir
            else None
        )

        # Use local if it exists
        if local_path and os.path.exists(local_path):
            paths[color_name] = local_path
            continue

        # Download if requested and local_path is valid
        if download and local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_path = f"{base_s3}/{band_id}.tif"
            if not quiet:
                print(f"Downloading {s3_path} to {local_path}...")
            src_ds = gdal.Open(s3_path)
            if src_ds is None:
                raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName("GTiff").CreateCopy(local_path, src_ds, callback=None)
            paths[color_name] = local_path
            continue

        # Fallback to streaming
        paths[color_name] = f"{base_s3}/{band_id}.tif"

    return paths


def parse_bbox(bbox: str) -> Tuple[float, float, float, float]:
    """Parse a bbox argument or exit with the existing CLI error message."""
    try:
        return tiler.parse_bbox_string(bbox)
    except ValueError:
        print(f"Error: Invalid bbox format: {bbox}")
        sys.exit(1)


def discover_mgrs_tiles_in_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> List[str]:
    """Discover the 100 km MGRS tiles touched by a bbox."""
    m_converter = mgrs.MGRS()
    discovered_mgrs: Set[str] = set()
    lon_steps = int((max_lon - min_lon) / 0.1) + 2
    lat_steps = int((max_lat - min_lat) / 0.1) + 2

    for i in range(lon_steps):
        lon = min(min_lon + i * 0.1, max_lon)
        for j in range(lat_steps):
            lat = min(min_lat + j * 0.1, max_lat)
            try:
                mgrs_res = m_converter.toMGRS(lat, lon, MGRSPrecision=0)
                mgrs_str = (
                    mgrs_res.decode("utf-8") if isinstance(mgrs_res, bytes) else mgrs_res
                )
                discovered_mgrs.add(mgrs_str)
            except mgrs_core.MGRSError:
                continue

    return list(discovered_mgrs)


def build_bbox_geometry(
    bbox: Tuple[float, float, float, float],
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

    dataset_srs = target_srs.Clone()
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    if hasattr(wgs84_srs, "SetAxisMappingStrategy"):
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dataset_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
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
        dataset_srs = target_srs.Clone()
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        if hasattr(wgs84_srs, "SetAxisMappingStrategy"):
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            dataset_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not (dataset_srs.IsSameGeogCS(wgs84_srs) and dataset_srs.IsGeographic()):
            polygon.Transform(osr.CoordinateTransformation(wgs84_srs, dataset_srs))

    return polygon


def discover_mgrs_tiles_from_ocean_mask(
    ocean_mask_src: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    candidate_mgrs_tiles: Optional[Set[str]] = None,
) -> Set[str]:
    """Discover MGRS tiles that contain land or shallow water from alpha-mask or GEBCO depth sources."""
    return land_mgrs_module.discover_mgrs_tiles_from_ocean_mask(
        ocean_mask_src,
        bbox=bbox,
        candidate_mgrs_tiles=candidate_mgrs_tiles,
        progress_factory=LiveProgressLine,
        update_count_progress_fn=update_count_progress,
    )


def discover_mgrs_bases(
    bbox: Optional[Tuple[float, float, float, float]],
    land_mgrs_source: Optional[str],
    land_mgrs_list_path: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> List[str]:
    """Resolve the requested MGRS tile list from bbox or the default all-tiles flow."""
    return land_mgrs_module.discover_mgrs_bases(
        bbox,
        land_mgrs_source,
        land_mgrs_list_path,
        force_refresh=force_refresh,
        s3_folder_cache=S3_FOLDER_CACHE,
        discover_mgrs_tiles_in_bbox_fn=discover_mgrs_tiles_in_bbox,
        discover_mgrs_tiles_from_ocean_mask_fn=discover_mgrs_tiles_from_ocean_mask,
    )


def expand_subtiles(mgrs_bases: List[str]) -> List[str]:
    """Expand each 100 km MGRS tile into its four processing subtiles."""
    return [
        f"{mgrs_tile}_{x}_{y}"
        for mgrs_tile in mgrs_bases
        for x, y in SUBTILE_OFFSETS
    ]


def find_resume_path(
    resume_arg: object,
    preferred_path: Optional[str] = None,
    *,
    allow_latest: bool = True,
) -> Optional[str]:
    """Resolve the explicit or latest state file path for --resume."""
    if isinstance(resume_arg, str) and os.path.exists(resume_arg):
        return resume_arg
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    if not allow_latest:
        return None

    states = glob.glob(".temp/state_*.json")
    if not states:
        return None
    return max(states, key=os.path.getmtime)


def load_saved_land_mgrs_list(
    land_mgrs_list_path: Optional[str],
    *,
    bbox: Optional[Tuple[float, float, float, float]],
    ocean_mask_source: Optional[str],
) -> Optional[Set[str]]:
    """Load a cached land-MGRS list via the shared land-MGRS helper module."""
    return land_mgrs_module.load_saved_land_mgrs_list(
        land_mgrs_list_path,
        bbox=bbox,
        ocean_mask_source=ocean_mask_source,
    )


def save_land_mgrs_list(
    land_mgrs_list_path: Optional[str],
    land_mgrs: Set[str],
    *,
    bbox: Optional[Tuple[float, float, float, float]],
    ocean_mask_source: Optional[str],
) -> None:
    """Persist the scanned land-MGRS list via the shared land-MGRS helper module."""
    land_mgrs_module.save_land_mgrs_list(
        land_mgrs_list_path,
        land_mgrs,
        bbox=bbox,
        ocean_mask_source=ocean_mask_source,
    )


def restore_resume_state(resume_path: str) -> Optional[Dict[str, Any]]:
    """Load a previous run state and keep only surviving intermediate files."""
    try:
        with open(resume_path, "r") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state file must contain a JSON object")
        unique_id = state["unique_id"]
        completed_units_raw = state["completed_units"]
        processed_tifs_raw = state["processed_tifs"]
        if not isinstance(unique_id, str):
            raise ValueError("unique_id must be a string")
        if not isinstance(completed_units_raw, list):
            raise ValueError("completed_units must be a list")
        if not isinstance(processed_tifs_raw, list):
            raise ValueError("processed_tifs must be a list")
        if not all(isinstance(unit_id, str) for unit_id in completed_units_raw):
            raise ValueError("completed_units entries must be strings")
        if not all(isinstance(path, str) for path in processed_tifs_raw):
            raise ValueError("processed_tifs entries must be strings")
    except (OSError, ValueError, TypeError, KeyError) as e:
        print(f"Warning: Could not load state file: {e}")
        return None

    completed_units = set(completed_units_raw)
    processed_tifs = [path for path in processed_tifs_raw if file_has_content(path)]
    print(f"Resuming from state file: {resume_path} (unique_id: {unique_id})")
    print(
        f"Already completed {len(completed_units)} sub-tile(s), "
        f"{len(processed_tifs)} TIFs found."
    )
    return {
        "state_file": resume_path,
        "unique_id": unique_id,
        "completed_units": completed_units,
        "processed_tifs": processed_tifs,
    }


def should_use_webp_cache_pipeline(args: argparse.Namespace) -> bool:
    """Return True when the run should use the max-zoom WebP cache pipeline."""
    return not args.vrt and args.format == "webp"


def recover_cached_work_outputs(
    work_units: tuple[LandWorkUnit, ...],
    output_path: str,
    unique_id: str,
) -> Set[str]:
    """Collect completed work units from cache markers."""
    recovered_units: Set[str] = set()
    for work_unit in work_units:
        marker_path = build_contributor_complete_marker(
            output_path,
            unique_id,
            work_unit.unit_id,
        )
        if file_has_content(marker_path):
            recovered_units.add(work_unit.unit_id)
    return recovered_units


def recover_pending_tile_cache_commits(
    output_path: str,
    unique_id: str,
) -> Set[str]:
    """Finish any interrupted staged final-tile publishes before resume checks."""
    commit_root = build_tile_cache_commit_root(output_path, unique_id)
    if not os.path.isdir(commit_root):
        return set()

    final_tile_tree = build_final_tile_cache_dir(output_path, unique_id)
    recovered_contributors: Set[str] = set()
    manifest_paths = sorted(glob.glob(os.path.join(commit_root, "*.json")))
    manifest_contributors: Set[str] = set()
    for manifest_path in manifest_paths:
        contributor_id, tile_relpaths = read_tile_cache_commit_manifest(manifest_path)
        manifest_contributors.add(contributor_id)
        stage_dir = build_tile_cache_commit_stage_dir(output_path, unique_id, contributor_id)
        tiler.publish_staged_webp_tree_commit(stage_dir, final_tile_tree, tile_relpaths)
        write_tile_cache_marker(
            build_contributor_complete_marker(output_path, unique_id, contributor_id),
            contributor_id,
            tile_relpaths,
        )
        remove_if_exists(manifest_path)
        recovered_contributors.add(contributor_id)

    for entry_name in os.listdir(commit_root):
        entry_path = os.path.join(commit_root, entry_name)
        if os.path.isdir(entry_path) and entry_name not in manifest_contributors:
            shutil.rmtree(entry_path)
    return recovered_contributors


def commit_raster_to_final_tile_cache(
    input_raster: str,
    output_path: str,
    unique_id: str,
    contributor_id: str,
    args: argparse.Namespace,
) -> List[str]:
    """Stage and publish one contributor directly into the shared final WebP tree."""
    final_tile_tree = build_final_tile_cache_dir(output_path, unique_id)
    stage_dir = build_tile_cache_commit_stage_dir(output_path, unique_id, contributor_id)
    manifest_path = build_tile_cache_commit_manifest_path(output_path, unique_id, contributor_id)
    marker_path = build_contributor_complete_marker(output_path, unique_id, contributor_id)

    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    remove_if_exists(manifest_path)

    tile_relpaths = tiler.stage_raster_to_webp_tree_commit(
        input_raster,
        final_tile_tree,
        stage_dir,
        args.max_zoom,
        args.blocksize,
        args.quality,
        args.resample_alg,
    )
    write_tile_cache_commit_manifest(manifest_path, contributor_id, tile_relpaths)
    tiler.publish_staged_webp_tree_commit(stage_dir, final_tile_tree, tile_relpaths)
    write_tile_cache_marker(marker_path, contributor_id, tile_relpaths)
    remove_if_exists(manifest_path)
    return tile_relpaths


def resolve_ocean_mask_source(ocean_background: str) -> Optional[str]:
    """Reuse the configured ocean background as the optional mask source when it has a usable mask band."""
    if not os.path.exists(ocean_background):
        return None

    try:
        dataset = gdal.Open(ocean_background)
    except RuntimeError as exc:
        print(f"Warning: Could not open ocean mask source {ocean_background}: {exc}")
        return None

    if dataset is None:
        print(f"Warning: Could not open ocean mask source {ocean_background}")
        return None

    if get_ocean_mask_band_index(dataset) is None:
        dataset = None
        return None

    dataset = None
    return ocean_background


def resolve_land_mgrs_source() -> Optional[str]:
    """Resolve the GEBCO source used for land/shallow-water MGRS discovery."""
    return land_mgrs_module.resolve_land_mgrs_source()


def prepare_ocean_background_for_output(
    ocean_background_path: str,
    requested_bbox: Optional[Tuple[float, float, float, float]],
    output_path: str,
    resample_alg: str,
    max_zoom: int,
    blocksize: int,
) -> Optional[str]:
    """Prepare the ocean background used in the final 3857 composite."""
    if not os.path.exists(ocean_background_path):
        return None
    if requested_bbox is None:
        return ocean_background_path

    snapped_bounds, pixel_size, _zoom = ocean.snapped_tile_grid_for_bbox(
        requested_bbox,
        max_zoom,
        tile_size=blocksize,
    )
    prepared_ocean_path = build_prepared_ocean_path(output_path)
    staged_ocean_path = build_staged_path(prepared_ocean_path)
    remove_if_exists(staged_ocean_path)
    dataset: Optional[gdal.Dataset] = None
    try:
        try:
            dataset = gdal.Open(ocean_background_path)
        except RuntimeError:
            dataset = None

        aligned_src_win = (
            get_aligned_web_mercator_src_win(dataset, snapped_bounds, pixel_size)
            if dataset is not None
            else None
        )
        if aligned_src_win is not None:
            prepared_ds = gdal.Translate(
                staged_ocean_path,
                dataset,
                options=gdal.TranslateOptions(
                    format="GTiff",
                    srcWin=aligned_src_win,
                    creationOptions=list(ocean.GTIFF_CREATION_OPTIONS),
                ),
            )
        else:
            warp_options = gdal.WarpOptions(
                format="GTiff",
                dstSRS="EPSG:3857",
                outputBounds=snapped_bounds,
                xRes=pixel_size,
                yRes=pixel_size,
                resampleAlg=resample_alg,
                multithread=True,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                creationOptions=list(ocean.GTIFF_CREATION_OPTIONS),
            )
            prepared_ds = gdal.Warp(staged_ocean_path, ocean_background_path, options=warp_options)
        if prepared_ds is None:
            raise RuntimeError(
                f"Could not prepare bbox ocean background from {ocean_background_path}"
            )
        prepared_ds = None
        publish_staged_path(staged_ocean_path, prepared_ocean_path)
        return prepared_ocean_path
    finally:
        dataset = None


def get_ocean_mask_band_index(dataset: gdal.Dataset) -> Optional[int]:
    """Return the explicit alpha band used as the ocean-mask handoff."""
    for band_index in range(1, dataset.RasterCount + 1):
        if dataset.GetRasterBand(band_index).GetColorInterpretation() == gdal.GCI_AlphaBand:
            return band_index

    return None


def get_bbox_scan_window(
    dataset: gdal.Dataset,
    bbox: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[int, int, int, int]]:
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


def get_aligned_web_mercator_src_win(
    dataset: gdal.Dataset,
    target_bounds: Tuple[float, float, float, float],
    pixel_size: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Return a direct crop window when a source raster already matches the requested 3857 grid."""
    dataset_srs = dataset.GetSpatialRef()
    if dataset_srs is None:
        return None

    dataset_srs = dataset_srs.Clone()
    web_mercator_srs = osr.SpatialReference()
    web_mercator_srs.ImportFromEPSG(3857)
    if hasattr(web_mercator_srs, "SetAxisMappingStrategy"):
        web_mercator_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dataset_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not dataset_srs.IsSame(web_mercator_srs):
        return None

    gt = dataset.GetGeoTransform()
    if abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        return None
    if abs(gt[1] - pixel_size) > 1e-9 or abs(abs(gt[5]) - pixel_size) > 1e-9:
        return None

    src_win = tiler.te_to_src_win(dataset, target_bounds)
    if src_win[2] <= 0 or src_win[3] <= 0:
        return None

    fitted_bounds = (
        gt[0] + src_win[0] * gt[1],
        gt[3] + (src_win[1] + src_win[3]) * gt[5],
        gt[0] + (src_win[0] + src_win[2]) * gt[1],
        gt[3] + src_win[1] * gt[5],
    )
    if not all(abs(actual - expected) <= 1e-6 for actual, expected in zip(fitted_bounds, target_bounds)):
        return None

    return src_win


def populate_s3_cache(date_paths: List[str]) -> None:
    """Cache remote folder listings for all-tiles runs."""
    print(f"Populating S3 folder cache for {len(date_paths)} date(s)...")
    total_folders = 0
    progress_line = LiveProgressLine()
    started_at = time.perf_counter()
    for index, date_path in enumerate(date_paths, start=1):
        s3_base = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}"
        update_count_progress(
            progress_line,
            "S3 cache progress:",
            index,
            len(date_paths),
            started_at,
            f"listing {date_path}...",
        )
        dirs = gdal.ReadDir(s3_base)
        if dirs:
            S3_FOLDER_CACHE[date_path] = set(dirs)
            total_folders += len(dirs)
        else:
            progress_line.finish()
            print(f"Warning: Could not list folders for {date_path}")
    progress_line.finish()
    print(
        f"S3 folder cache ready: {len(S3_FOLDER_CACHE)} date(s), "
        f"{total_folders} folders total."
    )


# --- Processing Layer (NumPy Manipulation) ---


def fill_nan_nearest(
    arr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    fill_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fill invalid pixels in a (C, H, W) array with the nearest valid pixel."""
    if valid_mask is None:
        valid_mask = np.all(np.isfinite(arr), axis=0)

    if valid_mask.shape != arr.shape[1:]:
        raise ValueError("valid_mask must match the spatial shape of arr")

    if fill_mask is not None and fill_mask.shape != arr.shape[1:]:
        raise ValueError("fill_mask must match the spatial shape of arr")

    target_mask = ~valid_mask
    if fill_mask is not None:
        target_mask &= fill_mask

    if not np.any(target_mask) or not np.any(valid_mask):
        return arr

    indices = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True
    )

    filled = np.array(arr, copy=True)
    for i in range(arr.shape[0]):
        nearest_band = arr[i][tuple(indices)]
        filled[i][target_mask] = nearest_band[target_mask]

    return filled


def load_tile_grid(red_path: str) -> TileGrid:
    """Read the common raster grid metadata from the red band."""
    dataset = gdal.Open(red_path)
    tile_grid = TileGrid(
        projection=dataset.GetProjection(),
        geotransform=dataset.GetGeoTransform(),
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
    )
    dataset = None
    return tile_grid


def build_projection_to_wgs84_transform(projection: str) -> osr.CoordinateTransformation:
    """Build a traditional GIS-order transform from the source projection to WGS84."""
    if not projection:
        raise RuntimeError("Tile grid projection is required for seasonal blending")

    source_srs = osr.SpatialReference()
    if source_srs.ImportFromWkt(projection) != 0:
        raise RuntimeError("Could not parse tile grid projection for seasonal blending")
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    if hasattr(source_srs, "SetAxisMappingStrategy"):
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(source_srs, wgs84_srs)


def compute_block_row_latitudes(
    tile_grid: TileGrid,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Return one WGS84 latitude per raster row using the block's horizontal center."""
    transform = build_projection_to_wgs84_transform(tile_grid.projection)
    center_col = xoff + (width / 2.0)
    latitudes = np.empty(height, dtype=np.float32)

    for row_index in range(height):
        center_row = yoff + row_index + 0.5
        x = tile_grid.geotransform[0] + (center_col * tile_grid.geotransform[1]) + (
            center_row * tile_grid.geotransform[2]
        )
        y = tile_grid.geotransform[3] + (center_col * tile_grid.geotransform[4]) + (
            center_row * tile_grid.geotransform[5]
        )
        _lon, lat, *_rest = transform.TransformPoint(float(x), float(y))
        latitudes[row_index] = float(lat)

    return latitudes


def smoothstep(values: np.ndarray) -> np.ndarray:
    """Clamp to [0, 1] and ease with a cubic smoothstep."""
    clipped = np.clip(values, 0.0, 1.0)
    return cast(np.ndarray, clipped * clipped * (3.0 - (2.0 * clipped)))


def build_local_season_date_weights(
    tile_grid: TileGrid,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
    *,
    winter: bool,
) -> np.ndarray:
    """Return first-date/second-date row weights for the equator-centered seasonal blend."""
    latitudes = compute_block_row_latitudes(tile_grid, xoff, yoff, width, height)
    primary_northward_weight = smoothstep(
        (latitudes + LOCAL_SEASON_TRANSITION_HALF_LATITUDE_DEGREES)
        / (2.0 * LOCAL_SEASON_TRANSITION_HALF_LATITUDE_DEGREES)
    ).astype(np.float32)
    if winter:
        primary_northward_weight = 1.0 - primary_northward_weight
    secondary_weight = 1.0 - primary_northward_weight
    return np.stack(
        (primary_northward_weight[:, np.newaxis], secondary_weight[:, np.newaxis])
    ).astype(np.float32)


def open_gebco_mask(
    gebco_src: Optional[str],
    tile_grid: TileGrid,
    mgrs_subtile: str,
) -> Optional[OceanMaskWarp]:
    """Warp GEBCO onto the tile grid so land/ocean blending can reuse it block-by-block."""
    if not gebco_src:
        return None

    try:
        gebco_ds = gdal.Open(gebco_src)
        if gebco_ds is None:
            raise RuntimeError(f"Could not open GEBCO source: {gebco_src}")
        band_index = get_ocean_mask_band_index(gebco_ds)
        if band_index is None:
            raise RuntimeError(f"Could not find a usable mask band in: {gebco_src}")

        try:
            mask_nodata = -1.0
            min_x = tile_grid.geotransform[0]
            max_y = tile_grid.geotransform[3]
            max_x = min_x + tile_grid.geotransform[1] * tile_grid.width
            min_y = max_y + tile_grid.geotransform[5] * tile_grid.height

            warped_mask_ds = gdal.Warp(
                "",
                gebco_ds,
                options=gdal.WarpOptions(
                    format="MEM",
                    dstSRS=tile_grid.projection,
                    outputBounds=(min_x, min_y, max_x, max_y),
                    width=tile_grid.width,
                    height=tile_grid.height,
                    resampleAlg="bilinear",
                    srcBands=[band_index],
                    srcAlpha=False,
                    dstNodata=mask_nodata,
                    workingType=gdal.GDT_Float32,
                    outputType=gdal.GDT_Float32,
                    multithread=True,
                    warpOptions=["NUM_THREADS=ALL_CPUS"],
                ),
            )
            if warped_mask_ds is None:
                raise RuntimeError(f"Could not warp GEBCO mask for {mgrs_subtile}")
            warped_mask_ds.GetRasterBand(1).SetNoDataValue(mask_nodata)
        finally:
            gebco_ds = None

        return OceanMaskWarp(
            alpha_band=warped_mask_ds.GetRasterBand(1),
            dataset=warped_mask_ds,
        )
    except RuntimeError as e:
        print(f"Warning: Could not apply GEBCO mask to {mgrs_subtile}: {e}")
        return None


def build_fill_allowed_mask(
    ocean_mask_alpha: np.ndarray,
    coverage_mask: Optional[np.ndarray] = None,
    nodata_value: Optional[float] = None,
) -> np.ndarray:
    """Return the pixels where Sentinel land rendering is allowed."""
    fill_allowed_mask = ocean_mask_alpha < OCEAN_MASK_ALPHA_THRESHOLD
    if nodata_value is not None:
        fill_allowed_mask &= ocean_mask_alpha != nodata_value

    if coverage_mask is not None:
        if coverage_mask.shape != fill_allowed_mask.shape:
            raise ValueError("coverage_mask must match the spatial shape of ocean_mask_alpha")
        fill_allowed_mask &= coverage_mask

    return fill_allowed_mask


def collect_ocean_mask_slabs(
    ocean_mask: OceanMaskWarp,
    tile_grid: TileGrid,
) -> tuple[ProcessingWindow, dict[int, OceanMaskSlab]] | None:
    """Read cropped mask slabs once so later stages can reuse them without rereads."""
    slabs: dict[int, OceanMaskSlab] = {}
    min_x = tile_grid.width
    max_x = 0
    min_y = tile_grid.height
    max_y = 0
    mask_nodata = ocean_mask.alpha_band.GetNoDataValue()

    for _xoff, yoff, block_width, block_height in iter_processing_windows(tile_grid):
        alpha_block = ocean_mask.alpha_band.ReadAsArray(0, yoff, block_width, block_height).astype(
            np.float32
        )
        if mask_nodata is None:
            coverage_block = np.ones(alpha_block.shape, dtype=bool)
        else:
            coverage_block = alpha_block != mask_nodata
        fill_allowed_block = build_fill_allowed_mask(
            alpha_block,
            coverage_block,
            nodata_value=mask_nodata,
        )
        slab_crop_block = coverage_block & binary_dilation(
            fill_allowed_block,
            structure=np.ones((3, 3), dtype=bool),
        )

        cols_with_land = np.any(slab_crop_block, axis=0)
        if not np.any(cols_with_land):
            continue

        x_start = int(np.argmax(cols_with_land))
        x_end = int(len(cols_with_land) - np.argmax(cols_with_land[::-1]))
        cropped = OceanMaskSlab(
            xoff=x_start,
            yoff=yoff,
            width=x_end - x_start,
            height=block_height,
            alpha_block=alpha_block[:, x_start:x_end],
            coverage_block=coverage_block[:, x_start:x_end],
            fill_allowed_block=fill_allowed_block[:, x_start:x_end],
        )
        slabs[yoff] = cropped
        min_x = min(min_x, cropped.xoff)
        max_x = max(max_x, cropped.xoff + cropped.width)
        min_y = min(min_y, cropped.yoff)
        max_y = max(max_y, cropped.yoff + cropped.height)

    if not slabs:
        return None

    return (
        ProcessingWindow(
            xoff=min_x,
            yoff=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        ),
        slabs,
    )


def prefetch_tile_bands_locally(
    folders: List[Tuple[str, str]],
    cache_dir: str,
) -> None:
    """Download all RGB inputs needed by one worker into a tile-scoped cache."""
    for folder_name, date_path in folders:
        get_tile_paths(folder_name, date_path, cache_dir, download=True, quiet=True)


def cleanup_prefetched_tile_bands(cache_dir: Optional[str]) -> None:
    """Remove the tile-scoped cache after a worker finishes processing."""
    if cache_dir is None:
        return
    shutil.rmtree(cache_dir, ignore_errors=True)


def estimate_subtile_land_percentage(
    mask_slabs: Optional[dict[int, OceanMaskSlab]],
    tile_grid: TileGrid,
) -> Optional[float]:
    """Estimate land coverage percentage from the already warped ocean-mask slabs."""
    if mask_slabs is None:
        return None

    total_pixels = tile_grid.width * tile_grid.height
    if total_pixels <= 0:
        return 0.0

    land_pixels = sum(
        int(np.count_nonzero(slab.fill_allowed_block))
        for slab in mask_slabs.values()
    )
    return (land_pixels * 100.0) / total_pixels


def should_prefetch_tile_bands(
    prefetch_if_land: float,
    mask_slabs: Optional[dict[int, OceanMaskSlab]],
    tile_grid: TileGrid,
) -> bool:
    """Return whether this worker should prefetch its date RGB bands."""
    if prefetch_if_land >= 100.0:
        return True
    if prefetch_if_land <= 0.0:
        return False

    land_percentage = estimate_subtile_land_percentage(mask_slabs, tile_grid)
    if land_percentage is None:
        return True
    return land_percentage >= prefetch_if_land


def open_date_band_sets(
    folders: List[Tuple[str, str]], cache_dir: str
) -> List[Tuple[List[gdal.Band], List[gdal.Dataset]]]:
    """Open RGB band handles for every date so block reads stay sequential."""
    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]] = []
    try:
        for folder_name, date_path in folders:
            paths = get_tile_paths(folder_name, date_path, cache_dir, download=False)
            datasets: List[gdal.Dataset] = []
            bands: List[gdal.Band] = []
            for band_id, color_name in RGB_BANDS:
                try:
                    dataset = gdal.Open(paths[color_name])
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Could not open {color_name} band {band_id} for {folder_name} ({date_path}): {exc}"
                    ) from exc
                if dataset is None:
                    raise RuntimeError(
                        f"Could not open {color_name} band {band_id} for {folder_name} ({date_path})"
                    )
                band = dataset.GetRasterBand(1)
                if band is None:
                    raise RuntimeError(
                        f"Could not read raster band 1 from {color_name} band {band_id} for {folder_name} ({date_path})"
                    )
                datasets.append(dataset)
                bands.append(band)
            date_band_sets.append((bands, datasets))
        return date_band_sets
    except RuntimeError:
        for bands, datasets in date_band_sets:
            bands.clear()
            datasets.clear()
        raise


def write_resume_state(
    state_file: str,
    unique_id: str,
    completed_units: Set[str],
    processed_tifs: List[str],
    args: argparse.Namespace,
) -> None:
    """Persist resume state atomically so interrupted writes do not corrupt the JSON file."""
    temp_state_file = f"{state_file}.tmp"
    with open(temp_state_file, "w") as f:
        json.dump(
            {
                "unique_id": unique_id,
                "completed_units": list(completed_units),
                "processed_tifs": processed_tifs,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    os.replace(temp_state_file, state_file)


def parse_tile_tree_relpath(relative_path: str) -> Tuple[int, int, int]:
        """Parse a z/x/y.webp path relative to a tile tree root."""
        parts = relative_path.split(os.sep)
        if len(parts) != 3:
            raise ValueError(f"Invalid tile tree relative path: {relative_path}")
        zoom = int(parts[0])
        tx = int(parts[1])
        ty = int(os.path.splitext(parts[2])[0])
        return zoom, tx, ty


def iter_processing_windows(tile_grid: TileGrid) -> Iterator[Tuple[int, int, int, int]]:
    """Yield full-width row slabs that match the striped source TIFF layout."""
    for yoff in range(0, tile_grid.height, PROCESS_SLAB_HEIGHT):
        yield 0, yoff, tile_grid.width, min(PROCESS_SLAB_HEIGHT, tile_grid.height - yoff)


def average_block(
    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]],
    xoff: int,
    yoff: int,
    width: int,
    height: int,
    *,
    date_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average one block across all dates while requiring complete RGB observations."""
    weighted_sum = np.zeros((3, height, width), dtype=np.float32)
    weight_totals = np.zeros((height, width), dtype=np.float32)
    fallback_sum = np.zeros((3, height, width), dtype=np.float32)
    fallback_counts = np.zeros((height, width), dtype=np.uint16)
    valid_source_mask = np.zeros((height, width), dtype=bool)

    for date_index, (bands, _) in enumerate(date_band_sets):
        rgb_block = np.empty((3, height, width), dtype=np.float32)
        for band_index, band in enumerate(bands):
            block = band.ReadAsArray(xoff, yoff, width, height).astype(np.float32)
            block[block == SENTINEL_NODATA] = np.nan
            rgb_block[band_index] = block
        complete_rgb_mask = np.all(np.isfinite(rgb_block), axis=0)
        if not np.any(complete_rgb_mask):
            continue

        valid_source_mask |= complete_rgb_mask
        fallback_counts[complete_rgb_mask] += 1
        if date_weights is None:
            block_weights = complete_rgb_mask.astype(np.float32)
        else:
            block_weights = np.where(complete_rgb_mask, date_weights[date_index], 0.0)
        np.add(weight_totals, block_weights, out=weight_totals)
        for band_index in range(rgb_block.shape[0]):
            np.add(
                fallback_sum[band_index],
                rgb_block[band_index],
                out=fallback_sum[band_index],
                where=complete_rgb_mask,
            )
            np.add(
                weighted_sum[band_index],
                np.where(complete_rgb_mask, rgb_block[band_index] * block_weights, 0.0),
                out=weighted_sum[band_index],
            )

    averaged_block = np.full((3, height, width), np.nan, dtype=np.float32)
    weighted_valid_mask = weight_totals > 0.0
    fallback_mask = valid_source_mask & ~weighted_valid_mask
    for band_index in range(averaged_block.shape[0]):
        np.divide(
            weighted_sum[band_index],
            weight_totals,
            out=averaged_block[band_index],
            where=weighted_valid_mask,
        )
        np.divide(
            fallback_sum[band_index],
            fallback_counts,
            out=averaged_block[band_index],
            where=fallback_mask,
        )

    return averaged_block, valid_source_mask


def average_tile_blocks(
    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]],
    processing_window: ProcessingWindow,
    mask_slabs: Optional[dict[int, OceanMaskSlab]],
    *,
    tile_grid: Optional[TileGrid] = None,
    winter: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Average the processing window and retain the matching output alpha blocks."""
    averaged = np.full(
        (3, processing_window.height, processing_window.width), np.nan, dtype=np.float32
    )
    source_valid_mask = np.zeros((processing_window.height, processing_window.width), dtype=bool)
    alpha_mask = np.zeros((processing_window.height, processing_window.width), dtype=np.uint8)
    fill_allowed_mask = (
        np.zeros((processing_window.height, processing_window.width), dtype=bool)
        if mask_slabs is not None
        else None
    )
    use_local_season_blend = tile_grid is not None and len(date_band_sets) == 2

    for relative_yoff in range(0, processing_window.height, PROCESS_SLAB_HEIGHT):
        yoff = processing_window.yoff + relative_yoff
        block_height = min(PROCESS_SLAB_HEIGHT, processing_window.height - relative_yoff)

        if mask_slabs is None:
            actual_xoff = processing_window.xoff
            actual_width = processing_window.width
            alpha_block = None
            coverage_block = None
            allowed_block = None
        else:
            slab = mask_slabs.get(yoff)
            if slab is None:
                continue
            actual_xoff = slab.xoff
            actual_width = slab.width
            alpha_block = slab.alpha_block
            coverage_block = slab.coverage_block
            allowed_block = slab.fill_allowed_block

        if use_local_season_blend:
            assert tile_grid is not None
            averaged_block, block_valid_sources = average_block(
                date_band_sets,
                actual_xoff,
                yoff,
                actual_width,
                block_height,
                date_weights=build_local_season_date_weights(
                    tile_grid,
                    actual_xoff,
                    yoff,
                    actual_width,
                    block_height,
                    winter=winter,
                ),
            )
        else:
            averaged_block, block_valid_sources = average_block(
                date_band_sets, actual_xoff, yoff, actual_width, block_height
            )
        if allowed_block is not None:
            averaged_block = np.where(allowed_block[np.newaxis, :, :], averaged_block, np.nan)
            block_valid_sources &= allowed_block
            if coverage_block is not None:
                seam_fill_block = coverage_block & ~allowed_block & binary_dilation(
                    block_valid_sources,
                    structure=np.ones((3, 3), dtype=bool),
                )
                allowed_block = allowed_block | seam_fill_block

        relative_xoff = actual_xoff - processing_window.xoff
        row_slice = slice(relative_yoff, relative_yoff + block_height)
        col_slice = slice(relative_xoff, relative_xoff + actual_width)
        averaged[:, row_slice, col_slice] = averaged_block
        source_valid_mask[row_slice, col_slice] = block_valid_sources

        alpha_mask[row_slice, col_slice] = build_alpha_block(
            alpha_block,
            block_valid_sources,
            coverage_block,
            allowed_block,
        )
        if fill_allowed_mask is not None and allowed_block is not None:
            fill_allowed_mask[row_slice, col_slice] = allowed_block

    return averaged, source_valid_mask, alpha_mask, fill_allowed_mask


def fill_missing_pixels(
    averaged: np.ndarray,
    source_valid_mask: np.ndarray,
    fill_allowed_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Fill gaps only when at least one missing pixel is eligible for coastal interpolation."""
    fill_mask = ~source_valid_mask
    if fill_allowed_mask is not None:
        fill_mask &= fill_allowed_mask
    if np.any(fill_mask):
        return fill_nan_nearest(
            averaged,
            valid_mask=source_valid_mask,
            fill_mask=fill_allowed_mask,
        )
    return averaged


def create_output_dataset(
    output_path: Optional[str],
    tile_grid: TileGrid,
    *,
    driver_name: str = "GTiff",
) -> Tuple[gdal.Dataset, List[gdal.Band], gdal.Band]:
    """Create a temporary RGBA dataset used before Web Mercator warping."""
    driver = gdal.GetDriverByName(driver_name)
    destination = output_path if output_path is not None else ""
    creation_options = ["COMPRESS=ZSTD", "TILED=YES"] if driver_name == "GTiff" else []
    dataset = driver.Create(
        destination,
        tile_grid.width,
        tile_grid.height,
        4,
        gdal.GDT_Byte,
        options=creation_options,
    )
    dataset.SetProjection(tile_grid.projection)
    dataset.SetGeoTransform(tile_grid.geotransform)

    color_bands = [dataset.GetRasterBand(index + 1) for index in range(3)]
    for band_index, color_name in enumerate(("RedBand", "GreenBand", "BlueBand")):
        color_bands[band_index].SetColorInterpretation(getattr(gdal, f"GCI_{color_name}"))
        color_bands[band_index].Fill(0)

    alpha_band = dataset.GetRasterBand(4)
    alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
    alpha_band.Fill(0)
    return dataset, color_bands, alpha_band


def build_alpha_block(
    ocean_mask_alpha: Optional[np.ndarray],
    source_valid_block: np.ndarray,
    coverage_block: Optional[np.ndarray] = None,
    fill_allowed_block: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the output alpha band for one block."""
    if ocean_mask_alpha is None:
        alpha_block = source_valid_block.astype(np.uint8) * 255
    elif fill_allowed_block is not None:
        alpha_block = fill_allowed_block.astype(np.uint8) * 255
    else:
        alpha_block = np.clip(255.0 - ocean_mask_alpha, 0.0, 255.0).astype(np.uint8)

    if fill_allowed_block is not None:
        if fill_allowed_block.shape != source_valid_block.shape:
            raise ValueError(
                "fill_allowed_block must match the spatial shape of source_valid_block"
            )
        # Filled seam pixels should prefer land coverage so the final composite never exposes
        # unexpected fully transparent gaps between land and ocean.
        alpha_block = np.where(fill_allowed_block & ~source_valid_block, 255, alpha_block)

    if coverage_block is not None:
        if coverage_block.shape != source_valid_block.shape:
            raise ValueError("coverage_block must match the spatial shape of source_valid_block")
        alpha_block = np.where(coverage_block, alpha_block, 0)

    return alpha_block.astype(np.uint8)


def write_processed_blocks(
    averaged: np.ndarray,
    alpha_mask: np.ndarray,
    processing_window: ProcessingWindow,
    args: argparse.Namespace,
    color_bands: List[gdal.Band],
    alpha_band: gdal.Band,
) -> None:
    """Convert averaged float data to byte RGB(A) output one block at a time."""
    source_min = args.stats_min if args.stats_min is not None else 0.0
    source_max = args.stats_max if args.stats_max is not None else 9000.0
    scale = source_max - source_min
    if scale <= 0.0:
        raise ValueError("stats_max must be greater than stats_min")

    if processing_window.width * processing_window.height <= max_in_memory_write_pixels():
        byte_block = tone_mapped_byte_block(averaged, args, source_min, scale)
        for band_index, out_band in enumerate(color_bands):
            out_band.WriteArray(byte_block[band_index], xoff=processing_window.xoff, yoff=processing_window.yoff)
        alpha_band.WriteArray(
            alpha_mask,
            xoff=processing_window.xoff,
            yoff=processing_window.yoff,
        )
        return

    for relative_yoff in range(0, processing_window.height, PROCESS_SLAB_HEIGHT):
        yoff = processing_window.yoff + relative_yoff
        block_height = min(PROCESS_SLAB_HEIGHT, processing_window.height - relative_yoff)
        averaged_block = averaged[
            :, relative_yoff : relative_yoff + block_height, :
        ]
        byte_block = tone_mapped_byte_block(averaged_block, args, source_min, scale)
        for band_index, out_band in enumerate(color_bands):
            out_band.WriteArray(byte_block[band_index], xoff=processing_window.xoff, yoff=yoff)
        alpha_band.WriteArray(
            alpha_mask[relative_yoff : relative_yoff + block_height, :],
            xoff=processing_window.xoff,
            yoff=yoff,
        )


def tone_mapped_byte_block(
    averaged_block: np.ndarray,
    args: argparse.Namespace,
    source_min: float,
    scale: float,
) -> np.ndarray:
    """Return an 8-bit RGB block after normalization, tonemapping, and grading."""
    normalized = np.clip((averaged_block - source_min) / scale, 0.0, 1.0)
    normalized[np.isnan(normalized)] = 0.0
    exposure = getattr(args, "exposure", tiler.DEFAULT_EXPOSURE)

    if getattr(args, "tonemap", False):
        toned_block = tiler.apply_soft_knee_numpy(
            normalized,
            shadow_break=getattr(args, "sb", tiler.SOFT_KNEE_SHADOW_BREAK),
            highlight_break=getattr(args, "hb", tiler.SOFT_KNEE_HIGHLIGHT_BREAK),
            shadow_slope=getattr(args, "ss", tiler.SOFT_KNEE_SHADOW_SLOPE),
            mid_slope=getattr(args, "ms", tiler.SOFT_KNEE_MID_SLOPE),
            highlight_slope=getattr(args, "hs", tiler.SOFT_KNEE_HIGHLIGHT_SLOPE),
            exposure=exposure,
        )
    else:
        toned_block = np.clip(normalized * exposure, 0.0, 1.0)

    if args.grade:
        toned_block = tiler.apply_preview_correction_numpy(
            toned_block,
            saturation=args.sat,
            darken_break=args.db,
            low_slope=args.ls,
            gamma=args.gamma,
            shoulder=getattr(args, "shoulder", tiler.DEFAULT_SHOULDER),
            highlight_break=args.ghb,
            mid_slope=args.gms,
            high_slope=args.ghs,
        )

    return np.nan_to_num(toned_block * 255.0, nan=0.0).astype(np.uint8)


def warp_to_web_mercator(
    source_path: str | gdal.Dataset,
    destination_path: str,
    resample_alg: str,
    max_zoom: int,
    blocksize: int,
) -> None:
    """Warp the temporary UTM GeoTIFF into the final EPSG:3857 intermediate."""
    remove_if_exists(destination_path)
    pixel_size = tiler.web_mercator_pixel_size_for_tile_size(max_zoom, blocksize)
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        xRes=pixel_size,
        yRes=pixel_size,
        resampleAlg=resample_alg,
        targetAlignedPixels=True,
        multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        creationOptions=["COMPRESS=ZSTD", "ZSTD_LEVEL=5", "TILED=YES", "BIGTIFF=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
    )
    gdal.Warp(destination_path, source_path, options=warp_options)


def recover_processed_tile_output(
    mgrs_subtile: str,
    args: argparse.Namespace,
) -> Optional[str]:
    """Reuse a finished temp output or recover one from a surviving UTM intermediate."""
    temp_utm_path, temp_3857_path = build_processed_tile_paths(mgrs_subtile, args.output)
    staged_3857_path = build_staged_path(temp_3857_path)

    if os.path.exists(temp_3857_path) and not file_has_content(temp_3857_path):
        remove_if_exists(temp_3857_path)
    if file_has_content(temp_3857_path):
        return temp_3857_path

    if os.path.exists(temp_utm_path) and not file_has_content(temp_utm_path):
        remove_if_exists(temp_utm_path)
    if not file_has_content(temp_utm_path):
        return None

    remove_if_exists(staged_3857_path)
    warp_to_web_mercator(
        temp_utm_path,
        staged_3857_path,
        args.resample_alg,
        args.max_zoom,
        args.blocksize,
    )
    publish_staged_path(staged_3857_path, temp_3857_path)
    remove_if_exists(temp_utm_path)
    return temp_3857_path


def recover_processed_work_outputs(
    work_units: tuple[LandWorkUnit, ...],
    args: argparse.Namespace,
) -> Tuple[Set[str], List[str]]:
    """Collect reusable temp outputs for the requested work units."""
    recovered_units: Set[str] = set()
    recovered_paths: List[str] = []
    for work_unit in work_units:
        recovered_path = recover_processed_tile_output(work_unit.unit_id, args)
        if recovered_path is None:
            continue
        recovered_units.add(work_unit.unit_id)
        recovered_paths.append(recovered_path)
    return recovered_units, recovered_paths


def build_ordered_webp_composite_rasters(
    plan: LandProcessingPlan,
    args: argparse.Namespace,
    prepared_ocean_background: Optional[str],
) -> List[str]:
    """Return the deterministic ocean-first raster list for final WebP compositing."""
    ordered_rasters: List[str] = []
    if prepared_ocean_background and file_has_content(prepared_ocean_background):
        ordered_rasters.append(prepared_ocean_background)

    for work_unit in sorted(plan.work_units, key=lambda current: current.unit_id):
        raster_path = build_work_unit_output_path(work_unit, args.output)
        if file_has_content(raster_path):
            ordered_rasters.append(raster_path)
    return ordered_rasters


def compose_final_webp_tile_tree(
    ordered_rasters: Sequence[str],
    output_dir: str,
    args: argparse.Namespace,
) -> int:
    """Compose the final max-zoom WebP tree one tile at a time in memory."""
    tile_inputs: Dict[str, List[str]] = {}
    for raster_path in ordered_rasters:
        for relative_path in tiler.iter_raster_tile_relpaths(raster_path, args.max_zoom):
            tile_inputs.setdefault(relative_path, []).append(raster_path)

    if not tile_inputs:
        return 0

    pending_tiles = [
        relative_path
        for relative_path in sorted(tile_inputs)
        if not file_has_content(os.path.join(output_dir, relative_path))
    ]
    if not pending_tiles:
        print("All final WebP tiles already composed.")
        return len(tile_inputs)

    print(
        f"Compositing {len(pending_tiles)} final tile(s) from {len(ordered_rasters)} raster(s)..."
    )
    progress_line = LiveProgressLine()
    started_at = time.perf_counter()
    completed_before_start = len(tile_inputs) - len(pending_tiles)

    def compose_one(relative_path: str) -> bool:
        zoom, tx, ty = parse_tile_tree_relpath(relative_path)
        return tiler.compose_webp_tile_from_rasters(
            tile_inputs[relative_path],
            os.path.join(output_dir, relative_path),
            zoom,
            tx,
            ty,
            args.blocksize,
            args.quality,
            args.resample_alg,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_path = {
                executor.submit(compose_one, relative_path): relative_path
                for relative_path in pending_tiles
            }
            processed_tiles = completed_before_start
            for future in as_completed(future_to_path):
                future.result()
                processed_tiles += 1
                update_count_progress(
                    progress_line,
                    "Final tile compositing progress:",
                    processed_tiles,
                    len(tile_inputs),
                    started_at,
                    f"{processed_tiles} tile(s) processed.",
                    completed_before_start=0,
                )
    finally:
        progress_line.finish()

    return len(tile_inputs)


def process_land_work_unit(
    work_unit: LandWorkUnit,
    date_paths: List[str],
    args: argparse.Namespace,
    gebco_src: Optional[str] = None,
    unique_id: Optional[str] = None,
) -> Optional[str]:
    """Dispatch one land-processing work unit to the appropriate strategy."""
    if should_use_webp_cache_pipeline(args):
        return process_single_tile(
            work_unit.unit_id,
            date_paths,
            args,
            gebco_src,
            use_in_memory_utm=True,
        )
    return process_single_tile(work_unit.unit_id, date_paths, args, gebco_src)


def process_single_tile(
    mgrs_subtile: str,
    date_paths: List[str],
    args: argparse.Namespace,
    gebco_src: Optional[str] = None,
    *,
    use_in_memory_utm: bool = False,
    completion_marker_path: Optional[str] = None,
) -> Optional[str]:
    """Process a single MGRS sub-tile: fetch dates, average, tone-map, and warp to Web Mercator."""
    cleanup_returned_raster = True
    folders = list_mosaic_folders_for_tile(mgrs_subtile, date_paths, args.cache)
    if not folders:
        if completion_marker_path is not None:
            write_tile_cache_marker(completion_marker_path, mgrs_subtile, [])
        return None

    if args.download:
        print(f"Downloading tile {mgrs_subtile} across {len(folders)} date(s)...")
        for folder_name, date_path in folders:
            get_tile_paths(folder_name, date_path, args.cache, download=True)
        return None

    prefetch_cache_dir = build_tile_prefetch_cache_dir(mgrs_subtile, args.output)
    first_folder, first_date = folders[0]
    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]] = []
    ocean_mask: Optional[OceanMaskWarp] = None
    fill_allowed_mask: Optional[np.ndarray] = None
    try:
        paths = get_tile_paths(first_folder, first_date, args.cache, download=False)
        tile_grid = load_tile_grid(paths["red"])

        ocean_mask = open_gebco_mask(gebco_src, tile_grid, mgrs_subtile)
        processing_window = ProcessingWindow(0, 0, tile_grid.width, tile_grid.height)
        mask_slabs: Optional[dict[int, OceanMaskSlab]] = None
        if ocean_mask is not None:
            collected = collect_ocean_mask_slabs(ocean_mask, tile_grid)
            if collected is None:
                if completion_marker_path is not None:
                    write_tile_cache_marker(completion_marker_path, mgrs_subtile, [])
                return None
            processing_window, mask_slabs = collected

        prefetch_if_land = float(
            getattr(args, "prefetch_if_land", DEFAULT_PREFETCH_IF_LAND)
        )
        band_cache_dir = args.cache
        if should_prefetch_tile_bands(prefetch_if_land, mask_slabs, tile_grid):
            prefetch_tile_bands_locally(folders, prefetch_cache_dir)
            band_cache_dir = prefetch_cache_dir

        date_band_sets = open_date_band_sets(folders, band_cache_dir)
        averaged, source_valid_mask, alpha_mask, fill_allowed_mask = average_tile_blocks(
            date_band_sets,
            processing_window,
            mask_slabs,
            tile_grid=tile_grid,
            winter=getattr(args, "winter", False),
        )

        averaged = fill_missing_pixels(averaged, source_valid_mask, fill_allowed_mask)
        temp_utm_path, temp_3857_path = build_processed_tile_paths(mgrs_subtile, args.output)
        staged_3857_path = build_staged_path(temp_3857_path)
        remove_if_exists(staged_3857_path)
        staged_utm_path = build_staged_path(temp_utm_path)
        in_memory_output = use_in_memory_utm or completion_marker_path is not None
        if in_memory_output:
            ds_out, color_bands, alpha_band = create_output_dataset(
                None,
                tile_grid,
                driver_name="MEM",
            )
        else:
            remove_if_exists(staged_utm_path)
            ds_out, color_bands, alpha_band = create_output_dataset(staged_utm_path, tile_grid)
        write_processed_blocks(
            averaged,
            alpha_mask,
            processing_window,
            args,
            color_bands,
            alpha_band,
        )
        ds_out.FlushCache()
        warp_source: str | gdal.Dataset
        if in_memory_output:
            warp_source = ds_out
        else:
            ds_out = None
            publish_staged_path(staged_utm_path, temp_utm_path)
            warp_source = temp_utm_path
        warp_to_web_mercator(
            warp_source,
            staged_3857_path,
            args.resample_alg,
            args.max_zoom,
            args.blocksize,
        )
        ds_out = None
        publish_staged_path(staged_3857_path, temp_3857_path)
        if not in_memory_output:
            os.remove(temp_utm_path)
        cleanup_returned_raster = False
        return temp_3857_path
    finally:
        for bands, datasets in date_band_sets:
            bands.clear()
            datasets.clear()
        date_band_sets.clear()
        ocean_mask = None
        fill_allowed_mask = None
        cleanup_prefetched_tile_bands(prefetch_cache_dir)
        remove_if_exists(build_staged_path(build_processed_tile_paths(mgrs_subtile, args.output)[0]))
        remove_if_exists(build_staged_path(build_processed_tile_paths(mgrs_subtile, args.output)[1]))
        if completion_marker_path is not None and cleanup_returned_raster:
            temp_utm_path, temp_3857_path = build_processed_tile_paths(mgrs_subtile, args.output)
            remove_if_exists(temp_utm_path)
            remove_if_exists(temp_3857_path)


def calculate_estimates(args: argparse.Namespace) -> None:
    """Calculate and print estimations for the given command."""
    requested_bbox = parse_bbox(args.bbox) if args.bbox else None
    date_paths = [date_path.strip() for date_path in args.date.split(",")]
    num_dates = len(date_paths)

    gebco_vrt_source = resolve_land_mgrs_source()
    land_mgrs_list_path = build_land_mgrs_list_path()

    if requested_bbox is None:
        populate_s3_cache(date_paths)

    mgrs_bases = discover_mgrs_bases(
        requested_bbox,
        gebco_vrt_source,
        land_mgrs_list_path,
    )
    plan = LandProcessingPlan(
        mgrs_bases=tuple(mgrs_bases),
        work_units=plan_subtile_work_units(
            mgrs_bases,
            discover_available_subtiles_from_s3_cache(mgrs_bases),
        ),
    )

    num_mgrs = len(plan.mgrs_bases)
    num_work_units = len(plan.work_units)
    unique_source_subtiles = {subtile for unit in plan.work_units for subtile in unit.source_subtiles}
    total_tile_dates = len(unique_source_subtiles) * num_dates

    # Rough network estimate based on 600MB per subtile-date (3 source bands).
    network_gb = (total_tile_dates * 600) / 1024
    ram_gb = args.parallel * 8

    total_seconds = (total_tile_dates * 15 / max(args.parallel, 1)) + (
        num_work_units * 30 / max(args.parallel, 1)
    )
    total_seconds += num_work_units * 2
    if args.format == "webp" and not args.vrt:
        # The WebP path keeps the processed 3857 rasters until the final per-tile
        # composite pass writes the max-zoom tree once.
        disk_peak_gb = (
            total_tile_dates * 60
            + num_work_units * 50
            + num_work_units * 5
            + 100
        ) / 1024
        disk_end_gb = (num_work_units * 4) / 1024
    else:
        disk_peak_gb = (total_tile_dates * 60 + num_work_units * 50 + 100) / 1024
        disk_end_gb = (num_work_units * 40) / 1024

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    print("--- Processing Estimates ---")
    print(f"MGRS Tiles (100km): {num_mgrs}")
    print(f"MGRS Sub-tiles (4x): {num_work_units}")
    print(f"Date(s):            {num_dates} ({', '.join(date_paths)})")
    print(f"Total tile-dates:   {total_tile_dates} (3 bands each)")
    print("---------------------------")
    print(f"Estimated Time:       {hours}h {minutes}m")
    print(f"Estimated RAM Usage:  {ram_gb:.1f} GB (peak)")
    print(f"Estimated Disk Peak:  {disk_peak_gb:.2f} GB")
    print(f"Estimated Disk End:   {disk_end_gb:.2f} GB")
    print(f"Estimated Network:    {network_gb:.2f} GB")
    print("---------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PMTiles from Sentinel-2 Global Mosaics on CDSE (NumPy Refactor)."
    )
    parser.add_argument(
        "--date",
        default="2025/07/01,2025/01/01",
        help=(
            "Mosaic date(s), comma-separated. With two dates, the first is blended north "
            "of the equator and the second south of it; --winter swaps the hemispheres."
        ),
    )
    parser.add_argument(
        "--output", "-o", default="output.pmtiles", help="Output PMTiles filename"
    )
    parser.add_argument(
        "--format",
        choices=["webp", "jpg", "png", "png8"],
        default="webp",
        help="Tile format",
    )
    parser.add_argument("--quality", type=int, default=74, help="Quality (0-100)")
    parser.add_argument(
        "--resample-alg",
        default="lanczos",
        choices=["bilinear", "average", "gauss", "lanczos"],
    )
    parser.add_argument(
        "--chunk-zoom",
        type=int,
        default=6,
        help="Zoom level to chunk the processing at",
    )
    parser.add_argument(
        "--parallel", type=int, default=12, help="Number of parallel processes"
    )
    parser.add_argument("--stats-min", type=float, help="Hardcoded source min")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source max")

    # Grading
    parser.add_argument(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land final grading",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="Global brightness multiplier",
    )
    parser.add_argument("--gamma", type=float, default=2.8)
    parser.add_argument(
        "--shoulder",
        type=float,
        default=0.6,
        help="Highlight shaping curve; values above 1 lift the top end",
    )
    parser.add_argument(
        "--sat", "--saturation", type=float, default=0.9
    )
    parser.add_argument(
        "--db", "--black-break", "--grade-low-break", type=float, default=0.08
    )
    parser.add_argument(
        "--ls", "--black-slope", "--grade-low-slope", type=float, default=0.7
    )
    parser.add_argument(
        "--ghb",
        "--grade-highlight-break",
        type=float,
        default=0.82,
        help="Upper breakpoint for the final grading curve",
    )
    parser.add_argument(
        "--gms",
        "--grade-mid-slope",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ghs",
        "--grade-highlight-slope",
        type=float,
        default=1.05,
        help="Highlight slope for the final grading curve",
    )
    parser.add_argument(
        "--tonemap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--land",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land tile processing",
    )

    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--cache", default=".cache", help="Cache directory")
    parser.add_argument(
        "--prefetch-if-land",
        type=parse_prefetch_if_land,
        default=DEFAULT_PREFETCH_IF_LAND,
        help=(
            "Prefetch RGB bands when land coverage is at least this percent. "
            "100 always prefetches without computing land percentage; "
            "0 never prefetches and also skips land-percentage calculation."
        ),
    )
    parser.add_argument(
        "--ocean-background",
        default="ocean.tif",
        help="Standalone ocean background GeoTIFF to use under bbox renders",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download S3 tiles to local cache and exit",
    )
    parser.add_argument(
        "--bbox", help="WGS84 bounding box as min_lon,min_lat,max_lon,max_lat"
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        choices=list(ocean.SUPPORTED_MAX_ZOOMS),
        default=ocean.DEFAULT_MAX_ZOOM,
        help="Target Web Mercator zoom used for output resolution",
    )
    parser.add_argument(
        "--vrt", action="store_true", help="Write final VRT and skip MBTiles"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print estimated time, RAM, disk space, and network size then exit",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="Resume from a previous run if a state file exists",
    )
    parser.add_argument(
        "--winter",
        action="store_true",
        help="Swap the two-date equator blend so the first date favors the south and the second the north",
    )
    parser.add_argument(
        "--delete-tifs",
        action="store_true",
        help="Delete durable land GeoTIFF intermediates after packaging when using the legacy raster pipeline",
    )
    land_mgrs_module.add_land_mgrs_cli_args(parser)
    args = parser.parse_args()

    setup_gdal_cdse()

    if args.estimate:
        calculate_estimates(args)
        return

    os.makedirs(".temp", exist_ok=True)

    requested_bbox = parse_bbox(args.bbox) if args.bbox else None
    date_paths = [date_path.strip() for date_path in args.date.split(",")]
    gebco_vrt_source = resolve_ocean_mask_source(args.ocean_background)
    land_mgrs_source = resolve_land_mgrs_source()
    land_mgrs_list_path = build_land_mgrs_list_path()
    if land_mgrs_module.handle_land_mgrs_refresh(
        args,
        requested_bbox,
        land_mgrs_source=land_mgrs_source,
        land_mgrs_list_path=land_mgrs_list_path,
    ):
        return

    if requested_bbox is None:
        populate_s3_cache(date_paths)

    unique_id = build_land_run_token(args, date_paths, requested_bbox, gebco_vrt_source)
    state_file = build_state_file_path(unique_id)
    use_webp_cache = should_use_webp_cache_pipeline(args)
    completed_units: Set[str] = set()
    processed_tifs: List[str] = []

    if args.resume:
        resume_path = find_resume_path(
            args.resume,
            preferred_path=state_file,
            allow_latest=not use_webp_cache,
        )
        if resume_path:
            resume_state = restore_resume_state(resume_path)
            if resume_state:
                state_file = cast(str, resume_state["state_file"])
                unique_id = cast(str, resume_state["unique_id"])
                completed_units = cast(Set[str], resume_state["completed_units"])
                processed_tifs = cast(List[str], resume_state["processed_tifs"])

    mgrs_bases = discover_mgrs_bases(
        requested_bbox,
        land_mgrs_source,
        land_mgrs_list_path,
    )
    plan = LandProcessingPlan(
        mgrs_bases=tuple(mgrs_bases),
        work_units=plan_subtile_work_units(
            mgrs_bases,
            discover_available_subtiles_from_s3_cache(mgrs_bases),
        ),
    )
    if use_webp_cache:
        recovered_units, recovered_tile_paths = recover_processed_work_outputs(plan.work_units, args)
        if recovered_units:
            print(f"Reusing {len(recovered_units)} existing temp raster(s).")
        completed_units &= recovered_units
        completed_units.update(recovered_units)
        land_output_paths = {build_work_unit_output_path(unit, args.output) for unit in plan.work_units}
        processed_tifs = recovered_tile_paths + [
            path
            for path in processed_tifs
            if path not in land_output_paths and file_has_content(path)
        ]
    else:
        recovered_units, recovered_tile_paths = recover_processed_work_outputs(plan.work_units, args)
        if recovered_units:
            print(f"Reusing {len(recovered_units)} existing temp raster(s).")
        completed_units &= recovered_units
        completed_units.update(recovered_units)
        land_output_paths = {build_work_unit_output_path(unit, args.output) for unit in plan.work_units}
        processed_tifs = recovered_tile_paths + [
            path
            for path in processed_tifs
            if path not in land_output_paths and file_has_content(path)
        ]
    print(describe_land_processing_plan(plan, len(date_paths)))

    prepared_ocean_background: Optional[str] = None
    ocean_cleanup_paths: List[str] = []
    if use_webp_cache and not args.download:
        prepared_ocean_background = prepare_ocean_background_for_output(
            args.ocean_background,
            requested_bbox,
            args.output,
            args.resample_alg,
            args.max_zoom,
            args.blocksize,
        )
        if prepared_ocean_background:
            print(f"Using ocean background: {prepared_ocean_background}")
            if os.path.abspath(prepared_ocean_background) != os.path.abspath(args.ocean_background):
                ocean_cleanup_paths.append(prepared_ocean_background)
        elif args.bbox:
            print(f"Warning: Ocean background not found, skipping: {args.ocean_background}")

    if args.land:
        work_units_to_process = [
            work_unit
            for work_unit in plan.work_units
            if work_unit.unit_id not in completed_units
        ]
        if work_units_to_process:
            completed_before_start = len(completed_units)
            print(
                f"Starting sub-tile processing for "
                f"{len(work_units_to_process)} sub-tile(s) "
                f"with {args.parallel} worker(s); {len(completed_units)} already complete."
            )
            progress_line = LiveProgressLine()
            started_at = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_work_unit = {
                    executor.submit(
                        process_land_work_unit,
                        work_unit,
                        date_paths,
                        args,
                        gebco_vrt_source,
                        unique_id,
                    ): work_unit
                    for work_unit in work_units_to_process
                }
                for future in as_completed(future_to_work_unit):
                    work_unit = future_to_work_unit[future]
                    res = future.result()
                    completed_units.add(work_unit.unit_id)
                    if res:
                        processed_tifs.append(res)
                    status_message = f"{len(processed_tifs)} raster(s) ready."
                    update_count_progress(
                        progress_line,
                        "Land processing progress:",
                        len(completed_units),
                        len(plan.work_units),
                        started_at,
                        status_message,
                        completed_before_start=completed_before_start,
                    )
                    write_resume_state(
                        state_file,
                        unique_id,
                        completed_units,
                        processed_tifs,
                        args,
                    )
            progress_line.finish()
        else:
            print("All sub-tiles already processed.")
    else:
        print("Skipping land tile processing (--no-land).")

    if args.download:
        print("Download complete.")
        return

    if use_webp_cache:
        ordered_rasters = build_ordered_webp_composite_rasters(
            plan,
            args,
            prepared_ocean_background,
        )
        final_tile_tree = build_final_tile_cache_dir(args.output, unique_id)
        compose_final_webp_tile_tree(ordered_rasters, final_tile_tree, args)
        final_tile_count = len(tiler.iter_tile_tree_paths(final_tile_tree))
        if final_tile_count <= 0:
            print("Error: No max-zoom tiles were generated.")
            sys.exit(1)

        temp_mbtiles = convert_tile_tree_to_pmtiles(
            final_tile_tree,
            args.output,
            resample_alg=args.resample_alg,
            max_zoom=args.max_zoom,
            name="Sentinel-2 Mosaic",
            description="Copernicus Sentinel data",
            requested_bbox=requested_bbox,
        )
        cleanup_temporary_files([temp_mbtiles] + processed_tifs + ocean_cleanup_paths)
    else:
        prepared_ocean_background = prepare_ocean_background_for_output(
            args.ocean_background,
            requested_bbox,
            args.output,
            args.resample_alg,
            args.max_zoom,
            args.blocksize,
        )
        if prepared_ocean_background:
            print(f"Using ocean background: {prepared_ocean_background}")
            if prepared_ocean_background not in processed_tifs:
                processed_tifs.insert(0, prepared_ocean_background)
        elif args.bbox:
            print(f"Warning: Ocean background not found, skipping: {args.ocean_background}")

        if not processed_tifs:
            print("Error: No data processed.")
            sys.exit(1)

        master_vrt = build_master_vrt_path(args.output, unique_id)
        staged_master_vrt = build_staged_path(master_vrt)
        pixel_size = tiler.web_mercator_pixel_size_for_tile_size(
            args.max_zoom,
            args.blocksize,
        )
        remove_if_exists(staged_master_vrt)
        progress_line = LiveProgressLine()
        master_vrt_callback = build_gdal_progress_callback(
            progress_line,
            f"Building master VRT from {len(processed_tifs)} raster(s)...",
            time.perf_counter(),
        )
        master_vrt_callback(0.0, "", None)
        try:
            gdal.BuildVRT(
                staged_master_vrt,
                processed_tifs,
                resolution="user",
                xRes=pixel_size,
                yRes=pixel_size,
                callback=master_vrt_callback,
            )
            master_vrt_callback(1.0, "", None)
        finally:
            progress_line.finish()
        publish_staged_path(staged_master_vrt, master_vrt)

        if args.vrt:
            print(f"Success! Master VRT: {master_vrt}")
            return

        land_output_abspaths = {
            os.path.abspath(build_work_unit_output_path(unit, args.output))
            for unit in plan.work_units
        }
        persistent_paths = {os.path.abspath(args.ocean_background)}
        cleanup_paths = [
            path
            for path in processed_tifs
            if os.path.abspath(path) not in persistent_paths
            and (
                args.delete_tifs
                or os.path.abspath(path) not in land_output_abspaths
            )
        ]
        packaged_tiles = convert_raster_to_pmtiles(
            master_vrt,
            args.output,
            tile_format=args.format,
            quality=args.quality,
            resample_alg=args.resample_alg,
            chunk_zoom=args.chunk_zoom,
            parallel=args.parallel,
            blocksize=args.blocksize,
            name="Sentinel-2 Mosaic",
            description="Copernicus Sentinel data",
            requested_bbox=requested_bbox,
            cleanup_input_paths=cleanup_paths,
        )
        cleanup_temporary_files(
            [packaged_tiles.temp_mbtiles] + packaged_tiles.tiling_artifacts.cleanup_paths
        )

    if os.path.exists(state_file):
        os.remove(state_file)


if __name__ == "__main__":
    main()
