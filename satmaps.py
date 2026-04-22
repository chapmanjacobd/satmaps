#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, cast

import mgrs
from mgrs import core as mgrs_core
import numpy as np
import ocean
from scipy.ndimage import distance_transform_edt
from osgeo import gdal

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
LAND_STAY_DEPTH = -42.0
OCEAN_FADE_DEPTH = -50.0


@dataclass(frozen=True)
class TileGrid:
    projection: str
    geotransform: Tuple[float, float, float, float, float, float]
    width: int
    height: int


@dataclass
class OceanMaskWarp:
    band: gdal.Band
    dataset: gdal.Dataset
    uses_alpha: bool
    coverage_band: gdal.Band
    coverage_dataset: gdal.Dataset


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
    gebco_block: np.ndarray
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
            # Fallback for non-global runs: check S3 directly for this specific folder
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
            print(f"Downloading {s3_path} to {local_path}...")
            src_ds = gdal.Open(s3_path)
            if src_ds is None:
                raise RuntimeError(f"Could not open {s3_path}")
            gdal.GetDriverByName("GTiff").CreateCopy(
                local_path, src_ds, callback=gdal.TermProgress_nocb
            )
            paths[color_name] = local_path
            continue

        # Fallback to streaming
        paths[color_name] = f"{base_s3}/{band_id}.tif"

    return paths


def load_land_tiles(land_only_file: str) -> Optional[Set[str]]:
    """Load the precomputed land tile list if present."""
    if not os.path.exists(land_only_file):
        return None

    with open(land_only_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


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


def filter_mgrs_tiles(
    discovered_mgrs: List[str],
    land_set: Optional[Set[str]],
    gebco_vrt_source: Optional[str],
) -> List[str]:
    """Keep only tiles that should participate in land rendering."""
    mgrs_bases = []
    for mgrs_tile in discovered_mgrs:
        is_land = False
        if land_set and mgrs_tile in land_set:
            is_land = True
        elif gebco_vrt_source and check_land_gebco(mgrs_tile, gebco_vrt_source):
            is_land = True
        elif land_set is None:
            is_land = True

        if is_land:
            mgrs_bases.append(mgrs_tile)

    return mgrs_bases


def discover_mgrs_bases(
    args: argparse.Namespace,
    land_set: Optional[Set[str]],
    gebco_vrt_source: Optional[str],
) -> List[str]:
    """Resolve the requested MGRS tile list from bbox, global, or explicit inputs."""
    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(args.bbox)
        discovered_mgrs = discover_mgrs_tiles_in_bbox(min_lon, min_lat, max_lon, max_lat)
        mgrs_bases = filter_mgrs_tiles(discovered_mgrs, land_set, gebco_vrt_source)
        print(
            f"Discovered {len(discovered_mgrs)} MGRS tiles in bbox, {len(mgrs_bases)} kept after ocean-mask/land filtering."
        )
        return mgrs_bases

    if args.all_tiles:
        if land_set is not None:
            return list(land_set)
        print("Error: --global requires HLS.land.tiles.txt to be present.")
        sys.exit(1)

    return [mgrs_tile.strip() for mgrs_tile in args.mgrs.split(",") if mgrs_tile.strip()]


def expand_subtiles(mgrs_bases: List[str]) -> List[str]:
    """Expand each 100 km MGRS tile into its four processing subtiles."""
    return [
        f"{mgrs_tile}_{x}_{y}"
        for mgrs_tile in mgrs_bases
        for x, y in SUBTILE_OFFSETS
    ]


def find_resume_path(resume_arg: object) -> Optional[str]:
    """Resolve the explicit or latest state file path for --resume."""
    if isinstance(resume_arg, str) and os.path.exists(resume_arg):
        return resume_arg

    states = glob.glob(".temp/state_*.json")
    if not states:
        return None
    return max(states, key=os.path.getmtime)


def restore_resume_state(resume_path: str) -> Optional[Dict[str, Any]]:
    """Load a previous run state and keep only surviving intermediate files."""
    try:
        with open(resume_path, "r") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state file must contain a JSON object")
        unique_id = state["unique_id"]
        completed_subtiles_raw = state["completed_subtiles"]
        processed_tifs_raw = state["processed_tifs"]
        if not isinstance(unique_id, str):
            raise ValueError("unique_id must be a string")
        if not isinstance(completed_subtiles_raw, list):
            raise ValueError("completed_subtiles must be a list")
        if not isinstance(processed_tifs_raw, list):
            raise ValueError("processed_tifs must be a list")
        if not all(isinstance(subtile, str) for subtile in completed_subtiles_raw):
            raise ValueError("completed_subtiles entries must be strings")
        if not all(isinstance(path, str) for path in processed_tifs_raw):
            raise ValueError("processed_tifs entries must be strings")
    except (OSError, ValueError, TypeError, KeyError) as e:
        print(f"Warning: Could not load state file: {e}")
        return None

    completed_subtiles = set(completed_subtiles_raw)
    processed_tifs = [path for path in processed_tifs_raw if os.path.exists(path)]
    print(f"Resuming from state file: {resume_path} (unique_id: {unique_id})")
    print(
        f"Already completed {len(completed_subtiles)} sub-tiles, {len(processed_tifs)} TIFs found."
    )
    return {
        "state_file": resume_path,
        "unique_id": unique_id,
        "completed_subtiles": completed_subtiles,
        "processed_tifs": processed_tifs,
    }


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

    if get_ocean_mask_band_details(dataset) is None:
        dataset = None
        return None

    dataset = None
    return ocean_background


def prepare_ocean_background_for_output(
    ocean_background_path: str,
    requested_bbox: Optional[Tuple[float, float, float, float]],
    unique_id: str,
    resample_alg: str,
    max_zoom: int,
) -> Optional[str]:
    """Prepare the ocean background used in the final 3857 composite."""
    if not os.path.exists(ocean_background_path):
        return None
    if requested_bbox is None:
        return ocean_background_path

    snapped_bounds, pixel_size, _zoom = ocean.snapped_tile_grid_for_bbox(requested_bbox, max_zoom)
    prepared_ocean_path = f".temp/ocean_{unique_id}_bbox.tif"
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        outputBounds=snapped_bounds,
        xRes=pixel_size,
        yRes=pixel_size,
        resampleAlg=resample_alg,
        creationOptions=list(ocean.GTIFF_CREATION_OPTIONS),
    )
    prepared_ds = gdal.Warp(prepared_ocean_path, ocean_background_path, options=warp_options)
    if prepared_ds is None:
        raise RuntimeError(
            f"Could not prepare bbox ocean background from {ocean_background_path}"
        )
    prepared_ds = None
    return prepared_ocean_path


def get_ocean_mask_band_details(dataset: gdal.Dataset) -> Optional[Tuple[int, bool]]:
    """Return the mask band index and whether it should be treated as an alpha mask."""
    if dataset.RasterCount == 1:
        return 1, False

    for band_index in range(1, dataset.RasterCount + 1):
        if dataset.GetRasterBand(band_index).GetColorInterpretation() == gdal.GCI_AlphaBand:
            return band_index, True

    return None


def populate_s3_cache(date_paths: List[str]) -> None:
    """Cache remote folder listings for global runs."""
    print("Populating S3 folder cache...")
    for date_path in date_paths:
        s3_base = f"/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/{date_path}"
        dirs = gdal.ReadDir(s3_base)
        if dirs:
            S3_FOLDER_CACHE[date_path] = set(dirs)
        else:
            print(f"Warning: Could not list folders for {date_path}")


# --- Processing Layer (NumPy Manipulation) ---


def check_land_gebco(mgrs_tile: str, gebco_src: str) -> bool:
    """Check if an MGRS tile contains land according to GEBCO (sampling)."""
    m = mgrs.MGRS()
    try:
        clat, clon = m.toLatLon(mgrs_tile)
    except mgrs_core.MGRSError:
        return True

    # 100km is roughly 1 degree. We'll sample a grid around the center.
    ds = gdal.Open(gebco_src)
    if ds is None:
        print(f"Warning: Could not open GEBCO source for land check: {gebco_src}")
        return True
    band_details = get_ocean_mask_band_details(ds)
    try:
        if band_details is None:
            return True
        band_index, uses_alpha = band_details
        band = ds.GetRasterBand(band_index)
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)

        # Sample a 1.2 degree box (13x13 points), but read the smallest
        # covering window once instead of issuing 169 tiny GDAL reads.
        sample_points: list[tuple[int, int]] = []
        for dlat in np.linspace(-0.6, 0.6, 13):
            for dlon in np.linspace(-0.6, 0.6, 13):
                px, py = gdal.ApplyGeoTransform(inv_gt, clon + dlon, clat + dlat)
                px, py = int(px), int(py)
                if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                    sample_points.append((px, py))
        if not sample_points:
            return False

        xoff = min(px for px, _ in sample_points)
        yoff = min(py for _, py in sample_points)
        xend = max(px for px, _ in sample_points) + 1
        yend = max(py for _, py in sample_points) + 1
        sampled = band.ReadAsArray(xoff, yoff, xend - xoff, yend - yoff).astype(np.float32)

        for px, py in sample_points:
            val = sampled[py - yoff, px - xoff]
            if (uses_alpha and val < 254.5) or (not uses_alpha and val > 0.001):
                return True
        return False
    finally:
        ds = None


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
        band_details = get_ocean_mask_band_details(gebco_ds)
        if band_details is None:
            raise RuntimeError(f"Could not find a usable mask band in: {gebco_src}")
        band_index, uses_alpha = band_details

        try:
            mem_driver = gdal.GetDriverByName("MEM")
            if mem_driver is None:
                raise RuntimeError("Could not load GDAL MEM driver")
            mask_data_type = gebco_ds.GetRasterBand(band_index).DataType

            warped_mask_ds = mem_driver.Create(
                "", tile_grid.width, tile_grid.height, 1, mask_data_type
            )
            if warped_mask_ds is None:
                raise RuntimeError("Could not create in-memory GEBCO mask dataset")
            warped_mask_ds.SetProjection(tile_grid.projection)
            warped_mask_ds.SetGeoTransform(tile_grid.geotransform)
            warped = gdal.Warp(
                warped_mask_ds,
                gebco_ds,
                options=gdal.WarpOptions(
                    resampleAlg="bilinear",
                    srcBands=[band_index],
                    dstBands=[1],
                ),
            )
            if warped is None:
                raise RuntimeError(f"Could not warp GEBCO mask for {mgrs_subtile}")

            coverage_source_ds = mem_driver.Create(
                "", gebco_ds.RasterXSize, gebco_ds.RasterYSize, 1, gdal.GDT_Byte
            )
            if coverage_source_ds is None:
                raise RuntimeError("Could not create source coverage dataset")
            coverage_source_ds.SetProjection(gebco_ds.GetProjection())
            coverage_source_ds.SetGeoTransform(gebco_ds.GetGeoTransform())
            coverage_source_ds.GetRasterBand(1).Fill(255)

            coverage_mem_ds = mem_driver.Create(
                "", tile_grid.width, tile_grid.height, 1, gdal.GDT_Byte
            )
            if coverage_mem_ds is None:
                raise RuntimeError("Could not create in-memory GEBCO coverage dataset")
            coverage_mem_ds.SetProjection(tile_grid.projection)
            coverage_mem_ds.SetGeoTransform(tile_grid.geotransform)
            coverage_mem_ds.GetRasterBand(1).Fill(0)
            warped_coverage = gdal.Warp(
                coverage_mem_ds,
                coverage_source_ds,
                options=gdal.WarpOptions(
                    resampleAlg="near",
                    srcBands=[1],
                    dstBands=[1],
                ),
            )
            coverage_source_ds = None
            if warped_coverage is None:
                raise RuntimeError(f"Could not warp GEBCO coverage for {mgrs_subtile}")
        finally:
            gebco_ds = None

        return OceanMaskWarp(
            band=warped_mask_ds.GetRasterBand(1),
            dataset=warped_mask_ds,
            uses_alpha=uses_alpha,
            coverage_band=coverage_mem_ds.GetRasterBand(1),
            coverage_dataset=coverage_mem_ds,
        )
    except RuntimeError as e:
        print(f"Warning: Could not apply GEBCO mask to {mgrs_subtile}: {e}")
        return None


def build_fill_allowed_mask(
    ocean_mask: np.ndarray,
    mask_uses_alpha: bool,
    coverage_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the pixels where Sentinel land rendering is allowed."""
    if mask_uses_alpha:
        fill_allowed_mask = ocean_mask < 254.5
    else:
        fill_allowed_mask = ocean_mask > OCEAN_FADE_DEPTH

    if coverage_mask is not None:
        if coverage_mask.shape != fill_allowed_mask.shape:
            raise ValueError("coverage_mask must match the spatial shape of ocean_mask")
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

    for _xoff, yoff, block_width, block_height in iter_processing_windows(tile_grid):
        gebco_block = ocean_mask.band.ReadAsArray(0, yoff, block_width, block_height).astype(
            np.float32
        )
        coverage_block = (
            ocean_mask.coverage_band.ReadAsArray(0, yoff, block_width, block_height) >= 254.5
        )
        fill_allowed_block = build_fill_allowed_mask(
            gebco_block,
            ocean_mask.uses_alpha,
            coverage_block,
        )
        cols_with_land = np.any(fill_allowed_block, axis=0)
        if not np.any(cols_with_land):
            continue

        x_start = int(np.argmax(cols_with_land))
        x_end = int(len(cols_with_land) - np.argmax(cols_with_land[::-1]))
        cropped = OceanMaskSlab(
            xoff=x_start,
            yoff=yoff,
            width=x_end - x_start,
            height=block_height,
            gebco_block=gebco_block[:, x_start:x_end],
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
    completed_subtiles: Set[str],
    processed_tifs: List[str],
    args: argparse.Namespace,
) -> None:
    """Persist resume state atomically so interrupted writes do not corrupt the JSON file."""
    temp_state_file = f"{state_file}.tmp"
    with open(temp_state_file, "w") as f:
        json.dump(
            {
                "unique_id": unique_id,
                "completed_subtiles": list(completed_subtiles),
                "processed_tifs": processed_tifs,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    os.replace(temp_state_file, state_file)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Average one block across all dates while requiring complete RGB observations."""
    summed = np.zeros((3, height, width), dtype=np.float32)
    valid_counts = np.zeros((height, width), dtype=np.uint16)
    valid_source_mask = np.zeros((height, width), dtype=bool)

    for bands, _ in date_band_sets:
        rgb_block = np.empty((3, height, width), dtype=np.float32)
        for band_index, band in enumerate(bands):
            block = band.ReadAsArray(xoff, yoff, width, height).astype(np.float32)
            block[block == SENTINEL_NODATA] = np.nan
            rgb_block[band_index] = block
        complete_rgb_mask = np.all(np.isfinite(rgb_block), axis=0)
        if not np.any(complete_rgb_mask):
            continue

        valid_source_mask |= complete_rgb_mask
        valid_counts[complete_rgb_mask] += 1
        for band_index in range(rgb_block.shape[0]):
            np.add(
                summed[band_index],
                rgb_block[band_index],
                out=summed[band_index],
                where=complete_rgb_mask,
            )

    averaged_block = np.full((3, height, width), np.nan, dtype=np.float32)
    for band_index in range(averaged_block.shape[0]):
        np.divide(
            summed[band_index],
            valid_counts,
            out=averaged_block[band_index],
            where=valid_source_mask,
        )

    return averaged_block, valid_source_mask


def average_tile_blocks(
    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]],
    processing_window: ProcessingWindow,
    mask_slabs: Optional[dict[int, OceanMaskSlab]],
    mask_uses_alpha: bool,
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

    for relative_yoff in range(0, processing_window.height, PROCESS_SLAB_HEIGHT):
        yoff = processing_window.yoff + relative_yoff
        block_height = min(PROCESS_SLAB_HEIGHT, processing_window.height - relative_yoff)

        if mask_slabs is None:
            actual_xoff = processing_window.xoff
            actual_width = processing_window.width
            gebco_block = None
            coverage_block = None
            allowed_block = None
        else:
            slab = mask_slabs.get(yoff)
            if slab is None:
                continue
            actual_xoff = slab.xoff
            actual_width = slab.width
            gebco_block = slab.gebco_block
            coverage_block = slab.coverage_block
            allowed_block = slab.fill_allowed_block

        averaged_block, block_valid_sources = average_block(
            date_band_sets, actual_xoff, yoff, actual_width, block_height
        )
        if allowed_block is not None:
            averaged_block = np.where(allowed_block[np.newaxis, :, :], averaged_block, np.nan)
            block_valid_sources &= allowed_block

        relative_xoff = actual_xoff - processing_window.xoff
        row_slice = slice(relative_yoff, relative_yoff + block_height)
        col_slice = slice(relative_xoff, relative_xoff + actual_width)
        averaged[:, row_slice, col_slice] = averaged_block
        source_valid_mask[row_slice, col_slice] = block_valid_sources

        alpha_mask[row_slice, col_slice] = build_alpha_block(
            gebco_block,
            block_valid_sources,
            mask_uses_alpha,
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
    output_path: str, tile_grid: TileGrid
) -> Tuple[gdal.Dataset, List[gdal.Band], gdal.Band]:
    """Create the temporary UTM RGBA GeoTIFF used before Web Mercator warping."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path,
        tile_grid.width,
        tile_grid.height,
        4,
        gdal.GDT_Byte,
        options=["COMPRESS=ZSTD", "TILED=YES"],
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
    gebco_block: Optional[np.ndarray],
    source_valid_block: np.ndarray,
    mask_uses_alpha: bool,
    coverage_block: Optional[np.ndarray] = None,
    fill_allowed_block: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the output alpha band for one block."""
    if gebco_block is None:
        alpha_block = source_valid_block.astype(np.uint8) * 255
    elif mask_uses_alpha:
        alpha_block = np.clip(255.0 - gebco_block, 0.0, 255.0).astype(np.uint8)
    else:
        alpha_block = np.clip(
            (gebco_block - OCEAN_FADE_DEPTH) / (LAND_STAY_DEPTH - OCEAN_FADE_DEPTH),
            0.0,
            1.0,
        )
        alpha_block = np.clip(alpha_block * 255.0, 0.0, 255.0).astype(np.uint8)

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

    for relative_yoff in range(0, processing_window.height, PROCESS_SLAB_HEIGHT):
        yoff = processing_window.yoff + relative_yoff
        block_height = min(PROCESS_SLAB_HEIGHT, processing_window.height - relative_yoff)
        averaged_block = averaged[
            :, relative_yoff : relative_yoff + block_height, :
        ]
        normalized = np.clip((averaged_block - source_min) / scale, 0.0, 1.0)
        normalized[np.isnan(normalized)] = 0.0

        if args.tonemap:
            toned_block = tiler.apply_soft_knee_numpy(
                normalized,
                shadow_break=args.sb,
                highlight_break=args.hb,
                shadow_slope=args.ss,
                mid_slope=args.ms,
                highlight_slope=args.hs,
                exposure=args.exposure,
            )
        else:
            toned_block = np.clip(normalized * args.exposure, 0.0, 1.0)

        if args.grade:
            toned_block = tiler.apply_preview_correction_numpy(
                toned_block,
                saturation=args.sat,
                darken_break=args.db,
                low_slope=args.ls,
                gamma=args.gamma,
            )

        byte_block = np.nan_to_num(toned_block * 255.0, nan=0.0).astype(np.uint8)
        for band_index, out_band in enumerate(color_bands):
            out_band.WriteArray(byte_block[band_index], xoff=processing_window.xoff, yoff=yoff)
        alpha_band.WriteArray(
            alpha_mask[relative_yoff : relative_yoff + block_height, :],
            xoff=processing_window.xoff,
            yoff=yoff,
        )


def warp_to_web_mercator(
    source_path: str, destination_path: str, resample_alg: str, max_zoom: int
) -> None:
    """Warp the temporary UTM GeoTIFF into the final EPSG:3857 intermediate."""
    pixel_size = tiler.web_mercator_pixel_size(max_zoom)
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        xRes=pixel_size,
        yRes=pixel_size,
        resampleAlg=resample_alg,
        creationOptions=["COMPRESS=ZSTD", "ZSTD_LEVEL=5", "TILED=YES", "BIGTIFF=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
    )
    gdal.Warp(destination_path, source_path, options=warp_options)


def process_single_tile(
    mgrs_subtile: str,
    date_paths: List[str],
    args: argparse.Namespace,
    unique_id: str,
    gebco_src: Optional[str] = None,
) -> Optional[str]:
    """Process a single MGRS sub-tile: fetch dates, average, tone-map, and warp to Web Mercator."""
    folders = list_mosaic_folders_for_tile(mgrs_subtile, date_paths, args.cache)
    if not folders:
        return None

    if args.download:
        for folder_name, date_path in folders:
            get_tile_paths(folder_name, date_path, args.cache, download=True)
        return None

    first_folder, first_date = folders[0]
    paths = get_tile_paths(first_folder, first_date, args.cache, download=False)
    tile_grid = load_tile_grid(paths["red"])

    ocean_mask = open_gebco_mask(gebco_src, tile_grid, mgrs_subtile)
    processing_window = ProcessingWindow(0, 0, tile_grid.width, tile_grid.height)
    mask_slabs: Optional[dict[int, OceanMaskSlab]] = None
    if ocean_mask is not None:
        collected = collect_ocean_mask_slabs(ocean_mask, tile_grid)
        if collected is None:
            return None
        processing_window, mask_slabs = collected

    date_band_sets: List[Tuple[List[gdal.Band], List[gdal.Dataset]]] = []
    try:
        print(f"Processing tile {mgrs_subtile} across {len(folders)} date(s)...")
        date_band_sets = open_date_band_sets(folders, args.cache)
        averaged, source_valid_mask, alpha_mask, fill_allowed_mask = average_tile_blocks(
            date_band_sets,
            processing_window,
            mask_slabs,
            ocean_mask.uses_alpha if ocean_mask is not None else False,
        )

        averaged = fill_missing_pixels(averaged, source_valid_mask, fill_allowed_mask)
        temp_utm_path = f".temp/processed_{mgrs_subtile}_{unique_id}_utm.tif"
        temp_3857_path = f".temp/processed_{mgrs_subtile}_{unique_id}_3857.tif"
        ds_out, color_bands, alpha_band = create_output_dataset(temp_utm_path, tile_grid)
        write_processed_blocks(
            averaged,
            alpha_mask,
            processing_window,
            args,
            color_bands,
            alpha_band,
        )
        ds_out.FlushCache()
        ds_out = None

        warp_to_web_mercator(temp_utm_path, temp_3857_path, args.resample_alg, args.max_zoom)
        os.remove(temp_utm_path)
        return temp_3857_path
    finally:
        for bands, datasets in date_band_sets:
            bands.clear()
            datasets.clear()
        date_band_sets.clear()
        ocean_mask = None
        fill_allowed_mask = None


def calculate_estimates(args: argparse.Namespace) -> None:
    """Calculate and print estimations for the given command."""
    date_paths = [date_path.strip() for date_path in args.date.split(",")]
    num_dates = len(date_paths)

    land_set = load_land_tiles("HLS.land.tiles.txt")

    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(args.bbox)
        discovered_mgrs = discover_mgrs_tiles_in_bbox(min_lon, min_lat, max_lon, max_lat)

        if land_set is not None:
            mgrs_bases = [m for m in discovered_mgrs if m in land_set]
        else:
            mgrs_bases = list(discovered_mgrs)
    elif args.all_tiles:
        if land_set is not None:
            mgrs_bases = list(land_set)
        else:
            print("Error: --global requires HLS.land.tiles.txt to be present.")
            sys.exit(1)
    else:
        mgrs_bases = [m.strip() for m in args.mgrs.split(",") if m.strip()]

    num_mgrs = len(mgrs_bases)
    num_subtiles = num_mgrs * 4
    total_tile_dates = num_subtiles * num_dates

    # Rough estimations based on 5000x5000 float32 subtiles
    # Network: ~200MB per band * 3 bands = 600MB per tile-date
    network_gb = (total_tile_dates * 600) / 1024

    # RAM: ~8GB per process for NumPy stack/mean operations
    ram_gb = args.parallel * 8

    # Time: ~15s per tile-date processing + ~30s per subtile for warping/tiling
    total_seconds = (total_tile_dates * 15 / args.parallel) + (
        num_subtiles * 30 / args.parallel
    )
    total_seconds += num_subtiles * 2

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    # Disk Space Peak (Optimized):
    # 1. Cached raw bands: 60MB * total_tile_dates (if using --cache)
    # 2. Processed intermediate TIFs: ~50MB * num_subtiles (ZSTD Predictor=2)
    # 3. MBTiles chunks: Now deleted during merge, peak is small (~100MB)
    disk_peak_gb = (total_tile_dates * 60 + num_subtiles * 50 + 100) / 1024

    # Disk Space End:
    # Final PMTiles: ~40MB * num_subtiles (very rough)
    disk_end_gb = (num_subtiles * 40) / 1024

    print("--- Processing Estimates ---")
    print(f"MGRS Tiles (100km): {num_mgrs}")
    print(f"MGRS Sub-tiles (4x): {num_subtiles}")
    print(f"Date(s):            {num_dates} ({', '.join(date_paths)})")
    print(f"Total tile-dates:    {total_tile_dates} (3 bands each)")
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
    parser.add_argument("mgrs", nargs="?", default="31TDF", help="MGRS tile ID")
    parser.add_argument(
        "--global",
        dest="all_tiles",
        action="store_true",
        help="Process all available tiles",
    )
    parser.add_argument(
        "--date",
        default="2025/07/01,2025/01/01",
        help="Mosaic date(s), comma-separated",
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
        default=4,
        help="Zoom level to chunk the processing at",
    )
    parser.add_argument(
        "--parallel", type=int, default=2, help="Number of parallel processes"
    )
    parser.add_argument("--stats-min", type=float, help="Hardcoded source min")
    parser.add_argument("--stats-max", type=float, help="Hardcoded source max")

    # Tone Mapping
    parser.add_argument("--exposure", type=float, default=tiler.DEFAULT_EXPOSURE)
    parser.add_argument(
        "--sb", "--shadow-break", type=float, default=tiler.SOFT_KNEE_SHADOW_BREAK
    )
    parser.add_argument(
        "--hb", "--highlight-break", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_BREAK
    )
    parser.add_argument(
        "--ss", "--shadow-slope", type=float, default=tiler.SOFT_KNEE_SHADOW_SLOPE
    )
    parser.add_argument(
        "--ms", "--mid-slope", type=float, default=tiler.SOFT_KNEE_MID_SLOPE
    )
    parser.add_argument(
        "--hs", "--highlight-slope", type=float, default=tiler.SOFT_KNEE_HIGHLIGHT_SLOPE
    )

    # Grading
    parser.add_argument("--gamma", type=float, default=2.6)
    parser.add_argument(
        "--sat", "--saturation", type=float, default=0.9
    )
    parser.add_argument(
        "--db", "--black-break", type=float, default=0.15
    )
    parser.add_argument(
        "--ls", "--black-slope", type=float, default=0.2
    )

    parser.add_argument(
        "--tonemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land tone mapping",
    )
    parser.add_argument(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable land final grading",
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
    args = parser.parse_args()

    if args.estimate:
        calculate_estimates(args)
        return

    setup_gdal_cdse()
    os.makedirs(".temp", exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    state_file = f".temp/state_{unique_id}.json"
    completed_subtiles: Set[str] = set()
    processed_tifs: List[str] = []

    if args.resume:
        resume_path = find_resume_path(args.resume)
        if resume_path:
            resume_state = restore_resume_state(resume_path)
            if resume_state:
                state_file = cast(str, resume_state["state_file"])
                unique_id = cast(str, resume_state["unique_id"])
                completed_subtiles = cast(Set[str], resume_state["completed_subtiles"])
                processed_tifs = cast(List[str], resume_state["processed_tifs"])

    requested_bbox = parse_bbox(args.bbox) if args.bbox else None
    date_paths = [date_path.strip() for date_path in args.date.split(",")]
    gebco_vrt_source = resolve_ocean_mask_source(args.ocean_background)

    if args.all_tiles:
        populate_s3_cache(date_paths)

    land_set = load_land_tiles("HLS.land.tiles.txt")
    mgrs_bases = discover_mgrs_bases(args, land_set, gebco_vrt_source)
    subtiles = expand_subtiles(mgrs_bases)

    if args.land:
        subtiles_to_process = [st for st in subtiles if st not in completed_subtiles]
        if subtiles_to_process:
            print(f"Starting processing for {len(subtiles_to_process)} sub-tiles...")
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_st = {
                    executor.submit(
                        process_single_tile,
                        st,
                        date_paths,
                        args,
                        unique_id,
                        gebco_vrt_source,
                    ): st
                    for st in subtiles_to_process
                }
                for future in as_completed(future_to_st):
                    st = future_to_st[future]
                    res = future.result()
                    completed_subtiles.add(st)
                    if res:
                        processed_tifs.append(res)
                    write_resume_state(
                        state_file, unique_id, completed_subtiles, processed_tifs, args
                    )
        else:
            print("All sub-tiles already processed.")
    else:
        print("Skipping land tile processing (--no-land).")

    if args.download:
        print("Download complete.")
        return

    # 3. Optional standalone ocean background
    prepared_ocean_background = prepare_ocean_background_for_output(
        args.ocean_background,
        requested_bbox,
        unique_id,
        args.resample_alg,
        args.max_zoom,
    )
    if prepared_ocean_background:
        if prepared_ocean_background not in processed_tifs:
            processed_tifs.insert(0, prepared_ocean_background)
    elif args.bbox:
        print(f"Warning: Ocean background not found, skipping: {args.ocean_background}")

    if not processed_tifs:
        print("Error: No data processed.")
        sys.exit(1)

    master_vrt = f".temp/master_{unique_id}.vrt"
    pixel_size = tiler.web_mercator_pixel_size(args.max_zoom)
    gdal.BuildVRT(master_vrt, processed_tifs, resolution="user", xRes=pixel_size, yRes=pixel_size)

    if args.vrt:
        print(f"Success! Master VRT: {master_vrt}")
        return

    temp_mbtiles = f".temp/tiles_{unique_id}.mbtiles"
    tiling_opts = {
        "format": args.format,
        "quality": args.quality,
        "resample_alg": args.resample_alg,
        "chunk_zoom": args.chunk_zoom,
        "processes": args.parallel,
        "blocksize": args.blocksize,
        "name": "Sentinel-2 Mosaic",
        "description": "Copernicus Sentinel data",
        "unique_id": unique_id,
    }
    if requested_bbox is not None:
        tiling_opts["chunk_bounds"] = tiler.lonlat_bbox_to_mercator_bounds(
            *requested_bbox
        )

    print("Generating MBTiles...")
    artifacts = tiler.run_tiling_simplified(master_vrt, temp_mbtiles, tiling_opts)

    print("Converting to PMTiles...")
    subprocess.run(["pmtiles", "convert", temp_mbtiles, args.output], check=True)
    print(f"Success! {args.output}")

    persistent_paths = {os.path.abspath(args.ocean_background)}
    cleanup_paths = [
        path
        for path in processed_tifs
        if os.path.abspath(path) not in persistent_paths
    ]

    for path in cleanup_paths + [master_vrt, temp_mbtiles] + artifacts.cleanup_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f"Warning: Could not remove temporary file {path}: {exc}")

    if os.path.exists(state_file):
        os.remove(state_file)


if __name__ == "__main__":
    main()
