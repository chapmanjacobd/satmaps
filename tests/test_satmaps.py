import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocean
import land_mgrs
import satmaps
import tiler
from satmaps import (
    fill_nan_nearest,
    get_tile_paths,
    list_mosaic_folders_for_tile,
    main,
)

class StubBand:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def ReadAsArray(self, xoff: int, yoff: int, width: int, height: int) -> np.ndarray:
        return self.data[yoff : yoff + height, xoff : xoff + width]

def test_list_mosaic_folders_for_tile_uses_cache(monkeypatch: object) -> None:
    # Pre-populate cache
    satmaps.S3_FOLDER_CACHE = {"2025/07/01": {"Sentinel-2_mosaic_2025_Q3_31TDF_0_0"}}

    found = list_mosaic_folders_for_tile("31TDF_0_0", ["2025/07/01"], ".cache")
    assert found == [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")]


def test_list_mosaic_folders_for_tile_first_match_only_short_circuits(monkeypatch: object) -> None:
    satmaps.S3_FOLDER_CACHE = {
        "2025/07/01": {"Sentinel-2_mosaic_2025_Q3_31TDF_0_0"},
        "2025/10/01": {"Sentinel-2_mosaic_2025_Q4_31TDF_0_0"},
    }
    checked_paths: list[str] = []

    def fail_if_called(path: str) -> list[str]:
        checked_paths.append(path)
        raise AssertionError("unexpected remote lookup")

    monkeypatch.setattr(satmaps.gdal, "ReadDir", fail_if_called)

    found = list_mosaic_folders_for_tile(
        "31TDF_0_0",
        ["2025/07/01", "2025/10/01"],
        ".cache",
        first_match_only=True,
    )

    assert found == [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")]
    assert checked_paths == []

def test_get_tile_paths_returns_s3_paths(monkeypatch: object) -> None:
    # Ensure local cache doesn't exist for this test
    monkeypatch.setattr("os.path.exists", lambda path: False)
    paths = get_tile_paths(
        "Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01", ".cache"
    )
    assert (
        paths["red"]
        == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B04.tif"
    )
    assert (
        paths["green"]
        == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B03.tif"
    )
    assert (
        paths["blue"]
        == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B02.tif"
    )

def test_fill_nan_nearest_fills_from_nearest_valid_pixel() -> None:
    arr = np.array(
        [
            [[1.0, np.nan], [np.nan, np.nan]],
            [[2.0, np.nan], [np.nan, np.nan]],
            [[3.0, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )

    filled = fill_nan_nearest(arr)

    expected = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(filled, expected)

def test_fill_nan_nearest_uses_explicit_valid_mask() -> None:
    arr = np.array(
        [
            [[10.0, 99.0, 99.0]],
            [[20.0, 88.0, 88.0]],
            [[30.0, 77.0, 77.0]],
        ],
        dtype=np.float32,
    )

    filled = fill_nan_nearest(arr, valid_mask=np.array([[True, False, False]]))

    expected = np.array(
        [
            [[10.0, 10.0, 10.0]],
            [[20.0, 20.0, 20.0]],
            [[30.0, 30.0, 30.0]],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(filled, expected)

def test_fill_nan_nearest_respects_fill_mask() -> None:
    arr = np.array(
        [
            [[10.0, 99.0, 50.0]],
            [[20.0, 88.0, 60.0]],
            [[30.0, 77.0, 70.0]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, False, False]])
    fill_mask = np.array([[False, True, False]])

    filled = fill_nan_nearest(arr, valid_mask=valid_mask, fill_mask=fill_mask)

    expected = np.array(
        [
            [[10.0, 10.0, 50.0]],
            [[20.0, 20.0, 60.0]],
            [[30.0, 30.0, 70.0]],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(filled, expected)

def test_create_gebco_ocean_vrt_masks_positive_values(tmp_path: Path) -> None:
    source_path = tmp_path / "gebco_source.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(source_path), 4, 1, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    ds.GetRasterBand(1).WriteArray(
        np.array([[-5.0, 0.0, 0.0005, 10.0]], dtype=np.float32)
    )
    ds = None

    source_vrt = tmp_path / "gebco_source.vrt"
    gdal.BuildVRT(str(source_vrt), [str(source_path)])

    ocean_vrt = tmp_path / "gebco_ocean.vrt"
    ocean.create_gebco_ocean_vrt(str(source_vrt), str(ocean_vrt))

    ds = gdal.Open(str(ocean_vrt))
    assert ds is not None
    arr = ds.ReadAsArray()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    assert nodata == -32767.0

    np.testing.assert_allclose(arr, np.array([[-5.0, 0.0, 0.0005, nodata]], dtype=np.float32))

def test_web_mercator_pixel_size_uses_output_zoom() -> None:
    assert satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM) == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM)
    )

def test_web_mercator_pixel_size_accepts_requested_zoom() -> None:
    assert satmaps.tiler.web_mercator_pixel_size(14) == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(14)
    )

def test_web_mercator_pixel_size_for_tile_size_adjusts_for_512px_tiles() -> None:
    assert satmaps.tiler.web_mercator_pixel_size_for_tile_size(7, 512) == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(8)
    )

def test_compute_in_memory_pixel_limit_uses_available_memory_and_cap(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(satmaps.tiler, "get_available_memory_bytes", lambda: 1_000_000)

    assert (
        satmaps.tiler.compute_in_memory_pixel_limit(
            10,
            usage_fraction=0.5,
            fallback_pixels=123,
            reserve_bytes=200_000,
            max_pixels=35_000,
        )
        == 35_000
    )

def test_compute_in_memory_pixel_limit_falls_back_when_memory_unknown(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(satmaps.tiler, "get_available_memory_bytes", lambda: 0)

    assert (
        satmaps.tiler.compute_in_memory_pixel_limit(
            10,
            usage_fraction=0.5,
            fallback_pixels=123,
        )
        == 123
    )

def test_runtime_memory_limits_scale_above_defaults(monkeypatch: object) -> None:
    monkeypatch.setattr(satmaps.tiler, "get_available_memory_bytes", lambda: 12 * 1024**3)

    assert satmaps.max_in_memory_write_pixels() > satmaps.DEFAULT_MAX_IN_MEMORY_WRITE_PIXELS
    assert ocean.max_in_memory_alpha_pixels() > ocean.DEFAULT_MAX_IN_MEMORY_ALPHA_PIXELS
    assert ocean.max_in_memory_color_pixels() > ocean.DEFAULT_MAX_IN_MEMORY_COLOR_PIXELS
    assert ocean.max_in_memory_sieve_mask_pixels() > ocean.DEFAULT_MAX_IN_MEMORY_SIEVE_MASK_PIXELS

def test_should_prefetch_tile_bands_fast_paths_skip_land_percentage_calc(
    monkeypatch: object,
) -> None:
    tile_grid = satmaps.TileGrid(
        projection="EPSG:32631",
        geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=10,
        height=10,
    )

    def fail(*args, **kwargs):
        raise AssertionError("land percentage should not be calculated")

    monkeypatch.setattr(satmaps, "estimate_subtile_land_percentage", fail)

    assert satmaps.should_prefetch_tile_bands(100.0, None, tile_grid)
    assert not satmaps.should_prefetch_tile_bands(0.0, None, tile_grid)

def test_should_prefetch_tile_bands_uses_land_percentage_threshold() -> None:
    tile_grid = satmaps.TileGrid(
        projection="EPSG:32631",
        geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=10,
        height=10,
    )
    mask_slabs = {
        0: satmaps.OceanMaskSlab(
            xoff=0,
            yoff=0,
            width=10,
            height=10,
            alpha_block=np.zeros((10, 10), dtype=np.float32),
            coverage_block=np.ones((10, 10), dtype=bool),
            fill_allowed_block=np.vstack(
                (
                    np.ones((3, 10), dtype=bool),
                    np.zeros((7, 10), dtype=bool),
                )
            ),
        )
    }

    assert satmaps.should_prefetch_tile_bands(20.0, mask_slabs, tile_grid)
    assert not satmaps.should_prefetch_tile_bands(40.0, mask_slabs, tile_grid)

def test_snapped_tile_grid_for_bbox_expands_to_tile_pixel_grid() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)

    snapped_bounds, pixel_size, zoom = ocean.snapped_tile_grid_for_bbox(bbox)

    assert zoom == ocean.DEFAULT_MAX_ZOOM
    assert pixel_size == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM)
    )

    mercator_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(*bbox)
    assert snapped_bounds[0] <= mercator_bounds[0]
    assert snapped_bounds[1] <= mercator_bounds[1]
    assert snapped_bounds[2] >= mercator_bounds[2]
    assert snapped_bounds[3] >= mercator_bounds[3]

def test_snapped_tile_grid_for_bbox_uses_requested_zoom() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)

    snapped_bounds, pixel_size, zoom = ocean.snapped_tile_grid_for_bbox(bbox, 14)

    assert zoom == 14
    assert pixel_size == pytest.approx(satmaps.tiler.web_mercator_pixel_size(14))
    mercator_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(*bbox)
    assert snapped_bounds[0] <= mercator_bounds[0]
    assert snapped_bounds[1] <= mercator_bounds[1]
    assert snapped_bounds[2] >= mercator_bounds[2]
    assert snapped_bounds[3] >= mercator_bounds[3]

def test_snapped_tile_grid_for_bbox_accepts_zoom_11() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)

    snapped_bounds, pixel_size, zoom = ocean.snapped_tile_grid_for_bbox(bbox, 11)

    assert zoom == 11
    assert pixel_size == pytest.approx(satmaps.tiler.web_mercator_pixel_size(11))
    mercator_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(*bbox)
    assert snapped_bounds[0] <= mercator_bounds[0]
    assert snapped_bounds[1] <= mercator_bounds[1]
    assert snapped_bounds[2] >= mercator_bounds[2]
    assert snapped_bounds[3] >= mercator_bounds[3]

def test_snapped_tile_grid_for_bbox_accepts_zoom_12() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)

    snapped_bounds, pixel_size, zoom = ocean.snapped_tile_grid_for_bbox(bbox, 12)

    assert zoom == 12
    assert pixel_size == pytest.approx(satmaps.tiler.web_mercator_pixel_size(12))
    mercator_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(*bbox)
    assert snapped_bounds[0] <= mercator_bounds[0]
    assert snapped_bounds[1] <= mercator_bounds[1]
    assert snapped_bounds[2] >= mercator_bounds[2]
    assert snapped_bounds[3] >= mercator_bounds[3]

def test_create_alpha_vrt_handles_near_nodata_and_shallow_values(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_near_nodata.tif"
    alpha_ds = driver.Create(str(alpha_source), 4, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    # -32766.95 is near nodata, and -49.0 is shallower than the fade threshold.
    alpha_ds.GetRasterBand(1).WriteArray(
        np.array([[-32767.0, -32766.95, -60.0, -49.0]], dtype=np.float32)
    )
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_near_nodata.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha_robust.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_allclose(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.array([[0.0, 0.0, 255.0, 0.0]], dtype=np.float32),
    )

def test_remove_small_enclosed_ocean_regions_prefers_land() -> None:
    ocean_mask = np.zeros((5, 5), dtype=bool)
    ocean_mask[2, 2] = True
    ocean_mask[0, 0] = True

    cleaned = ocean.remove_small_enclosed_ocean_regions(
        ocean_mask,
        (0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        150.0,
    )

    expected = np.zeros((5, 5), dtype=bool)
    expected[0, 0] = True
    np.testing.assert_array_equal(cleaned, expected)

def test_average_tile_blocks_skips_horizontal_ocean(monkeypatch: object) -> None:
    from satmaps import OceanMaskSlab, ProcessingWindow, average_tile_blocks

    processing_window = ProcessingWindow(xoff=3, yoff=0, width=4, height=4)
    mask_slabs = {
        0: OceanMaskSlab(
            xoff=3,
            yoff=0,
            width=4,
            height=4,
            alpha_block=np.zeros((4, 4), dtype=np.float32),
            coverage_block=np.ones((4, 4), dtype=bool),
            fill_allowed_block=np.ones((4, 4), dtype=bool),
        )
    }

    calls = []

    def mock_average_block(date_band_sets, xoff, yoff, width, height):
        calls.append((xoff, yoff, width, height))
        return np.zeros((3, height, width), dtype=np.float32), np.ones(
            (height, width), dtype=bool
        )

    monkeypatch.setattr("satmaps.average_block", mock_average_block)

    date_band_sets = []  # Empty for this test

    average_tile_blocks(date_band_sets, processing_window, mask_slabs)

    # It should have called mock_average_block with xoff=3 and width=4
    # Note: PROCESS_SLAB_HEIGHT is 24, so it should be one call for 4 rows
    assert len(calls) == 1
    assert calls[0] == (3, 0, 4, 4)

def test_build_local_season_date_weights_blends_across_equator() -> None:
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    tile_grid = satmaps.TileGrid(
        projection=wgs84.ExportToWkt(),
        geotransform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        width=1,
        height=4,
    )

    weights = satmaps.build_local_season_date_weights(
        tile_grid, 0, 0, 1, 4, winter=False
    )

    assert weights.shape == (2, 4, 1)
    np.testing.assert_allclose(weights[:, 0, 0], np.array([1.0, 0.0]), atol=1e-4)
    assert 0.5 < weights[0, 1, 0] < 1.0
    assert 0.0 < weights[0, 2, 0] < 0.5
    np.testing.assert_allclose(weights[:, 3, 0], np.array([0.0, 1.0]), atol=1e-4)

def test_build_local_season_date_weights_winter_flips_hemispheres() -> None:
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    tile_grid = satmaps.TileGrid(
        projection=wgs84.ExportToWkt(),
        geotransform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        width=1,
        height=4,
    )

    summer_weights = satmaps.build_local_season_date_weights(
        tile_grid, 0, 0, 1, 4, winter=False
    )
    winter_weights = satmaps.build_local_season_date_weights(
        tile_grid, 0, 0, 1, 4, winter=True
    )

    np.testing.assert_allclose(winter_weights[0], 1.0 - summer_weights[0], atol=1e-6)
    np.testing.assert_allclose(winter_weights[1], 1.0 - summer_weights[1], atol=1e-6)

def test_average_block_weighted_blend_falls_back_when_preferred_date_is_missing() -> None:
    north_missing = np.array(
        [
            [satmaps.SENTINEL_NODATA],
            [100.0],
        ],
        dtype=np.float32,
    )
    south_valid = np.array(
        [
            [20.0],
            [10.0],
        ],
        dtype=np.float32,
    )
    date_band_sets = [
        ([StubBand(north_missing), StubBand(north_missing), StubBand(north_missing)], []),
        ([StubBand(south_valid), StubBand(south_valid), StubBand(south_valid)], []),
    ]

    averaged, valid_mask = satmaps.average_block(
        date_band_sets,
        0,
        0,
        1,
        2,
        date_weights=np.array(
            [
                [[1.0], [1.0]],
                [[0.0], [0.0]],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(valid_mask, np.array([[True], [True]]))
    np.testing.assert_allclose(averaged[:, 0, 0], np.array([20.0, 20.0, 20.0]))
    np.testing.assert_allclose(averaged[:, 1, 0], np.array([100.0, 100.0, 100.0]))

def test_average_tile_blocks_skips_empty_slabs(monkeypatch: object) -> None:
    from satmaps import OceanMaskSlab, ProcessingWindow, average_tile_blocks

    processing_window = ProcessingWindow(xoff=0, yoff=24, width=10, height=24)
    mask_slabs = {
        24: OceanMaskSlab(
            xoff=0,
            yoff=24,
            width=10,
            height=24,
            alpha_block=np.zeros((24, 10), dtype=np.float32),
            coverage_block=np.ones((24, 10), dtype=bool),
            fill_allowed_block=np.ones((24, 10), dtype=bool),
        )
    }

    calls = []

    def mock_average_block(date_band_sets, xoff, yoff, width, height):
        calls.append((xoff, yoff, width, height))
        return np.zeros((3, height, width), dtype=np.float32), np.ones(
            (height, width), dtype=bool
        )

    monkeypatch.setattr("satmaps.average_block", mock_average_block)

    date_band_sets = []
    average_tile_blocks(date_band_sets, processing_window, mask_slabs)

    # SLAB_HEIGHT is 24.
    # Slab 0: 0-24 (Skip)
    # Slab 1: 24-48 (Contains 30-40) -> Processed
    # Slab 2: 48-72 (Skip)
    # Slab 3: 72-96 (Skip)
    # Slab 4: 96-100 (Skip)
    assert len(calls) == 1
    assert calls[0][1] == 24
    assert calls[0][3] == 24

def test_average_tile_blocks_masks_out_ocean_holes_inside_processed_slab(
    monkeypatch: object,
) -> None:
    from satmaps import OceanMaskSlab, ProcessingWindow, average_tile_blocks
    processing_window = ProcessingWindow(xoff=1, yoff=0, width=3, height=2)
    mask_slabs = {
        0: OceanMaskSlab(
            xoff=1,
            yoff=0,
            width=3,
            height=2,
            alpha_block=np.zeros((2, 3), dtype=np.float32),
            coverage_block=np.ones((2, 3), dtype=bool),
            fill_allowed_block=np.array(
                [
                    [True, False, True],
                    [True, False, True],
                ],
                dtype=bool,
            ),
        )
    }

    def mock_average_block(date_band_sets, xoff, yoff, width, height):
        assert (xoff, yoff, width, height) == (1, 0, 3, 2)
        averaged_block = np.ones((3, height, width), dtype=np.float32)
        source_valid = np.ones((height, width), dtype=bool)
        return averaged_block, source_valid

    monkeypatch.setattr("satmaps.average_block", mock_average_block)

    averaged, source_valid_mask, alpha_mask, fill_allowed_mask = average_tile_blocks([], processing_window, mask_slabs)

    expected_fill_allowed = np.array(
        [
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    expected_source_valid = np.array(
        [
            [True, False, True],
            [True, False, True],
        ],
        dtype=bool,
    )
    assert fill_allowed_mask is not None
    np.testing.assert_array_equal(fill_allowed_mask, expected_fill_allowed)
    np.testing.assert_array_equal(source_valid_mask, expected_source_valid)
    np.testing.assert_array_equal(
        alpha_mask,
        np.full((2, 3), 255, dtype=np.uint8),
    )
    np.testing.assert_array_equal(averaged[:, :, 0], np.ones((3, 2), dtype=np.float32))
    assert np.isnan(averaged[:, :, 1]).all()
    np.testing.assert_array_equal(averaged[:, :, 2], np.ones((3, 2), dtype=np.float32))

def test_alpha_mask_coastline_seam_pixel_is_filled_opaquely(
    monkeypatch: object,
) -> None:
    tile_grid = satmaps.TileGrid(
        projection="EPSG:32631",
        geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=3,
        height=1,
    )
    mask_dataset = gdal.GetDriverByName("MEM").Create("", 3, 1, 1, gdal.GDT_Float32)
    assert mask_dataset is not None
    mask_dataset.GetRasterBand(1).WriteArray(np.array([[0.0, 255.0, 255.0]], dtype=np.float32))

    collected = satmaps.collect_ocean_mask_slabs(
        satmaps.OceanMaskWarp(
            alpha_band=mask_dataset.GetRasterBand(1),
            dataset=mask_dataset,
        ),
        tile_grid,
    )
    assert collected is not None
    processing_window, mask_slabs = collected
    assert processing_window == satmaps.ProcessingWindow(xoff=0, yoff=0, width=2, height=1)

    def mock_average_block(date_band_sets, xoff, yoff, width, height):
        assert (xoff, yoff, width, height) == (0, 0, 2, 1)
        averaged_block = np.array(
            [[[1.0, 0.0]], [[0.5, 0.0]], [[0.25, 0.0]]],
            dtype=np.float32,
        )
        source_valid = np.array([[True, False]], dtype=bool)
        return averaged_block, source_valid

    monkeypatch.setattr("satmaps.average_block", mock_average_block)

    averaged, source_valid_mask, alpha_mask, fill_allowed_mask = satmaps.average_tile_blocks([], processing_window, mask_slabs)
    averaged = satmaps.fill_missing_pixels(averaged, source_valid_mask, fill_allowed_mask)

    dataset = gdal.GetDriverByName("MEM").Create("", 2, 1, 4, gdal.GDT_Byte)
    assert dataset is not None
    color_bands = [dataset.GetRasterBand(index + 1) for index in range(3)]
    alpha_band = dataset.GetRasterBand(4)
    satmaps.write_processed_blocks(
        averaged,
        alpha_mask,
        processing_window,
        argparse.Namespace(
            stats_min=0.0,
            stats_max=1.0,
            tonemap=False,
            grade=False,
            exposure=1.0,
            sb=0.3,
            hb=0.75,
            ss=1.4,
            ms=0.9,
            hs=0.5,
            sat=1.0,
            db=0.35,
            ls=0.35,
            ghb=None,
            gms=1.0,
            ghs=None,
            gamma=1.2,
        ),
        color_bands,
        alpha_band,
    )

    rgba = dataset.ReadAsArray()
    np.testing.assert_array_equal(rgba[:, 0, 0], np.array([255, 127, 63, 255], dtype=np.uint8))
    np.testing.assert_array_equal(rgba[:, 0, 1], np.array([255, 127, 63, 255], dtype=np.uint8))

def test_average_tile_blocks_limits_land_fill_to_nearby_source_pixels(
    monkeypatch: object,
) -> None:
    from satmaps import OceanMaskSlab, ProcessingWindow, average_tile_blocks

    processing_window = ProcessingWindow(xoff=0, yoff=0, width=6, height=1)
    mask_slabs = {
        0: OceanMaskSlab(
            xoff=0,
            yoff=0,
            width=6,
            height=1,
            alpha_block=np.zeros((1, 6), dtype=np.float32),
            coverage_block=np.ones((1, 6), dtype=bool),
            fill_allowed_block=np.ones((1, 6), dtype=bool),
        )
    }

    def mock_average_block(date_band_sets, xoff, yoff, width, height):
        assert (xoff, yoff, width, height) == (0, 0, 6, 1)
        averaged_block = np.array(
            [[[1.0, np.nan, np.nan, np.nan, np.nan, 2.0]]] * 3,
            dtype=np.float32,
        )
        source_valid = np.array([[True, False, False, False, False, True]], dtype=bool)
        return averaged_block, source_valid

    monkeypatch.setattr("satmaps.average_block", mock_average_block)

    averaged, source_valid_mask, alpha_mask, fill_allowed_mask = average_tile_blocks(
        [],
        processing_window,
        mask_slabs,
    )
    filled = satmaps.fill_missing_pixels(averaged, source_valid_mask, fill_allowed_mask)

    assert fill_allowed_mask is not None
    np.testing.assert_array_equal(
        fill_allowed_mask,
        np.array([[True, True, False, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        alpha_mask,
        np.array([[255, 255, 0, 0, 255, 255]], dtype=np.uint8),
    )
    np.testing.assert_allclose(filled[:, 0, 1], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.isnan(filled[:, 0, 2]).all()
    assert np.isnan(filled[:, 0, 3]).all()
    np.testing.assert_allclose(filled[:, 0, 4], np.array([2.0, 2.0, 2.0], dtype=np.float32))

def test_write_processed_blocks_block_path_matches_in_memory_path(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(satmaps, "max_in_memory_write_pixels", lambda: 1)
    processing_window = satmaps.ProcessingWindow(xoff=0, yoff=0, width=2, height=1)
    averaged = np.array(
        [[[1.0, 1.0]], [[0.5, 0.5]], [[0.25, 0.25]]],
        dtype=np.float32,
    )
    alpha_mask = np.full((1, 2), 255, dtype=np.uint8)

    dataset = gdal.GetDriverByName("MEM").Create("", 2, 1, 4, gdal.GDT_Byte)
    assert dataset is not None
    color_bands = [dataset.GetRasterBand(index + 1) for index in range(3)]
    alpha_band = dataset.GetRasterBand(4)
    satmaps.write_processed_blocks(
        averaged,
        alpha_mask,
        processing_window,
        argparse.Namespace(
            stats_min=0.0,
            stats_max=1.0,
            tonemap=False,
            grade=False,
            exposure=1.0,
            sb=0.3,
            hb=0.75,
            ss=1.4,
            ms=0.9,
            hs=0.5,
            sat=1.0,
            db=0.35,
            ls=0.35,
            ghb=None,
            gms=1.0,
            ghs=None,
            gamma=1.2,
        ),
        color_bands,
        alpha_band,
    )

    rgba = dataset.ReadAsArray()
    np.testing.assert_array_equal(rgba[:, 0, 0], np.array([255, 127, 63, 255], dtype=np.uint8))
    np.testing.assert_array_equal(rgba[:, 0, 1], np.array([255, 127, 63, 255], dtype=np.uint8))

def test_open_gebco_mask_avoids_update_mode_warning(tmp_path: Path) -> None:
    gebco_path = tmp_path / "gebco.tif"
    gebco_ds = gdal.GetDriverByName("GTiff").Create(str(gebco_path), 4, 3, 4, gdal.GDT_Byte)
    assert gebco_ds is not None
    gebco_ds.SetGeoTransform((0.0, 100.0, 0.0, 300.0, 0.0, -100.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    gebco_ds.SetProjection(srs.ExportToWkt())
    alpha_band = gebco_ds.GetRasterBand(4)
    alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
    alpha_band.WriteArray(np.full((3, 4), 255, dtype=np.uint8))
    gebco_ds = None

    tile_grid = satmaps.TileGrid(
        projection=srs.ExportToWkt(),
        geotransform=(0.0, 100.0, 0.0, 300.0, 0.0, -100.0),
        width=4,
        height=3,
    )

    messages: list[str] = []

    def handler(err_class: int, err_num: int, msg: str) -> None:
        del err_class, err_num
        messages.append(msg)

    gdal.PushErrorHandler(handler)
    try:
        mask = satmaps.open_gebco_mask(str(gebco_path), tile_grid, "31TDF_0_0")
    finally:
        gdal.PopErrorHandler()

    assert mask is not None
    assert mask.dataset.GetGeoTransform() == pytest.approx(tile_grid.geotransform)
    assert mask.dataset.RasterXSize == tile_grid.width
    assert mask.dataset.RasterYSize == tile_grid.height
    assert mask.alpha_band.GetNoDataValue() == -1.0
    assert not any("creation ignored in update mode" in message.lower() for message in messages)

def test_open_gebco_mask_crops_aligned_alpha_band_without_warp(monkeypatch: object) -> None:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    gebco_ds = gdal.GetDriverByName("MEM").Create("", 4, 3, 4, gdal.GDT_Byte)
    assert gebco_ds is not None
    gebco_ds.SetGeoTransform((0.0, 100.0, 0.0, 300.0, 0.0, -100.0))
    gebco_ds.SetProjection(srs.ExportToWkt())
    gebco_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    translate_options_calls: list[dict[str, object]] = []
    translated_mask_ds = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
    assert translated_mask_ds is not None

    monkeypatch.setattr("satmaps.gdal.Open", lambda path: gebco_ds)
    monkeypatch.setattr(
        "satmaps.gdal.TranslateOptions",
        lambda **kwargs: translate_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Translate",
        lambda destination, source, options=None: translated_mask_ds,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: (_ for _ in ()).throw(AssertionError("unexpected warp")),
    )

    tile_grid = satmaps.TileGrid(
        projection=srs.ExportToWkt(),
        geotransform=(0.0, 100.0, 0.0, 300.0, 0.0, -100.0),
        width=4,
        height=3,
    )
    mask = satmaps.open_gebco_mask("gebco.tif", tile_grid, "31TDF_0_0")

    assert mask is not None
    assert translate_options_calls[0]["srcWin"] == (0, 0, 4, 3)
    assert translate_options_calls[0]["bandList"] == [4]
    assert translate_options_calls[0]["outputType"] == gdal.GDT_Float32

def test_open_gebco_mask_crops_source_before_warp(monkeypatch: object) -> None:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    gebco_ds = gdal.GetDriverByName("MEM").Create("", 16, 16, 4, gdal.GDT_Byte)
    assert gebco_ds is not None
    gebco_ds.SetGeoTransform((0.0, 100.0, 0.0, 1600.0, 0.0, -100.0))
    gebco_ds.SetProjection(srs.ExportToWkt())
    gebco_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    translate_options_calls: list[dict[str, object]] = []
    warp_options_calls: list[dict[str, object]] = []
    cropped_mask_ds = gdal.GetDriverByName("MEM").Create("", 6, 8, 1, gdal.GDT_Float32)
    assert cropped_mask_ds is not None
    warped_mask_ds = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
    assert warped_mask_ds is not None

    monkeypatch.setattr("satmaps.gdal.Open", lambda path: gebco_ds)
    monkeypatch.setattr("satmaps.get_aligned_tile_grid_src_win", lambda dataset, tile_grid: None)
    monkeypatch.setattr(
        "satmaps.get_tile_grid_source_src_win",
        lambda dataset, tile_grid, halo_pixels=0: (2, 3, 6, 8),
    )
    monkeypatch.setattr(
        "satmaps.gdal.TranslateOptions",
        lambda **kwargs: translate_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Translate",
        lambda destination, source, options=None: cropped_mask_ds,
    )
    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: warped_mask_ds,
    )

    tile_grid = satmaps.TileGrid(
        projection=srs.ExportToWkt(),
        geotransform=(200.0, 100.0, 0.0, 1300.0, 0.0, -100.0),
        width=4,
        height=3,
    )
    mask = satmaps.open_gebco_mask("gebco.tif", tile_grid, "31TDF_0_0")

    assert mask is not None
    assert translate_options_calls[0]["srcWin"] == (2, 3, 6, 8)
    assert translate_options_calls[0]["bandList"] == [4]
    assert warp_options_calls[0]["srcBands"] == [1]
    assert warp_options_calls[0]["srcAlpha"] is False
    assert warp_options_calls[0]["multithread"] is True
    assert warp_options_calls[0]["warpOptions"] == ["NUM_THREADS=ALL_CPUS"]

def test_warp_band_dataset_to_tile_grid_keeps_lanczos_neighbors_continuous(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source_3857_shifted.tif"
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(source_path),
        64,
        64,
        1,
        gdal.GDT_Float32,
    )
    assert dataset is not None
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 640.0, 0.0, -10.0))
    dataset.GetRasterBand(1).WriteArray(
        np.add.outer(np.arange(64, dtype=np.float32) * 10.0, np.arange(64, dtype=np.float32))
    )
    dataset = None

    def render_neighbor(tile_index: int) -> np.ndarray:
        # Offset the target grid by half a source pixel so the warp path, not the aligned crop
        # path, renders each tile independently.
        tile_grid = satmaps.TileGrid(
            projection=srs.ExportToWkt(),
            geotransform=(5.0 + tile_index * 160.0, 10.0, 0.0, 635.0, 0.0, -10.0),
            width=16,
            height=16,
        )
        warped = satmaps.warp_band_dataset_to_tile_grid(str(source_path), tile_grid, "lanczos")
        array = warped.ReadAsArray()
        assert array is not None
        return array.astype(np.float32)

    left = render_neighbor(0)
    right = render_neighbor(1)
    seam_diff = right[:, 0] - left[:, -1]

    np.testing.assert_allclose(seam_diff, np.ones_like(seam_diff), atol=1e-5)

def test_build_land_output_tile_plan_adds_lanczos_halo() -> None:
    plan = satmaps.build_land_output_tile_plan("13/1/2.webp", 512, "lanczos")

    assert plan.halo_pixels == 3
    assert plan.width == 518
    assert plan.height == 518
    assert plan.core_src_win == (3, 3, 512, 512)
    assert plan.bounds == pytest.approx(tiler.get_web_mercator_bounds(13, 1, 2))

def test_build_output_tile_contributor_index_preserves_work_unit_order() -> None:
    work_units = (
        satmaps.LandWorkUnit("31TDF_1_0", ("31TDF_1_0",)),
        satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",)),
    )

    contributors_by_tile = satmaps.build_output_tile_contributor_index(
        work_units,
        {
            "31TDF_1_0": ("13/1/2.webp",),
            "31TDF_0_0": ("13/1/2.webp", "13/1/3.webp"),
        },
    )

    assert contributors_by_tile == {
        "13/1/2.webp": ("31TDF_1_0", "31TDF_0_0"),
        "13/1/3.webp": ("31TDF_0_0",),
    }

def test_build_output_tile_contributor_index_can_consume_candidates() -> None:
    work_units = (
        satmaps.LandWorkUnit("31TDF_1_0", ("31TDF_1_0",)),
        satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",)),
    )
    contributor_tile_candidates = {
        "31TDF_1_0": ("13/1/2.webp",),
        "31TDF_0_0": ("13/1/2.webp", "13/1/3.webp"),
    }

    contributors_by_tile = satmaps.build_output_tile_contributor_index(
        work_units,
        contributor_tile_candidates,
        consume_candidates=True,
    )

    assert contributors_by_tile == {
        "13/1/2.webp": ("31TDF_1_0", "31TDF_0_0"),
        "13/1/3.webp": ("31TDF_0_0",),
    }
    assert contributor_tile_candidates == {}

def test_create_alpha_vrt_masks_nodata_and_shallow_ocean(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_source.tif"
    alpha_ds = driver.Create(str(alpha_source), 4, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(
        np.array([[-60.0, -50.0, -5.0, -32767.0]], dtype=np.float32)
    )
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_source.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_allclose(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.array([[255.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

def test_create_alpha_vrt_block_path_matches_threshold_output(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_alpha_pixels", lambda: 1)
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_source_block.tif"
    alpha_ds = driver.Create(str(alpha_source), 4, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(
        np.array([[-60.0, -50.0, -5.0, -32767.0]], dtype=np.float32)
    )
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_source_block.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha_block.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_allclose(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.array([[255.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

def test_create_alpha_vrt_large_cleanup_removes_small_ocean_regions(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_alpha_pixels", lambda: 1)
    monkeypatch.setattr(ocean, "MAX_COMPONENT_CLEANUP_PIXELS", 0)
    monkeypatch.setattr(ocean, "SMALL_OCEAN_MAX_AREA_SQ_M", 2.0)
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_source_large_cleanup.tif"
    alpha_ds = driver.Create(str(alpha_source), 3, 3, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(
        np.array(
            [
                [-5.0, -5.0, -5.0],
                [-5.0, -60.0, -5.0],
                [-5.0, -5.0, -5.0],
            ],
            dtype=np.float32,
        )
    )
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_source_large_cleanup.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha_large_cleanup.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_array_equal(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.zeros((3, 3), dtype=np.uint8),
    )

def test_create_alpha_vrt_large_cleanup_preserves_land_holes(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_alpha_pixels", lambda: 1)
    monkeypatch.setattr(ocean, "MAX_COMPONENT_CLEANUP_PIXELS", 0)
    monkeypatch.setattr(ocean, "SMALL_OCEAN_MAX_AREA_SQ_M", 2.0)
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_source_large_hole.tif"
    alpha_ds = driver.Create(str(alpha_source), 3, 3, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(
        np.array(
            [
                [-60.0, -60.0, -60.0],
                [-60.0, -5.0, -60.0],
                [-60.0, -60.0, -60.0],
            ],
            dtype=np.float32,
        )
    )
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_source_large_hole.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha_large_hole.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_array_equal(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.array(
            [
                [255, 255, 255],
                [255, 0, 255],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )

def test_create_sieve_cleanup_mask_dataset_prefers_mem(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_sieve_mask_pixels", lambda: 9)
    driver = gdal.GetDriverByName("MEM")
    alpha_ds = driver.Create("", 3, 3, 1, gdal.GDT_Byte)

    cleanup_ds, cleanup_path = ocean.create_sieve_cleanup_mask_dataset(
        alpha_ds,
        tmp_path / "cleanup_mask.tif",
    )

    assert cleanup_ds.GetDriver().ShortName == "MEM"
    assert cleanup_path is None

def test_create_sieve_cleanup_mask_dataset_uses_uncompressed_gtiff(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_sieve_mask_pixels", lambda: 0)
    alpha_ds = gdal.GetDriverByName("MEM").Create("", 3, 3, 1, gdal.GDT_Byte)

    cleanup_ds, cleanup_path = ocean.create_sieve_cleanup_mask_dataset(
        alpha_ds,
        tmp_path / "cleanup_mask.tif",
    )

    assert cleanup_ds.GetDriver().ShortName == "GTiff"
    assert cleanup_path is not None
    assert cleanup_ds.GetMetadata("IMAGE_STRUCTURE").get("COMPRESSION") != "ZSTD"

    cleanup_ds = None
    ocean.remove_if_exists(cleanup_path)

def test_create_ocean_rgb_tif_colorizes_in_blocks(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    depth_path = tmp_path / "depth.tif"
    depth_ds = driver.Create(str(depth_path), 2, 2, 1, gdal.GDT_Float32)
    depth_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    depth_ds.SetProjection(srs.ExportToWkt())
    depth_ds.GetRasterBand(1).WriteArray(
        np.array([[-1000.0, -500.0], [-250.0, 0.0]], dtype=np.float32)
    )
    depth_ds = None

    hillshade_path = tmp_path / "hillshade.tif"
    hillshade_ds = driver.Create(str(hillshade_path), 2, 2, 1, gdal.GDT_Byte)
    hillshade_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    hillshade_ds.SetProjection(srs.ExportToWkt())
    hillshade_ds.GetRasterBand(1).WriteArray(
        np.array([[0, 64], [128, 255]], dtype=np.uint8)
    )
    hillshade_ds = None

    output_path = tmp_path / "ocean_rgb.tif"
    ocean.create_ocean_rgb_tif(
        str(depth_path),
        str(hillshade_path),
        str(output_path),
        ocean.OceanStyleOptions(tonemap=False, grade=False),
    )

    rgb_ds = gdal.Open(str(output_path))
    assert rgb_ds is not None
    arr = rgb_ds.ReadAsArray()
    assert arr.shape == (3, 2, 2)
    assert arr.dtype == np.uint8

    style = ocean.OceanStyleOptions(tonemap=False, grade=False)
    expected_rgb = ocean.colorize_ocean_depths(
        np.array([[-1000.0, -500.0], [-250.0, 0.0]], dtype=np.float32),
        style,
    )
    expected_shade = 0.35 + 0.65 * np.clip(
        np.array([[0, 64], [128, 255]], dtype=np.float32) / 255.0,
        0.0,
        1.0,
    )
    expected = (np.clip(expected_rgb * expected_shade[np.newaxis, :, :], 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    np.testing.assert_array_equal(arr, expected)

def test_create_ocean_rgb_tif_block_path_matches_full_read(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ocean, "max_in_memory_color_pixels", lambda: 1)
    driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    depth_path = tmp_path / "depth_block.tif"
    depth_ds = driver.Create(str(depth_path), 2, 2, 1, gdal.GDT_Float32)
    depth_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    depth_ds.SetProjection(srs.ExportToWkt())
    depth_ds.GetRasterBand(1).WriteArray(
        np.array([[-1000.0, -500.0], [-250.0, 0.0]], dtype=np.float32)
    )
    depth_ds = None

    hillshade_path = tmp_path / "hillshade_block.tif"
    hillshade_ds = driver.Create(str(hillshade_path), 2, 2, 1, gdal.GDT_Byte)
    hillshade_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    hillshade_ds.SetProjection(srs.ExportToWkt())
    hillshade_ds.GetRasterBand(1).WriteArray(
        np.array([[0, 64], [128, 255]], dtype=np.uint8)
    )
    hillshade_ds = None

    output_path = tmp_path / "ocean_rgb_block.tif"
    style = ocean.OceanStyleOptions(tonemap=False, grade=False)
    ocean.create_ocean_rgb_tif(
        str(depth_path),
        str(hillshade_path),
        str(output_path),
        style,
    )

    rgb_ds = gdal.Open(str(output_path))
    assert rgb_ds is not None
    expected_rgb = ocean.colorize_ocean_depths(
        np.array([[-1000.0, -500.0], [-250.0, 0.0]], dtype=np.float32),
        style,
    )
    expected_shade = 0.35 + 0.65 * np.clip(
        np.array([[0, 64], [128, 255]], dtype=np.float32) / 255.0,
        0.0,
        1.0,
    )
    np.testing.assert_array_equal(
        rgb_ds.ReadAsArray(),
        (np.clip(expected_rgb * expected_shade[np.newaxis, :, :], 0.0, 1.0) * 255.0).astype(
            np.uint8
        ),
    )

def test_create_rgb_with_alpha_vrt_preserves_alpha_values(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    rgb_path = tmp_path / "rgb.tif"
    rgb_ds = driver.Create(str(rgb_path), 2, 1, 3, gdal.GDT_Byte)
    rgb_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    rgb_ds.SetProjection(srs.ExportToWkt())
    for band_index, value in enumerate((10, 20, 30), start=1):
        rgb_ds.GetRasterBand(band_index).WriteArray(np.array([[value, value]], dtype=np.uint8))
    rgb_ds = None

    alpha_source = tmp_path / "alpha_source.tif"
    alpha_ds = driver.Create(str(alpha_source), 2, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(np.array([[-60.0, -32767.0]], dtype=np.float32))
    alpha_ds = None

    alpha_source_vrt = tmp_path / "alpha_source.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])
    alpha_vrt = tmp_path / "alpha.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    rgba_vrt = tmp_path / "rgba.vrt"
    ocean.create_rgb_with_alpha_vrt(str(rgb_path), str(alpha_vrt), str(rgba_vrt))
    assert not (tmp_path / "rgba.alpha.tif").exists()

    rgba_ds = gdal.Open(str(rgba_vrt))
    assert rgba_ds is not None
    np.testing.assert_array_equal(
        rgba_ds.GetRasterBand(4).ReadAsArray(),
        np.array([[255, 0]], dtype=np.uint8),
    )

def test_translate_rgba_vrt_preserves_alpha_values(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    rgb_path = tmp_path / "rgb.tif"
    rgb_ds = driver.Create(str(rgb_path), 2, 1, 3, gdal.GDT_Byte)
    rgb_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    rgb_ds.SetProjection(srs.ExportToWkt())
    for band_index, value in enumerate((10, 20, 30), start=1):
        rgb_ds.GetRasterBand(band_index).WriteArray(np.array([[value, value]], dtype=np.uint8))
    rgb_ds = None

    alpha_source = tmp_path / "alpha_source.tif"
    alpha_ds = driver.Create(str(alpha_source), 2, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(np.array([[-60.0, -32767.0]], dtype=np.float32))
    alpha_ds = None

    alpha_source_vrt = tmp_path / "alpha_source.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])
    alpha_vrt = tmp_path / "alpha.vrt"
    ocean.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    rgba_vrt = tmp_path / "rgba.vrt"
    ocean.create_rgb_with_alpha_vrt(str(rgb_path), str(alpha_vrt), str(rgba_vrt))
    output_tif = tmp_path / "out.tif"
    ocean.translate_rgba_vrt(str(rgba_vrt), str(output_tif))

    out_ds = gdal.Open(str(output_tif))
    assert out_ds is not None
    np.testing.assert_array_equal(
        out_ds.GetRasterBand(4).ReadAsArray(),
        np.array([[255, 0]], dtype=np.uint8),
    )
    assert not (tmp_path / ".temp_out.tif").exists()

def test_prepare_ocean_background_publishes_staged_tif_when_warping(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    source_path = tmp_path / "ocean.tif"
    source_path.write_text("ocean")
    destinations: list[str] = []
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: destinations.append(destination)
        or Path(destination).write_text("warped"),
    )

    prepared = satmaps.prepare_ocean_background_for_output(
        str(source_path),
        (0.0, 0.0, 1.0, 1.0),
        "output.pmtiles",
        "lanczos",
        13,
        256,
    )

    assert prepared == ".temp/output_ocean_bbox.tif"
    assert destinations == [".temp/.temp_output_ocean_bbox.tif"]
    assert Path(prepared).exists()
    assert not Path(".temp/.temp_output_ocean_bbox.tif").exists()
    assert warp_options_calls[0]["multithread"] is True
    assert warp_options_calls[0]["warpOptions"] == ["NUM_THREADS=ALL_CPUS"]

def test_prepare_ocean_background_crops_aligned_mercator_source_without_warp(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    bbox = (0.0, 0.0, 1.0, 1.0)
    snapped_bounds, pixel_size, _zoom = ocean.snapped_tile_grid_for_bbox(bbox, 13)
    width = int(round((snapped_bounds[2] - snapped_bounds[0]) / pixel_size))
    height = int(round((snapped_bounds[3] - snapped_bounds[1]) / pixel_size))

    source_path = tmp_path / "ocean.tif"
    source_ds = gdal.GetDriverByName("GTiff").Create(str(source_path), width, height, 4, gdal.GDT_Byte)
    assert source_ds is not None
    source_ds.SetGeoTransform((snapped_bounds[0], pixel_size, 0.0, snapped_bounds[3], 0.0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    source_ds.SetProjection(srs.ExportToWkt())
    source_ds.GetRasterBand(1).Fill(1)
    source_ds = None

    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: (_ for _ in ()).throw(AssertionError("unexpected warp")),
    )

    prepared = satmaps.prepare_ocean_background_for_output(
        str(source_path),
        bbox,
        "output.pmtiles",
        "lanczos",
        13,
        256,
    )

    assert prepared == ".temp/output_ocean_bbox.tif"
    prepared_ds = gdal.Open(prepared)
    assert prepared_ds is not None
    assert prepared_ds.GetGeoTransform() == pytest.approx(
        (snapped_bounds[0], pixel_size, 0.0, snapped_bounds[3], 0.0, -pixel_size)
    )
    prepared_ds = None

def test_get_bbox_scan_window_crops_before_scanning(tmp_path: Path) -> None:
    source_path = tmp_path / "scan_window.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(source_path), 100, 100, 1, gdal.GDT_Byte)
    assert dataset is not None
    dataset.SetGeoTransform((0.0, 0.1, 0.0, 10.0, 0.0, -0.1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    src_win = satmaps.get_bbox_scan_window(dataset, (1.0, 1.0, 2.0, 2.0))

    assert src_win == (10, 80, 10, 10)

def test_get_bbox_scan_window_returns_none_outside_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "scan_window_outside.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(source_path), 100, 100, 1, gdal.GDT_Byte)
    assert dataset is not None
    dataset.SetGeoTransform((0.0, 0.1, 0.0, 10.0, 0.0, -0.1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    assert satmaps.get_bbox_scan_window(dataset, (20.0, 20.0, 21.0, 21.0)) is None

def test_convert_raster_to_pmtiles_stages_final_output(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()
    (temp_dir / "output.mbtiles").write_text("mbtiles")
    converted: list[list[str]] = []

    monkeypatch.setattr(
        "satmaps.tiler.run_tiling_simplified",
        lambda input_raster, output_mbtiles, options: tiler.TilingArtifacts(
            final_vrt=input_raster,
            cleanup_paths=[],
        ),
    )

    def fake_subprocess_run(command: list[str], check: bool) -> None:
        assert check is True
        converted.append(command)
        Path(command[-1]).write_text("pmtiles")

    monkeypatch.setattr("satmaps.subprocess.run", fake_subprocess_run)

    packaged = satmaps.convert_raster_to_pmtiles(
        "input.vrt",
        "output.pmtiles",
        tile_format="webp",
        quality=80,
        resample_alg="lanczos",
        chunk_zoom=4,
        parallel=1,
        blocksize=512,
        name="Test",
        description="Test",
    )

    assert converted == [["pmtiles", "convert", ".temp/output.mbtiles", ".temp_output.pmtiles"]]
    assert Path("output.pmtiles").read_text() == "pmtiles"
    assert not Path(".temp_output.pmtiles").exists()
    assert packaged.temp_mbtiles == ".temp/output.mbtiles"

def test_convert_raster_to_pmtiles_cleans_inputs_after_mbtiles_before_pmtiles(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()
    cleanup_target = temp_dir / "land_31TDF_0_0_3857.tif"
    cleanup_target.write_text("tif")
    converted: list[list[str]] = []

    def fake_run_tiling_simplified(input_raster, output_mbtiles, options):
        Path(output_mbtiles).write_text("mbtiles")
        assert cleanup_target.exists()
        return tiler.TilingArtifacts(final_vrt=input_raster, cleanup_paths=[])

    def fake_subprocess_run(command: list[str], check: bool) -> None:
        assert check is True
        assert not cleanup_target.exists()
        converted.append(command)
        Path(command[-1]).write_text("pmtiles")

    monkeypatch.setattr("satmaps.tiler.run_tiling_simplified", fake_run_tiling_simplified)
    monkeypatch.setattr("satmaps.subprocess.run", fake_subprocess_run)

    satmaps.convert_raster_to_pmtiles(
        "input.vrt",
        "output.pmtiles",
        tile_format="webp",
        quality=80,
        resample_alg="lanczos",
        chunk_zoom=4,
        parallel=1,
        blocksize=512,
        name="Test",
        description="Test",
        cleanup_input_paths=[str(cleanup_target)],
    )

    assert converted == [["pmtiles", "convert", ".temp/output.mbtiles", ".temp_output.pmtiles"]]

def test_convert_tile_tree_to_pmtiles_uses_requested_bbox(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    (tmp_path / "tiles").mkdir()
    build_calls: list[dict[str, object]] = []
    overview_calls: list[tuple[str, str]] = []
    converted: list[list[str]] = []

    def fake_build_mbtiles_from_webp_tree(input_dir: str, output_mbtiles: str, **kwargs: object) -> int:
        build_calls.append(
            {
                "input_dir": input_dir,
                "output_mbtiles": output_mbtiles,
                **kwargs,
            }
        )
        Path(output_mbtiles).write_text("mbtiles")
        return 1

    def fake_build_mbtiles_overviews(mbtiles_path: str, resample_alg: str) -> None:
        overview_calls.append((mbtiles_path, resample_alg))

    def fake_subprocess_run(command: list[str], check: bool) -> None:
        assert check is True
        converted.append(command)
        Path(command[-1]).write_text("pmtiles")

    monkeypatch.setattr("satmaps.tiler.build_mbtiles_from_webp_tree", fake_build_mbtiles_from_webp_tree)
    monkeypatch.setattr("satmaps.tiler.build_mbtiles_overviews", fake_build_mbtiles_overviews)
    monkeypatch.setattr("satmaps.tiler.finalize_mbtiles_metadata", lambda mbtiles_path: None)
    monkeypatch.setattr("satmaps.subprocess.run", fake_subprocess_run)

    temp_mbtiles = satmaps.convert_tile_tree_to_pmtiles(
        "tiles",
        "output.pmtiles",
        resample_alg="lanczos",
        max_zoom=13,
        name="Test",
        description="Test",
        requested_bbox=(0.0, 1.0, 2.0, 3.0),
    )

    assert build_calls == [
        {
            "input_dir": "tiles",
            "output_mbtiles": ".temp/output.mbtiles",
            "name": "Test",
            "description": "Test",
            "maxzoom": 13,
            "bounds_wgs84": (0.0, 1.0, 2.0, 3.0),
        }
    ]
    assert overview_calls == [(".temp/output.mbtiles", "lanczos")]
    assert converted == [["pmtiles", "convert", ".temp/output.mbtiles", ".temp_output.pmtiles"]]
    assert temp_mbtiles == ".temp/output.mbtiles"

def test_create_hillshade_tif_uses_gdal_demprocessing(monkeypatch: object, tmp_path: Path) -> None:
    dem_processing_calls: list[tuple[str, str, str, object]] = []
    options_calls: list[dict[str, object]] = []

    class DummyDataset:
        pass

    monkeypatch.setattr(
        "ocean.gdal.DEMProcessingOptions",
        lambda **kwargs: options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "ocean.gdal.DEMProcessing",
        lambda destination, source, processing, options=None: dem_processing_calls.append(
            (destination, source, processing, options)
        )
        or Path(destination).write_text("hillshade")
        or DummyDataset(),
    )

    output_path = tmp_path / "hillshade.tif"
    result = ocean.create_hillshade_tif(str(tmp_path / "in.vrt"), str(output_path), 5.0)

    assert result == str(output_path)
    assert options_calls == [
        {
            "format": "GTiff",
            "creationOptions": list(ocean.GTIFF_CREATION_OPTIONS),
            "multiDirectional": True,
            "zFactor": 5.0,
        }
    ]
    assert dem_processing_calls == [
        (
            str(tmp_path / ".temp_hillshade.tif"),
            str(tmp_path / "in.vrt"),
            "hillshade",
            options_calls[0],
        )
    ]
    assert output_path.exists()

def test_ocean_main_uses_default_positionals(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            alpha_tif=".temp/alpha.tif",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif=str(kwargs["destination"]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ocean.py"],
    )
    monkeypatch.setattr(
        "ocean.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean.main()

    assert called["gebco_zip"] == ocean.DEFAULT_GEBCO_ZIP
    assert called["destination"] == ocean.DEFAULT_OUTPUT
    assert called["bbox"] is None
    assert called["vrt"] is False
    assert called["parallel"] == 40
    assert called["chunk_size"] == ocean.DEFAULT_OCEAN_CHUNK_SIZE
    assert called["style"] == ocean.OceanStyleOptions()

def test_ocean_main_parses_bbox_when_provided(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            alpha_tif=".temp/alpha.tif",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif=str(kwargs["destination"]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ocean.py", "--bbox", "0,0,1,1"],
    )
    monkeypatch.setattr(
        "ocean.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean.main()

    assert called["bbox"] == (0.0, 0.0, 1.0, 1.0)

def test_ocean_main_enables_vrt_mode(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            alpha_tif=".temp/alpha.tif",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif="ocean.vrt",
        )

    monkeypatch.setattr(sys, "argv", ["ocean.py", "--vrt"])
    monkeypatch.setattr(
        "ocean.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean.main()

    assert called["vrt"] is True

def test_ocean_main_passes_parallel_chunk_flags(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            alpha_tif=".temp/alpha.tif",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif="ocean.vrt",
        )

    monkeypatch.setattr(sys, "argv", ["ocean.py", "--parallel", "4", "--chunk-size", "2048"])
    monkeypatch.setattr("ocean.generate_ocean_background", fake_generate_ocean_background)

    ocean.main()

    assert called["parallel"] == 4
    assert called["chunk_size"] == 2048

def test_ocean_main_passes_zoom4(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            alpha_tif=".temp/alpha.tif",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif="ocean-z4.tif",
        )

    monkeypatch.setattr(sys, "argv", ["ocean.py", "--max-zoom", "4", "gebco.zip", "ocean-z4.tif"])
    monkeypatch.setattr("ocean.generate_ocean_background", fake_generate_ocean_background)

    ocean.main()

    assert called["max_zoom"] == 4

def test_build_ocean_output_plan_splits_aligned_chunks() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)
    plan = ocean.build_ocean_output_plan(bbox, max_zoom=13, chunk_size=1024)

    assert plan.width > 0
    assert plan.height > 0
    assert plan.total_pixels == plan.width * plan.height
    assert plan.chunk_size == 1024
    assert plan.halo_pixels >= 1
    assert plan.chunks

    first_chunk = plan.chunks[0]
    assert first_chunk.xoff == 0
    assert first_chunk.yoff == 0
    assert first_chunk.width <= plan.chunk_size
    assert first_chunk.height <= plan.chunk_size
    assert first_chunk.bounds[0] == pytest.approx(plan.bounds[0])
    assert first_chunk.bounds[3] == pytest.approx(plan.bounds[3])

def test_generate_ocean_without_bbox_processes_chunk_outputs(monkeypatch: object) -> None:
    plan_calls: list[tuple[tuple[float, float, float, float] | None, int, int]] = []
    processed_chunks: list[tuple[int, int]] = []
    merged_sources: list[str] = []
    translated: list[tuple[str, str]] = []

    plan = ocean.OceanBuildPlan(
        bounds=ocean.WEB_MERCATOR_WORLD_BOUNDS,
        pixel_size=satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM),
        zoom=ocean.DEFAULT_MAX_ZOOM,
        width=8,
        height=8,
        total_pixels=64,
        chunk_size=8,
        halo_pixels=2,
        chunks=(
            ocean.OceanChunkPlan(
                row=0,
                col=0,
                xoff=0,
                yoff=0,
                width=8,
                height=8,
                bounds=(0.0, 0.0, 8.0, 8.0),
                expanded_bounds=(0.0, 0.0, 8.0, 8.0),
                core_src_win=(0, 0, 8, 8),
            ),
        ),
    )

    monkeypatch.setattr("ocean.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("ocean.build_gebco_source_vrt", lambda gebco_zip, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.create_gebco_ocean_vrt", lambda source_vrt, output_vrt: output_vrt)
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox, *, max_zoom, chunk_size: (
            plan_calls.append((bbox, max_zoom, chunk_size)),
            plan,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.process_ocean_chunk",
        lambda **kwargs: processed_chunks.append((kwargs["chunk"].row, kwargs["chunk"].col))
        or f".temp/chunk_{kwargs['chunk'].row}_{kwargs['chunk'].col}.tif",
    )
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            merged_sources.extend(source_rasters),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            translated.append((rgba_vrt, destination)),
            destination,
        )[-1],
    )

    artifacts = ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        bbox=None,
    )

    assert plan_calls == [(None, ocean.DEFAULT_MAX_ZOOM, ocean.DEFAULT_OCEAN_CHUNK_SIZE)]
    assert processed_chunks == [(0, 0)]
    assert merged_sources == [".temp/chunk_0_0.tif"]
    assert translated == [(artifacts.rgba_vrt, "ocean.tif")]
    assert artifacts.masked_vrt.endswith("_masked.vrt")
    assert artifacts.warped_vrt.endswith("_depth_chunks.vrt")
    assert artifacts.alpha_tif.endswith("_alpha_chunks.vrt")
    assert artifacts.hillshade_tif.endswith("_hillshade_chunks.vrt")
    assert artifacts.color_tif.endswith("_color_chunks.vrt")
    assert artifacts.rgba_vrt.endswith("_rgba.vrt")

def test_generate_ocean_reuses_existing_chunk_outputs(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()
    plan = ocean.OceanBuildPlan(
        bounds=ocean.WEB_MERCATOR_WORLD_BOUNDS,
        pixel_size=satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM),
        zoom=ocean.DEFAULT_MAX_ZOOM,
        width=8,
        height=8,
        total_pixels=64,
        chunk_size=8,
        halo_pixels=2,
        chunks=(
            ocean.OceanChunkPlan(
                row=0,
                col=0,
                xoff=0,
                yoff=0,
                width=8,
                height=8,
                bounds=(0.0, 0.0, 8.0, 8.0),
                expanded_bounds=(0.0, 0.0, 8.0, 8.0),
                core_src_win=(0, 0, 8, 8),
            ),
        ),
    )
    unique_id = ocean.build_ocean_run_token(
        str(tmp_path / "ocean.tif"),
        None,
        max_zoom=ocean.DEFAULT_MAX_ZOOM,
        chunk_size=ocean.DEFAULT_OCEAN_CHUNK_SIZE,
        resample_alg="cubicspline",
        hillshade_z=5.0,
        style=ocean.OceanStyleOptions(),
    )
    existing_chunk = Path(
        ocean.build_ocean_chunk_artifacts(str(temp_dir), "ocean", unique_id, plan.chunks[0]).rgba_tif
    )
    existing_chunk.write_text("chunk")

    monkeypatch.setattr(
        "ocean.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: (Path(output_vrt).write_text("source"), output_vrt)[1],
    )
    monkeypatch.setattr(
        "ocean.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: (Path(output_vrt).write_text("masked"), output_vrt)[1],
    )
    monkeypatch.setattr("ocean.build_ocean_output_plan", lambda bbox, *, max_zoom, chunk_size: plan)
    monkeypatch.setattr(
        "ocean.process_ocean_chunk",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("chunk should have been reused")),
    )
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            Path(output_vrt).write_text("rgba"),
            output_vrt,
        )[-1],
    )
    translated: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            translated.append((rgba_vrt, destination)),
            Path(destination).write_text("final"),
            destination,
        )[-1],
    )

    artifacts = ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination=str(tmp_path / "ocean.tif"),
        temp_dir=str(temp_dir),
    )

    assert translated == [(artifacts.rgba_vrt, str(tmp_path / "ocean.tif"))]

def test_generate_ocean_without_bbox_reports_chunk_progress(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = ocean.OceanBuildPlan(
        bounds=ocean.WEB_MERCATOR_WORLD_BOUNDS,
        pixel_size=satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM),
        zoom=ocean.DEFAULT_MAX_ZOOM,
        width=16,
        height=8,
        total_pixels=128,
        chunk_size=8,
        halo_pixels=2,
        chunks=(
            ocean.OceanChunkPlan(
                row=0,
                col=0,
                xoff=0,
                yoff=0,
                width=8,
                height=8,
                bounds=(0.0, 0.0, 8.0, 8.0),
                expanded_bounds=(0.0, 0.0, 8.0, 8.0),
                core_src_win=(0, 0, 8, 8),
            ),
            ocean.OceanChunkPlan(
                row=0,
                col=1,
                xoff=8,
                yoff=0,
                width=8,
                height=8,
                bounds=(8.0, 0.0, 16.0, 8.0),
                expanded_bounds=(8.0, 0.0, 16.0, 8.0),
                core_src_win=(0, 0, 8, 8),
            ),
        ),
    )

    monkeypatch.setattr("ocean.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("ocean.build_gebco_source_vrt", lambda gebco_zip, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.create_gebco_ocean_vrt", lambda source_vrt, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.build_ocean_output_plan", lambda bbox, *, max_zoom, chunk_size: plan)
    monkeypatch.setattr(
        "ocean.process_ocean_chunk",
        lambda **kwargs: f".temp/chunk_{kwargs['chunk'].row}_{kwargs['chunk'].col}.tif",
    )
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            destination,
        )[-1],
    )

    ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        bbox=None,
        parallel=1,
    )

    out = capsys.readouterr().out
    assert "Ocean build: global run at z13 -> ocean.tif" in out
    assert "[3/6] Planning aligned Web Mercator chunks..." in out
    assert "Ocean target grid: 16x8 px (128 pixels)" in out
    assert "Processing 2 chunk(s) with 1 worker(s)..." in out
    assert "Ocean chunk progress: 2/2 (100%); Elapsed:" in out
    assert "[5/6] Building merged RGBA VRT... 100%; Elapsed:" in out
    assert "[6/6] Translating final RGBA GeoTIFF... 100%; Elapsed:" in out
    assert "Ocean build complete: ocean.tif" in out

def test_generate_ocean_with_bbox_builds_requested_plan(monkeypatch: object) -> None:
    plan_calls: list[tuple[tuple[float, float, float, float] | None, int, int]] = []
    bbox = (-4.0, 50.0, -3.0, 51.0)
    plan = ocean.OceanBuildPlan(
        bounds=(1.0, 2.0, 3.0, 4.0),
        pixel_size=19.0,
        zoom=13,
        width=1,
        height=1,
        total_pixels=1,
        chunk_size=512,
        halo_pixels=2,
        chunks=(
            ocean.OceanChunkPlan(
                row=0,
                col=0,
                xoff=0,
                yoff=0,
                width=1,
                height=1,
                bounds=(1.0, 2.0, 3.0, 4.0),
                expanded_bounds=(1.0, 2.0, 3.0, 4.0),
                core_src_win=(0, 0, 1, 1),
            ),
        ),
    )

    monkeypatch.setattr("ocean.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("ocean.build_gebco_source_vrt", lambda gebco_zip, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.create_gebco_ocean_vrt", lambda source_vrt, output_vrt: output_vrt)
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox_arg, *, max_zoom, chunk_size: (
            plan_calls.append((bbox_arg, max_zoom, chunk_size)),
            plan,
        )[-1],
    )
    monkeypatch.setattr("ocean.process_ocean_chunk", lambda **kwargs: ".temp/chunk_0_0.tif")
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            destination,
        )[-1],
    )

    ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        bbox=bbox,
        chunk_size=512,
    )

    assert plan_calls == [(bbox, ocean.DEFAULT_MAX_ZOOM, 512)]

def test_generate_ocean_uses_requested_zoom_in_plan(monkeypatch: object) -> None:
    plan_calls: list[tuple[tuple[float, float, float, float] | None, int, int]] = []
    plan = ocean.OceanBuildPlan(
        bounds=ocean.WEB_MERCATOR_WORLD_BOUNDS,
        pixel_size=satmaps.tiler.web_mercator_pixel_size(14),
        zoom=14,
        width=1,
        height=1,
        total_pixels=1,
        chunk_size=256,
        halo_pixels=2,
        chunks=(
            ocean.OceanChunkPlan(
                row=0,
                col=0,
                xoff=0,
                yoff=0,
                width=1,
                height=1,
                bounds=(0.0, 0.0, 1.0, 1.0),
                expanded_bounds=(0.0, 0.0, 1.0, 1.0),
                core_src_win=(0, 0, 1, 1),
            ),
        ),
    )

    monkeypatch.setattr("ocean.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("ocean.build_gebco_source_vrt", lambda gebco_zip, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.create_gebco_ocean_vrt", lambda source_vrt, output_vrt: output_vrt)
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox_arg, *, max_zoom, chunk_size: (
            plan_calls.append((bbox_arg, max_zoom, chunk_size)),
            plan,
        )[-1],
    )
    monkeypatch.setattr("ocean.process_ocean_chunk", lambda **kwargs: ".temp/chunk_0_0.tif")
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            destination,
        )[-1],
    )

    ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        max_zoom=14,
        chunk_size=256,
    )

    assert plan_calls == [(None, 14, 256)]

def test_build_source_raster_candidate_tile_relpaths_matches_real_warp(tmp_path: Path) -> None:
    source_path = tmp_path / "source_utm.tif"
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32604)
    dataset = gdal.GetDriverByName("GTiff").Create(str(source_path), 64, 64, 1, gdal.GDT_Byte)
    assert dataset is not None
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform((500000.0, 30.0, 0.0, 2300.0, 0.0, -30.0))
    dataset.GetRasterBand(1).WriteArray(np.full((64, 64), 255, dtype=np.uint8))
    dataset = None

    actual = satmaps.build_source_raster_candidate_tile_relpaths(
        str(source_path),
        13,
        tile_size=512,
        resample_alg="lanczos",
    )
    warped = gdal.Warp(
        "",
        str(source_path),
        options=satmaps.build_web_mercator_warp_options(
            None,
            "lanczos",
            13,
            512,
        ),
    )
    assert warped is not None
    expected = satmaps.build_relpaths_for_tile_bounds(tiler.get_dataset_bounds(warped), 13)

    assert actual == expected

def test_build_work_unit_candidate_tile_relpaths_from_sources_uses_actual_source_bounds(
    monkeypatch: object, tmp_path: Path
) -> None:
    work_unit = satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",))
    full_tile_geometry = satmaps.build_mgrs_tile_geometry("31TDF", satmaps.build_web_mercator_srs())
    assert full_tile_geometry is not None
    min_x, max_x, min_y, max_y = full_tile_geometry.GetEnvelope()
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0

    source_path = tmp_path / "source_3857_half.tif"
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset = gdal.GetDriverByName("GTiff").Create(str(source_path), 64, 64, 1, gdal.GDT_Byte)
    assert dataset is not None
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform((min_x, (mid_x - min_x) / 64.0, 0.0, mid_y, 0.0, -((mid_y - min_y) / 64.0)))
    dataset.GetRasterBand(1).WriteArray(np.full((64, 64), 255, dtype=np.uint8))
    dataset = None

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda mgrs_tile, date_paths, cache_dir, **kwargs: [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")],
    )
    monkeypatch.setattr(
        "satmaps.get_tile_band_path",
        lambda folder_name, date_path, band_id, cache_dir, download=False, quiet=False: str(source_path),
    )

    actual = set(
        satmaps.build_work_unit_candidate_tile_relpaths_from_sources(
            work_unit,
            ["2025/07/01"],
            ".cache",
            13,
        )
    )
    conservative = set(satmaps.build_work_unit_candidate_tile_relpaths(work_unit, 13))

    assert actual
    assert actual < conservative

def test_build_work_unit_candidate_tile_relpaths_from_sources_falls_back_on_inspection_error(
    monkeypatch: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work_unit = satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",))
    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda mgrs_tile, date_paths, cache_dir, **kwargs: [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")],
    )
    monkeypatch.setattr(
        "satmaps.get_tile_band_path",
        lambda folder_name, date_path, band_id, cache_dir, download=False, quiet=False: "broken.tif",
    )
    monkeypatch.setattr(
        "satmaps.gdal.Open",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    actual = satmaps.build_work_unit_candidate_tile_relpaths_from_sources(
        work_unit,
        ["2025/07/01"],
        ".cache",
        13,
    )

    assert actual == satmaps.build_work_unit_candidate_tile_relpaths(work_unit, 13)
    assert "Could not inspect source footprint for 31TDF_0_0" in capsys.readouterr().out

def test_resolve_work_unit_candidate_tile_relpaths_reuses_cached_subset_and_persists_missing(
    monkeypatch: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    work_units = (
        satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",)),
        satmaps.LandWorkUnit("31TDF_0_1", ("31TDF_0_1",)),
    )
    cache_path = satmaps.build_candidate_tile_cache_path("candidatecache")
    satmaps.write_candidate_tile_cache(
        cache_path,
        {"31TDF_0_0": ("13/1/1.webp",)},
    )
    seen_missing_units: list[str] = []

    def fake_precompute(
        missing_work_units,
        date_paths,
        cache_dir,
        zoom,
        *,
        tile_size=512,
        resample_alg="lanczos",
        parallel=1,
    ):
        del date_paths, cache_dir, zoom, tile_size, resample_alg, parallel
        seen_missing_units.extend(work_unit.unit_id for work_unit in missing_work_units)
        return {"31TDF_0_1": ("13/2/2.webp",)}

    monkeypatch.setattr(
        "satmaps.precompute_work_unit_candidate_tile_relpaths_from_sources",
        fake_precompute,
    )

    actual = satmaps.resolve_work_unit_candidate_tile_relpaths(
        work_units,
        ["2025/07/01"],
        ".cache",
        13,
        cache_path=cache_path,
    )

    assert seen_missing_units == ["31TDF_0_1"]
    assert actual == {
        "31TDF_0_0": ("13/1/1.webp",),
        "31TDF_0_1": ("13/2/2.webp",),
    }
    assert satmaps.read_candidate_tile_cache(cache_path) == actual
    out = capsys.readouterr().out
    assert f"Loaded candidate tile cache from {cache_path} with 1 sub-tile(s)." in out
    assert "Reusing cached candidate tile footprints for 1 sub-tile(s); computing 1 missing." in out


def test_precompute_work_unit_candidate_tile_relpaths_reports_progress(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    work_units = (
        satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",)),
        satmaps.LandWorkUnit("31TDF_0_1", ("31TDF_0_1",)),
    )
    monkeypatch.setattr(satmaps, "LAND_PROGRESS_HEARTBEAT_SECONDS", 0.01)

    def slow_build(*args: object, **kwargs: object) -> tuple[str, ...]:
        time.sleep(0.03)
        return ("13/1/1.webp", "13/1/2.webp")

    monkeypatch.setattr(
        "satmaps.build_work_unit_candidate_tile_relpaths_from_sources",
        slow_build,
    )

    actual = satmaps.precompute_work_unit_candidate_tile_relpaths_from_sources(
        work_units,
        ["2025/07/01"],
        ".cache",
        13,
        parallel=1,
    )

    assert actual == {
        "31TDF_0_0": ("13/1/1.webp", "13/1/2.webp"),
        "31TDF_0_1": ("13/1/1.webp", "13/1/2.webp"),
    }
    out = capsys.readouterr().out
    assert "Candidate footprint progress: 0/2 (0%);" in out
    assert "0 candidate tiles, 1 active." in out
    assert "Candidate footprint progress: 2/2 (100%); Elapsed:" in out
    assert "4 candidate tiles." in out

def test_generate_ocean_vrt_mode_skips_translate(monkeypatch: object) -> None:
    vrt_outputs: list[tuple[str, str]] = []

    monkeypatch.setattr("ocean.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("ocean.build_gebco_source_vrt", lambda gebco_zip, output_vrt: output_vrt)
    monkeypatch.setattr("ocean.create_gebco_ocean_vrt", lambda source_vrt, output_vrt: output_vrt)
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox, *, max_zoom, chunk_size: ocean.OceanBuildPlan(
            bounds=(0.0, 0.0, 1.0, 1.0),
            pixel_size=1.0,
            zoom=13,
            width=1,
            height=1,
            total_pixels=1,
            chunk_size=chunk_size,
            halo_pixels=1,
            chunks=(
                ocean.OceanChunkPlan(
                    row=0,
                    col=0,
                    xoff=0,
                    yoff=0,
                    width=1,
                    height=1,
                    bounds=(0.0, 0.0, 1.0, 1.0),
                    expanded_bounds=(0.0, 0.0, 1.0, 1.0),
                    core_src_win=(0, 0, 1, 1),
                ),
            ),
        ),
    )
    monkeypatch.setattr("ocean.process_ocean_chunk", lambda **kwargs: ".temp/chunk_0_0.tif")
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.write_rgba_vrt",
        lambda rgba_vrt, destination: vrt_outputs.append((rgba_vrt, destination)) or destination,
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (_ for _ in ()).throw(
            AssertionError("Translate should not be called")
        ),
    )

    artifacts = ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        vrt=True,
    )

    assert artifacts.output_tif == "ocean.vrt"
    assert vrt_outputs == [(artifacts.rgba_vrt, "ocean.vrt")]

def test_generate_ocean_deletes_heavy_non_output_tifs(tmp_path: Path, monkeypatch: object) -> None:
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()

    monkeypatch.setattr(
        "ocean.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: (Path(output_vrt).write_text("source"), output_vrt)[1],
    )
    monkeypatch.setattr(
        "ocean.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: (Path(output_vrt).write_text("masked"), output_vrt)[1],
    )
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox, *, max_zoom, chunk_size: ocean.OceanBuildPlan(
            bounds=(0.0, 0.0, 1.0, 1.0),
            pixel_size=1.0,
            zoom=13,
            width=1,
            height=1,
            total_pixels=1,
            chunk_size=chunk_size,
            halo_pixels=1,
            chunks=(
                ocean.OceanChunkPlan(
                    row=0,
                    col=0,
                    xoff=0,
                    yoff=0,
                    width=1,
                    height=1,
                    bounds=(0.0, 0.0, 1.0, 1.0),
                    expanded_bounds=(0.0, 0.0, 1.0, 1.0),
                    core_src_win=(0, 0, 1, 1),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ocean.process_ocean_chunk",
        lambda **kwargs: (
            Path(temp_dir / "chunk_0_0_rgba.tif").write_text("chunk"),
            str(temp_dir / "chunk_0_0_rgba.tif"),
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            Path(output_vrt).write_text("rgba"),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.translate_rgba_vrt",
        lambda rgba_vrt, destination, progress=None: (
            progress and progress(1.0),
            Path(destination).write_text("final"),
            destination,
        )[-1],
    )

    artifacts = ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination=str(tmp_path / "ocean.tif"),
        temp_dir=str(temp_dir),
    )

    assert Path(artifacts.output_tif).exists()
    assert not (temp_dir / "chunk_0_0_rgba.tif").exists()
    assert Path(artifacts.source_vrt).exists()
    assert Path(artifacts.masked_vrt).exists()
    assert Path(artifacts.rgba_vrt).exists()

def test_generate_ocean_vrt_mode_keeps_output_dependent_tifs(
    tmp_path: Path, monkeypatch: object
) -> None:
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()

    monkeypatch.setattr(
        "ocean.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: (Path(output_vrt).write_text("source"), output_vrt)[1],
    )
    monkeypatch.setattr(
        "ocean.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: (Path(output_vrt).write_text("masked"), output_vrt)[1],
    )
    monkeypatch.setattr(
        "ocean.build_ocean_output_plan",
        lambda bbox, *, max_zoom, chunk_size: ocean.OceanBuildPlan(
            bounds=(0.0, 0.0, 1.0, 1.0),
            pixel_size=1.0,
            zoom=13,
            width=1,
            height=1,
            total_pixels=1,
            chunk_size=chunk_size,
            halo_pixels=1,
            chunks=(
                ocean.OceanChunkPlan(
                    row=0,
                    col=0,
                    xoff=0,
                    yoff=0,
                    width=1,
                    height=1,
                    bounds=(0.0, 0.0, 1.0, 1.0),
                    expanded_bounds=(0.0, 0.0, 1.0, 1.0),
                    core_src_win=(0, 0, 1, 1),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "ocean.process_ocean_chunk",
        lambda **kwargs: (
            Path(temp_dir / "chunk_0_0_rgba.tif").write_text("chunk"),
            str(temp_dir / "chunk_0_0_rgba.tif"),
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.build_merged_vrt",
        lambda output_vrt, source_rasters, progress=None: (
            progress and progress(1.0),
            Path(output_vrt).write_text("rgba"),
            output_vrt,
        )[-1],
    )
    monkeypatch.setattr(
        "ocean.write_rgba_vrt",
        lambda rgba_vrt, destination: (Path(destination).write_text("final-vrt"), destination)[-1],
    )

    artifacts = ocean.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination=str(tmp_path / "ocean.tif"),
        temp_dir=str(temp_dir),
        vrt=True,
    )

    assert artifacts.output_tif.endswith(".vrt")
    assert Path(artifacts.output_tif).exists()
    assert Path(temp_dir / "chunk_0_0_rgba.tif").exists()
    assert Path(artifacts.source_vrt).exists()
    assert Path(artifacts.rgba_vrt).exists()

def test_build_ocean_ramp_colors_respects_style_flags() -> None:
    default_colors = ocean.build_ocean_ramp_colors(
        ocean.OceanStyleOptions()
    )
    ungraded_colors = ocean.build_ocean_ramp_colors(
        ocean.OceanStyleOptions(tonemap=False, grade=False, exposure=0.5)
    )

    assert default_colors.shape == (len(ocean.MAKO_RAMP), 3)
    np.testing.assert_allclose(
        ungraded_colors[0],
        (np.array(ocean.MAKO_RAMP[0][1:], dtype=np.float32) / 255.0) * 0.5,
    )
    assert np.all((default_colors >= 0.0) & (default_colors <= 1.0))
    assert np.all((ungraded_colors >= 0.0) & (ungraded_colors <= 1.0))

def test_build_land_run_token_changes_with_winter_flag() -> None:
    common_args = dict(
        output="render.pmtiles",
        max_zoom=13,
        resample_alg="lanczos",
        stats_min=0.0,
        stats_max=10000.0,
        tonemap=True,
        grade=True,
        exposure=1.0,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        ghb=None,
        gms=1.0,
        ghs=None,
    )

    summer_token = satmaps.build_land_run_token(
        argparse.Namespace(**common_args, winter=False),
        ["2025/07/01", "2025/01/01"],
        None,
        None,
    )
    winter_token = satmaps.build_land_run_token(
        argparse.Namespace(**common_args, winter=True),
        ["2025/07/01", "2025/01/01"],
        None,
        None,
    )

    assert summer_token != winter_token

def configure_main_defaults(
    monkeypatch: object,
    tmp_path: Path,
    argv: list[str],
    *,
    mgrs_bases: list[str],
    unique_id: str = "testrun",
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    monkeypatch.setattr(sys, "argv", ["satmaps.py", *argv])
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.build_land_run_token", lambda *args, **kwargs: unique_id)
    monkeypatch.setattr(
        "satmaps.build_candidate_tile_cache_token",
        lambda *args, **kwargs: unique_id,
    )
    monkeypatch.setattr(
        "satmaps.discover_mgrs_bases",
        lambda bbox, gebco_src, land_mgrs_list_path=None: mgrs_bases,
    )

def test_main_packages_webp_tiles(monkeypatch: object, tmp_path: Path) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--parallel", "1", "--date", "2025/07/01"],
        mgrs_bases=["31TDF"],
        unique_id="mainwebp",
    )
    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    render_calls: list[dict[str, object]] = []
    packaged: list[tuple[str, str, dict[str, object]]] = []

    def fake_convert(input_tile_tree: str, output_path: str, **kwargs: object) -> str:
        packaged.append((input_tile_tree, output_path, dict(kwargs)))
        return str(tmp_path / ".temp" / "output.mbtiles")

    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: ("13/1/2.webp",)
            for work_unit in work_units
        },
    )
    monkeypatch.setattr(
        "satmaps.render_land_output_tiles",
        lambda work_units, date_paths, args, unique_id, contributor_tile_candidates, **kwargs: render_calls.append(
            {
                "work_units": tuple(work_unit.unit_id for work_unit in work_units),
                "unique_id": unique_id,
                "contributor_tile_candidates": contributor_tile_candidates,
            }
        )
        or satmaps.LandOutputRenderStats(total_tiles=1, rendered_tiles=1, skipped_tiles=0),
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr("satmaps.convert_tile_tree_to_pmtiles", fake_convert)

    main()

    assert render_calls == [
        {
            "work_units": ("31TDF_0_0", "31TDF_0_1", "31TDF_1_0", "31TDF_1_1"),
            "unique_id": "mainwebp",
            "contributor_tile_candidates": {
                "31TDF_0_0": ("13/1/2.webp",),
                "31TDF_0_1": ("13/1/2.webp",),
                "31TDF_1_0": ("13/1/2.webp",),
                "31TDF_1_1": ("13/1/2.webp",),
            },
        }
    ]
    assert packaged == [
        (
            satmaps.build_final_tile_cache_dir("output.pmtiles", "mainwebp"),
            "output.pmtiles",
            {
                "resample_alg": "lanczos",
                "max_zoom": ocean.DEFAULT_MAX_ZOOM,
                "name": "Sentinel-2 Mosaic",
                "description": "Copernicus Sentinel data",
                "requested_bbox": None,
            },
        )
    ]
    assert list(tmp_path.glob(".temp/master_*.vrt")) == []

def test_main_reports_land_progress(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--parallel", "1", "--date", "2025/07/01"],
        mgrs_bases=["31TDF"],
    )
    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: (f"13/1/{index + 1}.webp",)
            for index, work_unit in enumerate(work_units)
        },
    )
    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "satmaps.render_final_output_tile",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    out = capsys.readouterr().out
    assert "Expanded 1 MGRS tiles into 4 sub-tiles across 1 date(s)." in out
    assert "Starting output-tile rendering for 4 sub-tile(s) with 1 worker(s)." in out
    assert "Land tile progress: 4/4 (100%); Elapsed:" in out
    assert "Rendered 4 land tile(s); reused 0 cached tile(s)." in out
    assert "Building master VRT" not in out

def test_render_land_output_tiles_emits_heartbeat_and_cached_counts(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(parallel=1, cache=".cache", output="output.pmtiles")
    work_units = [satmaps.LandWorkUnit(unit_id="31TDF_0_0", source_subtiles=("31TDF_0_0",))]
    contributor_tile_candidates = {"31TDF_0_0": ("13/1/1.webp",)}

    monkeypatch.setattr("satmaps.list_mosaic_folders_for_tile", lambda *args, **kwargs: [])
    monkeypatch.setattr(satmaps, "LAND_PROGRESS_HEARTBEAT_SECONDS", 0.01)

    def slow_render(*args: object, **kwargs: object) -> satmaps.LandTileRenderStatus:
        time.sleep(0.03)
        return satmaps.LandTileRenderStatus.CACHED

    monkeypatch.setattr("satmaps.render_final_output_tile", slow_render)

    stats = satmaps.render_land_output_tiles(
        work_units,
        ["2025/07/01"],
        args,
        "testrun",
        contributor_tile_candidates,
    )

    assert stats == satmaps.LandOutputRenderStats(
        total_tiles=1,
        rendered_tiles=0,
        skipped_tiles=1,
        cached_tiles=1,
        empty_tiles=0,
    )
    out = capsys.readouterr().out
    assert "0 rendered, 0 cached, 1 active." in out
    assert "0 rendered, 1 cached." in out

def test_render_land_output_tiles_submits_only_active_work(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(parallel=1, cache=".cache", output="output.pmtiles")
    work_units = [satmaps.LandWorkUnit(unit_id="31TDF_0_0", source_subtiles=("31TDF_0_0",))]
    contributor_tile_candidates = {
        "31TDF_0_0": ("13/1/1.webp", "13/1/2.webp", "13/1/3.webp")
    }
    monkeypatch.setattr("satmaps.list_mosaic_folders_for_tile", lambda *args, **kwargs: [])

    outstanding_submissions = 0
    max_outstanding_submissions = 0

    class FakeFuture:
        def __init__(self, status: satmaps.LandTileRenderStatus) -> None:
            self._status = status

        def result(self) -> satmaps.LandTileRenderStatus:
            return self._status

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 1

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def submit(self, fn: object, *args: object, **kwargs: object) -> FakeFuture:
            nonlocal outstanding_submissions, max_outstanding_submissions
            outstanding_submissions += 1
            max_outstanding_submissions = max(
                max_outstanding_submissions,
                outstanding_submissions,
            )
            return FakeFuture(satmaps.LandTileRenderStatus.RENDERED)

    def fake_wait(
        pending_futures: set[FakeFuture], timeout: float, return_when: object
    ) -> tuple[set[FakeFuture], set[FakeFuture]]:
        nonlocal outstanding_submissions
        completed = next(iter(pending_futures))
        outstanding_submissions -= 1
        return {completed}, {future for future in pending_futures if future is not completed}

    monkeypatch.setattr(satmaps, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(satmaps, "wait", fake_wait)

    stats = satmaps.render_land_output_tiles(
        work_units,
        ["2025/07/01"],
        args,
        "testrun",
        contributor_tile_candidates,
    )

    assert stats == satmaps.LandOutputRenderStats(
        total_tiles=3,
        rendered_tiles=3,
        skipped_tiles=0,
        cached_tiles=0,
        empty_tiles=0,
    )
    assert max_outstanding_submissions == 1

def test_main_low_zoom_uses_subtile_processing_strategy(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--parallel", "1", "--date", "2025/07/01", "--max-zoom", "4"],
        mgrs_bases=["31TDF"],
    )
    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: (f"4/1/{index + 1}.webp",)
            for index, work_unit in enumerate(work_units)
        },
    )
    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "satmaps.render_final_output_tile",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["4/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    out = capsys.readouterr().out
    assert "Expanded 1 MGRS tiles into 4 sub-tiles across 1 date(s)." in out
    assert "Starting output-tile rendering for 4 sub-tile(s) with 1 worker(s)." in out
    assert "Land tile progress: 4/4 (100%); Elapsed:" in out
    assert "Rendered 4 land tile(s); reused 0 cached tile(s)." in out

def test_main_passes_ocean_path_to_tile_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    custom_ocean_path = tmp_path / "custom-ocean.tif"
    custom_ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(custom_ocean_path), 1, 1, 4, gdal.GDT_Byte
    )
    assert custom_ocean_ds is not None
    custom_ocean_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    custom_ocean_ds.GetRasterBand(4).Fill(255)
    custom_ocean_ds = None
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        [
            "--parallel",
            "1",
            "--date",
            "2025/07/01",
            "--ocean-background",
            str(custom_ocean_path),
        ],
        mgrs_bases=["31TDF"],
    )
    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    gebco_sources: list[str | None] = []
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: ("13/1/2.webp",)
            for work_unit in work_units
        },
    )
    monkeypatch.setattr(
        "satmaps.render_land_output_tiles",
        lambda work_units, date_paths, args, unique_id, contributor_tile_candidates, **kwargs: gebco_sources.append(
            kwargs.get("gebco_src")
        )
        or satmaps.LandOutputRenderStats(total_tiles=1, rendered_tiles=1, skipped_tiles=0),
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    assert gebco_sources
    assert set(gebco_sources) == {str(custom_ocean_path)}

def test_main_passes_winter_flag_to_tile_processing(monkeypatch: object, tmp_path: Path) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--parallel", "1", "--winter"],
        mgrs_bases=["31TDF"],
    )
    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    winter_flags: list[bool] = []
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: ("13/1/2.webp",)
            for work_unit in work_units
        },
    )
    monkeypatch.setattr(
        "satmaps.render_land_output_tiles",
        lambda work_units, date_paths, args, unique_id, contributor_tile_candidates, **kwargs: winter_flags.append(
            args.winter
        )
        or satmaps.LandOutputRenderStats(total_tiles=1, rendered_tiles=1, skipped_tiles=0),
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    assert winter_flags
    assert set(winter_flags) == {True}

def test_main_bbox_prepares_and_commits_ocean_background(
    monkeypatch: object, tmp_path: Path
) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--bbox", "0,0,1,1", "--no-land", "--parallel", "1"],
        mgrs_bases=[],
        unique_id="bboxrun",
    )
    prepare_calls: list[tuple[object, ...]] = []
    commit_calls: list[tuple[object, ...]] = []
    package_calls: list[dict[str, object]] = []

    def fake_prepare(*args):
        prepare_calls.append(args)
        return ".temp/output_ocean_bbox.tif"

    def fake_commit(*args):
        commit_calls.append(args)
        return True

    def fake_convert(*args, **kwargs):
        package_calls.append(dict(kwargs))
        return str(tmp_path / ".temp" / "output.mbtiles")

    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", fake_prepare)
    monkeypatch.setattr("satmaps.commit_ocean_to_final_tile_cache", fake_commit)
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr("satmaps.convert_tile_tree_to_pmtiles", fake_convert)

    main()

    assert prepare_calls == [
        ("ocean.tif", (0.0, 0.0, 1.0, 1.0), "output.pmtiles", "lanczos", ocean.DEFAULT_MAX_ZOOM, 512)
    ]
    assert len(commit_calls) == 1
    assert commit_calls[0][:3] == (".temp/output_ocean_bbox.tif", "output.pmtiles", "bboxrun")
    assert isinstance(commit_calls[0][3], argparse.Namespace)
    assert package_calls == [
        {
            "resample_alg": "lanczos",
            "max_zoom": ocean.DEFAULT_MAX_ZOOM,
            "name": "Sentinel-2 Mosaic",
            "description": "Copernicus Sentinel data",
            "requested_bbox": (0.0, 0.0, 1.0, 1.0),
        }
    ]

def test_main_land_run_passes_prepared_ocean_to_output_tile_renderer_without_eager_commit(
    monkeypatch: object, tmp_path: Path
) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--bbox", "0,0,1,1", "--parallel", "1", "--date", "2025/07/01"],
        mgrs_bases=["31TDF"],
        unique_id="lazyocean",
    )
    render_calls: list[dict[str, object]] = []
    backfill_calls: list[tuple[str, str, str]] = []
    package_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.prepare_ocean_background_for_output",
        lambda *args, **kwargs: ".temp/output_ocean_bbox.tif",
    )
    monkeypatch.setattr(
        "satmaps.commit_ocean_to_final_tile_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("land runs should not eagerly commit ocean tiles")
        ),
    )
    monkeypatch.setattr(
        "satmaps.render_land_output_tiles",
        lambda work_units, date_paths, args, unique_id, contributor_tile_candidates, **kwargs: render_calls.append(
            {
                "unique_id": unique_id,
                "work_unit_count": len(work_units),
                "args_output": args.output,
                "quality": args.quality,
                "zoom": args.max_zoom,
                "date_paths": tuple(date_paths),
                "contributor_tile_candidates": contributor_tile_candidates,
                **kwargs,
            }
        )
        or satmaps.LandOutputRenderStats(total_tiles=4, rendered_tiles=4, skipped_tiles=0),
    )
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda work_units, *args, **kwargs: {
            work_unit.unit_id: ("13/1/2.webp",)
            for work_unit in work_units
        },
    )
    monkeypatch.setattr(
        "satmaps.fill_missing_ocean_to_final_tile_cache",
        lambda input_raster, output_path, unique_id, args: backfill_calls.append(
            (input_raster, output_path, unique_id)
        )
        or 1,
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: package_calls.append(dict(kwargs))
        or str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    assert len(render_calls) == 1
    assert render_calls[0]["args_output"] == "output.pmtiles"
    assert render_calls[0]["unique_id"] == "lazyocean"
    assert render_calls[0]["work_unit_count"] == 4
    assert render_calls[0]["quality"] == 74
    assert render_calls[0]["zoom"] == ocean.DEFAULT_MAX_ZOOM
    assert render_calls[0]["prepared_ocean_background"] == ".temp/output_ocean_bbox.tif"
    contributor_tile_candidates = render_calls[0]["contributor_tile_candidates"]
    assert isinstance(contributor_tile_candidates, dict)
    assert set(contributor_tile_candidates) == {
        "31TDF_0_0",
        "31TDF_0_1",
        "31TDF_1_0",
        "31TDF_1_1",
    }
    assert all(candidate_relpaths for candidate_relpaths in contributor_tile_candidates.values())
    assert backfill_calls == [
        (".temp/output_ocean_bbox.tif", "output.pmtiles", "lazyocean")
    ]
    assert package_calls == [
        {
            "resample_alg": "lanczos",
            "max_zoom": ocean.DEFAULT_MAX_ZOOM,
            "name": "Sentinel-2 Mosaic",
            "description": "Copernicus Sentinel data",
            "requested_bbox": (0.0, 0.0, 1.0, 1.0),
        }
    ]

def test_main_reuses_cached_candidate_tile_footprints(
    monkeypatch: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--parallel", "1", "--date", "2025/07/01"],
        mgrs_bases=["31TDF"],
        unique_id="cachedcandidates",
    )
    monkeypatch.setattr(
        "satmaps.build_candidate_tile_cache_token",
        lambda *args, **kwargs: "sharedcandidates",
    )
    cached_candidates = {
        "31TDF_0_0": ("13/1/1.webp",),
        "31TDF_0_1": ("13/1/2.webp",),
        "31TDF_1_0": ("13/1/3.webp",),
        "31TDF_1_1": ("13/1/4.webp",),
    }
    satmaps.write_candidate_tile_cache(
        satmaps.build_candidate_tile_cache_path("cachedcandidates"),
        cached_candidates,
    )
    render_calls: list[dict[str, object]] = []

    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "satmaps.precompute_work_unit_candidate_tile_relpaths_from_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expected candidate footprints to come from the cache")
        ),
    )
    monkeypatch.setattr(
        "satmaps.render_land_output_tiles",
        lambda work_units, date_paths, args, unique_id, contributor_tile_candidates, **kwargs: render_calls.append(
            {
                "unique_id": unique_id,
                "work_unit_count": len(work_units),
                "contributor_tile_candidates": contributor_tile_candidates,
            }
        )
        or satmaps.LandOutputRenderStats(total_tiles=4, rendered_tiles=4, skipped_tiles=0),
    )
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    assert len(render_calls) == 1
    assert render_calls[0]["contributor_tile_candidates"] == cached_candidates
    out = capsys.readouterr().out
    assert "Loaded candidate tile cache from .temp/candidate_tiles_cachedcandidates.json with 4 sub-tile(s)." in out
    assert "Reusing cached candidate tile footprints for 4 sub-tile(s)." in out
    assert satmaps.read_candidate_tile_cache(
        satmaps.build_candidate_tile_cache_path("sharedcandidates")
    ) == cached_candidates

@pytest.mark.parametrize("max_zoom", [ocean.DEFAULT_MAX_ZOOM, 14, 11, 12, 4])
def test_main_passes_requested_zoom_to_webp_pipeline(
    monkeypatch: object, tmp_path: Path, max_zoom: int
) -> None:
    argv = ["--no-land", "--parallel", "1"]
    if max_zoom != ocean.DEFAULT_MAX_ZOOM:
        argv.extend(["--max-zoom", str(max_zoom)])
    configure_main_defaults(monkeypatch, tmp_path, argv, mgrs_bases=[])
    ocean_zooms: list[int] = []
    package_zooms: list[int] = []

    def fake_prepare(
        ocean_background: str,
        requested_bbox: tuple[float, float, float, float] | None,
        output_path: str,
        resample_alg: str,
        requested_zoom: int,
        blocksize: int,
    ) -> str:
        ocean_zooms.append(requested_zoom)
        return "ocean.tif"

    def fake_convert(*args, **kwargs):
        package_zooms.append(kwargs["max_zoom"])
        return str(tmp_path / ".temp" / "output.mbtiles")

    monkeypatch.setattr("satmaps.prepare_ocean_background_for_output", fake_prepare)
    monkeypatch.setattr("satmaps.commit_ocean_to_final_tile_cache", lambda *args, **kwargs: False)
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr("satmaps.convert_tile_tree_to_pmtiles", fake_convert)

    main()

    assert ocean_zooms == [max_zoom]
    assert package_zooms == [max_zoom]

def test_fill_missing_ocean_to_final_tile_cache_writes_only_missing_tiles(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    final_tile_tree = Path(satmaps.build_final_tile_cache_dir("output.pmtiles", "oceanfill"))
    existing_tile = final_tile_tree / "13/1/2.webp"
    tiler.save_webp_image(Image.new("RGB", (8, 8), (1, 2, 3)), str(existing_tile), quality=100)
    original_bytes = existing_tile.read_bytes()

    args = argparse.Namespace(max_zoom=13, blocksize=512, resample_alg="lanczos", quality=74)
    fake_dataset = object()
    seen_dataset: list[object] = []
    rendered_bounds: list[tuple[float, float, float, float]] = []

    def fake_iter_dataset_tile_relpaths(dataset, zoom):
        seen_dataset.append(dataset)
        assert zoom == 13
        return iter(
            [
                "13/1/2.webp",
                "13/1/3.webp",
            ]
        )

    monkeypatch.setattr("satmaps.gdal.Open", lambda path: fake_dataset)
    monkeypatch.setattr("satmaps.tiler.iter_dataset_tile_relpaths", fake_iter_dataset_tile_relpaths)

    def fake_render_dataset_tile(dataset, bounds, tile_size, resample_alg):
        assert dataset is fake_dataset
        assert tile_size == 512
        assert resample_alg == "lanczos"
        rendered_bounds.append(bounds)
        return np.full((3, 8, 8), 60, dtype=np.uint8)

    monkeypatch.setattr("satmaps.tiler.render_dataset_tile", fake_render_dataset_tile)

    written_tiles = satmaps.fill_missing_ocean_to_final_tile_cache(
        "ocean.tif",
        "output.pmtiles",
        "oceanfill",
        args,
    )

    assert written_tiles == 1
    assert seen_dataset == [fake_dataset]
    assert rendered_bounds == [tiler.get_web_mercator_bounds(13, 1, 3)]
    assert existing_tile.read_bytes() == original_bytes
    new_tile_path = final_tile_tree / "13/1/3.webp"
    assert new_tile_path.exists()
    with Image.open(new_tile_path) as new_image:
        assert new_image.size == (8, 8)

def test_main_keeps_ocean_after_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")
    configure_main_defaults(
        monkeypatch,
        tmp_path,
        ["--no-land", "--parallel", "1"],
        mgrs_bases=[],
    )
    monkeypatch.setattr(
        "satmaps.prepare_ocean_background_for_output",
        lambda *args, **kwargs: str(ocean_path),
    )
    monkeypatch.setattr("satmaps.commit_ocean_to_final_tile_cache", lambda *args, **kwargs: False)
    monkeypatch.setattr("satmaps.tiler.iter_tile_tree_paths", lambda root: ["13/1/2.webp"])
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    main()

    assert ocean_path.exists()

def test_commit_raster_to_final_tile_cache_streams_tile_images(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    args = argparse.Namespace(
        max_zoom=13,
        blocksize=512,
        resample_alg="bilinear",
        quality=75,
    )
    fake_dataset = object()

    monkeypatch.setattr(
        "satmaps.tiler.render_raster_to_webp_tile_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expected streamed tile rendering, not eager dict rendering")
        ),
    )

    def fake_iter_dataset_webp_tile_images(dataset, zoom, tile_size, resample_alg):
        assert dataset is fake_dataset
        assert zoom == 13
        assert tile_size == 512
        assert resample_alg == "bilinear"
        return iter(
            [
                ("13/1/3.webp", Image.new("RGB", (8, 8), (40, 50, 60))),
                ("13/1/2.webp", Image.new("RGB", (8, 8), (10, 20, 30))),
            ]
        )

    monkeypatch.setattr("satmaps.tiler.iter_dataset_webp_tile_images", fake_iter_dataset_webp_tile_images)

    relpaths = satmaps.commit_raster_to_final_tile_cache(
        fake_dataset,
        "output.pmtiles",
        "streamraster",
        "31TDF_0_0",
        args,
    )

    assert relpaths == ["13/1/2.webp", "13/1/3.webp"]
    marker_path = Path(
        satmaps.build_contributor_complete_marker("output.pmtiles", "streamraster", "31TDF_0_0")
    )
    assert marker_path.exists()
    assert satmaps.read_tile_cache_marker(str(marker_path)) == ("31TDF_0_0", relpaths)
    for relative_path in relpaths:
        assert (
            Path(satmaps.build_final_tile_cache_dir("output.pmtiles", "streamraster")) / relative_path
        ).exists()

def test_render_final_output_tile_composites_ocean_and_ordered_contributors(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    args = argparse.Namespace(
        output="output.pmtiles",
        blocksize=8,
        resample_alg="lanczos",
        quality=100,
    )
    ocean_image = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    first_contributor = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    second_contributor = Image.new("RGBA", (8, 8), (0, 255, 0, 128))
    work_units_by_id = {
        "31TDF_0_0": satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",)),
        "31TDF_0_1": satmaps.LandWorkUnit("31TDF_0_1", ("31TDF_0_1",)),
    }

    monkeypatch.setattr(
        "satmaps.render_raster_tile_image",
        lambda *args, **kwargs: ocean_image.copy(),
    )

    def fake_render_land_contributor_output_tile(
        work_unit: satmaps.LandWorkUnit,
        folders: list[tuple[str, str]],
        tile_plan: satmaps.LandOutputTilePlan,
        args: argparse.Namespace,
        gebco_src: str | None = None,
    ) -> Image.Image:
        assert tile_plan.tile_size == 8
        if work_unit.unit_id == "31TDF_0_0":
            return first_contributor.copy()
        return second_contributor.copy()

    monkeypatch.setattr(
        "satmaps.render_land_contributor_output_tile",
        fake_render_land_contributor_output_tile,
    )

    wrote_tile = satmaps.render_final_output_tile(
        "13/1/2.webp",
        ("31TDF_0_0", "31TDF_0_1"),
        work_units_by_id,
        {"31TDF_0_0": [], "31TDF_0_1": []},
        args,
        "tilecompose",
        prepared_ocean_background="ocean.tif",
    )

    assert wrote_tile
    final_tile_path = Path(satmaps.build_final_tile_cache_dir("output.pmtiles", "tilecompose")) / "13/1/2.webp"
    assert final_tile_path.exists()

    expected = Image.alpha_composite(ocean_image, first_contributor)
    expected = Image.alpha_composite(expected, second_contributor)
    with Image.open(final_tile_path) as final_tile:
        final_pixel = final_tile.convert("RGBA").getpixel((0, 0))
    expected_pixel = expected.getpixel((0, 0))
    assert all(abs(actual - expected_channel) <= 10 for actual, expected_channel in zip(final_pixel, expected_pixel))

def test_render_final_output_tile_skips_existing_tile(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    args = argparse.Namespace(
        output="output.pmtiles",
        blocksize=8,
        resample_alg="lanczos",
        quality=74,
    )
    final_tile_path = Path(satmaps.build_final_tile_cache_dir("output.pmtiles", "tileskip")) / "13/1/2.webp"
    tiler.save_webp_image(Image.new("RGB", (8, 8), (10, 20, 30)), str(final_tile_path), quality=100)

    monkeypatch.setattr(
        "satmaps.render_land_contributor_output_tile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("existing tiles should skip rendering")),
    )

    render_status = satmaps.render_final_output_tile(
        "13/1/2.webp",
        ("31TDF_0_0",),
        {"31TDF_0_0": satmaps.LandWorkUnit("31TDF_0_0", ("31TDF_0_0",))},
        {"31TDF_0_0": []},
        args,
        "tileskip",
    )

    assert render_status == satmaps.LandTileRenderStatus.CACHED

def test_envelopes_overlap_rejects_touching_edges() -> None:
    assert not land_mgrs.envelopes_overlap(
        (0.0, 10.0, 0.0, 10.0),
        (10.0, 20.0, 0.0, 10.0),
    )
    assert land_mgrs.envelopes_overlap(
        (0.0, 10.0, 0.0, 10.0),
        (9.999, 20.0, 0.0, 10.0),
    )

def test_main_webp_resume_reuses_existing_final_tiles_without_latest_state_fallback(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--resume", "--parallel", "1", "--date", "2025/07/01"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.resolve_ocean_mask_source", lambda path: None)
    monkeypatch.setattr("satmaps.resolve_land_mgrs_source", lambda: None)
    monkeypatch.setattr(
        "satmaps.discover_mgrs_bases",
        lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"],
    )
    monkeypatch.setattr(
        "satmaps.prepare_ocean_background_for_output",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("satmaps.build_land_run_token", lambda *args, **kwargs: "webpresume")
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    work_units = satmaps.plan_subtile_work_units(["31TDF"])
    contributor_tile_candidates = {
        work_unit.unit_id: (f"13/1/{index + 1}.webp",)
        for index, work_unit in enumerate(work_units)
    }
    monkeypatch.setattr(
        "satmaps.resolve_work_unit_candidate_tile_relpaths",
        lambda *args, **kwargs: contributor_tile_candidates,
    )
    monkeypatch.setattr("satmaps.list_mosaic_folders_for_tile", lambda *args, **kwargs: [])
    for tile_relpaths in contributor_tile_candidates.values():
        final_tile = (
            Path(satmaps.build_final_tile_cache_dir("output.pmtiles", "webpresume"))
            / tile_relpaths[0]
        )
        tiler.save_webp_image(Image.new("RGB", (8, 8), (0, 0, 255)), str(final_tile), quality=100)

    stale_state = tmp_path / ".temp" / "state_stale.json"
    stale_state.write_text(
        '{"unique_id": "stale", "completed_units": ["stale"], "processed_tifs": [], "args": {}}'
    )

    main()

    out = capsys.readouterr().out
    assert "All land output tiles already rendered (reused from cache)." in out

def test_main_refresh_land_mgrs_list_force_regenerates_and_exits(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "satmaps.py",
            "--refresh-land-mgrs-list",
            "--bbox",
            "-158.0,20.8,-157.0,21.7",
        ],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.resolve_land_mgrs_source", lambda: ocean.DEFAULT_GEBCO_ZIP)
    monkeypatch.setattr(
        "land_mgrs.generate_land_mgrs_list",
        lambda gebco_zip, destination, bbox=None, force_refresh=False: satmaps.save_land_mgrs_list(
            destination,
            {"05QFJ"},
            bbox=bbox,
            ocean_mask_source=gebco_zip,
        )
        or destination,
    )
    monkeypatch.setattr(
        "satmaps.prepare_ocean_background_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expected --refresh-land-mgrs-list to exit before rendering")
        ),
    )

    main()

    assert satmaps.load_saved_land_mgrs_list(
        satmaps.build_land_mgrs_list_path(),
        bbox=(-158.0, 20.8, -157.0, 21.7),
        ocean_mask_source=ocean.DEFAULT_GEBCO_ZIP,
    ) == {"05QFJ"}
    assert list(tmp_path.glob(".temp/master_*.vrt")) == []

def test_main_refresh_land_mgrs_list_requires_mask_source(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--refresh-land-mgrs-list"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.resolve_land_mgrs_source", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)

    with pytest.raises(SystemExit, match="1"):
        main()

    assert (
        f"Error: --refresh-land-mgrs-list requires {ocean.DEFAULT_GEBCO_ZIP} in the current directory."
        in capsys.readouterr().out
    )
