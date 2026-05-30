import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocean
import satmaps
import tiler
from satmaps import (
    fill_nan_nearest,
    get_tile_paths,
    list_mosaic_folders_for_tile,
    main,
    process_single_tile,
)


def test_list_mosaic_folders_for_tile_uses_cache(monkeypatch: object) -> None:
    # Pre-populate cache
    satmaps.S3_FOLDER_CACHE = {"2025/07/01": {"Sentinel-2_mosaic_2025_Q3_31TDF_0_0"}}

    found = list_mosaic_folders_for_tile("31TDF_0_0", ["2025/07/01"], ".cache")
    assert found == [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")]


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
            gamma=1.2,
        ),
        color_bands,
        alpha_band,
    )

    rgba = dataset.ReadAsArray()
    np.testing.assert_array_equal(rgba[:, 0, 0], np.array([255, 127, 63, 255], dtype=np.uint8))
    np.testing.assert_array_equal(rgba[:, 0, 1], np.array([255, 127, 63, 255], dtype=np.uint8))


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


def test_open_gebco_mask_warps_alpha_band_directly(monkeypatch: object) -> None:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    gebco_ds = gdal.GetDriverByName("MEM").Create("", 4, 3, 4, gdal.GDT_Byte)
    assert gebco_ds is not None
    gebco_ds.SetGeoTransform((0.0, 100.0, 0.0, 300.0, 0.0, -100.0))
    gebco_ds.SetProjection(srs.ExportToWkt())
    gebco_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    warp_options_calls: list[dict[str, object]] = []
    warped_mask_ds = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
    assert warped_mask_ds is not None

    monkeypatch.setattr("satmaps.gdal.Open", lambda path: gebco_ds)
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
        geotransform=(0.0, 100.0, 0.0, 300.0, 0.0, -100.0),
        width=4,
        height=3,
    )
    mask = satmaps.open_gebco_mask("gebco.tif", tile_grid, "31TDF_0_0")

    assert mask is not None
    assert warp_options_calls[0]["srcBands"] == [4]
    assert warp_options_calls[0]["srcAlpha"] is False
    assert warp_options_calls[0]["multithread"] is True
    assert warp_options_calls[0]["warpOptions"] == ["NUM_THREADS=ALL_CPUS"]


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


def test_warp_to_web_mercator_uses_shared_zoom13_resolution(monkeypatch: object) -> None:
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr("satmaps.gdal.Warp", lambda destination, source, options=None: None)

    satmaps.warp_to_web_mercator("input.tif", "output.tif", "lanczos", 13, 256)

    assert warp_options_calls[0]["dstSRS"] == "EPSG:3857"
    assert warp_options_calls[0]["xRes"] == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM)
    )
    assert warp_options_calls[0]["yRes"] == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(ocean.DEFAULT_MAX_ZOOM)
    )
    assert warp_options_calls[0]["targetAlignedPixels"] is True
    assert warp_options_calls[0]["multithread"] is True
    assert warp_options_calls[0]["warpOptions"] == ["NUM_THREADS=ALL_CPUS"]


def test_warp_to_web_mercator_respects_requested_zoom(monkeypatch: object) -> None:
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr("satmaps.gdal.Warp", lambda destination, source, options=None: None)

    satmaps.warp_to_web_mercator("input.tif", "output.tif", "lanczos", 14, 256)

    assert warp_options_calls[0]["xRes"] == pytest.approx(satmaps.tiler.web_mercator_pixel_size(14))
    assert warp_options_calls[0]["yRes"] == pytest.approx(satmaps.tiler.web_mercator_pixel_size(14))
    assert warp_options_calls[0]["targetAlignedPixels"] is True
    assert warp_options_calls[0]["multithread"] is True
    assert warp_options_calls[0]["warpOptions"] == ["NUM_THREADS=ALL_CPUS"]


def test_warp_to_web_mercator_respects_blocksize(monkeypatch: object) -> None:
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr("satmaps.gdal.Warp", lambda destination, source, options=None: None)

    satmaps.warp_to_web_mercator("input.tif", "output.tif", "lanczos", 13, 512)

    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(13, 512)
    assert warp_options_calls[0]["xRes"] == pytest.approx(expected)
    assert warp_options_calls[0]["yRes"] == pytest.approx(expected)


def test_warp_to_web_mercator_aligns_adjacent_tiles_to_shared_pixel_grid(
    tmp_path: Path,
) -> None:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32604)
    pixel_size = satmaps.tiler.web_mercator_pixel_size(13)

    warped_datasets = []
    for index, x_origin in enumerate((500000.0, 501920.0), start=1):
        source_path = tmp_path / f"tile_{index}.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(str(source_path), 64, 64, 4, gdal.GDT_Byte)
        assert dataset is not None
        dataset.SetProjection(srs.ExportToWkt())
        dataset.SetGeoTransform((x_origin, 30.0, 0.0, 2300.0, 0.0, -30.0))
        for band_index, value in enumerate((40 * index, 80, 120), start=1):
            dataset.GetRasterBand(band_index).WriteArray(np.full((64, 64), value, dtype=np.uint8))
        dataset.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        dataset.GetRasterBand(4).WriteArray(np.full((64, 64), 255, dtype=np.uint8))
        dataset = None

        warped_path = tmp_path / f"tile_{index}_3857.tif"
        satmaps.warp_to_web_mercator(str(source_path), str(warped_path), "lanczos", 13, 256)
        warped_dataset = gdal.Open(str(warped_path))
        assert warped_dataset is not None
        warped_datasets.append(warped_dataset)

    gt_1 = warped_datasets[0].GetGeoTransform()
    gt_2 = warped_datasets[1].GetGeoTransform()

    for value in (gt_1[0], gt_1[3], gt_2[0], gt_2[3]):
        assert (value / pixel_size) == pytest.approx(round(value / pixel_size), abs=1e-6)
    assert gt_1[3] == pytest.approx(gt_2[3], abs=1e-6)


def test_warp_to_web_mercator_removes_existing_destination(monkeypatch: object) -> None:
    removed: list[str] = []

    monkeypatch.setattr("satmaps.remove_if_exists", lambda path: removed.append(path))
    monkeypatch.setattr("satmaps.gdal.WarpOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr("satmaps.gdal.Warp", lambda destination, source, options=None: None)

    satmaps.warp_to_web_mercator("input.tif", "output.tif", "lanczos", 14, 256)

    assert removed == ["output.tif"]


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


def test_process_single_tile_full_pipeline(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    # Mock data discovery
    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda tile, dates, cache: [
            ("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")
        ],
    )

    # Create fake input TIFs
    def fake_get_tile_paths(folder, date, cache, download=False, quiet=False):
        p = {}
        for b in ["red", "green", "blue"]:
            path = tmp_path / f"{b}.tif"
            if not path.exists():
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(path), 10, 10, 1, gdal.GDT_Int16)
                ds.SetGeoTransform((0, 1, 0, 10, 0, -1))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(32631)  # UTM zone 31N
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).Fill(1000)
                ds = None
            p[b] = str(path)
        return p

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)

    args = argparse.Namespace(
        cache=".cache",
        download=False,
        stats_min=0,
        stats_max=10000,
        tonemap=True,
        grade=True,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        exposure=1.0,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        resample_alg="bilinear",
        max_zoom=13,
        blocksize=256,
        output="render.pmtiles",
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, None)

    assert out_path is not None
    assert Path(out_path).exists()

    ds = gdal.Open(out_path)
    assert ds.RasterCount == 4
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    assert ds.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_AlphaBand
    # Check projection is 3857
    srs = osr.SpatialReference(ds.GetProjection())
    assert srs.GetAttrValue("AUTHORITY", 1) == "3857"
    out = capsys.readouterr().out
    assert "Processing tile 31TDF_0_0" not in out
    assert "Finished tile 31TDF_0_0" not in out


def test_process_single_tile_with_gebco_mask(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    raster_size = 100
    pixel_size = 100.0

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda tile, dates, cache: [
            ("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")
        ],
    )

    def fake_get_tile_paths(folder, date, cache, download=False, quiet=False):
        p = {}
        for b in ["red", "green", "blue"]:
            path = tmp_path / f"{b}.tif"
            if not path.exists():
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
                ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(32631)
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).Fill(1000)
                ds = None
            p[b] = str(path)
        return p

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)

    gebco_path = tmp_path / "gebco.tif"
    gebco_ds = gdal.GetDriverByName("GTiff").Create(
        str(gebco_path), raster_size, raster_size, 4, gdal.GDT_Byte
    )
    gebco_ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    gebco_ds.SetProjection(srs.ExportToWkt())
    gebco_values = np.full((raster_size, raster_size), 255, dtype=np.uint8)
    gebco_values[0, 0] = 0
    gebco_values[0, 1] = 0
    gebco_alpha_band = gebco_ds.GetRasterBand(4)
    gebco_alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
    gebco_alpha_band.WriteArray(gebco_values)
    gebco_ds = None

    args = argparse.Namespace(
        cache=".cache",
        download=False,
        stats_min=0,
        stats_max=10000,
        tonemap=True,
        grade=True,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        exposure=1.0,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        resample_alg="near",
        max_zoom=13,
        blocksize=256,
        output="render.pmtiles",
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, str(gebco_path))

    assert out_path is not None
    out_ds = gdal.Open(out_path)
    assert out_ds is not None
    alpha = out_ds.GetRasterBand(4).ReadAsArray()
    assert np.any(alpha == 0)
    assert np.any(alpha == 255)


def test_process_single_tile_with_rgba_ocean_alpha_mask(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    raster_size = 100
    pixel_size = 100.0

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda tile, dates, cache: [
            ("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")
        ],
    )

    def fake_get_tile_paths(folder, date, cache, download=False, quiet=False):
        p = {}
        for b in ["red", "green", "blue"]:
            path = tmp_path / f"{b}.tif"
            if not path.exists():
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
                ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(32631)
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).Fill(1000)
                ds = None
            p[b] = str(path)
        return p

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)

    ocean_path = tmp_path / "ocean_rgba.tif"
    ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(ocean_path), raster_size, raster_size, 4, gdal.GDT_Byte
    )
    ocean_ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    ocean_ds.SetProjection(srs.ExportToWkt())
    for band_index in range(1, 4):
        ocean_ds.GetRasterBand(band_index).Fill(0)
    alpha_values = np.full((raster_size, raster_size), 255, dtype=np.uint8)
    alpha_values[:, : raster_size // 2] = 0
    ocean_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    ocean_ds.GetRasterBand(4).WriteArray(alpha_values)
    ocean_ds = None

    args = argparse.Namespace(
        cache=".cache",
        download=False,
        stats_min=0,
        stats_max=10000,
        tonemap=True,
        grade=True,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        exposure=1.0,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        resample_alg="near",
        max_zoom=13,
        blocksize=256,
        output="render.pmtiles",
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, str(ocean_path))

    assert out_path is not None
    out_ds = gdal.Open(out_path)
    assert out_ds is not None
    alpha = out_ds.GetRasterBand(4).ReadAsArray()
    assert np.any(alpha == 255)
    assert np.any(alpha == 0)


def test_process_single_tile_prefetches_all_requested_dates_and_cleans_up(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    raster_size = 100
    pixel_size = 100.0
    band_requests: list[tuple[str, str, bool, bool]] = []
    prefetched_cache_dir: Path | None = None

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda tile, dates, cache: [
            ("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01"),
            ("Sentinel-2_mosaic_2025_Q4_31TDF_0_0", "2025/10/01"),
        ],
    )

    source_paths: dict[str, dict[str, str]] = {}
    for date in ["2025/07/01", "2025/10/01"]:
        source_paths[date] = {}
        for band_name in ["red", "green", "blue"]:
            path = tmp_path / f"source_{date.replace('/', '-')}_{band_name}.tif"
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
            ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(32631)
            ds.SetProjection(srs.ExportToWkt())
            ds.GetRasterBand(1).Fill(1000)
            ds = None
            source_paths[date][band_name] = str(path)

    def fake_get_tile_paths(folder, date, cache, download=False, quiet=False):
        nonlocal prefetched_cache_dir
        band_requests.append((str(cache), date, download, quiet))
        cache_path = Path(cache) if cache is not None else None
        if download and cache_path is not None:
            prefetched_cache_dir = cache_path
        paths = {}
        for band_name in ["red", "green", "blue"]:
            if cache_path is not None:
                cache_prefix = "_".join(folder.split("_")[4:])
                band_id = {"red": "B04", "green": "B03", "blue": "B02"}[band_name]
                path = cache_path / date.replace("/", "-") / f"{cache_prefix}_{band_id}.tif"
                if download:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    driver = gdal.GetDriverByName("GTiff")
                    ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
                    ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(32631)
                    ds.SetProjection(srs.ExportToWkt())
                    ds.GetRasterBand(1).Fill(1000)
                    ds = None
                if path.exists():
                    paths[band_name] = str(path)
                    continue
            paths[band_name] = source_paths[date][band_name]
        return paths

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)

    ocean_path = tmp_path / "ocean_rgba.tif"
    ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(ocean_path), raster_size, raster_size, 4, gdal.GDT_Byte
    )
    ocean_ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    ocean_ds.SetProjection(srs.ExportToWkt())
    alpha_values = np.full((raster_size, raster_size), 255, dtype=np.uint8)
    alpha_values[:30, :] = 0
    ocean_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    ocean_ds.GetRasterBand(4).WriteArray(alpha_values)
    ocean_ds = None

    args = argparse.Namespace(
        cache=".cache",
        download=False,
        prefetch_if_land=100.0,
        stats_min=0,
        stats_max=10000,
        tonemap=True,
        grade=True,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        exposure=1.0,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        resample_alg="near",
        max_zoom=13,
        blocksize=256,
        output="render.pmtiles",
    )

    out_path = process_single_tile(
        "31TDF_0_0",
        ["2025/07/01", "2025/10/01"],
        args,
        str(ocean_path),
    )

    assert out_path is not None
    assert [download for _cache, _date, download, _quiet in band_requests] == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert [date for _cache, date, download, _quiet in band_requests if download] == [
        "2025/07/01",
        "2025/10/01",
    ]
    assert prefetched_cache_dir is not None
    assert not prefetched_cache_dir.exists()


def test_process_single_tile_skips_prefetch_when_land_percentage_below_threshold(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    raster_size = 100
    pixel_size = 100.0
    band_requests: list[tuple[str, bool, bool]] = []
    prefetched_cache_dir: Path | None = None

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders_for_tile",
        lambda tile, dates, cache: [
            ("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")
        ],
    )

    source_paths = {}
    for band_name in ["red", "green", "blue"]:
        path = tmp_path / f"source_{band_name}.tif"
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
        ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32631)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).Fill(1000)
        ds = None
        source_paths[band_name] = str(path)

    def fake_get_tile_paths(folder, date, cache, download=False, quiet=False):
        nonlocal prefetched_cache_dir
        band_requests.append((str(cache), download, quiet))
        cache_path = Path(cache) if cache is not None else None
        if download and cache_path is not None:
            prefetched_cache_dir = cache_path
        paths = {}
        for band_name in ["red", "green", "blue"]:
            if cache_path is not None:
                cache_prefix = "_".join(folder.split("_")[4:])
                band_id = {"red": "B04", "green": "B03", "blue": "B02"}[band_name]
                path = cache_path / date.replace("/", "-") / f"{cache_prefix}_{band_id}.tif"
                if download:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    driver = gdal.GetDriverByName("GTiff")
                    ds = driver.Create(str(path), raster_size, raster_size, 1, gdal.GDT_Int16)
                    ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(32631)
                    ds.SetProjection(srs.ExportToWkt())
                    ds.GetRasterBand(1).Fill(1000)
                    ds = None
                if path.exists():
                    paths[band_name] = str(path)
                    continue
            paths[band_name] = source_paths[band_name]
        return paths

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)

    ocean_path = tmp_path / "ocean_rgba.tif"
    ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(ocean_path), raster_size, raster_size, 4, gdal.GDT_Byte
    )
    ocean_ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    ocean_ds.SetProjection(srs.ExportToWkt())
    alpha_values = np.full((raster_size, raster_size), 255, dtype=np.uint8)
    alpha_values[:10, :] = 0
    ocean_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    ocean_ds.GetRasterBand(4).WriteArray(alpha_values)
    ocean_ds = None

    args = argparse.Namespace(
        cache=".cache",
        download=False,
        prefetch_if_land=20.0,
        stats_min=0,
        stats_max=10000,
        tonemap=True,
        grade=True,
        sb=0.3,
        hb=0.75,
        ss=1.4,
        ms=0.9,
        hs=0.5,
        exposure=1.0,
        gamma=1.0,
        sat=0.9,
        db=0.7,
        ls=0.7,
        resample_alg="near",
        max_zoom=13,
        blocksize=256,
        output="render.pmtiles",
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, str(ocean_path))

    assert out_path is not None
    assert [download for _cache, download, _quiet in band_requests] == [False, False]
    assert prefetched_cache_dir is None


def test_main_vrt_mode(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--vrt", "--parallel", "1", "--date", "2025/07/01"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"])

    # Mock process_single_tile to return a fake path
    def fake_process_single_tile(st, dates, args, gebco_src=None):
        path = tmp_path / f"processed_{st}.tif"
        path.write_text("fake tif")
        return str(path)

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)

    # Mock gdal.BuildVRT
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    main()

    # Check that the master VRT was created
    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1


def test_main_reports_land_progress(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--vrt", "--parallel", "1", "--date", "2025/07/01"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"])

    def fake_process_single_tile(st, dates, args, gebco_src=None):
        path = tmp_path / f"processed_{st}.tif"
        path.write_text("fake tif")
        return str(path)

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    main()

    out = capsys.readouterr().out
    assert "Expanded 1 MGRS tiles into 4 sub-tiles across 1 date(s)." in out
    assert "Land processing progress: 4/4 (100%); Elapsed:" in out
    assert "4 raster(s) ready." in out
    assert "Building master VRT from 4 raster(s)... 100%; Elapsed:" in out


def test_main_low_zoom_uses_subtile_processing_strategy(
    monkeypatch: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--vrt", "--parallel", "1", "--date", "2025/07/01", "--max-zoom", "4"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"])
    processed_tiles: list[str] = []

    def fake_process_single_tile(tile_id, dates, args, gebco_src=None):
        processed_tiles.append(tile_id)
        path = tmp_path / f"{tile_id}.tif"
        path.write_text("fake tif")
        return str(path)

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    main()

    out = capsys.readouterr().out
    assert processed_tiles == ["31TDF_0_0", "31TDF_0_1", "31TDF_1_0", "31TDF_1_1"]
    assert "Expanded 1 MGRS tiles into 4 sub-tiles across 1 date(s)." in out
    assert "Starting sub-tile processing for 4 sub-tile(s) with 1 worker(s);" in out
    assert "Land processing progress: 4/4 (100%); Elapsed:" in out
    assert "4 raster(s) ready." in out


def test_main_passes_ocean_path_to_tile_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    custom_ocean_path = tmp_path / "custom-ocean.tif"
    custom_ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(custom_ocean_path), 1, 1, 4, gdal.GDT_Byte
    )
    assert custom_ocean_ds is not None
    custom_ocean_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    custom_ocean_ds.GetRasterBand(4).Fill(255)
    custom_ocean_ds = None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "satmaps.py",
            "--vrt",
            "--parallel",
            "1",
            "--date",
            "2025/07/01",
            "--ocean-background",
            str(custom_ocean_path),
        ],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"])
    gebco_sources: list[str | None] = []

    def fake_process_single_tile(st, dates, args, gebco_src=None):
        gebco_sources.append(gebco_src)
        path = tmp_path / f"processed_{st}.tif"
        path.write_text("fake tif")
        return str(path)

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    main()

    assert gebco_sources
    assert set(gebco_sources) == {str(custom_ocean_path)}


def test_main_uses_rgba_ocean_as_alpha_mask_source(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    custom_ocean_path = tmp_path / "custom-ocean.tif"
    custom_ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(custom_ocean_path), 1, 1, 4, gdal.GDT_Byte
    )
    assert custom_ocean_ds is not None
    for band_index in range(1, 5):
        custom_ocean_ds.GetRasterBand(band_index).Fill(band_index)
    custom_ocean_ds = None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "satmaps.py",
            "--vrt",
            "--parallel",
            "1",
            "--date",
            "2025/07/01",
            "--ocean-background",
            str(custom_ocean_path),
        ],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(satmaps, "S3_FOLDER_CACHE", {})
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"])
    gebco_sources: list[str | None] = []

    def fake_process_single_tile(st, dates, args, gebco_src=None):
        gebco_sources.append(gebco_src)
        path = tmp_path / f"processed_{st}.tif"
        path.write_text("fake tif")
        return str(path)

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    main()

    assert gebco_sources
    assert gebco_sources
    assert set(gebco_sources) == {str(custom_ocean_path)}


def test_main_bbox_uses_standalone_ocean(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--bbox", "0,0,1,1", "--no-land", "--vrt", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    buildvrt_sources: list[list[str]] = []
    warp_options_calls: list[dict[str, object]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_sources.append(list(src))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: Path(destination).write_text("fake ocean"),
    )
    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
    assert buildvrt_sources[-1][0] != "ocean.tif"
    assert Path(buildvrt_sources[-1][0]).name == "output_ocean_bbox.tif"
    snapped_bounds, pixel_size, _zoom = ocean.snapped_tile_grid_for_bbox(
        (0.0, 0.0, 1.0, 1.0),
        tile_size=512,
    )
    assert warp_options_calls[0]["outputBounds"] == pytest.approx(snapped_bounds)
    assert warp_options_calls[0]["xRes"] == pytest.approx(pixel_size)
    assert warp_options_calls[0]["yRes"] == pytest.approx(pixel_size)


def test_main_builds_master_vrt_with_shared_zoom13_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    assert kwargs["resolution"] == "user"
    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(
        ocean.DEFAULT_MAX_ZOOM,
        512,
    )
    assert kwargs["xRes"] == pytest.approx(expected)
    assert kwargs["yRes"] == pytest.approx(expected)


def test_main_builds_master_vrt_with_requested_zoom14_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1", "--max-zoom", "14"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(14, 512)
    assert kwargs["xRes"] == pytest.approx(expected)
    assert kwargs["yRes"] == pytest.approx(expected)


def test_main_builds_master_vrt_with_requested_zoom11_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1", "--max-zoom", "11"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(11, 512)
    assert kwargs["xRes"] == pytest.approx(expected)
    assert kwargs["yRes"] == pytest.approx(expected)


def test_main_builds_master_vrt_with_requested_zoom12_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1", "--max-zoom", "12"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(12, 512)
    assert kwargs["xRes"] == pytest.approx(expected)
    assert kwargs["yRes"] == pytest.approx(expected)


def test_main_builds_master_vrt_with_requested_zoom4_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1", "--max-zoom", "4"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    expected = satmaps.tiler.web_mercator_pixel_size_for_tile_size(4, 512)
    assert kwargs["xRes"] == pytest.approx(expected)
    assert kwargs["yRes"] == pytest.approx(expected)


def test_main_bbox_passes_chunk_bounds_to_tiler(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(ocean_path), 1, 1, 1, gdal.GDT_Byte
    )
    assert ocean_ds is not None
    ocean_ds.GetRasterBand(1).Fill(1)
    ocean_ds = None

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--bbox", "0,0,1,1", "--no-land", "--parallel", "1", "--format", "png"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )
    monkeypatch.setattr("satmaps.gdal.WarpOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        "satmaps.gdal.Warp",
        lambda destination, source, options=None: Path(destination).write_text("fake ocean"),
    )
    captured_options: dict[str, object] = {}

    def fake_run_tiling_simplified(input_vrt: str, output_mbtiles: str, options: dict[str, object]):
        captured_options.update(options)
        return satmaps.tiler.TilingArtifacts(final_vrt=input_vrt, cleanup_paths=[])

    monkeypatch.setattr("satmaps.tiler.run_tiling_simplified", fake_run_tiling_simplified)
    monkeypatch.setattr(
        "satmaps.subprocess.run",
        lambda cmd, check: Path(cmd[-1]).write_text("fake pmtiles"),
    )

    main()

    assert captured_options["chunk_bounds"] == satmaps.tiler.lonlat_bbox_to_mercator_bounds(
        0.0, 0.0, 1.0, 1.0
    )
    assert "unique_id" not in captured_options


def test_process_single_tile_writes_empty_marker_when_no_folders(
    monkeypatch: object, tmp_path: Path
) -> None:
    completion_marker = tmp_path / "tilecache" / "markers" / "31TDF_0_0.json"
    args = argparse.Namespace(cache=".cache", download=False)
    monkeypatch.setattr("satmaps.list_mosaic_folders_for_tile", lambda *args, **kwargs: [])

    result = process_single_tile(
        "31TDF_0_0",
        ["2025/07/01"],
        args,
        tile_cache_dir=str(tmp_path / "tilecache" / "contributors" / "31TDF_0_0"),
        completion_marker_path=str(completion_marker),
    )
    assert result is None
    assert completion_marker.exists()
    assert completion_marker.exists()
    marker = completion_marker.read_text()
    assert '"tile_count": 0' in marker
    assert '"tiles": []' in marker


def test_main_webp_resume_reuses_cache_markers_without_latest_state_fallback(
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
    monkeypatch.setattr("satmaps.tiler.merge_webp_trees", lambda dirs, final_tree, quality: 1)
    monkeypatch.setattr(
        "satmaps.convert_tile_tree_to_pmtiles",
        lambda *args, **kwargs: str(tmp_path / ".temp" / "output.mbtiles"),
    )

    work_units = satmaps.plan_subtile_work_units(["31TDF"])
    for work_unit in work_units:
        marker_path = satmaps.build_contributor_complete_marker(
            "output.pmtiles",
            "webpresume",
            work_unit.unit_id,
        )
        satmaps.write_tile_cache_marker(marker_path, work_unit.unit_id, [])
    contributor_dir = satmaps.build_contributor_tile_cache_dir(
        "output.pmtiles",
        "webpresume",
        work_units[0].unit_id,
    )
    Path(contributor_dir).mkdir(parents=True, exist_ok=True)

    stale_state = tmp_path / ".temp" / "state_stale.json"
    stale_state.write_text(
        '{"unique_id": "stale", "completed_units": ["stale"], "processed_tifs": [], "args": {}}'
    )

    processed_units: list[str] = []

    def fake_process_land_work_unit(*args, **kwargs):
        processed_units.append(args[0].unit_id)
        return None

    monkeypatch.setattr("satmaps.process_land_work_unit", fake_process_land_work_unit)

    main()

    assert processed_units == []
    out = capsys.readouterr().out
    assert "Reusing 4 existing tile cache contributor(s)." in out
    assert "All sub-tiles already processed." in out


def test_main_non_bbox_can_use_standalone_ocean(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    (tmp_path / "ocean.tif").write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    buildvrt_sources: list[list[str]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_sources.append(list(src))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
    assert buildvrt_sources[-1] == ["ocean.tif"]


def test_main_keeps_ocean_after_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_path = tmp_path / "ocean.tif"
    ocean_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--parallel", "1", "--format", "png"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr("satmaps.discover_mgrs_bases", lambda bbox, gebco_src, land_mgrs_list_path=None: [])
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    def fake_run_tiling_simplified(input_vrt: str, output_mbtiles: str, options: dict[str, object]):
        Path(output_mbtiles).write_text("fake mbtiles")
        return satmaps.tiler.TilingArtifacts(final_vrt=input_vrt, cleanup_paths=[])

    monkeypatch.setattr("satmaps.tiler.run_tiling_simplified", fake_run_tiling_simplified)
    monkeypatch.setattr(
        "satmaps.subprocess.run",
        lambda cmd, check: Path(cmd[-1]).write_text("fake pmtiles"),
    )

    main()

    assert ocean_path.exists()
    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
    assert master_vrts[0].exists()


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


@pytest.mark.parametrize(
    ("extra_args", "expect_land_tif_during_pmtiles", "expect_land_tif_after_main"),
    [
        ([], True, True),
        (["--delete-tifs"], False, False),
    ],
)
def test_main_delete_tifs_flag_controls_land_cleanup(
    monkeypatch: object,
    tmp_path: Path,
    extra_args: list[str],
    expect_land_tif_during_pmtiles: bool,
    expect_land_tif_after_main: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--parallel", "1", "--date", "2025/07/01", "--format", "png", *extra_args],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.populate_s3_cache", lambda date_paths: None)
    monkeypatch.setattr(
        "satmaps.discover_mgrs_bases",
        lambda bbox, gebco_src, land_mgrs_list_path=None: ["31TDF"],
    )
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    def fake_process_single_tile(st, dates, args, gebco_src=None):
        path = Path(".temp") / f"output_{st}_3857.tif"
        path.write_text("fake tif")
        return str(path)

    def fake_run_tiling_simplified(input_vrt: str, output_mbtiles: str, options: dict[str, object]):
        Path(output_mbtiles).write_text("fake mbtiles")
        return satmaps.tiler.TilingArtifacts(final_vrt=input_vrt, cleanup_paths=[])

    checked_inputs: list[bool] = []

    def fake_subprocess_run(cmd, check):
        land_tif = Path(".temp/output_31TDF_0_0_3857.tif")
        checked_inputs.append(land_tif.exists())
        Path(cmd[-1]).write_text("fake pmtiles")

    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    monkeypatch.setattr("satmaps.tiler.run_tiling_simplified", fake_run_tiling_simplified)
    monkeypatch.setattr("satmaps.subprocess.run", fake_subprocess_run)

    main()

    assert checked_inputs == [expect_land_tif_during_pmtiles]
    assert Path(".temp/output_31TDF_0_0_3857.tif").exists() is expect_land_tif_after_main
