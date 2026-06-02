import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
import ocean
import land_mgrs
import satmaps
import tiler
from satmaps import (
    build_alpha_block,
    build_fill_allowed_mask,
    build_bbox_geometry,
    build_candidate_tile_cache_token,
    build_land_run_token,
    expand_subtiles,
    find_resume_path,
    format_progress,
    get_ocean_mask_band_index,
    iter_processing_windows,
    parse_bbox,
    restore_resume_state,
)

QFJ_TILE_BOUNDS = (
    -158.039128863903,
    20.7971889977951,
    -157.06683030196527,
    21.692137914318558,
)
WIDE_HAWAII_BOUNDS = (-158.0, 20.8, -155.8, 22.2)


def build_projected_ocean_mask(
    mask_path: Path,
    land_patch: np.ndarray,
    bounds: tuple[float, float, float, float] = QFJ_TILE_BOUNDS,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bounds
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    if hasattr(wgs84, "SetAxisMappingStrategy"):
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        web_mercator.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(wgs84, web_mercator)

    min_x, min_y, _ = transform.TransformPoint(min_lon, min_lat)
    max_x, max_y, _ = transform.TransformPoint(max_lon, max_lat)
    left = min(min_x, max_x) - 1000.0
    right = max(min_x, max_x) + 1000.0
    bottom = min(min_y, max_y) - 1000.0
    top = max(min_y, max_y) + 1000.0

    height, width = land_patch.shape
    pixel_width = (right - left) / width
    pixel_height = (top - bottom) / height

    ds = gdal.GetDriverByName("GTiff").Create(str(mask_path), width, height, 4, gdal.GDT_Byte)
    assert ds is not None
    ds.SetGeoTransform((left, pixel_width, 0.0, top, 0.0, -pixel_height))
    ds.SetProjection(web_mercator.ExportToWkt())
    alpha_band = ds.GetRasterBand(4)
    alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
    alpha_band.Fill(255)
    alpha_band.WriteArray(land_patch)
    ds = None


def test_restore_resume_state_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "unique_id": "resume-id",
                "completed_units": ["31TDF_0_0", "31TDF_1_0"],
                "processed_tifs": [str(tmp_path / "legacy.tif")],
                "args": {"parallel": 2},
            }
        )
    )

    restored = restore_resume_state(str(state_file))
    assert restored == {
        "state_file": str(state_file),
        "unique_id": "resume-id",
        "completed_units": {"31TDF_0_0", "31TDF_1_0"},
    }
    assert "Resuming from state file" in capsys.readouterr().out


def test_restore_resume_state_accepts_files_without_legacy_processed_tifs(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "state.json"

    state_file.write_text(
        json.dumps(
            {
                "unique_id": "resume-id",
                "completed_units": ["31TDF_0_0"],
                "args": {"parallel": 2},
            }
        )
    )

    restored = restore_resume_state(str(state_file))

    assert restored is not None
    assert restored["completed_units"] == {"31TDF_0_0"}


def test_plan_subtile_work_units_expands_subtiles() -> None:
    work_units = satmaps.plan_subtile_work_units(["31TDF"])

    assert [unit.unit_id for unit in work_units] == [
        "31TDF_0_0",
        "31TDF_0_1",
        "31TDF_1_0",
        "31TDF_1_1",
    ]


def test_plan_subtile_work_units_filters_to_available_subtiles() -> None:
    work_units = satmaps.plan_subtile_work_units(
        ["31TDF"],
        {"31TDF_0_1", "31TDF_1_0"},
    )

    assert [unit.unit_id for unit in work_units] == [
        "31TDF_0_1",
        "31TDF_1_0",
    ]


def test_discover_mgrs_bases_reuses_saved_land_mgrs_list_ignoring_source_metadata(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    land_mgrs_list_path = satmaps.build_land_mgrs_list_path()
    satmaps.save_land_mgrs_list(
        land_mgrs_list_path,
        {"04QFJ", "05QFJ"},
        bbox=None,
        ocean_mask_source="gebco.vrt",
    )
    monkeypatch.setattr(
        satmaps,
        "S3_FOLDER_CACHE",
        {
            "2025/07/01": {
                "Sentinel-2_mosaic_2025_Q3_04QFJ_0_0",
                "Sentinel-2_mosaic_2025_Q3_31TDF_0_0",
            }
        },
    )
    monkeypatch.setattr(
        "satmaps.discover_mgrs_tiles_from_ocean_mask",
        lambda ocean_mask_src, bbox=None, candidate_mgrs_tiles=None: (_ for _ in ()).throw(
            AssertionError("expected cached land MGRS list to skip ocean scan")
        ),
    )

    assert satmaps.discover_mgrs_bases(None, "different-gebco.vrt", land_mgrs_list_path) == ["04QFJ"]


def test_restore_resume_state_returns_none_for_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("{not json")

    assert restore_resume_state(str(state_file)) is None
    assert "Warning: Could not load state file" in capsys.readouterr().out

def test_parse_bbox_parses_numbers_and_exits_on_invalid_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert parse_bbox("-10.5,20,30.25,40") == (-10.5, 20.0, 30.25, 40.0)

    with pytest.raises(SystemExit, match="1"):
        parse_bbox("not,a,bbox")

    assert "Error: Invalid bbox format: not,a,bbox" in capsys.readouterr().out


def test_discover_mgrs_bases_uses_bbox_clipped_ocean_mask_when_available(
    monkeypatch: object,
) -> None:
    calls: list[tuple[str, tuple[float, float, float, float] | None, set[str] | None]] = []

    def fake_discover(
        ocean_mask_src: str,
        bbox: tuple[float, float, float, float] | None = None,
        candidate_mgrs_tiles: set[str] | None = None,
    ) -> set[str]:
        calls.append((ocean_mask_src, bbox, candidate_mgrs_tiles))
        return {"05QFJ", "04QFJ"}

    monkeypatch.setattr("satmaps.discover_mgrs_tiles_from_ocean_mask", fake_discover)
    monkeypatch.setattr(
        "satmaps.discover_mgrs_tiles_in_bbox",
        lambda min_lon, min_lat, max_lon, max_lat: ["04QFJ", "05QFJ", "31TDF"],
    )

    assert satmaps.discover_mgrs_bases((-158.0, 20.8, -157.0, 21.7), "gebco.vrt") == [
        "04QFJ",
        "05QFJ",
    ]
    assert calls == [("gebco.vrt", (-158.0, 20.8, -157.0, 21.7), {"04QFJ", "05QFJ", "31TDF"})]


def test_discover_mgrs_bases_intersects_all_tiles_s3_cache_with_mask(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "satmaps.discover_mgrs_tiles_from_ocean_mask",
        lambda ocean_mask_src, bbox=None, candidate_mgrs_tiles=None: {"04QFJ", "99ZZZ"},
    )
    monkeypatch.setattr(
        satmaps,
        "S3_FOLDER_CACHE",
        {
            "2025/07/01": {
                "Sentinel-2_mosaic_2025_Q3_04QFJ_0_0",
                "Sentinel-2_mosaic_2025_Q3_31TDF_0_0",
            }
        },
    )

    assert satmaps.discover_mgrs_bases(None, "gebco.vrt") == ["04QFJ"]


def test_discover_mgrs_bases_ignores_bbox_scoped_cache_for_all_tiles(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    land_mgrs_list_path = satmaps.build_land_mgrs_list_path()
    satmaps.save_land_mgrs_list(
        land_mgrs_list_path,
        {"04QFJ"},
        bbox=(-158.0, 20.8, -157.0, 21.7),
        ocean_mask_source="gebco.vrt",
    )
    calls: list[tuple[str, tuple[float, float, float, float] | None, set[str] | None]] = []

    def fake_discover(
        ocean_mask_src: str,
        bbox: tuple[float, float, float, float] | None = None,
        candidate_mgrs_tiles: set[str] | None = None,
    ) -> set[str]:
        calls.append((ocean_mask_src, bbox, candidate_mgrs_tiles))
        return {"31TDF"}

    monkeypatch.setattr("satmaps.discover_mgrs_tiles_from_ocean_mask", fake_discover)
    monkeypatch.setattr(
        satmaps,
        "S3_FOLDER_CACHE",
        {
            "2025/07/01": {
                "Sentinel-2_mosaic_2025_Q3_31TDF_0_0",
            }
        },
    )

    assert satmaps.discover_mgrs_bases(None, "gebco.vrt", land_mgrs_list_path) == ["31TDF"]
    assert calls == [("gebco.vrt", None, {"31TDF"})]


def test_discover_mgrs_bases_force_refreshes_saved_land_mgrs_list(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    land_mgrs_list_path = satmaps.build_land_mgrs_list_path()
    satmaps.save_land_mgrs_list(
        land_mgrs_list_path,
        {"04QFJ"},
        bbox=(-158.0, 20.8, -157.0, 21.7),
        ocean_mask_source="gebco.vrt",
    )
    calls: list[tuple[str, tuple[float, float, float, float] | None, set[str] | None]] = []

    def fake_discover(
        ocean_mask_src: str,
        bbox: tuple[float, float, float, float] | None = None,
        candidate_mgrs_tiles: set[str] | None = None,
    ) -> set[str]:
        calls.append((ocean_mask_src, bbox, candidate_mgrs_tiles))
        return {"05QFJ"}

    monkeypatch.setattr("satmaps.discover_mgrs_tiles_from_ocean_mask", fake_discover)
    monkeypatch.setattr(
        "satmaps.discover_mgrs_tiles_in_bbox",
        lambda min_lon, min_lat, max_lon, max_lat: ["04QFJ", "05QFJ"],
    )

    assert satmaps.discover_mgrs_bases(
        (-158.0, 20.8, -157.0, 21.7),
        "gebco.vrt",
        land_mgrs_list_path,
        force_refresh=True,
    ) == ["05QFJ"]
    assert calls == [("gebco.vrt", (-158.0, 20.8, -157.0, 21.7), {"04QFJ", "05QFJ"})]
    assert satmaps.load_saved_land_mgrs_list(
        land_mgrs_list_path,
        bbox=(-158.0, 20.8, -157.0, 21.7),
        ocean_mask_source="gebco.vrt",
    ) == {"05QFJ"}


def test_expand_subtiles_and_find_resume_path(tmp_path: Path, monkeypatch: object) -> None:
    assert expand_subtiles(["31TDF", "32TLP"]) == [
        "31TDF_0_0",
        "31TDF_0_1",
        "31TDF_1_0",
        "31TDF_1_1",
        "32TLP_0_0",
        "32TLP_0_1",
        "32TLP_1_0",
        "32TLP_1_1",
    ]

    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".temp"
    state_dir.mkdir()
    older = state_dir / "state_old.json"
    newer = state_dir / "state_new.json"
    older.write_text("{}")
    newer.write_text("{}")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert find_resume_path(str(newer)) == str(newer)
    assert find_resume_path(True, preferred_path=str(older)) == str(older)
    assert find_resume_path(True) is None


def test_build_land_run_token_is_stable_for_matching_inputs() -> None:
    args = satmaps.argparse.Namespace(
        output="render.pmtiles",
        max_zoom=13,
        blocksize=256,
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
        gamma=2.6,
        shoulder=1.0,
        sat=0.9,
        db=0.15,
        ls=0.2,
        ghb=None,
        gms=1.0,
        ghs=None,
    )

    token_a = build_land_run_token(
        args,
        ["2025/07/01", "2025/10/01"],
        (0.0, 0.0, 1.0, 1.0),
        "ocean.tif",
    )
    token_b = build_land_run_token(
        args,
        ["2025/07/01", "2025/10/01"],
        (0.0, 0.0, 1.0, 1.0),
        "ocean.tif",
    )
    token_c = build_land_run_token(
        args,
        ["2025/07/01"],
        (0.0, 0.0, 1.0, 1.0),
        "ocean.tif",
    )

    assert token_a == token_b
    assert token_a != token_c


def test_build_candidate_tile_cache_token_ignores_output_path() -> None:
    args_a = satmaps.argparse.Namespace(
        output="render-a.pmtiles",
        max_zoom=13,
        blocksize=256,
        resample_alg="lanczos",
    )
    args_b = satmaps.argparse.Namespace(
        output="render-b.pmtiles",
        max_zoom=13,
        blocksize=256,
        resample_alg="lanczos",
    )
    args_c = satmaps.argparse.Namespace(
        output="render-a.pmtiles",
        max_zoom=14,
        blocksize=256,
        resample_alg="lanczos",
    )

    token_a = build_candidate_tile_cache_token(args_a)
    token_b = build_candidate_tile_cache_token(args_b)
    token_c = build_candidate_tile_cache_token(args_c)
    assert token_a == token_b
    assert token_a != token_c


def test_helper_modules_reexport_common_helpers() -> None:
    assert satmaps.build_staged_path is common.build_staged_path
    assert satmaps.publish_staged_path is common.publish_staged_path
    assert satmaps.remove_if_exists is common.remove_if_exists
    assert satmaps.format_eta is common.format_eta
    assert satmaps.LiveProgressLine is common.LiveProgressLine
    assert ocean.build_staged_path is common.build_staged_path
    assert ocean.publish_staged_path is common.publish_staged_path
    assert ocean.remove_if_exists is common.remove_if_exists
    assert ocean.format_eta is common.format_eta
    assert ocean.LiveProgressLine is common.LiveProgressLine
    assert tiler.build_staged_path is common.build_staged_path
    assert tiler.publish_staged_path is common.publish_staged_path
    assert tiler.remove_if_exists is common.remove_if_exists


def test_remove_if_exists_prefers_gdal_unlink(monkeypatch: object) -> None:
    calls: list[str] = []

    monkeypatch.setattr(common.os.path, "exists", lambda path: True)
    monkeypatch.setattr(common.gdal, "Unlink", lambda path: calls.append(path))
    monkeypatch.setattr(
        common.os,
        "remove",
        lambda path: (_ for _ in ()).throw(AssertionError("unexpected os.remove fallback")),
    )

    common.remove_if_exists("output.tif")

    assert calls == ["output.tif"]


def test_remove_if_exists_falls_back_to_os_remove(monkeypatch: object) -> None:
    calls: list[str] = []

    monkeypatch.setattr(common.os.path, "exists", lambda path: True)

    def fail_unlink(path: str) -> None:
        raise RuntimeError("unlink failed")

    monkeypatch.setattr(common.gdal, "Unlink", fail_unlink)
    monkeypatch.setattr(common.os, "remove", lambda path: calls.append(path))

    common.remove_if_exists("output.tif")

    assert calls == ["output.tif"]


def test_iter_processing_windows_uses_full_width_row_slabs() -> None:
    tile_grid = satmaps.TileGrid(
        projection="EPSG:32631",
        geotransform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        width=10008,
        height=50,
    )

    assert list(iter_processing_windows(tile_grid)) == [
        (0, 0, 10008, 24),
        (0, 24, 10008, 24),
        (0, 48, 10008, 2),
    ]


def test_format_progress_covers_long_runs() -> None:
    assert format_progress(3, 12) == "3/12 (25%)"


def test_populate_s3_cache_reports_progress(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    satmaps.S3_FOLDER_CACHE.clear()

    def fake_readdir(path: str) -> list[str] | None:
        if path.endswith("/2025/07/01"):
            return ["Sentinel-2_mosaic_2025_Q3_31TDF_0_0"]
        if path.endswith("/2025/01/01"):
            return ["Sentinel-2_mosaic_2025_Q1_31TDF_0_0", "Sentinel-2_mosaic_2025_Q1_32TLP_0_0"]
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(satmaps.gdal, "ReadDir", fake_readdir)

    satmaps.populate_s3_cache(["2025/07/01", "2025/01/01"])

    out = capsys.readouterr().out
    assert "Populating S3 folder cache for 2 date(s)..." in out
    assert "S3 cache progress: 1/2 (50%);" in out
    assert "listing 2025/07/01..." in out
    assert "S3 cache progress: 2/2 (100%); Elapsed:" in out
    assert "listing 2025/01/01..." in out
    assert "S3 folder cache ready: 2 date(s), 3 folders total." in out


def test_update_count_progress_bases_eta_on_new_work_after_resume(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(satmaps.time, "perf_counter", lambda: 100.0)
    progress_line = satmaps.LiveProgressLine()

    satmaps.update_count_progress(
        progress_line,
        "Land processing progress:",
        7,
        8,
        90.0,
        "7 raster(s) ready.",
        completed_before_start=6,
    )
    progress_line.finish()

    out = capsys.readouterr().out
    assert "Land processing progress: 7/8 (88%); ETA: 10s; 7 raster(s) ready." in out


def test_update_count_progress_uses_live_eta(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    perf_counter_values = iter([101.0, 106.0])
    monkeypatch.setattr(satmaps.time, "perf_counter", lambda: next(perf_counter_values))
    progress_line = satmaps.LiveProgressLine()

    satmaps.update_count_progress(
        progress_line,
        "Land processing progress:",
        1,
        4,
        100.0,
        "1 raster(s) ready.",
    )
    satmaps.update_count_progress(
        progress_line,
        "Land processing progress:",
        2,
        4,
        100.0,
        "2 raster(s) ready.",
    )
    progress_line.finish()

    out = capsys.readouterr().out
    assert "Land processing progress: 1/4 (25%); ETA: 3s; 1 raster(s) ready." in out
    assert "Land processing progress: 2/4 (50%); ETA: 6s; 2 raster(s) ready." in out


def test_build_alpha_block_uses_source_mask_without_gebco() -> None:
    source_valid_mask = np.array([[True, False, True]], dtype=bool)

    np.testing.assert_array_equal(
        build_alpha_block(None, source_valid_mask),
        np.array([[255, 0, 255]], dtype=np.uint8),
    )


def test_build_alpha_block_inverts_ocean_alpha() -> None:
    np.testing.assert_array_equal(
        build_alpha_block(
            np.array([[255.0, 128.0, 0.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=bool),
        ),
        np.array([[0, 127, 255]], dtype=np.uint8),
    )


def test_get_ocean_mask_band_index_requires_explicit_alpha_band() -> None:
    alpha_dataset = gdal.GetDriverByName("MEM").Create("", 1, 1, 4, gdal.GDT_Byte)
    assert alpha_dataset is not None
    alpha_dataset.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    assert get_ocean_mask_band_index(alpha_dataset) == 4

    single_band_dataset = gdal.GetDriverByName("MEM").Create("", 1, 1, 1, gdal.GDT_Byte)
    assert single_band_dataset is not None
    assert get_ocean_mask_band_index(single_band_dataset) is None


def test_build_fill_allowed_mask_uses_alpha_and_coverage() -> None:
    np.testing.assert_array_equal(
        build_fill_allowed_mask(
            np.array([[0.0, 255.0, 0.0]], dtype=np.float32),
            np.array([[True, True, False]], dtype=bool),
        ),
        np.array([[True, False, False]], dtype=bool),
    )


def test_build_fill_allowed_mask_excludes_nodata_pixels() -> None:
    np.testing.assert_array_equal(
        build_fill_allowed_mask(
            np.array([[0.0, -1.0, 255.0]], dtype=np.float32),
            nodata_value=-1.0,
        ),
        np.array([[True, False, False]], dtype=bool),
    )


def test_build_discovery_fill_allowed_mask_depth_mode_uses_shallow_cutoff() -> None:
    np.testing.assert_array_equal(
        land_mgrs.build_discovery_fill_allowed_mask(
            np.array([[-200.0, -50.0, -20.0, 10.0, -32767.0]], dtype=np.float32),
            nodata=-32767.0,
            depth_mode=True,
        ),
        np.array([[False, True, True, True, False]], dtype=bool),
    )


def test_build_bbox_geometry_reprojects_bbox_to_dataset_srs() -> None:
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    bbox_geom = build_bbox_geometry(QFJ_TILE_BOUNDS, web_mercator)

    min_lon, min_lat, max_lon, max_lat = QFJ_TILE_BOUNDS
    expected_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(
        min_lon, min_lat, max_lon, max_lat
    )
    envelope = bbox_geom.GetEnvelope()

    assert envelope == pytest.approx(
        (
            expected_bounds[0],
            expected_bounds[2],
            expected_bounds[1],
            expected_bounds[3],
        )
    )


def test_discover_mgrs_tiles_from_projected_ocean_mask_uses_wgs84_sampling(tmp_path: Path) -> None:
    ocean_mask_path = tmp_path / "ocean-mask-3857.tif"
    land_patch = np.full((100, 100), 255, dtype=np.uint8)
    land_patch[20:80, 20:80] = 0
    build_projected_ocean_mask(ocean_mask_path, land_patch)

    assert "04QFJ" in satmaps.discover_mgrs_tiles_from_ocean_mask(str(ocean_mask_path))


def test_discover_mgrs_tiles_from_projected_ocean_mask_respects_bbox_clip(
    tmp_path: Path,
) -> None:
    ocean_mask_path = tmp_path / "ocean-mask-3857.tif"
    land_patch = np.full((100, 100), 255, dtype=np.uint8)
    land_patch[10:30, 60:80] = 0
    build_projected_ocean_mask(ocean_mask_path, land_patch)

    min_lon, min_lat, max_lon, max_lat = QFJ_TILE_BOUNDS
    southwest_bbox = (min_lon, min_lat, (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)
    northeast_bbox = ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0, max_lon, max_lat)

    assert satmaps.discover_mgrs_tiles_from_ocean_mask(
        str(ocean_mask_path),
        bbox=southwest_bbox,
    ) == set()
    assert satmaps.discover_mgrs_tiles_from_ocean_mask(
        str(ocean_mask_path),
        bbox=northeast_bbox,
    ) == {"04QFJ"}


def test_discover_mgrs_tiles_from_projected_ocean_mask_misses_zigzag_hawaii_tiles(
    tmp_path: Path,
) -> None:
    ocean_mask_path = tmp_path / "ocean-mask-zigzag-3857.tif"
    land_patch = np.full((320, 480), 255, dtype=np.uint8)
    for y in range(land_patch.shape[0]):
        center_x = int((0.25 + 0.5 * np.sin(y / 22.0)) * land_patch.shape[1])
        land_patch[y, max(0, center_x - 8) : min(land_patch.shape[1], center_x + 9)] = 0
    build_projected_ocean_mask(
        ocean_mask_path,
        land_patch,
        bounds=WIDE_HAWAII_BOUNDS,
    )

    assert satmaps.discover_mgrs_tiles_from_ocean_mask(
        str(ocean_mask_path),
        bbox=WIDE_HAWAII_BOUNDS,
    ) == {
        "04QFJ",
        "04QFK",
        "04QGJ",
        "04QGK",
        "04QHJ",
        "04QHK",
        "05QJD",
        "05QJE",
        "05QKD",
        "05QKE",
    }


def test_discover_mgrs_tiles_from_projected_ocean_mask_uses_batched_reads(
    tmp_path: Path,
) -> None:
    ocean_mask_path = tmp_path / "ocean-mask-parallel-3857.tif"
    land_patch = np.full((320, 480), 255, dtype=np.uint8)
    for y in range(land_patch.shape[0]):
        center_x = int((0.25 + 0.5 * np.sin(y / 22.0)) * land_patch.shape[1])
        land_patch[y, max(0, center_x - 8) : min(land_patch.shape[1], center_x + 9)] = 0
    build_projected_ocean_mask(
        ocean_mask_path,
        land_patch,
        bounds=WIDE_HAWAII_BOUNDS,
    )

    assert satmaps.discover_mgrs_tiles_from_ocean_mask(
        str(ocean_mask_path),
        bbox=WIDE_HAWAII_BOUNDS,
    ) == {
        "04QFJ",
        "04QFK",
        "04QGJ",
        "04QGK",
        "04QHJ",
        "04QHK",
        "05QJD",
        "05QJE",
        "05QKD",
        "05QKE",
    }


def test_discover_mgrs_tiles_from_ocean_mask_targets_candidate_row_blocks(
    monkeypatch: object,
) -> None:
    class FakeBand:
        def __init__(self) -> None:
            self.read_calls: list[tuple[int, int, int, int]] = []

        def GetNoDataValue(self) -> None:
            return None

        def GetBlockSize(self) -> tuple[int, int]:
            return (10, 10)

        def ReadAsArray(
            self,
            xoff: int,
            yoff: int,
            width: int,
            height: int,
        ) -> np.ndarray:
            self.read_calls.append((xoff, yoff, width, height))
            return np.zeros((height, width), dtype=np.uint8)

    class FakeDataset:
        RasterXSize = 80
        RasterYSize = 40

        def __init__(self, band: FakeBand) -> None:
            self.band = band

        def GetRasterBand(self, band_index: int) -> FakeBand:
            assert band_index == 1
            return self.band

        def GetGeoTransform(self) -> tuple[float, float, float, float, float, float]:
            return (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

        def GetSpatialRef(self) -> None:
            return None

    fake_band = FakeBand()
    fake_ds = FakeDataset(fake_band)
    transform_builds = 0
    progress_updates: list[tuple[int, int, str]] = []

    monkeypatch.setattr(land_mgrs.gdal, "Open", lambda path: fake_ds)
    monkeypatch.setattr(land_mgrs, "get_ocean_mask_band_index", lambda ds: 1)
    monkeypatch.setattr(land_mgrs, "get_bbox_scan_window", lambda ds, bbox: (7, 0, 53, 40))
    monkeypatch.setattr(
        land_mgrs,
        "build_candidate_scan_envelopes",
        lambda dataset, candidate_tiles, bbox=None: [
            (7.0, 17.0, -10.0, 0.0),
            (7.0, 17.0, -20.0, -10.0),
        ],
    )

    def fake_build_wgs84_transform(ds: object) -> None:
        nonlocal transform_builds
        transform_builds += 1
        return None

    monkeypatch.setattr(
        land_mgrs,
        "build_wgs84_transform",
        fake_build_wgs84_transform,
    )
    monkeypatch.setattr(
        land_mgrs,
        "process_ocean_mask_window",
        lambda data, xoff, yoff, scan_window, geotransform, nodata, to_wgs84, mgrs_converter, bbox, candidate_tiles, depth_mode=False: set()
        if candidate_tiles is None
        else {"04QFJ"}
        if yoff == 0
        else {"04QFK"}
        if yoff == 10
        else set(),
    )
    monkeypatch.setattr(
        satmaps,
        "update_count_progress",
        lambda progress_line, label, current, total, started_at, detail, completed_before_start=0: progress_updates.append((current, total, detail)),
    )

    assert satmaps.discover_mgrs_tiles_from_ocean_mask(
        "fake-ocean-mask.tif",
        candidate_mgrs_tiles={"04QFJ", "04QFK"},
    ) == {"04QFJ", "04QFK"}
    assert fake_band.read_calls == [
        (7, 0, 53, 10),
        (7, 10, 53, 10),
    ]
    assert transform_builds == 1
    assert progress_updates == [
        (1, 2, "candidate row blocks 1/2; 1 tiles found so far."),
        (2, 2, "candidate row blocks 2/2; 2 tiles found so far."),
    ]


def test_land_mgrs_main_parses_bbox_and_prints_output(
    monkeypatch: object, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str, tuple[float, float, float, float] | None, bool]] = []

    def fake_generate(
        gebco_zip: str,
        destination: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        force_refresh: bool = False,
    ) -> str:
        calls.append((gebco_zip, destination, bbox, force_refresh))
        return destination

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "land_mgrs.py",
            "--bbox",
            "-158.0,20.8,-157.0,21.7",
            "--refresh",
            "custom-gebco.zip",
            "custom-land.list",
        ],
    )
    monkeypatch.setattr(land_mgrs, "generate_land_mgrs_list", fake_generate)

    land_mgrs.main()

    assert calls == [
        (
            "custom-gebco.zip",
            "custom-land.list",
            (-158.0, 20.8, -157.0, 21.7),
            True,
        )
    ]
    assert capsys.readouterr().out.strip() == "custom-land.list"


def test_build_alpha_block_masks_out_pixels_outside_ocean_render() -> None:
    np.testing.assert_array_equal(
        build_alpha_block(
            np.array([[0.0, 255.0, 0.0]], dtype=np.float32),
            np.array([[True, True, True]], dtype=bool),
            np.array([[True, True, False]], dtype=bool),
        ),
        np.array([[255, 0, 0]], dtype=np.uint8),
    )


def test_build_alpha_block_prefers_land_for_filled_seam_pixels() -> None:
    np.testing.assert_array_equal(
        build_alpha_block(
            np.array([[255.0, 255.0]], dtype=np.float32),
            np.array([[False, True]], dtype=bool),
            np.array([[True, True]], dtype=bool),
            np.array([[True, False]], dtype=bool),
        ),
        np.array([[255, 0]], dtype=np.uint8),
    )


def test_build_alpha_block_uses_hard_mask_when_fill_allowed_is_present() -> None:
    np.testing.assert_array_equal(
        build_alpha_block(
            np.array([[255.0, 200.0, 0.0]], dtype=np.float32),
            np.array([[False, False, False]], dtype=bool),
            np.array([[True, True, True]], dtype=bool),
            np.array([[False, True, False]], dtype=bool),
        ),
        np.array([[0, 255, 0]], dtype=np.uint8),
    )


def test_build_alpha_block_preserves_ocean_alpha_gradient() -> None:
    np.testing.assert_array_equal(
        build_alpha_block(
            np.array([[255.0, 200.0, 0.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=bool),
        ),
        np.array([[0, 55, 255]], dtype=np.uint8),
    )
