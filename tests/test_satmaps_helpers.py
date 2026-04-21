import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import satmaps
from satmaps import (
    build_alpha_block,
    delete_state,
    expand_subtiles,
    filter_mgrs_tiles,
    find_resume_path,
    get_source_scale,
    infer_quarter,
    iter_processing_windows,
    load_land_tiles,
    load_state,
    parse_bbox,
    parse_date_paths,
    restore_resume_state,
    save_state,
)


def test_infer_quarter_and_parse_date_paths_helpers() -> None:
    assert infer_quarter("2025/01/01") == "Q1"
    assert infer_quarter("2025/04/01") == "Q2"
    assert infer_quarter("2025/07/01") == "Q3"
    assert infer_quarter("2025/10/01") == "Q4"
    assert infer_quarter("2025/08/15") == "Q3"

    assert parse_date_paths("2025/07/01, 2025/01/01 ,2024/10/01") == [
        "2025/07/01",
        "2025/01/01",
        "2024/10/01",
    ]


def test_save_restore_and_delete_state_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_file = tmp_path / "state.json"
    kept_tif = tmp_path / "kept.tif"
    kept_tif.write_text("kept")

    args = argparse.Namespace(cache=".cache", parallel=2, download=False)
    save_state(
        str(state_file),
        "resume-id",
        {"31TDF_0_0", "31TDF_1_0"},
        [str(kept_tif), str(tmp_path / "missing.tif")],
        args,
    )

    loaded = load_state(str(state_file))
    assert loaded is not None
    assert loaded["unique_id"] == "resume-id"
    assert set(loaded["completed_subtiles"]) == {"31TDF_0_0", "31TDF_1_0"}
    assert loaded["args"]["parallel"] == 2

    restored = restore_resume_state(str(state_file))
    assert restored == satmaps.ResumeState(
        state_file=str(state_file),
        unique_id="resume-id",
        completed_subtiles={"31TDF_0_0", "31TDF_1_0"},
        processed_tifs=[str(kept_tif)],
    )
    assert "Resuming from state file" in capsys.readouterr().out

    delete_state(str(state_file))
    assert not state_file.exists()


def test_load_land_tiles_ignores_blank_lines_and_missing_file(tmp_path: Path) -> None:
    land_tiles = tmp_path / "land_tiles.txt"
    land_tiles.write_text("\n31TDF\n\n32TLP  \n31TDF\n")

    assert load_land_tiles(str(land_tiles)) == {"31TDF", "32TLP"}
    assert load_land_tiles(str(tmp_path / "missing.txt")) is None


def test_parse_bbox_parses_numbers_and_exits_on_invalid_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert parse_bbox("-10.5,20,30.25,40") == (-10.5, 20.0, 30.25, 40.0)

    with pytest.raises(SystemExit, match="1"):
        parse_bbox("not,a,bbox")

    assert "Error: Invalid bbox format: not,a,bbox" in capsys.readouterr().out


def test_filter_mgrs_tiles_respects_land_set_and_gebco_fallback(monkeypatch: object) -> None:
    calls: list[tuple[str, str]] = []

    def fake_check_land_gebco(mgrs_tile: str, gebco_src: str) -> bool:
        calls.append((mgrs_tile, gebco_src))
        return mgrs_tile == "32TLP"

    monkeypatch.setattr("satmaps.check_land_gebco", fake_check_land_gebco)

    assert filter_mgrs_tiles(["31TDF", "32TLP", "33TWN"], {"31TDF"}, "gebco.vrt") == [
        "31TDF",
        "32TLP",
    ]
    assert calls == [("32TLP", "gebco.vrt"), ("33TWN", "gebco.vrt")]

    calls.clear()
    assert filter_mgrs_tiles(["31TDF", "32TLP"], None, "gebco.vrt") == ["31TDF", "32TLP"]
    assert calls == [("31TDF", "gebco.vrt"), ("32TLP", "gebco.vrt")]
    assert filter_mgrs_tiles(["31TDF", "32TLP"], None, None) == ["31TDF", "32TLP"]


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
    assert find_resume_path(True) == str(Path(".temp") / "state_new.json")


def test_get_source_scale_uses_defaults_and_overrides() -> None:
    assert get_source_scale(argparse.Namespace(stats_min=None, stats_max=None)) == (0.0, 9000.0)
    assert get_source_scale(argparse.Namespace(stats_min=12.5, stats_max=345.0)) == (12.5, 345.0)


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


def test_build_alpha_block_uses_source_mask_without_gebco() -> None:
    source_valid_mask = np.array([[True, False, True]], dtype=bool)

    np.testing.assert_array_equal(
        build_alpha_block(None, source_valid_mask, 0, 0, 3, 1),
        np.array([[255, 0, 255]], dtype=np.uint8),
    )


def test_build_alpha_block_applies_gebco_fade_curve() -> None:
    dataset = gdal.GetDriverByName("MEM").Create("", 5, 1, 1, gdal.GDT_Float32)
    assert dataset is not None
    dataset.GetRasterBand(1).WriteArray(
        np.array([[-60.0, -50.0, -46.0, -42.0, 0.0]], dtype=np.float32)
    )

    np.testing.assert_array_equal(
        build_alpha_block(dataset.GetRasterBand(1), np.zeros((1, 5), dtype=bool), 0, 0, 5, 1),
        np.array([[0, 0, 127, 255, 255]], dtype=np.uint8),
    )
