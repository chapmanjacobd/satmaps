from pathlib import Path

import tuner_ui


def _create_sample(cache_root: Path, date_label: str, tile_prefix: str) -> None:
    sample_dir = cache_root / date_label
    sample_dir.mkdir(parents=True, exist_ok=True)
    for band in ("B02", "B03", "B04"):
        (sample_dir / f"{tile_prefix}_{band}.tif").touch()


def test_get_land_location_defaults_to_barcelona() -> None:
    location = tuner_ui.get_land_location(None)

    assert location.id == "barcelona"
    assert location.tile_prefix == "31TDF_0_0"


def test_find_land_samples_uses_requested_location(tmp_path: Path) -> None:
    _create_sample(tmp_path, "2025-01-01", "31TDF_0_0")
    _create_sample(tmp_path, "2025-07-01", "31TDF_0_0")
    _create_sample(tmp_path, "2025-01-01", "06VUN_0_0")
    _create_sample(tmp_path, "2025-07-01", "06VUN_0_0")

    samples = tuner_ui.find_land_samples(tuner_ui.LAND_LOCATIONS_BY_ID["anchorage"], tmp_path)

    assert [sample.date_label for sample in samples] == ["2025-07-01", "2025-01-01"]
    assert samples[0].paths["red"].endswith("06VUN_0_0_B04.tif")
    assert all("31TDF_0_0" not in path for sample in samples for path in sample.paths.values())


def test_land_locations_include_global_presets() -> None:
    assert set(tuner_ui.LAND_LOCATIONS_BY_ID) == {
        "barcelona",
        "anchorage",
        "shanghai",
        "singapore",
        "sao-paulo",
        "cape-town",
        "cairo",
        "mexico-city",
        "sydney",
        "nairobi",
    }

    assert tuner_ui.LAND_LOCATIONS_BY_ID["singapore"].tile_prefix == "48NUG_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["sao-paulo"].tile_prefix == "23KLP_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["cape-town"].tile_prefix == "34HBH_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["cairo"].tile_prefix == "36RUU_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["mexico-city"].tile_prefix == "14QMG_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["sydney"].tile_prefix == "56HLH_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["nairobi"].tile_prefix == "37MBU_0_0"


def test_build_land_view_defaults_to_previous_fixed_crop() -> None:
    land_view = tuner_ui.build_land_view(10980, 10980)

    assert land_view.crop_width == tuner_ui.LAND_SAMPLE_CROP_SIZE
    assert land_view.crop_height == tuner_ui.LAND_SAMPLE_CROP_SIZE
    assert land_view.xoff == tuner_ui.LAND_SAMPLE_DEFAULT_OFF_X
    assert land_view.yoff == tuner_ui.LAND_SAMPLE_DEFAULT_OFF_Y


def test_build_land_view_clamps_pan_to_tile_edges() -> None:
    land_view = tuner_ui.build_land_view(10980, 10980, pan_x=1.5, pan_y=-1.0)

    assert land_view.pan_x == 1.0
    assert land_view.pan_y == 0.0
    assert land_view.xoff == 10980 - tuner_ui.LAND_SAMPLE_CROP_SIZE
    assert land_view.yoff == 0
