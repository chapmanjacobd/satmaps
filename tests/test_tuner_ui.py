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
