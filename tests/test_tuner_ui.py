import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import satmaps
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


def test_get_land_samples_reads_cache_live(tmp_path: Path) -> None:
    assert tuner_ui.get_land_samples("barcelona", tmp_path) == ()

    _create_sample(tmp_path, "2025-07-01", "31TDF_0_0")

    samples = tuner_ui.get_land_samples("barcelona", tmp_path)

    assert [sample.date_label for sample in samples] == ["2025-07-01"]


def test_get_land_samples_prioritizes_july_and_january(tmp_path: Path) -> None:
    _create_sample(tmp_path, "2025-10-01", "31TDF_0_0")
    _create_sample(tmp_path, "2025-07-01", "31TDF_0_0")
    _create_sample(tmp_path, "2025-01-01", "31TDF_0_0")

    samples = tuner_ui.get_land_samples("barcelona", tmp_path)

    assert [sample.date_label for sample in samples] == ["2025-07-01", "2025-01-01"]


def test_land_locations_include_global_presets() -> None:
    assert set(tuner_ui.LAND_LOCATIONS_BY_ID) == {
        "barcelona",
        "anchorage",
        "banc-darguin",
        "bariloche",
        "shanghai",
        "singapore",
        "sao-paulo",
        "santiago",
        "cape-town",
        "cairo",
        "lesotho",
        "mexico-city",
        "queenstown-wanaka",
        "snowy-mountains",
        "sydney",
        "nairobi",
    }

    assert tuner_ui.LAND_LOCATIONS_BY_ID["bariloche"].tile_prefix == "19GBQ_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["singapore"].tile_prefix == "48NUG_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["sao-paulo"].tile_prefix == "23KLP_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["santiago"].tile_prefix == "19HCD_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["cape-town"].tile_prefix == "34HBH_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["cairo"].tile_prefix == "36RUU_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["banc-darguin"].tile_prefix == "28QCH_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["lesotho"].tile_prefix == "35JPJ_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["mexico-city"].tile_prefix == "14QMG_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["queenstown-wanaka"].tile_prefix == "59GLL_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["snowy-mountains"].tile_prefix == "55HFV_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["sydney"].tile_prefix == "56HLH_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["nairobi"].tile_prefix == "37MBU_0_0"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"].hemisphere == "north"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["banc-darguin"].hemisphere == "north"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["singapore"].hemisphere == "north"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["sydney"].hemisphere == "south"
    assert tuner_ui.LAND_LOCATIONS_BY_ID["nairobi"].hemisphere == "south"


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


def test_land_defaults_match_tuned_land_preset() -> None:
    defaults = tuner_ui.get_mode_defaults("land")

    assert defaults["exp"] == tuner_ui.LAND_DEFAULT_EXPOSURE
    assert defaults["sb"] == tuner_ui.tiler.SOFT_KNEE_SHADOW_BREAK
    assert defaults["hb"] == tuner_ui.tiler.SOFT_KNEE_HIGHLIGHT_BREAK
    assert defaults["ss"] == tuner_ui.tiler.SOFT_KNEE_SHADOW_SLOPE
    assert defaults["ms"] == tuner_ui.tiler.SOFT_KNEE_MID_SLOPE
    assert defaults["hs"] == tuner_ui.tiler.SOFT_KNEE_HIGHLIGHT_SLOPE
    assert defaults["gamma"] == tuner_ui.LAND_DEFAULT_GAMMA
    assert defaults["shoulder"] == tuner_ui.LAND_DEFAULT_SHOULDER
    assert defaults["sat"] == tuner_ui.LAND_DEFAULT_SATURATION
    assert defaults["db"] == tuner_ui.LAND_DEFAULT_GRADE_LOW_BREAK
    assert defaults["ghb"] == tuner_ui.LAND_DEFAULT_GRADE_HIGHLIGHT_BREAK
    assert defaults["ls"] == tuner_ui.LAND_DEFAULT_GRADE_LOW_SLOPE
    assert defaults["gms"] == tuner_ui.LAND_DEFAULT_GRADE_MID_SLOPE
    assert defaults["ghs"] == tuner_ui.LAND_DEFAULT_GRADE_HIGHLIGHT_SLOPE


def test_ocean_defaults_match_cli_defaults() -> None:
    defaults = tuner_ui.get_mode_defaults("ocean")

    assert defaults["exp"] == tuner_ui.ocean.OCEAN_DEFAULT_EXPOSURE
    assert defaults["sb"] == tuner_ui.ocean.OCEAN_DEFAULT_SHADOW_BREAK
    assert defaults["hb"] == tuner_ui.ocean.OCEAN_DEFAULT_HIGHLIGHT_BREAK
    assert defaults["ss"] == tuner_ui.ocean.OCEAN_DEFAULT_SHADOW_SLOPE
    assert defaults["ms"] == tuner_ui.ocean.OCEAN_DEFAULT_MID_SLOPE
    assert defaults["hs"] == tuner_ui.ocean.OCEAN_DEFAULT_HIGHLIGHT_SLOPE
    assert defaults["gamma"] == tuner_ui.ocean.OCEAN_DEFAULT_GAMMA
    assert defaults["shoulder"] == tuner_ui.ocean.OCEAN_DEFAULT_SHOULDER
    assert defaults["sat"] == tuner_ui.ocean.OCEAN_DEFAULT_SATURATION
    assert defaults["db"] == tuner_ui.ocean.OCEAN_DEFAULT_BLACK_BREAK
    assert defaults["ghb"] == tuner_ui.ocean.OCEAN_DEFAULT_BLACK_BREAK
    assert defaults["ls"] == tuner_ui.ocean.OCEAN_DEFAULT_BLACK_SLOPE
    assert defaults["gms"] == tuner_ui.tiler.PREVIEW_DARKEN_MID_SLOPE
    assert defaults["dmin"] == -11000.0
    assert defaults["dmax"] == 0.0


def test_get_grade_presets_returns_mode_specific_values() -> None:
    land_presets = tuner_ui.get_grade_presets("land")
    ocean_presets = tuner_ui.get_grade_presets("ocean")

    assert [preset["id"] for preset in land_presets] == ["balanced", "punchy", "matte", "vivid"]
    assert [preset["id"] for preset in ocean_presets] == ["balanced", "punchy", "muted", "glow"]
    assert land_presets[0]["values"] == tuner_ui.build_grade_preset_values(tuner_ui.get_mode_defaults("land"))
    assert ocean_presets[0]["values"] == tuner_ui.build_grade_preset_values(tuner_ui.get_mode_defaults("ocean"))


def test_get_land_season_blend_tracks_hemisphere() -> None:
    assert tuner_ui.get_land_season_blend(tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"], "summer") == 0.0
    assert tuner_ui.get_land_season_blend(tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"], "winter") == 1.0
    assert tuner_ui.get_land_season_blend(tuner_ui.LAND_LOCATIONS_BY_ID["sydney"], "summer") == 1.0
    assert tuner_ui.get_land_season_blend(tuner_ui.LAND_LOCATIONS_BY_ID["sydney"], "winter") == 0.0


def test_parse_request_params_uses_season_mode_defaults_and_explicit_blend() -> None:
    with tuner_ui.app.test_request_context("/"):
        north_summer_params = tuner_ui.parse_request_params("land", tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"], "summer")
    with tuner_ui.app.test_request_context("/"):
        south_summer_params = tuner_ui.parse_request_params(
            "land",
            tuner_ui.LAND_LOCATIONS_BY_ID["sydney"],
            "summer",
        )
    with tuner_ui.app.test_request_context("/"):
        north_winter_params = tuner_ui.parse_request_params("land", tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"], "winter")
    with tuner_ui.app.test_request_context("/?blend=0.25"):
        explicit_params = tuner_ui.parse_request_params(
            "land",
            tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"],
            "crossfade",
        )

    assert north_summer_params["blend"] == 0.0
    assert south_summer_params["blend"] == 1.0
    assert north_winter_params["blend"] == 1.0
    assert explicit_params["blend"] == 0.25


def test_get_neutral_grade_values_are_identity_like() -> None:
    assert tuner_ui.get_neutral_grade_values() == {
        "exp": 1.0,
        "gamma": 1.0,
        "shoulder": 1.0,
        "sat": 1.0,
        "db": 0.0,
        "ghb": 1.0,
        "ls": 1.0,
        "gms": 1.0,
        "ghs": 1.0,
    }


def test_get_output_histogram_data_uses_rgb_luma_for_rendered_output() -> None:
    rendered = tuner_ui.np.array(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=tuner_ui.np.float32,
    )

    hist = tuner_ui.get_output_histogram_data(rendered)

    assert hist[0] == 1.0
    assert hist[-1] == 1.0
    assert sum(value > 0 for value in hist) == 2


def test_index_exposes_shoulder_control() -> None:
    client = tuner_ui.app.test_client()

    response = client.get("/")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Shoulder" in html
    assert "--shoulder" in html
    assert "applyGradePreset" in html
    assert 'onclick=\'applyGradePreset("balanced")\'' in html
    assert '"exp": 1.0' in html
    assert "Balanced" in html
    assert "Punchy" in html
    assert "Matte" in html
    assert "Vivid" in html
    assert "Summer" in html
    assert "Winter" in html


def test_index_sets_up_histogram_hold_preview() -> None:
    client = tuner_ui.app.test_client()

    response = client.get("/")

    html = response.get_data(as_text=True)
    assert 'id="fg_hist_label"' in html
    assert "FINAL OUTPUT (hold for input)" in html
    assert "function setupHistogramHoldPreview()" in html
    assert "setHistogramPreview(true);" in html
    assert "finalGradeViz.setPointerCapture(event.pointerId);" in html
    assert "finalGradeViz.addEventListener('pointercancel', endPreview);" in html
    assert "inputHist = payload.raw_hist || [];" in html
    assert "finalHist = payload.final_hist || [];" in html


def test_index_uses_season_blend_modes(monkeypatch: object) -> None:
    client = tuner_ui.app.test_client()
    samples = (
        tuner_ui.LandSample("2025-07-01", {"red": "r", "green": "g", "blue": "b"}),
        tuner_ui.LandSample("2025-01-01", {"red": "r2", "green": "g2", "blue": "b2"}),
    )
    monkeypatch.setattr(tuner_ui, "find_land_samples", lambda location, cache_root=tuner_ui.LAND_CACHE_ROOT: samples)
    monkeypatch.setattr(
        tuner_ui,
        "get_land_view",
        lambda location_id, pan_x=None, pan_y=None: tuner_ui.build_land_view(10980, 10980, pan_x, pan_y),
    )
    monkeypatch.setattr(tuner_ui, "get_land_source", lambda *args, **kwargs: None)

    north_response = client.get("/?loc=barcelona")
    south_response = client.get("/?loc=sydney")
    winter_response = client.get("/?loc=barcelona&blend_mode=winter")

    north_html = north_response.get_data(as_text=True)
    south_html = south_response.get_data(as_text=True)
    winter_html = winter_response.get_data(as_text=True)

    assert 'Winter Season Swap' not in north_html
    assert '<option value="summer" selected' in north_html
    assert '<input type="range" id="blend" min="0" max="1" step="0.01" value="0.0">' in north_html
    assert '<input type="range" id="blend" min="0" max="1" step="0.01" value="1.0">' in south_html
    assert '<option value="winter" selected' in winter_html
    assert '<input type="range" id="blend" min="0" max="1" step="0.01" value="1.0">' in winter_html


def test_index_shows_sample_download_hint_when_cache_is_empty(monkeypatch: object) -> None:
    client = tuner_ui.app.test_client()
    monkeypatch.setattr(
        tuner_ui,
        "find_land_samples",
        lambda location, cache_root=tuner_ui.LAND_CACHE_ROOT: (),
    )

    response = client.get("/")

    html = response.get_data(as_text=True)
    assert "satmaps-tuner --download-samples" in html
    assert "then refresh" in html


def test_index_preserves_land_controls_when_switching_locations() -> None:
    client = tuner_ui.app.test_client()

    response = client.get("/")

    html = response.get_data(as_text=True)
    assert "function appendLandParams(params, locationId = currentLandLocation, includeBlend = true)" in html
    assert "function getSeasonalBlendForMode(blendModeId)" in html
    assert "landViewControls.forEach(id => params.set(id, document.getElementById(id).value));" in html
    assert "sharedControls.forEach(id => params.set(id, document.getElementById(id).value));" in html
    assert "appendLandParams(params, locationId, false);" in html
    assert "document.getElementById('blend').value = getSeasonalBlendForMode(defaultLandBlendMode);" in html
    assert "fg: document.getElementById('fg_on').checked ? '1' : '0'" in html
    assert "getBlendMode() === 'winter'" in html


def test_index_respects_fg_query_param() -> None:
    client = tuner_ui.app.test_client()

    response = client.get("/?fg=0")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert '<input type="checkbox" id="fg_on" >' in html
    assert '<input type="checkbox" id="fg_on" checked>' not in html


def test_histogram_route_returns_input_and_final_histograms(monkeypatch: object) -> None:
    client = tuner_ui.app.test_client()
    sample = tuner_ui.np.array(
        [
            [[0.02, 0.15, 0.45, 0.9]],
            [[0.02, 0.15, 0.45, 0.9]],
            [[0.02, 0.15, 0.45, 0.9]],
        ],
        dtype=tuner_ui.np.float32,
    )
    monkeypatch.setattr(tuner_ui, "get_land_source", lambda *args, **kwargs: sample)

    response = client.get("/histogram?mode=land")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload is not None
    assert set(payload) == {"raw_hist", "final_hist"}
    assert payload["raw_hist"] != payload["final_hist"]


def test_histogram_route_respects_fg_query_param(monkeypatch: object) -> None:
    client = tuner_ui.app.test_client()
    sample = tuner_ui.np.array(
        [
            [[0.02, 0.15, 0.45, 0.9]],
            [[0.02, 0.15, 0.45, 0.9]],
            [[0.02, 0.15, 0.45, 0.9]],
        ],
        dtype=tuner_ui.np.float32,
    )
    monkeypatch.setattr(tuner_ui, "get_land_source", lambda *args, **kwargs: sample)

    ungraded_response = client.get("/histogram?mode=land&fg=0")
    graded_response = client.get("/histogram?mode=land&fg=1")

    ungraded_payload = ungraded_response.get_json()
    graded_payload = graded_response.get_json()
    assert ungraded_response.status_code == 200
    assert graded_response.status_code == 200
    assert ungraded_payload is not None
    assert graded_payload is not None
    assert ungraded_payload["raw_hist"] == graded_payload["raw_hist"]
    assert ungraded_payload["final_hist"] != graded_payload["final_hist"]


def test_get_land_blend_mode_defaults_to_summer() -> None:
    blend_mode = tuner_ui.get_land_blend_mode(None)

    assert blend_mode.id == "summer"


def test_download_configured_land_samples_uses_satmaps_helpers(monkeypatch: object) -> None:
    setup_calls: list[bool] = []
    download_calls: list[tuple[tuple[satmaps.LandWorkUnit, ...], list[str], str, int]] = []

    monkeypatch.setattr(satmaps, "DEFAULT_DATE_PATHS", "2025/07/01,2025/01/01")
    monkeypatch.setattr(satmaps, "setup_gdal_cdse", lambda: setup_calls.append(True))
    monkeypatch.setattr(
        satmaps,
        "download_source_tiles_to_cache",
        lambda work_units, date_paths, cache_dir, parallel: (
            download_calls.append((tuple(work_units), list(date_paths), cache_dir, parallel)) or 5
        ),
    )

    downloaded = tuner_ui.download_configured_land_samples(cache_dir=".cache", parallel=7)

    assert downloaded == 5
    assert setup_calls == [True]
    captured_units, captured_dates, cache_dir, parallel = download_calls[0]
    assert [work_unit.unit_id for work_unit in captured_units] == list(tuner_ui.get_land_sample_tile_prefixes())
    assert all(work_unit.source_subtiles == (work_unit.unit_id,) for work_unit in captured_units)
    assert captured_dates == ["2025/07/01", "2025/01/01"]
    assert cache_dir == ".cache"
    assert parallel == 7


def test_download_configured_land_samples_raises_when_nothing_is_downloaded(monkeypatch: object) -> None:
    monkeypatch.setattr(satmaps, "DEFAULT_DATE_PATHS", "2025/07/01")
    monkeypatch.setattr(satmaps, "setup_gdal_cdse", lambda: None)
    monkeypatch.setattr(satmaps, "download_source_tiles_to_cache", lambda *args, **kwargs: 0)

    try:
        tuner_ui.download_configured_land_samples()
    except RuntimeError as exc:
        assert "No source tiles were downloaded into .cache" in str(exc)
    else:
        raise AssertionError("expected download_configured_land_samples to raise RuntimeError")


def test_main_download_samples_downloads_and_exits(monkeypatch: object, capsys: object) -> None:
    called: dict[str, object] = {}

    def fake_download_configured_land_samples(
        *,
        date_arg: str | None = None,
        cache_dir: str = ".cache",
        parallel: int = tuner_ui.DEFAULT_SAMPLE_DOWNLOAD_PARALLEL,
    ) -> int:
        called.update(
            {
                "date_arg": date_arg,
                "cache_dir": cache_dir,
                "parallel": parallel,
            }
        )
        return 5

    monkeypatch.setattr(tuner_ui, "download_configured_land_samples", fake_download_configured_land_samples)
    monkeypatch.setattr(tuner_ui.app, "run", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected app.run")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tuner_ui.py", "--download-samples", "--date", "2025/10/01,2025/07/01", "--parallel", "3"],
    )

    tuner_ui.main()

    assert called == {
        "date_arg": "2025/10/01,2025/07/01",
        "cache_dir": ".cache",
        "parallel": 3,
    }
    assert "Download complete. Cached 5 folder(s) for the tuner." in capsys.readouterr().out


def test_blend_land_samples_supports_all_requested_modes() -> None:
    primary = tuner_ui.np.zeros((3, 2, 4), dtype=tuner_ui.np.float32)
    secondary = tuner_ui.np.ones((3, 2, 4), dtype=tuner_ui.np.float32)
    samples = (primary, secondary)
    north_location = tuner_ui.LAND_LOCATIONS_BY_ID["barcelona"]
    south_location = tuner_ui.LAND_LOCATIONS_BY_ID["sydney"]

    summer_north = tuner_ui.blend_land_samples(north_location, samples, 0.25, "summer")
    winter_north = tuner_ui.blend_land_samples(north_location, samples, 0.25, "winter")
    summer_south = tuner_ui.blend_land_samples(south_location, samples, 0.25, "summer")
    winter_south = tuner_ui.blend_land_samples(south_location, samples, 0.25, "winter")
    crossfade = tuner_ui.blend_land_samples(north_location, samples, 0.25, "crossfade")
    difference = tuner_ui.blend_land_samples(north_location, samples, 0.25, "difference")
    lighten = tuner_ui.blend_land_samples(north_location, samples, 0.25, "lighten")
    darken = tuner_ui.blend_land_samples(north_location, samples, 0.25, "darken")
    swipe = tuner_ui.blend_land_samples(north_location, samples, 0.25, "swipe")

    assert summer_north is not None
    assert winter_north is not None
    assert summer_south is not None
    assert winter_south is not None
    assert crossfade is not None
    assert difference is not None
    assert lighten is not None
    assert darken is not None
    assert swipe is not None
    assert tuner_ui.np.allclose(summer_north, 0.0)
    assert tuner_ui.np.allclose(winter_north, 1.0)
    assert tuner_ui.np.allclose(summer_south, 1.0)
    assert tuner_ui.np.allclose(winter_south, 0.0)
    assert tuner_ui.np.allclose(crossfade, 0.25)
    assert tuner_ui.np.allclose(difference, 1.0)
    assert tuner_ui.np.allclose(lighten, 1.0)
    assert tuner_ui.np.allclose(darken, 0.0)
    assert tuner_ui.np.allclose(swipe[:, :, :3], 0.0)
    assert tuner_ui.np.allclose(swipe[:, :, 3:], 1.0)
