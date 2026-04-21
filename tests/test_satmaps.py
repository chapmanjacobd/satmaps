import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocean_background
import satmaps
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
    ocean_background.create_gebco_ocean_vrt(str(source_vrt), str(ocean_vrt))

    ds = gdal.Open(str(ocean_vrt))
    assert ds is not None
    arr = ds.ReadAsArray()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    assert nodata == -32767.0

    np.testing.assert_allclose(arr, np.array([[-5.0, 0.0, 0.0005, nodata]], dtype=np.float32))


def test_target_web_mercator_pixel_size_uses_output_zoom() -> None:
    assert ocean_background.target_web_mercator_pixel_size() == pytest.approx(
        satmaps.tiler.web_mercator_pixel_size(ocean_background.OUTPUT_WEB_MERCATOR_ZOOM)
    )


def test_snapped_tile_grid_for_bbox_expands_to_tile_pixel_grid() -> None:
    bbox = (-4.0, 50.0, -3.0, 51.0)

    snapped_bounds, pixel_size, zoom = ocean_background.snapped_tile_grid_for_bbox(bbox)

    assert zoom == ocean_background.OUTPUT_WEB_MERCATOR_ZOOM
    assert pixel_size == pytest.approx(ocean_background.target_web_mercator_pixel_size())

    mercator_bounds = satmaps.tiler.lonlat_bbox_to_mercator_bounds(*bbox)
    assert snapped_bounds[0] <= mercator_bounds[0]
    assert snapped_bounds[1] >= mercator_bounds[1]
    assert snapped_bounds[2] >= mercator_bounds[2]
    assert snapped_bounds[3] <= mercator_bounds[3]


def test_create_alpha_vrt_masks_nodata_with_expression(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")

    alpha_source = tmp_path / "ocean_source.tif"
    alpha_ds = driver.Create(str(alpha_source), 2, 1, 1, gdal.GDT_Float32)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).SetNoDataValue(-32767.0)
    alpha_ds.GetRasterBand(1).WriteArray(np.array([[-5.0, -32767.0]], dtype=np.float32))
    alpha_ds = None

    alpha_source_vrt = tmp_path / "ocean_source.vrt"
    gdal.BuildVRT(str(alpha_source_vrt), [str(alpha_source)])

    alpha_vrt = tmp_path / "alpha.vrt"
    ocean_background.create_alpha_vrt(str(alpha_source_vrt), str(alpha_vrt))

    np.testing.assert_allclose(
        gdal.Open(str(alpha_vrt)).ReadAsArray(),
        np.array([[255.0, 0.0]], dtype=np.float32),
    )


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
    ocean_background.create_ocean_rgb_tif(
        str(depth_path),
        str(hillshade_path),
        str(output_path),
        ocean_background.OceanStyleOptions(tonemap=False, grade=False),
    )

    rgb_ds = gdal.Open(str(output_path))
    assert rgb_ds is not None
    arr = rgb_ds.ReadAsArray()
    assert arr.shape == (3, 2, 2)
    assert arr.dtype == np.uint8

    style = ocean_background.OceanStyleOptions(tonemap=False, grade=False)
    expected_rgb = ocean_background.colorize_ocean_depths(
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


def test_build_hillshade_command_matches_expected_flags(tmp_path: Path) -> None:
    command = ocean_background.build_hillshade_command(
        str(tmp_path / "in.vrt"),
        str(tmp_path / "out.tif"),
        z_factor=5.0,
    )

    assert command[:7] == [
        "gdaldem",
        "hillshade",
        str(tmp_path / "in.vrt"),
        str(tmp_path / "out.tif"),
        "-multidirectional",
        "-z",
        "5.0",
    ]
    assert "BIGTIFF=YES" in command
    assert "COMPRESS=ZSTD" in command


def test_ocean_background_main_uses_default_positionals(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean_background.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif=str(kwargs["destination"]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ocean_background.py"],
    )
    monkeypatch.setattr(
        "ocean_background.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean_background.main()

    assert called["gebco_zip"] == ocean_background.DEFAULT_GEBCO_ZIP
    assert called["destination"] == ocean_background.DEFAULT_OUTPUT
    assert called["bbox"] is None
    assert called["vrt"] is False
    assert called["style"] == ocean_background.OceanStyleOptions(
        gamma=1.2,
        saturation=1.0,
        black_break=0.35,
        black_slope=0.35,
    )


def test_ocean_background_main_parses_bbox_when_provided(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean_background.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif=str(kwargs["destination"]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ocean_background.py", "--bbox", "0,0,1,1"],
    )
    monkeypatch.setattr(
        "ocean_background.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean_background.main()

    assert called["bbox"] == (0.0, 0.0, 1.0, 1.0)


def test_ocean_background_main_enables_vrt_mode(monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def fake_generate_ocean_background(**kwargs):
        called.update(kwargs)
        return ocean_background.OceanBackgroundArtifacts(
            source_vrt=".temp/source.vrt",
            masked_vrt=".temp/masked.vrt",
            warped_vrt=".temp/warped.vrt",
            alpha_vrt=".temp/alpha.vrt",
            hillshade_tif=".temp/ocean_hillshade.tif",
            color_tif=".temp/ocean_color.tif",
            rgba_vrt=".temp/ocean_rgba.vrt",
            output_tif="ocean.vrt",
        )

    monkeypatch.setattr(sys, "argv", ["ocean_background.py", "--vrt"])
    monkeypatch.setattr(
        "ocean_background.generate_ocean_background",
        fake_generate_ocean_background,
    )

    ocean_background.main()

    assert called["vrt"] is True


def test_generate_ocean_background_without_bbox_warps_global_raster(
    monkeypatch: object,
) -> None:
    commands: list[list[str]] = []
    translated: list[tuple[str, str]] = []
    warp_calls: list[tuple[str, str, object]] = []
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr("ocean_background.which", lambda cmd: "/usr/bin/gdaldem")
    monkeypatch.setattr("ocean_background.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ocean_background.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_alpha_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_ocean_rgb_tif",
        lambda depth_vrt, hillshade_tif, output_tif, style: output_tif,
    )
    monkeypatch.setattr(
        "ocean_background.create_rgb_with_alpha_vrt",
        lambda rgb_tif, alpha_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.translate_rgba_vrt",
        lambda rgba_vrt, destination: translated.append((rgba_vrt, destination)) or destination,
    )
    monkeypatch.setattr(
        "ocean_background.build_hillshade_command",
        lambda input_vrt, output_tif, z_factor: [input_vrt, output_tif, str(z_factor)],
    )
    monkeypatch.setattr(
        "ocean_background.subprocess.run",
        lambda cmd, check: commands.append(cmd),
    )
    monkeypatch.setattr(
        "ocean_background.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "ocean_background.gdal.Warp",
        lambda destination, source, options=None: warp_calls.append(
            (destination, source, options)
        ),
    )

    artifacts = ocean_background.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        bbox=None,
    )

    assert artifacts.masked_vrt.endswith("ocean_masked.vrt")
    assert artifacts.warped_vrt.endswith("ocean_3857.vrt")
    assert len(warp_calls) == 1
    assert warp_calls[0][0] == artifacts.warped_vrt
    assert warp_calls[0][1] == artifacts.masked_vrt
    assert warp_options_calls[0]["outputBounds"] == ocean_background.WEB_MERCATOR_WORLD_BOUNDS
    assert warp_options_calls[0]["xRes"] == pytest.approx(
        ocean_background.target_web_mercator_pixel_size()
    )
    assert warp_options_calls[0]["yRes"] == pytest.approx(
        ocean_background.target_web_mercator_pixel_size()
    )
    assert commands == [[artifacts.warped_vrt, artifacts.hillshade_tif, "5.0"]]
    assert translated == [(artifacts.rgba_vrt, "ocean.tif")]


def test_generate_ocean_background_with_bbox_sets_explicit_target_resolution(
    monkeypatch: object,
) -> None:
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr("ocean_background.which", lambda cmd: "/usr/bin/gdaldem")
    monkeypatch.setattr("ocean_background.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ocean_background.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_alpha_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_ocean_rgb_tif",
        lambda depth_vrt, hillshade_tif, output_tif, style: output_tif,
    )
    monkeypatch.setattr(
        "ocean_background.create_rgb_with_alpha_vrt",
        lambda rgb_tif, alpha_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr("ocean_background.translate_rgba_vrt", lambda rgba_vrt, destination: destination)
    monkeypatch.setattr(
        "ocean_background.build_hillshade_command",
        lambda input_vrt, output_tif, z_factor: [input_vrt, output_tif, str(z_factor)],
    )
    monkeypatch.setattr("ocean_background.subprocess.run", lambda cmd, check: None)
    monkeypatch.setattr(
        "ocean_background.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        "ocean_background.gdal.Warp",
        lambda destination, source, options=None: None,
    )

    bbox = (-4.0, 50.0, -3.0, 51.0)
    snapped_bounds, pixel_size, _zoom = ocean_background.snapped_tile_grid_for_bbox(bbox)
    ocean_background.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        bbox=bbox,
    )

    assert warp_options_calls[0]["outputBounds"] == pytest.approx(snapped_bounds)
    assert "outputBoundsSRS" not in warp_options_calls[0]
    assert warp_options_calls[0]["xRes"] == pytest.approx(pixel_size)
    assert warp_options_calls[0]["yRes"] == pytest.approx(pixel_size)


def test_warp_to_web_mercator_uses_shared_zoom13_resolution(monkeypatch: object) -> None:
    warp_options_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "satmaps.gdal.WarpOptions",
        lambda **kwargs: warp_options_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr("satmaps.gdal.Warp", lambda destination, source, options=None: None)

    satmaps.warp_to_web_mercator("input.tif", "output.tif", "lanczos")

    assert warp_options_calls[0]["dstSRS"] == "EPSG:3857"
    assert warp_options_calls[0]["xRes"] == pytest.approx(
        ocean_background.target_web_mercator_pixel_size()
    )
    assert warp_options_calls[0]["yRes"] == pytest.approx(
        ocean_background.target_web_mercator_pixel_size()
    )


def test_generate_ocean_background_vrt_mode_skips_translate(monkeypatch: object) -> None:
    vrt_outputs: list[tuple[str, str]] = []

    monkeypatch.setattr("ocean_background.which", lambda cmd: "/usr/bin/gdaldem")
    monkeypatch.setattr("ocean_background.os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ocean_background.build_gebco_source_vrt",
        lambda gebco_zip, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_gebco_ocean_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_alpha_vrt",
        lambda source_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.create_ocean_rgb_tif",
        lambda depth_vrt, hillshade_tif, output_tif, style: output_tif,
    )
    monkeypatch.setattr(
        "ocean_background.create_rgb_with_alpha_vrt",
        lambda rgb_tif, alpha_vrt, output_vrt: output_vrt,
    )
    monkeypatch.setattr(
        "ocean_background.write_rgba_vrt",
        lambda rgba_vrt, destination: vrt_outputs.append((rgba_vrt, destination)) or destination,
    )
    monkeypatch.setattr(
        "ocean_background.translate_rgba_vrt",
        lambda rgba_vrt, destination: (_ for _ in ()).throw(AssertionError("Translate should not be called")),
    )
    monkeypatch.setattr(
        "ocean_background.build_hillshade_command",
        lambda input_vrt, output_tif, z_factor: [input_vrt, output_tif, str(z_factor)],
    )
    monkeypatch.setattr("ocean_background.subprocess.run", lambda cmd, check: None)
    monkeypatch.setattr("ocean_background.gdal.WarpOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        "ocean_background.gdal.Warp",
        lambda destination, source, options=None: None,
    )

    artifacts = ocean_background.generate_ocean_background(
        gebco_zip="gebco.zip",
        destination="ocean.tif",
        vrt=True,
    )

    assert artifacts.output_tif == "ocean.vrt"
    assert vrt_outputs == [(artifacts.rgba_vrt, "ocean.vrt")]


def test_build_ocean_ramp_colors_respects_style_flags() -> None:
    default_colors = ocean_background.build_ocean_ramp_colors(
        ocean_background.OceanStyleOptions()
    )
    ungraded_colors = ocean_background.build_ocean_ramp_colors(
        ocean_background.OceanStyleOptions(tonemap=False, grade=False, exposure=0.5)
    )

    assert default_colors.shape == (len(ocean_background.MAKO_RAMP), 3)
    np.testing.assert_allclose(
        ungraded_colors[0],
        np.array(ocean_background.MAKO_RAMP[0][1:], dtype=np.float32) / 255.0,
    )
    assert np.all((default_colors >= 0.0) & (default_colors <= 1.0))
    assert np.all((ungraded_colors >= 0.0) & (ungraded_colors <= 1.0))


def test_process_single_tile_full_pipeline(monkeypatch: object, tmp_path: Path) -> None:
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
    def fake_get_tile_paths(folder, date, cache, download=False):
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
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, "abc", None)

    assert out_path is not None
    assert Path(out_path).exists()

    ds = gdal.Open(out_path)
    assert ds.RasterCount == 4
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    assert ds.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_AlphaBand
    # Check projection is 3857
    srs = osr.SpatialReference(ds.GetProjection())
    assert srs.GetAttrValue("AUTHORITY", 1) == "3857"


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

    def fake_get_tile_paths(folder, date, cache, download=False):
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
        str(gebco_path), raster_size, raster_size, 1, gdal.GDT_Float32
    )
    gebco_ds.SetGeoTransform((0, pixel_size, 0, raster_size * pixel_size, 0, -pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    gebco_ds.SetProjection(srs.ExportToWkt())
    gebco_values = np.full((raster_size, raster_size), 0.0, dtype=np.float32)
    gebco_values[0, 0] = -60.0
    gebco_values[0, 1] = -45.0
    gebco_ds.GetRasterBand(1).WriteArray(gebco_values)
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
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, "gebco", str(gebco_path))

    assert out_path is not None
    out_ds = gdal.Open(out_path)
    assert out_ds is not None
    alpha = out_ds.GetRasterBand(4).ReadAsArray()
    assert np.any(alpha == 0)
    assert np.any(alpha == 255)
    assert np.any((alpha > 0) & (alpha < 255))


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

    def fake_get_tile_paths(folder, date, cache, download=False):
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
    )

    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, "ocean", str(ocean_path))

    assert out_path is not None
    out_ds = gdal.Open(out_path)
    assert out_ds is not None
    alpha = out_ds.GetRasterBand(4).ReadAsArray()
    assert np.any(alpha == 255)
    assert np.any(alpha == 0)


def test_main_vrt_mode(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "31TDF", "--vrt", "--parallel", "1", "--date", "2025/07/01"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)

    # Mock process_single_tile to return a fake path
    def fake_process_single_tile(st, dates, args, uid, gebco_src=None):
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


def test_main_passes_ocean_background_path_to_tile_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    custom_ocean_path = tmp_path / "custom-ocean.tif"
    custom_ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(custom_ocean_path), 1, 1, 1, gdal.GDT_Byte
    )
    assert custom_ocean_ds is not None
    custom_ocean_ds.GetRasterBand(1).Fill(1)
    custom_ocean_ds = None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "satmaps.py",
            "31TDF",
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
    gebco_sources: list[str | None] = []

    def fake_process_single_tile(st, dates, args, uid, gebco_src=None):
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


def test_main_uses_rgba_ocean_background_as_alpha_mask_source(
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
            "31TDF",
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
    gebco_sources: list[str | None] = []

    def fake_process_single_tile(st, dates, args, uid, gebco_src=None):
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


def test_main_bbox_uses_standalone_ocean_background(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_background_path = tmp_path / "ocean.tif"
    ocean_background_path.write_text("fake ocean")

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
    assert Path(buildvrt_sources[-1][0]).name.startswith("ocean_")
    assert Path(buildvrt_sources[-1][0]).name.endswith("_bbox.tif")
    snapped_bounds, pixel_size, _zoom = ocean_background.snapped_tile_grid_for_bbox(
        (0.0, 0.0, 1.0, 1.0)
    )
    assert warp_options_calls[0]["outputBounds"] == pytest.approx(snapped_bounds)
    assert warp_options_calls[0]["xRes"] == pytest.approx(pixel_size)
    assert warp_options_calls[0]["yRes"] == pytest.approx(pixel_size)


def test_main_builds_master_vrt_with_shared_zoom13_resolution(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_background_path = tmp_path / "ocean.tif"
    ocean_background_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--no-land", "--vrt", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    buildvrt_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_calls.append((list(src), kwargs))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    assert buildvrt_calls
    _, kwargs = buildvrt_calls[-1]
    assert kwargs["resolution"] == "user"
    assert kwargs["xRes"] == pytest.approx(ocean_background.target_web_mercator_pixel_size())
    assert kwargs["yRes"] == pytest.approx(ocean_background.target_web_mercator_pixel_size())


def test_main_bbox_passes_chunk_bounds_to_tiler(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_background_path = tmp_path / "ocean.tif"
    ocean_ds = gdal.GetDriverByName("GTiff").Create(
        str(ocean_background_path), 1, 1, 1, gdal.GDT_Byte
    )
    assert ocean_ds is not None
    ocean_ds.GetRasterBand(1).Fill(1)
    ocean_ds = None

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "--bbox", "0,0,1,1", "--no-land", "--parallel", "1"],
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
    monkeypatch.setattr("satmaps.subprocess.run", lambda cmd, check: None)

    main()

    assert captured_options["chunk_bounds"] == satmaps.tiler.lonlat_bbox_to_mercator_bounds(
        0.0, 0.0, 1.0, 1.0
    )


def test_main_non_bbox_can_use_standalone_ocean_background(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    (tmp_path / "ocean.tif").write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "31TDF", "--no-land", "--vrt", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    buildvrt_sources: list[list[str]] = []

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_sources.append(list(src))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
    assert buildvrt_sources[-1] == ["ocean.tif"]


def test_main_keeps_ocean_background_after_processing(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    ocean_background_path = tmp_path / "ocean.tif"
    ocean_background_path.write_text("fake ocean")

    monkeypatch.setattr(
        sys,
        "argv",
        ["satmaps.py", "31TDF", "--no-land", "--parallel", "1"],
    )
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda out, src, **kwargs: Path(out).write_text("fake vrt"),
    )

    def fake_run_tiling_simplified(input_vrt: str, output_mbtiles: str, options: dict[str, object]):
        Path(output_mbtiles).write_text("fake mbtiles")
        return satmaps.tiler.TilingArtifacts(final_vrt=input_vrt, cleanup_paths=[])

    monkeypatch.setattr("satmaps.tiler.run_tiling_simplified", fake_run_tiling_simplified)
    monkeypatch.setattr("satmaps.subprocess.run", lambda cmd, check: None)

    main()

    assert ocean_background_path.exists()
