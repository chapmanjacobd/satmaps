import argparse
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocean_background
import satmaps
from satmaps import (
    fill_nan_nearest,
    get_tile_paths,
    list_mosaic_folders_for_tile,
    main,
    mosaic_date_stacks,
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


def test_mosaic_date_stacks_prefers_any_complete_date_over_neighbor_fill() -> None:
    date1 = np.array(
        [
            [[10.0, np.nan, np.nan]],
            [[20.0, np.nan, np.nan]],
            [[30.0, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    date2 = np.array(
        [
            [[np.nan, 40.0, np.nan]],
            [[50.0, 50.0, np.nan]],
            [[60.0, 60.0, np.nan]],
        ],
        dtype=np.float32,
    )

    averaged, valid_mask = mosaic_date_stacks([date1, date2])

    expected = np.array(
        [
            [[10.0, 40.0, np.nan]],
            [[20.0, 50.0, np.nan]],
            [[30.0, 60.0, np.nan]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(averaged[:, :, :2], expected[:, :, :2])
    assert np.isnan(averaged[:, :, 2]).all()
    np.testing.assert_array_equal(valid_mask, np.array([[True, True, False]]))


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


def test_resolution_for_utm_10m_3857_returns_positive_values() -> None:
    x_res, y_res = ocean_background.resolution_for_utm_10m_3857(-4.0, 50.0, -3.0, 51.0)

    assert x_res > 0.0
    assert y_res > 0.0
    assert np.isclose(x_res, y_res, rtol=0.05)


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


def test_create_hillshade_rgba_vrt_repeats_gray_and_uses_alpha_vrt(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")

    hillshade_path = tmp_path / "hillshade.tif"
    hillshade_ds = driver.Create(str(hillshade_path), 2, 1, 1, gdal.GDT_Byte)
    hillshade_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    hillshade_ds.SetProjection(srs.ExportToWkt())
    hillshade_ds.GetRasterBand(1).WriteArray(np.array([[10, 20]], dtype=np.uint8))
    hillshade_ds = None

    alpha_path = tmp_path / "alpha.tif"
    alpha_ds = driver.Create(str(alpha_path), 2, 1, 1, gdal.GDT_Byte)
    alpha_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    alpha_ds.SetProjection(srs.ExportToWkt())
    alpha_ds.GetRasterBand(1).WriteArray(np.array([[255, 0]], dtype=np.uint8))
    alpha_ds = None

    alpha_vrt = tmp_path / "alpha.vrt"
    gdal.BuildVRT(str(alpha_vrt), [str(alpha_path)])

    rgba_vrt = tmp_path / "ocean_rgba.vrt"
    ocean_background.create_hillshade_rgba_vrt(str(hillshade_path), str(alpha_vrt), str(rgba_vrt))

    rgba_ds = gdal.Open(str(rgba_vrt))
    assert rgba_ds is not None
    assert rgba_ds.RasterCount == 4
    np.testing.assert_array_equal(
        rgba_ds.ReadAsArray(),
        np.array([[[10, 20]], [[10, 20]], [[10, 20]], [[255, 0]]], dtype=np.uint8),
    )


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

    assert called["gebco_zip"] == ocean_background.DEFAULT_GEBCO_ZIP
    assert called["destination"] == ocean_background.DEFAULT_OUTPUT
    assert called["bbox"] == (0.0, 0.0, 1.0, 1.0)


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

    def fake_build_vrt(out, src, **kwargs):
        buildvrt_sources.append(list(src))
        Path(out).write_text("fake vrt")

    monkeypatch.setattr("satmaps.gdal.BuildVRT", fake_build_vrt)

    main()

    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
    assert buildvrt_sources[-1][0] == "ocean.tif"


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
