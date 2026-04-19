from argparse import Namespace
import sqlite3
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmaps import calculate_scaling_params, create_rgb_vrt, main, merge_date_vrts, prepare_group_sources, process_date
from tiler import (
    TilingArtifacts,
    create_byte_conversion_vrt,
    create_color_corrected_vrt,
    get_dataset_bounds,
    process_chunk,
    run_tiling,
)


def test_prepare_group_sources_collects_parallel_results(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    def fake_get_tile_paths(folder_name: str, date_path: str, cache_dir: str | None) -> tuple[dict[str, str], int, int]:
        return {"red": f"{folder_name}_r", "green": f"{folder_name}_g", "blue": f"{folder_name}_b"}, 1, 2

    def fake_create_rgb_vrt(band_paths: dict[str, str], output_vrt: str) -> None:
        Path(output_vrt).write_text("|".join(band_paths.values()))

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)
    monkeypatch.setattr("satmaps.create_rgb_vrt", fake_create_rgb_vrt)

    tile_vrts, cached, downloaded = prepare_group_sources(
        ["31TDF_0_0", "31TDF_1_0"],
        "2025/07/01",
        ".cache",
        "abc123",
        False,
        4,
    )

    assert [Path(path).name for path in tile_vrts] == [
        "step1_tile_31TDF_0_0_abc123.vrt",
        "step1_tile_31TDF_1_0_abc123.vrt",
    ]
    assert cached == 2
    assert downloaded == 4


def test_process_date_builds_group_vrts_from_prepared_sources(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    monkeypatch.setattr("satmaps.tiler.apply_rgb_color_interpretation_to_vrt", lambda path: None)

    build_calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders",
        lambda date, mgrs_filter=None, cache_dir=None: ["tile_a", "tile_b"],
    )
    monkeypatch.setattr(
        "satmaps.prepare_group_sources",
        lambda folders, date, cache_dir, unique_id, download_only, max_workers: (
            [f".temp/step1_tile_{folder}_{unique_id}.vrt" for folder in folders],
            3,
            6,
        ),
    )
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda output, sources, srcNodata=None, VRTNodata=None: build_calls.append((output, tuple(sources))),
    )

    args = Namespace(
        cache=".cache",
        all_tiles=False,
        mgrs="31TDF",
        download_only=False,
        processes=4,
    )

    date_vrts, tile_vrts = process_date("2025/07/01", args, "abc123", None)

    assert date_vrts == [".temp/step3_mosaic_2025-07-01_abc123.vrt"]
    assert tile_vrts == [
        ".temp/step1_tile_tile_a_abc123.vrt",
        ".temp/step1_tile_tile_b_abc123.vrt",
        ".temp/step2_group_2025-07-01_0_abc123.vrt",
    ]
    assert build_calls == [
        (
            ".temp/step2_group_2025-07-01_0_abc123.vrt",
            (".temp/step1_tile_tile_a_abc123.vrt", ".temp/step1_tile_tile_b_abc123.vrt"),
        ),
        (
            ".temp/step3_mosaic_2025-07-01_abc123.vrt",
            (".temp/step2_group_2025-07-01_0_abc123.vrt",),
        ),
    ]


def test_final_byte_vrt_chunk_packaging_produces_tiles(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tif"
    color_vrt = tmp_path / "color.vrt"
    byte_vrt = tmp_path / "byte.vrt"
    chunk_path = tmp_path / "chunk.mbtiles"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 256, 256, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 1000.0, 0.0, 256000.0, 0.0, -1000.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    for band_index, value in enumerate((1500, 2500, 3500), start=1):
        band = dataset.GetRasterBand(band_index)
        band.Fill(value)
        band.SetNoDataValue(0)
    dataset = None

    create_color_corrected_vrt(
        str(input_path),
        str(color_vrt),
        [[0, 4000, 0, 255]] * 3,
    )
    create_byte_conversion_vrt(str(color_vrt), str(byte_vrt))

    color_dataset = gdal.Open(str(color_vrt))
    assert color_dataset is not None
    assert [
        gdal.GetColorInterpretationName(color_dataset.GetRasterBand(index).GetColorInterpretation())
        for index in range(1, 4)
    ] == ["Red", "Green", "Blue"]

    byte_dataset = gdal.Open(str(byte_vrt))
    assert byte_dataset is not None
    assert byte_dataset.GetRasterBand(1).DataType == gdal.GDT_Byte
    assert [
        gdal.GetColorInterpretationName(byte_dataset.GetRasterBand(index).GetColorInterpretation())
        for index in range(1, 4)
    ] == ["Red", "Green", "Blue"]
    assert byte_dataset.ReadAsArray().max() > 0

    chunk_result = process_chunk(
        (
            str(byte_vrt),
            str(chunk_path),
            "png",
            {
                "name": "Test tiles",
                "description": "Nested Byte VRT chunk test",
                "quality": 75,
                "resample_alg": "bilinear",
                "blocksize": 256,
            },
            list(get_dataset_bounds(byte_dataset)),
        )
    )

    assert chunk_result == str(chunk_path)

    conn = sqlite3.connect(chunk_path)
    tile_count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    conn.close()

    assert tile_count > 0


def test_process_date_uses_step_prefixed_vrt_names(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    monkeypatch.setattr("satmaps.tiler.apply_rgb_color_interpretation_to_vrt", lambda path: None)

    monkeypatch.setattr(
        "satmaps.list_mosaic_folders",
        lambda date, mgrs_filter=None, cache_dir=None: ["tile_a"],
    )
    monkeypatch.setattr(
        "satmaps.prepare_group_sources",
        lambda folders, date, cache_dir, unique_id, download_only, max_workers: (
            [f".temp/step1_tile_{folder}_{unique_id}.vrt" for folder in folders],
            1,
            3,
        ),
    )

    build_calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda output, sources, srcNodata=None, VRTNodata=None: build_calls.append((output, tuple(sources))),
    )

    args = Namespace(
        cache=".cache",
        all_tiles=False,
        mgrs="31TDF",
        download_only=False,
        processes=1,
    )

    date_vrts, _tile_vrts = process_date("2025/07/01", args, "abc123", None)

    assert date_vrts == [".temp/step3_mosaic_2025-07-01_abc123.vrt"]
    assert build_calls[0][0] == ".temp/step2_group_2025-07-01_0_abc123.vrt"
    assert build_calls[1][0] == ".temp/step3_mosaic_2025-07-01_abc123.vrt"


def test_run_tiling_returns_final_byte_vrt_for_vrt_mode(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    input_path = tmp_path / "input.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 32, 32, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 1000.0, 0.0, 32000.0, 0.0, -1000.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    for band_index, value in enumerate((1500, 2500, 3500), start=1):
        dataset.GetRasterBand(band_index).Fill(value)
    dataset = None

    artifacts = run_tiling(
        str(input_path),
        str(tmp_path / "ignored.mbtiles"),
        "png",
        [[0, 4000, 0, 255]] * 3,
        {
            "chunk_zoom": 6,
            "processes": 1,
            "unique_id": "abc123",
            "vrt": True,
        },
    )

    assert artifacts.final_vrt == ".temp/step6_color_corrected_abc123.vrt"
    assert artifacts.cleanup_paths == [".temp/step6_tone_mapped_abc123.vrt"]
    assert not (tmp_path / "ignored.mbtiles").exists()

    final_dataset = gdal.Open(str(tmp_path / artifacts.final_vrt))
    assert final_dataset is not None
    assert final_dataset.GetRasterBand(1).DataType == gdal.GDT_Float32


def test_create_rgb_vrt_sets_rgb_color_interpretation(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")
    band_paths: dict[str, str] = {}
    for color_name, value in (("red", 1000), ("green", 2000), ("blue", 3000)):
        band_path = tmp_path / f"{color_name}.tif"
        dataset = driver.Create(str(band_path), 2, 2, 1, gdal.GDT_Int16)
        dataset.GetRasterBand(1).Fill(value)
        dataset = None
        band_paths[color_name] = str(band_path)

    output_path = tmp_path / "rgb.vrt"
    create_rgb_vrt(band_paths, str(output_path))

    dataset = gdal.Open(str(output_path))
    assert dataset is not None
    assert [
        gdal.GetColorInterpretationName(dataset.GetRasterBand(index).GetColorInterpretation())
        for index in range(1, 4)
    ] == ["Red", "Green", "Blue"]


def test_merge_date_vrts_sets_rgb_color_interpretation(tmp_path: Path) -> None:
    driver = gdal.GetDriverByName("GTiff")
    input_paths: list[str] = []
    for source_index, fill_values in enumerate(((1000, 2000, 3000), (1500, 2500, 3500)), start=1):
        input_path = tmp_path / f"input_{source_index}.tif"
        dataset = driver.Create(str(input_path), 2, 2, 3, gdal.GDT_Int16)
        dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        dataset.SetProjection(srs.ExportToWkt())

        for band_index, value in enumerate(fill_values, start=1):
            band = dataset.GetRasterBand(band_index)
            band.Fill(value)
            band.SetNoDataValue(-32768)
        dataset = None
        input_paths.append(str(input_path))

    output_path = tmp_path / "merged.vrt"
    merge_date_vrts(str(output_path), input_paths)

    dataset = gdal.Open(str(output_path))
    assert dataset is not None
    assert [
        gdal.GetColorInterpretationName(dataset.GetRasterBand(index).GetColorInterpretation())
        for index in range(1, 4)
    ] == ["Red", "Green", "Blue"]


def test_main_vrt_mode_skips_pmtiles_and_preserves_final_vrt(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    monkeypatch.setattr(sys, "argv", ["satmaps.py", "--vrt", "--date", "2025/07/01"])
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.uuid.uuid4", lambda: SimpleNamespace(hex="abc12345deadbeef"))
    monkeypatch.setattr("satmaps.tiler.apply_rgb_color_interpretation_to_vrt", lambda path: None)

    def fake_process_date(date: str, args: Namespace, unique_id: str, land_tiles: object) -> tuple[list[str], list[str]]:
        tile_vrt = Path(f".temp/step1_tile_tile_a_{unique_id}.vrt")
        group_vrt = Path(f".temp/step2_group_{date.replace('/', '-')}_0_{unique_id}.vrt")
        date_vrt = Path(f".temp/step3_mosaic_{date.replace('/', '-')}_{unique_id}.vrt")
        for path in (tile_vrt, group_vrt, date_vrt):
            path.write_text(path.name)
        return [str(date_vrt)], [str(tile_vrt), str(group_vrt)]

    monkeypatch.setattr("satmaps.process_date", fake_process_date)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda output, sources, srcNodata=None, VRTNodata=None: Path(output).write_text("|".join(sources)),
    )
    monkeypatch.setattr(
        "satmaps.run_warp",
        lambda input_vrt, output_vrt, materialize_geotiff=False: Path(output_vrt).write_text(input_vrt),
    )
    monkeypatch.setattr("satmaps.gdal.Open", lambda path: SimpleNamespace(RasterCount=3))
    monkeypatch.setattr("satmaps.calculate_scaling_params", lambda dataset: [[0, 4000, 0, 255]] * 3)

    pmtiles_calls: list[list[str]] = []
    monkeypatch.setattr("satmaps.subprocess.run", lambda cmd, check=True: pmtiles_calls.append(cmd))

    def fake_run_tiling(input_vrt: str, output_mbtiles: str, tile_format: str, scale_params: list[list[float]], options: dict[str, object]) -> TilingArtifacts:
        Path(".temp/step6_color_corrected_abc12345.vrt").write_text("color")
        return TilingArtifacts(
            final_vrt=".temp/step6_color_corrected_abc12345.vrt",
            cleanup_paths=[],
        )

    monkeypatch.setattr("satmaps.tiler.run_tiling", fake_run_tiling)

    main()

    assert pmtiles_calls == []
    assert Path(".temp/step1_tile_tile_a_abc12345.vrt").exists()
    assert Path(".temp/step2_group_2025-07-01_0_abc12345.vrt").exists()
    assert Path(".temp/step3_mosaic_2025-07-01_abc12345.vrt").exists()
    assert Path(".temp/step4_master_abc12345.vrt").exists()
    assert Path(".temp/step5_warped_3857_abc12345.vrt").exists()
    assert Path(".temp/step6_color_corrected_abc12345.vrt").exists()
    assert not Path(".temp/step7_byte_conversion_abc12345.vrt").exists()


def test_main_step5_mode_materializes_geotiff_and_preserves_it_for_vrt(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    monkeypatch.setattr(sys, "argv", ["satmaps.py", "--vrt", "--step5", "--date", "2025/07/01"])
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    monkeypatch.setattr("satmaps.uuid.uuid4", lambda: SimpleNamespace(hex="abc12345deadbeef"))
    monkeypatch.setattr("satmaps.tiler.apply_rgb_color_interpretation_to_vrt", lambda path: None)

    def fake_process_date(date: str, args: Namespace, unique_id: str, land_tiles: object) -> tuple[list[str], list[str]]:
        tile_vrt = Path(f".temp/step1_tile_tile_a_{unique_id}.vrt")
        group_vrt = Path(f".temp/step2_group_{date.replace('/', '-')}_0_{unique_id}.vrt")
        date_vrt = Path(f".temp/step3_mosaic_{date.replace('/', '-')}_{unique_id}.vrt")
        for path in (tile_vrt, group_vrt, date_vrt):
            path.write_text(path.name)
        return [str(date_vrt)], [str(tile_vrt), str(group_vrt)]

    monkeypatch.setattr("satmaps.process_date", fake_process_date)
    monkeypatch.setattr(
        "satmaps.gdal.BuildVRT",
        lambda output, sources, srcNodata=None, VRTNodata=None: Path(output).write_text("|".join(sources)),
    )
    monkeypatch.setattr("satmaps.gdal.Open", lambda path: SimpleNamespace(RasterCount=3))
    monkeypatch.setattr("satmaps.calculate_scaling_params", lambda dataset: [[0, 4000, 0, 255]] * 3)

    warp_calls: list[tuple[str, str, bool]] = []

    def fake_run_warp(input_vrt: str, output_path: str, materialize_geotiff: bool = False) -> None:
        warp_calls.append((input_vrt, output_path, materialize_geotiff))
        Path(output_path).write_text(input_vrt)

    monkeypatch.setattr("satmaps.run_warp", fake_run_warp)
    monkeypatch.setattr("satmaps.subprocess.run", lambda cmd, check=True: None)

    def fake_run_tiling(input_vrt: str, output_mbtiles: str, tile_format: str, scale_params: list[list[float]], options: dict[str, object]) -> TilingArtifacts:
        assert input_vrt == ".temp/step5_warped_3857_abc12345.tif"
        Path(".temp/step6_color_corrected_abc12345.vrt").write_text(input_vrt)
        return TilingArtifacts(
            final_vrt=".temp/step6_color_corrected_abc12345.vrt",
            cleanup_paths=[],
        )

    monkeypatch.setattr("satmaps.tiler.run_tiling", fake_run_tiling)

    main()

    assert warp_calls == [
        (
            ".temp/step4_master_abc12345.vrt",
            ".temp/step5_warped_3857_abc12345.tif",
            True,
        )
    ]
    assert Path(".temp/step5_warped_3857_abc12345.tif").exists()
    assert Path(".temp/step6_color_corrected_abc12345.vrt").exists()
    assert not Path(".temp/step7_byte_conversion_abc12345.vrt").exists()


def test_calculate_scaling_params_uses_996_percent_upper_bound(tmp_path: Path) -> None:
    input_path = tmp_path / "percentile_input.tif"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 1000, 1, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    values = np.full((1, 1000), 2000, dtype=np.int16)
    values[0, -4:] = 9000
    for band_index in range(1, 4):
        band = dataset.GetRasterBand(band_index)
        band.WriteArray(values)
        band.SetNoDataValue(0)
    dataset = None

    opened = gdal.Open(str(input_path))
    scale_params = calculate_scaling_params(opened)

    assert scale_params == [[0.0, 2000.0, 0, 255]] * 3
