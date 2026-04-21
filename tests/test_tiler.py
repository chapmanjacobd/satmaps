import sqlite3
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiler as tiler_module
from tiler import (
    WEB_MERCATOR_LIMIT,
    apply_preview_correction_numpy,
    apply_soft_knee_numpy,
    get_chunk_tile_range,
    get_web_mercator_bounds,
    intersect_proj_win,
    lonlat_bbox_to_mercator_bounds,
    merge_mbtiles,
    proj_win_to_src_win,
    run_tiling_simplified,
)


def test_tile_range_does_not_overselect_exact_tile_bounds() -> None:
    bounds = get_web_mercator_bounds(7, 3, 4)
    assert get_chunk_tile_range(bounds, 7) == (3, 4, 3, 4)


def test_tile_range_keeps_exact_south_boundary_in_same_tile() -> None:
    zoom = 7
    res = WEB_MERCATOR_LIMIT * 2 / (2**zoom)
    minx = -WEB_MERCATOR_LIMIT + (10.25 * res)
    maxx = minx + (0.5 * res)
    maxy = WEB_MERCATOR_LIMIT - (20.10 * res)
    miny = WEB_MERCATOR_LIMIT - (21.0 * res)

    assert get_chunk_tile_range((minx, maxy, maxx, miny), zoom) == (10, 20, 10, 20)


def test_intersect_proj_win_clamps_partial_overlap() -> None:
    assert intersect_proj_win((-5.0, 12.0, 8.0, -2.0), (0.0, 10.0, 10.0, 0.0)) == (
        0.0,
        10.0,
        8.0,
        0.0,
    )


def test_intersect_proj_win_rejects_touching_window() -> None:
    assert intersect_proj_win((10.0, 10.0, 20.0, 0.0), (0.0, 10.0, 10.0, 0.0)) is None


def test_lonlat_bbox_to_mercator_bounds_matches_chunk_tile() -> None:
    tile_bounds = get_web_mercator_bounds(4, 8, 7)
    bbox_bounds = lonlat_bbox_to_mercator_bounds(0.0, 0.0, 22.5, 21.943045533438177)
    np.testing.assert_allclose(bbox_bounds, tile_bounds, atol=1e-6)


def test_merge_mbtiles_merges_all_attached_tile_tables(tmp_path: Path) -> None:
    def write_chunk(
        path: Path, zoom_level: int, tile_column: int, tile_row: int, payload: bytes
    ) -> None:
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE tiles (
                zoom_level INTEGER,
                tile_column INTEGER,
                tile_row INTEGER,
                tile_data BLOB
            )
            """
        )
        conn.execute(
            "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
            (zoom_level, tile_column, tile_row, payload),
        )
        conn.commit()
        conn.close()

    chunk_paths = [
        tmp_path / "chunk_0.mbtiles",
        tmp_path / "chunk_1.mbtiles",
        tmp_path / "chunk_2.mbtiles",
    ]
    write_chunk(chunk_paths[0], 1, 0, 0, b"a")
    write_chunk(chunk_paths[1], 1, 1, 0, b"b")
    write_chunk(chunk_paths[2], 1, 2, 0, b"c")

    output_path = tmp_path / "merged.mbtiles"
    merge_mbtiles(str(output_path), [str(path) for path in chunk_paths])

    conn = sqlite3.connect(output_path)
    rows = conn.execute(
        "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles ORDER BY tile_column"
    ).fetchall()
    conn.close()

    assert rows == [
        (1, 0, 0, b"a"),
        (1, 1, 0, b"b"),
        (1, 2, 0, b"c"),
    ]


def test_apply_soft_knee_numpy_basics() -> None:
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    toned = apply_soft_knee_numpy(
        arr,
        shadow_break=0.3,
        highlight_break=0.75,
        shadow_slope=1.4,
        mid_slope=0.9,
        highlight_slope=0.5,
    )

    # Shadow: 0.0 * 1.4 = 0.0, 0.2 * 1.4 = 0.28
    # Mid: 0.3*1.4 + (0.5-0.3)*0.9 = 0.42 + 0.18 = 0.6
    # Highlight: 0.42 + (0.75-0.3)*0.9 + (0.8-0.75)*0.5 = 0.42 + 0.405 + 0.025 = 0.85
    # Max: 0.42 + 0.405 + (1.0-0.75)*0.5 = 0.825 + 0.125 = 0.95

    expected = np.array([0.0, 0.28, 0.6, 0.85, 0.95], dtype=np.float32)
    np.testing.assert_allclose(toned, expected, atol=1e-5)


def test_apply_preview_correction_numpy_basics() -> None:
    # (3, 1, 1) RGB stack
    rgb = np.array([[[0.5]], [[0.5]], [[0.5]]], dtype=np.float32)
    # Luma will be 0.5
    # Desaturate: 0.5 + (0.5 - 0.5) * 0.9 = 0.5
    # Darken: 0.5 < 0.7, so 0.5 * 0.7 = 0.35

    corrected = apply_preview_correction_numpy(
        rgb, saturation=0.9, darken_break=0.7, low_slope=0.7, gamma=1.0
    )
    expected = np.array([[[0.35]], [[0.35]], [[0.35]]], dtype=np.float32)
    np.testing.assert_allclose(corrected, expected, atol=1e-5)


def test_proj_win_to_src_win_converts_bounds_to_pixel_window(tmp_path: Path) -> None:
    input_path = tmp_path / "window.tif"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 10, 10, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 100.0, 0.0, -10.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    dataset = None

    opened = gdal.Open(str(input_path))
    assert proj_win_to_src_win(opened, (20.0, 90.0, 50.0, 40.0)) == (2, 1, 3, 5)


def test_run_tiling_simplified_creates_mbtiles(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    input_path = tmp_path / "input.tif"
    driver = gdal.GetDriverByName("GTiff")
    # Small image to speed up
    dataset = driver.Create(str(input_path), 16, 16, 3, gdal.GDT_Byte)
    dataset.SetGeoTransform((0.0, 1000.0, 0.0, 16000.0, 0.0, -1000.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    for i in range(1, 4):
        dataset.GetRasterBand(i).Fill(128)
    dataset = None

    output_mbtiles = tmp_path / "output.mbtiles"
    options = {
        "format": "png",
        "quality": 75,
        "resample_alg": "bilinear",
        "chunk_zoom": 10,
        "processes": 1,
        "unique_id": "test",
        "name": "Test",
        "description": "Test",
    }

    gdaladdo_calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        gdaladdo_calls.append(cmd)

    monkeypatch.setattr("subprocess.run", fake_run)

    artifacts = run_tiling_simplified(str(input_path), str(output_mbtiles), options)

    assert output_mbtiles.exists()
    assert artifacts.final_vrt == str(input_path)
    assert gdaladdo_calls == [
        [
            "gdaladdo",
            "-r",
            "bilinear",
            "--config",
            "GDAL_NUM_THREADS",
            "ALL_CPUS",
            str(output_mbtiles),
        ]
    ]

    # Check if MBTiles has data
    conn = sqlite3.connect(output_mbtiles)
    count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    conn.close()
    assert count > 0


def test_run_tiling_simplified_uses_explicit_chunk_bounds(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    input_path = tmp_path / "world.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(input_path), 16, 16, 3, gdal.GDT_Byte)
    dataset.SetGeoTransform(
        (
            -WEB_MERCATOR_LIMIT,
            (WEB_MERCATOR_LIMIT * 2) / 16,
            0.0,
            WEB_MERCATOR_LIMIT,
            0.0,
            -(WEB_MERCATOR_LIMIT * 2) / 16,
        )
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    dataset = None

    requested_bounds = get_web_mercator_bounds(4, 8, 7)
    seen_proj_wins: list[tuple[float, float, float, float]] = []

    def fake_process_chunk(task):
        seen_proj_wins.append(task[-1])
        return ""

    monkeypatch.setattr(tiler_module, "process_chunk", fake_process_chunk)
    monkeypatch.setattr(tiler_module, "merge_mbtiles", lambda output, chunks: None)
    monkeypatch.setattr(tiler_module, "finalize_mbtiles_metadata", lambda path: None)
    monkeypatch.setattr(tiler_module.subprocess, "run", lambda cmd, check: None)

    run_tiling_simplified(
        str(input_path),
        str(tmp_path / "output.mbtiles"),
        {
            "format": "png",
            "quality": 75,
            "resample_alg": "bilinear",
            "chunk_zoom": 4,
            "processes": 1,
            "unique_id": "test-explicit-bounds",
            "name": "Test",
            "description": "Test",
            "chunk_bounds": requested_bounds,
        },
    )

    assert seen_proj_wins == [requested_bounds]


@pytest.mark.skipif(shutil.which("gdaladdo") is None, reason="gdaladdo not available")
def test_run_tiling_simplified_builds_lower_zoom_levels(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    input_path = tmp_path / "input.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 256, 256, 3, gdal.GDT_Byte)
    dataset.SetGeoTransform((0.0, 1000.0, 0.0, 256000.0, 0.0, -1000.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    for i in range(1, 4):
        dataset.GetRasterBand(i).Fill(128)
    dataset = None

    output_mbtiles = tmp_path / "output.mbtiles"
    run_tiling_simplified(
        str(input_path),
        str(output_mbtiles),
        {
            "format": "png",
            "quality": 75,
            "resample_alg": "bilinear",
            "chunk_zoom": 4,
            "processes": 1,
            "unique_id": "test-overviews",
            "name": "Test",
            "description": "Test",
        },
    )

    conn = sqlite3.connect(output_mbtiles)
    zooms = [
        row[0]
        for row in conn.execute("SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level")
    ]
    metadata = dict(conn.execute("SELECT name, value FROM metadata"))
    conn.close()

    assert len(zooms) > 1
    assert min(zooms) < max(zooms)
    assert metadata["minzoom"] == str(min(zooms))
    assert metadata["maxzoom"] == str(max(zooms))


@pytest.mark.skipif(shutil.which("gdaladdo") is None, reason="gdaladdo not available")
def test_run_tiling_simplified_preserves_land_in_multi_chunk_overviews(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    width, height = 4096, 1024
    input_path = tmp_path / "input.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), width, height, 3, gdal.GDT_Byte)
    dataset.SetGeoTransform((0.0, 1000.0, 0.0, height * 1000.0, 0.0, -1000.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    ocean_r = np.full((height, width), 20, dtype=np.uint8)
    ocean_g = np.full((height, width), 50, dtype=np.uint8)
    ocean_b = np.full((height, width), 180, dtype=np.uint8)

    ocean_r[:, width // 2 :] = 70
    ocean_g[:, width // 2 :] = 180
    ocean_b[:, width // 2 :] = 50

    dataset.GetRasterBand(1).WriteArray(ocean_r)
    dataset.GetRasterBand(2).WriteArray(ocean_g)
    dataset.GetRasterBand(3).WriteArray(ocean_b)
    dataset = None

    output_mbtiles = tmp_path / "output.mbtiles"
    run_tiling_simplified(
        str(input_path),
        str(output_mbtiles),
        {
            "format": "png",
            "quality": 75,
            "resample_alg": "bilinear",
            "chunk_zoom": 4,
            "processes": 1,
            "unique_id": "test-multichunk-overviews",
            "name": "Test",
            "description": "Test",
        },
    )

    conn = sqlite3.connect(output_mbtiles)
    minzoom = int(
        conn.execute("SELECT value FROM metadata WHERE name = 'minzoom'").fetchone()[0]
    )
    conn.close()

    overview_ds = gdal.OpenEx(str(output_mbtiles), open_options=[f"ZOOM_LEVEL={minzoom}"])
    assert overview_ds is not None
    arr = overview_ds.ReadAsArray()
    overview_ds = None

    left_center = arr[:3, arr.shape[1] // 2, arr.shape[2] // 4]
    right_center = arr[:3, arr.shape[1] // 2, (3 * arr.shape[2]) // 4]

    np.testing.assert_array_equal(left_center, np.array([20, 50, 180], dtype=np.uint8))
    np.testing.assert_array_equal(right_center, np.array([70, 180, 50], dtype=np.uint8))


@pytest.mark.skipif(shutil.which("gdaladdo") is None, reason="gdaladdo not available")
def test_run_tiling_simplified_respects_alpha_masked_sources(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    def write_masked_tile(name: str, valid_slice: slice, color: tuple[int, int, int]) -> Path:
        path = tmp_path / f"{name}.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(path), 512, 256, 4, gdal.GDT_Byte)
        ds.SetGeoTransform((0.0, 1000.0, 0.0, 256000.0, 0.0, -1000.0))
        ds.SetProjection(srs.ExportToWkt())

        alpha = np.zeros((256, 512), dtype=np.uint8)
        alpha[:, valid_slice] = 255
        for i, value in enumerate(color, start=1):
            arr = np.zeros((256, 512), dtype=np.uint8)
            arr[:, valid_slice] = value
            ds.GetRasterBand(i).WriteArray(arr)
        ds.GetRasterBand(4).WriteArray(alpha)
        ds = None
        return path

    left = write_masked_tile("left", slice(0, 256), (70, 180, 50))
    right = write_masked_tile("right", slice(256, 512), (20, 50, 180))

    master_vrt = tmp_path / "master.vrt"
    gdal.BuildVRT(str(master_vrt), [str(left), str(right)], resolution="highest")

    output_mbtiles = tmp_path / "output.mbtiles"
    run_tiling_simplified(
        str(master_vrt),
        str(output_mbtiles),
        {
            "format": "png",
            "quality": 75,
            "resample_alg": "bilinear",
            "chunk_zoom": 4,
            "processes": 1,
            "unique_id": "test-alpha-mask",
            "name": "Test",
            "description": "Test",
        },
    )

    ds = gdal.OpenEx(str(output_mbtiles), open_options=["ZOOM_LEVEL=7"])
    assert ds is not None
    arr = ds.ReadAsArray()
    ds = None

    left_center = arr[:3, arr.shape[1] // 2, arr.shape[2] // 4]
    right_center = arr[:3, arr.shape[1] // 2, (3 * arr.shape[2]) // 4]

    np.testing.assert_array_equal(left_center, np.array([70, 180, 50], dtype=np.uint8))
    np.testing.assert_array_equal(right_center, np.array([20, 50, 180], dtype=np.uint8))
