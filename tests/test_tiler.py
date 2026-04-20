import sqlite3
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiler import (
    WEB_MERCATOR_LIMIT,
    apply_preview_correction_numpy,
    apply_soft_knee_numpy,
    get_chunk_tile_range,
    get_web_mercator_bounds,
    intersect_proj_win,
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

    # Mock gdaladdo as it might not be in the environment or could be slow
    monkeypatch.setattr("subprocess.run", lambda cmd, check: None)

    artifacts = run_tiling_simplified(str(input_path), str(output_mbtiles), options)

    assert output_mbtiles.exists()
    assert artifacts.final_vrt == str(input_path)

    # Check if MBTiles has data
    conn = sqlite3.connect(output_mbtiles)
    count = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    conn.close()
    assert count > 0
