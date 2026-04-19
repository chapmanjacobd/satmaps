import sqlite3
from pathlib import Path
import sys

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiler import (
    WEB_MERCATOR_LIMIT,
    create_color_corrected_vrt,
    get_chunk_tile_range,
    get_web_mercator_bounds,
    intersect_proj_win,
    merge_mbtiles,
    proj_win_to_src_win,
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
    assert intersect_proj_win((-5.0, 12.0, 8.0, -2.0), (0.0, 10.0, 10.0, 0.0)) == (0.0, 10.0, 8.0, 0.0)


def test_intersect_proj_win_rejects_touching_window() -> None:
    assert intersect_proj_win((10.0, 10.0, 20.0, 0.0), (0.0, 10.0, 10.0, 0.0)) is None


def test_merge_mbtiles_merges_all_attached_tile_tables(tmp_path: Path) -> None:
    def write_chunk(path: Path, zoom_level: int, tile_column: int, tile_row: int, payload: bytes) -> None:
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


def test_create_color_corrected_vrt_emits_nonzero_pixels(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "color.vrt"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 2, 2, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

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
        str(output_path),
        [[0, 4000, 0, 255]] * 3,
    )

    output = gdal.Open(str(output_path))
    assert output is not None
    assert output.GetRasterBand(1).DataType == gdal.GDT_Float32

    pixel_values = output.ReadAsArray()
    assert pixel_values.max() > 0


def test_create_color_corrected_vrt_preserves_zero_channel_dark_pixels(tmp_path: Path) -> None:
    input_path = tmp_path / "dark_input.tif"
    output_path = tmp_path / "dark_color.vrt"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 2, 1, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    band_values = (
        [[0, 0]],
        [[200, 0]],
        [[300, 0]],
    )
    for band_index, values in enumerate(band_values, start=1):
        band = dataset.GetRasterBand(band_index)
        band.WriteArray(np.array(values, dtype=np.int16))
        band.SetNoDataValue(0)
    dataset = None

    create_color_corrected_vrt(
        str(input_path),
        str(output_path),
        [[0, 4000, 0, 255]] * 3,
    )

    output = gdal.Open(str(output_path))
    assert output is not None

    pixel_values = output.ReadAsArray()
    assert pixel_values[:, 0, 0].max() > 0
    assert pixel_values[:, 0, 1].max() == 0

    vrt_text = output_path.read_text()
    assert "B1 &lt;= 0" not in vrt_text
    assert "B1 == 0.0" in vrt_text
    assert "B2 == 0.0" in vrt_text
    assert "B3 == 0.0" in vrt_text


def test_create_color_corrected_vrt_keeps_low_valid_values_above_zero(tmp_path: Path) -> None:
    input_path = tmp_path / "low_values_input.tif"
    output_path = tmp_path / "low_values_color.vrt"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 3, 1, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    low_values = np.array([[300, 400, 500]], dtype=np.int16)
    for band_index in range(1, 4):
        band = dataset.GetRasterBand(band_index)
        band.WriteArray(low_values)
        band.SetNoDataValue(0)
    dataset = None

    create_color_corrected_vrt(
        str(input_path),
        str(output_path),
        [[0, 9000, 0, 255]] * 3,
    )

    output = gdal.Open(str(output_path))
    assert output is not None

    pixel_values = output.ReadAsArray()
    assert pixel_values[:, 0, 0].min() > 0
    assert pixel_values[:, 0, 1].min() > 0
    assert pixel_values[:, 0, 2].min() > 0


def test_create_color_corrected_vrt_applies_soft_knee_tone_curve(tmp_path: Path) -> None:
    input_path = tmp_path / "tone_curve_input.tif"
    output_path = tmp_path / "tone_curve_color.vrt"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(input_path), 3, 1, 3, gdal.GDT_Int16)
    dataset.SetGeoTransform((0.0, 10.0, 0.0, 20.0, 0.0, -10.0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    values = np.array([[300, 4500, 9000]], dtype=np.int16)
    for band_index in range(1, 4):
        band = dataset.GetRasterBand(band_index)
        band.WriteArray(values)
        band.SetNoDataValue(0)
    dataset = None

    create_color_corrected_vrt(str(input_path), str(output_path), [[0, 9000, 0, 255]] * 3)

    output = gdal.Open(str(output_path))
    assert output is not None
    assert output.GetRasterBand(1).DataType == gdal.GDT_Float32

    pixel_values = output.ReadAsArray()
    expected = np.array([8.33, 107.1, 233.33], dtype=np.float32)
    np.testing.assert_allclose(pixel_values[0, 0, :], expected, atol=0.75)


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
