from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiler import (
    WEB_MERCATOR_LIMIT,
    get_chunk_tile_range,
    get_web_mercator_bounds,
    intersect_proj_win,
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
