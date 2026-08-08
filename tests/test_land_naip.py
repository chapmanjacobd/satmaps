import json
import sqlite3
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import land_naip


def _payload_bbox(payload: dict[str, Any]) -> land_naip.BBox:
    spatial_filter = payload["sceneFilter"]["spatialFilter"]
    lower_left = spatial_filter["lowerLeft"]
    upper_right = spatial_filter["upperRight"]
    return (
        lower_left["longitude"],
        lower_left["latitude"],
        upper_right["longitude"],
        upper_right["latitude"],
    )


def test_discover_naip_tiles_ee_splits_capped_searches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root_bbox = (-1.0, 0.0, 1.0, 2.0)
    queried_bboxes: list[land_naip.BBox] = []

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        assert endpoint == "scene-search"
        bbox = _payload_bbox(payload)
        queried_bboxes.append(bbox)
        if bbox == root_bbox:
            return {"results": [{"entityId": f"root-{index}"} for index in range(land_naip.EE_MAX_RESULTS)]}
        return {"results": [{"entityId": f"child-{bbox}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)

    scenes = land_naip.discover_naip_tiles_ee(
        root_bbox,
        "api-key",
        cache_dir=str(tmp_path),
    )

    assert queried_bboxes == [
        root_bbox,
        (-1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (-1.0, 1.0, 0.0, 2.0),
        (0.0, 1.0, 1.0, 2.0),
    ]
    assert [scene["entityId"] for scene in scenes] == [
        "child-(-1.0, 0.0, 0.0, 1.0)",
        "child-(0.0, 0.0, 1.0, 1.0)",
        "child-(-1.0, 1.0, 0.0, 2.0)",
        "child-(0.0, 1.0, 1.0, 2.0)",
    ]


def _l_shape_geometry():
    from osgeo import ogr

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat in [
        (0.5, 0.5),
        (0.5, 9.5),
        (9.5, 9.5),
        (9.5, 6.5),
        (3.5, 6.5),
        (3.5, 0.5),
        (0.5, 0.5),
    ]:
        ring.AddPoint(lon, lat)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def test_bbox_intersects_geometry_uses_exact_shape_not_envelope() -> None:
    geometry = _l_shape_geometry()

    assert land_naip._bbox_intersects_geometry((4.0, 1.0, 4.4, 1.9), geometry) is False
    assert land_naip._bbox_intersects_geometry((4.0, 7.0, 4.4, 7.9), geometry) is True
    assert land_naip._bbox_intersects_geometry((20.0, 20.0, 21.0, 21.0), geometry) is False


def test_discover_naip_tiles_ee_prunes_cells_outside_geometry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    queried_bboxes: list[land_naip.BBox] = []

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        assert endpoint == "scene-search"
        bbox = _payload_bbox(payload)
        queried_bboxes.append(bbox)
        return {"results": [{"entityId": f"scene-{bbox}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)

    land_naip.discover_naip_tiles_ee(
        (0.0, 0.0, 10.0, 10.0),
        "api-key",
        cache_dir=str(tmp_path),
        extent_geometry=_l_shape_geometry(),
    )

    assert queried_bboxes == [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 4.0, 2.0),
        (0.0, 2.0, 2.0, 4.0),
        (2.0, 2.0, 4.0, 4.0),
        (0.0, 4.0, 2.0, 6.0),
        (2.0, 4.0, 4.0, 6.0),
        (0.0, 6.0, 2.0, 8.0),
        (2.0, 6.0, 4.0, 8.0),
        (4.0, 6.0, 6.0, 8.0),
        (6.0, 6.0, 8.0, 8.0),
        (8.0, 6.0, 10.0, 8.0),
        (0.0, 8.0, 2.0, 10.0),
        (2.0, 8.0, 4.0, 10.0),
        (4.0, 8.0, 6.0, 10.0),
        (6.0, 8.0, 8.0, 10.0),
        (8.0, 8.0, 10.0, 10.0),
    ]


def test_discover_naip_tiles_ee_queries_each_extent_feature(
    tmp_path: Path, monkeypatch: Any
) -> None:
    queried_bboxes: list[land_naip.BBox] = []

    def polygon_geometry(
        min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> Any:
        from osgeo import ogr

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for lon, lat in [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]:
            ring.AddPoint(lon, lat)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        return polygon

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        assert endpoint == "scene-search"
        bbox = _payload_bbox(payload)
        queried_bboxes.append(bbox)
        return {"results": [{"entityId": f"scene-{bbox}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)

    scenes = land_naip.discover_naip_tiles_ee(
        (-180.0, -90.0, 180.0, 90.0),
        "api-key",
        cache_dir=str(tmp_path),
        extent_geometries=[
            polygon_geometry(-104.1, 29.3, -103.7, 29.8),
            polygon_geometry(-101.0, 31.0, -100.5, 31.5),
        ],
    )

    assert queried_bboxes == [
        (-104.1, 29.3, -103.7, 29.8),
        (-101.0, 31.0, -100.5, 31.5),
    ]
    assert len(scenes) == 2


def test_discover_naip_tiles_ee_reuses_metadata_until_expired(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = 0

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"results": [{"entityId": "scene-1"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)
    bbox = (-1.0, 0.0, 1.0, 1.0)

    land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    assert calls == 1

    db_path = tmp_path / "naip_metadata" / "manifest.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE manifest SET queried_at = ?",
            (
                time.time()
                - (land_naip.NAIP_METADATA_CACHE_MAX_AGE_DAYS + 1) * 24 * 60 * 60,
            ),
        )

    land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    assert calls == 2


def test_large_bbox_cache_is_reused_for_nearby_extent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = 0

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"results": [{"entityId": f"scene-{calls}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)
    first_bbox = (-30.0, 0.0, 30.0, 40.0)
    nearby_bbox = (-29.8, 0.2, 29.8, 39.8)

    land_naip.discover_naip_tiles_ee(first_bbox, "api-key", cache_dir=str(tmp_path))
    first_call_count = calls
    land_naip.discover_naip_tiles_ee(nearby_bbox, "api-key", cache_dir=str(tmp_path))

    assert first_call_count == len(land_naip._initial_query_bboxes(first_bbox))
    assert calls == first_call_count


def test_discover_naip_tiles_ee_negative_caches_zero_results(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[land_naip.BBox] = []

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        calls.append(_payload_bbox(payload))
        return {"results": []}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)
    bbox = (-68.0, 18.0, -66.0, 20.0)

    first = land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    second = land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))

    assert first == []
    assert second == []
    assert len(calls) == 1

    db_path = tmp_path / "naip_metadata" / "manifest.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT bbox_key, split, scenes FROM manifest").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == [bbox[0], bbox[1], bbox[2], bbox[3]]
    assert rows[0][1] == 0
    assert json.loads(rows[0][2]) == []


def test_discover_naip_tiles_ee_resumes_from_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = 0

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        bbox = _payload_bbox(payload)
        return {"results": [{"entityId": f"scene-{bbox}-{calls}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)
    bbox = (-1.0, 0.0, 1.0, 1.0)

    first = land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    assert calls == 1
    assert first

    second = land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))
    assert calls == 1
    assert second == first


def test_migrates_legacy_jsonl_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    calls = 0

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"results": [{"entityId": "should-not-be-queried"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)
    bbox = (-1.0, 0.0, 1.0, 1.0)

    manifest_path = tmp_path / "naip_metadata" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    record = {
        "bbox": list(bbox),
        "split": False,
        "queried_at": time.time(),
        "scenes": [{"entityId": "cached-scene-1"}],
    }
    manifest_path.write_text(json.dumps(record) + "\n")

    scenes = land_naip.discover_naip_tiles_ee(bbox, "api-key", cache_dir=str(tmp_path))

    assert calls == 0
    assert scenes == [{"entityId": "cached-scene-1"}]
    assert not manifest_path.exists()
    assert (tmp_path / "naip_metadata" / "manifest.db").exists()


def test_discover_naip_tiles_ee_resumes_split_without_requerying_parent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root_bbox = (-1.0, 0.0, 1.0, 2.0)
    queried_bboxes: list[land_naip.BBox] = []

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        assert endpoint == "scene-search"
        bbox = _payload_bbox(payload)
        queried_bboxes.append(bbox)
        if bbox == root_bbox:
            return {"results": [{"entityId": f"root-{index}"} for index in range(land_naip.EE_MAX_RESULTS)]}
        return {"results": [{"entityId": f"child-{bbox}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)

    first = land_naip.discover_naip_tiles_ee(root_bbox, "api-key", cache_dir=str(tmp_path))
    first_queries = list(queried_bboxes)
    assert root_bbox in first_queries
    assert len(first) == 4

    second = land_naip.discover_naip_tiles_ee(root_bbox, "api-key", cache_dir=str(tmp_path))
    assert queried_bboxes == first_queries
    assert second == first


def _incremental_discover_stub(scenes: list[Any], kwargs: dict[str, Any]) -> list[Any]:
    callback = kwargs.get("per_cell_callback")
    if callback:
        callback(scenes)
        return []
    return scenes


def test_handle_naip_workflow_combines_earth_explorer_and_island_sources(
    tmp_path: Path, monkeypatch: Any
) -> None:
    scene = {
        "entityId": "scene-1",
        "displayId": "scene-1",
        "spatialCoverage": {
            "type": "Polygon",
            "coordinates": [[
                [-1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.0],
            ]],
        },
    }
    dc_calls: list[land_naip.BBox] = []

    monkeypatch.setattr(land_naip, "m2m_login", lambda: "api-key")
    monkeypatch.setattr(land_naip, "m2m_logout", lambda api_key: None)
    monkeypatch.setattr(
        land_naip,
        "discover_naip_tiles_ee",
        lambda bbox, api_key, **kwargs: _incremental_discover_stub([scene], kwargs),
    )
    monkeypatch.setattr(
        land_naip,
        "fetch_naip_downloads",
        lambda scenes, api_key, cache_dir: ["earth-explorer.tif"],
    )

    def fake_digital_coast(bbox: land_naip.BBox, cache_dir: str) -> list[str]:
        dc_calls.append(bbox)
        return ["island-coverage.vrt"]

    monkeypatch.setattr(land_naip, "discover_naip_tiles_digitalcoast", fake_digital_coast)

    result = land_naip.handle_naip_workflow(
        Namespace(use_naip=True, cache=str(tmp_path), download=False, estimate=False),
        (-1.0, 0.0, 1.0, 1.0),
    )

    assert result == (True, ["earth-explorer.tif", "island-coverage.vrt"])
    assert dc_calls == [(-1.0, 0.0, 1.0, 1.0)]
