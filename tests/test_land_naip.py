import os
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
    root_bbox = (-2.0, 0.0, 2.0, 4.0)
    queried_bboxes: list[land_naip.BBox] = []

    def fake_request(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        assert endpoint == "scene-search"
        bbox = _payload_bbox(payload)
        queried_bboxes.append(bbox)
        if bbox == root_bbox:
            return {"results": [{"entityId": f"root-{index}"} for index in range(500)]}
        return {"results": [{"entityId": f"child-{bbox}"}]}

    monkeypatch.setattr(land_naip, "send_m2m_request", fake_request)

    scenes = land_naip.discover_naip_tiles_ee(
        root_bbox,
        "api-key",
        cache_dir=str(tmp_path),
    )

    assert queried_bboxes == [
        root_bbox,
        (-2.0, 0.0, 0.0, 4.0),
        (0.0, 0.0, 2.0, 4.0),
    ]
    assert [scene["entityId"] for scene in scenes] == [
        "child-(-2.0, 0.0, 0.0, 4.0)",
        "child-(0.0, 0.0, 2.0, 4.0)",
    ]


def test_discover_naip_tiles_ee_queries_each_extent_feature(
    tmp_path: Path, monkeypatch: Any
) -> None:
    queried_bboxes: list[land_naip.BBox] = []

    class FeatureGeometry:
        def __init__(self, envelope: tuple[float, float, float, float]) -> None:
            self.envelope = envelope

        def GetEnvelope(self) -> tuple[float, float, float, float]:
            return self.envelope

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
            FeatureGeometry((-104.1, -103.7, 29.3, 29.8)),
            FeatureGeometry((-101.0, -100.5, 31.0, 31.5)),
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

    cache_path = next(tmp_path.glob("naip_metadata/*.json"))
    expired_at = time.time() - (land_naip.NAIP_METADATA_CACHE_MAX_AGE_DAYS + 1) * 24 * 60 * 60
    os.utime(cache_path, (expired_at, expired_at))

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
        lambda bbox, api_key, **kwargs: [scene],
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
