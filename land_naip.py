#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Sequence, Tuple, Any

BBox = Tuple[float, float, float, float]

M2M_BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
NAIP_METADATA_CACHE_MAX_AGE_DAYS = 90
EE_MAX_RESULTS = 4000
EE_SPLIT_THRESHOLD = 3980
EE_MAX_SPLIT_DEPTH = 12
EE_INITIAL_MAX_BBOX_SPAN_DEGREES = 2.0

DIGITAL_COAST_DATASETS: List[dict[str, Any]] = [
    {
        "name": "HI_NAIP_2021",
        "id": "9668",
        "vrt_epsg_codes": ["26904", "26905"],
        "stac_url": "https://coastalimagery.blob.core.windows.net/digitalcoast/HI_NAIP_2021_9668/stac/noaa_imagery_item_collection_m9668.json",
        "base_url": "https://coastalimagery.blob.core.windows.net/digitalcoast/HI_NAIP_2021_9668",
        "bbox": (-160.3, 18.9, -154.7, 22.3),
    },
    {
        "name": "PR_NAIP_2021",
        "id": "9825",
        "vrt_epsg_codes": ["26919", "26920"],
        "stac_url": "https://coastalimagery.blob.core.windows.net/digitalcoast/PR_NAIP_2021_9825/stac/noaa_imagery_item_collection_m9825.json",
        "base_url": "https://coastalimagery.blob.core.windows.net/digitalcoast/PR_NAIP_2021_9825",
        "bbox": (-67.3, 17.6, -64.4, 18.6),
    },
]

def load_env() -> None:
    """Load variables from a .env file into os.environ if they don't already exist."""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = value

def send_m2m_request(endpoint: str, payload: dict, api_key: Optional[str] = None) -> Any:
    """Send a POST request to the M2M API."""
    url = f"{M2M_BASE_URL}/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-Auth-Token'] = api_key
        
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            if resp_data.get("errorCode"):
                raise RuntimeError(f"M2M Error: {resp_data.get('errorMessage')}")
            return resp_data.get("data")
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        raise RuntimeError(f"HTTP {e.code}: {error_msg}")

def m2m_login() -> str:
    """Authenticate with EarthExplorer M2M and return the API key."""
    token = os.environ.get("USGS_TOKEN")
    username = os.environ.get("USGS_USERNAME")
    
    if not token or not username:
        print("Error: USGS_USERNAME and USGS_TOKEN environment variables are required.")
        sys.exit(1)
        
    print("Authenticating with EarthExplorer M2M...")
    payload = {"username": username, "token": token}
    return send_m2m_request("login-token", payload)

def m2m_logout(api_key: str) -> None:
    """Log out from EarthExplorer M2M."""
    print("Logging out from EarthExplorer...")
    send_m2m_request("logout", {}, api_key=api_key)

def _scene_search_payload(bbox: BBox) -> dict[str, Any]:
    """Build the EarthExplorer scene-search request for one bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "datasetName": "NAIP",
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": min_lat, "longitude": min_lon},
                "upperRight": {"latitude": max_lat, "longitude": max_lon},
            }
        },
        "maxResults": EE_MAX_RESULTS,
    }


def _load_fresh_json_cache(cache_path: str, max_age_days: int) -> Optional[Any]:
    """Load a JSON cache entry when it is present and has not expired."""
    if not os.path.exists(cache_path):
        return None

    age_seconds = time.time() - os.path.getmtime(cache_path)
    if age_seconds >= max_age_days * 24 * 60 * 60:
        print(f"Cached metadata expired: {cache_path}")
        return None

    try:
        with open(cache_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Ignoring invalid cached metadata {cache_path}: {exc}")
        return None


def _write_json_cache(cache_path: str, data: Any) -> None:
    """Write a JSON cache entry atomically."""
    cache_parent = os.path.dirname(cache_path)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)
    temporary_path = f"{cache_path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump(data, f)
    os.replace(temporary_path, cache_path)


def _bbox_geometry(bbox: BBox) -> Any:
    """Build a WGS84 polygon for a bbox."""
    from osgeo import ogr

    min_lon, min_lat, max_lon, max_lat = bbox
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(min_lon, min_lat)
    ring.AddPoint(max_lon, min_lat)
    ring.AddPoint(max_lon, max_lat)
    ring.AddPoint(min_lon, max_lat)
    ring.AddPoint(min_lon, min_lat)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def _bbox_intersects_geometry(bbox: BBox, geometry: Any) -> bool:
    """Check whether a query bbox overlaps a render geometry using its exact
    shape, so cells that only touch the geometry's envelope corners are skipped."""
    envelope = geometry.GetEnvelope()
    geometry_bbox = (envelope[0], envelope[2], envelope[1], envelope[3])
    if not _bbox_overlaps(bbox, geometry_bbox):
        return False
    return _bbox_geometry(bbox).Intersects(geometry)


def _split_bbox(bbox: BBox) -> Optional[List[BBox]]:
    """Split a bbox into four quadrants, halving each axis.

    Sub-cells stay nested within the parent cell, so their metadata cache keys
    remain deterministic for nearby renders instead of depending on which axis
    happened to be longest.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    if lon_span <= 0 and lat_span <= 0:
        return None

    lon_midpoint = min_lon + lon_span / 2
    lat_midpoint = min_lat + lat_span / 2
    return [
        (min_lon, min_lat, lon_midpoint, lat_midpoint),
        (lon_midpoint, min_lat, max_lon, lat_midpoint),
        (min_lon, lat_midpoint, lon_midpoint, max_lat),
        (lon_midpoint, lat_midpoint, max_lon, max_lat),
    ]


def _initial_query_bboxes(bbox: BBox) -> List[BBox]:
    """Make a stable coarse grid so nearby large renders reuse metadata caches."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    if lon_span <= 0 or lat_span <= 0:
        return [bbox]

    if max(lon_span, lat_span) <= EE_INITIAL_MAX_BBOX_SPAN_DEGREES:
        return [bbox]

    grid_size = EE_INITIAL_MAX_BBOX_SPAN_DEGREES
    grid_min_lon = math.floor(min_lon / grid_size) * grid_size
    grid_min_lat = math.floor(min_lat / grid_size) * grid_size
    grid_max_lon = math.ceil(max_lon / grid_size) * grid_size
    grid_max_lat = math.ceil(max_lat / grid_size) * grid_size
    lon_parts = max(1, round((grid_max_lon - grid_min_lon) / grid_size))
    lat_parts = max(1, round((grid_max_lat - grid_min_lat) / grid_size))
    return [
        (
            grid_min_lon + lon_index * grid_size,
            grid_min_lat + lat_index * grid_size,
            grid_min_lon + (lon_index + 1) * grid_size,
            grid_min_lat + (lat_index + 1) * grid_size,
        )
        for lat_index in range(lat_parts)
        for lon_index in range(lon_parts)
    ]


def _deduplicate_scenes(scenes: List[Any]) -> List[Any]:
    """Remove duplicate scenes returned by adjacent spatial searches."""
    deduplicated: List[Any] = []
    seen: set[str] = set()
    for scene in scenes:
        identity = scene.get("entityId") or scene.get("displayId")
        if not identity:
            identity = json.dumps(scene, sort_keys=True, default=str)
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(scene)
    return deduplicated


def _query_naip_tiles_ee(
    bbox: BBox,
    api_key: str,
) -> List[Any]:
    """Query one EarthExplorer bbox for NAIP scene metadata."""
    print(f"Querying EarthExplorer for NAIP imagery in bbox {bbox}...")
    results = send_m2m_request(
        "scene-search",
        _scene_search_payload(bbox),
        api_key=api_key,
    )
    if not isinstance(results, dict):
        raise RuntimeError("EarthExplorer scene-search returned an invalid response.")
    scenes = results.get("results", [])
    if not isinstance(scenes, list):
        raise RuntimeError("EarthExplorer scene-search returned invalid scene results.")
    print(f"Found {len(scenes)} NAIP scenes in the bounding box.")
    return scenes


def _bbox_manifest_key(bbox: BBox) -> str:
    """Return a stable, human-readable key for a bbox in the metadata manifest."""
    return json.dumps(list(bbox), separators=(",", ":"))


def _manifest_path(cache_dir: str) -> str:
    """Return the path to the incremental NAIP metadata manifest."""
    return os.path.join(cache_dir, "naip_metadata", "manifest.jsonl")


def _load_manifest(cache_dir: str) -> dict[str, dict[str, Any]]:
    """Load fresh records from the incremental metadata manifest.

    Returns a dict keyed by bbox key. Malformed or expired lines are skipped so
    a partially-written tail line does not discard earlier progress.
    """
    path = _manifest_path(cache_dir)
    entries: dict[str, dict[str, Any]] = {}
    if not os.path.exists(path):
        return entries

    max_age_seconds = NAIP_METADATA_CACHE_MAX_AGE_DAYS * 24 * 60 * 60
    now = time.time()
    try:
        with open(path) as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Ignoring malformed manifest line {line_number} in {path}."
                    )
                    continue
                bbox = record.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                if now - record.get("queried_at", 0) >= max_age_seconds:
                    continue
                entries[_bbox_manifest_key(tuple(bbox))] = record
    except OSError as exc:
        print(f"Warning: Ignoring unreadable metadata manifest {path}: {exc}")
        return {}
    return entries


def _record_manifest_entry(
    cache_dir: str,
    bbox: BBox,
    split: bool,
    scenes: List[Any],
) -> None:
    """Append one bbox record to the incremental metadata manifest."""
    path = _manifest_path(cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record: dict[str, Any] = {
        "bbox": list(bbox),
        "split": split,
        "queried_at": time.time(),
    }
    if not split:
        record["scenes"] = scenes
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _discover_naip_tiles_ee_recursive(
    bbox: BBox,
    api_key: str,
    cache_dir: str,
    extent_geometry: Optional[Any],
    split_depth: int,
    manifest: dict[str, dict[str, Any]],
) -> List[Any]:
    """Query a bbox and recursively subdivide result sets near the API limit.

    Every queried bbox is recorded in the manifest so an interrupted run can
    resume without re-querying, including bboxes that returned zero scenes.
    """
    if extent_geometry is not None and not _bbox_intersects_geometry(bbox, extent_geometry):
        return []

    recorded = manifest.get(_bbox_manifest_key(bbox))
    if recorded is not None:
        if recorded.get("split"):
            split_bboxes = _split_bbox(bbox)
            if split_bboxes is None:
                return []
            child_scenes: List[Any] = []
            for child_bbox in split_bboxes:
                child_scenes.extend(
                    _discover_naip_tiles_ee_recursive(
                        child_bbox,
                        api_key,
                        cache_dir,
                        extent_geometry,
                        split_depth + 1,
                        manifest,
                    )
                )
            return _deduplicate_scenes(child_scenes)
        print(f"Using incremental metadata for bbox {bbox}.")
        scenes = recorded.get("scenes", [])
        return scenes if isinstance(scenes, list) else []

    scenes = _query_naip_tiles_ee(bbox, api_key)
    if len(scenes) <= EE_SPLIT_THRESHOLD:
        _record_manifest_entry(cache_dir, bbox, split=False, scenes=scenes)
        manifest[_bbox_manifest_key(bbox)] = {
            "bbox": list(bbox),
            "split": False,
            "queried_at": time.time(),
            "scenes": scenes,
        }
        return scenes

    split_bboxes = _split_bbox(bbox)
    if split_bboxes is None or split_depth >= EE_MAX_SPLIT_DEPTH:
        print(
            f"Warning: EarthExplorer returned {len(scenes)} scenes for bbox {bbox}; "
            "using the capped result set because it cannot be subdivided further."
        )
        _record_manifest_entry(cache_dir, bbox, split=False, scenes=scenes)
        manifest[_bbox_manifest_key(bbox)] = {
            "bbox": list(bbox),
            "split": False,
            "queried_at": time.time(),
            "scenes": scenes,
        }
        return scenes

    print(
        f"EarthExplorer returned {len(scenes)} scenes (over {EE_SPLIT_THRESHOLD}); "
        f"splitting bbox at depth {split_depth + 1}."
    )
    _record_manifest_entry(cache_dir, bbox, split=True, scenes=[])
    manifest[_bbox_manifest_key(bbox)] = {
        "bbox": list(bbox),
        "split": True,
        "queried_at": time.time(),
    }

    child_scenes = []
    for child_bbox in split_bboxes:
        child_scenes.extend(
            _discover_naip_tiles_ee_recursive(
                child_bbox,
                api_key,
                cache_dir,
                extent_geometry,
                split_depth + 1,
                manifest,
            )
        )
    return _deduplicate_scenes(child_scenes)


def discover_naip_tiles_ee(
    bbox: Optional[BBox],
    api_key: str,
    cache_dir: str = "cache",
    extent_geometry: Optional[Any] = None,
    extent_geometries: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """
    Query EarthExplorer for NAIP tiles, subdividing spatial searches near its
    500-result limit and caching each metadata response for 90 days.
    """
    if not bbox:
        print("Warning: No bounding box provided. Please provide --bbox.")
        return []

    scenes: List[Any] = []
    if extent_geometries:
        query_regions = list(extent_geometries)
    elif extent_geometry is not None:
        query_regions = [extent_geometry]
    else:
        query_regions = [None]

    manifest = _load_manifest(cache_dir)
    if manifest:
        print(
            f"Loaded {len(manifest)} cached bbox records from the NAIP metadata manifest."
        )

    for query_region in query_regions:
        query_bbox = bbox
        if query_region is not None:
            envelope = query_region.GetEnvelope()
            query_bbox = (envelope[0], envelope[2], envelope[1], envelope[3])
        if query_bbox is None:
            continue

        initial_bboxes = _initial_query_bboxes(query_bbox)
        if len(initial_bboxes) > 1:
            print(
                f"Splitting large NAIP search bbox into {len(initial_bboxes)} initial "
                f"cells (maximum span {EE_INITIAL_MAX_BBOX_SPAN_DEGREES:g} degrees)."
            )
        for initial_bbox in initial_bboxes:
            scenes.extend(
                _discover_naip_tiles_ee_recursive(
                    initial_bbox,
                    api_key,
                    cache_dir,
                    query_region,
                    split_depth=0,
                    manifest=manifest,
                )
            )
    return _deduplicate_scenes(scenes)

def _bbox_overlaps(bbox_a: BBox, bbox_b: BBox) -> bool:
    """Check if two bboxes overlap."""
    min_lon_a, min_lat_a, max_lon_a, max_lat_a = bbox_a
    min_lon_b, min_lat_b, max_lon_b, max_lat_b = bbox_b
    return (
        min_lon_a < max_lon_b and max_lon_a > min_lon_b
        and min_lat_a < max_lat_b and max_lat_a > min_lat_b
    )


def _find_digital_coast_datasets(bbox: BBox) -> List[dict[str, Any]]:
    """Return Digital Coast datasets whose extent overlaps the given bbox."""
    return [
        ds for ds in DIGITAL_COAST_DATASETS
        if _bbox_overlaps(bbox, (
            float(ds["bbox"][0]),
            float(ds["bbox"][1]),
            float(ds["bbox"][2]),
            float(ds["bbox"][3]),
        ))
    ]


def _fetch_json_cached(
    url: str,
    cache_path: str,
    max_age_days: int = NAIP_METADATA_CACHE_MAX_AGE_DAYS,
) -> Any:
    """Fetch a JSON URL, caching the result until it expires."""
    cached = _load_fresh_json_cache(cache_path, max_age_days)
    if cached is not None:
        print(f"Using cached {cache_path}")
        return cached

    print(f"Fetching {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
    _write_json_cache(cache_path, data)
    return data


def _bbox_overlaps_stac_item(bbox: BBox, item: dict) -> bool:
    """Check if a bbox overlaps with a STAC item's bbox."""
    item_bbox = item.get("bbox")
    if not item_bbox or len(item_bbox) != 4:
        return False
    return _bbox_overlaps(bbox, (item_bbox[0], item_bbox[1], item_bbox[2], item_bbox[3]))


def discover_naip_tiles_digitalcoast(bbox: Optional[BBox], cache_dir: str) -> List[str]:
    """
    Discover NOAA Digital Coast NAIP imagery for overlapping island datasets.
    Returns a list of raster paths (VRT paths referencing /vsicurl/ URLs).
    """
    if not bbox:
        return []

    datasets = _find_digital_coast_datasets(bbox)
    if not datasets:
        print("Bounding box does not overlap any Digital Coast NAIP dataset.")
        return []

    raster_paths = []
    for ds in datasets:
        print(f"Querying Digital Coast dataset: {ds['name']} (id={ds['id']})")
        stac_cache = os.path.join(cache_dir, f"dc_stac_{ds['id']}.json")
        collection = _fetch_json_cached(ds["stac_url"], stac_cache)

        features = collection.get("features", [])
        if not features:
            print(f"No features found in STAC collection for {ds['name']}")
            continue

        filtered = [f for f in features if _bbox_overlaps_stac_item(bbox, f)]
        if not filtered:
            print(f"No {ds['name']} tiles overlap the bounding box.")
            continue

        tif_urls = []
        for feature in filtered:
            assets = feature.get("assets", {})
            for key, asset in assets.items():
                href = asset.get("href", "")
                if href.endswith(".tif"):
                    tif_urls.append(href)
                    break

        if not tif_urls:
            print(f"No TIFF assets found for {ds['name']}")
            continue

        print(f"Found {len(tif_urls)} {ds['name']} tiles overlapping the bounding box.")

        jp2_gb = len(tif_urls) * 80 / 1024
        tif_gb = len(tif_urls) * 500 / 1024
        print(f"Estimated size: {jp2_gb:.1f} GB (JP2) / {tif_gb:.1f} GB (ZIP/TIF)")
        if len(tif_urls) > 50:
            ans = input(f"Warning: you are about to use {len(tif_urls)} tiles from {ds['name']}. Are you sure you want to proceed? (y/N) ")
            if ans.lower() not in ('y', 'yes'):
                print("Aborting Digital Coast NAIP fetch.")
                return []

        ds_cache_dir = os.path.join(cache_dir, f"dc_{ds['id']}")
        os.makedirs(ds_cache_dir, exist_ok=True)
        vrt_path = os.path.join(ds_cache_dir, f"{ds['name']}.vrt")

        if os.path.exists(vrt_path):
            print(f"Using cached VRT: {vrt_path}")
        else:
            from osgeo import gdal

            vrt_ds = gdal.BuildVRT(vrt_path, tif_urls, options=gdal.BuildVRTOptions(resolution="highest"))
            if vrt_ds is None:
                raise RuntimeError(f"Failed to build VRT for {ds['name']}")
            vrt_ds = None
            print(f"Built cached VRT: {vrt_path}")

        raster_paths.append(vrt_path)

    return raster_paths


def get_vrt_path_for_zip(path: str) -> str:
    import zipfile
    if path.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(path, 'r') as z:
                for name in z.namelist():
                    if name.lower().endswith('.tif'):
                        return f"/vsizip/{os.path.abspath(path)}/{name}"
        except Exception as e:
            print(f"Warning: Failed to inspect zip file {path}: {e}")
    return path

def fetch_naip_downloads(scenes: List[Any], api_key: str, cache_dir: str) -> List[str]:
    """Request download options, queue the downloads, save to cache_dir, and return file paths."""
    if not scenes:
        return []
        
    downloaded_paths = []
    scenes_to_fetch = []
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    for scene in scenes:
        display_id = scene.get("displayId", "").lower()
        jp2_path = os.path.join(cache_dir, f"{display_id}.jp2")
        zip_path = os.path.join(cache_dir, f"{display_id}.ZIP")
        
        if os.path.exists(jp2_path):
            print(f"Skipping API fetch for {display_id}.jp2 (already exists in cache)")
            downloaded_paths.append(jp2_path)
        elif os.path.exists(zip_path):
            print(f"Skipping API fetch for {display_id}.ZIP (already exists in cache)")
            downloaded_paths.append(get_vrt_path_for_zip(zip_path))
        else:
            scenes_to_fetch.append(scene)
            
    if not scenes_to_fetch:
        return downloaded_paths
        
    entity_ids = [scene["entityId"] for scene in scenes_to_fetch]
    print(f"Requesting download options for {len(entity_ids)} scenes...")
    
    # We must chunk entityIds if there are too many, but up to 50k is usually supported.
    payload = {
        "datasetName": "NAIP",
        "entityIds": entity_ids
    }
    options = send_m2m_request("download-options", payload, api_key=api_key)
    
    # Group available options by entityId, preferring JP2 (smaller) over ZIP/TIFF
    options_by_entity = {}
    for option in options:
        if option.get("available") and option.get("downloadSystem") in ("EE", "dds"):
            eid = option["entityId"]
            product_name = option.get("productName", "").lower()
            is_jp2 = "jp2" in product_name or "jpeg2000" in product_name or "jpeg 2000" in product_name
            
            if eid not in options_by_entity:
                options_by_entity[eid] = option
            elif is_jp2 and "jp2" not in options_by_entity[eid].get("productName", "").lower():
                options_by_entity[eid] = option
    
    downloads = [{"entityId": opt["entityId"], "productId": opt["id"]} for opt in options_by_entity.values()]
            
    if not downloads:
        print("No valid download options found for these scenes.")
        return []
        
    print(f"Requesting downloads for {len(downloads)} products...")
    req_payload = {
        "downloads": downloads,
        "label": "satmaps-naip-download"
    }
    req_resp = send_m2m_request("download-request", req_payload, api_key=api_key)
    
    available_downloads = req_resp.get("availableDownloads", [])
    preparing_downloads = req_resp.get("preparingDownloads", [])
    
    print(f"Initial request: {len(available_downloads)} available, {len(preparing_downloads)} preparing.")
    
    download_urls = {d["downloadId"]: d["url"] for d in available_downloads}
    
    # Poll for preparing downloads
    if preparing_downloads:
        print("Polling for preparing downloads (this may take a while)...")
        pending_ids = {d["downloadId"] for d in preparing_downloads}
        
        while pending_ids:
            time.sleep(10)
            print(f"Checking status for {len(pending_ids)} pending downloads...")
            retrieve_payload = {"label": "satmaps-naip-download"}
            retrieved = send_m2m_request("download-retrieve", retrieve_payload, api_key=api_key)
            
            new_available = retrieved.get("available", [])
            for item in new_available:
                if item["downloadId"] in pending_ids:
                    download_urls[item["downloadId"]] = item["url"]
                    pending_ids.remove(item["downloadId"])
                    print(f"Resolved URL for {item['downloadId']}")
                    
            if not pending_ids:
                break
    
    print(f"All {len(download_urls)} download URLs resolved. Starting fetch to {cache_dir}...")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    downloaded_paths = []

    def _download_worker(dl_id: str, url: str) -> Optional[str]:
        max_retries = 3
        timeout = 1200 # 20 minutes
        
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    filename = response.info().get_filename()
                    if not filename:
                        filename = os.path.basename(urllib.parse.urlparse(url).path)
                        if not filename or filename == "/":
                            filename = f"naip_{dl_id}.tif"
                        elif not filename.endswith(".tif"):
                            filename = f"naip_{dl_id}_{filename[:8]}.tif"
                        
                    out_path = os.path.join(cache_dir, filename)
                    
                    if os.path.exists(out_path):
                        print(f"Skipping {filename} (already exists in cache)")
                        return out_path
                        
                    print(f"Downloading {filename} (attempt {attempt + 1})...")
                    with open(out_path, 'wb') as f:
                        while True:
                            chunk = response.read(8192 * 16)
                            if not chunk:
                                break
                            f.write(chunk)
                return out_path
            except Exception as e:
                # 4. If a download fails, wait before re-attempting
                if attempt < max_retries - 1:
                    print(f"Download failed for {url} ({e}), waiting 10s before re-attempting...")
                    time.sleep(10)
                else:
                    print(f"Failed to download {url} after {max_retries} attempts: {e}")
                    return None

    # 5. Use multi-threading on download URLs, the recommended number of concurrent downloads should be 5 or less
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_download_worker, dl_id, url): dl_id for dl_id, url in download_urls.items()}
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            if path:
                downloaded_paths.append(get_vrt_path_for_zip(path))

    return downloaded_paths

def add_naip_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register NAIP-specific CLI flags."""
    add = parser.add_argument
    add(
        "--use-naip",
        action="store_true",
        help="Use NAIP imagery as the primary land data source instead of Sentinel-2 MGRS",
    )

def handle_naip_workflow(
    args: argparse.Namespace,
    requested_bbox: Optional[BBox],
    extent_geometry: Optional[Any] = None,
    extent_geometries: Optional[Sequence[Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    If NAIP workflow is requested, handle it and return (True, list_of_rasters).
    Otherwise return (False, []).
    Combines EarthExplorer scenes with NOAA Digital Coast imagery for
    overlapping Hawaii and Puerto Rico/USVI coverage.
    """
    if not hasattr(args, "use_naip") or not args.use_naip:
        return False, []
        
    print("NAIP workflow requested.")
    load_env()
    
    cache_dir = getattr(args, "cache", "cache")

    scenes: List[Any] = []
    ee_raster_paths: List[str] = []
    api_key: Optional[str] = None
    try:
        api_key = m2m_login()
        scenes = discover_naip_tiles_ee(
            requested_bbox,
            api_key,
            cache_dir=cache_dir,
            extent_geometry=extent_geometry,
            extent_geometries=extent_geometries,
        )

        if scenes and requested_bbox:
            from osgeo import ogr
            import json

            scenes.sort(key=lambda s: s.get('temporalCoverage', {}).get('startDate', ''), reverse=True)

            if extent_geometry is not None:
                target_poly = extent_geometry.Clone()
            else:
                min_lon, min_lat, max_lon, max_lat = requested_bbox
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(min_lon, min_lat)
                ring.AddPoint(max_lon, min_lat)
                ring.AddPoint(max_lon, max_lat)
                ring.AddPoint(min_lon, max_lat)
                ring.AddPoint(min_lon, min_lat)
                target_poly = ogr.Geometry(ogr.wkbPolygon)
                target_poly.AddGeometry(ring)

            coverage_union = ogr.Geometry(ogr.wkbPolygon)
            filtered_scenes = []

            for s in scenes:
                geom_dict = s.get("spatialCoverage")
                if not geom_dict:
                    continue
                try:
                    scene_poly = ogr.CreateGeometryFromJson(json.dumps(geom_dict))
                except Exception:
                    continue

                uncovered = target_poly.Difference(coverage_union)
                if scene_poly.Intersects(uncovered):
                    intersection = scene_poly.Intersection(uncovered)
                    if intersection and intersection.GetArea() > 1e-8:
                        filtered_scenes.append(s)
                        if coverage_union.IsEmpty():
                            coverage_union = scene_poly.Clone()
                        else:
                            coverage_union = coverage_union.Union(scene_poly)

                        if target_poly.Difference(coverage_union).GetArea() < 1e-8:
                            break

            desc = "extent geometry" if extent_geometry is not None else "bounding box"
            print(f"Greedy spatial fill selected {len(filtered_scenes)} out of {len(scenes)} scenes to cover the {desc}.")
            scenes = filtered_scenes

        if scenes:
            print("NAIP pipeline via EarthExplorer initiated.")
            if getattr(args, "download", False) or not getattr(args, "estimate", False):
                if len(scenes) > 50:
                    ans = input(f"Warning: you are about to download {len(scenes)} DOQs. Are you sure you want to proceed? (y/N) ")
                    if ans.lower() not in ('y', 'yes'):
                        print("Aborting NAIP download.")
                        sys.exit(0)
                ee_raster_paths = fetch_naip_downloads(scenes, api_key, cache_dir)
                if getattr(args, "download", False):
                    print("NAIP download-only workflow complete. Exiting.")
                    sys.exit(0)
            else:
                for scene in scenes[:5]:
                    print(f"Scene ID: {scene.get('entityId')} - Display ID: {scene.get('displayId')}")
                if len(scenes) > 5:
                    print(f"... and {len(scenes) - 5} more. Run with --download to fetch them.")
    except Exception as e:
        print(f"EarthExplorer API Error: {e}")
    finally:
        if api_key:
            try:
                m2m_logout(api_key)
            except Exception as e:
                print(f"Failed to logout gracefully: {e}")

    # Digital Coast covers Hawaii and Puerto Rico/USVI, so check each requested
    # extent feature independently, even if EarthExplorer returned mainland scenes.
    dc_rasters: List[str] = []
    if requested_bbox:
        dc_bboxes = [
            (
                feature.GetEnvelope()[0],
                feature.GetEnvelope()[2],
                feature.GetEnvelope()[1],
                feature.GetEnvelope()[3],
            )
            for feature in (extent_geometries or ())
        ]
        if not dc_bboxes:
            dc_bboxes = [requested_bbox]
        for dc_bbox in dc_bboxes:
            dc_rasters.extend(discover_naip_tiles_digitalcoast(dc_bbox, cache_dir))
        dc_rasters = list(dict.fromkeys(dc_rasters))
    raster_paths = ee_raster_paths + dc_rasters
    if raster_paths:
        return True, raster_paths

    if scenes:
        return True, []

    if not scenes:
        print("No NAIP imagery found in the bounding box; aborting NAIP pipeline.")
        if not getattr(args, "estimate", False) and not getattr(args, "download", False):
            sys.exit(0)

    return True, []
