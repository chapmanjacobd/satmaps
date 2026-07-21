#!/usr/bin/env python3
import argparse
import urllib.request
import urllib.parse
import json
import os
import sys
import time
import concurrent.futures
from typing import List, Optional, Tuple, Any

BBox = Tuple[float, float, float, float]

M2M_BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

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

def discover_naip_tiles_ee(bbox: Optional[BBox], api_key: str) -> List[Any]:
    """
    Query the EarthExplorer M2M API for NAIP tiles intersecting the given bbox.
    """
    if not bbox:
        print("Warning: No bounding box provided. Please provide --bbox.")
        return []

    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Construct an MBR (Minimum Bounding Rectangle) spatial filter
    payload = {
        "datasetName": "NAIP",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {"latitude": min_lat, "longitude": min_lon},
            "upperRight": {"latitude": max_lat, "longitude": max_lon}
        },
        "maxResults": 50000
    }
    
    print(f"Querying EarthExplorer for NAIP imagery in bbox {bbox}...")
    results = send_m2m_request("scene-search", payload, api_key=api_key)
    
    scenes = results.get("results", [])
    print(f"Found {len(scenes)} NAIP scenes in the bounding box.")
    return scenes

def fetch_naip_downloads(scenes: List[Any], api_key: str, cache_dir: str) -> List[str]:
    """Request download options, queue the downloads, save to cache_dir, and return file paths."""
    if not scenes:
        return []
        
    entity_ids = [scene["entityId"] for scene in scenes]
    print(f"Requesting download options for {len(entity_ids)} scenes...")
    
    # We must chunk entityIds if there are too many, but up to 50k is usually supported.
    payload = {
        "datasetName": "NAIP",
        "entityIds": entity_ids
    }
    options = send_m2m_request("download-options", payload, api_key=api_key)
    
    downloads = []
    for option in options:
        if option.get("available") and option.get("downloadSystem") in ("EE", "dds"):
            downloads.append({
                "entityId": option["entityId"],
                "productId": option["id"]
            })
            
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
                downloaded_paths.append(path)

    return downloaded_paths

def add_naip_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register NAIP-specific CLI flags."""
    add = parser.add_argument
    add(
        "--use-naip",
        action="store_true",
        help="Use NAIP imagery as the primary land data source instead of Sentinel-2 MGRS",
    )

def handle_naip_workflow(args: argparse.Namespace, requested_bbox: Optional[BBox]) -> Tuple[bool, List[str]]:
    """
    If NAIP workflow is requested, handle it and return (True, list_of_rasters).
    Otherwise return (False, []).
    """
    if not hasattr(args, "use_naip") or not args.use_naip:
        return False, []
        
    print("NAIP (EarthExplorer) workflow requested.")
    load_env()
    
    api_key = None
    try:
        api_key = m2m_login()
        scenes = discover_naip_tiles_ee(requested_bbox, api_key)
        
        if scenes:
            # Sort by date descending
            scenes.sort(key=lambda s: s.get('temporalCoverage', {}).get('startDate', ''), reverse=True)
            most_recent_date = scenes[0].get('temporalCoverage', {}).get('startDate', '')
            most_recent_year = most_recent_date[:4] if most_recent_date else None
            
            if most_recent_year:
                filtered_scenes = [s for s in scenes if s.get('temporalCoverage', {}).get('startDate', '').startswith(most_recent_year)]
                print(f"Filtering NAIP scenes to the most recent year ({most_recent_year}): {len(filtered_scenes)} of {len(scenes)} scenes.")
                scenes = filtered_scenes

            cache_dir = getattr(args, "cache", "cache")
            # When integrating into PMTiles, we MUST fetch to get local TIFFs
            # unless it's a dry run (where we just print and exit)
            if getattr(args, "download", False) or not getattr(args, "estimate", False):
                raster_paths = fetch_naip_downloads(scenes, api_key, cache_dir)
                if getattr(args, "download", False):
                    print("NAIP download-only workflow complete. Exiting.")
                    sys.exit(0)
                return True, raster_paths
            else:
                for scene in scenes[:5]:
                    print(f"Scene ID: {scene.get('entityId')} - Display ID: {scene.get('displayId')}")
                if len(scenes) > 5:
                    print(f"... and {len(scenes) - 5} more. Run with --download to fetch them.")
                
        print("NAIP pipeline via EarthExplorer initiated.")
    except Exception as e:
        print(f"EarthExplorer API Error: {e}")
    finally:
        if api_key:
            try:
                m2m_logout(api_key)
            except Exception as e:
                print(f"Failed to logout gracefully: {e}")
                
    return True, []
