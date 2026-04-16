import os
import subprocess
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# Configuration
MGRS_TILES = ["60HTB", "07VEK", "15TVG", "50HQJ", "49QGF", "45RVL", "32TQK", "12SUD", "40RCN", "28QCH"]
DATES = ["2025/07/01", "2025/01/01"]
FORMATS = ["webp", "jpg", "png"]
QUALITIES = [75, 80, 85]
RESAMPLING = ["bilinear", "average", "lanczos"]

OUTPUT_DIR = "combinations_output"
CACHE_DIR = "cache"
MAX_WORKERS = os.cpu_count() or 4

def run_satmaps(mgrs: str, date: str, fmt: str, quality: int, resample: str, cache_dir: str, output_path: str) -> bool:
    """Helper to run a single satmaps generation command."""
    cmd = [
        "python3", "satmaps.py", mgrs,
        "--date", date,
        "--format", fmt,
        "--quality", str(quality),
        "--resample-alg", resample,
        "--cache", cache_dir,
        "--output", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {os.path.basename(output_path)}: {e.stderr.decode()}")
        return False

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Phase 1: Download to cache (sequential to avoid overloading S3/bandwidth)
    print("--- Phase 1: Downloading source files to cache ---")
    for mgrs, date in product(MGRS_TILES, DATES):
        date_flat = date.replace("/", "-")
        date_cache_dir = os.path.join(CACHE_DIR, date_flat)
        os.makedirs(date_cache_dir, exist_ok=True)
        
        print(f"Checking/Downloading {mgrs} for {date}...")
        cmd = [
            "python3", "satmaps.py", mgrs,
            "--date", date,
            "--cache", date_cache_dir,
            "--download-only"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {mgrs} {date}: {e.stderr.decode()}")

    # Phase 2: Generation (Parallel)
    tasks = []
    for mgrs, date, fmt, quality, resample in product(MGRS_TILES, DATES, FORMATS, QUALITIES, RESAMPLING):
        date_flat = date.replace("/", "-")
        date_cache_dir = os.path.join(CACHE_DIR, date_flat)
        output_name = f"{mgrs}_{date_flat}_{fmt}_q{quality}_{resample}.pmtiles"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        if os.path.exists(output_path):
            continue
            
        tasks.append((mgrs, date, fmt, quality, resample, date_cache_dir, output_path))

    total_tasks = len(tasks)
    print(f"\n--- Phase 2: Generating {total_tasks} combinations (using {MAX_WORKERS} workers) ---")
    
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_satmaps, *task): task for task in tasks}
        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            output_name = os.path.basename(task[-1])
            if future.result():
                print(f"[{completed}/{total_tasks}] Success: {output_name}")
            else:
                print(f"[{completed}/{total_tasks}] Failed: {output_name}")
            
if __name__ == "__main__":
    main()
