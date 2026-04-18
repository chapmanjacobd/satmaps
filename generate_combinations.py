import os
import subprocess
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

MGRS_TILES = ["45RVL", "28QCH"]
DATES = [["2025/07/01", "2025/01/01"]]
FORMATS = ["webp"]
QUALITIES = [74]
RESAMPLING = ["lanczos"]
EXPONENTS = [0.3, 0.4, 0.5, 0.6, 0.8]
SCALING_RANGES = [(0, 6000), (0, 9000), (0, 11000), (0, 13000)]
MIN_ZOOM = 0
MAX_ZOOM = 14
BLOCKSIZE = 512

OUTPUT_DIR = "combinations_output"
CACHE_DIR = "cache"
MAX_WORKERS = min(8, os.cpu_count())

def run_satmaps(mgrs: str, dates: list, fmt: str, quality: int, resample: str, exponent: float, src_min: int, src_max: int, cache_dir: str, output_path: str) -> bool:
    """Helper to run a single satmaps generation command."""
    date_arg = ",".join(dates)
    cmd = [
        "python3", "satmaps.py", mgrs,
        "--date", date_arg,
        "--format", fmt,
        "--quality", str(quality),
        "--resample-alg", resample,
        "--exponent", str(exponent),
        "--stats-min", str(src_min),
        "--stats-max", str(src_max),
        "--minzoom", str(MIN_ZOOM),
        "--maxzoom", str(MAX_ZOOM),
        "--blocksize", str(BLOCKSIZE),
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
    unique_dates = set()
    for date_list in DATES:
        for d in date_list:
            unique_dates.add(d)

    for mgrs, date in product(MGRS_TILES, unique_dates):
        print(f"Checking/Downloading {mgrs} for {date}...")
        cmd = [
            "python3", "satmaps.py", mgrs,
            "--date", date,
            "--cache", CACHE_DIR,
            "--download-only"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {mgrs} {date}: {e.stderr.decode()}")

    # Phase 2: Generation (Parallel)
    tasks = []
    for mgrs, date_list, fmt, quality, resample, exponent, (src_min, src_max) in product(MGRS_TILES, DATES, FORMATS, QUALITIES, RESAMPLING, EXPONENTS, SCALING_RANGES):
        if len(date_list) == 1:
            date_flat = date_list[0].replace("/", "-")
        else:
            date_flat = "combined_" + "_".join([d.replace("/", "-") for d in date_list])
            
        output_name = f"{mgrs}_{date_flat}_{fmt}_q{quality}_{resample}_e{exponent}_s{src_min}-{src_max}.pmtiles"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        if os.path.exists(output_path):
            continue
            
        tasks.append((mgrs, date_list, fmt, quality, resample, exponent, src_min, src_max, CACHE_DIR, output_path))

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
