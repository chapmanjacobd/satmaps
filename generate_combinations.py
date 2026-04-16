import os
import subprocess
from itertools import product

# Configuration
MGRS_TILES = ["60HTB", "07VEK", "15TVG", "50HQJ", "49QGF", "45RVL", "32TQK", "12SUD", "40RCN", "28QCH"]
DATES = ["2025/07/01", "2025/01/01"]
FORMATS = ["webp", "jpg", "png"]
QUALITIES = [75, 80, 85]
RESAMPLING = ["bilinear", "average", "lanczos"]

OUTPUT_DIR = "combinations_output"
CACHE_DIR = "cache"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Phase 1: Download to cache
    print("--- Phase 1: Downloading source files to cache ---")
    for mgrs, date in product(MGRS_TILES, DATES):
        date_flat = date.replace("/", "-")
        date_cache_dir = os.path.join(CACHE_DIR, date_flat)
        os.makedirs(date_cache_dir, exist_ok=True)
        
        print(f"Downloading {mgrs} for {date} to {date_cache_dir}...")
        cmd = [
            "python3", "satmaps.py", mgrs,
            "--date", date,
            "--cache", date_cache_dir,
            "--download-only"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {mgrs} {date}: {e}")

    # Phase 2: Generation
    total_runs = len(MGRS_TILES) * len(DATES) * len(FORMATS) * len(QUALITIES) * len(RESAMPLING)
    print(f"\n--- Phase 2: Generating {total_runs} combinations ---")
    
    current_run = 0
    for mgrs, date, fmt, quality, resample in product(MGRS_TILES, DATES, FORMATS, QUALITIES, RESAMPLING):
        current_run += 1
        date_flat = date.replace("/", "-")
        date_cache_dir = os.path.join(CACHE_DIR, date_flat)
        output_name = f"{mgrs}_{date_flat}_{fmt}_q{quality}_{resample}.pmtiles"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        if os.path.exists(output_path):
            print(f"[{current_run}/{total_runs}] Skipping {output_name} (already exists)")
            continue
            
        print(f"[{current_run}/{total_runs}] Generating {output_name}...")
        
        cmd = [
            "python3", "satmaps.py", mgrs,
            "--date", date,
            "--format", fmt,
            "--quality", str(quality),
            "--resample-alg", resample,
            "--cache", date_cache_dir,
            "--output", output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating {output_name}: {e}")
            
if __name__ == "__main__":
    main()
