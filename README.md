# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE).

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It supports multiple image formats, quality levels, and resampling algorithms for comparison and production use.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, band merging (RGB), multi-date mosaicking, reprojection to Web Mercator (EPSG:3857), and MBTiles/PMTiles generation.
- `tiler.py`: Dedicated module for robust tiling using `gdal2tiles.py` for large-scale (global) processing.
- `generate_combinations.py`: A batch processing script to generate a matrix of comparisons across different MGRS tiles, dates, and compression settings.
- `viewer.html`: A side-by-side MapLibre GL JS viewer with dynamic configuration, synchronized views, and zoom shortcuts.

## Prerequisites

- GDAL: Must be compiled with S3 support. `gdal2tiles.py` is required for global runs.
- Python 3: With `gdal` bindings installed.
- PMTiles CLI: For converting MBTiles to PMTiles.
- AWS CLI: For tile discovery and S3 access.

### Configuration

You must configure an AWS profile named `cdse` to access the Copernicus data:

```bash
aws configure --profile cdse
```
Use your [CDSE](https://documentation.dataspace.copernicus.eu/APIs/S3.html) credentials.

## Usage

### Single or Multiple MGRS tiles

Generate PMTiles for one or more specific MGRS tiles:

```bash
# Single tile
python3 satmaps.py 31TDF --format webp --quality 75 -o output.pmtiles

# Multiple tiles (comma-separated)
python3 satmaps.py 31TCF,31TDF,31TCE,31TDE --date 2025/07/01 -o region.pmtiles
```

### Global Run

Process all available mosaic tiles. It is recommended to use a land-only filter:

```bash
python3 satmaps.py --global --land-only HLS.land.tiles.txt -o global.pmtiles
```

### Advanced Options

- `--date`: Comma-separated list of mosaic dates (e.g., `2025/07/01,2025/01/01`). Multiple dates are averaged to handle overlaps and reduce cloud artifacts.
- `--exponent`: Power-law scaling exponent (gamma correction). Default is `0.4`.
- `--stats-min`/`--stats-max`: Hardcoded source min/max values for consistent scaling across different runs.
- `--cache`: Local directory to store downloaded `.tif` files.

### Batch Comparison

The `generate_combinations.py` script runs in two phases:
1. Phase 1: Downloads required MGRS tiles for specified dates into a local `cache/` to avoid redundant S3 requests.
2. Phase 2: Iterates through permutations of format, quality, resampling, exponents, and scaling to generate comparison files in `combinations_output/`.

```bash
python3 generate_combinations.py
```

### Viewing Results

To compare the outputs, serve the directory via HTTP and open the viewer:

```bash
python3 serve.py
```
Then visit `http://localhost:8000/viewer.html`. The viewer dynamically reads the configuration from `generate_combinations.py` to populate its controls.

## Technical Details

- Source: `s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/`
- Projection: Source data is reprojected from UTM (MGRS) to Web Mercator (EPSG:3857).
- Processing Pipeline:
    1. Discovery: List S3 folders for requested MGRS tiles and dates.
    2. RGB Stacking: Create VRTs from B04, B03, B02 bands (VSI S3 or local cache).
    3. Mosaicking: Merge multiple dates using a mean pixel function (VRT derived band).
    4. Reprojection: `gdalwarp` to Web Mercator (EPSG:3857) using cubic resampling.
    5. Tiling & Packaging:
        - Individual MGRS Tile (`gdal_translate`):
            - One-pass conversion to MBTiles (8-bit scaling + power-law exponent).
            - `gdaladdo` to generate overviews inside the MBTiles.
        - Global Tiles takes a slightly different path by using `gdal2tiles` because it supports resuming:
            - Convert to 8-bit scaled VRT (applying exponent).
            - `gdal2tiles.py` for parallel, resumable tile pyramid generation.
            - Custom SQLite packing of raw tiles into MBTiles.
    6. Final Delivery: `pmtiles convert` for optimized distribution.

