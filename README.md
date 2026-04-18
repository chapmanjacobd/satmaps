# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE).

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It supports multiple image formats, quality levels, and resampling algorithms for comparison and production use.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, band merging (RGB), multi-date mosaicking, reprojection to Web Mercator (EPSG:3857), and MBTiles/PMTiles or inspection-VRT generation.
- `tiler.py`: Dedicated module for robust tiling using `gdal2tiles.py` for large-scale (global) processing.
- `testbench.py`: A batch processing script to generate a matrix of comparisons across different MGRS tiles, dates, and compression settings.
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
- `--stats-min`/`--stats-max`: Hardcoded source min/max values for consistent scaling across different runs. These now feed step 6's built-in soft-knee tone curve, so they are the main controls for balancing shadow lift against highlight preservation.
- `--cache`: Local directory to store downloaded `.tif` files.
- `--vrt`: Stop after generating the final Byte VRT in `.temp/` for inspection in QGIS, skipping MBTiles/PMTiles packaging.
- `--step5`: Materialize step 5 as a tiled ZSTD-compressed GeoTIFF so downstream VRT stages reference a real raster file instead of a warped VRT.

### Batch Comparison

The `testbench.py` script runs in two phases:
1. Phase 1: Downloads required MGRS tiles for specified dates into a local `cache/` to avoid redundant S3 requests.
2. Phase 2: Iterates through permutations of format, quality, resampling, exponents, and scaling to generate comparison files in `combinations_output/`.

```bash
python3 testbench.py
```

### Viewing Results

To compare the outputs, serve the directory via HTTP and open the viewer:

```bash
python3 serve.py
```
Then visit `http://localhost:8000/viewer.html`. The viewer dynamically reads the configuration from `testbench.py` to populate its controls.

## Technical Details

- Source: `s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/`
- Projection: Source data is reprojected from UTM (MGRS) to Web Mercator (EPSG:3857).
- Processing Pipeline:
    1. Discovery: List S3 folders for requested MGRS tiles and dates.
    2. Step 1: Create per-tile RGB VRTs from B04, B03, B02 bands (VSI S3 or local cache).
    3. Step 2: Build grouped VRT mosaics to keep the number of open files manageable.
    4. Step 3: Build per-date mosaic VRTs and average overlaps with a derived-band expression.
    5. Step 4: Merge the requested date mosaics into a master VRT.
    6. Step 5: Reproject the master VRT to Web Mercator (EPSG:3857), optionally materialized as a tiled ZSTD GeoTIFF with `--step5`.
    7. Step 6: Create a tone-mapped Float32 VRT with a built-in soft-knee curve that lifts shadows and rolls off highlights.
    8. Step 7: Create the final Byte-conversion VRT used either for QGIS inspection (`--vrt`) or chunked packaging.
    9. Packaging: Translate chunk MBTiles, merge them, build overviews, and optionally run `pmtiles convert` for distribution.
