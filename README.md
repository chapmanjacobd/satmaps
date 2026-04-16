# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE).

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It supports multiple image formats, quality levels, and resampling algorithms for comparison and production use.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, band merging (RGB), reprojection to Web Mercator (EPSG:3857), and MBTiles/PMTiles generation.
- `generate_combinations.py`: A batch processing script to generate a matrix of comparisons across different MGRS tiles, dates, and compression settings.
- `viewer.html`: A lightweight MapLibre GL JS viewer for side-by-side comparison of generated PMTiles.

## Prerequisites

- GDAL: Must be compiled with S3 support.
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

### Single MGRS tile test output

```bash
python3 satmaps.py 31TDF --format webp --quality 75 --resample-alg bilinear -o output.pmtiles
```

### Batch Comparison

The `generate_combinations.py` script runs in two phases:
1. Phase 1: Downloads required MGRS tiles for specified dates into a local `cache/` to avoid redundant S3 requests.
2. Phase 2: Iterates through permutations of format, quality, and resampling to generate comparison files in `combinations_output/`.

```bash
python3 generate_combinations.py
```

### Viewing Results

To compare the outputs, serve the directory via HTTP and open the viewer:

```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000/viewer.html`.

## Technical Details

- Source: `s3://eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/`
- Projection: Source data is reprojected from UTM (MGRS) to Web Mercator (EPSG:3857).
- Processing Pipeline:
    1. Direct VSI S3 reading of B04, B03, B02 bands.
    2. Virtual Raster (VRT) creation for RGB stacking.
    3. `gdalwarp` to EPSG:3857.
    4. `gdal_translate` to MBTiles with internal scale parameters (`0, 4000, 0, 255`).
    5. `gdaladdo` for overview generation.
    6. `pmtiles convert` for final packaging.
