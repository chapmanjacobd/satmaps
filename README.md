# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE), featuring a high-fidelity NumPy-based processing pipeline.

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It features a custom tone-mapping engine to handle the high dynamic range of satellite data, producing visually balanced results for web mapping.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, multi-date mosaicking, reprojection to Web Mercator (EPSG:3857), and PMTiles generation. Uses a NumPy-based pipeline for tone mapping and grading.
- `ocean_background.py`: Builds a standalone `ocean.tif` GEBCO hillshade background in EPSG:3857, matching the effective resolution of a 10m UTM source projected into Web Mercator.
- `tuner_ui.py`: A Flask-based interactive web interface to fine-tune tone mapping parameters (exposure, contrast, saturation) in real-time on sample data.
- `tiler.py`: Core logic for tile processing, tone mapping algorithms, and parallelized chunk execution.

## Prerequisites

- Python 3.11+
- GDAL: Must be compiled with S3 support.
- PMTiles CLI: For final conversion from MBTiles to PMTiles.
- AWS CLI: For tile discovery and S3 access.

### Configuration

Configure an AWS profile named `cdse` with your [CDSE S3 credentials](https://documentation.dataspace.copernicus.eu/APIs/S3.html):

```bash
aws configure --profile cdse
```

## Usage

### 1. Tune Visuals (Optional)

Before a large run, use the Tuner UI to find the best visual parameters:

```bash
python3 tuner_ui.py
```
Visit `http://localhost:5001` to adjust exposure, soft-knee curves, and saturation. Note: Requires some data in `.cache/2025-07-01` (e.g., from a small `satmaps.py` run).

### 2. Generate the Ocean Background

Prebuild the standalone ocean hillshade once and reuse it anywhere you want an ocean base layer:

```bash
python3 ocean_background.py --bbox -161,18,-154,23
```

The first positional argument is the GEBCO zip path if you need something other than `gebco_2025_sub_ice_topo_geotiff.zip`, and the optional second positional argument is the output path (default: `ocean.tif`).

### 3. Generate PMTiles

Generate PMTiles for specific MGRS tiles or a global run:

```bash
# Single tile (defaults to 2025/07/01 and 2025/01/01 mosaics)
python3 satmaps.py 31TDF -o barcelona.pmtiles

# Multiple tiles with custom quality and format
python3 satmaps.py 31TCF,31TDF,31TCE,31TDE --format webp --quality 80 -o region.pmtiles

# BBox render using the prebuilt standalone ocean background
python3 satmaps.py --bbox -161,18,-154,23 --ocean-background ocean.tif -o hawaii.pmtiles

# Global run using a land-only filter
python3 satmaps.py --global --land-only HLS.land.tiles.txt --ocean-background ocean.tif -o global.pmtiles
```

### 4. Estimate Resources

Before a global run, estimate the time and storage required:

```bash
python3 satmaps.py --global --land-only HLS.land.tiles.txt --estimate
```

## Advanced Options

- `--date`: Comma-separated list of mosaic dates (default: `2025/07/01,2025/01/01`). Overlapping areas are averaged.
- `--resample-alg`: Resampling algorithm (`lanczos`, `bilinear`, `average`, `gauss`).
- `--ocean-background`: Prebuilt standalone ocean background GeoTIFF (default: `ocean.tif`).
- `--no-soft-knee`: Disable the multi-segment tone mapping curve.
- `--no-grading`: Disable final saturation and gamma adjustments.
- `--cache`: Local directory for downloaded tiles (default: `.cache`).
- `--vrt`: Generate the final VRT and exit (useful for inspection in QGIS).

### Tone Mapping Parameters

You can override the defaults (tuned via `tuner_ui.py`):
- `--exposure`: Global brightness multiplier.
- `--sb`, `--hb`: Shadow and highlight "break" points for the soft-knee curve.
- `--ss`, `--ms`, `--hs`: Slopes for shadow, mid-tone, and highlight segments.
- `--sat`: Final saturation adjustment.
- `--gamma`: Final gamma correction.

## Technical Details

1.  Discovery: Lists S3 folders for requested MGRS tiles and dates.
2.  Mosaicking: Builds VRTs that combine bands (B04, B03, B02) and average multiple dates to reduce cloud artifacts.
3.  Reprojection: Warps data to Web Mercator (EPSG:3857).
4.  Processing (NumPy):
    - Soft-Knee Tone Mapping: A 3-segment linear curve to compress high dynamic range while preserving local contrast.
    - Color Grading: Saturation adjustment and gamma correction for a "natural" look.
5.  Packaging: Tiles are generated in chunks, merged into an MBTiles database, and converted to PMTiles.

## Datasets

- European Space Agency -- Copernicus Sentinel data 2025
- GEBCO Compilation Group -- GEBCO 2025 Grid (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)
