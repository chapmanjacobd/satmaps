# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE), featuring a high-fidelity NumPy-based processing pipeline.

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It features a custom tone-mapping engine to handle the high dynamic range of satellite data, producing visually balanced results for web mapping.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, multi-date mosaicking, reprojection to Web Mercator (EPSG:3857), and PMTiles generation. Uses a NumPy-based pipeline for tone mapping and grading.
- `ocean.py`: Builds a standalone styled Web Mercator ocean background from GEBCO, with both global and bbox exports targeting a shared Web Mercator zoom-13 output resolution.
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

## Installation

```bash
pip install git+https://github.com/chapmanjacobd/satmaps.git
```

## Usage

### 1. Tune Visuals (Optional)

Before a large run, use the Tuner UI to find the best visual parameters:

```bash
satmaps-tuner
```
Visit `http://localhost:5001` to adjust exposure, soft-knee curves, and saturation. Note: Requires some data in `.cache/2025-07-01` (e.g., from a small `satmaps` run).

### 2. Generate the Ocean Background

Build the standalone styled ocean background and reuse it as an ocean base layer:

```bash
# Global export from the full masked GEBCO source raster at Web Mercator zoom 13
ocean

# Crop and reproject to the same zoom-13 snapped tile-grid resolution used by bbox renders
ocean --bbox -161,18,-154,23

# Inspect the final styled RGBA VRT without translating to GeoTIFF
ocean --vrt
```

The first positional argument is the GEBCO zip path if you need something other than `gebco_2025_sub_ice_topo_geotiff.zip`, and the optional second positional argument is the output path (default: `ocean.tif`, or `ocean.vrt` when `--vrt` is used). Standalone ocean outputs target Web Mercator zoom 13 (~19.11 m/px at the equator).

### 3. Generate PMTiles

Generate PMTiles for specific MGRS tiles or a global run:

```bash
# Single tile (defaults to 2025/07/01 and 2025/01/01 mosaics)
satmaps 31TDF -o barcelona.pmtiles

# Multiple tiles with custom quality and format
satmaps 31TCF,31TDF,31TCE,31TDE --format webp --quality 80 -o region.pmtiles

# BBox render using either a global or bbox-matched standalone ocean background
satmaps --bbox -161,18,-154,23 --ocean-background ocean.tif -o hawaii.pmtiles

# Global run
satmaps --global --ocean-background ocean.tif -o global.pmtiles
```

### 4. Estimate Resources

Before a global run, estimate the time and storage required:

```bash
satmaps --global --estimate
```

## Advanced Options

- `--global`: Process all discovered land tiles from the source mosaics.
- `--bbox`: Discover MGRS tiles touched by a WGS84 bbox (`min_lon,min_lat,max_lon,max_lat`).
- `--date`: Comma-separated list of mosaic dates (default: `2025/07/01,2025/01/01`). Overlapping areas are averaged.
- `--format`: Output tile format (`webp`, `jpg`, `png`, `png8`).
- `--quality`: Output tile quality for lossy formats.
- `--resample-alg`: Resampling algorithm (`lanczos`, `bilinear`, `average`, `gauss`).
- `--chunk-zoom`: Chunking zoom used during MBTiles generation (default: `4`).
- `--parallel`: Number of worker processes/threads used for tile processing and chunk generation (default: `2`).
- `--blocksize`: GDAL tile block size used for MBTiles output (default: `512`).
- `--ocean-background`: Prebuilt standalone ocean background GeoTIFF (default: `ocean.tif`). Bbox runs use a bbox-local 3857 ocean raster snapped outward to the target Web Mercator tile pixel grid before chunk generation.
- Final Web Mercator land and ocean outputs target zoom 13 (~19.11 m/px at the equator) so merged VRTs and downstream tiling share one explicit output resolution.
- `--land` / `--no-land`: Enable or skip Sentinel-2 land tile processing entirely.
- `--tonemap` / `--no-tonemap`: Enable or disable the land tone-mapping stage.
- `--grade` / `--no-grade`: Enable or disable final land grading.
- `--cache`: Local directory for downloaded tiles (default: `.cache`).
- `--download`: Download source tiles into the cache and exit without building output tiles.
- `--resume [STATE_FILE]`: Resume a previous run from a saved `.temp/state_*.json`; without a path, the most recent state file is used.
- `--estimate`: Print estimated time, RAM, disk, and network usage, then exit.
- `--vrt`: Generate the final VRT and exit (useful for inspection in QGIS).

### Ocean Background Options

`ocean` supports the same tone-mapping controls as `satmaps`, plus:

- `--bbox`: Export a Web Mercator ocean background cropped to a WGS84 bbox and snapped outward to the zoom-13 Web Mercator tile grid used for bbox renders in `satmaps`.
- `--hillshade-z`: Vertical exaggeration passed to `gdaldem hillshade`.
- `--depth-min` / `--depth-max`: Depth range mapped onto the ocean color ramp.
- `--resample-alg`: GEBCO upscale kernel (`cubicspline` or `lanczos`).
- `--temp-dir`: Directory for intermediate rasters/VRTs.
- `--vrt`: Write the final styled RGBA VRT instead of translating to GeoTIFF.

### Tone Mapping Parameters

You can override the defaults (tuned via `satmaps-tuner`):
- `--exposure`: Global brightness multiplier.
- `--sb`, `--hb`: Shadow and highlight "break" points for the soft-knee curve.
- `--ss`, `--ms`, `--hs`: Slopes for shadow, mid-tone, and highlight segments.
- `--sat`: Final saturation adjustment.
- `--gamma`: Final gamma correction.

## Technical Details

1.  Discovery: Lists S3 folders for requested MGRS tiles and dates, or discovers touched MGRS tiles from `--bbox`.
2.  Mosaicking: Opens RGB bands (B04, B03, B02) for each date and averages complete observations to reduce cloud artifacts.
3.  Masking: Optionally warps the configured ocean background onto each tile grid so coastal alpha and fill decisions happen block-by-block.
4.  Reprojection: Warps processed land tiles to Web Mercator (EPSG:3857) and composites them with the standalone ocean background.
5.  Processing (NumPy):
    - Soft-Knee Tone Mapping: A 3-segment linear curve to compress high dynamic range while preserving local contrast.
    - Color Grading: Saturation adjustment and gamma correction for a "natural" look.
6.  Packaging: The merged Web Mercator VRT is split into XYZ-aligned chunks at `--chunk-zoom`, each chunk is translated into MBTiles, those MBTiles are merged, and the result is converted to PMTiles. For `--bbox` runs, chunk selection stays clipped to the requested bbox rather than the full background extent.

## Datasets

- European Space Agency -- Copernicus Sentinel data 2025
- GEBCO Compilation Group -- GEBCO 2025 Grid (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)
