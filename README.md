# Satmaps

A toolkit for generating high-performance PMTiles from Sentinel-2 Global Mosaics hosted on the Copernicus Data Space Ecosystem (CDSE), featuring a high-fidelity NumPy-based processing pipeline.

## Overview

This repository provides tools to fetch, process, and package Sentinel-2 mosaic data directly from CDSE S3 storage into reprojected, optimized PMTiles. It features a custom tone-mapping engine to handle the high dynamic range of satellite data, producing visually balanced results for web mapping.

## Core Components

- `satmaps.py`: The primary engine. Handles GDAL S3 configuration, multi-date mosaicking, reprojection to Web Mercator (EPSG:3857), and PMTiles generation. Uses a NumPy-based pipeline for tone mapping and grading.
- `land_mgrs.py`: Builds and refreshes the reusable `land_mgrs.list` cache from GEBCO so `satmaps` can skip repeat land-tile discovery scans.
- `ocean.py`: Builds a standalone styled Web Mercator ocean background from GEBCO, with both global and bbox exports targeting a shared configurable Web Mercator output resolution (default zoom 13).
- `terrain.py`: Builds Terrarium-encoded Web Mercator PMTiles from GEBCO elevation data for `raster-dem` terrain sources.
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
Visit `http://localhost:5001` to adjust exposure, soft-knee curves, and saturation.

If the tuner reports missing cached sample tiles, fetch the built-in sample set first:

```bash
satmaps-tuner --download-samples
```

You can also target different mosaic dates, for example `satmaps-tuner --download-samples --date 2025/10/01,2025/07/01`.

### 2. Prepare `land_mgrs.list`

Generate the reusable land-tile cache before a bbox run, or anytime `land_mgrs.list` is missing:

```bash
# Global land/shallow-water cache reused by default all-tiles runs
land-mgrs

# BBox-specific cache reused by matching satmaps --bbox runs
land-mgrs --bbox -161,18,-154,23
```

`land-mgrs` defaults to `gebco_2025_sub_ice_topo_geotiff.zip` and writes `land_mgrs.list` in the current working directory. Re-run it with `--refresh` to force a fresh scan, or pass a different GEBCO zip/output path as positional arguments.

### 3. Generate the Ocean Background

Build the standalone styled ocean background and reuse it as an ocean base layer:

```bash
# Global export from the full masked GEBCO source raster at the default Web Mercator zoom 13
ocean

# Lower-data global export at Web Mercator zoom 4
ocean --max-zoom 4 ocean-z4.tif

# Crop and reproject to the same snapped tile-grid resolution used by bbox renders
ocean --bbox -161,18,-154,23

# Inspect the final styled RGBA VRT without translating to GeoTIFF
ocean --vrt
```

The first positional argument is the GEBCO zip path if you need something other than `gebco_2025_sub_ice_topo_geotiff.zip`, and the optional second positional argument is the output path (default: `ocean.tif`, or `ocean.vrt` when `--vrt` is used). Standalone ocean outputs default to Web Mercator zoom 13 (~19.11 m/px at the equator), but can target coarser runs such as zoom 4 for much smaller outputs.

### 4. Generate Terrain PMTiles

Build Terrarium-encoded terrain PMTiles directly from GEBCO:

```bash
# Global Terrarium DEM PMTiles from the full GEBCO raster
terrain

# BBox subset snapped to the same Web Mercator pixel grid used elsewhere
terrain --bbox -161,18,-154,23 --max-zoom 14 gebco_2025_sub_ice_topo_geotiff.zip hawaii-terrain.pmtiles
```

`terrain` keeps the raw GEBCO elevations, including bathymetry, and encodes them as Terrarium PNG tiles suitable for MapLibre `raster-dem` sources such as maps.black.

### 5. Generate Imagery PMTiles

Generate PMTiles for either a bbox subset or the default all-tiles run:

```bash
# If land_mgrs.list is missing, build it once before imagery generation
land-mgrs

# BBox render using either a full-coverage or bbox-matched standalone ocean background
land-mgrs --bbox -161,18,-154,23
satmaps --bbox -161,18,-154,23 --ocean-background ocean.tif -o hawaii.pmtiles

# Lower-data global run using a coarser ocean background and imagery zoom
satmaps --max-zoom 4 --ocean-background ocean-z4.tif -o all-tiles-z4.pmtiles

# Makefile shortcut for the graded Hawaii bbox preset
make hawaii

# Default all-tiles run
satmaps --ocean-background ocean.tif -o all-tiles.pmtiles

# Optional raster-first variant: render full aligned 3857 land rasters first,
# then tile the merged master raster once
satmaps --full-render-first --ocean-background ocean.tif -o all-tiles-raster-first.pmtiles
```

### 6. Estimate Resources

Before an all-tiles run, estimate the time and storage required:

```bash
satmaps --estimate
```

## Advanced Options

- `--bbox`: Discover MGRS tiles touched by a WGS84 bbox (`min_lon,min_lat,max_lon,max_lat`). When `--bbox` is omitted, `satmaps` processes all discovered land tiles from the source mosaics.
- `--date`: Comma-separated list of mosaic dates (default: `2025/07/01,2025/01/01`). Overlapping areas are averaged.
- `--quality`: Output WebP quality.
- `--resample-alg`: Resampling algorithm (`lanczos`, `bilinear`, `average`, `gauss`).
- `--chunk-zoom`: Chunking zoom used during MBTiles generation (default: `4`).
- `--parallel`: Number of worker processes/threads used for tile processing and chunk generation (default: `2`).
- `--tile-batch-width` / `--ty`: Target number of contiguous output tiles rendered together within one row during the final land pass (default: `32`).
- `--full-render-first`: Render each land work unit into a full aligned EPSG:3857 GeoTIFF first, build a master VRT, then tile that merged raster once. This usually trades higher temporary disk/RAM for less repeated final-tile work.
- `--blocksize`: GDAL tile block size used for MBTiles output (default: `512`).
- `--ocean-background`: Prebuilt standalone ocean background GeoTIFF (default: `ocean.tif`). Bbox runs use a bbox-local 3857 ocean raster snapped outward to the target Web Mercator tile pixel grid before max-zoom tile caching. Coarser ocean masks (for example z4-z13) can still be reused under finer land renders (for example z13-z14), including the initial tile discovery pass.
- Final Web Mercator land outputs target `--max-zoom` (supported: 4-14; default zoom 13, ~19.11 m/px at the equator). Ocean backgrounds may be reused from the same or a coarser zoom level and are resampled onto that final output grid during composition. Low-resolution runs at `--max-zoom 7` and below use a coarse-grid-first land renderer to avoid the full per-subtile pipeline.
- `--land` / `--no-land`: Enable or skip Sentinel-2 land tile processing entirely.
- `--grade` / `--no-grade`: Enable or disable final land grading.
- `--exposure`: Global brightness multiplier.
- `--cache`: Local directory for downloaded tiles (default: `.cache`).
- `--prefetch-cache`: Ephemeral cache directory for prefetched RGB bands (default: `<cache>.temp`).
- `--temp-dir`: Directory for heavyweight intermediary files, resume state, and the staged MBTiles (default: `.temp`).
- `--output` / `-o`: Final output PMTiles path (default: `output.pmtiles`).
- `--download`: Download source tiles into the cache and exit without building output tiles.
- Runs always resume automatically: `satmaps` reuses any matching `<temp-dir>/state_*.json` and on-disk tiles from a previous run with the same parameters, so interrupted runs continue where they left off.
- `--refresh-land-mgrs-list`: Rebuild `land_mgrs.list` in the repository root from `gebco_2025_sub_ice_topo_geotiff.zip` using the same generator as `land-mgrs --refresh`, then exit.
- `--estimate`: Print estimated time, RAM, disk, and network usage, then exit.

### `land-mgrs` Options

- `--bbox`: Generate a bbox-scoped `land_mgrs.list` for a matching `satmaps --bbox` run.
- `--refresh`: Force regeneration even when an existing matching `land_mgrs.list` is already present.
- First positional argument: Optional GEBCO zip path (default: `gebco_2025_sub_ice_topo_geotiff.zip`).
- Second positional argument: Optional output list path (default: `land_mgrs.list`).

### Ocean Background Options

`ocean` supports the same grading controls as `satmaps`, plus:

- `--bbox`: Export a Web Mercator ocean background cropped to a WGS84 bbox and snapped outward to the requested Web Mercator tile grid used for bbox renders in `satmaps`.
- `--max-zoom`: Target Web Mercator zoom used for output resolution (`4` through `14`).
- `--hillshade-z`: Vertical exaggeration passed to `gdaldem hillshade`.
- `--depth-min` / `--depth-max`: Depth range mapped onto the ocean color ramp.
- `--resample-alg`: GEBCO upscale kernel (`cubicspline` or `lanczos`).
- `--temp-dir`: Directory for intermediate rasters/VRTs.
- `--vrt`: Write the final styled RGBA VRT instead of translating to GeoTIFF.

### Terrain Options

`terrain` supports:

- `--bbox`: Export a WGS84 bbox subset instead of the full GEBCO DEM.
- `--max-zoom`: Target Web Mercator zoom used for the DEM resolution (`4` through `14`).
- `--chunk-zoom`: Chunking zoom used during Terrarium MBTiles generation (default: `8`).
- `--parallel`: Number of chunk worker processes (default: `2`).
- `--blocksize`: GDAL tile block size used for MBTiles output (default: `512`).
- `--resample-alg`: GEBCO warp kernel (`bilinear`, `cubicspline`, or `lanczos`).
- `--temp-dir`: Directory for intermediate rasters/VRTs.

### Grading Parameters

You can override the defaults (tuned via `satmaps-tuner`):
- `--exposure`: Global brightness multiplier.
- `--sat`: Final saturation adjustment.
- `--gamma`: Final gamma correction.
- `--shoulder`: Highlight shaping curve for moving the top end.
- `--db`, `--ls`: Low-tone grading breakpoint and slope.
- `--ghb`, `--gms`, `--ghs`: Mid/high grading breakpoint and contrast slopes.

## Technical Details

1.  Discovery: Lists S3 folders for all available tiles and dates when no bbox is supplied, or discovers touched MGRS tiles from `--bbox`.
2.  Mosaicking: Opens RGB bands (B04, B03, B02) for each date and averages complete observations to reduce cloud artifacts.
3.  Masking: Optionally warps the configured ocean background onto each tile grid so coastal alpha and fill decisions happen block-by-block.
4.  Reprojection: Warps processed land tiles to Web Mercator (EPSG:3857) and composites them with the standalone ocean background.
5.  Processing (NumPy):
    - Soft-Knee Tone Mapping: A 3-segment linear curve to compress high dynamic range while preserving local contrast.
    - Color Grading: Exposure, gamma/shoulder shaping, and contrast controls for a "natural" look.
6.  Packaging: By default, `satmaps` renders each land work unit and the prepared ocean background into a resumable max-zoom `z/x/y.webp` cache, batching neighboring final land tiles together row-by-row when possible, then copies those WebP bytes into MBTiles, builds lower zooms with `gdaladdo`, and converts the archive to PMTiles. With `--full-render-first`, it instead writes full aligned 3857 land rasters first, builds a master VRT, and tiles that merged raster tree once before packaging.

## Datasets

- European Space Agency -- Copernicus Sentinel data 2025
- GEBCO Compilation Group -- GEBCO 2025 Grid (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)
