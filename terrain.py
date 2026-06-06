#!/usr/bin/env python3
import argparse
import os

from common import build_output_namespace, build_output_namespace_dir, build_staged_path, publish_staged_path, remove_if_exists
from osgeo import gdal

import ocean
import satmaps
import tiler

gdal.UseExceptions()

DEFAULT_OUTPUT = "terrain.pmtiles"
DEFAULT_CHUNK_ZOOM = 6


def generate_terrain_pmtiles(
    gebco_zip: str,
    destination: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    temp_dir: str = ".temp",
    resample_alg: str = "cubicspline",
    max_zoom: int = ocean.DEFAULT_MAX_ZOOM,
    chunk_zoom: int = DEFAULT_CHUNK_ZOOM,
    parallel: int = 2,
    blocksize: int = 512,
) -> str:
    """Generate Terrarium-encoded PMTiles from the full GEBCO elevation raster."""
    os.makedirs(temp_dir, exist_ok=True)
    stem = satmaps.temp_basename_from_output(destination)
    unique_id = build_output_namespace(destination, default_stem="terrain")
    output_temp_dir = build_output_namespace_dir(temp_dir, unique_id)
    os.makedirs(output_temp_dir, exist_ok=True)

    source_vrt = os.path.join(output_temp_dir, f"{stem}_source.vrt")
    warped_vrt = os.path.join(output_temp_dir, f"{stem}_3857.vrt")

    ocean.build_gebco_source_vrt(gebco_zip, source_vrt)

    warp_kwargs: dict[str, object] = {
        "format": "VRT",
        "dstSRS": "EPSG:3857",
        "resampleAlg": resample_alg,
        "multithread": True,
        "warpOptions": ["NUM_THREADS=ALL_CPUS"],
    }
    if bbox is None:
        warp_kwargs["outputBounds"] = ocean.WEB_MERCATOR_WORLD_BOUNDS
        warp_kwargs["xRes"] = tiler.web_mercator_pixel_size_for_tile_size(max_zoom, blocksize)
        warp_kwargs["yRes"] = tiler.web_mercator_pixel_size_for_tile_size(max_zoom, blocksize)
    else:
        snapped_bounds, pixel_size, _zoom = ocean.snapped_tile_grid_for_bbox(
            bbox,
            max_zoom,
            tile_size=blocksize,
        )
        warp_kwargs["outputBounds"] = snapped_bounds
        warp_kwargs["xRes"] = pixel_size
        warp_kwargs["yRes"] = pixel_size

    staged_warped_vrt = build_staged_path(warped_vrt)
    remove_if_exists(staged_warped_vrt)
    warped_ds = gdal.Warp(
        staged_warped_vrt,
        source_vrt,
        options=gdal.WarpOptions(**warp_kwargs),
    )
    if warped_ds is None:
        raise RuntimeError(f"Could not warp GEBCO terrain into Web Mercator: {warped_vrt}")
    warped_ds = None
    publish_staged_path(staged_warped_vrt, warped_vrt)

    packaged_tiles = satmaps.convert_raster_to_pmtiles(
        warped_vrt,
        destination,
        unique_id,
        tile_format="png",
        quality=100,
        resample_alg=resample_alg,
        chunk_zoom=chunk_zoom,
        parallel=parallel,
        blocksize=blocksize,
        name="GEBCO Terrain",
        description="GEBCO Terrarium DEM",
        requested_bbox=bbox,
        tiling_options={"elevation_encoding": "terrarium"},
    )

    for path in [
        source_vrt,
        warped_vrt,
        packaged_tiles.temp_mbtiles,
        *packaged_tiles.tiling_artifacts.cleanup_paths,
    ]:
        remove_if_exists(path)

    return destination


def build_terrain_argument_parser() -> argparse.ArgumentParser:
    """Build the terrain CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Terrarium-encoded PMTiles from GEBCO elevation data. "
            f"Defaults to Web Mercator zoom {ocean.DEFAULT_MAX_ZOOM}."
        )
    )
    add = parser.add_argument
    add(
        "gebco_zip",
        nargs="?",
        default=ocean.DEFAULT_GEBCO_ZIP,
        help="GEBCO zip archive",
    )
    add(
        "destination",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Output PMTiles path",
    )
    add(
        "--bbox",
        help=(
            "Optional WGS84 bbox as min_lon,min_lat,max_lon,max_lat. "
            "When omitted, exports the full GEBCO DEM in EPSG:3857."
        ),
    )
    add(
        "--max-zoom",
        type=int,
        choices=list(ocean.SUPPORTED_MAX_ZOOMS),
        default=ocean.DEFAULT_MAX_ZOOM,
        help="Target Web Mercator zoom used for output resolution",
    )
    add(
        "--chunk-zoom",
        type=int,
        default=DEFAULT_CHUNK_ZOOM,
        help="Zoom level to chunk the Terrarium PMTiles generation at",
    )
    add(
        "--parallel",
        type=int,
        default=2,
        help="Number of parallel chunk processes",
    )
    add(
        "--blocksize",
        type=int,
        default=512,
        help="Output tile size passed to the MBTiles driver",
    )
    add(
        "--temp-dir",
        default=".temp",
        help="Directory for intermediary files",
    )
    add(
        "--resample-alg",
        choices=["bilinear", "cubicspline", "lanczos"],
        default="cubicspline",
        help="Resampling algorithm for the GEBCO warp into EPSG:3857",
    )
    return parser


def main() -> None:
    args = build_terrain_argument_parser().parse_args()

    output = generate_terrain_pmtiles(
        args.gebco_zip,
        args.destination,
        bbox=ocean.parse_bbox(args.bbox) if args.bbox else None,
        temp_dir=args.temp_dir,
        resample_alg=args.resample_alg,
        max_zoom=args.max_zoom,
        chunk_zoom=args.chunk_zoom,
        parallel=args.parallel,
        blocksize=args.blocksize,
    )
    print(output)


if __name__ == "__main__":
    main()
