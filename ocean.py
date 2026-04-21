#!/usr/bin/env python3
import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile, which
from typing import Sequence
from xml.sax.saxutils import escape

import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import label

from tiler import (
    MAKO_RAMP,
    apply_preview_correction_numpy,
    apply_soft_knee_numpy,
    colorize_depth_numpy,
    lonlat_bbox_to_mercator_bounds,
    snap_bounds_to_pixel_grid,
    web_mercator_pixel_size,
)

gdal.UseExceptions()

DEFAULT_GEBCO_ZIP = "gebco_2025_sub_ice_topo_geotiff.zip"
DEFAULT_OUTPUT = "ocean.tif"
GEBCO_OCEAN_NODATA = -32767.0
WEB_MERCATOR_WORLD_BOUNDS = (
    -20037508.342789244,
    -20037508.342789244,
    20037508.342789244,
    20037508.342789244,
)
DEFAULT_MAX_ZOOM = 13
SUPPORTED_MAX_ZOOMS = (13, 14)
GTIFF_CREATION_OPTIONS = (
    "BIGTIFF=YES",
    "TILED=YES",
    "COMPRESS=ZSTD",
    "ZSTD_LEVEL=5",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)
OCEAN_DEFAULT_EXPOSURE = 1.0
OCEAN_DEFAULT_SHADOW_BREAK = 0.3
OCEAN_DEFAULT_HIGHLIGHT_BREAK = 0.75
OCEAN_DEFAULT_SHADOW_SLOPE = 1.4
OCEAN_DEFAULT_MID_SLOPE = 0.9
OCEAN_DEFAULT_HIGHLIGHT_SLOPE = 0.5
OCEAN_DEFAULT_GAMMA = 1.2
OCEAN_DEFAULT_SATURATION = 1.0
OCEAN_DEFAULT_BLACK_BREAK = 0.35
OCEAN_DEFAULT_BLACK_SLOPE = 0.35
OCEAN_FADE_DEPTH = -50.0
SMALL_OCEAN_MAX_AREA_SQ_M = 1_500_000.0
MAX_COMPONENT_CLEANUP_PIXELS = 120_000_000


@dataclass(frozen=True)
class OceanBackgroundArtifacts:
    source_vrt: str
    masked_vrt: str
    warped_vrt: str
    alpha_vrt: str
    hillshade_tif: str
    color_tif: str
    rgba_vrt: str
    output_tif: str


@dataclass(frozen=True)
class OceanStyleOptions:
    tonemap: bool = True
    grade: bool = True
    exposure: float = OCEAN_DEFAULT_EXPOSURE
    shadow_break: float = OCEAN_DEFAULT_SHADOW_BREAK
    highlight_break: float = OCEAN_DEFAULT_HIGHLIGHT_BREAK
    shadow_slope: float = OCEAN_DEFAULT_SHADOW_SLOPE
    mid_slope: float = OCEAN_DEFAULT_MID_SLOPE
    highlight_slope: float = OCEAN_DEFAULT_HIGHLIGHT_SLOPE
    gamma: float = OCEAN_DEFAULT_GAMMA
    saturation: float = OCEAN_DEFAULT_SATURATION
    black_break: float = OCEAN_DEFAULT_BLACK_BREAK
    black_slope: float = OCEAN_DEFAULT_BLACK_SLOPE
    depth_min: float = -11000.0
    depth_max: float = 0.0


def build_gebco_source_vrt(gebco_zip: str, output_vrt: str) -> str:
    """Build a source VRT from the GEBCO zip archive."""
    if not os.path.exists(gebco_zip):
        raise FileNotFoundError(f"GEBCO zip not found: {gebco_zip}")

    gebco_vsi = f"/vsizip/{gebco_zip}"
    files_in_zip = gdal.ReadDir(gebco_vsi) or []
    tif_paths = [
        f"{gebco_vsi}/{name}" for name in files_in_zip if name.lower().endswith(".tif")
    ]
    if not tif_paths:
        raise RuntimeError(f"No GeoTIFF files found in GEBCO zip: {gebco_zip}")

    gdal.BuildVRT(output_vrt, tif_paths)
    return output_vrt


def create_gebco_ocean_vrt(source_vrt: str, output_vrt: str) -> str:
    """Mask positive GEBCO values so only ocean elevations remain."""
    ds = gdal.Open(source_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open GEBCO source VRT: {source_vrt}")

    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = GEBCO_OCEAN_NODATA

    geotransform = ",".join(str(value) for value in ds.GetGeoTransform())
    projection = escape(ds.GetProjection())
    source_filename = escape(os.path.abspath(source_vrt))
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None

    with open(output_vrt, "w") as f:
        f.write(
            f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
    <NoDataValue>{nodata_value}</NoDataValue>
    <PixelFunctionType>expression</PixelFunctionType>
    <PixelFunctionArguments dialect="muparser" expression="B1 &gt; 0.001 ? {nodata_value} : B1"/>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    return output_vrt


def create_alpha_vrt(source_vrt: str, output_vrt: str) -> str:
    """Create an explicit alpha mask VRT from depth thresholds and land-preferred cleanup."""
    ds = gdal.Open(source_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open source VRT for alpha generation: {source_vrt}")

    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = GEBCO_OCEAN_NODATA

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    alpha_tif = str(Path(output_vrt).with_suffix(".tif"))
    driver = gdal.GetDriverByName("GTiff")
    alpha_ds = driver.Create(alpha_tif, xsize, ysize, 1, gdal.GDT_Byte, options=list(GTIFF_CREATION_OPTIONS))
    if alpha_ds is None:
        raise RuntimeError(f"Could not create alpha TIFF: {alpha_tif}")
    alpha_ds.SetProjection(ds.GetProjection())
    alpha_ds.SetGeoTransform(ds.GetGeoTransform())
    alpha_band = alpha_ds.GetRasterBand(1)

    block_width, block_height = band.GetBlockSize()
    if block_width <= 0:
        block_width = 512
    if block_height <= 0:
        block_height = 512

    for yoff in range(0, ysize, block_height):
        bh = min(block_height, ysize - yoff)
        for xoff in range(0, xsize, block_width):
            bw = min(block_width, xsize - xoff)
            depths = band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
            ocean_mask = (depths > nodata_value + 0.1) & (depths < OCEAN_FADE_DEPTH)
            alpha_band.WriteArray(ocean_mask.astype(np.uint8) * 255, xoff=xoff, yoff=yoff)

    alpha_band.FlushCache()
    if xsize * ysize <= MAX_COMPONENT_CLEANUP_PIXELS:
        ocean_mask = alpha_band.ReadAsArray().astype(bool)
        cleaned_mask = remove_small_enclosed_ocean_regions(
            ocean_mask,
            ds.GetGeoTransform(),
            SMALL_OCEAN_MAX_AREA_SQ_M,
        )
        alpha_band.WriteArray(cleaned_mask.astype(np.uint8) * 255)
        alpha_band.FlushCache()
    else:
        remove_small_enclosed_ocean_regions_vector(
            alpha_ds,
            alpha_band,
            ds.GetGeoTransform(),
            SMALL_OCEAN_MAX_AREA_SQ_M,
            Path(alpha_tif).with_suffix(".gpkg"),
        )

    alpha_ds = None
    ds = None
    gdal.BuildVRT(output_vrt, [alpha_tif])
    return output_vrt


def remove_small_enclosed_ocean_regions(
    ocean_mask: np.ndarray,
    geotransform: Sequence[float],
    max_area_sq_m: float,
) -> np.ndarray:
    """Remove enclosed ocean components smaller than the configured area threshold."""
    if ocean_mask.ndim != 2:
        raise ValueError("ocean_mask must be 2D")
    if ocean_mask.size == 0 or max_area_sq_m <= 0.0:
        return ocean_mask

    pixel_area_sq_m = abs(geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4])
    if pixel_area_sq_m <= 0.0:
        raise ValueError("geotransform must define a positive pixel area")

    labels, component_count = label(ocean_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return ocean_mask

    component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    edge_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    enclosed_small_labels = [
        label_id
        for label_id in range(1, component_count + 1)
        if label_id not in edge_labels and component_sizes[label_id] * pixel_area_sq_m < max_area_sq_m
    ]
    if not enclosed_small_labels:
        return ocean_mask

    cleaned_mask = ocean_mask.copy()
    cleaned_mask[np.isin(labels, enclosed_small_labels)] = False
    return cleaned_mask


def remove_small_enclosed_ocean_regions_vector(
    alpha_dataset: gdal.Dataset,
    alpha_band: gdal.Band,
    geotransform: Sequence[float],
    max_area_sq_m: float,
    vector_path: Path,
) -> None:
    """Remove small enclosed ocean polygons without loading the whole mask into memory."""
    vector_driver = ogr.GetDriverByName("GPKG")
    if vector_driver is None:
        raise RuntimeError("Could not load OGR GPKG driver for ocean mask cleanup")
    if vector_path.exists():
        vector_driver.DeleteDataSource(str(vector_path))

    vector_ds = vector_driver.CreateDataSource(str(vector_path))
    if vector_ds is None:
        raise RuntimeError(f"Could not create temporary vector dataset: {vector_path}")

    spatial_ref = alpha_dataset.GetSpatialRef()
    raw_layer = vector_ds.CreateLayer("raw_ocean", srs=spatial_ref, geom_type=ogr.wkbPolygon)
    kept_layer = vector_ds.CreateLayer("kept_ocean", srs=spatial_ref, geom_type=ogr.wkbPolygon)
    for layer in (raw_layer, kept_layer):
        if layer is None:
            raise RuntimeError(f"Could not create cleanup layer in {vector_path}")
    value_field = ogr.FieldDefn("value", ogr.OFTInteger)
    if raw_layer.CreateField(value_field) != 0:
        raise RuntimeError(f"Could not create polygon value field in {vector_path}")

    polygonize_result = gdal.Polygonize(alpha_band, None, raw_layer, 0, [], callback=None)
    if polygonize_result != 0:
        raise RuntimeError("Could not polygonize ocean alpha mask")

    xsize = alpha_dataset.RasterXSize
    ysize = alpha_dataset.RasterYSize
    min_x = geotransform[0]
    max_y = geotransform[3]
    max_x = geotransform[0] + geotransform[1] * xsize + geotransform[2] * ysize
    min_y = geotransform[3] + geotransform[4] * xsize + geotransform[5] * ysize
    edge_tolerance = max(abs(geotransform[1]), abs(geotransform[5]))
    kept_definition = kept_layer.GetLayerDefn()

    for feature in raw_layer:
        if feature.GetFieldAsInteger(0) != 255:
            continue
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        envelope = geometry.GetEnvelope()
        touches_edge = (
            envelope[0] <= min_x + edge_tolerance
            or envelope[1] >= max_x - edge_tolerance
            or envelope[2] <= min_y + edge_tolerance
            or envelope[3] >= max_y - edge_tolerance
        )
        if touches_edge or geometry.GetArea() >= max_area_sq_m:
            kept_feature = ogr.Feature(kept_definition)
            kept_feature.SetGeometry(geometry.Clone())
            if kept_layer.CreateFeature(kept_feature) != 0:
                raise RuntimeError(f"Could not write kept ocean feature into {vector_path}")
            kept_feature = None

    alpha_band.Fill(0)
    rasterize_result = gdal.RasterizeLayer(alpha_dataset, [1], kept_layer, burn_values=[255])
    if rasterize_result != 0:
        raise RuntimeError("Could not rasterize cleaned ocean alpha mask")
    alpha_band.FlushCache()
    vector_ds = None


def build_ocean_ramp_colors(style: OceanStyleOptions) -> np.ndarray:
    """Return the styled MAKO depth ramp as float32 RGB triples in [0, 1]."""
    mako_colors = np.array([c[1:] for c in MAKO_RAMP], dtype=np.float32) / 255.0
    mako_arr = mako_colors.T.reshape(3, -1, 1)

    if style.tonemap:
        toned_mako = apply_soft_knee_numpy(
            mako_arr,
            shadow_break=style.shadow_break,
            highlight_break=style.highlight_break,
            shadow_slope=style.shadow_slope,
            mid_slope=style.mid_slope,
            highlight_slope=style.highlight_slope,
            exposure=style.exposure,
        )
        mako_colors = toned_mako.reshape(3, -1).T
    else:
        mako_colors = np.clip(mako_colors * style.exposure, 0.0, 1.0)

    if style.grade:
        graded_mako = apply_preview_correction_numpy(
            mako_colors.T.reshape(3, -1, 1),
            saturation=style.saturation,
            darken_break=style.black_break,
            low_slope=style.black_slope,
            gamma=style.gamma,
        )
        mako_colors = graded_mako.reshape(3, -1).T

    return np.asarray(mako_colors, dtype=np.float32)


def colorize_ocean_depths(depths: np.ndarray, style: OceanStyleOptions) -> np.ndarray:
    """Colorize GEBCO depth values using the shared ocean styling pipeline."""
    return colorize_depth_numpy(
        depths,
        build_ocean_ramp_colors(style),
        style.depth_min,
        style.depth_max,
    )


def target_web_mercator_pixel_size(max_zoom: int = DEFAULT_MAX_ZOOM) -> float:
    """Return the shared Web Mercator output pixel size used across exports."""
    return web_mercator_pixel_size(max_zoom)


def snapped_tile_grid_for_bbox(
    bbox: tuple[float, float, float, float],
    max_zoom: int = DEFAULT_MAX_ZOOM,
) -> tuple[tuple[float, float, float, float], float, int]:
    """Snap a bbox outward to the target Web Mercator tile pixel grid."""
    pixel_size = target_web_mercator_pixel_size(max_zoom)
    zoom = max_zoom
    mercator_bounds = lonlat_bbox_to_mercator_bounds(*bbox)
    snapped_bounds = snap_bounds_to_pixel_grid(mercator_bounds, pixel_size)
    return snapped_bounds, pixel_size, zoom


def build_hillshade_command(
    input_vrt: str,
    output_tif: str,
    z_factor: float,
    creation_options: Sequence[str] = GTIFF_CREATION_OPTIONS,
) -> list[str]:
    """Build the gdaldem hillshade command line."""
    command = [
        "gdaldem",
        "hillshade",
        input_vrt,
        output_tif,
        "-multidirectional",
        "-z",
        str(z_factor),
    ]
    for option in creation_options:
        command.extend(["-co", option])
    return command


def create_ocean_rgb_tif(
    depth_vrt: str,
    hillshade_tif: str,
    output_tif: str,
    style: OceanStyleOptions,
) -> str:
    """Colorize warped GEBCO depths and modulate them with hillshade."""
    depth_ds = gdal.Open(depth_vrt)
    if depth_ds is None:
        raise RuntimeError(f"Could not open ocean depth VRT: {depth_vrt}")

    hillshade_ds = gdal.Open(hillshade_tif)
    if hillshade_ds is None:
        raise RuntimeError(f"Could not open hillshade TIFF: {hillshade_tif}")

    depth_band = depth_ds.GetRasterBand(1)
    hillshade_band = hillshade_ds.GetRasterBand(1)
    block_width, block_height = hillshade_band.GetBlockSize()
    if block_width <= 0:
        block_width = 512
    if block_height <= 0:
        block_height = 512
    ramp_colors = build_ocean_ramp_colors(style)

    driver = gdal.GetDriverByName("GTiff")
    color_ds = driver.Create(
        output_tif,
        hillshade_ds.RasterXSize,
        hillshade_ds.RasterYSize,
        3,
        gdal.GDT_Byte,
        options=list(GTIFF_CREATION_OPTIONS),
    )
    if color_ds is None:
        raise RuntimeError(f"Could not create colorized ocean TIFF: {output_tif}")

    color_ds.SetProjection(hillshade_ds.GetProjection())
    color_ds.SetGeoTransform(hillshade_ds.GetGeoTransform())

    for band_index, color_name in enumerate(("RedBand", "GreenBand", "BlueBand"), start=1):
        band = color_ds.GetRasterBand(band_index)
        band.SetColorInterpretation(getattr(gdal, f"GCI_{color_name}"))

    color_bands = [color_ds.GetRasterBand(index) for index in range(1, 4)]
    for yoff in range(0, hillshade_ds.RasterYSize, block_height):
        bh = min(block_height, hillshade_ds.RasterYSize - yoff)
        for xoff in range(0, hillshade_ds.RasterXSize, block_width):
            bw = min(block_width, hillshade_ds.RasterXSize - xoff)
            depths = depth_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
            hillshade = hillshade_band.ReadAsArray(xoff, yoff, bw, bh).astype(np.float32)
            rgb = colorize_depth_numpy(
                depths,
                ramp_colors,
                style.depth_min,
                style.depth_max,
            )
            shade = 0.35 + 0.65 * np.clip(hillshade / 255.0, 0.0, 1.0)
            shaded_rgb = np.clip(rgb * shade[np.newaxis, :, :], 0.0, 1.0)
            byte_arr = (shaded_rgb * 255.0).astype(np.uint8)
            for band_index, band in enumerate(color_bands):
                band.WriteArray(byte_arr[band_index], xoff=xoff, yoff=yoff)

    color_ds.FlushCache()
    color_ds = None
    depth_ds = None
    hillshade_ds = None
    return output_tif


def create_rgb_with_alpha_vrt(rgb_tif: str, alpha_vrt: str, output_vrt: str) -> str:
    """Attach an explicit alpha band to an RGB GeoTIFF via VRT."""
    rgb_ds = gdal.Open(rgb_tif)
    if rgb_ds is None:
        raise RuntimeError(f"Could not open RGB TIFF: {rgb_tif}")

    alpha_ds = gdal.Open(alpha_vrt)
    if alpha_ds is None:
        raise RuntimeError(f"Could not open alpha VRT: {alpha_vrt}")

    alpha_tif = str(Path(output_vrt).with_suffix(".alpha.tif"))
    translated_alpha = gdal.Translate(
        alpha_tif,
        alpha_vrt,
        options=gdal.TranslateOptions(
            format="GTiff",
            creationOptions=list(GTIFF_CREATION_OPTIONS),
        ),
    )
    if translated_alpha is None:
        raise RuntimeError(f"Could not materialize alpha TIFF: {alpha_tif}")
    translated_alpha = None

    xsize = rgb_ds.RasterXSize
    ysize = rgb_ds.RasterYSize
    geotransform = ",".join(str(value) for value in rgb_ds.GetGeoTransform())
    projection = escape(rgb_ds.GetProjection())
    rgb_filename = escape(os.path.abspath(rgb_tif))
    alpha_filename = escape(os.path.abspath(alpha_tif))
    rgb_ds = None
    alpha_ds = None

    with open(output_vrt, "w") as f:
        f.write(
            f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <ColorInterp>Red</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="2">
    <ColorInterp>Green</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="3">
    <ColorInterp>Blue</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{rgb_filename}</SourceFilename>
      <SourceBand>3</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="4">
    <ColorInterp>Alpha</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{alpha_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    return output_vrt


def translate_rgba_vrt(rgba_vrt: str, destination: str) -> str:
    """Materialize the RGBA VRT as a tiled GeoTIFF."""
    destination_dir = os.path.dirname(destination)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=list(GTIFF_CREATION_OPTIONS),
    )
    gdal.Translate(destination, rgba_vrt, options=options)
    return destination


def write_rgba_vrt(rgba_vrt: str, destination: str) -> str:
    """Persist the final RGBA VRT to a caller-visible output path."""
    destination_dir = os.path.dirname(destination)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    copyfile(rgba_vrt, destination)
    return destination


def generate_ocean_background(
    gebco_zip: str,
    destination: str,
    bbox: tuple[float, float, float, float] | None = None,
    temp_dir: str = ".temp",
    resample_alg: str = "cubicspline",
    hillshade_z: float = 5.0,
    style: OceanStyleOptions | None = None,
    vrt: bool = False,
    max_zoom: int = DEFAULT_MAX_ZOOM,
) -> OceanBackgroundArtifacts:
    """Generate a standalone RGBA ocean background output."""
    if which("gdaldem") is None:
        raise RuntimeError("gdaldem is required to generate the ocean background")

    os.makedirs(temp_dir, exist_ok=True)
    stem = Path(destination).stem or "ocean"

    source_vrt = os.path.join(temp_dir, f"{stem}_source.vrt")
    masked_vrt = os.path.join(temp_dir, f"{stem}_masked.vrt")
    warped_vrt = os.path.join(temp_dir, f"{stem}_3857.vrt")
    alpha_vrt = os.path.join(temp_dir, f"{stem}_alpha.vrt")
    hillshade_tif = os.path.join(temp_dir, f"{stem}_hillshade.tif")
    color_tif = os.path.join(temp_dir, f"{stem}_color.tif")
    rgba_vrt = os.path.join(temp_dir, f"{stem}_rgba.vrt")

    build_gebco_source_vrt(gebco_zip, source_vrt)
    create_gebco_ocean_vrt(source_vrt, masked_vrt)

    if style is None:
        style = OceanStyleOptions()

    warp_kwargs: dict[str, object] = {
        "format": "VRT",
        "dstSRS": "EPSG:3857",
        "resampleAlg": resample_alg,
    }
    if bbox is None:
        warp_kwargs["outputBounds"] = WEB_MERCATOR_WORLD_BOUNDS
        warp_kwargs["xRes"] = target_web_mercator_pixel_size(max_zoom)
        warp_kwargs["yRes"] = target_web_mercator_pixel_size(max_zoom)
    else:
        snapped_bounds, pixel_size, _zoom = snapped_tile_grid_for_bbox(bbox, max_zoom)
        warp_kwargs["outputBounds"] = snapped_bounds
        warp_kwargs["xRes"] = pixel_size
        warp_kwargs["yRes"] = pixel_size
    warp_options = gdal.WarpOptions(**warp_kwargs)
    gdal.Warp(warped_vrt, masked_vrt, options=warp_options)

    create_alpha_vrt(warped_vrt, alpha_vrt)
    subprocess.run(build_hillshade_command(warped_vrt, hillshade_tif, hillshade_z), check=True)
    create_ocean_rgb_tif(warped_vrt, hillshade_tif, color_tif, style)
    create_rgb_with_alpha_vrt(color_tif, alpha_vrt, rgba_vrt)
    if vrt:
        translate_path = str(Path(destination).with_suffix(".vrt"))
        write_rgba_vrt(rgba_vrt, translate_path)
        destination = translate_path
    else:
        translate_rgba_vrt(rgba_vrt, destination)

    return OceanBackgroundArtifacts(
        source_vrt=source_vrt,
        masked_vrt=masked_vrt,
        warped_vrt=warped_vrt,
        alpha_vrt=alpha_vrt,
        hillshade_tif=hillshade_tif,
        color_tif=color_tif,
        rgba_vrt=rgba_vrt,
        output_tif=destination,
    )


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bbox.split(","))
    if len(values) != 4:
        raise ValueError("bbox must contain four comma-separated values")
    return (values[0], values[1], values[2], values[3])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone GEBCO ocean hillshade GeoTIFF. "
            f"Defaults to Web Mercator zoom {DEFAULT_MAX_ZOOM} "
            f"(~{target_web_mercator_pixel_size(DEFAULT_MAX_ZOOM):.2f} m/px at the equator)."
        )
    )
    parser.add_argument(
        "gebco_zip", nargs="?", default=DEFAULT_GEBCO_ZIP, help="GEBCO zip archive"
    )
    parser.add_argument(
        "destination", nargs="?", default=DEFAULT_OUTPUT, help="Output GeoTIFF path"
    )
    parser.add_argument(
        "--bbox",
        help=(
            "Optional WGS84 bbox as min_lon,min_lat,max_lon,max_lat. "
            f"When omitted, exports the full masked source raster in EPSG:3857 at "
            f"Web Mercator zoom {DEFAULT_MAX_ZOOM}."
        ),
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        choices=list(SUPPORTED_MAX_ZOOMS),
        default=DEFAULT_MAX_ZOOM,
        help="Target Web Mercator zoom used for output resolution",
    )
    parser.add_argument("--temp-dir", default=".temp", help="Directory for intermediary files")
    parser.add_argument(
        "--resample-alg",
        choices=["cubicspline", "lanczos"],
        default="cubicspline",
        help="Resampling algorithm for the GEBCO upscale into EPSG:3857",
    )
    parser.add_argument(
        "--hillshade-z",
        type=float,
        default=5.0,
        help="Vertical exaggeration passed to gdaldem hillshade",
    )
    parser.add_argument(
        "--tonemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ocean tone mapping before colorization",
    )
    parser.add_argument(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ocean final grading before colorization",
    )
    parser.add_argument("--exposure", type=float, default=OCEAN_DEFAULT_EXPOSURE)
    parser.add_argument(
        "--sb", "--shadow-break", type=float, default=OCEAN_DEFAULT_SHADOW_BREAK
    )
    parser.add_argument(
        "--hb", "--highlight-break", type=float, default=OCEAN_DEFAULT_HIGHLIGHT_BREAK
    )
    parser.add_argument(
        "--ss", "--shadow-slope", type=float, default=OCEAN_DEFAULT_SHADOW_SLOPE
    )
    parser.add_argument(
        "--ms", "--mid-slope", type=float, default=OCEAN_DEFAULT_MID_SLOPE
    )
    parser.add_argument(
        "--hs", "--highlight-slope", type=float, default=OCEAN_DEFAULT_HIGHLIGHT_SLOPE
    )
    parser.add_argument("--gamma", type=float, default=OCEAN_DEFAULT_GAMMA)
    parser.add_argument(
        "--sat", "--saturation", type=float, default=OCEAN_DEFAULT_SATURATION
    )
    parser.add_argument(
        "--db", "--black-break", type=float, default=OCEAN_DEFAULT_BLACK_BREAK
    )
    parser.add_argument(
        "--ls", "--black-slope", type=float, default=OCEAN_DEFAULT_BLACK_SLOPE
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=-11000.0,
        help="Depth value mapped to the start of the ocean color ramp",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=0.0,
        help="Depth value mapped to the end of the ocean color ramp",
    )
    parser.add_argument(
        "--vrt",
        action="store_true",
        help="Write the final styled RGBA VRT instead of translating it to a GeoTIFF",
    )
    args = parser.parse_args()

    artifacts = generate_ocean_background(
        gebco_zip=args.gebco_zip,
        destination=args.destination,
        bbox=parse_bbox(args.bbox) if args.bbox else None,
        temp_dir=args.temp_dir,
        resample_alg=args.resample_alg,
        hillshade_z=args.hillshade_z,
        style=OceanStyleOptions(
            tonemap=args.tonemap,
            grade=args.grade,
            exposure=args.exposure,
            shadow_break=args.sb,
            highlight_break=args.hb,
            shadow_slope=args.ss,
            mid_slope=args.ms,
            highlight_slope=args.hs,
            gamma=args.gamma,
            saturation=args.sat,
            black_break=args.db,
            black_slope=args.ls,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        ),
        vrt=args.vrt,
        max_zoom=args.max_zoom,
    )
    print(artifacts.output_tif)


if __name__ == "__main__":
    main()
