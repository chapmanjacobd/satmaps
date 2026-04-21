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
from osgeo import gdal, osr

from tiler import (
    MAKO_RAMP,
    apply_preview_correction_numpy,
    apply_soft_knee_numpy,
    colorize_depth_numpy,
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
    """Create an explicit alpha mask VRT from a nodata comparison."""
    ds = gdal.Open(source_vrt)
    if ds is None:
        raise RuntimeError(f"Could not open source VRT for alpha generation: {source_vrt}")

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
    <PixelFunctionType>expression</PixelFunctionType>
    <PixelFunctionArguments dialect="muparser" expression="B1 == {nodata_value} ? 0 : 255"/>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    return output_vrt


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


def resolution_for_utm_10m_3857(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float]:
    """Approximate the EPSG:3857 pixel size of a 10m UTM grid at the bbox center."""
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    if center_lat < -80.0 or center_lat > 84.0:
        raise ValueError("UTM projection requires the bbox center to be between 80S and 84N")

    zone = max(1, min(60, int((center_lon + 180.0) / 6.0) + 1))
    epsg = (32600 if center_lat >= 0.0 else 32700) + zone

    wgs84 = osr.SpatialReference()
    utm = osr.SpatialReference()
    web_mercator = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    utm.ImportFromEPSG(epsg)
    web_mercator.ImportFromEPSG(3857)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    web_mercator.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    to_utm = osr.CoordinateTransformation(wgs84, utm)
    to_3857 = osr.CoordinateTransformation(utm, web_mercator)

    center_x, center_y, _ = to_utm.TransformPoint(center_lon, center_lat)
    center_merc_x, center_merc_y, _ = to_3857.TransformPoint(center_x, center_y)
    east_merc_x, _, _ = to_3857.TransformPoint(center_x + 10.0, center_y)
    _, north_merc_y, _ = to_3857.TransformPoint(center_x, center_y + 10.0)

    x_res = abs(east_merc_x - center_merc_x)
    y_res = abs(north_merc_y - center_merc_y)
    if x_res <= 0.0 or y_res <= 0.0:
        raise RuntimeError(
            "Could not derive a positive EPSG:3857 resolution from the bbox center"
        )
    return x_res, y_res


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

    xsize = rgb_ds.RasterXSize
    ysize = rgb_ds.RasterYSize
    geotransform = ",".join(str(value) for value in rgb_ds.GetGeoTransform())
    projection = escape(rgb_ds.GetProjection())
    rgb_filename = escape(os.path.abspath(rgb_tif))
    alpha_filename = escape(os.path.abspath(alpha_vrt))
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


def create_hillshade_rgba_vrt(hillshade_tif: str, alpha_vrt: str, output_vrt: str) -> str:
    """Repeat a grayscale hillshade into RGB and attach an explicit alpha band."""
    hillshade_ds = gdal.Open(hillshade_tif)
    if hillshade_ds is None:
        raise RuntimeError(f"Could not open hillshade TIFF: {hillshade_tif}")

    alpha_ds = gdal.Open(alpha_vrt)
    if alpha_ds is None:
        raise RuntimeError(f"Could not open alpha VRT: {alpha_vrt}")

    xsize = hillshade_ds.RasterXSize
    ysize = hillshade_ds.RasterYSize
    geotransform = ",".join(str(value) for value in hillshade_ds.GetGeoTransform())
    projection = escape(hillshade_ds.GetProjection())
    hillshade_filename = escape(os.path.abspath(hillshade_tif))
    alpha_filename = escape(os.path.abspath(alpha_vrt))
    hillshade_ds = None
    alpha_ds = None

    with open(output_vrt, "w") as f:
        f.write(
            f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <ColorInterp>Red</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{hillshade_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="2">
    <ColorInterp>Green</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{hillshade_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Byte" band="3">
    <ColorInterp>Blue</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{hillshade_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
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

    if bbox is None:
        warp_options = gdal.WarpOptions(
            format="VRT",
            dstSRS="EPSG:3857",
            outputBounds=WEB_MERCATOR_WORLD_BOUNDS,
            resampleAlg=resample_alg,
        )
    else:
        min_lon, min_lat, max_lon, max_lat = bbox
        x_res, y_res = resolution_for_utm_10m_3857(min_lon, min_lat, max_lon, max_lat)
        warp_options = gdal.WarpOptions(
            format="VRT",
            dstSRS="EPSG:3857",
            outputBounds=(min_lon, min_lat, max_lon, max_lat),
            outputBoundsSRS="EPSG:4326",
            xRes=x_res,
            yRes=y_res,
            resampleAlg=resample_alg,
        )
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
            "Without --bbox, export the full masked source raster in EPSG:3857."
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
            "When omitted, exports the full masked source raster in EPSG:3857."
        ),
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
    )
    print(artifacts.output_tif)


if __name__ == "__main__":
    main()
