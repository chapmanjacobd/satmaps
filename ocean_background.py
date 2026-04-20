#!/usr/bin/env python3
import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Sequence
from xml.sax.saxutils import escape

from osgeo import gdal, osr

gdal.UseExceptions()

DEFAULT_GEBCO_ZIP = "gebco_2025_sub_ice_topo_geotiff.zip"
DEFAULT_OUTPUT = "ocean.tif"
GEBCO_OCEAN_NODATA = -32767.0
GTIFF_CREATION_OPTIONS = (
    "BIGTIFF=YES",
    "TILED=YES",
    "COMPRESS=ZSTD",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)


@dataclass(frozen=True)
class OceanBackgroundArtifacts:
    source_vrt: str
    masked_vrt: str
    warped_vrt: str
    alpha_vrt: str
    hillshade_tif: str
    rgba_vrt: str
    output_tif: str


def build_gebco_source_vrt(gebco_zip: str, output_vrt: str) -> str:
    """Build a source VRT from the GEBCO zip archive."""
    if not os.path.exists(gebco_zip):
        raise FileNotFoundError(f"GEBCO zip not found: {gebco_zip}")

    gebco_vsi = f"/vsizip/{gebco_zip}"
    files_in_zip = gdal.ReadDir(gebco_vsi) or []
    tif_paths = [f"{gebco_vsi}/{name}" for name in files_in_zip if name.lower().endswith(".tif")]
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


def resolution_for_utm_10m_3857(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float]:
    """Approximate the EPSG:3857 pixel size of a 10m UTM grid at the bbox center."""
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    zone = int((center_lon + 180.0) / 6.0) + 1
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
        raise RuntimeError("Could not derive a positive EPSG:3857 resolution from the bbox center")
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


def generate_ocean_background(
    gebco_zip: str,
    destination: str,
    bbox: tuple[float, float, float, float],
    temp_dir: str = ".temp",
    resample_alg: str = "cubicspline",
    hillshade_z: float = 5.0,
) -> OceanBackgroundArtifacts:
    """Generate a standalone RGBA ocean background GeoTIFF."""
    if which("gdaldem") is None:
        raise RuntimeError("gdaldem is required to generate the ocean background")

    os.makedirs(temp_dir, exist_ok=True)
    stem = Path(destination).stem or "ocean"

    source_vrt = os.path.join(temp_dir, f"{stem}_source.vrt")
    masked_vrt = os.path.join(temp_dir, f"{stem}_masked.vrt")
    warped_vrt = os.path.join(temp_dir, f"{stem}_3857.vrt")
    alpha_vrt = os.path.join(temp_dir, f"{stem}_alpha.vrt")
    hillshade_tif = os.path.join(temp_dir, f"{stem}_hillshade.tif")
    rgba_vrt = os.path.join(temp_dir, f"{stem}_rgba.vrt")

    build_gebco_source_vrt(gebco_zip, source_vrt)
    create_gebco_ocean_vrt(source_vrt, masked_vrt)

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
    create_hillshade_rgba_vrt(hillshade_tif, alpha_vrt, rgba_vrt)
    translate_rgba_vrt(rgba_vrt, destination)

    return OceanBackgroundArtifacts(
        source_vrt=source_vrt,
        masked_vrt=masked_vrt,
        warped_vrt=warped_vrt,
        alpha_vrt=alpha_vrt,
        hillshade_tif=hillshade_tif,
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
        description="Generate a standalone GEBCO ocean hillshade GeoTIFF in EPSG:3857."
    )
    parser.add_argument("gebco_zip", nargs="?", default=DEFAULT_GEBCO_ZIP, help="GEBCO zip archive")
    parser.add_argument("destination", nargs="?", default=DEFAULT_OUTPUT, help="Output GeoTIFF path")
    parser.add_argument("--bbox", required=True, help="WGS84 bbox as min_lon,min_lat,max_lon,max_lat")
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
    args = parser.parse_args()

    artifacts = generate_ocean_background(
        gebco_zip=args.gebco_zip,
        destination=args.destination,
        bbox=parse_bbox(args.bbox),
        temp_dir=args.temp_dir,
        resample_alg=args.resample_alg,
        hillshade_z=args.hillshade_z,
    )
    print(artifacts.output_tif)


if __name__ == "__main__":
    main()
