#!/usr/bin/env python3
import argparse
import os
import subprocess
from dataclasses import dataclass
from shutil import which
from typing import Sequence
from xml.sax.saxutils import escape

from osgeo import gdal, osr

GEBCO_OCEAN_NODATA = -32767.0
HILLSHADE_CREATION_OPTIONS = (
    "BIGTIFF=YES",
    "TILED=YES",
    "COMPRESS=ZSTD",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)


@dataclass(frozen=True)
class OceanBackgroundOutputs:
    ocean_vrt: str
    hillshade_tif: str
    rgba_vrt: str


def create_gebco_ocean_vrt(source_vrt: str, output_vrt: str) -> str:
    """Create a VRT that masks positive GEBCO elevations for ocean rendering."""
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


def resolution_for_utm_10m_3857(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float]:
    """Approximate the EPSG:3857 resolution of a 10m UTM grid at the bbox center."""
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


def create_hillshade_rgba_vrt(
    hillshade_tif: str, alpha_source_vrt: str, output_vrt: str
) -> str:
    """Wrap a grayscale hillshade in an RGBA VRT so it can mosaic with land tiles."""
    hillshade_ds = gdal.Open(hillshade_tif)
    if hillshade_ds is None:
        raise RuntimeError(f"Could not open hillshade TIFF: {hillshade_tif}")

    alpha_ds = gdal.Open(alpha_source_vrt)
    if alpha_ds is None:
        raise RuntimeError(f"Could not open alpha source VRT: {alpha_source_vrt}")

    xsize = hillshade_ds.RasterXSize
    ysize = hillshade_ds.RasterYSize
    geotransform = ",".join(str(value) for value in hillshade_ds.GetGeoTransform())
    projection = escape(hillshade_ds.GetProjection())
    hillshade_filename = escape(os.path.abspath(hillshade_tif))
    alpha_filename = escape(os.path.abspath(alpha_source_vrt))

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
      <SourceBand>mask,1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
        )

    return output_vrt


def build_hillshade_command(
    input_vrt: str,
    output_tif: str,
    z_factor: float,
    creation_options: Sequence[str] = HILLSHADE_CREATION_OPTIONS,
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


def generate_ocean_background(
    gebco_source_vrt: str,
    bbox: tuple[float, float, float, float],
    unique_id: str,
    output_dir: str = ".temp",
    resample_alg: str = "cubicspline",
    hillshade_z: float = 5.0,
) -> OceanBackgroundOutputs:
    """Create a cropped, resampled GEBCO ocean hillshade for bbox rendering."""
    if which("gdaldem") is None:
        raise RuntimeError("gdaldem is required to generate the ocean background")

    os.makedirs(output_dir, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = bbox
    x_res, y_res = resolution_for_utm_10m_3857(min_lon, min_lat, max_lon, max_lat)

    masked_vrt = os.path.join(output_dir, f"gebco_ocean_source_{unique_id}.vrt")
    ocean_vrt = os.path.join(output_dir, f"gebco_ocean_{unique_id}.vrt")
    hillshade_tif = os.path.join(output_dir, f"ocean_{unique_id}.tif")
    rgba_vrt = os.path.join(output_dir, f"ocean_rgba_{unique_id}.vrt")

    create_gebco_ocean_vrt(gebco_source_vrt, masked_vrt)

    warp_options = gdal.WarpOptions(
        format="VRT",
        dstSRS="EPSG:3857",
        outputBounds=(min_lon, min_lat, max_lon, max_lat),
        outputBoundsSRS="EPSG:4326",
        xRes=x_res,
        yRes=y_res,
        resampleAlg=resample_alg,
    )
    gdal.Warp(ocean_vrt, masked_vrt, options=warp_options)

    subprocess.run(
        build_hillshade_command(ocean_vrt, hillshade_tif, hillshade_z),
        check=True,
    )
    create_hillshade_rgba_vrt(hillshade_tif, ocean_vrt, rgba_vrt)

    return OceanBackgroundOutputs(
        ocean_vrt=ocean_vrt,
        hillshade_tif=hillshade_tif,
        rgba_vrt=rgba_vrt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a GEBCO ocean hillshade background in EPSG:3857 for a WGS84 bbox."
    )
    parser.add_argument("gebco_source_vrt", help="Input GEBCO source VRT")
    parser.add_argument("bbox", help="WGS84 bbox as min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--output-dir", default=".temp", help="Directory for temp outputs")
    parser.add_argument("--unique-id", default="manual", help="Suffix for generated files")
    parser.add_argument(
        "--resample-alg",
        choices=["cubicspline", "lanczos", "bilinear", "average", "gauss"],
        default="cubicspline",
        help="Resampling algorithm for the GEBCO upscale step into EPSG:3857",
    )
    parser.add_argument(
        "--hillshade-z",
        type=float,
        default=5.0,
        help="Vertical exaggeration passed to gdaldem hillshade",
    )
    args = parser.parse_args()

    bbox_values = tuple(float(value) for value in args.bbox.split(","))
    if len(bbox_values) != 4:
        raise ValueError("bbox must contain four comma-separated values")
    bbox = (
        bbox_values[0],
        bbox_values[1],
        bbox_values[2],
        bbox_values[3],
    )

    outputs = generate_ocean_background(
        args.gebco_source_vrt,
        bbox,
        unique_id=args.unique_id,
        output_dir=args.output_dir,
        resample_alg=args.resample_alg,
        hillshade_z=args.hillshade_z,
    )
    print(outputs.hillshade_tif)


if __name__ == "__main__":
    main()
