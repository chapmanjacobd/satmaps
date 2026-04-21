import io
import os

import numpy as np
from flask import Flask, render_template, request, send_file
from osgeo import gdal
from PIL import Image

import ocean_background
import tiler

app = Flask(__name__)

# GEBCO discovery for Ocean Mode
GEBCO_ZIP = "gebco_2025_sub_ice_topo_geotiff.zip"


def find_sample_tile():
    cache_root = ".cache/2025-07-01"
    if not os.path.exists(cache_root):
        return None

    # 31TDF sub-tile covering downtown
    prefix = "31TDF_0_0"
    paths = {
        "red": os.path.join(cache_root, f"{prefix}_B04.tif"),
        "green": os.path.join(cache_root, f"{prefix}_B03.tif"),
        "blue": os.path.join(cache_root, f"{prefix}_B02.tif"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None


SAMPLE_PATHS = find_sample_tile()


def load_and_average_sample(paths):
    """Load sample RGB bands and return as float32 normalized numpy array (cropped for RAM)."""
    data = []
    # Crop to a 1024x1024 area in Barcelona Downtown (approx center of 31TDF_0_0)
    CROP_SIZE = 1024
    OFF_X, OFF_Y = 2000, 2000

    for band in ["red", "green", "blue"]:
        ds = gdal.Open(paths[band])
        # Use ReadAsArray with offsets to load only a small portion
        arr = (
            ds.GetRasterBand(1)
            .ReadAsArray(OFF_X, OFF_Y, CROP_SIZE, CROP_SIZE)
            .astype(np.float32)
        )
        # Handle nodata -32768
        arr[arr == -32768] = np.nan
        data.append(arr)

    rgb = np.stack(data)  # (3, 1024, 1024)
    # Standardize normalization for the tuner baseline
    v_min = 0.0
    v_max = 9000.0  # Match satmaps.py universal baseline
    rgb = np.clip((rgb - v_min) / (v_max - v_min), 0, 1)
    rgb[np.isnan(rgb)] = 0
    return rgb


RAW_RGB = load_and_average_sample(SAMPLE_PATHS) if SAMPLE_PATHS else None


def load_gebco_sample():
    """Load a sample of GEBCO data for ocean mode (Hawaii Area)."""
    if not os.path.exists(GEBCO_ZIP):
        return None

    # Hawaii is in n90.0_s0.0_w-180.0_e-90.0.tif
    gebco_vsi = f"/vsizip/{GEBCO_ZIP}/gebco_2025_sub_ice_n90.0_s0.0_w-180.0_e-90.0.tif"
    ds = gdal.Open(gebco_vsi)

    # Hawaii Center approx 19.5N, -159.5W
    # Tile origin: -180, 90. Resolution: 1/240 degree.
    # x_off = (-159.5 - (-180)) * 240 = 20.5 * 240 = 4920
    # y_off = (90 - 19.5) * 240 = 70.5 * 240 = 16920
    CROP_SIZE = 1024
    data = (
        ds.GetRasterBand(1)
        .ReadAsArray(4408, 16408, CROP_SIZE, CROP_SIZE)
        .astype(np.float32)
    )
    return data


RAW_GEBCO = load_gebco_sample()


def get_histogram_data(arr, mode="land", depth_min=-11000, depth_max=0):
    if arr is None:
        return []
    if mode == "land":
        # Use luma for histogram
        luma = 0.2126 * arr[0] + 0.7152 * arr[1] + 0.0722 * arr[2]
    else:
        # Normalize GEBCO band to 0-1 based on current depth range for histogram
        luma = np.clip((arr - depth_min) / (depth_max - depth_min), 0.0, 1.0)

    hist, _ = np.histogram(luma, bins=128, range=(0, 1))
    # Normalize to 0-1
    if hist.max() > 0:
        hist = hist / hist.max()
    return hist.tolist()


@app.route("/")
def index():
    mode = request.args.get("mode", "land")
    depth_min = float(request.args.get("dmin", -11000))
    depth_max = float(request.args.get("dmax", 0))

    hist = get_histogram_data(
        RAW_RGB if mode == "land" else RAW_GEBCO, mode, depth_min, depth_max
    )
    return render_template(
        "index.html",
        mode=mode,
        params={
            "exposure": tiler.DEFAULT_EXPOSURE,
            "shadow_break": tiler.SOFT_KNEE_SHADOW_BREAK,
            "highlight_break": tiler.SOFT_KNEE_HIGHLIGHT_BREAK,
            "shadow_slope": tiler.SOFT_KNEE_SHADOW_SLOPE,
            "mid_slope": tiler.SOFT_KNEE_MID_SLOPE,
            "highlight_slope": tiler.SOFT_KNEE_HIGHLIGHT_SLOPE,
            "gamma": tiler.DEFAULT_GAMMA,
            "saturation": tiler.PREVIEW_SATURATION,
            "darken_break": tiler.PREVIEW_DARKEN_BREAK,
            "low_slope": tiler.PREVIEW_DARKEN_LOW_SLOPE,
            "depth_min": depth_min,
            "depth_max": depth_max,
        },
        raw_hist=hist,
    )


@app.route("/render")
def render():
    mode = request.args.get("mode", "land")

    p = {
        "exp": float(request.args.get("exp", tiler.DEFAULT_EXPOSURE)),
        "sb": float(request.args.get("sb", tiler.SOFT_KNEE_SHADOW_BREAK)),
        "hb": float(request.args.get("hb", tiler.SOFT_KNEE_HIGHLIGHT_BREAK)),
        "ss": float(request.args.get("ss", tiler.SOFT_KNEE_SHADOW_SLOPE)),
        "ms": float(request.args.get("ms", tiler.SOFT_KNEE_MID_SLOPE)),
        "hs": float(request.args.get("hs", tiler.SOFT_KNEE_HIGHLIGHT_SLOPE)),
        "gamma": float(request.args.get("gamma", tiler.DEFAULT_GAMMA)),
        "sat": float(request.args.get("sat", tiler.PREVIEW_SATURATION)),
        "db": float(request.args.get("db", tiler.PREVIEW_DARKEN_BREAK)),
        "ls": float(request.args.get("ls", tiler.PREVIEW_DARKEN_LOW_SLOPE)),
        "dmin": float(request.args.get("dmin", -11000)),
        "dmax": float(request.args.get("dmax", 0)),
    }

    tm_on = request.args.get("tm", "1") == "1"
    fg_on = request.args.get("fg", "1") == "1"

    if mode == "land":
        if RAW_RGB is None:
            return "No sample data", 404
        # 1. Tone Mapping
        if tm_on:
            toned = tiler.apply_soft_knee_numpy(
                RAW_RGB, p["sb"], p["hb"], p["ss"], p["ms"], p["hs"], p["exp"]
            )
        else:
            toned = np.clip(RAW_RGB * p["exp"], 0.0, 1.0)

        # 2. Grading
        if fg_on:
            corrected = tiler.apply_preview_correction_numpy(
                toned, p["sat"], p["db"], p["ls"], p["gamma"]
            )
        else:
            corrected = toned
    else:
        # Ocean Mode
        if RAW_GEBCO is None:
            return "No GEBCO zip found", 404

        corrected = ocean_background.colorize_ocean_depths(
            RAW_GEBCO,
            ocean_background.OceanStyleOptions(
                tonemap=tm_on,
                grade=fg_on,
                exposure=p["exp"],
                shadow_break=p["sb"],
                highlight_break=p["hb"],
                shadow_slope=p["ss"],
                mid_slope=p["ms"],
                highlight_slope=p["hs"],
                gamma=p["gamma"],
                saturation=p["sat"],
                black_break=p["db"],
                black_slope=p["ls"],
                depth_min=p["dmin"],
                depth_max=p["dmax"],
            ),
        )

    # 3. Convert to Byte and JPEG
    byte_arr = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(np.transpose(byte_arr, (1, 2, 0)))

    img_io = io.BytesIO()
    img.save(img_io, "JPEG", quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
