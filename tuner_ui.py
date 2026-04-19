import os
import numpy as np
from flask import Flask, render_template, request, send_file
from osgeo import gdal
import io
from PIL import Image
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
        "blue": os.path.join(cache_root, f"{prefix}_B02.tif")
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
        arr = ds.GetRasterBand(1).ReadAsArray(OFF_X, OFF_Y, CROP_SIZE, CROP_SIZE).astype(np.float32)
        # Handle nodata -32768
        arr[arr == -32768] = np.nan
        data.append(arr)
    
    rgb = np.stack(data) # (3, 1024, 1024)
    # Standardize normalization for the tuner baseline
    v_min = 0.0
    v_max = 9000.0 # Match satmaps.py universal baseline
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
    data = ds.GetRasterBand(1).ReadAsArray(4408, 16408, CROP_SIZE, CROP_SIZE).astype(np.float32)
    return data

RAW_GEBCO = load_gebco_sample()

def get_histogram_data(arr, mode='land', depth_min=-11000, depth_max=0):
    if arr is None:
        return []
    if mode == 'land':
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

MAKO_RAMP = [
    (0.00, 11, 4, 5), (0.05, 25, 14, 24), (0.10, 38, 23, 43), (0.15, 49, 33, 64),
    (0.20, 56, 42, 84), (0.25, 62, 53, 107), (0.30, 65, 64, 130), (0.35, 62, 79, 148),
    (0.40, 57, 93, 156), (0.45, 54, 108, 160), (0.50, 53, 122, 162), (0.55, 52, 137, 166),
    (0.60, 52, 153, 170), (0.65, 55, 166, 172), (0.70, 63, 181, 173), (0.75, 75, 194, 173),
    (0.80, 101, 208, 173), (0.85, 136, 217, 177), (0.90, 171, 226, 190), (0.95, 198, 235, 209),
    (1.00, 222, 245, 229)
]

@app.route('/')
def index():
    mode = request.args.get('mode', 'land')
    depth_min = float(request.args.get('dmin', -11000))
    depth_max = float(request.args.get('dmax', 0))
    
    hist = get_histogram_data(RAW_RGB if mode == 'land' else RAW_GEBCO, mode, depth_min, depth_max)
    return render_template('index.html', 
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
                               "depth_max": depth_max
                           },
                           raw_hist=hist)

@app.route('/render')
def render():
    mode = request.args.get('mode', 'land')
    
    p = {
        "exp": float(request.args.get('exp', tiler.DEFAULT_EXPOSURE)),
        "sb": float(request.args.get('sb', tiler.SOFT_KNEE_SHADOW_BREAK)),
        "hb": float(request.args.get('hb', tiler.SOFT_KNEE_HIGHLIGHT_BREAK)),
        "ss": float(request.args.get('ss', tiler.SOFT_KNEE_SHADOW_SLOPE)),
        "ms": float(request.args.get('ms', tiler.SOFT_KNEE_MID_SLOPE)),
        "hs": float(request.args.get('hs', tiler.SOFT_KNEE_HIGHLIGHT_SLOPE)),
        "gamma": float(request.args.get('gamma', tiler.DEFAULT_GAMMA)),
        "sat": float(request.args.get('sat', tiler.PREVIEW_SATURATION)),
        "db": float(request.args.get('db', tiler.PREVIEW_DARKEN_BREAK)),
        "ls": float(request.args.get('ls', tiler.PREVIEW_DARKEN_LOW_SLOPE)),
        "dmin": float(request.args.get('dmin', -11000)),
        "dmax": float(request.args.get('dmax', 0))
    }
    
    tm_on = request.args.get('tm', '1') == '1'
    fg_on = request.args.get('fg', '1') == '1'

    if mode == 'land':
        if RAW_RGB is None: return "No sample data", 404
        # 1. Tone Mapping
        if tm_on:
            toned = tiler.apply_soft_knee_numpy(RAW_RGB, p["sb"], p["hb"], p["ss"], p["ms"], p["hs"], p["exp"])
        else:
            toned = np.clip(RAW_RGB * p["exp"], 0.0, 1.0)
        
        # 2. Grading
        if fg_on:
            corrected = tiler.apply_preview_correction_numpy(toned, p["sat"], p["db"], p["ls"], p["gamma"])
        else:
            corrected = toned
    else:
        # Ocean Mode
        if RAW_GEBCO is None: return "No GEBCO zip found", 404
        
        # Colorize Mako ramp based on depth
        mako_colors = np.array([c[1:] for c in MAKO_RAMP], dtype=np.float32) / 255.0
        mako_arr = mako_colors.T.reshape(3, -1, 1)
        
        if tm_on:
            toned_mako = tiler.apply_soft_knee_numpy(mako_arr, p["sb"], p["hb"], p["ss"], p["ms"], p["hs"], p["exp"])
        else:
            toned_mako = np.clip(mako_arr * p["exp"], 0.0, 1.0)
            
        if fg_on:
            graded_mako = tiler.apply_preview_correction_numpy(toned_mako, p["sat"], p["db"], p["ls"], p["gamma"])
        else:
            graded_mako = toned_mako
            
        mako_lut = graded_mako.reshape(3, -1).T # (21, 3)
        
        # Apply LUT to RAW_GEBCO
        normalized_gebco = np.clip((RAW_GEBCO - p["dmin"]) / (p["dmax"] - p["dmin"]), 0.0, 1.0)
        
        # Interp depth (0-1) to LUT indices
        fracs = np.array([c[0] for c in MAKO_RAMP])
        indices = np.arange(len(MAKO_RAMP))
        
        # Map pixels to continuous index
        pixel_indices = np.interp(normalized_gebco, fracs, indices)
        
        # Interpolate RGB values from the LUT
        corrected = np.zeros((3, 1024, 1024), dtype=np.float32)
        for i in range(3):
            corrected[i] = np.interp(pixel_indices, indices, mako_lut[:, i])

    # 3. Convert to Byte and JPEG
    byte_arr = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(np.transpose(byte_arr, (1, 2, 0)))
    
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
