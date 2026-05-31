import io
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from flask.typing import ResponseReturnValue
from numpy.typing import NDArray
from osgeo import gdal
from PIL import Image

import ocean
import tiler

app = Flask(__name__)

# GEBCO discovery for Ocean Mode
GEBCO_ZIP = "gebco_2025_sub_ice_topo_geotiff.zip"
LAND_CACHE_ROOT = Path(".cache")
LAND_SAMPLE_LIMIT = 2
LAND_SAMPLE_CROP_SIZE = 1024
LAND_SAMPLE_DEFAULT_OFF_X = 2000
LAND_SAMPLE_DEFAULT_OFF_Y = 2000
LAND_SAMPLE_MIN = 0.0
LAND_SAMPLE_MAX = 9000.0
LAND_DEFAULT_GAMMA = 2.6
LAND_DEFAULT_SHOULDER = 1.0
LAND_DEFAULT_SATURATION = 0.9
LAND_DEFAULT_GRADE_BREAK = 0.15
LAND_DEFAULT_GRADE_LOW_SLOPE = 0.2

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class LandSample:
    date_label: str
    paths: dict[str, str]


@dataclass(frozen=True)
class LandLocation:
    id: str
    name: str
    sample_label: str
    tile_prefix: str


@dataclass(frozen=True)
class LandView:
    pan_x: float
    pan_y: float
    xoff: int
    yoff: int
    crop_width: int
    crop_height: int
    full_width: int
    full_height: int


@dataclass(frozen=True)
class LandBlendMode:
    id: str
    label: str


LAND_LOCATIONS = (
    LandLocation(
        id="barcelona",
        name="Barcelona",
        sample_label="Barcelona Downtown (HLS L30 31TDF)",
        tile_prefix="31TDF_0_0",
    ),
    LandLocation(
        id="anchorage",
        name="Anchorage",
        sample_label="Anchorage (HLS L30 06VUN)",
        tile_prefix="06VUN_0_0",
    ),
    LandLocation(
        id="shanghai",
        name="Shanghai",
        sample_label="Shanghai (HLS L30 51RUQ)",
        tile_prefix="51RUQ_0_0",
    ),
    LandLocation(
        id="singapore",
        name="Singapore",
        sample_label="Singapore (HLS L30 48NUG)",
        tile_prefix="48NUG_0_0",
    ),
    LandLocation(
        id="sao-paulo",
        name="Sao Paulo",
        sample_label="Sao Paulo (HLS L30 23KLP)",
        tile_prefix="23KLP_0_0",
    ),
    LandLocation(
        id="cape-town",
        name="Cape Town",
        sample_label="Cape Town (HLS L30 34HBH)",
        tile_prefix="34HBH_0_0",
    ),
    LandLocation(
        id="cairo",
        name="Cairo",
        sample_label="Cairo (HLS L30 36RUU)",
        tile_prefix="36RUU_0_0",
    ),
    LandLocation(
        id="mexico-city",
        name="Mexico City",
        sample_label="Mexico City (HLS L30 14QMG)",
        tile_prefix="14QMG_0_0",
    ),
    LandLocation(
        id="sydney",
        name="Sydney",
        sample_label="Sydney (HLS L30 56HLH)",
        tile_prefix="56HLH_0_0",
    ),
    LandLocation(
        id="santiago",
        name="Santiago, Chile",
        sample_label="Santiago / Valle Nevado (HLS L30 19HCD)",
        tile_prefix="19HCD_0_0",
    ),
    LandLocation(
        id="bariloche",
        name="Bariloche, Argentina",
        sample_label="Bariloche / Cerro Catedral (HLS L30 19GBQ)",
        tile_prefix="19GBQ_0_0",
    ),
    LandLocation(
        id="queenstown-wanaka",
        name="Queenstown & Wanaka, New Zealand",
        sample_label="Queenstown / Wanaka (HLS L30 59GLL)",
        tile_prefix="59GLL_0_0",
    ),
    LandLocation(
        id="snowy-mountains",
        name="Snowy Mountains, Australia",
        sample_label="Snowy Mountains / Perisher (HLS L30 55HFV)",
        tile_prefix="55HFV_0_0",
    ),
    LandLocation(
        id="lesotho",
        name="Lesotho",
        sample_label="Lesotho / Afriski (HLS L30 35JPJ)",
        tile_prefix="35JPJ_0_0",
    ),
    LandLocation(
        id="nairobi",
        name="Nairobi",
        sample_label="Nairobi (HLS L30 37MBU)",
        tile_prefix="37MBU_0_0",
    ),
)
DEFAULT_LAND_LOCATION_ID = "barcelona"
LAND_LOCATIONS_BY_ID = {location.id: location for location in LAND_LOCATIONS}
LAND_BLEND_MODES = (
    LandBlendMode(id="crossfade", label="Crossfade"),
    LandBlendMode(id="difference", label="Difference"),
    LandBlendMode(id="lighten", label="Lighten"),
    LandBlendMode(id="darken", label="Darken"),
    LandBlendMode(id="swipe", label="Swipe"),
)
DEFAULT_LAND_BLEND_MODE = "crossfade"
LAND_BLEND_MODES_BY_ID = {blend_mode.id: blend_mode for blend_mode in LAND_BLEND_MODES}


def get_land_location(location_id: str | None) -> LandLocation:
    return LAND_LOCATIONS_BY_ID.get(location_id or DEFAULT_LAND_LOCATION_ID, LAND_LOCATIONS_BY_ID[DEFAULT_LAND_LOCATION_ID])


def get_land_blend_mode(blend_mode_id: str | None) -> LandBlendMode:
    return LAND_BLEND_MODES_BY_ID.get(
        blend_mode_id or DEFAULT_LAND_BLEND_MODE,
        LAND_BLEND_MODES_BY_ID[DEFAULT_LAND_BLEND_MODE],
    )


def find_land_samples(
    location: LandLocation,
    cache_root: Path = LAND_CACHE_ROOT,
) -> tuple[LandSample, ...]:
    if not cache_root.exists():
        return ()

    samples: list[LandSample] = []
    for date_dir in sorted((path for path in cache_root.iterdir() if path.is_dir()), reverse=True):
        paths = {
            "red": str(date_dir / f"{location.tile_prefix}_B04.tif"),
            "green": str(date_dir / f"{location.tile_prefix}_B03.tif"),
            "blue": str(date_dir / f"{location.tile_prefix}_B02.tif"),
        }
        if all(Path(path).exists() for path in paths.values()):
            samples.append(LandSample(date_label=date_dir.name, paths=paths))
        if len(samples) >= LAND_SAMPLE_LIMIT:
            break
    return tuple(samples)


def clamp_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def build_land_view(
    full_width: int,
    full_height: int,
    pan_x: float | None = None,
    pan_y: float | None = None,
) -> LandView:
    crop_width = min(LAND_SAMPLE_CROP_SIZE, full_width)
    crop_height = min(LAND_SAMPLE_CROP_SIZE, full_height)
    max_xoff = max(0, full_width - crop_width)
    max_yoff = max(0, full_height - crop_height)
    default_pan_x = 0.0 if max_xoff == 0 else min(LAND_SAMPLE_DEFAULT_OFF_X, max_xoff) / max_xoff
    default_pan_y = 0.0 if max_yoff == 0 else min(LAND_SAMPLE_DEFAULT_OFF_Y, max_yoff) / max_yoff
    resolved_pan_x = clamp_unit(default_pan_x if pan_x is None else pan_x)
    resolved_pan_y = clamp_unit(default_pan_y if pan_y is None else pan_y)
    return LandView(
        pan_x=resolved_pan_x,
        pan_y=resolved_pan_y,
        xoff=int(round(max_xoff * resolved_pan_x)),
        yoff=int(round(max_yoff * resolved_pan_y)),
        crop_width=crop_width,
        crop_height=crop_height,
        full_width=full_width,
        full_height=full_height,
    )


LAND_SAMPLE_SOURCES = {
    location.id: find_land_samples(location) for location in LAND_LOCATIONS
}


@lru_cache(maxsize=64)
def get_raster_size(path: str) -> tuple[int, int]:
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open sample band: {path}")
    size = (ds.RasterXSize, ds.RasterYSize)
    ds = None
    return size


def get_land_view(location_id: str, pan_x: float | None = None, pan_y: float | None = None) -> LandView:
    land_samples = LAND_SAMPLE_SOURCES[location_id]
    if not land_samples:
        return build_land_view(LAND_SAMPLE_CROP_SIZE, LAND_SAMPLE_CROP_SIZE, pan_x, pan_y)
    widths: list[int] = []
    heights: list[int] = []
    for sample in land_samples:
        width, height = get_raster_size(sample.paths["red"])
        widths.append(width)
        heights.append(height)
    return build_land_view(min(widths), min(heights), pan_x, pan_y)


@lru_cache(maxsize=8)
def load_land_sample_window(
    red_path: str,
    green_path: str,
    blue_path: str,
    xoff: int,
    yoff: int,
    crop_width: int,
    crop_height: int,
) -> FloatArray:
    """Load a land sample RGB crop and normalize it to the satmaps baseline."""
    data: list[FloatArray] = []
    for band_path in (red_path, green_path, blue_path):
        ds = gdal.Open(band_path)
        if ds is None:
            raise RuntimeError(f"Could not open sample band: {band_path}")
        arr = cast(
            FloatArray,
            ds.GetRasterBand(1)
            .ReadAsArray(
                xoff,
                yoff,
                crop_width,
                crop_height,
            )
            .astype(np.float32),
        )
        ds = None
        arr[arr == -32768] = np.nan
        data.append(arr)

    rgb = cast(FloatArray, np.stack(data))
    normalized = cast(
        FloatArray,
        np.clip((rgb - LAND_SAMPLE_MIN) / (LAND_SAMPLE_MAX - LAND_SAMPLE_MIN), 0.0, 1.0),
    )
    return cast(FloatArray, np.nan_to_num(normalized, nan=0.0))


def load_gebco_sample() -> FloatArray | None:
    """Load a sample of GEBCO data for ocean mode (Hawaii area)."""
    if not Path(GEBCO_ZIP).exists():
        return None

    gebco_vsi = f"/vsizip/{GEBCO_ZIP}/gebco_2025_sub_ice_n90.0_s0.0_w-180.0_e-90.0.tif"
    ds = gdal.Open(gebco_vsi)
    if ds is None:
        raise RuntimeError(f"Could not open GEBCO sample: {gebco_vsi}")

    data = cast(
        FloatArray,
        ds.GetRasterBand(1)
        .ReadAsArray(4408, 16408, LAND_SAMPLE_CROP_SIZE, LAND_SAMPLE_CROP_SIZE)
        .astype(np.float32),
    )
    ds = None
    return data


RAW_GEBCO = load_gebco_sample()


def build_grade_defaults(low_break: float, low_slope: float) -> dict[str, float]:
    highlight_break = low_break
    mid_slope = tiler.PREVIEW_DARKEN_MID_SLOPE
    return {
        "db": low_break,
        "ghb": highlight_break,
        "ls": low_slope,
        "gms": mid_slope,
        "ghs": tiler.derive_piecewise_high_slope(low_break, highlight_break, low_slope, mid_slope),
    }


def get_mode_defaults(mode: str) -> dict[str, float]:
    if mode == "ocean":
        params = {
            "exp": ocean.OCEAN_DEFAULT_EXPOSURE,
            "sb": ocean.OCEAN_DEFAULT_SHADOW_BREAK,
            "hb": ocean.OCEAN_DEFAULT_HIGHLIGHT_BREAK,
            "ss": ocean.OCEAN_DEFAULT_SHADOW_SLOPE,
            "ms": ocean.OCEAN_DEFAULT_MID_SLOPE,
            "hs": ocean.OCEAN_DEFAULT_HIGHLIGHT_SLOPE,
            "gamma": ocean.OCEAN_DEFAULT_GAMMA,
            "shoulder": ocean.OCEAN_DEFAULT_SHOULDER,
            "sat": ocean.OCEAN_DEFAULT_SATURATION,
            "dmin": -11000.0,
            "dmax": 0.0,
        }
        params.update(
            build_grade_defaults(ocean.OCEAN_DEFAULT_BLACK_BREAK, ocean.OCEAN_DEFAULT_BLACK_SLOPE)
        )
        return params

    params = {
        "exp": tiler.DEFAULT_EXPOSURE,
        "sb": tiler.SOFT_KNEE_SHADOW_BREAK,
        "hb": tiler.SOFT_KNEE_HIGHLIGHT_BREAK,
        "ss": tiler.SOFT_KNEE_SHADOW_SLOPE,
        "ms": tiler.SOFT_KNEE_MID_SLOPE,
        "hs": tiler.SOFT_KNEE_HIGHLIGHT_SLOPE,
        "gamma": LAND_DEFAULT_GAMMA,
        "shoulder": LAND_DEFAULT_SHOULDER,
        "sat": LAND_DEFAULT_SATURATION,
        "blend": 0.0,
    }
    params.update(build_grade_defaults(LAND_DEFAULT_GRADE_BREAK, LAND_DEFAULT_GRADE_LOW_SLOPE))
    return params


def parse_request_params(mode: str) -> dict[str, float]:
    defaults = get_mode_defaults(mode)
    params = {key: float(request.args.get(key, default)) for key, default in defaults.items()}
    if "blend" in params:
        params["blend"] = float(np.clip(params["blend"], 0.0, 1.0))
    return params


def get_land_pan_arg(name: str) -> float | None:
    raw_value = request.args.get(name)
    if raw_value is None:
        return None
    return clamp_unit(float(raw_value))


def get_cropped_land_samples(location_id: str, land_view: LandView) -> tuple[FloatArray, ...]:
    return tuple(
        load_land_sample_window(
            sample.paths["red"],
            sample.paths["green"],
            sample.paths["blue"],
            land_view.xoff,
            land_view.yoff,
            land_view.crop_width,
            land_view.crop_height,
        )
        for sample in LAND_SAMPLE_SOURCES[location_id]
    )


def blend_land_samples(
    cropped_samples: tuple[FloatArray, ...],
    blend: float,
    blend_mode_id: str,
) -> FloatArray | None:
    if not cropped_samples:
        return None
    if len(cropped_samples) == 1:
        return cropped_samples[0]

    primary = cropped_samples[0]
    secondary = cropped_samples[1]
    blend_mode = get_land_blend_mode(blend_mode_id).id
    mix = float(np.clip(blend, 0.0, 1.0))

    if blend_mode == "difference":
        return cast(FloatArray, np.abs(primary - secondary))
    if blend_mode == "lighten":
        return cast(FloatArray, np.maximum(primary, secondary))
    if blend_mode == "darken":
        return cast(FloatArray, np.minimum(primary, secondary))
    if blend_mode == "swipe":
        split = int(round(primary.shape[2] * (1.0 - mix)))
        if split <= 0:
            return secondary
        if split >= primary.shape[2]:
            return primary
        return cast(
            FloatArray,
            np.concatenate((primary[:, :, :split], secondary[:, :, split:]), axis=2),
        )

    return cast(
        FloatArray,
        np.clip(
            (primary * (1.0 - mix)) + (secondary * mix),
            0.0,
            1.0,
        ),
    )


def get_land_source(
    location_id: str,
    blend: float,
    blend_mode_id: str,
    land_view: LandView,
) -> FloatArray | None:
    cropped_samples = get_cropped_land_samples(location_id, land_view)
    if len(cropped_samples) == 1:
        return cropped_samples[0]
    return blend_land_samples(cropped_samples, blend, blend_mode_id)


def get_histogram_data(
    arr: FloatArray | None,
    mode: str,
    depth_min: float = -11000.0,
    depth_max: float = 0.0,
) -> list[float]:
    if arr is None:
        return []
    if mode == "land":
        luma = 0.2126 * arr[0] + 0.7152 * arr[1] + 0.0722 * arr[2]
    else:
        luma = cast(
            FloatArray,
            np.clip((arr - depth_min) / (depth_max - depth_min), 0.0, 1.0),
        )

    hist, _ = np.histogram(luma, bins=128, range=(0, 1))
    if hist.max() > 0:
        hist = hist / hist.max()
    return [float(value) for value in hist.tolist()]


@app.route("/")
def index() -> ResponseReturnValue:
    mode = request.args.get("mode", "land")
    if mode not in {"land", "ocean"}:
        mode = "land"

    land_location = get_land_location(request.args.get("loc"))
    land_samples = LAND_SAMPLE_SOURCES[land_location.id]
    land_blend_mode = get_land_blend_mode(request.args.get("blend_mode"))
    land_view = get_land_view(
        land_location.id,
        get_land_pan_arg("panx"),
        get_land_pan_arg("pany"),
    )
    params = parse_request_params(mode)
    source = (
        get_land_source(land_location.id, params.get("blend", 0.0), land_blend_mode.id, land_view)
        if mode == "land"
        else RAW_GEBCO
    )
    hist = get_histogram_data(source, mode, params.get("dmin", -11000.0), params.get("dmax", 0.0))
    return render_template(
        "index.html",
        mode=mode,
        params=params,
        defaults=get_mode_defaults(mode),
        raw_hist=hist,
        land_locations=LAND_LOCATIONS,
        land_blend_modes=LAND_BLEND_MODES,
        selected_land_blend_mode=land_blend_mode,
        default_land_blend_mode=DEFAULT_LAND_BLEND_MODE,
        selected_land_location=land_location,
        land_view=land_view,
        land_dates=[sample.date_label for sample in land_samples],
        has_land_blend=len(land_samples) > 1,
    )


@app.route("/histogram")
def histogram() -> ResponseReturnValue:
    mode = request.args.get("mode", "land")
    if mode not in {"land", "ocean"}:
        mode = "land"

    land_location = get_land_location(request.args.get("loc"))
    land_blend_mode = get_land_blend_mode(request.args.get("blend_mode"))
    land_view = get_land_view(
        land_location.id,
        get_land_pan_arg("panx"),
        get_land_pan_arg("pany"),
    )
    params = parse_request_params(mode)
    source = (
        get_land_source(land_location.id, params.get("blend", 0.0), land_blend_mode.id, land_view)
        if mode == "land"
        else RAW_GEBCO
    )
    return jsonify(
        {
            "hist": get_histogram_data(
                source,
                mode,
                params.get("dmin", -11000.0),
                params.get("dmax", 0.0),
            )
        }
    )


@app.route("/render")
def render() -> ResponseReturnValue:
    mode = request.args.get("mode", "land")
    if mode not in {"land", "ocean"}:
        mode = "land"

    land_location = get_land_location(request.args.get("loc"))
    land_blend_mode = get_land_blend_mode(request.args.get("blend_mode"))
    land_view = get_land_view(
        land_location.id,
        get_land_pan_arg("panx"),
        get_land_pan_arg("pany"),
    )
    p = parse_request_params(mode)
    tm_on = request.args.get("tm", "0") == "1"
    fg_on = request.args.get("fg", "1") == "1"

    if mode == "land":
        source_rgb = get_land_source(land_location.id, p.get("blend", 0.0), land_blend_mode.id, land_view)
        if source_rgb is None:
            return "No sample data", 404
        if tm_on:
            toned = tiler.apply_soft_knee_numpy(
                source_rgb,
                p["sb"],
                p["hb"],
                p["ss"],
                p["ms"],
                p["hs"],
                p["exp"],
            )
        else:
            toned = np.clip(source_rgb * p["exp"], 0.0, 1.0)

        if fg_on:
            corrected = tiler.apply_preview_correction_numpy(
                toned,
                saturation=p["sat"],
                darken_break=p["db"],
                low_slope=p["ls"],
                gamma=p["gamma"],
                shoulder=p["shoulder"],
                highlight_break=p["ghb"],
                mid_slope=p["gms"],
                high_slope=p["ghs"],
            )
        else:
            corrected = toned
    else:
        if RAW_GEBCO is None:
            return "No GEBCO zip found", 404

        corrected = ocean.colorize_ocean_depths(
            RAW_GEBCO,
            ocean.OceanStyleOptions(
                tonemap=tm_on,
                grade=fg_on,
                exposure=p["exp"],
                shadow_break=p["sb"],
                highlight_break=p["hb"],
                shadow_slope=p["ss"],
                mid_slope=p["ms"],
                highlight_slope=p["hs"],
                gamma=p["gamma"],
                shoulder=p["shoulder"],
                saturation=p["sat"],
                black_break=p["db"],
                black_slope=p["ls"],
                grade_high_break=p["ghb"],
                grade_mid_slope=p["gms"],
                grade_high_slope=p["ghs"],
                depth_min=p["dmin"],
                depth_max=p["dmax"],
            ),
        )

    byte_arr = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(np.transpose(byte_arr, (1, 2, 0)))

    img_io = io.BytesIO()
    img.save(img_io, "JPEG", quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


def main() -> None:
    app.run(debug=True, port=5001)


if __name__ == "__main__":
    main()
