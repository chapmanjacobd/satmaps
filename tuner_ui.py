import io
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
LAND_SAMPLE_OFF_X = 2000
LAND_SAMPLE_OFF_Y = 2000
LAND_SAMPLE_MIN = 0.0
LAND_SAMPLE_MAX = 9000.0
LAND_DEFAULT_GAMMA = 2.6
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
        id="nairobi",
        name="Nairobi",
        sample_label="Nairobi (HLS L30 37MBU)",
        tile_prefix="37MBU_0_0",
    ),
)
DEFAULT_LAND_LOCATION_ID = "barcelona"
LAND_LOCATIONS_BY_ID = {location.id: location for location in LAND_LOCATIONS}


def get_land_location(location_id: str | None) -> LandLocation:
    return LAND_LOCATIONS_BY_ID.get(location_id or DEFAULT_LAND_LOCATION_ID, LAND_LOCATIONS_BY_ID[DEFAULT_LAND_LOCATION_ID])


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


LAND_SAMPLE_SOURCES = {
    location.id: find_land_samples(location) for location in LAND_LOCATIONS
}


def load_land_sample(paths: dict[str, str]) -> FloatArray:
    """Load a land sample RGB crop and normalize it to the satmaps baseline."""
    data: list[FloatArray] = []
    for band in ("red", "green", "blue"):
        ds = gdal.Open(paths[band])
        if ds is None:
            raise RuntimeError(f"Could not open sample band: {paths[band]}")
        arr = cast(
            FloatArray,
            ds.GetRasterBand(1)
            .ReadAsArray(
                LAND_SAMPLE_OFF_X,
                LAND_SAMPLE_OFF_Y,
                LAND_SAMPLE_CROP_SIZE,
                LAND_SAMPLE_CROP_SIZE,
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


RAW_LAND_SAMPLES = {
    location_id: tuple(load_land_sample(sample.paths) for sample in samples)
    for location_id, samples in LAND_SAMPLE_SOURCES.items()
}


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
    params = {
        "exp": tiler.DEFAULT_EXPOSURE,
        "sb": tiler.SOFT_KNEE_SHADOW_BREAK,
        "hb": tiler.SOFT_KNEE_HIGHLIGHT_BREAK,
        "ss": tiler.SOFT_KNEE_SHADOW_SLOPE,
        "ms": tiler.SOFT_KNEE_MID_SLOPE,
        "hs": tiler.SOFT_KNEE_HIGHLIGHT_SLOPE,
    }
    if mode == "ocean":
        params.update(
            {
                "gamma": ocean.OCEAN_DEFAULT_GAMMA,
                "sat": ocean.OCEAN_DEFAULT_SATURATION,
                "dmin": -11000.0,
                "dmax": 0.0,
            }
        )
        params.update(
            build_grade_defaults(ocean.OCEAN_DEFAULT_BLACK_BREAK, ocean.OCEAN_DEFAULT_BLACK_SLOPE)
        )
        return params

    params.update(
        {
            "gamma": LAND_DEFAULT_GAMMA,
            "sat": LAND_DEFAULT_SATURATION,
            "blend": 0.0,
        }
    )
    params.update(build_grade_defaults(LAND_DEFAULT_GRADE_BREAK, LAND_DEFAULT_GRADE_LOW_SLOPE))
    return params


def parse_request_params(mode: str) -> dict[str, float]:
    defaults = get_mode_defaults(mode)
    params = {key: float(request.args.get(key, default)) for key, default in defaults.items()}
    if "blend" in params:
        params["blend"] = float(np.clip(params["blend"], 0.0, 1.0))
    return params


def get_land_source(location_id: str, blend: float) -> FloatArray | None:
    land_samples = RAW_LAND_SAMPLES[location_id]
    if not land_samples:
        return None
    if len(land_samples) == 1:
        return land_samples[0]
    mix = float(np.clip(blend, 0.0, 1.0))
    return cast(
        FloatArray,
        np.clip(
            (land_samples[0] * (1.0 - mix)) + (land_samples[1] * mix),
            0.0,
            1.0,
        ),
    )


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
    params = parse_request_params(mode)
    source = get_land_source(land_location.id, params.get("blend", 0.0)) if mode == "land" else RAW_GEBCO
    hist = get_histogram_data(source, mode, params.get("dmin", -11000.0), params.get("dmax", 0.0))
    return render_template(
        "index.html",
        mode=mode,
        params=params,
        defaults=get_mode_defaults(mode),
        raw_hist=hist,
        land_locations=LAND_LOCATIONS,
        selected_land_location=land_location,
        land_dates=[sample.date_label for sample in land_samples],
        has_land_blend=len(RAW_LAND_SAMPLES[land_location.id]) > 1,
    )


@app.route("/histogram")
def histogram() -> ResponseReturnValue:
    mode = request.args.get("mode", "land")
    if mode not in {"land", "ocean"}:
        mode = "land"

    land_location = get_land_location(request.args.get("loc"))
    params = parse_request_params(mode)
    source = get_land_source(land_location.id, params.get("blend", 0.0)) if mode == "land" else RAW_GEBCO
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
    p = parse_request_params(mode)
    tm_on = request.args.get("tm", "1") == "1"
    fg_on = request.args.get("fg", "1") == "1"

    if mode == "land":
        source_rgb = get_land_source(land_location.id, p.get("blend", 0.0))
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
