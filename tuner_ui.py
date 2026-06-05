import argparse
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

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
DEFAULT_SAMPLE_DOWNLOAD_PARALLEL = 12
LAND_SAMPLE_PREFERRED_DATE_ORDER = ("2025-07-01", "2025-01-01")
LAND_SAMPLE_MIN = 0.0
LAND_SAMPLE_MAX = 9000.0
LAND_DEFAULT_EXPOSURE = 2.5
LAND_DEFAULT_GAMMA = 2.2
LAND_DEFAULT_SHOULDER = 0.5
LAND_DEFAULT_SATURATION = 0.9
LAND_DEFAULT_GRADE_LOW_BREAK = 0.08
LAND_DEFAULT_GRADE_HIGHLIGHT_BREAK = 0.9
LAND_DEFAULT_GRADE_LOW_SLOPE = 0.0
LAND_DEFAULT_GRADE_MID_SLOPE = 1.0
LAND_DEFAULT_GRADE_HIGHLIGHT_SLOPE = 1.7

FloatArray = NDArray[np.float32]
Hemisphere = Literal["north", "south"]
GRADE_CONTROL_IDS = ("exp", "gamma", "shoulder", "sat", "db", "ghb", "ls", "gms", "ghs")


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
    hemisphere: Hemisphere


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
        sample_label="Barcelona Downtown (31TDF)",
        tile_prefix="31TDF_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="anchorage",
        name="Anchorage",
        sample_label="Anchorage (06VUN)",
        tile_prefix="06VUN_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="shanghai",
        name="Shanghai",
        sample_label="Shanghai (51RUQ)",
        tile_prefix="51RUQ_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="singapore",
        name="Singapore",
        sample_label="Singapore (48NUG)",
        tile_prefix="48NUG_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="sao-paulo",
        name="Sao Paulo",
        sample_label="Sao Paulo (23KLP)",
        tile_prefix="23KLP_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="cape-town",
        name="Cape Town",
        sample_label="Cape Town (34HBH)",
        tile_prefix="34HBH_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="cairo",
        name="Cairo",
        sample_label="Cairo (36RUU)",
        tile_prefix="36RUU_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="banc-darguin",
        name="Banc d'Arguin",
        sample_label="Banc d'Arguin, Mauritania (28QCH)",
        tile_prefix="28QCH_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="mexico-city",
        name="Mexico City",
        sample_label="Mexico City (14QMG)",
        tile_prefix="14QMG_0_0",
        hemisphere="north",
    ),
    LandLocation(
        id="sydney",
        name="Sydney",
        sample_label="Sydney (56HLH)",
        tile_prefix="56HLH_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="santiago",
        name="Santiago, Chile",
        sample_label="Santiago / Valle Nevado (19HCD)",
        tile_prefix="19HCD_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="bariloche",
        name="Bariloche, Argentina",
        sample_label="Bariloche / Cerro Catedral (19GBQ)",
        tile_prefix="19GBQ_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="queenstown-wanaka",
        name="Queenstown & Wanaka, New Zealand",
        sample_label="Queenstown / Wanaka (59GLL)",
        tile_prefix="59GLL_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="snowy-mountains",
        name="Snowy Mountains, Australia",
        sample_label="Snowy Mountains / Perisher (55HFV)",
        tile_prefix="55HFV_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="lesotho",
        name="Lesotho",
        sample_label="Lesotho / Afriski (35JPJ)",
        tile_prefix="35JPJ_0_0",
        hemisphere="south",
    ),
    LandLocation(
        id="nairobi",
        name="Nairobi",
        sample_label="Nairobi (37MBU)",
        tile_prefix="37MBU_0_0",
        hemisphere="south",
    ),
)
DEFAULT_LAND_LOCATION_ID = "barcelona"
LAND_LOCATIONS_BY_ID = {location.id: location for location in LAND_LOCATIONS}
LAND_BLEND_MODES = (
    LandBlendMode(id="summer", label="Summer"),
    LandBlendMode(id="winter", label="Winter"),
    LandBlendMode(id="crossfade", label="Crossfade"),
    LandBlendMode(id="difference", label="Difference"),
    LandBlendMode(id="lighten", label="Lighten"),
    LandBlendMode(id="darken", label="Darken"),
    LandBlendMode(id="swipe", label="Swipe"),
)
DEFAULT_LAND_BLEND_MODE = "summer"
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
    return sort_land_samples_for_tuner(tuple(samples))


def sort_land_samples_for_tuner(samples: tuple[LandSample, ...]) -> tuple[LandSample, ...]:
    preferred_dates = {date_label: index for index, date_label in enumerate(LAND_SAMPLE_PREFERRED_DATE_ORDER)}
    prioritized = sorted(
        (sample for sample in samples if sample.date_label in preferred_dates),
        key=lambda sample: preferred_dates[sample.date_label],
    )
    extras = sorted(
        (sample for sample in samples if sample.date_label not in preferred_dates),
        key=lambda sample: sample.date_label,
        reverse=True,
    )
    return tuple((prioritized + extras)[:LAND_SAMPLE_LIMIT])


def get_land_samples(
    location_id: str,
    cache_root: Path = LAND_CACHE_ROOT,
) -> tuple[LandSample, ...]:
    return find_land_samples(get_land_location(location_id), cache_root)


def get_land_sample_tile_prefixes() -> tuple[str, ...]:
    prefixes: list[str] = []
    seen: set[str] = set()
    for location in LAND_LOCATIONS:
        if location.tile_prefix in seen:
            continue
        seen.add(location.tile_prefix)
        prefixes.append(location.tile_prefix)
    return tuple(prefixes)


def get_land_season_blend(location: LandLocation, season_mode: str) -> float:
    if season_mode not in {"summer", "winter"}:
        raise ValueError(f"Unsupported season mode: {season_mode}")
    return 1.0 if ((location.hemisphere == "south") == (season_mode == "summer")) else 0.0


def download_configured_land_samples(
    *,
    date_arg: str | None = None,
    cache_dir: str = str(LAND_CACHE_ROOT),
    parallel: int = DEFAULT_SAMPLE_DOWNLOAD_PARALLEL,
) -> int:
    import satmaps

    date_value = satmaps.DEFAULT_DATE_PATHS if date_arg is None else date_arg
    date_paths = [date_path.strip() for date_path in date_value.split(",") if date_path.strip()]
    if not date_paths:
        raise ValueError("Expected at least one date path for tuner sample downloads.")

    satmaps.setup_gdal_cdse()
    work_units = tuple(
        satmaps.LandWorkUnit(unit_id=tile_prefix, source_subtiles=(tile_prefix,))
        for tile_prefix in get_land_sample_tile_prefixes()
    )
    downloaded = satmaps.download_source_tiles_to_cache(work_units, date_paths, cache_dir, parallel)
    if downloaded <= 0:
        raise RuntimeError(
            f"No source tiles were downloaded into {cache_dir}. Check the requested dates and CDSE access."
        )
    return downloaded


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


@lru_cache(maxsize=64)
def get_raster_size(path: str) -> tuple[int, int]:
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open sample band: {path}")
    size = (ds.RasterXSize, ds.RasterYSize)
    ds = None
    return size


def get_land_view(location_id: str, pan_x: float | None = None, pan_y: float | None = None) -> LandView:
    land_samples = get_land_samples(location_id)
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


def build_grade_defaults(
    low_break: float,
    low_slope: float,
    highlight_break: float | None = None,
    mid_slope: float = tiler.PREVIEW_DARKEN_MID_SLOPE,
    high_slope: float | None = None,
) -> dict[str, float]:
    resolved_highlight_break = low_break if highlight_break is None else highlight_break
    return {
        "db": low_break,
        "ghb": resolved_highlight_break,
        "ls": low_slope,
        "gms": mid_slope,
        "ghs": (
            tiler.derive_piecewise_high_slope(
                low_break,
                resolved_highlight_break,
                low_slope,
                mid_slope,
            )
            if high_slope is None
            else high_slope
        ),
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
        "exp": LAND_DEFAULT_EXPOSURE,
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
    params.update(
        build_grade_defaults(
            LAND_DEFAULT_GRADE_LOW_BREAK,
            LAND_DEFAULT_GRADE_LOW_SLOPE,
            highlight_break=LAND_DEFAULT_GRADE_HIGHLIGHT_BREAK,
            mid_slope=LAND_DEFAULT_GRADE_MID_SLOPE,
            high_slope=LAND_DEFAULT_GRADE_HIGHLIGHT_SLOPE,
        )
    )
    return params


def build_grade_preset_values(mode_defaults: dict[str, float], **overrides: float) -> dict[str, float]:
    values = {key: float(mode_defaults[key]) for key in GRADE_CONTROL_IDS}
    values.update(overrides)
    return values


def get_neutral_grade_values() -> dict[str, float]:
    return {
        "exp": 1.0,
        "gamma": 1.0,
        "shoulder": 1.0,
        "sat": 1.0,
        "db": 0.0,
        "ghb": 1.0,
        "ls": 1.0,
        "gms": 1.0,
        "ghs": 1.0,
    }


def get_grade_presets(mode: str) -> tuple[dict[str, object], ...]:
    defaults = get_mode_defaults(mode)
    if mode == "ocean":
        return (
            {"id": "balanced", "label": "Balanced", "values": build_grade_preset_values(defaults)},
            {
                "id": "punchy",
                "label": "Punchy",
                "values": build_grade_preset_values(
                    defaults,
                    exp=1.1,
                    gamma=2.2,
                    shoulder=1.0,
                    sat=1.1,
                    db=0.14,
                    ghb=0.74,
                    ls=0.15,
                    gms=1.1,
                    ghs=1.5,
                ),
            },
            {
                "id": "muted",
                "label": "Muted",
                "values": build_grade_preset_values(
                    defaults,
                    exp=0.9,
                    gamma=2.5,
                    shoulder=0.6,
                    sat=0.75,
                    db=0.06,
                    ghb=0.82,
                    ls=0.0,
                    gms=0.9,
                    ghs=1.2,
                ),
            },
            {
                "id": "glow",
                "label": "Glow",
                "values": build_grade_preset_values(
                    defaults,
                    exp=1.2,
                    gamma=2.0,
                    shoulder=1.3,
                    sat=0.95,
                    db=0.1,
                    ghb=0.7,
                    ls=0.05,
                    gms=1.0,
                    ghs=1.35,
                ),
            },
        )
    return (
        {"id": "balanced", "label": "Balanced", "values": build_grade_preset_values(defaults)},
        {
            "id": "punchy",
            "label": "Punchy",
            "values": build_grade_preset_values(
                defaults,
                exp=2.8,
                gamma=2.0,
                shoulder=0.7,
                sat=1.1,
                db=0.12,
                ghb=0.86,
                ls=0.15,
                gms=1.1,
                ghs=1.9,
            ),
        },
        {
            "id": "matte",
            "label": "Matte",
            "values": build_grade_preset_values(
                defaults,
                exp=2.3,
                gamma=2.5,
                shoulder=0.35,
                sat=0.8,
                db=0.05,
                ghb=0.94,
                ls=0.0,
                gms=0.9,
                ghs=1.3,
            ),
        },
        {
            "id": "vivid",
            "label": "Vivid",
            "values": build_grade_preset_values(
                defaults,
                exp=3.0,
                gamma=2.1,
                shoulder=0.6,
                sat=1.25,
                db=0.1,
                ghb=0.88,
                ls=0.05,
                gms=1.15,
                ghs=1.8,
            ),
        },
    )


def parse_request_params(
    mode: str,
    land_location: LandLocation | None = None,
    blend_mode_id: str | None = None,
) -> dict[str, float]:
    defaults = get_mode_defaults(mode)
    if mode == "land" and land_location is not None and blend_mode_id in {"summer", "winter"}:
        defaults["blend"] = get_land_season_blend(land_location, blend_mode_id)
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
        for sample in get_land_samples(location_id)
    )


def blend_land_samples(
    location: LandLocation,
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

    if blend_mode in {"summer", "winter"}:
        return secondary if get_land_season_blend(location, blend_mode) >= 0.5 else primary
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
    location: LandLocation,
    blend: float,
    blend_mode_id: str,
    land_view: LandView,
) -> FloatArray | None:
    cropped_samples = get_cropped_land_samples(location.id, land_view)
    if len(cropped_samples) == 1:
        return cropped_samples[0]
    return blend_land_samples(location, cropped_samples, blend, blend_mode_id)


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

    fg_on = request.args.get("fg", "1") == "1"
    land_location = get_land_location(request.args.get("loc"))
    land_samples = get_land_samples(land_location.id)
    land_blend_mode = get_land_blend_mode(request.args.get("blend_mode"))
    land_view = get_land_view(
        land_location.id,
        get_land_pan_arg("panx"),
        get_land_pan_arg("pany"),
    )
    params = parse_request_params(mode, land_location, land_blend_mode.id)
    source = (
        get_land_source(land_location, params.get("blend", 0.0), land_blend_mode.id, land_view)
        if mode == "land"
        else RAW_GEBCO
    )
    hist = get_histogram_data(source, mode, params.get("dmin", -11000.0), params.get("dmax", 0.0))
    return render_template(
        "index.html",
        mode=mode,
        params=params,
        defaults=get_mode_defaults(mode),
        grade_presets=get_grade_presets(mode),
        neutral_grade_values=get_neutral_grade_values(),
        raw_hist=hist,
        land_locations=LAND_LOCATIONS,
        land_blend_modes=LAND_BLEND_MODES,
        selected_land_blend_mode=land_blend_mode,
        default_land_blend_mode=DEFAULT_LAND_BLEND_MODE,
        selected_land_location=land_location,
        land_view=land_view,
        land_dates=[sample.date_label for sample in land_samples],
        has_land_blend=len(land_samples) > 1,
        fg_on=fg_on,
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
    params = parse_request_params(mode, land_location, land_blend_mode.id)
    source = (
        get_land_source(land_location, params.get("blend", 0.0), land_blend_mode.id, land_view)
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
    p = parse_request_params(mode, land_location, land_blend_mode.id)
    tm_on = request.args.get("tm", "0") == "1"
    fg_on = request.args.get("fg", "1") == "1"

    if mode == "land":
        source_rgb = get_land_source(land_location, p.get("blend", 0.0), land_blend_mode.id, land_view)
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
    parser = argparse.ArgumentParser(
        description="Start the satmaps style tuner or download its configured sample tiles."
    )
    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Download the configured tuner land sample tiles into the cache and exit",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Mosaic date(s), comma-separated. Only used with --download-samples.",
    )
    parser.add_argument(
        "--cache",
        default=str(LAND_CACHE_ROOT),
        help="Cache directory for tuner sample downloads (default: .cache).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_SAMPLE_DOWNLOAD_PARALLEL,
        help="Parallel sample downloads to run with --download-samples.",
    )
    args = parser.parse_args()

    if args.download_samples:
        downloaded = download_configured_land_samples(
            date_arg=args.date,
            cache_dir=args.cache,
            parallel=args.parallel,
        )
        print(f"Download complete. Cached {downloaded} folder(s) for the tuner.")
        return

    app.run(debug=True, port=5001)


if __name__ == "__main__":
    main()
