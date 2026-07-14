#!/usr/bin/env python3
"""
Compare two ways of simulating IWR for one ideal 1 km x 1 km pixel.

METHOD A - CURRENT FRACTIONAL PIXEL MODEL
-----------------------------------------
The crop mixture is represented by one weighted-average crop:

    Kc_pixel = sum(f_i * Kc_i) / sum(f_i)
    Zr_pixel = sum(f_i * Zr_i) / sum(f_i)
    S_threshold_pixel = AWC * 1000 *
                        sum(f_i * Zr_i * (1 - p_i)) / sum(f_i)

One soil-water balance is solved for the mixed pixel.

METHOD B - CROP-BY-CROP MODEL
-----------------------------
A separate soil-water balance is solved for each crop using its own Kc,
root depth and p. Daily crop IWR values are aggregated only at the end:

    IWR_pixel_mm = sum(f_i * IWR_i_mm)

The raw crop IWR depths must NOT be summed without area weighting.
For a 1 km2 pixel:

    volume_i_m3 = f_i * IWR_i_mm / 1000 * 1_000_000
                = f_i * IWR_i_mm * 1000

The daily calculation order and equations reproduce the uploaded iwr_model.py:
- effective precipitation = 0.95 * P
- ETc = Kc * PET/ET0
- green-water Ks is based on soil storage at the beginning of the day
- green ET = Ks * ETc
- blue IWR = max(ETc - green ET, 0)
- deep percolation is calculated from previous storage, threshold, TAW, Fmax
- runoff occurs when storage would exceed TAW
- irrigation is consumed on the same day and does not create residual blue storage

Inputs
------
1. A forcing CSV containing date, P and PET columns.
2. A crop parameter CSV containing:
       crop_name, root_depth_max_m, Kc_ini, Kc_mid, Kc_end, p
3. The crop fractions and phenology settings in the USER SETTINGS section below.

Outputs
-------
- daily_comparison.csv
- crop_daily_details.csv
- summary_by_method.csv
- summary_by_crop.csv
- daily_iwr_comparison.png
- cumulative_iwr_comparison.png
- soil_storage_comparison.png

Run
---
    python compare_fractional_vs_cropwise_iwr.py

Optional path overrides:
    python compare_fractional_vs_cropwise_iwr.py \
        --forcing /path/to/forcing.csv \
        --crop-parameters /path/to/crop_parameters.csv \
        --output-dir /path/to/output_folder
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

FORCING_CSV = Path("/share/data/DAO/input/forcing_pixel.csv")
CROP_PARAMETERS_CSV = Path("/share/data/DAO/static/processed/crop_parameters_aida_irrigated.csv")
OUTPUT_DIR = Path("/share/data/DAO/output_pixel_comparison")

# Ideal pixel requested by the user: 1 km x 1 km.
PIXEL_AREA_M2 = 1_000_000.0

# Soil parameters for the ideal pixel.
# AWC is FC - WP, in m3/m3.
AVAILABLE_WATER_CONTENT = 0.16
FMAX_MM_DAY = 125.0
INITIAL_STORAGE_FRACTION = 0.50
EFFECTIVE_PRECIPITATION_FACTOR = 0.95

# This reproduces the current code, where inactive phenology has Kc = 0.5.
INACTIVE_KC = 0.50

# In the existing raster model, irrigation is a pixel-level mask. Set this to
# True to reproduce an irrigated pixel in the old fractional calculation.
OLD_PIXEL_IS_IRRIGATED = True

# Crop mixture. Names must match crop_name in CROP_PARAMETERS_CSV.
# Fractions are fractions of the full 1 km2 pixel and may sum to less than 1.
# To isolate only the fractional-vs-cropwise aggregation effect, use the same
# phenology for all crops (seasons=None) and irrigated=True for every crop.
CROP_MIX = [
    {
        "crop_name": "maize_i",
        "fraction": 0.50,
        "irrigated": True,
        "seasons": None,
    },
    {
        "crop_name": "barley_i",
        "fraction": 0.30,
        "irrigated": True,
        "seasons": None,
    },
]

# Phenology modes:
#   "asap_dekad"    -> reproduces phenology_functions.py. Values may be 1..108.
#   "calendar_dates" -> uses MM-DD boundaries.
PHENOLOGY_MODE = "asap_dekad"

# Shared seasons used when a crop has seasons=None.
# ASAP example: add a second dictionary for a second season.
SHARED_SEASONS = [
    {"sos": 7, "tom": 13, "sen": 24, "eos": 29},
]

# Calendar-date alternative example:
# PHENOLOGY_MODE = "calendar_dates"
# SHARED_SEASONS = [
#     {"sos": "03-01", "tom": "05-01", "sen": "08-15", "eos": "10-15"},
# ]

# If True, crop fractions above 1 are normalized to sum to 1, matching the
# preparation logic in the raster model. A sum below 1 is left unchanged.
NORMALIZE_FRACTIONS_ABOVE_ONE = True

# Input checks.
MAX_P_MM_DAY = 300.0
MAX_PET_MM_DAY = 20.0


# =============================================================================
# CONSTANTS AND DATA STRUCTURES
# =============================================================================

PHENOLOGY_INACTIVE = 0
PHENOLOGY_GROWING = 1
PHENOLOGY_MAXIMUM = 2
PHENOLOGY_SENESCENCE = 3

STAGE_NAMES = {
    PHENOLOGY_INACTIVE: "inactive",
    PHENOLOGY_GROWING: "growing",
    PHENOLOGY_MAXIMUM: "maximum",
    PHENOLOGY_SENESCENCE: "senescence",
}

REQUIRED_CROP_COLUMNS = {
    "crop_name",
    "root_depth_max_m",
    "Kc_ini",
    "Kc_mid",
    "Kc_end",
    "p",
}


@dataclass(frozen=True)
class CropDefinition:
    crop_name: str
    fraction: float
    irrigated: bool
    root_depth_m: float
    kc_ini: float
    kc_mid: float
    kc_end: float
    p: float
    seasons: list[dict[str, Any]]


@dataclass
class BalanceState:
    storage_mm: float


@dataclass(frozen=True)
class StaticHydrology:
    taw_mm: float
    storage_threshold_mm: float
    root_depth_m: float
    p_equivalent: float


# =============================================================================
# INPUT READING AND VALIDATION
# =============================================================================


def _normalise_column_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def read_forcing_csv(path: Path) -> pd.DataFrame:
    """Read date, P and PET/ET0 forcing data and return standardized columns."""

    if not path.exists():
        raise FileNotFoundError(f"Forcing CSV not found: {path}")

    df = pd.read_csv(path)
    normalized = {_normalise_column_name(c): c for c in df.columns}

    aliases = {
        "date": ["date", "time", "datetime"],
        "P": ["p", "precipitation", "precip", "rain", "rainfall"],
        "PET": ["pet", "et0", "eto", "reference_et", "reference_evapotranspiration"],
    }

    selected: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                selected[target] = normalized[candidate]
                break
        if target not in selected:
            raise ValueError(
                f"Could not identify the '{target}' column in {path}. "
                f"Available columns: {list(df.columns)}"
            )

    out = df[[selected["date"], selected["P"], selected["PET"]]].copy()
    out.columns = ["date", "P", "PET"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["P"] = pd.to_numeric(out["P"], errors="coerce")
    out["PET"] = pd.to_numeric(out["PET"], errors="coerce")

    if out.isna().any().any():
        bad = out[out.isna().any(axis=1)]
        raise ValueError(
            "Forcing CSV contains invalid date/P/PET values. "
            f"First invalid rows:\n{bad.head(10).to_string(index=False)}"
        )

    if out["date"].duplicated().any():
        duplicates = out.loc[out["date"].duplicated(keep=False), "date"]
        raise ValueError(f"Duplicate forcing dates found: {duplicates.dt.date.tolist()[:10]}")

    out = out.sort_values("date").reset_index(drop=True)

    if (out["P"] < 0).any() or (out["P"] > MAX_P_MM_DAY).any():
        raise ValueError(f"P must be in [0, {MAX_P_MM_DAY}] mm/day.")

    if (out["PET"] < 0).any() or (out["PET"] > MAX_PET_MM_DAY).any():
        raise ValueError(f"PET must be in [0, {MAX_PET_MM_DAY}] mm/day.")

    expected = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    missing = expected.difference(out["date"])
    if len(missing) > 0:
        print(
            f"Warning: forcing has {len(missing)} missing calendar day(s). "
            "The model will advance only on rows present in the CSV."
        )

    return out


def read_crop_parameters(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Crop parameter CSV not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_CROP_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing crop parameter columns: {missing}")

    df = df.copy()
    df["crop_name"] = df["crop_name"].astype(str).str.strip()

    for column in REQUIRED_CROP_COLUMNS.difference({"crop_name"}):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Non-numeric values found in crop column '{column}'.")

    if df["crop_name"].duplicated().any():
        names = df.loc[df["crop_name"].duplicated(keep=False), "crop_name"].tolist()
        raise ValueError(f"Duplicate crop_name values found: {names}")

    if (df["root_depth_max_m"] <= 0).any():
        raise ValueError("All root_depth_max_m values must be > 0.")

    if ((df["p"] <= 0) | (df["p"] > 1)).any():
        raise ValueError("All p values must satisfy 0 < p <= 1.")

    return df


def build_crop_definitions(crop_df: pd.DataFrame) -> list[CropDefinition]:
    if not CROP_MIX:
        raise ValueError("CROP_MIX is empty.")

    crop_lookup = crop_df.set_index("crop_name")
    requested_names = [str(item["crop_name"]).strip() for item in CROP_MIX]
    unknown = [name for name in requested_names if name not in crop_lookup.index]
    if unknown:
        raise ValueError(
            f"Crops not found in crop parameter CSV: {unknown}. "
            f"Available names: {crop_df['crop_name'].tolist()}"
        )

    fractions = np.array([float(item["fraction"]) for item in CROP_MIX], dtype=float)
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("All crop fractions must be finite and >= 0.")

    fraction_sum = float(fractions.sum())
    if fraction_sum <= 0:
        raise ValueError("The crop fractions must sum to a positive value.")

    if fraction_sum > 1.01:
        if not NORMALIZE_FRACTIONS_ABOVE_ONE:
            raise ValueError(f"Crop fractions sum to {fraction_sum:.6f}, greater than 1.")
        print(
            f"Warning: crop fractions sum to {fraction_sum:.6f}; "
            "normalizing them to 1.0."
        )
        fractions = fractions / fraction_sum

    crops: list[CropDefinition] = []
    for item, fraction in zip(CROP_MIX, fractions):
        if fraction == 0:
            continue

        name = str(item["crop_name"]).strip()
        row = crop_lookup.loc[name]
        seasons = item.get("seasons")
        if seasons is None:
            seasons = SHARED_SEASONS

        validate_seasons(seasons, PHENOLOGY_MODE, crop_name=name)

        crops.append(
            CropDefinition(
                crop_name=name,
                fraction=float(fraction),
                irrigated=bool(item.get("irrigated", True)),
                root_depth_m=float(row["root_depth_max_m"]),
                kc_ini=float(row["Kc_ini"]),
                kc_mid=float(row["Kc_mid"]),
                kc_end=float(row["Kc_end"]),
                p=float(row["p"]),
                seasons=[dict(s) for s in seasons],
            )
        )

    return crops


# =============================================================================
# PHENOLOGY
# =============================================================================


def date_to_dekad(value: pd.Timestamp | datetime | date) -> int:
    day = value.day
    dekad_in_month = 1 if day <= 10 else 2 if day <= 20 else 3
    return (value.month - 1) * 3 + dekad_in_month


def validate_seasons(
    seasons: Iterable[dict[str, Any]],
    mode: str,
    crop_name: str,
) -> None:
    seasons = list(seasons)
    if not seasons:
        raise ValueError(f"No phenology season supplied for crop '{crop_name}'.")

    required = {"sos", "tom", "sen", "eos"}
    for season_index, season in enumerate(seasons, start=1):
        missing = required.difference(season)
        if missing:
            raise ValueError(
                f"Crop '{crop_name}', season {season_index}: missing {sorted(missing)}"
            )

        if mode == "asap_dekad":
            values = [int(season[k]) for k in ("sos", "tom", "sen", "eos")]
            if not all(1 <= value <= 108 for value in values):
                raise ValueError(
                    f"Crop '{crop_name}', season {season_index}: ASAP values must be 1..108."
                )
            if not (values[0] <= values[1] <= values[2] <= values[3]):
                raise ValueError(
                    f"Crop '{crop_name}', season {season_index}: expected sos <= tom <= sen <= eos."
                )
        elif mode == "calendar_dates":
            for key in ("sos", "tom", "sen", "eos"):
                try:
                    datetime.strptime(str(season[key]), "%m-%d")
                except ValueError as exc:
                    raise ValueError(
                        f"Crop '{crop_name}', season {season_index}: "
                        f"'{key}' must use MM-DD format."
                    ) from exc
        else:
            raise ValueError(
                f"Unknown PHENOLOGY_MODE '{mode}'. Use 'asap_dekad' or 'calendar_dates'."
            )


def _status_asap_dekad(current_date: pd.Timestamp, seasons: list[dict[str, Any]]) -> int:
    """Reproduce the d, d+36, d+72 logic in phenology_functions.py."""

    current = date_to_dekad(current_date)
    status = PHENOLOGY_INACTIVE

    for season in seasons:
        sos = int(season["sos"])
        tom = int(season["tom"])
        sen = int(season["sen"])
        eos = int(season["eos"])

        for candidate in (current, current + 36, current + 72):
            if sos <= candidate <= eos:
                if sos <= candidate < tom:
                    status = PHENOLOGY_GROWING
                elif tom <= candidate < sen:
                    status = PHENOLOGY_MAXIMUM
                elif sen <= candidate <= eos:
                    status = PHENOLOGY_SENESCENCE

    return status


def _month_day_ordinal(value: str) -> int:
    # Leap reference year supports 02-29 and gives a stable cyclic calendar.
    return datetime.strptime(f"2000-{value}", "%Y-%m-%d").timetuple().tm_yday


def _status_calendar_dates(
    current_date: pd.Timestamp,
    seasons: list[dict[str, Any]],
) -> int:
    current = datetime(2000, current_date.month, current_date.day).timetuple().tm_yday
    year_length = 366
    status = PHENOLOGY_INACTIVE

    for season in seasons:
        sos = _month_day_ordinal(str(season["sos"]))
        tom = _month_day_ordinal(str(season["tom"]))
        sen = _month_day_ordinal(str(season["sen"]))
        eos = _month_day_ordinal(str(season["eos"]))

        # Unwrap boundaries so cross-year seasons are monotonic.
        while tom < sos:
            tom += year_length
        while sen < tom:
            sen += year_length
        while eos < sen:
            eos += year_length

        for candidate in (current, current + year_length):
            if sos <= candidate <= eos:
                if sos <= candidate < tom:
                    status = PHENOLOGY_GROWING
                elif tom <= candidate < sen:
                    status = PHENOLOGY_MAXIMUM
                elif sen <= candidate <= eos:
                    status = PHENOLOGY_SENESCENCE

    return status


def phenology_status(current_date: pd.Timestamp, crop: CropDefinition) -> int:
    if PHENOLOGY_MODE == "asap_dekad":
        return _status_asap_dekad(current_date, crop.seasons)
    return _status_calendar_dates(current_date, crop.seasons)


def kc_for_status(crop: CropDefinition, status: int) -> float:
    if status == PHENOLOGY_GROWING:
        return crop.kc_ini
    if status == PHENOLOGY_MAXIMUM:
        return crop.kc_mid
    if status == PHENOLOGY_SENESCENCE:
        return crop.kc_end
    return INACTIVE_KC


# =============================================================================
# HYDROLOGICAL EQUATIONS - SCALAR VERSION OF iwr_model.py
# =============================================================================


def make_crop_hydrology(crop: CropDefinition) -> StaticHydrology:
    taw = AVAILABLE_WATER_CONTENT * crop.root_depth_m * 1000.0
    threshold = AVAILABLE_WATER_CONTENT * crop.root_depth_m * (1.0 - crop.p) * 1000.0
    return StaticHydrology(
        taw_mm=taw,
        storage_threshold_mm=threshold,
        root_depth_m=crop.root_depth_m,
        p_equivalent=crop.p,
    )


def make_fractional_hydrology(crops: list[CropDefinition]) -> StaticHydrology:
    fraction_sum = sum(c.fraction for c in crops)

    root_depth = sum(c.fraction * c.root_depth_m for c in crops) / fraction_sum
    threshold_root_depth = (
        sum(c.fraction * c.root_depth_m * (1.0 - c.p) for c in crops)
        / fraction_sum
    )

    taw = AVAILABLE_WATER_CONTENT * root_depth * 1000.0
    threshold = AVAILABLE_WATER_CONTENT * threshold_root_depth * 1000.0

    # Equivalent p is diagnostic only because averaging Zr and Zr*(1-p)
    # does not generally equal applying one simple arithmetic-average p.
    p_equivalent = 1.0 - threshold_root_depth / root_depth

    return StaticHydrology(
        taw_mm=taw,
        storage_threshold_mm=threshold,
        root_depth_m=root_depth,
        p_equivalent=p_equivalent,
    )


def compute_ks(storage_mm: float, storage_threshold_mm: float) -> float:
    if storage_threshold_mm <= 0:
        return 1.0
    if storage_mm < storage_threshold_mm:
        return float(np.clip(storage_mm / storage_threshold_mm, 0.0, 1.0))
    return 1.0


def compute_deep_percolation(
    storage_previous_mm: float,
    storage_threshold_mm: float,
    taw_mm: float,
) -> float:
    if taw_mm <= storage_threshold_mm:
        return 0.0
    if storage_previous_mm < storage_threshold_mm:
        return 0.0
    if storage_previous_mm > taw_mm:
        return 0.0

    theoretical = (
        FMAX_MM_DAY
        * (storage_previous_mm - storage_threshold_mm)
        / (taw_mm - storage_threshold_mm)
    )
    available = max(storage_previous_mm - storage_threshold_mm, 0.0)
    return min(theoretical, available)


def one_day_balance(
    storage_previous_mm: float,
    p_mm: float,
    pet_mm: float,
    kc: float,
    hydro: StaticHydrology,
    irrigated: bool,
) -> dict[str, float]:
    """Perform one daily step in the same order as the raster model."""

    p_effective = EFFECTIVE_PRECIPITATION_FACTOR * p_mm
    etc = pet_mm * kc

    # Important: this uses beginning-of-day storage, before today's rainfall.
    ks_green = compute_ks(storage_previous_mm, hydro.storage_threshold_mm)
    green_et = etc * ks_green
    iwr = max(etc - green_et, 0.0) if irrigated else 0.0

    et_for_balance = etc if irrigated else green_et
    deep_percolation = compute_deep_percolation(
        storage_previous_mm=storage_previous_mm,
        storage_threshold_mm=hydro.storage_threshold_mm,
        taw_mm=hydro.taw_mm,
    )

    # In the uploaded code, deficit scaling is applied only to non-irrigated pixels.
    if not irrigated:
        available = storage_previous_mm + p_effective
        outgoing = et_for_balance + deep_percolation
        if outgoing > 0 and available < outgoing:
            scale = max(available / outgoing, 0.0)
            et_for_balance *= scale
            green_et = et_for_balance
            deep_percolation *= scale

    # Irrigation is intentionally excluded from the runoff test.
    balance_before_runoff = (
        storage_previous_mm + p_effective - et_for_balance - deep_percolation
    )
    runoff = max(balance_before_runoff - hydro.taw_mm, 0.0)

    storage_new = (
        storage_previous_mm
        + p_effective
        - et_for_balance
        - deep_percolation
        - runoff
        + iwr
    )
    storage_new = float(np.clip(storage_new, 0.0, hydro.taw_mm))

    residual = (
        storage_previous_mm
        + p_effective
        + iwr
        - et_for_balance
        - deep_percolation
        - runoff
        - storage_new
    )

    return {
        "P_effective_mm": p_effective,
        "Kc": kc,
        "ETc_mm": etc,
        "Ks_green": ks_green,
        "green_ET_mm": green_et,
        "IWR_mm": iwr,
        "ET_for_balance_mm": et_for_balance,
        "deep_percolation_mm": deep_percolation,
        "runoff_mm": runoff,
        "storage_start_mm": storage_previous_mm,
        "storage_end_mm": storage_new,
        "water_balance_residual_mm": residual,
    }


# =============================================================================
# SIMULATION
# =============================================================================


def run_comparison(
    forcing: pd.DataFrame,
    crops: list[CropDefinition],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fraction_sum = sum(c.fraction for c in crops)
    fractional_hydro = make_fractional_hydrology(crops)

    old_state = BalanceState(
        storage_mm=INITIAL_STORAGE_FRACTION * fractional_hydro.taw_mm
    )

    crop_hydrology = {c.crop_name: make_crop_hydrology(c) for c in crops}
    crop_states = {
        c.crop_name: BalanceState(
            storage_mm=INITIAL_STORAGE_FRACTION * crop_hydrology[c.crop_name].taw_mm
        )
        for c in crops
    }

    daily_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []

    for forcing_row in forcing.itertuples(index=False):
        current_date = pd.Timestamp(forcing_row.date)
        p_mm = float(forcing_row.P)
        pet_mm = float(forcing_row.PET)

        crop_status: dict[str, int] = {}
        crop_kc: dict[str, float] = {}
        for crop in crops:
            status = phenology_status(current_date, crop)
            crop_status[crop.crop_name] = status
            crop_kc[crop.crop_name] = kc_for_status(crop, status)

        # Current fractional method: one daily weighted Kc and one storage.
        kc_old = sum(c.fraction * crop_kc[c.crop_name] for c in crops) / fraction_sum
        old = one_day_balance(
            storage_previous_mm=old_state.storage_mm,
            p_mm=p_mm,
            pet_mm=pet_mm,
            kc=kc_old,
            hydro=fractional_hydro,
            irrigated=OLD_PIXEL_IS_IRRIGATED,
        )
        old_state.storage_mm = old["storage_end_mm"]

        # Crop-by-crop method: independent states, then area-weight final IWR.
        cropwise_pixel_iwr_mm = 0.0
        cropwise_volume_m3 = 0.0
        cropwise_green_et_pixel_mm = 0.0
        cropwise_etc_pixel_mm = 0.0
        cropwise_storage_pixel_mm = 0.0

        for crop in crops:
            hydro = crop_hydrology[crop.crop_name]
            state = crop_states[crop.crop_name]
            result = one_day_balance(
                storage_previous_mm=state.storage_mm,
                p_mm=p_mm,
                pet_mm=pet_mm,
                kc=crop_kc[crop.crop_name],
                hydro=hydro,
                irrigated=crop.irrigated,
            )
            state.storage_mm = result["storage_end_mm"]

            iwr_pixel_contribution = crop.fraction * result["IWR_mm"]
            volume_contribution = (
                result["IWR_mm"] / 1000.0 * crop.fraction * PIXEL_AREA_M2
            )

            cropwise_pixel_iwr_mm += iwr_pixel_contribution
            cropwise_volume_m3 += volume_contribution
            cropwise_green_et_pixel_mm += crop.fraction * result["green_ET_mm"]
            cropwise_etc_pixel_mm += crop.fraction * result["ETc_mm"]
            cropwise_storage_pixel_mm += crop.fraction * result["storage_end_mm"]

            crop_rows.append(
                {
                    "date": current_date,
                    "crop_name": crop.crop_name,
                    "fraction_of_pixel": crop.fraction,
                    "crop_area_m2": crop.fraction * PIXEL_AREA_M2,
                    "irrigated": crop.irrigated,
                    "phenology_status": crop_status[crop.crop_name],
                    "phenology_stage": STAGE_NAMES[crop_status[crop.crop_name]],
                    "root_depth_m": crop.root_depth_m,
                    "p": crop.p,
                    "TAW_mm_crop_area": hydro.taw_mm,
                    "storage_threshold_mm_crop_area": hydro.storage_threshold_mm,
                    **result,
                    "IWR_pixel_contribution_mm": iwr_pixel_contribution,
                    "IWR_volume_m3": volume_contribution,
                    "green_ET_pixel_contribution_mm": crop.fraction * result["green_ET_mm"],
                    "ETc_pixel_contribution_mm": crop.fraction * result["ETc_mm"],
                    "storage_end_pixel_contribution_mm": crop.fraction * result["storage_end_mm"],
                }
            )

        # Old model returns depth over cropped area. Convert it to a full-pixel
        # equivalent before comparing it with the cropwise area-weighted sum.
        old_pixel_iwr_mm = fraction_sum * old["IWR_mm"]
        old_volume_m3 = old_pixel_iwr_mm / 1000.0 * PIXEL_AREA_M2
        cropwise_cropped_area_iwr_mm = cropwise_pixel_iwr_mm / fraction_sum

        daily_rows.append(
            {
                "date": current_date,
                "P_mm": p_mm,
                "PET_mm": pet_mm,
                "crop_fraction_sum": fraction_sum,
                "cropped_area_m2": fraction_sum * PIXEL_AREA_M2,
                "old_root_depth_m": fractional_hydro.root_depth_m,
                "old_p_equivalent": fractional_hydro.p_equivalent,
                "old_TAW_mm_crop_area": fractional_hydro.taw_mm,
                "old_storage_threshold_mm_crop_area": fractional_hydro.storage_threshold_mm,
                "old_Kc": old["Kc"],
                "old_ETc_mm_crop_area": old["ETc_mm"],
                "old_Ks_green": old["Ks_green"],
                "old_green_ET_mm_crop_area": old["green_ET_mm"],
                "old_IWR_mm_crop_area": old["IWR_mm"],
                "old_IWR_mm_full_pixel": old_pixel_iwr_mm,
                "old_IWR_volume_m3": old_volume_m3,
                "old_storage_start_mm_crop_area": old["storage_start_mm"],
                "old_storage_end_mm_crop_area": old["storage_end_mm"],
                "old_deep_percolation_mm_crop_area": old["deep_percolation_mm"],
                "old_runoff_mm_crop_area": old["runoff_mm"],
                "old_water_balance_residual_mm": old["water_balance_residual_mm"],
                "cropwise_ETc_mm_full_pixel": cropwise_etc_pixel_mm,
                "cropwise_green_ET_mm_full_pixel": cropwise_green_et_pixel_mm,
                "cropwise_IWR_mm_full_pixel": cropwise_pixel_iwr_mm,
                "cropwise_IWR_mm_cropped_area": cropwise_cropped_area_iwr_mm,
                "cropwise_IWR_volume_m3": cropwise_volume_m3,
                "cropwise_storage_end_mm_full_pixel": cropwise_storage_pixel_mm,
                "difference_cropwise_minus_old_mm_full_pixel": (
                    cropwise_pixel_iwr_mm - old_pixel_iwr_mm
                ),
                "difference_cropwise_minus_old_m3": (
                    cropwise_volume_m3 - old_volume_m3
                ),
            }
        )

    daily = pd.DataFrame(daily_rows)
    crop_daily = pd.DataFrame(crop_rows)

    daily["old_cumulative_IWR_mm_full_pixel"] = daily[
        "old_IWR_mm_full_pixel"
    ].cumsum()
    daily["cropwise_cumulative_IWR_mm_full_pixel"] = daily[
        "cropwise_IWR_mm_full_pixel"
    ].cumsum()
    daily["cumulative_difference_mm_full_pixel"] = (
        daily["cropwise_cumulative_IWR_mm_full_pixel"]
        - daily["old_cumulative_IWR_mm_full_pixel"]
    )

    daily["old_cumulative_IWR_volume_m3"] = daily["old_IWR_volume_m3"].cumsum()
    daily["cropwise_cumulative_IWR_volume_m3"] = daily[
        "cropwise_IWR_volume_m3"
    ].cumsum()

    total_old_mm = float(daily["old_IWR_mm_full_pixel"].sum())
    total_cropwise_mm = float(daily["cropwise_IWR_mm_full_pixel"].sum())
    absolute_difference = total_cropwise_mm - total_old_mm
    relative_difference = (
        100.0 * absolute_difference / total_old_mm if total_old_mm > 0 else np.nan
    )

    summary_method = pd.DataFrame(
        [
            {
                "method": "old_fractional_pixel",
                "IWR_total_mm_full_pixel": total_old_mm,
                "IWR_total_mm_cropped_area": float(daily["old_IWR_mm_crop_area"].sum()),
                "IWR_total_volume_m3": float(daily["old_IWR_volume_m3"].sum()),
                "difference_from_old_mm_full_pixel": 0.0,
                "difference_from_old_percent": 0.0,
            },
            {
                "method": "new_crop_by_crop",
                "IWR_total_mm_full_pixel": total_cropwise_mm,
                "IWR_total_mm_cropped_area": float(
                    daily["cropwise_IWR_mm_cropped_area"].sum()
                ),
                "IWR_total_volume_m3": float(
                    daily["cropwise_IWR_volume_m3"].sum()
                ),
                "difference_from_old_mm_full_pixel": absolute_difference,
                "difference_from_old_percent": relative_difference,
            },
        ]
    )

    summary_crop = (
        crop_daily.groupby(
            ["crop_name", "fraction_of_pixel", "crop_area_m2", "irrigated"],
            as_index=False,
        )
        .agg(
            IWR_total_mm_crop_area=("IWR_mm", "sum"),
            IWR_total_mm_full_pixel_contribution=("IWR_pixel_contribution_mm", "sum"),
            IWR_total_volume_m3=("IWR_volume_m3", "sum"),
            ETc_total_mm_crop_area=("ETc_mm", "sum"),
            green_ET_total_mm_crop_area=("green_ET_mm", "sum"),
            deep_percolation_total_mm_crop_area=("deep_percolation_mm", "sum"),
            runoff_total_mm_crop_area=("runoff_mm", "sum"),
        )
        .sort_values("IWR_total_mm_full_pixel_contribution", ascending=False)
        .reset_index(drop=True)
    )

    return daily, crop_daily, summary_method, summary_crop


# =============================================================================
# OUTPUTS
# =============================================================================


def save_plots(daily: pd.DataFrame, crop_daily: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily["date"], daily["old_IWR_mm_full_pixel"], label="Old fractional pixel")
    ax.plot(daily["date"], daily["cropwise_IWR_mm_full_pixel"], label="Crop by crop")
    ax.set_title("Daily IWR: old fractional pixel vs crop-by-crop")
    ax.set_xlabel("Date")
    ax.set_ylabel("IWR (mm/day, full-pixel equivalent)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "daily_iwr_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        daily["date"],
        daily["old_cumulative_IWR_mm_full_pixel"],
        label="Old fractional pixel",
    )
    ax.plot(
        daily["date"],
        daily["cropwise_cumulative_IWR_mm_full_pixel"],
        label="Crop by crop",
    )
    ax.set_title("Cumulative IWR comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IWR (mm, full-pixel equivalent)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_iwr_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        daily["date"],
        daily["old_storage_end_mm_crop_area"],
        label="Old mixed storage",
        linewidth=2.2,
    )
    for crop_name, group in crop_daily.groupby("crop_name"):
        ax.plot(group["date"], group["storage_end_mm"], label=f"{crop_name} storage")
    ax.set_title("Independent crop storages versus mixed-pixel storage")
    ax.set_xlabel("Date")
    ax.set_ylabel("Soil water storage (mm over corresponding crop/root zone)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "soil_storage_comparison.png", dpi=180)
    plt.close(fig)


def write_run_configuration(
    output_dir: Path,
    forcing_path: Path,
    crop_parameters_path: Path,
    crops: list[CropDefinition],
) -> None:
    config = {
        "forcing_csv": str(forcing_path.resolve()),
        "crop_parameters_csv": str(crop_parameters_path.resolve()),
        "pixel_area_m2": PIXEL_AREA_M2,
        "available_water_content_fc_minus_wp": AVAILABLE_WATER_CONTENT,
        "fmax_mm_day": FMAX_MM_DAY,
        "initial_storage_fraction": INITIAL_STORAGE_FRACTION,
        "effective_precipitation_factor": EFFECTIVE_PRECIPITATION_FACTOR,
        "inactive_kc": INACTIVE_KC,
        "old_pixel_is_irrigated": OLD_PIXEL_IS_IRRIGATED,
        "phenology_mode": PHENOLOGY_MODE,
        "shared_seasons": SHARED_SEASONS,
        "crop_mix": [
            {
                "crop_name": c.crop_name,
                "fraction": c.fraction,
                "irrigated": c.irrigated,
                "root_depth_m": c.root_depth_m,
                "Kc_ini": c.kc_ini,
                "Kc_mid": c.kc_mid,
                "Kc_end": c.kc_end,
                "p": c.p,
                "seasons": c.seasons,
            }
            for c in crops
        ],
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def print_summary(
    crops: list[CropDefinition],
    summary_method: pd.DataFrame,
    summary_crop: pd.DataFrame,
) -> None:
    fraction_sum = sum(c.fraction for c in crops)
    print("\nPixel setup")
    print("-----------")
    print(f"Pixel area:       {PIXEL_AREA_M2:,.0f} m2")
    print(f"Crop fraction:    {fraction_sum:.6f}")
    print(f"Cropped area:     {fraction_sum * PIXEL_AREA_M2:,.0f} m2")
    print(f"AWC (FC-WP):      {AVAILABLE_WATER_CONTENT:.4f} m3/m3")
    print(f"Fmax:             {FMAX_MM_DAY:.2f} mm/day")
    print(f"Phenology mode:   {PHENOLOGY_MODE}")

    print("\nMethod comparison")
    print("-----------------")
    print(summary_method.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nCrop-by-crop contributions")
    print("--------------------------")
    display_columns = [
        "crop_name",
        "fraction_of_pixel",
        "IWR_total_mm_crop_area",
        "IWR_total_mm_full_pixel_contribution",
        "IWR_total_volume_m3",
    ]
    print(
        summary_crop[display_columns].to_string(
            index=False, float_format=lambda x: f"{x:,.4f}"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fractional-pixel and crop-by-crop IWR for one 1 km2 pixel."
    )
    parser.add_argument("--forcing", type=Path, default=FORCING_CSV)
    parser.add_argument("--crop-parameters", type=Path, default=CROP_PARAMETERS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    forcing = read_forcing_csv(args.forcing)
    crop_df = read_crop_parameters(args.crop_parameters)
    crops = build_crop_definitions(crop_df)

    daily, crop_daily, summary_method, summary_crop = run_comparison(
        forcing=forcing,
        crops=crops,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "daily_comparison.csv", index=False)
    crop_daily.to_csv(args.output_dir / "crop_daily_details.csv", index=False)
    summary_method.to_csv(args.output_dir / "summary_by_method.csv", index=False)
    summary_crop.to_csv(args.output_dir / "summary_by_crop.csv", index=False)

    save_plots(daily=daily, crop_daily=crop_daily, output_dir=args.output_dir)
    write_run_configuration(
        output_dir=args.output_dir,
        forcing_path=args.forcing,
        crop_parameters_path=args.crop_parameters,
        crops=crops,
    )
    print_summary(crops, summary_method, summary_crop)

    max_old_residual = float(daily["old_water_balance_residual_mm"].abs().max())
    max_crop_residual = float(crop_daily["water_balance_residual_mm"].abs().max())
    print("\nBalance diagnostics")
    print("-------------------")
    print(f"Max absolute old-model residual:      {max_old_residual:.3e} mm")
    print(f"Max absolute cropwise residual:       {max_crop_residual:.3e} mm")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
