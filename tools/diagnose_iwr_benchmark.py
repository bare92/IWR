#!/usr/bin/env python3
"""
Diagnostic script for the IWR benchmark model outputs.

Reads existing config/config.json and model outputs (no model re-run).
Produces CSV tables and PNG plots for interpretation and sharing.

Usage:
    python tools/diagnose_iwr_benchmark.py --config config/config.json
    python tools/diagnose_iwr_benchmark.py --config config/config.json --aida-mcm 1831.1
    python tools/diagnose_iwr_benchmark.py --config config/config.json --aida-mcm 1831.1 --out /my/output/folder
"""

import argparse
import json
import sys
import warnings
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------------------------
# Path setup: allow imports from simple_code/ regardless of launch directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SIMPLE_CODE_DIR = _PROJECT_ROOT / "simple_code"

for _p in [str(_SIMPLE_CODE_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crop_functions import check_crop_raster_and_csv  # noqa: E402
from iwr_model import create_kc_pixel, prepare_crop_fractions  # noqa: E402
from phenology_functions import (  # noqa: E402
    PHENOLOGY_GROWING,
    PHENOLOGY_INACTIVE,
    PHENOLOGY_MAXIMUM,
    PHENOLOGY_SENESCENCE,
    create_phenology_status_mask_from_date,
    load_phenology_layers,
)
from utilities import read_forcing_geotiff_day  # noqa: E402

# ---------------------------------------------------------------------------
# Nodata constants and helpers
# ---------------------------------------------------------------------------
NODATA = -9999.0
_FILL_SENTINELS = [(-9999.0, 1.0), (9999.0, 1.0)]


def is_valid(arr):
    """Return boolean mask: True where arr is a real value (not nodata/nan/inf)."""
    mask = np.isfinite(arr)
    for sentinel, tol in _FILL_SENTINELS:
        mask &= np.abs(arr - sentinel) > tol
    return mask


def valid_values(arr):
    return arr[is_valid(arr)]


def weighted_mean(arr, weights):
    """Weighted mean. Both arr and weights must be 2-D arrays."""
    valid = is_valid(arr) & is_valid(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(arr[valid] * weights[valid]) / np.sum(weights[valid]))


def pct_over_valid(arr, q):
    """Unweighted percentile over valid pixels."""
    v = valid_values(arr)
    if v.size == 0:
        return np.nan
    return float(np.percentile(v, q))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg, config_path


def resolve_path(p, config_dir):
    """Return an absolute Path, resolving relative paths against config_dir."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


# ---------------------------------------------------------------------------
# Pixel area
# ---------------------------------------------------------------------------

def compute_pixel_area_m2(profile):
    transform = profile["transform"]
    crs = profile.get("crs")
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    if crs is not None and not crs.is_projected:
        warnings.warn(
            "CRS is geographic (degrees). Pixel area is in degrees^2, not m^2. "
            "Area results will be incorrect for volume calculations.",
            UserWarning,
            stacklevel=2,
        )
    return pixel_width * pixel_height


# ---------------------------------------------------------------------------
# IWR file discovery
# ---------------------------------------------------------------------------

def find_daily_iwr_files(iwr_output_folder):
    """Return dict: date -> Path for all iwr_YYYYMMDD.tif files."""
    folder = Path(iwr_output_folder)
    result = {}
    for f in sorted(folder.glob("iwr_????????.tif")):
        stem = f.stem  # iwr_YYYYMMDD
        date_str = stem[4:]
        try:
            d = date_cls(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            result[d] = f
        except ValueError:
            pass
    return result


def read_iwr_day(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


# ---------------------------------------------------------------------------
# Output 1: area_diagnostics.csv and crop_area_by_band.csv
# ---------------------------------------------------------------------------

def compute_area_diagnostics(
    crop_fraction_data_prepared,
    crop_fraction_sum,
    irrigation_mask,
    pixel_area_m2,
    crop_df,
    band_descriptions,
    out_folder,
    log_lines,
):
    irr_binary = (irrigation_mask == 1).astype(np.float32)
    crop_pixels = crop_fraction_sum > 0
    irrigated_pixels_mask = irr_binary > 0
    irrigated_crop_pixels = crop_pixels & (irr_binary > 0)

    total_grid_area_km2 = float(pixel_area_m2 * crop_fraction_sum.size / 1e6)
    crop_area_km2 = float(pixel_area_m2 * np.sum(crop_fraction_sum[crop_pixels]) / 1e6)
    irrigated_mask_area_km2 = float(pixel_area_m2 * np.sum(irr_binary) / 1e6)
    irrigated_crop_area_km2 = float(
        pixel_area_m2
        * np.sum(crop_fraction_sum[irrigated_crop_pixels] * irr_binary[irrigated_crop_pixels])
        / 1e6
    )

    n_crop_pixels = int(np.sum(crop_pixels))
    n_irrigated_pixels = int(np.sum(irrigated_pixels_mask))
    n_irrigated_crop_pixels = int(np.sum(irrigated_crop_pixels))

    def frac_stats(arr, mask):
        v = arr[mask]
        if v.size == 0:
            return {k: np.nan for k in ["min", "p05", "p50", "p95", "p99", "max"]}
        return {
            "min": float(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
        }

    cs_stats_crop = frac_stats(crop_fraction_sum, crop_pixels)
    cs_stats_irr_crop = frac_stats(crop_fraction_sum, irrigated_crop_pixels)

    rows = [
        ("pixel_area_m2", pixel_area_m2),
        ("total_grid_area_km2", total_grid_area_km2),
        ("crop_area_km2", crop_area_km2),
        ("irrigated_mask_area_km2", irrigated_mask_area_km2),
        ("irrigated_crop_area_km2", irrigated_crop_area_km2),
        ("n_crop_pixels", n_crop_pixels),
        ("n_irrigated_pixels", n_irrigated_pixels),
        ("n_irrigated_crop_pixels", n_irrigated_crop_pixels),
    ]
    for key, val in cs_stats_crop.items():
        rows.append((f"crop_fraction_sum_{key}_over_crop_pixels", round(val, 6) if np.isfinite(val) else val))
    for key, val in cs_stats_irr_crop.items():
        rows.append((f"crop_fraction_sum_{key}_over_irrigated_crop_pixels", round(val, 6) if np.isfinite(val) else val))

    area_df = pd.DataFrame(rows, columns=["metric", "value"])
    area_path = out_folder / "area_diagnostics.csv"
    area_df.to_csv(area_path, index=False)
    log_lines.append(f"Written: {area_path}")

    # Per-crop areas
    irr_flat = irr_binary.ravel()
    total_irrigated_crop_area_m2 = float(pixel_area_m2 * np.sum(crop_fraction_sum.ravel() * irr_flat))

    crop_rows = []
    for i in range(crop_fraction_data_prepared.shape[0]):
        band = crop_fraction_data_prepared[i]
        crop_name = (
            str(band_descriptions[i]).strip()
            if band_descriptions and band_descriptions[i] not in (None, "")
            else crop_df.iloc[i]["crop_name"]
        )
        total_area_km2 = float(pixel_area_m2 * float(np.sum(band)) / 1e6)
        irr_area_km2 = float(pixel_area_m2 * float(np.sum(band * irr_binary)) / 1e6)
        share = (
            100.0 * irr_area_km2 * 1e6 / total_irrigated_crop_area_m2
            if total_irrigated_crop_area_m2 > 0
            else np.nan
        )
        crop_rows.append({
            "band": i + 1,
            "crop_name": crop_name,
            "total_crop_area_km2": round(total_area_km2, 4),
            "irrigated_crop_area_km2": round(irr_area_km2, 4),
            "share_of_irrigated_crop_area_percent": (
                round(share, 4) if np.isfinite(share) else np.nan
            ),
        })

    crop_area_df = pd.DataFrame(crop_rows)
    crop_area_path = out_folder / "crop_area_by_band.csv"
    crop_area_df.to_csv(crop_area_path, index=False)
    log_lines.append(f"Written: {crop_area_path}")

    return {
        "irrigated_crop_area_m2": irrigated_crop_area_km2 * 1e6,
        "irrigated_mask_area_m2": irrigated_mask_area_km2 * 1e6,
        "irrigated_crop_area_km2": irrigated_crop_area_km2,
        "irrigated_mask_area_km2": irrigated_mask_area_km2,
    }


# ---------------------------------------------------------------------------
# Outputs 2 & 3: annual and monthly IWR volume diagnostics
# ---------------------------------------------------------------------------

def compute_iwr_volume_diagnostics(
    iwr_files,
    crop_fraction_sum,
    irrigation_mask,
    pixel_area_m2,
    area_stats,
    aida_mcm,
    out_folder,
    log_lines,
):
    irr_binary = (irrigation_mask == 1).astype(np.float32)
    irrigated_crop_area_m2 = area_stats["irrigated_crop_area_m2"]
    irrigated_mask_area_m2 = area_stats["irrigated_mask_area_m2"]

    day_records = []
    for d, fpath in sorted(iwr_files.items()):
        iwr_arr, _ = read_iwr_day(fpath)
        valid = is_valid(iwr_arr)
        iwr_clean = np.where(valid, iwr_arr, 0.0)

        # Correct benchmark volume: mm/day / 1000 * m2 * crop_fraction * irrigation_mask
        vol_cropped_m3 = float(
            np.sum(iwr_clean / 1000.0 * pixel_area_m2 * crop_fraction_sum * irr_binary)
        )
        # Diagnostic alternative: full irrigated pixel
        vol_full_irr_m3 = float(
            np.sum(iwr_clean / 1000.0 * pixel_area_m2 * irr_binary)
        )
        iwr_depth_mm = (
            vol_cropped_m3 / irrigated_crop_area_m2 * 1000.0
            if irrigated_crop_area_m2 > 0
            else np.nan
        )

        day_records.append({
            "date": d,
            "year": d.year,
            "month": d.month,
            "vol_cropped_m3": vol_cropped_m3,
            "vol_full_irr_m3": vol_full_irr_m3,
            "iwr_depth_mm_over_irrigated_crop_area": iwr_depth_mm,
        })

    if not day_records:
        log_lines.append("WARNING: No daily IWR files found. Volume diagnostics empty.")
        empty = pd.DataFrame()
        return empty, empty, empty

    day_df = pd.DataFrame(day_records)

    # ----- Monthly -----
    monthly = (
        day_df.groupby(["year", "month"])[["vol_cropped_m3", "vol_full_irr_m3"]]
        .sum()
        .reset_index()
    )
    monthly["monthly_iwr_mcm_cropped_area"] = monthly["vol_cropped_m3"] / 1e6
    monthly["monthly_iwr_mcm_full_irrigated_pixel"] = monthly["vol_full_irr_m3"] / 1e6
    monthly["monthly_iwr_depth_mm_over_irrigated_crop_area"] = (
        monthly["vol_cropped_m3"] / irrigated_crop_area_m2 * 1000.0
        if irrigated_crop_area_m2 > 0
        else np.nan
    )
    monthly_out = monthly[[
        "year", "month",
        "monthly_iwr_mcm_cropped_area",
        "monthly_iwr_mcm_full_irrigated_pixel",
        "monthly_iwr_depth_mm_over_irrigated_crop_area",
    ]]
    monthly_path = out_folder / "monthly_iwr_volume_diagnostics.csv"
    monthly_out.to_csv(monthly_path, index=False)
    log_lines.append(f"Written: {monthly_path}")

    # ----- Annual -----
    annual = (
        day_df.groupby("year")[["vol_cropped_m3", "vol_full_irr_m3"]]
        .sum()
        .reset_index()
    )
    annual["annual_iwr_mcm_cropped_area"] = annual["vol_cropped_m3"] / 1e6
    annual["annual_iwr_mcm_full_irrigated_pixel"] = annual["vol_full_irr_m3"] / 1e6
    annual["mean_iwr_depth_mm_over_irrigated_crop_area"] = (
        annual["vol_cropped_m3"] / irrigated_crop_area_m2 * 1000.0
        if irrigated_crop_area_m2 > 0
        else np.nan
    )
    annual["mean_iwr_depth_mm_over_full_irrigated_mask"] = (
        annual["vol_full_irr_m3"] / irrigated_mask_area_m2 * 1000.0
        if irrigated_mask_area_m2 > 0
        else np.nan
    )

    if aida_mcm is not None:
        annual["ratio_model_to_aida"] = annual["annual_iwr_mcm_cropped_area"] / aida_mcm
        annual["missing_mcm_to_aida"] = aida_mcm - annual["annual_iwr_mcm_cropped_area"]
        annual["aida_equivalent_depth_mm_over_irrigated_crop_area"] = (
            aida_mcm * 1e6 / irrigated_crop_area_m2 * 1000.0
            if irrigated_crop_area_m2 > 0
            else np.nan
        )

    annual_path = out_folder / "annual_iwr_volume_diagnostics.csv"
    annual.to_csv(annual_path, index=False)
    log_lines.append(f"Written: {annual_path}")

    return annual, monthly_out, day_df


# ---------------------------------------------------------------------------
# Output 4: forcing_and_demand_diagnostics.csv
# ---------------------------------------------------------------------------

def compute_forcing_and_demand_diagnostics(
    start_date,
    end_date,
    precip_folder,
    et0_folder,
    crop_fraction_data_prepared,
    crop_fraction_sum,
    irrigation_mask,
    pixel_area_m2,
    phenology,
    crop_df,
    out_folder,
    log_lines,
):
    irr_binary = (irrigation_mask == 1).astype(np.float32)
    irr_crop_mask = (crop_fraction_sum > 0) & (irrigation_mask == 1)
    # Base weights for area-weighted means over irrigated crop pixels
    weights_base = pixel_area_m2 * crop_fraction_sum * irr_binary

    day_records = []
    missing_precip = []
    missing_et0 = []

    current = start_date
    while current <= end_date:
        # Progress indicator
        if current.day == 1:
            print(f"  Forcing/demand: {current}", flush=True)

        # --- Precipitation ---
        try:
            precip = read_forcing_geotiff_day(
                geotiff_folder=precip_folder,
                date=current,
                min_value=0.0,
                max_value=None,
                nodata=NODATA,
                min_valid_fraction=0.0,
                variable_name="precipitation",
            )
        except (FileNotFoundError, ValueError):
            missing_precip.append(str(current))
            current += timedelta(days=1)
            continue

        # --- ET0 ---
        try:
            et0 = read_forcing_geotiff_day(
                geotiff_folder=et0_folder,
                date=current,
                min_value=0.0,
                max_value=None,
                nodata=NODATA,
                min_valid_fraction=0.0,
                variable_name="et0",
            )
        except (FileNotFoundError, ValueError):
            missing_et0.append(str(current))
            current += timedelta(days=1)
            continue

        # --- Phenology and Kc ---
        phenology_status = create_phenology_status_mask_from_date(
            current_date=current,
            phenology=phenology,
            nodata=-9999,
        )
        kc_pixel = create_kc_pixel(
            current_date=current,
            phenology=phenology,
            crop_fraction_data=crop_fraction_data_prepared,
            crop_df=crop_df,
            nodata=-9999.0,
        )

        # --- Model-valid mask ---
        forcing_valid = is_valid(precip) & is_valid(et0)
        model_valid = forcing_valid & irr_crop_mask

        # --- ETc (only where model valid) ---
        etc = np.where(model_valid, kc_pixel * et0, 0.0).astype(np.float32)

        # Weights restricted to model-valid pixels
        w = np.where(model_valid, weights_base, 0.0)
        w_sum = float(np.sum(w))
        n_valid = int(np.sum(model_valid))

        def wmean(arr):
            if w_sum <= 0:
                return np.nan
            arr_clean = np.where(model_valid & is_valid(arr), arr, 0.0)
            return float(np.sum(arr_clean * w) / w_sum)

        def pctile(arr, q):
            v = arr[model_valid & is_valid(arr)]
            if v.size == 0:
                return np.nan
            return float(np.percentile(v, q))

        def area_pct(bool_mask):
            if n_valid == 0:
                return np.nan
            return 100.0 * float(np.sum(bool_mask & model_valid)) / n_valid

        row = {
            "date": str(current),
            "year": current.year,
            "month": current.month,
            "precipitation_mean_irrigated_crop_area": round(wmean(precip), 4),
            "precipitation_p95_irrigated_crop_area": round(pctile(precip, 95), 4),
            "p_eff_mean_irrigated_crop_area": round(0.95 * wmean(precip), 4),
            "et0_mean_irrigated_crop_area": round(wmean(et0), 4),
            "et0_p95_irrigated_crop_area": round(pctile(et0, 95), 4),
            "kc_mean_irrigated_crop_area": round(wmean(kc_pixel), 4),
            "kc_p50_irrigated_crop_area": round(pctile(kc_pixel, 50), 4),
            "kc_p95_irrigated_crop_area": round(pctile(kc_pixel, 95), 4),
            "etc_mean_irrigated_crop_area": round(wmean(etc), 4),
            "etc_p95_irrigated_crop_area": round(pctile(etc, 95), 4),
            "active_irrigated_crop_area_percent": round(area_pct(kc_pixel > 0), 2),
            "high_kc_irrigated_crop_area_percent": round(area_pct(kc_pixel >= 0.8), 2),
            "phenology_inactive_percent_irrigated_crop_area": round(
                area_pct(phenology_status == PHENOLOGY_INACTIVE), 2
            ),
            "phenology_growing_percent_irrigated_crop_area": round(
                area_pct(phenology_status == PHENOLOGY_GROWING), 2
            ),
            "phenology_maximum_percent_irrigated_crop_area": round(
                area_pct(phenology_status == PHENOLOGY_MAXIMUM), 2
            ),
            "phenology_senescence_percent_irrigated_crop_area": round(
                area_pct(phenology_status == PHENOLOGY_SENESCENCE), 2
            ),
        }
        day_records.append(row)
        current += timedelta(days=1)

    if missing_precip:
        log_lines.append(
            f"WARNING: Missing precipitation for {len(missing_precip)} dates: "
            f"{missing_precip[:5]}" + ("..." if len(missing_precip) > 5 else "")
        )
    if missing_et0:
        log_lines.append(
            f"WARNING: Missing ET0 for {len(missing_et0)} dates: "
            f"{missing_et0[:5]}" + ("..." if len(missing_et0) > 5 else "")
        )

    forcing_df = pd.DataFrame(day_records)
    path = out_folder / "forcing_and_demand_diagnostics.csv"
    forcing_df.to_csv(path, index=False)
    log_lines.append(f"Written: {path}")
    return forcing_df


# ---------------------------------------------------------------------------
# Output 5: annual_forcing_and_demand_summary.csv
# ---------------------------------------------------------------------------

def compute_annual_forcing_summary(forcing_df, monthly_iwr_df, out_folder, log_lines):
    if forcing_df.empty:
        log_lines.append("WARNING: Forcing diagnostics empty, skipping annual summary.")
        return pd.DataFrame()

    annual_rows = []
    for year, grp in forcing_df.groupby("year"):
        monthly_etc = grp.groupby("month")["etc_mean_irrigated_crop_area"].sum()
        month_max_etc = int(monthly_etc.idxmax()) if not monthly_etc.empty else np.nan

        month_max_iwr = np.nan
        if monthly_iwr_df is not None and not monthly_iwr_df.empty:
            yr_iwr = monthly_iwr_df[monthly_iwr_df["year"] == year]
            if not yr_iwr.empty:
                month_max_iwr = int(
                    yr_iwr.loc[
                        yr_iwr["monthly_iwr_mcm_cropped_area"].idxmax(), "month"
                    ]
                )

        row = {
            "year": year,
            "annual_precipitation_mean_mm": round(
                grp["precipitation_mean_irrigated_crop_area"].sum(), 2
            ),
            "annual_p_eff_mean_mm": round(
                grp["p_eff_mean_irrigated_crop_area"].sum(), 2
            ),
            "annual_et0_mean_mm": round(
                grp["et0_mean_irrigated_crop_area"].sum(), 2
            ),
            "annual_etc_mean_mm": round(
                grp["etc_mean_irrigated_crop_area"].sum(), 2
            ),
            "mean_active_irrigated_crop_area_percent": round(
                grp["active_irrigated_crop_area_percent"].mean(), 2
            ),
            "days_with_active_area_gt_10_percent": int(
                (grp["active_irrigated_crop_area_percent"] > 10).sum()
            ),
            "days_with_active_area_gt_50_percent": int(
                (grp["active_irrigated_crop_area_percent"] > 50).sum()
            ),
            "days_with_kc_p95_gt_0_8": int(
                (grp["kc_p95_irrigated_crop_area"] > 0.8).sum()
            ),
            "month_of_max_etc": month_max_etc,
            "month_of_max_iwr": month_max_iwr,
        }
        annual_rows.append(row)

    annual_df = pd.DataFrame(annual_rows)
    path = out_folder / "annual_forcing_and_demand_summary.csv"
    annual_df.to_csv(path, index=False)
    log_lines.append(f"Written: {path}")
    return annual_df


# ---------------------------------------------------------------------------
# Output 6: debug_csv_summary.csv
# ---------------------------------------------------------------------------

def compute_debug_csv_summary(iwr_output_folder, out_folder, log_lines):
    debug_csv = Path(iwr_output_folder) / "iwr_debug_daily_stats.csv"
    if not debug_csv.exists():
        log_lines.append(
            f"WARNING: Debug CSV not found at {debug_csv}. Skipping debug_csv_summary."
        )
        return None

    log_lines.append(f"Debug CSV found: {debug_csv}")
    df = pd.read_csv(debug_csv)

    # Expected columns: (friendly_name, csv_column, aggregation)
    # aggregation: 'mean_max' or 'mean_only' or 'year_end'
    col_specs = [
        ("kc_pixel_p95", "kc_pixel_p95", "mean_max"),
        ("potential_evapotranspiration_p95", "potential_evapotranspiration_p95", "mean_max"),
        ("green_water_stress_coefficient_p50", "green_water_stress_coefficient_p50", "mean_only"),
        ("green_water_stress_coefficient_p95", "green_water_stress_coefficient_p95", "mean_only"),
        ("blue_iwr_watneeds_p95", "blue_iwr_watneeds_p95", "mean_max"),
        ("irrigation_p95", "irrigation_p95", "mean_max"),
        ("cumulative_irrigation_p95", "cumulative_irrigation_p95", "year_end"),
        ("soil_moisture_p50", "soil_moisture_p50", "mean_only"),
        ("soil_moisture_p95", "soil_moisture_p95", "mean_only"),
        ("runoff_p95", "runoff_p95", "mean_max"),
        ("deep_percolation_p95", "deep_percolation_p95", "mean_max"),
    ]

    missing_cols = [col for _, col, _ in col_specs if col not in df.columns]
    if missing_cols:
        log_lines.append(f"WARNING: Missing columns in debug CSV: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    rows = []
    for year, grp in df.groupby("year"):
        row = {"year": year}
        for friendly, col, agg in col_specs:
            if col not in df.columns:
                if agg == "year_end":
                    row[f"{friendly}_year_end"] = np.nan
                elif agg == "mean_max":
                    row[f"{friendly}_mean"] = np.nan
                    row[f"{friendly}_max"] = np.nan
                else:
                    row[f"{friendly}_mean"] = np.nan
            else:
                vals = pd.to_numeric(grp[col], errors="coerce")
                if agg == "year_end":
                    row[f"{friendly}_year_end"] = (
                        float(vals.iloc[-1]) if not vals.empty else np.nan
                    )
                elif agg == "mean_max":
                    row[f"{friendly}_mean"] = float(vals.mean())
                    row[f"{friendly}_max"] = float(vals.max())
                else:
                    row[f"{friendly}_mean"] = float(vals.mean())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    path = out_folder / "debug_csv_summary.csv"
    summary_df.to_csv(path, index=False)
    log_lines.append(f"Written: {path}")
    return summary_df


# ---------------------------------------------------------------------------
# Output 7: Plots
# ---------------------------------------------------------------------------

def make_plots(
    annual_df,
    monthly_df,
    forcing_df,
    day_iwr_df,
    area_stats,
    aida_mcm,
    out_folder,
    log_lines,
):
    plt.rcParams.update({"figure.autolayout": True, "font.size": 10})

    # A. annual_iwr_vs_aida.png
    if not annual_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        years = annual_df["year"].astype(str)
        ax.bar(
            years,
            annual_df["annual_iwr_mcm_cropped_area"],
            color="steelblue",
            label="Model IWR (MCM)",
        )
        if aida_mcm is not None:
            ax.axhline(
                aida_mcm,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"AIDA {aida_mcm:.1f} MCM",
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Volume (MCM)")
        ax.set_title("Annual IWR vs AIDA reference")
        ax.legend()
        fig.savefig(out_folder / "annual_iwr_vs_aida.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: annual_iwr_vs_aida.png")

    # B. monthly_iwr_by_year.png
    if not monthly_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for year, grp in monthly_df.groupby("year"):
            ax.plot(
                grp["month"],
                grp["monthly_iwr_mcm_cropped_area"],
                marker="o",
                markersize=4,
                label=str(year),
            )
        ax.set_xlabel("Month")
        ax.set_ylabel("Volume (MCM)")
        ax.set_title("Monthly IWR (MCM, cropped area) by year")
        ax.set_xticks(range(1, 13))
        ax.legend()
        fig.savefig(out_folder / "monthly_iwr_by_year.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: monthly_iwr_by_year.png")

    # C. annual_mean_depth_vs_aida_equivalent.png
    if not annual_df.empty and "mean_iwr_depth_mm_over_irrigated_crop_area" in annual_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        years = annual_df["year"].astype(str)
        ax.bar(
            years,
            annual_df["mean_iwr_depth_mm_over_irrigated_crop_area"],
            color="steelblue",
            label="Model mean IWR depth (mm)",
        )
        if (
            aida_mcm is not None
            and "aida_equivalent_depth_mm_over_irrigated_crop_area" in annual_df.columns
        ):
            aida_depth = float(
                annual_df["aida_equivalent_depth_mm_over_irrigated_crop_area"].iloc[0]
            )
            if np.isfinite(aida_depth):
                ax.axhline(
                    aida_depth,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"AIDA equiv. depth {aida_depth:.0f} mm",
                )
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean IWR depth (mm/year)")
        ax.set_title("Annual mean IWR depth over irrigated crop area vs AIDA")
        ax.legend()
        fig.savefig(out_folder / "annual_mean_depth_vs_aida_equivalent.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: annual_mean_depth_vs_aida_equivalent.png")

    # D. monthly_etc_and_iwr.png
    if not forcing_df.empty and not monthly_df.empty:
        monthly_etc = (
            forcing_df.groupby(["year", "month"])["etc_mean_irrigated_crop_area"]
            .sum()
            .reset_index()
            .rename(columns={"etc_mean_irrigated_crop_area": "monthly_etc_mm"})
        )
        avg_etc = monthly_etc.groupby("month")["monthly_etc_mm"].mean()
        avg_iwr = monthly_df.groupby("month")["monthly_iwr_depth_mm_over_irrigated_crop_area"].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.35
        ax.bar(
            avg_etc.index - w / 2,
            avg_etc.values,
            width=w,
            label="Mean ETc (mm/month)",
            color="orange",
        )
        ax.bar(
            avg_iwr.index + w / 2,
            avg_iwr.values,
            width=w,
            label="Mean IWR depth (mm/month)",
            color="steelblue",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("mm/month")
        ax.set_title("Monthly mean ETc and IWR depth (average across years)")
        ax.set_xticks(range(1, 13))
        ax.legend()
        fig.savefig(out_folder / "monthly_etc_and_iwr.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: monthly_etc_and_iwr.png")

    # E. active_area_by_month.png
    if not forcing_df.empty:
        monthly_active = (
            forcing_df.groupby("month")["active_irrigated_crop_area_percent"].mean()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(monthly_active.index, monthly_active.values, color="seagreen")
        ax.set_xlabel("Month")
        ax.set_ylabel("Active irrigated crop area (%)")
        ax.set_title("Mean active irrigated crop area by month")
        ax.set_xticks(range(1, 13))
        ax.set_ylim(0, 100)
        fig.savefig(out_folder / "active_area_by_month.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: active_area_by_month.png")

    # F. kc_p95_by_date.png
    if not forcing_df.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        dates = pd.to_datetime(forcing_df["date"])
        ax.plot(
            dates,
            forcing_df["kc_p95_irrigated_crop_area"],
            linewidth=0.8,
            color="darkorange",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Kc p95 (irrigated crop area)")
        ax.set_title("Daily Kc p95 over irrigated crop area")
        fig.savefig(out_folder / "kc_p95_by_date.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: kc_p95_by_date.png")

    # G. et0_etc_iwr_daily.png
    if not forcing_df.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(forcing_df["date"])
        ax.plot(
            dates,
            forcing_df["et0_mean_irrigated_crop_area"],
            linewidth=0.7,
            label="ET0 mean (mm/day)",
            alpha=0.85,
            color="royalblue",
        )
        ax.plot(
            dates,
            forcing_df["etc_mean_irrigated_crop_area"],
            linewidth=0.7,
            label="ETc mean (mm/day)",
            alpha=0.85,
            color="darkorange",
        )
        # IWR daily depth if available
        if day_iwr_df is not None and not day_iwr_df.empty:
            iwr_dates = pd.to_datetime(day_iwr_df["date"])
            ax.plot(
                iwr_dates,
                day_iwr_df["iwr_depth_mm_over_irrigated_crop_area"],
                linewidth=0.7,
                label="IWR depth (mm/day, over irr. crop area)",
                alpha=0.85,
                color="steelblue",
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("mm/day")
        ax.set_title("Daily ET0, ETc and IWR depth over irrigated crop area")
        ax.legend(fontsize=8)
        fig.savefig(out_folder / "et0_etc_iwr_daily.png", dpi=150)
        plt.close(fig)
        log_lines.append("Written: et0_etc_iwr_daily.png")


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def write_log(log_lines, out_folder):
    log_path = out_folder / "diagnostics_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(str(line) for line in log_lines) + "\n")
    print(f"Log written: {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Non-invasive diagnostics for the IWR benchmark model outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--aida-mcm",
        type=float,
        default=None,
        dest="aida_mcm",
        help="AIDA reference annual volume in million m3/year",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output diagnostic folder (default: <iwr_output_folder>/diagnostics_benchmark)",
    )
    args = parser.parse_args()

    log_lines = []
    log_lines.append("=== IWR Benchmark Diagnostics ===")
    log_lines.append(f"Config path: {Path(args.config).resolve()}")
    log_lines.append(f"AIDA MCM provided: {args.aida_mcm}")

    # ---- Load config ----
    cfg, config_path = load_config(args.config)
    config_dir = config_path.parent

    def rp(key):
        return resolve_path(cfg[key], config_dir)

    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(cfg["end_date"], "%Y-%m-%d").date()
    log_lines.append(f"Date range: {start_date} to {end_date}")

    iwr_output_folder = rp("iwr_output_folder")
    log_lines.append(f"IWR output folder: {iwr_output_folder}")

    # ---- Output folder ----
    if args.out:
        out_folder = Path(args.out)
    else:
        out_folder = iwr_output_folder / "diagnostics_benchmark"
    out_folder.mkdir(parents=True, exist_ok=True)
    log_lines.append(f"Diagnostics output folder: {out_folder}")

    # ---- Find daily IWR files ----
    iwr_files = find_daily_iwr_files(iwr_output_folder)
    log_lines.append(f"Daily IWR files found: {len(iwr_files)}")

    # Check missing dates in expected range
    expected_dates = []
    cur = start_date
    while cur <= end_date:
        expected_dates.append(cur)
        cur += timedelta(days=1)
    missing_dates = [str(d) for d in expected_dates if d not in iwr_files]
    if missing_dates:
        log_lines.append(
            f"WARNING: Missing IWR output for {len(missing_dates)} dates: "
            f"{missing_dates[:10]}" + ("..." if len(missing_dates) > 10 else "")
        )
    else:
        log_lines.append("No missing IWR output dates.")

    # ---- Load static data ----
    print("Loading static data...", flush=True)

    irrigation_mask_path = rp("irrigated_areas_path")
    crop_fraction_path = rp("crop_fraction_path")
    crop_parameters_csv = rp("crop_parameters_csv")
    precip_folder = rp("precipitation_geotiff_folder")
    et0_folder = rp("et0_geotiff_folder")

    # Read irrigation mask and get profile/CRS
    with rasterio.open(irrigation_mask_path) as src:
        irrigation_mask = src.read(1)
        output_profile = src.profile.copy()
        crs = src.crs

    log_lines.append(f"CRS: {crs}")
    pixel_area_m2 = compute_pixel_area_m2(output_profile)
    log_lines.append(f"Pixel area: {pixel_area_m2:.2f} m2")

    # Read and prepare crop fractions (consistent with model)
    crop_df, crop_fraction_data_raw, crop_profile, band_descriptions = check_crop_raster_and_csv(
        crop_fraction_path=crop_fraction_path,
        crop_parameters_csv=crop_parameters_csv,
    )
    crop_fraction_data_prepared, crop_fraction_sum = prepare_crop_fractions(crop_fraction_data_raw)
    log_lines.append(
        f"Crops: {len(crop_df)} ({', '.join(crop_df['crop_name'].tolist())})"
    )

    # Load phenology layers
    phenology_paths_cfg = {
        name: resolve_path(path, config_dir)
        for name, path in cfg["phenology_paths"].items()
    }
    phenology = load_phenology_layers(phenology_paths_cfg)
    log_lines.append(f"Phenology layers loaded: {list(phenology.keys())}")

    # ---- Output 1: Area diagnostics ----
    print("Computing area diagnostics...", flush=True)
    area_stats = compute_area_diagnostics(
        crop_fraction_data_prepared=crop_fraction_data_prepared,
        crop_fraction_sum=crop_fraction_sum,
        irrigation_mask=irrigation_mask,
        pixel_area_m2=pixel_area_m2,
        crop_df=crop_df,
        band_descriptions=band_descriptions,
        out_folder=out_folder,
        log_lines=log_lines,
    )
    log_lines.append(
        f"Irrigated crop area: {area_stats['irrigated_crop_area_km2']:.2f} km2 "
        f"({area_stats['irrigated_crop_area_m2']:.0f} m2)"
    )
    log_lines.append(f"Irrigated mask area: {area_stats['irrigated_mask_area_km2']:.2f} km2")

    # ---- Outputs 2 & 3: IWR volume diagnostics ----
    print("Computing IWR volume diagnostics...", flush=True)
    annual_df, monthly_df, day_iwr_df = compute_iwr_volume_diagnostics(
        iwr_files=iwr_files,
        crop_fraction_sum=crop_fraction_sum,
        irrigation_mask=irrigation_mask,
        pixel_area_m2=pixel_area_m2,
        area_stats=area_stats,
        aida_mcm=args.aida_mcm,
        out_folder=out_folder,
        log_lines=log_lines,
    )

    # ---- Output 4: Forcing and demand diagnostics ----
    print("Computing forcing and demand diagnostics (looping over all days)...", flush=True)
    forcing_df = compute_forcing_and_demand_diagnostics(
        start_date=start_date,
        end_date=end_date,
        precip_folder=precip_folder,
        et0_folder=et0_folder,
        crop_fraction_data_prepared=crop_fraction_data_prepared,
        crop_fraction_sum=crop_fraction_sum,
        irrigation_mask=irrigation_mask,
        pixel_area_m2=pixel_area_m2,
        phenology=phenology,
        crop_df=crop_df,
        out_folder=out_folder,
        log_lines=log_lines,
    )

    # ---- Output 5: Annual forcing summary ----
    print("Aggregating annual forcing summary...", flush=True)
    annual_forcing_df = compute_annual_forcing_summary(
        forcing_df=forcing_df,
        monthly_iwr_df=monthly_df if not monthly_df.empty else None,
        out_folder=out_folder,
        log_lines=log_lines,
    )

    # ---- Output 6: Debug CSV summary ----
    print("Summarising debug CSV...", flush=True)
    debug_csv_path = iwr_output_folder / "iwr_debug_daily_stats.csv"
    log_lines.append(f"Debug CSV present: {debug_csv_path.exists()}")
    compute_debug_csv_summary(
        iwr_output_folder=iwr_output_folder,
        out_folder=out_folder,
        log_lines=log_lines,
    )

    # ---- Output 7: Plots ----
    print("Creating plots...", flush=True)
    make_plots(
        annual_df=annual_df,
        monthly_df=monthly_df,
        forcing_df=forcing_df,
        day_iwr_df=day_iwr_df,
        area_stats=area_stats,
        aida_mcm=args.aida_mcm,
        out_folder=out_folder,
        log_lines=log_lines,
    )

    # ---- Final summary in log ----
    log_lines.append("\n=== FINAL SUMMARY ===")
    log_lines.append(f"Irrigated crop area: {area_stats['irrigated_crop_area_km2']:.2f} km2")
    if not annual_df.empty:
        for _, row in annual_df.iterrows():
            year = int(row["year"])
            mcm = float(row["annual_iwr_mcm_cropped_area"])
            depth = float(row.get("mean_iwr_depth_mm_over_irrigated_crop_area", np.nan))
            log_lines.append(
                f"  Year {year}: model MCM = {mcm:.3f}, "
                f"mean IWR depth = {depth:.1f} mm/year"
            )
            if args.aida_mcm is not None:
                ratio = float(row.get("ratio_model_to_aida", np.nan))
                missing = float(row.get("missing_mcm_to_aida", np.nan))
                aida_depth = float(
                    row.get("aida_equivalent_depth_mm_over_irrigated_crop_area", np.nan)
                )
                log_lines.append(
                    f"    AIDA MCM = {args.aida_mcm:.3f}, "
                    f"model/AIDA ratio = {ratio:.4f}, "
                    f"missing = {missing:.3f} MCM"
                )
                if np.isfinite(aida_depth):
                    log_lines.append(
                        f"    AIDA equiv. depth over irrigated crop area = {aida_depth:.1f} mm/year"
                    )

    # ---- Write log ----
    write_log(log_lines, out_folder)

    # ---- Output 9: Print sharing summary ----
    print("\nDiagnostics completed.")
    print("Please share:")
    for fname in [
        "annual_iwr_volume_diagnostics.csv",
        "annual_forcing_and_demand_summary.csv",
        "area_diagnostics.csv",
        "crop_area_by_band.csv",
        "diagnostics_log.txt",
        "annual_iwr_vs_aida.png",
        "monthly_iwr_by_year.png",
        "active_area_by_month.png",
    ]:
        fpath = out_folder / fname
        status = "(present)" if fpath.exists() else "(not created)"
        print(f"  - {fpath}  {status}")


if __name__ == "__main__":
    main()
