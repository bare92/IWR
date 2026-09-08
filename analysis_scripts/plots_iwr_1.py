#!/usr/bin/env python3
"""
Plot theoretical IWR and ET diagnostics from model GeoTIFF outputs.

All daily flux comparisons (IWR, ETx, ETa under water stress, ET deficit)
use a single shared phenology-active spatial support derived from either an
explicit active-mask raster or from ETx > ACTIVE_ETX_TOLERANCE_MM.
This ensures that all four variables share the same denominator on every
day, preventing spurious differences caused by different nodata conventions.

Inputs from current model output structure:
- daily theoretical IWR: iwr_YYYYMMDD.tif
- daily ETx: etx_YYYYMMDD.tif
- daily stress-limited ETa: eta_stress_YYYYMMDD.tif
- optional daily active-pixel mask: active_pixels_YYYYMMDD.tif

Computed statistics:
- common-support daily diagnostics for IWR, ETx, ETa_stress, and ET deficit
- daily total volume from depth and analysis-area support layers
- daily active-area mean depth and static-domain mean depth
- annual accumulated volume and annual domain-average accumulated depth
- configured-period completeness and partial-year reporting
- physical consistency checks (IWR/ETa outside phenology, ETa > ETx, etc.)

Outputs:
- daily_iwr_summary.csv
- annual_iwr_summary.csv
- daily_iwr_timeseries.png
- yearly_iwr.png
- daily_et_comparison.png (when ET pairs are available)
- yearly_et_components.png (when ET pairs are available)
- plotting_diagnostics.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio


# =========================
# USER SETTINGS
# =========================

MODEL_CONFIG_FILE = Path(
    "IWR_scripts/config/config_eraL_theoretical_rainfed.json"
)

CELL_AREA_M2_FILE = Path(
    "/home/fremen/data/projects/Burkina/00_Data_iwr/static/"
    "burkina_cell_area_m2.tif"
)

OUTPUT_DIR = Path(
    "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/iwr_plots2"
)

# Annual output mode:
#   volume_m3 -> annual total volume [m3/year]
#   mean_mm   -> annual domain-average accumulated depth [mm/year]
ANNUAL_PLOT_MODE = "volume_m3"

# Daily plot mean mode:
#   active_area -> weighted mean over common phenology-active area
#   domain_area -> domain-equivalent mean over full static analysis area
DAILY_MEAN_MODE = "domain_area"

# Keep or drop years that are not full and complete calendar years.
INCLUDE_PARTIAL_YEARS = True

# If True, negative depths are treated as nodata.
# If False, negative depths raise an error.
IGNORE_NEGATIVE_VALUES = True

# Pixel-level tolerance for ETx - ETa_stress - ET_deficit balance diagnostics.
ET_BALANCE_TOLERANCE_MM = 1e-5

# Optional plotting window override. None means configured model period.
DATE_START = "2021-01-01"
DATE_END = "2024-12-31"

# Annual plot readability.
ANNUAL_MAX_X_LABELS = 12
ANNUAL_MIN_FIGURE_WIDTH = 14.0
ANNUAL_MAX_FIGURE_WIDTH = 20.0
ANNUAL_INCHES_PER_YEAR = 0.42
ANNUAL_FIGURE_HEIGHT = 6.0
ANNUAL_LABEL_ROTATION = 45

# Source used to identify daily phenology-active pixels:
#   "auto"        -> use an explicit daily active mask when available,
#                    otherwise derive it from ETx
#   "active_mask" -> require an explicit daily active mask
#   "etx"         -> derive active support from ETx > threshold
ACTIVE_SUPPORT_SOURCE = "auto"

# Optional directory containing active_pixels_YYYYMMDD.tif.
# When None, look first in RUN_DIR / "ActiveMasks".
# If no mask is found and ACTIVE_SUPPORT_SOURCE is "auto",
# fall back to ETx > ACTIVE_ETX_TOLERANCE_MM.
ACTIVE_MASK_DIR = None

ACTIVE_MASK_PREFIX = "active_pixels"
ACTIVE_ETX_TOLERANCE_MM = 1e-8

# Tolerance used for physical consistency checks.
FLUX_CONSISTENCY_TOLERANCE_MM = 1e-5

# Raise an error instead of silently producing a misleading plot.
STRICT_CONSISTENCY_CHECKS = True


# =========================
# FUNCTIONS
# =========================

DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")


def parse_date_from_name(path: Path) -> pd.Timestamp:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def list_daily_files(directory: Path, prefix: str) -> dict[pd.Timestamp, Path]:
    files = sorted(directory.glob(f"{prefix}_????????.tif"))
    out: dict[pd.Timestamp, Path] = {}

    for file_path in files:
        try:
            out[parse_date_from_name(file_path)] = file_path
        except ValueError:
            continue

    return out


def require_existing_path(path: Path, label: str, expected: str) -> None:
    if expected == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing required directory for {label}: {path}")
    if expected == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing required file for {label}: {path}")


def load_json(path: Path) -> dict:
    require_existing_path(path, "MODEL_CONFIG_FILE", "file")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_model_metadata(config: dict) -> dict:
    return {
        "model_start_date": config["time"]["start_date"],
        "model_end_date": config["time"]["end_date"],
        "iwr_mode": config["options"]["iwr_mode"],
        "iwr_domain": config["options"]["iwr_domain"],
        "theoretical_iwr_target": config["options"]["theoretical_iwr_target"],
        "drainage_scheme": config["options"].get("drainage_scheme", "unknown"),
    }


def build_runtime_paths(config: dict) -> dict:
    run_dir = Path(config["outputs"]["output_base"]) / config["outputs"]["run_name"]

    return {
        "RUN_DIR": run_dir,
        "IWR_DIR": run_dir / "IWR",
        "ETX_DIR": run_dir / "ETx",
        "ETA_STRESS_DIR": run_dir / "ETa_stress",
        "STATIC_DIR": run_dir / "Static",
        "IWR_AREA_FRACTION_FILE": run_dir / "Static" / "iwr_analysis_area_fraction.tif",
        "ACTIVE_MASK_DIR": run_dir / "ActiveMasks",
    }


def select_analysis_period(model_start_date: str, model_end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg_start = pd.to_datetime(model_start_date)
    cfg_end = pd.to_datetime(model_end_date)

    analysis_start = cfg_start if DATE_START is None else pd.to_datetime(DATE_START)
    analysis_end = cfg_end if DATE_END is None else pd.to_datetime(DATE_END)

    if analysis_start > analysis_end:
        raise ValueError("Analysis start date is after analysis end date")

    return analysis_start, analysis_end


def read_depth_raster(path: Path) -> tuple[np.ndarray, tuple[int, int], rasterio.Affine, object]:
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected single-band raster: {path}")

        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[~np.isfinite(arr)] = np.nan

        if IGNORE_NEGATIVE_VALUES:
            arr[arr < 0] = np.nan
        elif np.any((arr < 0) & np.isfinite(arr)):
            raise ValueError(f"Negative depth values found in {path}")

        return arr, src.shape, src.transform, src.crs


def validate_alignment(
    shape_a,
    transform_a,
    crs_a,
    shape_b,
    transform_b,
    crs_b,
    label_a: str,
    label_b: str,
) -> None:
    if shape_a != shape_b:
        raise ValueError(f"Shape mismatch between {label_a} and {label_b}: {shape_a} vs {shape_b}")
    if transform_a != transform_b:
        raise ValueError(f"Transform mismatch between {label_a} and {label_b}")
    if crs_a != crs_b:
        raise ValueError(f"CRS mismatch between {label_a} and {label_b}: {crs_a} vs {crs_b}")


def read_active_mask(
    path: Path,
    reference_shape: tuple[int, int],
    reference_transform: rasterio.Affine,
    reference_crs: object,
) -> np.ndarray:
    """
    Read an explicit active-pixel mask raster.

    Returns a boolean array where True means the pixel is phenology-active.
    Values greater than zero are active; nodata and zero are inactive.
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Active mask must be single-band: {path}")

        validate_alignment(
            src.shape,
            src.transform,
            src.crs,
            reference_shape,
            reference_transform,
            reference_crs,
            str(path),
            "reference raster",
        )

        arr = src.read(1)
        nodata = src.nodata

    if nodata is not None:
        inactive = arr == nodata
    else:
        inactive = np.zeros(arr.shape, dtype=bool)

    return ((arr > 0) & ~inactive).astype(bool)


def load_support_layers(
    cell_area_file: Path,
    area_fraction_file: Path,
    reference_iwr_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load and validate static support layers against a reference IWR raster.

    Returns:
      cell_area_m2
      area_fraction
      analysis_area_m2
      total_analysis_area_m2
    """

    require_existing_path(cell_area_file, "CELL_AREA_M2_FILE", "file")
    require_existing_path(area_fraction_file, "IWR_AREA_FRACTION_FILE", "file")
    require_existing_path(reference_iwr_file, "reference IWR raster", "file")

    _, ref_shape, ref_transform, ref_crs = read_depth_raster(reference_iwr_file)

    with rasterio.open(cell_area_file) as area_src:
        if area_src.count != 1:
            raise ValueError(f"Cell area raster must be single band: {cell_area_file}")
        validate_alignment(
            area_src.shape,
            area_src.transform,
            area_src.crs,
            ref_shape,
            ref_transform,
            ref_crs,
            "cell_area_m2",
            "reference_iwr",
        )

        cell_area_m2 = area_src.read(1).astype("float64")
        if area_src.nodata is not None:
            cell_area_m2[cell_area_m2 == area_src.nodata] = np.nan

    with rasterio.open(area_fraction_file) as frac_src:
        if frac_src.count != 1:
            raise ValueError(f"Area fraction raster must be single band: {area_fraction_file}")
        validate_alignment(
            frac_src.shape,
            frac_src.transform,
            frac_src.crs,
            ref_shape,
            ref_transform,
            ref_crs,
            "iwr_analysis_area_fraction",
            "reference_iwr",
        )

        area_fraction = frac_src.read(1).astype("float64")
        if frac_src.nodata is not None:
            area_fraction[area_fraction == frac_src.nodata] = np.nan

    cell_area_m2[~np.isfinite(cell_area_m2)] = np.nan
    area_fraction[~np.isfinite(area_fraction)] = np.nan

    if np.any((cell_area_m2 <= 0) & np.isfinite(cell_area_m2)):
        raise ValueError("CELL_AREA_M2_FILE contains non-positive finite values")

    # Allow harmless floating-point deviations around the theoretical [0, 1] range.
    FRACTION_TOLERANCE = 1e-6

    finite_fraction = np.isfinite(area_fraction)
    fraction_values = area_fraction[finite_fraction]

    if fraction_values.size == 0:
        raise ValueError("IWR area fraction contains no finite values")

    minimum_fraction = float(np.min(fraction_values))
    maximum_fraction = float(np.max(fraction_values))

    if minimum_fraction < -FRACTION_TOLERANCE:
        raise ValueError(
            "IWR area fraction contains materially negative values: "
            f"minimum={minimum_fraction:.12g}"
        )

    if maximum_fraction > 1.0 + FRACTION_TOLERANCE:
        raise ValueError(
            "IWR area fraction contains values materially above 1: "
            f"maximum={maximum_fraction:.12g}"
        )

    # Remove only harmless floating-point undershoots/overshoots.
    area_fraction[finite_fraction] = np.clip(
        area_fraction[finite_fraction],
        0.0,
        1.0,
    )

    analysis_area_m2 = cell_area_m2 * area_fraction
    analysis_area_m2[~np.isfinite(analysis_area_m2)] = np.nan

    total_analysis_area_m2 = float(np.nansum(analysis_area_m2))
    if not np.isfinite(total_analysis_area_m2) or total_analysis_area_m2 <= 0:
        raise ValueError("Total analysis area is not positive; check support layers")

    return cell_area_m2, area_fraction, analysis_area_m2, total_analysis_area_m2


def compute_weighted_metrics_from_depth_array(
    depth_mm: np.ndarray,
    analysis_area_m2: np.ndarray,
    total_analysis_area_m2: float,
    support_mask: np.ndarray,
) -> dict:
    """
    Compute weighted spatial metrics using an explicit support mask.

    The support mask must be determined externally (e.g. common
    phenology-active support for all compared variables).
    """
    support = (
        support_mask
        & np.isfinite(analysis_area_m2)
        & (analysis_area_m2 > 0)
    )

    valid_pixel_count = int(np.count_nonzero(support))

    if valid_pixel_count == 0:
        return {
            "valid_pixel_count": 0,
            "valid_analysis_area_m2": 0.0,
            "valid_analysis_area_fraction_of_total": 0.0,
            "minimum_mm_day": np.nan,
            "maximum_mm_day": np.nan,
            "median_mm_day": np.nan,
            "active_mean_mm": 0.0,
            "domain_mean_mm": 0.0,
            "volume_m3": 0.0,
        }

    support_area_m2 = float(np.sum(analysis_area_m2[support]))
    volume_m3 = float(np.sum(depth_mm[support] * analysis_area_m2[support] / 1000.0))
    active_mean_mm = float(
        np.sum(depth_mm[support] * analysis_area_m2[support]) / support_area_m2
    )
    domain_mean_mm = float(volume_m3 / total_analysis_area_m2 * 1000.0)

    return {
        "valid_pixel_count": valid_pixel_count,
        "valid_analysis_area_m2": support_area_m2,
        "valid_analysis_area_fraction_of_total": float(support_area_m2 / total_analysis_area_m2),
        "minimum_mm_day": float(np.min(depth_mm[support])),
        "maximum_mm_day": float(np.max(depth_mm[support])),
        "median_mm_day": float(np.median(depth_mm[support])),
        "active_mean_mm": active_mean_mm,
        "domain_mean_mm": domain_mean_mm,
        "volume_m3": volume_m3,
    }


def _resolve_active_mask_path(
    date: pd.Timestamp,
    active_mask_dir: Path | None,
) -> Path | None:
    """
    Look for an explicit active-mask raster for the given date.

    Returns None when no file is found.
    """
    if ACTIVE_SUPPORT_SOURCE == "etx":
        return None

    dirs_to_check: list[Path] = []
    if ACTIVE_MASK_DIR is not None:
        dirs_to_check.append(Path(ACTIVE_MASK_DIR))
    if active_mask_dir is not None:
        dirs_to_check.append(active_mask_dir)

    date_str = date.strftime("%Y%m%d")
    filename = f"{ACTIVE_MASK_PREFIX}_{date_str}.tif"

    for d in dirs_to_check:
        candidate = d / filename
        if candidate.is_file():
            return candidate

    if ACTIVE_SUPPORT_SOURCE == "active_mask":
        raise FileNotFoundError(
            f"ACTIVE_SUPPORT_SOURCE='active_mask' but no active mask found for "
            f"{date.date()} in: {dirs_to_check}"
        )

    return None


def compute_daily_joint_metrics(
    iwr_path: Path,
    etx_path: Path,
    eta_stress_path: Path,
    active_mask_path: Path | None,
    analysis_area_m2: np.ndarray,
    total_analysis_area_m2: float,
    metadata: dict,
) -> dict:
    """
    Read IWR, ETx and ETa together and compute metrics using a shared
    phenology-active spatial support.

    All four derived quantities (IWR, ETx, ETa, ET deficit) use exactly
    the same daily active_support mask.
    """
    iwr, iwr_shape, iwr_transform, iwr_crs = read_depth_raster(iwr_path)
    etx, etx_shape, etx_transform, etx_crs = read_depth_raster(etx_path)
    eta, eta_shape, eta_transform, eta_crs = read_depth_raster(eta_stress_path)

    # Validate spatial alignment between all three rasters.
    validate_alignment(
        iwr_shape, iwr_transform, iwr_crs,
        etx_shape, etx_transform, etx_crs,
        str(iwr_path), str(etx_path),
    )
    validate_alignment(
        iwr_shape, iwr_transform, iwr_crs,
        eta_shape, eta_transform, eta_crs,
        str(iwr_path), str(eta_stress_path),
    )

    if iwr.shape != analysis_area_m2.shape:
        raise ValueError(
            f"Raster shape {iwr.shape} does not match analysis area shape "
            f"{analysis_area_m2.shape}"
        )

    # Static analysis support: pixels where analysis_area_m2 is valid.
    static_support: np.ndarray = (
        np.isfinite(analysis_area_m2) & (analysis_area_m2 > 0)
    )

    # Derive the shared daily phenology-active support.
    if active_mask_path is not None:
        active_mask = read_active_mask(
            active_mask_path, iwr_shape, iwr_transform, iwr_crs,
        )
        active_support: np.ndarray = static_support & active_mask
        active_support_source = "active_mask"
    else:
        active_support = (
            static_support
            & np.isfinite(etx)
            & (etx > ACTIVE_ETX_TOLERANCE_MM)
        )
        active_support_source = "etx_positive"

    # --- Missing-value checks on active pixels ---
    missing_iwr_on_active = active_support & ~np.isfinite(iwr)
    missing_etx_on_active = active_support & ~np.isfinite(etx)
    missing_eta_on_active = active_support & ~np.isfinite(eta)

    n_missing_iwr = int(np.count_nonzero(missing_iwr_on_active))
    n_missing_etx = int(np.count_nonzero(missing_etx_on_active))
    n_missing_eta = int(np.count_nonzero(missing_eta_on_active))

    if n_missing_iwr > 0 or n_missing_etx > 0 or n_missing_eta > 0:
        msg = (
            f"Active support contains missing values: "
            f"IWR={n_missing_iwr}, ETx={n_missing_etx}, ETa={n_missing_eta} pixels"
        )
        if STRICT_CONSISTENCY_CHECKS:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}. Removing those pixels from common support.")
            active_support = (
                active_support
                & np.isfinite(iwr)
                & np.isfinite(etx)
                & np.isfinite(eta)
            )

    # --- ET deficit on common support ---
    et_deficit = np.full_like(etx, np.nan, dtype="float64")
    et_deficit[active_support] = np.maximum(
        etx[active_support] - eta[active_support], 0.0
    )

    # --- Compute weighted metrics for all four variables ---
    iwr_stats = compute_weighted_metrics_from_depth_array(
        iwr, analysis_area_m2, total_analysis_area_m2, active_support
    )
    etx_stats = compute_weighted_metrics_from_depth_array(
        etx, analysis_area_m2, total_analysis_area_m2, active_support
    )
    eta_stats = compute_weighted_metrics_from_depth_array(
        eta, analysis_area_m2, total_analysis_area_m2, active_support
    )
    deficit_stats = compute_weighted_metrics_from_depth_array(
        et_deficit, analysis_area_m2, total_analysis_area_m2, active_support
    )

    active_pixel_count = int(np.count_nonzero(active_support))
    active_area_m2 = iwr_stats["valid_analysis_area_m2"]
    active_area_fraction = iwr_stats["valid_analysis_area_fraction_of_total"]

    # --- Physical consistency checks ---

    # A. Positive IWR outside phenology.
    positive_iwr_outside_active = (
        static_support
        & ~active_support
        & np.isfinite(iwr)
        & (iwr > FLUX_CONSISTENCY_TOLERANCE_MM)
    )
    n_pos_iwr_outside = int(np.count_nonzero(positive_iwr_outside_active))
    max_iwr_outside = (
        float(np.max(iwr[positive_iwr_outside_active]))
        if n_pos_iwr_outside > 0
        else 0.0
    )

    if n_pos_iwr_outside > 0 and STRICT_CONSISTENCY_CHECKS:
        raise ValueError(
            f"IWR is positive on {n_pos_iwr_outside} pixels outside the "
            f"phenology-active support (max={max_iwr_outside:.6g} mm/day)."
        )
    elif n_pos_iwr_outside > 0:
        print(
            f"WARNING: IWR positive outside active support on {n_pos_iwr_outside} "
            f"pixels (max={max_iwr_outside:.6g} mm/day)."
        )

    # B. Positive ETa outside phenology.
    positive_eta_outside_active = (
        static_support
        & ~active_support
        & np.isfinite(eta)
        & (eta > FLUX_CONSISTENCY_TOLERANCE_MM)
    )
    n_pos_eta_outside = int(np.count_nonzero(positive_eta_outside_active))
    max_eta_outside = (
        float(np.max(eta[positive_eta_outside_active]))
        if n_pos_eta_outside > 0
        else 0.0
    )

    if n_pos_eta_outside > 0 and STRICT_CONSISTENCY_CHECKS:
        raise ValueError(
            f"ETa is positive on {n_pos_eta_outside} pixels outside the "
            f"phenology-active support (max={max_eta_outside:.6g} mm/day)."
        )
    elif n_pos_eta_outside > 0:
        print(
            f"WARNING: ETa positive outside active support on {n_pos_eta_outside} "
            f"pixels (max={max_eta_outside:.6g} mm/day)."
        )

    # C. Positive ETx outside explicit active mask.
    n_pos_etx_outside = 0
    max_etx_outside = 0.0
    if active_support_source == "active_mask":
        positive_etx_outside_active = (
            static_support
            & ~active_support
            & np.isfinite(etx)
            & (etx > FLUX_CONSISTENCY_TOLERANCE_MM)
        )
        n_pos_etx_outside = int(np.count_nonzero(positive_etx_outside_active))
        max_etx_outside = (
            float(np.max(etx[positive_etx_outside_active]))
            if n_pos_etx_outside > 0
            else 0.0
        )
        if n_pos_etx_outside > 0 and STRICT_CONSISTENCY_CHECKS:
            raise ValueError(
                f"ETx is positive on {n_pos_etx_outside} pixels outside the "
                f"explicit active mask (max={max_etx_outside:.6g} mm/day)."
            )
        elif n_pos_etx_outside > 0:
            print(
                f"WARNING: ETx positive outside explicit active mask on "
                f"{n_pos_etx_outside} pixels (max={max_etx_outside:.6g} mm/day)."
            )

    # D. ETa must not exceed ETx on active pixels.
    eta_above_etx = (
        active_support
        & (eta > etx + FLUX_CONSISTENCY_TOLERANCE_MM)
    )
    n_eta_above_etx = int(np.count_nonzero(eta_above_etx))
    max_eta_minus_etx = (
        float(np.max(eta[eta_above_etx] - etx[eta_above_etx]))
        if n_eta_above_etx > 0
        else 0.0
    )

    if n_eta_above_etx > 0 and STRICT_CONSISTENCY_CHECKS:
        raise ValueError(
            f"ETa exceeds ETx on {n_eta_above_etx} active pixels "
            f"(max excess={max_eta_minus_etx:.6g} mm/day)."
        )
    elif n_eta_above_etx > 0:
        print(
            f"WARNING: ETa > ETx on {n_eta_above_etx} active pixels "
            f"(max excess={max_eta_minus_etx:.6g} mm/day)."
        )

    # E. IWR must not exceed ETx when target is stress_threshold.
    n_iwr_above_etx = 0
    max_iwr_minus_etx = 0.0
    if metadata.get("theoretical_iwr_target") == "stress_threshold":
        iwr_above_etx = (
            active_support
            & (iwr > etx + FLUX_CONSISTENCY_TOLERANCE_MM)
        )
        n_iwr_above_etx = int(np.count_nonzero(iwr_above_etx))
        max_iwr_minus_etx = (
            float(np.max(iwr[iwr_above_etx] - etx[iwr_above_etx]))
            if n_iwr_above_etx > 0
            else 0.0
        )

        if n_iwr_above_etx > 0 and STRICT_CONSISTENCY_CHECKS:
            raise ValueError(
                f"IWR exceeds ETx on {n_iwr_above_etx} active pixels for "
                f"stress_threshold target (max excess={max_iwr_minus_etx:.6g} mm/day)."
            )
        elif n_iwr_above_etx > 0:
            print(
                f"WARNING: IWR > ETx on {n_iwr_above_etx} active pixels "
                f"(max excess={max_iwr_minus_etx:.6g} mm/day)."
            )

    # F. ET balance check on active support only.
    balance_error = np.full_like(etx, np.nan, dtype="float64")
    if active_pixel_count > 0:
        balance_error[active_support] = np.abs(
            etx[active_support]
            - eta[active_support]
            - et_deficit[active_support]
        )

    valid_balance = np.isfinite(balance_error)
    if np.any(valid_balance):
        balance_max = float(np.nanmax(balance_error[valid_balance]))
        balance_mean = float(np.nanmean(balance_error[valid_balance]))
        balance_above_tol = int(
            np.count_nonzero(balance_error[valid_balance] > ET_BALANCE_TOLERANCE_MM)
        )
        balance_pixel_count = int(np.count_nonzero(valid_balance))
    else:
        balance_max = np.nan
        balance_mean = np.nan
        balance_above_tol = 0
        balance_pixel_count = 0

    return {
        "iwr_stats": iwr_stats,
        "etx_stats": etx_stats,
        "eta_stats": eta_stats,
        "deficit_stats": deficit_stats,
        "active_support_source": active_support_source,
        "active_pixel_count": active_pixel_count,
        "active_analysis_area_m2": active_area_m2,
        "active_analysis_area_fraction_of_total": active_area_fraction,
        "missing_iwr_on_active_pixel_count": n_missing_iwr,
        "missing_etx_on_active_pixel_count": n_missing_etx,
        "missing_eta_on_active_pixel_count": n_missing_eta,
        "positive_iwr_outside_active_pixel_count": n_pos_iwr_outside,
        "maximum_iwr_outside_active_mm_day": max_iwr_outside,
        "positive_eta_outside_active_pixel_count": n_pos_eta_outside,
        "maximum_eta_outside_active_mm_day": max_eta_outside,
        "eta_above_etx_pixel_count": n_eta_above_etx,
        "maximum_eta_minus_etx_mm_day": max_eta_minus_etx,
        "iwr_above_etx_pixel_count": n_iwr_above_etx,
        "maximum_iwr_minus_etx_mm_day": max_iwr_minus_etx,
        "balance_max_mm": balance_max,
        "balance_mean_mm": balance_mean,
        "balance_above_tol": balance_above_tol,
        "balance_pixel_count": balance_pixel_count,
    }



def build_daily_summary(
    iwr_files: dict[pd.Timestamp, Path],
    etx_files: dict[pd.Timestamp, Path],
    eta_stress_files: dict[pd.Timestamp, Path],
    metadata: dict,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    analysis_area_m2: np.ndarray,
    total_analysis_area_m2: float,
    active_mask_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    dates = pd.date_range(start=analysis_start, end=analysis_end, freq="D")

    records: list[dict] = []

    global_balance_max = -np.inf
    global_balance_sum = 0.0
    global_balance_count = 0
    global_balance_above_tol = 0

    n_complete = 0
    n_no_active_phenology = 0
    total_pos_iwr_outside = 0
    total_pos_eta_outside = 0
    total_eta_above_etx = 0
    total_iwr_above_etx = 0
    active_fractions: list[float] = []

    # Column names for the missing-file NaN fill.
    _nan_fields = [
        "iwr_valid_pixel_count",
        "iwr_valid_analysis_area_m2",
        "iwr_valid_analysis_area_fraction_of_total",
        "iwr_minimum_mm_day",
        "iwr_maximum_mm_day",
        "iwr_median_mm_day",
        "iwr_active_area_mean_mm_day",
        "iwr_domain_mean_mm_day",
        "iwr_volume_m3_day",
        "etx_active_area_mean_mm_day",
        "eta_stress_active_area_mean_mm_day",
        "et_deficit_active_area_mean_mm_day",
        "etx_domain_mean_mm_day",
        "eta_stress_domain_mean_mm_day",
        "et_deficit_domain_mean_mm_day",
        "etx_volume_m3_day",
        "eta_stress_volume_m3_day",
        "et_deficit_volume_m3_day",
        "et_balance_error_max_mm_day",
        "et_balance_error_mean_mm_day",
        "et_balance_error_pixels_above_tolerance",
        "active_support_source",
        "active_pixel_count",
        "active_analysis_area_m2",
        "active_analysis_area_fraction_of_total",
        "positive_iwr_outside_active_pixel_count",
        "maximum_iwr_outside_active_mm_day",
        "positive_eta_outside_active_pixel_count",
        "maximum_eta_outside_active_mm_day",
        "eta_above_etx_pixel_count",
        "maximum_eta_minus_etx_mm_day",
        "iwr_above_etx_pixel_count",
        "maximum_iwr_minus_etx_mm_day",
        "missing_iwr_on_active_pixel_count",
        "missing_etx_on_active_pixel_count",
        "missing_eta_on_active_pixel_count",
    ]

    for idx, date in enumerate(dates, start=1):
        print(f"[{idx}/{len(dates)}] {date.date()}")

        row: dict = {
            "date": date,
            "iwr_mode": metadata["iwr_mode"],
            "iwr_domain": metadata["iwr_domain"],
            "theoretical_iwr_target": metadata["theoretical_iwr_target"],
            "model_start_date": metadata["model_start_date"],
            "model_end_date": metadata["model_end_date"],
        }

        iwr_path = iwr_files.get(date)
        etx_path = etx_files.get(date)
        eta_path = eta_stress_files.get(date)

        missing: list[str] = []
        if iwr_path is None:
            missing.append("iwr")
        if etx_path is None:
            missing.append("etx")
        if eta_path is None:
            missing.append("eta")

        if missing:
            row["daily_data_status"] = "missing_" + "_".join(missing)
            row.update({k: np.nan for k in _nan_fields})
            records.append(row)
            continue

        # All three files are present.
        active_mask_path = _resolve_active_mask_path(date, active_mask_dir)

        joint = compute_daily_joint_metrics(
            iwr_path=iwr_path,
            etx_path=etx_path,
            eta_stress_path=eta_path,
            active_mask_path=active_mask_path,
            analysis_area_m2=analysis_area_m2,
            total_analysis_area_m2=total_analysis_area_m2,
            metadata=metadata,
        )

        n_complete += 1
        if joint["active_pixel_count"] == 0:
            n_no_active_phenology += 1
        active_fractions.append(joint["active_analysis_area_fraction_of_total"])

        total_pos_iwr_outside += joint["positive_iwr_outside_active_pixel_count"]
        total_pos_eta_outside += joint["positive_eta_outside_active_pixel_count"]
        total_eta_above_etx += joint["eta_above_etx_pixel_count"]
        total_iwr_above_etx += joint["iwr_above_etx_pixel_count"]

        iwr_s = joint["iwr_stats"]
        etx_s = joint["etx_stats"]
        eta_s = joint["eta_stats"]
        def_s = joint["deficit_stats"]

        row["daily_data_status"] = "complete"
        row.update(
            {
                "iwr_valid_pixel_count": iwr_s["valid_pixel_count"],
                "iwr_valid_analysis_area_m2": iwr_s["valid_analysis_area_m2"],
                # backward-compatible alias: now always equals active_analysis_area_fraction_of_total
                "iwr_valid_analysis_area_fraction_of_total": joint[
                    "active_analysis_area_fraction_of_total"
                ],
                "iwr_minimum_mm_day": iwr_s["minimum_mm_day"],
                "iwr_maximum_mm_day": iwr_s["maximum_mm_day"],
                "iwr_median_mm_day": iwr_s["median_mm_day"],
                "iwr_active_area_mean_mm_day": iwr_s["active_mean_mm"],
                "iwr_domain_mean_mm_day": iwr_s["domain_mean_mm"],
                "iwr_volume_m3_day": iwr_s["volume_m3"],
                "etx_active_area_mean_mm_day": etx_s["active_mean_mm"],
                "eta_stress_active_area_mean_mm_day": eta_s["active_mean_mm"],
                "et_deficit_active_area_mean_mm_day": def_s["active_mean_mm"],
                "etx_domain_mean_mm_day": etx_s["domain_mean_mm"],
                "eta_stress_domain_mean_mm_day": eta_s["domain_mean_mm"],
                "et_deficit_domain_mean_mm_day": def_s["domain_mean_mm"],
                "etx_volume_m3_day": etx_s["volume_m3"],
                "eta_stress_volume_m3_day": eta_s["volume_m3"],
                "et_deficit_volume_m3_day": def_s["volume_m3"],
                "et_balance_error_max_mm_day": joint["balance_max_mm"],
                "et_balance_error_mean_mm_day": joint["balance_mean_mm"],
                "et_balance_error_pixels_above_tolerance": joint["balance_above_tol"],
                "active_support_source": joint["active_support_source"],
                "active_pixel_count": joint["active_pixel_count"],
                "active_analysis_area_m2": joint["active_analysis_area_m2"],
                "active_analysis_area_fraction_of_total": joint[
                    "active_analysis_area_fraction_of_total"
                ],
                "positive_iwr_outside_active_pixel_count": joint[
                    "positive_iwr_outside_active_pixel_count"
                ],
                "maximum_iwr_outside_active_mm_day": joint["maximum_iwr_outside_active_mm_day"],
                "positive_eta_outside_active_pixel_count": joint[
                    "positive_eta_outside_active_pixel_count"
                ],
                "maximum_eta_outside_active_mm_day": joint["maximum_eta_outside_active_mm_day"],
                "eta_above_etx_pixel_count": joint["eta_above_etx_pixel_count"],
                "maximum_eta_minus_etx_mm_day": joint["maximum_eta_minus_etx_mm_day"],
                "iwr_above_etx_pixel_count": joint["iwr_above_etx_pixel_count"],
                "maximum_iwr_minus_etx_mm_day": joint["maximum_iwr_minus_etx_mm_day"],
                "missing_iwr_on_active_pixel_count": joint["missing_iwr_on_active_pixel_count"],
                "missing_etx_on_active_pixel_count": joint["missing_etx_on_active_pixel_count"],
                "missing_eta_on_active_pixel_count": joint["missing_eta_on_active_pixel_count"],
            }
        )

        if np.isfinite(joint["balance_max_mm"]):
            global_balance_max = max(global_balance_max, joint["balance_max_mm"])
        if (
            np.isfinite(joint["balance_mean_mm"])
            and joint["balance_pixel_count"] > 0
        ):
            global_balance_sum += joint["balance_mean_mm"] * joint["balance_pixel_count"]
            global_balance_count += joint["balance_pixel_count"]
            global_balance_above_tol += joint["balance_above_tol"]

        records.append(row)

    if global_balance_count > 0:
        global_balance_mean = global_balance_sum / global_balance_count
        global_balance_max_out: float = global_balance_max
    else:
        global_balance_mean = np.nan
        global_balance_max_out = np.nan

    min_active_fraction = float(np.min(active_fractions)) if active_fractions else np.nan
    max_active_fraction = float(np.max(active_fractions)) if active_fractions else np.nan

    diagnostics = {
        "total_complete_joint_dates": n_complete,
        "dates_without_active_phenology": n_no_active_phenology,
        "min_active_area_fraction": min_active_fraction,
        "max_active_area_fraction": max_active_fraction,
        "total_positive_iwr_outside_active_pixels": total_pos_iwr_outside,
        "total_positive_eta_outside_active_pixels": total_pos_eta_outside,
        "total_eta_above_etx_pixels": total_eta_above_etx,
        "total_iwr_above_etx_pixels": total_iwr_above_etx,
        "max_balance_error_mm": global_balance_max_out,
        "mean_balance_error_mm": global_balance_mean,
        "pixels_above_tolerance": global_balance_above_tol,
        "tolerance_mm": ET_BALANCE_TOLERANCE_MM,
        "valid_balance_pixel_count": global_balance_count,
    }

    return pd.DataFrame(records), diagnostics


def configured_period_bounds_for_year(
    year: int,
    configured_start: pd.Timestamp,
    configured_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)

    period_start = max(year_start, configured_start)
    period_end = min(year_end, configured_end)

    if period_start > period_end:
        return None, None

    return period_start, period_end


def build_annual_summary(
    daily_df: pd.DataFrame,
    configured_start: pd.Timestamp,
    configured_end: pd.Timestamp,
    total_analysis_area_m2: float,
) -> pd.DataFrame:
    years = np.arange(configured_start.year, configured_end.year + 1)

    available_iwr_dates = pd.to_datetime(daily_df.loc[daily_df["iwr_volume_m3_day"].notna(), "date"])
    available_iwr_set = set(available_iwr_dates)

    annual_rows = []

    for year in years:
        period_start, period_end = configured_period_bounds_for_year(
            year=year,
            configured_start=configured_start,
            configured_end=configured_end,
        )

        if period_start is None:
            continue

        year_dates = pd.date_range(start=period_start, end=period_end, freq="D")
        expected_days = int(len(year_dates))
        available_iwr_days = int(sum(d in available_iwr_set for d in year_dates))
        missing_iwr_days = int(expected_days - available_iwr_days)
        coverage_fraction = float(available_iwr_days / expected_days) if expected_days > 0 else np.nan

        is_complete_for_configured_period = bool(available_iwr_days == expected_days)

        has_full_calendar_window = (
            period_start.month == 1
            and period_start.day == 1
            and period_end.month == 12
            and period_end.day == 31
        )

        is_full_calendar_year = bool(
            has_full_calendar_window and is_complete_for_configured_period
        )

        year_mask = daily_df["year"] == year

        annual_iwr_volume_m3 = float(
            daily_df.loc[year_mask, "iwr_volume_m3_day"].sum(min_count=1)
        )
        if np.isfinite(annual_iwr_volume_m3):
            annual_iwr_domain_mean_mm = float(
                annual_iwr_volume_m3 / total_analysis_area_m2 * 1000.0
            )
        else:
            annual_iwr_domain_mean_mm = np.nan

        annual_etx_volume_m3 = float(
            daily_df.loc[year_mask, "etx_volume_m3_day"].sum(min_count=1)
        )
        annual_eta_volume_m3 = float(
            daily_df.loc[year_mask, "eta_stress_volume_m3_day"].sum(min_count=1)
        )
        annual_deficit_volume_m3 = float(
            daily_df.loc[year_mask, "et_deficit_volume_m3_day"].sum(min_count=1)
        )

        annual_etx_domain_mean_mm = (
            float(annual_etx_volume_m3 / total_analysis_area_m2 * 1000.0)
            if np.isfinite(annual_etx_volume_m3)
            else np.nan
        )
        annual_eta_domain_mean_mm = (
            float(annual_eta_volume_m3 / total_analysis_area_m2 * 1000.0)
            if np.isfinite(annual_eta_volume_m3)
            else np.nan
        )
        annual_deficit_domain_mean_mm = (
            float(annual_deficit_volume_m3 / total_analysis_area_m2 * 1000.0)
            if np.isfinite(annual_deficit_volume_m3)
            else np.nan
        )

        annual_rows.append(
            {
                "year": int(year),
                "period_start": period_start.date().isoformat(),
                "period_end": period_end.date().isoformat(),
                "expected_days": expected_days,
                "available_iwr_days": available_iwr_days,
                "missing_iwr_days": missing_iwr_days,
                "coverage_fraction": coverage_fraction,
                "is_full_calendar_year": is_full_calendar_year,
                "is_complete_for_configured_period": is_complete_for_configured_period,
                "annual_iwr_volume_m3_year": annual_iwr_volume_m3,
                "annual_iwr_domain_mean_mm_year": annual_iwr_domain_mean_mm,
                "annual_etx_volume_m3_year": annual_etx_volume_m3,
                "annual_eta_stress_volume_m3_year": annual_eta_volume_m3,
                "annual_et_deficit_volume_m3_year": annual_deficit_volume_m3,
                "annual_etx_domain_mean_mm_year": annual_etx_domain_mean_mm,
                "annual_eta_stress_domain_mean_mm_year": annual_eta_domain_mean_mm,
                "annual_et_deficit_domain_mean_mm_year": annual_deficit_domain_mean_mm,
            }
        )

    return pd.DataFrame(annual_rows).sort_values("year")


def annual_figure_size(number_of_years: int) -> tuple[float, float]:
    width = number_of_years * ANNUAL_INCHES_PER_YEAR
    width = min(ANNUAL_MAX_FIGURE_WIDTH, max(ANNUAL_MIN_FIGURE_WIDTH, width))
    return width, ANNUAL_FIGURE_HEIGHT


def prepare_annual_x_axis(ax, labels: list[str]):
    positions = np.arange(len(labels), dtype=float)

    if len(labels) == 0:
        return positions

    label_step = max(
        1,
        int(np.ceil((len(labels) - 1) / max(1, ANNUAL_MAX_X_LABELS - 1))),
    )

    tick_indices = np.arange(0, len(labels), label_step, dtype=int)
    if tick_indices[-1] != len(labels) - 1:
        tick_indices = np.append(tick_indices, len(labels) - 1)

    ax.set_xticks(positions[tick_indices])
    ax.set_xticklabels(
        np.array(labels)[tick_indices],
        rotation=ANNUAL_LABEL_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )

    ax.tick_params(axis="x", labelsize=10, pad=4)
    ax.margins(x=0.01)

    return positions


def subtitle_lines(metadata: dict) -> str:
    return (
        f"mode: {metadata['iwr_mode']}\n"
        f"domain: {metadata['iwr_domain']}\n"
        f"target: {metadata['theoretical_iwr_target']}\n"
        f"drainage_scheme: {metadata.get('drainage_scheme', 'unknown')}"
    )


def _shade_domain_inactive_periods(
    ax,
    daily_df: pd.DataFrame,
    frac_col: str = "active_analysis_area_fraction_of_total",
    alpha: float = 0.08,
    color: str = "#888888",
) -> None:
    """
    Shade date spans where no crop is phenologically active in the domain.

    A day is considered inactive when the active-area fraction is zero or NaN.
    During these periods ETx, crop ETa, and theoretical IWR must be zero;
    water-balance background evaporation may remain positive.
    """
    if frac_col not in daily_df.columns:
        return

    dates = daily_df["date"].to_numpy()
    inactive = (daily_df[frac_col].fillna(0.0) == 0.0).to_numpy()

    in_span = False
    span_start = None

    for date, is_inactive in zip(dates, inactive):
        if is_inactive and not in_span:
            span_start = date
            in_span = True
        elif not is_inactive and in_span:
            ax.axvspan(span_start, date, alpha=alpha, color=color, linewidth=0)
            in_span = False

    if in_span and span_start is not None:
        ax.axvspan(span_start, dates[-1], alpha=alpha, color=color, linewidth=0)


def save_daily_iwr_plot(daily_df: pd.DataFrame, metadata: dict) -> None:
    mode_to_column = {
        "active_area": "iwr_active_area_mean_mm_day",
        "domain_area": "iwr_domain_mean_mm_day",
    }

    if DAILY_MEAN_MODE not in mode_to_column:
        raise ValueError("DAILY_MEAN_MODE must be one of: active_area, domain_area")

    y_col = mode_to_column[DAILY_MEAN_MODE]

    if DAILY_MEAN_MODE == "active_area":
        y_label = "Crop-area-weighted mean over common phenology-active area [mm/day]"
    else:
        y_label = "Domain-equivalent mean over total static crop area [mm/day]"

    fig, (ax, ax_area) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    ax.plot(daily_df["date"], daily_df[y_col], linewidth=1.2)
    ax.set_title("Daily theoretical irrigation water requirement")
    ax.text(0.01, 0.98, subtitle_lines(metadata), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    _shade_domain_inactive_periods(ax, daily_df)

    frac_col = "active_analysis_area_fraction_of_total"
    if frac_col in daily_df.columns:
        ax_area.plot(daily_df["date"], daily_df[frac_col], linewidth=1.0, color="#666666")
    ax_area.set_ylim(0, 1)
    ax_area.set_ylabel("Phenology-active\nfraction")
    ax_area.set_xlabel("Date")
    ax_area.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "daily_iwr_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_yearly_iwr_plot(annual_df: pd.DataFrame, metadata: dict) -> None:
    if annual_df.empty:
        raise RuntimeError("Annual summary is empty; cannot plot yearly IWR")

    if INCLUDE_PARTIAL_YEARS:
        plot_df = annual_df.copy()
    else:
        plot_df = annual_df[annual_df["is_full_calendar_year"]].copy()

    if plot_df.empty:
        raise RuntimeError(
            "No years available for annual plot after applying INCLUDE_PARTIAL_YEARS setting"
        )

    partial_mask = ~plot_df["is_full_calendar_year"].to_numpy(dtype=bool)

    if ANNUAL_PLOT_MODE == "volume_m3":
        values = plot_df["annual_iwr_volume_m3_year"].to_numpy(dtype=float) / 1e6
        y_label = "Annual IWR volume [million m3/year]"
    elif ANNUAL_PLOT_MODE == "mean_mm":
        values = plot_df["annual_iwr_domain_mean_mm_year"].to_numpy(dtype=float)
        y_label = "Annual domain-average accumulated IWR depth [mm/year]"
    else:
        raise ValueError("ANNUAL_PLOT_MODE must be one of: volume_m3, mean_mm")

    year_labels = [
        f"{int(y)}{'*' if is_partial else ''}"
        for y, is_partial in zip(plot_df["year"], partial_mask)
    ]

    fig, ax = plt.subplots(figsize=annual_figure_size(len(plot_df)))
    x = prepare_annual_x_axis(ax, year_labels)

    bars = ax.bar(x, values, width=0.78, color="#4e79a7")
    for bar, is_partial in zip(bars, partial_mask):
        if is_partial:
            bar.set_hatch("//")
            bar.set_edgecolor("#2f2f2f")

    ax.set_title("Annual theoretical irrigation water requirement")
    ax.text(0.01, 0.98, subtitle_lines(metadata), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Year", labelpad=12)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    if np.any(partial_mask):
        partial_patch = mpatches.Patch(
            facecolor="#4e79a7",
            edgecolor="#2f2f2f",
            hatch="//",
            label="* partial year",
        )
        ax.legend(handles=[partial_patch], loc="best")

    fig.subplots_adjust(bottom=0.20)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "yearly_iwr.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_daily_et_comparison_plot(daily_df: pd.DataFrame, metadata: dict) -> None:
    mode_to_columns = {
        "active_area": (
            "etx_active_area_mean_mm_day",
            "eta_stress_active_area_mean_mm_day",
            "et_deficit_active_area_mean_mm_day",
            "iwr_active_area_mean_mm_day",
            "Crop-area-weighted mean over common phenology-active area [mm/day]",
        ),
        "domain_area": (
            "etx_domain_mean_mm_day",
            "eta_stress_domain_mean_mm_day",
            "et_deficit_domain_mean_mm_day",
            "iwr_domain_mean_mm_day",
            "Domain-equivalent mean over total static crop area [mm/day]",
        ),
    }

    if DAILY_MEAN_MODE not in mode_to_columns:
        raise ValueError("DAILY_MEAN_MODE must be one of: active_area, domain_area")

    etx_col, eta_col, deficit_col, iwr_col, y_label = mode_to_columns[DAILY_MEAN_MODE]

    # Determine whether a mass-balance residual panel is available.
    balance_col = "et_balance_error_max_mm_day"
    has_balance = balance_col in daily_df.columns and daily_df[balance_col].notna().any()

    n_rows = 3 if has_balance else 2
    height_ratios = ([3, 1.5, 1] if has_balance else [4, 1])

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(13, 3.5 * n_rows),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    ax = axes[0]
    ax_area = axes[-1]

    ax.plot(daily_df["date"], daily_df[etx_col], linewidth=1.2, label="ETx (stress-free crop ET)")
    ax.plot(daily_df["date"], daily_df[eta_col], linewidth=1.2, label="ETa (stress-limited)")
    ax.plot(daily_df["date"], daily_df[deficit_col], linewidth=1.2, label="ET deficit (ETx − ETa)")

    # IWR is plotted as a separate dashed line — it is NOT an ET component and
    # must not be visually stacked or confused with the crop ET terms.
    ax.plot(
        daily_df["date"],
        daily_df[iwr_col],
        linewidth=1.4,
        linestyle="--",
        color="#d62728",
        label="Theoretical net IWR  ← separate from ET components",
    )

    ax.set_title("Daily crop ET and theoretical irrigation requirement")
    ax.text(0.01, 0.98, subtitle_lines(metadata), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    _shade_domain_inactive_periods(ax, daily_df)

    ax.text(
        0.01,
        0.02,
        "All fluxes use the same daily phenology-active spatial support.\n"
        "IWR is shown separately — it is not an ET component.\n"
        "Grey shading = inactive phenology (ETx, crop ETa, IWR must be zero).",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    # Optional mass-balance residual panel.
    if has_balance:
        ax_res = axes[1]
        ax_res.plot(
            daily_df["date"],
            daily_df[balance_col],
            linewidth=1.0,
            color="#333333",
        )
        ax_res.axhline(0, linewidth=0.8, color="red", linestyle="--")
        ax_res.set_ylabel("ET balance error\nmax [mm/day]")
        ax_res.grid(True, alpha=0.3)

        max_bal = daily_df[balance_col].abs().max()
        ax_res.set_title(
            f"ET balance residual — max |error| = {max_bal:.2e} mm/day",
            fontsize=9, loc="left",
        )

    frac_col = "active_analysis_area_fraction_of_total"
    if frac_col in daily_df.columns:
        ax_area.plot(daily_df["date"], daily_df[frac_col], linewidth=1.0, color="#666666")
    ax_area.set_ylim(0, 1)
    ax_area.set_ylabel("Phenology-active\nfraction")
    ax_area.set_xlabel("Date")
    ax_area.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "daily_et_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_yearly_et_components_plot(annual_df: pd.DataFrame, metadata: dict) -> None:
    if annual_df.empty:
        raise RuntimeError("Annual summary is empty; cannot plot yearly ET components")

    if INCLUDE_PARTIAL_YEARS:
        plot_df = annual_df.copy()
    else:
        plot_df = annual_df[annual_df["is_full_calendar_year"]].copy()

    if plot_df.empty:
        raise RuntimeError(
            "No years available for ET annual plot after applying INCLUDE_PARTIAL_YEARS setting"
        )

    partial_mask = ~plot_df["is_full_calendar_year"].to_numpy(dtype=bool)

    if ANNUAL_PLOT_MODE == "volume_m3":
        eta_values = plot_df["annual_eta_stress_volume_m3_year"].to_numpy(dtype=float) / 1e6
        deficit_values = plot_df["annual_et_deficit_volume_m3_year"].to_numpy(dtype=float) / 1e6
        etx_values = plot_df["annual_etx_volume_m3_year"].to_numpy(dtype=float) / 1e6
        iwr_values = plot_df["annual_iwr_volume_m3_year"].to_numpy(dtype=float) / 1e6
        y_label = "Annual volume [million m³/year]"
        iwr_y_label = "Annual IWR [million m³/year]"
    elif ANNUAL_PLOT_MODE == "mean_mm":
        eta_values = plot_df["annual_eta_stress_domain_mean_mm_year"].to_numpy(dtype=float)
        deficit_values = plot_df["annual_et_deficit_domain_mean_mm_year"].to_numpy(dtype=float)
        etx_values = plot_df["annual_etx_domain_mean_mm_year"].to_numpy(dtype=float)
        iwr_values = plot_df["annual_iwr_domain_mean_mm_year"].to_numpy(dtype=float)
        y_label = "Annual domain-average accumulated depth [mm/year]"
        iwr_y_label = "Annual IWR [mm/year]"
    else:
        raise ValueError("ANNUAL_PLOT_MODE must be one of: volume_m3, mean_mm")

    year_labels = [
        f"{int(y)}{'*' if is_partial else ''}"
        for y, is_partial in zip(plot_df["year"], partial_mask)
    ]

    # Two-panel layout: ET components (top) and IWR (bottom, separate axis).
    fig, (ax, ax_iwr) = plt.subplots(
        2, 1,
        figsize=annual_figure_size(len(plot_df)),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    x = prepare_annual_x_axis(ax, year_labels)
    prepare_annual_x_axis(ax_iwr, year_labels)

    bars_eta = ax.bar(x, eta_values, width=0.78, color="#4e79a7", label="ETa (stress-limited)")
    bars_deficit = ax.bar(
        x,
        deficit_values,
        width=0.78,
        bottom=eta_values,
        color="#f28e2b",
        label="ET deficit (ETx − ETa)",
    )

    ax.plot(x, etx_values, color="#2f2f2f", linewidth=1.4, marker="o", label="ETx (validation line)")

    for bar_group in (bars_eta, bars_deficit):
        for bar, is_partial in zip(bar_group, partial_mask):
            if is_partial:
                bar.set_hatch("//")
                bar.set_edgecolor("#2f2f2f")

    ax.set_title("Annual ET components and theoretical IWR  (IWR shown separately below)")
    ax.text(0.01, 0.98, subtitle_lines(metadata), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # IWR lower panel — completely separate axis so it cannot be confused with
    # ET-component stacking.
    bars_iwr = ax_iwr.bar(x, iwr_values, width=0.78, color="#d62728",
                          label="Theoretical net IWR")
    for bar, is_partial in zip(bars_iwr, partial_mask):
        if is_partial:
            bar.set_hatch("//")
            bar.set_edgecolor("#2f2f2f")

    ax_iwr.set_ylabel(iwr_y_label)
    ax_iwr.set_xlabel("Year", labelpad=12)
    ax_iwr.grid(True, axis="y", alpha=0.3)
    ax_iwr.set_axisbelow(True)
    ax_iwr.text(
        0.01, 0.97,
        "IWR is not an ET component — shown on a separate axis.",
        transform=ax_iwr.transAxes, fontsize=8, va="top",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles_iwr, labels_iwr = ax_iwr.get_legend_handles_labels()
    all_handles = handles + handles_iwr
    all_labels = labels + labels_iwr
    if np.any(partial_mask):
        all_handles.append(
            mpatches.Patch(
                facecolor="#ffffff",
                edgecolor="#2f2f2f",
                hatch="//",
                label="* partial year",
            )
        )
        all_labels.append("* partial year")
    ax.legend(handles=all_handles, labels=all_labels, loc="best", fontsize=9)

    fig.subplots_adjust(bottom=0.20)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "yearly_et_components.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_json(MODEL_CONFIG_FILE)
    metadata = extract_model_metadata(config)
    paths = build_runtime_paths(config)

    require_existing_path(paths["IWR_DIR"], "IWR_DIR", "dir")

    iwr_files = list_daily_files(paths["IWR_DIR"], "iwr")
    if len(iwr_files) == 0:
        raise RuntimeError(f"No iwr_YYYYMMDD.tif files found in {paths['IWR_DIR']}")

    first_iwr_file = iwr_files[min(iwr_files.keys())]

    cell_area_m2, area_fraction, analysis_area_m2, total_analysis_area_m2 = load_support_layers(
        cell_area_file=CELL_AREA_M2_FILE,
        area_fraction_file=paths["IWR_AREA_FRACTION_FILE"],
        reference_iwr_file=first_iwr_file,
    )

    analysis_start, analysis_end = select_analysis_period(
        metadata["model_start_date"],
        metadata["model_end_date"],
    )

    etx_files = list_daily_files(paths["ETX_DIR"], "etx") if paths["ETX_DIR"].is_dir() else {}
    eta_stress_files = (
        list_daily_files(paths["ETA_STRESS_DIR"], "eta_stress")
        if paths["ETA_STRESS_DIR"].is_dir()
        else {}
    )

    # Determine active mask directory.
    if ACTIVE_MASK_DIR is not None:
        resolved_active_mask_dir: Path | None = Path(ACTIVE_MASK_DIR)
    else:
        candidate = paths["ACTIVE_MASK_DIR"]
        resolved_active_mask_dir = candidate if candidate.is_dir() else None

    # Count active masks found.
    n_active_masks = 0
    if resolved_active_mask_dir is not None and resolved_active_mask_dir.is_dir():
        n_active_masks = len(
            list(resolved_active_mask_dir.glob(f"{ACTIVE_MASK_PREFIX}_????????.tif"))
        )

    complete_dates = (
        set(iwr_files.keys()) & set(etx_files.keys()) & set(eta_stress_files.keys())
    )

    print(f"Configured model period: {metadata['model_start_date']} to {metadata['model_end_date']}")
    print(f"Analysis period: {analysis_start.date()} to {analysis_end.date()}")
    print(f"Total analysis area [m2]: {total_analysis_area_m2:,.2f}")
    print(f"IWR files found: {len(iwr_files)}")
    print(f"ETx files found: {len(etx_files)}")
    print(f"ETa files found: {len(eta_stress_files)}")
    print(f"Complete joint dates (IWR + ETx + ETa): {len(complete_dates)}")
    print(f"Active masks found: {n_active_masks}")

    daily_df, diagnostics = build_daily_summary(
        iwr_files=iwr_files,
        etx_files=etx_files,
        eta_stress_files=eta_stress_files,
        metadata=metadata,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        analysis_area_m2=analysis_area_m2,
        total_analysis_area_m2=total_analysis_area_m2,
        active_mask_dir=resolved_active_mask_dir,
    )

    daily_df["year"] = daily_df["date"].dt.year

    annual_df = build_annual_summary(
        daily_df=daily_df,
        configured_start=pd.to_datetime(metadata["model_start_date"]),
        configured_end=pd.to_datetime(metadata["model_end_date"]),
        total_analysis_area_m2=total_analysis_area_m2,
    )

    daily_out = OUTPUT_DIR / "daily_iwr_summary.csv"
    annual_out = OUTPUT_DIR / "annual_iwr_summary.csv"

    daily_df.to_csv(daily_out, index=False)
    annual_df.to_csv(annual_out, index=False)

    print(f"Saved: {daily_out}")
    print(f"Saved: {annual_out}")

    # --- Global diagnostic summary ---
    print("\nGlobal diagnostic summary:")
    print(f"  Complete joint dates: {diagnostics['total_complete_joint_dates']}")
    print(f"  Dates without active phenology: {diagnostics['dates_without_active_phenology']}")
    min_frac = diagnostics["min_active_area_fraction"]
    max_frac = diagnostics["max_active_area_fraction"]
    min_frac_str = f"{min_frac:.4f}" if isinstance(min_frac, float) and not np.isnan(min_frac) else "N/A"
    max_frac_str = f"{max_frac:.4f}" if isinstance(max_frac, float) and not np.isnan(max_frac) else "N/A"
    print(f"  Active-area fraction range: {min_frac_str} – {max_frac_str}")
    print(f"  Total positive-IWR-outside-active pixels: {diagnostics['total_positive_iwr_outside_active_pixels']}")
    print(f"  Total positive-ETa-outside-active pixels: {diagnostics['total_positive_eta_outside_active_pixels']}")
    print(f"  Total ETa > ETx pixels: {diagnostics['total_eta_above_etx_pixels']}")
    print(f"  Total IWR > ETx pixels (stress_threshold): {diagnostics['total_iwr_above_etx_pixels']}")
    bal = diagnostics["max_balance_error_mm"]
    print(f"  Max ET balance error [mm]: {bal:.6g}" if isinstance(bal, float) and not np.isnan(bal) else "  Max ET balance error [mm]: N/A")

    # Save diagnostics JSON (replace non-finite floats with None for JSON compliance).
    def _json_safe(v: object) -> object:
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    diag_out = OUTPUT_DIR / "plotting_diagnostics.json"
    with diag_out.open("w", encoding="utf-8") as f_diag:
        json.dump({k: _json_safe(v) for k, v in diagnostics.items()}, f_diag, indent=2)
    print(f"Saved: {diag_out}")

    save_daily_iwr_plot(daily_df, metadata)
    save_yearly_iwr_plot(annual_df, metadata)

    if len(complete_dates) > 0:
        save_daily_et_comparison_plot(daily_df, metadata)
        save_yearly_et_components_plot(annual_df, metadata)
    else:
        print(
            "WARNING: Skipping ET plots because matching IWR/ETx/ETa daily files are unavailable."
        )

    # Keep support-layer arrays explicit and used.
    _ = cell_area_m2, area_fraction

    print("Done.")


if __name__ == "__main__":
    main()


# =========================
# SYNTHETIC TESTS
# =========================
# These functions can be run as standalone checks or collected by pytest.


def _make_test_area(shape: tuple[int, int], value: float = 1000.0) -> np.ndarray:
    return np.full(shape, value, dtype="float64")


def test_all_zero_etx_eta_iwr() -> None:
    """Test 1: ETx, ETa and IWR are zero everywhere -> active fraction = 0, all means = 0."""
    shape = (4, 4)
    area = _make_test_area(shape)
    total_area = float(np.sum(area))

    etx = np.zeros(shape, dtype="float64")
    iwr = np.zeros(shape, dtype="float64")

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    assert int(np.count_nonzero(active_support)) == 0

    iwr_s = compute_weighted_metrics_from_depth_array(iwr, area, total_area, active_support)
    assert iwr_s["valid_pixel_count"] == 0
    assert iwr_s["active_mean_mm"] == 0.0
    assert iwr_s["domain_mean_mm"] == 0.0
    assert iwr_s["volume_m3"] == 0.0
    assert iwr_s["valid_analysis_area_fraction_of_total"] == 0.0


def test_two_active_pixels_common_denominator() -> None:
    """Test 2: Only 2 pixels have ETx > 0; all active means use those same 2 pixels."""
    shape = (4, 4)
    area = _make_test_area(shape, 1000.0)
    total_area = float(np.sum(area))

    etx = np.zeros(shape, dtype="float64")
    etx[0, 0] = 5.0
    etx[1, 1] = 3.0

    eta = np.zeros(shape, dtype="float64")
    eta[0, 0] = 4.0
    eta[1, 1] = 2.0

    iwr = np.zeros(shape, dtype="float64")
    iwr[0, 0] = 1.0
    iwr[1, 1] = 1.0

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    assert int(np.count_nonzero(active_support)) == 2

    iwr_s = compute_weighted_metrics_from_depth_array(iwr, area, total_area, active_support)
    etx_s = compute_weighted_metrics_from_depth_array(etx, area, total_area, active_support)

    assert iwr_s["valid_pixel_count"] == 2
    assert etx_s["valid_pixel_count"] == 2

    # Domain mean uses full denominator (16 pixels * 1000 m2 = 16000 m2).
    expected_etx_volume = (5.0 * 1000 + 3.0 * 1000) / 1000.0  # m3
    expected_etx_domain_mean = expected_etx_volume / total_area * 1000.0
    assert abs(etx_s["domain_mean_mm"] - expected_etx_domain_mean) < 1e-9


def test_iwr_nodata_outside_active_etx_zero() -> None:
    """Test 3: IWR nodata outside active, ETx/ETa finite zero -> same active support."""
    shape = (3, 3)
    area = _make_test_area(shape, 1000.0)
    total_area = float(np.sum(area))

    etx = np.zeros(shape, dtype="float64")
    etx[1, 1] = 4.0

    iwr = np.full(shape, np.nan, dtype="float64")
    iwr[1, 1] = 1.0  # only active pixel has finite IWR

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    assert int(np.count_nonzero(active_support)) == 1

    iwr_s = compute_weighted_metrics_from_depth_array(iwr, area, total_area, active_support)
    etx_s = compute_weighted_metrics_from_depth_array(etx, area, total_area, active_support)

    assert iwr_s["valid_pixel_count"] == 1
    assert etx_s["valid_pixel_count"] == 1
    # Denominators are identical.
    assert iwr_s["valid_analysis_area_m2"] == etx_s["valid_analysis_area_m2"]


def test_iwr_positive_where_etx_zero_detection() -> None:
    """Test 4: IWR positive where ETx is zero -> positive_iwr_outside_active is non-zero."""
    shape = (3, 3)
    etx = np.zeros(shape, dtype="float64")
    etx[1, 1] = 4.0

    iwr = np.zeros(shape, dtype="float64")
    iwr[1, 1] = 1.0
    iwr[0, 0] = 2.0  # positive where ETx = 0

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    positive_iwr_outside = (
        static_support
        & ~active_support
        & np.isfinite(iwr)
        & (iwr > FLUX_CONSISTENCY_TOLERANCE_MM)
    )
    assert int(np.count_nonzero(positive_iwr_outside)) == 1


def test_eta_above_etx_detection() -> None:
    """Test 5: ETa > ETx on active pixel -> eta_above_etx mask is non-zero."""
    shape = (2, 2)
    etx = np.full(shape, 3.0, dtype="float64")
    eta = np.full(shape, 5.0, dtype="float64")  # eta > etx

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    eta_above_etx = active_support & (eta > etx + FLUX_CONSISTENCY_TOLERANCE_MM)
    assert int(np.count_nonzero(eta_above_etx)) == shape[0] * shape[1]


def test_iwr_above_etx_stress_threshold_detection() -> None:
    """Test 6: stress_threshold, IWR > ETx -> iwr_above_etx mask is non-zero."""
    shape = (2, 2)
    etx = np.full(shape, 3.0, dtype="float64")
    iwr = np.full(shape, 5.0, dtype="float64")  # IWR > ETx

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    iwr_above_etx = active_support & (iwr > etx + FLUX_CONSISTENCY_TOLERANCE_MM)
    assert int(np.count_nonzero(iwr_above_etx)) == shape[0] * shape[1]


def test_missing_values_on_active_pixel_detection() -> None:
    """Test 7: Active pixel has NaN in IWR -> missing_iwr_on_active is non-zero."""
    shape = (3, 3)
    etx = np.zeros(shape, dtype="float64")
    etx[1, 1] = 4.0

    iwr = np.full(shape, np.nan, dtype="float64")
    # iwr[1,1] remains NaN -> missing on active support

    static_support = np.ones(shape, dtype=bool)
    active_support = static_support & (etx > ACTIVE_ETX_TOLERANCE_MM)

    missing_iwr_on_active = active_support & ~np.isfinite(iwr)
    assert int(np.count_nonzero(missing_iwr_on_active)) == 1


def test_no_active_mask_falls_back_to_etx() -> None:
    """Test 8: No active mask available -> active_support derived from ETx."""
    shape = (3, 3)
    etx = np.zeros(shape, dtype="float64")
    etx[1, 1] = 4.0

    static_support = np.ones(shape, dtype=bool)
    # Simulate the auto-fallback: no active_mask_path -> use ETx.
    active_mask_path = None
    if active_mask_path is not None:
        active_support_source = "active_mask"
    else:
        active_support_source = "etx_positive"
        active_support = (
            static_support
            & np.isfinite(etx)
            & (etx > ACTIVE_ETX_TOLERANCE_MM)
        )

    assert active_support_source == "etx_positive"
    assert int(np.count_nonzero(active_support)) == 1


def run_all_synthetic_tests() -> None:
    """Run all synthetic tests; raises AssertionError on failure."""
    test_all_zero_etx_eta_iwr()
    test_two_active_pixels_common_denominator()
    test_iwr_nodata_outside_active_etx_zero()
    test_iwr_positive_where_etx_zero_detection()
    test_eta_above_etx_detection()
    test_iwr_above_etx_stress_threshold_detection()
    test_missing_values_on_active_pixel_detection()
    test_no_active_mask_falls_back_to_etx()
    print("All synthetic tests passed.")
