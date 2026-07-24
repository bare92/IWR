#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_iwr_anomalies import (
    ConfigError,
    PeriodKey,
    RasterReference,
    aggregate_period,
    aggregation_context,
    aggregation_window_bounds,
    build_period_groups,
    configure_logging,
    discover_daily_files,
    get_reference,
    read_float_raster,
    save_aggregated_period,
    select_baseline_periods,
    select_target_periods,
    validate_config,
    write_count_raster,
    write_float_raster,
)


LOGGER = logging.getLogger("iwr_anomaly_gamma")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    validate_gamma_config(config)
    return config


def validate_gamma_config(config: dict) -> None:
    climatology_cfg = config["climatology"]
    minimum_positive_years = int(climatology_cfg.get("minimum_positive_years", 3))
    if minimum_positive_years <= 0:
        raise ConfigError("climatology.minimum_positive_years must be a positive integer")

    gamma_epsilon = float(climatology_cfg.get("gamma_epsilon", 1.0e-12))
    if gamma_epsilon <= 0.0:
        raise ConfigError("climatology.gamma_epsilon must be > 0")

    cdf_clip = float(climatology_cfg.get("cdf_clip", 1.0e-8))
    if not 0.0 < cdf_clip < 0.5:
        raise ConfigError("climatology.cdf_clip must be in the interval (0, 0.5)")


def gamma_climatology_paths(
    climatology_dir: Path,
    month: int,
    dekad: int,
) -> Tuple[Path, Path, Path, Path]:
    stem = f"iwr_climatology_m{month:02d}_monthly" if dekad == 0 else f"iwr_climatology_m{month:02d}_d{dekad}"
    return (
        climatology_dir / f"{stem}_shape_gamma.tif",
        climatology_dir / f"{stem}_scale_gamma.tif",
        climatology_dir / f"{stem}_pzero_gamma.tif",
        climatology_dir / f"{stem}_count_gamma.tif",
    )


def fit_gamma_climatology_arrays(
    total_count: np.ndarray,
    zero_count: np.ndarray,
    positive_count: np.ndarray,
    positive_sum: np.ndarray,
    positive_log_sum: np.ndarray,
    positive_sq_sum: np.ndarray,
    minimum_years: int,
    minimum_positive_years: int,
    gamma_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.full(total_count.shape, np.nan, dtype=np.float64)
    scale = np.full(total_count.shape, np.nan, dtype=np.float64)
    pzero = np.full(total_count.shape, np.nan, dtype=np.float64)

    sufficient_total = total_count >= minimum_years
    pzero[sufficient_total] = zero_count[sufficient_total] / total_count[sufficient_total]

    fit_mask = sufficient_total & (positive_count >= minimum_positive_years)
    if not np.any(fit_mask):
        return shape, scale, pzero

    positive_mean = positive_sum[fit_mask] / positive_count[fit_mask]
    positive_mean_log = positive_log_sum[fit_mask] / positive_count[fit_mask]
    positive_variance = (
        positive_sq_sum[fit_mask] / positive_count[fit_mask]
        - positive_mean * positive_mean
    )

    a_value = np.log(positive_mean) - positive_mean_log
    shape_values = np.full(positive_mean.shape, np.nan, dtype=np.float64)

    mle_mask = a_value > gamma_epsilon
    shape_values[mle_mask] = (
        1.0 + np.sqrt(1.0 + (4.0 * a_value[mle_mask] / 3.0))
    ) / (4.0 * a_value[mle_mask])

    moment_mask = (~mle_mask) & (positive_variance > gamma_epsilon)
    shape_values[moment_mask] = (
        positive_mean[moment_mask] * positive_mean[moment_mask]
    ) / positive_variance[moment_mask]

    valid_shape = np.isfinite(shape_values) & (shape_values > 0.0)
    scale_values = np.full(positive_mean.shape, np.nan, dtype=np.float64)
    scale_values[valid_shape] = positive_mean[valid_shape] / shape_values[valid_shape]

    valid_scale = np.isfinite(scale_values) & (scale_values > 0.0)
    fitted = valid_shape & valid_scale
    if not np.any(fitted):
        return shape, scale, pzero

    shape_block = np.full(positive_mean.shape, np.nan, dtype=np.float64)
    scale_block = np.full(positive_mean.shape, np.nan, dtype=np.float64)
    shape_block[fitted] = shape_values[fitted]
    scale_block[fitted] = scale_values[fitted]
    shape[fit_mask] = shape_block
    scale[fit_mask] = scale_block
    return shape, scale, pzero


def gamma_cdf_to_standard_normal(
    values: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    pzero: np.ndarray,
    cdf_clip: float,
) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0.0) & np.isfinite(pzero)
    if not np.any(valid):
        return output

    cdf = np.full(values.shape, np.nan, dtype=np.float64)

    zero_obs = valid & (values == 0.0)
    cdf[zero_obs] = pzero[zero_obs]

    positive_obs = valid & (values > 0.0)
    positive_fit = (
        positive_obs
        & np.isfinite(shape)
        & np.isfinite(scale)
        & (shape > 0.0)
        & (scale > 0.0)
    )
    if np.any(positive_fit):
        gamma_cdf = stats.gamma.cdf(
            values[positive_fit],
            a=shape[positive_fit],
            loc=0.0,
            scale=scale[positive_fit],
        )
        cdf[positive_fit] = pzero[positive_fit] + (1.0 - pzero[positive_fit]) * gamma_cdf

    all_zero_climatology = valid & (pzero >= 1.0 - cdf_clip)
    output[all_zero_climatology & zero_obs] = 0.0
    cdf[all_zero_climatology & positive_obs] = 1.0 - cdf_clip

    cdf_ready = np.isfinite(cdf) & ~np.isfinite(output)
    if np.any(cdf_ready):
        output[cdf_ready] = stats.norm.ppf(np.clip(cdf[cdf_ready], cdf_clip, 1.0 - cdf_clip))
    return output


def compute_gamma_climatology_for_key(
    key: Tuple[int, int],
    periods: List[Tuple[PeriodKey, Dict]],
    all_daily_files: Dict,
    reference: RasterReference,
    config: dict,
) -> None:
    month, dekad = key
    climatology_cfg = config["climatology"]
    climatology_dir = Path(climatology_cfg["directory"])
    shape_path, scale_path, pzero_path, count_path = gamma_climatology_paths(
        climatology_dir,
        month,
        dekad,
    )
    overwrite = bool(climatology_cfg.get("overwrite", False))

    if (
        shape_path.exists()
        and scale_path.exists()
        and pzero_path.exists()
        and count_path.exists()
        and not overwrite
    ):
        LOGGER.info("Gamma climatology exists for month %02d dekad %d; reusing it", month, dekad)
        return

    total_count = np.zeros((reference.height, reference.width), dtype=np.uint16)
    zero_count = np.zeros((reference.height, reference.width), dtype=np.uint16)
    positive_count = np.zeros((reference.height, reference.width), dtype=np.uint16)
    positive_sum = np.zeros((reference.height, reference.width), dtype=np.float64)
    positive_log_sum = np.zeros((reference.height, reference.width), dtype=np.float64)
    positive_sq_sum = np.zeros((reference.height, reference.width), dtype=np.float64)

    used_periods = 0
    for period, _ in sorted(periods, key=lambda item: item[0]):
        aggregated = aggregate_period(period, all_daily_files, reference, config)
        if aggregated is None:
            continue

        valid = np.isfinite(aggregated)
        zero = valid & (aggregated == 0.0)
        positive = valid & (aggregated > 0.0)

        total_count[valid] += 1
        zero_count[zero] += 1
        positive_count[positive] += 1
        positive_sum[positive] += aggregated[positive]
        positive_log_sum[positive] += np.log(aggregated[positive])
        positive_sq_sum[positive] += aggregated[positive] * aggregated[positive]
        used_periods += 1

    if used_periods == 0:
        raise RuntimeError(
            f"No usable baseline periods were available for month {month:02d}, dekad {dekad}"
        )

    minimum_years = int(climatology_cfg.get("minimum_years", 3))
    minimum_positive_years = int(climatology_cfg.get("minimum_positive_years", 3))
    gamma_epsilon = float(climatology_cfg.get("gamma_epsilon", 1.0e-12))
    shape_output, scale_output, pzero_output = fit_gamma_climatology_arrays(
        total_count,
        zero_count,
        positive_count,
        positive_sum,
        positive_log_sum,
        positive_sq_sum,
        minimum_years,
        minimum_positive_years,
        gamma_epsilon,
    )

    start_year = climatology_cfg.get("start_year")
    end_year = climatology_cfg.get("end_year")
    aggregation_label, _, aggregated_units = aggregation_context(config)
    common_tags = {
        "variable": "iwr_gamma_climatology",
        "distribution": "gamma",
        "zero_probability_included": True,
        "aggregation": aggregation_label,
        "aggregation_operation": "sum",
        "month": month,
        "dekad": dekad,
        "baseline_start_year": start_year if start_year is not None else "all",
        "baseline_end_year": end_year if end_year is not None else "all",
        "minimum_years": minimum_years,
        "minimum_positive_years": minimum_positive_years,
        "source_units": "mm/day",
        "aggregated_units": aggregated_units,
    }

    write_float_raster(
        shape_path,
        shape_output,
        reference,
        config,
        {**common_tags, "statistic": "gamma_shape", "units": "dimensionless"},
        overwrite=True,
    )
    write_float_raster(
        scale_path,
        scale_output,
        reference,
        config,
        {**common_tags, "statistic": "gamma_scale", "units": aggregated_units},
        overwrite=True,
    )
    write_float_raster(
        pzero_path,
        pzero_output,
        reference,
        config,
        {**common_tags, "statistic": "zero_probability", "units": "probability"},
        overwrite=True,
    )
    write_count_raster(
        count_path,
        total_count,
        reference,
        config,
        {**common_tags, "statistic": "valid_year_count", "units": "count"},
        overwrite=True,
    )

    LOGGER.info(
        "Wrote gamma climatology for month %02d dekad %d using %d baseline periods",
        month,
        dekad,
        used_periods,
    )


def ensure_gamma_climatology(
    target_periods: Dict[PeriodKey, Dict],
    baseline_by_key: Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict]]],
    all_daily_files: Dict,
    reference: RasterReference,
    config: dict,
) -> None:
    climatology_cfg = config["climatology"]
    climatology_dir = Path(climatology_cfg["directory"])
    compute = bool(climatology_cfg.get("compute_climatology", False))
    overwrite = bool(climatology_cfg.get("overwrite", False))
    climatology_dir.mkdir(parents=True, exist_ok=True)

    needed_keys = sorted({period.climatology_key for period in target_periods})
    missing_keys: List[Tuple[int, int]] = []

    for month, dekad in needed_keys:
        shape_path, scale_path, pzero_path, count_path = gamma_climatology_paths(
            climatology_dir,
            month,
            dekad,
        )
        exists = (
            shape_path.exists()
            and scale_path.exists()
            and pzero_path.exists()
            and count_path.exists()
        )
        if overwrite or not exists:
            missing_keys.append((month, dekad))

    if missing_keys and not compute:
        missing_text = ", ".join(f"m{month:02d}-d{dekad}" for month, dekad in missing_keys)
        raise FileNotFoundError(
            "Gamma climatology is missing for the following periods and compute_climatology is false: "
            f"{missing_text}"
        )

    for key in missing_keys:
        periods = baseline_by_key.get(key, [])
        if not periods:
            raise RuntimeError(
                f"Cannot compute gamma climatology for month {key[0]:02d}, dekad {key[1]}: "
                "no baseline periods are available"
            )
        compute_gamma_climatology_for_key(key, periods, all_daily_files, reference, config)


def compute_all_gamma_climatology(
    baseline_by_key: Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict]]],
    all_daily_files: Dict,
    reference: RasterReference,
    config: dict,
) -> None:
    climatology_dir = Path(config["climatology"]["directory"])
    climatology_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_by_key:
        raise RuntimeError("No input periods fall inside the configured climatology baseline")

    for key in sorted(baseline_by_key):
        compute_gamma_climatology_for_key(key, baseline_by_key[key], all_daily_files, reference, config)


def compute_gamma_anomalies(
    target_periods: Dict[PeriodKey, Dict],
    all_daily_files: Dict,
    reference: RasterReference,
    config: dict,
) -> None:
    output_cfg = config["output"]
    climatology_cfg = config["climatology"]
    output_dir = Path(output_cfg["directory"])
    climatology_dir = Path(climatology_cfg["directory"])
    filename_template = output_cfg.get(
        "filename_template",
        "iwr_spi_{year}{month:02d}_d{dekad}_gamma.tif",
    )
    overwrite = bool(output_cfg.get("overwrite", False))
    cdf_clip = float(climatology_cfg.get("cdf_clip", 1.0e-8))
    aggregation_label, aggregation_token, _ = aggregation_context(config)

    climatology_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    written = 0
    skipped = 0

    for period, _ in sorted(target_periods.items()):
        window_start, window_end = aggregation_window_bounds(period, config)
        output_path = output_dir / filename_template.format(
            year=period.year,
            month=period.month,
            dekad=period.dekad,
            start=window_start.strftime("%Y%m%d"),
            end=window_end.strftime("%Y%m%d"),
        )

        if output_path.exists() and not overwrite:
            LOGGER.info("Output exists, skipping: %s", output_path)
            skipped += 1
            continue

        aggregated = aggregate_period(period, all_daily_files, reference, config)
        if aggregated is None:
            skipped += 1
            continue

        save_aggregated_period(period, aggregated, reference, config)

        key = period.climatology_key
        if key not in climatology_cache:
            shape_path, scale_path, pzero_path, _ = gamma_climatology_paths(climatology_dir, *key)
            climatology_cache[key] = (
                read_float_raster(shape_path, reference),
                read_float_raster(scale_path, reference),
                read_float_raster(pzero_path, reference),
            )

        climatology_shape, climatology_scale, climatology_pzero = climatology_cache[key]
        anomaly = gamma_cdf_to_standard_normal(
            aggregated,
            climatology_shape,
            climatology_scale,
            climatology_pzero,
            cdf_clip,
        )

        write_float_raster(
            output_path,
            anomaly,
            reference,
            config,
            {
                "variable": "iwr_gamma_spi_anomaly",
                "distribution": "gamma",
                "zero_probability_included": True,
                "units": "standard_normal",
                "aggregation": aggregation_label,
                "aggregation_operation": "sum",
                "period_start": window_start.isoformat(),
                "period_end": window_end.isoformat(),
                "release_step_month": period.month,
                "release_step_dekad": period.dekad,
                "climatology_month": period.month,
                "climatology_dekad": period.dekad,
                "baseline_start_year": climatology_cfg.get("start_year", "all"),
                "baseline_end_year": climatology_cfg.get("end_year", "all"),
                "gamma_formula": (
                    f"norm.ppf(p_zero + (1-p_zero)*gamma_cdf({aggregation_token}; shape, scale))"
                ),
                "positive_interpretation": "above-normal irrigation water requirement",
            },
            overwrite=True,
        )
        LOGGER.info("Wrote gamma anomaly: %s", output_path)
        written += 1

    LOGGER.info(
        "Finished: %d gamma anomaly rasters written, %d periods skipped",
        written,
        skipped,
    )


def run(config_path: Path, climatology_only: bool = False) -> None:
    config = load_config(config_path)
    configure_logging(config.get("processing", {}).get("log_level", "INFO"))

    files_by_date = discover_daily_files(config)
    groups = build_period_groups(files_by_date, config)
    baseline_by_key = select_baseline_periods(groups, config)

    first_path = files_by_date[min(files_by_date)]
    reference = get_reference(first_path)

    if climatology_only:
        LOGGER.info(
            "Climatology-only mode enabled; computing gamma climatology rasters without anomaly outputs"
        )
        LOGGER.info(
            "Climatology baseline: %s-%s",
            config["climatology"].get("start_year", "all"),
            config["climatology"].get("end_year", "all"),
        )
        compute_all_gamma_climatology(baseline_by_key, files_by_date, reference, config)
        return

    target_periods = select_target_periods(groups, config)

    LOGGER.info("Target periods: %d", len(target_periods))
    LOGGER.info(
        "Climatology baseline: %s-%s",
        config["climatology"].get("start_year", "all"),
        config["climatology"].get("end_year", "all"),
    )

    ensure_gamma_climatology(target_periods, baseline_by_key, files_by_date, reference, config)
    compute_gamma_anomalies(target_periods, files_by_date, reference, config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate daily IWR GeoTIFFs and compute pixel-wise gamma-based SPI-like "
            "anomalies with configurable release_step and aggregation windows."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the JSON configuration file")
    parser.add_argument(
        "--climatology-only",
        action="store_true",
        help="Compute gamma climatology rasters only and skip anomaly generation.",
    )
    args = parser.parse_args()

    try:
        run(args.config, climatology_only=args.climatology_only)
    except Exception as exc:
        LOGGER.exception("Processing failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())