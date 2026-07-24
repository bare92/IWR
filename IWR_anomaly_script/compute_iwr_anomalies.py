#!/usr/bin/env python3

import argparse
import calendar
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio


LOGGER = logging.getLogger("iwr_anomaly")


@dataclass(frozen=True, order=True)
class PeriodKey:
    year: int
    month: int
    dekad: int

    @property
    def climatology_key(self) -> Tuple[int, int]:
        return self.month, self.dekad

    @property
    def start_date(self) -> date:
        if self.dekad == 0:
            return date(self.year, self.month, 1)
        if self.dekad == 1:
            day = 1
        elif self.dekad == 2:
            day = 11
        else:
            day = 21
        return date(self.year, self.month, day)

    @property
    def end_date(self) -> date:
        if self.dekad == 0:
            return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])
        if self.dekad == 1:
            day = 10
        elif self.dekad == 2:
            day = 20
        else:
            day = calendar.monthrange(self.year, self.month)[1]
        return date(self.year, self.month, day)

    @property
    def expected_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


@dataclass
class RasterReference:
    profile: dict
    width: int
    height: int
    transform: object
    crs: object


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for section in ("input", "aggregation", "climatology", "output"):
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")

    aggregation_cfg = config["aggregation"]

    window_value = aggregation_cfg.get("window", "dekad")
    # Accept both "month" and "monthly" for backward compatibility.
    if window_value == "monthly":
        window_value = "month"
        aggregation_cfg["window"] = "month"

    if window_value not in {"dekad", "month"}:
        raise ConfigError("aggregation.window must be 'dekad' or 'month'")

    release_step = aggregation_cfg.get("release_step")
    if release_step is not None and release_step not in {"dekad", "month"}:
        raise ConfigError("aggregation.release_step must be 'dekad' or 'month'")

    effective_release_step = release_step if release_step is not None else window_value

    window_days = aggregation_cfg.get("window_days")
    if window_days is not None:
        if int(window_days) <= 0:
            raise ConfigError("aggregation.window_days must be a positive integer")

    output_cfg = config["output"]
    output_template = output_cfg.get(
        "filename_template",
        "iwr_zscore_{year}{month:02d}_d{dekad}.tif",
    )
    aggregated_template = output_cfg.get(
        "aggregated_filename_template",
        "iwr_sum_{year}{month:02d}_d{dekad}.tif",
    )

    if effective_release_step == "dekad":
        if "{dekad}" not in output_template:
            raise ConfigError(
                "output.filename_template must include '{dekad}' when aggregation.release_step='dekad'"
            )

        if bool(output_cfg.get("save_aggregated_iwr", False)) and "{dekad}" not in aggregated_template:
            raise ConfigError(
                "output.aggregated_filename_template must include '{dekad}' when "
                "aggregation.release_step='dekad' and save_aggregated_iwr=true"
            )

    if config["aggregation"].get("operation", "sum") != "sum":
        raise ConfigError("This version supports aggregation.operation='sum' only")

    policy = config["aggregation"].get("incomplete_period_policy", "skip")
    if policy not in {"skip", "error"}:
        raise ConfigError("incomplete_period_policy must be 'skip' or 'error'")

    valid_fraction = float(config["aggregation"].get("minimum_valid_fraction", 1.0))
    if not 0.0 < valid_fraction <= 1.0:
        raise ConfigError("minimum_valid_fraction must be in the interval (0, 1]")

    ddof = int(config["climatology"].get("ddof", 1))
    if ddof < 0:
        raise ConfigError("climatology.ddof must be >= 0")

    zero_std_policy = config["climatology"].get("zero_std_policy", "nodata")
    if zero_std_policy not in {"nodata", "zero"}:
        raise ConfigError("zero_std_policy must be 'nodata' or 'zero'")


def parse_optional_date(value: Optional[str]) -> Optional[date]:
    if value in (None, ""):
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def dekad_from_day(day: int) -> int:
    if day <= 10:
        return 1
    if day <= 20:
        return 2
    return 3


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def discover_daily_files(config: dict) -> Dict[date, Path]:
    input_cfg = config["input"]
    input_dir = Path(input_cfg["directory"])
    glob_pattern = input_cfg.get("glob", "iwr_*.tif")
    regex = re.compile(input_cfg.get("date_regex", r"iwr_(?P<date>\d{8})\.tif$"))
    date_format = input_cfg.get("date_format", "%Y%m%d")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files_by_date: Dict[date, Path] = {}
    unmatched: List[Path] = []

    for path in sorted(input_dir.glob(glob_pattern)):
        match = regex.search(path.name)
        if not match:
            unmatched.append(path)
            continue

        try:
            file_date = datetime.strptime(match.group("date"), date_format).date()
        except (ValueError, IndexError) as exc:
            raise ConfigError(
                f"Cannot parse date from '{path.name}' using regex '{regex.pattern}' "
                f"and date format '{date_format}'"
            ) from exc

        if file_date in files_by_date:
            raise RuntimeError(
                f"Duplicate input date {file_date}: {files_by_date[file_date]} and {path}"
            )
        files_by_date[file_date] = path

    if unmatched:
        LOGGER.debug("Ignored %d files that did not match the date regex", len(unmatched))

    if not files_by_date:
        raise FileNotFoundError(
            f"No input files matching '{glob_pattern}' and the configured date regex were found in {input_dir}"
        )

    LOGGER.info(
        "Discovered %d daily rasters from %s to %s",
        len(files_by_date),
        min(files_by_date),
        max(files_by_date),
    )
    return files_by_date


def build_period_groups(files_by_date: Dict[date, Path], config: dict) -> Dict[PeriodKey, Dict[date, Path]]:
    aggregation_cfg = config["aggregation"]
    release_step = aggregation_cfg.get("release_step")
    if release_step is None:
        # Backward-compatible behavior: if no explicit release_step is set,
        # the release cadence follows aggregation.window.
        release_step = aggregation_cfg.get("window", "dekad")

    groups: Dict[PeriodKey, Dict[date, Path]] = defaultdict(dict)
    for file_date, path in files_by_date.items():
        if release_step == "month":
            key = PeriodKey(file_date.year, file_date.month, 0)
        else:
            key = PeriodKey(file_date.year, file_date.month, dekad_from_day(file_date.day))
        groups[key][file_date] = path
    return dict(groups)


def aggregation_window_bounds(period: PeriodKey, config: dict) -> Tuple[date, date]:
    aggregation_cfg = config["aggregation"]
    window_days = aggregation_cfg.get("window_days")

    if window_days is None:
        return period.start_date, period.end_date

    end_date = period.end_date
    start_date = end_date - timedelta(days=int(window_days) - 1)
    return start_date, end_date


def aggregation_context(config: dict) -> Tuple[str, str, str]:
    """Return aggregation label, formula token, and units suffix."""
    aggregation_cfg = config["aggregation"]
    window_days = aggregation_cfg.get("window_days")
    if window_days is not None:
        days = int(window_days)
        return f"rolling_{days}d", f"rolling_{days}day_sum", f"mm/{days}day"

    window = aggregation_cfg.get("window", "dekad")
    if window == "month":
        return "month", "monthly_sum", "mm/month"
    return "dekad", "dekadal_sum", "mm/dekad"


def get_reference(path: Path) -> RasterReference:
    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError(f"Expected one band in {path}, found {src.count}")
        profile = src.profile.copy()
        return RasterReference(
            profile=profile,
            width=src.width,
            height=src.height,
            transform=src.transform,
            crs=src.crs,
        )


def validate_raster_against_reference(src: rasterio.io.DatasetReader, path: Path, reference: RasterReference) -> None:
    if src.count != 1:
        raise RuntimeError(f"Expected one band in {path}, found {src.count}")
    if src.width != reference.width or src.height != reference.height:
        raise RuntimeError(
            f"Grid size mismatch in {path}: {(src.width, src.height)} != "
            f"{(reference.width, reference.height)}"
        )
    if src.transform != reference.transform:
        raise RuntimeError(f"Geotransform mismatch in {path}")
    if src.crs != reference.crs:
        raise RuntimeError(f"CRS mismatch in {path}")


def valid_data_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(data)
    if nodata is not None:
        if np.isnan(nodata):
            mask &= ~np.isnan(data)
        else:
            mask &= data != nodata
    return mask


def handle_incomplete_period(message: str, config: dict) -> None:
    policy = config["aggregation"].get("incomplete_period_policy", "skip")
    if policy == "error":
        raise RuntimeError(message)
    LOGGER.warning("%s; skipping period", message)


def aggregate_period(
    period: PeriodKey,
    all_daily_files: Dict[date, Path],
    reference: RasterReference,
    config: dict,
) -> Optional[np.ndarray]:
    aggregation_cfg = config["aggregation"]
    require_complete = bool(aggregation_cfg.get("require_complete_period", True))
    minimum_valid_fraction = float(aggregation_cfg.get("minimum_valid_fraction", 1.0))

    window_start, window_end = aggregation_window_bounds(period, config)
    expected_dates = list(daterange(window_start, window_end))
    missing_dates = [day for day in expected_dates if day not in all_daily_files]

    if missing_dates and require_complete:
        missing_text = ", ".join(day.isoformat() for day in missing_dates)
        handle_incomplete_period(
            (
                f"Incomplete {period} aggregation window "
                f"({window_start} to {window_end}): missing {len(missing_dates)} day(s): {missing_text}"
            ),
            config,
        )
        return None

    available_dates = [day for day in expected_dates if day in all_daily_files]
    if not available_dates:
        handle_incomplete_period(f"No daily rasters available for {period}", config)
        return None

    total = np.zeros((reference.height, reference.width), dtype=np.float64)
    valid_count = np.zeros((reference.height, reference.width), dtype=np.uint16)

    for day in available_dates:
        path = all_daily_files[day]
        with rasterio.open(path) as src:
            validate_raster_against_reference(src, path, reference)
            data = src.read(1).astype(np.float64, copy=False)
            valid = valid_data_mask(data, src.nodata)
            total[valid] += data[valid]
            valid_count[valid] += 1

    denominator = len(expected_dates) if require_complete else len(available_dates)
    minimum_valid_days = max(1, math.ceil(denominator * minimum_valid_fraction))
    valid_period = valid_count >= minimum_valid_days

    aggregated = np.full((reference.height, reference.width), np.nan, dtype=np.float64)
    aggregated[valid_period] = total[valid_period]
    return aggregated


def climatology_paths(climatology_dir: Path, month: int, dekad: int) -> Tuple[Path, Path, Path]:
    stem = f"iwr_climatology_m{month:02d}_monthly" if dekad == 0 else f"iwr_climatology_m{month:02d}_d{dekad}"
    return (
        climatology_dir / f"{stem}_mean.tif",
        climatology_dir / f"{stem}_std.tif",
        climatology_dir / f"{stem}_count.tif",
    )


def output_profile(reference: RasterReference, config: dict, dtype: str, nodata: float) -> dict:
    output_cfg = config["output"]
    profile = reference.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress=output_cfg.get("compress", "LZW"),
        tiled=bool(output_cfg.get("tiled", True)),
        BIGTIFF=output_cfg.get("bigtiff", "IF_SAFER"),
    )
    return profile


def write_float_raster(
    path: Path,
    array: np.ndarray,
    reference: RasterReference,
    config: dict,
    tags: dict,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        LOGGER.info("Output exists, skipping: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = float(config["output"].get("nodata", -9999.0))
    dtype = config["output"].get("dtype", "float32")
    profile = output_profile(reference, config, dtype=dtype, nodata=nodata)

    output = np.full(array.shape, nodata, dtype=dtype)
    valid = np.isfinite(array)
    output[valid] = array[valid].astype(dtype, copy=False)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)
        dst.update_tags(**{key: str(value) for key, value in tags.items()})


def write_count_raster(
    path: Path,
    count: np.ndarray,
    reference: RasterReference,
    config: dict,
    tags: dict,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        LOGGER.info("Output exists, skipping: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint16",
        nodata=0,
        compress=config["output"].get("compress", "LZW"),
        tiled=bool(config["output"].get("tiled", True)),
        BIGTIFF=config["output"].get("bigtiff", "IF_SAFER"),
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(count.astype(np.uint16, copy=False), 1)
        dst.update_tags(**{key: str(value) for key, value in tags.items()})


def update_welford(
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> None:
    valid = np.isfinite(values)
    if not np.any(valid):
        return

    old_count = count[valid].astype(np.float64)
    new_count = old_count + 1.0
    delta = values[valid] - mean[valid]
    mean[valid] += delta / new_count
    delta2 = values[valid] - mean[valid]
    m2[valid] += delta * delta2
    count[valid] += 1


def compute_climatology_for_key(
    key: Tuple[int, int],
    periods: List[Tuple[PeriodKey, Dict[date, Path]]],
    all_daily_files: Dict[date, Path],
    reference: RasterReference,
    config: dict,
) -> None:
    month, dekad = key
    climatology_cfg = config["climatology"]
    climatology_dir = Path(climatology_cfg["directory"])
    mean_path, std_path, count_path = climatology_paths(climatology_dir, month, dekad)
    overwrite = bool(climatology_cfg.get("overwrite", False))

    if mean_path.exists() and std_path.exists() and count_path.exists() and not overwrite:
        LOGGER.info("Climatology exists for month %02d dekad %d; reusing it", month, dekad)
        return

    count = np.zeros((reference.height, reference.width), dtype=np.uint16)
    mean = np.zeros((reference.height, reference.width), dtype=np.float64)
    m2 = np.zeros((reference.height, reference.width), dtype=np.float64)

    used_periods = 0
    for period, period_files in sorted(periods, key=lambda item: item[0]):
        aggregated = aggregate_period(period, all_daily_files, reference, config)
        if aggregated is None:
            continue
        update_welford(count, mean, m2, aggregated)
        used_periods += 1

    if used_periods == 0:
        raise RuntimeError(
            f"No usable baseline periods were available for month {month:02d}, dekad {dekad}"
        )

    ddof = int(climatology_cfg.get("ddof", 1))
    minimum_years = int(climatology_cfg.get("minimum_years", max(2, ddof + 1)))
    sufficient = (count >= minimum_years) & (count > ddof)

    mean_output = np.full(mean.shape, np.nan, dtype=np.float64)
    std_output = np.full(mean.shape, np.nan, dtype=np.float64)
    mean_output[sufficient] = mean[sufficient]
    std_output[sufficient] = np.sqrt(m2[sufficient] / (count[sufficient] - ddof))

    start_year = climatology_cfg.get("start_year")
    end_year = climatology_cfg.get("end_year")
    aggregation_label, _, aggregated_units = aggregation_context(config)
    common_tags = {
        "variable": "iwr_climatology",
        "aggregation": aggregation_label,
        "aggregation_operation": "sum",
        "month": month,
        "dekad": dekad,
        "baseline_start_year": start_year if start_year is not None else "all",
        "baseline_end_year": end_year if end_year is not None else "all",
        "ddof": ddof,
        "minimum_years": minimum_years,
        "source_units": "mm/day",
        "aggregated_units": aggregated_units,
    }

    write_float_raster(
        mean_path,
        mean_output,
        reference,
        config,
        {**common_tags, "statistic": "mean", "units": aggregated_units},
        overwrite=True,
    )
    write_float_raster(
        std_path,
        std_output,
        reference,
        config,
        {**common_tags, "statistic": "standard_deviation", "units": aggregated_units},
        overwrite=True,
    )
    write_count_raster(
        count_path,
        count,
        reference,
        config,
        {**common_tags, "statistic": "valid_year_count", "units": "count"},
        overwrite=True,
    )

    LOGGER.info(
        "Wrote climatology for month %02d dekad %d using %d baseline periods",
        month,
        dekad,
        used_periods,
    )


def read_float_raster(path: Path, reference: RasterReference) -> np.ndarray:
    with rasterio.open(path) as src:
        validate_raster_against_reference(src, path, reference)
        data = src.read(1).astype(np.float64, copy=False)
        valid = valid_data_mask(data, src.nodata)
        output = np.full(data.shape, np.nan, dtype=np.float64)
        output[valid] = data[valid]
        return output


def select_target_periods(
    groups: Dict[PeriodKey, Dict[date, Path]],
    config: dict,
) -> Dict[PeriodKey, Dict[date, Path]]:
    processing_cfg = config.get("processing", {})
    start_date = parse_optional_date(processing_cfg.get("start_date"))
    end_date = parse_optional_date(processing_cfg.get("end_date"))

    selected: Dict[PeriodKey, Dict[date, Path]] = {}
    for period, files in groups.items():
        if start_date is not None and period.end_date < start_date:
            continue
        if end_date is not None and period.start_date > end_date:
            continue
        selected[period] = files

    if not selected:
        raise RuntimeError("No aggregation periods fall inside the configured processing date range")
    return selected


def select_baseline_periods(
    groups: Dict[PeriodKey, Dict[date, Path]],
    config: dict,
) -> Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict[date, Path]]]]:
    climatology_cfg = config["climatology"]
    start_year = climatology_cfg.get("start_year")
    end_year = climatology_cfg.get("end_year")

    by_key: Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict[date, Path]]]] = defaultdict(list)
    for period, files in groups.items():
        if start_year is not None and period.year < int(start_year):
            continue
        if end_year is not None and period.year > int(end_year):
            continue
        by_key[period.climatology_key].append((period, files))

    if not by_key:
        raise RuntimeError("No input periods fall inside the configured climatology baseline")
    return dict(by_key)


def ensure_climatology(
    target_periods: Dict[PeriodKey, Dict[date, Path]],
    baseline_by_key: Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict[date, Path]]]],
    all_daily_files: Dict[date, Path],
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
        mean_path, std_path, count_path = climatology_paths(climatology_dir, month, dekad)
        exists = mean_path.exists() and std_path.exists() and count_path.exists()
        if overwrite or not exists:
            missing_keys.append((month, dekad))

    if missing_keys and not compute:
        missing_text = ", ".join(f"m{month:02d}-d{dekad}" for month, dekad in missing_keys)
        raise FileNotFoundError(
            "Climatology is missing for the following periods and compute_climatology is false: "
            f"{missing_text}"
        )

    for key in missing_keys:
        periods = baseline_by_key.get(key, [])
        if not periods:
            raise RuntimeError(
                f"Cannot compute climatology for month {key[0]:02d}, dekad {key[1]}: "
                "no baseline periods are available"
            )
        compute_climatology_for_key(key, periods, all_daily_files, reference, config)


def compute_all_climatology(
    baseline_by_key: Dict[Tuple[int, int], List[Tuple[PeriodKey, Dict[date, Path]]]],
    all_daily_files: Dict[date, Path],
    reference: RasterReference,
    config: dict,
) -> None:
    climatology_cfg = config["climatology"]
    climatology_dir = Path(climatology_cfg["directory"])
    climatology_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_by_key:
        raise RuntimeError("No input periods fall inside the configured climatology baseline")

    for key in sorted(baseline_by_key):
        compute_climatology_for_key(key, baseline_by_key[key], all_daily_files, reference, config)


def save_aggregated_period(
    period: PeriodKey,
    aggregated: np.ndarray,
    reference: RasterReference,
    config: dict,
) -> None:
    output_cfg = config["output"]
    if not bool(output_cfg.get("save_aggregated_iwr", False)):
        return

    directory_value = output_cfg.get("aggregated_directory")
    if not directory_value:
        directory_value = str(Path(output_cfg["directory"]) / "aggregated_iwr")
    directory = Path(directory_value)
    template = output_cfg.get(
        "aggregated_filename_template",
        "iwr_sum_{year}{month:02d}_d{dekad}.tif",
    )
    window_start, window_end = aggregation_window_bounds(period, config)
    aggregation_label, _, aggregated_units = aggregation_context(config)
    path = directory / template.format(
        year=period.year,
        month=period.month,
        dekad=period.dekad,
        start=window_start.strftime("%Y%m%d"),
        end=window_end.strftime("%Y%m%d"),
    )

    write_float_raster(
        path,
        aggregated,
        reference,
        config,
        {
            "variable": "aggregated_iwr",
            "units": aggregated_units,
            "aggregation": aggregation_label,
            "aggregation_operation": "sum",
            "period_start": window_start.isoformat(),
            "period_end": window_end.isoformat(),
            "release_step_month": period.month,
            "release_step_dekad": period.dekad,
        },
        overwrite=bool(output_cfg.get("overwrite", False)),
    )


def compute_anomalies(
    target_periods: Dict[PeriodKey, Dict[date, Path]],
    all_daily_files: Dict[date, Path],
    reference: RasterReference,
    config: dict,
) -> None:
    output_cfg = config["output"]
    climatology_cfg = config["climatology"]
    output_dir = Path(output_cfg["directory"])
    climatology_dir = Path(climatology_cfg["directory"])
    filename_template = output_cfg.get(
        "filename_template",
        "iwr_zscore_{year}{month:02d}_d{dekad}.tif",
    )
    overwrite = bool(output_cfg.get("overwrite", False))
    epsilon = float(climatology_cfg.get("standard_deviation_epsilon", 1.0e-6))
    zero_std_policy = climatology_cfg.get("zero_std_policy", "nodata")
    aggregation_label, aggregation_token, _ = aggregation_context(config)

    climatology_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    written = 0
    skipped = 0

    for period, period_files in sorted(target_periods.items()):
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
            mean_path, std_path, _ = climatology_paths(climatology_dir, *key)
            climatology_cache[key] = (
                read_float_raster(mean_path, reference),
                read_float_raster(std_path, reference),
            )

        climatology_mean, climatology_std = climatology_cache[key]
        valid = (
            np.isfinite(aggregated)
            & np.isfinite(climatology_mean)
            & np.isfinite(climatology_std)
        )
        usable_std = valid & (climatology_std > epsilon)

        anomaly = np.full(aggregated.shape, np.nan, dtype=np.float64)
        anomaly[usable_std] = (
            aggregated[usable_std] - climatology_mean[usable_std]
        ) / climatology_std[usable_std]

        if zero_std_policy == "zero":
            zero_std = valid & (climatology_std <= epsilon)
            anomaly[zero_std] = 0.0

        write_float_raster(
            output_path,
            anomaly,
            reference,
            config,
            {
                "variable": "iwr_zscore_anomaly",
                "units": "standard_deviation",
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
                "zscore_formula": f"({aggregation_token}-climatology_mean)/climatology_std",
                "positive_interpretation": "above-normal irrigation water requirement",
            },
            overwrite=True,
        )
        LOGGER.info("Wrote anomaly: %s", output_path)
        written += 1

    LOGGER.info("Finished: %d anomaly rasters written, %d periods skipped", written, skipped)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ConfigError(f"Invalid log level: {level_name}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
            "Climatology-only mode enabled; computing climatology rasters without anomaly outputs"
        )
        LOGGER.info(
            "Climatology baseline: %s-%s",
            config["climatology"].get("start_year", "all"),
            config["climatology"].get("end_year", "all"),
        )
        compute_all_climatology(baseline_by_key, files_by_date, reference, config)
        return

    target_periods = select_target_periods(groups, config)

    LOGGER.info("Target periods: %d", len(target_periods))
    LOGGER.info(
        "Climatology baseline: %s-%s",
        config["climatology"].get("start_year", "all"),
        config["climatology"].get("end_year", "all"),
    )

    ensure_climatology(target_periods, baseline_by_key, files_by_date, reference, config)
    compute_anomalies(target_periods, files_by_date, reference, config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate daily IWR GeoTIFFs and compute pixel-wise z-score anomalies "
            "with configurable release_step and aggregation windows."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the JSON configuration file")
    parser.add_argument(
        "--climatology-only",
        action="store_true",
        help="Compute climatology rasters only and skip anomaly generation.",
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
