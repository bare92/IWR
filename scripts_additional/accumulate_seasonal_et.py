#!/usr/bin/env python3
"""Accumulate daily ETx and stress-limited ETa over pixel-specific seasons.

The script is designed for the daily GeoTIFF outputs written by the attached
IWR model. It reads the ASAP phenology paths and model-output location from the
same JSON configuration used for the run, then writes one ETx total and one ETa
total for each calendar year, combining both phenological seasons in that year.

ASAP timings use an extended 1--108 dekad axis. A season year is defined here
as the calendar year containing that pixel's start of season (SOS). Therefore,
a season beginning in October 2000 and ending in April 2001 is labelled 2000.

Default output layout::

    <output_base>/<run_name>/Seasonal_ET/
        ETx/etx_sum_1993.tif
        ETa_stress/eta_stress_sum_1993.tif

Only complete pixel-year totals are written as valid data. A pixel is set to
nodata when the input time series does not cover the relevant SOS--EOS interval
for either season in that year, or either daily ET input is invalid on any
active day.

Example::

    python accumulate_seasonal_et.py \
        --config config/config_eraL_theoretical_rainfed.json
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


DATE_PATTERNS = (
    re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)"),
    re.compile(r"(?<!\d)((?:19|20)\d{2}-\d{2}-\d{2})(?!\d)"),
)
NEEDED_PHENOLOGY_KEYS = (
    "phenonseasons",
    "phenos1",
    "phenoe1",
    "phenos2",
    "phenoe2",
)


@dataclass(frozen=True)
class GridSpec:
    """Spatial grid used to validate all input rasters."""

    height: int
    width: int
    transform: object
    crs: object


@dataclass(frozen=True)
class SeasonInfo:
    """Validated, precomputed ASAP timing arrays for one season."""

    number: int
    valid: np.ndarray
    sos: np.ndarray
    eos: np.ndarray
    sos_block: np.ndarray
    eos_block: np.ndarray
    sos_calendar_dekad: np.ndarray
    eos_calendar_dekad: np.ndarray


@dataclass
class SeasonalAccumulator:
    """Sums and quality flags for one season number and SOS year."""

    etx_sum: np.ndarray
    eta_sum: np.ndarray
    active_day_count: np.ndarray
    bad_etx: np.ndarray
    bad_eta: np.ndarray


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sum daily ETx and stress-limited ETa within each pixel's ASAP "
            "phenological season."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="IWR JSON configuration containing phenology and output paths.",
    )
    parser.add_argument(
        "--etx-dir",
        type=Path,
        default=None,
        help="Override the default <run output>/ETx directory.",
    )
    parser.add_argument(
        "--eta-dir",
        type=Path,
        default=None,
        help="Override the default <run output>/ETa_stress directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the default <run output>/Seasonal_ET directory.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First SOS year to write (default: all complete seasons).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last SOS year to write, inclusive (default: all complete seasons).",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Nodata value for output rasters (default: -9999).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace seasonal GeoTIFFs that already exist.",
    )
    parser.add_argument(
        "--progress-days",
        type=int,
        default=365,
        help="Print progress every N input days; 0 disables it (default: 365).",
    )
    args = parser.parse_args(argv)

    if args.start_year is not None and args.end_year is not None:
        if args.start_year > args.end_year:
            parser.error("--start-year must be <= --end-year")
    if args.progress_days < 0:
        parser.error("--progress-days must be >= 0")
    if not np.isfinite(args.nodata):
        parser.error("--nodata must be finite")
    return args


def _nested(config: dict, *keys: str):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(keys)
            raise KeyError(f"Missing required configuration key: {joined}")
        value = value[key]
    return value


def _resolve_config_path(raw_path: str | Path, config_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Path], Path, Path, Path]:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    config_dir = config_path.parent
    raw_phenology = _nested(config, "datasets", "static", "phenology_paths")
    missing = [key for key in NEEDED_PHENOLOGY_KEYS if key not in raw_phenology]
    if missing:
        raise KeyError(f"Missing phenology path(s) in configuration: {missing}")
    phenology_paths = {
        key: _resolve_config_path(raw_phenology[key], config_dir)
        for key in NEEDED_PHENOLOGY_KEYS
    }

    output_base = _resolve_config_path(
        _nested(config, "outputs", "output_base"), config_dir
    )
    run_name = str(_nested(config, "outputs", "run_name"))
    run_output = output_base / run_name

    etx_dir = args.etx_dir.expanduser().resolve() if args.etx_dir else run_output / "ETx"
    eta_dir = args.eta_dir.expanduser().resolve() if args.eta_dir else run_output / "ETa_stress"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_output / "Seasonal_ET"
    )
    return phenology_paths, etx_dir, eta_dir, output_dir


def _date_from_filename(path: Path) -> date:
    matches: set[date] = set()
    for pattern in DATE_PATTERNS:
        for token in pattern.findall(path.stem):
            try:
                matches.add(date.fromisoformat(token) if "-" in token else date(
                    int(token[0:4]), int(token[4:6]), int(token[6:8])
                ))
            except ValueError as exc:
                raise ValueError(f"Invalid date token '{token}' in {path.name}") from exc

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one YYYYMMDD or YYYY-MM-DD date in {path.name}; "
            f"found {len(matches)}."
        )
    return next(iter(matches))


def index_daily_geotiffs(folder: Path, label: str) -> dict[date, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {folder}")

    index: dict[date, Path] = {}
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {label} directory: {folder}")

    for path in paths:
        raster_date = _date_from_filename(path)
        if raster_date in index:
            raise ValueError(
                f"Duplicate {label} raster date {raster_date}: "
                f"{index[raster_date]} and {path}"
            )
        index[raster_date] = path
    return index


def validate_daily_indices(
    etx_index: dict[date, Path], eta_index: dict[date, Path]
) -> list[date]:
    etx_dates = set(etx_index)
    eta_dates = set(eta_index)
    if etx_dates != eta_dates:
        only_etx = sorted(etx_dates - eta_dates)
        only_eta = sorted(eta_dates - etx_dates)
        raise ValueError(
            "ETx and ETa_stress do not contain the same dates. "
            f"Only in ETx (first 10): {only_etx[:10]}; "
            f"only in ETa_stress (first 10): {only_eta[:10]}."
        )

    dates = sorted(etx_dates)
    expected_days = (dates[-1] - dates[0]).days + 1
    if len(dates) != expected_days:
        available = set(dates)
        missing: list[date] = []
        current = dates[0]
        while current <= dates[-1] and len(missing) < 20:
            if current not in available:
                missing.append(current)
            current += timedelta(days=1)
        raise ValueError(
            "The common ET time series is not daily-continuous. "
            f"First missing dates: {missing}."
        )
    return dates


def _grid_from_dataset(src: rasterio.io.DatasetReader) -> GridSpec:
    return GridSpec(src.height, src.width, src.transform, src.crs)


def _check_grid(src: rasterio.io.DatasetReader, reference: GridSpec, path: Path) -> None:
    if (src.height, src.width) != (reference.height, reference.width):
        raise ValueError(
            f"Raster shape mismatch for {path}: found {(src.height, src.width)}, "
            f"expected {(reference.height, reference.width)}."
        )
    if not src.transform.almost_equals(reference.transform):
        raise ValueError(f"Raster transform mismatch for {path}")
    if src.crs != reference.crs:
        raise ValueError(
            f"Raster CRS mismatch for {path}: found {src.crs}, expected {reference.crs}"
        )


def _valid_values(array: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        valid &= array != nodata
    valid &= np.abs(array) < 1.0e19
    return valid


def read_et_day(path: Path, reference: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one band in {path}; found {src.count}")
        _check_grid(src, reference, path)
        data = src.read(1).astype(np.float32)
        valid = _valid_values(data, src.nodata)

    strongly_negative = valid & (data < -1.0e-5)
    if np.any(strongly_negative):
        minimum = float(np.min(data[strongly_negative]))
        raise ValueError(f"Negative ET value in {path}: minimum={minimum}")
    tiny_negative = valid & (data < 0.0)
    data[tiny_negative] = 0.0
    return data, valid


def load_phenology(
    paths: dict[str, Path], reference: GridSpec
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Phenology raster not found ({key}): {path}")
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(f"Expected one band in {path}; found {src.count}")
            _check_grid(src, reference, path)
            array = src.read(1).astype(np.float32)
            arrays[key] = array
            validity[key] = _valid_values(array, src.nodata)
    return arrays, validity


def prepare_season_info(
    number: int,
    phenology: dict[str, np.ndarray],
    phenology_validity: dict[str, np.ndarray],
) -> SeasonInfo:
    nseasons = phenology["phenonseasons"]
    sos_key = f"phenos{number}"
    eos_key = f"phenoe{number}"
    sos_raw = phenology[sos_key]
    eos_raw = phenology[eos_key]

    integer_timing = (
        np.isclose(sos_raw, np.rint(sos_raw), atol=1.0e-4)
        & np.isclose(eos_raw, np.rint(eos_raw), atol=1.0e-4)
    )
    valid = (
        phenology_validity["phenonseasons"]
        & phenology_validity[sos_key]
        & phenology_validity[eos_key]
        & (nseasons >= number)
        & integer_timing
        & (sos_raw >= 1)
        & (sos_raw <= 108)
        & (eos_raw >= 1)
        & (eos_raw <= 108)
        & (sos_raw <= eos_raw)
    )

    sos = np.where(valid, np.rint(sos_raw), 1).astype(np.int16)
    eos = np.where(valid, np.rint(eos_raw), 1).astype(np.int16)

    # A recurrent season lasting one year or longer cannot be assigned
    # unambiguously to separate annual occurrences.
    overlong = valid & ((eos - sos) >= 36)
    if np.any(overlong):
        raise ValueError(
            f"Season {number} has {int(np.sum(overlong))} pixel(s) with a "
            "SOS--EOS interval of at least 36 dekads. Annual occurrences "
            "would overlap and cannot be accumulated unambiguously."
        )

    declared = phenology_validity["phenonseasons"] & (nseasons >= number)
    invalid_declared = declared & ~valid
    if np.any(invalid_declared):
        print(
            f"Warning: season {number} is declared but has invalid SOS/EOS at "
            f"{int(np.sum(invalid_declared))} pixel(s); those pixels will be nodata.",
            file=sys.stderr,
        )

    sos_block = ((sos - 1) // 36).astype(np.int8)
    eos_block = ((eos - 1) // 36).astype(np.int8)
    return SeasonInfo(
        number=number,
        valid=valid,
        sos=sos,
        eos=eos,
        sos_block=sos_block,
        eos_block=eos_block,
        sos_calendar_dekad=(((sos - 1) % 36) + 1).astype(np.int8),
        eos_calendar_dekad=(((eos - 1) % 36) + 1).astype(np.int8),
    )
def date_to_dekad(current_date: date) -> int:
    if current_date.day <= 10:
        part = 1
    elif current_date.day <= 20:
        part = 2
    else:
        part = 3
    return (current_date.month - 1) * 3 + part


def season_contributions(
    current_date: date, info: SeasonInfo
) -> tuple[list[tuple[int, np.ndarray]], np.ndarray]:
    """Return (SOS year, mask) contributions and today's union active mask."""
    calendar_dekad = date_to_dekad(current_date)
    contributions: list[tuple[int, np.ndarray]] = []
    active_today = np.zeros(info.valid.shape, dtype=bool)

    for candidate_block in range(3):
        extended_dekad = calendar_dekad + 36 * candidate_block
        active = (
            info.valid
            & (extended_dekad >= info.sos)
            & (extended_dekad <= info.eos)
        )
        if not np.any(active):
            continue

        # SOS blocks may vary spatially. Split them so every mask has one
        # unambiguous start-year label.
        for start_block in range(3):
            mask = active & (info.sos_block == start_block)
            if not np.any(mask):
                continue
            season_year = current_date.year - (candidate_block - start_block)
            contributions.append((season_year, mask))
            active_today |= mask

    return contributions, active_today


def new_accumulator(shape: tuple[int, int]) -> SeasonalAccumulator:
    return SeasonalAccumulator(
        etx_sum=np.zeros(shape, dtype=np.float64),
        eta_sum=np.zeros(shape, dtype=np.float64),
        active_day_count=np.zeros(shape, dtype=np.uint16),
        bad_etx=np.zeros(shape, dtype=bool),
        bad_eta=np.zeros(shape, dtype=bool),
    )


def _year_selected(year: int, args: argparse.Namespace) -> bool:
    return not (
        (args.start_year is not None and year < args.start_year)
        or (args.end_year is not None and year > args.end_year)
    )


def accumulate(
    dates: list[date],
    etx_index: dict[date, Path],
    eta_index: dict[date, Path],
    reference: GridSpec,
    seasons: tuple[SeasonInfo, SeasonInfo],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int], SeasonalAccumulator], int]:
    accumulators: dict[tuple[int, int], SeasonalAccumulator] = {}
    overlap_pixel_days = 0
    shape = (reference.height, reference.width)

    for day_number, current_date in enumerate(dates, start=1):
        etx, etx_valid = read_et_day(etx_index[current_date], reference)
        eta, eta_valid = read_et_day(eta_index[current_date], reference)

        active_masks: list[np.ndarray] = []
        for info in seasons:
            contributions, active_today = season_contributions(current_date, info)
            active_masks.append(active_today)

            for season_year, mask in contributions:
                if not _year_selected(season_year, args):
                    continue
                key = (info.number, season_year)
                accumulator = accumulators.get(key)
                if accumulator is None:
                    accumulator = new_accumulator(shape)
                    accumulators[key] = accumulator

                accumulator.active_day_count[mask] += 1

                good_etx = mask & etx_valid
                good_eta = mask & eta_valid
                accumulator.etx_sum[good_etx] += etx[good_etx]
                accumulator.eta_sum[good_eta] += eta[good_eta]
                accumulator.bad_etx |= mask & ~etx_valid
                accumulator.bad_eta |= mask & ~eta_valid

        overlap_pixel_days += int(np.count_nonzero(active_masks[0] & active_masks[1]))

        if args.progress_days and (
            day_number == 1
            or day_number % args.progress_days == 0
            or day_number == len(dates)
        ):
            print(
                f"Processed {day_number:>6}/{len(dates)} days "
                f"({current_date.isoformat()})"
            )

    return accumulators, overlap_pixel_days


def _dekad_boundary_ordinal(year: int, dekad: int, end: bool) -> int:
    month = (dekad - 1) // 3 + 1
    part = (dekad - 1) % 3
    if end:
        day = (10, 20, calendar.monthrange(year, month)[1])[part]
    else:
        day = (1, 11, 21)[part]
    return date(year, month, day).toordinal()


def complete_season_mask(
    info: SeasonInfo,
    season_year: int,
    first_input_date: date,
    last_input_date: date,
    observed_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return completeness mask and expected active-day count per pixel."""
    start_table = np.zeros(37, dtype=np.int32)
    for dekad in range(1, 37):
        start_table[dekad] = _dekad_boundary_ordinal(season_year, dekad, end=False)
    start_ordinal = start_table[info.sos_calendar_dekad]

    span = info.eos_block - info.sos_block
    end_ordinal = np.zeros(info.valid.shape, dtype=np.int32)
    for year_span in range(3):
        mask = info.valid & (span == year_span)
        if not np.any(mask):
            continue
        end_table = np.zeros(37, dtype=np.int32)
        for dekad in range(1, 37):
            end_table[dekad] = _dekad_boundary_ordinal(
                season_year + year_span, dekad, end=True
            )
        end_ordinal[mask] = end_table[info.eos_calendar_dekad[mask]]

    expected_days_i32 = end_ordinal - start_ordinal + 1
    expected_days = np.clip(expected_days_i32, 0, np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    covered_by_input_range = (
        (start_ordinal >= first_input_date.toordinal())
        & (end_ordinal <= last_input_date.toordinal())
    )
    complete = (
        info.valid
        & covered_by_input_range
        & (expected_days_i32 > 0)
        & (observed_days == expected_days)
    )
    return complete, expected_days


def _output_profile(reference_profile: dict, nodata: float) -> dict:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=float(nodata),
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    return profile


def write_sum(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    profile: dict,
    nodata: float,
    tags: dict[str, str],
    band_description: str,
    overwrite: bool,
) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Use --overwrite to replace existing files."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.full(values.shape, nodata, dtype=np.float32)
    output[valid] = values[valid].astype(np.float32)
    with rasterio.open(path, "w", **_output_profile(profile, nodata)) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, band_description)
        dst.update_tags(**tags)
    return int(np.count_nonzero(valid))


def aggregate_yearly_season_totals(
    accumulators: dict[tuple[int, int], SeasonalAccumulator],
    seasons: tuple[SeasonInfo, SeasonInfo],
    dates: list[date],
) -> dict[int, dict[str, np.ndarray]]:
    season_lookup = {info.number: info for info in seasons}
    yearly_totals: dict[int, dict[str, np.ndarray]] = {}

    for (season_number, season_year), accumulator in accumulators.items():
        info = season_lookup[season_number]
        complete, _ = complete_season_mask(
            info=info,
            season_year=season_year,
            first_input_date=dates[0],
            last_input_date=dates[-1],
            observed_days=accumulator.active_day_count,
        )
        etx_valid = complete & ~accumulator.bad_etx
        eta_valid = complete & ~accumulator.bad_eta

        accumulated = yearly_totals.setdefault(
            season_year,
            {
                "etx_sum": np.zeros(accumulator.etx_sum.shape, dtype=np.float64),
                "eta_sum": np.zeros(accumulator.eta_sum.shape, dtype=np.float64),
                "etx_valid": np.zeros(accumulator.etx_sum.shape, dtype=bool),
                "eta_valid": np.zeros(accumulator.eta_sum.shape, dtype=bool),
            },
        )

        accumulated["etx_sum"][etx_valid] += accumulator.etx_sum[etx_valid]
        accumulated["eta_sum"][eta_valid] += accumulator.eta_sum[eta_valid]
        accumulated["etx_valid"] |= etx_valid
        accumulated["eta_valid"] |= eta_valid

    return yearly_totals


def write_outputs(
    accumulators: dict[tuple[int, int], SeasonalAccumulator],
    seasons: tuple[SeasonInfo, SeasonInfo],
    reference_profile: dict,
    dates: list[date],
    output_dir: Path,
    etx_dir: Path,
    eta_dir: Path,
    args: argparse.Namespace,
    overlap_pixel_days: int,
) -> tuple[int, int]:
    yearly_totals = aggregate_yearly_season_totals(
        accumulators=accumulators,
        seasons=seasons,
        dates=dates,
    )

    files_written = 0
    skipped_empty = 0

    for season_year in sorted(yearly_totals):
        annual = yearly_totals[season_year]
        etx_valid = annual["etx_valid"]
        eta_valid = annual["eta_valid"]

        if not np.any(etx_valid) and not np.any(eta_valid):
            skipped_empty += 1
            print(
                f"Skipping SOS year {season_year}: "
                "no complete valid pixel-years after combining both seasons."
            )
            continue

        common_tags = {
            "aggregation": "sum",
            "units": "mm/year",
            "season_year": str(season_year),
            "season_year_definition": "calendar year containing pixel-specific SOS for both seasons",
            "season_index": "1+2 combined",
            "season_start": "phenos1 and phenos2 combined (inclusive)",
            "season_end": "phenoe1 and phenoe2 combined (inclusive)",
            "phenology_time_axis": "ASAP extended dekads 1-108",
            "completeness_rule": (
                "valid only when the relevant pixel-season intervals are fully "
                "covered and all active daily ET values are valid"
            ),
            "input_first_date": dates[0].isoformat(),
            "input_last_date": dates[-1].isoformat(),
            "two_season_overlap_policy": (
                "daily ET is included in both season totals where seasons overlap; "
                "annual totals sum both season totals for the same year"
            ),
            "overlap_pixel_days_in_input": str(overlap_pixel_days),
            "mode": "annual_total_including_both_seasons",
        }

        etx_path = output_dir / "ETx" / f"etx_sum_{season_year}.tif"
        eta_path = output_dir / "ETa_stress" / f"eta_stress_sum_{season_year}.tif"

        if not args.overwrite:
            existing = [path for path in (etx_path, eta_path) if path.exists()]
            if existing:
                raise FileExistsError(
                    "Output file(s) already exist: "
                    + ", ".join(str(path) for path in existing)
                    + ". Use --overwrite to replace them."
                )

        if np.any(etx_valid):
            tags = dict(common_tags)
            tags.update(
                variable="annual_potential_crop_evapotranspiration_without_stress",
                short_name="ETx",
                source_directory=str(etx_dir),
            )
            valid_count = write_sum(
                path=etx_path,
                values=annual["etx_sum"],
                valid=etx_valid,
                profile=reference_profile,
                nodata=args.nodata,
                tags=tags,
                band_description="Annual ETx sum including both seasons (mm/year)",
                overwrite=args.overwrite,
            )
            files_written += 1
            print(f"Wrote {etx_path} ({valid_count} valid pixels)")

        if np.any(eta_valid):
            tags = dict(common_tags)
            tags.update(
                variable="annual_actual_crop_evapotranspiration_under_water_stress",
                short_name="ETa",
                source_directory=str(eta_dir),
            )
            valid_count = write_sum(
                path=eta_path,
                values=annual["eta_sum"],
                valid=eta_valid,
                profile=reference_profile,
                nodata=args.nodata,
                tags=tags,
                band_description="Annual stress-limited ETa sum including both seasons (mm/year)",
                overwrite=args.overwrite,
            )
            files_written += 1
            print(f"Wrote {eta_path} ({valid_count} valid pixels)")

    return files_written, skipped_empty


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    phenology_paths, etx_dir, eta_dir, output_dir = load_configuration(args)

    print(f"ETx input:       {etx_dir}")
    print(f"ETa input:       {eta_dir}")
    print(f"Seasonal output: {output_dir}")

    etx_index = index_daily_geotiffs(etx_dir, "ETx")
    eta_index = index_daily_geotiffs(eta_dir, "ETa_stress")
    dates = validate_daily_indices(etx_index, eta_index)
    print(
        f"Input coverage:  {dates[0].isoformat()} to {dates[-1].isoformat()} "
        f"({len(dates)} daily pairs)"
    )

    with rasterio.open(etx_index[dates[0]]) as src:
        if src.count != 1:
            raise ValueError(
                f"Expected one band in {etx_index[dates[0]]}; found {src.count}"
            )
        reference = _grid_from_dataset(src)
        reference_profile = src.profile.copy()

    phenology, phenology_validity = load_phenology(phenology_paths, reference)
    seasons = (
        prepare_season_info(1, phenology, phenology_validity),
        prepare_season_info(2, phenology, phenology_validity),
    )
    print(
        "Phenology pixels: "
        f"season 1={int(np.count_nonzero(seasons[0].valid))}, "
        f"season 2={int(np.count_nonzero(seasons[1].valid))}"
    )

    accumulators, overlap_pixel_days = accumulate(
        dates=dates,
        etx_index=etx_index,
        eta_index=eta_index,
        reference=reference,
        seasons=seasons,
        args=args,
    )
    if overlap_pixel_days:
        print(
            "Warning: phenological seasons overlap for "
            f"{overlap_pixel_days} pixel-day(s). Because the daily model output "
            "does not separate overlapping seasons, those daily ET values are "
            "included in both seasonal totals.",
            file=sys.stderr,
        )

    files_written, skipped_empty = write_outputs(
        accumulators=accumulators,
        seasons=seasons,
        reference_profile=reference_profile,
        dates=dates,
        output_dir=output_dir,
        etx_dir=etx_dir,
        eta_dir=eta_dir,
        args=args,
        overlap_pixel_days=overlap_pixel_days,
    )
    print(
        f"Done: {files_written} GeoTIFF(s) written; "
        f"{skipped_empty} boundary/incomplete season-year(s) skipped."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
