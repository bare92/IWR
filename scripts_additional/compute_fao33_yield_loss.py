#!/usr/bin/env python3
"""Compute seasonal relative yield loss with the FAO-33 relationship.

This script is the post-processing step for ``accumulate_seasonal_et.py``.
For every matching seasonal ETx/ETa pair it applies, crop by crop,

    1 - Ya/Yx = Ky * (1 - ETa/ETx)

and then computes the crop-area-weighted mean relative loss in each pixel.
It also writes the shared water-deficit term, relative yield, crop-specific
loss bands, and static support rasters for the crop fraction and effective Ky.

The crop-fraction GeoTIFF band order must match the row order of the crop CSV.
If all raster bands have descriptions, the names are checked as well. Inputs
may contain fractions (0--1) or percentages (0--100); percentage detection and
small-overshoot normalization mirror the IWR model's preparation logic.

Default input layout::

    <output_base>/<run_name>/Seasonal_ET/
        ETx/etx_sum_season1_1993.tif
        ETa_stress/eta_stress_sum_season1_1993.tif

Default output layout::

    <output_base>/<run_name>/Seasonal_Yield_Loss_FAO33/
        Support/crop_fraction_sum.tif
        Support/effective_ky.tif
        ET_deficit_fraction/et_deficit_fraction_season1_1993.tif
        Yield_loss_fraction/yield_loss_fraction_season1_1993.tif
        Relative_yield_fraction/relative_yield_fraction_season1_1993.tif
        Crop_yield_loss_fraction/crop_yield_loss_fraction_season1_1993.tif

All loss and relative-yield outputs are fractions in [0, 1]. The multiband
crop output contains one band per CSV crop and is nodata where that crop's
fraction is zero. The single-band yield loss is an area-weighted relative loss
over the cropped part of the pixel, not tonnes/ha or tonnes/pixel.

Example::

    python compute_fao33_yield_loss.py \
        --config IWR_scripts/config/config_eraL_theoretical_rainfed.json \
        --start-year 1993 --end-year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio


ETX_PATTERN = re.compile(
    r"^etx_sum(?:_season(?P<season>\d+))?_(?P<year>\d{4})$", re.IGNORECASE
)
ETA_PATTERN = re.compile(
    r"^eta_stress_sum(?:_season(?P<season>\d+))?_(?P<year>\d{4})$",
    re.IGNORECASE,
)
FRACTION_TOLERANCE = 1.0e-6
MAX_FRACTION_OVERSHOOT = 0.01
DEFAULT_NODATA = -9999.0
_MISSING_CRS_WARNINGS: set[tuple[str, str]] = set()


@dataclass(frozen=True, order=True)
class SeasonKey:
    """One phenological-season index and SOS year."""

    season: int
    year: int


@dataclass(frozen=True)
class GridSpec:
    """Grid properties used for strict alignment checks."""

    height: int
    width: int
    transform: object
    crs: object


@dataclass(frozen=True)
class CropTable:
    """Crop names, seasonal Ky values, and provenance strings."""

    names: tuple[str, ...]
    ky: np.ndarray
    source_types: tuple[str, ...]
    references: tuple[str, ...]
    confidence: tuple[str, ...]


@dataclass(frozen=True)
class CropFractions:
    """Prepared crop fractions and their spatial support."""

    values: np.ndarray
    total: np.ndarray
    footprint: np.ndarray
    units_detected: str
    normalized_pixel_count: int


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the seasonal FAO-33 yield-response equation to paired "
            "seasonal ETx and stress-limited ETa GeoTIFFs."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="IWR JSON configuration used for the preceding model run.",
    )
    parser.add_argument(
        "--seasonal-et-dir",
        type=Path,
        default=None,
        help="Override <run output>/Seasonal_ET.",
    )
    parser.add_argument(
        "--etx-dir",
        type=Path,
        default=None,
        help="Override <seasonal ET>/ETx.",
    )
    parser.add_argument(
        "--eta-dir",
        type=Path,
        default=None,
        help="Override <seasonal ET>/ETa_stress.",
    )
    parser.add_argument(
        "--crop-fraction-path",
        type=Path,
        default=None,
        help="Override datasets.static.crop.crop_fraction_path.",
    )
    parser.add_argument(
        "--crop-parameters-csv",
        type=Path,
        default=None,
        help="Override datasets.static.crop.crop_parameters_csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override <run output>/Seasonal_Yield_Loss_FAO33.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First SOS year to process (default: all available pairs).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last SOS year to process, inclusive (default: all pairs).",
    )
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        default=None,
        help="Season index to process; repeat for multiple indices.",
    )
    parser.add_argument(
        "--min-etx",
        type=float,
        default=1.0e-6,
        help="ETx totals <= this value are output as nodata (default: 1e-6 mm).",
    )
    parser.add_argument(
        "--fao33-deficit-limit",
        type=float,
        default=0.5,
        help=(
            "Warn when 1-ETa/ETx exceeds this approximate FAO-33 linear "
            "calibration range (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=DEFAULT_NODATA,
        help="Output nodata value (default: -9999).",
    )
    parser.add_argument(
        "--strict-et-order",
        action="store_true",
        help="Fail instead of clipping pixels where ETa is greater than ETx.",
    )
    parser.add_argument(
        "--no-crop-specific",
        action="store_true",
        help="Do not write the 10-band crop-specific yield-loss GeoTIFFs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs that already exist.",
    )
    args = parser.parse_args(argv)

    if args.start_year is not None and args.end_year is not None:
        if args.start_year > args.end_year:
            parser.error("--start-year must be <= --end-year")
    if args.season is not None and any(value < 1 for value in args.season):
        parser.error("--season values must be positive integers")
    if not math.isfinite(args.min_etx) or args.min_etx < 0:
        parser.error("--min-etx must be finite and >= 0")
    if not math.isfinite(args.fao33_deficit_limit):
        parser.error("--fao33-deficit-limit must be finite")
    if not 0 < args.fao33_deficit_limit <= 1:
        parser.error("--fao33-deficit-limit must be in (0, 1]")
    if not math.isfinite(args.nodata):
        parser.error("--nodata must be finite")
    return args


def _nested(config: dict, dotted_key: str):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_key)
        value = value[key]
    return value


def _first_config_value(config: dict, keys: Sequence[str]):
    for key in keys:
        try:
            return _nested(config, key)
        except KeyError:
            continue
    raise KeyError(" or ".join(keys))


def _resolve_path(path: str | Path, base: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def load_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    config_dir = config_path.parent
    output_base = _resolve_path(
        _first_config_value(config, ("outputs.output_base", "output_base")),
        config_dir,
    )
    run_name = str(
        _first_config_value(config, ("outputs.run_name", "run_name"))
    ).strip()
    if not run_name:
        raise ValueError("outputs.run_name must not be empty")
    run_output = output_base / run_name

    seasonal_root = (
        args.seasonal_et_dir.expanduser().resolve()
        if args.seasonal_et_dir
        else run_output / "Seasonal_ET"
    )
    etx_dir = (
        args.etx_dir.expanduser().resolve()
        if args.etx_dir
        else seasonal_root / "ETx"
    )
    eta_dir = (
        args.eta_dir.expanduser().resolve()
        if args.eta_dir
        else seasonal_root / "ETa_stress"
    )
    crop_fraction_path = (
        args.crop_fraction_path.expanduser().resolve()
        if args.crop_fraction_path
        else _resolve_path(
            _first_config_value(
                config,
                (
                    "datasets.static.crop.crop_fraction_path",
                    "datasets.static.crop_fraction_path",
                    "crop_fraction_path",
                ),
            ),
            config_dir,
        )
    )
    crop_parameters_csv = (
        args.crop_parameters_csv.expanduser().resolve()
        if args.crop_parameters_csv
        else _resolve_path(
            _first_config_value(
                config,
                (
                    "datasets.static.crop.crop_parameters_csv",
                    "datasets.static.crop_parameters_csv",
                    "crop_parameters_csv",
                ),
            ),
            config_dir,
        )
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_output / "Seasonal_Yield_Loss_FAO33"
    )
    return etx_dir, eta_dir, crop_fraction_path, crop_parameters_csv, output_dir


def _selected(key: SeasonKey, args: argparse.Namespace) -> bool:
    if args.start_year is not None and key.year < args.start_year:
        return False
    if args.end_year is not None and key.year > args.end_year:
        return False
    if args.season is not None and key.season not in set(args.season):
        return False
    return True


def index_seasonal_rasters(
    directory: Path,
    pattern: re.Pattern[str],
    label: str,
    args: argparse.Namespace,
) -> dict[SeasonKey, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {directory}")

    result: dict[SeasonKey, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = pattern.fullmatch(path.stem)
        if match is None:
            continue
        season = match.groupdict().get("season")
        # Backward compatibility: accept year-only files such as etx_sum_1993.tif.
        key = SeasonKey(int(season) if season is not None else 1, int(match.group("year")))
        if not _selected(key, args):
            continue
        if key in result:
            raise ValueError(
                f"Duplicate {label} raster for season {key.season}, year "
                f"{key.year}: {result[key]} and {path}"
            )
        result[key] = path

    if not result:
        raise FileNotFoundError(
            f"No selected {label} seasonal GeoTIFFs matching {pattern.pattern} "
            f"in {directory}"
        )
    return result


def pair_inputs(
    etx_index: dict[SeasonKey, Path], eta_index: dict[SeasonKey, Path]
) -> list[tuple[SeasonKey, Path, Path]]:
    etx_keys = set(etx_index)
    eta_keys = set(eta_index)
    if etx_keys != eta_keys:
        only_etx = sorted(etx_keys - eta_keys)
        only_eta = sorted(eta_keys - etx_keys)
        raise ValueError(
            "Seasonal ETx and ETa_stress pairs do not match. "
            f"Only ETx (first 10): {only_etx[:10]}; "
            f"only ETa_stress (first 10): {only_eta[:10]}."
        )
    return [(key, etx_index[key], eta_index[key]) for key in sorted(etx_keys)]


def _clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def read_crop_table(path: Path) -> CropTable:
    if not path.is_file():
        raise FileNotFoundError(f"Crop parameter CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        missing = [name for name in ("crop_name", "Ky") if name not in fieldnames]
        if missing:
            raise ValueError(f"Crop CSV is missing required column(s): {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Crop CSV contains no crop rows: {path}")

    names: list[str] = []
    ky_values: list[float] = []
    source_types: list[str] = []
    references: list[str] = []
    confidence: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        name = (row.get("crop_name") or "").strip()
        if not name:
            raise ValueError(f"Empty crop_name at CSV line {line_number}")
        raw_ky = (row.get("Ky") or "").strip()
        try:
            ky = float(raw_ky)
        except ValueError as exc:
            raise ValueError(
                f"Invalid or missing Ky for crop '{name}' at CSV line "
                f"{line_number}: {raw_ky!r}"
            ) from exc
        if not math.isfinite(ky) or ky <= 0:
            raise ValueError(
                f"Ky must be finite and > 0 for crop '{name}'; found {ky}"
            )
        names.append(name)
        ky_values.append(ky)
        source_types.append((row.get("Ky_source_type") or "unspecified").strip())
        references.append((row.get("Ky_reference") or "unspecified").strip())
        confidence.append((row.get("Ky_confidence") or "unspecified").strip())

    cleaned = [_clean_name(name) for name in names]
    duplicates = sorted({name for name in cleaned if cleaned.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate normalized crop names in CSV: {duplicates}")

    return CropTable(
        names=tuple(names),
        ky=np.asarray(ky_values, dtype=np.float64),
        source_types=tuple(source_types),
        references=tuple(references),
        confidence=tuple(confidence),
    )


def _grid_from_dataset(src: rasterio.io.DatasetReader) -> GridSpec:
    return GridSpec(src.height, src.width, src.transform, src.crs)


def _check_grid(src: rasterio.io.DatasetReader, reference: GridSpec, path: Path) -> None:
    if (src.height, src.width) != (reference.height, reference.width):
        raise ValueError(
            f"Raster shape mismatch for {path}: found "
            f"{(src.height, src.width)}, expected "
            f"{(reference.height, reference.width)}"
        )
    if not src.transform.almost_equals(reference.transform):
        raise ValueError(f"Raster transform mismatch for {path}")
    if src.crs == reference.crs:
        return
    if src.crs is None or reference.crs is None:
        warning_key = (str(src.crs), str(reference.crs))
        if warning_key not in _MISSING_CRS_WARNINGS:
            _MISSING_CRS_WARNINGS.add(warning_key)
            print(
                "Warning: accepting an exact shape/transform match even though "
                f"one CRS is missing (found {src.crs}, expected {reference.crs}).",
                file=sys.stderr,
            )
        return
    if src.crs != reference.crs:
        raise ValueError(
            f"Raster CRS mismatch for {path}: found {src.crs}, "
            f"expected {reference.crs}"
        )


def _finite_valid(
    array: np.ndarray, nodata: float | None, mask: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(array) & mask
    if nodata is not None:
        if math.isnan(float(nodata)):
            valid &= ~np.isnan(array)
        else:
            valid &= array != nodata
    valid &= np.abs(array) < 1.0e19
    return valid


def read_single_band(
    path: Path, reference: GridSpec | None = None
) -> tuple[np.ndarray, np.ndarray, GridSpec, dict, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one band in {path}; found {src.count}")
        if reference is not None:
            _check_grid(src, reference, path)
        grid = _grid_from_dataset(src)
        profile = src.profile.copy()
        tags = {str(key): str(value) for key, value in src.tags().items()}
        array = src.read(1).astype(np.float64)
        mask = src.read_masks(1) > 0
        valid = _finite_valid(array, src.nodata, mask)
    return array, valid, grid, profile, tags


def read_crop_fractions(
    path: Path, crops: CropTable, reference: GridSpec
) -> CropFractions:
    if not path.is_file():
        raise FileNotFoundError(f"Crop fraction raster not found: {path}")
    with rasterio.open(path) as src:
        _check_grid(src, reference, path)
        if src.count != len(crops.names):
            raise ValueError(
                f"Crop raster has {src.count} bands but the CSV has "
                f"{len(crops.names)} crop rows"
            )
        descriptions = src.descriptions
        raw = src.read().astype(np.float64)
        masks = src.read_masks() > 0
        nodata = src.nodata

    nonempty_descriptions = [bool(value and value.strip()) for value in descriptions]
    if any(nonempty_descriptions) and not all(nonempty_descriptions):
        raise ValueError(
            "Crop raster has only partial band descriptions. Supply all crop "
            "names or clear all descriptions and rely on CSV row order."
        )
    if all(nonempty_descriptions):
        raster_names = tuple(_clean_name(value or "") for value in descriptions)
        csv_names = tuple(_clean_name(value) for value in crops.names)
        if raster_names != csv_names:
            raise ValueError(
                "Crop raster band descriptions do not match CSV row order. "
                f"Raster: {raster_names}; CSV: {csv_names}"
            )
        print("Crop-to-band mapping verified from raster band descriptions.")
    else:
        print(
            "Crop raster has no band descriptions; using CSV row order as the "
            "crop-to-band mapping."
        )

    valid_source = np.isfinite(raw) & masks
    if nodata is not None and not math.isnan(float(nodata)):
        valid_source &= raw != nodata
    footprint = np.any(valid_source, axis=0)
    negative_count = int(np.count_nonzero(valid_source & (raw < 0)))
    if negative_count:
        print(
            f"Warning: treating {negative_count} negative crop-fraction "
            "cell-band value(s) as zero.",
            file=sys.stderr,
        )
    fractions = np.where(valid_source & (raw > 0), raw, 0.0)
    maximum = float(np.max(fractions))
    units_detected = "fraction_0_1"
    if maximum > 1.0 + FRACTION_TOLERANCE:
        if maximum > 100.0 + FRACTION_TOLERANCE:
            raise ValueError(
                "A crop-fraction value exceeds 100; expected either 0--1 "
                f"fractions or 0--100 percentages. Maximum={maximum:.12g}"
            )
        fractions /= 100.0
        units_detected = "percent_0_100"
        print(
            f"Crop fractions detected as percentages (maximum={maximum:.6g}); "
            "dividing all bands by 100."
        )

    total = np.sum(fractions, axis=0, dtype=np.float64)
    maximum_total = float(np.max(total))
    too_high = total > 1.0 + FRACTION_TOLERANCE
    normalized_count = int(np.count_nonzero(too_high))
    if normalized_count:
        if maximum_total > 1.0 + MAX_FRACTION_OVERSHOOT:
            raise ValueError(
                "Crop fractions sum to more than 101% in at least one pixel; "
                f"maximum sum={maximum_total:.12g}"
            )
        scale = np.ones_like(total)
        scale[too_high] = 1.0 / total[too_high]
        fractions *= scale[np.newaxis, :, :]
        total = np.sum(fractions, axis=0, dtype=np.float64)
        print(
            f"Warning: normalized {normalized_count} pixel(s) whose crop "
            f"fractions summed slightly above 1 (maximum={maximum_total:.8f}).",
            file=sys.stderr,
        )

    print("Crop fraction diagnostics:")
    print(f"  bands/crops:                 {len(crops.names)}")
    print(f"  positive cropped pixels:    {int(np.count_nonzero(total > 0))}")
    print(f"  maximum total fraction:     {float(np.max(total)):.8f}")
    if np.any(total > 0):
        print(f"  mean positive fraction:     {float(np.mean(total[total > 0])):.8f}")

    return CropFractions(
        values=fractions,
        total=total,
        footprint=footprint,
        units_detected=units_detected,
        normalized_pixel_count=normalized_count,
    )


def effective_ky(crop_fractions: CropFractions, crops: CropTable) -> np.ndarray:
    weighted_sum = np.sum(
        crop_fractions.values * crops.ky[:, np.newaxis, np.newaxis],
        axis=0,
        dtype=np.float64,
    )
    result = np.full(crop_fractions.total.shape, np.nan, dtype=np.float64)
    cropped = crop_fractions.total > 0
    result[cropped] = weighted_sum[cropped] / crop_fractions.total[cropped]
    return result


def _output_profile(reference_profile: dict, count: int, nodata: float) -> dict:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=count,
        nodata=float(nodata),
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _string_tags(tags: dict[str, object]) -> dict[str, str]:
    return {str(key): str(value) for key, value in tags.items()}


def write_single_band(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    reference_profile: dict,
    nodata: float,
    description: str,
    tags: dict[str, object],
    overwrite: bool,
) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.full(values.shape, nodata, dtype=np.float32)
    output[valid] = values[valid].astype(np.float32)
    with rasterio.open(path, "w", **_output_profile(reference_profile, 1, nodata)) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, description)
        dst.update_tags(**_string_tags(tags))
    return int(np.count_nonzero(valid))


def write_crop_bands(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    reference_profile: dict,
    nodata: float,
    crops: CropTable,
    tags: dict[str, object],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.full(values.shape, nodata, dtype=np.float32)
    output[valid] = values[valid].astype(np.float32)
    with rasterio.open(
        path,
        "w",
        **_output_profile(reference_profile, len(crops.names), nodata),
    ) as dst:
        dst.write(output)
        dst.update_tags(**_string_tags(tags))
        for index, name in enumerate(crops.names, start=1):
            dst.set_band_description(index, f"{name}: relative yield loss")
            dst.update_tags(
                index,
                crop_name=name,
                Ky=f"{crops.ky[index - 1]:.12g}",
                Ky_source_type=crops.source_types[index - 1],
                Ky_reference=crops.references[index - 1],
                Ky_confidence=crops.confidence[index - 1],
                crop_fraction_band=str(index),
                units="fraction_0_1",
            )


def output_paths(
    output_dir: Path, key: SeasonKey, write_crop_specific: bool
) -> tuple[Path, ...]:
    suffix = f"season{key.season}_{key.year}.tif"
    paths = [
        output_dir / "ET_deficit_fraction" / f"et_deficit_fraction_{suffix}",
        output_dir / "Yield_loss_fraction" / f"yield_loss_fraction_{suffix}",
        output_dir / "Relative_yield_fraction" / f"relative_yield_fraction_{suffix}",
    ]
    if write_crop_specific:
        paths.append(
            output_dir
            / "Crop_yield_loss_fraction"
            / f"crop_yield_loss_fraction_{suffix}"
        )
    return tuple(paths)


def preflight_outputs(
    output_dir: Path,
    pairs: Sequence[tuple[SeasonKey, Path, Path]],
    write_crop_specific: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    candidates = [
        output_dir / "Support" / "crop_fraction_sum.tif",
        output_dir / "Support" / "effective_ky.tif",
    ]
    for key, _, _ in pairs:
        candidates.extend(output_paths(output_dir, key, write_crop_specific))
    existing = [path for path in candidates if path.exists()]
    if existing:
        preview = "\n  ".join(str(path) for path in existing[:20])
        remainder = len(existing) - min(len(existing), 20)
        extra = f"\n  ... and {remainder} more" if remainder else ""
        raise FileExistsError(
            "Output files already exist; no files were written. Use "
            f"--overwrite to replace them:\n  {preview}{extra}"
        )


def process_pair(
    key: SeasonKey,
    etx_path: Path,
    eta_path: Path,
    reference: GridSpec,
    reference_profile: dict,
    crop_fractions: CropFractions,
    crops: CropTable,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    etx, etx_valid, _, _, etx_tags = read_single_band(etx_path, reference)
    eta, eta_valid, _, _, eta_tags = read_single_band(eta_path, reference)

    negative_etx = etx_valid & (etx < -1.0e-5)
    negative_eta = eta_valid & (eta < -1.0e-5)
    if np.any(negative_etx):
        raise ValueError(
            f"Negative ETx in {etx_path}; minimum={float(np.min(etx[negative_etx]))}"
        )
    if np.any(negative_eta):
        raise ValueError(
            f"Negative ETa in {eta_path}; minimum={float(np.min(eta[negative_eta]))}"
        )
    etx[(etx_valid) & (etx < 0)] = 0.0
    eta[(eta_valid) & (eta < 0)] = 0.0

    cropped = crop_fractions.total > 0
    valid = etx_valid & eta_valid & (etx > args.min_etx) & cropped
    eta_above_etx = valid & (eta > etx + 1.0e-5)
    eta_above_count = int(np.count_nonzero(eta_above_etx))
    if eta_above_count and args.strict_et_order:
        maximum_excess = float(np.max((eta - etx)[eta_above_etx]))
        raise ValueError(
            f"ETa exceeds ETx at {eta_above_count} pixel(s) for season "
            f"{key.season}, {key.year}; maximum excess={maximum_excess:.8g} mm"
        )

    deficit = np.zeros(etx.shape, dtype=np.float64)
    deficit[valid] = 1.0 - eta[valid] / etx[valid]
    deficit[valid] = np.clip(deficit[valid], 0.0, 1.0)

    outside_fao_count = int(
        np.count_nonzero(valid & (deficit > args.fao33_deficit_limit))
    )
    ky_cube = crops.ky[:, np.newaxis, np.newaxis]
    crop_loss = np.clip(ky_cube * deficit[np.newaxis, :, :], 0.0, 1.0)
    crop_band_valid = (
        valid[np.newaxis, :, :] & (crop_fractions.values > 0)
    )
    uncapped_crop_loss = ky_cube * deficit[np.newaxis, :, :]
    capped_any = valid & np.any(
        (crop_fractions.values > 0) & (uncapped_crop_loss > 1.0), axis=0
    )
    capped_pixel_count = int(np.count_nonzero(capped_any))

    weighted_loss_sum = np.sum(
        crop_fractions.values * crop_loss, axis=0, dtype=np.float64
    )
    yield_loss = np.zeros(etx.shape, dtype=np.float64)
    yield_loss[valid] = weighted_loss_sum[valid] / crop_fractions.total[valid]
    yield_loss[valid] = np.clip(yield_loss[valid], 0.0, 1.0)
    relative_yield = np.zeros(etx.shape, dtype=np.float64)
    relative_yield[valid] = 1.0 - yield_loss[valid]

    paths = output_paths(output_dir, key, not args.no_crop_specific)
    et_deficit_path, yield_loss_path, relative_yield_path = paths[:3]
    crop_loss_path = paths[3] if len(paths) == 4 else None

    common_tags: dict[str, object] = {
        "method": "FAO Irrigation and Drainage Paper 33 seasonal relationship",
        "formula": "1-Ya/Yx = Ky*(1-ETa/ETx)",
        "season_index": key.season,
        "season_year": key.year,
        "season_year_definition": "calendar year containing pixel-specific SOS",
        "units": "fraction_0_1",
        "source_etx": str(etx_path),
        "source_eta": str(eta_path),
        "source_etx_units": etx_tags.get("units", "unknown"),
        "source_eta_units": eta_tags.get("units", "unknown"),
        "ky_csv": str(args._crop_parameters_csv),
        "crop_fraction_raster": str(args._crop_fraction_path),
        "crop_fraction_input_units_detected": crop_fractions.units_detected,
        "crop_fraction_normalized_pixel_count": crop_fractions.normalized_pixel_count,
        "mixed_crop_aggregation": (
            "crop-specific loss calculated first then area-weighted over cropped fraction"
        ),
        "et_ratio_scope": (
            "same upstream pixel-level ETa/ETx ratio applied to every crop band"
        ),
        "crop_mix_season_assignment": (
            "same static crop-fraction mix applied independently to every season index"
        ),
        "yield_loss_cap": "each crop loss clipped to 0-1 before aggregation",
        "eta_above_etx_policy": (
            "error" if args.strict_et_order else "ET deficit clipped to zero"
        ),
        "eta_above_etx_pixel_count": eta_above_count,
        "fao33_approx_linear_deficit_limit": args.fao33_deficit_limit,
        "pixels_above_fao33_deficit_limit": outside_fao_count,
        "pixels_with_crop_loss_capped_at_one": capped_pixel_count,
        "valid_pixel_count": int(np.count_nonzero(valid)),
    }

    write_single_band(
        et_deficit_path,
        deficit,
        valid,
        reference_profile,
        args.nodata,
        "Seasonal ET deficit: 1 - ETa/ETx",
        {**common_tags, "variable": "seasonal_relative_et_deficit"},
        args.overwrite,
    )
    write_single_band(
        yield_loss_path,
        yield_loss,
        valid,
        reference_profile,
        args.nodata,
        "Crop-area-weighted relative yield loss: 1 - Ya/Yx",
        {
            **common_tags,
            "variable": "crop_area_weighted_relative_yield_loss",
            "interpretation": (
                "relative loss over cropped area; not absolute yield or pixel production"
            ),
        },
        args.overwrite,
    )
    write_single_band(
        relative_yield_path,
        relative_yield,
        valid,
        reference_profile,
        args.nodata,
        "Crop-area-weighted relative yield: Ya/Yx",
        {
            **common_tags,
            "variable": "crop_area_weighted_relative_yield",
            "formula": "Ya/Yx = 1 - crop-area-weighted relative yield loss",
        },
        args.overwrite,
    )
    if crop_loss_path is not None:
        write_crop_bands(
            crop_loss_path,
            crop_loss,
            crop_band_valid,
            reference_profile,
            args.nodata,
            crops,
            {
                **common_tags,
                "variable": "crop_specific_relative_yield_loss",
                "band_order": "same as crop parameter CSV and crop fraction raster",
            },
            args.overwrite,
        )

    print(
        f"Season {key.season}, {key.year}: wrote {len(paths)} file(s); "
        f"valid pixels={int(np.count_nonzero(valid))}; "
        f"ETa>ETx clipped={eta_above_count}; "
        f"deficit>{args.fao33_deficit_limit:g}={outside_fao_count}; "
        f"loss capped={capped_pixel_count}"
    )
    return len(paths)


def write_support_rasters(
    output_dir: Path,
    reference_profile: dict,
    crop_fractions: CropFractions,
    crops: CropTable,
    effective_ky_values: np.ndarray,
    crop_fraction_path: Path,
    crop_parameters_csv: Path,
    args: argparse.Namespace,
) -> int:
    support_dir = output_dir / "Support"
    ky_summary = ";".join(
        f"{name}={ky:.12g}" for name, ky in zip(crops.names, crops.ky)
    )
    common = {
        "crop_fraction_raster": str(crop_fraction_path),
        "ky_csv": str(crop_parameters_csv),
        "crop_order_and_ky": ky_summary,
        "crop_fraction_input_units_detected": crop_fractions.units_detected,
        "crop_fraction_normalized_pixel_count": crop_fractions.normalized_pixel_count,
    }
    write_single_band(
        support_dir / "crop_fraction_sum.tif",
        crop_fractions.total,
        crop_fractions.footprint,
        reference_profile,
        args.nodata,
        "Sum of prepared crop fractions",
        {
            **common,
            "variable": "prepared_crop_fraction_sum",
            "units": "fraction_0_1",
            "aggregation": "sum across prepared crop-fraction bands",
        },
        args.overwrite,
    )
    cropped = crop_fractions.total > 0
    write_single_band(
        support_dir / "effective_ky.tif",
        effective_ky_values,
        cropped,
        reference_profile,
        args.nodata,
        "Crop-area-weighted seasonal Ky",
        {
            **common,
            "variable": "crop_area_weighted_seasonal_ky",
            "units": "dimensionless",
            "formula": "sum(f_i*Ky_i)/sum(f_i)",
            "note": (
                "Exact aggregate slope before physical crop-level loss caps are applied"
            ),
        },
        args.overwrite,
    )
    return 2


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    (
        etx_dir,
        eta_dir,
        crop_fraction_path,
        crop_parameters_csv,
        output_dir,
    ) = load_paths(args)
    # Retain resolved provenance paths for process_pair output metadata.
    args._crop_fraction_path = crop_fraction_path
    args._crop_parameters_csv = crop_parameters_csv

    print(f"Seasonal ETx input:  {etx_dir}")
    print(f"Seasonal ETa input:  {eta_dir}")
    print(f"Crop fractions:       {crop_fraction_path}")
    print(f"Crop parameters/Ky:   {crop_parameters_csv}")
    print(f"FAO-33 output:        {output_dir}")

    etx_index = index_seasonal_rasters(etx_dir, ETX_PATTERN, "ETx", args)
    eta_index = index_seasonal_rasters(eta_dir, ETA_PATTERN, "ETa_stress", args)
    pairs = pair_inputs(etx_index, eta_index)
    print(f"Selected seasonal pairs: {len(pairs)}")

    first_key, first_etx_path, _ = pairs[0]
    _, _, reference, reference_profile, _ = read_single_band(first_etx_path)
    print(
        f"Reference grid from season {first_key.season}, {first_key.year}: "
        f"{reference.width} x {reference.height}; CRS={reference.crs}"
    )

    crops = read_crop_table(crop_parameters_csv)
    print("Seasonal Ky values:")
    for name, ky, source_type, confidence in zip(
        crops.names, crops.ky, crops.source_types, crops.confidence
    ):
        print(
            f"  {name:<12} Ky={ky:<6g} source={source_type:<20} "
            f"confidence={confidence}"
        )

    crop_fractions = read_crop_fractions(crop_fraction_path, crops, reference)
    effective_ky_values = effective_ky(crop_fractions, crops)
    cropped = crop_fractions.total > 0
    if np.any(cropped):
        print(
            "Effective Ky over cropped pixels: "
            f"min={float(np.min(effective_ky_values[cropped])):.6g}, "
            f"mean={float(np.mean(effective_ky_values[cropped])):.6g}, "
            f"max={float(np.max(effective_ky_values[cropped])):.6g}"
        )

    preflight_outputs(
        output_dir,
        pairs,
        write_crop_specific=not args.no_crop_specific,
        overwrite=args.overwrite,
    )
    files_written = write_support_rasters(
        output_dir,
        reference_profile,
        crop_fractions,
        crops,
        effective_ky_values,
        crop_fraction_path,
        crop_parameters_csv,
        args,
    )
    for key, etx_path, eta_path in pairs:
        files_written += process_pair(
            key,
            etx_path,
            eta_path,
            reference,
            reference_profile,
            crop_fractions,
            crops,
            output_dir,
            args,
        )

    print(f"Done: wrote {files_written} GeoTIFF(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        csv.Error,
        json.JSONDecodeError,
        rasterio.errors.RasterioError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
