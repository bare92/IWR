#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing monthly precipitation NetCDF files.
P_NETCDF_FOLDER = Path(
    "/share/data/DAO/input/Forcing_micromet_corrected/P"
)

# Folder containing monthly potential evapotranspiration NetCDF files.
PET_NETCDF_FOLDER = Path(
    "/share/data/DAO/input/Forcing_micromet_corrected/ET_HS"
)

# Output root folder. The script creates P and PET subfolders automatically.
OUTPUT_ROOT = Path(
    "/share/data/DAO/input/output_geotiffs_micromet"
)

# Reference grid used for all GeoTIFF outputs.
WORKING_GRID_PATH = Path(
    "/share/data/DAO/static/processed/working_grid_3035_1km_precip_valid.tif"
)

# Variable names inside the NetCDF files.
P_VARIABLE_NAME = "P"
PET_VARIABLE_NAME = "PET"

# Input file pattern.
NETCDF_PATTERN = "*.nc"

# Output nodata value.
OUTPUT_NODATA = -9999.0

# Optional date filter.
# Use None to process the complete available period.
START_DATE = None
END_DATE = None

# Examples:
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# Existing-file behaviour.
OVERWRITE_EXISTING = False

# If True, fail when the dates decoded from a NetCDF do not belong to the
# year and month written in the NetCDF filename.
STRICT_DATE_CHECK = False

# If True, compare the complete sets of P and PET dates at the end.
CHECK_P_PET_DATE_CONSISTENCY = True

# Fallback CRS used only when no valid CRS can be read from the NetCDF.
FALLBACK_CRS = "EPSG:3035"

# Output compression.
GEOTIFF_COMPRESSION = "lzw"

# Resampling used when source and target grids differ.
REPROJECT_RESAMPLING = "bilinear"

# =============================================================================
# END USER SETTINGS
# =============================================================================


MONTH_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[_-](?P<month>0[1-9]|1[0-2])"
)


@dataclass(frozen=True)
class ProductConfig:
    label: str
    variable: str
    output_subfolder: str
    filename_prefix: str
    units_hint: str


@dataclass(frozen=True)
class GridDefinition:
    width: int
    height: int
    transform: Affine
    crs: CRS


P_CONFIG = ProductConfig(
    label="precipitation",
    variable=P_VARIABLE_NAME,
    output_subfolder="P",
    filename_prefix="P_P_sim",
    units_hint="mm",
)

PET_CONFIG = ProductConfig(
    label="potential evapotranspiration",
    variable=PET_VARIABLE_NAME,
    output_subfolder="PET",
    filename_prefix="PET_PET_sim",
    units_hint="mm day-1",
)


def parse_optional_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None

    timestamp = pd.Timestamp(value).normalize()

    if pd.isna(timestamp):
        raise ValueError(f"Invalid date: {value}")

    return timestamp


def validate_settings() -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_date = parse_optional_date(START_DATE)
    end_date = parse_optional_date(END_DATE)

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError(
                "START_DATE must be earlier than or equal to END_DATE."
            )

    return start_date, end_date


def list_netcdf_files(folder: Path, pattern: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    files = sorted(
        path for path in folder.glob(pattern)
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files matching '{pattern}' were found in {folder}"
        )

    return files


def parse_year_month_from_filename(
    path: Path,
) -> tuple[int, int] | None:
    match = MONTH_PATTERN.search(path.stem)

    if match is None:
        return None

    return int(match.group("year")), int(match.group("month"))


def choose_variable(
    dataset: xr.Dataset,
    requested_name: str,
    label: str,
) -> str:
    if requested_name in dataset.data_vars:
        return requested_name

    case_insensitive = {
        str(name).lower(): str(name)
        for name in dataset.data_vars
    }

    matched = case_insensitive.get(requested_name.lower())

    if matched is not None:
        print(
            f"  Warning: variable '{requested_name}' was not found exactly; "
            f"using '{matched}'."
        )
        return matched

    candidates = [
        str(name)
        for name, variable in dataset.data_vars.items()
        if "time" in variable.dims and variable.ndim >= 3
    ]

    if len(candidates) == 1:
        print(
            f"  Warning: variable '{requested_name}' was not found. "
            f"Using the only time-dependent raster variable: "
            f"'{candidates[0]}'."
        )
        return candidates[0]

    raise ValueError(
        f"Variable '{requested_name}' was not found in the {label} NetCDF.\n"
        f"Available data variables: {list(dataset.data_vars)}"
    )


def find_xy_dimension_names(
    variable: xr.DataArray,
) -> tuple[str, str]:
    dims = list(variable.dims)

    x_candidates = [
        name for name in dims
        if name.lower() in {
            "x",
            "lon",
            "longitude",
            "easting",
        }
    ]

    y_candidates = [
        name for name in dims
        if name.lower() in {
            "y",
            "lat",
            "latitude",
            "northing",
        }
    ]

    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]

    spatial_dims = [
        name for name in dims
        if name != "time"
    ]

    if len(spatial_dims) != 2:
        raise ValueError(
            f"Could not identify two spatial dimensions in {variable.dims}."
        )

    # Most CF-compliant rasters use dimensions (..., y, x).
    return spatial_dims[-1], spatial_dims[-2]


def read_crs(
    dataset: xr.Dataset,
    variable: xr.DataArray,
) -> CRS:
    candidate_attributes: list[str] = []

    grid_mapping_name = variable.attrs.get("grid_mapping")

    if grid_mapping_name and grid_mapping_name in dataset.variables:
        grid_mapping = dataset[grid_mapping_name]

        for key in ("crs_wkt", "spatial_ref"):
            value = grid_mapping.attrs.get(key)

            if value:
                candidate_attributes.append(str(value))

    for variable_name in ("spatial_ref", "crs"):
        if variable_name in dataset.variables:
            for key in ("crs_wkt", "spatial_ref"):
                value = dataset[variable_name].attrs.get(key)

                if value:
                    candidate_attributes.append(str(value))

    for key in ("crs_wkt", "spatial_ref"):
        value = dataset.attrs.get(key)

        if value:
            candidate_attributes.append(str(value))

    for value in candidate_attributes:
        try:
            return CRS.from_user_input(value)
        except Exception:
            continue

    print(
        f"  Warning: no usable CRS was found in the NetCDF. "
        f"Using {FALLBACK_CRS}."
    )

    return CRS.from_user_input(FALLBACK_CRS)


def regular_spacing(
    values: np.ndarray,
    coordinate_name: str,
) -> float:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"Coordinate '{coordinate_name}' must be one-dimensional "
            "and contain at least two values."
        )

    differences = np.diff(values)

    if not np.all(np.isfinite(differences)):
        raise ValueError(
            f"Coordinate '{coordinate_name}' contains invalid spacing."
        )

    spacing = float(np.median(np.abs(differences)))

    if spacing <= 0:
        raise ValueError(
            f"Coordinate '{coordinate_name}' has zero or invalid spacing."
        )

    tolerance = max(spacing * 1e-6, 1e-8)

    if not np.allclose(
        np.abs(differences),
        spacing,
        rtol=1e-6,
        atol=tolerance,
    ):
        raise ValueError(
            f"Coordinate '{coordinate_name}' is not regularly spaced."
        )

    return spacing


def build_grid_definition(
    dataset: xr.Dataset,
    variable: xr.DataArray,
    x_name: str,
    y_name: str,
) -> GridDefinition:
    if x_name not in dataset.coords or y_name not in dataset.coords:
        raise ValueError(
            f"Spatial coordinates '{x_name}' and '{y_name}' must be "
            "available as NetCDF coordinates."
        )

    x_values = np.asarray(
        dataset[x_name].values,
        dtype=np.float64,
    )

    y_values = np.asarray(
        dataset[y_name].values,
        dtype=np.float64,
    )

    x_resolution = regular_spacing(x_values, x_name)
    y_resolution = regular_spacing(y_values, y_name)

    west = float(
        np.min(x_values) - x_resolution / 2.0
    )

    north = float(
        np.max(y_values) + y_resolution / 2.0
    )

    transform = from_origin(
        west,
        north,
        x_resolution,
        y_resolution,
    )

    return GridDefinition(
        width=int(x_values.size),
        height=int(y_values.size),
        transform=transform,
        crs=read_crs(dataset, variable),
    )


def load_grid_from_geotiff(path: Path) -> GridDefinition:
    if not path.exists():
        raise FileNotFoundError(
            f"Working grid file does not exist: {path}"
        )

    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(
                f"Working grid has no CRS: {path}"
            )

        return GridDefinition(
            width=int(source.width),
            height=int(source.height),
            transform=source.transform,
            crs=source.crs,
        )


def parse_resampling(name: str) -> Resampling:
    normalized = name.strip().lower()

    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }

    if normalized not in mapping:
        raise ValueError(
            "REPROJECT_RESAMPLING must be one of: "
            f"{', '.join(mapping.keys())}. Got '{name}'."
        )

    return mapping[normalized]


def align_to_target_grid(
    data: np.ndarray,
    source_grid: GridDefinition,
    target_grid: GridDefinition,
    nodata: float,
    resampling: Resampling,
) -> np.ndarray:
    if grid_is_equivalent(source_grid, target_grid):
        return data

    destination = np.full(
        (target_grid.height, target_grid.width),
        np.float32(nodata),
        dtype=np.float32,
    )

    reproject(
        source=data,
        destination=destination,
        src_transform=source_grid.transform,
        src_crs=source_grid.crs,
        src_nodata=np.float32(nodata),
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        dst_nodata=np.float32(nodata),
        resampling=resampling,
    )

    return destination


def grid_is_equivalent(
    first: GridDefinition,
    second: GridDefinition,
    transform_tolerance: float = 0.1,
) -> bool:
    same_transform = all(
        abs(a - b) <= transform_tolerance
        for a, b in zip(first.transform, second.transform)
    )

    same_crs = first.crs == second.crs

    if not same_crs:
        try:
            same_crs = (
                first.crs.to_epsg()
                == second.crs.to_epsg()
            )
        except Exception:
            same_crs = False

    return (
        first.width == second.width
        and first.height == second.height
        and same_transform
        and same_crs
    )


def normalize_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        timestamp = pd.Timestamp(str(value))

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert("UTC")
            .tz_localize(None)
        )

    return timestamp.normalize()


def derive_dates(
    dataset: xr.Dataset,
    source_path: Path,
    number_of_steps: int,
) -> list[pd.Timestamp]:
    """
    Prefer decoded CF time coordinates.

    If time decoding is unavailable, use the year/month in the filename and
    assign bands consecutively from the first day of that month.
    """

    if "time" in dataset.coords:
        values = dataset["time"].values

        if np.asarray(values).size != number_of_steps:
            raise ValueError(
                f"Time coordinate length "
                f"({np.asarray(values).size}) does not match the number "
                f"of raster steps ({number_of_steps}) in "
                f"{source_path.name}."
            )

        try:
            return [
                normalize_timestamp(value)
                for value in values
            ]
        except Exception as exc:
            print(
                f"  Warning: CF time decoding failed in "
                f"{source_path.name}: {exc}. "
                "Falling back to filename year/month."
            )

    parsed = parse_year_month_from_filename(source_path)

    if parsed is None:
        raise ValueError(
            "Could not decode time and could not extract YYYY_MM or "
            f"YYYY-MM from filename: {source_path.name}"
        )

    year, month = parsed
    month_start = pd.Timestamp(
        year=year,
        month=month,
        day=1,
    )

    return [
        month_start + pd.Timedelta(days=index)
        for index in range(number_of_steps)
    ]


def open_dataset_with_fallback(
    path: Path,
) -> xr.Dataset:
    try:
        return xr.open_dataset(
            path,
            decode_times=True,
            mask_and_scale=True,
        )
    except Exception as first_error:
        print(
            f"  Warning: normal NetCDF opening failed for "
            f"{path.name}: {first_error}"
        )
        print("  Retrying with decode_times=False.")

        try:
            return xr.open_dataset(
                path,
                decode_times=False,
                mask_and_scale=True,
            )
        except Exception as second_error:
            raise RuntimeError(
                f"Could not open NetCDF file {path}\n"
                f"First error: {first_error}\n"
                f"Second error: {second_error}"
            ) from second_error


def prepare_daily_array(
    variable: xr.DataArray,
    time_index: int,
    x_name: str,
    y_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    nodata: float,
) -> np.ndarray:
    daily = variable.isel(time=time_index)

    extra_dimensions = [
        dimension
        for dimension in daily.dims
        if dimension not in {x_name, y_name}
    ]

    if extra_dimensions:
        non_singleton = [
            dimension
            for dimension in extra_dimensions
            if daily.sizes[dimension] != 1
        ]

        if non_singleton:
            raise ValueError(
                "Unexpected non-spatial dimensions after selecting time: "
                f"{non_singleton}"
            )

        daily = daily.squeeze(
            extra_dimensions,
            drop=True,
        )

    daily = daily.transpose(y_name, x_name)

    data = np.asarray(
        daily.values,
        dtype=np.float32,
    )

    if data.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional daily raster, "
            f"got shape {data.shape}."
        )

    # GeoTIFF rows must run north to south.
    if y_values[0] < y_values[-1]:
        data = np.flipud(data)

    # GeoTIFF columns must run west to east.
    if x_values[0] > x_values[-1]:
        data = np.fliplr(data)

    valid = np.isfinite(data)

    cleaned = np.full(
        data.shape,
        nodata,
        dtype=np.float32,
    )

    cleaned[valid] = data[valid]

    return cleaned


def output_filename(
    config: ProductConfig,
    date: pd.Timestamp,
) -> str:
    return (
        f"{config.filename_prefix}_"
        f"{date.strftime('%Y-%m-%d')}_00_00_00.tif"
    )


def write_daily_geotiff(
    output_path: Path,
    data: np.ndarray,
    grid: GridDefinition,
    nodata: float,
    variable_name: str,
    date: pd.Timestamp,
    units: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.float32(nodata),
        "compress": GEOTIFF_COMPRESSION,
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }

    # Use tiling only when the raster is large enough.
    if grid.width >= 16 and grid.height >= 16:
        profile["tiled"] = True
        profile["blockxsize"] = min(
            256,
            max(16, (grid.width // 16) * 16),
        )
        profile["blockysize"] = min(
            256,
            max(16, (grid.height // 16) * 16),
        )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as destination:
        destination.write(data, 1)
        destination.set_band_description(
            1,
            variable_name,
        )
        destination.update_tags(
            variable=variable_name,
            date=date.strftime("%Y-%m-%d"),
            units=units,
        )


def date_is_selected(
    date: pd.Timestamp,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> bool:
    if start_date is not None and date < start_date:
        return False

    if end_date is not None and date > end_date:
        return False

    return True


def validate_dates_against_filename(
    path: Path,
    dates: Iterable[pd.Timestamp],
    strict: bool,
) -> None:
    parsed = parse_year_month_from_filename(path)

    if parsed is None:
        return

    year, month = parsed

    mismatches = [
        date
        for date in dates
        if date.year != year or date.month != month
    ]

    if not mismatches:
        return

    message = (
        f"{path.name}: {len(mismatches)} decoded date(s) fall outside "
        f"{year:04d}-{month:02d}. First mismatch: "
        f"{mismatches[0].strftime('%Y-%m-%d')}"
    )

    if strict:
        raise ValueError(message)

    print(f"  Warning: {message}")


def convert_product(
    input_folder: Path,
    output_root: Path,
    config: ProductConfig,
    pattern: str,
    nodata: float,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    overwrite: bool,
    strict_date_check: bool,
    reference_grid: GridDefinition | None,
) -> tuple[set[pd.Timestamp], GridDefinition]:
    files = list_netcdf_files(
        input_folder,
        pattern,
    )

    output_folder = (
        output_root
        / config.output_subfolder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dates: set[pd.Timestamp] = set()
    current_reference_grid = reference_grid
    source_reference_grid: GridDefinition | None = None
    reproject_resampling = parse_resampling(
        REPROJECT_RESAMPLING
    )

    if current_reference_grid is None:
        raise ValueError(
            "A reference working grid must be provided."
        )

    print()
    print("=" * 78)
    print(f"Processing {config.label}")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"NetCDF files:  {len(files)}")
    print("=" * 78)

    for file_number, source_path in enumerate(
        files,
        start=1,
    ):
        print(
            f"[{file_number}/{len(files)}] "
            f"{source_path.name}"
        )

        with open_dataset_with_fallback(
            source_path
        ) as dataset:
            variable_name = choose_variable(
                dataset=dataset,
                requested_name=config.variable,
                label=config.label,
            )

            variable = dataset[variable_name]

            if "time" not in variable.dims:
                raise ValueError(
                    f"Variable '{variable_name}' in "
                    f"{source_path.name} does not have "
                    "a time dimension."
                )

            x_name, y_name = find_xy_dimension_names(
                variable
            )

            grid = build_grid_definition(
                dataset=dataset,
                variable=variable,
                x_name=x_name,
                y_name=y_name,
            )

            if source_reference_grid is None:
                source_reference_grid = grid

                print(
                    "  Source grid: "
                    f"{grid.width} x {grid.height}, "
                    f"{grid.transform.a:g} m, "
                    f"{grid.crs}"
                )
                print(
                    "  Target grid: "
                    f"{current_reference_grid.width} x "
                    f"{current_reference_grid.height}, "
                    f"{current_reference_grid.transform.a:g} m, "
                    f"{current_reference_grid.crs}"
                )
            elif not grid_is_equivalent(
                source_reference_grid,
                grid,
            ):
                print(
                    "  Warning: source grid differs from the first "
                    "source file; this file will be reprojected "
                    "to the target working grid."
                )

            number_of_steps = int(
                variable.sizes["time"]
            )

            dates = derive_dates(
                dataset=dataset,
                source_path=source_path,
                number_of_steps=number_of_steps,
            )

            validate_dates_against_filename(
                path=source_path,
                dates=dates,
                strict=strict_date_check,
            )

            if len(set(dates)) != len(dates):
                raise ValueError(
                    f"Duplicate dates were found inside "
                    f"{source_path.name}."
                )

            x_values = np.asarray(
                dataset[x_name].values,
                dtype=np.float64,
            )

            y_values = np.asarray(
                dataset[y_name].values,
                dtype=np.float64,
            )

            units = str(
                variable.attrs.get(
                    "units",
                    config.units_hint,
                )
            )

            outputs_from_file = 0
            skipped_from_file = 0
            filtered_from_file = 0

            for time_index, date in enumerate(dates):
                if not date_is_selected(
                    date,
                    start_date,
                    end_date,
                ):
                    filtered_from_file += 1
                    continue

                output_path = (
                    output_folder
                    / output_filename(config, date)
                )

                if date in processed_dates:
                    raise ValueError(
                        f"Duplicate date "
                        f"{date.strftime('%Y-%m-%d')} "
                        f"was found across multiple "
                        f"{config.label} NetCDF files."
                    )

                if (
                    output_path.exists()
                    and not overwrite
                ):
                    processed_dates.add(date)
                    skipped_from_file += 1
                    continue

                data = prepare_daily_array(
                    variable=variable,
                    time_index=time_index,
                    x_name=x_name,
                    y_name=y_name,
                    x_values=x_values,
                    y_values=y_values,
                    nodata=nodata,
                )

                data = align_to_target_grid(
                    data=data,
                    source_grid=grid,
                    target_grid=current_reference_grid,
                    nodata=nodata,
                    resampling=reproject_resampling,
                )

                write_daily_geotiff(
                    output_path=output_path,
                    data=data,
                    grid=current_reference_grid,
                    nodata=nodata,
                    variable_name=variable_name,
                    date=date,
                    units=units,
                )

                processed_dates.add(date)
                outputs_from_file += 1

            print(
                f"  Dates in file: "
                f"{dates[0].strftime('%Y-%m-%d')} to "
                f"{dates[-1].strftime('%Y-%m-%d')}"
            )

            print(
                f"  Written: {outputs_from_file}; "
                f"already existing: {skipped_from_file}; "
                f"filtered out: {filtered_from_file}"
            )

    if current_reference_grid is None:
        raise RuntimeError(
            f"No grid was generated for {config.label}."
        )

    return processed_dates, current_reference_grid


def report_date_consistency(
    precipitation_dates: set[pd.Timestamp],
    pet_dates: set[pd.Timestamp],
) -> None:
    only_precipitation = sorted(
        precipitation_dates - pet_dates
    )

    only_pet = sorted(
        pet_dates - precipitation_dates
    )

    print()
    print("=" * 78)
    print("P/PET date consistency")
    print("=" * 78)
    print(
        f"Precipitation dates: "
        f"{len(precipitation_dates)}"
    )
    print(
        f"PET dates:           "
        f"{len(pet_dates)}"
    )
    print(
        f"Common dates:        "
        f"{len(precipitation_dates & pet_dates)}"
    )

    if only_precipitation:
        preview = ", ".join(
            date.strftime("%Y-%m-%d")
            for date in only_precipitation[:10]
        )

        print(
            f"Warning: {len(only_precipitation)} date(s) "
            f"occur only in P. First values: {preview}"
        )

    if only_pet:
        preview = ", ".join(
            date.strftime("%Y-%m-%d")
            for date in only_pet[:10]
        )

        print(
            f"Warning: {len(only_pet)} date(s) "
            f"occur only in PET. First values: {preview}"
        )

    if not only_precipitation and not only_pet:
        print(
            "The P and PET date sets match exactly."
        )


def main() -> int:
    start_date, end_date = validate_settings()
    working_grid = load_grid_from_geotiff(
        WORKING_GRID_PATH
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Configuration")
    print("-------------")
    print(f"P folder:       {P_NETCDF_FOLDER}")
    print(f"PET folder:     {PET_NETCDF_FOLDER}")
    print(f"Output root:    {OUTPUT_ROOT}")
    print(f"P variable:     {P_VARIABLE_NAME}")
    print(f"PET variable:   {PET_VARIABLE_NAME}")
    print(f"Working grid:   {WORKING_GRID_PATH}")
    print(f"Start date:     {start_date}")
    print(f"End date:       {end_date}")
    print(f"Overwrite:      {OVERWRITE_EXISTING}")
    print(f"Output nodata:  {OUTPUT_NODATA}")
    print(f"Resampling:     {REPROJECT_RESAMPLING}")

    precipitation_dates, reference_grid = convert_product(
        input_folder=P_NETCDF_FOLDER,
        output_root=OUTPUT_ROOT,
        config=P_CONFIG,
        pattern=NETCDF_PATTERN,
        nodata=OUTPUT_NODATA,
        start_date=start_date,
        end_date=end_date,
        overwrite=OVERWRITE_EXISTING,
        strict_date_check=STRICT_DATE_CHECK,
        reference_grid=working_grid,
    )

    pet_dates, _ = convert_product(
        input_folder=PET_NETCDF_FOLDER,
        output_root=OUTPUT_ROOT,
        config=PET_CONFIG,
        pattern=NETCDF_PATTERN,
        nodata=OUTPUT_NODATA,
        start_date=start_date,
        end_date=end_date,
        overwrite=OVERWRITE_EXISTING,
        strict_date_check=STRICT_DATE_CHECK,
        reference_grid=reference_grid,
    )

    if CHECK_P_PET_DATE_CONSISTENCY:
        report_date_consistency(
            precipitation_dates=precipitation_dates,
            pet_dates=pet_dates,
        )

    print()
    print("Conversion completed.")
    print(
        f"P output:   "
        f"{OUTPUT_ROOT / P_CONFIG.output_subfolder}"
    )
    print(
        f"PET output: "
        f"{OUTPUT_ROOT / PET_CONFIG.output_subfolder}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
