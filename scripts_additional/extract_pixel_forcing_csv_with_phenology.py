from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_CODE_FOLDER = PROJECT_ROOT / "simple_code"
PRECIPITATION_FOLDER = Path("/share/data/DAO/input/output_geotiffs/P")
PET_FOLDER = Path("/share/data/DAO/input/output_geotiffs/PET")
OUTPUT_CSV = Path("/share/data/DAO/input/forcing_pixel.csv")

PHENOLOGY_PATHS = {
    "phenoe1": Path("/share/data/DAO/static/processed/phenoe1_v04_aligned.tif"),
    "phenoe2": Path("/share/data/DAO/static/processed/phenoe2_v04_aligned.tif"),
    "phenom1": Path("/share/data/DAO/static/processed/phenom1_v04_aligned.tif"),
    "phenom2": Path("/share/data/DAO/static/processed/phenom2_v04_aligned.tif"),
    "phenonseasons": Path("/share/data/DAO/static/processed/phenonseasons_v04_aligned.tif"),
    "phenos1": Path("/share/data/DAO/static/processed/phenos1_v04_aligned.tif"),
    "phenos2": Path("/share/data/DAO/static/processed/phenos2_v04_aligned.tif"),
    "phenosen1": Path("/share/data/DAO/static/processed/phenosen1_v04_aligned.tif"),
    "phenosen2": Path("/share/data/DAO/static/processed/phenosen2_v04_aligned.tif"),
}

ROW = 85
COL = 195
START_DATE = None
END_DATE = None
MAX_P_MM_DAY = 300.0
MAX_PET_MM_DAY = 20.0
ALLOW_GAPS = False
CLIP_NEGATIVE_TO_ZERO = True
NODATA = -9999.0

for path in (PROJECT_CODE_FOLDER, PROJECT_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from simple_code.phenology_functions import (
        PHENOLOGY_GROWING,
        PHENOLOGY_INACTIVE,
        PHENOLOGY_MAXIMUM,
        PHENOLOGY_SENESCENCE,
        create_phenology_status_mask_from_date,
        date_to_dekad,
        load_phenology_layers,
    )
except ModuleNotFoundError:
    from phenology_functions import (  # type: ignore[reportMissingImports]
        PHENOLOGY_GROWING,
        PHENOLOGY_INACTIVE,
        PHENOLOGY_MAXIMUM,
        PHENOLOGY_SENESCENCE,
        create_phenology_status_mask_from_date,
        date_to_dekad,
        load_phenology_layers,
    )

STATUS_LABELS = {
    PHENOLOGY_INACTIVE: "inactive",
    PHENOLOGY_GROWING: "growing",
    PHENOLOGY_MAXIMUM: "maximum",
    PHENOLOGY_SENESCENCE: "senescence",
}

DATE_PATTERNS = (
    (re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)"), "%Y-%m-%d"),
    (re.compile(r"(?<!\d)(\d{8})(?!\d)"), "%Y%m%d"),
)


def parse_date_from_name(path):
    for pattern, date_format in DATE_PATTERNS:
        match = pattern.search(path.name)
        if match:
            return datetime.strptime(match.group(1), date_format).date()
    raise ValueError(f"No date found in filename: {path.name}")


def index_folder(folder):
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files found in {folder}")

    indexed = {}
    for path in files:
        try:
            current_date = parse_date_from_name(path)
        except ValueError:
            continue
        indexed.setdefault(current_date, path)

    if not indexed:
        raise ValueError(f"No dated GeoTIFF files found in {folder}")

    return indexed


def read_pixel(path, expected_grid=None):
    with rasterio.open(path) as src:
        if not (0 <= ROW < src.height and 0 <= COL < src.width):
            raise IndexError(
                f"Pixel row={ROW}, col={COL} is outside {path.name} "
                f"with shape ({src.height}, {src.width})"
            )

        grid = (src.height, src.width, src.transform, src.crs)

        if expected_grid is not None:
            expected_height, expected_width, expected_transform, expected_crs = expected_grid
            if (src.height, src.width) != (expected_height, expected_width):
                raise ValueError(f"Shape mismatch in {path}")
            if not src.transform.almost_equals(expected_transform):
                raise ValueError(f"Transform mismatch in {path}")
            if src.crs != expected_crs:
                raise ValueError(f"CRS mismatch in {path}")

        value = src.read(
            1,
            window=Window(col_off=COL, row_off=ROW, width=1, height=1),
            masked=True,
        )[0, 0]

        if np.ma.is_masked(value):
            return np.nan, grid

        return float(value), grid


def clean_forcing(value, maximum, variable_name, current_date):
    if not np.isfinite(value):
        return 0.0
    if CLIP_NEGATIVE_TO_ZERO and value < 0.0:
        value = 0.0
    if value > maximum:
        raise ValueError(
            f"{variable_name}={value} exceeds {maximum} on {current_date}"
        )
    return float(value)


def parse_optional_date(value):
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_selected_dates(p_files, pet_files):
    common_dates = sorted(set(p_files) & set(pet_files))
    if not common_dates:
        raise ValueError("P and PET folders have no dates in common")

    start_date = parse_optional_date(START_DATE) or common_dates[0]
    end_date = parse_optional_date(END_DATE) or common_dates[-1]

    if end_date < start_date:
        raise ValueError("END_DATE must be on or after START_DATE")

    selected = [d for d in common_dates if start_date <= d <= end_date]
    if not selected:
        raise ValueError("No forcing dates found in the selected period")

    expected = {
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    }
    missing = sorted(expected - set(selected))

    if missing and not ALLOW_GAPS:
        raise ValueError(
            f"Missing {len(missing)} forcing dates; first missing date: {missing[0]}"
        )

    return selected


def validate_phenology_grid(reference_grid):
    for name, path in PHENOLOGY_PATHS.items():
        _, grid = read_pixel(path, expected_grid=reference_grid)
        if grid != reference_grid:
            raise ValueError(f"Grid mismatch in phenology layer {name}")


def load_pixel_phenology():
    phenology_full = load_phenology_layers(PHENOLOGY_PATHS)
    phenology_pixel = {}
    static_values = {}

    for name, data in phenology_full.items():
        value = float(data[ROW, COL])
        if not np.isfinite(value) or value == NODATA:
            raise ValueError(f"Invalid value in phenology layer {name} at row={ROW}, col={COL}")
        phenology_pixel[name] = np.array([[value]], dtype=data.dtype)
        static_values[name] = int(round(value))

    return phenology_pixel, static_values


def main():
    p_files = index_folder(PRECIPITATION_FOLDER)
    pet_files = index_folder(PET_FOLDER)
    dates = get_selected_dates(p_files, pet_files)

    _, reference_grid = read_pixel(p_files[dates[0]])
    validate_phenology_grid(reference_grid)
    phenology, static_phenology = load_pixel_phenology()

    rows = []

    for index, current_date in enumerate(dates, start=1):
        p_value, _ = read_pixel(p_files[current_date], expected_grid=reference_grid)
        pet_value, _ = read_pixel(pet_files[current_date], expected_grid=reference_grid)

        p_value = clean_forcing(p_value, MAX_P_MM_DAY, "P", current_date)
        pet_value = clean_forcing(pet_value, MAX_PET_MM_DAY, "PET", current_date)

        status = int(
            create_phenology_status_mask_from_date(
                current_date=current_date,
                phenology=phenology,
                nodata=NODATA,
            )[0, 0]
        )

        row = {
            "date": current_date.strftime("%Y-%m-%d"),
            "P": p_value,
            "PET": pet_value,
            "current_dekad": date_to_dekad(current_date),
            "phenology_status": status,
            "phenology_label": STATUS_LABELS[status],
        }
        row.update(static_phenology)
        rows.append(row)

        if index == 1 or index % 250 == 0 or index == len(dates):
            print(f"Processed {index}/{len(dates)}")

    columns = [
        "date",
        "P",
        "PET",
        "current_dekad",
        "phenology_status",
        "phenology_label",
        "phenonseasons",
        "phenos1",
        "phenom1",
        "phenosen1",
        "phenoe1",
        "phenos2",
        "phenom2",
        "phenosen2",
        "phenoe2",
    ]

    result = pd.DataFrame(rows)[columns]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")

    print(f"Saved {len(result)} rows to {OUTPUT_CSV}")
    print(result["phenology_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
