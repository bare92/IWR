#!/usr/bin/env python3
"""
Distribution Check for IWR Dekadal Cumulative Data

This script:
1. Loads dekadal cumulative IWR raster files (already aggregated)
2. Groups rasters by climatological decade (month + dekad) across all years
3. Generates histograms with overlaid gamma distribution
4. Saves PNG visualizations for each of the 36 climatological decades
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import stats


# ============================================================
# USER SETTINGS - CONFIGURE HERE
# ============================================================

# Input directory containing dekadal cumulative IWR raster files
IWR_DIRECTORY = Path("/share/data/DAO/output_eraLand/IWR_aggregated/dekad")

# Output directory for plots
OUTPUT_DIRECTORY = Path("/share/archive/DAO/IWRanomaly/climatology/distribution_plots_gamma")

# Date range to analyze
START_DATE = "1991-01-01"
END_DATE = "2020-12-31"

# Optional month filter (inclusive).
# Use None to process all months, or set like (4, 9) for Apr-Sep.
MONTH_WINDOW = None

# Optional explicit nodata sentinels to remove in addition to raster nodata masks.
# Keep empty if nodata is correctly encoded in the raster metadata.
ADDITIONAL_NODATA_VALUES: List[float] = [0]

# File pattern and metadata extraction for:
# iwr_sum_YYYYMM_dD.tif (example: iwr_sum_199101_d3.tif)
IWR_FILENAME_PATTERN = "iwr_sum_*.tif"
IWR_FILE_REGEX = re.compile(
    r"iwr_sum_(?P<year>\d{4})(?P<month>\d{2})_d(?P<dekad>[123])\.tif$"
)

# Plot settings
FIGURE_SIZE = (14, 10)
FIGURE_DPI = 150
BINS = 50
ALPHA_HISTOGRAM = 0.7
ALPHA_GAMMA_CURVE = 1.0

# Histogram color
HISTOGRAM_COLOR = "steelblue"
GAMMA_CURVE_COLOR = "red"
GAMMA_CURVE_LINEWIDTH = 2.5

# Statistics displayed on plot
FONT_SIZE_TITLE = 16
FONT_SIZE_STATS = 11
FONT_SIZE_AXIS = 12

# Skip decades with too few valid data points
MIN_VALID_POINTS = 50

# ============================================================
# FUNCTIONS
# ============================================================


def parse_date(value: str) -> date:
    """Parse date string."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_in_window(month: int, month_window: Tuple[int, int] | None) -> bool:
    """Check if month is inside optional month window."""
    if month_window is None:
        return True

    start_month, end_month = month_window
    if start_month <= end_month:
        return start_month <= month <= end_month

    # Wrapped window, e.g. (11, 2) -> Nov to Feb.
    return month >= start_month or month <= end_month


def discover_dekad_files(input_dir: Path) -> Dict[Tuple[int, int, int], Path]:
    """Discover dekadal cumulative IWR raster files mapped by (year, month, dekad)."""
    files_by_key: Dict[Tuple[int, int, int], Path] = {}

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for path in sorted(input_dir.glob(IWR_FILENAME_PATTERN)):
        match = IWR_FILE_REGEX.fullmatch(path.name)
        if not match:
            continue

        try:
            year = int(match.group("year"))
            month = int(match.group("month"))
            dekad = int(match.group("dekad"))
        except (ValueError, IndexError) as exc:
            print(f"Warning: Cannot parse metadata from '{path.name}': {exc}")
            continue

        key = (year, month, dekad)
        if key in files_by_key:
            print(f"Warning: Duplicate file for {key}, using first occurrence")
        else:
            files_by_key[key] = path

    return files_by_key


def load_raster_data(file_path: Path) -> np.ma.MaskedArray:
    """Load raster data from GeoTIFF file as a masked array."""
    with rasterio.open(file_path) as src:
        # masked=True automatically masks pixels marked as nodata by raster metadata.
        data = src.read(1, masked=True)
    return data


def extract_valid_values(data: np.ma.MaskedArray) -> np.ndarray:
    """Extract valid values from raster, excluding nodata and invalid numbers."""
    # Flatten and remove masked (nodata) cells first.
    valid = data.compressed()

    # Remove NaN/Inf.
    valid = valid[np.isfinite(valid)]

    # Remove any user-provided extra nodata sentinels.
    if ADDITIONAL_NODATA_VALUES:
        invalid_mask = np.isin(valid, np.asarray(ADDITIONAL_NODATA_VALUES, dtype=valid.dtype))
        valid = valid[~invalid_mask]

    return valid


def create_gamma_plot(
    data: np.ndarray,
    dekad_key: Tuple[int, int],
    years_covered: Tuple[int, int],
    output_path: Path,
) -> None:
    """Create histogram with gamma curve overlay and summary statistics."""
    month, dekad = dekad_key
    start_year, end_year = years_covered

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.suptitle(
        f"IWR Gamma Check - Month {month:02d} Dekad {dekad} ({start_year}-{end_year})",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
    )

    # ========== Plot 1: Histogram with gamma curve ==========
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(
        data,
        bins=BINS,
        color=HISTOGRAM_COLOR,
        alpha=ALPHA_HISTOGRAM,
        edgecolor="black",
        density=True,
    )

    # Fit gamma distribution. Constrain location to zero only when data is strictly positive.
    if np.all(data > 0):
        shape, loc, scale = stats.gamma.fit(data, floc=0)
    else:
        shape, loc, scale = stats.gamma.fit(data)

    # Overlay gamma distribution
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 200)
    gamma_curve = stats.gamma.pdf(x, shape, loc=loc, scale=scale)
    ax.plot(
        x,
        gamma_curve,
        color=GAMMA_CURVE_COLOR,
        linewidth=GAMMA_CURVE_LINEWIDTH,
        label="Gamma Distribution",
    )

    ax.set_xlabel("IWR (mm)", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Density", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Histogram with Gamma Curve", fontsize=FONT_SIZE_STATS)
    ax.legend(fontsize=FONT_SIZE_STATS)
    ax.grid(True, alpha=0.3)

    # ========== Plot 2: Q-Q Plot ==========
    ax = axes[0, 1]
    stats.probplot(data, dist=stats.gamma, sparams=(shape, loc, scale), plot=ax)
    ax.set_title("Q-Q Plot", fontsize=FONT_SIZE_STATS)
    ax.grid(True, alpha=0.3)

    # ========== Plot 3: Box Plot ==========
    ax = axes[1, 0]
    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="darkred", linewidth=2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(HISTOGRAM_COLOR)
        patch.set_alpha(ALPHA_HISTOGRAM)
    ax.set_ylabel("IWR (mm)", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Box Plot", fontsize=FONT_SIZE_STATS)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(["IWR"])

    # ========== Plot 4: Statistics Text ==========
    ax = axes[1, 1]
    ax.axis("off")

    # Prepare statistics text
    stats_text = f"Sample Statistics:\n"
    stats_text += f"{'─' * 40}\n"
    stats_text += f"N (valid points): {len(data):,}\n"
    stats_text += f"Mean: {mu:.4f} mm\n"
    stats_text += f"Std Dev: {sigma:.4f} mm\n"
    stats_text += f"Min: {data.min():.4f} mm\n"
    stats_text += f"Q1 (25%): {np.percentile(data, 25):.4f} mm\n"
    stats_text += f"Median: {np.median(data):.4f} mm\n"
    stats_text += f"Q3 (75%): {np.percentile(data, 75):.4f} mm\n"
    stats_text += f"Max: {data.max():.4f} mm\n"
    stats_text += f"Skewness: {stats.skew(data):.4f}\n"
    stats_text += f"Kurtosis: {stats.kurtosis(data):.4f}\n"
    stats_text += "\nGamma Fit Parameters:\n"
    stats_text += f"shape (k): {shape:.4f}\n"
    stats_text += f"loc: {loc:.4f}\n"
    stats_text += f"scale (theta): {scale:.4f}\n"

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_STATS,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("IWR Gamma Distribution Check by Climatological Dekad")
    print("=" * 70)

    # Parse date range
    start_date = parse_date(START_DATE)
    end_date = parse_date(END_DATE)

    # Create output directory
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {IWR_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Month window: {MONTH_WINDOW if MONTH_WINDOW is not None else 'ALL'}")
    print(f"Minimum valid points per decade: {MIN_VALID_POINTS}")
    print()

    # Discover files
    print("Discovering dekadal cumulative IWR files...")
    files_by_key = discover_dekad_files(IWR_DIRECTORY)
    print(f"Found {len(files_by_key)} dekadal files")
    print()

    if not files_by_key:
        print("ERROR: No IWR files found!")
        return

    # Prepare climatological dekads: (month, dekad), optionally filtered by month window.
    climatological_decades = [
        (month, dekad)
        for month in range(1, 13)
        for dekad in (1, 2, 3)
        if month_in_window(month, MONTH_WINDOW)
    ]

    print(f"Processing {len(climatological_decades)} climatological decades...")
    print()

    processed_count = 0
    skipped_count = 0

    for month, dekad in climatological_decades:
        # Collect dekadal cumulative values for this month/dekad across all years.
        decade_arrays: List[np.ndarray] = []
        for (year, file_month, file_dekad), file_path in files_by_key.items():
            if year < start_date.year or year > end_date.year:
                continue
            if file_month != month or file_dekad != dekad:
                continue
            try:
                raster_data = load_raster_data(file_path)
                valid_values = extract_valid_values(raster_data)
            except Exception as e:
                print(f"Warning: Failed to load {file_path.name}: {e}")
                continue

            if valid_values.size > 0:
                decade_arrays.append(valid_values)

        if decade_arrays:
            decade_data = np.concatenate(decade_arrays)
        else:
            decade_data = np.array([])

        if len(decade_data) < MIN_VALID_POINTS:
            print(
                f"⊘ Month {month:02d} Dekad {dekad}: Skipped "
                f"({len(decade_data)} valid points < {MIN_VALID_POINTS})"
            )
            skipped_count += 1
            continue

        # Create output filename
        output_filename = f"iwr_gamma_climatology_m{month:02d}_d{dekad}.png"
        output_path = OUTPUT_DIRECTORY / output_filename

        # Create plot
        create_gamma_plot(
            decade_data,
            (month, dekad),
            (start_date.year, end_date.year),
            output_path,
        )

        processed_count += 1

    print()
    print("=" * 70)
    print(f"Processing Complete!")
    print(f"Processed: {processed_count} decades")
    print(f"Skipped: {skipped_count} decades (insufficient data)")
    print("=" * 70)


if __name__ == "__main__":
    main()
