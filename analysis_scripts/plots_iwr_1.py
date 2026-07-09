#!/usr/bin/env python3
"""
Plot IWR time series and yearly totals from daily GeoTIFF outputs.

Expected input filenames:
  iwr_YYYYMMDD.tif
  blue_et_YYYYMMDD.tif
  green_et_YYYYMMDD.tif

The script computes:
  - spatial mean in mm/day
  - spatial total volume in m3/day, if raster CRS is projected in metres

Outputs:
  - CSV summary
  - daily IWR plot
  - yearly IWR plot with AIDA reference
  - daily blue/green plot
  - yearly blue/green stacked plot
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================

INPUT_DIR = Path("/share/data/DAO/output_corine_ecm/IWR")
OUTPUT_DIR = Path("/share/data/DAO/output_corine_ecm/IWR_plots")

# Choose what to plot for annual totals:
#   "volume_m3" = total water volume over the model domain
#   "mean_mm"   = spatial mean depth
ANNUAL_PLOT_MODE = "volume_m3"

# AIDA reference layers created with iwr_AIDA.py
AIDA_IWR_MM_FILE = Path(
    "/share/data/DAO/auxiliary/AIDA_dataset/derived_iwr/"
    "aligned_to_model_grid/aida_iwr_blue_mm_yr_irrigated_area_aligned.tif"
)

AIDA_IWR_VOLUME_FILE = Path(
    "/share/data/DAO/auxiliary/AIDA_dataset/derived_iwr/"
    "aida_iwr_blue_volume_m3_yr.tif"
)

PLOT_AIDA_REFERENCE = True

# Mask aligned to the model grid: 1 = inside SIGRIAN irrigated districts, 0/nodata = outside
AIDA_MASK_FILE = Path(
    "/share/data/DAO/static/processed/"
    "distretti_irrigui_SIGRIAN_3035_1km_mask_aligned.tif"
)

# Valid precipitation mask on the same working grid:
# keep only pixels where value == 1
VALID_PRECIP_MASK_FILE = Path(
    "/share/data/DAO/static/processed/working_grid_3035_1km_precip_valid.tif"
)

# If True, negative values are ignored.
IGNORE_NEGATIVE_VALUES = True

# Optional: restrict analysis period.
# Use None to process all available dates.
DATE_START = None   # example: "2021-01-01"
DATE_END = None     # example: "2024-12-31"


# =========================
# FUNCTIONS
# =========================

DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")


def parse_date_from_name(path: Path) -> pd.Timestamp:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def list_daily_files(prefix: str) -> dict:
    """
    Return dictionary {date: path} for files like prefix_YYYYMMDD.tif.
    """
    files = sorted(INPUT_DIR.glob(f"{prefix}_????????.tif"))
    out = {}

    for f in files:
        try:
            date = parse_date_from_name(f)
            out[date] = f
        except ValueError:
            continue

    return out


def pixel_area_m2(src: rasterio.io.DatasetReader) -> float:
    """
    Estimate pixel area.

    This is correct if CRS is projected in metres, e.g. EPSG:3035.
    """
    transform = src.transform
    return abs(transform.a * transform.e)


def read_raster_stats(path: Path) -> dict:
    """
    Read one raster and return spatial mean and total volume.

    Assumes raster values are water depth in mm/day.
    Volume is calculated as:
        sum(mm) / 1000 * pixel_area_m2
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")

        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        arr[~np.isfinite(arr)] = np.nan

        if IGNORE_NEGATIVE_VALUES:
            arr[arr < 0] = np.nan

        valid_count = np.count_nonzero(np.isfinite(arr))

        if valid_count == 0:
            mean_mm = np.nan
            total_volume_m3 = np.nan
        else:
            mean_mm = np.nanmean(arr)
            area_m2 = pixel_area_m2(src)
            total_volume_m3 = np.nansum(arr) / 1000.0 * area_m2

        crs = src.crs.to_string() if src.crs else "unknown"

    return {
        "mean_mm": mean_mm,
        "volume_m3": total_volume_m3,
        "valid_pixels": valid_count,
        "crs": crs,
    }


def read_single_band_array(path: Path) -> np.ndarray:
    """
    Read a single-band raster and return a cleaned float64 array.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[~np.isfinite(arr)] = np.nan

        if IGNORE_NEGATIVE_VALUES:
            arr[arr < 0] = np.nan

    return arr


def read_aida_reference_value() -> float | None:
    """
    Read the masked AIDA reference value for the annual IWR plot.

    The AIDA layer used here is:
      aida_iwr_blue_mm_yr_irrigated_area_aligned.tif

    It is already aligned to the model grid. Before calculating the reference
    value, it is masked with:
      distretti_irrigui_SIGRIAN_3035_1km_mask_aligned.tif

        Additionally, analysis is restricted to pixels where:
            working_grid_3035_1km_precip_valid.tif == 1

        If ANNUAL_PLOT_MODE == "volume_m3":
        returns masked AIDA blue-water IWR [million m3/year]

    If ANNUAL_PLOT_MODE == "mean_mm":
        returns masked spatial mean AIDA blue-water IWR [mm/year]
    """

    if not PLOT_AIDA_REFERENCE:
        return None

    if not AIDA_IWR_MM_FILE.exists():
        print(f"WARNING: AIDA mm reference file not found: {AIDA_IWR_MM_FILE}")
        return None

    if not AIDA_MASK_FILE.exists():
        print(f"WARNING: AIDA mask file not found: {AIDA_MASK_FILE}")
        return None

    if not VALID_PRECIP_MASK_FILE.exists():
        print(f"WARNING: valid precipitation mask file not found: {VALID_PRECIP_MASK_FILE}")
        return None

    with (
        rasterio.open(AIDA_IWR_MM_FILE) as aida_src,
        rasterio.open(AIDA_MASK_FILE) as irrig_mask_src,
        rasterio.open(VALID_PRECIP_MASK_FILE) as precip_valid_src,
    ):

        if aida_src.shape != irrig_mask_src.shape:
            raise ValueError(
                "AIDA reference and mask have different shapes:\n"
                f"  AIDA: {aida_src.shape}\n"
                f"  Irrigation mask: {irrig_mask_src.shape}"
            )

        if aida_src.shape != precip_valid_src.shape:
            raise ValueError(
                "AIDA reference and precip-valid mask have different shapes:\n"
                f"  AIDA: {aida_src.shape}\n"
                f"  Precip-valid mask: {precip_valid_src.shape}"
            )

        if aida_src.transform != irrig_mask_src.transform:
            raise ValueError(
                "AIDA reference and mask have different transforms. "
                "They must be aligned to the same grid."
            )

        if aida_src.transform != precip_valid_src.transform:
            raise ValueError(
                "AIDA reference and precip-valid mask have different transforms. "
                "They must be aligned to the same grid."
            )

        aida = aida_src.read(1).astype("float64")
        irrig_mask = irrig_mask_src.read(1).astype("float64")
        precip_valid_mask = precip_valid_src.read(1).astype("float64")

        if aida_src.nodata is not None:
            aida[aida == aida_src.nodata] = np.nan

        if irrig_mask_src.nodata is not None:
            irrig_mask[irrig_mask == irrig_mask_src.nodata] = np.nan

        if precip_valid_src.nodata is not None:
            precip_valid_mask[precip_valid_mask == precip_valid_src.nodata] = np.nan

        aida[~np.isfinite(aida)] = np.nan
        irrig_mask[~np.isfinite(irrig_mask)] = np.nan
        precip_valid_mask[~np.isfinite(precip_valid_mask)] = np.nan

        if IGNORE_NEGATIVE_VALUES:
            aida[aida < 0] = np.nan

        # Keep only pixels inside SIGRIAN irrigation districts and where
        # precipitation forcing is marked valid (value == 1).
        valid_mask = (irrig_mask > 0) & (precip_valid_mask == 1)
        aida_masked = np.where(valid_mask, aida, np.nan)

        valid_count = np.count_nonzero(np.isfinite(aida_masked))

        if valid_count == 0:
            print(
                "WARNING: no valid AIDA pixels inside SIGRIAN mask with precip-valid == 1."
            )
            return None

        if ANNUAL_PLOT_MODE == "mean_mm":
            # Mean AIDA IWR inside SIGRIAN mask [mm/year]
            return float(np.nanmean(aida_masked))

        elif ANNUAL_PLOT_MODE == "volume_m3":
            # Convert masked AIDA depth to volume:
            # mm/year / 1000 * pixel_area_m2 = m3/year
            area_m2 = pixel_area_m2(aida_src)
            volume_m3 = np.nansum(aida_masked) / 1000.0 * area_m2

            # Convert to million m3/year, matching your bar plot
            return float(volume_m3 / 1e6)

        else:
            raise ValueError(f"Unknown ANNUAL_PLOT_MODE: {ANNUAL_PLOT_MODE}")


def build_timeseries() -> pd.DataFrame:
    iwr_files = list_daily_files("iwr")
    blue_files = list_daily_files("blue_et")
    green_files = list_daily_files("green_et")

    all_dates = sorted(set(iwr_files) | set(blue_files) | set(green_files))

    if DATE_START is not None:
        all_dates = [d for d in all_dates if d >= pd.to_datetime(DATE_START)]
    if DATE_END is not None:
        all_dates = [d for d in all_dates if d <= pd.to_datetime(DATE_END)]

    records = []

    print(f"Found {len(iwr_files)} IWR files")
    print(f"Found {len(blue_files)} blue ET files")
    print(f"Found {len(green_files)} green ET files")
    print(f"Processing {len(all_dates)} dates")

    for i, date in enumerate(all_dates, start=1):
        print(f"[{i}/{len(all_dates)}] {date.date()}")

        row = {"date": date}

        if date in iwr_files:
            stats = read_raster_stats(iwr_files[date])
            row["iwr_mean_mm"] = stats["mean_mm"]
            row["iwr_volume_m3"] = stats["volume_m3"]
            row["valid_pixels"] = stats["valid_pixels"]
            row["crs"] = stats["crs"]
        else:
            row["iwr_mean_mm"] = np.nan
            row["iwr_volume_m3"] = np.nan

        if date in blue_files:
            stats = read_raster_stats(blue_files[date])
            row["blue_mean_mm"] = stats["mean_mm"]
            row["blue_volume_m3"] = stats["volume_m3"]
        else:
            row["blue_mean_mm"] = np.nan
            row["blue_volume_m3"] = np.nan

        if date in green_files:
            stats = read_raster_stats(green_files[date])
            row["green_mean_mm"] = stats["mean_mm"]
            row["green_volume_m3"] = stats["volume_m3"]
        else:
            row["green_mean_mm"] = np.nan
            row["green_volume_m3"] = np.nan

        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("date")
    df["year"] = df["date"].dt.year

    return df


def save_daily_iwr_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["date"], df["iwr_mean_mm"], linewidth=1.2)

    ax.set_title("Daily irrigation water requirement")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spatial mean IWR [mm/day]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "daily_iwr_timeseries.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def save_yearly_iwr_plot(df: pd.DataFrame):
    """
    Annual model IWR bar plot with AIDA reference line.
    """

    aida_value = read_aida_reference_value()

    if ANNUAL_PLOT_MODE == "volume_m3":
        annual = df.groupby("year", as_index=False)["iwr_volume_m3"].sum()
        y = annual["iwr_volume_m3"] / 1e6
        ylabel = "Annual IWR [million m3/year]"
        title = "Annual irrigation water requirement"
        aida_label = "AIDA blue-water IWR"

    else:
        annual = df.groupby("year", as_index=False)["iwr_mean_mm"].sum()
        y = annual["iwr_mean_mm"]
        ylabel = "Annual spatial mean IWR [mm/year]"
        title = "Annual irrigation water requirement"
        aida_label = "AIDA blue-water IWR"

    years = annual["year"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(years, y, label="Model IWR")

    if aida_value is not None:
        ax.axhline(
            aida_value,
            linestyle="--",
            linewidth=2,
            label=f"{aida_label}: {aida_value:.1f}",
        )

        ax.text(
            x=len(years) - 0.5,
            y=aida_value,
            s=f"AIDA = {aida_value:.1f}",
            va="bottom",
            ha="right",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out = OUTPUT_DIR / "yearly_iwr_with_aida.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def save_daily_blue_green_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["date"], df["green_mean_mm"], label="Green water ET", linewidth=1.2)
    ax.plot(df["date"], df["blue_mean_mm"], label="Blue water ET", linewidth=1.2)
    ax.plot(df["date"], df["iwr_mean_mm"], label="IWR", linewidth=1.2, linestyle="--")

    ax.set_title("Daily IWR, green water and blue water")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spatial mean [mm/day]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "daily_iwr_blue_green.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def save_yearly_blue_green_plot(df: pd.DataFrame):
    if ANNUAL_PLOT_MODE == "volume_m3":
        annual = df.groupby("year", as_index=False)[
            ["green_volume_m3", "blue_volume_m3", "iwr_volume_m3"]
        ].sum()

        green = annual["green_volume_m3"] / 1e6
        blue = annual["blue_volume_m3"] / 1e6
        iwr = annual["iwr_volume_m3"] / 1e6
        ylabel = "Annual volume [million m3/year]"

    else:
        annual = df.groupby("year", as_index=False)[
            ["green_mean_mm", "blue_mean_mm", "iwr_mean_mm"]
        ].sum()

        green = annual["green_mean_mm"]
        blue = annual["blue_mean_mm"]
        iwr = annual["iwr_mean_mm"]
        ylabel = "Annual spatial mean [mm/year]"

    years = annual["year"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(years, green, label="Green water")
    ax.bar(years, blue, bottom=green, label="Blue water")
    ax.plot(years, iwr, marker="o", label="IWR")

    ax.set_title("Annual green and blue water components")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "yearly_iwr_blue_green.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


# =========================
# MAIN
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_timeseries()

    if df.empty:
        raise RuntimeError("No valid daily files found. Check INPUT_DIR and filenames.")

    csv_out = OUTPUT_DIR / "iwr_timeseries_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    crs_values = df["crs"].dropna().unique() if "crs" in df.columns else []
    if len(crs_values) > 0:
        print(f"Detected CRS: {crs_values[0]}")
        if "4326" in crs_values[0] or "longlat" in crs_values[0].lower():
            print(
                "WARNING: CRS appears geographic. Volume estimates may be wrong. "
                "Use mean_mm plots or reproject rasters to a metric CRS first."
            )

    save_daily_iwr_plot(df)
    save_yearly_iwr_plot(df)
    save_daily_blue_green_plot(df)
    save_yearly_blue_green_plot(df)

    print("Done.")


if __name__ == "__main__":
    main()



