#!/usr/bin/env python3

import calendar
import glob
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# ============================================================
# USER SETTINGS
# ============================================================

ANOMALY_DIRECTORY = Path("/share/archive/DAO/IWRanomaly/dekads/maps_gamma")
OUTPUT_FILE = Path(
    "/share/archive/DAO/IWRanomaly/dekads/plots03gamma/"
    "iwr_anomaly_dekads_timeseries.png"
)

# Use None to plot all available dates.
START_DATE = "2003-04-01"           # example: "2021-01-01"
END_DATE = "2003-09-30"              # example: "2025-12-31"

# Restrict to a window of calendar months (inclusive on both ends).
# Set to None to include all months.
MONTH_WINDOW = (4, 9)       # example: (4, 9) keeps April–September only

FIGURE_SIZE = (14, 6)
FIGURE_DPI = 200
OVERWRITE = True

# Fixed anomaly scale.
Z_MIN = -3.0
Z_MAX = 3.0

# Expected filenames:
# iwr_spi_199101_d1_gamma.tif

# FILENAME_REGEX = re.compile(
# r"iwr_zscore_(?P<year>\d{4})(?P<month>\d{2})_d(?P<dekad>[123]).tif$"


FILENAME_REGEX = re.compile(
    r"iwr_spi_(?P<year>\d{4})(?P<month>\d{2})_d(?P<dekad>[123])_gamma\.tif$"
)


# ============================================================
# FUNCTIONS
# ============================================================

def parse_optional_date(value):
    if value in (None, ""):
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def dekad_start_end(year, month, dekad):
    if dekad == 1:
        return date(year, month, 1), date(year, month, 10)
    if dekad == 2:
        return date(year, month, 11), date(year, month, 20)

    return date(year, month, 21), date(
        year, month, calendar.monthrange(year, month)[1]
    )


def dekad_midpoint(year, month, dekad):
    period_start, period_end = dekad_start_end(year, month, dekad)
    return date.fromordinal(
        (period_start.toordinal() + period_end.toordinal()) // 2
    )


def discover_anomaly_rasters():
    if not ANOMALY_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Anomaly directory does not exist: {ANOMALY_DIRECTORY}"
        )

    rasters = []

    tif_paths = sorted(glob.glob(str(ANOMALY_DIRECTORY / "*.tif")))
    for path_str in tif_paths:
        path = Path(path_str)
        match = FILENAME_REGEX.fullmatch(path.name)

        if match is None:
            continue

        year = int(match.group("year"))
        month = int(match.group("month"))
        dekad = int(match.group("dekad"))

        period_start, period_end = dekad_start_end(year, month, dekad)

        rasters.append(
            {
                "path": path,
                "year": year,
                "month": month,
                "dekad": dekad,
                "start_date": period_start,
                "end_date": period_end,
                "midpoint": dekad_midpoint(year, month, dekad),
            }
        )

    if not rasters:
        raise FileNotFoundError(
            f"No anomaly GeoTIFFs matching '{FILENAME_REGEX.pattern}' "
            f"were found in {ANOMALY_DIRECTORY}"
        )

    return sorted(rasters, key=lambda item: item["midpoint"])


def filter_by_date(rasters):
    start_date = parse_optional_date(START_DATE)
    end_date = parse_optional_date(END_DATE)

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise ValueError("START_DATE must not be later than END_DATE")

    selected = []

    for item in rasters:
        if start_date is not None and item["end_date"] < start_date:
            continue

        if end_date is not None and item["start_date"] > end_date:
            continue

        selected.append(item)

    if not selected:
        raise RuntimeError(
            "No anomaly rasters fall within the selected date range"
        )

    return selected


def spatial_mean_anomaly(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(np.float64)

    values = np.asarray(data.filled(np.nan), dtype=np.float64)
    valid = np.isfinite(values)

    if not np.any(valid):
        return np.nan

    return float(np.nanmean(values[valid]))


def build_timeseries(rasters):
    dates = []
    anomalies = []

    for index, item in enumerate(rasters, start=1):
        anomaly = spatial_mean_anomaly(item["path"])

        if not np.isfinite(anomaly):
            print(f"Skipping raster with no valid pixels: {item['path']}")
            continue

        dates.append(item["midpoint"])
        anomalies.append(anomaly)

        print(
            f"[{index}/{len(rasters)}] "
            f"{item['year']}-{item['month']:02d}-d{item['dekad']}: {anomaly:.3f}"
        )

    if not dates:
        raise RuntimeError(
            "No valid anomaly values were available for plotting"
        )

    return np.asarray(dates), np.asarray(anomalies, dtype=np.float64)


def add_anomaly_background(axis, dates):
    x_start = mdates.date2num(dates.min())
    x_end = mdates.date2num(dates.max())

    if x_start == x_end:
        x_start -= 1.0
        x_end += 1.0

    gradient = np.linspace(Z_MIN, Z_MAX, 512).reshape(-1, 1)

    image = axis.imshow(
        gradient,
        extent=[x_start, x_end, Z_MIN, Z_MAX],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=Z_MIN,
        vmax=Z_MAX,
        alpha=0.45,
        interpolation="bicubic",
        zorder=0,
    )

    return image


def plot_timeseries(dates, anomalies):
    if OUTPUT_FILE.exists() and not OVERWRITE:
        print(f"Output already exists, skipping: {OUTPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    background = add_anomaly_background(axis, dates)

    if MONTH_WINDOW is not None:
        month_start, month_end = MONTH_WINDOW
        in_window = np.array([month_start <= d.month <= month_end for d in dates])

        # In-window: real anomaly values; gaps elsewhere
        in_vals = np.where(in_window, anomalies, np.nan)
        # Out-of-window: flat zero line; gaps elsewhere
        out_vals = np.where(~in_window, 0.0, np.nan)

        axis.plot(dates, in_vals, color="black", linewidth=1.5, zorder=3)

        if np.any(~in_window):
            axis.plot(
                dates, out_vals,
                color="dimgray", linewidth=1.0, linestyle="--",
                alpha=0.55, zorder=2,
            )

        # Vertical lines at the start and end of each window period
        years = sorted({d.year for d in dates})
        for year in years:
            win_start = date(year, month_start, 1)
            win_end = date(
                year, month_end,
                calendar.monthrange(year, month_end)[1],
            )
            for boundary in (win_start, win_end):
                axis.axvline(
                    boundary,
                    color="steelblue", linewidth=0.9,
                    linestyle="-", alpha=0.55, zorder=4,
                )
    else:
        axis.plot(dates, anomalies, color="black", linewidth=1.5, zorder=3)

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        zorder=2,
    )

    axis.set_ylim(Z_MIN, Z_MAX)
    axis.set_xlim(dates.min(), dates.max())

    if MONTH_WINDOW is None:
        title = "Dekadal IWR anomaly – all months"
    else:
        m_start = calendar.month_abbr[MONTH_WINDOW[0]]
        m_end = calendar.month_abbr[MONTH_WINDOW[1]]
        title = f"Dekadal IWR anomaly – {m_start}–{m_end} (all years)"
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Spatial mean IWR anomaly (z-score)")

    locator = mdates.AutoDateLocator(
        minticks=6,
        maxticks=12,
    )
    formatter = mdates.ConciseDateFormatter(locator)

    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)

    axis.grid(
        True,
        axis="x",
        alpha=0.18,
        linewidth=0.7,
        zorder=1,
    )

    colorbar = figure.colorbar(
        background,
        ax=axis,
        pad=0.02,
        fraction=0.04,
    )
    colorbar.set_label("IWR anomaly (z-score)")
    colorbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    figure.tight_layout()
    figure.savefig(
        OUTPUT_FILE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {OUTPUT_FILE}")


def main():
    rasters = discover_anomaly_rasters()
    rasters = filter_by_date(rasters)

    print(f"Selected {len(rasters)} anomaly rasters")
    print(
        f"Period: {rasters[0]['start_date']} "
        f"to {rasters[-1]['end_date']}"
    )

    dates, anomalies = build_timeseries(rasters)
    plot_timeseries(dates, anomalies)


if __name__ == "__main__":
    main()