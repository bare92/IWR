"""
check_forcing_files.py
======================
Scan all GeoTIFFs in the precipitation and ET0 forcing folders and report:

  - Total number of files
  - Files with 0 valid pixels
  - Files with valid percentage below the configured threshold
  - Files with values above the configured maximum thresholds
  - Per-file statistics: min / p50 / p95 / p99 / max

A summary is printed to stdout and a full per-file report is written to
``forcing_file_report.csv`` inside the current working directory (or to the
IWR output folder when ``iwr_output_folder`` is configured).

Usage
-----
    python check_forcing_files.py                 # uses config/config.json
    python check_forcing_files.py path/to/cfg.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

_MISSING = object()


def _nested_get(config, path, default=_MISSING):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _cfg_get(config, nested_path, fallback_keys=(), default=_MISSING):
    value = _nested_get(config, nested_path, default=_MISSING)
    if value is not _MISSING:
        return value

    for key in fallback_keys:
        if key in config:
            return config[key]

    if default is not _MISSING:
        return default

    tried = [nested_path, *fallback_keys]
    raise KeyError(f"Missing required config key(s): {tried}")

# ---------------------------------------------------------------------------
# Helpers (local copies so the script can run standalone)
# ---------------------------------------------------------------------------

_FILL_SENTINELS = [
    (-9999, 1.0),  # covers -9999, -9999.0, -9999.9
    ( 9999, 1.0),  # covers  9999,  9999.0,  9999.9
]


def _mask_fill_values(data, nodata=-9999.0):
    """Apply common fill-value masks in-place and return the array."""
    data = data.astype(np.float32)
    data[data > 1e19] = nodata
    data[data < -1e19] = nodata
    for fv, tol in _FILL_SENTINELS:
        data[np.abs(data - fv) <= tol] = nodata
    return data


def _array_stats(arr, nodata=-9999.0):
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr) & (arr != nodata)]
    _nan = float("nan")
    if valid.size == 0:
        return dict(valid_count=0, valid_percent=0.0,
                    min=_nan, p50=_nan, p95=_nan, p99=_nan, max=_nan)
    return dict(
        valid_count=int(valid.size),
        valid_percent=100.0 * valid.size / arr.size,
        min=float(np.min(valid)),
        p50=float(np.percentile(valid, 50)),
        p95=float(np.percentile(valid, 95)),
        p99=float(np.percentile(valid, 99)),
        max=float(np.max(valid)),
    )


def _scan_folder(folder, label, nodata, max_value, min_valid_pct):
    """
    Scan all .tif files in *folder*.

    Returns
    -------
    rows : list[dict]   per-file statistics rows
    issues : list[str]  human-readable problem lines
    """
    folder = Path(folder)
    tif_files = sorted(folder.glob("*.tif"))

    rows = []
    issues = []

    for tif in tif_files:
        with rasterio.open(tif) as src:
            raw = src.read(1).astype(np.float32)
            src_nodata = src.nodata

        # Convert declared nodata, then fill-value sentinels.
        if src_nodata is not None:
            raw[raw == src_nodata] = nodata
        raw = _mask_fill_values(raw, nodata=nodata)

        stats = _array_stats(raw, nodata=nodata)
        row = {"folder": label, "file": tif.name, **stats}

        # Flag problems
        file_issues = []
        if stats["valid_count"] == 0:
            file_issues.append("NO VALID PIXELS")
        elif stats["valid_percent"] < min_valid_pct:
            file_issues.append(
                f"low valid ({stats['valid_percent']:.2f}% < {min_valid_pct:.2f}%)"
            )
        if max_value is not None and stats["max"] != stats["max"]:
            pass  # NaN – already flagged above
        elif max_value is not None and stats["max"] > max_value:
            file_issues.append(
                f"max={stats['max']:.2f} exceeds threshold={max_value}"
            )

        row["issues"] = "; ".join(file_issues) if file_issues else ""

        if file_issues:
            for msg in file_issues:
                issues.append(f"  [{label}] {tif.name}: {msg}")

        rows.append(row)

    return rows, issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    with open(config_path) as fh:
        config = json.load(fh)

    nodata = -9999.0
    max_precip = _cfg_get(
        config,
        "options.max_precipitation_mm_day",
        fallback_keys=("max_precipitation_mm_day",),
        default=300,
    )
    max_et0 = _cfg_get(
        config,
        "options.max_et0_mm_day",
        fallback_keys=("max_et0_mm_day",),
        default=20,
    )
    min_valid_pct = _cfg_get(
        config,
        "options.min_valid_forcing_fraction",
        fallback_keys=("min_valid_forcing_fraction",),
        default=0.01,
    ) * 100.0
    precip_folder = _cfg_get(
        config,
        "datasets.forcing.precipitation_geotiff_folder",
        fallback_keys=("precipitation_geotiff_folder", "precipitation_folder"),
        default=None,
    )
    et0_folder = _cfg_get(
        config,
        "datasets.forcing.et0_geotiff_folder",
        fallback_keys=("et0_geotiff_folder", "et0_folder"),
        default=None,
    )

    explicit_output_folder = _cfg_get(
        config,
        "outputs.iwr_output_folder",
        fallback_keys=("iwr_output_folder",),
        default=None,
    )
    if explicit_output_folder is not None:
        output_folder = Path(explicit_output_folder)
    else:
        output_base = _cfg_get(config, "outputs.output_base", fallback_keys=("output_base",), default=None)
        run_name = _cfg_get(config, "outputs.run_name", fallback_keys=("run_name",), default=None)
        output_folder = Path(output_base) / run_name if output_base and run_name else Path(".")

    print("=" * 70)
    print("FORCING FILE DIAGNOSTIC SCAN")
    print("=" * 70)
    print(f"Precipitation folder : {precip_folder}")
    print(f"ET0 folder           : {et0_folder}")
    print(f"Max precipitation    : {max_precip} mm/day")
    print(f"Max ET0              : {max_et0} mm/day")
    print(f"Min valid fraction   : {min_valid_pct:.2f}%")
    print()

    all_rows   = []
    all_issues = []

    for folder, label, max_val in [
        (precip_folder, "precipitation", max_precip),
        (et0_folder,    "et0",           max_et0),
    ]:
        if folder is None:
            print(f"WARNING: folder not configured for {label}, skipping.")
            continue
        rows, issues = _scan_folder(folder, label, nodata, max_val, min_valid_pct)
        all_rows.extend(rows)
        all_issues.extend(issues)
        n_files   = len(rows)
        n_empty   = sum(1 for r in rows if r["valid_count"] == 0)
        n_low     = sum(1 for r in rows if 0 < r["valid_percent"] < min_valid_pct)
        n_toohigh = sum(
            1 for r in rows
            if r["max"] == r["max"] and r["max"] > max_val
        )
        print(f"[{label}]")
        print(f"  Total files        : {n_files}")
        print(f"  Empty (0 valid px) : {n_empty}")
        print(f"  Low valid (<{min_valid_pct:.0f}%)   : {n_low}")
        print(f"  Above max value    : {n_toohigh}")
        print()

    if all_issues:
        print("PROBLEMS DETECTED:")
        for line in all_issues:
            print(line)
        print()
    else:
        print("No problems detected.")
        print()

    # Write CSV report
    csv_path = output_folder / "forcing_file_report.csv"
    output_folder.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "folder", "file",
        "valid_count", "valid_percent",
        "min", "p50", "p95", "p99", "max",
        "issues",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Full report written to: {csv_path}")


if __name__ == "__main__":
    main()
