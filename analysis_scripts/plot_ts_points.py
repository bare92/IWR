#!/usr/bin/env python3
"""
Extract pixel time series from IWR/debug GeoTIFF folders at point locations.

For each point in a point shapefile:
  - extracts daily values from one or more GeoTIFF folders
  - saves one CSV time series
  - saves one PNG plot with one subplot per selected variable

Expected filename examples:
  IWR/iwr_20210101.tif
  IWR_debug/actual_evapotranspiration_for_balance/actual_evapotranspiration_for_balance_20210101.tif
  IWR_debug/actual_deep_percolation/actual_deep_percolation_20210101.tif
  IWR_debug/actual_runoff/actual_runoff_20210101.tif
  IWR_debug/soil_saturation/soil_saturation_20210101.tif

Usage
-----
Provide paths via CLI arguments or edit RUN_CONFIG below.

Minimal example (CLI)::

    python plot_ts_points.py \\
        --points /path/to/points.shp \\
        --out-dir /path/to/output \\
        --iwr-folder /path/to/model_output/IWR \\
        --debug-folder /path/to/model_output/IWR_debug \\
        --precipitation-folder /path/to/forcing/P \\
        --et0-folder /path/to/forcing/PET
"""

from pathlib import Path
import re
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==========================================================
# VARIABLE METADATA
# Static per-variable properties: pattern, label, units, group.
# Folder paths are provided at runtime via RUN_CONFIG or CLI args.
#
# groups:
#   model   — IWR model outputs (one subplot per variable)
#   forcing — meteorological inputs (written to a separate forcing plot)
#   overlay — plotted as a secondary axis on another variable's subplot
# ==========================================================

VARIABLE_METADATA = {
    "iwr": {
        "pattern": "iwr_*.tif",
        "label": "IWR",
        "units": "mm/day",
        "group": "model",
    },
    "actual_evapotranspiration": {
        "pattern": "actual_evapotranspiration_for_balance_*.tif",
        "label": "Actual evapotranspiration for balance",
        "units": "mm/day",
        "group": "model",
    },
    "kc_pixel": {
        "pattern": "kc_pixel_*.tif",
        "label": "Crop coefficient",
        "units": "-",
        "group": "overlay",
    },
    "deep_percolation": {
        "pattern": "actual_deep_percolation_*.tif",
        "label": "Deep percolation (actual)",
        "units": "mm/day",
        "group": "model",
    },
    "reference_deep_percolation": {
        "pattern": "reference_deep_percolation_*.tif",
        "label": "Deep percolation (reference)",
        "units": "mm/day",
        "group": "model",
    },
    "runoff": {
        "pattern": "actual_runoff_*.tif",
        "label": "Runoff",
        "units": "mm/day",
        "group": "model",
    },
    # Available root-zone water expressed as a fraction of TAW.
    # The debug GeoTIFF is written under the folder "soil_saturation" but the
    # data are actually available_water_fraction = soil_moisture / TAW.
    "available_water_fraction": {
        "pattern": "soil_saturation_*.tif",
        "label": "Available root-zone water / TAW",
        "units": "fraction",
        "group": "model",
    },
    # Kept for backward-compatibility; same data as available_water_fraction.
    "soil_saturation": {
        "pattern": "soil_saturation_*.tif",
        "label": "Available root-zone water / TAW",
        "units": "fraction",
        "group": "model",
    },
    # No-stress threshold as a fraction of TAW (= raw_fraction_of_taw in debug).
    "no_stress_threshold_fraction_of_taw": {
        "pattern": "raw_fraction_of_taw_*.tif",
        "label": "No-stress storage threshold / TAW",
        "units": "fraction",
        "group": "model",
    },
    # Soil storage state variables in mm.
    "actual_soil_storage": {
        "pattern": "actual_soil_moisture_*.tif",
        "label": "Actual soil storage",
        "units": "mm",
        "group": "model",
    },
    "reference_soil_storage": {
        "pattern": "reference_soil_moisture_*.tif",
        "label": "Reference soil storage",
        "units": "mm",
        "group": "model",
    },
    # Static soil-parameter layers (single-date rasters read once).
    "no_stress_storage_threshold_mm": {
        "pattern": "no_stress_storage_threshold_mm.tif",
        "label": "No-stress storage threshold",
        "units": "mm",
        "group": "static",
    },
    "taw_mm": {
        "pattern": "taw_mm.tif",
        "label": "TAW / field-capacity storage",
        "units": "mm",
        "group": "static",
    },
    # FAO-56 drainage diagnostic variables.
    "effective_precipitation": {
        "pattern": "effective_precipitation_*.tif",
        "label": "Effective precipitation",
        "units": "mm/day",
        "group": "model",
    },
    "provisional_storage_before_drainage": {
        "pattern": "provisional_storage_before_drainage_*.tif",
        "label": "Provisional storage before drainage",
        "units": "mm",
        "group": "model",
    },
    "excess_above_field_capacity": {
        "pattern": "excess_above_field_capacity_*.tif",
        "label": "Excess above field capacity",
        "units": "mm",
        "group": "model",
    },
    # Mass-balance residual (debug check).
    "actual_mass_balance_residual": {
        "pattern": "actual_mass_balance_residual_*.tif",
        "label": "Mass-balance residual",
        "units": "mm",
        "group": "model",
    },
    "precipitation": {
        "pattern": "*.tif",
        "label": "Precipitation",
        "units": "mm/day",
        "group": "forcing",
    },
    "et0": {
        "pattern": "*.tif",
        "label": "ET0",
        "units": "mm/day",
        "group": "forcing",
    },
    "etx": {
        "pattern": "etx_*.tif",
        "label": "ETx (stress-free crop ET)",
        "units": "mm/day",
        "group": "model",
    },
    "eta_stress": {
        "pattern": "eta_stress_*.tif",
        "label": "ETa (stress-limited)",
        "units": "mm/day",
        "group": "model",
    },
}


# ==========================================================
# SCRIPT RUN CONFIGURATION
# Set values here to run without CLI arguments.
# Any CLI argument provided will override these values.
#
# Required:
#   points     — path to a point shapefile
#   out_dir    — where CSVs and plots are written
#   iwr_folder — folder containing iwr_YYYYMMDD.tif files
#
# Optional:
#   debug_folder         — base folder of IWR debug variable subfolders
#   precipitation_folder — folder containing daily precipitation GeoTIFFs
#   et0_folder           — folder containing daily ET0/PET GeoTIFFs
# ==========================================================

RUN_CONFIG = {
    # Required inputs
    "points": "/home/fremen/data/projects/Burkina/00_Data_iwr/shapefiles/burkina_shp_iwr_check.shp",       # e.g. "/path/to/points.shp"
    "out_dir": Path(
        "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/iwr_plot_points"
    ),  # e.g. "/path/to/output"

    # Optional runtime controls
    "variables": [
        "iwr",
        "etx",
        "eta_stress",
        "actual_evapotranspiration",
        "kc_pixel",
        "deep_percolation",
        "reference_deep_percolation",
        "runoff",
        "available_water_fraction",
        "no_stress_threshold_fraction_of_taw",
        "actual_soil_storage",
        "reference_soil_storage",
        "no_stress_storage_threshold_mm",
        "taw_mm",
        "effective_precipitation",
        "provisional_storage_before_drainage",
        "excess_above_field_capacity",
        "actual_mass_balance_residual",
        "precipitation",
        "et0",
    ],
    "id_field": None,

    # Data folders (set to None to use CLI args)
    "iwr_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed/IWR",
    "debug_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed_debug",
    "etx_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed/ETx",           # e.g. "/path/to/model_output/ETx"
    "eta_stress_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed/ETa_stress",    # e.g. "/path/to/model_output/ETa_stress"
    "static_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed/Static",     # contains no_stress_storage_threshold_mm.tif, taw_mm.tif
    "precipitation_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/forcings/P",
    "et0_folder": "/home/fremen/data/projects/Burkina/00_Data_iwr/forcings/PET",
    "area_fraction_file": "/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output/IWR_theoretical_rainfed/Static/iwr_analysis_area_fraction.tif",   # e.g. "/path/to/model_output/Static/iwr_analysis_area_fraction.tif"
    "crs": None,  # e.g. "EPSG:4326"; used if rasters do not carry CRS metadata
    "start_date": "2022-01-01",  # e.g. "2021-01-01"
    "end_date": "2022-12-31",    # e.g. "2021-12-31"
}


DATE_REGEX_COMPACT = re.compile(r"(\d{8})")
DATE_REGEX_DASHED = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date_from_filename(path):
    """
    Extract date from filename and return pandas Timestamp.

    Supported formats:
      YYYYMMDD
      YYYY-MM-DD
    """
    name = path.name

    dashed_match = DATE_REGEX_DASHED.search(name)
    if dashed_match is not None:
        return pd.to_datetime(dashed_match.group(1), format="%Y-%m-%d")

    compact_match = DATE_REGEX_COMPACT.search(name)
    if compact_match is not None:
        return pd.to_datetime(compact_match.group(1), format="%Y%m%d")

    return None


def list_rasters_for_variable(variable_name, variable_config):
    """
    Return a dataframe with date and raster path for one variable.
    """
    folder = Path(variable_config["folder"])
    pattern = variable_config["pattern"]

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found for variable '{variable_name}': {folder}"
        )

    rows = []

    for tif_path in sorted(folder.glob(pattern)):
        if "cumulative" in tif_path.name.lower():
            continue

        date = parse_date_from_filename(tif_path)

        if date is None:
            print(f"Warning: skipping file without YYYYMMDD date: {tif_path}")
            continue

        rows.append(
            {
                "date": date,
                "variable": variable_name,
                "path": tif_path,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No daily GeoTIFF files found for variable '{variable_name}' "
            f"in {folder} with pattern '{pattern}'. "
            "Check that debug_mode=True was used and that the variable folder exists."
        )

    return pd.DataFrame(rows).sort_values("date")


def folder_has_matching_rasters(folder, pattern):
    folder = Path(folder)
    if not folder.exists():
        return False
    return any(folder.glob(pattern))


def build_raster_inventory(selected_variables, variables_config):
    """
    Build a dictionary:
      variable_name -> dataframe(date, variable, path)

    Static-group variables (single raster, no date series) are skipped here;
    they are read separately via ``extract_static_value_for_point``.
    """
    inventory = {}

    for variable_name in selected_variables:
        if variable_name not in variables_config:
            available = ", ".join(variables_config.keys())
            raise ValueError(
                f"Unknown variable '{variable_name}'. Available variables: {available}"
            )

        # Static-group variables are not time series; skip them here.
        if variables_config[variable_name].get("group") == "static":
            continue

        inventory[variable_name] = list_rasters_for_variable(
            variable_name=variable_name,
            variable_config=variables_config[variable_name],
        )

    return inventory


def read_reference_crs(inventory, crs_override=None, points_path_for_fallback=None):
    """
    Read CRS from rasters found in the inventory.

    If *crs_override* is provided (e.g. ``'EPSG:3035'``), it is returned
    directly without inspecting the raster.  Use this when the raster CRS
    is stored as a LOCAL_CS that rasterio/pyproj cannot resolve to an EPSG
    code.

    If no raster has CRS metadata and *points_path_for_fallback* is provided,
    the CRS of the points layer is used as a fallback.
    """
    if crs_override is not None:
        print(f"Using CRS override: {crs_override}")
        return crs_override

    checked_paths = 0
    first_non_epsg_crs = None
    first_non_epsg_path = None

    def _maybe_use_raster_crs(path, variable_name):
        nonlocal checked_paths, first_non_epsg_crs, first_non_epsg_path

        checked_paths += 1
        with rasterio.open(path) as src:
            raster_crs = src.crs

        if raster_crs is None:
            return None

        epsg = raster_crs.to_epsg()
        if epsg is not None:
            print(
                f"Using CRS from variable '{variable_name}': {path} -> EPSG:{epsg}",
                flush=True,
            )
            return f"EPSG:{epsg}"

        if first_non_epsg_crs is None:
            first_non_epsg_crs = raster_crs
            first_non_epsg_path = path

        return None

    # Fast path: check only the first raster of each variable.
    for variable_name, variable_df in inventory.items():
        if variable_df.empty:
            continue
        candidate = variable_df.iloc[0]["path"]
        found = _maybe_use_raster_crs(candidate, variable_name)
        if found is not None:
            return found

    # Fallback: sample a small number of additional rasters per variable.
    sample_limit_per_variable = 5
    for variable_name, variable_df in inventory.items():
        if variable_df.empty:
            continue
        n = min(sample_limit_per_variable, len(variable_df))
        for idx in range(1, n):
            candidate = variable_df.iloc[idx]["path"]
            found = _maybe_use_raster_crs(candidate, variable_name)
            if found is not None:
                return found

    if first_non_epsg_crs is not None:
        print(
            "Warning: could not resolve an EPSG code from rasters, "
            "but found a non-EPSG CRS. Using it directly. "
            "If reprojection fails, pass --crs EPSG:XXXX explicitly.",
            flush=True,
        )
        print(f"Using non-EPSG CRS from: {first_non_epsg_path}", flush=True)
        return first_non_epsg_crs

    if points_path_for_fallback is not None:
        try:
            points = gpd.read_file(points_path_for_fallback)
            points_crs = points.crs
        except Exception as exc:
            raise ValueError(
                "Could not find CRS metadata in rasters and failed to read point "
                f"layer CRS from: {points_path_for_fallback}"
            ) from exc

        if points_crs is not None:
            points_epsg = points_crs.to_epsg()
            if points_epsg is not None:
                fallback = f"EPSG:{points_epsg}"
            else:
                fallback = points_crs

            print(
                "Warning: no raster CRS metadata found. Falling back to point "
                f"layer CRS: {fallback}",
                flush=True,
            )
            return fallback

    raise ValueError(
        "Could not find any raster with CRS metadata in the selected inventory.\n"
        f"Checked {checked_paths} raster(s).\n"
        "Pass --crs EPSG:XXXX on the command line to force the target CRS, "
        "or ensure the point layer has a valid CRS for fallback."
    )


def load_points(point_shapefile, target_crs, id_field=None):
    """
    Load points and reproject them to raster CRS.
    """
    points = gpd.read_file(point_shapefile)

    if points.empty:
        raise ValueError(f"No features found in point shapefile: {point_shapefile}")

    if points.crs is None:
        raise ValueError(
            "Point shapefile has no CRS. Define its CRS before running extraction."
        )

    points_epsg = points.crs.to_epsg() if points.crs is not None else None

    if isinstance(target_crs, str) and target_crs.upper().startswith("EPSG:"):
        target_epsg = int(target_crs.split(":")[1])
    else:
        target_epsg = target_crs.to_epsg() if hasattr(target_crs, "to_epsg") else None

    print(f"Point CRS: {points.crs}")
    print(f"Point EPSG: {points_epsg}")
    print(f"Raster target CRS: {target_crs}")
    print(f"Raster target EPSG: {target_epsg}")

    if points_epsg is not None and target_epsg is not None and points_epsg == target_epsg:
        print("Point CRS already matches raster CRS; no reprojection needed.")
    else:
        try:
            print(f"Reprojecting points from {points.crs} to {target_crs}")
            points = points.to_crs(target_crs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to reproject the point shapefile to the raster CRS. "
                "This is usually caused by a malformed CRS definition or a GDAL/PROJ "
                "environment issue. Check that the point shapefile has a valid CRS and "
                "that the raster CRS can be interpreted. "
                "You can pass --crs EPSG:XXXX to override the raster CRS."
            ) from exc

    if id_field is not None and id_field not in points.columns:
        raise ValueError(
            f"id_field '{id_field}' not found in shapefile columns: {list(points.columns)}"
        )

    if id_field is None:
        points["point_id"] = [f"point_{i + 1}" for i in range(len(points))]
    else:
        points["point_id"] = points[id_field].astype(str)

    # Ensure geometries are points
    non_point = points.geometry.geom_type != "Point"
    if non_point.any():
        raise ValueError("The input shapefile must contain only point geometries.")

    return points


def extract_value_from_raster(raster_path, point_geom, nodata_to_nan=True):
    """
    Extract one pixel value at one point.
    """
    x = point_geom.x
    y = point_geom.y

    with rasterio.open(raster_path) as src:
        value = next(src.sample([(x, y)]))[0]
        nodata = src.nodata

    if nodata_to_nan and nodata is not None and value == nodata:
        return float("nan")

    return float(value)


def extract_static_value_for_point(static_path, point_geom):
    """
    Extract a single scalar from a static (non-time-series) raster.

    Returns float or NaN if the path does not exist or the pixel is nodata.
    """
    static_path = Path(static_path)
    if not static_path.is_file():
        return float("nan")
    return extract_value_from_raster(static_path, point_geom)


def read_drainage_scheme_from_debug_folder(debug_base):
    """
    Probe the first available debug GeoTIFF in *debug_base* and read the
    ``drainage_scheme`` tag from its GDAL metadata.

    Returns the scheme string (e.g. ``'fao56_excess_above_field_capacity'``)
    or ``None`` when the tag is absent.
    """
    debug_base = Path(debug_base)
    if not debug_base.is_dir():
        return None

    probe_patterns = [
        "soil_saturation/soil_saturation_*.tif",
        "actual_deep_percolation/actual_deep_percolation_*.tif",
        "actual_soil_moisture/actual_soil_moisture_*.tif",
    ]

    for pattern in probe_patterns:
        candidates = sorted(debug_base.glob(pattern))
        if candidates:
            try:
                with rasterio.open(candidates[0]) as src:
                    tags = src.tags()
                scheme = tags.get("drainage_scheme")
                if scheme:
                    return scheme
            except Exception:
                pass

    return None


def extract_timeseries_for_point(point_row, inventory, variables_config):
    """
    Extract all selected variables for one point.
    Output dataframe:
      date, variable_1, variable_2, ...
    """
    point_geom = point_row.geometry

    series_by_variable = []

    for variable_name, files_df in inventory.items():
        print(
            f"Extracting {variable_name} for {point_row['point_id']} "
            f"({len(files_df)} files)",
            flush=True,
        )
        values = []

        for i, (_, file_row) in enumerate(files_df.iterrows(), start=1):
            if i == 1 or i % 100 == 0 or i == len(files_df):
                print(
                    f"  {variable_name}: {i}/{len(files_df)}",
                    flush=True,
                )
            value = extract_value_from_raster(
                raster_path=file_row["path"],
                point_geom=point_geom,
            )

            values.append(
                {
                    "date": file_row["date"],
                    variable_name: value,
                }
            )

        variable_df = pd.DataFrame(values)
        series_by_variable.append(variable_df)

    out_df = series_by_variable[0]

    for variable_df in series_by_variable[1:]:
        out_df = out_df.merge(variable_df, on="date", how="outer")

    out_df = out_df.sort_values("date").reset_index(drop=True)

    return out_df


def split_variables_by_group(selected_variables, variables_config):
    """
    Split selected variables into model, forcing, overlay and static variables.
    """
    model_variables = []
    forcing_variables = []
    overlay_variables = []
    static_variables = []

    for variable_name in selected_variables:
        group = variables_config[variable_name].get("group", "model")

        if group == "forcing":
            forcing_variables.append(variable_name)
        elif group == "overlay":
            overlay_variables.append(variable_name)
        elif group == "static":
            static_variables.append(variable_name)
        else:
            model_variables.append(variable_name)

    return model_variables, forcing_variables, overlay_variables, static_variables


def plot_timeseries_for_point(
    df,
    point_id,
    selected_variables,
    variables_config,
    output_png,
    plot_title=None,
    overlay_variables=None,
):
    """
    Make one plot with one subplot per variable.
    """
    if overlay_variables is None:
        overlay_variables = []

    n_vars = len(selected_variables)

    fig, axes = plt.subplots(
        n_vars,
        1,
        figsize=(12, 3.2 * n_vars),
        sharex=True,
    )

    if n_vars == 1:
        axes = [axes]

    for ax, variable_name in zip(axes, selected_variables):
        label = variables_config[variable_name]["label"]
        units = variables_config[variable_name]["units"]

        if variable_name == "precipitation":
            ax.bar(df["date"], df[variable_name], width=1.0)
            ax.set_ylabel(f"{label}\n({units})")
        else:
            main_line = ax.plot(
                df["date"],
                df[variable_name],
                linewidth=1.4,
                label=label,
            )[0]
            ax.set_ylabel(f"{label}\n({units})")

            if (
                variable_name == "actual_evapotranspiration"
                and "kc_pixel" in overlay_variables
                and "kc_pixel" in df.columns
            ):
                kc_series = df["kc_pixel"]
                if kc_series.notna().any():
                    kc_label = variables_config["kc_pixel"]["label"]
                    kc_units = variables_config["kc_pixel"]["units"]

                    ax_kc = ax.twinx()
                    kc_line = ax_kc.plot(
                        df["date"],
                        kc_series,
                        linestyle="--",
                        linewidth=1.2,
                        color="tab:orange",
                        label=kc_label,
                    )[0]
                    ax_kc.set_ylabel(f"{kc_label}\n({kc_units})")
                    ax_kc.set_ylim(bottom=0.0)

                    ax.legend(
                        handles=[main_line, kc_line],
                        labels=[label, kc_label],
                        loc="upper right",
                    )
                else:
                    print(
                        f"Warning: no valid Kc values for point {point_id}; "
                        "Kc overlay skipped.",
                        flush=True,
                    )

        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Date")

    if plot_title is None:
        plot_title = f"Time series - {point_id}"

    fig.suptitle(plot_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_iwr1_comparison_for_point(
    df,
    point_id,
    output_png,
    area_fraction_value=None,
    plot_title=None,
):
    """
    Reproduce the plots_iwr_1.py style for a single pixel.

    Two-panel figure:
      Top   — ETx, ETa (stress-limited), ET deficit (= max(ETx - ETa, 0))
               and IWR (dashed line).  Mirrors daily_et_comparison.png.
      Bottom — Kc_pixel as a proxy for phenology-active state, same as
               the active-fraction panel in plots_iwr_1.py.

    Only ETx and eta_stress are required; iwr and kc_pixel are optional.
    Missing columns are silently skipped.
    """
    has_etx = "etx" in df.columns and df["etx"].notna().any()
    has_eta = "eta_stress" in df.columns and df["eta_stress"].notna().any()

    if not (has_etx and has_eta):
        print(
            f"Skipping iwr1-style plot for {point_id}: "
            "etx and/or eta_stress values are all NaN.",
            flush=True,
        )
        return

    has_iwr = "iwr" in df.columns and df["iwr"].notna().any()
    has_kc = "kc_pixel" in df.columns and df["kc_pixel"].notna().any()

    # Compute ET deficit on rows where both ETx and ETa are valid.
    et_deficit = pd.Series(float("nan"), index=df.index)
    both_valid = df["etx"].notna() & df["eta_stress"].notna()
    et_deficit[both_valid] = (df["etx"] - df["eta_stress"]).clip(lower=0.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    ax, ax_bottom = axes

    ax.plot(df["date"], df["etx"],       linewidth=1.2, label="ETx (stress-free crop ET)")
    ax.plot(df["date"], df["eta_stress"], linewidth=1.2, label="ETa (stress-limited)")
    ax.plot(df["date"], et_deficit,       linewidth=1.2, label="ET deficit (ETx − ETa)")
    if has_iwr:
        ax.plot(
            df["date"], df["iwr"],
            linewidth=1.2, linestyle="--", label="IWR",
        )

    ax.set_ylabel("mm/day")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Annotation matching plots_iwr_1.py style.
    annotation_lines = ["Pixel-level time series"]
    if area_fraction_value is not None and not (area_fraction_value != area_fraction_value):
        annotation_lines.append(f"analysis area fraction: {area_fraction_value:.4f}")
    annotation_lines.append("IWR is shown separately — it is not an ET component.")
    ax.text(
        0.01, 0.02,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    # Shade inactive periods (Kc == 0).
    _shade_inactive_periods(ax, df)

    # Bottom panel: Kc as proxy for phenology-active state.
    if has_kc:
        ax_bottom.plot(df["date"], df["kc_pixel"], linewidth=1.0, color="#666666")
        ax_bottom.set_ylabel("Kc pixel\n(-)")
    else:
        ax_bottom.set_ylabel("Kc pixel\n(not available)")

    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(True, alpha=0.3)

    if plot_title is None:
        plot_title = f"ET comparison (plots_iwr_1 style) — {point_id}"

    fig.suptitle(plot_title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _shade_inactive_periods(ax, df, kc_col="kc_pixel", alpha=0.08, color="#888888"):
    """
    Shade date spans where the crop is phenologically inactive (Kc == 0 or NaN).

    Uses ``kc_col`` as the proxy.  When the column is absent, nothing is drawn.
    """
    if kc_col not in df.columns:
        return

    kc = df[kc_col].fillna(0.0)
    dates = df["date"].to_numpy()
    inactive = (kc == 0.0).to_numpy()

    in_span = False
    span_start = None

    for i, (date, is_inactive) in enumerate(zip(dates, inactive)):
        if is_inactive and not in_span:
            span_start = date
            in_span = True
        elif not is_inactive and in_span:
            ax.axvspan(span_start, date, alpha=alpha, color=color, linewidth=0)
            in_span = False

    if in_span and span_start is not None:
        ax.axvspan(span_start, dates[-1], alpha=alpha, color=color, linewidth=0)


def plot_water_balance_diagnostic_for_point(
    df,
    point_id,
    output_png,
    static_values=None,
    drainage_scheme=None,
    area_fraction_value=None,
    plot_title=None,
):
    """
    Comprehensive water-balance diagnostic plot for one pixel.

    Panels (always shown when data are available):
      A. Soil storage [mm]: actual, reference, no-stress threshold, TAW
      B. Normalised fractions: available_water_fraction, no_stress_threshold_fraction
      C. Deep percolation [mm/day]: actual vs reference (separate lines)
      D. FAO-56 drainage detail (only when drainage_scheme is fao56_*):
            effective precipitation, provisional storage, excess above FC, deep percolation
      E. ET components [mm/day]: ETx and ETa (crop), AET for balance, background ET;
            IWR shown as a separate dashed line — NOT stacked with ET
      F. Mass-balance residual [mm] (when available) — zero line + max residual in title

    Inactive phenology periods are shaded in all panels.
    A warning annotation is added when drainage_scheme is unknown or mixed.
    """
    if static_values is None:
        static_values = {}

    dates = df["date"]

    # ---- helpers ----
    def col(name):
        """Return series or None."""
        if name in df.columns and df[name].notna().any():
            return df[name]
        return None

    def static(name):
        v = static_values.get(name)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
        return None

    # ---- decide which panels to show ----
    is_fao56 = drainage_scheme is not None and "fao56" in drainage_scheme.lower()

    has_soil_mm = any(
        col(c) is not None
        for c in ("actual_soil_storage", "reference_soil_storage")
    )
    has_frac = any(
        col(c) is not None
        for c in ("available_water_fraction", "soil_saturation",
                  "no_stress_threshold_fraction_of_taw")
    )
    has_percol = any(
        col(c) is not None
        for c in ("deep_percolation", "reference_deep_percolation")
    )
    has_fao56_detail = is_fao56 and any(
        col(c) is not None
        for c in ("effective_precipitation", "provisional_storage_before_drainage",
                  "excess_above_field_capacity")
    )
    has_et = any(
        col(c) is not None
        for c in ("etx", "eta_stress", "actual_evapotranspiration")
    )
    has_residual = col("actual_mass_balance_residual") is not None

    panel_names = []
    height_ratios = []

    if has_soil_mm:
        panel_names.append("soil_mm")
        height_ratios.append(3)
    if has_frac:
        panel_names.append("frac")
        height_ratios.append(2)
    if has_percol:
        panel_names.append("percol")
        height_ratios.append(2)
    if has_fao56_detail:
        panel_names.append("fao56")
        height_ratios.append(2)
    if has_et:
        panel_names.append("et")
        height_ratios.append(3)
    if has_residual:
        panel_names.append("residual")
        height_ratios.append(1.5)

    n_panels = len(panel_names)
    if n_panels == 0:
        print(
            f"Skipping water-balance diagnostic for {point_id}: no data available.",
            flush=True,
        )
        return

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 2.8 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    panel_axes = dict(zip(panel_names, axes))

    # ---- drainage scheme annotation / warning ----
    scheme_label = drainage_scheme if drainage_scheme else "unknown"
    warning_text = None
    if drainage_scheme is None:
        warning_text = (
            "WARNING: drainage_scheme could not be determined from debug outputs.\n"
            "Verify that outputs from a single consistent run are being used."
        )

    # ---- title ----
    if plot_title is None:
        plot_title = f"Water-balance diagnostic — {point_id}"

    title_parts = [plot_title, f"drainage_scheme={scheme_label}"]
    if area_fraction_value is not None and not np.isnan(area_fraction_value):
        title_parts.append(f"area_fraction={area_fraction_value:.4f}")

    # ---- Panel A: soil storage in mm ----
    if "soil_mm" in panel_axes:
        ax = panel_axes["soil_mm"]
        taw_val = static("taw_mm")
        threshold_val = static("no_stress_storage_threshold_mm")

        # Draw TAW and no-stress threshold as horizontal lines (static values)
        if taw_val is not None:
            ax.axhline(
                taw_val, linestyle="--", linewidth=1.2, color="#555555",
                label=f"TAW / FC storage ({taw_val:.1f} mm)",
            )
        if threshold_val is not None:
            ax.axhline(
                threshold_val, linestyle=":", linewidth=1.2, color="#aa6600",
                label=f"No-stress threshold ({threshold_val:.1f} mm)",
            )

        s_actual = col("actual_soil_storage")
        s_ref = col("reference_soil_storage")

        if s_actual is not None:
            ax.plot(dates, s_actual, linewidth=1.4, color="#1f77b4",
                    label="Actual soil storage")
        if s_ref is not None:
            ax.plot(dates, s_ref, linewidth=1.4, color="#2ca02c",
                    linestyle="--", label="Reference soil storage")

        ax.set_ylabel("Soil storage\n(mm)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        _shade_inactive_periods(ax, df)

    # ---- Panel B: normalised fractions ----
    if "frac" in panel_axes:
        ax = panel_axes["frac"]

        awf = col("available_water_fraction")
        if awf is None:
            awf = col("soil_saturation")
        nst = col("no_stress_threshold_fraction_of_taw")

        if nst is not None:
            # Plot as a time series in case it varies (e.g. multi-crop pixels)
            ax.plot(dates, nst, linewidth=1.0, linestyle=":",
                    color="#aa6600", label="No-stress threshold / TAW")
        if awf is not None:
            ax.plot(dates, awf, linewidth=1.4, color="#1f77b4",
                    label="Available root-zone water / TAW")

        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("Available root-zone\nwater / TAW")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        _shade_inactive_periods(ax, df)

    # ---- Panel C: deep percolation (actual vs reference) ----
    if "percol" in panel_axes:
        ax = panel_axes["percol"]

        dp_act = col("deep_percolation")
        dp_ref = col("reference_deep_percolation")

        if dp_act is not None:
            ax.plot(dates, dp_act, linewidth=1.4, color="#d62728",
                    label="Deep percolation (actual)")
        if dp_ref is not None:
            ax.plot(dates, dp_ref, linewidth=1.4, color="#9467bd",
                    linestyle="--", label="Deep percolation (reference)")

        ax.set_ylabel("Deep percolation\n(mm/day)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        _shade_inactive_periods(ax, df)

    # ---- Panel D: FAO-56 drainage detail ----
    if "fao56" in panel_axes:
        ax = panel_axes["fao56"]

        taw_val = static("taw_mm")
        prov = col("provisional_storage_before_drainage")
        exc = col("excess_above_field_capacity")
        dp_act = col("deep_percolation")
        peff = col("effective_precipitation")

        if peff is not None:
            ax.bar(dates, peff, width=1.0, alpha=0.4, color="#1f77b4",
                   label="Effective precipitation (mm/day)")
        if prov is not None:
            ax.plot(dates, prov, linewidth=1.2, color="#ff7f0e",
                    label="Provisional storage (mm)")
        if taw_val is not None:
            ax.axhline(taw_val, linestyle="--", linewidth=1.0, color="#555555",
                       label=f"TAW ({taw_val:.1f} mm)")
        if exc is not None:
            ax.fill_between(dates, 0, exc, alpha=0.35, color="#d62728",
                            label="Excess above FC (mm)")
        if dp_act is not None:
            ax.plot(dates, dp_act, linewidth=1.4, color="#d62728",
                    label="Deep percolation (mm/day)")

        ax.annotate(
            "Deep percolation > 0 only when\nprovisional storage > TAW",
            xy=(0.01, 0.97), xycoords="axes fraction",
            fontsize=8, va="top",
        )
        ax.set_ylabel("FAO-56 drainage\ndiagnostic (mm)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        _shade_inactive_periods(ax, df)

    # ---- Panel E: ET components + IWR (separate) ----
    if "et" in panel_axes:
        ax = panel_axes["et"]

        etx = col("etx")
        eta = col("eta_stress")
        aet = col("actual_evapotranspiration")
        iwr = col("iwr")

        if etx is not None:
            ax.plot(dates, etx, linewidth=1.4, color="#2ca02c",
                    label="ETx (stress-free crop ET)")
        if eta is not None:
            ax.plot(dates, eta, linewidth=1.4, color="#1f77b4",
                    label="ETa (crop, stress-limited)")
        if aet is not None:
            ax.plot(dates, aet, linewidth=1.0, color="#9467bd", linestyle="-.",
                    label="AET for balance (incl. background)")

        if iwr is not None:
            ax.plot(dates, iwr, linewidth=1.4, color="#d62728",
                    linestyle="--",
                    label="Theoretical IWR (not an ET component)")

        ax.set_ylabel("ET / IWR\n(mm/day)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        _shade_inactive_periods(ax, df)

        # Small annotation making separation explicit
        ax.annotate(
            "IWR is shown separately — it is not an ET component.",
            xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=8, va="bottom",
        )

    # ---- Panel F: mass-balance residual ----
    if "residual" in panel_axes:
        ax = panel_axes["residual"]
        residual = col("actual_mass_balance_residual")
        if residual is not None:
            ax.plot(dates, residual, linewidth=1.0, color="#333333")
            max_abs = residual.abs().max()
            ax.axhline(0, linewidth=0.8, color="red", linestyle="--")
            ax.set_ylabel("Mass-balance\nresidual (mm)")
            ax.set_title(
                f"Mass-balance residual — max |residual| = {max_abs:.2e} mm",
                fontsize=9, loc="left",
            )
            ax.grid(True, alpha=0.3)

    # ---- warning annotation ----
    if warning_text is not None:
        axes[0].annotate(
            warning_text,
            xy=(0.5, 1.01), xycoords="axes fraction",
            fontsize=9, ha="center", va="bottom", color="darkred",
        )

    axes[-1].set_xlabel("Date")
    fig.suptitle("\n".join(title_parts), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def make_safe_filename(value):
    """
    Clean point id for filenames.
    """
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    return value.strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Extract IWR/debug GeoTIFF time series at point locations."
    )

    parser.add_argument(
        "--points",
        default=RUN_CONFIG["points"],
        help="Path to point shapefile.",
    )

    parser.add_argument(
        "--out-dir",
        default=RUN_CONFIG["out_dir"],
        help="Output folder for CSV and plots.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=RUN_CONFIG["variables"],
        help=(
            "Variables to extract/plot. Available: "
            f"{' '.join(VARIABLE_METADATA.keys())}"
        ),
    )

    parser.add_argument(
        "--id-field",
        default=RUN_CONFIG["id_field"],
        help="Optional point attribute to use as point ID.",
    )

    parser.add_argument(
        "--iwr-folder",
        default=RUN_CONFIG["iwr_folder"],
        help="Folder containing iwr_YYYYMMDD.tif files.",
    )

    parser.add_argument(
        "--debug-folder",
        default=RUN_CONFIG["debug_folder"],
        help="Base folder containing debug variable subfolders.",
    )

    parser.add_argument(
        "--precipitation-folder",
        default=RUN_CONFIG["precipitation_folder"],
        help="Folder containing daily precipitation GeoTIFFs.",
    )

    parser.add_argument(
        "--et0-folder",
        default=RUN_CONFIG["et0_folder"],
        help="Folder containing daily ET0/PET GeoTIFFs.",
    )

    parser.add_argument(
        "--etx-folder",
        default=RUN_CONFIG["etx_folder"],
        help="Folder containing daily ETx GeoTIFFs (etx_YYYYMMDD.tif), written by the IWR model.",
    )

    parser.add_argument(
        "--eta-stress-folder",
        default=RUN_CONFIG["eta_stress_folder"],
        help="Folder containing daily ETa_stress GeoTIFFs (eta_stress_YYYYMMDD.tif), written by the IWR model.",
    )

    parser.add_argument(
        "--static-folder",
        default=RUN_CONFIG.get("static_folder"),
        help=(
            "Path to the model Static output folder (e.g. run_dir/Static). "
            "Used to read no_stress_storage_threshold_mm.tif and taw_mm.tif."
        ),
    )

    parser.add_argument(
        "--area-fraction-file",
        default=RUN_CONFIG["area_fraction_file"],
        help=(
            "Path to the static iwr_analysis_area_fraction.tif raster. "
            "When provided, the cropped-area fraction at each point is extracted "
            "and annotated on plots."
        ),
    )

    parser.add_argument(
        "--crs",
        default=RUN_CONFIG["crs"],
        help=(
            "Override the raster CRS used for reprojecting points, e.g. 'EPSG:3035'. "
            "Use this when the raster CRS is stored as a LOCAL_CS that rasterio cannot "
            "resolve to an EPSG code."
        ),
    )

    parser.add_argument(
        "--start-date",
        default=RUN_CONFIG["start_date"],
        help="Optional start date YYYY-MM-DD. If provided, only rasters from this date onward are used.",
    )

    parser.add_argument(
        "--end-date",
        default=RUN_CONFIG["end_date"],
        help="Optional end date YYYY-MM-DD. If provided, only rasters up to this date are used.",
    )

    args = parser.parse_args()

    if not args.points:
        raise ValueError(
            "No point shapefile provided. Set RUN_CONFIG['points'] in the script "
            "or pass --points on the command line."
        )

    if not args.out_dir:
        raise ValueError(
            "No output directory provided. Set RUN_CONFIG['out_dir'] in the script "
            "or pass --out-dir on the command line."
        )

    output_dir = Path(args.out_dir)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"

    # Build variables_config from static metadata + runtime folder paths.
    variables_config = {
        name: meta.copy() for name, meta in VARIABLE_METADATA.items()
    }

    # Assign folders from CLI / RUN_CONFIG.
    if args.iwr_folder:
        variables_config["iwr"]["folder"] = args.iwr_folder

    if args.debug_folder:
        debug_base = Path(args.debug_folder)

        # actual_evapotranspiration: prefer the *_for_balance subfolder,
        # fall back to the plain subfolder name if present.
        aet_folder_for_balance = debug_base / "actual_evapotranspiration_for_balance"
        aet_pattern_for_balance = "actual_evapotranspiration_for_balance_*.tif"
        aet_folder_plain = debug_base / "actual_evapotranspiration"
        aet_pattern_plain = "actual_evapotranspiration_*.tif"

        if folder_has_matching_rasters(aet_folder_for_balance, aet_pattern_for_balance):
            variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_for_balance)
            variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_for_balance
            variables_config["actual_evapotranspiration"]["label"] = "Actual evapotranspiration for balance"
        elif folder_has_matching_rasters(aet_folder_plain, aet_pattern_plain):
            variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_plain)
            variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_plain
            variables_config["actual_evapotranspiration"]["label"] = "Actual evapotranspiration"
        else:
            # Keep the preferred path so the FileNotFoundError is informative.
            variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_for_balance)
            variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_for_balance

        kc_legacy_folder = debug_base / "kc_pixel"
        kc_legacy_pattern = "kc_pixel_*.tif"
        kc_crop_output_folder = debug_base / "kc_crop_output"
        kc_crop_output_pattern = "kc_crop_output_*.tif"

        if folder_has_matching_rasters(kc_legacy_folder, kc_legacy_pattern):
            variables_config["kc_pixel"]["folder"] = str(kc_legacy_folder)
            variables_config["kc_pixel"]["pattern"] = kc_legacy_pattern
        elif folder_has_matching_rasters(kc_crop_output_folder, kc_crop_output_pattern):
            # Backward-compatible alias: the legacy plotting variable name
            # kc_pixel now reads the crop-facing coefficient written as
            # kc_crop_output by the current model.
            variables_config["kc_pixel"]["folder"] = str(kc_crop_output_folder)
            variables_config["kc_pixel"]["pattern"] = kc_crop_output_pattern
        else:
            variables_config["kc_pixel"]["folder"] = str(kc_crop_output_folder)
            variables_config["kc_pixel"]["pattern"] = kc_crop_output_pattern

        # Debug subfolders follow <debug_base>/<subfolder_name>/<variable_name>_*.tif.
        # Each variable's subfolder name matches the debug output written by iwr_model.py.
        debug_subfolder_map = {
            "deep_percolation":                 ("actual_deep_percolation",          "actual_deep_percolation_*.tif"),
            "reference_deep_percolation":       ("reference_deep_percolation",       "reference_deep_percolation_*.tif"),
            "runoff":                           ("actual_runoff",                    "actual_runoff_*.tif"),
            "soil_saturation":                  ("soil_saturation",                  "soil_saturation_*.tif"),
            "available_water_fraction":         ("soil_saturation",                  "soil_saturation_*.tif"),
            "no_stress_threshold_fraction_of_taw": ("raw_fraction_of_taw",          "raw_fraction_of_taw_*.tif"),
            "actual_soil_storage":              ("actual_soil_moisture",             "actual_soil_moisture_*.tif"),
            "reference_soil_storage":           ("reference_soil_moisture",          "reference_soil_moisture_*.tif"),
            "effective_precipitation":          ("effective_precipitation",          "effective_precipitation_*.tif"),
            "provisional_storage_before_drainage": ("provisional_storage_before_drainage", "provisional_storage_before_drainage_*.tif"),
            "excess_above_field_capacity":      ("excess_above_field_capacity",      "excess_above_field_capacity_*.tif"),
            "actual_mass_balance_residual":     ("actual_mass_balance_residual",     "actual_mass_balance_residual_*.tif"),
        }
        for var_name, (subfolder, pattern) in debug_subfolder_map.items():
            variables_config[var_name]["folder"] = str(debug_base / subfolder)
            variables_config[var_name]["pattern"] = pattern

    # Static-group variables: read once from the Static output folder.
    static_folder = getattr(args, "static_folder", None)
    if static_folder:
        static_dir = Path(static_folder)
        variables_config["no_stress_storage_threshold_mm"]["folder"] = str(static_dir)
        variables_config["no_stress_storage_threshold_mm"]["pattern"] = "no_stress_storage_threshold_mm.tif"
        variables_config["taw_mm"]["folder"] = str(static_dir)
        variables_config["taw_mm"]["pattern"] = "taw_mm.tif"

    if args.precipitation_folder:
        variables_config["precipitation"]["folder"] = args.precipitation_folder

    if args.et0_folder:
        variables_config["et0"]["folder"] = args.et0_folder

    if args.etx_folder:
        variables_config["etx"]["folder"] = args.etx_folder

    if args.eta_stress_folder:
        variables_config["eta_stress"]["folder"] = args.eta_stress_folder

    selected_variables = args.variables
    model_variables, forcing_variables, overlay_variables, static_variables = split_variables_by_group(
        selected_variables=selected_variables,
        variables_config=variables_config,
    )

    inventory = build_raster_inventory(
        selected_variables=selected_variables,
        variables_config=variables_config,
    )

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    if start_date is not None or end_date is not None:
        for variable_name, files_df in inventory.items():
            filtered = files_df.copy()
            if start_date is not None:
                filtered = filtered[filtered["date"] >= start_date]
            if end_date is not None:
                filtered = filtered[filtered["date"] <= end_date]

            if filtered.empty:
                raise FileNotFoundError(
                    f"No files left for variable '{variable_name}' after applying "
                    f"date filter start={args.start_date}, end={args.end_date}."
                )

            inventory[variable_name] = filtered.reset_index(drop=True)

    print("Raster inventory:", flush=True)
    for variable_name, files_df in inventory.items():
        print(
            f"  {variable_name}: {len(files_df)} files "
            f"from {files_df['date'].min().date()} "
            f"to {files_df['date'].max().date()} "
            f"| folder={variables_config[variable_name]['folder']} "
            f"| pattern={variables_config[variable_name]['pattern']}",
            flush=True,
        )

    raster_crs = read_reference_crs(
        inventory,
        crs_override=args.crs,
        points_path_for_fallback=args.points,
    )
    print(f"Using raster target CRS: {raster_crs}", flush=True)

    points = load_points(
        point_shapefile=args.points,
        target_crs=raster_crs,
        id_field=args.id_field,
    )

    print(f"Loaded {len(points)} point(s)", flush=True)
    print(f"Selected variables: {', '.join(selected_variables)}", flush=True)
    print(f"Raster CRS: {raster_crs}", flush=True)

    # Detect drainage_scheme from debug folder metadata once (same for all points).
    drainage_scheme = None
    if args.debug_folder:
        drainage_scheme = read_drainage_scheme_from_debug_folder(args.debug_folder)
        if drainage_scheme:
            print(f"Detected drainage_scheme from debug outputs: {drainage_scheme}", flush=True)
        else:
            print(
                "Warning: could not detect drainage_scheme from debug folder metadata.",
                flush=True,
            )

    # Extract the static area-fraction value once per point (if file provided).
    area_fraction_file = getattr(args, "area_fraction_file", None)
    area_fraction_by_point = {}
    if area_fraction_file and Path(area_fraction_file).is_file():
        print(f"Extracting area-fraction values from: {area_fraction_file}", flush=True)
        for _, point_row in points.iterrows():
            pid = point_row["point_id"]
            area_fraction_by_point[pid] = extract_value_from_raster(
                raster_path=Path(area_fraction_file),
                point_geom=point_row.geometry,
            )
    elif area_fraction_file:
        print(
            f"Warning: area_fraction_file not found, skipping: {area_fraction_file}",
            flush=True,
        )

    # Determine paths to static soil-parameter rasters.
    static_raster_paths = {}
    resolved_static_folder = getattr(args, "static_folder", None)
    if resolved_static_folder:
        static_dir = Path(resolved_static_folder)
        static_raster_paths["no_stress_storage_threshold_mm"] = (
            static_dir / "no_stress_storage_threshold_mm.tif"
        )
        static_raster_paths["taw_mm"] = static_dir / "taw_mm.tif"

    for _, point_row in points.iterrows():
        point_id = point_row["point_id"]
        safe_id = make_safe_filename(point_id)

        df = extract_timeseries_for_point(
            point_row=point_row,
            inventory=inventory,
            variables_config=variables_config,
        )

        # Extract static values (per-point scalar, not time series).
        static_values_for_point = {}
        for static_var, raster_path in static_raster_paths.items():
            static_values_for_point[static_var] = extract_static_value_for_point(
                raster_path, point_row.geometry,
            )

        csv_path = csv_dir / f"{safe_id}_timeseries.csv"

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        print(f"Written CSV:  {csv_path}", flush=True)

        if model_variables:
            model_png_path = plot_dir / f"{safe_id}_timeseries.png"

            plot_timeseries_for_point(
                df=df,
                point_id=point_id,
                selected_variables=model_variables,
                variables_config=variables_config,
                output_png=model_png_path,
                plot_title=f"IWR/debug time series - {point_id}",
                overlay_variables=overlay_variables,
            )

            print(f"Written model plot: {model_png_path}", flush=True)

        if forcing_variables:
            forcing_png_path = plot_dir / f"{safe_id}_forcing_timeseries.png"

            plot_timeseries_for_point(
                df=df,
                point_id=point_id,
                selected_variables=forcing_variables,
                variables_config=variables_config,
                output_png=forcing_png_path,
                plot_title=f"Forcing time series - {point_id}",
                overlay_variables=[],
            )

            print(f"Written forcing plot: {forcing_png_path}", flush=True)

        # -- plots_iwr_1.py style: ETx / ETa / ET-deficit / IWR comparison --
        iwr1_png_path = plot_dir / f"{safe_id}_et_comparison.png"
        area_frac = area_fraction_by_point.get(point_id)
        plot_iwr1_comparison_for_point(
            df=df,
            point_id=point_id,
            output_png=iwr1_png_path,
            area_fraction_value=area_frac,
            plot_title=f"ET / IWR comparison — {point_id}",
        )
        if iwr1_png_path.exists():
            print(f"Written iwr1-style plot: {iwr1_png_path}", flush=True)

        # -- Comprehensive water-balance diagnostic plot --
        wb_png_path = plot_dir / f"{safe_id}_water_balance_diagnostic.png"
        plot_water_balance_diagnostic_for_point(
            df=df,
            point_id=point_id,
            output_png=wb_png_path,
            static_values=static_values_for_point,
            drainage_scheme=drainage_scheme,
            area_fraction_value=area_frac,
            plot_title=f"Water-balance diagnostic — {point_id}",
        )
        if wb_png_path.exists():
            print(f"Written water-balance diagnostic: {wb_png_path}", flush=True)


if __name__ == "__main__":
    main()