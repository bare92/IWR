#!/usr/bin/env python3
"""
Extract pixel time series from IWR/debug GeoTIFF folders at point locations.

For each point in a point shapefile:
  - extracts daily values from one or more GeoTIFF folders
  - saves one CSV time series
  - saves one PNG plot with one subplot per selected variable

Expected filename examples:
  IWR/iwr_20210101.tif
  IWR_debug/actual_evapotranspiration/actual_evapotranspiration_20210101.tif
  IWR_debug/deep_percolation/deep_percolation_20210101.tif
  IWR_debug/runoff/runoff_20210101.tif
  IWR_debug/soil_saturation/soil_saturation_20210101.tif
"""

from pathlib import Path
import re
import argparse

import geopandas as gpd
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ==========================================================
# DEFAULT VARIABLE CONFIGURATION
# Modify these paths or pass them from command line if needed.
# ==========================================================

DEFAULT_VARIABLES = {
    "iwr": {
        "folder": "/share/data/DAO/output_aida/IWR",
        "pattern": "iwr_*.tif",
        "label": "IWR",
        "units": "mm/day",
        "group": "model",
    },
    "actual_evapotranspiration": {
        "folder": "/share/data/DAO/output_aida/IWR_debug/actual_evapotranspiration_for_balance",
        "pattern": "actual_evapotranspiration_for_balance_*.tif",
        "label": "Actual evapotranspiration for balance",
        "units": "mm/day",
        "group": "model",
    },
    "kc_pixel": {
        "folder": "/share/data/DAO/output_aida_dynamic_Kc/IWR_debug/kc_pixel",
        "pattern": "kc_pixel_*.tif",
        "label": "Crop coefficient",
        "units": "-",
        "group": "overlay",
    },
    "deep_percolation": {
        "folder": "/share/data/DAO/output_aida/IWR_debug/deep_percolation",
        "pattern": "deep_percolation_*.tif",
        "label": "Deep percolation",
        "units": "mm/day",
        "group": "model",
    },
    "runoff": {
        "folder": "/share/data/DAO/output_aida/IWR_debug/runoff",
        "pattern": "runoff_*.tif",
        "label": "Runoff",
        "units": "mm/day",
        "group": "model",
    },
    "soil_saturation": {
        "folder": "/share/data/DAO/output_aida/IWR_debug/soil_saturation",
        "pattern": "soil_saturation_*.tif",
        "label": "Soil saturation",
        "units": "fraction",
        "group": "model",
    },
    "precipitation": {
        "folder": "/share/data/DAO/input/output_geotiffs/P",
        "pattern": "*.tif",
        "label": "Precipitation",
        "units": "mm/day",
        "group": "forcing",
    },
    "et0": {
        "folder": "/share/data/DAO/input/output_geotiffs/PET",
        "pattern": "*.tif",
        "label": "ET0",
        "units": "mm/day",
        "group": "forcing",
    },
}


# ==========================================================
# SCRIPT RUN CONFIGURATION
# Define all runtime variables here if you want to run without CLI args.
# Any CLI argument provided will override these values.
# ==========================================================

RUN_CONFIG = {
    # Required inputs
    "points": "/share/data/DAO/auxiliary/shapefile_checks/point_check_crops.shp",  # e.g. "/path/to/points.shp"
    "out_dir": "/share/data/DAO/output_aida_dynamic_Kc/IWR_plot_point_ts",

    # Optional runtime controls
    "variables": [
        "iwr",
        "actual_evapotranspiration",
        "kc_pixel",
        "deep_percolation",
        "runoff",
        "soil_saturation",
        "precipitation",
        "et0",
    ],
    "id_field": None,

    # Data folders
    "iwr_folder": DEFAULT_VARIABLES["iwr"]["folder"],
    "debug_folder": "/share/data/DAO/output_aida_dynamic_Kc/IWR_debug",
    "precipitation_folder": DEFAULT_VARIABLES["precipitation"]["folder"],
    "et0_folder": DEFAULT_VARIABLES["et0"]["folder"],
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
    """
    inventory = {}

    for variable_name in selected_variables:
        if variable_name not in variables_config:
            available = ", ".join(variables_config.keys())
            raise ValueError(
                f"Unknown variable '{variable_name}'. Available variables: {available}"
            )

        inventory[variable_name] = list_rasters_for_variable(
            variable_name=variable_name,
            variable_config=variables_config[variable_name],
        )

    return inventory


def read_reference_crs(inventory):
    """
    Read CRS from the first raster found in the inventory.

    In this project, some model GeoTIFFs are written with a LOCAL_CS
    definition named 'ETRS89-extended / LAEA Europe'. This is actually
    the modelling grid in EPSG:3035, but rasterio/pyproj cannot always
    recover the EPSG code from the LOCAL_CS WKT.

    Therefore, when the raster CRS has no EPSG code and contains LAEA Europe
    or ETRS89-extended, force EPSG:3035.
    """
    first_variable = next(iter(inventory.keys()))
    first_path = inventory[first_variable].iloc[0]["path"]

    with rasterio.open(first_path) as src:
        raster_crs = src.crs

    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {first_path}")

    epsg = raster_crs.to_epsg()

    if epsg is not None:
        return f"EPSG:{epsg}"

    crs_text = raster_crs.to_wkt()

    if "LAEA Europe" in crs_text or "ETRS89-extended" in crs_text:
        print(
            "Raster CRS is stored as LOCAL_CS for ETRS89 / LAEA Europe. "
            "Forcing target CRS to EPSG:3035."
        )
        return "EPSG:3035"

    raise ValueError(
        "Could not determine raster EPSG code. "
        f"Raster CRS is:\n{raster_crs}"
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
                "that the raster CRS can be interpreted. For this IWR project, the model "
                "grid should normally be EPSG:3035."
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
    Split selected variables into model, forcing and overlay variables.
    """
    model_variables = []
    forcing_variables = []
    overlay_variables = []

    for variable_name in selected_variables:
        group = variables_config[variable_name].get("group", "model")

        if group == "forcing":
            forcing_variables.append(variable_name)
        elif group == "overlay":
            overlay_variables.append(variable_name)
        else:
            model_variables.append(variable_name)

    return model_variables, forcing_variables, overlay_variables


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
        help="Path to point shapefile. Defaults to RUN_CONFIG['points'].",
    )

    parser.add_argument(
        "--out-dir",
        default=RUN_CONFIG["out_dir"],
        help="Output folder for CSV and plots. Defaults to RUN_CONFIG['out_dir'].",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=RUN_CONFIG["variables"],
        help=(
            "Variables to extract/plot. Available: "
            f"{' '.join(DEFAULT_VARIABLES.keys())}"
        ),
    )

    parser.add_argument(
        "--id-field",
        default=RUN_CONFIG["id_field"],
        help="Optional point attribute to use as point ID. Defaults to RUN_CONFIG['id_field'].",
    )

    parser.add_argument(
        "--iwr-folder",
        default=RUN_CONFIG["iwr_folder"],
        help="Folder containing iwr_YYYYMMDD.tif files. Defaults to RUN_CONFIG['iwr_folder'].",
    )

    parser.add_argument(
        "--debug-folder",
        default=RUN_CONFIG["debug_folder"],
        help="Base folder containing debug variable subfolders. Defaults to RUN_CONFIG['debug_folder'].",
    )

    parser.add_argument(
        "--precipitation-folder",
        default=RUN_CONFIG["precipitation_folder"],
        help="Folder containing daily precipitation GeoTIFFs. Defaults to RUN_CONFIG['precipitation_folder'].",
    )

    parser.add_argument(
        "--et0-folder",
        default=RUN_CONFIG["et0_folder"],
        help="Folder containing daily ET0/PET GeoTIFFs. Defaults to RUN_CONFIG['et0_folder'].",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date YYYY-MM-DD. If provided, only rasters from this date onward are used.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
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

    variables_config = {
        name: config.copy() for name, config in DEFAULT_VARIABLES.items()
    }

    # Override paths from command line
    variables_config["iwr"]["folder"] = args.iwr_folder

    debug_base = Path(args.debug_folder)

    aet_folder_for_balance = debug_base / "actual_evapotranspiration_for_balance"
    aet_pattern_for_balance = "actual_evapotranspiration_for_balance_*.tif"

    aet_folder_old = debug_base / "actual_evapotranspiration"
    aet_pattern_old = "actual_evapotranspiration_*.tif"

    if folder_has_matching_rasters(aet_folder_for_balance, aet_pattern_for_balance):
        variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_for_balance)
        variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_for_balance
        variables_config["actual_evapotranspiration"]["label"] = "Actual evapotranspiration for balance"
    elif folder_has_matching_rasters(aet_folder_old, aet_pattern_old):
        variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_old)
        variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_old
        variables_config["actual_evapotranspiration"]["label"] = "Actual evapotranspiration"
    else:
        # Keep the preferred new path so the later FileNotFoundError is informative.
        variables_config["actual_evapotranspiration"]["folder"] = str(aet_folder_for_balance)
        variables_config["actual_evapotranspiration"]["pattern"] = aet_pattern_for_balance

    variables_config["deep_percolation"]["folder"] = str(
        debug_base / "deep_percolation"
    )
    variables_config["kc_pixel"]["folder"] = str(
        debug_base / "kc_pixel"
    )
    variables_config["kc_pixel"]["pattern"] = "kc_pixel_*.tif"
    variables_config["runoff"]["folder"] = str(
        debug_base / "runoff"
    )
    variables_config["soil_saturation"]["folder"] = str(
        debug_base / "soil_saturation"
    )
    variables_config["precipitation"]["folder"] = args.precipitation_folder
    variables_config["et0"]["folder"] = args.et0_folder

    selected_variables = args.variables
    model_variables, forcing_variables, overlay_variables = split_variables_by_group(
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

    raster_crs = read_reference_crs(inventory)
    print(f"Using raster target CRS: {raster_crs}", flush=True)

    points = load_points(
        point_shapefile=args.points,
        target_crs=raster_crs,
        id_field=args.id_field,
    )

    print(f"Loaded {len(points)} point(s)", flush=True)
    print(f"Selected variables: {', '.join(selected_variables)}", flush=True)
    print(f"Raster CRS: {raster_crs}", flush=True)

    for _, point_row in points.iterrows():
        point_id = point_row["point_id"]
        safe_id = make_safe_filename(point_id)

        df = extract_timeseries_for_point(
            point_row=point_row,
            inventory=inventory,
            variables_config=variables_config,
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


if __name__ == "__main__":
    main()