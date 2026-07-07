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
        "folder": "/share/data/DAO/output/IWR",
        "pattern": "iwr_*.tif",
        "label": "IWR",
        "units": "mm/day",
        "group": "model",
    },
    "actual_evapotranspiration": {
        "folder": "/share/data/DAO/output/IWR_debug/actual_evapotranspiration",
        "pattern": "actual_evapotranspiration_*.tif",
        "label": "Actual evapotranspiration",
        "units": "mm/day",
        "group": "model",
    },
    "deep_percolation": {
        "folder": "/share/data/DAO/output/IWR_debug/deep_percolation",
        "pattern": "deep_percolation_*.tif",
        "label": "Deep percolation",
        "units": "mm/day",
        "group": "model",
    },
    "runoff": {
        "folder": "/share/data/DAO/output/IWR_debug/runoff",
        "pattern": "runoff_*.tif",
        "label": "Runoff",
        "units": "mm/day",
        "group": "model",
    },
    "soil_saturation": {
        "folder": "/share/data/DAO/output/IWR_debug/soil_saturation",
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
        values = []

        for _, file_row in files_df.iterrows():
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
    Split selected variables into model/debug variables and forcing variables.
    """
    model_variables = []
    forcing_variables = []

    for variable_name in selected_variables:
        group = variables_config[variable_name].get("group", "model")

        if group == "forcing":
            forcing_variables.append(variable_name)
        else:
            model_variables.append(variable_name)

    return model_variables, forcing_variables


def plot_timeseries_for_point(
    df,
    point_id,
    selected_variables,
    variables_config,
    output_png,
    plot_title=None,
):
    """
    Make one plot with one subplot per variable.
    """
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
        else:
            ax.plot(df["date"], df[variable_name], linewidth=1.4)
        ax.set_ylabel(f"{label}\n({units})")
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
        required=True,
        help="Path to point shapefile.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for CSV and plots.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "iwr",
            "actual_evapotranspiration",
            "deep_percolation",
            "runoff",
            "soil_saturation",
            "precipitation",
            "et0",
        ],
        help=(
            "Variables to extract/plot. Available: "
            "iwr actual_evapotranspiration deep_percolation runoff "
            "soil_saturation precipitation et0"
        ),
    )

    parser.add_argument(
        "--id-field",
        default=None,
        help="Optional point attribute to use as point ID.",
    )

    parser.add_argument(
        "--iwr-folder",
        default=DEFAULT_VARIABLES["iwr"]["folder"],
        help="Folder containing iwr_YYYYMMDD.tif files.",
    )

    parser.add_argument(
        "--debug-folder",
        default="/share/data/DAO/output/IWR_debug",
        help="Base folder containing debug variable subfolders.",
    )

    parser.add_argument(
        "--precipitation-folder",
        default=DEFAULT_VARIABLES["precipitation"]["folder"],
        help="Folder containing daily precipitation GeoTIFFs.",
    )

    parser.add_argument(
        "--et0-folder",
        default=DEFAULT_VARIABLES["et0"]["folder"],
        help="Folder containing daily ET0/PET GeoTIFFs.",
    )

    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"

    variables_config = DEFAULT_VARIABLES.copy()

    # Override paths from command line
    variables_config["iwr"]["folder"] = args.iwr_folder

    debug_base = Path(args.debug_folder)
    variables_config["actual_evapotranspiration"]["folder"] = str(
        debug_base / "actual_evapotranspiration"
    )
    variables_config["deep_percolation"]["folder"] = str(
        debug_base / "deep_percolation"
    )
    variables_config["runoff"]["folder"] = str(
        debug_base / "runoff"
    )
    variables_config["soil_saturation"]["folder"] = str(
        debug_base / "soil_saturation"
    )
    variables_config["precipitation"]["folder"] = args.precipitation_folder
    variables_config["et0"]["folder"] = args.et0_folder

    selected_variables = args.variables
    model_variables, forcing_variables = split_variables_by_group(
        selected_variables=selected_variables,
        variables_config=variables_config,
    )

    inventory = build_raster_inventory(
        selected_variables=selected_variables,
        variables_config=variables_config,
    )

    raster_crs = read_reference_crs(inventory)
    print(f"Using raster target CRS: {raster_crs}")

    points = load_points(
        point_shapefile=args.points,
        target_crs=raster_crs,
        id_field=args.id_field,
    )

    print(f"Loaded {len(points)} point(s)")
    print(f"Selected variables: {', '.join(selected_variables)}")
    print(f"Raster CRS: {raster_crs}")

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

        print(f"Written CSV:  {csv_path}")

        if model_variables:
            model_png_path = plot_dir / f"{safe_id}_timeseries.png"

            plot_timeseries_for_point(
                df=df,
                point_id=point_id,
                selected_variables=model_variables,
                variables_config=variables_config,
                output_png=model_png_path,
                plot_title=f"IWR/debug time series - {point_id}",
            )

            print(f"Written model plot: {model_png_path}")

        if forcing_variables:
            forcing_png_path = plot_dir / f"{safe_id}_forcing_timeseries.png"

            plot_timeseries_for_point(
                df=df,
                point_id=point_id,
                selected_variables=forcing_variables,
                variables_config=variables_config,
                output_png=forcing_png_path,
                plot_title=f"Forcing time series - {point_id}",
            )

            print(f"Written forcing plot: {forcing_png_path}")


if __name__ == "__main__":
    main()