#!/usr/bin/env python3
"""
Plot ET and PET time series for every point in a shapefile.

Inputs:
  - point shapefile in EPSG:3035 or another CRS
  - FEST ET NetCDF with variable ET_sim
  - FEST PET NetCDF with variable PET_sim

Outputs:
  - one PNG per point
  - one CSV per point
  - one combined CSV for all points
"""

from pathlib import Path
import os
import re
import warnings

# ---------------------------------------------------------------------
# PROJ fix: useful in conda environments with mixed PROJ installations
# ---------------------------------------------------------------------
try:
    import pyproj

    proj_data = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_data
    os.environ["PROJ_DATA"] = proj_data
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

POINTS_PATH = Path("/share/data/DAO/auxiliary/shapefile_checks/point_check_crops.shp")

ET_NC_PATH = Path("/share/data/DAO/input/FEST_storico_2021_2025/ET.nc")
PET_NC_PATH = Path("/share/data/DAO/input/FEST_storico_2021_2025/PET.nc")

ET_VAR_NAME = "ET_sim"
PET_VAR_NAME = "PET_sim"

OUT_DIR = Path("/share/data/DAO/input/point_check_crops_ET_PET_timeseries")

# Leave as None to plot the full available period.
# Example:
# START_DATE = "2021-01-01"
# END_DATE = "2025-12-31"
START_DATE = None
END_DATE = None

# Used only if the NetCDF time coordinate cannot be decoded.
# Your files have 1827 daily steps, so this is a safe fallback.
DATA_START_DATE_IF_TIME_MISSING = "2021-01-01"

# Used only if the shapefile has no CRS.
POINTS_CRS_IF_MISSING = "EPSG:3035"

# If None, the script searches automatically.
# For your uploaded shapefile, "crop" exists and will be used.
POINT_ID_FIELD = None

# If True, CSV files are also written.
SAVE_CSV = True


# =============================================================================
# FUNCTIONS
# =============================================================================

def safe_name(value: str) -> str:
    """Make a safe filename component."""
    value = str(value)
    value = value.strip()
    value = re.sub(r"[^\w\-\.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "point"


def choose_id_field(gdf: gpd.GeoDataFrame) -> str | None:
    """Choose a useful point ID field."""
    if POINT_ID_FIELD is not None:
        if POINT_ID_FIELD not in gdf.columns:
            raise ValueError(
                f"POINT_ID_FIELD='{POINT_ID_FIELD}' not found. "
                f"Available fields: {list(gdf.columns)}"
            )
        return POINT_ID_FIELD

    candidates = [
        "crop",
        "id",
        "ID",
        "Id",
        "name",
        "Name",
        "NAME",
        "label",
        "Label",
        "point_id",
        "POINT_ID",
    ]

    for field in candidates:
        if field in gdf.columns:
            return field

    return None


def find_variable(ds: xr.Dataset, preferred_name: str) -> str:
    """Find the main 3D variable in a NetCDF file."""
    if preferred_name in ds.data_vars:
        return preferred_name

    # Fallback: search by partial name.
    preferred_lower = preferred_name.lower()
    candidates = []
    for name, da in ds.data_vars.items():
        if preferred_lower in name.lower() and da.ndim >= 3:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    # Second fallback: any 3D variable that is not lat/lon/z.
    excluded = {"lat", "latitude", "lon", "longitude", "z"}
    candidates = [
        name
        for name, da in ds.data_vars.items()
        if da.ndim >= 3 and name.lower() not in excluded
    ]

    if len(candidates) == 1:
        warnings.warn(
            f"Variable '{preferred_name}' not found. Using '{candidates[0]}' instead."
        )
        return candidates[0]

    raise ValueError(
        f"Could not identify variable '{preferred_name}'. "
        f"Available data variables: {list(ds.data_vars)}"
    )


def find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Find latitude and longitude variables."""
    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]

    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in ds.variables:
            lat_name = name
            break

    for name in lon_candidates:
        if name in ds.variables:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        raise ValueError(
            "Could not find lat/lon variables in NetCDF. "
            f"Available variables: {list(ds.variables)}"
        )

    return lat_name, lon_name


def find_time_dim(da: xr.DataArray) -> str:
    """Find the time dimension of a DataArray."""
    for dim in da.dims:
        if "time" in dim.lower():
            return dim

    # FEST files shown by gdalinfo are [time, y, x], so fallback to first dimension.
    if da.ndim >= 3:
        return da.dims[0]

    raise ValueError(f"Could not identify time dimension for variable {da.name}.")


def make_time_index(ds: xr.Dataset, da: xr.DataArray, time_dim: str) -> pd.DatetimeIndex:
    """Create a pandas DatetimeIndex from the NetCDF time coordinate."""
    n_time = da.sizes[time_dim]

    if time_dim in ds.variables:
        raw = ds[time_dim].values

        # If numeric survived decoding, avoid interpreting it as ns since 1970.
        if not np.issubdtype(np.asarray(raw).dtype, np.number):
            try:
                return pd.DatetimeIndex(pd.to_datetime(raw))
            except Exception:
                try:
                    return pd.DatetimeIndex([pd.Timestamp(str(v)) for v in raw])
                except Exception:
                    pass

    if DATA_START_DATE_IF_TIME_MISSING is not None:
        warnings.warn(
            f"Could not decode NetCDF time for variable '{da.name}'. "
            f"Using fallback daily dates starting from {DATA_START_DATE_IF_TIME_MISSING}."
        )
        return pd.date_range(DATA_START_DATE_IF_TIME_MISSING, periods=n_time, freq="D")

    raise ValueError(
        f"Could not decode time coordinate for variable '{da.name}'. "
        "Set DATA_START_DATE_IF_TIME_MISSING."
    )


def get_spatial_dims(
    da: xr.DataArray,
    lat_da: xr.DataArray,
    time_dim: str,
) -> tuple[str, str]:
    """
    Determine the two spatial dimensions corresponding to the 2D lat/lon arrays.
    """
    if lat_da.ndim == 2 and all(dim in da.dims for dim in lat_da.dims):
        return tuple(lat_da.dims)

    spatial_dims = [dim for dim in da.dims if dim != time_dim]
    if len(spatial_dims) < 2:
        raise ValueError(f"Could not identify spatial dimensions for {da.name}.")

    return spatial_dims[0], spatial_dims[1]


def prepare_nc(nc_path: Path, variable_name: str) -> dict:
    """Open NetCDF, find variable, time, lat/lon grid, and build nearest-neighbour tree."""
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    print(f"Opening {nc_path}")
    ds = xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True)

    var_name = find_variable(ds, variable_name)
    da = ds[var_name]

    lat_name, lon_name = find_lat_lon_names(ds)
    lat_da = ds[lat_name]
    lon_da = ds[lon_name]

    if lat_da.ndim != 2 or lon_da.ndim != 2:
        raise ValueError(
            f"Expected 2D lat/lon arrays. Got {lat_name}.ndim={lat_da.ndim}, "
            f"{lon_name}.ndim={lon_da.ndim}."
        )

    time_dim = find_time_dim(da)
    time_index = make_time_index(ds, da, time_dim)
    spatial_dims = get_spatial_dims(da, lat_da, time_dim)

    lat = np.asarray(lat_da.values, dtype=float)
    lon = np.asarray(lon_da.values, dtype=float)

    if lat.shape != lon.shape:
        raise ValueError(f"lat/lon shape mismatch: {lat.shape} vs {lon.shape}")

    valid = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        raise ValueError(f"No valid lat/lon values found in {nc_path}")

    flat_valid_indices = np.flatnonzero(valid.ravel())
    lon_flat = lon.ravel()[flat_valid_indices]
    lat_flat = lat.ravel()[flat_valid_indices]

    tree = cKDTree(np.column_stack([lon_flat, lat_flat]))

    print(f"  variable: {var_name}")
    print(f"  dims: {da.dims}")
    print(f"  time dim: {time_dim}, n={len(time_index)}")
    print(f"  spatial dims: {spatial_dims}")
    print(f"  grid shape: {lat.shape}")

    return {
        "path": nc_path,
        "ds": ds,
        "da": da,
        "var_name": var_name,
        "time_dim": time_dim,
        "time_index": time_index,
        "spatial_dims": spatial_dims,
        "lat": lat,
        "lon": lon,
        "tree": tree,
        "flat_valid_indices": flat_valid_indices,
        "lon_min": np.nanmin(lon),
        "lon_max": np.nanmax(lon),
        "lat_min": np.nanmin(lat),
        "lat_max": np.nanmax(lat),
        "units": da.attrs.get("units", ""),
    }


def nearest_cell(bundle: dict, point_lon: float, point_lat: float) -> dict:
    """Find nearest NetCDF grid cell to a lon/lat point."""
    query_lon = float(point_lon)
    query_lat = float(point_lat)

    # Handle grids stored in 0..360 longitude.
    if bundle["lon_min"] >= 0 and query_lon < 0:
        query_lon += 360.0

    dist_deg, pos = bundle["tree"].query([query_lon, query_lat])
    flat_index = bundle["flat_valid_indices"][pos]
    row, col = np.unravel_index(flat_index, bundle["lat"].shape)

    nearest_lon = float(bundle["lon"][row, col])
    nearest_lat = float(bundle["lat"][row, col])

    return {
        "row": int(row),
        "col": int(col),
        "nearest_lon": nearest_lon,
        "nearest_lat": nearest_lat,
        "distance_deg": float(dist_deg),
        "distance_km_approx": float(dist_deg * 111.0),
    }


def extract_series(bundle: dict, time_positions: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract one time series from a NetCDF variable."""
    y_dim, x_dim = bundle["spatial_dims"]

    series = bundle["da"].isel(
        {
            bundle["time_dim"]: time_positions,
            y_dim: row,
            x_dim: col,
        }
    )

    values = np.asarray(series.values, dtype=float).reshape(-1)
    values[~np.isfinite(values)] = np.nan
    return values


def load_points(points_path: Path) -> gpd.GeoDataFrame:
    """Load point shapefile and convert coordinates to lon/lat."""
    if not points_path.exists():
        raise FileNotFoundError(f"Point shapefile not found: {points_path}")

    gdf = gpd.read_file(points_path)

    if gdf.empty:
        raise ValueError(f"No features found in {points_path}")

    if gdf.crs is None:
        if POINTS_CRS_IF_MISSING is None:
            raise ValueError(
                "Point shapefile has no CRS. Set POINTS_CRS_IF_MISSING."
            )
        warnings.warn(
            f"Point shapefile has no CRS. Assuming {POINTS_CRS_IF_MISSING}."
        )
        gdf = gdf.set_crs(POINTS_CRS_IF_MISSING)

    # Make sure geometries are points.
    non_point = ~gdf.geometry.geom_type.isin(["Point"])
    if non_point.any():
        warnings.warn(
            "Some geometries are not points. Using representative points."
        )
        gdf = gdf.copy()
        gdf.loc[non_point, "geometry"] = gdf.loc[non_point, "geometry"].representative_point()

    gdf_ll = gdf.to_crs("EPSG:4326")

    gdf = gdf.copy()
    gdf["point_lon"] = gdf_ll.geometry.x
    gdf["point_lat"] = gdf_ll.geometry.y

    id_field = choose_id_field(gdf)

    if id_field is not None:
        gdf["point_label"] = gdf[id_field].astype(str)
    else:
        gdf["point_label"] = [f"point_{i + 1:03d}" for i in range(len(gdf))]

    # Add index prefix to guarantee unique filenames.
    gdf["point_id"] = [
        f"{i + 1:03d}_{safe_name(label)}"
        for i, label in enumerate(gdf["point_label"])
    ]

    print(f"Loaded {len(gdf)} points from {points_path}")
    print(f"Point CRS: {gdf.crs}")
    if id_field:
        print(f"Using point label field: {id_field}")

    return gdf


def select_common_time_positions(
    et_bundle: dict,
    pet_bundle: dict,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Select common dates between ET and PET, then apply optional date filters."""
    et_time = et_bundle["time_index"]
    pet_time = pet_bundle["time_index"]

    if et_time.equals(pet_time):
        common_time = et_time
        et_pos = np.arange(len(et_time))
        pet_pos = np.arange(len(pet_time))
    else:
        common_time = et_time.intersection(pet_time)
        if len(common_time) == 0:
            raise ValueError("ET and PET have no common timestamps.")

        et_pos = et_time.get_indexer(common_time)
        pet_pos = pet_time.get_indexer(common_time)

        if np.any(et_pos < 0) or np.any(pet_pos < 0):
            raise ValueError("Failed to align ET/PET time coordinates.")

    mask = np.ones(len(common_time), dtype=bool)

    if START_DATE is not None:
        start_ts = pd.Timestamp(START_DATE)
        mask &= common_time >= start_ts

    if END_DATE is not None:
        end_ts = pd.Timestamp(END_DATE)
        mask &= common_time <= end_ts

    selected_time = common_time[mask]
    selected_et_pos = et_pos[mask]
    selected_pet_pos = pet_pos[mask]

    if len(selected_time) == 0:
        raise ValueError(
            "No timesteps selected. Check START_DATE and END_DATE."
        )

    print(
        f"Selected period: {selected_time[0].date()} to {selected_time[-1].date()} "
        f"({len(selected_time)} timesteps)"
    )

    return selected_time, selected_et_pos, selected_pet_pos


def plot_point_timeseries(
    df: pd.DataFrame,
    point_id: str,
    point_label: str,
    et_cell: dict,
    pet_cell: dict,
    out_png: Path,
    units_label: str,
):
    """Create one ET/PET plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["date"], df["ET"], linewidth=1.0, label="ET")
    ax.plot(df["date"], df["PET"], linewidth=1.0, label="PET")

    ax.set_title(
        f"{point_label} | ET/PET time series\n"
        f"ET cell row={et_cell['row']}, col={et_cell['col']}, "
        f"nearest distance ~{et_cell['distance_km_approx']:.2f} km"
    )

    ax.set_ylabel(units_label)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_DIR / "png"
    csv_dir = OUT_DIR / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(POINTS_PATH)

    et_bundle = prepare_nc(ET_NC_PATH, ET_VAR_NAME)
    pet_bundle = prepare_nc(PET_NC_PATH, PET_VAR_NAME)

    selected_time, et_time_pos, pet_time_pos = select_common_time_positions(
        et_bundle,
        pet_bundle,
    )

    units = et_bundle["units"] or pet_bundle["units"]
    units_label = f"ET / PET ({units})" if units else "ET / PET"

    all_records = []

    for _, point in points.iterrows():
        point_id = point["point_id"]
        point_label = point["point_label"]
        point_lon = float(point["point_lon"])
        point_lat = float(point["point_lat"])

        print(f"\nProcessing {point_id}: lon={point_lon:.6f}, lat={point_lat:.6f}")

        et_cell = nearest_cell(et_bundle, point_lon, point_lat)
        pet_cell = nearest_cell(pet_bundle, point_lon, point_lat)

        et_values = extract_series(
            et_bundle,
            et_time_pos,
            et_cell["row"],
            et_cell["col"],
        )
        pet_values = extract_series(
            pet_bundle,
            pet_time_pos,
            pet_cell["row"],
            pet_cell["col"],
        )

        df = pd.DataFrame(
            {
                "date": selected_time,
                "point_id": point_id,
                "point_label": point_label,
                "point_lon": point_lon,
                "point_lat": point_lat,
                "ET": et_values,
                "PET": pet_values,
                "ET_row": et_cell["row"],
                "ET_col": et_cell["col"],
                "ET_nearest_lon": et_cell["nearest_lon"],
                "ET_nearest_lat": et_cell["nearest_lat"],
                "ET_distance_km_approx": et_cell["distance_km_approx"],
                "PET_row": pet_cell["row"],
                "PET_col": pet_cell["col"],
                "PET_nearest_lon": pet_cell["nearest_lon"],
                "PET_nearest_lat": pet_cell["nearest_lat"],
                "PET_distance_km_approx": pet_cell["distance_km_approx"],
            }
        )

        out_png = png_dir / f"{point_id}_ET_PET.png"
        plot_point_timeseries(
            df=df,
            point_id=point_id,
            point_label=point_label,
            et_cell=et_cell,
            pet_cell=pet_cell,
            out_png=out_png,
            units_label=units_label,
        )

        print(f"  saved plot: {out_png}")

        if SAVE_CSV:
            out_csv = csv_dir / f"{point_id}_ET_PET.csv"
            df.to_csv(out_csv, index=False)
            print(f"  saved csv:  {out_csv}")

        all_records.append(df)

    if SAVE_CSV and all_records:
        combined = pd.concat(all_records, ignore_index=True)
        combined_csv = OUT_DIR / "all_points_ET_PET_timeseries.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nSaved combined CSV: {combined_csv}")

    print(f"\nDone. Outputs are in: {OUT_DIR}")


if __name__ == "__main__":
    main()