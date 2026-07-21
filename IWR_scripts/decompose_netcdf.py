#!/usr/bin/env python3
"""
Convert NetCDF layers to GeoTIFF aligned to a reference raster.

Fixes included:
- masks huge NetCDF fill values such as 3.4028235e38
- writes clean GeoTIFF nodata = -9999
- keeps CRS before reprojection
- aligns output to reference raster
"""

from pathlib import Path
import re
import os

# Try to reduce PROJ conflicts when conda and venv are both active.
# Best solution remains: use only one environment.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)
os.environ.pop("GDAL_DATA", None)

from pyproj.datadir import get_data_dir

os.environ["PROJ_LIB"] = get_data_dir()
os.environ["PROJ_DATA"] = get_data_dir()

import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling


# ============================================================
# USER SETTINGS
# ============================================================

NETCDF_PATHS = [
    "/share/data/DAO/input/FEST_storico_2021_2025/P.nc",
    "/share/data/DAO/input/FEST_storico_2021_2025/PET.nc",
]

REFERENCE_RASTER = "/share/data/DAO/static/processed/working_grid_3035_1km.tif"

OUTPUT_DIR = "/share/data/DAO/input/output_geotiffs"

RESAMPLING_METHOD = "bilinear"

# Set to None to export all spatial variables.
# Use this once if you need to inspect variable names:
# VARIABLES_TO_EXPORT = None
VARIABLES_TO_EXPORT = ["P_sim", "PET_sim"]

X_DIM_CANDIDATES = ["x", "lon", "longitude"]
Y_DIM_CANDIDATES = ["y", "lat", "latitude"]

DEFAULT_INPUT_CRS = "EPSG:32633"
REFERENCE_CRS_OVERRIDE = "EPSG:3035"

OUTPUT_NODATA = -9999.0
OUTPUT_COMPRESS = "deflate"

# Any value with absolute magnitude above this is treated as invalid/fill.
# This catches 3.4028235e38 and similar NetCDF fill values.
HUGE_VALUE_THRESHOLD = 1.0e20

# Optional physical sanity limits.
# Set to None if you do not want physical clipping.
# For daily precipitation/PET these are usually safe, but adjust if needed.
PHYSICAL_MIN = 0.0
PHYSICAL_MAX = 10000.0

# ============================================================
# END USER SETTINGS
# ============================================================


RESAMPLING_LOOKUP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
}


def clean_name(value) -> str:
    """Create safe filenames."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text)
    text = text.strip("_")
    return text


def find_spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    """Detect x/y spatial dimensions."""
    x_dim = None
    y_dim = None

    for candidate in X_DIM_CANDIDATES:
        if candidate in ds.dims or candidate in ds.coords:
            x_dim = candidate
            break

    for candidate in Y_DIM_CANDIDATES:
        if candidate in ds.dims or candidate in ds.coords:
            y_dim = candidate
            break

    if x_dim is None or y_dim is None:
        raise ValueError(
            "Could not detect spatial dimensions.\n"
            f"Available dims: {list(ds.dims)}\n"
            f"Available coords: {list(ds.coords)}\n"
            "Update X_DIM_CANDIDATES and Y_DIM_CANDIDATES."
        )

    return x_dim, y_dim


def is_spatial_variable(da: xr.DataArray, x_dim: str, y_dim: str) -> bool:
    """Check if a variable has spatial dimensions."""
    return x_dim in da.dims and y_dim in da.dims


def format_layer_suffix(layer_value) -> str:
    """Create readable filename suffix from time or other dimension values."""
    if isinstance(layer_value, tuple):
        return "_".join(clean_name(v) for v in layer_value)

    return clean_name(layer_value)


def remove_encoding_conflicts(da: xr.DataArray) -> xr.DataArray:
    """
    Remove metadata keys that commonly create conflicts when writing GeoTIFFs.
    Do this only after masking original fill values.
    """
    da = da.copy()

    conflict_keys = [
        "_FillValue",
        "missing_value",
    ]

    for key in conflict_keys:
        da.attrs.pop(key, None)
        da.encoding.pop(key, None)

    return da


def mask_invalid_values(da: xr.DataArray) -> xr.DataArray:
    """
    Convert NetCDF fill values and absurd values to NaN before reprojection.

    This is the key fix for QGIS showing values like 3.4028235e38.
    """

    da = da.astype("float32")

    # Collect fill values from attrs and encoding before removing them
    fill_values = []

    for key in ["_FillValue", "missing_value"]:
        if key in da.attrs:
            fill_values.append(da.attrs[key])
        if key in da.encoding:
            fill_values.append(da.encoding[key])

    # Mask explicit fill values
    for fill_value in fill_values:
        try:
            fv = float(fill_value)
            da = da.where(da != fv)
        except Exception:
            pass

    # Mask huge values, e.g. float32 max nodata
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < HUGE_VALUE_THRESHOLD)

    # Optional physical sanity filtering
    if PHYSICAL_MIN is not None:
        da = da.where(da >= PHYSICAL_MIN)

    if PHYSICAL_MAX is not None:
        da = da.where(da <= PHYSICAL_MAX)

    da = remove_encoding_conflicts(da)

    return da


def write_clean_geotiff(layer_da: xr.DataArray, out_path: Path):
    """
    Write one 2D layer as GeoTIFF with clean nodata handling.
    """

    layer_da = layer_da.squeeze(drop=True)
    layer_da = mask_invalid_values(layer_da)

    # Convert NaN to explicit nodata
    layer_da = layer_da.fillna(OUTPUT_NODATA).astype("float32")

    # Remove encoding conflicts again after fillna/astype
    layer_da = remove_encoding_conflicts(layer_da)

    # Write nodata metadata
    layer_da = layer_da.rio.write_nodata(OUTPUT_NODATA, inplace=False)

    layer_da.rio.to_raster(
        out_path,
        compress=OUTPUT_COMPRESS,
        dtype="float32",
    )

    print(f"Saved: {out_path}")


def export_dataarray_layers(
    da: xr.DataArray,
    variable_name: str,
    nc_stem: str,
    output_dir: Path,
    reference_da: xr.DataArray,
    resampling_method: str,
    input_crs: str,
):
    """
    Export a DataArray to one or more GeoTIFFs aligned to the reference raster.
    """

    if resampling_method not in RESAMPLING_LOOKUP:
        raise ValueError(
            f"Unknown resampling method: {resampling_method}. "
            f"Choose from: {list(RESAMPLING_LOOKUP)}"
        )

    resampling = RESAMPLING_LOOKUP[resampling_method]

    # Important: mask invalid values BEFORE reprojection,
    # otherwise bilinear interpolation can spread huge fill values.
    da = mask_invalid_values(da)

    # Force CRS directly on the data variable
    da = da.rio.write_crs(input_crs, inplace=False)

    print(f"DataArray CRS before reprojection: {da.rio.crs}")

    da_aligned = da.rio.reproject_match(
        reference_da,
        resampling=resampling,
        nodata=np.nan,
    )

    da_aligned = mask_invalid_values(da_aligned)
    da_aligned = da_aligned.rio.write_nodata(OUTPUT_NODATA, inplace=False)

    non_spatial_dims = [
        dim
        for dim in da_aligned.dims
        if dim not in [da_aligned.rio.x_dim, da_aligned.rio.y_dim]
    ]

    # Case 1: simple 2D raster
    if not non_spatial_dims:
        out_name = f"{nc_stem}_{variable_name}.tif"
        out_path = output_dir / out_name
        write_clean_geotiff(da_aligned, out_path)
        return

    # Case 2: raster with time or other extra dimensions
    stacked = da_aligned.stack(layer=non_spatial_dims)

    for layer_value in stacked.layer.values:
        layer_da = stacked.sel(layer=layer_value).squeeze(drop=True)

        suffix = format_layer_suffix(layer_value)

        out_name = f"{nc_stem}_{variable_name}_{suffix}.tif"
        out_path = output_dir / out_name

        write_clean_geotiff(layer_da, out_path)


def process_netcdf(
    nc_path: str,
    reference_da: xr.DataArray,
    output_root: str,
    resampling_method: str,
    variables_to_export=None,
):
    """Process one NetCDF file."""

    nc_path = Path(nc_path)
    output_root = Path(output_root)

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    output_dir = output_root / nc_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Processing: {nc_path}")
    print(f"Output folder: {output_dir}")

    # decode_cf=True is default, but explicit is clearer.
    # mask_and_scale=True helps with NetCDF scale/fill values.
    ds = xr.open_dataset(
        nc_path,
        decode_cf=True,
        mask_and_scale=True,
    )

    print(f"Dataset dims: {dict(ds.sizes)}")
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset coords: {list(ds.coords)}")

    x_dim, y_dim = find_spatial_dims(ds)

    ds = ds.rio.set_spatial_dims(
        x_dim=x_dim,
        y_dim=y_dim,
        inplace=False,
    )

    if ds.rio.crs is None:
        if DEFAULT_INPUT_CRS is None:
            raise ValueError(
                f"No CRS found in {nc_path}. "
                "Set DEFAULT_INPUT_CRS, for example 'EPSG:3035'."
            )

        print(f"No CRS found in NetCDF. Assigning {DEFAULT_INPUT_CRS}")
        ds = ds.rio.write_crs(DEFAULT_INPUT_CRS, inplace=False)
        input_crs = DEFAULT_INPUT_CRS
    else:
        print(f"NetCDF CRS: {ds.rio.crs}")
        input_crs = str(ds.rio.crs)

    for var_name in ds.data_vars:
        if variables_to_export is not None and var_name not in variables_to_export:
            print(f"Skipping variable not requested: {var_name}")
            continue

        da = ds[var_name]

        if not is_spatial_variable(da, x_dim, y_dim):
            print(f"Skipping non-spatial variable: {var_name}")
            continue

        da = da.rio.set_spatial_dims(
            x_dim=x_dim,
            y_dim=y_dim,
            inplace=False,
        )

        da = da.rio.write_crs(input_crs, inplace=False)

        print(f"Exporting variable: {var_name}")
        print(f"Variable dims: {da.dims}")
        print(f"Variable shape: {da.shape}")
        print(f"Variable CRS: {da.rio.crs}")

        # Quick diagnostic on first slice
        try:
            sample = da.isel({dim: 0 for dim in da.dims if dim not in [x_dim, y_dim]})
            sample = mask_invalid_values(sample)
            print(
                f"Sample valid min/max after masking: "
                f"{float(sample.min(skipna=True).values)} / "
                f"{float(sample.max(skipna=True).values)}"
            )
        except Exception as exc:
            print(f"Could not compute sample min/max: {exc}")

        export_dataarray_layers(
            da=da,
            variable_name=clean_name(var_name),
            nc_stem=clean_name(nc_path.stem),
            output_dir=output_dir,
            reference_da=reference_da,
            resampling_method=resampling_method,
            input_crs=input_crs,
        )

    ds.close()


def main():
    reference_path = Path(REFERENCE_RASTER)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {REFERENCE_RASTER}")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_da = rioxarray.open_rasterio(REFERENCE_RASTER)

    if REFERENCE_CRS_OVERRIDE is not None:
        reference_da = reference_da.rio.write_crs(
            REFERENCE_CRS_OVERRIDE,
            inplace=False,
        )

    print(f"Reference raster: {REFERENCE_RASTER}")
    print(f"Reference CRS: {reference_da.rio.crs}")
    print(f"Reference shape: {reference_da.rio.height} rows x {reference_da.rio.width} cols")
    print(f"Reference resolution: {reference_da.rio.resolution()}")

    for nc_path in NETCDF_PATHS:
        process_netcdf(
            nc_path=nc_path,
            reference_da=reference_da,
            output_root=OUTPUT_DIR,
            resampling_method=RESAMPLING_METHOD,
            variables_to_export=VARIABLES_TO_EXPORT,
        )

    reference_da.close()

    print("\nDone.")


if __name__ == "__main__":
    main()