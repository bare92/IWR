from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from forcing_alignment import reproject_forcing_to_grid


def open_forcing_dataset(nc_path, variable_name, chunks=None):
    """
    Open a NetCDF forcing dataset without loading everything in memory.
    """

    nc_path = Path(nc_path)

    if chunks is None:
        chunks = {
            "time": 1
        }

    dataset = xr.open_dataset(
        nc_path,
        chunks=chunks,
        decode_times=True
    )

    if variable_name not in dataset:
        raise ValueError(
            f"Variable '{variable_name}' not found in {nc_path}. "
            f"Available variables: {list(dataset.data_vars)}"
        )

    return dataset


def clean_forcing_array(data, min_value=0.0, nodata=-9999.0):
    """
    Clean one forcing layer.

    - NaN and infinite values are set to nodata
    - values below min_value are set to min_value
    """

    data = np.asarray(data, dtype=np.float32)

    data[~np.isfinite(data)] = nodata

    valid_mask = data != nodata

    if min_value is not None:
        data[valid_mask & (data < min_value)] = min_value

    return data.astype(np.float32)


def read_forcing_day(
    dataset,
    variable_name,
    date,
    min_value=0.0,
    target_profile=None,
    align_to_grid=False,
    resampling_method="bilinear",
    source_nodata=-9999.9,
    target_nodata=-9999.0,
):
    """
    Read one daily forcing layer from an open dataset.

    If align_to_grid=True, reproject the daily layer to target_profile.
    """

    data = dataset[variable_name].sel(time=date).values.astype(np.float32)

    if align_to_grid:
        if target_profile is None:
            raise ValueError("target_profile is required when align_to_grid=True.")

        data = reproject_forcing_to_grid(
            data=data,
            source_dataset=dataset,
            target_profile=target_profile,
            resampling_method=resampling_method,
            source_nodata=source_nodata,
            target_nodata=target_nodata,
        )

    return clean_forcing_array(
        data=data,
        min_value=min_value,
        nodata=target_nodata,
    )


def read_forcing_geotiff_day(
    geotiff_folder,
    date,
    min_value=0.0,
    reference_profile=None,
    nodata=-9999.0,
):
    """
    Read one daily forcing GeoTIFF from a folder.

    The function searches for files containing either:
        YYYY-MM-DD
    or:
        YYYYMMDD

    Example matching filenames:
        P_P_sim_2021-01-01_00_00_00.tif
        PET_PET_sim_2021-01-01_00_00_00.tif
        P_20210101.tif

    Because these GeoTIFFs are assumed to be already aligned to the
    modelling grid, no reprojection or resampling is done here.
    """

    geotiff_folder = Path(geotiff_folder)

    date_string_dash = date.strftime("%Y-%m-%d")
    date_string_compact = date.strftime("%Y%m%d")

    candidate_files = sorted(
        list(geotiff_folder.glob(f"*{date_string_dash}*.tif"))
        + list(geotiff_folder.glob(f"*{date_string_compact}*.tif"))
    )

    if not candidate_files:
        raise FileNotFoundError(
            f"No forcing GeoTIFF found in {geotiff_folder} for date {date_string_dash}"
        )

    if len(candidate_files) > 1:
        print(
            f"Warning: multiple GeoTIFFs found for {date_string_dash}. "
            f"Using: {candidate_files[0]}"
        )

    geotiff_path = candidate_files[0]

    with rasterio.open(geotiff_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        source_nodata = src.nodata

    if reference_profile is not None:
        expected_shape = (
            reference_profile["height"],
            reference_profile["width"],
        )

        if data.shape != expected_shape:
            raise ValueError(
                f"Forcing GeoTIFF shape mismatch for {geotiff_path}\n"
                f"Expected shape: {expected_shape}\n"
                f"Found shape:    {data.shape}\n"
                "This means the forcing GeoTIFF is not aligned to the model grid."
            )

        if profile["transform"] != reference_profile["transform"]:
            raise ValueError(
                f"Forcing GeoTIFF transform mismatch for {geotiff_path}\n"
                "This means the forcing GeoTIFF is not aligned to the model grid."
            )

        if profile["crs"] != reference_profile["crs"]:
            raise ValueError(
                f"Forcing GeoTIFF CRS mismatch for {geotiff_path}\n"
                f"Expected CRS: {reference_profile['crs']}\n"
                f"Found CRS:    {profile['crs']}"
            )

    if source_nodata is not None:
        data[data == source_nodata] = nodata

    return clean_forcing_array(
        data=data,
        min_value=min_value,
        nodata=nodata,
    )