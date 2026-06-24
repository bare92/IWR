from pathlib import Path

import numpy as np
import xarray as xr


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


def clean_forcing_array(data, min_value=0.0):
    """
    Clean one forcing layer.

    - NaN and infinite values are set to 0
    - values below min_value are set to min_value
    """

    data = np.asarray(data, dtype=np.float32)

    data[~np.isfinite(data)] = 0.0

    if min_value is not None:
        data[data < min_value] = min_value

    return data.astype(np.float32)


def read_forcing_day(dataset, variable_name, date, min_value=0.0):
    """
    Read one daily forcing layer from an open dataset.

    Returns a cleaned NumPy array for one day only.
    """

    data = dataset[variable_name].sel(time=date)

    return clean_forcing_array(
        data=data.values,
        min_value=min_value,
    )