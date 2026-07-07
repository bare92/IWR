import warnings
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

import matplotlib.pyplot as plt

from forcing_alignment import reproject_forcing_to_grid

# Fill-value sentinels treated as nodata regardless of what the GeoTIFF header says.
# (fv, tolerance) pairs.  |-9999 - (-9999.9)| = 0.9, so tol=1.0 covers all variants.
_FILL_SENTINELS = [
    (-9999, 1.0),  # covers -9999, -9999.0, -9999.9
    ( 9999, 1.0),  # covers  9999,  9999.0,  9999.9
]


def debug_imshow(
    arr,
    title="Debug raster",
    nodata=-9999,
    cmap="viridis",
    vmin=None,
    vmax=None,
    save_path=None,
):
    arr = np.asarray(arr).astype("float32")

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, shrink=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved debug plot to: {save_path}")
    else:
        plt.show()

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
    - values below min_value are clipped, but nodata pixels are never touched
    """

    data = np.asarray(data, dtype=np.float32)

    data[~np.isfinite(data)] = nodata

    # Only clip valid pixels; nodata pixels must not be altered.
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
    max_value=None,
    reference_profile=None,
    nodata=-9999.0,
    min_valid_fraction=0.01,
    variable_name="forcing",
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

    Parameters
    ----------
    max_value : float or None
        If set, any valid pixel above this threshold raises a ValueError.
    min_valid_fraction : float
        Fraction of pixels that must be valid (not nodata/nan/inf).
        Raises ValueError when the fraction is below this threshold.
    variable_name : str
        Used in error messages to identify the variable.
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

    # --- Convert declared source nodata ---
    if source_nodata is not None:
        data[data == source_nodata] = nodata

    # --- Mask common undeclared fill values ---
    # Check for extreme float sentinel values first.
    data[data > 1e19] = nodata
    data[data < -1e19] = nodata
    # Then mask near-integer fill sentinels (e.g. -9999, -9999.9, 9999, 9999.9).
    for fv, tol in _FILL_SENTINELS:
        data[np.abs(data - fv) <= tol] = nodata

    # --- Fail-fast: check valid pixel fraction ---
    valid_mask = np.isfinite(data) & (data != nodata)
    valid_fraction = float(np.sum(valid_mask)) / data.size

    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"Variable '{variable_name}' for {date_string_dash} has only "
            f"{100.0 * valid_fraction:.2f}% valid pixels "
            f"(minimum required: {100.0 * min_valid_fraction:.2f}%). "
            f"File: {geotiff_path}"
        )

    # --- Fail-fast: check max_value ---
    if max_value is not None:
        too_large_mask = valid_mask & (data > max_value)
        if np.any(too_large_mask):
            n_bad = int(np.sum(too_large_mask))
            example_vals = list(data[too_large_mask].flat[:5])
            raise ValueError(
                f"Variable '{variable_name}' for {date_string_dash}: "
                f"{n_bad} pixel(s) exceed max_value={max_value}. "
                f"Example values: {[round(float(v), 4) for v in example_vals]}. "
                f"File: {geotiff_path}"
            )

    return clean_forcing_array(
        data=data,
        min_value=min_value,
        nodata=nodata,
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def array_stats(arr, nodata=-9999.0):
    """
    Return basic statistics for *arr*, ignoring nodata / NaN / inf.

    Returns
    -------
    dict with keys:
        valid_count, valid_percent, min, p01, p05, p50, p95, p99, max, mean
    All statistic values are NaN when there are no valid pixels.
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr) & (arr != nodata)]

    _nan = float("nan")

    if valid.size == 0:
        return {
            "valid_count": 0,
            "valid_percent": 0.0,
            "min": _nan, "p01": _nan, "p05": _nan, "p50": _nan,
            "p95": _nan, "p99": _nan, "max": _nan, "mean": _nan,
        }

    return {
        "valid_count": int(valid.size),
        "valid_percent": 100.0 * valid.size / arr.size,
        "min":  float(np.min(valid)),
        "p01":  float(np.percentile(valid,  1)),
        "p05":  float(np.percentile(valid,  5)),
        "p50":  float(np.percentile(valid, 50)),
        "p95":  float(np.percentile(valid, 95)),
        "p99":  float(np.percentile(valid, 99)),
        "max":  float(np.max(valid)),
        "mean": float(np.mean(valid)),
    }


def assert_reasonable_range(
    name,
    arr,
    min_allowed,
    max_allowed,
    nodata=-9999.0,
    date=None,
    raise_error=True,
):
    """
    Raise ValueError (or emit a warning when raise_error=False) if any valid
    pixel in *arr* falls outside [min_allowed, max_allowed].

    nodata / NaN / inf values are ignored.
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid_pixel_mask = np.isfinite(arr) & (arr != nodata)

    if not np.any(valid_pixel_mask):
        return

    bad_mask = valid_pixel_mask & ((arr < min_allowed) | (arr > max_allowed))

    if not np.any(bad_mask):
        return

    bad_rows, bad_cols = np.where(bad_mask)
    first_row = int(bad_rows[0])
    first_col = int(bad_cols[0])
    bad_val   = float(arr[first_row, first_col])
    date_str  = date.strftime("%Y-%m-%d") if date is not None else "unknown"
    valid_min = float(np.min(arr[valid_pixel_mask]))
    valid_max = float(np.max(arr[valid_pixel_mask]))

    msg = (
        f"Variable '{name}' on {date_str} has {int(np.sum(bad_mask))} pixel(s) "
        f"outside [{min_allowed}, {max_allowed}]. "
        f"First bad pixel: row={first_row}, col={first_col}, value={bad_val:.4f}. "
        f"Observed valid range: [{valid_min:.4f}, {valid_max:.4f}]."
    )

    if raise_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg, stacklevel=2)