import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


def get_netcdf_crs(dataset):
    """
    Read CRS from the NetCDF crs variable.

    Expected:
        crs:epsg_code = "EPSG:32633"
    """

    if "crs" not in dataset:
        raise ValueError("No 'crs' variable found in NetCDF dataset.")

    crs_attrs = dataset["crs"].attrs

    if "epsg_code" in crs_attrs:
        return CRS.from_string(crs_attrs["epsg_code"])

    if "crs_wkt" in crs_attrs:
        return CRS.from_wkt(crs_attrs["crs_wkt"])

    raise ValueError("Could not identify CRS from NetCDF crs attributes.")


def get_netcdf_transform_and_data_orientation(dataset, data):
    """
    Create affine transform from 1D x/y coordinates.

    If y is ascending, flip the data vertically because raster rows are expected
    from north to south.
    """

    x = dataset["x"].values
    y = dataset["y"].values

    dx = float(abs(np.median(np.diff(x))))
    dy = float(abs(np.median(np.diff(y))))

    # If y increases from first row to last row, data is south-to-north.
    # Flip it to north-to-south for rasterio.
    if y[0] < y[-1]:
        data = np.flipud(data)
        y_top = float(y[-1] + dy / 2.0)
    else:
        y_top = float(y[0] + dy / 2.0)

    x_left = float(x[0] - dx / 2.0)

    transform = from_origin(
        west=x_left,
        north=y_top,
        xsize=dx,
        ysize=dy,
    )

    return transform, data


def reproject_forcing_to_grid(
    data,
    source_dataset,
    target_profile,
    resampling_method="bilinear",
    source_nodata=-9999.9,
    target_nodata=-9999.0,
):
    """
    Reproject one forcing layer to the static model grid.

    Parameters
    ----------
    data : numpy.ndarray
        Daily forcing layer, 2D.

    source_dataset : xarray.Dataset
        Open NetCDF dataset containing x, y and crs.

    target_profile : dict
        Rasterio profile from the target grid.

    resampling_method : str
        "bilinear" or "nearest".

    Returns
    -------
    aligned_data : numpy.ndarray
        Daily forcing aligned to the target grid.
    """

    source_crs = get_netcdf_crs(source_dataset)

    source_transform, data = get_netcdf_transform_and_data_orientation(
        dataset=source_dataset,
        data=data,
    )

    target_crs = target_profile["crs"]
    target_transform = target_profile["transform"]
    target_height = target_profile["height"]
    target_width = target_profile["width"]

    aligned_data = np.full(
        (target_height, target_width),
        target_nodata,
        dtype=np.float32,
    )

    if resampling_method == "nearest":
        resampling = Resampling.nearest
    elif resampling_method == "bilinear":
        resampling = Resampling.bilinear
    else:
        raise ValueError(f"Unsupported resampling method: {resampling_method}")

    reproject(
        source=data.astype(np.float32),
        destination=aligned_data,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=target_nodata,
        resampling=resampling,
    )

    aligned_data[~np.isfinite(aligned_data)] = target_nodata

    return aligned_data
