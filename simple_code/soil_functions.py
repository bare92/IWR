from pathlib import Path

import numpy as np
import rasterio



# SOIL_PARAMETERS = {
#     1:  {"name": "Clay heavy",       "wp": 0.39, "fc": 0.54, "fmax": 5},
#     2:  {"name": "Silty clay",       "wp": 0.32, "fc": 0.50, "fmax": 12},
#     3:  {"name": "Clay",             "wp": 0.39, "fc": 0.54, "fmax": 7},
#     4:  {"name": "Silty clay loam",  "wp": 0.23, "fc": 0.44, "fmax": 24},
#     5:  {"name": "Clay loam",        "wp": 0.23, "fc": 0.39, "fmax": 24},
#     6:  {"name": "Silt",             "wp": 0.09, "fc": 0.33, "fmax": 60},
#     7:  {"name": "Silt loam",        "wp": 0.13, "fc": 0.33, "fmax": 82},
#     8:  {"name": "Sandy clay",       "wp": 0.27, "fc": 0.39, "fmax": 14},
#     9:  {"name": "Loam",             "wp": 0.15, "fc": 0.31, "fmax": 158},
#     10: {"name": "Sandy clay loam",  "wp": 0.20, "fc": 0.32, "fmax": 36},
#     11: {"name": "Sandy loam",       "wp": 0.10, "fc": 0.22, "fmax": 262},
#     12: {"name": "Loamy sand",       "wp": 0.08, "fc": 0.16, "fmax": 500},
#     13: {"name": "Sand",             "wp": 0.06, "fc": 0.13, "fmax": 720},
# }

SOIL_PARAMETERS = {
    1:  {"name": "Clay heavy",       "wp": 0.39, "fc": 0.54, "fmax": 35},
    2:  {"name": "Silty clay",       "wp": 0.32, "fc": 0.50, "fmax": 100},
    3:  {"name": "Clay",             "wp": 0.39, "fc": 0.54, "fmax": 35},
    4:  {"name": "Silty clay loam",  "wp": 0.23, "fc": 0.44, "fmax": 150},
    5:  {"name": "Clay loam",        "wp": 0.23, "fc": 0.39, "fmax": 125},
    6:  {"name": "Silt",             "wp": 0.09, "fc": 0.33, "fmax": 500},
    7:  {"name": "Silt loam",        "wp": 0.13, "fc": 0.33, "fmax": 575},
    8:  {"name": "Sandy clay",       "wp": 0.27, "fc": 0.39, "fmax": 35},
    9:  {"name": "Loam",             "wp": 0.15, "fc": 0.31, "fmax": 500},
    10: {"name": "Sandy clay loam",  "wp": 0.20, "fc": 0.32, "fmax": 225},
    11: {"name": "Sandy loam",       "wp": 0.10, "fc": 0.22, "fmax": 1200},
    12: {"name": "Loamy sand",       "wp": 0.08, "fc": 0.16, "fmax": 2200},
    13: {"name": "Sand",             "wp": 0.06, "fc": 0.13, "fmax": 3000},
}

def soil_texture_to_arrays(soil_texture, nodata=-9999.0):
    """
    Convert a soil texture array into soil hydraulic parameter arrays.

    Parameters
    ----------
    soil_texture : numpy.ndarray
        Input soil texture raster values.
        Expected values are integer USDA texture classes from 1 to 13.

    nodata : float
        Output nodata value.

    Returns
    -------
    field_capacity : numpy.ndarray
        Field capacity layer, in m3/m3.

    wilting_point : numpy.ndarray
        Wilting point layer, in m3/m3.

    total_available_water : numpy.ndarray
        Total available water layer, in m3/m3.

    fmax : numpy.ndarray
        Maximum infiltration rate layer, in mm/day.
    """

    field_capacity = np.full(soil_texture.shape, nodata, dtype=np.float32)
    wilting_point = np.full(soil_texture.shape, nodata, dtype=np.float32)
    fmax = np.full(soil_texture.shape, nodata, dtype=np.float32)

    for soil_code, params in SOIL_PARAMETERS.items():
        mask = soil_texture == soil_code

        field_capacity[mask] = params["fc"]
        wilting_point[mask] = params["wp"]
        fmax[mask] = params["fmax"]

    total_available_water = field_capacity - wilting_point

    nodata_mask = field_capacity == nodata
    total_available_water[nodata_mask] = nodata

    return field_capacity, wilting_point, total_available_water, fmax


def create_soil_parameter_rasters(
    soil_texture_path,
    output_folder,
    nodata=-9999.0,
):
    """
    Create soil parameter rasters from a soil texture raster.

    Outputs
    -------
    field_capacity.tif
    wilting_point.tif
    total_available_water.tif
    fmax.tif
    """

    soil_texture_path = Path(soil_texture_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with rasterio.open(soil_texture_path) as src:
        soil_texture = src.read(1)
        profile = src.profile.copy()

    field_capacity, wilting_point, total_available_water, fmax = soil_texture_to_arrays(
        soil_texture,
        nodata=nodata,
    )

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw",
    )

    output_paths = {
        "field_capacity": output_folder / "field_capacity.tif",
        "wilting_point": output_folder / "wilting_point.tif",
        "total_available_water": output_folder / "total_available_water.tif",
        "fmax": output_folder / "fmax.tif",
    }

    with rasterio.open(output_paths["field_capacity"], "w", **profile) as dst:
        dst.write(field_capacity, 1)

    with rasterio.open(output_paths["wilting_point"], "w", **profile) as dst:
        dst.write(wilting_point, 1)

    with rasterio.open(output_paths["total_available_water"], "w", **profile) as dst:
        dst.write(total_available_water, 1)

    with rasterio.open(output_paths["fmax"], "w", **profile) as dst:
        dst.write(fmax, 1)

    return output_paths