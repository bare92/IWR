import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio



SOIL_PARAMETERS = {
    1:  {"name": "Clay heavy",       "wp": 0.39, "fc": 0.54, "fmax": 5},
    2:  {"name": "Silty clay",       "wp": 0.32, "fc": 0.50, "fmax": 12},
    3:  {"name": "Clay",             "wp": 0.39, "fc": 0.54, "fmax": 7},
    4:  {"name": "Silty clay loam",  "wp": 0.23, "fc": 0.44, "fmax": 24},
    5:  {"name": "Clay loam",        "wp": 0.23, "fc": 0.39, "fmax": 24},
    6:  {"name": "Silt",             "wp": 0.09, "fc": 0.33, "fmax": 60},
    7:  {"name": "Silt loam",        "wp": 0.13, "fc": 0.33, "fmax": 82},
    8:  {"name": "Sandy clay",       "wp": 0.27, "fc": 0.39, "fmax": 14},
    9:  {"name": "Loam",             "wp": 0.15, "fc": 0.31, "fmax": 158},
    10: {"name": "Sandy clay loam",  "wp": 0.20, "fc": 0.32, "fmax": 36},
    11: {"name": "Sandy loam",       "wp": 0.10, "fc": 0.22, "fmax": 262},
    12: {"name": "Loamy sand",       "wp": 0.08, "fc": 0.16, "fmax": 500},
    13: {"name": "Sand",             "wp": 0.06, "fc": 0.13, "fmax": 720},
}

# SOIL_PARAMETERS = {
#     1:  {"name": "Clay heavy",       "wp": 0.39, "fc": 0.54, "fmax": 35},
#     2:  {"name": "Silty clay",       "wp": 0.32, "fc": 0.50, "fmax": 100},
#     3:  {"name": "Clay",             "wp": 0.39, "fc": 0.54, "fmax": 35},
#     4:  {"name": "Silty clay loam",  "wp": 0.23, "fc": 0.44, "fmax": 150},
#     5:  {"name": "Clay loam",        "wp": 0.23, "fc": 0.39, "fmax": 125},
#     6:  {"name": "Silt",             "wp": 0.09, "fc": 0.33, "fmax": 500},
#     7:  {"name": "Silt loam",        "wp": 0.13, "fc": 0.33, "fmax": 575},
#     8:  {"name": "Sandy clay",       "wp": 0.27, "fc": 0.39, "fmax": 35},
#     9:  {"name": "Loam",             "wp": 0.15, "fc": 0.31, "fmax": 500},
#     10: {"name": "Sandy clay loam",  "wp": 0.20, "fc": 0.32, "fmax": 225},
#     11: {"name": "Sandy loam",       "wp": 0.10, "fc": 0.22, "fmax": 1200},
#     12: {"name": "Loamy sand",       "wp": 0.08, "fc": 0.16, "fmax": 2200},
#     13: {"name": "Sand",             "wp": 0.06, "fc": 0.13, "fmax": 3000},
# }

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


def validate_soil_texture_codes(soil_texture, nodata=-9999.0):
    """
    Inspect unique soil texture codes and validate them against SOIL_PARAMETERS.

    Prints code, soil-class name and pixel count for each class found.
    Raises ValueError if any code is not present in SOIL_PARAMETERS.

    Parameters
    ----------
    soil_texture : numpy.ndarray
        Soil texture raster values (integer USDA texture classes).
    nodata : float
        Raster nodata value; excluded from inspection.

    Notes
    -----
    This check validates encoding values only.  It cannot prove that the
    source dataset's class legend semantically matches the lookup table used
    here.  Manual verification against the original dataset documentation is
    always recommended.
    """
    unique_codes = np.unique(soil_texture)

    # Exclude nodata with a dtype-aware comparison so int rasters are handled
    # correctly (e.g. int16 -9999 is not equal to float64 -9999.0 after cast).
    nodata_as_dtype = soil_texture.dtype.type(nodata)
    valid_codes = unique_codes[unique_codes != nodata_as_dtype]

    unknown = [int(c) for c in valid_codes if int(c) not in SOIL_PARAMETERS]
    if unknown:
        raise ValueError(
            f"Soil texture raster contains unknown class code(s): {unknown}. "
            f"Expected codes: {sorted(SOIL_PARAMETERS.keys())}."
        )

    print("Soil texture class validation:")
    for code in sorted(valid_codes):
        icode = int(code)
        name = SOIL_PARAMETERS[icode]["name"]
        count = int(np.sum(soil_texture == code))
        print(f"  Code {icode:2d}  {name:<22s}  {count:8d} pixels")

    print(
        "  [Note] This check validates encoding values only.  It cannot prove "
        "that the source dataset's class legend semantically matches the "
        "lookup table used here."
    )


def _write_soil_parameters_metadata(output_folder, soil_texture_path, nodata):
    """Write soil_parameters_metadata.json to *output_folder*."""
    soil_texture_path = Path(soil_texture_path)
    output_folder = Path(output_folder)

    try:
        src_mtime = datetime.fromtimestamp(
            soil_texture_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except OSError:
        src_mtime = None

    metadata = {
        "software_note": (
            "Cached derived products generated by IWR soil_functions.py. "
            "Do not edit manually. To regenerate, set "
            "options.force_regenerate_soil_parameters=true in the config."
        ),
        "source_soil_texture_path": str(soil_texture_path),
        "source_file_modification_time": src_mtime,
        "generation_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "nodata_value": nodata,
        "soil_parameters_lookup": {str(k): v for k, v in SOIL_PARAMETERS.items()},
    }

    metadata_path = output_folder / "soil_parameters_metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata_path


def check_cached_soil_parameters_metadata(output_folder):
    """
    Load soil_parameters_metadata.json and compare the saved lookup table
    with the current SOIL_PARAMETERS definition.

    Prints a strong warning (and a recommendation to regenerate) when the
    two lookup tables differ.

    Parameters
    ----------
    output_folder : str or Path
        Folder that is expected to contain soil_parameters_metadata.json.

    Returns
    -------
    dict or None
        The metadata dict, or None when the file does not exist.
    """
    metadata_path = Path(output_folder) / "soil_parameters_metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)

    saved_lookup = metadata.get("soil_parameters_lookup", {})
    current_lookup = {str(k): v for k, v in SOIL_PARAMETERS.items()}

    if saved_lookup != current_lookup:
        print(
            "WARNING: The cached soil parameter rasters were generated with a "
            "different SOIL_PARAMETERS lookup table than the one currently "
            "defined in soil_functions.py."
        )
        changed_keys = sorted(
            set(saved_lookup) | set(current_lookup),
            key=lambda x: int(x),
        )
        for k in changed_keys:
            saved_entry = saved_lookup.get(k, "<missing>")
            current_entry = current_lookup.get(k, "<missing>")
            if saved_entry != current_entry:
                print(f"  Code {k:>2s}: saved={saved_entry}  current={current_entry}")
        print(
            "  Recommendation: set options.force_regenerate_soil_parameters=true "
            "in the config to regenerate the rasters with the current lookup table."
        )
    else:
        print(
            "Soil parameter cache: lookup table matches current SOIL_PARAMETERS. OK."
        )

    return metadata


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
    soil_parameters_metadata.json
    """

    soil_texture_path = Path(soil_texture_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with rasterio.open(soil_texture_path) as src:
        soil_texture = src.read(1)
        profile = src.profile.copy()
        raster_nodata = src.nodata

    # Use the raster's own nodata when available for the validation step so
    # that integer nodata values (e.g. 255, -9999) are handled correctly.
    validation_nodata = raster_nodata if raster_nodata is not None else nodata
    validate_soil_texture_codes(soil_texture, nodata=validation_nodata)

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

    _write_soil_parameters_metadata(
        output_folder=output_folder,
        soil_texture_path=soil_texture_path,
        nodata=nodata,
    )

    return output_paths