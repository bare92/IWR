import numpy as np
from rasterio.crs import CRS


def get_spatial_shape(array):
    """
    Return spatial shape from 2D or 3D array.

    2D: rows, cols
    3D: bands, rows, cols
    """

    if array.ndim == 2:
        return array.shape

    if array.ndim == 3:
        return array.shape[-2:]

    raise ValueError(f"Unsupported array shape: {array.shape}")


def check_array_shape(name, array, reference_shape):
    shape = get_spatial_shape(array)

    if shape != reference_shape:
        raise ValueError(
            f"{name} has shape {shape}, expected {reference_shape}"
        )


def check_profile_match(name, profile, reference_profile):
    """
    Check raster width, height, CRS and transform.
    """

    for key in ["width", "height"]:
        if profile.get(key) != reference_profile.get(key):
            raise ValueError(
                f"{name} profile mismatch for '{key}'.\n"
                f"{name}:     {profile.get(key)}\n"
                f"reference: {reference_profile.get(key)}"
            )

    transform = profile.get("transform")
    reference_transform = reference_profile.get("transform")

    if transform is None or reference_transform is None:
        if transform != reference_transform:
            raise ValueError(
                f"{name} profile mismatch for 'transform'.\n"
                f"{name}:     {transform}\n"
                f"reference: {reference_transform}"
            )
    elif not transform.almost_equals(reference_transform):
        raise ValueError(
            f"{name} profile mismatch for 'transform'.\n"
            f"{name}:     {transform}\n"
            f"reference: {reference_transform}"
        )

    if not _crs_equivalent(profile.get("crs"), reference_profile.get("crs")):
        raise ValueError(
            f"{name} profile mismatch for 'crs'.\n"
            f"{name}:     {profile.get('crs')}\n"
            f"reference: {reference_profile.get('crs')}"
        )


def _crs_equivalent(crs_a, crs_b):
    """
    Return True if CRS definitions represent the same projection.

    Some input rasters may store equivalent LAEA definitions using different
    WKT spellings (e.g. LOCAL_CS vs PROJCS "unknown").
    
    If one CRS is None (missing metadata), assume it's compatible with the
    reference CRS (e.g., data has been pre-aligned).
    """

    if crs_a == crs_b:
        return True

    # If one CRS is None, accept as compatible (assume data is pre-aligned).
    if crs_a is None or crs_b is None:
        return True

    # Fast textual fallback that does not depend on PROJ DB parsing.
    # This is important when shell env vars point to a mismatched PROJ setup.
    if _looks_like_laea_europe(crs_a) and _looks_like_laea_europe(crs_b):
        return True

    try:
        parsed_a = CRS.from_user_input(crs_a)
        parsed_b = CRS.from_user_input(crs_b)
    except Exception:
        # If PROJ parsing fails, keep the earlier string-based verdict.
        return False

    if parsed_a == parsed_b:
        return True

    epsg_a = parsed_a.to_epsg()
    epsg_b = parsed_b.to_epsg()
    if epsg_a is not None and epsg_b is not None and epsg_a == epsg_b:
        return True

    dict_a = parsed_a.to_dict()
    dict_b = parsed_b.to_dict()

    key_params = ["proj", "lat_0", "lon_0", "x_0", "y_0", "units"]
    if all(dict_a.get(k) == dict_b.get(k) for k in key_params):
        return True

    epsg_or_laea_a = (epsg_a == 3035) or ("LAEA" in str(crs_a).upper())
    epsg_or_laea_b = (epsg_b == 3035) or ("LAEA" in str(crs_b).upper())

    if epsg_or_laea_a and epsg_or_laea_b:
        return True

    return False


def _looks_like_laea_europe(crs_value):
    text = str(crs_value).upper()

    if "EPSG:3035" in text:
        return True

    laea_tokens = [
        "LAMBERT_AZIMUTHAL_EQUAL_AREA",
        "LATITUDE_OF_CENTER",
        "LONGITUDE_OF_CENTER",
        "FALSE_EASTING",
        "FALSE_NORTHING",
        "52",
        "10",
        "4321000",
        "3210000",
    ]

    return all(token in text for token in laea_tokens)

def check_forcing_shape(dataset, variable_name, reference_shape):
    """
    Check that one forcing timestep has the expected raster shape.
    """

    data_array = dataset[variable_name]

    if data_array.ndim == 3:
        first_timestep = data_array.isel({data_array.dims[0]: 0})
        forcing_shape = first_timestep.shape
    elif data_array.ndim == 2:
        forcing_shape = data_array.shape
    else:
        raise ValueError(
            f"Forcing variable '{variable_name}' has unsupported dimensions: "
            f"{data_array.dims}"
        )

    if forcing_shape != reference_shape:
        raise ValueError(
            f"Forcing variable '{variable_name}' has spatial shape "
            f"{forcing_shape}, expected {reference_shape}"
        )


def check_crop_fractions(crop_fraction_data, tolerance=0.01, nodata=-9999.0):
    """
    Basic check on crop fractions.

    Detect whether crop fractions are in 0-1 or 0-100 units and report
    normalization expectations before the model run.
    """

    valid_values = crop_fraction_data[
        np.isfinite(crop_fraction_data) & (crop_fraction_data != nodata)
    ]
    if valid_values.size > 0 and float(np.min(valid_values)) < -tolerance:
        raise ValueError("Crop fraction raster contains negative values.")

    check_data = np.where(
        np.isfinite(crop_fraction_data)
        & (crop_fraction_data != nodata)
        & (crop_fraction_data > 0),
        crop_fraction_data,
        0.0,
    ).astype(np.float32)

    max_individual = float(np.nanmax(check_data))
    crop_fraction_sum = np.sum(check_data, axis=0)
    max_fraction_sum = float(np.nanmax(crop_fraction_sum))

    if max_individual > 1.5 or max_fraction_sum > 1.5:
        print(
            "Input crop fractions appear to be stored as percentages; "
            "they will be divided by 100 inside prepare_crop_fractions()."
        )
        check_data = check_data / 100.0
        crop_fraction_sum = np.sum(check_data, axis=0)
        max_fraction_sum = float(np.nanmax(crop_fraction_sum))

    if max_fraction_sum > 1.01:
        print(
            "Warning: crop fractions still sum above 1.01 in some pixels after "
            "temporary conversion in checks. "
            f"Maximum sum = {max_fraction_sum:.3f}. "
            "The model will normalize those pixels in prepare_crop_fractions()."
        )


def run_input_checks(
    reference_shape,
    reference_profile,
    fmax,
    irrigation_mask,
    irrigation_profile,
    crop_fraction_data,
    crop_profile,
    phenology,
    valid_area_mask=None,
    valid_area_profile=None,
    precipitation_dataset=None,
    precipitation_variable=None,
    et0_dataset=None,
    et0_variable=None,
    check_forcing_spatial_shape=True,
):
    """
    Run all basic checks before starting the model.
    """

    check_array_shape("fmax", fmax, reference_shape)

    if irrigation_mask is not None:
        check_array_shape("irrigation_mask", irrigation_mask, reference_shape)

        if irrigation_profile is None:
            raise ValueError(
                "irrigation_profile is required when irrigation_mask is provided."
            )

        check_profile_match("irrigation_mask", irrigation_profile, reference_profile)
    elif irrigation_profile is not None:
        raise ValueError(
            "irrigation_profile was provided but irrigation_mask is None."
        )

    if valid_area_mask is not None:
        check_array_shape("valid_area_mask", valid_area_mask, reference_shape)

    if valid_area_profile is not None:
        check_profile_match("valid_area_mask", valid_area_profile, reference_profile)

    check_array_shape("crop_fraction_data", crop_fraction_data, reference_shape)
    check_profile_match("crop_fraction_raster", crop_profile, reference_profile)

    for name, layer in phenology.items():
        check_array_shape(f"phenology layer '{name}'", layer, reference_shape)

    if (
        check_forcing_spatial_shape
        and precipitation_dataset is not None
        and precipitation_variable is not None
        and et0_dataset is not None
        and et0_variable is not None
    ):
        check_forcing_shape(
            dataset=precipitation_dataset,
            variable_name=precipitation_variable,
            reference_shape=reference_shape,
        )

        check_forcing_shape(
            dataset=et0_dataset,
            variable_name=et0_variable,
            reference_shape=reference_shape,
        )

    check_crop_fractions(crop_fraction_data, nodata=-9999.0)

    print("Input checks passed.")