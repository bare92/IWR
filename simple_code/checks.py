import numpy as np


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

    keys = ["width", "height", "crs", "transform"]

    for key in keys:
        if profile.get(key) != reference_profile.get(key):
            raise ValueError(
                f"{name} profile mismatch for '{key}'.\n"
                f"{name}:     {profile.get(key)}\n"
                f"reference: {reference_profile.get(key)}"
            )


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


def check_crop_fractions(crop_fraction_data, tolerance=0.01):
    """
    Basic check on crop fractions.

    Fractions should be >= 0 and normally their sum should not be much above 1.
    """

    if np.nanmin(crop_fraction_data) < -tolerance:
        raise ValueError("Crop fraction raster contains negative values.")

    crop_fraction_sum = np.nansum(
        np.where(crop_fraction_data > 0, crop_fraction_data, 0.0),
        axis=0,
    )

    max_fraction_sum = np.nanmax(crop_fraction_sum)

    if max_fraction_sum > 1.0 + tolerance:
        print(
            "Warning: crop fractions sum above 1 in some pixels. "
            f"Maximum sum = {max_fraction_sum:.3f}. "
            "The model will normalize values above 1."
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
    precipitation_dataset,
    precipitation_variable,
    et0_dataset,
    et0_variable,
):
    """
    Run all basic checks before starting the model.
    """

    check_array_shape("fmax", fmax, reference_shape)

    check_array_shape("irrigation_mask", irrigation_mask, reference_shape)
    check_profile_match("irrigation_mask", irrigation_profile, reference_profile)

    check_array_shape("crop_fraction_data", crop_fraction_data, reference_shape)
    check_profile_match("crop_fraction_raster", crop_profile, reference_profile)

    for name, layer in phenology.items():
        check_array_shape(f"phenology layer '{name}'", layer, reference_shape)

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

    check_crop_fractions(crop_fraction_data)

    print("Input checks passed.")