import csv
import warnings
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from utilities import array_stats, assert_reasonable_range, read_forcing_geotiff_day, debug_imshow
from utilities import build_forcing_file_index
from phenology_functions import (
    create_phenology_status_mask_from_date,
    create_dynamic_kc_curve_from_date,
)


IWR_MODE_WATNEEDS_BLUE_ET = "watneeds_blue_et"
IWR_MODE_THEORETICAL_NET_IRRIGATION = "theoretical_net_irrigation"

IWR_DOMAIN_IRRIGATED = "irrigated"
IWR_DOMAIN_NON_IRRIGATED = "non_irrigated"
IWR_DOMAIN_ALL_CROPPED = "all_cropped"

THEORETICAL_IWR_TARGET_STRESS_THRESHOLD = "stress_threshold"
THEORETICAL_IWR_TARGET_FIELD_CAPACITY = "field_capacity"

VALID_IWR_MODES = {
    IWR_MODE_WATNEEDS_BLUE_ET,
    IWR_MODE_THEORETICAL_NET_IRRIGATION,
}

VALID_IWR_DOMAINS = {
    IWR_DOMAIN_IRRIGATED,
    IWR_DOMAIN_NON_IRRIGATED,
    IWR_DOMAIN_ALL_CROPPED,
}

VALID_THEORETICAL_IWR_TARGETS = {
    THEORETICAL_IWR_TARGET_STRESS_THRESHOLD,
    THEORETICAL_IWR_TARGET_FIELD_CAPACITY,
}


def normalize_iwr_configuration(
    iwr_mode,
    iwr_domain,
    theoretical_iwr_target,
):
    mode = (iwr_mode or IWR_MODE_WATNEEDS_BLUE_ET).strip().lower()
    target = (theoretical_iwr_target or THEORETICAL_IWR_TARGET_STRESS_THRESHOLD).strip().lower()

    if mode not in VALID_IWR_MODES:
        raise ValueError(
            f"Unsupported iwr_mode '{iwr_mode}'. "
            f"Valid values: {sorted(VALID_IWR_MODES)}"
        )

    if target not in VALID_THEORETICAL_IWR_TARGETS:
        raise ValueError(
            f"Unsupported theoretical_iwr_target '{theoretical_iwr_target}'. "
            f"Valid values: {sorted(VALID_THEORETICAL_IWR_TARGETS)}"
        )

    if iwr_domain is None or str(iwr_domain).strip() == "":
        if mode == IWR_MODE_WATNEEDS_BLUE_ET:
            domain = IWR_DOMAIN_IRRIGATED
        else:
            domain = IWR_DOMAIN_ALL_CROPPED
    else:
        domain = str(iwr_domain).strip().lower()

    if domain not in VALID_IWR_DOMAINS:
        raise ValueError(
            f"Unsupported iwr_domain '{iwr_domain}'. "
            f"Valid values: {sorted(VALID_IWR_DOMAINS)}"
        )

    if mode == IWR_MODE_WATNEEDS_BLUE_ET and domain != IWR_DOMAIN_IRRIGATED:
        raise ValueError(
            "iwr_mode='watneeds_blue_et' only supports iwr_domain='irrigated'."
        )

    return mode, domain, target


def create_iwr_domain_mask(
    iwr_domain,
    valid_area_pixels,
    irrigation_mask=None,
    nodata=-9999.0,
):
    if iwr_domain == IWR_DOMAIN_ALL_CROPPED:
        return valid_area_pixels.astype(bool)

    if irrigation_mask is None:
        raise ValueError(
            f"iwr_domain='{iwr_domain}' requires a valid irrigation_mask raster."
        )

    irrigation_valid = (
        np.isfinite(irrigation_mask)
        & (irrigation_mask != nodata)
    )

    if iwr_domain == IWR_DOMAIN_IRRIGATED:
        return (irrigation_mask == 1) & irrigation_valid & valid_area_pixels

    if iwr_domain == IWR_DOMAIN_NON_IRRIGATED:
        return (irrigation_mask != 1) & irrigation_valid & valid_area_pixels

    raise ValueError(f"Unsupported iwr_domain '{iwr_domain}'.")


def create_theoretical_iwr_target_storage(
    raw,
    total_available_water_pixel,
    theoretical_iwr_target,
    nodata=-9999.0,
):
    target_storage = np.full_like(raw, nodata, dtype=np.float32)

    valid_mask = (
        np.isfinite(raw)
        & np.isfinite(total_available_water_pixel)
        & (raw != nodata)
        & (total_available_water_pixel != nodata)
        & (total_available_water_pixel >= 0)
    )

    if theoretical_iwr_target == THEORETICAL_IWR_TARGET_STRESS_THRESHOLD:
        target_storage[valid_mask] = raw[valid_mask]
    elif theoretical_iwr_target == THEORETICAL_IWR_TARGET_FIELD_CAPACITY:
        target_storage[valid_mask] = total_available_water_pixel[valid_mask]
    else:
        raise ValueError(
            f"Unsupported theoretical_iwr_target '{theoretical_iwr_target}'."
        )

    target_storage[valid_mask] = np.clip(
        target_storage[valid_mask],
        0.0,
        total_available_water_pixel[valid_mask],
    )

    return target_storage.astype(np.float32)


def compute_theoretical_net_irrigation_requirement(
    soil_moisture_previous,
    precipitation_effective,
    potential_evapotranspiration,
    deep_percolation,
    runoff,
    target_storage,
    total_available_water_pixel,
    demand_mask,
    nodata=-9999.0,
    valid_mask=None,
):
    theoretical_iwr = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)
    soil_moisture_new = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)
    pre_irrigation_storage = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)

    reference_valid = (
        np.isfinite(soil_moisture_previous)
        & np.isfinite(precipitation_effective)
        & np.isfinite(potential_evapotranspiration)
        & np.isfinite(deep_percolation)
        & np.isfinite(runoff)
        & np.isfinite(target_storage)
        & np.isfinite(total_available_water_pixel)
        & (soil_moisture_previous != nodata)
        & (precipitation_effective != nodata)
        & (potential_evapotranspiration != nodata)
        & (deep_percolation != nodata)
        & (runoff != nodata)
        & (target_storage != nodata)
        & (total_available_water_pixel != nodata)
        & (total_available_water_pixel >= 0)
    )

    if valid_mask is not None:
        reference_valid = reference_valid & valid_mask

    pre_irrigation_storage[reference_valid] = (
        soil_moisture_previous[reference_valid]
        + precipitation_effective[reference_valid]
        - potential_evapotranspiration[reference_valid]
        - deep_percolation[reference_valid]
        - runoff[reference_valid]
    ).astype(np.float32)

    theoretical_iwr[reference_valid] = 0.0

    irrigation_demand_mask = reference_valid & demand_mask
    theoretical_iwr[irrigation_demand_mask] = np.maximum(
        target_storage[irrigation_demand_mask] - pre_irrigation_storage[irrigation_demand_mask],
        0.0,
    )

    soil_moisture_new[reference_valid] = np.clip(
        pre_irrigation_storage[reference_valid] + theoretical_iwr[reference_valid],
        0.0,
        total_available_water_pixel[reference_valid],
    ).astype(np.float32)

    return (
        theoretical_iwr.astype(np.float32),
        soil_moisture_new.astype(np.float32),
        pre_irrigation_storage.astype(np.float32),
    )


def prepare_crop_fractions(crop_fraction_data):
    """
    Prepare crop fractions.

    The model expects crop fractions in 0-1 units.
    If the input appears to be stored as percentages in 0-100,
    convert to 0-1 before any normalization.

    Input shape:
        crop, rows, cols

    Returns:
        crop_fraction_data, crop_fraction_sum
    """

    crop_fraction_data = crop_fraction_data.astype(np.float32)

    # Clean invalid values first. The uploaded land-cover raster uses NaN nodata.
    crop_fraction_data = np.where(
        np.isfinite(crop_fraction_data) & (crop_fraction_data > 0),
        crop_fraction_data,
        0.0,
    ).astype(np.float32)

    crop_fraction_sum_before = np.sum(crop_fraction_data, axis=0)

    max_individual_before = float(np.nanmax(crop_fraction_data))
    max_sum_before = float(np.nanmax(crop_fraction_sum_before))

    print("Crop fraction diagnostics before preparation:")
    print(f"  max individual value: {max_individual_before:.6f}")
    print(f"  max sum across bands: {max_sum_before:.6f}")

    converted_from_percent = False

    if max_individual_before > 1.5 or max_sum_before > 1.5:
        print("Crop fractions appear to be stored as percentages. Dividing by 100.")
        crop_fraction_data = crop_fraction_data / 100.0
        converted_from_percent = True

    max_individual_after_conversion = float(np.nanmax(crop_fraction_data))

    if max_individual_after_conversion > 1.01:
        raise ValueError(
            "Crop fraction raster still contains values > 1 after percent conversion. "
            f"Maximum individual value = {max_individual_after_conversion:.6f}"
        )

    crop_fraction_sum = np.sum(crop_fraction_data, axis=0)

    too_high_mask = crop_fraction_sum > 1.01

    if np.any(too_high_mask):
        print(
            "Warning: crop fractions still sum above 1.01 in some pixels after conversion. "
            f"Maximum sum = {float(np.nanmax(crop_fraction_sum)):.6f}. "
            "Normalizing only those pixels to 1."
        )

        scale = np.ones_like(crop_fraction_sum, dtype=np.float32)
        scale[too_high_mask] = 1.0 / crop_fraction_sum[too_high_mask]

        crop_fraction_data = crop_fraction_data * scale[np.newaxis, :, :]
        crop_fraction_sum = np.sum(crop_fraction_data, axis=0)

    print("Crop fraction diagnostics after preparation:")
    print(f"  converted from percent: {converted_from_percent}")
    print(f"  max individual value: {float(np.nanmax(crop_fraction_data)):.6f}")
    print(f"  max sum across bands: {float(np.nanmax(crop_fraction_sum)):.6f}")

    return crop_fraction_data.astype(np.float32), crop_fraction_sum.astype(np.float32)


def create_effective_root_depth(
    crop_fraction_data,
    crop_df,
):
    """
    Create one effective root depth per pixel.

    The effective root depth is calculated as the crop-fraction-weighted
    average considering only the crop fractions present in the pixel.

    This avoids diluting root depth by non-crop or empty pixel fractions.

    Example:
        If a pixel has only maize with fraction 0.05 and maize root depth is 1.2 m,
        the output root depth is 1.2 m, not 0.05 * 1.2 = 0.06 m.

    Formula:
        root_depth_eff = sum(crop_fraction_i * root_depth_i) / sum(crop_fraction_i)

    Output unit:
        m
    """

    root_depths = crop_df["root_depth_max_m"].to_numpy(dtype=np.float32)

    weighted_root_depth_sum = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    crop_fraction_sum = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    for crop_index, root_depth in enumerate(root_depths):
        crop_fraction = crop_fraction_data[crop_index, :, :]

        weighted_root_depth_sum += crop_fraction * root_depth
        crop_fraction_sum += crop_fraction

    effective_root_depth = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    valid_pixels = crop_fraction_sum > 0

    effective_root_depth[valid_pixels] = (
        weighted_root_depth_sum[valid_pixels]
        / crop_fraction_sum[valid_pixels]
    )

    return effective_root_depth


def create_total_available_water_pixel(
    total_available_water,
    crop_fraction_data,
    crop_df,
    nodata=-9999.0,
):
    """
    Create one TAW value per pixel.

    total_available_water input is:
        FC - WP, in m3/m3

    Pixel storage:
        TAW_mm = (FC - WP) * effective_root_depth_m * 1000

    Output unit:
        mm
    """

    effective_root_depth = create_effective_root_depth(
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
    )

    total_available_water_pixel = (
        total_available_water
        * effective_root_depth
        * 1000.0
    ).astype(np.float32)

    total_available_water_pixel[total_available_water == nodata] = nodata

    return total_available_water_pixel


def create_raw_pixel(
    total_available_water,
    crop_fraction_data,
    crop_df,
    nodata=-9999.0,
):
    """
    Create one storage threshold value per pixel (returned as raw for backward compatibility).

    total_available_water input is:
        FC - WP, in m3/m3

    For multiple crops in one pixel, compute the storage-equivalent threshold:
        RAW_storage_threshold_mm = (FC - WP) * 1000 *
                                   [sum(crop_fraction_i * root_depth_i * (1 - p_i))
                                    / sum(crop_fraction_i)]

    In FAO-56, p is a depletion fraction and RAW_depletion = p * TAW.
    This model uses S as available water storage, not depletion.
    Since depletion Dr = TAW - S, the equivalent storage threshold is:
        S_threshold = TAW - RAW_depletion = (1 - p) * TAW.
    Therefore the stress threshold used by the model is (1 - p) * TAW.

    This avoids diluting RAW by non-crop or empty pixel fractions.

    Output unit:
        mm
    """

    root_depths = crop_df["root_depth_max_m"].to_numpy(dtype=np.float32)
    depletion_factors = crop_df["p"].to_numpy(dtype=np.float32)

    weighted_root_depth_storage_threshold_sum = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    crop_fraction_sum = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    for crop_index, (root_depth, p) in enumerate(zip(root_depths, depletion_factors)):
        crop_fraction = crop_fraction_data[crop_index, :, :]

        storage_threshold_factor = 1.0 - p

        weighted_root_depth_storage_threshold_sum += (
            crop_fraction * root_depth * storage_threshold_factor
        )
        crop_fraction_sum += crop_fraction

    effective_root_depth_storage_threshold = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    valid_crop_pixels = crop_fraction_sum > 0

    effective_root_depth_storage_threshold[valid_crop_pixels] = (
        weighted_root_depth_storage_threshold_sum[valid_crop_pixels]
        / crop_fraction_sum[valid_crop_pixels]
    )

    raw_pixel = (
        total_available_water
        * effective_root_depth_storage_threshold
        * 1000.0
    ).astype(np.float32)

    raw_pixel[total_available_water == nodata] = nodata

    # Optional: keep non-crop pixels as nodata instead of 0
    raw_pixel[~valid_crop_pixels] = nodata

    return raw_pixel


def initialize_soil_moisture(
    total_available_water_pixel,
    initial_fraction=0.5,
    nodata=-9999.0,
):
    """
    Initialize one soil moisture value per pixel.

    S0 = initial_fraction * TAW
    """

    soil_moisture = total_available_water_pixel * initial_fraction
    soil_moisture[total_available_water_pixel == nodata] = nodata

    return soil_moisture.astype(np.float32)


def create_kc_pixel(
    current_date,
    phenology,
    crop_fraction_data,
    crop_df,
    nodata=-9999.0,
):
    """
    Create one crop coefficient per pixel using cropped-area convention.

    Kc_pixel = sum(crop_fraction_i * Kc_i) / sum(crop_fraction_i)

    Kc_i is a continuous FAO-56-style daily curve for each crop.
    Output unit: dimensionless
    """
    shape = crop_fraction_data.shape[1:]
    kc_weighted_sum = np.zeros(shape, dtype=np.float32)
    crop_fraction_sum = np.sum(crop_fraction_data, axis=0).astype(np.float32)

    for crop_index in range(crop_fraction_data.shape[0]):
        crop_fraction = crop_fraction_data[crop_index, :, :]
        kc_ini = float(crop_df.iloc[crop_index]["Kc_ini"])
        kc_mid = float(crop_df.iloc[crop_index]["Kc_mid"])
        kc_end = float(crop_df.iloc[crop_index]["Kc_end"])

        kc_crop = create_dynamic_kc_curve_from_date(
            current_date=current_date,
            phenology=phenology,
            kc_ini=kc_ini,
            kc_mid=kc_mid,
            kc_end=kc_end,
            nodata=nodata,
            inactive_kc=0.0,
        )

        kc_weighted_sum += crop_fraction * kc_crop

    kc_pixel = np.zeros(shape, dtype=np.float32)
    valid = crop_fraction_sum > 0
    kc_pixel[valid] = kc_weighted_sum[valid] / crop_fraction_sum[valid]

    return kc_pixel.astype(np.float32)


def compute_deep_percolation(
    soil_moisture_previous,
    raw,
    total_available_water_pixel,
    fmax_pixel,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute deep percolation D.

    D = Fmax * (S - RAW) / (TAW - RAW), if RAW <= S <= TAW
    D = 0, if S < RAW

    If valid_mask is provided, output is nodata outside valid_mask.
    """

    deep_percolation = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)

    base_mask = (
        np.isfinite(soil_moisture_previous)
        & (soil_moisture_previous != nodata)
        & (total_available_water_pixel != nodata)
        & (fmax_pixel != nodata)
        & (total_available_water_pixel > raw)
    )
    if valid_mask is not None:
        base_mask = base_mask & valid_mask

    # All valid pixels start at zero percolation.
    deep_percolation[base_mask] = 0.0

    percolation_mask = (
        base_mask
        & (soil_moisture_previous >= raw)
        & (soil_moisture_previous <= total_available_water_pixel)
    )

    theoretical_deep_percolation = np.zeros_like(
        soil_moisture_previous,
        dtype=np.float32,
    )

    theoretical_deep_percolation[percolation_mask] = (
        fmax_pixel[percolation_mask]
        * (
            soil_moisture_previous[percolation_mask]
            - raw[percolation_mask]
        )
        / (
            total_available_water_pixel[percolation_mask]
            - raw[percolation_mask]
        )
    )

    available_for_percolation = np.zeros_like(
        soil_moisture_previous,
        dtype=np.float32,
    )

    available_for_percolation[percolation_mask] = np.maximum(
        soil_moisture_previous[percolation_mask]
        - raw[percolation_mask],
        0.0,
    )

    deep_percolation[percolation_mask] = np.minimum(
        theoretical_deep_percolation[percolation_mask],
        available_for_percolation[percolation_mask],
    )

    return deep_percolation


def scale_fluxes_if_water_deficit(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    irrigated_pixels,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    If S_prev + Peff - ETa - D is negative,
    scale ETa and D proportionally to close the balance.

    This is applied only to non-irrigated pixels.
    Irrigated pixels receive irrigation instead.

    If valid_mask is provided, scaling is only applied inside valid_mask;
    output retains the incoming values outside valid_mask.
    """

    out_eta = actual_evapotranspiration.copy()
    out_dp  = deep_percolation.copy()

    active_mask = (
        (soil_moisture_previous != nodata)
        & (actual_evapotranspiration != nodata)
        & (deep_percolation != nodata)
        & (precipitation_effective != nodata)
    )
    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    available_water = soil_moisture_previous + precipitation_effective
    outgoing_water  = actual_evapotranspiration + deep_percolation

    deficit_mask = (
        active_mask
        & (available_water < outgoing_water)
        & (outgoing_water > 0)
        & (~irrigated_pixels)
    )

    scale_factor = np.ones_like(soil_moisture_previous, dtype=np.float32)
    scale_factor[deficit_mask] = (
        available_water[deficit_mask]
        / outgoing_water[deficit_mask]
    )

    out_eta[active_mask] = actual_evapotranspiration[active_mask] * scale_factor[active_mask]
    out_dp[active_mask]  = deep_percolation[active_mask]          * scale_factor[active_mask]

    return out_eta, out_dp


def compute_irrigation(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    irrigated_pixels,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute one irrigation value per pixel.

    I is the water needed to avoid a negative balance.
    Applied only where irrigated_pixels == True AND valid_mask (if given).

    Outside valid pixels: nodata.
    Valid non-irrigated pixels: 0.
    Valid irrigated pixels: max(deficit, 0).
    """

    irrigation = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)

    valid_pixel_mask = (
        (soil_moisture_previous != nodata)
        & (precipitation_effective != nodata)
        & (actual_evapotranspiration != nodata)
        & (deep_percolation != nodata)
    )
    if valid_mask is not None:
        valid_pixel_mask = valid_pixel_mask & valid_mask

    # All valid pixels start at 0 irrigation.
    irrigation[valid_pixel_mask] = 0.0

    # Only compute irrigation for valid irrigated pixels.
    irrigated_valid = valid_pixel_mask & irrigated_pixels

    balance = (
        soil_moisture_previous[irrigated_valid]
        + precipitation_effective[irrigated_valid]
        - actual_evapotranspiration[irrigated_valid]
        - deep_percolation[irrigated_valid]
    )
    irrigation[irrigated_valid] = np.maximum(-balance, 0.0)

    return irrigation.astype(np.float32)


def compute_blue_water_requirement_watneeds(
    potential_evapotranspiration,
    green_evapotranspiration,
    irrigated_pixels,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute WATNEEDS-like blue water requirement.

    Blue water requirement is the missing water needed to move from
    green/rainfed stressed ET to unstressed crop ET.

    IWR_blue = max(ETc - ET_green, 0)

    It is computed only on irrigated pixels.
    Valid non-irrigated pixels are set to 0.
    Outside valid pixels: nodata.

    Output:
        mm/day over cropped/irrigated crop area
    """

    blue_iwr = np.full_like(potential_evapotranspiration, nodata, dtype=np.float32)

    active_mask = (
        np.isfinite(potential_evapotranspiration)
        & np.isfinite(green_evapotranspiration)
        & (potential_evapotranspiration != nodata)
        & (green_evapotranspiration != nodata)
    )

    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    blue_iwr[active_mask] = 0.0

    irrigated_valid = active_mask & irrigated_pixels

    blue_iwr[irrigated_valid] = np.maximum(
        potential_evapotranspiration[irrigated_valid]
        - green_evapotranspiration[irrigated_valid],
        0.0,
    )

    return blue_iwr.astype(np.float32)


def compute_subsurface_runoff(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    total_available_water_pixel,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute one runoff value per pixel.

    If:
        S_prev + Peff - ETa - D > TAW

    then:
        R = balance - TAW

    In the WATNEEDS-style green/blue decomposition used here, irrigation is not
    added to the runoff balance because blue water is assumed to satisfy the ET
    deficit on the same day and does not create residual soil storage.

    If valid_mask is provided, output is nodata outside valid_mask.
    """

    runoff = np.full_like(soil_moisture_previous, nodata, dtype=np.float32)

    active_mask = (
        (total_available_water_pixel != nodata)
        & (soil_moisture_previous != nodata)
        & (precipitation_effective != nodata)
        & (actual_evapotranspiration != nodata)
        & (deep_percolation != nodata)
    )
    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    runoff[active_mask] = 0.0

    balance = (
        soil_moisture_previous[active_mask]
        + precipitation_effective[active_mask]
        - actual_evapotranspiration[active_mask]
        - deep_percolation[active_mask]
    )
    runoff[active_mask] = np.maximum(
        balance - total_available_water_pixel[active_mask],
        0.0,
    )

    return runoff.astype(np.float32)


def water_balance_step(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    runoff,
    irrigation,
    total_available_water_pixel,
    delta_t=1.0,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    One daily water balance step.

    S_t = S_t-1 + delta_t * (P_eff - ETa - D - R + I)

    If valid_mask is provided, output is nodata outside valid_mask.
    """

    # Carry forward previous soil moisture as the default (preserves nodata).
    soil_moisture = soil_moisture_previous.copy()

    active_mask = (
        (total_available_water_pixel != nodata)
        & (soil_moisture_previous != nodata)
        & (precipitation_effective != nodata)
        & (actual_evapotranspiration != nodata)
        & (deep_percolation != nodata)
        & (runoff != nodata)
        & (irrigation != nodata)
    )
    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    new_sm = (
        soil_moisture_previous[active_mask]
        + delta_t
        * (
            precipitation_effective[active_mask]
            - actual_evapotranspiration[active_mask]
            - deep_percolation[active_mask]
            - runoff[active_mask]
            + irrigation[active_mask]
        )
    )
    new_sm = np.maximum(new_sm, 0.0)
    new_sm = np.minimum(new_sm, total_available_water_pixel[active_mask])

    soil_moisture[active_mask] = new_sm

    return soil_moisture.astype(np.float32)


def write_active_pixel_mask_geotiff(
    output_path,
    data,
    profile,
    date=None,
    iwr_domain=None,
):
    """
    Write one daily active-pixel mask as uint8 GeoTIFF.

    Pixel values:
        1 — phenology active AND crop fraction > 0 AND irrigation mask == 1 AND valid area
        0 — any condition above is not met, or nodata
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_profile = profile.copy()
    mask_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=None,
        compress="lzw",
    )

    if profile.get("crs") is not None:
        mask_profile["crs"] = profile["crs"]
    if profile.get("transform") is not None:
        mask_profile["transform"] = profile["transform"]

    description = (
        "1=phenology_active+crop_fraction>0+selected_iwr_domain+valid_area, "
        "0=inactive/nodata"
    )

    tags = {
        "variable": "active_pixel_mask",
        "description": description,
    }
    if date is not None:
        tags["date"] = date.strftime("%Y-%m-%d")
    if iwr_domain is not None:
        tags["iwr_domain"] = str(iwr_domain)

    with rasterio.open(output_path, "w", **mask_profile) as dst:
        dst.write(data.astype(np.uint8), 1)
        dst.update_tags(**tags)


def write_daily_geotiff(
    output_path,
    data,
    profile,
    nodata=-9999.0,
    date=None,
    metadata=None,
):
    """
    Write one daily output GeoTIFF.

    CRS and transform are always taken from *profile* and written explicitly.
    Optional *date* and *metadata* dict are stored as raster tags.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw",
    )

    # Ensure CRS and transform are present and explicit.
    if profile.get("crs") is not None:
        output_profile["crs"] = profile["crs"]
    if profile.get("transform") is not None:
        output_profile["transform"] = profile["transform"]

    tags = {
        "variable": "daily_irrigation_water_requirement",
        "units": "mm/day",
    }
    if date is not None:
        tags["date"] = date.strftime("%Y-%m-%d")
    if metadata is not None:
        tags.update(metadata)

    data_to_write = data.astype("float32")

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(data_to_write, 1)
        dst.update_tags(**tags)


def partition_flux_by_water_source(
    flux,
    green_storage_available,
    blue_storage_available,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Partition a flux into green-water and blue-water components
    according to the relative contribution of green and blue water
    in the available root-zone storage.
    """

    green_flux = np.full_like(flux, nodata, dtype=np.float32)
    blue_flux = np.full_like(flux, nodata, dtype=np.float32)

    active_mask = (
        np.isfinite(flux)
        & np.isfinite(green_storage_available)
        & np.isfinite(blue_storage_available)
        & (flux != nodata)
        & (green_storage_available != nodata)
        & (blue_storage_available != nodata)
    )

    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    total_available = green_storage_available + blue_storage_available

    positive_storage_mask = active_mask & (total_available > 0)

    green_fraction = np.zeros_like(flux, dtype=np.float32)
    blue_fraction = np.zeros_like(flux, dtype=np.float32)

    green_fraction[positive_storage_mask] = (
        green_storage_available[positive_storage_mask]
        / total_available[positive_storage_mask]
    )

    blue_fraction[positive_storage_mask] = (
        blue_storage_available[positive_storage_mask]
        / total_available[positive_storage_mask]
    )

    green_flux[active_mask] = 0.0
    blue_flux[active_mask] = 0.0

    green_flux[positive_storage_mask] = (
        flux[positive_storage_mask]
        * green_fraction[positive_storage_mask]
    )

    blue_flux[positive_storage_mask] = (
        flux[positive_storage_mask]
        * blue_fraction[positive_storage_mask]
    )

    return green_flux.astype(np.float32), blue_flux.astype(np.float32)


def compute_water_stress_coefficient(
    soil_moisture,
    raw,
    irrigated_pixels,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute water stress coefficient Ks.

    For rainfed / non-irrigated pixels:
        Ks = S / RAW, if S < RAW
        Ks = 1,       if S >= RAW

    For irrigated pixels:
        Ks = 1

    Output:
        Ks, dimensionless
    """

    ks = np.full_like(soil_moisture, nodata, dtype=np.float32)

    active_mask = (
        np.isfinite(soil_moisture)
        & np.isfinite(raw)
        & (soil_moisture != nodata)
        & (raw != nodata)
        & (raw > 0)
    )

    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    ks[active_mask] = 1.0

    stressed_mask = (
        active_mask
        & (~irrigated_pixels)
        & (soil_moisture < raw)
    )

    ks[stressed_mask] = soil_moisture[stressed_mask] / raw[stressed_mask]

    ks[active_mask] = np.clip(ks[active_mask], 0.0, 1.0)

    return ks.astype(np.float32)


def compute_green_water_stress_coefficient(
    soil_moisture_green,
    raw,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute the water stress coefficient for the green/rainfed branch.

    This function must NOT force Ks = 1 on irrigated pixels.
    It represents how much of ETc can be supplied by precipitation
    and green soil moisture only.

    Ks = S_green / RAW, if S_green < RAW
    Ks = 1,             if S_green >= RAW

    Output:
        dimensionless
    """

    ks = np.full_like(soil_moisture_green, nodata, dtype=np.float32)

    active_mask = (
        np.isfinite(soil_moisture_green)
        & np.isfinite(raw)
        & (soil_moisture_green != nodata)
        & (raw != nodata)
        & (raw > 0)
    )

    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    ks[active_mask] = 1.0

    stressed_mask = active_mask & (soil_moisture_green < raw)

    ks[stressed_mask] = soil_moisture_green[stressed_mask] / raw[stressed_mask]

    ks[active_mask] = np.clip(ks[active_mask], 0.0, 1.0)

    return ks.astype(np.float32)


def compute_potential_evapotranspiration(
    et0,
    kc_pixel,
):
    """
    Compute potential crop evapotranspiration.

    ET = Kc * ET0
    """

    potential_evapotranspiration = et0 * kc_pixel

    return potential_evapotranspiration.astype(np.float32)


def compute_actual_evapotranspiration(
    potential_evapotranspiration,
    water_stress_coefficient,
    nodata=-9999.0,
    valid_mask=None,
):
    """
    Compute actual evapotranspiration.

    ETa = Ks * ET
    """

    actual_evapotranspiration = np.full_like(
        potential_evapotranspiration,
        nodata,
        dtype=np.float32,
    )

    active_mask = (
        np.isfinite(potential_evapotranspiration)
        & np.isfinite(water_stress_coefficient)
        & (potential_evapotranspiration != nodata)
        & (water_stress_coefficient != nodata)
    )

    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    actual_evapotranspiration[active_mask] = (
        potential_evapotranspiration[active_mask]
        * water_stress_coefficient[active_mask]
    ).astype(np.float32)

    return actual_evapotranspiration


def run_iwr_model(
    start_date,
    end_date,
    total_available_water,
    fmax,
    irrigation_mask,
    valid_area_mask,
    crop_fraction_data,
    crop_df,
    phenology,
    precipitation_geotiff_folder,
    et0_geotiff_folder,
    output_folder=None,
    output_profile=None,
    nodata=-9999.0,
    strict_checks=True,
    write_debug_csv=True,
    write_cumulative_iwr=True,
    write_green_blue_outputs=False,
    write_daily_green_blue_outputs=False,
    write_active_pixel_masks=False,
    debug_mode=False,
    debug_output_folder=None,
    max_precipitation_mm_day=300,
    max_et0_mm_day=20,
    max_iwr_mm_day=100,
    min_valid_forcing_fraction=0.01,
    debug_csv_frequency_days=1,
    iwr_mode=IWR_MODE_WATNEEDS_BLUE_ET,
    iwr_domain=None,
    theoretical_iwr_target=THEORETICAL_IWR_TARGET_STRESS_THRESHOLD,
    initial_soil_moisture_fraction=0.5,
):
    """
    Run daily IWR water balance using the cropped-area-depth convention.

    - The model uses cropped-area-depth convention throughout.
    - P_eff = 0.95 * P (mm/day over cropped area).
    - Kc_pixel = sum(f_i * Kc_i) / sum(f_i) — average Kc over the cropped fraction.
    - TAW and RAW are in mm of water over the cropped root zone.
    - The model supports two IWR modes:
        watneeds_blue_et:
            IWR = max(ETc - ET_green, 0) on irrigated pixels only.
        theoretical_net_irrigation:
            IWR is the reference-state net irrigation needed to restore
            selected target storage after ETc and non-ET losses.
    - ET_green is the actual ET supplied by green water (rainfall + stored green moisture),
      computed using a stress coefficient based on green soil moisture.
    - Output mm values must be converted to volume using irrigated crop area,
      not full pixel area:
          volume_m3 = IWR_mm / 1000 * irrigated_crop_area_m2
    - D is computed from RAW, TAW and Fmax.
    - R is computed when storage exceeds TAW.
    - nodata is propagated safely; no arithmetic on -9999 values.
    - Daily irrigation maps are written as GeoTIFFs if output_folder is provided.
    - Computation is restricted to valid_area_mask == 1.
        - Before writing daily IWR, pixels that are not active are set to nodata.
            Active means: phenology active AND crop fraction > 0 AND selected IWR domain
            AND valid area.
        - Optional uint8 active-pixel mask files can still be written with
            write_active_pixel_masks=True.
    """

    crop_fraction_data, crop_fraction_sum = prepare_crop_fractions(
        crop_fraction_data=crop_fraction_data,
    )

    iwr_mode, iwr_domain, theoretical_iwr_target = normalize_iwr_configuration(
        iwr_mode=iwr_mode,
        iwr_domain=iwr_domain,
        theoretical_iwr_target=theoretical_iwr_target,
    )

    if not (0.0 <= float(initial_soil_moisture_fraction) <= 1.0):
        raise ValueError(
            "initial_soil_moisture_fraction must be in the range [0, 1]."
        )

    print(
        "IWR configuration:",
        f"mode={iwr_mode},",
        f"domain={iwr_domain},",
        f"target={theoretical_iwr_target},",
        f"initial_soil_moisture_fraction={float(initial_soil_moisture_fraction):.3f}",
    )

    total_available_water_pixel = create_total_available_water_pixel(
        total_available_water=total_available_water,
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
        nodata=nodata,
    )

    raw = create_raw_pixel(
        total_available_water=total_available_water,
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
        nodata=nodata,
    )

    # Despite the historical variable name "raw", this array is used as a
    # storage threshold in the S-based water balance.
    # Since crop_df["p"] is the FAO depletion fraction, the storage threshold is:
    #     S_threshold = (1 - p) * TAW
    # not:
    #     p * TAW

    fmax_pixel = fmax

    # Diagnostic: compute raw as a fraction of TAW for validation
    raw_fraction_of_taw = np.full_like(raw, nodata, dtype=np.float32)

    raw_fraction_mask = (
        np.isfinite(raw)
        & np.isfinite(total_available_water_pixel)
        & (raw != nodata)
        & (total_available_water_pixel != nodata)
        & (total_available_water_pixel > 0)
    )

    raw_fraction_of_taw[raw_fraction_mask] = (
        raw[raw_fraction_mask]
        / total_available_water_pixel[raw_fraction_mask]
    ).astype(np.float32)

    soil_moisture = initialize_soil_moisture(
        total_available_water_pixel=total_available_water_pixel,
        initial_fraction=float(initial_soil_moisture_fraction),
        nodata=nodata,
    )

    valid_area_pixels = np.isfinite(valid_area_mask) & (valid_area_mask == 1)

    iwr_domain_pixels = create_iwr_domain_mask(
        iwr_domain=iwr_domain,
        valid_area_pixels=valid_area_pixels,
        irrigation_mask=irrigation_mask,
        nodata=nodata,
    )

    theoretical_target_storage = create_theoretical_iwr_target_storage(
        raw=raw,
        total_available_water_pixel=total_available_water_pixel,
        theoretical_iwr_target=theoretical_iwr_target,
        nodata=nodata,
    )

    cumulative_irrigation = np.zeros_like(soil_moisture, dtype=np.float32)
    # Actual state used by the water balance (rainfed-only in theoretical mode).
    soil_moisture_actual = soil_moisture

    if iwr_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION:
        soil_moisture_reference = theoretical_target_storage.copy()
    else:
        soil_moisture_reference = soil_moisture_actual.copy()

    soil_moisture_green = soil_moisture_actual.copy()
    soil_moisture_blue = np.zeros_like(soil_moisture, dtype=np.float32)
    soil_moisture_blue[soil_moisture_actual == nodata] = nodata

    cumulative_green_et = np.zeros_like(soil_moisture, dtype=np.float32)
    cumulative_blue_et = np.zeros_like(soil_moisture, dtype=np.float32)
    cumulative_green_deep_percolation = np.zeros_like(soil_moisture, dtype=np.float32)
    cumulative_blue_deep_percolation = np.zeros_like(soil_moisture, dtype=np.float32)
    cumulative_green_runoff = np.zeros_like(soil_moisture, dtype=np.float32)
    cumulative_blue_runoff = np.zeros_like(soil_moisture, dtype=np.float32)

    static_valid_mask = (soil_moisture_actual != nodata) & valid_area_pixels
    cumulative_static_mask = static_valid_mask & iwr_domain_pixels & (crop_fraction_sum > 0)
    daily_stats_rows = []

    # Prepare active-pixel-mask output folder at the same level as output_folder.
    active_pixel_masks_folder = None
    if write_active_pixel_masks and output_folder is not None:
        active_pixel_masks_folder = Path(output_folder).parent / "IWR active pixel masks"
        active_pixel_masks_folder.mkdir(parents=True, exist_ok=True)

    if debug_csv_frequency_days < 1:
        raise ValueError("debug_csv_frequency_days must be >= 1")

    print(f"Building precipitation file index from: {precipitation_geotiff_folder}")
    precipitation_file_index = build_forcing_file_index(
        precipitation_geotiff_folder
    )
    print(f"Indexed precipitation files: {len(precipitation_file_index)}")

    print(f"Building ET0 file index from: {et0_geotiff_folder}")
    et0_file_index = build_forcing_file_index(
        et0_geotiff_folder
    )
    print(f"Indexed ET0 files: {len(et0_file_index)}")

    # Pre-compute TAW max for soil_moisture sanity check.
    taw_valid = total_available_water_pixel[total_available_water_pixel != nodata]
    taw_max = float(np.max(taw_valid)) * 1.05 if taw_valid.size > 0 else 10000.0

    current_date = start_date
    day_index = 0

    total_days = (end_date - start_date).days + 1
    print(
        f"Starting daily processing from {start_date.strftime('%Y-%m-%d')} "
        f"to {end_date.strftime('%Y-%m-%d')} ({total_days} days)"
    )

    while current_date <= end_date:

        date_str = current_date.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------ #
        # 1. Read forcing (fail-fast on empty / exceeded thresholds)           #
        # ------------------------------------------------------------------ #
        precipitation = read_forcing_geotiff_day(
            geotiff_folder=precipitation_geotiff_folder,
            date=current_date,
            min_value=0.0,
            max_value=max_precipitation_mm_day if strict_checks else None,
            reference_profile=output_profile,
            nodata=nodata,
            min_valid_fraction=min_valid_forcing_fraction,
            variable_name="precipitation",
            file_index=precipitation_file_index,
        )

        et0 = read_forcing_geotiff_day(
            geotiff_folder=et0_geotiff_folder,
            date=current_date,
            min_value=0.0,
            max_value=max_et0_mm_day if strict_checks else None,
            reference_profile=output_profile,
            nodata=nodata,
            min_valid_fraction=min_valid_forcing_fraction,
            variable_name="et0",
            file_index=et0_file_index,
        )

        # ------------------------------------------------------------------ #
        # 2. Build validity masks                                              #
        # ------------------------------------------------------------------ #
        forcing_valid_mask = (
            np.isfinite(precipitation)
            & np.isfinite(et0)
            & (precipitation != nodata)
            & (et0 != nodata)
        )

        if not np.any(forcing_valid_mask):
            raise ValueError(
                f"No valid forcing pixels for {date_str}. "
                "Precipitation and/or ET0 are entirely nodata."
            )

        model_valid_mask = (
            forcing_valid_mask
            & valid_area_pixels
            & (soil_moisture_actual != nodata)
            & (total_available_water_pixel != nodata)
            & (raw != nodata)
            & (fmax_pixel != nodata)
            & (crop_fraction_sum > 0)
        )

        # ------------------------------------------------------------------ #
        # 3. Phenology and Kc                                                  #
        # ------------------------------------------------------------------ #
        kc_pixel = create_kc_pixel(
            current_date=current_date,
            phenology=phenology,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            nodata=nodata,
        )
        # Kc is undefined outside valid mask; zero outside is safe.
        kc_pixel[~model_valid_mask] = 0.0

        # ------------------------------------------------------------------ #
        # 4. Sanity checks on forcing inputs                                   #
        # ------------------------------------------------------------------ #
        if strict_checks:
            assert_reasonable_range(
                "precipitation", precipitation, 0.0, max_precipitation_mm_day,
                nodata=nodata, date=current_date,
            )
            assert_reasonable_range(
                "et0", et0, 0.0, max_et0_mm_day,
                nodata=nodata, date=current_date,
            )
            assert_reasonable_range(
                "kc_pixel", kc_pixel, 0.0, 1.5,
                nodata=nodata, date=current_date, raise_error=False,
            )

        # ------------------------------------------------------------------ #
        # 5. Effective precipitation  (masked)                                 #
        # ------------------------------------------------------------------ #
        precipitation_effective = np.full_like(precipitation, nodata, dtype=np.float32)
        precipitation_effective[model_valid_mask] = (
            0.95 * precipitation[model_valid_mask]
        ).astype(np.float32)

        # ------------------------------------------------------------------ #
        # 6. Potential ET  (masked)                                            #
        # ------------------------------------------------------------------ #
        potential_evapotranspiration = compute_potential_evapotranspiration(
            et0=et0,
            kc_pixel=kc_pixel,
        )
        potential_evapotranspiration[~model_valid_mask] = nodata

        if strict_checks:
            assert_reasonable_range(
                "potential_evapotranspiration", potential_evapotranspiration,
                0.0, 30.0, nodata=nodata, date=current_date, raise_error=False,
            )

        # ------------------------------------------------------------------ #
        # 7. Green-water stress and WATNEEDS-like ET decomposition             #
        # ------------------------------------------------------------------ #
        green_water_stress_coefficient = compute_green_water_stress_coefficient(
            soil_moisture_green=soil_moisture_green,
            raw=raw,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        green_evapotranspiration_watneeds = compute_actual_evapotranspiration(
            potential_evapotranspiration=potential_evapotranspiration,
            water_stress_coefficient=green_water_stress_coefficient,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )
        green_evapotranspiration_watneeds[~model_valid_mask] = nodata

        blue_iwr_watneeds = compute_blue_water_requirement_watneeds(
            potential_evapotranspiration=potential_evapotranspiration,
            green_evapotranspiration=green_evapotranspiration_watneeds,
            irrigated_pixels=iwr_domain_pixels,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        actual_evapotranspiration_for_balance = np.full_like(
            potential_evapotranspiration,
            nodata,
            dtype=np.float32,
        )
        actual_evapotranspiration_for_balance[model_valid_mask] = (
            green_evapotranspiration_watneeds[model_valid_mask]
        )

        if iwr_mode == IWR_MODE_WATNEEDS_BLUE_ET:
            irrigated_valid = model_valid_mask & iwr_domain_pixels
            actual_evapotranspiration_for_balance[irrigated_valid] = (
                potential_evapotranspiration[irrigated_valid]
            )
            actual_irrigation_pixels = iwr_domain_pixels
        else:
            actual_irrigation_pixels = np.zeros_like(iwr_domain_pixels, dtype=bool)

        # ------------------------------------------------------------------ #
        # 8. Deep percolation                                                  #
        # ------------------------------------------------------------------ #
        deep_percolation = compute_deep_percolation(
            soil_moisture_previous=soil_moisture_actual,
            raw=raw,
            total_available_water_pixel=total_available_water_pixel,
            fmax_pixel=fmax_pixel,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        actual_evapotranspiration_for_balance, deep_percolation = scale_fluxes_if_water_deficit(
            soil_moisture_previous=soil_moisture_actual,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration_for_balance,
            deep_percolation=deep_percolation,
            irrigated_pixels=actual_irrigation_pixels,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        valid_dp = deep_percolation[model_valid_mask]
        if valid_dp.size > 0 and float(np.max(valid_dp)) > 100.0:
            dp_display = np.where(model_valid_mask, deep_percolation, 0.0)
            r_dp, c_dp = np.unravel_index(np.argmax(dp_display), dp_display.shape)
            print(
                f"\n[DEEP PERC TRACEBACK] Suspicious deep percolation on {date_str}: "
                f"max={dp_display[r_dp, c_dp]:.2f} mm/day at row={r_dp}, col={c_dp}"
            )
            for label, val in [
                ("soil_moisture_actual_previous", soil_moisture_actual[r_dp, c_dp]),
                ("raw", raw[r_dp, c_dp]),
                ("total_available_water_pixel", total_available_water_pixel[r_dp, c_dp]),
                ("fmax_pixel", fmax_pixel[r_dp, c_dp]),
                ("deep_percolation", deep_percolation[r_dp, c_dp]),
                ("precipitation_effective", precipitation_effective[r_dp, c_dp]),
                ("actual_evapotranspiration_for_balance", actual_evapotranspiration_for_balance[r_dp, c_dp]),
                ("actual_irrigation_pixels", actual_irrigation_pixels[r_dp, c_dp]),
            ]:
                print(f"  {label:<30} = {val}")

        if strict_checks:
            assert_reasonable_range(
                "actual_evapotranspiration_for_balance", actual_evapotranspiration_for_balance,
                0.0, 30.0, nodata=nodata, date=current_date, raise_error=False,
            )
            assert_reasonable_range(
                "deep_percolation", deep_percolation,
                0.0, 5000.0, nodata=nodata, date=current_date, raise_error=False,
            )

        if iwr_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION:
            actual_irrigation_input = np.full_like(
                potential_evapotranspiration,
                nodata,
                dtype=np.float32,
            )
            actual_irrigation_input[model_valid_mask] = 0.0
        else:
            actual_irrigation_input = blue_iwr_watneeds

        # ------------------------------------------------------------------ #
        # 9. Irrigation output variable                                        #
        # ------------------------------------------------------------------ #
        if iwr_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION:
            reference_valid_mask = (
                forcing_valid_mask
                & valid_area_pixels
                & (crop_fraction_sum > 0)
                & np.isfinite(soil_moisture_reference)
                & np.isfinite(total_available_water_pixel)
                & np.isfinite(raw)
                & np.isfinite(fmax_pixel)
                & (soil_moisture_reference != nodata)
                & (total_available_water_pixel != nodata)
                & (raw != nodata)
                & (fmax_pixel != nodata)
            )

            reference_deep_percolation = compute_deep_percolation(
                soil_moisture_previous=soil_moisture_reference,
                raw=raw,
                total_available_water_pixel=total_available_water_pixel,
                fmax_pixel=fmax_pixel,
                nodata=nodata,
                valid_mask=reference_valid_mask,
            )

            reference_runoff = compute_subsurface_runoff(
                soil_moisture_previous=soil_moisture_reference,
                precipitation_effective=precipitation_effective,
                actual_evapotranspiration=potential_evapotranspiration,
                deep_percolation=reference_deep_percolation,
                total_available_water_pixel=total_available_water_pixel,
                nodata=nodata,
                valid_mask=reference_valid_mask,
            )

            theoretical_demand_mask = (
                reference_valid_mask
                & iwr_domain_pixels
                & (kc_pixel > 0)
            )

            irrigation, soil_moisture_reference, reference_pre_irrigation_storage = (
                compute_theoretical_net_irrigation_requirement(
                    soil_moisture_previous=soil_moisture_reference,
                    precipitation_effective=precipitation_effective,
                    potential_evapotranspiration=potential_evapotranspiration,
                    deep_percolation=reference_deep_percolation,
                    runoff=reference_runoff,
                    target_storage=theoretical_target_storage,
                    total_available_water_pixel=total_available_water_pixel,
                    demand_mask=theoretical_demand_mask,
                    nodata=nodata,
                    valid_mask=reference_valid_mask,
                )
            )
        else:
            irrigation = blue_iwr_watneeds
            reference_deep_percolation = np.full_like(irrigation, nodata, dtype=np.float32)
            reference_runoff = np.full_like(irrigation, nodata, dtype=np.float32)
            reference_pre_irrigation_storage = np.full_like(irrigation, nodata, dtype=np.float32)

        # ------------------------------------------------------------------ #
        # 10. Debug traceback for suspicious IWR                               #
        # ------------------------------------------------------------------ #
        valid_irrigated_mask = model_valid_mask & iwr_domain_pixels
        valid_irrigation_vals = irrigation[valid_irrigated_mask]
        if valid_irrigation_vals.size > 0 and float(np.max(valid_irrigation_vals)) > max_iwr_mm_day:
            phenology_status = create_phenology_status_mask_from_date(
                current_date=current_date,
                phenology=phenology,
                nodata=nodata,
            )

            irr_display = np.where(valid_irrigated_mask, irrigation, 0.0)
            r, c = np.unravel_index(np.argmax(irr_display), irr_display.shape)
            print(
                f"\n[IWR TRACEBACK] Suspicious IWR on {date_str}: "
                f"max={irr_display[r, c]:.2f} mm/day at row={r}, col={c}"
            )
            for label, val in [
                ("precipitation",                   precipitation[r, c]),
                ("et0",                             et0[r, c]),
                ("crop_fraction_sum",               crop_fraction_sum[r, c]),
                ("kc_pixel",                        kc_pixel[r, c]),
                ("precipitation_effective",         precipitation_effective[r, c]),
                ("potential_ET",                    potential_evapotranspiration[r, c]),
                ("green_water_stress_coeff",        green_water_stress_coefficient[r, c]),
                ("green_ET",                        green_evapotranspiration_watneeds[r, c]),
                ("actual_ET_for_balance",           actual_evapotranspiration_for_balance[r, c]),
                ("deep_percolation",                deep_percolation[r, c]),
                ("blue_iwr_watneeds",               blue_iwr_watneeds[r, c]),
                ("soil_moisture_actual (prev)",     soil_moisture_actual[r, c]),
                ("soil_moisture_green (prev)",      soil_moisture_green[r, c]),
                ("soil_moisture_reference (prev)",  soil_moisture_reference[r, c]),
                ("total_available_water",           total_available_water_pixel[r, c]),
                ("raw",                             raw[r, c]),
                ("fmax_pixel",                      fmax_pixel[r, c]),
                ("iwr_domain_pixels",               iwr_domain_pixels[r, c]),
                ("actual_irrigation_input",         actual_irrigation_input[r, c]),
                ("reference_pre_irrigation_storage", reference_pre_irrigation_storage[r, c]),
                ("reference_deep_percolation",      reference_deep_percolation[r, c]),
                ("reference_runoff",                reference_runoff[r, c]),
                ("phenology_status",                phenology_status[r, c]),
            ]:
                print(f"  {label:<30} = {val}")

        if strict_checks:
            assert_reasonable_range(
                "irrigation", irrigation, 0.0, max_iwr_mm_day,
                nodata=nodata, date=current_date,
            )

        # ------------------------------------------------------------------ #
        # 11. Runoff and soil moisture update                                  #
        # ------------------------------------------------------------------ #
        runoff_balance_before_threshold = None
        runoff_excess_before_threshold = None

        if debug_mode:
            runoff_balance_before_threshold = np.full_like(
                soil_moisture_actual,
                nodata,
                dtype=np.float32,
            )
            runoff_excess_before_threshold = np.full_like(
                soil_moisture_actual,
                nodata,
                dtype=np.float32,
            )

            runoff_balance_mask = (
                model_valid_mask
                & np.isfinite(soil_moisture_actual)
                & np.isfinite(precipitation_effective)
                & np.isfinite(actual_evapotranspiration_for_balance)
                & np.isfinite(deep_percolation)
                & np.isfinite(total_available_water_pixel)
                & (soil_moisture_actual != nodata)
                & (precipitation_effective != nodata)
                & (actual_evapotranspiration_for_balance != nodata)
                & (deep_percolation != nodata)
                & (total_available_water_pixel != nodata)
            )

            runoff_balance_before_threshold[runoff_balance_mask] = (
                soil_moisture_actual[runoff_balance_mask]
                + precipitation_effective[runoff_balance_mask]
                - actual_evapotranspiration_for_balance[runoff_balance_mask]
                - deep_percolation[runoff_balance_mask]
            ).astype(np.float32)

            runoff_excess_before_threshold[runoff_balance_mask] = (
                runoff_balance_before_threshold[runoff_balance_mask]
                - total_available_water_pixel[runoff_balance_mask]
            ).astype(np.float32)

        runoff = compute_subsurface_runoff(
            soil_moisture_previous=soil_moisture_actual,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration_for_balance,
            deep_percolation=deep_percolation,
            total_available_water_pixel=total_available_water_pixel,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        # WATNEEDS-style green/blue ET outputs.
        # Green ET is supplied by rainfall-derived soil moisture (stress-limited).
        # Blue water requirement (IWR) is the gap between potential ETc and green ET.
        green_et = green_evapotranspiration_watneeds
        blue_et = blue_iwr_watneeds

        # In the WATNEEDS approach, blue water is consumed as ET on the same day
        # (no residual blue storage). All percolation and runoff comes from green water.
        green_deep_percolation = deep_percolation
        blue_deep_percolation = np.zeros_like(deep_percolation, dtype=np.float32)
        blue_deep_percolation[deep_percolation == nodata] = nodata
        green_runoff = runoff
        blue_runoff = np.zeros_like(runoff, dtype=np.float32)
        blue_runoff[runoff == nodata] = nodata

        soil_moisture_actual = water_balance_step(
            soil_moisture_previous=soil_moisture_actual,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration_for_balance,
            deep_percolation=deep_percolation,
            runoff=runoff,
            irrigation=actual_irrigation_input,
            total_available_water_pixel=total_available_water_pixel,
            delta_t=1.0,
            nodata=nodata,
            valid_mask=model_valid_mask,
        )

        # The actual-state storage follows the realized water balance.
        # In theoretical mode, it remains precipitation-only (no irrigation input).
        soil_moisture_green = soil_moisture_actual.copy()
        soil_moisture_blue = np.zeros_like(soil_moisture_actual, dtype=np.float32)
        soil_moisture_blue[soil_moisture_actual == nodata] = nodata

        soil_saturation = np.full_like(soil_moisture_actual, nodata, dtype=np.float32)

        saturation_mask = (
            model_valid_mask
            & np.isfinite(soil_moisture_actual)
            & np.isfinite(total_available_water_pixel)
            & (soil_moisture_actual != nodata)
            & (total_available_water_pixel != nodata)
            & (total_available_water_pixel > 0)
        )

        soil_saturation[saturation_mask] = (
            soil_moisture_actual[saturation_mask]
            / total_available_water_pixel[saturation_mask]
        ).astype(np.float32)

        soil_saturation[saturation_mask] = np.clip(
            soil_saturation[saturation_mask],
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------ #
        # 12. Write daily outputs                                              #
        # ------------------------------------------------------------------ #
        active_pixel_mask = (
            (kc_pixel > 0)
            & (crop_fraction_sum > 0)
            & iwr_domain_pixels
            & valid_area_pixels
            & (total_available_water_pixel != nodata)
        )

        irrigation_to_write = np.full_like(irrigation, nodata, dtype=np.float32)
        irrigation_to_write[active_pixel_mask] = irrigation[active_pixel_mask]

        if output_folder is not None and output_profile is not None:
            output_file = Path(output_folder) / f"iwr_{current_date.strftime('%Y%m%d')}.tif"

            if iwr_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION:
                daily_metadata = {
                    "variable": "daily_theoretical_net_irrigation_requirement",
                    "units": "mm/day over cropped area",
                    "iwr_mode": iwr_mode,
                    "iwr_domain": iwr_domain,
                    "actual_water_input": "effective_precipitation_only",
                    "target_storage": theoretical_iwr_target,
                }
            else:
                daily_metadata = {
                    "variable": "daily_blue_water_requirement",
                    "units": "mm/day over cropped/irrigated crop area",
                    "iwr_mode": iwr_mode,
                    "iwr_domain": iwr_domain,
                }

            write_daily_geotiff(
                output_path=output_file,
                data=irrigation_to_write,
                profile=output_profile,
                nodata=nodata,
                date=current_date,
                metadata=daily_metadata,
            )

        if active_pixel_masks_folder is not None and output_profile is not None:
            active_pixel_data = active_pixel_mask.astype(np.uint8)
            mask_file = (
                active_pixel_masks_folder
                / f"active_pixels_{current_date.strftime('%Y%m%d')}.tif"
            )
            write_active_pixel_mask_geotiff(
                output_path=mask_file,
                data=active_pixel_data,
                profile=output_profile,
                date=current_date,
                iwr_domain=iwr_domain,
            )

        if debug_mode and output_profile is not None:
            date_token = current_date.strftime("%Y%m%d")

            if debug_output_folder is None:
                if output_folder is None:
                    raise ValueError(
                        "debug_mode=True requires either debug_output_folder or output_folder."
                    )

                output_folder_path = Path(output_folder)

                debug_base_folder = (
                    output_folder_path.parent
                    / f"{output_folder_path.name}_debug"
                )
            else:
                debug_base_folder = Path(debug_output_folder)

            debug_outputs = [
                ("kc_pixel", kc_pixel, "kc_pixel", "dimensionless"),
                (
                    "actual_soil_moisture",
                    soil_moisture_actual,
                    "actual_soil_moisture",
                    "mm",
                ),
                (
                    "actual_irrigation_input",
                    actual_irrigation_input,
                    "actual_irrigation_input",
                    "mm/day",
                ),
                (
                    "actual_evapotranspiration_for_balance",
                    actual_evapotranspiration_for_balance,
                    "actual_evapotranspiration_for_balance",
                    "mm/day",
                ),
                (
                    "actual_deep_percolation",
                    deep_percolation,
                    "actual_deep_percolation",
                    "mm/day",
                ),
                ("actual_runoff", runoff, "actual_runoff", "mm/day"),
                ("reported_iwr", irrigation, "reported_iwr", "mm/day"),
                (
                    "reference_soil_moisture",
                    soil_moisture_reference,
                    "reference_soil_moisture",
                    "mm",
                ),
                (
                    "reference_pre_irrigation_storage",
                    reference_pre_irrigation_storage,
                    "reference_pre_irrigation_storage",
                    "mm",
                ),
                (
                    "reference_deep_percolation",
                    reference_deep_percolation,
                    "reference_deep_percolation",
                    "mm/day",
                ),
                ("reference_runoff", reference_runoff, "reference_runoff", "mm/day"),
                (
                    "theoretical_target_storage",
                    theoretical_target_storage,
                    "theoretical_target_storage",
                    "mm",
                ),
                (
                    "runoff_balance_before_threshold",
                    runoff_balance_before_threshold,
                    "runoff_balance_before_threshold",
                    "mm",
                ),
                (
                    "runoff_excess_before_threshold",
                    runoff_excess_before_threshold,
                    "runoff_excess_before_threshold",
                    "mm",
                ),
                ("soil_saturation", soil_saturation, "soil_saturation", "fraction"),
                ("raw_fraction_of_taw", raw_fraction_of_taw, "raw_fraction_of_taw", "fraction"),
            ]

            for folder_name, data_array, variable_name, units in debug_outputs:
                debug_path = (
                    debug_base_folder
                    / folder_name
                    / f"{variable_name}_{date_token}.tif"
                )

                write_daily_geotiff(
                    output_path=debug_path,
                    data=data_array,
                    profile=output_profile,
                    nodata=nodata,
                    date=current_date,
                    metadata={
                        "variable": variable_name,
                        "units": units,
                        "debug_mode": "true",
                        "iwr_mode": iwr_mode,
                        "iwr_domain": iwr_domain,
                    },
                )

        if output_folder is not None and output_profile is not None and write_daily_green_blue_outputs:
            date_token = current_date.strftime('%Y%m%d')
            write_daily_geotiff(
                output_path=Path(output_folder) / f"green_et_{date_token}.tif",
                data=green_et,
                profile=output_profile,
                nodata=nodata,
                date=current_date,
                metadata={"variable": "green_evapotranspiration", "units": "mm/day"},
            )
            write_daily_geotiff(
                output_path=Path(output_folder) / f"blue_et_{date_token}.tif",
                data=blue_et,
                profile=output_profile,
                nodata=nodata,
                date=current_date,
                metadata={"variable": "blue_evapotranspiration", "units": "mm/day"},
            )
            write_daily_geotiff(
                output_path=Path(output_folder) / f"green_storage_{date_token}.tif",
                data=soil_moisture_green,
                profile=output_profile,
                nodata=nodata,
                date=current_date,
                metadata={"variable": "green_storage", "units": "mm"},
            )
            write_daily_geotiff(
                output_path=Path(output_folder) / f"blue_storage_{date_token}.tif",
                data=soil_moisture_blue,
                profile=output_profile,
                nodata=nodata,
                date=current_date,
                metadata={"variable": "blue_storage", "units": "mm"},
            )

        if strict_checks:
            assert_reasonable_range(
                "soil_moisture", soil_moisture_actual, 0.0, taw_max,
                nodata=nodata, date=current_date, raise_error=False,
            )

        # Accumulate irrigation only where daily IWR output is active.
        cumulative_irrigation += np.where(
            active_pixel_mask & (irrigation != nodata), irrigation, 0.0
        ).astype(np.float32)
        cumulative_green_et += np.where(
            model_valid_mask & (green_et != nodata), green_et, 0.0
        ).astype(np.float32)
        cumulative_blue_et += np.where(
            model_valid_mask & (blue_et != nodata), blue_et, 0.0
        ).astype(np.float32)
        cumulative_green_deep_percolation += np.where(
            model_valid_mask & (green_deep_percolation != nodata), green_deep_percolation, 0.0
        ).astype(np.float32)
        cumulative_blue_deep_percolation += np.where(
            model_valid_mask & (blue_deep_percolation != nodata), blue_deep_percolation, 0.0
        ).astype(np.float32)
        cumulative_green_runoff += np.where(
            model_valid_mask & (green_runoff != nodata), green_runoff, 0.0
        ).astype(np.float32)
        cumulative_blue_runoff += np.where(
            model_valid_mask & (blue_runoff != nodata), blue_runoff, 0.0
        ).astype(np.float32)

        # ------------------------------------------------------------------ #
        # 13. Collect daily diagnostics                                        #
        # ------------------------------------------------------------------ #
        if write_debug_csv:
            if day_index % debug_csv_frequency_days == 0:
                forcing_valid_pct = 100.0 * float(np.sum(forcing_valid_mask)) / forcing_valid_mask.size
                model_valid_pct   = 100.0 * float(np.sum(model_valid_mask))   / model_valid_mask.size
                row = {
                    "date":               date_str,
                    "forcing_valid_pct":  round(forcing_valid_pct, 2),
                    "model_valid_pct":    round(model_valid_pct, 2),
                    "iwr_mode":           iwr_mode,
                    "iwr_domain":         iwr_domain,
                }
                for var_name, var_arr in [
                    ("precipitation",                         precipitation),
                    ("et0",                                   et0),
                    ("crop_fraction_sum",                     crop_fraction_sum),
                    ("kc_pixel",                              kc_pixel),
                    ("precipitation_effective",               precipitation_effective),
                    ("potential_evapotranspiration",          potential_evapotranspiration),
                    ("green_water_stress_coefficient",        green_water_stress_coefficient),
                    ("green_evapotranspiration_watneeds",     green_evapotranspiration_watneeds),
                    ("blue_iwr_watneeds",                     blue_iwr_watneeds),
                    ("actual_evapotranspiration_for_balance", actual_evapotranspiration_for_balance),
                    ("green_et",                              green_et),
                    ("blue_et",                               blue_et),
                    ("actual_irrigation_input",               actual_irrigation_input),
                    ("actual_deep_percolation",               deep_percolation),
                    ("reported_iwr",                          irrigation),
                    ("actual_soil_moisture",                  soil_moisture_actual),
                    ("actual_runoff",                         runoff),
                    ("reference_soil_moisture",               soil_moisture_reference),
                    ("reference_pre_irrigation_storage",      reference_pre_irrigation_storage),
                    ("reference_deep_percolation",            reference_deep_percolation),
                    ("reference_runoff",                      reference_runoff),
                    ("cumulative_irrigation",                 cumulative_irrigation),
                ]:
                    stats = array_stats(var_arr, nodata=nodata)
                    for stat_key in ("min", "p50", "p95", "p99", "max"):
                        v = stats[stat_key]
                        row[f"{var_name}_{stat_key}"] = round(v, 4) if v == v else "nan"
                daily_stats_rows.append(row)

        print(f"Processed: {date_str}")

        day_index += 1
        current_date = current_date + timedelta(days=1)

    # ---------------------------------------------------------------------- #
    # End-of-run outputs                                                       #
    # ---------------------------------------------------------------------- #

    if write_debug_csv and output_folder is not None and daily_stats_rows:
        csv_path = Path(output_folder) / "iwr_debug_daily_stats.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(daily_stats_rows[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(daily_stats_rows)
        print(f"Debug CSV written: {csv_path}")

    if write_cumulative_iwr and output_folder is not None and output_profile is not None:
        cumulative_path = Path(output_folder) / "iwr_cumulative_total.tif"

        if iwr_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION:
            cumulative_metadata = {
                "variable": "cumulative_theoretical_net_irrigation_requirement",
                "units": "mm over cropped area",
                "iwr_mode": iwr_mode,
                "iwr_domain": iwr_domain,
                "target_storage": theoretical_iwr_target,
            }
        else:
            cumulative_metadata = {
                "variable": "cumulative_blue_water_requirement",
                "units": "mm over cropped/irrigated crop area",
                "iwr_mode": iwr_mode,
                "iwr_domain": iwr_domain,
            }

        write_daily_geotiff(
            output_path=cumulative_path,
            data=np.where(cumulative_static_mask, cumulative_irrigation, nodata).astype(np.float32),
            profile=output_profile,
            nodata=nodata,
            metadata=cumulative_metadata,
        )
        print(f"Cumulative IWR written: {cumulative_path}")

    if write_green_blue_outputs and output_folder is not None and output_profile is not None:
        cumulative_outputs = [
            (
                "green_et_cumulative_total.tif",
                cumulative_green_et,
                "green_evapotranspiration_cumulative",
                "mm",
            ),
            (
                "blue_et_cumulative_total.tif",
                cumulative_blue_et,
                "blue_evapotranspiration_cumulative",
                "mm",
            ),
            (
                "green_deep_percolation_cumulative_total.tif",
                cumulative_green_deep_percolation,
                "green_deep_percolation_cumulative",
                "mm",
            ),
            (
                "blue_deep_percolation_cumulative_total.tif",
                cumulative_blue_deep_percolation,
                "blue_deep_percolation_cumulative",
                "mm",
            ),
            (
                "green_runoff_cumulative_total.tif",
                cumulative_green_runoff,
                "green_runoff_cumulative",
                "mm",
            ),
            (
                "blue_runoff_cumulative_total.tif",
                cumulative_blue_runoff,
                "blue_runoff_cumulative",
                "mm",
            ),
        ]

        for out_name, out_data, variable, units in cumulative_outputs:
            out_path = Path(output_folder) / out_name
            write_daily_geotiff(
                output_path=out_path,
                data=np.where(static_valid_mask, out_data, nodata).astype(np.float32),
                profile=output_profile,
                nodata=nodata,
                metadata={"variable": variable, "units": units},
            )
            print(f"Cumulative output written: {out_path}")

    return soil_moisture_actual, cumulative_irrigation
