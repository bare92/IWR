from datetime import timedelta
from pathlib import Path

import numpy as np
import rasterio

from utilities import read_forcing_day
from phenology_functions import (
    create_phenology_status_mask_from_date,
    PHENOLOGY_INACTIVE,
    PHENOLOGY_GROWING,
    PHENOLOGY_MAXIMUM,
    PHENOLOGY_SENESCENCE,
)


def prepare_crop_fractions(crop_fraction_data):
    """
    Prepare crop fractions.

    Input shape:
        crop, rows, cols

    Output:
        cleaned crop fractions with values >= 0
        crop_fraction_sum with shape rows, cols

    If crop fractions sum above 1 because of small inconsistencies,
    they are normalized back to 1.
    """

    crop_fraction_data = crop_fraction_data.astype(np.float32)

    crop_fraction_data = np.where(
        np.isfinite(crop_fraction_data) & (crop_fraction_data > 0),
        crop_fraction_data,
        0.0,
    )

    crop_fraction_sum = np.sum(crop_fraction_data, axis=0)

    scale = np.ones_like(crop_fraction_sum, dtype=np.float32)
    scale[crop_fraction_sum > 1.0] = 1.0 / crop_fraction_sum[crop_fraction_sum > 1.0]

    crop_fraction_data = crop_fraction_data * scale[np.newaxis, :, :]
    crop_fraction_sum = np.sum(crop_fraction_data, axis=0)

    return crop_fraction_data, crop_fraction_sum


def create_effective_root_depth(
    crop_fraction_data,
    crop_df,
):
    """
    Create one effective root depth per pixel.

    root_depth_eff = sum(crop_fraction_i * root_depth_i)

    Output unit:
        m
    """

    root_depths = crop_df["root_depth_max_m"].to_numpy(dtype=np.float32)

    effective_root_depth = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    for crop_index, root_depth in enumerate(root_depths):
        effective_root_depth += crop_fraction_data[crop_index, :, :] * root_depth

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
    Create one RAW value per pixel.

    total_available_water input is:
        FC - WP, in m3/m3

    For multiple crops in one pixel:
        RAW_mm = (FC - WP) * 1000 * sum(crop_fraction_i * root_depth_i * p_i)

    Output unit:
        mm
    """

    root_depths = crop_df["root_depth_max_m"].to_numpy(dtype=np.float32)
    depletion_factors = crop_df["p"].to_numpy(dtype=np.float32)

    weighted_root_depth_p = np.zeros(
        crop_fraction_data.shape[1:],
        dtype=np.float32,
    )

    for crop_index, (root_depth, p) in enumerate(zip(root_depths, depletion_factors)):
        weighted_root_depth_p += (
            crop_fraction_data[crop_index, :, :]
            * root_depth
            * p
        )

    raw_pixel = (
        total_available_water
        * weighted_root_depth_p
        * 1000.0
    ).astype(np.float32)

    raw_pixel[total_available_water == nodata] = nodata

    return raw_pixel


def create_fmax_pixel(
    fmax,
    crop_fraction_sum,
    nodata=-9999.0,
):
    """
    Create one Fmax value per pixel.

    Since the balance is crop-fraction weighted, Fmax is also weighted
    by the total crop fraction in the pixel.

    Output unit:
        mm/day
    """

    fmax_pixel = (fmax * crop_fraction_sum).astype(np.float32)
    fmax_pixel[fmax == nodata] = nodata

    return fmax_pixel


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
    phenology_status,
    crop_fraction_data,
    crop_df,
):
    """
    Create one crop coefficient per pixel.

    Kc_pixel = sum(crop_fraction_i * Kc_i)

    Kc rules:
        inactive    -> 0
        growing     -> Kc_ini
        maximum     -> Kc_mid
        senescence  -> Kc_end
    """

    kc_pixel = np.zeros(phenology_status.shape, dtype=np.float32)

    for crop_index in range(crop_fraction_data.shape[0]):
        crop_fraction = crop_fraction_data[crop_index, :, :]

        kc_ini = crop_df.iloc[crop_index]["Kc_ini"]
        kc_mid = crop_df.iloc[crop_index]["Kc_mid"]
        kc_end = crop_df.iloc[crop_index]["Kc_end"]

        kc_crop = np.zeros(phenology_status.shape, dtype=np.float32)

        kc_crop[phenology_status == PHENOLOGY_GROWING] = kc_ini
        kc_crop[phenology_status == PHENOLOGY_MAXIMUM] = kc_mid
        kc_crop[phenology_status == PHENOLOGY_SENESCENCE] = kc_end
        kc_crop[phenology_status == PHENOLOGY_INACTIVE] = 0.0

        kc_pixel += crop_fraction * kc_crop

    return kc_pixel


def compute_deep_percolation(
    soil_moisture_previous,
    raw,
    total_available_water_pixel,
    fmax_pixel,
    nodata=-9999.0,
):
    """
    Compute deep percolation D.

    D = Fmax * (S - RAW) / (TAW - RAW), if RAW <= S <= TAW
    D = 0, if S < RAW
    """

    deep_percolation = np.zeros_like(soil_moisture_previous, dtype=np.float32)

    valid_mask = (
        (total_available_water_pixel != nodata)
        & (fmax_pixel != nodata)
        & (total_available_water_pixel > raw)
    )

    percolation_mask = (
        valid_mask
        & (soil_moisture_previous >= raw)
        & (soil_moisture_previous <= total_available_water_pixel)
    )

    deep_percolation[percolation_mask] = (
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

    deep_percolation[total_available_water_pixel == nodata] = nodata

    return deep_percolation


def scale_fluxes_if_water_deficit(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    irrigated_pixels,
    nodata=-9999.0,
):
    """
    If S_prev + Peff - ETa - D is negative,
    scale ETa and D proportionally to close the balance.

    This is applied only to non-irrigated pixels.
    Irrigated pixels receive irrigation instead.
    """

    available_water = soil_moisture_previous + precipitation_effective
    outgoing_water = actual_evapotranspiration + deep_percolation

    deficit_mask = (
        (available_water < outgoing_water)
        & (outgoing_water > 0)
        & (~irrigated_pixels)
        & (soil_moisture_previous != nodata)
    )

    scale_factor = np.ones_like(soil_moisture_previous, dtype=np.float32)

    scale_factor[deficit_mask] = (
        available_water[deficit_mask]
        / outgoing_water[deficit_mask]
    )

    actual_evapotranspiration = actual_evapotranspiration * scale_factor
    deep_percolation = deep_percolation * scale_factor

    actual_evapotranspiration[soil_moisture_previous == nodata] = nodata
    deep_percolation[soil_moisture_previous == nodata] = nodata

    return actual_evapotranspiration, deep_percolation


def compute_irrigation(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    irrigated_pixels,
    nodata=-9999.0,
):
    """
    Compute one irrigation value per pixel.

    I is the water needed to avoid a negative balance.

    Applied only where irrigation_mask == 1.
    """

    balance_without_irrigation = (
        soil_moisture_previous
        + precipitation_effective
        - actual_evapotranspiration
        - deep_percolation
    )

    irrigation = np.maximum(-balance_without_irrigation, 0.0)

    irrigation[~irrigated_pixels] = 0.0
    irrigation[soil_moisture_previous == nodata] = nodata

    return irrigation.astype(np.float32)


def compute_subsurface_runoff(
    soil_moisture_previous,
    precipitation_effective,
    actual_evapotranspiration,
    deep_percolation,
    irrigation,
    total_available_water_pixel,
    nodata=-9999.0,
):
    """
    Compute one runoff value per pixel.

    If:
        S_prev + Peff - ETa - D + I > TAW

    then:
        R = balance - TAW
    """

    balance_before_runoff = (
        soil_moisture_previous
        + precipitation_effective
        - actual_evapotranspiration
        - deep_percolation
        + irrigation
    )

    runoff = np.maximum(
        balance_before_runoff - total_available_water_pixel,
        0.0,
    )

    runoff[total_available_water_pixel == nodata] = nodata

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
):
    """
    One daily water balance step.

    S_t = S_t-1 + delta_t * (P_eff - ETa - D - R + I)
    """

    soil_moisture = (
        soil_moisture_previous
        + delta_t
        * (
            precipitation_effective
            - actual_evapotranspiration
            - deep_percolation
            - runoff
            + irrigation
        )
    )

    soil_moisture = np.maximum(soil_moisture, 0.0)
    soil_moisture = np.minimum(soil_moisture, total_available_water_pixel)

    soil_moisture[total_available_water_pixel == nodata] = nodata

    return soil_moisture.astype(np.float32)


def write_daily_geotiff(output_path, data, profile, nodata=-9999.0):
    """
    Write one daily output GeoTIFF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = profile.copy()
    output_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw",
    )

    data_to_write = data.astype("float32")

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(data_to_write, 1)


def compute_water_stress_coefficient(
    soil_moisture,
    raw,
    irrigated_pixels,
    nodata=-9999.0,
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

    ks = np.ones_like(soil_moisture, dtype=np.float32)

    valid_mask = (
        (soil_moisture != nodata)
        & (raw > 0)
    )

    stressed_mask = (
        valid_mask
        & (~irrigated_pixels)
        & (soil_moisture < raw)
    )

    ks[stressed_mask] = soil_moisture[stressed_mask] / raw[stressed_mask]

    ks = np.clip(ks, 0.0, 1.0)
    ks[soil_moisture == nodata] = nodata

    return ks


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
):
    """
    Compute actual evapotranspiration.

    ETa = Ks * ET
    """

    actual_evapotranspiration = (
        water_stress_coefficient
        * potential_evapotranspiration
    )

    actual_evapotranspiration[water_stress_coefficient == nodata] = nodata

    return actual_evapotranspiration.astype(np.float32)


def run_iwr_model(
    start_date,
    end_date,
    total_available_water,
    fmax,
    irrigation_mask,
    crop_fraction_data,
    crop_df,
    phenology,
    precipitation_dataset,
    precipitation_variable,
    et0_dataset,
    et0_variable,
    output_folder=None,
    output_profile=None,
    nodata=-9999.0,
):
    """
    Run daily IWR water balance with one S and one I per pixel.

    Current implementation:
    - crop layers are aggregated before the balance
    - P_eff = 0.95 * P * total crop fraction
    - Kc_pixel = sum(crop_fraction_i * Kc_i)
    - ETa = Ks * Kc * PET (with water stress Ks)
    - D is computed from RAW, TAW and Fmax
    - I is computed only where irrigation_mask == 1
    - R is computed when storage exceeds TAW
    - Daily irrigation maps are written as GeoTIFFs if output_folder is provided
    """

    crop_fraction_data, crop_fraction_sum = prepare_crop_fractions(
        crop_fraction_data=crop_fraction_data,
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

    fmax_pixel = create_fmax_pixel(
        fmax=fmax,
        crop_fraction_sum=crop_fraction_sum,
        nodata=nodata,
    )

    soil_moisture = initialize_soil_moisture(
        total_available_water_pixel=total_available_water_pixel,
        initial_fraction=0.5,
        nodata=nodata,
    )

    irrigated_pixels = irrigation_mask == 1

    cumulative_irrigation = np.zeros_like(soil_moisture, dtype=np.float32)

    current_date = start_date

    while current_date <= end_date:

        precipitation = read_forcing_day(
            dataset=precipitation_dataset,
            variable_name=precipitation_variable,
            date=current_date,
        )

        et0 = read_forcing_day(
            dataset=et0_dataset,
            variable_name=et0_variable,
            date=current_date,
        )

        phenology_status = create_phenology_status_mask_from_date(
            current_date=current_date,
            phenology=phenology,
            nodata=nodata,
        )

        kc_pixel = create_kc_pixel(
            phenology_status=phenology_status,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
        )

        precipitation_effective = (
            0.95
            * precipitation
            * crop_fraction_sum
        ).astype(np.float32)

        potential_evapotranspiration = compute_potential_evapotranspiration(
            et0=et0,
            kc_pixel=kc_pixel,
        )

        water_stress_coefficient = compute_water_stress_coefficient(
            soil_moisture=soil_moisture,
            raw=raw,
            irrigated_pixels=irrigated_pixels,
            nodata=nodata,
        )

        actual_evapotranspiration = compute_actual_evapotranspiration(
            potential_evapotranspiration=potential_evapotranspiration,
            water_stress_coefficient=water_stress_coefficient,
            nodata=nodata,
        )

        deep_percolation = compute_deep_percolation(
            soil_moisture_previous=soil_moisture,
            raw=raw,
            total_available_water_pixel=total_available_water_pixel,
            fmax_pixel=fmax_pixel,
            nodata=nodata,
        )

        actual_evapotranspiration, deep_percolation = scale_fluxes_if_water_deficit(
            soil_moisture_previous=soil_moisture,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration,
            deep_percolation=deep_percolation,
            irrigated_pixels=irrigated_pixels,
            nodata=nodata,
        )

        irrigation = compute_irrigation(
            soil_moisture_previous=soil_moisture,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration,
            deep_percolation=deep_percolation,
            irrigated_pixels=irrigated_pixels,
            nodata=nodata,
        )

        if output_folder is not None and output_profile is not None:
            output_file = (
                Path(output_folder)
                / f"iwr_{current_date.strftime('%Y%m%d')}.tif"
            )

            write_daily_geotiff(
                output_path=output_file,
                data=irrigation,
                profile=output_profile,
                nodata=nodata,
            )

        runoff = compute_subsurface_runoff(
            soil_moisture_previous=soil_moisture,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration,
            deep_percolation=deep_percolation,
            irrigation=irrigation,
            total_available_water_pixel=total_available_water_pixel,
            nodata=nodata,
        )

        soil_moisture = water_balance_step(
            soil_moisture_previous=soil_moisture,
            precipitation_effective=precipitation_effective,
            actual_evapotranspiration=actual_evapotranspiration,
            deep_percolation=deep_percolation,
            runoff=runoff,
            irrigation=irrigation,
            total_available_water_pixel=total_available_water_pixel,
            delta_t=1.0,
            nodata=nodata,
        )

        cumulative_irrigation += np.where(irrigation == nodata, 0.0, irrigation)

        print("Processed:", current_date.strftime("%Y-%m-%d"))

        current_date = current_date + timedelta(days=1)

    return soil_moisture, cumulative_irrigation
