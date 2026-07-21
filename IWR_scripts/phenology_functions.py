import calendar
from pathlib import Path

import numpy as np
import rasterio


PHENOLOGY_INACTIVE = 0
PHENOLOGY_GROWING = 1
PHENOLOGY_MAXIMUM = 2
PHENOLOGY_SENESCENCE = 3


def doy_to_dekad(doy):
    """
    Convert day of year to ASAP dekad from 1 to 36.

    Simple approximation:
    days 1-10   -> dekad 1
    days 11-20  -> dekad 2
    days 21-31  -> dekad 3
    etc.
    """

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    day_counter = 0

    for month_index, month_days in enumerate(month_lengths):
        month_start = day_counter + 1
        month_end = day_counter + month_days

        if month_start <= doy <= month_end:
            day_in_month = doy - day_counter

            if day_in_month <= 10:
                dekad_in_month = 1
            elif day_in_month <= 20:
                dekad_in_month = 2
            else:
                dekad_in_month = 3

            return month_index * 3 + dekad_in_month

        day_counter += month_days

    raise ValueError(f"Invalid day of year: {doy}")


def date_to_dekad(date):
    """
    Convert a datetime/date object to ASAP dekad from 1 to 36.
    """

    month = date.month
    day = date.day

    if day <= 10:
        dekad_in_month = 1
    elif day <= 20:
        dekad_in_month = 2
    else:
        dekad_in_month = 3

    return (month - 1) * 3 + dekad_in_month


def date_to_continuous_dekad(current_date):
    """
    Return the ASAP dekad position as a float with intra-dekad resolution.

    The first day of each dekad maps to the integer dekad value.
    Later days produce a fractional value between the current and next dekad.
    """
    month = current_date.month
    day = current_date.day
    days_in_month = calendar.monthrange(current_date.year, month)[1]

    if day <= 10:
        first_day = 1
        dekad_length = 10
    elif day <= 20:
        first_day = 11
        dekad_length = 10
    else:
        first_day = 21
        dekad_length = days_in_month - 20

    fraction = (day - first_day) / dekad_length
    return float(date_to_dekad(current_date) + fraction)


def load_phenology_layers(phenology_paths):
    """
    Load ASAP phenology rasters.

    Expected keys:
    phenos1, phenos2
    phenom1, phenom2
    phenosen1, phenosen2
    phenoe1, phenoe2
    phenonseasons
    """

    phenology = {}

    for name, path in phenology_paths.items():
        path = Path(path)

        with rasterio.open(path) as src:
            phenology[name] = src.read(1)

    return phenology


def _update_status_for_one_season(
    status,
    current_dekad,
    sos,
    tom,
    sen,
    eos,
    valid_mask,
):
    """
    Update phenological status for one growing season.

    ASAP phenology values are expressed on a 1-108 dekad scale.
    The current calendar dekad is tested as:
    d, d + 36, d + 72
    """

    current_dekad_candidates = [
        current_dekad,
        current_dekad + 36,
        current_dekad + 72,
    ]

    for current_v in current_dekad_candidates:

        active_mask = (
            valid_mask
            & (current_v >= sos)
            & (current_v <= eos)
        )

        growing_mask = (
            active_mask
            & (current_v >= sos)
            & (current_v < tom)
        )

        maximum_mask = (
            active_mask
            & (current_v >= tom)
            & (current_v < sen)
        )

        senescence_mask = (
            active_mask
            & (current_v >= sen)
            & (current_v <= eos)
        )

        status[growing_mask] = PHENOLOGY_GROWING
        status[maximum_mask] = PHENOLOGY_MAXIMUM
        status[senescence_mask] = PHENOLOGY_SENESCENCE

    return status


def create_phenology_status_mask(
    current_dekad,
    phenology,
    nodata=-9999,
):
    """
    Create a phenological status mask for the current dekad.

    Output values:
    0 = inactive / outside growing season
    1 = growing phase, from SOS to TOM
    2 = maximum phase, from TOM to SEN
    3 = senescence phase, from SEN to EOS
    """

    number_of_seasons = phenology["phenonseasons"]

    status = np.zeros(number_of_seasons.shape, dtype=np.uint8)

    valid_mask = number_of_seasons != nodata

    # Season 1
    season_1_mask = valid_mask & (number_of_seasons >= 1)

    status = _update_status_for_one_season(
        status=status,
        current_dekad=current_dekad,
        sos=phenology["phenos1"],
        tom=phenology["phenom1"],
        sen=phenology["phenosen1"],
        eos=phenology["phenoe1"],
        valid_mask=season_1_mask,
    )

    # Season 2
    season_2_mask = valid_mask & (number_of_seasons >= 2)

    status = _update_status_for_one_season(
        status=status,
        current_dekad=current_dekad,
        sos=phenology["phenos2"],
        tom=phenology["phenom2"],
        sen=phenology["phenosen2"],
        eos=phenology["phenoe2"],
        valid_mask=season_2_mask,
    )

    return status


def create_phenology_status_mask_from_date(
    current_date,
    phenology,
    nodata=-9999,
):
    """
    Same as create_phenology_status_mask(),
    but using a datetime/date object instead of dekad.
    """

    current_dekad = date_to_dekad(current_date)

    return create_phenology_status_mask(
        current_dekad=current_dekad,
        phenology=phenology,
        nodata=nodata,
    )


def _calculate_dynamic_kc_for_one_season(
    current_dekad_position,
    sos,
    tom,
    sen,
    eos,
    valid_mask,
    kc_ini,
    kc_mid,
    kc_end,
):
    """
    Compute a continuous FAO-56-style Kc curve for one growing season.

    Uses the extended 1-108 dekad calendar: position is tested at
    d, d+36 and d+72 to cover all three possible calendar placements.

    Returns
    -------
    kc : float32 array
    season_active : bool array  (True where this season is currently active)
    """
    sos = sos.astype(np.float32)
    tom = tom.astype(np.float32)
    sen = sen.astype(np.float32)
    eos = eos.astype(np.float32)

    kc = np.zeros(sos.shape, dtype=np.float32)
    season_active = np.zeros(sos.shape, dtype=bool)

    stage_valid = (
        valid_mask
        & (sos <= tom)
        & (tom <= sen)
        & (sen <= eos)
    )

    for offset in (0, 36, 72):
        pos = float(current_dekad_position) + offset

        active = stage_valid & (pos >= sos) & (pos < eos + 1)

        dev_mask = active & (pos >= sos) & (pos < tom)
        dev_len = np.maximum(tom - sos, 1e-6)
        dev_progress = np.clip((pos - sos) / dev_len, 0.0, 1.0)
        kc[dev_mask] = (kc_ini + dev_progress * (kc_mid - kc_ini)).astype(np.float32)[dev_mask]

        mid_mask = active & (pos >= tom) & (pos < sen)
        kc[mid_mask] = np.float32(kc_mid)

        late_mask = active & (pos >= sen) & (pos < eos + 1)
        late_len = np.maximum((eos + 1) - sen, 1e-6)
        late_progress = np.clip((pos - sen) / late_len, 0.0, 1.0)
        kc[late_mask] = (kc_mid + late_progress * (kc_end - kc_mid)).astype(np.float32)[late_mask]

        season_active |= active

    return kc.astype(np.float32), season_active


def create_dynamic_kc_curve_from_date(
    current_date,
    phenology,
    kc_ini,
    kc_mid,
    kc_end,
    nodata=-9999,
    inactive_kc=0.0,
):
    """
    Compute a continuous FAO-56-style Kc raster for current_date.

    Processes up to two growing seasons per pixel. Where two seasons
    overlap, the maximum active Kc is kept. Pixels outside all active
    seasons receive inactive_kc. Returns float32.
    """
    pos = date_to_continuous_dekad(current_date)
    nseasons = phenology["phenonseasons"].astype(np.float32)
    shape = nseasons.shape

    base_valid = np.isfinite(nseasons) & (nseasons != nodata)

    kc = np.full(shape, inactive_kc, dtype=np.float32)
    any_active = np.zeros(shape, dtype=bool)

    for season_num in (1, 2):
        suffix = str(season_num)
        season_mask = base_valid & (nseasons >= season_num)

        sos_arr = phenology[f"phenos{suffix}"].astype(np.float32)
        tom_arr = phenology[f"phenom{suffix}"].astype(np.float32)
        sen_arr = phenology[f"phenosen{suffix}"].astype(np.float32)
        eos_arr = phenology[f"phenoe{suffix}"].astype(np.float32)

        valid = (
            season_mask
            & np.isfinite(sos_arr) & (sos_arr != nodata)
            & np.isfinite(tom_arr) & (tom_arr != nodata)
            & np.isfinite(sen_arr) & (sen_arr != nodata)
            & np.isfinite(eos_arr) & (eos_arr != nodata)
            & (sos_arr <= tom_arr)
            & (tom_arr <= sen_arr)
            & (sen_arr <= eos_arr)
        )

        kc_season, active = _calculate_dynamic_kc_for_one_season(
            current_dekad_position=pos,
            sos=sos_arr,
            tom=tom_arr,
            sen=sen_arr,
            eos=eos_arr,
            valid_mask=valid,
            kc_ini=kc_ini,
            kc_mid=kc_mid,
            kc_end=kc_end,
        )

        new_only = (~any_active) & active
        overlap = any_active & active

        kc[new_only] = kc_season[new_only]
        kc[overlap] = np.maximum(kc[overlap], kc_season[overlap])
        any_active |= active

    return kc.astype(np.float32)
