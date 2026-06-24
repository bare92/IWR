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
