from pathlib import Path

import pandas as pd
import rasterio


REQUIRED_CROP_COLUMNS = [
    "crop_name",
    "root_depth_max_m",
    "Kc_ini",
    "Kc_mid",
    "Kc_end",
    "p",
]

NUMERIC_CROP_COLUMNS = [
    "root_depth_max_m",
    "Kc_ini",
    "Kc_mid",
    "Kc_end",
    "p",
]


def _clean_name(value):
    return str(value).strip().lower().replace(" ", "_")


def read_crop_parameters(crop_parameters_csv):
    """
    Read crop parameters from CSV.

    Expected columns:
    crop_name, root_depth_max_m, Kc_ini, Kc_mid, Kc_end, p
    """

    crop_parameters_csv = Path(crop_parameters_csv)
    crop_df = pd.read_csv(crop_parameters_csv)

    missing_columns = [
        col for col in REQUIRED_CROP_COLUMNS
        if col not in crop_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in crop parameter CSV: {missing_columns}"
        )

    for col in NUMERIC_CROP_COLUMNS:
        crop_df[col] = pd.to_numeric(crop_df[col], errors="coerce")

        if crop_df[col].isna().any():
            bad_rows = crop_df[crop_df[col].isna()].index.tolist()
            raise ValueError(
                f"Column '{col}' contains non-numeric values in rows: {bad_rows}"
            )

    if (crop_df["root_depth_max_m"] <= 0).any():
        raise ValueError("All root_depth_max_m values must be > 0.")

    if ((crop_df["p"] <= 0) | (crop_df["p"] > 1)).any():
        raise ValueError("All depletion factor p values must be in the range 0 < p <= 1.")

    return crop_df


def read_crop_fraction_raster(crop_fraction_path):
    """
    Read multiband crop fraction GeoTIFF.

    Each band is one crop.
    Each pixel value is the crop fraction.
    """

    crop_fraction_path = Path(crop_fraction_path)

    with rasterio.open(crop_fraction_path) as src:
        crop_fraction_data = src.read()
        profile = src.profile.copy()
        band_descriptions = src.descriptions

    return crop_fraction_data, profile, band_descriptions


def check_crop_raster_and_csv(crop_fraction_path, crop_parameters_csv):
    """
    Check that the number of GeoTIFF bands matches the number of crops in the CSV.

    If the GeoTIFF has band descriptions, also check that they match crop_name.
    """

    crop_df = read_crop_parameters(crop_parameters_csv)

    crop_fraction_data, profile, band_descriptions = read_crop_fraction_raster(
        crop_fraction_path
    )

    n_bands = crop_fraction_data.shape[0]
    n_crops = len(crop_df)

    if n_bands != n_crops:
        raise ValueError(
            f"Crop raster has {n_bands} bands, but CSV has {n_crops} crops."
        )

    descriptions_are_available = (
        band_descriptions is not None
        and all(desc not in [None, ""] for desc in band_descriptions)
    )

    if descriptions_are_available:
        raster_crop_names = [_clean_name(desc) for desc in band_descriptions]
        csv_crop_names = [_clean_name(name) for name in crop_df["crop_name"]]

        if raster_crop_names != csv_crop_names:
            raise ValueError(
                "Crop band names do not match CSV crop_name order.\n"
                f"Raster bands: {raster_crop_names}\n"
                f"CSV crops:    {csv_crop_names}"
            )

    return crop_df, crop_fraction_data, profile, band_descriptions
