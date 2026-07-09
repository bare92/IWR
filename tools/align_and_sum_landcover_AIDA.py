#!/usr/bin/env python3
"""
Align rainfed AIDA crop percentages to the irrigated 1 km DAO grid,
sum rainfed + irrigated percentages by crop, and create a quality mask.

This version:
  - keeps all values in percentage units;
  - avoids CRS.from_epsg(), because the environment has a PROJ database conflict;
  - forces the target CRS using an explicit EPSG:3035 PROJ string;
  - forces the source CRS using an explicit WGS84 PROJ string;
  - writes output rasters on the exact grid of italy_areas_irrigated_DAO_1km.tif.

Inputs:
  - italy_areas_irrigated_DAO_1km.tif  -> reference grid, 1 km, EPSG:3035
  - italy_areas_rainfed.tif            -> source raster, WGS84

Outputs:
  - italy_areas_rainfed_DAO_1km.tif
  - combined_crop_percent.tif
  - total_crop_percent_sum.tif
  - quality_mask_crop_sum_gt_100.tif
  - quality_summary.csv

Quality mask values:
  0   = crop percentage sum <= 100
  1   = crop percentage sum > 100
  255 = nodata / outside valid domain
"""

from pathlib import Path
import os
import re

# Important when a venv is accidentally picking up the wrong Conda PROJ database.
# This helps Rasterio/GDAL avoid using the incompatible Conda proj.db.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# USER SETTINGS
# ============================================================

RAINFED_PATH = Path("/share/data/DAO/AIDA/italy_areas_rainfed.tif")

REFERENCE_IRRIGATED_PATH = Path(
    "/share/data/DAO/AIDA/italy_areas_irrigated_DAO_1km.tif"
)

OUTPUT_DIR = Path("/share/data/DAO/AIDA/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAINFED_ALIGNED = OUTPUT_DIR / "italy_areas_rainfed_DAO_1km.tif"
OUT_COMBINED_PERCENT = OUTPUT_DIR / "italy_areas_irr_rainfed_DAO_1km_clipped_bilinear.tif"
OUT_TOTAL_PERCENT_SUM = OUTPUT_DIR / "total_crop_percent_sum.tif"
OUT_QUALITY_MASK = OUTPUT_DIR / "quality_mask_crop_sum_gt_100.tif"
OUT_SUMMARY_CSV = OUTPUT_DIR / "quality_summary.csv"

# Do NOT use CRS.from_epsg() here, because your environment has a PROJ database conflict.
# EPSG:3035 / ETRS89-extended / LAEA Europe as explicit PROJ string.
REFERENCE_CRS_OVERRIDE = (
    "+proj=laea "
    "+lat_0=52 "
    "+lon_0=10 "
    "+x_0=4321000 "
    "+y_0=3210000 "
    "+ellps=GRS80 "
    "+units=m "
    "+no_defs "
    "+type=crs"
)

# WGS84 geographic CRS as explicit PROJ string.
RAINFED_CRS_OVERRIDE = (
    "+proj=longlat "
    "+ellps=WGS84 "
    "+datum=WGS84 "
    "+no_defs "
    "+type=crs"
)

# For continuous percentage layers, average is appropriate when resampling.
RESAMPLING_METHOD = Resampling.bilinear

FLOAT_NODATA = np.nan
QUALITY_NODATA = 255

# Quality threshold in percentage.
QUALITY_THRESHOLD_PERCENT = 100.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_nan_value(value) -> bool:
    """Safely test whether a value is NaN."""
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def clean_crop_name(description: str, fallback: str) -> str:
    """
    Convert band descriptions such as:
      maize_i -> maize
      maize_r -> maize
      other_cereals_i -> other_cereals
      other_cereals_r -> other_cereals

    This allows irrigated and rainfed layers of the same crop to be summed.
    """
    if description is None or str(description).strip() == "":
        name = fallback
    else:
        name = str(description).strip()

    # Remove final _i or _r only when it is at the end of the name.
    name = re.sub(r"_(i|r)$", "", name)

    return name


def get_band_names(src: rasterio.DatasetReader) -> list[str]:
    """Return cleaned crop names from raster band descriptions."""
    names = []

    for band_idx in range(1, src.count + 1):
        desc = src.descriptions[band_idx - 1]
        fallback = f"band_{band_idx:02d}"
        names.append(clean_crop_name(desc, fallback))

    return names


def read_band_as_float(src: rasterio.DatasetReader, band_idx: int) -> np.ndarray:
    """
    Read one band as float32 and replace nodata with NaN.
    Values are kept in their original units, here percentage.
    """
    arr = src.read(band_idx).astype("float32")

    if src.nodata is not None and not is_nan_value(src.nodata):
        arr[arr == src.nodata] = np.nan

    return arr


def write_multiband_float_tif(
    output_path: Path,
    reference_src: rasterio.DatasetReader,
    output_crs,
    data_by_band: list[np.ndarray],
    band_names: list[str],
    nodata_value=np.nan,
):
    """Write a multiband float32 GeoTIFF using the reference grid."""
    profile = reference_src.profile.copy()

    profile.update(
        driver="GTiff",
        height=reference_src.height,
        width=reference_src.width,
        count=len(data_by_band),
        dtype="float32",
        nodata=nodata_value,
        crs=output_crs,
        transform=reference_src.transform,
        compress="LZW",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, arr in enumerate(data_by_band, start=1):
            dst.write(arr.astype("float32"), idx)
            dst.set_band_description(idx, band_names[idx - 1])


def write_singleband_float_tif(
    output_path: Path,
    reference_src: rasterio.DatasetReader,
    output_crs,
    arr: np.ndarray,
    description: str,
    nodata_value=np.nan,
):
    """Write a single-band float32 GeoTIFF using the reference grid."""
    profile = reference_src.profile.copy()

    profile.update(
        driver="GTiff",
        height=reference_src.height,
        width=reference_src.width,
        count=1,
        dtype="float32",
        nodata=nodata_value,
        crs=output_crs,
        transform=reference_src.transform,
        compress="LZW",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.set_band_description(1, description)


def write_quality_mask_tif(
    output_path: Path,
    reference_src: rasterio.DatasetReader,
    output_crs,
    quality_mask: np.ndarray,
):
    """Write uint8 quality mask."""
    profile = reference_src.profile.copy()

    profile.update(
        driver="GTiff",
        height=reference_src.height,
        width=reference_src.width,
        count=1,
        dtype="uint8",
        nodata=QUALITY_NODATA,
        crs=output_crs,
        transform=reference_src.transform,
        compress="LZW",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(quality_mask.astype("uint8"), 1)
        dst.set_band_description(1, "crop_sum_gt_100")


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    with rasterio.open(REFERENCE_IRRIGATED_PATH) as ref_src, rasterio.open(RAINFED_PATH) as rain_src:

        target_crs = REFERENCE_CRS_OVERRIDE
        source_crs = RAINFED_CRS_OVERRIDE

        print("Reference raster:")
        print(f"  Path: {REFERENCE_IRRIGATED_PATH}")
        print(f"  Original CRS: {ref_src.crs}")
        print(f"  CRS used for processing: {target_crs}")
        print(f"  Shape: {ref_src.height}, {ref_src.width}")
        print(f"  Pixel size: {ref_src.transform.a}, {ref_src.transform.e}")
        print(f"  Bands: {ref_src.count}")

        print("\nRainfed source raster:")
        print(f"  Path: {RAINFED_PATH}")
        print(f"  Original CRS: {rain_src.crs}")
        print(f"  CRS used for processing: {source_crs}")
        print(f"  Shape: {rain_src.height}, {rain_src.width}")
        print(f"  Pixel size: {rain_src.transform.a}, {rain_src.transform.e}")
        print(f"  Bands: {rain_src.count}")

        irrigated_crop_names = get_band_names(ref_src)
        rainfed_crop_names = get_band_names(rain_src)

        print("\nIrrigated crop names:")
        print(irrigated_crop_names)

        print("\nRainfed crop names:")
        print(rainfed_crop_names)

        # ------------------------------------------------------------
        # 1. Reproject / crop / resample rainfed raster to reference grid
        # ------------------------------------------------------------

        rainfed_aligned_percent_bands = []

        for band_idx in range(1, rain_src.count + 1):
            crop_name = rainfed_crop_names[band_idx - 1]
            print(f"Reprojecting rainfed band {band_idx}/{rain_src.count}: {crop_name}")

            src_arr = read_band_as_float(rain_src, band_idx)

            dst_arr = np.full(
                shape=(ref_src.height, ref_src.width),
                fill_value=np.nan,
                dtype="float32",
            )

            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=rain_src.transform,
                src_crs=source_crs,
                src_nodata=np.nan,
                dst_transform=ref_src.transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=RESAMPLING_METHOD,
            )

            rainfed_aligned_percent_bands.append(dst_arr)

        write_multiband_float_tif(
            output_path=OUT_RAINFED_ALIGNED,
            reference_src=ref_src,
            output_crs=target_crs,
            data_by_band=rainfed_aligned_percent_bands,
            band_names=[f"{name}_r_aligned_percent" for name in rainfed_crop_names],
            nodata_value=np.nan,
        )

        print(f"\nSaved aligned rainfed raster:\n  {OUT_RAINFED_ALIGNED}")

        # ------------------------------------------------------------
        # 2. Read irrigated raster as percentage
        # ------------------------------------------------------------

        irrigated_percent_by_crop = {}
        rainfed_percent_by_crop = {}

        for band_idx, crop_name in enumerate(irrigated_crop_names, start=1):
            arr_percent = read_band_as_float(ref_src, band_idx)
            irrigated_percent_by_crop[crop_name] = arr_percent

        for crop_name, arr_percent in zip(rainfed_crop_names, rainfed_aligned_percent_bands):
            rainfed_percent_by_crop[crop_name] = arr_percent

        # ------------------------------------------------------------
        # 3. Sum irrigated + rainfed by crop, keeping percentage units
        # ------------------------------------------------------------

        all_crop_names = sorted(
            set(irrigated_percent_by_crop.keys())
            | set(rainfed_percent_by_crop.keys())
        )

        combined_percent_bands = []
        summary_records = []

        for crop_name in all_crop_names:
            irr = irrigated_percent_by_crop.get(crop_name)
            rain = rainfed_percent_by_crop.get(crop_name)

            if irr is None:
                irr = np.full((ref_src.height, ref_src.width), np.nan, dtype="float32")

            if rain is None:
                rain = np.full((ref_src.height, ref_src.width), np.nan, dtype="float32")

            valid = np.isfinite(irr) | np.isfinite(rain)

            combined = np.full(
                shape=(ref_src.height, ref_src.width),
                fill_value=np.nan,
                dtype="float32",
            )

            combined[valid] = (
                np.nan_to_num(irr[valid], nan=0.0)
                + np.nan_to_num(rain[valid], nan=0.0)
            )

            combined_percent_bands.append(combined.astype("float32"))

            summary_records.append({
                "crop": crop_name,
                "max_percent": float(np.nanmax(combined)) if np.isfinite(combined).any() else np.nan,
                "mean_percent": float(np.nanmean(combined)) if np.isfinite(combined).any() else np.nan,
            })

        write_multiband_float_tif(
            output_path=OUT_COMBINED_PERCENT,
            reference_src=ref_src,
            output_crs=target_crs,
            data_by_band=combined_percent_bands,
            band_names=[f"{name}_percent" for name in all_crop_names],
            nodata_value=np.nan,
        )

        print(f"\nSaved combined crop percent raster:\n  {OUT_COMBINED_PERCENT}")

        # ------------------------------------------------------------
        # 4. Sum all crop percentages per pixel
        # ------------------------------------------------------------

        stack = np.stack(combined_percent_bands, axis=0)

        any_valid_crop = np.isfinite(stack).any(axis=0)

        total_crop_percent_sum = np.full(
            shape=(ref_src.height, ref_src.width),
            fill_value=np.nan,
            dtype="float32",
        )

        total_crop_percent_sum[any_valid_crop] = np.nansum(
            stack[:, any_valid_crop],
            axis=0,
        )

        write_singleband_float_tif(
            output_path=OUT_TOTAL_PERCENT_SUM,
            reference_src=ref_src,
            output_crs=target_crs,
            arr=total_crop_percent_sum,
            description="sum_all_crop_percentages",
            nodata_value=np.nan,
        )

        print(f"Saved total crop percentage sum raster:\n  {OUT_TOTAL_PERCENT_SUM}")

        # ------------------------------------------------------------
        # 5. Quality mask: 1 where sum of crop percentages exceeds 100
        # ------------------------------------------------------------

        quality_mask = np.full(
            shape=(ref_src.height, ref_src.width),
            fill_value=QUALITY_NODATA,
            dtype="uint8",
        )

        quality_mask[any_valid_crop] = 0
        quality_mask[any_valid_crop & (total_crop_percent_sum > QUALITY_THRESHOLD_PERCENT)] = 1

        write_quality_mask_tif(
            output_path=OUT_QUALITY_MASK,
            reference_src=ref_src,
            output_crs=target_crs,
            quality_mask=quality_mask,
        )

        print(f"Saved quality mask:\n  {OUT_QUALITY_MASK}")

        # ------------------------------------------------------------
        # 6. CSV summary
        # ------------------------------------------------------------

        n_valid_pixels = int(np.sum(any_valid_crop))
        n_bad_pixels = int(np.sum(quality_mask == 1))

        pct_bad_pixels = (
            100.0 * n_bad_pixels / n_valid_pixels
            if n_valid_pixels > 0
            else np.nan
        )

        summary_df = pd.DataFrame(summary_records)

        extra_rows = pd.DataFrame([
            {
                "crop": "__ALL_PIXELS__",
                "max_percent": float(np.nanmax(total_crop_percent_sum))
                if np.isfinite(total_crop_percent_sum).any()
                else np.nan,
                "mean_percent": float(np.nanmean(total_crop_percent_sum))
                if np.isfinite(total_crop_percent_sum).any()
                else np.nan,
                "valid_pixels": n_valid_pixels,
                "pixels_sum_gt_100": n_bad_pixels,
                "percent_pixels_sum_gt_100": pct_bad_pixels,
            }
        ])

        summary_df = pd.concat([summary_df, extra_rows], ignore_index=True)
        summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

        print(f"Saved summary CSV:\n  {OUT_SUMMARY_CSV}")

        print("\nQuality summary:")
        print(f"  Valid pixels: {n_valid_pixels}")
        print(f"  Pixels where crop sum > 100%: {n_bad_pixels}")
        print(f"  Percentage of valid pixels where crop sum > 100%: {pct_bad_pixels:.3f}%")

        print("\nFinished.")


if __name__ == "__main__":
    main()