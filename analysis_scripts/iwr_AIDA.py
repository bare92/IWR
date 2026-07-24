#!/usr/bin/env python3
"""
Create an AIDA-derived irrigation water requirement layer for comparison
with modelled IWR outputs.

Inputs from AIDA:
  italy_areas_irrigated.tif  -> crop-specific irrigated area [ha]
  italy_i_bw_mm_yr.tif      -> crop-specific annual blue water demand [mm/year]

Outputs on original AIDA grid:
  aida_iwr_blue_mm_yr_irrigated_area.tif
      Area-weighted blue water demand [mm/year over irrigated crop area]

  aida_iwr_blue_volume_m3_yr.tif
      Total blue water demand [m3/year per pixel]

  aida_irrigated_area_total_ha.tif
      Total irrigated area [ha per pixel]

Output aligned to model grid:
  aida_iwr_blue_mm_yr_irrigated_area_aligned.tif
      AIDA blue-water IWR [mm/year] reprojected to your model grid.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# USER SETTINGS
# ============================================================

AIDA_DIR = Path("/share/data/DAO/auxiliary/AIDA_dataset")
OUT_DIR = AIDA_DIR / "derived_iwr"

AREA_FILE = AIDA_DIR / "italy_areas_irrigated.tif"
BLUE_FILE = AIDA_DIR / "italy_i_bw_mm_yr.tif"

TARGET_GRID = Path("/share/data/DAO/static/processed/working_grid_3035_1km.tif")

IGNORE_NEGATIVE_VALUES = True
MIN_IRRIGATED_AREA_HA = 1e-9
NODATA_OUT = -9999.0


# Explicit CRS definitions, avoiding EPSG database lookup
AIDA_SRC_CRS = (
    "+proj=longlat "
    "+datum=WGS84 "
    "+no_defs "
    "+type=crs"
)

MODEL_TARGET_CRS = (
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


# ============================================================
# FUNCTIONS
# ============================================================

def clean_array(arr, nodata=None):
    """Convert raster band to float64 and mask nodata/invalid values."""
    arr = arr.astype("float64")

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    if IGNORE_NEGATIVE_VALUES:
        arr[arr < 0] = np.nan

    return arr


def remove_existing_raster(path):
    """Remove an existing output file and common sidecar files."""
    path = Path(path)

    for p in [
        path,
        Path(str(path) + ".aux.xml"),
        Path(str(path) + ".ovr"),
        Path(str(path) + ".msk"),
    ]:
        if p.exists():
            p.unlink()


def make_output_profile(base_profile, array, dtype="float32", nodata=NODATA_OUT):
    """Create a safe single-band GeoTIFF profile."""
    profile = base_profile.copy()

    for key in [
        "blockxsize",
        "blockysize",
        "tiled",
        "interleave",
        "compress",
        "predictor",
        "photometric",
    ]:
        profile.pop(key, None)

    profile.update(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        nodata=nodata,
        BIGTIFF="IF_SAFER",
    )

    return profile


def write_geotiff(path, array, base_profile, nodata=NODATA_OUT, dtype="float32"):
    """Write a single-band GeoTIFF safely."""
    path = Path(path)
    remove_existing_raster(path)

    out_profile = make_output_profile(
        base_profile=base_profile,
        array=array,
        dtype=dtype,
        nodata=nodata,
    )

    out = array.astype(dtype)
    out[~np.isfinite(out)] = nodata

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)


def reproject_iwr_to_target_grid(
    src_array,
    src_profile,
    target_grid,
    out_path,
):
    """
    Reproject AIDA IWR [mm/year] to the target model grid.

    Uses explicit CRS strings to avoid PROJ/EPSG database conflicts.
    """
    target_grid = Path(target_grid)

    with rasterio.open(target_grid) as target:
        target_profile = target.profile.copy()
        target_transform = target.transform
        target_shape = (target.height, target.width)

    dst_array = np.full(target_shape, np.nan, dtype="float64")

    src_for_warp = src_array.astype("float64").copy()
    src_for_warp[~np.isfinite(src_for_warp)] = NODATA_OUT

    reproject(
        source=src_for_warp,
        destination=dst_array,
        src_transform=src_profile["transform"],
        src_crs=AIDA_SRC_CRS,
        src_nodata=NODATA_OUT,
        dst_transform=target_transform,
        dst_crs=MODEL_TARGET_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )

    target_profile["crs"] = MODEL_TARGET_CRS

    write_geotiff(out_path, dst_array, target_profile)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not AREA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {AREA_FILE}")

    if not BLUE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {BLUE_FILE}")

    with rasterio.open(AREA_FILE) as area_src, rasterio.open(BLUE_FILE) as blue_src:

        if area_src.count != blue_src.count:
            raise ValueError(
                f"Band mismatch: {AREA_FILE.name} has {area_src.count} bands, "
                f"{BLUE_FILE.name} has {blue_src.count} bands."
            )

        if area_src.shape != blue_src.shape:
            raise ValueError("Area and blue-water rasters have different shapes.")

        if area_src.transform != blue_src.transform:
            raise ValueError("Area and blue-water rasters have different transforms.")

        profile = area_src.profile.copy()

        # Force AIDA CRS metadata to WGS84 without EPSG lookup.
        profile["crs"] = AIDA_SRC_CRS

        print(f"Reading {area_src.count} crop bands")
        print(f"Grid: {area_src.width} x {area_src.height}")
        print(f"Original CRS: {area_src.crs}")
        print(f"Forced source CRS: {AIDA_SRC_CRS}")
        print(f"Resolution: {area_src.res}")

        total_irrigated_area_ha = np.zeros(area_src.shape, dtype="float64")
        total_blue_volume_m3 = np.zeros(area_src.shape, dtype="float64")

        for band in range(1, area_src.count + 1):
            print(f"Processing band {band}/{area_src.count}")

            area_ha = clean_array(area_src.read(band), area_src.nodata)
            blue_mm_yr = clean_array(blue_src.read(band), blue_src.nodata)

            area_ha = np.nan_to_num(area_ha, nan=0.0)
            blue_mm_yr = np.nan_to_num(blue_mm_yr, nan=0.0)

            # 1 mm over 1 ha = 10 m3
            blue_volume_m3 = blue_mm_yr * area_ha * 10.0

            total_irrigated_area_ha += area_ha
            total_blue_volume_m3 += blue_volume_m3

        iwr_blue_mm_yr = np.full(area_src.shape, np.nan, dtype="float64")

        valid = total_irrigated_area_ha > MIN_IRRIGATED_AREA_HA

        iwr_blue_mm_yr[valid] = total_blue_volume_m3[valid] / (
            total_irrigated_area_ha[valid] * 10.0
        )

        total_irrigated_area_ha[~valid] = np.nan
        total_blue_volume_m3[~valid] = np.nan

    # ------------------------------------------------------------
    # Save outputs on original AIDA grid
    # ------------------------------------------------------------

    out_mm = OUT_DIR / "aida_iwr_blue_mm_yr_irrigated_area.tif"
    out_vol = OUT_DIR / "aida_iwr_blue_volume_m3_yr.tif"
    out_area = OUT_DIR / "aida_irrigated_area_total_ha.tif"

    write_geotiff(out_mm, iwr_blue_mm_yr, profile)
    write_geotiff(out_vol, total_blue_volume_m3, profile)
    write_geotiff(out_area, total_irrigated_area_ha, profile)

    print(f"Saved: {out_mm}")
    print(f"Saved: {out_vol}")
    print(f"Saved: {out_area}")

    # ------------------------------------------------------------
    # Align IWR mm/year layer to model grid
    # ------------------------------------------------------------

    if TARGET_GRID is not None:
        if not TARGET_GRID.exists():
            raise FileNotFoundError(f"TARGET_GRID not found: {TARGET_GRID}")

        aligned_dir = OUT_DIR / "aligned_to_model_grid"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        out_aligned = aligned_dir / "aida_iwr_blue_mm_yr_irrigated_area_aligned.tif"

        print(f"Aligning AIDA IWR to target grid: {TARGET_GRID}")

        reproject_iwr_to_target_grid(
            src_array=iwr_blue_mm_yr,
            src_profile=profile,
            target_grid=TARGET_GRID,
            out_path=out_aligned,
        )

        print(f"Saved aligned IWR layer: {out_aligned}")

    print("Done.")


if __name__ == "__main__":
    main()