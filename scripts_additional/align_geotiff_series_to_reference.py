#!/usr/bin/env python3
"""
Align a series of GeoTIFF layers to a reference raster grid.

Inputs are configured directly in this script under USER SETTINGS.

What this script does for each input GeoTIFF:
- reprojects to the reference CRS if needed
- resamples to the reference resolution
- aligns to the exact reference extent/transform
- writes aligned output to OUTPUT_FOLDER
"""

import glob
import os
from pathlib import Path

import numpy as np

# Avoid mixed conda/venv PROJ database conflicts.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject


# ============================================================
# USER SETTINGS
# ============================================================

# Reference raster that defines target CRS, transform, width and height.
REFERENCE_MAP = Path("/home/fremen/data/projects/Burkina/00_Data_iwr/static/burkina_valid_mask.tif")

# List of GeoTIFFs to align.
# INPUT_GEOTIFFS = [
#     Path("/path/to/input_01.tif"),
#     Path("/path/to/input_02.tif"),
# ]
INPUT_GEOTIFFS = [
    Path(p) for p in
      glob.glob("/home/fremen/data/basedata/WORLD/ASAP_pheno/pheno*.tif")]  

INPUT_GEOTIFFS = [
    Path(p) for p in
      glob.glob("/home/fremen/data/basedata/WORLD/ASAP_pheno/pheno*.tif")] 

# Folder where aligned rasters will be written.
OUTPUT_FOLDER = Path("/home/fremen/data/projects/Burkina/00_Data_iwr/static")

# Optional nodata override used for both input and output nodata handling.
# Set to None to use each source file nodata as-is.
NODATA_VALUE = None

# Optional CRS override for the reference map when REFERENCE_MAP has no CRS metadata.
# Examples: "EPSG:4326", "EPSG:3035", or a PROJ string.
REFERENCE_CRS_OVERRIDE = "+proj=longlat +datum=WGS84 +no_defs +type=crs"

# Optional CRS override for source rasters when inputs have no CRS metadata.
# Examples: "EPSG:4326", "EPSG:3035", or a PROJ string.
SOURCE_CRS_OVERRIDE = "+proj=longlat +datum=WGS84 +no_defs +type=crs"

# Resampling method: "nearest" or "bilinear".
# Use nearest for categorical rasters, bilinear for continuous rasters.
RESAMPLING_METHOD = "nearest"


# ============================================================
# IMPLEMENTATION
# ============================================================


def get_resampling(method_name: str) -> Resampling:
    method_name = method_name.lower().strip()
    if method_name == "nearest":
        return Resampling.nearest
    if method_name == "bilinear":
        return Resampling.bilinear
    raise ValueError(f"Unsupported RESAMPLING_METHOD: {method_name}")


def parse_crs(crs_value):
    """Parse CRS from None, CRS object, or string/int EPSG-like value."""
    if crs_value is None:
        return None
    if isinstance(crs_value, CRS):
        return crs_value
    return CRS.from_user_input(crs_value)


def aligned_output_profile(
    src: rasterio.io.DatasetReader,
    reference: rasterio.io.DatasetReader,
    target_crs: CRS,
    dst_nodata,
    resampling: Resampling,
) -> dict:
    """Build an output profile based on the reference grid and source band count."""
    if resampling == Resampling.nearest:
        dst_dtype = src.dtypes[0]
    else:
        dst_dtype = "float32"

    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=src.count,
        dtype=dst_dtype,
        crs=target_crs,
        transform=reference.transform,
        width=reference.width,
        height=reference.height,
        nodata=dst_nodata,
        compress="LZW",
    )
    return profile


def nodata_fill_value(dst_nodata, dst_dtype):
    """Choose a safe fill value for destination array initialization."""
    if dst_nodata is not None:
        return dst_nodata

    if np.issubdtype(np.dtype(dst_dtype), np.floating):
        return np.nan

    return 0


def align_one_geotiff(
    src_path: Path,
    reference: rasterio.io.DatasetReader,
    target_crs: CRS,
    output_folder: Path,
    resampling: Resampling,
    nodata_override,
    source_crs_override,
) -> Path:
    """Align one source GeoTIFF to the reference grid and write output."""
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else nodata_override
        dst_nodata = nodata_override if nodata_override is not None else src.nodata
        src_crs = src.crs if src.crs is not None else source_crs_override

        if src_crs is None:
            raise ValueError(
                f"Input raster has no CRS metadata and SOURCE_CRS_OVERRIDE is None: {src_path}"
            )

        profile = aligned_output_profile(
            src=src,
            reference=reference,
            target_crs=target_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

        output_path = output_folder / f"{src_path.stem}_aligned.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                fill_value = nodata_fill_value(profile["nodata"], profile["dtype"])
                destination = np.full(
                    (reference.height, reference.width),
                    fill_value,
                    dtype=profile["dtype"],
                )

                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=reference.transform,
                    dst_crs=target_crs,
                    dst_nodata=profile["nodata"],
                    resampling=resampling,
                )

                dst.write(destination, band_idx)

                band_description = None
                if src.descriptions and band_idx - 1 < len(src.descriptions):
                    band_description = src.descriptions[band_idx - 1]
                if band_description:
                    dst.set_band_description(band_idx, band_description)

    return output_path


def validate_settings() -> None:
    if not REFERENCE_MAP.exists():
        raise FileNotFoundError(f"Reference map not found: {REFERENCE_MAP}")

    if not INPUT_GEOTIFFS:
        raise ValueError("INPUT_GEOTIFFS is empty. Provide at least one GeoTIFF path.")

    missing_inputs = [str(path) for path in INPUT_GEOTIFFS if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Some input GeoTIFFs were not found:\n- " + "\n- ".join(missing_inputs)
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:
    validate_settings()
    resampling = get_resampling(RESAMPLING_METHOD)
    reference_crs_override = parse_crs(REFERENCE_CRS_OVERRIDE)
    source_crs_override = parse_crs(SOURCE_CRS_OVERRIDE)

    print(f"Reference map: {REFERENCE_MAP}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Input files: {len(INPUT_GEOTIFFS)}")
    print(f"Resampling: {RESAMPLING_METHOD}")
    print(f"Nodata override: {NODATA_VALUE}")
    print(f"Reference CRS override: {REFERENCE_CRS_OVERRIDE}")
    print(f"Source CRS override: {SOURCE_CRS_OVERRIDE}")

    with rasterio.open(REFERENCE_MAP) as reference:
        target_crs = reference.crs if reference.crs is not None else reference_crs_override
        if target_crs is None:
            raise ValueError(
                "Reference raster has no CRS metadata. Set REFERENCE_CRS_OVERRIDE in USER SETTINGS."
            )

        for src_path in INPUT_GEOTIFFS:
            print(f"Aligning: {src_path}")
            out_path = align_one_geotiff(
                src_path=src_path,
                reference=reference,
                target_crs=target_crs,
                output_folder=OUTPUT_FOLDER,
                resampling=resampling,
                nodata_override=NODATA_VALUE,
                source_crs_override=source_crs_override,
            )
            print(f"  -> wrote: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
