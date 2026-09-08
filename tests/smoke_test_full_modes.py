import sys
import tempfile
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Prevent cross-environment PROJ database conflicts in test environments.
for _proj_var in ("PROJ_LIB", "PROJ_DATA"):
    _proj_path = os.environ.get(_proj_var)
    if _proj_path and "miniconda3/envs/" in _proj_path:
        os.environ.pop(_proj_var, None)

import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import (  # noqa: E402
    IWR_MODE_WATNEEDS_BLUE_ET,
    compute_actual_evapotranspiration,
    compute_blue_water_requirement_watneeds,
    compute_green_water_stress_coefficient,
    compute_potential_evapotranspiration,
    create_iwr_domain_mask,
    create_kc_pixel,
    create_raw_pixel,
    create_total_available_water_pixel,
    initialize_soil_moisture,
    run_iwr_model,
)

NODATA = -9999.0


def write_tif(path, array, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def build_profile(height, width):
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "nodata": NODATA,
        "crs": "EPSG:3035",
        "transform": Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0),
        "compress": "lzw",
    }


def test_legacy_mode_smoke():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p_dir = td / "P"
        et0_dir = td / "PET"
        out_dir = td / "out"
        p_dir.mkdir()
        et0_dir.mkdir()
        out_dir.mkdir()

        shape = (1, 2)
        profile = build_profile(*shape)

        precipitation = np.array([[0.0, 0.0]], dtype=np.float32)
        et0 = np.array([[10.0, 10.0]], dtype=np.float32)

        write_tif(p_dir / "P_20210101.tif", precipitation, profile)
        write_tif(et0_dir / "PET_20210101.tif", et0, profile)

        total_available_water = np.array([[0.1, 0.1]], dtype=np.float32)
        fmax = np.array([[5.0, 5.0]], dtype=np.float32)
        irrigation_mask = np.array([[1.0, 0.0]], dtype=np.float32)
        valid_mask = np.array([[1.0, 1.0]], dtype=np.float32)
        crop_fraction_data = np.array([[[1.0, 1.0]]], dtype=np.float32)

        crop_df = pd.DataFrame(
            {
                "crop_name": ["crop_a"],
                "root_depth_max_m": [1.0],
                "Kc_ini": [1.0],
                "Kc_mid": [1.0],
                "Kc_end": [1.0],
                "p": [0.5],
            }
        )

        phenology = {
            "phenonseasons": np.ones(shape, dtype=np.float32),
            "phenos1": np.ones(shape, dtype=np.float32),
            "phenom1": np.ones(shape, dtype=np.float32) * 2,
            "phenosen1": np.ones(shape, dtype=np.float32) * 3,
            "phenoe1": np.ones(shape, dtype=np.float32) * 36,
            "phenos2": np.zeros(shape, dtype=np.float32),
            "phenom2": np.zeros(shape, dtype=np.float32),
            "phenosen2": np.zeros(shape, dtype=np.float32),
            "phenoe2": np.zeros(shape, dtype=np.float32),
        }

        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 1)

        run_iwr_model(
            start_date=start,
            end_date=end,
            total_available_water=total_available_water,
            fmax=fmax,
            irrigation_mask=irrigation_mask,
            valid_area_mask=valid_mask,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            phenology=phenology,
            precipitation_geotiff_folder=p_dir,
            et0_geotiff_folder=et0_dir,
            output_folder=out_dir,
            output_profile=profile,
            strict_checks=False,
            write_debug_csv=False,
            write_cumulative_iwr=True,
            debug_mode=False,
            iwr_mode=IWR_MODE_WATNEEDS_BLUE_ET,
            iwr_domain="irrigated",
            theoretical_iwr_target="stress_threshold",
            initial_soil_moisture_fraction=0.2,
        )

        # Compute explicit legacy expected value from helper equations.
        taw_pixel = create_total_available_water_pixel(
            total_available_water=total_available_water,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            nodata=NODATA,
        )
        raw = create_raw_pixel(
            total_available_water=total_available_water,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            nodata=NODATA,
        )
        soil_init = initialize_soil_moisture(
            total_available_water_pixel=taw_pixel,
            initial_fraction=0.2,
            nodata=NODATA,
        )
        valid_area_pixels = valid_mask == 1
        iwr_domain_pixels = create_iwr_domain_mask(
            iwr_domain="irrigated",
            valid_area_pixels=valid_area_pixels,
            irrigation_mask=irrigation_mask,
            nodata=NODATA,
        )

        kc = create_kc_pixel(
            current_date=start,
            phenology=phenology,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            nodata=NODATA,
        )
        et_c = compute_potential_evapotranspiration(et0=et0, kc_pixel=kc)
        ks = compute_green_water_stress_coefficient(
            soil_moisture_green=soil_init,
            raw=raw,
            nodata=NODATA,
            valid_mask=valid_area_pixels,
        )
        et_green = compute_actual_evapotranspiration(
            potential_evapotranspiration=et_c,
            water_stress_coefficient=ks,
            nodata=NODATA,
            valid_mask=valid_area_pixels,
        )
        expected_iwr = compute_blue_water_requirement_watneeds(
            potential_evapotranspiration=et_c,
            green_evapotranspiration=et_green,
            irrigated_pixels=iwr_domain_pixels,
            nodata=NODATA,
            valid_mask=valid_area_pixels,
        )

        daily_tif = out_dir / "IWR" / "iwr_20210101.tif"
        with rasterio.open(daily_tif) as src:
            produced = src.read(1)

        active = (kc > 0) & iwr_domain_pixels & valid_area_pixels
        expected_written = np.full_like(expected_iwr, NODATA, dtype=np.float32)
        expected_written[active] = expected_iwr[active]

        assert np.allclose(produced, expected_written, equal_nan=False)


def main():
    test_legacy_mode_smoke()
    print("smoke_test_full_modes.py: PASS")


if __name__ == "__main__":
    main()
