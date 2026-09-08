from datetime import date, datetime
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import (  # noqa: E402
    IWR_MODE_WATNEEDS_BLUE_ET,
    compute_potential_evapotranspiration,
    compute_theoretical_net_irrigation_requirement,
    create_kc_balance_pixels,
    resolve_offseason_water_balance_kc,
    run_iwr_model,
)


NODATA = -9999.0


def _crop_df():
    return pd.DataFrame(
        {
            "crop_name": ["crop_a"],
            "root_depth_max_m": [1.0],
            "Kc_ini": [1.0],
            "Kc_mid": [1.0],
            "Kc_end": [1.0],
            "p": [0.5],
        }
    )


def _phenology(active: bool):
    shape = (1, 1)
    if active:
        sos, tom, sen, eos = 1, 1, 1, 1
        current = date(2020, 1, 1)
    else:
        sos, tom, sen, eos = 7, 7, 7, 7
        current = date(2020, 1, 1)

    ones = np.ones(shape, dtype=np.float32)
    return current, {
        "phenonseasons": ones.copy(),
        "phenos1": ones * sos,
        "phenom1": ones * tom,
        "phenosen1": ones * sen,
        "phenoe1": ones * eos,
        "phenos2": np.zeros(shape, dtype=np.float32),
        "phenom2": np.zeros(shape, dtype=np.float32),
        "phenosen2": np.zeros(shape, dtype=np.float32),
        "phenoe2": np.zeros(shape, dtype=np.float32),
    }


def _profile():
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 1,
        "height": 1,
        "count": 1,
        "crs": None,
        "transform": rasterio.transform.from_bounds(0, 0, 1000, 1000, 1, 1),
        "nodata": NODATA,
    }


def _write_forcing(folder: Path, prefix: str, stamp: str, value: float):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{stamp}.tif"
    with rasterio.open(path, "w", **_profile()) as dst:
        dst.write(np.array([[value]], dtype=np.float32), 1)
    return path


def test_inactive_day_uses_balance_kc_without_cropping_outputs():
    current_date, phenology = _phenology(active=False)
    crop_fraction_data = np.array([[[1.0]]], dtype=np.float32)
    model_valid_mask = np.array([[True]], dtype=bool)
    kc_crop_output, kc_water_balance = create_kc_balance_pixels(
        current_date=current_date,
        phenology=phenology,
        crop_fraction_data=crop_fraction_data,
        crop_df=_crop_df(),
        model_valid_mask=model_valid_mask,
        phenology_active_mask=np.array([[False]], dtype=bool),
        offseason_water_balance_kc=0.5,
        nodata=NODATA,
    )

    et0 = np.array([[6.0]], dtype=np.float32)
    crop_et = compute_potential_evapotranspiration(et0=et0, kc_pixel=kc_crop_output)
    balance_et = compute_potential_evapotranspiration(et0=et0, kc_pixel=kc_water_balance)

    np.testing.assert_allclose(crop_et, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(balance_et, np.array([[3.0]], dtype=np.float32))

    irrigation, soil_new, _ = compute_theoretical_net_irrigation_requirement(
        soil_moisture_previous=np.array([[10.0]], dtype=np.float32),
        precipitation_effective=np.array([[0.0]], dtype=np.float32),
        potential_evapotranspiration=balance_et,
        deep_percolation=np.array([[0.0]], dtype=np.float32),
        runoff=np.array([[0.0]], dtype=np.float32),
        target_storage=np.array([[5.0]], dtype=np.float32),
        total_available_water_pixel=np.array([[100.0]], dtype=np.float32),
        demand_mask=np.array([[False]], dtype=bool),
        nodata=NODATA,
        valid_mask=np.array([[True]], dtype=bool),
    )

    np.testing.assert_allclose(irrigation, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(soil_new, np.array([[7.0]], dtype=np.float32))


def test_active_day_crop_and_balance_kc_match():
    current_date, phenology = _phenology(active=True)
    crop_fraction_data = np.array([[[1.0]]], dtype=np.float32)
    model_valid_mask = np.array([[True]], dtype=bool)
    kc_crop_output, kc_water_balance = create_kc_balance_pixels(
        current_date=current_date,
        phenology=phenology,
        crop_fraction_data=crop_fraction_data,
        crop_df=_crop_df(),
        model_valid_mask=model_valid_mask,
        phenology_active_mask=np.array([[True]], dtype=bool),
        offseason_water_balance_kc=0.5,
        nodata=NODATA,
    )

    np.testing.assert_allclose(kc_crop_output, kc_water_balance)
    assert float(kc_crop_output[0, 0]) > 0.0


def test_legacy_watneeds_default_offseason_kc_matches_explicit_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        precip_dir = tmpdir / "P"
        et0_dir = tmpdir / "PET"
        out_default = tmpdir / "out_default"
        out_explicit = tmpdir / "out_explicit"

        stamp = "20200101"
        _write_forcing(precip_dir, "P", stamp, 0.0)
        _write_forcing(et0_dir, "PET", stamp, 6.0)

        total_available_water = np.array([[100.0]], dtype=np.float32)
        fmax = np.array([[5.0]], dtype=np.float32)
        irrigation_mask = np.array([[1.0]], dtype=np.float32)
        valid_mask = np.array([[1.0]], dtype=np.float32)
        crop_fraction_data = np.array([[[1.0]]], dtype=np.float32)
        crop_df = _crop_df()
        current_date = datetime(2020, 1, 1)
        phenology = {
            "phenonseasons": np.ones((1, 1), dtype=np.float32),
            "phenos1": np.array([[7.0]], dtype=np.float32),
            "phenom1": np.array([[8.0]], dtype=np.float32),
            "phenosen1": np.array([[9.0]], dtype=np.float32),
            "phenoe1": np.array([[10.0]], dtype=np.float32),
            "phenos2": np.zeros((1, 1), dtype=np.float32),
            "phenom2": np.zeros((1, 1), dtype=np.float32),
            "phenosen2": np.zeros((1, 1), dtype=np.float32),
            "phenoe2": np.zeros((1, 1), dtype=np.float32),
        }

        common_kwargs = dict(
            start_date=current_date,
            end_date=current_date,
            total_available_water=total_available_water,
            fmax=fmax,
            irrigation_mask=irrigation_mask,
            valid_area_mask=valid_mask,
            crop_fraction_data=crop_fraction_data,
            crop_df=crop_df,
            phenology=phenology,
            precipitation_geotiff_folder=precip_dir,
            et0_geotiff_folder=et0_dir,
            output_profile=_profile(),
            strict_checks=False,
            write_debug_csv=False,
            write_cumulative_iwr=False,
            debug_mode=False,
            iwr_mode=IWR_MODE_WATNEEDS_BLUE_ET,
            iwr_domain="irrigated",
            theoretical_iwr_target="stress_threshold",
            initial_soil_moisture_fraction=0.5,
            drainage_scheme="auto",
        )

        default_state = run_iwr_model(output_folder=out_default, **common_kwargs)
        explicit_state = run_iwr_model(
            output_folder=out_explicit,
            offseason_water_balance_kc=0.0,
            **common_kwargs,
        )

        np.testing.assert_allclose(default_state[0], explicit_state[0])
        np.testing.assert_allclose(default_state[1], explicit_state[1])

        with rasterio.open(out_default / "IWR" / "iwr_20200101.tif") as src:
            default_iwr = src.read(1)
        with rasterio.open(out_explicit / "IWR" / "iwr_20200101.tif") as src:
            explicit_iwr = src.read(1)

        np.testing.assert_allclose(default_iwr, explicit_iwr)
        np.testing.assert_allclose(default_iwr, np.array([[0.0]], dtype=np.float32))
