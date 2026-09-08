import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import IWR_MODE_THEORETICAL_NET_IRRIGATION, run_iwr_model  # noqa: E402
import analysis_scripts.plots_iwr_1 as plots  # noqa: E402

NODATA = -9999.0


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


def write_single_band_tif(path, array, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def test_static_support_rasters_alignment_and_values():
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

        write_single_band_tif(p_dir / "P_20210101.tif", np.zeros(shape, dtype=np.float32), profile)
        write_single_band_tif(et0_dir / "PET_20210101.tif", np.ones(shape, dtype=np.float32), profile)

        total_available_water = np.array([[0.1, 0.1]], dtype=np.float32)
        fmax = np.array([[5.0, 5.0]], dtype=np.float32)
        valid_mask = np.array([[1.0, 1.0]], dtype=np.float32)

        # Percent-like fractions with first pixel summing to 120% => normalized to 1.0
        crop_fraction_data = np.array(
            [
                [[60.0, 0.0]],
                [[60.0, 0.0]],
            ],
            dtype=np.float32,
        )

        crop_df = pd.DataFrame(
            {
                "crop_name": ["a", "b"],
                "root_depth_max_m": [1.0, 1.0],
                "Kc_ini": [1.0, 1.0],
                "Kc_mid": [1.0, 1.0],
                "Kc_end": [1.0, 1.0],
                "p": [0.5, 0.5],
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

        run_iwr_model(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 1, 1),
            total_available_water=total_available_water,
            fmax=fmax,
            irrigation_mask=None,
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
            write_cumulative_iwr=False,
            write_static_support_layers=True,
            iwr_mode=IWR_MODE_THEORETICAL_NET_IRRIGATION,
            iwr_domain="all_cropped",
            theoretical_iwr_target="stress_threshold",
            initial_soil_moisture_fraction=0.5,
        )

        prepared_path = out_dir / "Static" / "prepared_crop_fraction_sum.tif"
        area_frac_path = out_dir / "Static" / "iwr_analysis_area_fraction.tif"
        iwr_path = out_dir / "IWR" / "iwr_20210101.tif"

        with rasterio.open(prepared_path) as src_pre, rasterio.open(area_frac_path) as src_area, rasterio.open(iwr_path) as src_iwr:
            prepared = src_pre.read(1)
            area_frac = src_area.read(1)
            iwr = src_iwr.read(1)

            # Alignment checks.
            assert src_pre.shape == src_iwr.shape
            assert src_area.shape == src_iwr.shape
            assert src_pre.transform == src_iwr.transform
            assert src_area.transform == src_iwr.transform
            assert src_pre.crs == src_iwr.crs
            assert src_area.crs == src_iwr.crs

        # Fractions in [0, 1] on valid values.
        pre_valid = prepared != NODATA
        area_valid = area_frac != NODATA
        assert np.all((prepared[pre_valid] >= 0.0) & (prepared[pre_valid] <= 1.0))
        assert np.all((area_frac[area_valid] >= 0.0) & (area_frac[area_valid] <= 1.0))

        # Exact normalized crop fractions used internally.
        assert np.isclose(prepared[0, 0], 1.0)
        assert np.isclose(prepared[0, 1], 0.0)
        assert np.isclose(area_frac[0, 0], 1.0)
        assert area_frac[0, 1] == NODATA


def test_volume_conversion_and_weighted_mean_math():
    # Criterion 2: 5 mm/day over 20% of 1,000,000 m2 => 1,000 m3/day
    depth = np.array([[5.0]], dtype=np.float64)
    analysis_area_m2 = np.array([[200000.0]], dtype=np.float64)
    stats = plots.compute_weighted_metrics_from_depth_array(depth, analysis_area_m2, 200000.0)
    assert np.isclose(stats["volume_m3"], 1000.0)

    # Fails if full cell area is used: would be 5000 instead of 1000.
    assert not np.isclose(stats["volume_m3"], 5000.0)

    # Criterion 3: weighted mean = 6 mm.
    depth2 = np.array([[4.0, 10.0]], dtype=np.float64)
    area2 = np.array([[1000000.0, 500000.0]], dtype=np.float64)
    stats2 = plots.compute_weighted_metrics_from_depth_array(depth2, area2, 1500000.0)
    assert np.isclose(stats2["active_mean_mm"], 6.0)


def test_annual_depth_from_volume_not_sum_of_daily_means_and_missing_year_nan():
    # 2025 entirely missing; 2026 has two days with non-null volumes.
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "year": [2026, 2026],
            "iwr_volume_m3_day": [1000.0, 3000.0],
            # Deliberately unrelated active means; annual depth must ignore these.
            "iwr_active_area_mean_mm_day": [100.0, 100.0],
            "etx_volume_m3_day": [2000.0, 2000.0],
            "eta_stress_volume_m3_day": [1000.0, 1200.0],
            "et_deficit_volume_m3_day": [1000.0, 800.0],
        }
    )

    annual = plots.build_annual_summary(
        daily_df=daily_df,
        configured_start=pd.Timestamp("2025-01-01"),
        configured_end=pd.Timestamp("2026-04-30"),
        total_analysis_area_m2=200000.0,
    )

    row_2026 = annual.loc[annual["year"] == 2026].iloc[0]
    expected_mm = (4000.0 / 200000.0) * 1000.0
    assert np.isclose(row_2026["annual_iwr_domain_mean_mm_year"], expected_mm)

    row_2025 = annual.loc[annual["year"] == 2025].iloc[0]
    assert np.isnan(row_2025["annual_iwr_volume_m3_year"])


def test_et_consistency_and_balance_diagnostics():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        profile = build_profile(1, 2)
        etx_path = td / "etx_20210101.tif"
        eta_path = td / "eta_stress_20210101.tif"

        write_single_band_tif(etx_path, np.array([[5.0, 8.0]], dtype=np.float32), profile)
        write_single_band_tif(eta_path, np.array([[3.0, 8.0]], dtype=np.float32), profile)

        analysis_area_m2 = np.array([[1000000.0, 500000.0]], dtype=np.float64)
        total_analysis_area_m2 = float(np.nansum(analysis_area_m2))

        _, _, _, balance = plots.compute_et_pair_metrics(
            etx_path=etx_path,
            eta_stress_path=eta_path,
            analysis_area_m2=analysis_area_m2,
            total_analysis_area_m2=total_analysis_area_m2,
        )

        assert balance["max_balance_error_mm"] <= 1e-12
        assert balance["mean_balance_error_mm"] <= 1e-12
        assert balance["pixels_above_tolerance"] == 0


def test_partial_year_marking_and_zero_pixel_validity_and_missing_day_reporting():
    # Zero-valued IWR pixel remains valid.
    depth = np.array([[0.0]], dtype=np.float64)
    area = np.array([[1000000.0]], dtype=np.float64)
    stats = plots.compute_weighted_metrics_from_depth_array(depth, area, 1000000.0)
    assert stats["valid_pixel_count"] == 1
    assert np.isclose(stats["volume_m3"], 0.0)
    assert np.isclose(stats["active_mean_mm"], 0.0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        iwr_dir = td / "IWR"
        iwr_dir.mkdir()
        profile = build_profile(1, 1)
        write_single_band_tif(iwr_dir / "iwr_20260101.tif", np.array([[0.0]], dtype=np.float32), profile)

        iwr_files = {pd.Timestamp("2026-01-01"): iwr_dir / "iwr_20260101.tif"}

        daily, _ = plots.build_daily_summary(
            iwr_files=iwr_files,
            etx_files={},
            eta_stress_files={},
            metadata={
                "iwr_mode": "theoretical_net_irrigation",
                "iwr_domain": "all_cropped",
                "theoretical_iwr_target": "stress_threshold",
                "model_start_date": "2026-01-01",
                "model_end_date": "2026-01-02",
            },
            analysis_start=pd.Timestamp("2026-01-01"),
            analysis_end=pd.Timestamp("2026-01-02"),
            analysis_area_m2=np.array([[1000000.0]], dtype=np.float64),
            total_analysis_area_m2=1000000.0,
        )

        missing_row = daily.loc[daily["date"] == pd.Timestamp("2026-01-02")].iloc[0]
        assert np.isnan(missing_row["iwr_volume_m3_day"])

        # Criterion 6: 2026 partial calendar year for configured end at April 30.
        all_dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
        daily_for_annual = pd.DataFrame(
            {
                "date": all_dates,
                "year": [2026] * len(all_dates),
                "iwr_volume_m3_day": [1000.0] * len(all_dates),
                "etx_volume_m3_day": [2000.0] * len(all_dates),
                "eta_stress_volume_m3_day": [1500.0] * len(all_dates),
                "et_deficit_volume_m3_day": [500.0] * len(all_dates),
            }
        )

        annual = plots.build_annual_summary(
            daily_df=daily_for_annual,
            configured_start=pd.Timestamp("1991-01-01"),
            configured_end=pd.Timestamp("2026-04-30"),
            total_analysis_area_m2=1000000.0,
        )

        row_2026 = annual.loc[annual["year"] == 2026].iloc[0]
        assert row_2026["period_start"] == "2026-01-01"
        assert row_2026["period_end"] == "2026-04-30"
        assert bool(row_2026["is_full_calendar_year"]) is False
        assert bool(row_2026["is_complete_for_configured_period"]) is True


def test_main_generates_expected_output_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        run_dir = td / "iwr_output" / "run_test"
        iwr_dir = run_dir / "IWR"
        etx_dir = run_dir / "ETx"
        eta_dir = run_dir / "ETa_stress"
        static_dir = run_dir / "Static"

        iwr_dir.mkdir(parents=True)
        etx_dir.mkdir(parents=True)
        eta_dir.mkdir(parents=True)
        static_dir.mkdir(parents=True)

        profile = build_profile(1, 1)

        # Two days to ensure annual and daily outputs are non-empty.
        for d in ["20260101", "20260102"]:
            write_single_band_tif(iwr_dir / f"iwr_{d}.tif", np.array([[5.0]], dtype=np.float32), profile)
            write_single_band_tif(etx_dir / f"etx_{d}.tif", np.array([[7.0]], dtype=np.float32), profile)
            write_single_band_tif(eta_dir / f"eta_stress_{d}.tif", np.array([[6.0]], dtype=np.float32), profile)

        # Static support rasters.
        write_single_band_tif(static_dir / "iwr_analysis_area_fraction.tif", np.array([[0.2]], dtype=np.float32), profile)
        write_single_band_tif(td / "cell_area.tif", np.array([[1000000.0]], dtype=np.float32), profile)

        cfg = {
            "time": {"start_date": "2026-01-01", "end_date": "2026-01-02"},
            "outputs": {"output_base": str(td / "iwr_output"), "run_name": "run_test"},
            "options": {
                "iwr_mode": "theoretical_net_irrigation",
                "iwr_domain": "all_cropped",
                "theoretical_iwr_target": "stress_threshold",
            },
        }
        cfg_path = td / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        original_cfg = plots.MODEL_CONFIG_FILE
        original_area = plots.CELL_AREA_M2_FILE
        original_out = plots.OUTPUT_DIR

        try:
            plots.MODEL_CONFIG_FILE = cfg_path
            plots.CELL_AREA_M2_FILE = td / "cell_area.tif"
            plots.OUTPUT_DIR = td / "plots_out"

            plots.main()

            expected_files = [
                plots.OUTPUT_DIR / "daily_iwr_summary.csv",
                plots.OUTPUT_DIR / "annual_iwr_summary.csv",
                plots.OUTPUT_DIR / "daily_iwr_timeseries.png",
                plots.OUTPUT_DIR / "yearly_iwr.png",
                plots.OUTPUT_DIR / "daily_et_comparison.png",
                plots.OUTPUT_DIR / "yearly_et_components.png",
            ]

            for output_file in expected_files:
                assert output_file.exists(), f"Missing expected output: {output_file}"
        finally:
            plots.MODEL_CONFIG_FILE = original_cfg
            plots.CELL_AREA_M2_FILE = original_area
            plots.OUTPUT_DIR = original_out
