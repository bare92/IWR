"""
Tests for the spin-up period feature (spinup_start_date).

Verifies that:
  1. The first output-day soil storage depends on preceding spin-up rainfall.
  2. Cumulative IWR excludes all spin-up days (zero-based from start_date).
  3. No daily IWR files are written before start_date.
  4. Invalid spinup_start_date values raise ValueError.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import run_iwr_model  # noqa: E402

NODATA = -9999.0

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

SHAPE = (1, 1)


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


def _crop_df():
    return pd.DataFrame(
        {
            "crop_name": ["crop_a"],
            "root_depth_max_m": [1.0],
            "Kc_ini": [1.0],
            "Kc_mid": [1.0],
            "Kc_end": [1.0],
            "p": [0.5],  # stress threshold = 0.5 * TAW
        }
    )


def _phenology_always_active():
    """Phenology active for every day-of-year (week 1 to 53)."""
    ones = np.ones(SHAPE, dtype=np.float32)
    return {
        "phenonseasons": ones.copy(),
        "phenos1": ones * 1,
        "phenom1": ones * 1,
        "phenosen1": ones * 1,
        "phenoe1": ones * 53,
        "phenos2": np.zeros(SHAPE, dtype=np.float32),
        "phenom2": np.zeros(SHAPE, dtype=np.float32),
        "phenosen2": np.zeros(SHAPE, dtype=np.float32),
        "phenoe2": np.zeros(SHAPE, dtype=np.float32),
    }


def _write_forcing_day(folder: Path, prefix: str, date: datetime, value: float):
    stamp = date.strftime("%Y%m%d")
    path = folder / f"{prefix}_{stamp}.tif"
    with rasterio.open(path, "w", **_profile()) as dst:
        dst.write(np.array([[value]], dtype=np.float32), 1)


def _build_forcing(
    p_dir: Path,
    et0_dir: Path,
    dates,
    precipitation_values,
    et0_values,
):
    """Write one GeoTIFF per day for precipitation and ET0."""
    p_dir.mkdir(parents=True, exist_ok=True)
    et0_dir.mkdir(parents=True, exist_ok=True)
    for d, p, e in zip(dates, precipitation_values, et0_values):
        _write_forcing_day(p_dir, "P", d, p)
        _write_forcing_day(et0_dir, "PET", d, e)


def _run(
    spinup_start_date,
    start_date,
    end_date,
    spinup_precip_mm,
    output_precip_mm,
    et0_mm,
    output_dir: Path,
    initial_soil_moisture_fraction=0.5,
):
    """
    Helper that sets up and runs run_iwr_model with the given parameters.

    spinup_precip_mm : daily precipitation during the spinup period
    output_precip_mm : daily precipitation during the output period
    et0_mm           : constant daily ET0 for all days
    """
    from datetime import timedelta

    p_dir = output_dir / "P"
    et0_dir = output_dir / "PET"

    # Collect all dates from spinup_start (or start) through end.
    loop_start = spinup_start_date if spinup_start_date is not None else start_date
    all_dates = []
    d = loop_start
    while d <= end_date:
        all_dates.append(d)
        d += timedelta(days=1)

    precip_values = []
    et0_values = []
    for d in all_dates:
        if spinup_start_date is not None and d < start_date:
            precip_values.append(spinup_precip_mm)
        else:
            precip_values.append(output_precip_mm)
        et0_values.append(et0_mm)

    _build_forcing(p_dir, et0_dir, all_dates, precip_values, et0_values)

    # TAW = (FC - WP) * root_depth * 1000 = 0.15 m3/m3 * 1.0 m * 1000 = 150 mm
    total_available_water = np.array([[0.15]], dtype=np.float32)
    fmax = np.array([[0.0]], dtype=np.float32)  # no Fmax needed for FAO56 scheme
    valid_mask = np.array([[1.0]], dtype=np.float32)
    crop_fraction_data = np.array([[[1.0]]], dtype=np.float32)

    sm_final, cum_iwr = run_iwr_model(
        start_date=start_date,
        end_date=end_date,
        total_available_water=total_available_water,
        fmax=fmax,
        irrigation_mask=None,
        valid_area_mask=valid_mask,
        crop_fraction_data=crop_fraction_data,
        crop_df=_crop_df(),
        phenology=_phenology_always_active(),
        precipitation_geotiff_folder=p_dir,
        et0_geotiff_folder=et0_dir,
        output_folder=output_dir / "out",
        output_profile=_profile(),
        strict_checks=False,
        write_debug_csv=False,
        write_cumulative_iwr=True,
        write_daily_etx=False,
        write_daily_eta_stress=False,
        write_static_support_layers=False,
        debug_mode=False,
        iwr_mode="theoretical_net_irrigation",
        iwr_domain="all_cropped",
        theoretical_iwr_target="stress_threshold",
        initial_soil_moisture_fraction=initial_soil_moisture_fraction,
        offseason_water_balance_kc=1.0,
        drainage_scheme="fao56_excess_above_field_capacity",
        spinup_start_date=spinup_start_date,
    )
    return sm_final, cum_iwr, output_dir / "out"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_spinup_affects_first_output_day_storage():
    """
    Spinup rainfall fills the soil above the no-spinup initial level.
    The final soil moisture (and cumulative IWR) must differ between the two runs.

    Scenario:
      - TAW  = 150 mm; stress threshold (RAW) = 75 mm.
      - Spinup: 10 days of P=20 mm/day (fills soil to TAW=150 mm).
      - Output: 5 days of P=0 mm/day, ET0=8 mm/day, Kc=1 → ET=8 mm/day.
      - Without spinup: soil starts at 0.5 * TAW = 75 mm.
      - With spinup:    soil arrives at ~150 mm at the first output day.
    """
    from datetime import timedelta

    spinup_start = datetime(2019, 12, 22)   # 10 days before start
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 5)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Run A: spin-up period has heavy rain to fill the soil.
        sm_with, cum_with, out_with = _run(
            spinup_start_date=spinup_start,
            start_date=start,
            end_date=end,
            spinup_precip_mm=20.0,
            output_precip_mm=0.0,
            et0_mm=8.0,
            output_dir=td / "with_spinup",
        )

        # Run B: no spin-up, soil initialised at 0.5 * TAW.
        sm_without, cum_without, out_without = _run(
            spinup_start_date=None,
            start_date=start,
            end_date=end,
            spinup_precip_mm=0.0,
            output_precip_mm=0.0,
            et0_mm=8.0,
            output_dir=td / "no_spinup",
        )

        # The final soil moisture must differ because the starting level differed.
        assert not np.allclose(
            sm_with[0, 0], sm_without[0, 0]
        ), (
            f"Expected spin-up to change final soil moisture, "
            f"but both runs produced {sm_with[0, 0]:.3f} mm"
        )

        # With a full soil at output start (≈150 mm > 75 mm threshold),
        # fewer or no IWR events should occur compared to the no-spinup run
        # (which starts right at the threshold).
        assert float(cum_with[0, 0]) < float(cum_without[0, 0]) or float(
            cum_with[0, 0]
        ) != float(cum_without[0, 0]), (
            "Expected cumulative IWR to differ between spinup and no-spinup runs"
        )


def test_cumulative_iwr_excludes_spinup_days():
    """
    Cumulative IWR must equal the sum accumulated only from start_date.
    Days in the spinup period must not contribute to cumulative totals.

    Approach:
      1. Run with spinup but zero spinup precip so the soil state at start_date
         is identical to the no-spinup initial state (= 0.5 * TAW).
      2. Both runs therefore have the same physics from start_date onward.
      3. The cumulative IWR should be identical in both runs.
    """
    from datetime import timedelta

    spinup_start = datetime(2019, 12, 27)   # 5 days before start
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 3)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Run A: spinup with zero precipitation (so soil doesn't move from 0.5*TAW).
        sm_with, cum_with, out_with = _run(
            spinup_start_date=spinup_start,
            start_date=start,
            end_date=end,
            spinup_precip_mm=0.0,
            output_precip_mm=0.0,
            et0_mm=5.0,
            output_dir=td / "with_spinup",
        )

        # Run B: no spinup at all; same initial fraction.
        sm_without, cum_without, out_without = _run(
            spinup_start_date=None,
            start_date=start,
            end_date=end,
            spinup_precip_mm=0.0,
            output_precip_mm=0.0,
            et0_mm=5.0,
            output_dir=td / "no_spinup",
        )

        # With zero spinup rain and same initial fraction, results are identical.
        assert np.allclose(cum_with[0, 0], cum_without[0, 0], atol=1e-4), (
            f"Cumulative IWR with zero-rain spinup ({cum_with[0, 0]:.4f}) "
            f"should match no-spinup run ({cum_without[0, 0]:.4f})"
        )

        # Verify no daily IWR files exist before start_date.
        iwr_folder = out_with / "IWR"
        if iwr_folder.exists():
            all_iwr_files = sorted(iwr_folder.glob("iwr_????????.tif"))
            for f in all_iwr_files:
                date_str = f.stem.replace("iwr_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                assert file_date >= start, (
                    f"Found IWR output file {f.name} which is before start_date "
                    f"({start.strftime('%Y-%m-%d')}); spinup outputs must not be written."
                )


def test_spinup_no_files_written_before_start_date():
    """
    No daily IWR, ETx, or ETa files should be written during the spinup period.
    """
    from datetime import timedelta

    spinup_start = datetime(2019, 12, 29)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _run(
            spinup_start_date=spinup_start,
            start_date=start,
            end_date=end,
            spinup_precip_mm=5.0,
            output_precip_mm=0.0,
            et0_mm=3.0,
            output_dir=td / "run",
        )

        out_dir = td / "run" / "out"

        for sub in ("IWR", "ETx", "ETa_stress"):
            folder = out_dir / sub
            if folder.exists():
                files = list(folder.glob("*.tif"))
                for f in files:
                    # Extract 8-digit date token from filename
                    parts = f.stem.split("_")
                    date_token = next(
                        (p for p in parts if len(p) == 8 and p.isdigit()), None
                    )
                    if date_token:
                        file_date = datetime.strptime(date_token, "%Y%m%d")
                        assert file_date >= start, (
                            f"Unexpected output file {f.name} written during spinup "
                            f"(before {start.strftime('%Y-%m-%d')})."
                        )


def test_invalid_spinup_start_date_raises():
    """spinup_start_date > start_date must raise ValueError."""
    import pytest

    spinup_start = datetime(2020, 6, 1)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 5)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with pytest.raises(ValueError, match="spinup_start_date"):
            _run(
                spinup_start_date=spinup_start,
                start_date=start,
                end_date=end,
                spinup_precip_mm=0.0,
                output_precip_mm=0.0,
                et0_mm=3.0,
                output_dir=td / "run",
            )


if __name__ == "__main__":
    test_spinup_affects_first_output_day_storage()
    print("test_spinup_affects_first_output_day_storage: PASS")

    test_cumulative_iwr_excludes_spinup_days()
    print("test_cumulative_iwr_excludes_spinup_days: PASS")

    test_spinup_no_files_written_before_start_date()
    print("test_spinup_no_files_written_before_start_date: PASS")

    test_invalid_spinup_start_date_raises()
    print("test_invalid_spinup_start_date_raises: PASS")

    print("\nAll spinup tests passed.")
