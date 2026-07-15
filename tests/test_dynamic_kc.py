"""
Standalone tests for the dynamic FAO-56 Kc curve.

No external raster files required. Uses tiny numpy arrays.
Run with:  python tests/test_dynamic_kc.py
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simple_code"))

from phenology_functions import (
    _calculate_dynamic_kc_for_one_season,
    create_dynamic_kc_curve_from_date,
    date_to_continuous_dekad,
)
from iwr_model import create_kc_pixel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KC_INI = 0.3
KC_MID = 1.1
KC_END = 0.5
NODATA = -9999.0


def _scalar_phenology(sos, tom, sen, eos, nseasons=1):
    """Build a 1x1 phenology dict with given scalar values."""
    ones = np.ones((1, 1), dtype=np.float32)
    return {
        "phenonseasons": np.full((1, 1), nseasons, dtype=np.float32),
        "phenos1":    ones * sos,
        "phenom1":    ones * tom,
        "phenosen1":  ones * sen,
        "phenoe1":    ones * eos,
        "phenos2":    ones * sos,
        "phenom2":    ones * tom,
        "phenosen2":  ones * sen,
        "phenoe2":    ones * eos,
    }


def _kc_for_dekad_position(pos, sos, tom, sen, eos):
    """Direct helper using _calculate_dynamic_kc_for_one_season at a float position."""
    shape = (1, 1)
    kc, active = _calculate_dynamic_kc_for_one_season(
        current_dekad_position=pos,
        sos=np.full(shape, sos, dtype=np.float32),
        tom=np.full(shape, tom, dtype=np.float32),
        sen=np.full(shape, sen, dtype=np.float32),
        eos=np.full(shape, eos, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
        kc_ini=KC_INI,
        kc_mid=KC_MID,
        kc_end=KC_END,
    )
    return float(kc[0, 0]), bool(active[0, 0])


# ---------------------------------------------------------------------------
# Test 1: before SOS
# ---------------------------------------------------------------------------
def test_before_sos():
    kc, active = _kc_for_dekad_position(pos=4.0, sos=6, tom=12, sen=18, eos=24)
    assert not active, "Should be inactive before SOS"
    assert kc == 0.0, f"Kc before SOS must be 0, got {kc}"


# ---------------------------------------------------------------------------
# Test 2: at SOS (first instant)
# ---------------------------------------------------------------------------
def test_at_sos():
    kc, active = _kc_for_dekad_position(pos=6.0, sos=6, tom=12, sen=18, eos=24)
    assert active, "Should be active at SOS"
    assert abs(kc - KC_INI) < 1e-5, f"Kc at SOS must be Kc_ini={KC_INI}, got {kc}"


# ---------------------------------------------------------------------------
# Test 3: between SOS and TOM -> Kc between Kc_ini and Kc_mid
# ---------------------------------------------------------------------------
def test_development_phase():
    kc, active = _kc_for_dekad_position(pos=9.0, sos=6, tom=12, sen=18, eos=24)
    assert active
    assert KC_INI < kc < KC_MID, f"Dev-phase Kc={kc} not in ({KC_INI}, {KC_MID})"


# ---------------------------------------------------------------------------
# Test 4: at TOM -> Kc_mid
# ---------------------------------------------------------------------------
def test_at_tom():
    kc, active = _kc_for_dekad_position(pos=12.0, sos=6, tom=12, sen=18, eos=24)
    assert active
    assert abs(kc - KC_MID) < 1e-5, f"Kc at TOM must be Kc_mid={KC_MID}, got {kc}"


# ---------------------------------------------------------------------------
# Test 5: between TOM and SEN -> Kc_mid
# ---------------------------------------------------------------------------
def test_mid_season():
    kc, active = _kc_for_dekad_position(pos=15.0, sos=6, tom=12, sen=18, eos=24)
    assert active
    assert abs(kc - KC_MID) < 1e-5, f"Mid-season Kc must be Kc_mid={KC_MID}, got {kc}"


# ---------------------------------------------------------------------------
# Test 6: at SEN -> late-season interpolation starts from Kc_mid
# ---------------------------------------------------------------------------
def test_at_sen():
    kc, active = _kc_for_dekad_position(pos=18.0, sos=6, tom=12, sen=18, eos=24)
    assert active
    # progress = (18 - 18) / (25 - 18) = 0 -> Kc_mid
    assert abs(kc - KC_MID) < 1e-5, f"Kc at SEN start must be Kc_mid={KC_MID}, got {kc}"


# ---------------------------------------------------------------------------
# Test 7: during late season -> Kc moves toward Kc_end
# ---------------------------------------------------------------------------
def test_late_season():
    kc, active = _kc_for_dekad_position(pos=21.0, sos=6, tom=12, sen=18, eos=24)
    assert active
    assert KC_END < kc < KC_MID, f"Late-season Kc={kc} not between Kc_end and Kc_mid"


# ---------------------------------------------------------------------------
# Test 8: after EOS dekad -> Kc = 0
# ---------------------------------------------------------------------------
def test_after_eos():
    # pos >= eos + 1 is outside the window
    kc, active = _kc_for_dekad_position(pos=25.0, sos=6, tom=12, sen=18, eos=24)
    assert not active, "Should be inactive after EOS"
    assert kc == 0.0, f"Kc after EOS must be 0, got {kc}"


# ---------------------------------------------------------------------------
# Test 9: second-season position using +36 offset
# ---------------------------------------------------------------------------
def test_second_season_offset():
    # Season stored at 1-108 scale; place it at dekad 40-70 (in +36 range)
    sos, tom, sen, eos = 40, 50, 60, 70
    # Calendar dekad 4 + 36 = 40  -> at SOS of that season
    kc, active = _kc_for_dekad_position(pos=4.0, sos=sos, tom=tom, sen=sen, eos=eos)
    assert active, "Should be active via +36 offset"
    assert abs(kc - KC_INI) < 1e-5, f"At SOS via offset, Kc must be Kc_ini={KC_INI}, got {kc}"


# ---------------------------------------------------------------------------
# Test 10: crop-fraction-weighted average
# ---------------------------------------------------------------------------
def test_crop_fraction_weighted_average():
    """
    Two crops, each with 1x1 shape.
    Crop 0: fraction=0.4, Kc_ini=0.3, Kc_mid=1.1, Kc_end=0.5
    Crop 1: fraction=0.6, Kc_ini=0.5, Kc_mid=1.3, Kc_end=0.7

    Place both at mid-season so each crop Kc = Kc_mid.
    Expected kc_pixel = (0.4 * Kc_mid_0 + 0.6 * Kc_mid_1) / (0.4 + 0.6)
    """
    import pandas as pd

    kc_mid_0 = 1.1
    kc_mid_1 = 1.3
    f0, f1 = 0.4, 0.6

    crop_df = pd.DataFrame({
        "crop_name": ["crop_a", "crop_b"],
        "Kc_ini":    [0.3,     0.5],
        "Kc_mid":    [kc_mid_0, kc_mid_1],
        "Kc_end":    [0.5,     0.7],
        "root_depth_max_m": [1.0, 1.0],
        "p":         [0.5,     0.5],
    })

    # SOS=6, TOM=12, SEN=18, EOS=24; dekad 15 is mid-season
    pheno = _scalar_phenology(sos=6, tom=12, sen=18, eos=24, nseasons=1)

    # crop_fraction_data shape: (2, 1, 1)
    crop_fraction_data = np.array([[[f0]], [[f1]]], dtype=np.float32)

    # dekad 15 is mid-season: day 21 of month 5 -> dekad (5-1)*3+3 = 15
    current = date(2020, 5, 21)
    assert date_to_continuous_dekad(current) == 15.0

    kc_pixel = create_kc_pixel(
        current_date=current,
        phenology=pheno,
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
        nodata=NODATA,
    )

    expected = (f0 * kc_mid_0 + f1 * kc_mid_1) / (f0 + f1)
    assert abs(float(kc_pixel[0, 0]) - expected) < 1e-4, (
        f"Weighted Kc={float(kc_pixel[0,0]):.6f} != expected={expected:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 11: nodata / invalid phenology -> no NaN, no inf, no warnings
# ---------------------------------------------------------------------------
def test_nodata_safe():
    import warnings

    shape = (2, 2)
    pheno = {
        "phenonseasons": np.array([[1, NODATA], [1, 1]], dtype=np.float32),
        "phenos1":   np.array([[6,  NODATA], [6, np.nan]], dtype=np.float32),
        "phenom1":   np.array([[12, NODATA], [12, 12]], dtype=np.float32),
        "phenosen1": np.array([[18, NODATA], [18, 18]], dtype=np.float32),
        "phenoe1":   np.array([[24, NODATA], [24, 24]], dtype=np.float32),
        "phenos2":   np.zeros(shape, dtype=np.float32),
        "phenom2":   np.zeros(shape, dtype=np.float32),
        "phenosen2": np.zeros(shape, dtype=np.float32),
        "phenoe2":   np.zeros(shape, dtype=np.float32),
    }
    current = date(2020, 5, 21)  # mid-season for valid pixels

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kc = create_dynamic_kc_curve_from_date(
            current_date=current,
            phenology=pheno,
            kc_ini=KC_INI,
            kc_mid=KC_MID,
            kc_end=KC_END,
            nodata=NODATA,
            inactive_kc=0.0,
        )

    assert np.all(np.isfinite(kc)), f"NaN or Inf in output: {kc}"
    assert kc.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 12: output dtype is float32
# ---------------------------------------------------------------------------
def test_output_dtype():
    current = date(2020, 5, 21)
    pheno = _scalar_phenology(sos=6, tom=12, sen=18, eos=24)
    kc = create_dynamic_kc_curve_from_date(
        current_date=current,
        phenology=pheno,
        kc_ini=KC_INI,
        kc_mid=KC_MID,
        kc_end=KC_END,
        nodata=NODATA,
    )
    assert kc.dtype == np.float32, f"Expected float32, got {kc.dtype}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_before_sos,
        test_at_sos,
        test_development_phase,
        test_at_tom,
        test_mid_season,
        test_at_sen,
        test_late_season,
        test_after_eos,
        test_second_season_offset,
        test_crop_fraction_weighted_average,
        test_nodata_safe,
        test_output_dtype,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
