import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import (  # noqa: E402
    DRAINAGE_SCHEME_FAO56_EXCESS_ABOVE_FIELD_CAPACITY,
    DRAINAGE_SCHEME_WATNEEDS_LINEAR,
    IWR_MODE_THEORETICAL_NET_IRRIGATION,
    IWR_MODE_WATNEEDS_BLUE_ET,
    apply_theoretical_irrigation_to_target,
    compute_deep_percolation,
    compute_deep_percolation_watneeds_linear,
    compute_fao56_natural_root_zone_balance,
    normalize_drainage_scheme,
)

NODATA = -9999.0


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _s(v):
    """1×1 float32 array."""
    return np.array([[v]], dtype=np.float32)


def _b(v):
    """1×1 bool array."""
    return np.array([[v]])


# --------------------------------------------------------------------------- #
# A. Legacy watneeds regression                                                 #
# --------------------------------------------------------------------------- #

def test_watneeds_linear_regression():
    """
    S=200, RAW=90, TAW=204, Fmax=262
    theoretical = 262*(200-90)/(204-90) ≈ 252.8
    available   = 200 - 90 = 110
    expected    = min(252.8, 110) = 110
    """
    dp = compute_deep_percolation_watneeds_linear(
        soil_moisture_previous=_s(200.0),
        raw=_s(90.0),
        total_available_water_pixel=_s(204.0),
        fmax_pixel=_s(262.0),
        nodata=NODATA,
    )
    assert np.isclose(dp[0, 0], 110.0, atol=1e-4)


def test_compute_deep_percolation_wrapper_matches_linear():
    """Backward-compat wrapper must return identical value."""
    kwargs = dict(
        soil_moisture_previous=_s(200.0),
        raw=_s(90.0),
        total_available_water_pixel=_s(204.0),
        fmax_pixel=_s(262.0),
        nodata=NODATA,
    )
    dp_named = compute_deep_percolation_watneeds_linear(**kwargs)
    dp_wrap = compute_deep_percolation(**kwargs)
    assert np.isclose(dp_named[0, 0], dp_wrap[0, 0])


# --------------------------------------------------------------------------- #
# B. FAO-56 rainfall retained below field capacity                              #
# --------------------------------------------------------------------------- #

def test_fao56_retained_below_fc():
    """
    S=90, P=109.25, ET=5, TAW=204
    provisional = 194.25  →  dp = 0,  bounded = 194.25
    """
    dp, runoff, pre_irr, bounded, prov = compute_fao56_natural_root_zone_balance(
        soil_moisture_previous=_s(90.0),
        precipitation_effective=_s(109.25),
        evapotranspiration=_s(5.0),
        total_available_water_pixel=_s(204.0),
        nodata=NODATA,
    )
    assert np.isclose(prov[0, 0], 194.25, atol=1e-4)
    assert np.isclose(dp[0, 0], 0.0, atol=1e-4)
    assert np.isclose(bounded[0, 0], 194.25, atol=1e-4)
    assert np.isclose(runoff[0, 0], 0.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# C. FAO-56 excess above field capacity                                         #
# --------------------------------------------------------------------------- #

def test_fao56_excess_above_fc():
    """
    S=204, P=109.25, ET=5, TAW=204
    provisional = 308.25  →  dp = 104.25,  bounded = 204
    """
    dp, runoff, pre_irr, bounded, prov = compute_fao56_natural_root_zone_balance(
        soil_moisture_previous=_s(204.0),
        precipitation_effective=_s(109.25),
        evapotranspiration=_s(5.0),
        total_available_water_pixel=_s(204.0),
        nodata=NODATA,
    )
    assert np.isclose(prov[0, 0], 308.25, atol=1e-4)
    assert np.isclose(dp[0, 0], 104.25, atol=1e-4)
    assert np.isclose(bounded[0, 0], 204.0, atol=1e-4)
    assert np.isclose(runoff[0, 0], 0.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# D. Dry active day at stress threshold — demand_mask True                      #
# --------------------------------------------------------------------------- #

def test_dry_day_demand_true():
    """
    target=91.8, S=91.8, P=0, ET=5, TAW=204
    pre_irr = 86.8  →  irrigation = 5,  final = 91.8
    """
    _, _, pre_irr, _, _ = compute_fao56_natural_root_zone_balance(
        soil_moisture_previous=_s(91.8),
        precipitation_effective=_s(0.0),
        evapotranspiration=_s(5.0),
        total_available_water_pixel=_s(204.0),
        nodata=NODATA,
    )
    assert np.isclose(pre_irr[0, 0], 86.8, atol=1e-4)

    irr, final_s = apply_theoretical_irrigation_to_target(
        pre_irrigation_storage=pre_irr,
        target_storage=_s(91.8),
        total_available_water_pixel=_s(204.0),
        demand_mask=_b(True),
        nodata=NODATA,
    )
    assert np.isclose(irr[0, 0], 5.0, atol=1e-4)
    assert np.isclose(final_s[0, 0], 91.8, atol=1e-4)


# --------------------------------------------------------------------------- #
# E. Same dry day — demand_mask False                                           #
# --------------------------------------------------------------------------- #

def test_dry_day_demand_false():
    """
    Same conditions as D but demand_mask=False.
    irrigation = 0,  final = 86.8
    """
    _, _, pre_irr, _, _ = compute_fao56_natural_root_zone_balance(
        soil_moisture_previous=_s(91.8),
        precipitation_effective=_s(0.0),
        evapotranspiration=_s(5.0),
        total_available_water_pixel=_s(204.0),
        nodata=NODATA,
    )
    irr, final_s = apply_theoretical_irrigation_to_target(
        pre_irrigation_storage=pre_irr,
        target_storage=_s(91.8),
        total_available_water_pixel=_s(204.0),
        demand_mask=_b(False),
        nodata=NODATA,
    )
    assert np.isclose(irr[0, 0], 0.0, atol=1e-4)
    assert np.isclose(final_s[0, 0], 86.8, atol=1e-4)


# --------------------------------------------------------------------------- #
# F. Storage below zero before irrigation                                       #
# --------------------------------------------------------------------------- #

def test_pre_irrigation_storage_negative():
    """
    target=91.8, S=0, P=0, ET=5, TAW=204, demand_mask=True
    pre_irr = -5  →  irrigation = 96.8,  final = 91.8
    """
    _, _, pre_irr, _, _ = compute_fao56_natural_root_zone_balance(
        soil_moisture_previous=_s(0.0),
        precipitation_effective=_s(0.0),
        evapotranspiration=_s(5.0),
        total_available_water_pixel=_s(204.0),
        nodata=NODATA,
    )
    assert np.isclose(pre_irr[0, 0], -5.0, atol=1e-4)

    irr, final_s = apply_theoretical_irrigation_to_target(
        pre_irrigation_storage=pre_irr,
        target_storage=_s(91.8),
        total_available_water_pixel=_s(204.0),
        demand_mask=_b(True),
        nodata=NODATA,
    )
    assert np.isclose(irr[0, 0], 96.8, atol=1e-4)
    assert np.isclose(final_s[0, 0], 91.8, atol=1e-4)


# --------------------------------------------------------------------------- #
# normalize_drainage_scheme                                                     #
# --------------------------------------------------------------------------- #

def test_normalize_auto_watneeds():
    scheme = normalize_drainage_scheme(IWR_MODE_WATNEEDS_BLUE_ET, "auto")
    assert scheme == DRAINAGE_SCHEME_WATNEEDS_LINEAR


def test_normalize_none_theoretical():
    scheme = normalize_drainage_scheme(IWR_MODE_THEORETICAL_NET_IRRIGATION, None)
    assert scheme == DRAINAGE_SCHEME_FAO56_EXCESS_ABOVE_FIELD_CAPACITY


def test_normalize_explicit_override():
    scheme = normalize_drainage_scheme(
        IWR_MODE_THEORETICAL_NET_IRRIGATION,
        DRAINAGE_SCHEME_WATNEEDS_LINEAR,
    )
    assert scheme == DRAINAGE_SCHEME_WATNEEDS_LINEAR


def test_normalize_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported drainage_scheme"):
        normalize_drainage_scheme(IWR_MODE_WATNEEDS_BLUE_ET, "invalid_scheme")


# --------------------------------------------------------------------------- #
# G. 2-day sequence: rainfall retained on following dry day                   #
# --------------------------------------------------------------------------- #

def test_fao56_two_day_sequence_rainfall_retained():
        """
        Demonstrate that rainfall absorbed in Day 1 remains available on Day 2.

        Day 1: S=90, P=109.25, ET=5, TAW=204
            provisional = 90 + 109.25 - 5 = 194.25
            deep_percolation = max(194.25 - 204, 0) = 0
            next storage = 194.25                           (all rain retained)

        Day 2: S=194.25, P=0, ET=5, TAW=204
            provisional = 194.25 + 0 - 5 = 189.25
            deep_percolation = max(189.25 - 204, 0) = 0
            next storage = 189.25                           (still no drainage)

        The WATNEEDS-linear scheme would drain back to ~RAW on Day 1 because
        Fmax * (S - RAW) / (TAW - RAW) > 0 once S > RAW.  This test verifies
        that the FAO-56 scheme does NOT do that.
        """
        taw = 204.0
        et = 5.0
        mask = np.array([[True]])

        # ---------- Day 1 ----------
        dp1, _r1, pre_irr1, bounded1, prov1 = compute_fao56_natural_root_zone_balance(
                soil_moisture_previous=np.array([[90.0]], dtype=np.float32),
                precipitation_effective=np.array([[109.25]], dtype=np.float32),
                evapotranspiration=np.array([[et]], dtype=np.float32),
                total_available_water_pixel=np.array([[taw]], dtype=np.float32),
                nodata=NODATA,
                valid_mask=mask,
        )
        assert np.isclose(prov1[0, 0], 194.25, atol=1e-4), f"Day 1 provisional={prov1[0,0]}"
        assert np.isclose(dp1[0, 0], 0.0, atol=1e-4), f"Day 1 drainage={dp1[0,0]} (should be 0)"
        assert np.isclose(bounded1[0, 0], 194.25, atol=1e-4), f"Day 1 storage={bounded1[0,0]}"

        # ---------- Day 2 (uses Day 1 output as initial storage) ----------
        dp2, _r2, pre_irr2, bounded2, prov2 = compute_fao56_natural_root_zone_balance(
                soil_moisture_previous=bounded1,                               # 194.25
                precipitation_effective=np.array([[0.0]], dtype=np.float32),
                evapotranspiration=np.array([[et]], dtype=np.float32),
                total_available_water_pixel=np.array([[taw]], dtype=np.float32),
                nodata=NODATA,
                valid_mask=mask,
        )
        assert np.isclose(prov2[0, 0], 189.25, atol=1e-4), f"Day 2 provisional={prov2[0,0]}"
        assert np.isclose(dp2[0, 0], 0.0, atol=1e-4), f"Day 2 drainage={dp2[0,0]} (should be 0)"
        assert np.isclose(bounded2[0, 0], 189.25, atol=1e-4), f"Day 2 storage={bounded2[0,0]}"
