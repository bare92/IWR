import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_model import (  # noqa: E402
    IWR_DOMAIN_ALL_CROPPED,
    IWR_DOMAIN_IRRIGATED,
    IWR_DOMAIN_NON_IRRIGATED,
    IWR_MODE_THEORETICAL_NET_IRRIGATION,
    IWR_MODE_WATNEEDS_BLUE_ET,
    compute_theoretical_net_irrigation_requirement,
    create_iwr_domain_mask,
    normalize_iwr_configuration,
    water_balance_step,
)

NODATA = -9999.0


def test_theoretical_requirement_function():
    previous = np.array([[50.0]], dtype=np.float32)
    target = np.array([[50.0]], dtype=np.float32)
    precip = np.array([[0.0]], dtype=np.float32)
    etc = np.array([[10.0]], dtype=np.float32)
    dp = np.array([[0.0]], dtype=np.float32)
    runoff = np.array([[0.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)
    valid = np.array([[True]])

    iwr, s_new, s_pre = compute_theoretical_net_irrigation_requirement(
        soil_moisture_previous=previous,
        precipitation_effective=precip,
        potential_evapotranspiration=etc,
        deep_percolation=dp,
        runoff=runoff,
        target_storage=target,
        total_available_water_pixel=taw,
        demand_mask=valid,
        nodata=NODATA,
        valid_mask=valid,
    )

    assert np.isclose(s_pre[0, 0], 40.0)
    assert np.isclose(iwr[0, 0], 10.0)
    assert np.isclose(s_new[0, 0], 50.0)


def test_no_repeated_deficit_and_state_separation():
    target = np.array([[50.0]], dtype=np.float32)
    taw = np.array([[200.0]], dtype=np.float32)
    precip = np.array([[0.0]], dtype=np.float32)
    etc = np.array([[10.0]], dtype=np.float32)
    dp = np.array([[0.0]], dtype=np.float32)
    runoff = np.array([[0.0]], dtype=np.float32)
    mask = np.array([[True]])

    s_ref = target.copy()
    s_actual = target.copy()
    cumulative_iwr = 0.0

    for _ in range(3):
        iwr, s_ref, _ = compute_theoretical_net_irrigation_requirement(
            soil_moisture_previous=s_ref,
            precipitation_effective=precip,
            potential_evapotranspiration=etc,
            deep_percolation=dp,
            runoff=runoff,
            target_storage=target,
            total_available_water_pixel=taw,
            demand_mask=mask,
            nodata=NODATA,
            valid_mask=mask,
        )
        cumulative_iwr += float(iwr[0, 0])

        zero_irrigation = np.array([[0.0]], dtype=np.float32)
        s_actual = water_balance_step(
            soil_moisture_previous=s_actual,
            precipitation_effective=precip,
            actual_evapotranspiration=etc,
            deep_percolation=dp,
            runoff=runoff,
            irrigation=zero_irrigation,
            total_available_water_pixel=taw,
            nodata=NODATA,
            valid_mask=mask,
        )

        assert np.isclose(s_ref[0, 0], 50.0, atol=1e-6)

    assert np.isclose(cumulative_iwr, 30.0, atol=1e-6)
    assert np.isclose(s_actual[0, 0], 20.0, atol=1e-6)


def test_replacement_of_non_et_losses():
    previous = np.array([[50.0]], dtype=np.float32)
    target = np.array([[50.0]], dtype=np.float32)
    precip = np.array([[0.0]], dtype=np.float32)
    etc = np.array([[5.0]], dtype=np.float32)
    dp = np.array([[10.0]], dtype=np.float32)
    runoff = np.array([[0.0]], dtype=np.float32)
    taw = np.array([[500.0]], dtype=np.float32)
    valid = np.array([[True]])

    iwr, _, pre = compute_theoretical_net_irrigation_requirement(
        soil_moisture_previous=previous,
        precipitation_effective=precip,
        potential_evapotranspiration=etc,
        deep_percolation=dp,
        runoff=runoff,
        target_storage=target,
        total_available_water_pixel=taw,
        demand_mask=valid,
        nodata=NODATA,
        valid_mask=valid,
    )

    assert np.isclose(pre[0, 0], 35.0)
    assert np.isclose(iwr[0, 0], 15.0)


def test_domain_masking():
    valid_area = np.array(
        [[True, True, True, False]],
        dtype=bool,
    )
    irrigation_mask = np.array(
        [[1.0, 0.0, -9999.0, 1.0]],
        dtype=np.float32,
    )

    all_cropped = create_iwr_domain_mask(
        iwr_domain=IWR_DOMAIN_ALL_CROPPED,
        valid_area_pixels=valid_area,
        irrigation_mask=None,
        nodata=NODATA,
    )
    assert np.array_equal(all_cropped, valid_area)

    irrigated = create_iwr_domain_mask(
        iwr_domain=IWR_DOMAIN_IRRIGATED,
        valid_area_pixels=valid_area,
        irrigation_mask=irrigation_mask,
        nodata=NODATA,
    )
    assert np.array_equal(irrigated, np.array([[True, False, False, False]]))

    non_irrigated = create_iwr_domain_mask(
        iwr_domain=IWR_DOMAIN_NON_IRRIGATED,
        valid_area_pixels=valid_area,
        irrigation_mask=irrigation_mask,
        nodata=NODATA,
    )
    assert np.array_equal(non_irrigated, np.array([[False, True, False, False]]))


def test_configuration_validation():
    ok_mode, ok_domain, ok_target = normalize_iwr_configuration(
        iwr_mode=IWR_MODE_THEORETICAL_NET_IRRIGATION,
        iwr_domain=None,
        theoretical_iwr_target="stress_threshold",
    )
    assert ok_mode == IWR_MODE_THEORETICAL_NET_IRRIGATION
    assert ok_domain == IWR_DOMAIN_ALL_CROPPED
    assert ok_target == "stress_threshold"

    try:
        normalize_iwr_configuration(
            iwr_mode="bad_mode",
            iwr_domain=None,
            theoretical_iwr_target="stress_threshold",
        )
        raise AssertionError("Expected ValueError for bad iwr_mode")
    except ValueError as exc:
        assert "Unsupported iwr_mode" in str(exc)

    try:
        normalize_iwr_configuration(
            iwr_mode=IWR_MODE_WATNEEDS_BLUE_ET,
            iwr_domain=IWR_DOMAIN_NON_IRRIGATED,
            theoretical_iwr_target="stress_threshold",
        )
        raise AssertionError("Expected ValueError for incompatible mode/domain")
    except ValueError as exc:
        assert "only supports iwr_domain='irrigated'" in str(exc)

    try:
        normalize_iwr_configuration(
            iwr_mode=IWR_MODE_THEORETICAL_NET_IRRIGATION,
            iwr_domain=IWR_DOMAIN_ALL_CROPPED,
            theoretical_iwr_target="bad_target",
        )
        raise AssertionError("Expected ValueError for bad theoretical_iwr_target")
    except ValueError as exc:
        assert "Unsupported theoretical_iwr_target" in str(exc)


def main():
    test_theoretical_requirement_function()
    test_no_repeated_deficit_and_state_separation()
    test_replacement_of_non_et_losses()
    test_domain_masking()
    test_configuration_validation()
    print("smoke_test_theoretical_iwr.py: PASS")


if __name__ == "__main__":
    main()
