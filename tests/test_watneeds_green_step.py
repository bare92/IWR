import numpy as np

from iwr_processing.iwr_core_process import watneeds_green_step


def test_no_stress_condition():
    result = watneeds_green_step(
        s_prev_mm=np.array([[80.0]], dtype=np.float32),
        precipitation_mm=np.array([[0.0]], dtype=np.float32),
        etc_mm=np.array([[5.0]], dtype=np.float32),
        taw_mm=np.array([[100.0]], dtype=np.float32),
        p=0.5,
        fmax_mm_day=np.array([[0.0]], dtype=np.float32),
        irrigated_mask=np.array([[True]]),
    )

    assert np.isclose(result["ks"][0, 0], 1.0)
    assert np.isclose(result["green_et_mm"][0, 0], 5.0)
    assert np.isclose(result["blue_water_mm"][0, 0], 0.0)


def test_stress_condition_and_blue_water_definition():
    result = watneeds_green_step(
        s_prev_mm=np.array([[20.0, 20.0]], dtype=np.float32),
        precipitation_mm=np.array([[0.0, 0.0]], dtype=np.float32),
        etc_mm=np.array([[10.0, 10.0]], dtype=np.float32),
        taw_mm=np.array([[100.0, 100.0]], dtype=np.float32),
        p=0.5,
        fmax_mm_day=np.array([[0.0, 0.0]], dtype=np.float32),
        irrigated_mask=np.array([[True, False]]),
    )

    assert np.isclose(result["ks"][0, 0], 0.4)
    assert np.isclose(result["green_et_mm"][0, 0], 4.0)
    assert np.isclose(result["blue_water_mm"][0, 0], 6.0)
    assert np.isclose(result["blue_water_mm"][0, 1], 0.0)


def test_deep_percolation_and_overflow_runoff():
    result = watneeds_green_step(
        s_prev_mm=np.array([[100.0]], dtype=np.float32),
        precipitation_mm=np.array([[10.0]], dtype=np.float32),
        etc_mm=np.array([[0.0]], dtype=np.float32),
        taw_mm=np.array([[100.0]], dtype=np.float32),
        p=0.5,
        fmax_mm_day=np.array([[8.0]], dtype=np.float32),
    )

    assert np.isclose(result["deep_perc_mm"][0, 0], 8.0)
    assert np.isclose(result["overflow_runoff_mm"][0, 0], 1.5)
    assert np.isclose(result["surface_runoff_mm"][0, 0], 0.5)
    assert np.isclose(result["s_next_mm"][0, 0], 100.0)


def test_negative_balance_scaling_and_mass_balance():
    result = watneeds_green_step(
        s_prev_mm=np.array([[1.0]], dtype=np.float32),
        precipitation_mm=np.array([[0.0]], dtype=np.float32),
        etc_mm=np.array([[100.0]], dtype=np.float32),
        taw_mm=np.array([[100.0]], dtype=np.float32),
        p=0.5,
        fmax_mm_day=np.array([[100.0]], dtype=np.float32),
    )

    total_in = 1.0
    total_out = (
        result["green_et_mm"][0, 0]
        + result["deep_perc_mm"][0, 0]
        + result["s_next_mm"][0, 0]
        + result["overflow_runoff_mm"][0, 0]
    )
    assert np.isclose(total_out, total_in)
    assert np.isclose(result["s_next_mm"][0, 0], 0.0)


def test_valid_mask_zeroes_outputs():
    result = watneeds_green_step(
        s_prev_mm=np.array([[80.0]], dtype=np.float32),
        precipitation_mm=np.array([[1.0]], dtype=np.float32),
        etc_mm=np.array([[5.0]], dtype=np.float32),
        taw_mm=np.array([[100.0]], dtype=np.float32),
        p=0.5,
        fmax_mm_day=np.array([[2.0]], dtype=np.float32),
        valid_mask=np.array([[False]]),
    )

    for key in ("s_next_mm", "green_et_mm", "deep_perc_mm", "blue_water_mm", "ks", "total_runoff_mm"):
        assert np.isclose(result[key][0, 0], 0.0)