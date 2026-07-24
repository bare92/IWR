import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "IWR_anomaly_script" / "compute_iwr_anomalies_gamma.py"
SPEC = importlib.util.spec_from_file_location("compute_iwr_anomalies_gamma", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fit_gamma_climatology_arrays_tracks_zero_probability_and_fit():
    samples = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )

    total_count = np.full(3, samples.shape[0], dtype=np.uint16)
    zero_count = np.sum(samples == 0.0, axis=0).astype(np.uint16)
    positive_count = np.sum(samples > 0.0, axis=0).astype(np.uint16)
    positive_sum = np.sum(np.where(samples > 0.0, samples, 0.0), axis=0)
    positive_safe = np.where(samples > 0.0, samples, 1.0)
    positive_log_sum = np.sum(np.where(samples > 0.0, np.log(positive_safe), 0.0), axis=0)
    positive_sq_sum = np.sum(np.where(samples > 0.0, samples * samples, 0.0), axis=0)

    shape, scale, pzero = MODULE.fit_gamma_climatology_arrays(
        total_count=total_count,
        zero_count=zero_count,
        positive_count=positive_count,
        positive_sum=positive_sum,
        positive_log_sum=positive_log_sum,
        positive_sq_sum=positive_sq_sum,
        minimum_years=3,
        minimum_positive_years=2,
        gamma_epsilon=1.0e-12,
    )

    assert np.isfinite(shape[0]) and shape[0] > 0.0
    assert np.isfinite(scale[0]) and scale[0] > 0.0
    assert np.isfinite(shape[1]) and shape[1] > 0.0
    assert np.isfinite(scale[1]) and scale[1] > 0.0
    assert np.isnan(shape[2])
    assert np.isnan(scale[2])
    assert np.allclose(pzero, np.array([0.0, 0.5, 1.0]))


def test_gamma_cdf_to_standard_normal_handles_zero_mass_cases():
    values = np.array([0.0, 2.0, 0.0, 3.0], dtype=np.float64)
    shape = np.array([2.0, 2.0, np.nan, np.nan], dtype=np.float64)
    scale = np.array([1.0, 1.0, np.nan, np.nan], dtype=np.float64)
    pzero = np.array([0.0, 0.25, 1.0, 1.0], dtype=np.float64)

    anomaly = MODULE.gamma_cdf_to_standard_normal(values, shape, scale, pzero, cdf_clip=1.0e-8)

    assert anomaly[0] < -5.0
    assert np.isfinite(anomaly[1])
    assert anomaly[2] == 0.0
    assert anomaly[3] > 5.0