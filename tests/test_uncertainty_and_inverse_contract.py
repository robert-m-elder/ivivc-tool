from __future__ import annotations

import numpy as np

from engine import _inverse_response_times
from models.exponential import model_function as exponential_model
from utilities.evaluation import generate_confidence_bands, generate_prediction_bands


def test_confidence_band_contract_returns_pointwise_summary_arrays():
    np.random.seed(12345)
    x = np.array([0.0, 1.0, 2.0])
    result = generate_confidence_bands(
        exponential_model,
        x,
        {'a': 1.0, 'b': -0.2},
        np.diag([1e-4, 1e-4]),
        n_samples=200,
    )

    assert set(result) == {'lower', 'upper', 'mean'}
    assert all(np.asarray(result[key]).shape == x.shape for key in result)
    assert np.all(result['lower'] <= result['mean'])
    assert np.all(result['mean'] <= result['upper'])


def test_legacy_prediction_band_name_now_returns_parameter_only_summary():
    np.random.seed(12345)
    x = np.array([0.0, 1.0, 2.0])
    result = generate_prediction_bands(
        exponential_model,
        x,
        {'a': 1.0, 'b': -0.2},
        np.diag([1e-4, 1e-4]),
        residuals=np.array([0.02, -0.01, 0.01, -0.02, 0.0, 0.01]),
        n_samples=200,
    )

    assert set(result) == {'lower', 'upper', 'mean', 'std'}
    assert all(np.asarray(result[key]).shape == x.shape for key in result)
    assert np.all(result['std'] >= 0)
    assert np.all(result['lower'] <= result['mean'])
    assert np.all(result['mean'] <= result['upper'])


def test_inverse_response_mapping_is_bounded_by_observed_calibration_range():
    targets = np.array([1.0, np.exp(-0.5), np.exp(-1.0), 1.2, np.nan])
    mapped, status, reason = _inverse_response_times(
        exponential_model,
        {'a': 1.0, 'b': -0.1},
        targets,
        (0.0, 10.0),
        observed_responses=np.exp(-0.1 * np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])),
    )

    assert reason is None
    np.testing.assert_allclose(mapped[:3], np.array([0.0, 5.0, 10.0]), atol=1e-7)
    assert np.isnan(mapped[3])
    assert np.isnan(mapped[4])
    assert status.tolist() == [
        'mapped',
        'mapped',
        'mapped',
        'outside observed in vivo response range',
        'nonfinite input response',
    ]
