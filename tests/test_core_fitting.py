from __future__ import annotations

import numpy as np

from ivivc_core import ProfileFit, fit_profile
from models import models


def test_fit_profile_returns_reusable_numerical_result(fast_analysis_config):
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    response = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 99.0])

    result = fit_profile(
        time,
        response,
        model_name='linear_zero_intercept',
        selected_metrics=['r_squared', 'rmse'],
        analysis_config=fast_analysis_config,
        approach_id='approach1',
    )

    assert isinstance(result, ProfileFit)
    assert result.model_name == 'linear_zero_intercept'
    assert result.parameter_names == ('a',)
    assert list(result.parameters) == ['a']
    assert np.isclose(result.parameters['a'], 2.0, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(result.time, np.arange(5.0))
    np.testing.assert_array_equal(result.response, 2.0 * np.arange(5.0))
    np.testing.assert_allclose(result.fitted_values, result.response, atol=1e-7)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.parameter_vector, np.array([2.0]), rtol=1e-5)
    assert result.covariance.shape == (1, 1)
    assert result.initial_parameters.shape == (1,)
    assert result.n_parameters == 1
    assert result.residual_dof == 4
    assert np.isclose(result.fit_metrics['r_squared'], 1.0)
    assert result.fit_metrics['rmse'] < 1e-7
    assert set(result.cv_scores) == {'r_squared', 'rmse'}
    assert np.isnan(result.cv_means['r_squared'])
    assert np.isnan(result.cv_means['rmse'])
    assert result.cv_parameter_sets == []
    assert result.tau_estimate is None
    assert result.tau_sd is None


def test_fit_profile_can_calculate_tau_for_one_in_vivo_profile(
    monkeypatch,
    fast_analysis_config,
):
    time = np.linspace(0.0, 10.0, 8)
    response = np.exp(-0.10 * time)

    def exponential_initial_guess(x, y):
        rate = np.log(y[-1] / y[0]) / (x[-1] - x[0])
        return np.array([y[0], rate])

    monkeypatch.setitem(models['exponential'], 'initial_guess', exponential_initial_guess)
    np.random.seed(12345)
    result = fit_profile(
        time,
        response,
        model_name='exponential',
        selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        approach_id='approach2',
        calculate_tau=True,
    )

    assert result.parameter_names == ('a', 'b')
    assert np.isclose(result.parameters['a'], 1.0, rtol=1e-4)
    assert np.isclose(result.parameters['b'], -0.10, rtol=1e-4)
    assert result.residual_dof == 6
    assert np.isfinite(result.tau_estimate)
    assert np.isclose(result.tau_estimate, 10.0, rtol=1e-3)
    assert np.isfinite(result.tau_sd)
    assert result.tau_sd >= 0


def test_fit_profile_retains_cv_training_parameter_sets_when_requested(
    fast_analysis_config,
):
    config = dict(fast_analysis_config)
    config.update({
        'cv_scheme': 'leave_one_timepoint_out',
        'include_cv_fit_variability_plot': True,
    })
    time = np.arange(5.0)
    response = 2.0 * time

    result = fit_profile(
        time,
        response,
        model_name='linear_zero_intercept',
        selected_metrics=['rmse'],
        analysis_config=config,
        approach_id='approach1',
    )

    assert len(result.cv_parameter_sets) == 5
    assert all(
        values is not None and np.asarray(values).shape == (1,)
        for values in result.cv_parameter_sets
    )
    assert result.cv_scores['rmse'].shape == (5,)
