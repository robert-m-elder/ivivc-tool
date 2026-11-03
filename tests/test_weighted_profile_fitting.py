"""Numerical contracts for absolute observation uncertainty in profile fits."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import curve_fit
from sklearn.model_selection import LeaveOneOut

from ivivc_core import ObservationUncertainty, fit_profile
from models import generic_fit_model, models
from utilities.evaluation import auto_grid_search, cross_validation_curve_fit


def _fit(time, response, config, uncertainty=None, **kwargs):
    return fit_profile(
        time, response,
        model_name='linear_zero_intercept', selected_metrics=['rmse'],
        analysis_config=config, observation_uncertainty=uncertainty, **kwargs,
    )


def _weighted_slope(time, response, sigma):
    precision = 1.0 / np.asarray(sigma, dtype=float) ** 2
    denominator = np.sum(precision * time ** 2)
    return np.sum(precision * time * response) / denominator, 1.0 / denominator


def test_weighted_profile_matches_analytic_slope_and_absolute_covariance(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    sigma = np.array([0.1, 0.2, 0.15, 0.2, 3.0])
    expected, expected_variance = _weighted_slope(time, response, sigma)

    fit = _fit(time, response, fast_analysis_config,
               ObservationUncertainty(sigma, response_kind='mean', source='reported SEM'))

    np.testing.assert_allclose(fit.parameter_vector, [expected], rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, [[expected_variance]], rtol=1e-5)
    np.testing.assert_array_equal(fit.observation_uncertainty.sigma, sigma)
    assert fit.observation_uncertainty.source == 'reported SEM'
    assert fit.fit_diagnostics['objective'] == 'weighted_least_squares'
    assert fit.fit_diagnostics['covariance_scaling'] == 'absolute'
    assert fit.fit_diagnostics['absolute_sigma'] is True
    expected_chi_squared = np.sum(((response - expected * time) / sigma) ** 2)
    assert np.isclose(fit.fit_diagnostics['chi_squared'], expected_chi_squared)
    assert np.isclose(fit.fit_diagnostics['reduced_chi_squared'], expected_chi_squared / 4)
    assert fit.fit_diagnostics['covariance_finite'] is True


def test_no_uncertainty_preserves_unweighted_fit_and_residual_scaled_covariance(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    expected_slope = np.sum(time * response) / np.sum(time ** 2)
    residuals = response - expected_slope * time
    expected_variance = np.sum(residuals ** 2) / (len(time) - 1) / np.sum(time ** 2)

    fit = _fit(time, response, fast_analysis_config)

    np.testing.assert_allclose(fit.parameter_vector, [expected_slope], rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, [[expected_variance]], rtol=1e-5)
    assert fit.observation_uncertainty is None
    assert fit.fit_diagnostics['objective'] == 'unweighted_least_squares'
    assert fit.fit_diagnostics['covariance_scaling'] == 'residual_scaled'
    assert fit.fit_diagnostics['absolute_sigma'] is False


def test_equal_absolute_errors_preserve_parameters_but_not_residual_covariance(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    unweighted = _fit(time, response, fast_analysis_config)
    weighted = _fit(time, response, fast_analysis_config,
                    ObservationUncertainty(0.2, response_kind='individual'))

    np.testing.assert_allclose(weighted.parameter_vector, unweighted.parameter_vector, rtol=1e-6)
    np.testing.assert_allclose(weighted.covariance, [[0.2 ** 2 / np.sum(time ** 2)]], rtol=1e-5)
    assert not np.isclose(weighted.covariance[0, 0], unweighted.covariance[0, 0])


def test_absolute_error_scale_changes_covariance_without_changing_parameters(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    sigma = np.array([0.1, 0.2, 0.15, 0.2, 3.0])
    base = _fit(time, response, fast_analysis_config,
                ObservationUncertainty(sigma, response_kind='mean'))
    scaled = _fit(time, response, fast_analysis_config,
                  ObservationUncertainty(3.0 * sigma, response_kind='mean'))

    np.testing.assert_allclose(scaled.parameter_vector, base.parameter_vector, rtol=1e-6)
    np.testing.assert_allclose(scaled.covariance, 9.0 * base.covariance, rtol=1e-5)


def test_exact_fit_retains_nonzero_absolute_parameter_uncertainty(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    fit = _fit(time, 2.0 * time, fast_analysis_config,
               ObservationUncertainty(0.3, response_kind='mean'))

    np.testing.assert_allclose(fit.parameter_vector, [2.0], rtol=1e-7)
    np.testing.assert_allclose(fit.covariance, [[0.3 ** 2 / np.sum(time ** 2)]], rtol=1e-5)
    assert fit.covariance[0, 0] > 0.0


def test_heterogeneous_errors_reduce_influence_of_imprecise_observation(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
    unweighted = _fit(time, response, fast_analysis_config)
    weighted = _fit(time, response, fast_analysis_config,
                    ObservationUncertainty([0.1, 0.1, 0.1, 0.1, 10.0], response_kind='mean'))

    assert abs(weighted.parameters['a'] - 1.0) < 0.01
    assert abs(unweighted.parameters['a'] - 1.0) > 1.0


def test_constant_column_and_sem_counts_have_identical_fit_information(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 5.3])
    specifications = [
        ObservationUncertainty(0.2, response_kind='individual'),
        ObservationUncertainty([0.2] * 5, response_kind='mean'),
        ObservationUncertainty(0.2, response_kind='mean', replicate_counts=[2, 3, 4, 5, 6]),
    ]
    fits = [_fit(time, response, fast_analysis_config, value) for value in specifications]

    for fit in fits[1:]:
        np.testing.assert_allclose(fit.parameter_vector, fits[0].parameter_vector, rtol=1e-7)
        np.testing.assert_allclose(fit.covariance, fits[0].covariance, rtol=1e-6)
        assert fit.residual_dof == 4
    np.testing.assert_array_equal(fits[-1].observation_uncertainty.replicate_counts, [2, 3, 4, 5, 6])


def test_missing_response_rows_mask_uncertainty_before_numerical_validation(fast_analysis_config):
    time = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    response = np.array([1.1, np.nan, 30.0, 4.1, 9.0])
    sigma = np.array([0.1, np.nan, np.nan, 0.2, 2.0])
    fit = _fit(time, response, fast_analysis_config,
               ObservationUncertainty(sigma, response_kind='mean', replicate_counts=[2, np.nan, np.nan, 3, 4]))
    expected, variance = _weighted_slope(time[[0, 3, 4]], response[[0, 3, 4]], sigma[[0, 3, 4]])

    np.testing.assert_array_equal(fit.time, [1.0, 4.0, 5.0])
    np.testing.assert_array_equal(fit.observation_uncertainty.sigma, [0.1, 0.2, 2.0])
    np.testing.assert_array_equal(fit.observation_uncertainty.replicate_counts, [2, 3, 4])
    np.testing.assert_allclose(fit.parameter_vector, [expected], rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, [[variance]], rtol=1e-5)
    assert fit.residual_dof == 2


@pytest.mark.parametrize('sigma', [[0.1, np.nan], [0.1, 0.0], [0.1], [0.1, np.inf]])
def test_invalid_profile_uncertainty_is_rejected_before_disabled_cv(fast_analysis_config, sigma):
    with pytest.raises(ValueError):
        _fit(np.array([1.0, 2.0]), np.array([1.0, 2.1]), fast_analysis_config,
             ObservationUncertainty(sigma, response_kind='mean'))


def test_cv_parameter_sets_use_only_the_training_observation_weights(fast_analysis_config):
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    sigma = np.array([0.1, 0.2, 0.15, 0.2, 3.0])
    config = dict(fast_analysis_config, cv_scheme='leave_one_timepoint_out',
                  include_cv_fit_variability_plot=True)
    fit = _fit(time, response, config, ObservationUncertainty(sigma, response_kind='mean'))

    assert len(fit.cv_parameter_sets) == len(time)
    for held_out, params in enumerate(fit.cv_parameter_sets):
        keep = np.arange(len(time)) != held_out
        expected, _ = _weighted_slope(time[keep], response[keep], sigma[keep])
        assert params is not None
        np.testing.assert_allclose(params, [expected], rtol=1e-6)
    assert np.all(np.isfinite(fit.cv_scores['rmse']))


def test_grid_search_minimizes_the_same_weighted_objective():
    time = np.arange(1.0, 6.0)
    response = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
    sigma = np.array([0.1, 0.1, 0.1, 0.1, 10.0])
    expected, _ = _weighted_slope(time, response, sigma)
    params = auto_grid_search(
        models['linear_zero_intercept']['model_function'], time, response,
        num_points=3, num_cores=1, param_min=-2.0, param_max=2.0,
        random_state=12345, sigma=sigma,
    )

    np.testing.assert_allclose(params, [expected], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('sigma', [0.0, [0.1], [0.1, np.nan], [0.1, -0.2]])
def test_invalid_direct_helper_sigma_is_not_hidden_as_disabled_or_failed_cv(sigma):
    with pytest.raises(ValueError):
        cross_validation_curve_fit(
            np.array([1.0, 2.0]), np.array([1.0, 2.1]),
            models['linear_zero_intercept']['model_function'], None, ['rmse'],
            p0=[1.0], sigma=sigma, absolute_sigma=True,
        )


def test_weighted_fit_does_not_mutate_model_registry_options(fast_analysis_config):
    entry = models['linear_zero_intercept']
    before = deepcopy(entry['fit_kwargs'])
    options_object = entry['fit_kwargs']
    _fit(np.arange(1.0, 6.0), np.array([1.1, 2.0, 2.8, 4.2, 5.3]),
         fast_analysis_config, ObservationUncertainty(0.2, response_kind='mean'))

    assert entry['fit_kwargs'] is options_object
    assert entry['fit_kwargs'] == before
    assert 'sigma' not in entry['fit_kwargs']
    assert 'absolute_sigma' not in entry['fit_kwargs']


def test_cv_legacy_sigma_options_are_subset_without_mutating_the_callers_dictionary():
    time = np.arange(1.0, 6.0)
    response = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    sigma = np.array([0.1, 0.2, 0.15, 0.2, 3.0])
    options = {'sigma': sigma, 'absolute_sigma': True, 'maxfev': 10000}
    scores, parameter_sets = cross_validation_curve_fit(
        time, response, models['linear_zero_intercept']['model_function'],
        LeaveOneOut(), ['rmse', 'aic'], p0=[1.0], kwargs=options,
        return_fit_params=True,
    )

    assert options['sigma'] is sigma
    assert options['absolute_sigma'] is True
    assert options['maxfev'] == 10000
    np.testing.assert_array_equal(sigma, [0.1, 0.2, 0.15, 0.2, 3.0])
    for held_out, params in enumerate(parameter_sets):
        keep = np.arange(len(time)) != held_out
        expected, _ = _weighted_slope(time[keep], response[keep], sigma[keep])
        np.testing.assert_allclose(params, [expected], rtol=1e-6)
    assert np.all(np.isfinite(scores['rmse']))
    assert np.all(np.isnan(scores['aic']))


def test_duplicate_observation_error_options_are_rejected_before_cv():
    time = np.arange(1.0, 4.0)
    response = time.copy()
    function = models['linear_zero_intercept']['model_function']
    with pytest.raises(ValueError):
        cross_validation_curve_fit(
            time, response, function, None, ['rmse'], p0=[1.0],
            sigma=0.1, kwargs={'sigma': 0.1},
        )
    with pytest.raises(ValueError):
        cross_validation_curve_fit(
            time, response, function, None, ['rmse'], p0=[1.0],
            sigma=0.1, absolute_sigma=True, kwargs={'absolute_sigma': False},
        )


@pytest.mark.parametrize('approach_id', ['approach1', 'approach3'])
def test_experimental_uncertainty_does_not_enable_other_ivivc_approaches(
    fast_analysis_config, approach_id,
):
    with pytest.raises(ValueError):
        _fit(np.arange(1.0, 6.0), np.arange(1.0, 6.0), fast_analysis_config,
             ObservationUncertainty(0.2, response_kind='mean'), approach_id=approach_id)


def test_weighted_information_criteria_are_withheld_until_likelihood_support(fast_analysis_config):
    fit = fit_profile(
        np.arange(1.0, 6.0), np.array([1.1, 2.0, 2.8, 4.2, 5.3]),
        model_name='linear_zero_intercept',
        selected_metrics=['rmse', 'aic', 'aicc', 'bic'],
        analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty(0.2, response_kind='mean'),
    )

    assert np.isfinite(fit.fit_metrics['rmse'])
    for metric in ('aic', 'aicc', 'bic'):
        assert np.isnan(fit.fit_metrics[metric])
        assert metric in fit.fit_diagnostics['unavailable_metrics']


def test_weighted_generic_fit_preserves_nonlinear_model_bounds_and_options():
    def exponential(time, rate):
        return np.exp(rate * time)

    time = np.linspace(0.0, 4.0, 8)
    response = np.exp(0.1 * time)
    sigma = np.linspace(0.1, 0.3, len(time))
    options = {'bounds': ([-0.5], [0.0]), 'maxfev': 10000}
    before = deepcopy(options)
    result = generic_fit_model(exponential, time, response, p0=[-0.1],
                               kwargs=options, sigma=sigma, absolute_sigma=True)
    expected, expected_covariance = curve_fit(
        exponential, time, response, p0=[-0.1], sigma=sigma,
        absolute_sigma=True, **options,
    )

    assert -0.5 <= result['params']['rate'] <= 0.0
    np.testing.assert_allclose([result['params']['rate']], expected, atol=1e-8)
    np.testing.assert_allclose(result['pcov'], expected_covariance, rtol=1e-6)
    assert options == before
