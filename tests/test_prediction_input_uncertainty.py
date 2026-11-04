"""Approach 2 propagates entered response uncertainty once into joint draws."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import norm

from ivivc_core import (
    ObservationUncertainty, TimeCourse, fit_ivivc, generate_trajectory_ensembles,
    predict_in_vivo, prepare_data, prepare_time_course, resolve_observation_uncertainty,
)
from ivivc_core.predictions import (
    predict_inverse_response, predict_tau_ratio, predict_value_ratio_with_uncertainty,
)
from ivivc_core.uncertainty import (
    sample_inverse_response_trajectories, sample_tau_ratio_trajectories,
    sample_value_ratio_trajectories,
)


def _prediction(sigma=(0.02, 0.04), *, kind='mean', counts=None, values=(0.8, 0.6)):
    value = np.asarray(values, dtype=float)
    uncertainty = None if sigma is None else resolve_observation_uncertainty(
        ObservationUncertainty(sigma, response_kind=kind, replicate_counts=counts), len(value),
    )
    return TimeCourse(np.arange(1, len(value) + 1, dtype=float), value,
                      observation_uncertainty=uncertainty)


def _ratio_options(**changes):
    options = dict(model_name='exponential',
                   in_vitro_parameters={'a': 1.0, 'b': 0.0},
                   in_vivo_parameters={'a': 2.0, 'b': 0.0},
                   in_vitro_covariance=np.zeros((2, 2)),
                   in_vivo_covariance=np.zeros((2, 2)), n_samples=128, random_state=19)
    options.update(changes)
    return options


def _tau_options(**changes):
    options = _ratio_options(in_vitro_parameters={'a': 1.0, 'b': -0.2},
                             in_vivo_parameters={'a': 1.0, 'b': -0.1})
    options.update(changes)
    return options


def _inverse_options(**changes):
    options = dict(model_name='linear', parameters={'a': 1.0, 'b': -0.1},
                   covariance=np.zeros((2, 2)), calibration_time_range=(0.0, 10.0),
                   n_samples=128, random_state=19)
    options.update(changes)
    return options


def _assert_input_metadata(metadata, prediction):
    assert metadata['prediction_input_uncertainty'] == prediction.observation_uncertainty.to_metadata()
    assert metadata['prediction_inputs_fixed'] is False
    assert metadata['prediction_times_fixed'] is True
    assert metadata['prediction_input_uncertainty_propagated'] is True
    assert metadata['future_observation_variability_included'] is False


def test_value_ratio_multiplies_each_sampled_input_once_and_uses_its_percentiles():
    prediction = _prediction()
    options = _ratio_options()
    ensemble = sample_value_ratio_trajectories(prediction, **options)
    result = predict_value_ratio_with_uncertainty(prediction, **options)
    inputs = ensemble.metadata['prediction_response_draws']
    np.testing.assert_array_equal(ensemble.value_draws, 2.0 * inputs)
    np.testing.assert_array_equal(ensemble.time_draws, np.broadcast_to(prediction.time, inputs.shape))
    np.testing.assert_array_equal(result.in_vivo_value, 2.0 * prediction.value)
    bounds = np.percentile(ensemble.value_draws, [2.5, 97.5], axis=0)
    np.testing.assert_array_equal(result.value_lower, bounds[0])
    np.testing.assert_array_equal(result.value_upper, bounds[1])
    assert result.auxiliary['interval_kind'] == 'parameter_and_input_uncertainty'
    assert np.all(result.auxiliary['interval_valid_samples'] == options['n_samples'])
    assert ensemble.valid_draws.all()
    _assert_input_metadata(result.auxiliary, prediction)
    _assert_input_metadata(ensemble.metadata, prediction)


@pytest.mark.parametrize('value', [-0.2, 0.0, 0.2])
def test_response_uncertainty_is_not_multiplied_by_the_entered_response_twice(value):
    prediction = _prediction(0.1, values=(value,))
    ensemble = sample_value_ratio_trajectories(prediction, **_ratio_options())
    np.testing.assert_array_equal(ensemble.value_draws, 2.0 * ensemble.metadata['prediction_response_draws'])
    assert np.std(ensemble.value_draws) > 0.1
    np.testing.assert_array_equal(ensemble.central.in_vivo_value, [2.0 * value])


def test_original_row_errors_are_independent_including_duplicate_times():
    prediction = _prediction((0.02, 0.02), values=(0.8, 0.8))
    prediction.time[:] = 1.0
    ensemble = sample_value_ratio_trajectories(prediction, **_ratio_options(n_samples=4096))
    response_draws = ensemble.metadata['prediction_response_draws']
    centered = response_draws - prediction.value
    np.testing.assert_allclose(np.std(centered, axis=0), [0.02, 0.02], rtol=0.04)
    assert abs(np.corrcoef(centered.T)[0, 1]) < 0.06
    assert not np.array_equal(response_draws[:, 0], response_draws[:, 1])


@pytest.mark.parametrize('method', ['value_ratio', 'tau_ratio', 'observed_response_inverse'])
def test_constant_pointwise_sd_and_sem_counts_share_the_same_resolved_error_model(method):
    samplers = {'value_ratio': (sample_value_ratio_trajectories, _ratio_options),
                'tau_ratio': (sample_tau_ratio_trajectories, _tau_options),
                'observed_response_inverse': (sample_inverse_response_trajectories, _inverse_options)}
    sampler, options = samplers[method]
    courses = [_prediction(0.03, kind='individual'),
               _prediction([0.03, 0.03], kind='mean'),
               _prediction(0.03, kind='mean', counts=[4, 16])]
    ensembles = [sampler(course, **options(n_samples=32)) for course in courses]
    for actual in ensembles[1:]:
        np.testing.assert_array_equal(actual.time_draws, ensembles[0].time_draws)
        np.testing.assert_array_equal(actual.value_draws, ensembles[0].value_draws)
        np.testing.assert_array_equal(actual.metadata['prediction_response_draws'],
                                      ensembles[0].metadata['prediction_response_draws'])
    assert ensembles[0].metadata['prediction_input_uncertainty']['uncertainty_kind'] == 'sd'
    assert ensembles[-1].metadata['prediction_input_uncertainty']['uncertainty_kind'] == 'sem'
    assert ensembles[-1].metadata['prediction_input_uncertainty']['replicate_counts'] == [4, 16]


def test_tau_ratio_response_errors_leave_time_draws_unchanged_and_add_vertical_bounds():
    prediction = _prediction()
    options = _tau_options(in_vitro_covariance=np.diag([0.0, 1e-5]),
                            in_vivo_covariance=np.diag([0.0, 1e-6]))
    uncertain = sample_tau_ratio_trajectories(prediction, **options)
    fixed = sample_tau_ratio_trajectories(_prediction(None), **options)
    np.testing.assert_array_equal(uncertain.time_draws, fixed.time_draws)
    np.testing.assert_array_equal(uncertain.metadata['in_vitro_parameter_samples'],
                                  fixed.metadata['in_vitro_parameter_samples'])
    np.testing.assert_array_equal(uncertain.metadata['in_vivo_parameter_samples'],
                                  fixed.metadata['in_vivo_parameter_samples'])
    np.testing.assert_array_equal(uncertain.value_draws, uncertain.metadata['prediction_response_draws'])
    assert np.std(uncertain.value_draws[:, 0]) > 0.0
    result = predict_tau_ratio(prediction, tau_values=[SimpleNamespace(n=5.0, s=0.0),
                                                      SimpleNamespace(n=10.0, s=0.0)], **options)
    baseline = predict_tau_ratio(_prediction(None), tau_values=[SimpleNamespace(n=5.0, s=0.0),
                                                                 SimpleNamespace(n=10.0, s=0.0)], **options)
    np.testing.assert_array_equal(result.time_lower, baseline.time_lower)
    np.testing.assert_array_equal(result.time_upper, baseline.time_upper)
    np.testing.assert_array_equal(result.in_vivo_value, prediction.value)
    sigma = prediction.observation_uncertainty.sigma
    np.testing.assert_allclose(result.value_lower, prediction.value - norm.ppf(0.975) * sigma)
    np.testing.assert_allclose(result.value_upper, prediction.value + norm.ppf(0.975) * sigma)
    assert result.auxiliary['interval_kind'] == 'parameter_uncertainty'
    assert result.auxiliary['value_interval_kind'] == 'prediction_input_uncertainty'
    assert result.auxiliary['value_interval_method'] == 'normal'
    _assert_input_metadata(result.auxiliary, prediction)


def test_tau_vertical_input_intervals_remain_available_when_calibration_sampling_fails():
    prediction = _prediction()
    result = predict_tau_ratio(
        prediction, tau_values=[SimpleNamespace(n=5.0, s=0.0), SimpleNamespace(n=10.0, s=0.0)],
        **_tau_options(in_vitro_covariance=np.diag([-0.01, 0.0])),
    )
    assert result.available
    np.testing.assert_array_equal(result.in_vivo_time, 2.0 * prediction.time)
    assert np.isnan(result.time_lower).all() and np.isnan(result.time_upper).all()
    sigma = prediction.observation_uncertainty.sigma
    np.testing.assert_allclose(result.value_lower, prediction.value - norm.ppf(0.975) * sigma)
    np.testing.assert_allclose(result.value_upper, prediction.value + norm.ppf(0.975) * sigma)
    assert result.auxiliary['value_interval_kind'] == 'prediction_input_uncertainty'
    _assert_input_metadata(result.auxiliary, prediction)


def test_added_input_uncertainty_preserves_both_full_calibration_parameter_draws():
    prediction = _prediction()
    options = _ratio_options(
        in_vitro_parameters={'a': 1.0, 'b': -0.2}, in_vivo_parameters={'a': 0.9, 'b': -0.1},
        in_vitro_covariance=np.array([[1e-4, -2e-5], [-2e-5, 1e-5]]),
        in_vivo_covariance=np.array([[2e-4, -1e-5], [-1e-5, 2e-5]]),
    )
    uncertain = sample_value_ratio_trajectories(prediction, **options)
    fixed = sample_value_ratio_trajectories(_prediction(None), **options)
    for source in ('in_vitro', 'in_vivo'):
        np.testing.assert_array_equal(uncertain.metadata[source + '_parameter_samples'],
                                      fixed.metadata[source + '_parameter_samples'])
    np.testing.assert_array_equal(uncertain.metadata['ratio_draws'], fixed.metadata['ratio_draws'])
    np.testing.assert_array_equal(uncertain.value_draws,
        uncertain.metadata['prediction_response_draws'] * uncertain.metadata['ratio_draws'])


@pytest.mark.parametrize('rng_kind', ['generator', 'random_state'])
def test_numpy_random_objects_reproduce_input_draws_without_perturbing_calibration_draws(rng_kind):
    factory = np.random.default_rng if rng_kind == 'generator' else np.random.RandomState
    options = _ratio_options(
        in_vitro_parameters={'a': 1.0, 'b': -0.2}, in_vivo_parameters={'a': 0.9, 'b': -0.1},
        in_vitro_covariance=np.array([[1e-4, -2e-5], [-2e-5, 1e-5]]),
        in_vivo_covariance=np.array([[2e-4, -1e-5], [-1e-5, 2e-5]]),
    )
    uncertain = sample_value_ratio_trajectories(_prediction(), **dict(options, random_state=factory(19)))
    repeated = sample_value_ratio_trajectories(_prediction(), **dict(options, random_state=factory(19)))
    fixed = sample_value_ratio_trajectories(_prediction(None), **dict(options, random_state=factory(19)))
    np.testing.assert_array_equal(uncertain.value_draws, repeated.value_draws)
    np.testing.assert_array_equal(uncertain.metadata['prediction_response_draws'],
                                  repeated.metadata['prediction_response_draws'])
    for source in ('in_vitro', 'in_vivo'):
        np.testing.assert_array_equal(uncertain.metadata[source + '_parameter_samples'],
                                      fixed.metadata[source + '_parameter_samples'])


def test_inverse_maps_the_same_sampled_response_retained_in_each_joint_draw():
    prediction = _prediction()
    options = _inverse_options()
    ensemble = sample_inverse_response_trajectories(prediction, **options)
    np.testing.assert_array_equal(ensemble.value_draws, ensemble.metadata['prediction_response_draws'])
    np.testing.assert_allclose(ensemble.time_draws, 10.0 * (1.0 - ensemble.value_draws), atol=1e-8)
    assert ensemble.valid_draws.all()
    result = predict_inverse_response(prediction, residuals=np.ones(5), **options)
    bounds = np.percentile(ensemble.time_draws, [2.5, 97.5], axis=0)
    np.testing.assert_allclose(result.time_lower, bounds[0], atol=1e-8)
    np.testing.assert_allclose(result.time_upper, bounds[1], atol=1e-8)
    np.testing.assert_allclose(result.in_vivo_time, [2.0, 4.0], atol=1e-8)
    assert result.auxiliary['interval_kind'] == 'parameter_and_input_uncertainty'
    sigma = prediction.observation_uncertainty.sigma
    np.testing.assert_allclose(result.value_lower, prediction.value - norm.ppf(0.975) * sigma)
    np.testing.assert_allclose(result.value_upper, prediction.value + norm.ppf(0.975) * sigma)
    assert result.auxiliary['value_interval_kind'] == 'prediction_input_uncertainty'
    _assert_input_metadata(ensemble.metadata, prediction)


def test_inverse_unreachable_input_draws_are_retained_and_not_clipped_or_retried():
    prediction = _prediction((0.6, 0.001), values=(0.8, 0.5))
    options = _inverse_options(n_samples=64)
    ensemble = sample_inverse_response_trajectories(prediction, **options)
    response_draws = ensemble.metadata['prediction_response_draws']
    assert np.any(response_draws[:, 0] > 1.0)
    assert np.any(response_draws[:, 0] < 0.0)
    assert ensemble.n_draws == options['n_samples']
    invalid = response_draws[:, 0] > 1.0
    assert np.isnan(ensemble.time_draws[invalid, 0]).all()
    assert np.isfinite(ensemble.time_draws[:, 1]).all()
    np.testing.assert_array_equal(ensemble.valid_draws, np.all(np.isfinite(ensemble.time_draws), axis=1))
    assert 0 < ensemble.n_valid_draws < ensemble.n_draws
    result = predict_inverse_response(prediction, residuals=np.zeros(3), interval_policy='strict', **options)
    assert np.isfinite(result.in_vivo_time).all()
    assert np.isnan(result.time_lower[0]) and np.isnan(result.time_upper[0])
    assert np.isfinite(result.time_lower[1]) and np.isfinite(result.time_upper[1])
    assert result.auxiliary['interval_valid_samples'][0] == np.count_nonzero(~invalid)
    assert result.auxiliary['interval_valid_samples'][1] == options['n_samples']


def test_inverse_input_perturbation_cannot_create_a_missing_nominal_prediction():
    prediction = _prediction((0.6, 0.001), values=(1.2, 0.5))
    options = _inverse_options(n_samples=64)
    ensemble = sample_inverse_response_trajectories(prediction, **options)
    assert np.isnan(ensemble.central.in_vivo_time[0])
    assert np.any(ensemble.metadata['prediction_response_draws'][:, 0] < 1.0)
    assert np.isnan(ensemble.time_draws[:, 0]).all()
    assert np.isfinite(ensemble.time_draws[:, 1]).all()
    assert not ensemble.valid_draws.any()
    result = predict_inverse_response(prediction, residuals=np.zeros(3), **options)
    assert np.isnan(result.in_vivo_time[0])
    assert np.isnan(result.time_lower[0]) and np.isnan(result.time_upper[0])
    assert np.isfinite(result.in_vivo_time[1])
    assert np.isfinite(result.time_lower[1]) and np.isfinite(result.time_upper[1])


def test_input_errors_do_not_bypass_the_ratio_denominator_zero_crossing_safeguard():
    prediction = _prediction()
    result = predict_value_ratio_with_uncertainty(prediction, **_ratio_options(
        in_vitro_parameters={'a': 0.05, 'b': 0.0},
        in_vitro_covariance=np.diag([0.04, 0.0]), n_samples=512,
    ))
    assert np.isfinite(result.in_vivo_value).all()
    assert np.isnan(result.value_lower).all() and np.isnan(result.value_upper).all()
    assert np.all(result.auxiliary['denominator_lower'] < 0.0)
    assert np.all(result.auxiliary['denominator_upper'] > 0.0)
    assert np.all(result.auxiliary['interval_status'] == 'unavailable (denominator interval includes zero)')


def test_fixed_initial_normalization_rescales_input_uncertainty_without_freezing_the_first_response():
    prediction, metadata = prepare_time_course(
        [0.0, 1.0], [10.0, 8.0], selected_normalizations=['initial_value'],
        observation_uncertainty=ObservationUncertainty([0.5, 0.8], response_kind='mean'),
    )
    np.testing.assert_allclose(prediction.observation_uncertainty.sigma, [0.05, 0.08])
    assert metadata.normalizations[0].in_vitro_parameters['constants_treated_as_fixed'] is True
    ensemble = sample_value_ratio_trajectories(prediction, **_ratio_options())
    np.testing.assert_array_equal(ensemble.central.in_vivo_value, [2.0, 1.6])
    assert np.std(ensemble.metadata['prediction_response_draws'][:, 0]) > 0.0
    _assert_input_metadata(ensemble.metadata, prediction)


def _analysis(config, *, calibration_uncertainty=False, mixed_approaches=False):
    time = np.arange(6.0)
    response = np.exp(-0.2 * time) + np.array([0.0, 0.001, -0.002, 0.001, 0.001, -0.001])
    supplied = ObservationUncertainty(0.02, response_kind='mean') if calibration_uncertainty else None
    prepared = prepare_data(time, response, 2.0 * time, response,
                            in_vitro_uncertainty=supplied, in_vivo_uncertainty=supplied)
    selections = ['approach2:exponential_normalized']
    if mixed_approaches:
        selections = ['approach1:linear', *selections, 'approach3']
    analysis = fit_ivivc(prepared, selections=selections, selected_metrics=['rmse'], analysis_config=config)
    assert analysis.errors == {}
    return analysis


def test_public_apis_keep_approach2_results_when_other_approaches_cannot_use_input_uncertainty(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, mixed_approaches=True)
    prediction = _prediction()
    methods = {'approach1': ('time_mapping',),
               'approach2': ('value_ratio', 'observed_response_inverse'),
               'approach3': ('value_scaling',)}
    for operation in (predict_in_vivo, generate_trajectory_ensembles):
        batch = operation(analysis, prediction, methods=methods, n_samples=12, random_state=19)
        assert batch.successful_keys == ('approach2:exponential_normalized:value_ratio',
                                         'approach2:exponential_normalized:observed_response_inverse')
        assert batch.failed_keys == ('approach1:linear:time_mapping', 'approach3:value_scaling')
        assert all(error.exception_type == 'ValueError' for error in batch.errors.values())
        with pytest.raises(ValueError, match='prediction-response uncertainty'):
            operation(analysis, prediction, methods=methods, n_samples=12, fail_fast=True)


def test_public_complete_ensembles_preserve_both_uncertainty_sources_and_do_not_refit(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, calibration_uncertainty=True)
    fit = analysis['approach2:exponential_normalized']
    snapshots = [(part.parameter_vector.copy(), part.covariance.copy())
                 for part in (fit.in_vitro_fit, fit.in_vivo_fit)]
    prediction = _prediction()
    with patch('ivivc_core.api.fit_profile', side_effect=AssertionError('Prediction must not refit calibration.')):
        batch = generate_trajectory_ensembles(analysis, prediction,
            methods={'approach2': ('value_ratio', 'tau_ratio', 'observed_response_inverse')},
            n_samples=12, random_state=19)
    assert batch.errors == {}
    assert len(batch.results) == 3
    for output in batch.results.values():
        _assert_input_metadata(output.metadata, prediction)
        _assert_input_metadata(output.central.auxiliary, prediction)
        for source in ('in_vitro', 'in_vivo'):
            assert output.metadata['calibration_uncertainty'][source]['covariance_scaling'] == 'absolute'
    for part, (parameters, covariance) in zip((fit.in_vitro_fit, fit.in_vivo_fit), snapshots):
        np.testing.assert_array_equal(part.parameter_vector, parameters)
        np.testing.assert_array_equal(part.covariance, covariance)


def test_public_apis_reject_the_same_calibration_course_as_an_independent_uncertain_input(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, calibration_uncertainty=True)
    reused = analysis.prepared_data.in_vitro
    for operation in (predict_in_vivo, generate_trajectory_ensembles):
        with pytest.raises(ValueError, match='calibration|independen|reus'):
            operation(analysis, reused, methods={'approach2': ('value_ratio',)})


def test_invalid_attached_input_uncertainty_fails_before_consuming_the_random_generator():
    prediction = _prediction()
    prediction.observation_uncertainty = resolve_observation_uncertainty(
        ObservationUncertainty(0.03, response_kind='mean'), 3,
    )
    rng = np.random.default_rng(19)
    state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        sample_value_ratio_trajectories(prediction, **_ratio_options(random_state=rng))
    assert rng.bit_generator.state == state


def test_appending_an_unsupported_input_does_not_change_an_existing_inverse_interval():
    first = predict_inverse_response(
        _prediction(0.001, values=(0.5,)), residuals=[], **_inverse_options(),
    )
    together = predict_inverse_response(
        _prediction([0.001, 0.4], values=(0.5, 1.5)), residuals=[], **_inverse_options(),
    )
    for field in ('in_vivo_time', 'time_lower', 'time_upper'):
        np.testing.assert_array_equal(getattr(first, field), getattr(together, field)[:1])
    assert np.isnan(together.in_vivo_time[1])
    assert np.isnan(together.time_lower[1])


def test_combined_inverse_interval_warning_describes_propagated_uncertainty():
    result = predict_inverse_response(
        _prediction(0.09, values=(0.1,)), residuals=[], **_inverse_options(),
    )
    assert not result.auxiliary['extrapolated'][0]
    assert result.auxiliary['interval_extrapolated'][0]
    warning = next(item for item in result.auxiliary['extrapolation_warnings']
                   if item['label'] == 'Interval extends beyond calibration')
    assert 'propagated-uncertainty intervals' in warning['message']


def test_nonfinite_response_bounds_do_not_mark_a_finite_prediction_unavailable():
    result = predict_tau_ratio(
        _prediction(1.e308, values=(0.5,)),
        tau_values=[SimpleNamespace(n=5.0, s=0.0), SimpleNamespace(n=10.0, s=0.0)],
        **_tau_options(n_samples=8),
    )
    assert result.available
    assert np.isfinite(result.in_vivo_time[0])
    assert np.isnan(result.value_lower[0]) and np.isnan(result.value_upper[0])
    assert result.auxiliary['value_interval_status'][0] == 'unavailable (nonfinite interval bounds)'


@pytest.mark.parametrize('method', ['tau_ratio', 'observed_response_inverse'])
def test_unavailable_nominal_predictions_retain_input_metadata_without_claiming_propagation(method):
    prediction = _prediction(0.1, values=(1.5,))
    if method == 'tau_ratio':
        result = predict_tau_ratio(prediction, tau_values=None, **_tau_options(n_samples=8))
    else:
        result = predict_inverse_response(prediction, residuals=[], **_inverse_options(n_samples=8))
    assert not result.available
    assert np.isnan(result.in_vivo_time).all()
    assert result.auxiliary['prediction_input_uncertainty'] == prediction.observation_uncertainty.to_metadata()
    assert result.auxiliary['prediction_input_uncertainty_propagated'] is False
