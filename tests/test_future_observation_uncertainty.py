"""Future measurement errors are separate from propagated input/fit errors."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from ivivc_core import (
    IVIVCAnalysis, IVIVCModelFit, ObservationUncertainty, PreparedIVIVCData,
    TimeCourse, predict_in_vivo, prepare_time_course, resolve_observation_uncertainty,
)
from ivivc_core.future_observations import resolve_future_observation_uncertainty
from ivivc_core.prediction_inputs import sample_prediction_responses
from ivivc_core.predictions import (
    predict_inverse_response, predict_tau_ratio, predict_value_ratio_with_uncertainty,
)


def _course(values=(.8, .6), sigma=.03, *, kind='individual', counts=None, time=None):
    uncertainty = None if sigma is None else resolve_observation_uncertainty(
        ObservationUncertainty(sigma, response_kind=kind, replicate_counts=counts), len(values))
    return TimeCourse(np.arange(1, len(values) + 1.) if time is None else time,
                      values, observation_uncertainty=uncertainty)


def _predict(method, prediction, **changes):
    options = dict(n_samples=256, random_state=12345)
    options.update(changes)
    if method == 'inverse':
        common = dict(model_name='exponential',
            parameters={'a': 1., 'b': -.1}, covariance=np.zeros((2, 2)), residuals=[],
            calibration_time_range=(0., 10.))
        common.update(options)
        return predict_inverse_response(prediction, **common)
    common = dict(model_name='exponential', in_vitro_covariance=np.zeros((2, 2)),
                  in_vivo_covariance=np.zeros((2, 2)))
    if method == 'ratio':
        common.update(in_vitro_parameters={'a': 1., 'b': 0.},
                      in_vivo_parameters={'a': 2., 'b': 0.})
        common.update(options)
        return predict_value_ratio_with_uncertainty(prediction, **common)
    common.update(in_vitro_parameters={'a': 1., 'b': -.2},
                  in_vivo_parameters={'a': 1., 'b': -.1},
                  tau_values=[SimpleNamespace(n=5., s=0.), SimpleNamespace(n=10., s=0.)])
    common.update(options)
    return predict_tau_ratio(prediction, **common)


@pytest.mark.parametrize('method', ['ratio', 'tau', 'inverse'])
def test_future_option_preserves_centers_time_bounds_and_previous_response_bounds(method):
    prediction = _course()
    covariance = np.diag([1e-5, 1e-6])
    options = ({'covariance': covariance} if method == 'inverse' else
               {'in_vitro_covariance': covariance, 'in_vivo_covariance': covariance})
    old = _predict(method, prediction, **options)
    off = _predict(method, prediction, future_observation_uncertainty=None, **options)
    future = _predict(method, prediction,
        future_observation_uncertainty=ObservationUncertainty(.04, response_kind='individual'), **options)
    for name in ('in_vivo_time', 'in_vivo_value', 'time_lower', 'time_upper'):
        np.testing.assert_array_equal(getattr(old, name), getattr(future, name))
    np.testing.assert_array_equal(old.value_lower, off.value_lower)
    np.testing.assert_array_equal(old.value_upper, off.value_upper)
    np.testing.assert_array_equal(old.value_lower, future.auxiliary['propagated_value_lower'])
    np.testing.assert_array_equal(old.value_upper, future.auxiliary['propagated_value_upper'])
    assert future.auxiliary['future_observation_variability_included'] is True
    assert future.auxiliary['value_interval_kind'] == 'future_observation_prediction'
    if method == 'ratio':
        assert future.auxiliary['interval_kind'] == 'future_observation_prediction'


@pytest.mark.parametrize('rng_kind', ['generator', 'randomstate', 'global'])
def test_future_noise_does_not_advance_the_source_random_stream(rng_kind):
    prediction = _course()
    states = []
    original_global_state = np.random.get_state()
    try:
        for future in (None, ObservationUncertainty(.04, response_kind='individual')):
            if rng_kind == 'global':
                np.random.seed(19)
                rng = None
            else:
                rng = np.random.default_rng(19) if rng_kind == 'generator' else np.random.RandomState(19)
            _predict('ratio', prediction, random_state=rng, future_observation_uncertainty=future)
            states.append(deepcopy(rng.bit_generator.state if rng_kind == 'generator'
                                   else (np.random if rng is None else rng).get_state()))
        if rng_kind == 'generator':
            assert states[0] == states[1]
        else:
            assert states[0][0] == states[1][0] and states[0][2:] == states[1][2:]
            np.testing.assert_array_equal(states[0][1], states[1][1])
    finally:
        np.random.set_state(original_global_state)


@pytest.mark.parametrize('method', ['ratio', 'tau', 'inverse'])
def test_reusing_input_precision_matches_explicit_sd_or_sem_without_count_conversion(method):
    prediction = _course(kind='mean', counts=[4, 9])
    versions = ['prediction', ObservationUncertainty(.03, response_kind='individual'),
                ObservationUncertainty(.03, response_kind='mean', replicate_counts=[2, 16])]
    results = [_predict(method, prediction, future_observation_uncertainty=value) for value in versions]
    for other in results[1:]:
        np.testing.assert_array_equal(results[0].value_lower, other.value_lower)
        np.testing.assert_array_equal(results[0].value_upper, other.value_upper)
    assert results[0].auxiliary['future_observation_uncertainty']['uncertainty_kind'] == 'sem'


def test_combined_ratio_bounds_include_mapped_input_error_and_unscaled_future_error_once():
    prediction = _course(values=(.8,), sigma=.03)
    result = _predict('ratio', prediction, n_samples=20000,
        future_observation_uncertainty=ObservationUncertainty(.04, response_kind='individual'))
    expected = 1.6 + norm.ppf([.025, .975]) * np.sqrt((2 * .03) ** 2 + .04 ** 2)
    np.testing.assert_allclose([result.value_lower[0], result.value_upper[0]], expected, atol=.003)


def test_zero_parameter_and_input_uncertainty_leaves_only_future_measurement_scatter():
    prediction = _course(values=(.8,), sigma=None)
    result = _predict('ratio', prediction, n_samples=20000,
        future_observation_uncertainty=ObservationUncertainty(.04, response_kind='individual'))
    expected = 1.6 + norm.ppf([.025, .975]) * .04
    np.testing.assert_allclose([result.value_lower[0], result.value_upper[0]], expected, atol=.002)
    np.testing.assert_array_equal(result.auxiliary['propagated_value_lower'], [1.6])


def test_nearest_calibration_errors_use_nominal_times_ties_and_endpoint_extension():
    calibration = _course(values=(.4, 1., .1), sigma=[.02, .01, .03], time=[10., 0., 20.])
    prediction = _course(values=(.8, .6, .4, .2), sigma=None)
    resolved, metadata = resolve_future_observation_uncertainty(
        'calibration', prediction, [-2., 5., 12., 22.], calibration_in_vivo=calibration)
    np.testing.assert_array_equal(resolved.sigma, [.01, .01, .02, .03])
    assert metadata['future_observation_uncertainty_source'] == 'calibration'
    assert metadata['future_observation_calibration_indices'] == [1, 1, 0, 2]
    assert metadata['future_observation_lookup_extrapolated'] == [True, False, False, True]


def test_calibration_reuse_preserves_normalized_sem_on_the_analyzed_time_scale():
    calibration, _ = prepare_time_course([1., 10., 100.], [50., 40., 30.],
        selected_normalizations=['initial_value'], selected_scalings=['log_x'],
        observation_uncertainty=ObservationUncertainty([5., 4., 3.], response_kind='mean',
                                                      replicate_counts=[4, 9, 16]))
    resolved, _ = resolve_future_observation_uncertainty(
        'calibration', _course(sigma=None), [.9, 2.], calibration_in_vivo=calibration)
    np.testing.assert_array_equal(resolved.sigma, [.08, .06])
    assert resolved.uncertainty_kind == 'sem'
    assert resolved.response_units == 'dimensionless'
    np.testing.assert_array_equal(resolved.replicate_counts, [9, 16])


def test_endpoint_reuse_is_independent_of_overflow_in_distance_calculations():
    calibration = _course(values=(1., .8), sigma=[.01, .02], time=[-1e308, -9e307])
    resolved, _ = resolve_future_observation_uncertainty(
        'calibration', _course(sigma=None), [-1.7e308, 1e308], calibration_in_vivo=calibration)
    np.testing.assert_array_equal(resolved.sigma, [.01, .02])


def test_inverse_future_bounds_use_the_same_successful_input_mappings():
    prediction = _course(values=(.9, .2), sigma=.05, kind='mean')
    options = dict(n_samples=100, random_state=12345, min_valid_fraction=.95)
    future = ObservationUncertainty(.03, response_kind='individual')
    result = _predict('inverse', prediction, interval_policy='conditional',
                      future_observation_uncertainty=future, **options)
    baseline = _predict('inverse', prediction, interval_policy='conditional', **options)
    inputs = sample_prediction_responses(prediction, 100, 12345)
    valid = (inputs > np.exp(-8.)) & (inputs <= 1.)
    assert 95 <= np.count_nonzero(valid[:, 0]) < 100
    # Independent child stream is separate from calibration/input streams.
    rng = np.random.default_rng(np.random.SeedSequence(12345, spawn_key=(3,)))
    draws = inputs + .03 * rng.standard_normal((2, 100)).T
    for index in range(2):
        expected = np.percentile(draws[valid[:, index], index], [2.5, 97.5])
        np.testing.assert_allclose([result.value_lower[index], result.value_upper[index]], expected)
    np.testing.assert_array_equal(result.time_lower, baseline.time_lower)
    np.testing.assert_array_equal(result.time_upper, baseline.time_upper)
    np.testing.assert_array_equal(result.auxiliary['value_interval_valid_samples'], valid.sum(axis=0))
    assert result.auxiliary['value_interval_conditional'].tolist() == [True, False]
    strict = _predict('inverse', prediction, interval_policy='strict',
                      future_observation_uncertainty=future, **options)
    assert np.isnan(strict.value_lower[0]) and np.isfinite(strict.value_lower[1])


def test_future_measurement_error_cannot_supply_missing_parameter_uncertainty():
    result = _predict('ratio', _course(), in_vivo_covariance=np.full((2, 2), np.nan),
        future_observation_uncertainty=ObservationUncertainty(.03, response_kind='individual'))
    assert np.isfinite(result.in_vivo_value).all()
    assert np.isnan(result.value_lower).all() and np.isnan(result.value_upper).all()


def test_future_noise_overflow_does_not_become_conditional_inverse_success():
    result = _predict('inverse', _course(values=(.8,), sigma=None), n_samples=100,
        future_observation_uncertainty=ObservationUncertainty(8e307, response_kind='individual'))
    assert np.isfinite(result.time_lower).all()
    assert np.isnan(result.value_lower).all() and np.isnan(result.value_upper).all()
    assert 'nonfinite future-response draws' in result.auxiliary['value_interval_status'][0]


@pytest.mark.parametrize('source', ['prediction', 'calibration', 'unknown', .03])
def test_unavailable_or_invalid_future_sources_raise(source):
    with pytest.raises((TypeError, ValueError)):
        resolve_future_observation_uncertainty(source, _course(sigma=None), [1., 2.])


def test_conflicting_duplicate_calibration_errors_require_an_explicit_future_model():
    with pytest.raises(ValueError):
        resolve_future_observation_uncertainty('calibration', _course(sigma=None), [1., 2.],
            calibration_in_vivo=_course(values=(1., .8, .7), sigma=[.01, .02, .03], time=[0., 1., 1.]))


def _analysis(approach='approach2'):
    course = _course(values=(1., np.exp(-1.)), sigma=.02, time=[0., 10.])
    fit = SimpleNamespace(parameters={'a': 1., 'b': -.1}, covariance=np.zeros((2, 2)),
                          residuals=np.zeros(2), observation_uncertainty=course.observation_uncertainty)
    if approach == 'approach2':
        model = IVIVCModelFit(approach, 'exponential', in_vitro_fit=fit, in_vivo_fit=fit)
    elif approach == 'approach1':
        model = IVIVCModelFit(approach, 'linear', time_mapping_fit=fit)
    else:
        model = IVIVCModelFit(approach, None)
    return IVIVCAnalysis(PreparedIVIVCData(course, course), {model.key: model})


def test_public_api_keeps_future_and_calibration_metadata_together():
    analysis = _analysis()
    batch = predict_in_vivo(analysis, _course(), methods={'approach2': ['value_ratio']},
        n_samples=256, future_observation_uncertainty='calibration')
    assert not batch.errors
    result = next(iter(batch.results.values()))
    assert result.auxiliary['future_observation_variability_included'] is True
    assert result.auxiliary['calibration_uncertainty']['in_vivo']['observation_uncertainty'] is not None


@pytest.mark.parametrize('approach,method', [('approach1', 'time_mapping'), ('approach3', 'value_scaling')])
def test_future_observation_options_are_limited_to_approach2(approach, method):
    with pytest.raises(ValueError):
        predict_in_vivo(_analysis(approach), _course(sigma=None), methods={approach: [method]},
                       future_observation_uncertainty=ObservationUncertainty(.03, response_kind='individual'))
