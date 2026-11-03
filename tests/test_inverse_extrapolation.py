"""Numerical contracts for inverse search, shared intervals and domain cautions."""
from types import SimpleNamespace

import numpy as np
import pytest

from ivivc_core.inverse import inverse_response_times
from ivivc_core.predictions import (
    predict_inverse_response, inverse_response_prediction_intervals,
    predict_direct_value_scaling, predict_direct_time_scaling,
)
from ivivc_core.extrapolation import annotate_prediction_domain
from ivivc_core.types import TimeCourse, PredictedTrajectory, ParameterSamples
from ivivc_core.uncertainty import sample_inverse_response_trajectories
from models.exponential import model_function as exponential
from models.mass_loss_surface_erosion import model_function as erosion

PARAMS = {'a': 1., 'b': -.1}


def mapping(targets, time_range=(0., 10.), **kwargs):
    return inverse_response_times(exponential, PARAMS, targets, time_range, **kwargs)


@pytest.mark.parametrize('target', [np.exp(-.5), np.exp(-1.), .2, .01, 1e-20])
def test_exponential_inverse_agrees_with_exact_solution_inside_and_outside_calibration(target):
    # Give this accuracy check enough search range even if app defaults change.
    values, status, reason = mapping([target], max_expansions=16)
    assert reason is None
    np.testing.assert_allclose(values, [-10. * np.log(target)], rtol=1e-8, atol=1e-9)
    assert ('extrapolated' in status[0]) == (values[0] > 10.)


def test_backward_extrapolation_reaches_time_zero_without_negative_physical_times():
    result, status, _ = mapping([.9, 1., 1.2], time_range=(5., 10.))
    np.testing.assert_allclose(result[:2], [-10. * np.log(.9), 0.], atol=1e-8)
    assert np.isnan(result[2])
    assert 'allowed time domain' in status[2]


def test_log_time_coordinates_can_be_negative_when_explicitly_allowed():
    # A linear curve fitted against log10(time) can map to a negative coordinate.
    def model(t, a, b):
        return a + b * np.asarray(t)
    result, _, reason = inverse_response_times(model, {'a': 1., 'b': -1.}, [2.], (0., 1.), minimum_time=None)
    assert reason is None
    np.testing.assert_allclose(result, [-1.])


def test_observed_response_range_does_not_clip_fitted_inverse():
    # .9 lies in the observed response range but above the fitted calibration
    # endpoint at t=5; it must map to 1.0536, never be clipped to t=5.
    result, status, _ = mapping([.9], (5., 10.), observed_responses=[1., .2])
    assert result[0] == pytest.approx(-10. * np.log(.9))
    assert 'extrapolated' in status[0]
    result2, _, _ = mapping([.5], observed_responses=[.8, .9])
    assert result2[0] == pytest.approx(-10. * np.log(.5))


def test_extrapolation_can_be_disabled_without_restoring_endpoint_clipping():
    result, status, _ = mapping([.9], (5., 10.), allow_extrapolation=False)
    assert np.isnan(result[0])
    assert 'disabled' in status[0]


@pytest.mark.parametrize('time_range', [(0., 5.), (0., 10.), (0., 20.)])
def test_surface_erosion_plateau_is_not_a_unique_or_first_passage_inverse(time_range):
    result, status, _ = inverse_response_times(
        erosion, {'y0': 1., 'k': .1, 'n': 1., 'A':0.}, [0., .5], time_range)
    assert np.isnan(result[0])
    assert 'flat' in status[0]
    assert result[1] == pytest.approx(5.)


@pytest.mark.parametrize('target', [0., -.1])
def test_exponential_asymptote_or_unreachable_response_is_not_a_fabricated_time(target):
    result, status, _ = mapping([target])
    assert np.isnan(result[0])
    assert any(word in status[0] for word in ('flat', 'limit', 'not reached'))


def test_finite_search_limit_is_not_reported_as_a_physical_endpoint():
    result, status, _ = mapping([np.exp(-10.)], max_expansions=1)
    assert np.isnan(result[0])
    assert status[0] == 'inverse not resolved within search limit'


def test_nonmonotonic_calibration_and_continuation_are_not_inverted_arbitrarily():
    def curved(t):
        return (np.asarray(t) - 2.) ** 2
    result, _, reason = inverse_response_times(curved, {}, [1.], (0., 4.))
    assert np.isnan(result).all() and reason is not None
    # Calibration decreasing on [0,1]; reaching a target below its minimum
    # would traverse a turning point. Refuse that branch rather than pick a root.
    result, status, reason = inverse_response_times(curved, {}, [-1.], (0., 1.))
    assert reason is None and np.isnan(result[0])
    assert status[0] == 'inverse continuation is not monotonic'


def test_nonfinite_gap_is_not_bridged_to_find_a_root():
    def gap(t):
        t = np.asarray(t)
        return np.where((t > 2.) & (t < 4.), np.nan, 10. - t)
    result, status, _ = inverse_response_times(gap, {}, [5.], (0., 1.))
    assert np.isnan(result[0]) and 'nonfinite' in status[0]


def test_discontinuous_function_is_not_accepted_just_because_it_has_a_sign_change():
    def step(t):
        return np.where(np.asarray(t) < .5, 1., -1.)
    result, status, _ = inverse_response_times(step, {}, [.2], (0., 1.))
    assert np.isnan(result[0])
    assert 'tolerance' in status[0]


def test_exact_root_and_nonfinite_target_have_explicit_statuses():
    result, status, reason = mapping([1., np.nan, np.inf])
    assert reason is None and result[0] == 0.
    assert status.tolist() == ['mapped', 'nonfinite input response', 'nonfinite input response']


@pytest.mark.parametrize('factor', [1e-12, 1., 1e12])
def test_inverse_does_not_depend_on_response_units(factor):
    result, _, _ = inverse_response_times(exponential, {'a': factor, 'b': -.1},
                                          np.array([.6, .2]) * factor, (0., 10.))
    np.testing.assert_allclose(result, -10. * np.log([.6, .2]), rtol=1e-9)


def inverse(responses=(.6, .2), covariance=None):
    return predict_inverse_response(
        TimeCourse(np.arange(len(responses)), np.asarray(responses)), model_name='exponential',
        parameters=PARAMS, covariance=np.zeros((2, 2)) if covariance is None else covariance,
        residuals=np.ones(5), calibration_time_range=(0., 10.),
        observed_responses=np.exp(-.1 * np.linspace(0, 10, 5)), n_samples=30,
    )


def test_zero_covariance_collapses_intervals_even_for_extrapolated_points():
    result = inverse()
    np.testing.assert_array_equal(result.in_vivo_time, result.time_lower)
    np.testing.assert_array_equal(result.in_vivo_time, result.time_upper)
    assert result.auxiliary['interval_status'].tolist() == ['available', 'available']
    assert result.auxiliary['inverse_extrapolated_samples'].tolist() == [0, 30]
    assert result.auxiliary['extrapolated'].tolist() == [False, True]


def test_distant_unsupported_target_does_not_change_supported_intervals():
    cov = np.diag([1e-5, 1e-6])
    first = inverse((.2,), cov)
    together = inverse((.2, 0.), cov)
    assert first.time_lower[0] == together.time_lower[0]
    assert first.time_upper[0] == together.time_upper[0]
    assert np.isnan(together.in_vivo_time[1])
    assert inverse((.2,), cov).time_lower[0] == first.time_lower[0]


def test_failed_inverse_draws_are_counted_and_not_silently_conditioned(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', lambda *a, **k:
                        ParameterSamples(('a', 'b'), np.array([[1., -.1], [.5, -.1], [1., -.2]])))
    interval, note = inverse_response_prediction_intervals(
        exponential, PARAMS, np.eye(2), [], [.8, .2], (0., 10.), n_samples=3)
    assert interval['valid_counts'].tolist() == [2, 3]
    assert np.isnan(interval['lower'][0]) and np.isfinite(interval['lower'][1])
    assert sum(interval['inverse_failed_draw_reasons'][0].values()) == 1
    assert interval['inverse_failed_draw_reasons'][1] == {}
    assert note is not None


def test_ensemble_and_prediction_paths_use_the_same_inverse_policy():
    prediction = TimeCourse(np.array([10., 20.]), np.array([.6, .2]))
    ensemble = sample_inverse_response_trajectories(prediction, model_name='exponential',
        parameters=PARAMS, covariance=np.zeros((2, 2)), calibration_time_range=(0., 10.),
        observed_responses=[1., np.exp(-1.)], n_samples=10, random_state=12345)
    result = inverse()
    np.testing.assert_allclose(ensemble.central.in_vivo_time, result.in_vivo_time)
    np.testing.assert_allclose(ensemble.time_draws, np.tile(result.in_vivo_time, (10, 1)))
    assert ensemble.valid_draws.all()


def trajectory(method, input_time=(1.,), output_time=(2.,), response=(.5,)):
    return PredictedTrajectory(source_approach='approach1' if method == 'time_mapping' else 'approach2',
        source_method=method, in_vitro_time=np.array(input_time), in_vitro_value=np.array(response),
        in_vivo_time=np.array(output_time), in_vivo_value=np.array(response))


def annotate(result, **kwargs):
    return annotate_prediction_domain(result, in_vitro_time=[0., 10.], in_vivo_time=[0., 20.],
                                      in_vivo_response=[.2, 1.], **kwargs)


def test_approach1_checks_actual_paired_fit_ranges_not_unpaired_observations():
    result = annotate(trajectory('time_mapping', input_time=(8.,), output_time=(16.,)),
                      paired_in_vitro_time=[2., 6.], paired_in_vivo_time=[4., 12.])
    assert result.auxiliary['extrapolated'][0]
    assert 'paired' in result.auxiliary['extrapolation_reason'][0]


def test_inverse_ignores_unused_input_time_and_in_vitro_calibration_range():
    result = annotate(trajectory('observed_response_inverse', input_time=(9999.,), output_time=(10.,)))
    assert not result.auxiliary['extrapolated'][0]
    assert result.auxiliary['extrapolation_warnings'] == []


def test_inverse_flags_observed_response_range_even_when_time_is_inside():
    result = annotate(trajectory('observed_response_inverse', output_time=(1.,), response=(1.1,)))
    assert result.auxiliary['extrapolated'][0]
    assert 'response' in result.auxiliary['extrapolation_reason'][0]


def test_value_ratio_checks_both_evaluation_time_domains_not_scaled_response_magnitude():
    result = annotate(trajectory('value_ratio', input_time=(15.,), output_time=(15.,), response=(99.,)))
    assert result.auxiliary['extrapolated'][0]
    assert result.auxiliary['extrapolation_reason'][0] == 'evaluation time outside in vitro calibration range'


def test_interval_only_caution_does_not_label_the_central_point_extrapolated():
    result = trajectory('tau_ratio', output_time=(19.,))
    result.time_lower, result.time_upper = np.array([18.]), np.array([22.])
    annotate(result)
    assert not result.auxiliary['extrapolated'][0]
    assert result.auxiliary['interval_extrapolated'][0]
    assert result.auxiliary['extrapolation_warnings'][0]['label'] == 'Interval extends beyond calibration'


def test_unavailable_prediction_is_not_silently_labelled_in_range():
    result = annotate(trajectory('tau_ratio', input_time=(999.,), output_time=(np.nan,)))
    assert result.auxiliary['extrapolation_status'][0] == 'unavailable prediction'
    assert result.auxiliary['extrapolation_warnings'] == []


def test_direct_lookup_extrapolation_remains_unavailable_but_output_time_is_flagged():
    prediction = TimeCourse(np.array([.5, 2.]), np.array([.5, .5]))
    values = predict_direct_value_scaling(prediction, time_grid=[0., 1.],
        in_vitro_response_on_time_grid=[1., .5], in_vivo_response_on_time_grid=[1., .8])
    assert np.isfinite(values.in_vivo_value[0]) and np.isnan(values.in_vivo_value[1])
    assert 'lookup' in values.status[1]
    times = predict_direct_time_scaling(prediction, response_grid=[0., 1.],
        in_vitro_time_at_response=[1., .5], in_vivo_time_at_response=[2., 1.])
    annotate_prediction_domain(times, in_vitro_time=[0, 1], in_vivo_time=[0, 2])
    assert times.in_vivo_time[1] == 4.
    assert times.auxiliary['extrapolated'][1]


@pytest.mark.parametrize('kwargs', [{'minimum_time': 1.}, {'minimum_time': np.inf}, {'max_expansions': -1}, {'max_expansions': .5}])
def test_invalid_search_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        mapping([.5], **kwargs)


def test_public_api_applies_log_time_domain_and_preserves_annotations():
    from ivivc_core import IVIVCAnalysis, IVIVCModelFit, PreparedIVIVCData, PreprocessingMetadata
    from ivivc_core import predict_in_vivo, generate_trajectory_ensembles
    fit = SimpleNamespace(parameters={'a': 1., 'b': -1.}, covariance=np.zeros((2, 2)), residuals=np.zeros(3))
    prepared = PreparedIVIVCData(
        TimeCourse(np.array([0., .5, 1.]), np.array([1., .5, 0.])),
        TimeCourse(np.array([0., .5, 1.]), np.array([1., .5, 0.])),
        preprocessing=PreprocessingMetadata(applied_scalings=('log_x',)))
    result = IVIVCModelFit('approach2', 'linear', in_vitro_fit=fit, in_vivo_fit=fit)
    analysis = IVIVCAnalysis(prepared, {result.key: result})
    prediction = TimeCourse(np.array([7.]), np.array([2.]))
    methods = {'approach2': ['observed_response_inverse']}
    batch = predict_in_vivo(analysis, prediction, methods=methods, n_samples=4)
    assert not batch.errors
    output = next(iter(batch.results.values()))
    np.testing.assert_allclose(output.in_vivo_time, [-1.])
    assert output.auxiliary['extrapolated'][0]
    ensemble = generate_trajectory_ensembles(analysis, prediction, methods=methods, n_samples=4)
    assert not ensemble.errors
    np.testing.assert_allclose(next(iter(ensemble.results.values())).time_draws, -1.)


def test_rounded_logistic_plateau_is_not_accepted_as_a_finite_time():
    from models.mass_loss_one_phase import model_function
    result, status, _ = inverse_response_times(
        model_function, {'y0': 1., 'A': .8, 't50': 4., 's': 2.}, [.2], (0., 8.))
    assert np.isnan(result[0])
    assert 'flat' in status[0] or 'limit' in status[0]


def test_increasing_curve_can_be_extrapolated_without_assuming_degradation_direction():
    result, status, reason = inverse_response_times(
        lambda t, a, b: a + b * np.asarray(t), {'a': 1., 'b': 2.}, [1., 5., 25.], (0., 10.))
    assert reason is None
    np.testing.assert_allclose(result, [0., 2., 12.])
    assert 'extrapolated' in status[-1]


@pytest.mark.parametrize('scale', [1e-9, 1e9])
def test_inverse_time_unit_conversion_preserves_the_physical_solution(scale):
    result, _, _ = inverse_response_times(exponential, {'a': 1., 'b': -.1 / scale}, [.2], (0., 10. * scale))
    np.testing.assert_allclose(result / scale, [-10. * np.log(.2)], rtol=1e-9)
