"""Inverse-only conditional intervals and backend configuration contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from ivivc_core import IVIVCAnalysis, IVIVCModelFit, PreparedIVIVCData, predict_in_vivo
from ivivc_core.intervals import parameter_interval_summary
from ivivc_core.predictions import inverse_response_prediction_intervals, predict_inverse_response
from ivivc_core.types import ParameterSamples, TimeCourse
from models.exponential import model_function


PARAMS = {'a': 1., 'b': -.1}


def _draws(monkeypatch, successful, count=100):
    import ivivc_core.uncertainty as uncertainty
    values = np.column_stack((np.linspace(.9, 1.1, count), np.full(count, -.1)))
    values[successful:, 0] = .5  # .8 then requires a disallowed negative time.
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets',
                        lambda *a, **k: ParameterSamples(('a', 'b'), values))
    return values


def _interval(**kwargs):
    return inverse_response_prediction_intervals(
        model_function, PARAMS, np.eye(2), [], [.8, .2], (0., 10.),
        n_samples=100, **kwargs)


@pytest.mark.parametrize('successful,policy,fraction,available,conditional', [
    (100, 'conditional', .95, True, False),
    (95, 'conditional', .95, True, True),
    (94, 'conditional', .95, False, False),
    (0, 'conditional', .95, False, False),
    (50, 'conditional', .5, True, True),
    (49, 'conditional', .5, False, False),
    (95, 'strict', .95, False, False),
    (95, 'conditional', .99, False, False),
    (94, 'conditional', .94, True, True),
    (1, 'conditional', .01, False, False),
])
def test_success_threshold_is_pointwise_and_preserves_percentiles(
        monkeypatch, successful, policy, fraction, available, conditional):
    values = _draws(monkeypatch, successful)
    result, note = _interval(interval_policy=policy, min_valid_fraction=fraction)
    assert bool(np.isfinite(result['lower'][0])) == available
    assert bool(result['interval_conditional'][0]) == conditional
    assert result['valid_counts'].tolist() == [successful, 100]
    assert np.isfinite(result['lower'][1])
    assert result['interval_status'][1] == 'available'
    assert sum(result['inverse_failed_draw_reasons'][0].values()) == 100 - successful
    assert result['inverse_failed_draw_reasons'][1] == {}
    if available:
        expected = np.percentile(10 * np.log(values[:successful, 0] / .8), [2.5, 97.5])
        np.testing.assert_allclose([result['lower'][0], result['upper'][0]], expected)
    if conditional:
        assert 'conditional' in result['interval_status'][0]
        assert 'successful mappings only' in note
        assert 'could not be calculated' not in note
    elif successful == 100:
        assert note is None


def test_missing_draws_use_requested_count_not_returned_count(monkeypatch):
    _draws(monkeypatch, 94, count=94)
    result, _ = _interval(interval_policy='conditional', min_valid_fraction=.95)
    assert result['interval_requested_samples'] == 100
    assert result['interval_total_samples'] == 94
    assert result['valid_counts'].tolist() == [94, 94]
    assert np.isnan(result['lower']).all()


def test_strict_summary_remains_default_and_sampling_failure_cannot_supply_bounds():
    draws = np.arange(100., dtype=float).reshape(-1, 1)
    draws[-1] = np.nan
    strict = parameter_interval_summary(draws, [1.], n_samples=100)
    assert np.isnan(strict['lower']).all()
    conditional = parameter_interval_summary(draws, [1.], n_samples=100, min_valid_fraction=.95)
    assert np.isfinite(conditional['lower']).all()
    failed = parameter_interval_summary(np.ones((100, 1)), [1.], n_samples=100,
                                       min_valid_fraction=.95, sampling_error='invalid covariance')
    assert np.isnan(failed['lower']).all()
    assert 'sampling failed' in failed['interval_status'][0]


def _prediction(**kwargs):
    return predict_inverse_response(
        TimeCourse([1., 2.], [.8, .2]), model_name='exponential', parameters=PARAMS,
        covariance=np.eye(2), residuals=[], calibration_time_range=(0., 10.),
        n_samples=100, **kwargs)


@pytest.mark.parametrize('configured_fraction', [.95, .5])
def test_backend_defaults_are_read_at_call_time_and_centers_remain_nominal(monkeypatch, configured_fraction):
    import ivivc_core.predictions as predictions
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_INTERVAL_POLICY', 'conditional')
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_MIN_VALID_FRACTION', configured_fraction)
    _draws(monkeypatch, 95)
    default = _prediction()
    np.testing.assert_allclose(default.in_vivo_time, -10 * np.log([.8, .2]))
    assert default.auxiliary['interval_policy'] == 'conditional'
    assert default.auxiliary['interval_min_valid_fraction'] == configured_fraction
    assert default.auxiliary['interval_conditional'].tolist() == [True, False]
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_MIN_VALID_FRACTION', .99)
    assert np.isnan(_prediction().time_lower[0])
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_INTERVAL_POLICY', 'strict')
    strict = _prediction()
    np.testing.assert_array_equal(default.in_vivo_time, strict.in_vivo_time)
    assert np.isnan(strict.time_lower[0])
    assert strict.auxiliary['interval_min_valid_fraction'] == 1.
    assert np.isfinite(_prediction(interval_policy='conditional', min_valid_fraction=.95).time_lower[0])


def test_unavailable_nominal_and_parameter_sampling_stay_unavailable(monkeypatch):
    _draws(monkeypatch, 100)
    prediction = predict_inverse_response(
        TimeCourse([1., 2.], [1.2, .2]), model_name='exponential', parameters=PARAMS,
        covariance=np.eye(2), residuals=[], calibration_time_range=(0., 10.), n_samples=100)
    assert np.isnan(prediction.in_vivo_time[0]) and np.isnan(prediction.time_lower[0])
    assert np.isfinite(prediction.time_lower[1])
    import ivivc_core.uncertainty as uncertainty
    def fail(*args, **kwargs):
        raise ValueError('invalid covariance')
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', fail)
    result = _prediction()
    assert np.isfinite(result.in_vivo_time).all()
    assert np.isnan(result.time_lower).all()
    assert not result.auxiliary['interval_conditional'].any()
    assert 'invalid covariance' == result.auxiliary['interval_sampling_error']


@pytest.mark.parametrize('options', [
    {'interval_policy': 'unknown'}, {'min_valid_fraction': 0},
    {'min_valid_fraction': 1.01}, {'min_valid_fraction': np.nan},
    {'min_valid_fraction': True}, {'min_valid_fraction': '0.95'},
    {'min_valid_fraction': .95 + 0j},
])
def test_backend_settings_are_validated_even_without_a_nominal_inverse(options):
    with pytest.raises(ValueError):
        predict_inverse_response(
            TimeCourse([1.], [1.2]), model_name='exponential', parameters=PARAMS,
            covariance=np.eye(2), residuals=[], calibration_time_range=(0., 10.), **options)


def test_public_api_forwards_inverse_settings(monkeypatch):
    import ivivc_core.predictions as predictions
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_INTERVAL_POLICY', 'conditional')
    monkeypatch.setattr(predictions, 'DEFAULT_INVERSE_MIN_VALID_FRACTION', .95)
    _draws(monkeypatch, 95)
    course = TimeCourse([0., 10.], [1., np.exp(-1.)])
    fit = SimpleNamespace(parameters=PARAMS, covariance=np.eye(2), residuals=np.zeros(2))
    model = IVIVCModelFit('approach2', 'exponential', in_vitro_fit=fit, in_vivo_fit=fit)
    analysis = IVIVCAnalysis(PreparedIVIVCData(course, course), {model.key: model})
    prediction = TimeCourse([1.], [.8])
    options = dict(methods={'approach2': ['observed_response_inverse']}, n_samples=100)
    default = predict_in_vivo(analysis, prediction, **options)
    strict = predict_in_vivo(analysis, prediction, inverse_interval_policy='strict', **options)
    higher = predict_in_vivo(analysis, prediction, inverse_min_valid_fraction=.99, **options)
    assert not default.errors and not strict.errors and not higher.errors
    key = model.key + ':observed_response_inverse'
    assert np.isfinite(default[key].time_lower[0])
    assert np.isnan(strict[key].time_lower[0]) and np.isnan(higher[key].time_lower[0])


def test_conditional_input_draws_preserve_nominal_center_and_unconditional_response_bounds():
    from scipy.stats import norm
    from ivivc_core import ObservationUncertainty, resolve_observation_uncertainty
    from ivivc_core.prediction_inputs import sample_prediction_responses

    prediction = TimeCourse([1.], [.9], observation_uncertainty=resolve_observation_uncertainty(
        ObservationUncertainty(.05, response_kind='mean'), 1))
    sampled = sample_prediction_responses(prediction, 100, 12345)[:, 0]
    valid = (sampled > 0) & (sampled <= 1)
    assert 95 <= np.count_nonzero(valid) < 100
    result = predict_inverse_response(
        prediction, model_name='exponential', parameters=PARAMS,
        covariance=np.zeros((2, 2)), residuals=[], calibration_time_range=(0., 10.),
        n_samples=100, random_state=12345, interval_policy='conditional', min_valid_fraction=.95)
    expected_time = np.percentile(-10 * np.log(sampled[valid]), [2.5, 97.5])
    np.testing.assert_allclose([result.time_lower[0], result.time_upper[0]], expected_time)
    np.testing.assert_allclose(result.in_vivo_time, [-10 * np.log(.9)])
    np.testing.assert_allclose(result.value_lower, [.9 - norm.ppf(.975) * .05])
    np.testing.assert_allclose(result.value_upper, [.9 + norm.ppf(.975) * .05])
    assert result.auxiliary['interval_conditional'].tolist() == [True]
    assert result.auxiliary['interval_valid_samples'][0] == np.count_nonzero(valid)
