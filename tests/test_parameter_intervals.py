"""Common conditional parameter-only intervals, independent of presentation."""
from types import SimpleNamespace

import numpy as np
import pytest

from ivivc_core import TimeCourse, ParameterSamples
from ivivc_core.intervals import curve_parameter_intervals, parameter_interval_summary
from ivivc_core.predictions import (
    predict_time_mapping, predict_tau_ratio, predict_inverse_response,
    inverse_response_prediction_intervals, inverse_response_times,
)
from utilities.evaluation import generate_confidence_bands, generate_prediction_bands
from models.exponential import model_function as exponential


def linear(t, a, b):
    return a + b * np.asarray(t)


def tau_prediction(*, covariance=None, random_state=12345, n_samples=100, times=(0., 1., 2.)):
    covariance = np.diag([0., 1e-5]) if covariance is None else covariance
    return predict_tau_ratio(
        TimeCourse(np.asarray(times), np.full(len(times), .5)), model_name='exponential',
        tau_values=[SimpleNamespace(n=5., s=.1), SimpleNamespace(n=10., s=.2)],
        in_vitro_parameters={'a': 1., 'b': -.2}, in_vitro_covariance=covariance,
        in_vivo_parameters={'a': 1., 'b': -.1}, in_vivo_covariance=covariance,
        n_samples=n_samples, random_state=random_state,
    )


@pytest.mark.parametrize('residuals', [None, [], [0., 0.], [10., -30., 15.], [np.nan]])
def test_legacy_band_alias_adds_no_noise_or_residual_degrees_of_freedom(residuals):
    args = (linear, np.array([1., 3.]), {'a': 2., 'b': 3.}, np.diag([.1, .2]))
    actual = generate_prediction_bands(*args, residuals=residuals, n_samples=100)
    expected = curve_parameter_intervals(*args, n_samples=100)
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_zero_covariance_collapses_fitting_and_approach1_intervals():
    x = np.array([0., 1., 2.])
    params = {'a': 1., 'b': 2.}
    expected = linear(x, **params)
    band = curve_parameter_intervals(linear, x, params, np.zeros((2, 2)), n_samples=20)
    prediction = predict_time_mapping(
        TimeCourse(x, np.ones(3)), model_name='linear', parameters=params,
        covariance=np.zeros((2, 2)), include_uncertainty=True, n_samples=20,
    )
    for values in (band['lower'], band['upper'], prediction.time_lower, prediction.time_upper):
        np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(band['std'], 0.)
    assert prediction.auxiliary['interval_kind'] == 'parameter_uncertainty'


def test_percentiles_propagate_the_full_covariance_not_independent_parameters():
    from ivivc_core.uncertainty import sample_parameter_sets
    params = {'a': 1., 'b': 2.}
    covariance = np.array([[.04, -.03], [-.03, .09]])
    x = np.array([0., 1., 5.])
    draws = sample_parameter_sets(params, covariance, n_samples=400, random_state=17).values
    expected = np.percentile(draws[:, :1] + draws[:, 1:] * x, [2.5, 97.5], axis=0)
    result = curve_parameter_intervals(linear, x, params, covariance, n_samples=400, random_state=17)
    np.testing.assert_array_equal(result['lower'], expected[0])
    np.testing.assert_array_equal(result['upper'], expected[1])


@pytest.mark.parametrize('raises', [False, True])
def test_bad_requested_point_does_not_condition_intervals_at_other_points(raises):
    def function(t, a):
        t = np.asarray(t)
        if raises and a < 0 and np.any(t == 0):
            raise ValueError('undefined only at zero for this draw')
        return np.where((t == 0) & (a < 0), np.nan, t * (1 + a))
    args = ({'a': 0.}, np.array([[1.]]))
    first = curve_parameter_intervals(function, [2.], *args, n_samples=100)
    together = curve_parameter_intervals(function, [0., 2.], *args, n_samples=100)
    assert 0 < together['valid_counts'][0] < 100
    assert np.isnan(together['lower'][0])
    assert 'unavailable' in together['interval_status'][0]
    for key in ('lower', 'upper', 'mean', 'std'):
        assert first[key][0] == together[key][1]


@pytest.mark.parametrize('covariance', [None, np.full((2, 2), np.inf), np.eye(3), np.diag([1., -1.])])
def test_bad_covariance_preserves_approach1_point_and_reports_missing_interval(covariance):
    result = predict_time_mapping(
        TimeCourse(np.array([1., 2.]), np.array([.8, .5])), model_name='linear',
        parameters={'a': 1., 'b': 2.}, covariance=covariance, include_uncertainty=True,
        n_samples=10,
    )
    np.testing.assert_array_equal(result.in_vivo_time, [3., 5.])
    assert np.isnan(result.time_lower).all()
    assert result.auxiliary['interval_total_samples'] == 0
    assert result.auxiliary['interval_sampling_error']
    assert result.note


@pytest.mark.parametrize('n_samples,confidence_level', [(1, .95), (2.5, .95), (10, 0.), (10, 1.), (10, np.nan)])
def test_invalid_settings_fail_explicitly(n_samples, confidence_level):
    with pytest.raises(ValueError):
        curve_parameter_intervals(linear, [1.], {'a': 1., 'b': 2.}, np.eye(2),
                                  n_samples=n_samples, confidence_level=confidence_level)


def test_short_or_nonfinite_draws_are_not_silently_dropped():
    result = parameter_interval_summary([[1., 2.], [1., np.nan]], [1., 2.], n_samples=2)
    assert result['lower'][0] == 1.
    assert np.isnan(result['lower'][1])
    np.testing.assert_array_equal(result['valid_counts'], [2, 1])
    short = parameter_interval_summary([[1., 2.], [1., 2.]], [1., 2.], n_samples=3)
    assert np.isnan(short['lower']).all()
    assert short['interval_total_samples'] == 2


def test_seeded_fitting_bands_do_not_use_or_modify_global_random_state():
    args = (linear, [1., 2.], {'a': 1., 'b': 2.}, np.eye(2))
    np.random.seed(7)
    state = np.random.get_state()
    first = generate_confidence_bands(*args, n_samples=20)
    actual = np.random.rand()
    np.random.set_state(state)
    assert actual == np.random.rand()
    np.random.seed(999)
    second = generate_confidence_bands(*args, n_samples=20)
    np.testing.assert_array_equal(first['lower'], second['lower'])


def test_tau_zero_covariance_collapses_to_deterministic_fitted_time_scaling():
    result = tau_prediction(covariance=np.zeros((2, 2)), n_samples=10)
    np.testing.assert_allclose(result.in_vivo_time, [0., 2., 4.])
    np.testing.assert_allclose(result.time_lower, result.in_vivo_time, atol=1e-12)
    np.testing.assert_allclose(result.time_upper, result.in_vivo_time, atol=1e-12)
    assert result.auxiliary['interval_total_samples'] == 10


def test_tau_intervals_are_direct_nonlinear_percentiles(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    vitro = np.array([[1., -.2], [1., -.4], [1., -.2], [1., -.1]])
    vivo = np.array([[1., -.1], [1., -.1], [1., -.4], [1., -.1]])
    calls = []
    def sample(parameters, covariance, **kwargs):
        values = vitro if len(calls) == 0 else vivo
        calls.append(kwargs)
        return ParameterSamples(('a', 'b'), values)
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', sample)
    result = tau_prediction(n_samples=4, times=(1., 3.))
    expected = np.percentile((vitro[:, 1] / vivo[:, 1])[:, None] * np.array([1., 3.]),
                             [2.5, 97.5], axis=0)
    np.testing.assert_allclose(result.time_lower, expected[0])
    np.testing.assert_allclose(result.time_upper, expected[1])
    np.testing.assert_array_equal(result.in_vivo_time, [2., 6.])
    assert len(calls) == 2


def test_one_invalid_sampled_tau_withholds_interval_not_point(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    samples = np.array([[1., -.2], [1., .1], [1., -.3]])
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets',
                        lambda *a, **k: ParameterSamples(('a', 'b'), samples))
    result = tau_prediction(n_samples=3)
    assert result.available
    assert np.isnan(result.time_lower).all()
    np.testing.assert_array_equal(result.auxiliary['interval_valid_samples'], 2)
    assert not result.auxiliary['interval_available']


def test_tau_center_independent_of_seed_and_interval_asymmetric():
    first = tau_prediction(random_state=3)
    second = tau_prediction(random_state=17)
    np.testing.assert_array_equal(first.in_vivo_time, second.in_vivo_time)
    assert first.time_lower[1] != second.time_lower[1]
    assert not np.allclose(first.time_error_lower[1:], first.time_error_upper[1:])


def inverse(*, responses=(.8, .6), covariance=None, residuals=(), n_samples=30, random_state=12345):
    covariance = np.diag([1e-5, 1e-6]) if covariance is None else covariance
    return predict_inverse_response(
        TimeCourse(np.arange(len(responses), dtype=float), np.asarray(responses)),
        model_name='exponential', parameters={'a': 1., 'b': -.1}, covariance=covariance,
        residuals=residuals, calibration_time_range=(0., 20.),
        observed_responses=np.exp(-.1 * np.linspace(0., 20., 10)),
        n_samples=n_samples, random_state=random_state,
    )


@pytest.mark.parametrize('residuals', [(), (0.,), (100., -200., 300.), (np.nan,)])
def test_inverse_holds_response_fixed_and_does_not_require_residual_df(residuals):
    result = inverse(covariance=np.zeros((2, 2)), residuals=residuals, n_samples=10)
    np.testing.assert_allclose(result.in_vivo_time, -10 * np.log([.8, .6]), atol=1e-9)
    np.testing.assert_array_equal(result.time_lower, result.in_vivo_time)
    np.testing.assert_array_equal(result.time_upper, result.in_vivo_time)
    assert result.auxiliary['assumed_response_sd'] == 0.


def test_inverse_pointwise_intervals_are_reproducible_and_do_not_depend_on_other_targets():
    single = inverse(responses=(.6,))
    multiple = inverse(responses=(.6, 1.5))  # Second response would require a negative physical time.
    assert single.time_lower[0] == multiple.time_lower[0]
    assert single.time_upper[0] == multiple.time_upper[0]
    assert np.isnan(multiple.in_vivo_time[1])
    assert np.isnan(multiple.time_lower[1])
    assert inverse(responses=(.6,)).time_lower[0] == single.time_lower[0]


def test_inverse_zero_covariance_does_not_make_a_plateau_unique():
    # A zero-response plateau has no unique inverse, even with exact parameters.
    from models.mass_loss_surface_erosion import model_function
    params = {'y0': 1., 'k': .1, 'n': 1., 'A':0}
    central, _, _ = inverse_response_times(model_function, params, [0.], (0., 20.), observed_responses=[0., 1.])
    interval, _ = inverse_response_prediction_intervals(
        model_function, params, np.zeros((3, 3)), [], [0.], (0., 20.),
        observed_responses=[0., 1.], n_samples=10,
    )
    assert np.isnan(central).all()
    assert np.isnan(interval['lower']).all()
    assert np.isnan(interval['upper']).all()


def test_inverse_failed_draw_counts_are_not_hidden(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', lambda *a, **k:
                        ParameterSamples(('a',), np.array([[1.], [-1.], [2.]])))
    def mapper(function, params, targets, time_range, **kwargs):
        targets = np.asarray(targets)
        values = targets * params['a']
        values[(targets > 1) & (params['a'] < 0)] = np.nan
        return values, np.full(targets.shape, 'mapped'), None
    result, note = inverse_response_prediction_intervals(
        None, {'a': 1.}, np.eye(1), [], [1., 2.], (0., 10.), n_samples=3,
        inverse_mapper=mapper,
    )
    assert np.isfinite(result['lower'][0])
    assert np.isnan(result['lower'][1])
    np.testing.assert_array_equal(result['valid_counts'], [3, 2])
    assert note == (
        'An uncertainty interval could not be calculated for 1 of 2 predictions because '
        'some parameter estimates did not produce a unique inverse mapping.'
    )


def test_inverse_missing_point_does_not_duplicate_interval_warning():
    result = inverse(responses=(.6, 1.5), covariance=np.zeros((2, 2)), n_samples=3)
    assert result.note == (
        '1 of 2 predictions could not be calculated because the response does not have '
        'a unique inverse mapping to in vivo time.'
    )


def test_response_unit_conversion_preserves_conditional_curve_intervals():
    x = np.array([1., 2., 5.])
    covariance = np.diag([.04, 1e-5])
    original = curve_parameter_intervals(exponential, x, {'a': 1., 'b': -.1}, covariance, n_samples=100)
    transform = np.diag([100., 1.])
    scaled = curve_parameter_intervals(exponential, x, {'a': 100., 'b': -.1},
                                       transform @ covariance @ transform, n_samples=100)
    # Sampling uses the same Gaussian directions for this diagonal scaling.
    for key in ('lower', 'upper'):
        np.testing.assert_allclose(scaled[key], original[key] * 100., rtol=1e-12)
