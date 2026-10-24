"""Numerical contracts for conditional response-ratio parameter uncertainty."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from ivivc_core.predictions import predict_value_ratio, predict_value_ratio_with_uncertainty
from ivivc_core.types import ParameterSamples, TimeCourse
from ivivc_core.uncertainty import (
    evaluate_curve_draws,
    sample_value_ratio_trajectories,
)


def arguments(**changes):
    result = dict(
        model_name='exponential',
        in_vitro_parameters={'a': 1.0, 'b': 0.0},
        in_vitro_covariance=np.zeros((2, 2)),
        in_vivo_parameters={'a': 4.0, 'b': 0.0},
        in_vivo_covariance=np.diag([0.04, 0.0]),
        n_samples=257,
        random_state=123,
    )
    result.update(changes)
    return result


def test_bounds_are_percentiles_of_input_times_ratio_not_prediction_times_ratio():
    inputs = TimeCourse(np.array([1., 2.]), np.array([.2, .1]))
    kwargs = arguments()
    result = predict_value_ratio_with_uncertainty(inputs, **kwargs)
    draws = sample_value_ratio_trajectories(inputs, **kwargs)
    expected = np.percentile(draws.value_draws, [2.5, 97.5], axis=0)
    np.testing.assert_allclose(result.in_vivo_value, [.8, .4])
    np.testing.assert_allclose(result.value_lower, expected[0])
    np.testing.assert_allclose(result.value_upper, expected[1])
    ratio_quantiles = np.percentile(draws.metadata['ratio_draws'], [2.5, 97.5], axis=0)
    np.testing.assert_allclose(expected, inputs.value * ratio_quantiles)
    assert not np.allclose(expected, result.in_vivo_value * ratio_quantiles)
    assert np.all(result.auxiliary['interval_status'] == 'available')
    assert result.auxiliary['interval_kind'] == 'parameter_uncertainty'
    assert result.auxiliary['interval_random_seed'] == 123


@pytest.mark.parametrize('level', [1.0, 1e-4, 1e-12])
def test_identical_exact_profiles_have_ratio_one_without_a_denominator_floor(level):
    inputs = TimeCourse(np.array([0., 1.]), np.array([.3, .2]))
    result = predict_value_ratio_with_uncertainty(inputs, **arguments(
        in_vitro_parameters={'a': level, 'b': 0.},
        in_vivo_parameters={'a': level, 'b': 0.},
        in_vivo_covariance=np.zeros((2, 2)),
    ))
    np.testing.assert_array_equal(result.scaling_factor, [1., 1.])
    np.testing.assert_array_equal(result.in_vivo_value, inputs.value)
    np.testing.assert_array_equal(result.value_lower, inputs.value)
    np.testing.assert_array_equal(result.value_upper, inputs.value)


@pytest.mark.parametrize('input_value', [.2, 0., -.2])
def test_fixed_input_is_scaled_once_and_percentile_order_survives_negative_values(input_value):
    inputs = TimeCourse(np.array([1.]), np.array([input_value]))
    result = predict_value_ratio_with_uncertainty(inputs, **arguments())
    ensemble = sample_value_ratio_trajectories(inputs, **arguments())
    expected = np.percentile(ensemble.value_draws, [2.5, 97.5], axis=0)
    np.testing.assert_allclose(result.value_lower, expected[0])
    np.testing.assert_allclose(result.value_upper, expected[1])
    assert result.value_lower[0] <= result.value_upper[0]


def test_point_prediction_is_still_the_fitted_parameter_result():
    inputs = TimeCourse(np.array([2.]), np.array([.2]))
    kwargs = arguments(in_vitro_covariance=np.diag([.09, 0.]))
    interval = predict_value_ratio_with_uncertainty(inputs, **kwargs)
    point = predict_value_ratio(inputs, **{key: kwargs[key] for key in (
        'model_name', 'in_vitro_parameters', 'in_vivo_parameters',
    )})
    np.testing.assert_array_equal(interval.in_vivo_value, point.in_vivo_value)


def test_seeded_results_do_not_depend_on_other_requested_times():
    alone = TimeCourse(np.array([1.]), np.array([.2]))
    together = TimeCourse(np.array([0., 1., 2.]), np.array([.9, .2, .1]))
    one = predict_value_ratio_with_uncertainty(alone, **arguments())
    many = predict_value_ratio_with_uncertainty(together, **arguments())
    repeated = predict_value_ratio_with_uncertainty(alone, **arguments())
    for key in ('in_vivo_value', 'value_lower', 'value_upper'):
        np.testing.assert_array_equal(getattr(one, key), getattr(repeated, key))
        np.testing.assert_array_equal(getattr(one, key), getattr(many, key)[1:2])


def test_invalid_late_time_does_not_discard_valid_earlier_draws():
    kwargs = arguments(
        in_vitro_parameters={'a': 1., 'b': -.2},
        in_vitro_covariance=np.diag([0., .01]),
    )
    inputs = TimeCourse(np.array([0., 10000.]), np.array([.2, .1]))
    ensemble = sample_value_ratio_trajectories(inputs, **kwargs)
    result = predict_value_ratio_with_uncertainty(inputs, **kwargs)
    alone = predict_value_ratio_with_uncertainty(TimeCourse(np.array([0.]), np.array([.2])), **kwargs)
    assert ensemble.n_valid_draws < ensemble.n_draws
    assert np.isfinite(ensemble.value_draws[:, 0]).all()
    assert result.auxiliary['interval_valid_samples'][0] == kwargs['n_samples']
    assert result.auxiliary['interval_status'][0] == 'available'
    np.testing.assert_array_equal(result.value_lower[:1], alone.value_lower)
    np.testing.assert_array_equal(result.value_upper[:1], alone.value_upper)
    assert np.isnan(result.value_lower[1])


def test_pointwise_curve_evaluation_handles_a_vector_domain_exception():
    def limited(x, a):
        if np.any(x > 1.):
            raise ValueError('Outside model domain')
        return a * np.ones_like(x)

    samples = ParameterSamples(('a',), np.array([[1.], [2.]]))
    partial = evaluate_curve_draws(limited, [0., 2.], samples, require_finite=False)
    np.testing.assert_array_equal(partial.values[:, 0], [1., 2.])
    assert np.isnan(partial.values[:, 1]).all()
    assert not partial.valid_draws.any()
    strict = evaluate_curve_draws(limited, [0., 2.], samples)
    assert np.isnan(strict.values).all()


def test_response_unit_change_scales_points_and_bounds_not_the_ratio():
    inputs = TimeCourse(np.array([1., 2.]), np.array([.2, .1]))
    kwargs = arguments(in_vitro_covariance=np.diag([.0025, 0.]))
    original = predict_value_ratio_with_uncertainty(inputs, **kwargs)
    scale = 1000.
    scaled_kwargs = dict(kwargs)
    for side in ('in_vitro', 'in_vivo'):
        parameters = dict(kwargs[f'{side}_parameters'])
        parameters['a'] *= scale
        scaled_kwargs[f'{side}_parameters'] = parameters
        transform = np.diag([scale, 1.])
        scaled_kwargs[f'{side}_covariance'] = transform @ kwargs[f'{side}_covariance'] @ transform
    scaled = predict_value_ratio_with_uncertainty(
        TimeCourse(inputs.time, inputs.value * scale), **scaled_kwargs,
    )
    np.testing.assert_allclose(scaled.scaling_factor, original.scaling_factor)
    for key in ('in_vivo_value', 'value_lower', 'value_upper'):
        np.testing.assert_allclose(getattr(scaled, key), scale * getattr(original, key), rtol=1e-12)


def test_denominator_zero_crossing_withholds_bounds_not_finite_point():
    result = predict_value_ratio_with_uncertainty(
        TimeCourse(np.array([0.]), np.array([.2])),
        **arguments(in_vitro_parameters={'a': .05, 'b': 0.},
                    in_vitro_covariance=np.diag([.04, 0.])),
    )
    assert np.isfinite(result.in_vivo_value[0])
    assert np.isnan(result.value_lower[0]) and np.isnan(result.value_upper[0])
    assert result.auxiliary['denominator_lower'][0] < 0 < result.auxiliary['denominator_upper'][0]
    assert result.auxiliary['interval_valid_samples'][0] == 257
    assert 'denominator interval includes zero' in result.auxiliary['interval_status'][0]
    assert result.note is not None


@pytest.mark.parametrize('denominator', [0., np.inf, np.nan])
def test_undefined_central_denominator_is_not_silently_repaired(denominator):
    with np.errstate(all='ignore'):
        result = predict_value_ratio_with_uncertainty(
            TimeCourse(np.array([0.]), np.array([.2])),
            **arguments(in_vitro_parameters={'a': denominator, 'b': 0.}),
        )
    assert np.isnan(result.in_vivo_value[0])
    assert np.isnan(result.value_lower[0])
    assert not result.available
    assert 'zero or nonfinite fitted in vitro' in result.status[0]


@pytest.mark.parametrize('covariance', [None, np.ones((1, 1)), np.full((2, 2), np.nan),
                                        np.array([[1., 2.], [2., 1.]])])
def test_sampling_failure_keeps_central_prediction_and_explicit_diagnostics(covariance):
    result = predict_value_ratio_with_uncertainty(
        TimeCourse(np.array([0.]), np.array([.2])),
        **arguments(in_vitro_covariance=covariance),
    )
    np.testing.assert_allclose(result.in_vivo_value, [.8])
    assert np.isnan(result.value_lower[0])
    assert result.auxiliary['interval_total_samples'] == 0
    assert result.auxiliary['interval_requested_samples'] == 257
    assert result.auxiliary['interval_sampling_error']
    assert 'sampling failed' in result.auxiliary['interval_status'][0]


def test_single_bad_draw_is_not_silently_dropped(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    real_sampler = uncertainty.sample_value_ratio_trajectories

    def one_bad_draw(*args, **kwargs):
        ensemble = real_sampler(*args, **kwargs)
        ensemble.value_draws[0, 1] = np.nan
        ensemble.valid_draws[0] = False
        return ensemble

    monkeypatch.setattr(uncertainty, 'sample_value_ratio_trajectories', one_bad_draw)
    result = predict_value_ratio_with_uncertainty(
        TimeCourse(np.array([0., 1.]), np.array([.2, .1])), **arguments(),
    )
    assert result.auxiliary['interval_status'][0] == 'available'
    assert result.auxiliary['interval_valid_samples'][1] == 256
    assert np.isnan(result.value_lower[1])
    assert 'missing Monte Carlo draws' in result.auxiliary['interval_status'][1]


@pytest.mark.parametrize('changes', [{'confidence_level': 0.}, {'confidence_level': 1.},
                                     {'n_samples': 1}, {'n_samples': 2.5}])
def test_invalid_interval_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        predict_value_ratio_with_uncertainty(
            TimeCourse(np.array([0.]), np.array([.2])), **arguments(**changes),
        )


def test_interval_api_has_no_residual_or_future_measurement_noise_inputs():
    parameters = inspect.signature(predict_value_ratio_with_uncertainty).parameters
    assert not any('residual' in name or name.endswith('_sd') for name in parameters)


def test_public_value_ratio_api_returns_the_same_interval_contract(fast_analysis_config):
    from ivivc_core import fit_ivivc, predict_in_vivo, prepare_data
    time = np.linspace(0., 5., 6)
    perturbation = np.array([1., .99, 1.01, .995, 1.005, 1.])
    prepared = prepare_data(time, np.exp(-.2 * time) * perturbation,
                            time, np.exp(-.1 * time) * perturbation,
                            selected_interpolation=['none'])
    analysis = fit_ivivc(prepared, selections=['approach2:exponential'],
                         selected_metrics=['rmse'], analysis_config=fast_analysis_config)
    assert not analysis.errors
    batch = predict_in_vivo(analysis, TimeCourse(np.array([1., 2.]), np.array([.8, .6])),
                            methods={'approach2': 'value_ratio'}, n_samples=40)
    assert not batch.errors
    result = batch['approach2:exponential:value_ratio']
    assert result.value_lower.shape == (2,)
    assert result.auxiliary['interval_kind'] == 'parameter_uncertainty'
    assert result.auxiliary['interval_requested_samples'] == 40
