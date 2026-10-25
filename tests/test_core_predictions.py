from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import engine
from ivivc_core import (
    PredictedTrajectory,
    TimeCourse,
    predict_direct_time_scaling,
    predict_direct_value_scaling,
    predict_inverse_response,
    predict_tau_ratio,
    predict_time_mapping,
    predict_value_ratio,
)


def test_predicted_trajectory_copies_inputs_and_validates_lengths():
    time = np.array([0.0, 1.0])
    value = np.array([1.0, 0.5])
    trajectory = PredictedTrajectory(
        source_approach='approach1',
        source_method='time_mapping',
        in_vitro_time=time,
        in_vitro_value=value,
        in_vivo_time=2.0 * time,
        in_vivo_value=value,
        auxiliary={'grid': time},
    )

    time[0] = 99.0
    value[0] = 99.0
    np.testing.assert_array_equal(trajectory.in_vitro_time, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(trajectory.in_vitro_value, np.array([1.0, 0.5]))
    np.testing.assert_array_equal(trajectory.auxiliary['grid'], np.array([0.0, 1.0]))

    try:
        PredictedTrajectory(
            source_approach='approach1',
            source_method='time_mapping',
            in_vitro_time=np.array([0.0, 1.0]),
            in_vitro_value=np.array([1.0]),
            in_vivo_time=np.array([0.0, 2.0]),
            in_vivo_value=np.array([1.0, 0.5]),
        )
    except ValueError as exc:
        assert 'must match the input length' in str(exc)
    else:
        raise AssertionError('Mismatched trajectory arrays should be rejected.')


def test_time_mapping_and_value_ratio_use_common_trajectory_representation():
    prediction = TimeCourse(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.7, 0.4]),
    )

    time_mapping = predict_time_mapping(
        prediction,
        model_name='linear',
        parameters={'a': 1.0, 'b': 2.0},
    )
    assert time_mapping.source_method == 'time_mapping'
    np.testing.assert_allclose(time_mapping.in_vivo_time, np.array([1.0, 3.0, 5.0]))
    np.testing.assert_array_equal(time_mapping.in_vivo_value, prediction.value)

    value_ratio = predict_value_ratio(
        prediction,
        model_name='exponential',
        in_vitro_parameters={'a': 1.0, 'b': -0.2},
        in_vivo_parameters={'a': 1.0, 'b': -0.1},
    )
    expected_ratio = np.exp(0.1 * prediction.time)
    np.testing.assert_allclose(value_ratio.scaling_factor, expected_ratio)
    np.testing.assert_allclose(value_ratio.in_vivo_value, prediction.value * expected_ratio)
    np.testing.assert_array_equal(value_ratio.in_vivo_time, prediction.time)
    assert len(value_ratio.auxiliary['plot_time']) == 50


def test_tau_ratio_returns_time_uncertainty_or_structured_skip_reason():
    prediction = TimeCourse(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.8, 0.6]),
    )
    result = predict_tau_ratio(
        prediction,
        model_name='exponential',
        tau_values=[SimpleNamespace(n=5.0, s=0.1), SimpleNamespace(n=10.0, s=0.2)],
        residual_dof=[4, 6],
        in_vitro_parameters={'a': 1., 'b': -.2},
        in_vitro_covariance=np.diag([0., 1e-6]),
        in_vivo_parameters={'a': 1., 'b': -.1},
        in_vivo_covariance=np.diag([0., 1e-6]),
    )

    assert result.available is True
    assert result.skip_reason is None
    np.testing.assert_allclose(result.in_vivo_time, 2.0 * prediction.time)
    np.testing.assert_allclose(result.scaling_factor, 2.0)
    assert result.time_error_upper[0] == 0.0
    assert np.all(result.time_error_upper[1:] > 0.0)
    assert result.auxiliary['interval_kind'] == 'parameter_uncertainty'
    assert result.auxiliary['interval_requested_samples'] == 1000
    assert np.all(result.time_lower <= result.time_upper)

    unavailable = predict_tau_ratio(
        prediction,
        model_name='exponential',
        tau_values=[SimpleNamespace(n=0.0, s=1.0), SimpleNamespace(n=10.0, s=0.2)],
    )
    assert unavailable.available is False
    assert 'In vitro tau is not positive' in unavailable.skip_reason
    assert np.all(np.isnan(unavailable.in_vivo_time))


def test_inverse_response_prediction_retains_bounds_status_and_interval_metadata():
    t_pred = np.array([0.0, 2.5, 5.0])
    prediction = TimeCourse(t_pred, np.exp(-0.2 * t_pred))
    observed_response = np.exp(-0.1 * np.linspace(0.0, 10.0, 6))

    def fixed_intervals(
        model_function,
        params,
        pcov,
        residuals,
        target_responses,
        time_range,
        observed_responses=None,
        confidence_level=0.95,
        n_samples=1000,
        grid_points=1001,
        **kwargs,
    ):
        mapped = 2.0 * t_pred
        return {
            'lower': mapped - 0.2,
            'upper': mapped + 0.2,
            'valid_counts': np.full(mapped.shape, n_samples, dtype=int),
            'n_samples': n_samples,
            'residual_std': 0.0,
            'interval_kind': 'parameter_uncertainty',
            'interval_confidence_level': confidence_level,
            'interval_status': np.full(mapped.shape, 'available', dtype=object),
            'interval_valid_samples': np.full(mapped.shape, n_samples, dtype=int),
            'interval_total_samples': n_samples,
            'interval_requested_samples': n_samples,
            'interval_random_seed': 12345,
        }, None

    result = predict_inverse_response(
        prediction,
        model_name='exponential',
        parameters={'a': 1.0, 'b': -0.1},
        covariance=np.eye(2) * 1e-6,
        residuals=np.full(6, 0.01),
        calibration_time_range=(0.0, 10.0),
        observed_responses=observed_response,
        interval_estimator=fixed_intervals,
    )

    assert result.available is True
    assert result.skip_reason is None
    assert result.note is None
    np.testing.assert_allclose(result.in_vivo_time, 2.0 * t_pred, atol=1e-7)
    np.testing.assert_allclose(result.time_lower, 2.0 * t_pred - 0.2)
    np.testing.assert_allclose(result.time_upper, 2.0 * t_pred + 0.2)
    np.testing.assert_allclose(result.time_error_lower, 0.2)
    np.testing.assert_allclose(result.time_error_upper, 0.2)
    assert result.status.tolist() == ['mapped', 'mapped', 'mapped']
    assert result.auxiliary['interval_total_samples'] == 1000
    np.testing.assert_array_equal(result.auxiliary['interval_valid_samples'], 1000)


def test_direct_scaling_methods_preserve_current_approach3_calculations():
    prediction = TimeCourse(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    value_result = predict_direct_value_scaling(
        prediction,
        time_grid=np.array([0.0, 1.0, 2.0]),
        in_vitro_response_on_time_grid=np.array([1.0, 2.0, 3.0]),
        in_vivo_response_on_time_grid=np.array([2.0, 4.0, 6.0]),
    )
    np.testing.assert_allclose(value_result.scaling_factor, 2.0)
    np.testing.assert_allclose(value_result.in_vivo_value, np.array([2.0, 4.0, 6.0]))

    time_result = predict_direct_time_scaling(
        prediction,
        response_grid=np.array([3.0, 2.0, 1.0]),
        in_vitro_time_at_response=np.array([2.0, 1.0, 0.0]),
        in_vivo_time_at_response=np.array([4.0, 2.0, 0.0]),
    )
    np.testing.assert_allclose(time_result.in_vivo_time, np.array([0.0, 2.0, 4.0]))
    assert np.isnan(time_result.scaling_factor[0])
    np.testing.assert_allclose(time_result.scaling_factor[1:], 2.0)


def test_process_predictions_delegates_approach1_to_reusable_core(monkeypatch):
    t1 = np.array([0.0, 1.0, 2.0])
    m1 = np.array([1.0, 0.7, 0.4])
    data = (
        t1,
        m1,
        2.0 * t1,
        m1.copy(),
        m1[::-1],
        t1,
        t1[::-1],
        2.0 * t1[::-1],
        m1,
        m1,
    )
    prediction_data = [t1, m1]
    result = {
        'approach1:linear': {
            'params': {'a': 1.0, 'b': 2.0},
            'pcov': np.eye(2),
            'residuals': np.zeros(3),
        }
    }

    original = engine.predict_time_mapping
    calls = []

    def recording_prediction(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, 'predict_time_mapping', recording_prediction)
    monkeypatch.setattr(
        engine,
        'create_prediction_plot_a1',
        lambda *args, **kwargs: {'plot': 'prediction', 'table_html': 'table'},
    )

    output = engine.process_predictions(
        data,
        prediction_data,
        result,
        ['approach1'],
    )

    assert len(calls) == 1
    assert calls[0]['model_name'] == 'linear'
    assert calls[0]['parameters'] == {'a': 1.0, 'b': 2.0}
    np.testing.assert_allclose(
        output['approach1']['linear']['predictions']['predicted_in_vivo_time'],
        np.array([1.0, 3.0, 5.0]),
    )


def test_process_predictions_delegates_all_approach2_numerical_methods(monkeypatch):
    t1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    m1 = np.exp(-0.2 * t1)
    t2 = 2.0 * t1
    m2 = np.exp(-0.1 * t2)
    response_grid = np.array(sorted(set(m1).union(set(m2)), reverse=True))
    time_grid = np.array(sorted(set(t1).union(set(t2))))
    data = (
        t1,
        m1,
        t2,
        m2,
        response_grid,
        time_grid,
        np.interp(response_grid, m1[::-1], t1[::-1], left=np.nan, right=np.nan),
        np.interp(response_grid, m2[::-1], t2[::-1], left=np.nan, right=np.nan),
        np.interp(time_grid, t1, m1, left=np.nan, right=np.nan),
        np.interp(time_grid, t2, m2, left=np.nan, right=np.nan),
    )
    prediction_data = [np.array([0.0, 2.5, 5.0]), np.exp(-0.2 * np.array([0.0, 2.5, 5.0]))]
    model_result = {
        'params': [
            {'a': 1.0, 'b': -0.2},
            {'a': 1.0, 'b': -0.1},
        ],
        'pcov': [np.eye(2) * 1e-6, np.eye(2) * 1e-6],
        'residuals': [np.zeros(6), np.full(6, 0.01)],
        'residual_dof': [4, 4],
        'tau': [SimpleNamespace(n=5.0, s=0.1), SimpleNamespace(n=10.0, s=0.2)],
    }

    calls = {'value': 0, 'tau': 0, 'inverse': 0}
    for name, key in (
        ('predict_value_ratio_with_uncertainty', 'value'),
        ('predict_tau_ratio', 'tau'),
        ('predict_inverse_response', 'inverse'),
    ):
        original = getattr(engine, name)

        def make_wrapper(function, call_key):
            def wrapper(*args, **kwargs):
                calls[call_key] += 1
                return function(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(engine, name, make_wrapper(original, key))

    def fixed_intervals(
        model_function,
        params,
        pcov,
        residuals,
        target_responses,
        time_range,
        observed_responses=None,
        confidence_level=0.95,
        n_samples=1000,
        grid_points=1001,
        **kwargs,
    ):
        mapped = 2.0 * prediction_data[0]
        return {
            'lower': mapped - 0.2,
            'upper': mapped + 0.2,
            'valid_counts': np.full(mapped.shape, n_samples, dtype=int),
            'n_samples': n_samples,
            'residual_std': 0.0,
            'interval_kind': 'parameter_uncertainty',
            'interval_confidence_level': confidence_level,
            'interval_status': np.full(mapped.shape, 'available', dtype=object),
            'interval_valid_samples': np.full(mapped.shape, n_samples, dtype=int),
            'interval_total_samples': n_samples,
            'interval_requested_samples': n_samples,
            'interval_random_seed': 12345,
        }, None

    monkeypatch.setattr(engine, '_inverse_response_prediction_intervals', fixed_intervals)
    monkeypatch.setattr(
        engine,
        'create_prediction_plot_a2',
        lambda *args, **kwargs: {
            'plot': kwargs.get('prediction_mode', 'value_ratio'),
            'table_html': 'table',
        },
    )

    output = engine.process_predictions(
        data,
        prediction_data,
        {'approach2:exponential': model_result},
        ['approach2'],
    )

    assert calls == {'value': 1, 'tau': 1, 'inverse': 1}
    assert 'error' not in output['approach2']['exponential']
