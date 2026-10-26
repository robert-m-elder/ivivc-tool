from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import engine
from engine import process_predictions


def _calibration_data():
    t1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    m1 = np.exp(-0.2 * t1)
    t2 = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    m2 = np.exp(-0.1 * t2)
    mm = np.array(sorted(set(m1).union(set(m2)), reverse=True))
    tt = np.array(sorted(set(t1).union(set(t2))))
    ti1 = np.interp(mm, m1[::-1], t1[::-1], left=np.nan, right=np.nan)
    ti2 = np.interp(mm, m2[::-1], t2[::-1], left=np.nan, right=np.nan)
    mi1 = np.interp(tt, t1, m1, left=np.nan, right=np.nan)
    mi2 = np.interp(tt, t2, m2, left=np.nan, right=np.nan)
    return t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2


def test_approach1_prediction_contract(monkeypatch):
    captured = {}

    def fake_plot(data, t_pred, m_pred, t_pred_vivo, model_info, include_bands=True, **kwargs):
        captured['t_pred_vivo'] = np.asarray(t_pred_vivo)
        return {'plot': 'a1-prediction', 'table_html': 'table'}

    monkeypatch.setattr(engine, 'create_prediction_plot_a1', fake_plot)
    data = _calibration_data()
    prediction_data = [np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.7, 0.4])]
    results = {
        'approach1:linear': {
            'params': {'a': 1.0, 'b': 2.0},
            'pcov': np.eye(2),
            'residuals': np.zeros(6),
        }
    }

    output = process_predictions(data, prediction_data, results, ['approach1'])
    prediction = output['approach1']['linear']

    np.testing.assert_allclose(captured['t_pred_vivo'], np.array([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(
        prediction['predictions']['predicted_in_vivo_time'],
        np.array([1.0, 3.0, 5.0]),
    )
    assert prediction['plot']['plot'] == 'a1-prediction'


def test_approach2_prediction_modes_and_tau_mutation(monkeypatch):
    plotted = {}
    interval_settings = {}

    def fake_plot(*args, **kwargs):
        mode = kwargs.get('prediction_mode', 'value_ratio')
        plotted[mode] = np.asarray(args[3], dtype=float)
        return {'plot': mode, 'table_html': 'table'}

    def fake_inverse_intervals(
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
        interval_settings['n_samples'] = n_samples
        mapped, _, _ = engine._inverse_response_times(
            model_function,
            params,
            target_responses,
            time_range,
            observed_responses=observed_responses,
        )
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

    monkeypatch.setattr(engine, 'create_prediction_plot_a2', fake_plot)
    monkeypatch.setattr(engine, '_inverse_response_prediction_intervals', fake_inverse_intervals)

    data = _calibration_data()
    t_pred = np.array([0.0, 2.5, 5.0])
    m_pred = np.exp(-0.2 * t_pred)
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
    results = {'approach2:exponential': model_result}

    output = process_predictions(data, [t_pred, m_pred], results, ['approach2'])
    prediction = output['approach2']['exponential']

    expected_value_ratio = m_pred * np.exp(0.1 * t_pred)
    np.testing.assert_allclose(plotted['value_ratio'], expected_value_ratio)
    np.testing.assert_allclose(plotted['tau_ratio'], 2.0 * t_pred)
    np.testing.assert_allclose(plotted['observed_response_inverse'], 2.0 * t_pred, atol=1e-7)
    assert model_result['prediction_skip_reasons']['tau_ratio'] is None
    assert prediction['plot_tau_skip_reason'] is None
    inverse = prediction['inverse_response_predictions']
    np.testing.assert_allclose(inverse['predicted_in_vivo_time'], 2.0 * t_pred, atol=1e-7)
    np.testing.assert_allclose(inverse['lower_95'], 2.0 * t_pred - 0.2, atol=1e-7)
    np.testing.assert_allclose(inverse['upper_95'], 2.0 * t_pred + 0.2, atol=1e-7)
    assert inverse['interval_total_samples'] == interval_settings['n_samples']
    assert inverse['status'].tolist() == ['mapped', 'mapped', 'mapped']


def test_undefined_tau_is_recorded_on_legacy_model_result(monkeypatch):
    monkeypatch.setattr(
        engine,
        'create_prediction_plot_a2',
        lambda *args, **kwargs: {'plot': kwargs.get('prediction_mode', 'value_ratio'), 'table_html': 'table'},
    )
    monkeypatch.setattr(
        engine,
        '_inverse_response_times',
        lambda *args, **kwargs: (
            np.array([np.nan]),
            np.array(['outside observed in vivo response range'], dtype=object),
            'inverse unavailable',
        ),
    )

    data = _calibration_data()
    model_result = {
        'params': [
            {'a': 1.0, 'b': -0.2},
            {'a': 1.0, 'b': -0.1},
        ],
        'pcov': [np.eye(2), np.eye(2)],
        'residuals': [np.zeros(6), np.zeros(6)],
        'residual_dof': [4, 4],
        'tau': [SimpleNamespace(n=0.0, s=1.0), SimpleNamespace(n=10.0, s=0.2)],
    }

    output = process_predictions(
        data,
        [np.array([1.0]), np.array([2.0])],
        {'approach2:exponential': model_result},
        ['approach2'],
    )

    reason = model_result['prediction_skip_reasons']['tau_ratio']
    assert reason == 'Time-constant rescaling was not performed because the in vitro time constant could not be calculated.'
    assert output['approach2']['exponential']['plot_tau'] is None
    assert output['approach2']['exponential']['plot_tau_skip_reason'] == reason


def test_approach3_direct_scaling_contract(monkeypatch):
    monkeypatch.setattr(
        engine,
        'create_prediction_plot_a3',
        lambda *args, **kwargs: {'plot': kwargs['ptype'], 'table_html': 'table'},
    )

    t1 = np.array([0.0, 1.0, 2.0])
    m1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([0.0, 2.0, 4.0])
    m2 = np.array([2.0, 4.0, 6.0])
    mm = np.array([3.0, 2.0, 1.0])
    tt = np.array([0.0, 1.0, 2.0])
    ti1 = np.array([2.0, 1.0, 0.0])
    ti2 = np.array([4.0, 2.0, 0.0])
    mi1 = np.array([1.0, 2.0, 3.0])
    mi2 = np.array([2.0, 4.0, 6.0])
    data = (t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2)
    prediction_data = [np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])]

    output = process_predictions(
        data,
        prediction_data,
        {'approach3': {'plot_v': {}, 'plot_t': {}}},
        ['approach3'],
        interpolated_data=data,
    )['approach3']

    value_result = output['method1_value_scaling']['predictions']
    np.testing.assert_allclose(value_result['scaling_factor'], 2.0)
    np.testing.assert_allclose(value_result['predicted_in_vivo_value'], np.array([2.0, 4.0, 6.0]))

    time_result = output['method2_time_scaling']['predictions']
    np.testing.assert_allclose(time_result['predicted_in_vivo_time'], np.array([0.0, 2.0, 4.0]))
    assert np.isnan(time_result['scaling_factor'][0])
    np.testing.assert_allclose(time_result['scaling_factor'][1:], 2.0)
