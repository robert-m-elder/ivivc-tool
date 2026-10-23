from __future__ import annotations

import numpy as np

from models import models
import engine
from engine import preprocess_data, process_data


REQUIRED_APPROACH1_KEYS = {
    'params',
    'uparams',
    'parameter_intervals',
    'pcov',
    'residuals',
    'residual_x',
    'residual_y',
    'cv_fit_params',
    'stats',
    'predictions',
    'evidence_data',
    'stats_table',
    'parameter_diagnostics_html',
    'plot',
}

REQUIRED_APPROACH2_KEYS = {
    'params',
    'uparams',
    'parameter_intervals',
    'tau',
    'tau_intervals',
    'stats',
    'predictions',
    'pcov',
    'residuals',
    'residual_x',
    'residual_y',
    'cv_fit_params',
    'residual_dof',
    'evidence_data',
    'stats_table',
    'parameter_diagnostics_html',
    'plot',
}


def test_approach1_fit_preserves_legacy_result_contract(
    paired_degradation_data,
    fast_analysis_config,
    stub_fit_plots,
):
    t1, m1, t2, m2 = paired_degradation_data
    data = preprocess_data(t1, m1, t2, m2, selected_interpolation=['none'])

    results = process_data(
        data,
        selected_models=['approach1:linear_zero_intercept'],
        selected_approaches=['approach1'],
        selected_metrics=['r_squared', 'rmse'],
        analysis_config=fast_analysis_config,
    )
    result = results['approach1:linear_zero_intercept']

    assert 'error' not in result
    assert REQUIRED_APPROACH1_KEYS <= set(result)
    assert list(result['params']) == ['a']
    assert np.isclose(result['params']['a'], 2.0, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result['predictions'], result['residual_y'], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result['residuals'], 0.0, atol=1e-7)
    assert result['pcov'].shape == (1, 1)
    assert result['stats'].columns.tolist() == [
        'r_squared',
        'rmse',
        'CV r_squared',
        'CV rmse',
    ]
    assert set(result['evidence_data']['fit']) == {'n', 'rss', 'n_params'}
    assert result['evidence_data']['fit']['n'] == 5
    assert result['evidence_data']['fit']['n_params'] == 1
    assert result['evidence_data']['fit']['rss'] < 1e-12
    assert result['cv_fit_params'] == []
    assert result['plot'] == stub_fit_plots['approach1']
    assert isinstance(result['stats_table'], str)
    assert isinstance(result['parameter_diagnostics_html'], str)


def test_approach2_fit_preserves_two_profile_contract(
    monkeypatch,
    fast_analysis_config,
    stub_fit_plots,
):
    t1 = np.linspace(0.0, 5.0, 8)
    m1 = np.exp(-0.20 * t1)
    t2 = np.linspace(0.0, 10.0, 8)
    m2 = np.exp(-0.10 * t2)
    data = preprocess_data(t1, m1, t2, m2, selected_interpolation=['default'])

    def exponential_initial_guess(x, y):
        rate = np.log(y[-1] / y[0]) / (x[-1] - x[0])
        return np.array([y[0], rate])

    monkeypatch.setitem(models['exponential'], 'initial_guess', exponential_initial_guess)

    results = process_data(
        data,
        selected_models=['approach2:exponential'],
        selected_approaches=['approach2'],
        selected_metrics=['r_squared', 'rmse'],
        analysis_config=fast_analysis_config,
    )
    result = results['approach2:exponential']

    assert 'error' not in result
    assert REQUIRED_APPROACH2_KEYS <= set(result)
    assert len(result['params']) == 2
    assert list(result['params'][0]) == ['a', 'b']
    assert list(result['params'][1]) == ['a', 'b']
    assert np.isclose(result['params'][0]['a'], 1.0, rtol=1e-4)
    assert np.isclose(result['params'][0]['b'], -0.20, rtol=1e-4)
    assert np.isclose(result['params'][1]['a'], 1.0, rtol=1e-4)
    assert np.isclose(result['params'][1]['b'], -0.10, rtol=1e-4)
    assert len(result['predictions']) == 2
    assert len(result['residuals']) == 2
    np.testing.assert_allclose(result['residuals'][0], 0.0, atol=1e-7)
    np.testing.assert_allclose(result['residuals'][1], 0.0, atol=1e-7)
    assert [matrix.shape for matrix in result['pcov']] == [(2, 2), (2, 2)]
    assert result['residual_dof'] == [6, 6]
    assert len(result['tau']) == 2
    assert all(np.isfinite(value.n) and value.n > 0 for value in result['tau'])
    assert all(np.isfinite(value.s) and value.s >= 0 for value in result['tau'])
    assert len(result['tau_intervals']) == 2
    assert result['evidence_data']['in_vitro']['n'] == 8
    assert result['evidence_data']['in_vivo']['n'] == 8
    assert result['plot'] == stub_fit_plots['approach2']


def test_approach3_result_remains_plot_only(
    paired_degradation_data,
    fast_analysis_config,
    stub_fit_plots,
):
    t1, m1, t2, m2 = paired_degradation_data
    data = preprocess_data(t1, m1, t2, m2, selected_interpolation=['default'])

    result = process_data(
        data,
        selected_models=['approach3'],
        selected_approaches=['approach3'],
        selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
    )['approach3']

    assert result == {
        'plot_v': stub_fit_plots['approach3_value'],
        'plot_t': stub_fit_plots['approach3_time'],
    }


def test_cv_training_parameter_sets_are_retained_when_requested(
    monkeypatch,
    paired_degradation_data,
    fast_analysis_config,
    stub_fit_plots,
):
    t1, m1, t2, m2 = paired_degradation_data
    data = preprocess_data(t1, m1, t2, m2, selected_interpolation=['none'])
    config = dict(fast_analysis_config)
    config.update({
        'cv_scheme': 'leave_one_timepoint_out',
        'include_cv_fit_variability_plot': True,
    })
    sentinel = {'plot': 'cv-fit-variability', 'table_html': 'table'}
    monkeypatch.setattr(engine, 'create_cv_fit_variability_plotly', lambda *args, **kwargs: sentinel)

    result = process_data(
        data,
        selected_models=['approach1:linear_zero_intercept'],
        selected_approaches=['approach1'],
        selected_metrics=['rmse'],
        analysis_config=config,
    )['approach1:linear_zero_intercept']

    assert len(result['cv_fit_params']) == 5
    assert all(params is not None and np.asarray(params).shape == (1,) for params in result['cv_fit_params'])
    assert result['cv_fit_variability_plot'] == sentinel


def test_process_data_delegates_profile_fits_to_reusable_core(
    monkeypatch,
    fast_analysis_config,
    stub_fit_plots,
):
    t1 = np.linspace(0.0, 5.0, 8)
    m1 = np.exp(-0.20 * t1)
    t2 = np.linspace(0.0, 10.0, 8)
    m2 = np.exp(-0.10 * t2)
    data = preprocess_data(t1, m1, t2, m2, selected_interpolation=['none'])

    def exponential_initial_guess(x, y):
        rate = np.log(y[-1] / y[0]) / (x[-1] - x[0])
        return np.array([y[0], rate])

    monkeypatch.setitem(models['exponential'], 'initial_guess', exponential_initial_guess)

    original_fit_profile = engine.fit_profile
    calls = []

    def recording_fit_profile(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_fit_profile(*args, **kwargs)

    monkeypatch.setattr(engine, 'fit_profile', recording_fit_profile)
    result = process_data(
        data,
        selected_models=['approach2:exponential'],
        selected_approaches=['approach2'],
        selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
    )['approach2:exponential']

    assert 'error' not in result
    assert len(calls) == 2
    assert all(call['model_name'] == 'exponential' for call in calls)
    assert all(call['approach_id'] == 'approach2' for call in calls)
    assert all(call['calculate_tau'] is True for call in calls)
