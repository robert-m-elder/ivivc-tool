"""Exact numerical intervals are reused by either plot axis and by reports."""
from io import StringIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import engine
from ivivc_core import PredictedTrajectory
from ivivc_core.intervals import attach_parameter_interval, parameter_interval_summary
from utilities.report_content import build_report_sections


def time_trajectory(method):
    times = np.array([1., 2., 3.])
    response = np.array([.8, .5, .2])
    result = PredictedTrajectory(
        source_approach='approach1' if method == 'time_mapping' else 'approach2',
        source_method=method, model_name='exponential', in_vitro_time=times,
        in_vitro_value=response, in_vivo_time=2. * times, in_vivo_value=response,
    )
    interval = parameter_interval_summary(np.tile(2. * times, (10, 1)), 2. * times, n_samples=10)
    interval['lower'] = np.array([1.5, 5., np.nan])
    interval['upper'] = np.array([2.8, 5.7, np.nan])
    interval['interval_status'][2] = 'unavailable (test failure)'
    return attach_parameter_interval(result, interval, output_kind='time')


def render(monkeypatch, trajectory, include_bands=True):
    captured = []
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **k: captured.append(fig) or '<plot>')
    def forbidden(*a, **k):
        raise AssertionError('Rendering a supplied trajectory must not calculate intervals')
    monkeypatch.setattr(engine, 'curve_parameter_intervals', forbidden)
    t, y = trajectory.in_vitro_time, trajectory.in_vitro_value
    data = (t, y, t, y, y, t, t, t, y, y)
    if trajectory.source_approach == 'approach1':
        output = engine.create_prediction_plot_a1(data, t, y, trajectory.in_vivo_time,
                    {'model_name': 'Example'}, trajectory=trajectory, include_bands=include_bands)
    else:
        output = engine.create_prediction_plot_a2(data, t, y, trajectory.in_vivo_time, t, y, y,
                    {'model_name': 'Example'}, time_prediction=trajectory,
                    prediction_mode=trajectory.source_method, include_bands=include_bands)
    return output, captured[0]


@pytest.mark.parametrize('method', ['time_mapping', 'tau_ratio', 'observed_response_inverse'])
def test_time_plot_and_download_keep_noncentered_bounds_on_the_time_axis(monkeypatch, method):
    result = time_trajectory(method)
    output, figure = render(monkeypatch, result)
    points = next(t for t in figure.data if t.name == 'In Vivo (Predicted)')
    np.testing.assert_array_equal(points.x, result.in_vivo_time)
    assert points.error_x.arrayminus[0] == .5
    assert points.error_x.array[0] == pytest.approx(.8)
    assert np.isnan(points.error_x.array[1])
    interval = next(t for t in figure.data if t.name == '95% parameter-uncertainty interval')
    assert list(interval.x) == [5., 5.7, None]
    assert list(interval.y) == [.5, .5, None]
    assert output['prediction_records'] == engine._parameter_prediction_records(result, 'time')
    table = pd.read_html(StringIO(output['table_html']))[0]
    for bound in ('lower', 'upper'):
        column = engine._plot_table_column_label('Time', 'In Vivo (Predicted)', bound)
        np.testing.assert_allclose(table[column][:3], getattr(result, 'time_' + bound), equal_nan=True)
        assert engine._plot_table_column_label('Value', 'In Vivo (Predicted)', bound) not in table
    json.dumps(output['prediction_records'], allow_nan=False)


@pytest.mark.parametrize('method', ['time_mapping', 'tau_ratio', 'observed_response_inverse'])
def test_disabled_time_intervals_do_not_mutate_numerical_results(monkeypatch, method):
    result = time_trajectory(method)
    output, fig = render(monkeypatch, result, include_bands=False)
    assert all(row['lower_95'] is None for row in output['prediction_records'])
    assert not any(t.name == '95% parameter-uncertainty interval' for t in fig.data)
    assert result.time_lower[0] == 1.5


def test_small_time_bounds_keep_download_precision(monkeypatch):
    result = time_trajectory('time_mapping')
    result.time_lower[:2] = [1.25e-10, 3.5e-9]
    result.time_upper[:2] = [2.75e-10, 4.5e-9]
    output, _ = render(monkeypatch, result)
    table = pd.read_html(StringIO(output['table_html']))[0]
    for bound in ('lower', 'upper'):
        column = engine._plot_table_column_label('Time', 'In Vivo (Predicted)', bound)
        np.testing.assert_allclose(table[column][:2], getattr(result, 'time_' + bound)[:2], atol=0., rtol=1e-13)


@pytest.mark.parametrize('approach', ['approach1', 'approach2'])
def test_fit_bands_are_parameter_only_even_with_large_residuals(monkeypatch, approach):
    figures = []
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **k: figures.append(fig) or '<plot>')
    x = np.array([1., 2., 3.])
    def model(t, a, b):
        return a + b * np.asarray(t)
    params = {'a': 1e-9, 'b': 2e-9}
    y = model(x, **params)
    if approach == 'approach1':
        output = engine.create_plotly_a1(x, y, {'model_name': 'Example', 'model_function': model,
            'params': params, 'pcov': np.zeros((2, 2)), 'residuals': np.ones(3) * 1000})
    else:
        output = engine.create_plotly_a2(x, y, x, y, {'model_name': 'Example', 'model_function': model,
            'params1': params, 'params2': params, 'pcov1': np.zeros((2, 2)),
            'pcov2': np.zeros((2, 2)), 'residuals1': np.ones(3) * 1000,
            'residuals2': np.ones(3) * 2000})
    bands = [t for t in figures[0].data if 'Parameter-Uncertainty Band' in t.name]
    assert len(bands) == (2 if approach == 'approach1' else 4)
    for trace in bands:
        np.testing.assert_allclose(trace.y, model(np.asarray(trace.x), **params), atol=0., rtol=1e-12)
    assert not any('Prediction Band' in t.name for t in figures[0].data)
    table = pd.read_html(StringIO(output['table_html']))[0]
    column = engine._plot_table_column_label(figures[0].layout.yaxis.title.text, bands[0].name)
    np.testing.assert_allclose(table[column].dropna(), bands[0].y, atol=0., rtol=1e-12)


@pytest.mark.parametrize('approach,method', [
    ('approach1','direct_time_correlation'), ('approach2','tau_ratio'),
    ('approach2','observed_response_inverse'), ('approach2','value_ratio'),
])
def test_report_uses_one_interval_interpretation_and_keeps_numerical_bounds(approach, method):
    from test_report_content import fixture, table, text
    context, option = fixture(approach=approach, method=method)
    trajectory = time_trajectory('time_mapping')
    records = engine._parameter_prediction_records(trajectory, 'time')
    if method == 'value_ratio':
        for row in records:
            row['predicted_in_vivo_value'] = row.pop('predicted_in_vivo_time')
    context['prediction_source'] = {'filename': 'pred.csv'}
    context['prediction_results'] = {approach: {'example': {'prediction_tables': {method: records}}}}
    sections = build_report_sections(context, option)
    rows = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
    assert rows[1][3:6] == ['2', '1.5', '2.8']
    assert rows[2][3:6] == ['4', '5', '5.7']
    content = text(sections)
    assert 'not prediction intervals for future observations' in content
    assert '1,000 parameter draws and uncertainty seed 12345' in content
    assert 'surrogate for' not in content
    assert 'covariance and response perturbations' not in content
    json.dumps(sections, allow_nan=False)


def test_description_sources_and_template_no_longer_promise_observation_bands():
    from utilities.descriptions import APPROACH_DESCRIPTIONS
    for approach in ('approach1','approach2'):
        for key in ('fit_uncertainty_html','prediction_error_html'):
            value = APPROACH_DESCRIPTIONS[approach][key]
            assert 'No additional residual or observation noise is added' in value
            assert 'not observation prediction bands' in value
    template = Path('templates/results.html').read_text()
    assert "assumed response uncertainty equal to" not in template
    assert 'additional scatter of future observations' in template
