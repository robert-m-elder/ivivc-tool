"""One numerical response-ratio interval is reused by plots, tables and reports."""
from __future__ import annotations

from io import StringIO
import json

import numpy as np
import pandas as pd
import pytest

import engine
from ivivc_core.predictions import predict_value_ratio_with_uncertainty
from ivivc_core.types import TimeCourse


def trajectory():
    result = predict_value_ratio_with_uncertainty(
        TimeCourse(np.array([1., 2., 3.]), np.array([.2, .1, .05])),
        model_name='exponential',
        in_vitro_parameters={'a': 1., 'b': 0.},
        in_vivo_parameters={'a': 4., 'b': 0.},
        in_vitro_covariance=np.zeros((2, 2)),
        in_vivo_covariance=np.zeros((2, 2)),
        n_samples=12, random_state=9,
    )
    # Includes an asymmetric interval, one excluding the central estimate,
    # and an explicitly unavailable interval.
    result.value_lower = np.array([.7, .5, np.nan])
    result.value_upper = np.array([1.1, .6, np.nan])
    result.auxiliary['interval_status'] = np.array(
        ['available', 'available', 'unavailable (denominator interval includes zero)'], dtype=object,
    )
    result.note = 'One interval is unavailable.'
    return result


def plot(monkeypatch, result, **changes):
    captured = []
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **kw: captured.append(fig) or '<plot>')
    t, y = result.in_vitro_time, result.in_vitro_value
    data = (t, y, t, y, y, t, t, t, y, y)
    kwargs = {'value_prediction': result}
    kwargs.update(changes)
    output = engine.create_prediction_plot_a2(
        data, t, y, result.in_vivo_value, t, y, y, {'model_name': 'Example'}, **kwargs,
    )
    return output, captured[0]


def test_plot_uses_actual_bounds_and_keeps_point_outside_percentile_interval(monkeypatch):
    result = trajectory()
    output, figure = plot(monkeypatch, result)
    points = next(trace for trace in figure.data if trace.name == 'In Vivo (Predicted)')
    np.testing.assert_allclose(points.y, [.8, .4, .2])
    assert points.error_y.arrayminus[0] == pytest.approx(.1)
    assert points.error_y.array[0] == pytest.approx(.3)
    assert np.isnan(points.error_y.array[1])
    segments = next(trace for trace in figure.data if trace.name == '95% parameter-uncertainty interval')
    assert list(segments.x) == [2., 2., None]
    assert list(segments.y) == [.5, .6, None]
    assert output['prediction_records'] == engine._value_ratio_prediction_records(result)
    rows = output['prediction_records']
    assert rows[1]['predicted_in_vivo_value'] == .4
    assert (rows[1]['lower_95'], rows[1]['upper_95']) == (.5, .6)
    assert rows[2]['lower_95'] is None
    assert 'denominator interval includes zero' in rows[2]['interval_status']
    json.dumps(rows, allow_nan=False)


def test_download_uses_numerical_bounds_even_when_interval_excludes_center(monkeypatch):
    output, _ = plot(monkeypatch, trajectory())
    table = pd.read_html(StringIO(output['table_html']))[0]
    for bound, expected in [('lower', [.7, .5, np.nan]), ('upper', [1.1, .6, np.nan])]:
        column = engine._plot_table_column_label('Value', 'In Vivo (Predicted)', bound)
        np.testing.assert_allclose(table[column][:3], expected, equal_nan=True)
    assert 'denominator interval includes zero' in table['Interval status (In Vivo (Predicted))'][2]
    assert not any('parameter-uncertainty interval' in str(column) for column in table.columns)


def test_plot_does_not_sample_and_disabled_bounds_remain_absent(monkeypatch):
    result = trajectory()

    def forbidden(*args, **kwargs):
        raise AssertionError('Plotting must not sample parameters')

    monkeypatch.setattr(engine, 'predict_value_ratio_with_uncertainty', forbidden)
    output, figure = plot(monkeypatch, result, include_bands=False)
    assert all(row['lower_95'] is None for row in output['prediction_records'])
    assert not any(trace.name == '95% parameter-uncertainty interval' for trace in figure.data)
    assert not np.isnan(result.value_lower[0])  # Display controls did not mutate the core result.


def test_old_plot_call_without_numeric_intervals_never_fabricates_them(monkeypatch):
    output, _ = plot(monkeypatch, trajectory(), value_prediction=None)
    assert all(row['lower_95'] is None for row in output['prediction_records'])


def test_process_predictions_reports_numeric_bounds_even_with_stub_plot_helpers(monkeypatch):
    result = trajectory()
    calls = []

    def calculated(*args, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(engine, 'predict_value_ratio_with_uncertainty', calculated)
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *a, **k: {'plot': 'plot', 'table_html': ''})
    # Keep this integration check independent of tau's uncertainty dependency.
    t = np.array([0., 1., 2., 3.])
    vitro, vivo = np.exp(-.2 * t), np.exp(-.1 * t)
    data = (t, vitro, t, vivo, vitro, t, t, t, vitro, vivo)
    fitted = {'params': [{'a': 1., 'b': -.2}, {'a': 1., 'b': -.1}],
              'pcov': [np.zeros((2, 2)), np.zeros((2, 2))],
              'residuals': [np.zeros(4), np.zeros(4)], 'residual_dof': [2, 2], 'tau': None}
    output = engine.process_predictions(
        data, [result.in_vitro_time, result.in_vitro_value],
        {'approach2:exponential': fitted}, ['approach2'],
    )['approach2']['exponential']
    assert 'error' not in output
    assert len(calls) == 1
    assert calls[0]['random_state'] == 12345
    assert output['prediction_tables']['value_ratio'] == engine._value_ratio_prediction_records(result)
    assert output['plot_value_ratio_note'] == result.note
    assert 'observed_response_inverse' in output['prediction_tables']


def test_small_response_bounds_retain_precision_in_download(monkeypatch):
    result = trajectory()
    result.value_lower[:2] = [1.25e-10, 3.5e-9]
    result.value_upper[:2] = [2.75e-10, 4.5e-9]
    output, _ = plot(monkeypatch, result)
    table = pd.read_html(StringIO(output['table_html']))[0]
    for bound in ('lower', 'upper'):
        column = engine._plot_table_column_label('Value', 'In Vivo (Predicted)', bound)
        np.testing.assert_allclose(table[column][:2], getattr(result, f'value_{bound}')[:2], rtol=1e-14, atol=0.)
