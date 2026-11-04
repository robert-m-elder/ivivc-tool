"""Prediction adapters and plot downloads preserve future-response results."""
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import engine
from ivivc_core import ObservationUncertainty, prepare_data
from models.exponential import model_function
from test_future_observation_uncertainty import _course, _predict


METHODS = [('ratio', 'value_ratio'), ('tau', 'tau_ratio'), ('inverse', 'observed_response_inverse')]


@pytest.mark.parametrize('method,mode', METHODS)
def test_future_records_keep_combined_and_propagated_bounds_separate(method, mode):
    result = _predict(method, _course(sigma=None),
        future_observation_uncertainty=ObservationUncertainty(.04, response_kind='mean'))
    kind = 'value' if method == 'ratio' else 'time'
    rows = engine._parameter_prediction_records(result, kind)
    np.testing.assert_array_equal([row['lower_95'] for row in rows], getattr(result, kind + '_lower'))
    np.testing.assert_array_equal([row['upper_95'] for row in rows], getattr(result, kind + '_upper'))
    for index, row in enumerate(rows):
        assert row['future_observation_variability_included'] is True
        assert row['future_standard_uncertainty'] == .04
        assert row['future_uncertainty_kind'] == 'sem' and row['future_response_kind'] == 'mean'
        assert row['value_interval_kind'] == 'future_observation_prediction'
        assert row['value_interval_valid_samples'] == 256
        previous = result.auxiliary['propagated_value_lower']
        assert row['propagated_response_lower_95'] == (None if previous is None else previous[index])
        if kind == 'time':
            assert row['response_lower_95'] == result.value_lower[index]
            assert row['response_upper_95'] == result.value_upper[index]
    json.dumps(rows, allow_nan=False)
    off = engine._parameter_prediction_records(_predict(method, _course(sigma=None)), kind)
    assert all('future_standard_uncertainty' not in row for row in off)


@pytest.mark.parametrize('method,mode', METHODS)
def test_future_plot_and_download_use_existing_bounds_without_resampling(monkeypatch, method, mode):
    result = _predict(method, _course(sigma=None),
        future_observation_uncertainty=ObservationUncertainty(.04, response_kind='individual'))
    import ivivc_core.uncertainty as uncertainty
    def no_sampling(*args, **kwargs):
        raise AssertionError('Presentation must not resample uncertainty.')
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', no_sampling)
    figures = []
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **kwargs: figures.append(fig) or '<plot>')
    t = np.linspace(0., 10., 6)
    y = np.exp(-.1 * t)
    data = (t, y, t, y, y, t, t, t, y, y)
    info = {'model_name': 'Exponential', 'model_function': model_function,
            'params': [{'a': 1., 'b': -.1}, {'a': 1., 'b': -.1}]}
    center = result.in_vivo_value if method == 'ratio' else result.in_vivo_time
    argument = {'value_prediction': result} if method == 'ratio' else {'time_prediction': result}
    plot = engine.create_prediction_plot_a2(data, result.in_vitro_time, result.in_vitro_value,
        center, t, y, y, info, prediction_mode=mode, **argument)
    trace = next(value for value in figures[0].data if value.name == 'In Vivo (Predicted)')
    np.testing.assert_allclose(np.asarray(trace.y) - trace.error_y.arrayminus, result.value_lower)
    np.testing.assert_allclose(np.asarray(trace.y) + trace.error_y.array, result.value_upper)
    table = pd.read_html(StringIO(plot['table_html']))[0]
    source = next(column for column in table if column.startswith('Future error source'))
    precision = next(column for column in table if column.startswith('Future SD/SEM'))
    interpretation = next(column for column in table if column.startswith('Response interval interpretation'))
    prediction_rows = table.iloc[:len(result.in_vivo_time)]
    assert prediction_rows[source].tolist() == ['supplied', 'supplied']
    np.testing.assert_array_equal(prediction_rows[precision], [.04, .04])
    assert prediction_rows[interpretation].tolist() == ['future_observation_prediction'] * 2
    assert plot['prediction_records'] == engine._parameter_prediction_records(
        result, 'value' if method == 'ratio' else 'time')


def test_prediction_adapter_reuses_calibration_precision_for_all_methods(monkeypatch):
    t = np.array([0., 10.])
    prepared = prepare_data(t, np.exp(-.2 * t), t, np.exp(-.1 * t),
        in_vivo_uncertainty=ObservationUncertainty([.02, .04], response_kind='mean', replicate_counts=[4, 9]))
    model = {'params': [{'a': 1., 'b': -.2}, {'a': 1., 'b': -.1}],
             'pcov': [np.zeros((2, 2)), np.zeros((2, 2))], 'residuals': [np.zeros(2)] * 2,
             'tau': [SimpleNamespace(n=5., s=0.), SimpleNamespace(n=10., s=0.)],
             'tau_diagnostics': [{}, {}], 'residual_dof': [0, 0],
             'observation_uncertainty': [None, prepared.in_vivo.observation_uncertainty.to_metadata()]}
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *args, **kwargs:
                        {'plot': '<plot>', 'table_html': ''})
    arguments = (prepared, _course(sigma=None), {'approach2:exponential': model}, ['approach2'])
    off = engine.process_predictions(*arguments)['approach2']['exponential']['prediction_tables']
    payload = engine.process_predictions(*arguments, future_observation_uncertainty='calibration')['approach2']['exponential']
    assert 'error' not in payload
    assert set(payload['prediction_tables']) == {mode for _, mode in METHODS}
    for mode, rows in payload['prediction_tables'].items():
        assert payload['uncertainty_metadata'][mode]['future_observation_variability_included'] is True
        for index, row in enumerate(rows):
            assert row['future_observation_uncertainty_source'] == 'calibration'
            assert row['future_uncertainty_kind'] == 'sem'
            predicted_time = row.get('predicted_in_vivo_time', row['in_vitro_time'])
            closest = int(np.argmin(abs(t - predicted_time)))
            assert row['future_calibration_time'] == t[closest]
            assert row['future_standard_uncertainty'] == [.02, .04][closest]
            for key in ('predicted_in_vivo_time', 'predicted_in_vivo_value'):
                if key in off[mode][index]:
                    assert row[key] == off[mode][index][key]
    json.dumps(payload['prediction_tables'], allow_nan=False)
