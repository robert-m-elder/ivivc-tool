"""The presentation facade retains and displays the core uncertainty model."""
from __future__ import annotations

from html.parser import HTMLParser

import numpy as np
import pytest
from scipy.stats import norm

import engine
from ivivc_core import ObservationUncertainty, PredictedTrajectory, prepare_time_course


SELECTION = 'approach2:exponential_normalized'


def _table_rows(html):
    class Rows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.row, self.cell = [], [], None

        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self.row = []
            elif tag in ('th', 'td'):
                self.cell = []

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

        def handle_endtag(self, tag):
            if tag in ('th', 'td') and self.cell is not None:
                self.row.append(''.join(self.cell).strip())
                self.cell = None
            elif tag == 'tr' and self.row:
                self.rows.append(self.row)

    parser = Rows()
    parser.feed(html)
    return parser.rows


def _prepared():
    time = np.arange(6.)
    response = np.exp(-0.2 * time) + np.array([0., 0.002, -0.001, 0.001, -0.002, 0.001])
    return engine.preprocess_data(
        time, response, 2. * time, response,
        in_vitro_uncertainty=ObservationUncertainty(0.04, response_kind='mean'),
        return_prepared=True,
    )


def _fit(monkeypatch, fast_analysis_config):
    monkeypatch.setattr(engine, 'create_plotly_a2', lambda *args, **kwargs: {'plot': '<fit>', 'table_html': ''})
    prepared = _prepared()
    results = engine.process_data(prepared, [SELECTION], ['approach2'], ['rmse', 'aic', 'aicc'],
                                  analysis_config=fast_analysis_config)
    assert 'error' not in results[SELECTION]
    return prepared, results


def test_typed_preprocessing_keeps_resolved_errors_and_fixed_normalization_metadata():
    prepared, metadata = engine.preprocess_data(
        [0., 1., np.nan, 3.], [10., 8., 5., 4.],
        [0., 2., 4., np.nan], [100., 80., 60., 30.],
        selected_normalizations=['initial_value'], return_metadata=True, return_prepared=True,
        in_vitro_uncertainty=ObservationUncertainty([0.5, 0.8, np.nan, 0.4], response_kind='mean'),
    )
    np.testing.assert_allclose(prepared.in_vitro.observation_uncertainty.sigma, [0.05, 0.08, 0.04])
    np.testing.assert_array_equal(prepared.preprocessing.sources['in_vitro'].retained_rows, [0, 1, 3])
    assert metadata['normalization_records'][0]['constants_treated_as_fixed'] is True
    assert metadata['observation_uncertainty']['in_vitro']['uncertainty_kind'] == 'sem'
    assert metadata['observation_uncertainty']['in_vivo'] is None
    display = engine.data_for_display(prepared)
    np.testing.assert_array_equal(display[0], prepared.in_vitro.time)
    display[0][0] = -99.
    assert prepared.in_vitro.time[0] == 0.
    with pytest.raises(ValueError, match='uncertainty'):
        prepared.to_legacy_data()


def test_paired_facade_passes_each_source_uncertainty_and_rejects_other_approaches(monkeypatch, fast_analysis_config):
    original = engine.fit_profile
    calls = []
    def fit(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, 'fit_profile', fit)
    monkeypatch.setattr(engine, 'create_plotly_a2', lambda *args, **kwargs: {'plot': '<fit>', 'table_html': ''})
    prepared = _prepared()
    results = engine.process_data(prepared, ['approach1:linear', SELECTION, 'approach3'],
                                  ['approach1', 'approach2', 'approach3'], ['rmse'],
                                  analysis_config=fast_analysis_config)
    assert len(calls) == 2
    assert calls[0]['observation_uncertainty'] is prepared.in_vitro.observation_uncertainty
    assert 'observation_uncertainty' not in calls[1]
    assert 'error' not in results[SELECTION]
    assert 'Approach 2' in results['approach1:linear']['error']
    assert 'Approach 2' in results['approach3']['error']
    assert results[SELECTION]['observation_uncertainty'][0]['uncertainty_kind'] == 'sem'
    assert results[SELECTION]['observation_uncertainty'][1] is None


def test_uncertain_prediction_facade_exports_both_axes_without_refitting(monkeypatch, fast_analysis_config):
    prepared, results = _fit(monkeypatch, fast_analysis_config)
    prediction, _ = prepare_time_course(
        [1., 2.], [0.8, 0.6],
        observation_uncertainty=ObservationUncertainty([0.01, 0.02], response_kind='mean', replicate_counts=[4, 16]),
    )
    monkeypatch.setattr(engine, 'create_prediction_plot_a2',
                        lambda *args, **kwargs: {'plot': '<prediction>', 'table_html': ''})
    def no_refitting(*args, **kwargs):
        raise AssertionError('Prediction must not repeat calibration fitting.')
    monkeypatch.setattr(engine, 'fit_profile', no_refitting)
    result = engine.process_predictions(prepared, prediction, results, ['approach2'])['approach2']['exponential_normalized']
    assert 'error' not in result
    for method in ('value_ratio', 'tau_ratio', 'observed_response_inverse'):
        metadata = result['uncertainty_metadata'][method]
        assert metadata['prediction_inputs_fixed'] is False
        assert metadata['future_observation_variability_included'] is False
        assert metadata['calibration_uncertainty']['in_vitro']['covariance_scaling'] == 'absolute'
        assert metadata['calibration_uncertainty']['in_vivo']['covariance_scaling'] == 'residual_scaled'
    for method in ('tau_ratio', 'observed_response_inverse'):
        records = result['prediction_tables'][method]
        for index, row in enumerate(records):
            sigma = [0.01, 0.02][index]
            assert row['input_standard_uncertainty'] == sigma
            assert row['input_uncertainty_kind'] == 'sem'
            assert row['input_replicate_count'] == [4, 16][index]
            assert row['value_interval_kind'] == 'prediction_input_uncertainty'
            np.testing.assert_allclose(row['response_lower_95'], prediction.value[index] - norm.ppf(0.975) * sigma)
            np.testing.assert_allclose(row['response_upper_95'], prediction.value[index] + norm.ppf(0.975) * sigma)
    assert result['prediction_tables']['value_ratio'][0]['interval_kind'] == 'parameter_and_input_uncertainty'
    assert result['prediction_tables']['tau_ratio'][0]['interval_kind'] == 'parameter_uncertainty'


def _trajectory():
    prediction, _ = prepare_time_course([1.], [0.5],
        observation_uncertainty=ObservationUncertainty(0.1, response_kind='mean'))
    return PredictedTrajectory(
        'approach2', 'observed_response_inverse', prediction.time, prediction.value,
        np.array([10.]), prediction.value, time_lower=np.array([8.]), time_upper=np.array([9.]),
        value_lower=np.array([0.3]), value_upper=np.array([0.7]),
        auxiliary={'prediction_input_uncertainty': prediction.observation_uncertainty.to_metadata(),
                   'interval_kind': 'parameter_and_input_uncertainty',
                   'value_interval_kind': 'prediction_input_uncertainty',
                   'value_interval_status': np.array(['available'])},
    )


def test_time_prediction_renderer_keeps_exact_primary_bounds_and_secondary_response_bars():
    trajectory = _trajectory()
    records = engine._parameter_prediction_records(trajectory, 'time')
    figure = engine.go.Figure()
    engine._add_parameter_prediction_trace(figure, trajectory.in_vitro_time, trajectory.in_vitro_value,
                                           trajectory.in_vivo_time, records, 'time')
    point, interval = figure.data
    assert point.error_x.visible is False
    np.testing.assert_allclose(point.error_y.array, [0.2])
    np.testing.assert_allclose(point.error_y.arrayminus, [0.2])
    assert list(interval.x) == [8., 9., None]
    assert interval.name == '95% propagated-uncertainty interval'
    assert records[0]['lower_95'] == 8.
    assert records[0]['upper_95'] == 9.
    figure.update_layout(xaxis_title='Time', yaxis_title='Value')
    table = engine.extract_plotly_data_for_table(figure)
    assert 'Input standard uncertainty' in table
    assert 'Response interval interpretation' in table
    headers, values = _table_rows(table)
    cells = dict(zip(headers, values))
    for axis, lower, upper in (('Time', 8., 9.), ('Value', 0.3, 0.7)):
        for bound, expected in (('lower', lower), ('upper', upper)):
            column = engine._plot_table_column_label(axis, 'In Vivo (Predicted)', bound)
            assert float(cells[column]) == expected


def test_noncontaining_response_intervals_are_segments_without_moving_their_bounds():
    trajectory = _trajectory()
    trajectory.value_lower = np.array([0.6])
    trajectory.value_upper = np.array([0.7])
    records = engine._parameter_prediction_records(trajectory, 'time')
    figure = engine.go.Figure()
    engine._add_parameter_prediction_trace(figure, trajectory.in_vitro_time, trajectory.in_vitro_value,
                                           trajectory.in_vivo_time, records, 'time')
    assert figure.data[0].error_y.visible is False
    response_interval = next(trace for trace in figure.data if trace.name == '95% input-response uncertainty interval')
    assert list(response_interval.x) == [10., 10., None]
    assert list(response_interval.y) == [0.6, 0.7, None]
    assert records[0]['response_lower_95'] == 0.6
    assert records[0]['response_upper_95'] == 0.7


def test_disabling_intervals_preserves_centers_and_hides_both_axes():
    trajectory = _trajectory()
    records = engine._parameter_prediction_records(trajectory, 'time')
    figure = engine.go.Figure()
    engine._add_parameter_prediction_trace(figure, trajectory.in_vitro_time, trajectory.in_vitro_value,
                                           trajectory.in_vivo_time, records, 'time', include_bands=False)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [10.]
    assert list(figure.data[0].y) == [0.5]
    assert figure.data[0].error_x.visible is False
    assert figure.data[0].error_y.visible is False
    assert records[0]['lower_95'] is None and records[0]['response_lower_95'] is None


def test_calibration_error_bars_retain_fitted_parameter_bands(monkeypatch):
    captured = {}
    def capture(figure, **kwargs):
        captured['figure'] = figure
        return '<fit>'
    monkeypatch.setattr(engine.pio, 'to_html', capture)
    time = np.arange(3.)
    values = np.exp(-0.2 * time)
    info = {'model_name': 'Exponential', 'model_function': lambda t, b: np.exp(b * np.asarray(t)),
            'params1': {'b': -0.2}, 'params2': {'b': -0.1},
            'pcov1': np.zeros((1, 1)), 'pcov2': np.zeros((1, 1)),
            'observation_uncertainty1': {'standard_uncertainty': [0.1, 0.1, 0.1], 'uncertainty_kind': 'sem'}}
    engine.create_plotly_a2(time, values, time, np.exp(-0.1 * time), info)
    figure = captured['figure']
    trace = next(trace for trace in figure.data if trace.name == 'In Vitro Data (± SEM)')
    np.testing.assert_array_equal(trace.error_y.array, [0.1] * 3)
    assert trace.meta['observation_uncertainty_kind'] == 'sem'
    assert any('Parameter-Uncertainty Band' in trace.name for trace in figure.data)
    assert figure.layout.yaxis.range[1] > 1.1
