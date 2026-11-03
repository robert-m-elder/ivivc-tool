"""CLI uncertainty options reach typed fitting/prediction and CSV records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli


def frame():
    time = np.arange(6.)
    response = np.exp(-0.2 * time)
    return pd.DataFrame({
        't1': time, 'y1': 10. * response, 't2': 2. * time, 'y2': 20. * response,
        'vitro_sem': [1.] * 6, 'vivo_sem': [2.] * 6, 'n': [4] * 6,
        'notes': ['Entered error bars'] * 6,
    })


def options(**changes):
    values = {'in_vitro_uncertainty_mode': 'column', 'in_vitro_uncertainty_column': 'vitro_sem',
              'in_vitro_response_kind': 'mean', 'in_vitro_replicate_column': 'n',
              'in_vivo_uncertainty_mode': 'constant', 'in_vivo_uncertainty_value': 2.,
              'in_vivo_response_kind': 'mean', 'uncertainty_independence_confirmed': True}
    values.update(changes)
    return values


def run(frame_data, config, **changes):
    arguments = dict(approach='approach2', model='exponential_normalized',
                     metrics=['rmse', 'aic', 'aicc'], interpolation='none', scalings=[],
                     normalizations=['initial_value'], analysis_config=config)
    arguments.update(changes)
    return cli.run_one_df(Path('calibration.csv'), None, None, frame_data, **arguments)


def test_cli_parser_distinguishes_sd_sem_and_optional_prediction_columns():
    parsed = cli.build_parser().parse_args([
        'calibration.csv', '--approach', 'approach2', '--model', 'exponential_normalized',
        '--metric', 'rmse', '--in-vitro-uncertainty-mode', 'constant',
        '--in-vitro-uncertainty-value', '0.1', '--in-vitro-response-kind', 'individual',
        '--prediction', 'new.csv', '--prediction-uncertainty-mode', 'column',
        '--prediction-uncertainty-column', 'SEM', '--prediction-response-kind', 'mean',
        '--prediction-replicate-column', 'n', '--independent-errors',
    ])
    assert parsed.in_vitro_uncertainty_value == 0.1
    assert parsed.in_vitro_response_kind == 'individual'
    assert parsed.prediction_response_kind == 'mean'
    assert parsed.prediction_replicate_column == 'n'
    assert parsed.uncertainty_independence_confirmed is True


def test_cli_weighted_rows_record_analyzed_sem_without_dividing_by_counts(fast_analysis_config, stub_fit_plots):
    data = frame()
    data.loc[5, ['t1', 'y1', 't2', 'y2', 'vitro_sem', 'n']] = np.nan
    rows = run(data, fast_analysis_config, uncertainty_options=options())
    assert len(rows) == 2
    for row in rows:
        assert 'error' not in row
        assert row['response_kind'] == 'mean' and row['uncertainty_kind'] == 'sem'
        np.testing.assert_allclose(json.loads(row['standard_uncertainty']), [0.1] * 5)
        assert row['covariance_scaling'] == 'absolute'
        assert row['normalization_constants_fixed'] is True
        assert np.isfinite(row['aicc'])
    assert json.loads(rows[0]['replicate_counts']) == [4] * 5
    assert json.loads(rows[0]['entered_standard_uncertainty'])[-1] is None
    assert rows[0]['uncertainty_column'] == 'vitro_sem'
    assert rows[1]['uncertainty_mode'] == 'constant'


def test_cli_unspecified_uncertainty_preserves_the_fit_row_schema(fast_analysis_config, stub_fit_plots):
    rows = run(frame(), fast_analysis_config)
    assert len(rows) == 2
    assert set(rows[0]) == {'file', 'sheet_index', 'sheet_name', 'model_key', 'dataset',
                            'rmse', 'aic', 'aicc', 'CV rmse', 'CV aic', 'CV aicc',
                            'tau_nominal', 'tau_sd'}


def test_cli_rejects_uncertainty_for_approach1_before_running_preprocessing(monkeypatch, fast_analysis_config):
    def unexpected(*args, **kwargs):
        raise AssertionError('Unsupported uncertainty must be rejected before preprocessing.')
    monkeypatch.setattr(cli, 'preprocess_data', unexpected)
    with pytest.raises(ValueError, match='Approach 2'):
        run(frame(), fast_analysis_config, approach='approach1', model='linear', uncertainty_options=options())


def test_cli_prediction_rows_export_input_errors_and_both_axes(monkeypatch, fast_analysis_config, stub_fit_plots):
    import engine
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *args, **kwargs: {'plot': '<plot>', 'table_html': ''})
    data = frame()
    data['y1'] /= 10.
    data['y2'] /= 20.
    prediction = pd.DataFrame({'time': [1., 2.], 'value': [0.8, 0.6], 'SEM': [0.02, 0.03]})
    prediction_rows = []
    rows = run(data, fast_analysis_config, normalizations=[],
               uncertainty_options=options(in_vitro_uncertainty_value=0.02,
                    in_vitro_uncertainty_mode='constant', in_vivo_uncertainty_value=0.02,
                    prediction_uncertainty_mode='column', prediction_uncertainty_column='SEM',
                    prediction_response_kind='mean'),
               prediction_frame=prediction, prediction_file='new.csv', prediction_rows=prediction_rows)
    assert len(rows) == 2 and all('error' not in row for row in rows)
    assert {row['method'] for row in prediction_rows} == {'value_ratio', 'tau_ratio', 'observed_response_inverse'}
    assert len(prediction_rows) == 6
    for row in prediction_rows:
        assert row['input_uncertainty_kind'] == 'sem'
        assert row['prediction_uncertainty_column'] == 'SEM'
        assert row['prediction_inputs_fixed'] is False
        assert row['future_observation_variability_included'] is False
        if row['method'] in ('tau_ratio', 'observed_response_inverse'):
            assert np.isfinite(row['response_lower_95']) and np.isfinite(row['response_upper_95'])
            assert row['value_interval_kind'] == 'prediction_input_uncertainty'


def test_cli_prediction_failure_preserves_completed_fit_summaries(fast_analysis_config, stub_fit_plots):
    prediction_rows = []
    rows = run(frame(), fast_analysis_config, prediction_frame=pd.DataFrame({'t': [1.], 'y': [0.8]}),
               prediction_rows=prediction_rows,
               uncertainty_options=options(prediction_uncertainty_mode='constant',
                    prediction_uncertainty_value=0., prediction_response_kind='mean'))
    assert len(rows) == 2 and all('error' not in row for row in rows)
    assert len(prediction_rows) == 1
    assert 'positive' in prediction_rows[0]['error']


def test_cli_json_columns_preserve_missing_values_as_null():
    value = json.loads(cli._json_values({'sigma': np.array([0.1, np.nan]), 'count': np.int64(4)}))
    assert value == {'sigma': [0.1, None], 'count': 4}
