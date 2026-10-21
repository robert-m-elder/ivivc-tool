from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import cli


def test_parse_sheet_spec_preserves_python_slice_and_deduplication_semantics():
    sheet_names = ['A', 'B', 'C', 'D', 'E']

    assert cli.parse_sheet_spec('0,2:-1,-1,0', sheet_names) == [
        (0, 'A'),
        (2, 'C'),
        (3, 'D'),
        (4, 'E'),
    ]


def test_cli_run_one_df_returns_current_numeric_summary_schema(
    paired_degradation_data,
    fast_analysis_config,
    stub_fit_plots,
):
    t1, m1, t2, m2 = paired_degradation_data
    frame = pd.DataFrame({
        'time1': t1,
        'value1': m1,
        'time2': t2,
        'value2': m2,
    })

    rows = cli.run_one_df(
        Path('synthetic.csv'),
        None,
        None,
        frame,
        approach='approach1',
        model='linear_zero_intercept',
        metrics=['r_squared', 'rmse'],
        interpolation='none',
        scalings=[],
        normalizations=[],
        analysis_config=fast_analysis_config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row['file'] == 'synthetic.csv'
    assert row['sheet_index'] == ''
    assert row['sheet_name'] == ''
    assert row['model_key'] == 'approach1:linear_zero_intercept'
    assert np.isclose(row['r_squared'], 1.0)
    assert row['rmse'] < 1e-7
    assert np.isnan(row['CV r_squared'])
    assert np.isnan(row['CV rmse'])
