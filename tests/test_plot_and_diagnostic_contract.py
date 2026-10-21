from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine import (
    create_initial_plotly,
    create_residual_plotly,
    create_residual_qq_plotly,
    extract_plotly_data_for_table,
)


def _read_single_html_table(html):
    return pd.read_html(StringIO(html))[0]


def test_plot_table_includes_uncertainty_bounds_and_applies_accessible_marker_style():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1.0, 2.0],
        y=[10.0, 20.0],
        mode='markers',
        name='In Vivo Predicted',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[2.0, 3.0],
            arrayminus=[1.0, 1.5],
            visible=True,
        ),
    ))
    fig.update_layout(xaxis_title='Time', yaxis_title='Value')

    table_html = extract_plotly_data_for_table(fig)
    table = _read_single_html_table(table_html)

    assert fig.data[0].marker.symbol == 'triangle-up'
    assert fig.data[0].marker.line.color == '#1f1f1f'
    lower_columns = [column for column in table.columns if 'lower bound' in column]
    upper_columns = [column for column in table.columns if 'upper bound' in column]
    assert len(lower_columns) == 1
    assert len(upper_columns) == 1
    np.testing.assert_allclose(table[lower_columns[0]], np.array([9.0, 18.5]))
    np.testing.assert_allclose(table[upper_columns[0]], np.array([12.0, 23.0]))


def test_initial_plot_contract_contains_plot_and_downloadable_raw_series():
    result = create_initial_plotly(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.5]),
        np.array([0.0, 2.0]),
        np.array([1.0, 0.5]),
    )

    assert set(result) == {'plot', 'table_html'}
    assert 'plotly-graph-div' in result['plot']
    table = _read_single_html_table(result['table_html'])
    assert table.shape == (2, 4)
    assert any('in vitro' in column.lower() for column in table.columns)
    assert any('in vivo' in column.lower() for column in table.columns)


def test_raw_residual_plot_reports_independent_scale_metadata():
    result = create_residual_plotly(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.01, -0.02, 0.015]),
        np.array([1.0, 0.8, 0.6]),
        'Residual characterization',
    )

    assert set(result) == {'plot', 'table_html', 'scale_info'}
    assert result['scale_info']['axis_limit'] > 0
    assert result['scale_info']['max_abs_residual'] == 0.02
    assert 'plotly-graph-div' in result['plot']


def test_qq_plot_is_suppressed_when_residual_variation_is_negligible():
    result = create_residual_qq_plotly(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.8, 0.6, 0.4]),
        'Q-Q characterization',
    )

    assert result['available'] is False
    assert result['plot'] is None
    assert 'too small' in result['message']
