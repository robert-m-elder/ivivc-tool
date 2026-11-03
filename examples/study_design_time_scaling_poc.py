"""Empirical ratio input and reporting for the study-design POC."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from study_design_core import EmpiricalTimeScaleDistribution, time_scaling_weight_summary


def load_time_scale_distribution(path=None):
    """Read ratio records without dropping malformed rows or fitting a distribution.

    CSV columns are ``ratio`` and optional ``weight``/``source_id``. Other
    columns, including ``group``, are retained as record provenance. Missing
    weight columns mean equal weights; missing cells in a supplied column do
    not. A missing file or invalid table never falls back to the illustration.
    """
    if path is None:
        table = pd.DataFrame({
            'ratio': [1.6, 1.8, 2.0, 2.2, 2.4],
            'weight': np.ones(5),
            'source_id': [f'illustrative-{letter}' for letter in 'ABCDE'],
            'group': ['illustrative'] * 5,
        })
        source = 'illustrative_synthetic'
        source_path = None
    else:
        source_path = str(Path(path).resolve())
        try:
            with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream, strict=True)
                header = [column.strip() for column in next(reader)]
                rows = list(reader)
            if not header or any(not column for column in header) or len(header) != len(set(header)):
                raise ValueError('CSV column names must be nonempty and unique.')
            for index, row in enumerate(rows, start=2):
                if len(row) != len(header):
                    raise ValueError(f'Time-scale CSV record {index} must contain {len(header)} fields.')
            table = pd.DataFrame(rows, columns=header)
        except (OSError, UnicodeError, StopIteration, csv.Error) as exception:
            raise ValueError(f'Could not read the time-scale ratio CSV {source_path!r}: {exception}') from exception
        if 'ratio' not in table:
            raise ValueError('The time-scale CSV must contain a ratio column.')
        try:
            table['ratio'] = pd.to_numeric(table['ratio'], errors='raise').astype(float)
            if 'weight' in table:
                table['weight'] = pd.to_numeric(table['weight'], errors='raise').astype(float)
            else:
                table['weight'] = 1.0
        except (ValueError, TypeError) as exception:
            raise ValueError('Every supplied ratio and weight must be numeric; blank cells are invalid.') from exception
        if 'source_id' not in table:
            table['source_id'] = [f'record-{index + 1}' for index in range(len(table))]
        source = 'csv'
    extras = [column for column in table if column not in ('ratio', 'weight', 'source_id')]
    distribution = EmpiricalTimeScaleDistribution(
        table['ratio'].to_numpy(dtype=float),
        weights=table['weight'].to_numpy(dtype=float),
        source_ids=table['source_id'].tolist(),
        metadata={
            'input_source': source,
            'input_path': source_path,
            'record_metadata': table[extras].to_dict(orient='records'),
            'ratio_definition': 'tau_in_vivo / tau_in_vitro',
        },
    )
    return distribution, table.copy()


def _is_time_scaling(definition):
    return definition.reference.metadata.get('reference_source') == 'empirical_time_scaling'


def time_scaling_tables(definition):
    """Return empirical-specific output tables, retaining input record order."""
    if not _is_time_scaling(definition):
        return {}
    summary = time_scaling_weight_summary(definition.reference)
    inputs = pd.DataFrame(definition.metadata['time_scale_input_records']).copy()
    weights = pd.DataFrame({
        'record_index': np.arange(len(summary['ratios'])),
        'ratio': summary['ratios'],
        'input_weight': inputs['weight'].to_numpy(dtype=float),
        'prior_weight': summary['prior_weights'],
        'current_weight': summary['current_weights'],
        'source_id': inputs['source_id'].to_numpy(),
    })
    for column in inputs:
        if column in ('ratio', 'weight', 'source_id'):
            continue
        destination = column
        while destination in weights:
            destination = 'input_' + destination
        weights[destination] = inputs[column].to_numpy()
    prior = EmpiricalTimeScaleDistribution(summary['ratios'], weights=summary['prior_weights'])
    current = EmpiricalTimeScaleDistribution(summary['ratios'], weights=summary['current_weights'])
    active = (summary['prior_unique_weights'] > 0) | (summary['current_unique_weights'] > 0)
    support = summary['unique_ratios'][active]
    tables = {
        'time_scale_input.csv': inputs,
        'time_scale_weights.csv': weights,
        'time_scale_ecdf.csv': pd.DataFrame({
            'ratio': support,
            'prior_cdf': prior.cdf(support),
            'current_cdf': current.cdf(support),
        }),
    }
    agreement = definition.metadata.get('time_scaling_observation_agreement')
    if agreement is not None:
        tables['observation_agreement.csv'] = pd.DataFrame([{
            key: value for key, value in agreement.items() if np.ndim(value) == 0
        }])
    return tables


def _step_coordinates(distribution):
    support, cumulative = distribution.ecdf()
    padding = max(float(support[-1] - support[0]), float(support[-1])) * .05
    x = [max(0., float(support[0] - padding))]
    y = [0.]
    previous = 0.
    for ratio, probability in zip(support, cumulative):
        x.extend([float(ratio), float(ratio)])
        y.extend([previous, float(probability)])
        previous = float(probability)
    last = float(support[-1])
    x.append(last + min(padding, np.finfo(float).max - last))
    y.append(1.)
    return x, y


def _html_table(table):
    return '<div style="overflow-x:auto">' + table.to_html(
        index=False, escape=True, na_rep='—', float_format=lambda value: f'{value:.6g}',
    ) + '</div>'


def time_scaling_summary_html(definition, go, to_html, reference_table_fn):
    """Show empirical inputs and prior/current reference uncertainty on POC pages."""
    if not _is_time_scaling(definition):
        return ''
    reference = definition.reference
    prior_reference = getattr(definition, 'prior_reference', None)
    if prior_reference is None:
        prior_reference = reference
    prior_summary = time_scaling_weight_summary(prior_reference)
    current_summary = time_scaling_weight_summary(reference)
    provenance = reference.metadata.get('time_scale_metadata', {})
    source = provenance.get('input_source', 'unspecified')
    source_text = (
        'Illustrative synthetic ratio records; these are not calibrated estimates for an implant or polymer class.'
        if source == 'illustrative_synthetic' else
        'User-supplied ratio CSV: ' + str(provenance.get('input_path', 'unspecified'))
    )
    ecdf = go.Figure()
    for label, summary, color, dash in (
        ('Prior', prior_summary, '#4477AA', 'dash'),
        ('Current planning reference', current_summary, '#228833', 'solid'),
    ):
        distribution = EmpiricalTimeScaleDistribution(summary['ratios'], weights=summary['current_weights'])
        x, y = _step_coordinates(distribution)
        ecdf.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label,
                                 line={'color': color, 'dash': dash}))
    ecdf.update_layout(title='Weighted empirical time-scale distribution',
                       xaxis_title='Time ratio r = tau_IVV / tau_IV',
                       yaxis_title='Cumulative weight', template='plotly_white')
    curves = go.Figure()
    for label, candidate, color, dash in (
        ('Prior', prior_reference, '68,119,170', 'dash'),
        ('Current', reference, '34,136,51', 'solid'),
    ):
        table = reference_table_fn(candidate)
        curves.add_trace(go.Scatter(x=table['time'], y=table['lower_95'], mode='lines',
                                   line={'width': 0}, showlegend=False))
        curves.add_trace(go.Scatter(x=table['time'], y=table['upper_95'], mode='lines',
                                   line={'width': 0}, fill='tonexty',
                                   fillcolor=f'rgba({color},0.12)', name=f'{label} 95% reference envelope'))
        curves.add_trace(go.Scatter(x=table['time'], y=table['central_response'], mode='lines',
                                   line={'color': f'rgb({color})', 'dash': dash}, name=f'{label} central reference'))
    if definition.existing_observations is not None:
        observations = definition.existing_observations
        curves.add_trace(go.Scatter(x=observations.time, y=observations.response,
                                   mode='markers', name='Completed target-study observations'))
    curves.update_layout(title='Prior and current in vivo reference uncertainty',
                         xaxis_title='Time', yaxis_title='Response', template='plotly_white')
    effective_sizes = pd.DataFrame([
        {
            'Reference': label,
            'Trajectory-draw ESS': summary['draw_effective_sample_size'],
            'Ratio-record ESS': summary['record_effective_sample_size'],
            'Unique-scale ESS': summary['unique_scale_effective_sample_size'],
        }
        for label, summary in (('Prior', prior_summary), ('Current', current_summary))
    ])
    tables = time_scaling_tables(definition)
    model = reference.metadata.get('source_model_name') or 'external time course'
    parameters = reference.metadata.get('source_parameters') or {}
    parameter_text = '; '.join(f'{name}={value:.6g}' for name, value in parameters.items()) or 'not applicable'
    source_range = reference.metadata['source_time_range']
    lower, upper = reference.supported_time_range
    extrapolated = np.asarray(reference.metadata['extrapolated_points_by_ratio'], dtype=bool)
    adaptive = bool(reference.metadata.get('adaptive_reweighting'))
    state_text = (
        'Completed target-study observations reweight the original prior once; the curves and available ratio values are retained.'
        if adaptive else
        'Prospective planning uses the supplied prior; prior and current distributions and reference envelopes coincide.'
    )
    agreement_html = ''
    if 'observation_agreement.csv' in tables:
        agreement_html = (
            '<h3>Agreement with completed observations</h3>'
            '<p>Standardized residual RMS uses individual replicate observation SD. '
            'A concentrated posterior can still fit poorly; these diagnostics do not impose an adequacy threshold.</p>'
            + _html_table(tables['observation_agreement.csv'])
        )
    return (
        '<section><h2>Empirical time-scale reference</h2>'
        f'<p>{escape(source_text)}</p>'
        '<p>Each complete curve uses one r = tau_IVV / tau_IV, with f_IVV(t) = f_IV(t / r). '
        'The weighted ECDF is used directly, without fitting a parametric scale distribution.</p>'
        f'<p>{escape(state_text)}</p>'
        f'<p><strong>In vitro source model:</strong> <code>{escape(str(model))}</code>; '
        f'<strong>parameters:</strong> {escape(parameter_text)}; '
        f'<strong>joint parameter draws:</strong> {int(reference.metadata["n_parameter_draws"])} '
        '(zero means a fixed fitted curve). '
        f'<strong>Source time range:</strong> {source_range[0]:g} to {source_range[1]:g}; '
        f'<strong>in vivo reference range:</strong> {lower:g} to {upper:g}. '
        f'<strong>Scaled ratio/time evaluations outside source coverage:</strong> '
        f'{int(np.count_nonzero(extrapolated))} of {extrapolated.size}.</p>'
        + to_html(ecdf, full_html=False, include_plotlyjs=False)
        + to_html(curves, full_html=False, include_plotlyjs=False)
        + '<h3>Input ratio records and planning weights</h3>'
        + _html_table(tables['time_scale_weights.csv'])
        + '<h3>Effective sample sizes</h3><p>Draw, record, and unique-scale ESS are separate measures of weight concentration. '
        'Parameter-draw expansion adds no historical studies; none of these ESS values is a study count.</p>'
        + _html_table(effective_sizes) + agreement_html + '</section>'
    )
