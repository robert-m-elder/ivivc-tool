"""Tables and optional offline HTML for the calibration-sampling POC."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


INTERPRETATION = (
    'Candidates are specified examples, not optimized schedules. Historical observations stay fixed; '
    'only proposed calibration measurements are regenerated. Widths and RMSE are conditional on '
    'available calculations, with counts supplied. Penalized mean curve-recovery IRMSE also includes '
    'failed fits using the failure penalty recorded in metadata. Pooled pointwise coverage includes every experiment '
    'with known truth: unavailable intervals count as not covering. Conditional coverage considers '
    'available intervals only. These are pointwise summaries, not simultaneous whole-curve coverage. '
    'Generating trajectories and noise models are assumptions; this is not evidence of transferability '
    'or a universal criterion for a sufficiently informative IVIVC. Future calibration measurement '
    'noise is simulated separately from reference uncertainty. Additional scatter of future downstream '
    'in vivo observations is excluded from the prediction intervals.'
)


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def _point(value, index, default=None):
    if value is None:
        return default
    if isinstance(value, (str, dict)) or np.ndim(value) == 0:
        return value
    return value[index] if index < len(value) else default


def _course_rows(course, **context):
    if course is None:
        return []
    uncertainty = course.observation_uncertainty
    return [dict(context, row=index, time=float(time), response=float(value),
                 standard_uncertainty=None if uncertainty is None else float(uncertainty.sigma[index]),
                 uncertainty_kind=None if uncertainty is None else uncertainty.uncertainty_kind,
                 response_kind=None if uncertainty is None else uncertainty.response_kind,
                 replicate_count=None if uncertainty is None else _point(uncertainty.replicate_counts, index),
                 precision_source=None if uncertainty is None else uncertainty.source)
            for index, (time, value) in enumerate(zip(course.time, course.value))]


def calibration_poc_tables(scenario):
    """Export backend results without recomputing scores or dropping failures."""
    designs, pointwise, outcomes, measurements, evaluations = [], [], [], [], []
    for label, result in scenario['calibration_results'].items():
        summary = scenario['calibration_summaries'][label]
        plan = scenario['designs'][label]
        row = dict(design=label, display_name=plan.get('display_name', label.replace('_', ' ')),
                   n_simulations=result.n_simulations,
                   calibration_fit_failure_rate=summary['fit_failure_rate'])
        for side in ('in_vitro', 'in_vivo'):
            row[side + '_times'] = json.dumps(_json_safe(plan[side + '_times']))
            row[side + '_replicates_per_time'] = plan[side + '_replicates_per_time']
            row[side + '_added_measurements'] = plan[side + '_added_measurements']
        row['total_added_measurements'] = row['in_vitro_added_measurements'] + row['in_vivo_added_measurements']
        recovery = scenario.get('recovery_summaries', {}).get(label)
        if recovery is not None:
            row.update(curve_mean_irmse=recovery.mean_irmse,
                       curve_penalized_mean_irmse=recovery.penalized_mean_irmse,
                       curve_p90_irmse=recovery.p90_irmse, curve_fit_failure_rate=recovery.failure_rate)
        for output in ('relationship', 'prediction'):
            item = summary.get(output)
            if item is None:
                continue
            row[output + '_quantity'] = item['quantity']
            for key in ('mean_interval_width', 'rmse', 'complete_point_rate', 'complete_interval_rate',
                        'coverage', 'conditional_coverage'):
                row[output + '_' + key] = item[key]
            row[output + '_truth_points'] = int(np.sum(item['truth_counts']))
            row[output + '_interval_truth_points'] = int(np.sum(item['interval_truth_counts']))
            for column, coordinate in enumerate(item['coordinate']):
                values = dict(design=label, output=output, quantity=item['quantity'],
                              coordinate=coordinate, n_simulations=result.n_simulations,
                              coverage_scope=item['coverage_scope'])
                for key in ('point_counts', 'interval_counts', 'truth_counts', 'interval_truth_counts',
                            'error_counts', 'width_counts', 'point_availability', 'interval_availability',
                            'pointwise_mean_interval_width', 'pointwise_rmse', 'pointwise_coverage',
                            'pointwise_conditional_coverage'):
                    values[key] = _point(item.get(key), column)
                pointwise.append(values)
        designs.append(row)
        for experiment, assessment in enumerate(result.assessments):
            failure = {stage: dict(message=error.message, exception_type=error.exception_type)
                       for stage, error in assessment.errors.items()}
            trial = dict(design=label, experiment=experiment, calibration_fit_available=assessment.fit_available,
                         calculation_errors=json.dumps(_json_safe(failure), sort_keys=True))
            curve = scenario.get('recovery_results', {}).get(label)
            if curve is not None:
                trial.update(curve_fit_available=bool(curve.fit_success[experiment]),
                             curve_fit_message=curve.fit_messages[experiment])
                for key in ('integrated_rmse', 'normalized_integrated_rmse', 'maximum_absolute_error',
                            'mean_curve_interval_width', 'penalized_integrated_rmse'):
                    values = getattr(curve, key, None)
                    if values is not None:
                        trial['curve_' + key] = values[experiment]
            for side in ('in_vitro', 'in_vivo'):
                measurements.extend(_course_rows(getattr(result, side + '_data')[experiment],
                                                design=label, experiment=experiment, environment=side))
            for output in ('relationship', 'prediction'):
                item = summary.get(output)
                if item is None:
                    continue
                evaluation = getattr(assessment, output)
                trial[output + '_all_points_available'] = bool(
                    evaluation is not None and len(evaluation.estimate) and np.all(evaluation.point_available))
                trial[output + '_all_intervals_available'] = bool(
                    evaluation is not None and len(evaluation.estimate) and np.all(evaluation.interval_available))
                truth = result.truth.get(output)
                for column, coordinate in enumerate(item['coordinate']):
                    record = dict(design=label, experiment=experiment, output=output,
                                  quantity=item['quantity'], coordinate=coordinate,
                                  estimate=np.nan, lower=np.nan, upper=np.nan,
                                  truth=np.nan if truth is None else truth[experiment, column],
                                  point_available=False, interval_available=False,
                                  calculation_status='unavailable (no evaluation)',
                                  interval_status='unavailable (no evaluation)',
                                  interval_conditional=False, interval_valid_samples=None,
                                  interval_requested_samples=result.metadata.get('propagation_samples'),
                                  extrapolated=None, extrapolation_status=None, extrapolation_reason=None,
                                  interval_extrapolated=None, interval_extrapolation_status=None)
                    if evaluation is not None:
                        aux = evaluation.trajectory.auxiliary
                        axis = 'value' if item['quantity'] in ('value_ratio', 'in_vivo_response') else 'time'
                        status = _point(aux.get(axis + '_interval_status', aux.get('interval_status')), column)
                        record.update(estimate=evaluation.estimate[column], lower=evaluation.lower[column],
                                      upper=evaluation.upper[column],
                                      point_available=bool(evaluation.point_available[column]),
                                      interval_available=bool(evaluation.interval_available[column]),
                                      calculation_status=_point(evaluation.trajectory.status, column,
                                                               'available' if evaluation.point_available[column] else 'unavailable'),
                                      interval_status=status or ('available' if evaluation.interval_available[column] else 'unavailable'),
                                      interval_conditional=bool(_point(aux.get(axis + '_interval_conditional',
                                                                           aux.get('interval_conditional')), column, False)),
                                      interval_valid_samples=_point(aux.get('interval_valid_samples'), column),
                                      interval_requested_samples=aux.get('interval_requested_samples',
                                                                        result.metadata.get('propagation_samples')))
                        for field in ('extrapolated', 'extrapolation_status', 'extrapolation_reason',
                                      'interval_extrapolated', 'interval_extrapolation_status'):
                            record[field] = _point(aux.get(field), column)
                        trial[output + '_extrapolation_warnings'] = json.dumps(
                            _json_safe(aux.get('extrapolation_warnings', [])))
                    evaluations.append(record)
            outcomes.append(trial)
    fixed = []
    for environment, course in scenario.get('existing_data', {}).items():
        fixed.extend(_course_rows(course, role='fixed_calibration', environment=environment))
    fixed.extend(_course_rows(scenario.get('target'), role='target_prediction', environment='in_vitro'))
    references = []
    for environment, source in scenario.get('references', {}).items():
        for index, (values, weight) in enumerate(zip(source.value_draws, source.draw_weights)):
            references.extend(dict(environment=environment, draw=index, weight=weight,
                                   valid=bool(source.valid_draws[index]), time=time, true_response=value)
                              for time, value in zip(source.time_grid, values))
    return {
        'design_summary.csv': pd.DataFrame(designs),
        'pointwise_performance.csv': pd.DataFrame(pointwise),
        'experiment_outcomes.csv': pd.DataFrame(outcomes),
        'fixed_inputs.csv': pd.DataFrame(fixed),
        'simulated_calibration_data.csv': pd.DataFrame(measurements),
        'evaluations.csv': pd.DataFrame(evaluations),
        'references.csv': pd.DataFrame(references),
    }


def _plot_html(table, title):
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return '<p>Interactive plots require Plotly. All numerical results are available in the tables and downloads.</p>', False
    first = ('curve_penalized_mean_irmse', 'Curve recovery: penalized mean IRMSE') if 'curve_penalized_mean_irmse' in table else (
        'total_added_measurements', 'Additional independent measurements')
    metrics = [first, ('relationship_mean_interval_width', 'Relationship: mean interval width'),
               ('prediction_mean_interval_width', 'Target prediction: mean interval width')]
    figure = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in metrics])
    for column, (key, label) in enumerate(metrics, 1):
        values = [_json_safe(value) for value in table.get(key, pd.Series(np.nan, index=table.index))]
        figure.add_trace(go.Bar(x=[escape(str(item)) for item in table['display_name']], y=values, name=label), row=1, col=column)
    figure.update_layout(title=escape(title), showlegend=False, height=480, margin=dict(b=150))
    return to_html(figure, full_html=False, include_plotlyjs=True), True


def _document(title, body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><style>'
            'body{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#18222b}'
            'table{border-collapse:collapse;font-size:.85rem}th,td{padding:.5rem;border:1px solid #ccc;text-align:left}'
            'th{background:#eef3f6}pre{white-space:pre-wrap}.table-wrap{overflow:auto}a{color:#075b9a}'
            '</style></head><body>' + body + '</body></html>')


def _comparison_tables(table):
    """Small reader-facing views; CSV retains every machine-readable column."""
    labels = {
        'display_name': 'Candidate', 'n_simulations': 'Requested experiments',
        'in_vitro_times': 'Additional in vitro times', 'in_vitro_replicates_per_time': 'In vitro n/time',
        'in_vitro_added_measurements': 'Additional in vitro measurements',
        'in_vivo_times': 'Additional in vivo times', 'in_vivo_replicates_per_time': 'In vivo n/time',
        'in_vivo_added_measurements': 'Additional in vivo measurements',
        'total_added_measurements': 'Total additional measurements',
        'curve_mean_irmse': 'Mean IRMSE: successful fits', 'curve_penalized_mean_irmse': 'Penalized mean IRMSE',
        'curve_p90_irmse': '90th percentile IRMSE', 'curve_fit_failure_rate': 'Curve fit failure fraction',
        'calibration_fit_failure_rate': 'IVIVC fit failure fraction',
    }
    groups = [('Proposed sampling', ['display_name', 'in_vitro_times', 'in_vitro_replicates_per_time',
                'in_vitro_added_measurements', 'in_vivo_times', 'in_vivo_replicates_per_time',
                'in_vivo_added_measurements', 'total_added_measurements'])]
    if 'curve_mean_irmse' in table:
        groups.append(('Curve recovery', ['display_name', 'n_simulations', 'curve_mean_irmse',
            'curve_penalized_mean_irmse', 'curve_p90_irmse', 'curve_fit_failure_rate']))
    for output, title in (('relationship', 'IVIVC relationship'), ('prediction', 'Target predictions')):
        if output + '_mean_interval_width' not in table:
            continue
        metrics = ['mean_interval_width', 'rmse', 'complete_interval_rate', 'coverage',
                   'conditional_coverage', 'truth_points', 'interval_truth_points']
        names = ['Mean pointwise interval width', 'RMSE: available estimates',
                 'All requested intervals available: fraction of experiments', 'Pooled pointwise coverage',
                 'Coverage among available intervals', 'Truth-known evaluations', 'Available intervals with truth']
        labels.update({output + '_' + key: name for key, name in zip(metrics, names)})
        quantity = table[output + '_quantity'].iloc[0]
        title = {'value_ratio': 'In vivo/in vitro response ratio (dimensionless)',
                 'time_constant_ratio': 'In vivo/in vitro time-constant ratio (dimensionless)',
                 'in_vivo_response': 'Target in vivo response', 'in_vivo_time': 'In vivo time'}.get(quantity, title)
        groups.append((title, ['display_name', 'calibration_fit_failure_rate'] + [output + '_' + key for key in metrics]))
    pieces = []
    for title, columns in groups:
        display = table[[key for key in columns if key in table]].rename(columns=labels)
        display = display.replace([np.inf, -np.inf], np.nan)
        pieces.append(f'<h3>{title}</h3><div class="table-wrap">' + display.to_html(
            index=False, escape=True, na_rep='Unavailable', float_format=lambda x: f'{x:.5g}') + '</div>')
    return ''.join(pieces)


def write_calibration_poc_outputs(run, output_directory, *, include_html=True):
    """Write CSV, strict JSON, and optional self-contained HTML summaries."""
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'settings': _json_safe(run['settings']), 'scenarios': list(run['scenarios']),
                'interpretation': INTERPRETATION, 'files': {}, 'html_requested': bool(include_html)}
    links = []
    for name, scenario in run['scenarios'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Scenario identifiers must use letters, digits, underscores, or hyphens.')
        directory = root / name
        directory.mkdir(exist_ok=True)
        tables = calibration_poc_tables(scenario)
        for filename, table in tables.items():
            table.replace([np.inf, -np.inf], np.nan).to_csv(directory / filename, index=False)
        metadata = {
            'name': name, 'title': scenario['title'], 'description': scenario['description'],
            'assumptions': scenario['assumptions'], 'designs': scenario['designs'],
            'calibration_summaries': scenario['calibration_summaries'],
            'recovery_summaries': scenario.get('recovery_summaries', {}),
            'calibration_metadata': {label: result.metadata for label, result in scenario['calibration_results'].items()},
            'calculation_failures': {label: [dict(experiment=index,
                errors={stage: dict(message=error.message, exception_type=error.exception_type)
                        for stage, error in assessment.errors.items()})
                for index, assessment in enumerate(result.assessments) if assessment.errors]
                for label, result in scenario['calibration_results'].items()},
            'fixed_inputs': {key: _course_rows(course) for key, course in scenario.get('existing_data', {}).items()},
            'target_inputs': _course_rows(scenario.get('target')),
            'reference_metadata': {key: dict(metadata=source.metadata, draw_weights=source.draw_weights,
                                            supported_time_range=source.supported_time_range,
                                            future_measurement_noise_included=False)
                                   for key, source in scenario.get('references', {}).items()},
            'interpretation': INTERPRETATION,
        }
        (directory / 'metadata.json').write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False), encoding='utf-8')
        filenames = list(tables) + ['metadata.json']
        if include_html:
            chart, available = _plot_html(tables['design_summary.csv'], scenario['title'])
            download_links = ' | '.join(f'<a href="{escape(filename, quote=True)}">{escape(filename)}</a>' for filename in filenames)
            first_result = next(iter(scenario['calibration_results'].values()))
            confidence = first_result.metadata.get('confidence_level', .95)
            propagation = first_result.metadata.get('propagation_samples', run['settings'].get('n_interval_samples'))
            range_text = ''
            curve_range = scenario['assumptions'].get('curve_recovery_evaluation_range')
            if curve_range is not None:
                range_text = f'<p>Curve recovery is assessed over times {escape(str(curve_range[0]))} to {escape(str(curve_range[1]))}.</p>'
            ivivc_times = scenario['assumptions'].get('ivivc_evaluation_times')
            if ivivc_times is not None:
                range_text += f'<p>Relationship and target predictions are assessed at times {escape(str(ivivc_times))}.</p>'
            if run['settings'].get('response_scale'):
                range_text += f'<p>Response scale: {escape(str(run["settings"]["response_scale"]))}; '
                range_text += f'time units: {escape(str(run["settings"].get("time_units", "input time units")))}.</p>'
            body = (f'<p><a href="../index.html">All demonstrations</a></p><h1>{escape(scenario["title"])}</h1>'
                    f'<p>{escape(scenario["description"])}</p><h2>Assumptions</h2>'
                    f'<p>{first_result.n_simulations} simulated experiments per candidate; '
                    f'{escape(str(propagation))} uncertainty-propagation draws per experiment.</p>{range_text}'
                    f'<pre>{escape(json.dumps(_json_safe(scenario["assumptions"]), indent=2))}</pre>'
                    f'<p>{escape(INTERPRETATION)}</p>'
                    '<p>Calibration-range extrapolation flags and reasons are retained in evaluations.csv. '
                    'A finite interval does not establish that extrapolation is reliable.</p>'
                    f'{chart}<h2>Candidate comparison</h2>'
                    f'<p>Widths summarize approximate {100*confidence:g}% pointwise intervals among available calculations. '
                    'Sampling times and counts below describe additional measurements only; '
                    'held-fixed historical observations are recorded in fixed_inputs.csv.</p>'
                    '<p>Failure, availability, and coverage are proportions from 0 to 1; '
                    'errors and widths retain their output units. Missing values mean unavailable, not zero. '
                    'In vitro and in vivo measurements need not have equal cost.</p>'
                    f'{_comparison_tables(tables["design_summary.csv"])}'
                    '<h2>Downloads</h2><p>simulated_calibration_data.csv contains the assembled datasets actually fitted, '
                    'including historical rows repeated unchanged across experiments.</p>'
                    f'<p>{download_links}</p>')
            (directory / 'summary.html').write_text(_document(scenario['title'], body), encoding='utf-8')
            filenames.append('summary.html')
            manifest.setdefault('plots_available', {})[name] = available
            links.append(f'<li><a href="{name}/summary.html">{escape(scenario["title"])}</a></li>')
        else:
            (directory / 'summary.html').unlink(missing_ok=True)
        manifest['files'][name] = [f'{name}/{filename}' for filename in filenames]
    (root / 'manifest.json').write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False), encoding='utf-8')
    if include_html:
        (root / 'index.html').write_text(_document('Calibration sampling demonstrations',
            '<h1>Calibration sampling demonstrations</h1>'
            '<p>Compare specified in vivo schedules with optional IVIVC assessment, or additions to existing calibration datasets.</p>'
            f'<p>{escape(INTERPRETATION)}</p><ul>{"".join(links)}</ul>'
            '<p><a href="manifest.json">Run settings and file manifest</a></p>'), encoding='utf-8')
    else:
        (root / 'index.html').unlink(missing_ok=True)
    return root
