"""Render shared calibration-comparison outcomes without recalculating scores."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .design_planning_results import _number, _percent, _plot, _reference_band
from .design_calibration_recovery import build_calibration_recovery


SIDES = ('in_vitro', 'in_vivo')
SIDE_LABELS = {'in_vitro': 'In vitro', 'in_vivo': 'In vivo'}


def _at(values, index, default=np.nan):
    if values is None:
        return default
    if np.ndim(values) == 0:
        return values
    return values[index] if index < len(values) else default


def _count(values):
    return int(np.sum(values)) if values is not None else 0


def _percentage(value):
    return _percent(value) if value is not None and np.isfinite(value) else '—'


def _quantity_labels(quantity, output, method):
    if quantity == 'time_constant_ratio':
        return 'In vivo / in vitro time-constant ratio (dimensionless)', 'Constant ratio'
    if quantity == 'value_ratio':
        return 'In vivo / in vitro response ratio (dimensionless)', 'In vitro time'
    if quantity == 'in_vivo_response':
        return 'Predicted in vivo response (analyzed scale)', 'Target in vitro time'
    coordinate = ('In vivo response level (analyzed scale)' if method == 'observed_response_inverse'
                  else 'Target in vitro time' if output == 'prediction' else 'In vitro time')
    return 'Predicted in vivo time', coordinate


def _observation_description(observation):
    if observation is None:
        return 'No new measurements'
    if observation.kind == 'constant_sd':
        return 'Individual SD = ' + _number(observation.value)
    if observation.kind == 'constant_cv':
        return ('Individual CV = ' + _number(observation.value) +
                '; minimum SD = ' + _number(observation.minimum_sd))
    return (f'Time-dependent individual SD: {len(observation.profile_times)} profile times, '
            f'{_number(observation.profile_times[0])}–{_number(observation.profile_times[-1])}; '
            f'SD {_number(np.min(observation.profile_sd))}–{_number(np.max(observation.profile_sd))}; '
            'nearest-time lookup with endpoint extension')


def _diagnostics(simulated, output, quantity):
    """Retain stage/point diagnostics, including conditional inverse intervals."""
    statuses, conditional, valid_samples, fractions = Counter(), 0, [], set()
    axis = 'value' if quantity in ('value_ratio', 'in_vivo_response') else 'time'
    for assessment in simulated.assessments:
        evaluation = getattr(assessment, output)
        if evaluation is None:
            continue
        trajectory = evaluation.trajectory
        if trajectory.status is not None:
            statuses.update(str(value) for value in np.atleast_1d(trajectory.status))
        auxiliary = trajectory.auxiliary
        conditional += int(np.count_nonzero(auxiliary.get(
            axis + '_interval_conditional', auxiliary.get('interval_conditional', False))))
        counts = auxiliary.get(axis + '_interval_valid_samples', auxiliary.get('interval_valid_samples'))
        if counts is not None:
            valid_samples.extend(float(value) for value in np.atleast_1d(counts)
                                 if np.isfinite(value))
        fraction = auxiliary.get(axis + '_interval_min_valid_fraction', auxiliary.get('interval_min_valid_fraction'))
        if fraction is not None and np.isfinite(fraction):
            fractions.add(float(fraction))
    return {
        'statuses': [{'label': label, 'count': count} for label, count in statuses.items()],
        'conditional_count': conditional,
        'valid_samples': ('—' if not valid_samples else
                          f'{_number(min(valid_samples))}–{_number(max(valid_samples))}'),
        'min_valid_fraction': ', '.join(_percentage(value) for value in sorted(fractions)),
    }


def _output_result(label, output, item, simulated, method):
    quantity_label, coordinate_label = _quantity_labels(item['quantity'], output, method)
    empirical = 'empirical_quantiles' in item
    points = []
    for index, coordinate in enumerate(item['coordinate']):
        record = {
            'coordinate': 'Constant' if item['quantity'] == 'time_constant_ratio' else _number(coordinate),
            'point_count': int(_at(item.get('point_counts'), index, 0)),
            'interval_count': int(_at(item.get('interval_counts'), index, 0)),
            'truth_count': int(_at(item.get('truth_counts'), index, 0)),
            'error_count': int(_at(item.get('error_counts'), index, 0)),
            'width_count': int(_at(item.get('width_counts'), index, 0)),
            'interval_truth_count': int(_at(item.get('interval_truth_counts'), index, 0)),
            'width': _number(_at(item.get('pointwise_mean_interval_width'), index)),
            'rmse': _number(_at(item.get('pointwise_rmse'), index)),
            'coverage': _percentage(_at(item.get('pointwise_coverage'), index)),
            'conditional_coverage': _percentage(_at(item.get('pointwise_conditional_coverage'), index)),
        }
        if empirical:
            for kind in ('estimate', 'error'):
                quantiles = np.asarray(item['pointwise_' + kind + '_quantiles'])[:, index]
                record[kind + '_range'] = f'{_number(quantiles[0])}–{_number(quantiles[-1])}'
                record[kind + '_median'] = _number(quantiles[1])
        points.append(record)
    example = None
    for experiment, assessment in enumerate(simulated.assessments):
        evaluation = getattr(assessment, output)
        if evaluation is None or not np.any(evaluation.point_available):
            continue
        truth = simulated.truth.get(output)
        auxiliary = evaluation.trajectory.auxiliary
        axis = 'value' if item['quantity'] in ('value_ratio', 'in_vivo_response') else 'time'
        interval_status = auxiliary.get(axis + '_interval_status', auxiliary.get('interval_status'))
        example = {'experiment': experiment + 1, 'points': [
            {'coordinate': 'Constant' if item['quantity'] == 'time_constant_ratio' else _number(coordinate),
             'estimate': _number(evaluation.estimate[index]),
             'lower': _number(evaluation.lower[index]), 'upper': _number(evaluation.upper[index]),
             'truth': _number(None if truth is None else truth[experiment, index]),
             'point_available': bool(evaluation.point_available[index]),
             'interval_available': bool(evaluation.interval_available[index]),
             'status': _at(evaluation.trajectory.status, index, 'Unavailable'),
             'interval_status': _at(interval_status, index,
                                    'Available' if evaluation.interval_available[index] else 'Unavailable'),
             'conditional': bool(_at(auxiliary.get(axis + '_interval_conditional',
                                                   auxiliary.get('interval_conditional')), index, False)),
             'valid_samples': _number(_at(auxiliary.get('interval_valid_samples'), index))}
            for index, coordinate in enumerate(evaluation.coordinate)]}
        break
    return {
        'name': label, 'quantity': item['quantity'], 'quantity_label': quantity_label,
        'coordinate_label': coordinate_label, 'n_simulations': simulated.n_simulations,
        'complete_point_rate': _percentage(item.get('complete_point_rate')),
        'complete_interval_rate': _percentage(item.get('complete_interval_rate')),
        'width': _number(item.get('mean_interval_width')), 'width_count': _count(item.get('width_counts')),
        'rmse': _number(item.get('rmse')), 'error_count': _count(item.get('error_counts')),
        'coverage': _percentage(item.get('coverage')), 'truth_count': _count(item.get('truth_counts')),
        'conditional_coverage': _percentage(item.get('conditional_coverage')),
        'interval_truth_count': _count(item.get('interval_truth_counts')),
        'point_count': _count(item.get('point_counts')), 'interval_count': _count(item.get('interval_counts')),
        'points': points, 'empirical': empirical, 'example': example,
        'diagnostics': _diagnostics(simulated, output, item['quantity']),
    }


def _existing_rows(existing):
    rows = []
    for side in SIDES:
        course = existing.get(side)
        if course is None:
            rows.append({'side': SIDE_LABELS[side], 'rows': 0, 'times': 'None',
                         'units': '0', 'precision': 'No existing calibration observations'})
            continue
        precision = course.observation_uncertainty
        individual = precision is None or precision.response_kind == 'individual'
        units = (str(len(course.time)) if individual else
                 str(sum(int(value) for value in precision.replicate_counts))
                 if precision.replicate_counts is not None else 'Not reported for means')
        interpretation = ('Unspecified observation precision' if precision is None else
                          'SEM of each reported mean; counts are retained without another division'
                          if precision.response_kind == 'mean' else 'Individual observation SD')
        if precision is not None and precision.source == 'future_model_assumption':
            interpretation = 'Individual SD explicitly assumed from the future variability model'
        rows.append({'side': SIDE_LABELS[side], 'rows': len(course.time),
                     'times': ', '.join(_number(value) for value in np.unique(course.time)),
                     'units': units, 'precision': interpretation})
    return rows


def _source_previews(result):
    previews = []
    for side in SIDES:
        details = result.source_details.get(side, {})
        reference = result.references.get(side)
        fixed = result.analyzed_existing.get(side)
        course = details.get('analyzed_source')
        series, band = [], None
        if reference is not None:
            series.append({'label': 'Central generating reference', 'color': '#24535c',
                           'dash': '', 'join': True,
                           'values': list(zip(reference.time_grid, reference.central_curve))})
            if np.count_nonzero(reference.valid_draws & (reference.draw_weights > 0)) > 1:
                lower, upper = _reference_band(reference)
                band = list(zip(reference.time_grid, lower, upper))
        if fixed is not None:
            series.append({'label': 'Fixed calibration observations', 'color': '#a75328',
                           'dash': '', 'join': False, 'values': list(zip(fixed.time, fixed.value))})
        observations = (list(zip(course.time, course.value)) if course is not None
                        and details.get('source_kind') == 'file' else None)
        normalization = details.get('normalization', {})
        fit = result.source_fits.get(side)
        parameters = []
        if fit is not None:
            for index, (name, value) in enumerate(fit.parameters.items()):
                variance = fit.covariance[index, index]
                parameters.append({'name': name, 'value': _number(value),
                                   'se': _number(np.sqrt(variance) if np.isfinite(variance)
                                                 and variance >= 0 else None)})
        previews.append({
            'side': SIDE_LABELS[side], 'identifier': 'calibration-source-' + side,
            'mode': details.get('mode', 'fixed'), 'source_kind': details.get('source_kind', 'none'),
            'model': details.get('model') or '—', 'parameters': parameters,
            'has_reference': reference is not None,
            'draws': '—' if reference is None else f'{reference.n_valid_draws} / {reference.n_draws}',
            'range': ('—' if reference is None else
                      '–'.join(_number(value) for value in reference.supported_time_range)),
            'slope': _number(normalization.get('slope', 1.)),
            'intercept': _number(normalization.get('intercept', 0.)),
            'generation_grid': details.get('generation_grid'),
            'plot': _plot(series, band=band, observations=observations),
        })
    return previews


def present_calibration_result(result, *, source_summaries=None, profile_summaries=None):
    """Format every candidate and preserve the backend availability denominators."""
    settings = result.settings
    method, approach = settings['method'], settings['approach_id']
    rows, sections = [], {'relationship': [], 'prediction': []}
    inspection = build_calibration_recovery(result)
    identifiers = {candidate['name']: candidate['id'] for candidate in inspection['candidates']}
    for label, summary in result.comparison.summaries.items():
        design, simulated = result.designs[label], result.comparison.results[label]
        side_rows = {}
        for side in SIDES:
            times = getattr(design, side + '_times')
            observation = getattr(design, side + '_observation_model')
            count = 0 if observation is None else int(observation.replicates_per_time)
            side_rows[side] = {'times': ', '.join(_number(value) for value in times) or 'None',
                               'time_count': len(times), 'replicates': count,
                               'units': len(times) * count,
                               'observation': _observation_description(observation)}
        row = {'id': identifiers[label], 'name': label, 'sides': side_rows,
               'new_units': sum(side_rows[side]['units'] for side in SIDES),
               'n_fit_successful': summary['n_fit_successful'],
               'n_simulations': summary['n_simulations'],
               'fit_failure_rate': _percentage(summary['fit_failure_rate']),
               'stages': [{'stage': stage, 'count': count}
                          for stage, count in summary.get('error_counts', {}).items()],
               'error_examples': [], 'outputs': {}}
        seen = set()
        for assessment in simulated.assessments:
            for stage, error in assessment.errors.items():
                record = (stage, error.exception_type, error.message)
                if record not in seen and len(row['error_examples']) < 4:
                    row['error_examples'].append({'stage': stage, 'type': error.exception_type,
                                                  'message': error.message})
                    seen.add(record)
        for output in sections:
            if output == 'relationship' and not settings.get('relationship', True):
                continue
            if output == 'prediction' and not settings.get('assess_target', False):
                continue
            item = summary.get(output)
            if item is not None:
                formatted = _output_result(label, output, item, simulated, method)
                sections[output].append(formatted)
                row['outputs'][output] = formatted
        rows.append(row)
    output_sections = []
    for output, candidates in sections.items():
        if not candidates:
            continue
        plot_series = [{'label': row['name'], 'color': color, 'join': True, 'dash': '',
                        'values': list(zip(result.comparison.summaries[row['name']][output]['coordinate'],
                                           result.comparison.summaries[row['name']][output]['pointwise_mean_interval_width']))}
                       for row, color in zip(candidates, ('#24535c', '#a75328', '#6f4a92', '#2c7b50', '#9c4966', '#556280'))]
        output_sections.append({'key': output,
                                'title': 'IVIVC relationship' if output == 'relationship' else 'Specified target predictions',
                                'quantity_label': candidates[0]['quantity_label'],
                                'coordinate_label': candidates[0]['coordinate_label'],
                                'candidates': candidates,
                                'width_plot': (None if candidates[0]['quantity'] == 'time_constant_ratio'
                                               else _plot(plot_series))})
    generation = []
    for entry in result.generation_outcomes:
        generation.append({**entry, 'side_label': SIDE_LABELS.get(entry.get('side'), entry.get('side', '')),
                           'times_text': ', '.join(_number(value) for value in entry.get('times', ())) or 'None',
                           'criterion_text': _number(entry.get('criterion'))})
    normalizations = {'none': 'None', 'initial_value': 'Fixed initial-value scaling',
                      'min_max': 'Fixed min–max scaling'}
    normalized = normalizations.get(settings.get('normalization', 'none'), settings.get('normalization', 'none'))
    if settings.get('one_minus'):
        normalized += '; then one minus response'
    setting_rows = [('Analysis task', 'Compare calibration sampling'),
                    ('IVIVC approach', approach), ('IVIVC model', settings['model_name']),
                    ('Prediction method', method), ('Calibration interpolation', settings.get('interpolation', 'none')),
                    ('Varied environment', {
                        'in_vivo': 'In vivo only; keep in vitro data fixed',
                        'in_vitro': 'In vitro only; keep in vivo data fixed',
                        'both': 'Both environments',
                    }.get(settings['varied_environment'], settings['varied_environment'])),
                    ('Common response preprocessing', normalized),
                    ('Source parameter draws per environment', str(settings['reference_draws'])),
                    ('Simulated calibration studies per candidate', str(settings['n_simulations'])),
                    ('Propagation draws per fitted IVIVC', str(settings['interval_draws'])),
                    ('Random seed', str(settings['random_seed']))]
    target_transform = settings.get('target_normalization')
    if target_transform is not None:
        setting_rows.append(('Fixed target response transformation',
                             'Analyzed response = ' + _number(target_transform['slope']) +
                             ' × supplied response + ' + _number(target_transform['intercept'])))
    return {
        'calibration_inspection': inspection,
        'rows': rows, 'output_sections': output_sections, 'source_previews': _source_previews(result),
        'existing_rows': _existing_rows(result.analyzed_existing), 'generation_outcomes': generation,
        'warnings': list(result.warnings), 'is_approach1': approach == 'approach1',
        'source_summaries': source_summaries or [], 'profile_summaries': profile_summaries or [],
        'settings_rows': setting_rows,
    }
