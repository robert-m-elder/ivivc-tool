"""Finite display summaries of retained IVIVC evaluations and target trajectories.

No observations, fits, interval draws, or new prediction coordinates are created.
Time-mapping target ranges stay horizontal at each supplied target response;
they are not interpreted as response uncertainty on a common in vivo time grid.
"""
from __future__ import annotations

import numpy as np

from .design_recovery_inspection import _pointwise_summary, _values


def _labels(quantity, output, method):
    if quantity == 'time_constant_ratio':
        return 'In vivo / in vitro time-constant ratio (dimensionless)', 'Constant ratio'
    if quantity == 'value_ratio':
        return 'In vivo / in vitro response ratio (dimensionless)', 'In vitro time'
    if quantity == 'in_vivo_response':
        return 'Predicted in vivo response (analyzed scale)', 'Target in vitro time'
    return 'Predicted in vivo time', (
        'In vivo response level (analyzed scale)' if method == 'observed_response_inverse'
        else 'Target in vitro time' if output == 'prediction' else 'In vitro time')


def _ordered(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return bool(len(values) < 2 or np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))


def _spec(result, output, method):
    for name, simulated in result.comparison.results.items():
        summary = result.comparison.summaries.get(name, {}).get(output)
        if summary is not None:
            return summary['quantity'], np.asarray(summary['coordinate'], dtype=float)
        for assessment in simulated.assessments:
            evaluation = getattr(assessment, output, None)
            if evaluation is not None:
                return evaluation.quantity, np.asarray(evaluation.coordinate, dtype=float)
        spec = simulated.metadata.get('evaluation_spec', {}).get(output)
        if spec is not None:
            return spec['quantity'], np.asarray(spec['coordinate'], dtype=float)
    quantity = (('in_vivo_response' if method == 'value_ratio' else 'in_vivo_time')
                if output == 'prediction' else 'time_constant_ratio' if method == 'tau_ratio'
                else 'value_ratio' if method == 'value_ratio' else 'in_vivo_time')
    return quantity, np.array([], dtype=float)


def _retained(simulated, output, quantity, coordinate):
    count, width = len(simulated.assessments), len(coordinate)
    shape = (count, width)
    arrays = {name: np.full(shape, np.nan) for name in ('estimate', 'lower', 'upper', 'tx', 'ty')}
    points, intervals = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for index, assessment in enumerate(simulated.assessments):
        evaluation = getattr(assessment, output, None)
        if (evaluation is None or evaluation.quantity != quantity or
                not np.array_equal(evaluation.coordinate, coordinate, equal_nan=True)):
            continue
        points[index] = evaluation.point_available
        intervals[index] = evaluation.interval_available
        arrays['estimate'][index] = np.where(points[index], evaluation.estimate, np.nan)
        arrays['lower'][index] = np.where(intervals[index], evaluation.lower, np.nan)
        arrays['upper'][index] = np.where(intervals[index], evaluation.upper, np.nan)
        for name, attr in (('tx', 'in_vivo_time'), ('ty', 'in_vivo_value')):
            values = np.asarray(getattr(evaluation.trajectory, attr), dtype=float)
            if values.shape == (width,):
                arrays[name][index] = values
    truth = simulated.truth.get(output)
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != shape:
            truth = None
    return {**arrays, 'points': points, 'intervals': intervals, 'truth': truth}


def _summary(values):
    return _pointwise_summary(values, np.ones(len(values), dtype=bool))


def _truth(values, independent, axis):
    if values is None:
        return None
    summary = _summary(values)
    dependent = summary['median']
    x, y = (dependent, _values(independent)) if axis == 'x' else (_values(independent), dependent)
    return {'x': x, 'y': y, 'counts': summary['available_counts']}


def _example_message(simulated, output, points):
    if np.any(points):
        return 'Study 1: retained point estimates and fitted intervals where available.'
    errors = getattr(simulated.assessments[0], 'errors', {}) if simulated.assessments else {}
    error = errors.get(output) or errors.get('fit') or errors.get('preprocessing')
    suffix = f' {getattr(error, "message", str(error))}' if error is not None else ''
    return f'Study 1 has no available {output} estimates; no replacement study is substituted.' + suffix


def _view(simulated, output, retained, independent, dependent, axis='y', *, available=True, message=''):
    points = retained['points'] & np.isfinite(independent)[None, :] & np.isfinite(dependent)
    dependent = np.where(points, dependent, np.nan)
    if not available:
        dependent[:] = np.nan
    summary = _summary(dependent)
    x, y = (summary['median'], _values(independent)) if axis == 'x' else (_values(independent), summary['median'])
    count, width = dependent.shape
    first = dependent[0] if count else np.full(width, np.nan)
    example_x, example_y = (_values(first), _values(independent)) if axis == 'x' else (_values(independent), _values(first))
    first_points = points[0] if count else np.zeros(width, dtype=bool)
    first_intervals = retained['intervals'][0] & first_points if count else first_points
    return {
        'available': available, 'message': message, 'x': x, 'y': y,
        'lower': summary['lower'], 'upper': summary['upper'], 'counts': summary['available_counts'],
        'interval_counts': np.sum(retained['intervals'] & points, axis=0).astype(int).tolist(),
        'truth': _truth(retained['truth'], independent, axis) if available else None,
        'example': {
            'x': example_x, 'y': example_y,
            'lower': _values(np.where(first_intervals, retained['lower'][0], np.nan)) if count else [None] * width,
            'upper': _values(np.where(first_intervals, retained['upper'][0], np.nan)) if count else [None] * width,
            'point_counts': first_points.astype(int).tolist(),
            'interval_counts': first_intervals.astype(int).tolist(),
            'truth': (_truth(retained['truth'][:1], independent, axis)
                      if retained['truth'] is not None and available else None),
            'message': _example_message(simulated, output, first_points),
        },
    }


def _fixed_axis(retained_values, name, width):
    """Verify a shared physical target coordinate instead of medianing both axes."""
    fixed = np.full(width, np.nan)
    consistent = True
    for retained in retained_values:
        for row in np.where(retained['points'], retained[name], np.nan):
            common = np.isfinite(fixed) & np.isfinite(row)
            if np.any(fixed[common] != row[common]):
                consistent = False
            missing = ~np.isfinite(fixed) & np.isfinite(row)
            fixed[missing] = row[missing]
    return fixed, consistent and bool(np.any(np.isfinite(fixed)))


def build_calibration_outputs(result):
    """Build plot coordinates from every retained requested evaluation."""
    if result.comparison is None:
        return {}
    method = result.settings.get('method', 'value_ratio')
    outputs = {}
    for output in ('relationship', 'prediction'):
        if not result.settings.get('relationship' if output == 'relationship' else 'assess_target', output == 'relationship'):
            continue
        quantity, coordinate = _spec(result, output, method)
        quantity_label, coordinate_label = _labels(quantity, output, method)
        title = 'IVIVC relationship' if output == 'relationship' else 'Target predictions'
        scalar = quantity == 'time_constant_ratio'
        message = ('Medians and pointwise central 95% ranges describe variation across available simulated studies. '
                   'They are not confidence intervals within one fit. A range needs two usable estimates. '
                   'Counts retain failed and unavailable studies in their denominators.')
        metadata = {
            'title': title, 'quantity': quantity, 'quantity_label': quantity_label,
            'coordinate_label': coordinate_label, 'scalar': scalar, 'message': message,
            'views': {'native': {
                'label': 'Relationship output' if output == 'relationship' else 'Prediction output',
                'title': title, 'x_label': 'Design' if scalar else coordinate_label,
                'y_label': quantity_label, 'range_axis': 'y', 'scalar': scalar,
                'connect_points': _ordered(coordinate), 'available': True, 'help': message,
            }}, 'candidates': [],
        }
        simulated_items = list(result.comparison.results.items())
        retained = [_retained(item, output, quantity, coordinate) for _, item in simulated_items]
        response_axis, fixed, safe_response = 'y', None, False
        if output == 'prediction':
            response_axis = 'y' if quantity == 'in_vivo_response' else 'x'
            fixed, safe_response = _fixed_axis(retained, 'tx' if response_axis == 'y' else 'ty', len(coordinate))
            response_help = ('Time ranges are horizontal at each supplied target response; they are not response bands on a common in vivo time grid.'
                             if response_axis == 'x' else 'Response ranges are vertical at the retained target times.')
            if not safe_response:
                response_help = ('A response-versus-time summary is unavailable because no shared fixed target coordinate could be verified. '
                                 'The native prediction output remains available.')
            metadata['views']['response'] = {
                'label': 'Response versus in vivo time', 'title': 'Predicted target response versus in vivo time',
                'x_label': 'Predicted in vivo time', 'y_label': 'Predicted in vivo response (analyzed scale)',
                'range_axis': response_axis, 'scalar': False, 'connect_points': _ordered(fixed),
                'available': safe_response, 'help': response_help + ' ' + message,
            }
        for index, ((name, simulated), values) in enumerate(zip(simulated_items, retained)):
            candidate = {
                'id': f'calibration-design-{index}', 'name': name,
                'n_simulations': len(simulated.assessments),
                'n_fit_successful': sum(bool(item.fit_available) for item in simulated.assessments),
                'views': {'native': _view(simulated, output, values, coordinate, values['estimate'])},
            }
            if output == 'prediction':
                dependent = values['ty'] if response_axis == 'y' else values['tx']
                candidate['views']['response'] = _view(
                    simulated, output, values, fixed, dependent, response_axis,
                    available=safe_response, message=metadata['views']['response']['help'],
                )
                # Examples retain their own physical trajectory, even when a
                # common target coordinate cannot support an across-study view.
                if len(simulated.assessments):
                    mask = values['points'][0] & np.isfinite(values['tx'][0]) & np.isfinite(values['ty'][0])
                    example = candidate['views']['response']['example']
                    example['x'] = _values(np.where(mask, values['tx'][0], np.nan))
                    example['y'] = _values(np.where(mask, values['ty'][0], np.nan))
                    example['point_counts'] = mask.astype(int).tolist()
                    example['message'] = _example_message(simulated, output, mask)
            metadata['candidates'].append(candidate)
        outputs[output] = metadata
    return outputs
