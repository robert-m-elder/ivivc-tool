"""Compact response-curve inspection from retained calibration simulations.

No models are fitted or observations generated here. The empirical ranges
describe pointwise variation across simulated studies, not confidence intervals
within an individual fit. Study 1 is fixed before fit availability is examined.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ivivc_core import get_model_spec

from .design_recovery_inspection import _pointwise_summary, _values


_SIDES = ('in_vitro', 'in_vivo')
_LABELS = {'in_vitro': 'In vitro', 'in_vivo': 'In vivo'}


def _points(course):
    if course is None:
        return []
    return [{'time': float(time), 'response': float(response)}
            for time, response in zip(course.time, course.value)
            if np.isfinite(time) and np.isfinite(response)]


def _new_points(course, existing):
    """Remove original rows as a multiset, preserving repeated-time additions."""
    original = Counter((point['time'], point['response']) for point in _points(existing))
    new = []
    for point in _points(course):
        pair = point['time'], point['response']
        if original[pair]:
            original[pair] -= 1
        else:
            new.append(point)
    return new


def _grid(result, side):
    reference = result.references.get(side)
    if reference is not None:
        lower, upper = reference.supported_time_range
        retained = np.asarray(reference.time_grid, dtype=float)
        return np.union1d([lower, upper], retained[(retained >= lower) & (retained <= upper)])
    times = []
    existing = result.analyzed_existing.get(side)
    if existing is not None:
        times.extend(np.asarray(existing.time, dtype=float))
    for simulated in result.comparison.results.values():
        for course in getattr(simulated, side + '_data'):
            if course is not None:
                times.extend(np.asarray(course.time, dtype=float))
                # A candidate's measurement times are identical across studies.
                break
    times = np.asarray(times, dtype=float)
    times = np.unique(times[np.isfinite(times)])
    if len(times) < 2:
        return times
    return np.linspace(times[0], times[-1], 101)


def _profile_curve(assessment, side, model_name, grid):
    unavailable = np.full(len(grid), np.nan)
    if not assessment.fit_available:
        return unavailable
    model = assessment.analysis.model_results.get('approach2:' + model_name)
    if model is None:
        return unavailable
    profile = getattr(model, side + '_fit', None)
    if profile is None:
        return unavailable
    try:
        spec = get_model_spec(profile.model_name)
        with np.errstate(all='ignore'):
            curve = np.asarray(spec.model_function(grid, **profile.parameters), dtype=float)
        return curve if curve.shape == grid.shape else unavailable
    except (KeyError, TypeError, ValueError, FloatingPointError, OverflowError):
        return unavailable


def _example_pairing(result):
    candidates = list(result.comparison.results.values())
    if not candidates or any(not len(item.assessments) for item in candidates):
        return False, 'No shared first simulated study is available.'
    for side in _SIDES:
        reference = result.references.get(side)
        if reference is None:
            continue
        indices = []
        for simulated in candidates:
            values = np.asarray(simulated.metadata.get(side + '_draw_indices', ()), dtype=int)
            if not len(values) or not 0 <= values[0] < reference.n_draws:
                return False, 'Generating-curve selections for study 1 are unavailable.'
            indices.append(int(values[0]))
        if len(set(indices)) != 1:
            return False, 'Study 1 does not use matching generating curves across designs.'
    return True, 'Study 1 is predetermined for every selected design, including failed fits.'


def _truth(result, simulated, side, grid, shared):
    reference = result.references.get(side)
    if reference is None or not shared:
        return None
    indices = np.asarray(simulated.metadata.get(side + '_draw_indices', ()), dtype=int)
    if not len(indices):
        return None
    return _values(reference.evaluate(grid, draw_indices=indices[:1])[0])


def _failure_message(assessment):
    if assessment.fit_available:
        return 'The retained fit supplies no usable reconstructed response on this grid.'
    errors = getattr(assessment, 'errors', {})
    if errors:
        stage, error = next(iter(errors.items()))
        return f'{stage}: {getattr(error, "message", str(error))}'
    return 'The IVIVC fit was unavailable in simulated study 1.'


def build_calibration_recovery(result):
    """Return finite inspection values using only retained profile parameters."""
    approach = result.settings['approach_id']
    supported = approach == 'approach2'
    inspection = {
        'schema_version': 1, 'approach_id': approach, 'supported': supported,
        'message': ('Response profiles are reconstructed from retained Approach 2 fits.'
                    if supported else 'Approach 1 fits a time-to-time mapping rather than separate '
                    'response-versus-time profiles. Reconstructed calibration curves are not '
                    'available for this approach; its relationship and prediction results remain below.'),
        'default_environment': ('in_vitro' if result.settings['varied_environment'] == 'in_vitro'
                                else 'in_vivo'),
        'environments': {}, 'candidates': [],
        'example': {'study_number': 1, 'available': False, 'message': ''},
    }
    comparison = result.comparison
    if not supported:
        if comparison is not None:
            inspection['candidates'] = [
                {'id': f'calibration-design-{index}', 'name': name,
                 'n_simulations': len(simulated.assessments),
                 'n_fit_successful': sum(bool(item.fit_available) for item in simulated.assessments),
                 'sides': {}}
                for index, (name, simulated) in enumerate(comparison.results.items())
            ]
        inspection['example']['message'] = inspection['message']
        return inspection
    if comparison is None:
        inspection['example']['message'] = 'No simulated calibration comparison is available.'
        return inspection
    shared, shared_message = _example_pairing(result)
    inspection['example'].update(available=shared, message=shared_message)
    grids = {side: _grid(result, side) for side in _SIDES}
    fixed_curves = {side: [] for side in _SIDES}
    for side in _SIDES:
        reference = result.references.get(side)
        fixed = (result.source_details.get(side, {}).get('mode') == 'fixed' or
                 result.settings['varied_environment'] not in (side, 'both'))
        inspection['environments'][side] = {
            'label': _LABELS[side], 'time': _values(grids[side]), 'fixed': fixed,
            'reference': (None if reference is None else
                          {'central': _values(reference.evaluate_central(grids[side]))}),
            'fixed_curve': None, 'fixed_curve_shared': False,
            'existing_points': _points(result.analyzed_existing.get(side)),
        }
    for index, (name, simulated) in enumerate(comparison.results.items()):
        successful = np.asarray([item.fit_available for item in simulated.assessments], dtype=bool)
        candidate = {
            'id': f'calibration-design-{index}', 'name': name,
            'n_simulations': len(successful), 'n_fit_successful': int(np.count_nonzero(successful)),
            'sides': {},
        }
        for side in _SIDES:
            grid = grids[side]
            curves = np.asarray([_profile_curve(item, side, result.settings['model_name'], grid)
                                 for item in simulated.assessments], dtype=float).reshape((len(successful), len(grid)))
            summary = _pointwise_summary(curves, successful)
            usable = successful & np.any(np.isfinite(curves), axis=1)
            if inspection['environments'][side]['fixed']:
                fixed_curves[side].extend(curves[usable])
            first = simulated.assessments[0] if len(successful) else None
            curve = curves[0] if len(successful) else np.full(len(grid), np.nan)
            available = bool(first is not None and first.fit_available and np.any(np.isfinite(curve)))
            datasets = getattr(simulated, side + '_data')
            first_data = datasets[0] if len(datasets) else None
            candidate['sides'][side] = {
                **summary, 'n_curves': int(np.count_nonzero(usable)),
                'example': {
                    'fit_available': available, 'curve': _values(curve),
                    'truth': _truth(result, simulated, side, grid, shared),
                    'new_points': _new_points(first_data, result.analyzed_existing.get(side)),
                    'message': ('Fit available for simulated study 1.' if available else
                                _failure_message(first) if first is not None else 'No simulated study is available.'),
                },
            }
        inspection['candidates'].append(candidate)
    for side, curves in fixed_curves.items():
        if curves and all(np.array_equal(curves[0], curve, equal_nan=True) for curve in curves[1:]):
            inspection['environments'][side].update(
                fixed_curve=_values(curves[0]), fixed_curve_shared=True,
            )
    return inspection
