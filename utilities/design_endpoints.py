"""Endpoint summaries from retained planning curves, without new fitting or draws.

Positive infinity denotes a valid curve that does not reach its endpoint within
the assessment horizon. NaN denotes an invalid curve; failed fits are counted
separately. Only the JSON-facing summaries replace these sentinels with nulls.
"""
from __future__ import annotations

import numpy as np

from ivivc_core import get_model_spec


_PROBABILITIES = (.025, .5, .975)


def endpoint_definitions(settings):
    """Absent settings preserve T50/T90; an explicitly empty list selects none."""
    defaults = [dict(kind='relative_decrease', value=value, direction='decreasing',
                     label=f'T{value:g}') for value in (50., 90.)]
    return [dict(item) for item in settings.get('recovery_endpoints', defaults)]


def _crossings(grid, curves, definition):
    """First threshold attainment, interpolated between retained grid values."""
    curves = np.asarray(curves, dtype=float)
    times = np.full(len(curves), np.nan)
    relative = definition['kind'] == 'relative_decrease'
    for index, curve in enumerate(curves):
        if np.any(~np.isfinite(curve)) or not len(curve):
            continue
        if relative:
            if curve[0] <= 0:
                continue
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                values = 1. - curve / curve[0]
            if np.any(~np.isfinite(values)):
                continue
            threshold = definition['value'] / 100.
            reached = values >= threshold
        else:
            values, threshold = curve, definition['value']
            reached = (values <= threshold if definition['direction'] == 'decreasing'
                       else values >= threshold)
        positions = np.flatnonzero(reached)
        if not len(positions):
            times[index] = np.inf
            continue
        position = int(positions[0])
        if position == 0:
            times[index] = grid[0]
            continue
        y0, y1 = values[position - 1:position + 1]
        scale = max(abs(float(y0)), abs(float(y1)), abs(float(threshold)))
        fraction = ((threshold / scale - y0 / scale) /
                    (y1 / scale - y0 / scale))
        fraction = np.clip(fraction, 0., 1.)
        times[index] = (1. - fraction) * grid[position - 1] + fraction * grid[position]
    return times


def _quantiles(values, weights=None, *, invalid_status='unavailable'):
    values = np.asarray(values, dtype=float)
    usable = ~np.isnan(values)
    values = values[usable]
    weights = (np.ones(len(values)) if weights is None else
               np.asarray(weights, dtype=float)[usable])
    if len(values):
        order = np.argsort(values, kind='stable')
        values = values[order]
        cumulative = np.cumsum(weights[order])
        cumulative /= cumulative[-1]
        cumulative[-1] = 1.
    summaries = []
    for probability in _PROBABILITIES:
        value = (float(values[np.searchsorted(cumulative, probability, side='left')])
                 if len(values) else np.nan)
        summaries.append({
            'probability': probability,
            'value': value if np.isfinite(value) else None,
            'status': ('resolved' if np.isfinite(value) else 'beyond_horizon'
                       if np.isposinf(value) else invalid_status),
        })
    return summaries


def reference_endpoint_summary(reference, definition, baseline_time):
    """Weighted reference quantiles, retaining all positive noncrossing weight."""
    lower, upper = reference.supported_time_range
    active = np.flatnonzero(reference.valid_draws & (reference.draw_weights > 0))
    weights = np.asarray(reference.draw_weights[active], dtype=float)
    weights = weights / weights.sum()
    if not np.isfinite(baseline_time) or not lower <= baseline_time <= upper:
        times = np.full(len(active), np.nan)
    else:
        grid = np.union1d([baseline_time, upper], reference.time_grid[
            (reference.time_grid > baseline_time) & (reference.time_grid < upper)])
        times = _crossings(grid, reference.evaluate(grid, draw_indices=active), definition)
    invalid_weight = float(weights[np.isnan(times)].sum())
    invalid = invalid_weight > 0
    return {
        'quantiles': _quantiles(np.full_like(times, np.nan) if invalid else times,
                                weights, invalid_status='invalid_reference'),
        'reached_weight': None if invalid else float(weights[np.isfinite(times)].sum()),
        'invalid_weight': invalid_weight,
        'range_available': bool(len(active) > 1 and not invalid),
    }


def _assessment_curves(result, simulated, baseline_time):
    """Insert exact baseline/horizon values without changing the IRMSE grid."""
    lower, upper = result.reference.supported_time_range
    original = np.asarray(simulated.evaluation_times, dtype=float)
    grid = np.union1d([baseline_time, upper], original[
        (original > baseline_time) & (original < upper)])
    shape = (len(simulated.fit_success), len(grid))
    truth, fitted = np.full(shape, np.nan), np.full(shape, np.nan)
    if not np.isfinite(baseline_time) or not lower <= baseline_time <= upper:
        return grid, truth, fitted
    known = np.searchsorted(original, grid)
    present = (known < len(original))
    present[present] &= original[known[present]] == grid[present]
    truth[:, present] = simulated.true_curves[:, known[present]]
    fitted[:, present] = simulated.fitted_curves[:, known[present]]
    missing = ~present
    if np.any(missing):
        truth[:, missing] = result.reference.evaluate(
            grid[missing], draw_indices=simulated.trajectory_draw_indices)
        spec = get_model_spec(simulated.model_name)
        for index in np.flatnonzero(simulated.fit_success):
            parameters = dict(zip(spec.parameter_names, simulated.parameter_estimates[index]))
            try:
                with np.errstate(all='ignore'):
                    fitted[index, missing] = spec.model_function(grid[missing], **parameters)
            except (ValueError, TypeError, OverflowError, FloatingPointError):
                # A successful original fit may still be invalid at the baseline.
                continue
    return grid, truth, fitted


def _study_summary(times, success=None):
    active = np.ones(len(times), dtype=bool) if success is None else success
    valid = active & ~np.isnan(times)
    result = {
        'n_valid': int(np.count_nonzero(valid)),
        'n_invalid': int(np.count_nonzero(active & np.isnan(times))),
        'n_reached': int(np.count_nonzero(active & np.isfinite(times))),
        'n_beyond': int(np.count_nonzero(active & np.isposinf(times))),
        'quantiles': _quantiles(times[active]),
        'range_available': bool(np.count_nonzero(valid) > 1),
    }
    if success is not None:
        result['n_failed'] = int(np.count_nonzero(~success))
    return result


def _paired_summary(truth, recovered, success):
    valid = success & ~np.isnan(truth) & ~np.isnan(recovered)
    true_finite, recovered_finite = np.isfinite(truth), np.isfinite(recovered)
    finite = valid & true_finite & recovered_finite
    with np.errstate(over='ignore', invalid='ignore'):
        errors = recovered[finite] - truth[finite]
    # Time bounds are finite and nonnegative in planning, so these differences
    # remain finite; filter defensively rather than emitting invalid JSON.
    errors = errors[np.isfinite(errors)]
    return {
        'n_finite': int(len(errors)),
        'median_absolute_error': float(np.median(np.abs(errors))) if len(errors) else None,
        'p90_absolute_error': float(np.quantile(np.abs(errors), .9)) if len(errors) else None,
        'median_signed_error': float(np.median(errors)) if len(errors) else None,
        'true_only': int(np.count_nonzero(valid & true_finite & ~recovered_finite)),
        'recovered_only': int(np.count_nonzero(valid & ~true_finite & recovered_finite)),
        'neither': int(np.count_nonzero(valid & ~true_finite & ~recovered_finite)),
        'n_invalid': int(np.count_nonzero(success & ~valid)),
        'n_failed': int(np.count_nonzero(~success)),
    }


def build_endpoint_inspection(result):
    """Summarize each recovered endpoint against its own simulated truth.

    Simulated studies already sample the reference weights and therefore have
    equal summary weights. Time quantiles use an inverse empirical CDF so that
    noncrossing studies retain their mass beyond the assessment horizon.
    """
    baseline_time = float(result.settings.get('baseline_time',
                          result.reference.supported_time_range[0]))
    inspection = {'baseline_time': baseline_time,
                  'horizon': float(result.reference.supported_time_range[1]), 'schedules': []}
    definitions = endpoint_definitions(result.settings)
    if result.comparison is None or not definitions:
        return inspection
    identifiers = {name: f'schedule-{index}' for index, name in enumerate(result.schedules)}
    for name, simulated in result.comparison.results.items():
        grid, true_curves, recovered_curves = _assessment_curves(result, simulated, baseline_time)
        success = np.asarray(simulated.fit_success, dtype=bool)
        endpoints = []
        for definition in definitions:
            truth = _crossings(grid, true_curves, definition)
            recovered = _crossings(grid, recovered_curves, definition)
            endpoints.append({
                'definition': dict(definition), 'n_simulations': int(len(success)),
                'truth': _study_summary(truth),
                'recovered': _study_summary(recovered, success),
                'paired': _paired_summary(truth, recovered, success),
            })
        inspection['schedules'].append({'id': identifiers[name], 'name': name,
                                        'endpoints': endpoints})
    return inspection
