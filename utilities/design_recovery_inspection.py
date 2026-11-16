"""Compact inspection data from retained study-planning simulation outputs.

No models are fitted and no observations are generated here. Empirical ranges
describe pointwise variation across successful simulated studies, separately
from any confidence intervals reported by a fitted model.
"""
from __future__ import annotations

import numpy as np


_EXAMPLE_INDEX = 0
_QUANTILES = (.025, .5, .975)


def _values(values):
    """Keep unavailable coordinates as JSON null so plots can retain gaps."""
    return [float(value) if np.isfinite(value) else None
            for value in np.asarray(values, dtype=float)]


def _pointwise_summary(values, success):
    values = np.asarray(values, dtype=float)
    success = np.asarray(success, dtype=bool)
    lower, median, upper, counts = [], [], [], []
    for column in values.T:
        finite = column[success & np.isfinite(column)]
        counts.append(int(len(finite)))
        if not len(finite):
            lower.append(None)
            median.append(None)
            upper.append(None)
            continue
        with np.errstate(invalid='ignore', over='ignore'):
            quantiles = np.quantile(finite, _QUANTILES)
            if not np.isfinite(quantiles).all():
                # Linear interpolation can overflow its intermediate difference
                # for finite endpoints of opposite sign. Rescale only those
                # affected outputs, retaining ordinary quantile precision.
                scale = np.max(np.abs(finite))
                stable = np.quantile(finite / scale, _QUANTILES) * scale
                quantiles = np.where(np.isfinite(quantiles), quantiles, stable)
        low, middle, high = _values(quantiles)
        median.append(middle)
        # One recovered value gives a median, but cannot demonstrate spread.
        lower.append(low if len(finite) > 1 else None)
        upper.append(high if len(finite) > 1 else None)
    return {'lower': lower, 'median': median, 'upper': upper,
            'available_counts': counts}


def _completed_rows(result):
    completed = getattr(result, 'completed_observations', None)
    if completed is None:
        return []
    precision = completed.observation_uncertainty
    rows = []
    for index, (time, response) in enumerate(zip(completed.time, completed.response)):
        count = (None if precision is None or precision.replicate_counts is None
                 else int(precision.replicate_counts[index]))
        rows.append({
            'time': float(time), 'response': float(response), 'role': 'completed',
            'response_kind': completed.response_kind,
            'observation_sigma': (None if precision is None else
                                  _values([precision.sigma[index]])[0]),
            'uncertainty_kind': (None if precision is None else
                                 'SEM' if completed.response_kind == 'mean' else 'SD'),
            'reported_replicate_count': count,
        })
    return rows


def _shared_example(comparison):
    description = {
        'available': False, 'simulation_index': _EXAMPLE_INDEX,
        'study_number': _EXAMPLE_INDEX + 1, 'trajectory_draw_index': None,
        'reason': 'No simulated schedule comparison is available.',
    }
    if comparison is None or not comparison.results:
        return description
    expected = np.asarray(comparison.simulation_plan.trajectory_draw_indices)
    if not len(expected) or any(
        not np.array_equal(simulated.trajectory_draw_indices, expected)
        for simulated in comparison.results.values()
    ):
        description['reason'] = ('The retained schedules do not have matching shared '
                                 'trajectory selections; no shared example is supplied.')
        return description
    description.update(available=True, trajectory_draw_index=int(expected[_EXAMPLE_INDEX]),
                       reason='Study 1 is predetermined for every schedule, including failed fits.')
    return description


def _example(simulated, result, shared):
    if not shared['available']:
        return None
    index = _EXAMPLE_INDEX
    success = bool(simulated.fit_success[index])
    fitted = np.asarray(simulated.fitted_curves[index], dtype=float)
    truth = np.asarray(simulated.true_curves[index], dtype=float)
    # A failed fit must stay unavailable even if an incidental value remains.
    recovered = fitted if success else np.full_like(fitted, np.nan)
    with np.errstate(invalid='ignore', over='ignore'):
        errors = recovered - truth
    completed = getattr(result, 'completed_observations', None)
    completed_times = [] if completed is None else completed.unique_times
    replicates = int(simulated.metadata.get('replicates_per_time',
                     result.settings['replicates_per_time']))
    means = []
    for column, time in enumerate(simulated.design_times):
        is_completed = bool(np.any(np.isclose(completed_times, time, rtol=0., atol=1e-12)))
        means.append({
            'time': float(time),
            'response': _values([simulated.observed_means[index, column]])[0],
            'expected_response': _values([simulated.expected_values[index, column]])[0],
            'role': 'completed' if is_completed else 'new',
            'planned_replicate_count': None if is_completed else replicates,
        })
    return {
        'simulation_index': index, 'study_number': index + 1,
        'trajectory_draw_index': int(simulated.trajectory_draw_indices[index]),
        'fit_success': success,
        'fit_message': (None if success else
                        str(simulated.fit_messages[index] or 'The simulated study could not be fitted.')),
        'true_response': _values(truth), 'recovered_response': _values(recovered),
        'recovery_error': _values(errors), 'observed_time_means': means,
    }


def build_recovery_inspection(result):
    """Summarize retained curves and keep one shared, predetermined example.

    Monte Carlo draws already sample the reference weights. These empirical
    quantiles therefore use equal simulation weights. Errors are paired with
    each simulated study's own true trajectory before calculating quantiles.
    Every point carries its usable count; fewer than two values suppress the
    range, while zero usable values also suppress the median.
    """
    comparison = result.comparison
    shared = _shared_example(comparison)
    inspection = {
        'schema_version': 1,
        'quantiles': list(_QUANTILES),
        'quantile_method': 'linear', 'minimum_range_count': 2,
        'range_interpretation': ('Pointwise central 95% ranges across successful simulated '
                                 'studies; not fitted confidence intervals or simultaneous bands.'),
        'error_definition': 'Recovered response minus that simulated study\u2019s own true response.',
        'observation_interpretation': ('New points are retained per-time means of simulated replicates. '
                                       'Completed time means are for display only; original completed '
                                       'rows remain separate and were held fixed during fitting.'),
        'shared_example': shared,
        'completed_observations': _completed_rows(result),
        'fit_reported_intervals': {
            'available': False,
            'reason': ('Pointwise fitted confidence bounds are not retained by this simulation output. '
                       'The existing mean fitted curve interval width is a separate summary.'),
        },
        'schedules': [],
    }
    if comparison is None:
        return inspection
    identifiers = {name: f'schedule-{index}' for index, name in enumerate(result.schedules)}
    for name, simulated in comparison.results.items():
        fitted = np.asarray(simulated.fitted_curves, dtype=float)
        with np.errstate(invalid='ignore', over='ignore'):
            errors = fitted - np.asarray(simulated.true_curves, dtype=float)
        success = np.asarray(simulated.fit_success, dtype=bool)
        inspection['schedules'].append({
            'id': identifiers[name], 'name': name,
            'evaluation_times': _values(simulated.evaluation_times),
            'n_simulations': int(len(success)), 'n_successful': int(np.count_nonzero(success)),
            'n_failed': int(np.count_nonzero(~success)),
            'n_complete_recovered_curves': int(np.count_nonzero(
                success & np.all(np.isfinite(fitted), axis=1))),
            'recovered_response': _pointwise_summary(fitted, success),
            'recovery_error': _pointwise_summary(errors, success),
            'example': _example(simulated, result, shared),
        })
    return inspection
