"""Display measures from already calculated curve and endpoint summaries.

This adapter performs no simulation, fitting, or endpoint calculation. It keeps
the numerical table and interactive figure on the same selected measure while
retaining noncrossing statuses separately from finite plotting coordinates.
"""
from __future__ import annotations

import numpy as np

from .design_endpoints import endpoint_definitions


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _measures(definitions):
    measures = [{
        'id': 'irmse', 'label': 'Whole curve — IRMSE', 'kind': 'curve',
        'views': {'error': {
            'title': 'Simulated curve recovery error versus study size',
            'y_label': 'IRMSE (analyzed response units)',
            'primary_label': 'Mean IRMSE', 'secondary_label': '90th percentile IRMSE',
            'help': 'IRMSE compares each recovered curve with its simulated true curve '
                    'across the assessment range. Mean and 90th-percentile errors use '
                    'successful fits with finite errors. Lower error means closer '
                    'recovery under the supplied assumptions; read errors with fit counts.',
        }},
    }]
    for index, definition in enumerate(definitions):
        label = definition['label']
        measures.append({
            'id': f'endpoint-{index}', 'label': label, 'kind': 'endpoint',
            'views': {
                'error': {
                    'title': f'{label}: absolute timing error versus study size',
                    'y_label': 'Absolute timing error (in vivo time units)',
                    'primary_label': 'Median absolute time error',
                    'secondary_label': '90th percentile absolute time error',
                    'help': 'Timing errors compare each recovered endpoint with that '
                            'simulated study’s own true endpoint. Median and 90th-percentile '
                            'absolute errors use only pairs with two finite times. Read '
                            'errors with the finite-pair and recovered-reaching counts; '
                            'two noncrossings do not imply zero error.',
                },
                'time': {
                    'title': f'{label}: recovered endpoint time versus study size',
                    'y_label': 'Recovered endpoint time (in vivo time)',
                    'primary_label': 'Recovered median time',
                    'secondary_label': 'Central 95% recovered time range',
                    'help': 'Recovered endpoint times use equally weighted simulated '
                            'studies with valid endpoint outcomes, including noncrossings '
                            'beyond the assessment horizon. Error bars show the empirical '
                            'central 95% range across studies, not a confidence interval '
                            'within one fit. A range needs two valid outcomes and both '
                            'bounds must be finite to draw it. Unresolved percentiles '
                            'remain explicit in the table and are never replaced by the '
                            'horizon. Times are in vivo coordinates, not durations since baseline.',
                },
            },
        })
    return measures


def _endpoint_values(endpoint):
    if endpoint is None:
        endpoint = {}
    recovered, paired = endpoint.get('recovered', {}), endpoint.get('paired', {})
    counts = {name: int(recovered.get(name, 0))
              for name in ('n_valid', 'n_reached', 'n_invalid', 'n_failed', 'n_beyond')}
    quantiles = recovered.get('quantiles', [])
    unavailable = {'value': None, 'status': 'unavailable'}
    lower, median, upper = quantiles if len(quantiles) == 3 else [unavailable] * 3
    if counts['n_valid'] < 2:
        lower = upper = unavailable
    time = {
        **counts, 'primary': _finite(median['value']), 'secondary': None,
        'lower': _finite(lower['value']), 'upper': _finite(upper['value']),
        'primary_status': median['status'], 'lower_status': lower['status'],
        'upper_status': upper['status'],
    }
    error = {
        **counts, 'primary': _finite(paired.get('median_absolute_error')),
        'secondary': _finite(paired.get('p90_absolute_error')),
        'n_pairs': int(paired.get('n_finite', 0)),
    }
    return {'error': error, 'time': time}


def build_performance_inspection(result, rows, endpoint_recovery, *, x_label):
    """Return finite, JSON-safe display values using the existing schedule IDs."""
    definitions = endpoint_definitions(result.settings)
    inspection = {'x_label': x_label, 'measures': _measures(definitions), 'schedules': []}
    comparison = result.comparison
    if comparison is None:
        return inspection
    endpoint_schedules = {schedule['id']: schedule
                          for schedule in endpoint_recovery.get('schedules', [])}
    adaptive = getattr(result, 'completed_observations', None) is not None
    for row in rows:
        name = row['name']
        summary, simulated = comparison.summaries[name], comparison.results[name]
        successful = np.asarray(simulated.fit_success, dtype=bool)
        finite_errors = successful & np.isfinite(simulated.integrated_rmse)
        measures = {'irmse': {'error': {
            'primary': _finite(summary.mean_irmse), 'secondary': _finite(summary.p90_irmse),
            'n_pairs': int(np.count_nonzero(finite_errors)),
        }}}
        endpoints = endpoint_schedules.get(row['id'], {}).get('endpoints', [])
        for index, _ in enumerate(definitions):
            measures[f'endpoint-{index}'] = _endpoint_values(
                endpoints[index] if index < len(endpoints) else None)
        inspection['schedules'].append({
            'id': row['id'], 'name': name, 'kind': row['kind'],
            'x': int(row['number_of_new_times'] if adaptive else row['number_of_times']),
            'n_simulations': int(row['n_simulations']), 'n_successful': int(row['n_successful']),
            'interval_count': int(row['interval_count']),
            'curve_interval_width': _finite(summary.mean_curve_interval_width),
            'measures': measures,
        })
    return inspection
