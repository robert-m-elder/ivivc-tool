"""Optional future-response variability, separate from fitted trajectories.

Errors and lookup times use analyzed coordinates. Calibration lookup is fixed at
the nominal predicted time; no variance model is fitted or sampled.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .intervals import parameter_interval_summary
from .observations import (
    ObservationUncertainty, ResolvedObservationUncertainty,
    resolve_observation_uncertainty,
)
from .prediction_inputs import prediction_uncertainty


def resolve_future_observation_uncertainty(spec, prediction, in_vivo_time, *,
                                         calibration_in_vivo=None):
    """Resolve supplied errors or reuse prediction/calibration precision.

    Calibration ties select the earlier time (then the first original row).
    Duplicate times require identical errors. Undefined nominal times get no
    reported lookup; their internal placeholder errors never support an interval.
    """
    if spec is None:
        return None, {}
    times = np.asarray(in_vivo_time, dtype=float)
    count = len(prediction.value)
    if times.ndim != 1 or len(times) != count:
        raise ValueError('Future-error times must match prediction response rows.')
    lookup = {}
    if isinstance(spec, str):
        if spec == 'prediction':
            uncertainty = prediction_uncertainty(prediction)
            if uncertainty is None:
                raise ValueError('Reusing prediction uncertainty requires supplied prediction SD/SEM.')
        elif spec == 'calibration':
            if calibration_in_vivo is None:
                raise ValueError('Calibration reuse requires the in vivo calibration time course.')
            source = prediction_uncertainty(calibration_in_vivo)
            if source is None:
                raise ValueError('Calibration reuse requires supplied in vivo SD/SEM.')
            source_times = np.asarray(calibration_in_vivo.time, dtype=float)
            if not len(source_times) or not np.isfinite(source_times).all():
                raise ValueError('Calibration-error lookup requires finite calibration times.')
            order = np.argsort(source_times, kind='stable')
            grid, errors = source_times[order], source.sigma[order]
            if np.any((np.diff(grid) == 0) & (np.diff(errors) != 0)):
                raise ValueError('Calibration reuse requires one SD/SEM per time; duplicate times have conflicting errors.')
            order = order[np.unique(grid, return_index=True)[1]]
            grid = source_times[order]
            finite = np.isfinite(times)
            query = np.where(finite, times, grid[0])
            right = np.clip(np.searchsorted(grid, query), 0, len(grid) - 1)
            left = np.maximum(right - 1, 0)
            with np.errstate(over='ignore', invalid='ignore'):
                nearest = np.where(abs(grid[right] - query) < abs(query - grid[left]), right, left)
            nearest[query <= grid[0]] = 0
            nearest[query >= grid[-1]] = len(grid) - 1
            indices = order[nearest]
            uncertainty = ResolvedObservationUncertainty(
                source.sigma[indices], response_kind=source.response_kind,
                replicate_counts=None if source.replicate_counts is None else source.replicate_counts[indices],
                source=source.source, response_units=source.response_units,
            )
            lookup = {
                'future_observation_lookup_rule': 'nearest_analyzed_time_with_endpoint_extension',
                'future_observation_calibration_indices': [int(i) if valid else None for i, valid in zip(indices, finite)],
                'future_observation_lookup_extrapolated': (finite & ((times < grid[0]) | (times > grid[-1]))).tolist(),
            }
        else:
            raise ValueError('Future uncertainty source must be prediction or calibration.')
        source_name = spec
    elif isinstance(spec, (ObservationUncertainty, ResolvedObservationUncertainty)):
        uncertainty = resolve_observation_uncertainty(spec, count)
        source_name = 'supplied'
    else:
        raise TypeError('Future uncertainty must be supplied/resolved SD/SEM, prediction, calibration, or None.')
    record = uncertainty.to_metadata()
    if source_name == 'calibration':
        for index in np.flatnonzero(~np.isfinite(times)):
            record['standard_uncertainty'][index] = None
            if record['replicate_counts'] is not None:
                record['replicate_counts'][index] = None
    return uncertainty, {
        'future_observation_variability_included': True,
        'future_observation_uncertainty_source': source_name,
        'future_observation_uncertainty': record,
        'future_observation_errors_independent': True,
        'future_observation_profile_fixed': True,
        **lookup,
    }


def _future_noise_rng(random_state):
    """Derive an independent stream without advancing the source RNG."""
    if isinstance(random_state, (int, np.integer)):
        entropy = int(random_state)
    elif isinstance(random_state, np.random.Generator):
        entropy = deepcopy(random_state).integers(0, 2**32, size=4, dtype=np.uint32)
    else:
        source = np.random if random_state is None else random_state
        if not isinstance(source, np.random.RandomState) and source is not np.random:
            raise TypeError('random_state must be an integer, Generator, RandomState, or None.')
        clone = np.random.RandomState()
        clone.set_state(source.get_state())
        entropy = clone.randint(0, 2**32, size=4, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(3,)))


def future_response_interval(value_draws, prediction, in_vivo_time, central_response, *,
                             future_observation_uncertainty, calibration_in_vivo=None,
                             confidence_level=.95, n_samples=1000, random_state=12345,
                             sampling_error=None, min_valid_fraction=1.0):
    """Add independent future error once to existing propagated response draws.

    Callers retain unsuccessful mapping entries as NaN. Their requested-draw
    denominator and conditional policy therefore also apply to the response.
    """
    uncertainty, metadata = resolve_future_observation_uncertainty(
        future_observation_uncertainty, prediction, in_vivo_time,
        calibration_in_vivo=calibration_in_vivo,
    )
    if uncertainty is None:
        return None
    values = np.asarray(value_draws, dtype=float)
    central = np.asarray(central_response, dtype=float)
    if central.ndim != 1 or len(central) != len(prediction.value):
        raise ValueError('Future-response centers must match prediction rows.')
    if values.ndim != 2 or values.shape[1] != len(central):
        raise ValueError('Future-response draws must be a sample-by-point matrix.')
    rng = _future_noise_rng(random_state)
    noise = rng.standard_normal((len(central), len(values))).T
    with np.errstate(over='ignore', invalid='ignore'):
        combined = values + noise * uncertainty.sigma
    supported = np.isfinite(in_vivo_time) & np.isfinite(central)
    combined[:, ~supported] = np.nan
    summary = parameter_interval_summary(
        combined, np.where(supported, central, np.nan), confidence_level=confidence_level,
        n_samples=n_samples, random_state=random_state, sampling_error=sampling_error,
        min_valid_fraction=min_valid_fraction,
    )
    numerical_failure = np.any(np.isfinite(values) & supported & ~np.isfinite(combined), axis=0)
    for key in ('lower', 'upper', 'mean', 'std'):
        summary[key][numerical_failure] = np.nan
    summary['interval_status'][numerical_failure] = 'unavailable (nonfinite future-response draws)'
    available = np.isfinite(summary['lower']) & np.isfinite(summary['upper'])
    summary.update(metadata)
    summary.update({
        'interval_kind': 'future_observation_prediction',
        'interval_method': 'propagation_and_future_measurement_monte_carlo',
        'interval_min_valid_fraction': float(min_valid_fraction),
        'interval_conditional': available & (summary['valid_counts'] < n_samples),
    })
    return summary


def attach_future_response_interval(trajectory, summary):
    """Attach future-response bounds while retaining propagated-only bounds."""
    if summary is None:
        return trajectory
    metadata = trajectory.auxiliary
    ratio = trajectory.source_method == 'value_ratio'
    primary_lower = trajectory.value_lower if ratio else trajectory.time_lower
    primary_upper = trajectory.value_upper if ratio else trajectory.time_upper
    count = len(trajectory.in_vivo_value)
    supported = (np.zeros(count, dtype=bool) if primary_lower is None or primary_upper is None
                 else np.isfinite(primary_lower) & np.isfinite(primary_upper))
    previous_status = metadata.get('value_interval_status', metadata.get('interval_status') if ratio else None)
    metadata.update({
        'propagated_value_lower': None if trajectory.value_lower is None else trajectory.value_lower.copy(),
        'propagated_value_upper': None if trajectory.value_upper is None else trajectory.value_upper.copy(),
        'propagated_value_interval_status': None if previous_status is None else previous_status.copy(),
        'propagated_value_interval_kind': metadata.get('value_interval_kind', metadata.get('interval_kind') if ratio else 'fixed_response'),
    })
    lower, upper = summary['lower'].copy(), summary['upper'].copy()
    status = summary['interval_status'].copy()
    lower[~supported], upper[~supported] = np.nan, np.nan
    primary_status = metadata.get('interval_status', np.full(count, 'unavailable (propagation unavailable)', dtype=object))
    status[~supported] = primary_status[~supported]
    conditional = summary['interval_conditional'] & supported
    future_metadata = {key: value for key, value in summary.items() if key.startswith('future_')}
    interval = {key: value for key, value in summary.items() if key.startswith('interval_')}
    interval.update(interval_status=status, interval_conditional=conditional)
    metadata.update(future_metadata)
    metadata.update({'value_' + key: value for key, value in interval.items()})
    trajectory.value_lower, trajectory.value_upper = lower, upper
    if ratio:
        metadata.update(interval)
        metadata['interval_available'] = bool(np.isfinite(lower).any())
        metadata['interval_displayable'] = np.isfinite(lower) & np.isfinite(upper)
    unavailable = supported & ~(np.isfinite(lower) & np.isfinite(upper))
    if unavailable.any():
        note = f'{int(unavailable.sum())} future-response prediction interval(s) are unavailable; see pointwise status.'
        trajectory.note = ' '.join(filter(None, (trajectory.note, note)))
    return trajectory
