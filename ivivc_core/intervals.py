"""Pointwise fitted-parameter intervals shared by plots and predictions.

These are conditional intervals, not prediction intervals for future observations.
The ensemble samplers retain their separate whole-trajectory validity masks.
"""
from __future__ import annotations

import numpy as np

DEFAULT_INTERVAL_SAMPLES = 1000
DEFAULT_INTERVAL_SEED = 12345


def validate_interval_settings(confidence_level, n_samples):
    confidence_level = float(confidence_level)
    if not 0. < confidence_level < 1.:
        raise ValueError('confidence_level must be between zero and one.')
    if int(n_samples) != n_samples or n_samples < 2:
        raise ValueError('n_samples must be an integer of at least two.')
    return confidence_level, int(n_samples)


def parameter_interval_summary(draws, central, *, confidence_level=.95,
                               n_samples=DEFAULT_INTERVAL_SAMPLES,
                               random_state=DEFAULT_INTERVAL_SEED,
                               sampling_error=None):
    """Summarize without silently conditioning on successful draws at a point.

    All requested draws must be finite at a point for a bounded interval to be
    displayed there. Failure elsewhere does not change that point's interval.
    Raw finite-draw counts are retained even when the interval is unavailable.
    """
    from .uncertainty import summarize_draws

    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    central = np.asarray(central, dtype=float).reshape(-1)
    values = np.asarray(draws, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(central):
        raise ValueError('Draws must be a sample-by-point matrix matching the central values.')
    with np.errstate(over='ignore', invalid='ignore'):
        summary = summarize_draws(values, confidence_level=confidence_level)
    complete = (summary.valid_counts == n_samples) & (len(values) == n_samples)
    supported = complete & np.isfinite(summary.lower) & np.isfinite(summary.upper)
    supported &= np.isfinite(central)
    status = np.full(len(central), 'unavailable (nonfinite or missing Monte Carlo draws)', dtype=object)
    status[complete] = 'unavailable (nonfinite percentile bounds)'
    if sampling_error is not None:
        status[:] = 'unavailable (parameter sampling failed)'
    status[supported] = 'available'
    status[~np.isfinite(central)] = 'unavailable (point prediction unavailable)'
    return {
        'lower': np.where(supported, summary.lower, np.nan),
        'upper': np.where(supported, summary.upper, np.nan),
        'mean': np.where(supported, summary.mean, np.nan),
        'std': np.where(supported, summary.std, np.nan),
        'valid_counts': summary.valid_counts,
        'n_samples': n_samples,
        'interval_kind': 'parameter_uncertainty',
        'interval_confidence_level': confidence_level,
        'interval_status': status,
        'interval_valid_samples': summary.valid_counts,
        'interval_total_samples': len(values),
        'interval_requested_samples': n_samples,
        'interval_random_seed': int(random_state) if isinstance(random_state, (int, np.integer)) else None,
        'interval_sampling_error': sampling_error,
    }


def curve_parameter_intervals(model_function, x_values, parameters, covariance, *,
                              confidence_level=.95, n_samples=DEFAULT_INTERVAL_SAMPLES,
                              random_state=DEFAULT_INTERVAL_SEED):
    """Evaluate covariance-aware draws with no added residual/observation noise."""
    from .uncertainty import sample_parameter_sets, evaluate_curve_draws

    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    x_values = np.asarray(x_values, dtype=float)
    if x_values.ndim != 1:
        raise ValueError('x_values must be one-dimensional.')
    # The center is the fitted-parameter result, never the Monte Carlo mean.
    from .types import ParameterSamples
    nominal = ParameterSamples(tuple(parameters), np.asarray([list(parameters.values())], dtype=float))
    central = evaluate_curve_draws(model_function, x_values, nominal, require_finite=False).values[0]
    sampling_error = None
    try:
        samples = sample_parameter_sets(parameters, covariance, n_samples=n_samples,
                                        random_state=random_state)
        draws = evaluate_curve_draws(model_function, x_values, samples, require_finite=False).values
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        draws = np.empty((0, len(x_values)))
        sampling_error = str(exc)
    return parameter_interval_summary(draws, central, confidence_level=confidence_level,
                                      n_samples=n_samples, random_state=random_state,
                                      sampling_error=sampling_error)


def attach_parameter_interval(trajectory, interval, *, output_kind):
    """Attach authoritative bounds and presentation-independent diagnostics."""
    if output_kind not in ('time', 'value'):
        raise ValueError('output_kind must be time or value.')
    lower, upper = interval['lower'], interval['upper']
    setattr(trajectory, output_kind + '_lower', lower)
    setattr(trajectory, output_kind + '_upper', upper)
    trajectory.auxiliary.update({key: value for key, value in interval.items()
                                if key.startswith('interval_')})
    supported = np.isfinite(lower) & np.isfinite(upper)
    trajectory.auxiliary['interval_available'] = bool(np.any(supported))
    trajectory.auxiliary['interval_displayable'] = supported
    if output_kind == 'time':
        center = trajectory.in_vivo_time
        contains = supported & (lower <= center) & (center <= upper)
        # Bars require a containing interval; the renderer draws other finite
        # intervals as separate segments instead of moving their bounds.
        trajectory.time_error_lower = np.where(contains, center - lower, np.nan)
        trajectory.time_error_upper = np.where(contains, upper - center, np.nan)
        if not np.any(supported):
            trajectory.time_error_lower = trajectory.time_error_upper = None
    count = int(np.count_nonzero(~supported))
    if count:
        message = (f'{count} of {len(lower)} parameter-uncertainty intervals are unavailable. '
                   'See the prediction data table for pointwise reasons.')
        if interval.get('interval_sampling_error'):
            message += ' Parameter sampling failed: ' + interval['interval_sampling_error']
        trajectory.note = ' '.join(filter(None, (trajectory.note, message)))
    return trajectory
