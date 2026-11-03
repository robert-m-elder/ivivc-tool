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


def _profile_interval_metadata(fit, confidence_level):
    """Select a local coefficient-interval convention for the fitted scale."""
    from scipy import stats

    confidence_level = float(confidence_level)
    if not 0. < confidence_level < 1.:
        raise ValueError('confidence_level must be between zero and one.')
    probability = (1. + confidence_level) / 2.
    if fit.observation_uncertainty is not None:
        return {
            'confidence_level': confidence_level,
            'multiplier': float(stats.norm.ppf(probability)),
            'distribution': 'normal',
            'degrees_of_freedom': None,
            'covariance_scaling': 'absolute',
            'scale_source': 'supplied_observation_uncertainty',
            'method': 'normal_wald',
        }
    try:
        degrees_of_freedom = int(fit.residual_dof)
    except (TypeError, ValueError, OverflowError):
        degrees_of_freedom = 0
    multiplier = stats.t.ppf(probability, degrees_of_freedom) if degrees_of_freedom > 0 else np.nan
    use_student_t = bool(np.isfinite(multiplier))
    if not use_student_t:
        # Retain the exact historical 95% fallback used by app parameter tables.
        multiplier = 1.96 if confidence_level == .95 else stats.norm.ppf(probability)
    return {
        'confidence_level': confidence_level,
        'multiplier': float(multiplier),
        'distribution': 'student_t' if use_student_t else 'normal_fallback',
        'degrees_of_freedom': degrees_of_freedom,
        'covariance_scaling': 'residual_scaled',
        'scale_source': 'residuals',
        'method': 'student_t_wald' if use_student_t else 'normal_wald_fallback',
    }


def profile_parameter_intervals(fit, *, confidence_level=.95):
    """Summarize approximate local coefficient intervals from one profile fit.

    Supplied absolute SD/SEM uses a normal multiplier, conditional on those
    supplied error estimates. The historical residual-scaled path uses its
    residual-df t multiplier and normal fallback. Invalid covariance retains
    central estimates but makes uncertainty unavailable; no variance is clipped.
    Output remains numerical and contains no table or display formatting.
    """
    from scipy import stats

    metadata = _profile_interval_metadata(fit, confidence_level)
    names = tuple(fit.parameter_names)
    estimate = np.asarray(fit.parameter_vector, dtype=float).copy()
    standard_error = np.full(len(names), np.nan, dtype=float)
    covariance_status = 'unavailable (invalid covariance)'
    try:
        covariance = np.asarray(fit.covariance, dtype=float)
    except (TypeError, ValueError, OverflowError):
        covariance = np.empty((0, 0))
    if covariance.shape == (len(names), len(names)):
        if np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) >= 0):
            # SciPy's sampler checks PSD with scale-sensitive eigenvalue
            # tolerance and accepts singular covariance. Validate against that
            # same model without drawing or altering any random state.
            scale = float(np.max(np.abs(covariance))) if covariance.size else 0.
            symmetry_tolerance = 1.e6 * np.finfo(float).eps * scale
            symmetric = np.allclose(covariance, covariance.T, rtol=0., atol=symmetry_tolerance)
            if symmetric:
                try:
                    stats.multivariate_normal(
                        mean=np.zeros(len(names)), cov=covariance, allow_singular=True,
                    )
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    pass
                else:
                    standard_error = np.sqrt(np.diag(covariance))
                    covariance_status = 'available'
    with np.errstate(over='ignore', invalid='ignore'):
        half_width = metadata['multiplier'] * standard_error
        lower = estimate - half_width
        upper = estimate + half_width
    available = (
        np.isfinite(estimate) & np.isfinite(standard_error)
        & np.isfinite(lower) & np.isfinite(upper)
    )
    return {
        'parameter_names': names,
        'estimate': estimate,
        'standard_error': standard_error,
        'lower': np.where(available, lower, np.nan),
        'upper': np.where(available, upper, np.nan),
        'available': available,
        'covariance_status': covariance_status,
        **metadata,
    }


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


def profile_curve_intervals(fit, time_grid, *, confidence_level=.95,
                            n_samples=DEFAULT_INTERVAL_SAMPLES,
                            random_state=DEFAULT_INTERVAL_SEED):
    """Evaluate the existing full-covariance fitted-curve draws with provenance.

    Curve bounds remain pointwise multivariate-normal Monte Carlo percentiles,
    including on the residual-scaled path whose coefficient table uses t
    multipliers. No observation noise or normalization-reference variability is
    added, and existing pointwise failed-draw rules remain in force.
    """
    from .registry import get_model_spec

    spec = get_model_spec(fit.model_name)
    parameters = {name: fit.parameters[name] for name in fit.parameter_names}
    parameter_summary = profile_parameter_intervals(fit, confidence_level=confidence_level)
    if parameter_summary['covariance_status'] == 'available':
        interval = curve_parameter_intervals(
            spec.model_function, time_grid, parameters, fit.covariance,
            confidence_level=confidence_level, n_samples=n_samples,
            random_state=random_state,
        )
    else:
        from .types import ParameterSamples
        from .uncertainty import evaluate_curve_draws

        confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
        grid = np.asarray(time_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError('time_grid must be one-dimensional.')
        nominal = ParameterSamples(tuple(parameters), np.asarray([list(parameters.values())], dtype=float))
        central = evaluate_curve_draws(spec.model_function, grid, nominal, require_finite=False).values[0]
        interval = parameter_interval_summary(
            np.empty((0, len(grid))), central,
            confidence_level=confidence_level, n_samples=n_samples,
            random_state=random_state,
            sampling_error='The fitted parameter covariance is invalid for joint uncertainty.',
        )
    supplied = fit.observation_uncertainty is not None
    interval.update({
        'interval_covariance_scaling': 'absolute' if supplied else 'residual_scaled',
        'interval_scale_source': 'supplied_observation_uncertainty' if supplied else 'residuals',
        'interval_distribution': 'multivariate_normal',
        'interval_method': 'covariance_monte_carlo',
        'interval_covariance_status': parameter_summary['covariance_status'],
        'observation_noise_included': False,
    })
    return interval


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
