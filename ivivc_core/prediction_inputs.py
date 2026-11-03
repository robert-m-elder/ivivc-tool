"""Independent response uncertainty at the original prediction-input rows.

These errors describe measured input values, not future in vivo observations.
Times and normalization constants are fixed. Calibration measurements must be
independent of the prediction measurements; shared errors are not modeled.
"""
from __future__ import annotations

import numpy as np

from .observations import ResolvedObservationUncertainty


def prediction_uncertainty(prediction) -> ResolvedObservationUncertainty | None:
    """Validate optional resolved SD/SEM without changing legacy fixed inputs."""
    uncertainty = getattr(prediction, 'observation_uncertainty', None)
    if uncertainty is None:
        return None
    if not isinstance(uncertainty, ResolvedObservationUncertainty):
        raise TypeError('Prediction uncertainty must be resolved; use prepare_time_course.')
    if uncertainty.n_observations != len(prediction.value):
        raise ValueError('Prediction uncertainty must align with its response rows.')
    return uncertainty


def prediction_input_metadata(prediction, *, propagated=True):
    """Describe supplied input precision without adding a second noise layer."""
    uncertainty = prediction_uncertainty(prediction)
    if uncertainty is None:
        return {}
    return {
        'prediction_input_uncertainty': uncertainty.to_metadata(),
        'prediction_inputs_fixed': False,
        'prediction_times_fixed': True,
        'prediction_input_distribution': 'normal',
        'prediction_input_errors_independent': True,
        'prediction_calibration_independence_assumed': True,
        'prediction_input_uncertainty_propagated': bool(propagated),
        'future_observation_variability_included': False,
    }


def sample_prediction_responses(prediction, n_samples, random_state=None):
    """Draw each reported response once per trajectory, conditional on SD/SEM.

    Call after calibration parameter sampling. Integer seeds use a separate
    child stream, leaving the existing calibration draws unchanged. Generators,
    RandomState objects and the global RNG are consumed sequentially. Drawing
    columns in order keeps earlier row draws unchanged when later rows are
    appended. No truncation, clipping or retries are applied to response draws.
    """
    uncertainty = prediction_uncertainty(prediction)
    count = int(n_samples)
    if count < 1:
        raise ValueError('n_samples must be at least one.')
    values = np.asarray(prediction.value, dtype=float)
    if uncertainty is None:
        return np.broadcast_to(values, (count, len(values))).copy()
    if isinstance(random_state, (int, np.integer)):
        child = np.random.SeedSequence(int(random_state)).spawn(3)[2]
        rng = np.random.default_rng(child)
    else:
        rng = np.random if random_state is None else random_state
    standard = rng.standard_normal((len(values), count)).T
    with np.errstate(over='ignore', invalid='ignore'):
        return values + standard * uncertainty.sigma


def attach_input_response_interval(trajectory, prediction, *, confidence_level=.95):
    """Attach input-only normal bounds on the unchanged response axis.

    Time-constant and inverse predictions retain the measured response as their
    response coordinate. Its marginal normal interval is available analytically;
    the corresponding time bounds use joint Monte Carlo propagation separately.
    """
    uncertainty = prediction_uncertainty(prediction)
    if uncertainty is None:
        return trajectory
    from scipy.stats import norm

    confidence_level = float(confidence_level)
    if not 0. < confidence_level < 1.:
        raise ValueError('confidence_level must be between zero and one.')
    multiplier = norm.ppf((1. + confidence_level) / 2.)
    with np.errstate(over='ignore', invalid='ignore'):
        lower = trajectory.in_vivo_value - multiplier * uncertainty.sigma
        upper = trajectory.in_vivo_value + multiplier * uncertainty.sigma
    point_available = np.isfinite(trajectory.in_vivo_time) & np.isfinite(trajectory.in_vivo_value)
    available = point_available & np.isfinite(lower) & np.isfinite(upper)
    status = np.full(len(lower), 'unavailable (nonfinite interval bounds)', dtype=object)
    status[~point_available] = 'unavailable (point prediction unavailable)'
    status[available] = 'available'
    trajectory.value_lower = np.where(available, lower, np.nan)
    trajectory.value_upper = np.where(available, upper, np.nan)
    trajectory.auxiliary.update(prediction_input_metadata(
        prediction, propagated=bool(np.any(point_available)),
    ))
    trajectory.auxiliary.update({
        'value_interval_kind': 'prediction_input_uncertainty',
        'value_interval_method': 'normal',
        'value_interval_confidence_level': confidence_level,
        'value_interval_status': status,
    })
    return trajectory
