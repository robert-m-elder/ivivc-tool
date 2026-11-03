"""Empirical-scale reference weights for prospective and adaptive planning."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .adaptive import reweight_reference_ensemble
from .time_scaling import EmpiricalTimeScaleDistribution
from .types import (
    ExistingObservations,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
    ReferenceUpdateResult,
)


def _distribution_and_indices(reference: ReferenceTrajectoryEnsemble):
    if not isinstance(reference, ReferenceTrajectoryEnsemble):
        raise TypeError('reference must be a ReferenceTrajectoryEnsemble.')
    metadata = reference.metadata
    if metadata.get('reference_source') != 'empirical_time_scaling':
        raise ValueError('The reference must originate from reference_from_time_scaling().')
    try:
        distribution = EmpiricalTimeScaleDistribution(
            metadata['time_scale_ratios'], weights=metadata['time_scale_weights'],
            source_ids=metadata.get('time_scale_source_ids'),
            metadata=metadata.get('time_scale_metadata', {}),
        )
        indices = np.asarray(metadata['draw_ratio_indices'])
    except KeyError as exception:
        raise ValueError('The reference is missing empirical time-scale provenance.') from exception
    if (indices.ndim != 1 or len(indices) != reference.n_draws
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(distribution.ratios))):
        raise ValueError('draw_ratio_indices must identify one valid ratio record per reference draw.')
    weights = np.asarray(reference.draw_weights, dtype=float).copy()
    if (weights.shape != (reference.n_draws,) or np.any(~np.isfinite(weights))
            or np.any(weights < 0)):
        raise ValueError('Reference draw weights must be finite and nonnegative.')
    weights[~reference.valid_draws] = 0.0
    if not np.any(weights > 0):
        raise ValueError('The reference must retain positive draw weight.')
    if np.any((weights > 0) & (distribution.weights[indices] == 0)):
        raise ValueError('A draw cannot assign weight to a zero-weight original ratio record.')
    weights /= np.max(weights)
    weights /= np.sum(weights)
    return distribution, indices, weights


def _effective_size(weights) -> float:
    return float(1.0 / np.sum(np.asarray(weights, dtype=float) ** 2))


def time_scaling_weight_summary(reference: ReferenceTrajectoryEnsemble) -> dict:
    """Summarize original and current scale weights without changing the reference.

    Record marginals sum all parameter draws belonging to that input record.
    Unique-scale marginals additionally combine duplicate numerical ratios.
    Effective sizes are reported separately at draw, record, and unique-scale
    levels; none is a count of independent historical studies.

    This works both before and after reweighting. The original empirical
    weights remain in provenance, while current weights come from the complete
    reference draws. Zero-weight records remain visible in their input order.
    """
    distribution, indices, weights = _distribution_and_indices(reference)
    current = np.bincount(indices, weights=weights, minlength=len(distribution.ratios))
    unique, inverse = np.unique(distribution.ratios, return_inverse=True)
    prior_unique = np.bincount(inverse, weights=distribution.weights, minlength=len(unique))
    current_unique = np.bincount(inverse, weights=current, minlength=len(unique))
    return {
        'input_mode': ('adaptive-time-scaling' if reference.metadata.get('adaptive_reweighting')
                       else 'prospective-time-scaling'),
        'ratios': distribution.ratios.copy(),
        'source_ids': distribution.source_ids,
        'prior_weights': distribution.weights.copy(),
        'current_weights': current,
        'unique_ratios': unique,
        'prior_unique_weights': prior_unique,
        'current_unique_weights': current_unique,
        'draw_effective_sample_size': _effective_size(weights),
        'record_effective_sample_size': _effective_size(current),
        'unique_scale_effective_sample_size': _effective_size(current_unique),
        'record_count': len(distribution.ratios),
        'positive_prior_record_count': int(np.count_nonzero(distribution.weights)),
        'positive_current_record_count': int(np.count_nonzero(current)),
    }


def _row_rms(values: np.ndarray) -> np.ndarray:
    """Compute residual RMS without squaring unscaled, potentially large values."""
    scale = np.max(np.abs(values), axis=1)
    normalized = np.zeros_like(values)
    usable = np.isfinite(scale) & (scale > 0)
    np.divide(values, scale[:, None], out=normalized, where=usable[:, None])
    result = scale.copy()
    result[usable] = scale[usable] * np.sqrt(np.mean(normalized[usable] ** 2, axis=1))
    return result


def _weighted_rms(rms: np.ndarray, weights: np.ndarray) -> float:
    active = weights > 0
    values = rms[active]
    scale = float(np.max(values))
    if scale == 0 or not np.isfinite(scale):
        return scale
    return float(scale * np.sqrt(np.average((values / scale) ** 2, weights=weights[active])))


def _observation_agreement(prior, observations, observation_model, update) -> dict:
    active = np.flatnonzero(prior.valid_draws & (update.prior_weights > 0))
    predicted = prior.evaluate(observations.time, draw_indices=active)
    sd = observation_model.standard_deviation(predicted)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        residual = observations.response[None, :] - predicted
        standardized = residual / sd
    rmse = np.full(prior.n_draws, np.nan)
    standardized_rms = np.full(prior.n_draws, np.nan)
    rmse[active] = _row_rms(residual)
    standardized_rms[active] = _row_rms(standardized)
    return {
        'observation_count': observations.n_observations,
        'residual_scale': 'individual_replicate_observation_sd',
        'rmse_by_draw': rmse,
        'standardized_residual_rms_by_draw': standardized_rms,
        'best_standardized_residual_rms': float(np.min(standardized_rms[active])),
        'prior_weighted_rmse': _weighted_rms(rmse, update.prior_weights),
        'posterior_weighted_rmse': _weighted_rms(rmse, update.posterior_weights),
        'prior_weighted_standardized_residual_rms': _weighted_rms(standardized_rms, update.prior_weights),
        'posterior_weighted_standardized_residual_rms': _weighted_rms(standardized_rms, update.posterior_weights),
    }


def reweight_time_scaled_reference(
    original_prior: ReferenceTrajectoryEnsemble,
    cumulative_observations: ExistingObservations,
    observation_model: ObservationModel,
) -> ReferenceUpdateResult:
    """Condition an original empirical-scale prior on all completed observations.

    Reuses the existing replicate-level likelihood calculation. Only weights
    change; trajectories and scale support are not widened or refitted. Always
    supply the original prior and the cumulative dataset. An already-updated
    reference, including a generic reweighting result, is rejected to prevent
    accidental reuse of completed observations in a cumulative update.

    Weight summaries and continuous observation-agreement diagnostics are
    attached to both the result and its reference metadata. No adequacy cutoff
    or fallback is introduced. Use ``update.reference`` with the ordinary
    analysis-model fitting, timing, optimization, and simulation APIs, keeping
    the same completed observations fixed in constraints and simulation.
    """
    _distribution_and_indices(original_prior)
    if original_prior.metadata.get('adaptive_reweighting'):
        raise ValueError(
            'Use the original empirical time-scale prior with cumulative observations, '
            'not an already-updated reference.'
        )
    if not isinstance(cumulative_observations, ExistingObservations):
        raise TypeError('cumulative_observations must be ExistingObservations.')
    if not isinstance(observation_model, ObservationModel):
        raise TypeError('observation_model must be an ObservationModel.')
    update = reweight_reference_ensemble(original_prior, cumulative_observations, observation_model)
    diagnostics = {
        'planning_input_mode': 'adaptive-time-scaling',
        'time_scale_update': 'weights_only',
        'time_scaling_weight_summary': time_scaling_weight_summary(update.reference),
        'observation_agreement': _observation_agreement(
            original_prior, cumulative_observations, observation_model, update,
        ),
    }
    update.metadata.update(deepcopy(diagnostics))
    update.reference.metadata = {**update.reference.metadata, **deepcopy(diagnostics)}
    return update
