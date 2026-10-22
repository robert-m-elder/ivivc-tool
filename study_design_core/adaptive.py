"""Adaptive study-design helpers.

This module keeps already-completed observations separate from future
simulated measurements and provides likelihood reweighting of complete
reference trajectories by those completed observations.
"""

from __future__ import annotations

import numpy as np

from .types import (
    ExistingObservations,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
    ReferenceUpdateResult,
)


def reweight_reference_ensemble(
    reference: ReferenceTrajectoryEnsemble,
    existing_observations: ExistingObservations,
    observation_model: ObservationModel,
) -> ReferenceUpdateResult:
    """Update trajectory weights using completed replicate-level observations.

    The trajectory shapes themselves are not changed.  Each complete curve is
    assigned a likelihood under the completed observations, multiplied by its
    prior weight, and then normalized.  This permits the same reference
    trajectory ensemble to be used before and after target-study observations
    become available.
    """
    lower, upper = reference.supported_time_range
    if (
        np.min(existing_observations.time) < lower - 1e-12
        or np.max(existing_observations.time) > upper + 1e-12
    ):
        raise ValueError(
            'Existing observations extend beyond the supported reference domain.'
        )

    valid = reference.valid_draws & (reference.draw_weights > 0)
    valid_indices = np.flatnonzero(valid)
    predicted = reference.evaluate(
        existing_observations.time,
        draw_indices=valid_indices,
    )
    sd = observation_model.standard_deviation(predicted)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(
            'Observation-model SD must be finite and positive for adaptive reweighting.'
        )

    residual = existing_observations.response[None, :] - predicted
    log_likelihood_valid = np.sum(
        -0.5 * (residual / sd) ** 2
        - np.log(sd)
        - 0.5 * np.log(2.0 * np.pi),
        axis=1,
    )

    prior = reference.draw_weights.copy()
    full_log_likelihood = np.full(reference.n_draws, -np.inf, dtype=float)
    full_log_likelihood[valid_indices] = log_likelihood_valid

    log_prior = np.log(prior[valid_indices])
    log_unnormalized = log_prior + log_likelihood_valid
    maximum = float(np.max(log_unnormalized))
    scaled = np.exp(log_unnormalized - maximum)
    denominator = float(np.sum(scaled))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            'Completed observations are incompatible with all positive-weight reference trajectories.'
        )
    posterior_valid = scaled / denominator
    posterior = np.zeros(reference.n_draws, dtype=float)
    posterior[valid_indices] = posterior_valid

    effective_sample_size = float(1.0 / np.sum(posterior_valid**2))
    log_normalizing_constant = maximum + float(np.log(denominator))
    metadata = dict(reference.metadata)
    metadata.update({
        'adaptive_reweighting': True,
        'existing_observation_count': existing_observations.n_observations,
        'existing_time_point_count': existing_observations.n_time_points,
        'latest_existing_time': existing_observations.latest_time,
        'effective_sample_size': effective_sample_size,
        'effective_sample_size_fraction': effective_sample_size / len(valid_indices),
        'observation_model_kind': observation_model.kind,
        'observation_model_value': observation_model.value,
    })

    updated = ReferenceTrajectoryEnsemble(
        time_grid=reference.time_grid,
        value_draws=reference.value_draws,
        valid_draws=reference.valid_draws,
        draw_weights=posterior,
        central_curve=None,
        supported_time_range=reference.supported_time_range,
        interpolation=reference.interpolation,
        metadata=metadata,
    )
    return ReferenceUpdateResult(
        reference=updated,
        prior_weights=prior,
        posterior_weights=posterior,
        log_likelihood=full_log_likelihood,
        effective_sample_size=effective_sample_size,
        log_normalizing_constant=log_normalizing_constant,
        metadata={
            'effective_sample_size_fraction': effective_sample_size / len(valid_indices),
            'positive_prior_draws': int(len(valid_indices)),
        },
    )
