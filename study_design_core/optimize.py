"""Discrete information-based optimization of degradation sampling times."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ivivc_core import get_model_spec

from .information import (
    finite_difference_sensitivities,
    score_information_matrices,
    trapezoidal_integration_weights,
)
from .types import (
    CoordinateExchangeResult,
    DesignConstraints,
    DesignInformationScore,
    DesignOptimizationResult,
    DesignStep,
    ObservationModel,
    ReferenceModelEnsemble,
)


@dataclass
class _InformationContext:
    reference_model: ReferenceModelEnsemble
    observation_model: ObservationModel
    constraints: DesignConstraints
    all_times: np.ndarray
    evaluation_times: np.ndarray
    integration_weights: np.ndarray
    information_contributions: np.ndarray
    evaluation_sensitivities: np.ndarray
    rank_tolerance: float | None

    def score(self, times) -> DesignInformationScore:
        times = np.unique(np.asarray(times, dtype=float))
        indices = []
        for value in times:
            matches = np.flatnonzero(np.isclose(self.all_times, value, rtol=0.0, atol=1e-12))
            if len(matches) != 1:
                raise ValueError(f'Design time {value:g} is not available in this optimization context.')
            indices.append(int(matches[0]))
        matrices = np.sum(self.information_contributions[:, indices], axis=1)
        return score_information_matrices(
            matrices,
            self.evaluation_sensitivities,
            self.integration_weights,
            valid_draws=self.reference_model.valid_draws,
            draw_weights=self.reference_model.draw_weights,
            rank_tolerance=self.rank_tolerance,
        )


def _evaluate_model(model_name: str, times: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    spec = get_model_spec(model_name)
    with np.errstate(all='ignore'):
        values = np.asarray(spec.model_function(times, *parameters), dtype=float)
    if values.shape != times.shape or np.any(~np.isfinite(values)):
        raise ValueError('Model evaluation returned invalid values during design optimization.')
    return values


def _prepare_evaluation_grid(
    reference_model: ReferenceModelEnsemble,
    evaluation_times,
    evaluation_weights,
) -> tuple[np.ndarray, np.ndarray]:
    if evaluation_times is None:
        times = reference_model.time_grid.copy()
    else:
        times = np.unique(np.asarray(evaluation_times, dtype=float))
    if times.ndim != 1 or len(times) < 2:
        raise ValueError('evaluation_times must contain at least two distinct values.')
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('evaluation_times must be finite and strictly increasing.')
    lower, upper = reference_model.supported_time_range
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    if times[0] < lower - tolerance or times[-1] > upper + tolerance:
        raise ValueError('evaluation_times extend beyond the supported reference domain.')
    weights = trapezoidal_integration_weights(times, point_weights=evaluation_weights)
    return times, weights


def _build_information_context(
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> _InformationContext:
    all_times = constraints.all_available_times
    lower, upper = reference_model.supported_time_range
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    if all_times[0] < lower - tolerance or all_times[-1] > upper + tolerance:
        raise ValueError('Candidate or fixed times extend beyond the supported reference domain.')

    evaluation_times, integration_weights = _prepare_evaluation_grid(
        reference_model,
        evaluation_times,
        evaluation_weights,
    )
    draws = reference_model.n_draws
    parameters = reference_model.n_parameters
    contributions = np.zeros(
        (draws, len(all_times), parameters, parameters),
        dtype=float,
    )
    evaluation_sensitivities = np.zeros(
        (draws, len(evaluation_times), parameters),
        dtype=float,
    )

    replicates = float(constraints.replicates_per_time)
    for draw_index in np.flatnonzero(reference_model.valid_draws):
        theta = reference_model.parameter_draws[draw_index]
        expected = _evaluate_model(reference_model.model_name, all_times, theta)
        variance = observation_model.standard_deviation(expected) ** 2 / replicates
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError(
                'Information optimization requires strictly positive observation variance '
                'at every available time.'
            )
        sensitivities = finite_difference_sensitivities(
            reference_model.model_name,
            all_times,
            theta,
            relative_step=relative_step,
            absolute_step=absolute_step,
        )
        for time_index, (gradient, var) in enumerate(zip(sensitivities, variance)):
            contributions[draw_index, time_index] = np.outer(gradient, gradient) / var
        evaluation_sensitivities[draw_index] = finite_difference_sensitivities(
            reference_model.model_name,
            evaluation_times,
            theta,
            relative_step=relative_step,
            absolute_step=absolute_step,
        )

    return _InformationContext(
        reference_model=reference_model,
        observation_model=observation_model,
        constraints=constraints,
        all_times=all_times,
        evaluation_times=evaluation_times,
        integration_weights=integration_weights,
        information_contributions=contributions,
        evaluation_sensitivities=evaluation_sensitivities,
        rank_tolerance=rank_tolerance,
    )


def _score_is_better(
    candidate: DesignInformationScore,
    incumbent: DesignInformationScore | None,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if incumbent is None:
        return True
    if candidate.full_rank_fraction > incumbent.full_rank_fraction + tolerance:
        return True
    if candidate.full_rank_fraction < incumbent.full_rank_fraction - tolerance:
        return False
    if candidate.min_rank != incumbent.min_rank:
        return candidate.min_rank > incumbent.min_rank
    if candidate.mean_rank > incumbent.mean_rank + tolerance:
        return True
    if candidate.mean_rank < incumbent.mean_rank - tolerance:
        return False

    if candidate.is_full_rank and incumbent.is_full_rank:
        if candidate.criterion < incumbent.criterion - tolerance:
            return True
        if candidate.criterion > incumbent.criterion + tolerance:
            return False
    else:
        candidate_logdet = candidate.mean_log_pseudodeterminant
        incumbent_logdet = incumbent.mean_log_pseudodeterminant
        if candidate_logdet > incumbent_logdet + tolerance:
            return True
        if candidate_logdet < incumbent_logdet - tolerance:
            return False

    return False


def evaluate_i_optimal_design(
    design_times,
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> DesignInformationScore:
    """Evaluate one user-specified design using the optimizer's common context."""
    design = constraints.validate_design(design_times)
    context = _build_information_context(
        reference_model,
        observation_model,
        constraints,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        relative_step=relative_step,
        absolute_step=absolute_step,
        rank_tolerance=rank_tolerance,
    )
    return context.score(design)


def _greedy_designs_from_context(context: _InformationContext) -> tuple[DesignStep, ...]:
    constraints = context.constraints
    selected = constraints.fixed_times.copy()
    steps: list[DesignStep] = []

    if len(selected) >= constraints.min_number_of_times:
        steps.append(
            DesignStep(
                times=selected,
                score=context.score(selected),
                method='greedy_nested',
                added_time=None,
            )
        )

    while len(selected) < constraints.max_number_of_times:
        available = constraints.available_candidates(selected)
        if len(available) == 0:
            raise ValueError(
                'The minimum-spacing and fixed-time constraints do not permit '
                'construction of the requested maximum design size.'
            )

        best_time = None
        best_score = None
        for candidate in available:
            trial = np.union1d(selected, [candidate])
            score = context.score(trial)
            if _score_is_better(score, best_score):
                best_time = float(candidate)
                best_score = score
        if best_time is None or best_score is None:
            raise RuntimeError('No feasible candidate time could be selected.')

        selected = np.union1d(selected, [best_time])
        if len(selected) >= constraints.min_number_of_times:
            steps.append(
                DesignStep(
                    times=selected,
                    score=best_score,
                    method='greedy_nested',
                    added_time=best_time,
                )
            )

    return tuple(steps)


def greedy_i_optimal_designs(
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> tuple[DesignStep, ...]:
    """Construct a nested, rank-aware sequence of uncertainty-averaged designs.

    Before the information matrix is full rank, candidates are selected by
    robust rank and pseudo-determinant diagnostics.  Once all positive-weight
    parameter draws are full rank, selection minimizes the ensemble-averaged
    I-optimality criterion.
    """
    context = _build_information_context(
        reference_model,
        observation_model,
        constraints,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        relative_step=relative_step,
        absolute_step=absolute_step,
        rank_tolerance=rank_tolerance,
    )
    return _greedy_designs_from_context(context)


def _coordinate_exchange_from_context(
    initial_times,
    context: _InformationContext,
    *,
    max_iterations: int,
) -> CoordinateExchangeResult:
    constraints = context.constraints
    current = constraints.validate_design(initial_times)
    initial = current.copy()
    current_score = context.score(current)
    exchanges: list[tuple[float, float]] = []
    converged = False

    for _ in range(max_iterations):
        best_times = current
        best_score = current_score
        best_exchange = None

        replaceable = [
            value
            for value in current
            if not np.any(np.isclose(constraints.fixed_times, value, rtol=0.0, atol=1e-12))
        ]
        for old_time in replaceable:
            retained = current[~np.isclose(current, old_time, rtol=0.0, atol=1e-12)]
            for new_time in constraints.candidate_times:
                if np.any(np.isclose(current, new_time, rtol=0.0, atol=1e-12)):
                    continue
                trial = np.union1d(retained, [new_time])
                try:
                    trial = constraints.validate_design(trial)
                except ValueError:
                    continue
                score = context.score(trial)
                if _score_is_better(score, best_score):
                    best_times = trial
                    best_score = score
                    best_exchange = (float(old_time), float(new_time))

        if best_exchange is None:
            converged = True
            break
        current = best_times
        current_score = best_score
        exchanges.append(best_exchange)

    return CoordinateExchangeResult(
        initial_times=initial,
        times=current,
        score=current_score,
        exchanges=exchanges,
        converged=converged,
    )


def coordinate_exchange_i_optimal_design(
    initial_times,
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
    max_iterations: int = 25,
) -> CoordinateExchangeResult:
    """Refine a fixed-size design by exchanging one nonfixed time at a time."""
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError('max_iterations must be at least one.')
    context = _build_information_context(
        reference_model,
        observation_model,
        constraints,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        relative_step=relative_step,
        absolute_step=absolute_step,
        rank_tolerance=rank_tolerance,
    )
    return _coordinate_exchange_from_context(
        initial_times,
        context,
        max_iterations=max_iterations,
    )


def optimize_i_optimal_designs(
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
    refine: bool = True,
    max_exchange_iterations: int = 25,
) -> DesignOptimizationResult:
    """Return nested greedy designs and optional non-nested refinements."""
    context = _build_information_context(
        reference_model,
        observation_model,
        constraints,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        relative_step=relative_step,
        absolute_step=absolute_step,
        rank_tolerance=rank_tolerance,
    )
    nested = _greedy_designs_from_context(context)
    refined_steps: list[DesignStep] = []
    if refine:
        max_exchange_iterations = int(max_exchange_iterations)
        if max_exchange_iterations < 1:
            raise ValueError('max_exchange_iterations must be at least one.')
        for step in nested:
            result = _coordinate_exchange_from_context(
                step.times,
                context,
                max_iterations=max_exchange_iterations,
            )
            refined_steps.append(
                DesignStep(
                    times=result.times,
                    score=result.score,
                    method='coordinate_exchange',
                    added_time=None,
                    n_exchanges=len(result.exchanges),
                    metadata={
                        'initial_nested_times': step.times.copy(),
                        'exchanges': result.exchanges,
                        'converged': result.converged,
                    },
                )
            )

    metadata: dict[str, object] = {
        'criterion': 'uncertainty_averaged_i_optimality',
        'rank_aware_initialization': True,
        'replicates_per_time': constraints.replicates_per_time,
        'observation_model_kind': observation_model.kind,
        'observation_model_value': observation_model.value,
        'minimum_spacing': constraints.minimum_spacing,
        'fixed_times': constraints.fixed_times.copy(),
        'candidate_time_count': len(constraints.candidate_times),
        'reference_parameter_draw_count': reference_model.n_valid_draws,
    }
    return DesignOptimizationResult(
        model_name=reference_model.model_name,
        nested_steps=nested,
        refined_steps=refined_steps,
        evaluation_times=context.evaluation_times,
        evaluation_weights=context.integration_weights,
        metadata=metadata,
    )
