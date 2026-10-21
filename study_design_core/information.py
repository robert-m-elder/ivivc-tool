"""Information-based calculations for degradation time-point design.

The functions in this module are independent of presentation code.  They use
finite-difference model sensitivities so every model registered by the IVIVC app
can participate without adding model-specific derivative implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np

from ivivc_core import get_model_spec

from .fitting import fit_model_once
from .types import (
    DesignInformationScore,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
)


def _coerce_parameter_vector(
    model_name: str,
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    spec = get_model_spec(model_name)
    names = spec.parameter_names
    if isinstance(parameters, Mapping):
        missing = [name for name in names if name not in parameters]
        if missing:
            raise ValueError('Parameter mapping is missing: ' + ', '.join(missing))
        vector = np.asarray([parameters[name] for name in names], dtype=float)
    else:
        vector = np.asarray(parameters, dtype=float)
    if vector.ndim != 1 or len(vector) != len(names):
        raise ValueError('Parameters must contain one value per model parameter.')
    if np.any(~np.isfinite(vector)):
        raise ValueError('Parameters must be finite.')
    return vector.copy(), names


def _parameter_bounds(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_model_spec(model_name)
    bounds = spec.fit_kwargs.get('bounds')
    count = len(spec.parameter_names)
    if bounds is None:
        return (
            np.full(count, -np.inf, dtype=float),
            np.full(count, np.inf, dtype=float),
        )
    if len(bounds) != 2:
        raise ValueError('Model bounds must contain lower and upper values.')
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (count,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (count,)).copy()
    return lower, upper


def _evaluate_model(model_name: str, times: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    spec = get_model_spec(model_name)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', category=RuntimeWarning)
        values = np.asarray(spec.model_function(times, *parameters), dtype=float)
    if values.shape != times.shape or np.any(~np.isfinite(values)):
        raise ValueError('Model evaluation returned nonfinite or incorrectly shaped values.')
    return values


def finite_difference_sensitivities(
    model_name: str,
    times,
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    max_step_halvings: int = 8,
) -> np.ndarray:
    """Return dy/dtheta for one model at the requested times.

    Central differences are used where both perturbations are feasible.  Near
    model bounds, or when one perturbation produces an invalid model value, the
    function falls back to a one-sided difference.  The perturbation is reduced
    repeatedly before a parameter is declared numerically non-differentiable.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError('Sensitivity times must be a nonempty one-dimensional array.')
    if np.any(~np.isfinite(times)):
        raise ValueError('Sensitivity times must be finite.')
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError('relative_step must be finite and positive.')
    if not np.isfinite(absolute_step) or absolute_step <= 0:
        raise ValueError('absolute_step must be finite and positive.')
    max_step_halvings = int(max_step_halvings)
    if max_step_halvings < 0:
        raise ValueError('max_step_halvings must be nonnegative.')

    theta, names = _coerce_parameter_vector(model_name, parameters)
    lower, upper = _parameter_bounds(model_name)
    base = _evaluate_model(model_name, times, theta)
    sensitivities = np.empty((len(times), len(names)), dtype=float)

    for index, value in enumerate(theta):
        initial_step = max(abs(value) * relative_step, absolute_step)
        derivative = None
        for halving in range(max_step_halvings + 1):
            step = initial_step / (2 ** halving)
            plus_step = min(step, upper[index] - value)
            minus_step = min(step, value - lower[index])
            plus_values = None
            minus_values = None

            if plus_step > 0 and np.isfinite(plus_step):
                plus = theta.copy()
                plus[index] += plus_step
                try:
                    plus_values = _evaluate_model(model_name, times, plus)
                except Exception:
                    plus_values = None
            if minus_step > 0 and np.isfinite(minus_step):
                minus = theta.copy()
                minus[index] -= minus_step
                try:
                    minus_values = _evaluate_model(model_name, times, minus)
                except Exception:
                    minus_values = None

            if plus_values is not None and minus_values is not None:
                derivative = (plus_values - minus_values) / (plus_step + minus_step)
            elif plus_values is not None:
                derivative = (plus_values - base) / plus_step
            elif minus_values is not None:
                derivative = (base - minus_values) / minus_step

            if derivative is not None and np.all(np.isfinite(derivative)):
                break
            derivative = None

        if derivative is None:
            raise ValueError(
                f'Could not calculate a finite sensitivity for parameter {names[index]!r}.'
            )
        sensitivities[:, index] = derivative

    return sensitivities


def _coerce_replicates(replicates_per_time, count: int) -> np.ndarray:
    if replicates_per_time is None:
        replicates = np.ones(count, dtype=float)
    else:
        replicates = np.asarray(replicates_per_time, dtype=float)
        if replicates.ndim == 0:
            replicates = np.full(count, float(replicates), dtype=float)
        else:
            replicates = np.broadcast_to(replicates, (count,)).astype(float, copy=True)
    if np.any(~np.isfinite(replicates)) or np.any(replicates <= 0):
        raise ValueError('replicates_per_time must be finite and positive.')
    return replicates


def information_matrix(
    model_name: str,
    design_times,
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
    observation_model: ObservationModel,
    *,
    replicates_per_time=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
) -> np.ndarray:
    """Calculate the local Fisher-information approximation for one design."""
    design_times = np.asarray(design_times, dtype=float)
    if design_times.ndim != 1 or len(design_times) == 0:
        raise ValueError('design_times must be a nonempty one-dimensional array.')
    theta, _ = _coerce_parameter_vector(model_name, parameters)
    expected = _evaluate_model(model_name, design_times, theta)
    sensitivities = finite_difference_sensitivities(
        model_name,
        design_times,
        theta,
        relative_step=relative_step,
        absolute_step=absolute_step,
    )
    replicates = _coerce_replicates(
        observation_model.replicates_per_time
        if replicates_per_time is None
        else replicates_per_time,
        len(design_times),
    )
    variance = observation_model.standard_deviation(expected) ** 2 / replicates
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError(
            'Information calculations require a strictly positive observation variance.'
        )
    weighted = sensitivities / variance[:, None]
    matrix = sensitivities.T @ weighted
    return 0.5 * (matrix + matrix.T)


def trapezoidal_integration_weights(times, *, point_weights=None) -> np.ndarray:
    """Return normalized trapezoidal weights on a strictly increasing grid."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError('Integration times must contain at least two points.')
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Integration times must be finite and strictly increasing.')

    weights = np.empty(len(times), dtype=float)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    if len(times) > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])

    if point_weights is not None:
        emphasis = np.asarray(point_weights, dtype=float)
        if emphasis.shape != times.shape:
            raise ValueError('point_weights must match the integration time grid.')
        if np.any(~np.isfinite(emphasis)) or np.any(emphasis < 0):
            raise ValueError('point_weights must be finite and nonnegative.')
        weights *= emphasis

    total = float(np.sum(weights))
    if not total > 0:
        raise ValueError('Integration weights must have positive total weight.')
    return weights / total


def _matrix_rank_and_log_pseudodeterminant(
    matrix: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> tuple[int, float]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0
    tolerance = (
        float(rank_tolerance)
        if rank_tolerance is not None
        else max(matrix.shape) * np.finfo(float).eps * max(1.0, maximum)
    )
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    log_pseudodeterminant = (
        float(np.sum(np.log(eigenvalues[positive])))
        if rank
        else float('-inf')
    )
    return rank, log_pseudodeterminant


def _i_criterion_from_matrix(
    matrix: np.ndarray,
    evaluation_sensitivities: np.ndarray,
    integration_weights: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> tuple[float, int, float]:
    rank, log_pseudodeterminant = _matrix_rank_and_log_pseudodeterminant(
        matrix,
        rank_tolerance=rank_tolerance,
    )
    parameter_count = matrix.shape[0]
    if rank < parameter_count:
        return float('inf'), rank, log_pseudodeterminant
    inverse = np.linalg.pinv(matrix, hermitian=True)
    prediction_variance = np.einsum(
        'ij,jk,ik->i',
        evaluation_sensitivities,
        inverse,
        evaluation_sensitivities,
    )
    prediction_variance = np.maximum(prediction_variance, 0.0)
    criterion = float(np.sum(integration_weights * prediction_variance))
    return criterion, rank, log_pseudodeterminant


def local_i_optimality(
    model_name: str,
    design_times,
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
    observation_model: ObservationModel,
    *,
    evaluation_times,
    evaluation_weights=None,
    replicates_per_time=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> float:
    """Return integrated fitted-mean variance for one parameter vector."""
    theta, _ = _coerce_parameter_vector(model_name, parameters)
    evaluation_times = np.asarray(evaluation_times, dtype=float)
    weights = trapezoidal_integration_weights(
        evaluation_times,
        point_weights=evaluation_weights,
    )
    matrix = information_matrix(
        model_name,
        design_times,
        theta,
        observation_model,
        replicates_per_time=replicates_per_time,
        relative_step=relative_step,
        absolute_step=absolute_step,
    )
    evaluation_sensitivities = finite_difference_sensitivities(
        model_name,
        evaluation_times,
        theta,
        relative_step=relative_step,
        absolute_step=absolute_step,
    )
    criterion, _, _ = _i_criterion_from_matrix(
        matrix,
        evaluation_sensitivities,
        weights,
        rank_tolerance=rank_tolerance,
    )
    return criterion


def score_information_matrices(
    matrices: np.ndarray,
    evaluation_sensitivities: np.ndarray,
    integration_weights: np.ndarray,
    *,
    valid_draws: np.ndarray,
    draw_weights: np.ndarray,
    rank_tolerance: float | None = None,
) -> DesignInformationScore:
    """Aggregate per-draw information matrices into one robust score."""
    matrices = np.asarray(matrices, dtype=float)
    evaluation_sensitivities = np.asarray(evaluation_sensitivities, dtype=float)
    integration_weights = np.asarray(integration_weights, dtype=float)
    valid_draws = np.asarray(valid_draws, dtype=bool)
    draw_weights = np.asarray(draw_weights, dtype=float)

    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError('matrices must have shape (draws, parameters, parameters).')
    draw_count, parameter_count, _ = matrices.shape
    if evaluation_sensitivities.ndim != 3:
        raise ValueError(
            'evaluation_sensitivities must have shape (draws, times, parameters).'
        )
    if evaluation_sensitivities.shape[0] != draw_count or evaluation_sensitivities.shape[2] != parameter_count:
        raise ValueError('evaluation_sensitivities does not match the information matrices.')
    if integration_weights.ndim != 1 or len(integration_weights) != evaluation_sensitivities.shape[1]:
        raise ValueError('integration_weights does not match the evaluation time grid.')
    if valid_draws.shape != (draw_count,) or draw_weights.shape != (draw_count,):
        raise ValueError('valid_draws and draw_weights must contain one value per draw.')

    per_draw_criteria = np.full(draw_count, np.nan, dtype=float)
    per_draw_ranks = np.zeros(draw_count, dtype=int)
    per_draw_logdet = np.full(draw_count, np.nan, dtype=float)
    for index in np.flatnonzero(valid_draws):
        criterion, rank, logdet = _i_criterion_from_matrix(
            matrices[index],
            evaluation_sensitivities[index],
            integration_weights,
            rank_tolerance=rank_tolerance,
        )
        per_draw_criteria[index] = criterion
        per_draw_ranks[index] = rank
        per_draw_logdet[index] = logdet

    positive_weights = draw_weights.copy()
    positive_weights[~valid_draws] = 0.0
    positive_weights /= np.sum(positive_weights)
    if np.any(np.isinf(per_draw_criteria[valid_draws]) & (positive_weights[valid_draws] > 0)):
        aggregate = float('inf')
    else:
        aggregate = float(
            np.sum(positive_weights[valid_draws] * per_draw_criteria[valid_draws])
        )

    return DesignInformationScore(
        criterion=aggregate,
        parameter_count=parameter_count,
        per_draw_criteria=per_draw_criteria,
        per_draw_ranks=per_draw_ranks,
        per_draw_log_pseudodeterminants=per_draw_logdet,
        valid_draws=valid_draws,
        draw_weights=positive_weights,
    )


def ensemble_i_optimality(
    reference_model: ReferenceModelEnsemble,
    design_times,
    observation_model: ObservationModel,
    *,
    evaluation_times=None,
    evaluation_weights=None,
    replicates_per_time=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> DesignInformationScore:
    """Evaluate uncertainty-averaged I-optimality for one candidate design."""
    design_times = np.asarray(design_times, dtype=float)
    if design_times.ndim != 1 or len(design_times) == 0:
        raise ValueError('design_times must be a nonempty one-dimensional array.')
    if evaluation_times is None:
        evaluation_times = reference_model.time_grid
    evaluation_times = np.unique(np.asarray(evaluation_times, dtype=float))
    weights = trapezoidal_integration_weights(
        evaluation_times,
        point_weights=evaluation_weights,
    )

    draw_count = reference_model.n_draws
    parameter_count = reference_model.n_parameters
    matrices = np.zeros((draw_count, parameter_count, parameter_count), dtype=float)
    evaluation_sensitivities = np.zeros(
        (draw_count, len(evaluation_times), parameter_count),
        dtype=float,
    )
    for index in np.flatnonzero(reference_model.valid_draws):
        theta = reference_model.parameter_draws[index]
        matrices[index] = information_matrix(
            reference_model.model_name,
            design_times,
            theta,
            observation_model,
            replicates_per_time=replicates_per_time,
            relative_step=relative_step,
            absolute_step=absolute_step,
        )
        evaluation_sensitivities[index] = finite_difference_sensitivities(
            reference_model.model_name,
            evaluation_times,
            theta,
            relative_step=relative_step,
            absolute_step=absolute_step,
        )

    return score_information_matrices(
        matrices,
        evaluation_sensitivities,
        weights,
        valid_draws=reference_model.valid_draws,
        draw_weights=reference_model.draw_weights,
        rank_tolerance=rank_tolerance,
    )


def fit_reference_model_ensemble(
    reference: ReferenceTrajectoryEnsemble,
    *,
    model_name: str,
    initial_parameters: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    draw_indices: Sequence[int] | None = None,
    perform_grid_search: bool = False,
    fallback_grid_search: bool = True,
    grid_search_config: Mapping[str, object] | None = None,
    minimum_successful_draws: int = 1,
) -> ReferenceModelEnsemble:
    """Fit one analysis model to the central curve and each selected draw."""
    spec = get_model_spec(model_name)
    if draw_indices is None:
        selected = np.flatnonzero(reference.valid_draws & (reference.draw_weights > 0))
    else:
        selected = np.asarray(draw_indices, dtype=int)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError('draw_indices must be a nonempty one-dimensional sequence.')
        if np.any(selected < 0) or np.any(selected >= reference.n_draws):
            raise IndexError('draw_indices contains an out-of-range draw index.')
        selected = np.unique(selected)
        selected = selected[
            reference.valid_draws[selected] & (reference.draw_weights[selected] > 0)
        ]
    if len(selected) == 0:
        raise ValueError('No selected reference draw is valid.')

    central_fit = fit_model_once(
        reference.time_grid,
        reference.central_curve,
        model_name=model_name,
        initial_parameters=initial_parameters,
        perform_grid_search=perform_grid_search,
        fallback_grid_search=fallback_grid_search,
        grid_search_config=grid_search_config,
    )
    if not central_fit.success:
        raise ValueError(
            f'Could not fit model {model_name!r} to the central reference curve: '
            f'{central_fit.message}'
        )

    parameter_draws = np.full((len(selected), len(spec.parameter_names)), np.nan, dtype=float)
    valid_draws = np.zeros(len(selected), dtype=bool)
    fit_messages: list[str | None] = []
    for output_index, source_index in enumerate(selected):
        result = fit_model_once(
            reference.time_grid,
            reference.value_draws[source_index],
            model_name=model_name,
            initial_parameters=central_fit.parameter_vector,
            perform_grid_search=False,
            fallback_grid_search=fallback_grid_search,
            grid_search_config=grid_search_config,
        )
        if result.success:
            parameter_draws[output_index] = result.parameter_vector
            valid_draws[output_index] = True
            fit_messages.append(None)
        else:
            fit_messages.append(result.message)

    minimum_successful_draws = int(minimum_successful_draws)
    if minimum_successful_draws < 1:
        raise ValueError('minimum_successful_draws must be at least one.')
    successful = int(np.count_nonzero(valid_draws))
    if successful < minimum_successful_draws:
        raise ValueError(
            f'Only {successful} reference trajectories could be fitted; '
            f'{minimum_successful_draws} were required.'
        )

    source_weights = reference.draw_weights[selected].copy()
    source_weights[~valid_draws] = 0.0
    source_weight_total = float(np.sum(source_weights))
    if not source_weight_total > 0:
        raise ValueError('No positive-weight reference trajectory was fitted successfully.')
    source_weights /= source_weight_total
    metadata = dict(reference.metadata)
    metadata.update({
        'analysis_model_name': model_name,
        'analysis_parameter_names': spec.parameter_names,
        'source_draw_indices': selected.copy(),
        'requested_draw_count': int(len(selected)),
        'successful_model_fit_count': successful,
        'failed_model_fit_count': int(len(selected) - successful),
        'central_model_fit_used_grid_search': central_fit.used_grid_search,
    })

    return ReferenceModelEnsemble(
        model_name=model_name,
        parameter_names=spec.parameter_names,
        parameter_draws=parameter_draws,
        valid_draws=valid_draws,
        draw_weights=source_weights,
        central_parameters=central_fit.parameter_vector,
        time_grid=reference.time_grid,
        supported_time_range=reference.supported_time_range,
        fit_messages=fit_messages,
        metadata=metadata,
    )
