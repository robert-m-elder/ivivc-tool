"""Monte Carlo evaluation of proposed degradation-study time designs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fitting import fit_model_once
from .information import fit_reference_model_ensemble, trapezoidal_integration_weights
from .metrics import (
    approximate_mean_curve_interval_width,
    integrated_root_mean_square_error,
    maximum_absolute_error,
    parameter_near_bounds,
    reference_response_scale,
    threshold_crossing_time,
)
from .types import (
    DesignConstraints,
    DesignPerformanceSummary,
    DesignSimulationComparison,
    DesignSimulationResult,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
    SimulationPlan,
)


def _prepare_times(times, *, label: str) -> np.ndarray:
    values = np.unique(np.asarray(times, dtype=float))
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f'{label} must be a nonempty finite one-dimensional array.')
    return values


def _prepare_evaluation_grid(
    reference: ReferenceTrajectoryEnsemble,
    evaluation_times=None,
    evaluation_weights=None,
) -> tuple[np.ndarray, np.ndarray]:
    if evaluation_times is None:
        times = reference.time_grid.copy()
    else:
        times = np.asarray(evaluation_times, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError('evaluation_times must contain at least two points.')
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError('evaluation_times must be finite and strictly increasing.')
        lower, upper = reference.supported_time_range
        if times[0] < lower - 1e-12 or times[-1] > upper + 1e-12:
            raise ValueError('evaluation_times extend beyond the supported reference domain.')
    return times, trapezoidal_integration_weights(
        times,
        point_weights=evaluation_weights,
    )


def create_simulation_plan(
    reference: ReferenceTrajectoryEnsemble,
    observation_model: ObservationModel,
    master_times,
    *,
    n_simulations: int = 1000,
    random_state=None,
) -> SimulationPlan:
    """Create shared trajectory selections and standard-normal errors.

    Reusing one plan across proposed designs gives common random numbers at all
    shared candidate times, reducing Monte Carlo noise in design comparisons.
    """
    master_times = _prepare_times(master_times, label='master_times')
    lower, upper = reference.supported_time_range
    if master_times[0] < lower - 1e-12 or master_times[-1] > upper + 1e-12:
        raise ValueError('master_times extend beyond the supported reference domain.')
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError('n_simulations must be at least one.')

    valid_indices = np.flatnonzero(reference.valid_draws & (reference.draw_weights > 0))
    probabilities = reference.draw_weights[valid_indices]
    probabilities = probabilities / np.sum(probabilities)
    rng = np.random.default_rng(random_state)
    draw_indices = rng.choice(
        valid_indices,
        size=n_simulations,
        replace=True,
        p=probabilities,
    )
    errors = rng.standard_normal((
        n_simulations,
        int(observation_model.replicates_per_time),
        len(master_times),
    ))
    return SimulationPlan(
        master_times=master_times,
        trajectory_draw_indices=draw_indices,
        standard_normal_errors=errors,
        metadata={
            'random_state': random_state,
            'observation_model_kind': observation_model.kind,
            'observation_model_value': observation_model.value,
            'replicates_per_time': observation_model.replicates_per_time,
            'common_random_numbers': True,
        },
    )


def _parameter_start_for_source_draw(
    reference_model: ReferenceModelEnsemble,
    source_draw_index: int,
) -> np.ndarray:
    """Return the generating draw's parameters for explicit oracle diagnostics."""
    source_indices = reference_model.metadata.get('source_draw_indices')
    if source_indices is not None:
        source_indices = np.asarray(source_indices, dtype=int)
        matches = np.flatnonzero(source_indices == int(source_draw_index))
        if len(matches):
            row = int(matches[0])
            if reference_model.valid_draws[row]:
                return reference_model.parameter_draws[row].copy()
    return reference_model.central_parameters.copy()


def _fit_start_parameters(
    mode: str,
    reference_model: ReferenceModelEnsemble,
    source_draw_index: int,
) -> np.ndarray | None:
    mode = str(mode).strip().lower()
    if mode == 'data_driven':
        return None
    if mode == 'central_reference':
        return reference_model.central_parameters.copy()
    if mode == 'oracle':
        return _parameter_start_for_source_draw(reference_model, source_draw_index)
    raise ValueError(
        "fit_initialization must be 'data_driven', 'central_reference', or 'oracle'."
    )


def simulate_design(
    reference: ReferenceTrajectoryEnsemble,
    design_times,
    observation_model: ObservationModel,
    *,
    model_name: str,
    constraints: DesignConstraints | None = None,
    reference_model: ReferenceModelEnsemble | None = None,
    simulation_plan: SimulationPlan | None = None,
    n_simulations: int = 1000,
    random_state=None,
    evaluation_times=None,
    evaluation_weights=None,
    target_response: float | None = None,
    target_direction: str = 'auto',
    confidence_level: float = 0.95,
    failure_penalty: float | None = None,
    fallback_grid_search: bool = True,
    grid_search_config: Mapping[str, object] | None = None,
    fit_initialization: str = 'data_driven',
) -> DesignSimulationResult:
    """Simulate, refit, and score one proposed time-point design.

    The default ``data_driven`` initialization uses only the simulated
    dataset and the registered model heuristic. ``central_reference`` uses
    one common prior starting vector. ``oracle`` is retained only as an
    explicit diagnostic because it uses the unknown generating parameters.
    """
    fit_initialization = str(fit_initialization).strip().lower()
    if fit_initialization not in {'data_driven', 'central_reference', 'oracle'}:
        raise ValueError(
            "fit_initialization must be 'data_driven', 'central_reference', or 'oracle'."
        )
    if constraints is not None:
        design = constraints.validate_design(design_times)
        replicates = constraints.replicates_per_time
        if replicates != observation_model.replicates_per_time:
            raise ValueError(
                'DesignConstraints and ObservationModel specify different replicate counts.'
            )
        master_times = constraints.all_available_times
    else:
        design = _prepare_times(design_times, label='design_times')
        replicates = observation_model.replicates_per_time
        master_times = design

    lower, upper = reference.supported_time_range
    if design[0] < lower - 1e-12 or design[-1] > upper + 1e-12:
        raise ValueError('Design times extend beyond the supported reference domain.')

    if simulation_plan is None:
        simulation_plan = create_simulation_plan(
            reference,
            observation_model,
            master_times,
            n_simulations=n_simulations,
            random_state=random_state,
        )
    if simulation_plan.n_replicates < replicates:
        raise ValueError('Simulation plan does not contain enough replicate error draws.')
    design_indices = simulation_plan.indices_for_times(design)
    draw_indices = simulation_plan.trajectory_draw_indices
    n_simulations = simulation_plan.n_simulations

    evaluation_times, evaluation_weights = _prepare_evaluation_grid(
        reference,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
    )
    true_at_design = reference.evaluate(design, draw_indices=draw_indices)
    true_curves = reference.evaluate(evaluation_times, draw_indices=draw_indices)
    standard_deviation = observation_model.standard_deviation(true_at_design)
    errors = simulation_plan.standard_normal_errors[
        :, :replicates, :
    ][:, :, design_indices]
    observations = true_at_design[:, None, :] + standard_deviation[:, None, :] * errors
    observed_means = np.mean(observations, axis=1)

    if reference_model is None:
        reference_model = fit_reference_model_ensemble(reference, model_name=model_name)
    if reference_model.model_name != model_name:
        raise ValueError('reference_model does not match the requested analysis model.')

    parameter_count = reference_model.n_parameters
    fitted_curves = np.full_like(true_curves, np.nan, dtype=float)
    parameter_estimates = np.full((n_simulations, parameter_count), np.nan, dtype=float)
    parameter_bound_hits = np.zeros((n_simulations, parameter_count), dtype=bool)
    fit_success = np.zeros(n_simulations, dtype=bool)
    fit_messages: list[str | None] = []
    interval_widths = np.full(n_simulations, np.nan, dtype=float)

    repeated_times = np.tile(design, replicates)
    for simulation_index in range(n_simulations):
        response = observations[simulation_index].reshape(-1)
        start = _fit_start_parameters(
            fit_initialization,
            reference_model,
            int(draw_indices[simulation_index]),
        )
        fit = fit_model_once(
            repeated_times,
            response,
            model_name=model_name,
            initial_parameters=start,
            perform_grid_search=False,
            fallback_grid_search=fallback_grid_search,
            grid_search_config=grid_search_config,
        )
        if not fit.success:
            fit_messages.append(fit.message)
            continue
        try:
            predicted = fit.predict(evaluation_times)
        except Exception as exception:
            fit_messages.append(str(exception))
            continue
        if predicted.shape != evaluation_times.shape or np.any(~np.isfinite(predicted)):
            fit_messages.append('The fitted model returned invalid evaluation-grid values.')
            continue

        fit_success[simulation_index] = True
        fit_messages.append(None)
        fitted_curves[simulation_index] = predicted
        parameter_estimates[simulation_index] = fit.parameter_vector
        parameter_bound_hits[simulation_index] = parameter_near_bounds(
            model_name,
            fit.parameter_vector,
        )
        interval_widths[simulation_index] = approximate_mean_curve_interval_width(
            model_name,
            fit.parameter_vector,
            fit.covariance,
            evaluation_times,
            confidence_level=confidence_level,
        )

    irmse = integrated_root_mean_square_error(
        true_curves,
        fitted_curves,
        weights=evaluation_weights,
    )
    max_error = maximum_absolute_error(true_curves, fitted_curves)
    response_scale = reference_response_scale(true_curves)
    normalized_irmse = irmse / response_scale

    if target_response is None:
        true_target_times = np.full(n_simulations, np.nan, dtype=float)
        fitted_target_times = np.full(n_simulations, np.nan, dtype=float)
        target_time_errors = np.full(n_simulations, np.nan, dtype=float)
    else:
        true_target_times = threshold_crossing_time(
            evaluation_times,
            true_curves,
            float(target_response),
            direction=target_direction,
        )
        fitted_target_times = threshold_crossing_time(
            evaluation_times,
            fitted_curves,
            float(target_response),
            direction=target_direction,
        )
        target_time_errors = fitted_target_times - true_target_times

    if failure_penalty is None:
        failure_penalty = 2.0 * response_scale
    failure_penalty = float(failure_penalty)
    if not np.isfinite(failure_penalty) or failure_penalty < 0:
        raise ValueError('failure_penalty must be finite and nonnegative.')
    penalized_irmse = irmse.copy()
    penalized_irmse[~fit_success | ~np.isfinite(penalized_irmse)] = failure_penalty

    return DesignSimulationResult(
        design_times=design,
        model_name=model_name,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        trajectory_draw_indices=draw_indices,
        expected_values=true_at_design,
        observed_means=observed_means,
        true_curves=true_curves,
        fitted_curves=fitted_curves,
        parameter_estimates=parameter_estimates,
        parameter_bound_hits=parameter_bound_hits,
        fit_success=fit_success,
        fit_messages=fit_messages,
        integrated_rmse=irmse,
        normalized_integrated_rmse=normalized_irmse,
        maximum_absolute_error=max_error,
        mean_curve_interval_width=interval_widths,
        true_target_times=true_target_times,
        fitted_target_times=fitted_target_times,
        target_time_errors=target_time_errors,
        response_scale=response_scale,
        failure_penalty=failure_penalty,
        penalized_integrated_rmse=penalized_irmse,
        metadata={
            'target_response': target_response,
            'target_direction': target_direction,
            'confidence_level': confidence_level,
            'replicates_per_time': replicates,
            'fit_initialization': fit_initialization,
            'common_random_numbers': bool(
                simulation_plan.metadata.get('common_random_numbers', False)
            ),
        },
    )


def _finite_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(np.mean(finite)),
        float(np.median(finite)),
        float(np.percentile(finite, 90)),
        float(np.percentile(finite, 95)),
    )


def summarize_simulation(result: DesignSimulationResult) -> DesignPerformanceSummary:
    """Summarize successful-fit accuracy and all-study penalized performance."""
    success = result.fit_success
    mean_irmse, median_irmse, p90_irmse, p95_irmse = _finite_summary(
        result.integrated_rmse[success]
    )
    mean_nirmse, median_nirmse, p90_nirmse, _ = _finite_summary(
        result.normalized_integrated_rmse[success]
    )
    mean_max_error, _, p90_max_error, _ = _finite_summary(
        result.maximum_absolute_error[success]
    )
    mean_interval_width, median_interval_width, _, _ = _finite_summary(
        result.mean_curve_interval_width[success]
    )
    target_absolute = np.abs(result.target_time_errors[success])
    _, median_target_error, p90_target_error, _ = _finite_summary(target_absolute)
    penalized_mean_irmse = float(np.mean(result.penalized_integrated_rmse))
    penalized_mean_nirmse = penalized_mean_irmse / result.response_scale
    if np.any(success):
        boundary_hit_rate = float(np.mean(np.any(result.parameter_bound_hits[success], axis=1)))
    else:
        boundary_hit_rate = np.nan

    return DesignPerformanceSummary(
        design_times=result.design_times,
        n_simulations=result.n_simulations,
        n_successful=result.n_successful,
        failure_rate=result.failure_rate,
        response_scale=result.response_scale,
        failure_penalty=result.failure_penalty,
        mean_irmse=mean_irmse,
        median_irmse=median_irmse,
        p90_irmse=p90_irmse,
        p95_irmse=p95_irmse,
        mean_normalized_irmse=mean_nirmse,
        median_normalized_irmse=median_nirmse,
        p90_normalized_irmse=p90_nirmse,
        penalized_mean_irmse=penalized_mean_irmse,
        penalized_mean_normalized_irmse=penalized_mean_nirmse,
        mean_maximum_absolute_error=mean_max_error,
        p90_maximum_absolute_error=p90_max_error,
        mean_curve_interval_width=mean_interval_width,
        median_curve_interval_width=median_interval_width,
        median_absolute_target_time_error=median_target_error,
        p90_absolute_target_time_error=p90_target_error,
        parameter_bound_hit_rate=boundary_hit_rate,
        metadata=dict(result.metadata),
    )


def compare_designs_by_simulation(
    reference: ReferenceTrajectoryEnsemble,
    designs: Mapping[str, Sequence[float]],
    observation_model: ObservationModel,
    *,
    model_name: str,
    constraints: DesignConstraints,
    reference_model: ReferenceModelEnsemble | None = None,
    simulation_plan: SimulationPlan | None = None,
    n_simulations: int = 1000,
    random_state=None,
    **simulation_kwargs: Any,
) -> DesignSimulationComparison:
    """Evaluate multiple designs using one shared Monte Carlo plan."""
    if not designs:
        raise ValueError('At least one named design is required.')
    if simulation_plan is None:
        simulation_plan = create_simulation_plan(
            reference,
            observation_model,
            constraints.all_available_times,
            n_simulations=n_simulations,
            random_state=random_state,
        )
    if reference_model is None:
        reference_model = fit_reference_model_ensemble(reference, model_name=model_name)

    results: dict[str, DesignSimulationResult] = {}
    summaries: dict[str, DesignPerformanceSummary] = {}
    for label, times in designs.items():
        label = str(label)
        if label in results:
            raise ValueError(f'Duplicate design label: {label!r}.')
        result = simulate_design(
            reference,
            times,
            observation_model,
            model_name=model_name,
            constraints=constraints,
            reference_model=reference_model,
            simulation_plan=simulation_plan,
            **simulation_kwargs,
        )
        results[label] = result
        summaries[label] = summarize_simulation(result)

    return DesignSimulationComparison(
        simulation_plan=simulation_plan,
        results=results,
        summaries=summaries,
        metadata={
            'model_name': model_name,
            'common_random_numbers': True,
            'design_labels': tuple(results),
        },
    )


def _construct_target_baseline(
    target_times,
    constraints: DesignConstraints,
    number_of_times: int,
) -> np.ndarray:
    number_of_times = int(number_of_times)
    if number_of_times < constraints.min_number_of_times or number_of_times > constraints.max_number_of_times:
        raise ValueError('Requested baseline size is outside the allowed design-size range.')
    if number_of_times < len(constraints.fixed_times):
        raise ValueError('Requested baseline size is smaller than the number of fixed times.')

    targets = list(np.sort(np.asarray(target_times, dtype=float)))
    selected = constraints.fixed_times.copy()
    for fixed in selected:
        if targets:
            nearest = int(np.argmin(np.abs(np.asarray(targets) - fixed)))
            targets.pop(nearest)

    for target in targets:
        if len(selected) >= number_of_times:
            break
        available = constraints.available_candidates(selected)
        if len(available) == 0:
            break
        distances = np.abs(available - target)
        candidate = float(available[np.lexsort((available, distances))[0]])
        selected = np.union1d(selected, [candidate])

    while len(selected) < number_of_times:
        available = constraints.available_candidates(selected)
        if len(available) == 0:
            raise ValueError('Constraints do not permit the requested baseline design size.')
        if len(selected) == 0:
            candidate = float(available[0])
        else:
            spacing = np.min(np.abs(available[:, None] - selected[None, :]), axis=1)
            candidate = float(available[np.lexsort((available, -spacing))[0]])
        selected = np.union1d(selected, [candidate])
    return constraints.validate_design(selected)


def equally_spaced_design(
    constraints: DesignConstraints,
    number_of_times: int,
) -> np.ndarray:
    """Return the nearest feasible design to equal calendar-time spacing."""
    targets = np.linspace(constraints.min_time, constraints.max_time, int(number_of_times))
    return _construct_target_baseline(targets, constraints, number_of_times)


def log_spaced_design(
    constraints: DesignConstraints,
    number_of_times: int,
) -> np.ndarray:
    """Return a shifted-log-spaced design that also supports time zero."""
    count = int(number_of_times)
    span = constraints.max_time - constraints.min_time
    targets = constraints.min_time + np.expm1(
        np.linspace(0.0, np.log1p(span), count)
    )
    return _construct_target_baseline(targets, constraints, count)


def equal_response_spaced_design(
    reference: ReferenceTrajectoryEnsemble,
    constraints: DesignConstraints,
    number_of_times: int,
) -> np.ndarray:
    """Return a feasible design approximately equally spaced in central response."""
    dense_times = np.linspace(constraints.min_time, constraints.max_time, 1001)
    response = reference.evaluate_central(dense_times)
    differences = np.diff(response)
    scale = max(1.0, float(np.max(np.abs(response))))
    tolerance = 1e-10 * scale
    increasing = np.all(differences >= -tolerance)
    decreasing = np.all(differences <= tolerance)
    if not (increasing or decreasing):
        raise ValueError('Equal-response spacing requires a monotonic central reference curve.')

    target_responses = np.linspace(response[0], response[-1], int(number_of_times))
    if decreasing:
        inverse_response = response[::-1]
        inverse_time = dense_times[::-1]
    else:
        inverse_response = response
        inverse_time = dense_times
    unique_response, unique_indices = np.unique(inverse_response, return_index=True)
    if len(unique_response) < 2:
        raise ValueError('The central reference response is effectively constant.')
    target_times = np.interp(
        target_responses,
        unique_response,
        inverse_time[unique_indices],
    )
    return _construct_target_baseline(target_times, constraints, number_of_times)
