"""Central information optimization subject to timing requirements.

Each requested final size is planned independently. Feasibility is checked
separately from the heuristic information search; a feasible result does not
assert globally optimal information. The mixed API reserves joint
latest-before points before optimizing the remaining sampling times.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .anchors import (
    LatestBeforeSelectionResult,
    _lock_latest_before_times,
    select_latest_before_times,
)
from .feasibility import check_schedule_feasibility
from .optimize import (
    _build_local_information_context,
    _prepare_evaluation_grid,
    _score_is_better,
    _selection_rule,
)
from .requirements import (
    LatestBefore,
    ResolvedTimingRequirements,
    TimingRequirementsEvaluation,
    _signature,
    evaluate_timing_requirements,
)
from .types import DesignConstraints, DesignInformationScore, ObservationModel, ReferenceModelEnsemble


@dataclass(frozen=True)
class WindowDesignResult:
    """One requested size, with feasibility and optimization reported separately.

    ``status='feasible'`` always includes a complete, checked schedule. If an
    improvement search reaches a limit, that schedule is retained and the
    separate ``optimization_status`` records the interruption.
    """

    number_of_times: int
    status: str
    message: str
    times: tuple[float, ...] = ()
    score: DesignInformationScore | None = None
    initial_times: tuple[float, ...] = ()
    requirement_evaluation: TimingRequirementsEvaluation | None = None
    optimization_status: str = 'not_run'
    feasibility_states_visited: int = 0
    feasibility_checks: int = 0
    exchanges: tuple[tuple[float, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    latest_before_selection: LatestBeforeSelectionResult | None = None


@dataclass(frozen=True)
class WindowDesignOptimizationResult:
    """Independent constrained schedules; they are not a nested sequence."""

    model_name: str
    designs: tuple[WindowDesignResult, ...]
    evaluation_times: tuple[float, ...]
    evaluation_weights: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    resolved_requirements: ResolvedTimingRequirements | None = None

    def result_for_n(self, number_of_times: int) -> WindowDesignResult:
        """Return the outcome, including an unavailable outcome, for one size."""
        for result in self.designs:
            if result.number_of_times == number_of_times:
                return result
        raise KeyError(f'No constrained result is available for N={number_of_times}.')

    def design_for_n(self, number_of_times: int) -> WindowDesignResult:
        """Return a feasible design or raise when that size has no schedule."""
        result = self.result_for_n(number_of_times)
        if result.status != 'feasible':
            raise KeyError(f'No feasible constrained design for N={number_of_times}: {result.status}.')
        return result


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def _requested_sizes(number_of_times, constraints: DesignConstraints) -> tuple[int, ...]:
    if number_of_times is None:
        return tuple(range(constraints.min_number_of_times, constraints.max_number_of_times + 1))
    if isinstance(number_of_times, (int, np.integer)):
        values = (number_of_times,)
    else:
        try:
            values = tuple(number_of_times)
        except TypeError as exc:
            raise ValueError('number_of_times must be an integer or a sequence of integers.') from exc
    sizes = tuple(sorted({_positive_integer(value, 'number_of_times') for value in values}))
    if not sizes:
        raise ValueError('At least one final schedule size is required.')
    if any(
        size < constraints.min_number_of_times or size > constraints.max_number_of_times
        for size in sizes
    ):
        raise ValueError('Requested schedule sizes must be within the design constraints.')
    return sizes


def _eligible_additions(selected, resolved: ResolvedTimingRequirements, constraints: DesignConstraints):
    available = constraints.available_candidates(selected)
    return available[available > resolved.context.planning_start]


def _greedy_with_completion(
    initial_witness,
    context,
    resolved: ResolvedTimingRequirements,
    constraints: DesignConstraints,
    number_of_times: int,
    max_search_states: int,
):
    """Add informative points only while a full feasible completion exists."""
    selected = constraints.fixed_times.copy()
    witness = tuple(initial_witness)
    checks = states = 0
    interrupted = False
    while len(selected) < number_of_times:
        best_trial = best_score = best_completion = None
        for candidate in _eligible_additions(selected, resolved, constraints):
            trial = np.union1d(selected, [candidate])
            completion = check_schedule_feasibility(
                resolved, constraints, number_of_times,
                required_times=trial, max_search_states=max_search_states,
            )
            checks += 1
            states += completion.states_visited
            if completion.status == 'search_limit':
                interrupted = True
                break
            if completion.status != 'feasible':
                continue
            score = context.score(trial)
            if _score_is_better(score, best_score):
                best_trial, best_score, best_completion = trial, score, completion.times
        if interrupted:
            # The preceding iteration supplied a checked complete schedule.
            # Incomplete exploration must not become a claim of infeasibility.
            break
        if best_trial is None:
            raise RuntimeError('A feasible witness exists but no feasible next addition was found.')
        selected, witness = best_trial, tuple(best_completion)
    if not interrupted:
        witness = tuple(float(value) for value in selected)
    return witness, interrupted, checks, states


def _refine_windows(initial_times, context, resolved, constraints, max_iterations):
    """Coordinate exchange over complete schedules that preserve all windows."""
    current = np.asarray(initial_times, dtype=float)
    current_score = context.score(current)
    exchanges: list[tuple[float, float]] = []
    for _ in range(max_iterations):
        best_times, best_score, best_exchange = current, current_score, None
        for old_time in current:
            if np.any(np.isclose(constraints.fixed_times, old_time, rtol=0.0, atol=1e-12)):
                continue
            retained = current[~np.isclose(current, old_time, rtol=0.0, atol=1e-12)]
            for new_time in resolved.candidate_times:
                if np.any(np.isclose(current, new_time, rtol=0.0, atol=1e-12)):
                    continue
                trial = np.union1d(retained, [new_time])
                if len(trial) != len(current):
                    continue
                try:
                    assessment = evaluate_timing_requirements(trial, resolved, constraints)
                except ValueError:
                    continue
                if assessment.status != 'satisfied':
                    continue
                score = context.score(trial)
                if _score_is_better(score, best_score):
                    best_times, best_score = trial, score
                    best_exchange = (float(old_time), float(new_time))
        if best_exchange is None:
            return tuple(float(value) for value in current), current_score, tuple(exchanges), True
        current, current_score = best_times, best_score
        exchanges.append(best_exchange)
    return tuple(float(value) for value in current), current_score, tuple(exchanges), False


def optimize_window_constrained_designs(
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    resolved_requirements: ResolvedTimingRequirements,
    *,
    number_of_times: int | Sequence[int] | None = None,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
    refine: bool = True,
    max_exchange_iterations: int = 25,
    max_search_states: int = 100000,
) -> WindowDesignOptimizationResult:
    """Plan complete schedules with minimum counts in resolved sampling windows.

    ``number_of_times`` specifies total distinct times, including completed,
    baseline, and fixed future points. By default every size allowed by the
    constraints is requested. Each size is solved independently. Completed
    observations and future commitments remain fixed; other additions must
    occur after the resolved planning start.

    An exact-or-limited feasibility search precedes information optimization.
    The greedy information search only accepts additions with feasible full
    completions, then optional coordinate exchange preserves every window.
    The original evaluation horizon and central rank/I-optimal score are
    retained. Information optimization is heuristic, not a global guarantee.

    ``max_search_states`` limits each feasibility call. If a later call reaches
    that limit, a known feasible schedule is returned with
    ``optimization_status='search_limit'``. Without an initial feasible witness
    the result instead has ``status='search_limit'`` and no schedule. LatestBefore
    requests are rejected rather than treated as ordinary coverage windows.
    """
    if not isinstance(resolved_requirements, ResolvedTimingRequirements):
        raise TypeError('resolved_requirements must be a ResolvedTimingRequirements result.')
    if _signature(constraints) != resolved_requirements.constraints_signature:
        raise ValueError('Use the same design constraints that resolved the requirements.')
    all_times = constraints.all_available_times
    if np.any(np.isclose(all_times[:-1], all_times[1:], rtol=0.0, atol=1e-12)):
        raise ValueError('Fixed and candidate times must be unambiguous at the information-matching tolerance.')
    if any(isinstance(row.request, LatestBefore) for row in resolved_requirements.requirements):
        raise ValueError('LatestBefore scheduling is not implemented; supply SamplingWindow requirements only.')
    sizes = _requested_sizes(number_of_times, constraints)
    max_search_states = _positive_integer(max_search_states, 'max_search_states')
    if refine:
        max_exchange_iterations = _positive_integer(max_exchange_iterations, 'max_exchange_iterations')
    lower, upper = reference_model.supported_time_range
    source_domain = getattr(resolved_requirements, 'supported_time_range', None)
    if source_domain is not None and not np.allclose(
        reference_model.supported_time_range, source_domain, rtol=0.0, atol=1e-12,
    ):
        raise ValueError('The analysis model and resolved requirements must have the same supported domain.')
    if constraints.min_time < lower or constraints.max_time > upper:
        raise ValueError('Design constraints extend beyond the analysis-model domain.')
    times, weights = _prepare_evaluation_grid(reference_model, evaluation_times, evaluation_weights)
    context = None
    results: list[WindowDesignResult] = []
    for size in sizes:
        feasibility = check_schedule_feasibility(
            resolved_requirements, constraints, size, max_search_states=max_search_states,
        )
        if feasibility.status != 'feasible':
            results.append(WindowDesignResult(
                size, feasibility.status, feasibility.message,
                feasibility_states_visited=feasibility.states_visited,
                feasibility_checks=1,
            ))
            continue
        if context is None:
            context = _build_local_information_context(
                reference_model, observation_model, constraints,
                evaluation_times=times, evaluation_weights=evaluation_weights,
                relative_step=relative_step, absolute_step=absolute_step,
                rank_tolerance=rank_tolerance,
            )
        initial, interrupted, checks, states = _greedy_with_completion(
            feasibility.times, context, resolved_requirements, constraints,
            size, max_search_states,
        )
        score = context.score(initial)
        witness_score = context.score(feasibility.times)
        seed_selection = 'greedy_completion'
        if _score_is_better(witness_score, score):
            initial, score = feasibility.times, witness_score
            seed_selection = 'initial_feasibility_witness'
        final, exchanges, converged = initial, (), True
        if refine:
            final, score, exchanges, converged = _refine_windows(
                initial, context, resolved_requirements, constraints, max_exchange_iterations,
            )
        assessment = evaluate_timing_requirements(final, resolved_requirements, constraints)
        if assessment.status != 'satisfied' or len(final) != size:
            raise RuntimeError('Constrained optimization failed to preserve a complete feasible schedule.')
        optimization_status = 'search_limit' if interrupted else ('complete' if converged else 'exchange_limit')
        message = 'A feasible schedule was found; the configured information search completed.'
        if optimization_status == 'search_limit':
            message = 'A feasible schedule is retained; a feasibility check limited the information search.'
        elif optimization_status == 'exchange_limit':
            message = 'A feasible schedule is retained; coordinate exchange reached its iteration limit.'
        results.append(WindowDesignResult(
            size, 'feasible', message, final, score, initial, assessment,
            optimization_status, feasibility.states_visited + states, 1 + checks, exchanges,
            {
                'selection_rule': _selection_rule(score),
                'seed_selection': seed_selection,
                'refined': bool(refine),
                'coordinate_exchange_converged': converged if refine else None,
                'global_information_optimum_verified': False,
            },
        ))
    return WindowDesignOptimizationResult(
        reference_model.model_name, tuple(results),
        tuple(float(value) for value in times), tuple(float(value) for value in weights),
        {
            'criterion': 'local_i_optimality',
            'parameter_source': 'central_reference_fit',
            'independent_final_sizes': True,
            'nested_schedules': False,
            'max_search_states_per_check': max_search_states,
            'global_information_optimum_verified': False,
        },
        resolved_requirements,
    )


def _evaluate_assigned_points(times, resolved, constraints, selection):
    """Add the joint selector's distinct-point evidence to schedule accounting."""
    assessment = evaluate_timing_requirements(times, resolved, constraints)
    points = {point.label: point for point in selection.points}
    rows = []
    for row in assessment.requirements:
        if row.kind == 'latest_before':
            point = points[row.label]
            if point.selected_time not in times:
                raise RuntimeError('An optimized schedule omitted a reserved latest-before point.')
            row = replace(
                row, qualifying_times=(point.selected_time,), achieved_count=1,
                status='satisfied', deadline_slack=point.deadline_slack,
                message='A distinct point was selected jointly in the reported latest-before priority order.',
            )
        rows.append(row)
    if any(row.status != 'satisfied' for row in rows):
        raise RuntimeError('Mixed optimization failed to preserve a timing requirement.')
    return replace(assessment, requirements=tuple(rows), status='satisfied',
                   message='Window counts and all distinct latest-before selections are satisfied.')


def optimize_timing_constrained_designs(
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    resolved_requirements: ResolvedTimingRequirements,
    *,
    number_of_times: int | Sequence[int] | None = None,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
    refine: bool = True,
    max_exchange_iterations: int = 25,
    max_search_states: int = 100000,
    max_anchor_search_states: int = 100000,
) -> WindowDesignOptimizationResult:
    """Combine window coverage and distinct latest-before points for each N.

    Latest-before points are chosen first, with later deadlines taking
    priority and request order breaking equal-deadline ties. Each selection
    preserves feasibility of every remaining request and the complete study.
    The points are then locked while the existing central information search
    optimizes the remaining times. A different final N can change the anchors.

    The anchor budget covers the entire joint selection for each N. A limit
    before latestness is established returns no selected schedule. The window
    budget still limits each information-search feasibility check. If that
    search stops, the selector's complete feasible witness remains available.
    Window-only calls delegate to optimize_window_constrained_designs().
    """
    if not isinstance(resolved_requirements, ResolvedTimingRequirements):
        raise TypeError('resolved_requirements must be a ResolvedTimingRequirements result.')
    if _signature(constraints) != resolved_requirements.constraints_signature:
        raise ValueError('Use the same design constraints that resolved the requirements.')
    sizes = _requested_sizes(number_of_times, constraints)
    max_search_states = _positive_integer(max_search_states, 'max_search_states')
    max_anchor_search_states = _positive_integer(max_anchor_search_states, 'max_anchor_search_states')
    if refine:
        max_exchange_iterations = _positive_integer(max_exchange_iterations, 'max_exchange_iterations')
    kwargs = dict(
        evaluation_times=evaluation_times, evaluation_weights=evaluation_weights,
        relative_step=relative_step, absolute_step=absolute_step,
        rank_tolerance=rank_tolerance, refine=refine,
        max_exchange_iterations=max_exchange_iterations, max_search_states=max_search_states,
    )
    if not any(isinstance(row.request, LatestBefore) for row in resolved_requirements.requirements):
        return optimize_window_constrained_designs(
            reference_model, observation_model, constraints, resolved_requirements,
            number_of_times=sizes, **kwargs,
        )
    source_domain = resolved_requirements.supported_time_range
    if source_domain is not None and not np.allclose(
        reference_model.supported_time_range, source_domain, rtol=0.0, atol=1e-12,
    ):
        raise ValueError('The analysis model and resolved requirements must have the same supported domain.')
    lower, upper = reference_model.supported_time_range
    if constraints.min_time < lower or constraints.max_time > upper:
        raise ValueError('Design constraints extend beyond the analysis-model domain.')
    times, weights = _prepare_evaluation_grid(reference_model, evaluation_times, evaluation_weights)
    results = []
    for size in sizes:
        selection = select_latest_before_times(
            resolved_requirements, constraints, size, max_search_states=max_anchor_search_states,
        )
        if selection.status != 'feasible':
            results.append(WindowDesignResult(
                size, selection.status, selection.message,
                feasibility_states_visited=selection.states_visited,
                feasibility_checks=selection.feasibility_checks, latest_before_selection=selection,
            ))
            continue
        windows, locked = _lock_latest_before_times(
            resolved_requirements, constraints, selection.selected_times,
        )
        optimized = optimize_window_constrained_designs(
            reference_model, observation_model, locked, windows, number_of_times=size, **kwargs,
        ).result_for_n(size)
        if optimized.status == 'search_limit':
            # The anchor selection already supplied a complete feasible study.
            # A tighter information-search budget must not discard that study.
            context = _build_local_information_context(
                reference_model, observation_model, constraints,
                evaluation_times=times, evaluation_weights=evaluation_weights,
                relative_step=relative_step, absolute_step=absolute_step,
                rank_tolerance=rank_tolerance,
            )
            fallback_score = context.score(selection.times)
            optimized = replace(
                optimized, status='feasible', times=selection.times,
                initial_times=selection.times, score=fallback_score,
                optimization_status='search_limit',
                message='The latest-before selection is established; its feasible witness is retained after an information-search limit.',
                metadata={'selection_rule': _selection_rule(fallback_score),
                          'refined': False, 'coordinate_exchange_converged': None,
                          'seed_selection': 'latest_before_feasibility_witness',
                          'global_information_optimum_verified': False},
            )
        elif optimized.status != 'feasible':
            raise RuntimeError('A verified joint selection lost its feasible completion during optimization.')
        assessment = _evaluate_assigned_points(
            optimized.times, resolved_requirements, constraints, selection,
        )
        results.append(replace(
            optimized, requirement_evaluation=assessment, latest_before_selection=selection,
            feasibility_states_visited=optimized.feasibility_states_visited + selection.states_visited,
            feasibility_checks=optimized.feasibility_checks + selection.feasibility_checks,
            metadata={**optimized.metadata, 'latest_before_priority': selection.priority_labels,
                      'locked_latest_before_times': selection.selected_times,
                      'latest_before_locations_verified': True},
        ))
    return WindowDesignOptimizationResult(
        reference_model.model_name, tuple(results), tuple(map(float, times)), tuple(map(float, weights)),
        {'criterion': 'local_i_optimality', 'parameter_source': 'central_reference_fit',
         'independent_final_sizes': True, 'nested_schedules': False,
         'latest_before_priority': 'later_deadlines_first_then_request_order',
         'max_anchor_search_states_per_size': max_anchor_search_states,
         'max_search_states_per_check': max_search_states,
         'global_information_optimum_verified': False},
        resolved_requirements,
    )
