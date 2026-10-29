"""Joint latest-before selections with complete-schedule feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .feasibility import _compatible, _positive_integer, check_schedule_feasibility
from .requirements import (
    LatestBefore,
    ResolvedTimingRequirements,
    _contains,
    _qualifying_times,
    _signature,
)
from .types import DesignConstraints


@dataclass(frozen=True)
class LatestBeforePoint:
    """One requested point and its displacement from the latest eligible time.

    Eligibility accounts for the original fixed times and grid. It does not
    account for the final schedule size or competing timing requirements.
    """

    label: str
    priority: int
    lower_bound: float | None
    deadline: float | None
    eligible_times: tuple[float, ...]
    latest_eligible_time: float | None
    selected_time: float | None
    status: str
    message: str

    @property
    def deadline_slack(self) -> float | None:
        if self.selected_time is None:
            return None
        return self.deadline - self.selected_time

    @property
    def displacement(self) -> float | None:
        if self.selected_time is None:
            return None
        return self.latest_eligible_time - self.selected_time


@dataclass(frozen=True)
class LatestBeforeSelectionResult:
    """A complete joint selection or an explicit unresolved outcome.

    A feasible result certifies lexicographically latest selections in the
    reported priority order. Information optimality is a separate operation.
    """

    status: str
    message: str
    number_of_times: int
    points: tuple[LatestBeforePoint, ...] = ()
    times: tuple[float, ...] = ()
    states_visited: int = 0
    feasibility_checks: int = 0
    resolved_requirements: ResolvedTimingRequirements | None = None

    @property
    def selected_times(self) -> tuple[float, ...]:
        return tuple(sorted(point.selected_time for point in self.points
                            if point.selected_time is not None))

    @property
    def priority_labels(self) -> tuple[str, ...]:
        return tuple(point.label for point in sorted(self.points, key=lambda point: point.priority))


def _windows_only(resolved: ResolvedTimingRequirements) -> ResolvedTimingRequirements:
    rows = tuple(row for row in resolved.requirements if not isinstance(row.request, LatestBefore))
    ready = all(row.status in ('ready', 'already_satisfied') for row in rows)
    return replace(resolved, requirements=rows, status='ready' if ready else 'unavailable')


def _lock_latest_before_times(
    resolved: ResolvedTimingRequirements,
    constraints: DesignConstraints,
    selected_times,
) -> tuple[ResolvedTimingRequirements, DesignConstraints]:
    """Rebind resolved windows after reserving new points, without new targets."""
    locked = replace(constraints, fixed_times=np.union1d(constraints.fixed_times, selected_times))
    windows = _windows_only(resolved)
    candidates = locked.available_candidates()
    candidates = candidates[candidates > resolved.context.planning_start]
    remaining = locked.max_number_of_times - len(locked.fixed_times)
    rows = []
    for row in windows.requirements:
        fixed = _qualifying_times(locked.fixed_times, resolved.context,
                                  row.effective_lower_bound, row.upper_bound, row.count_scope)
        eligible = _qualifying_times(candidates, resolved.context,
                                     row.effective_lower_bound, row.upper_bound, row.count_scope)
        rows.append(replace(
            row, fixed_qualifying_times=fixed, eligible_times=eligible,
            candidate_count_upper_bound=len(fixed) + min(len(eligible), remaining),
            status='already_satisfied' if len(fixed) >= row.minimum_count else 'ready',
            message='Window accounting includes the reserved latest-before points.',
        ))
    return replace(
        windows, requirements=tuple(rows), candidate_times=tuple(map(float, candidates)),
        constraints_signature=_signature(locked),
    ), locked


def _matching_count(pools) -> int:
    """Maximum distinct-point matching, used only as a safe pruning bound."""
    owners, matched = {}, {}
    for initial in range(len(pools)):
        stack, parents, seen_values = [initial], {initial: None}, set()
        augmented = False
        while stack and not augmented:
            index = stack.pop()
            for value in pools[index]:
                if value in seen_values:
                    continue
                seen_values.add(value)
                if value in owners:
                    owner = owners[value]
                    if owner not in parents:
                        parents[owner] = index
                        stack.append(owner)
                    continue
                # Reverse the alternating path without recursive calls.
                current, next_value = index, value
                while current is not None:
                    previous_value = matched.get(current)
                    matched[current] = next_value
                    owners[next_value] = current
                    current, next_value = parents[current], previous_value
                augmented = True
                break
    return len(matched)


def select_latest_before_times(
    resolved: ResolvedTimingRequirements,
    constraints: DesignConstraints,
    number_of_times: int,
    *,
    max_search_states: int = 100000,
) -> LatestBeforeSelectionResult:
    """Select distinct latest-before points while preserving a complete study.

    Later deadlines have priority; equal deadlines retain request order. At
    each priority the latest feasible point is preferred, conditional on all
    higher-priority selections and satisfaction of every remaining request.
    Feasibility includes windows, fixed times, spacing, and the exact final N.
    Fixed future points may fill requests; baseline and completed times cannot.

    The budget covers assignment states and all window-feasibility states in
    this call. If any higher-priority possibility remains unresolved when the
    limit is reached, no selections are certified or returned.
    """
    if not isinstance(resolved, ResolvedTimingRequirements):
        raise TypeError('resolved must be ResolvedTimingRequirements.')
    if _signature(constraints) != resolved.constraints_signature:
        raise ValueError('Use the same design constraints that resolved the requirements.')
    size = _positive_integer(number_of_times, 'number_of_times')
    budget = _positive_integer(max_search_states, 'max_search_states')
    if not constraints.min_number_of_times <= size <= constraints.max_number_of_times:
        raise ValueError('number_of_times must lie within the design-size limits.')
    grid = constraints.all_available_times
    if np.any(np.isclose(grid[:-1], grid[1:], rtol=0.0, atol=1e-12)):
        raise ValueError('Fixed and candidate times must be unambiguous at the time-matching tolerance.')
    requests = tuple(row for row in resolved.requirements if isinstance(row.request, LatestBefore))
    order = sorted(range(len(requests)), key=lambda index: (
        requests[index].upper_bound is None,
        -requests[index].upper_bound if requests[index].upper_bound is not None else 0.0,
        index,
    ))
    priorities = {index: rank + 1 for rank, index in enumerate(order)}
    pools = tuple(tuple(sorted(set(row.fixed_qualifying_times + row.eligible_times)))
                  for row in requests)
    states = checks = 0

    def result(status, message, selections=(), times=()):
        by_index = dict(zip(order, selections)) if status == 'feasible' else {}
        points = tuple(LatestBeforePoint(
            row.label, priorities[index], row.effective_lower_bound, row.upper_bound,
            pools[index], max(pools[index]) if pools[index] else None,
            by_index.get(index),
            'selected' if index in by_index else (
                row.status if row.status not in ('ready', 'already_satisfied') else 'unresolved'
            ),
            ('Latest feasible distinct point in the reported priority order.'
             if index in by_index else message),
        ) for index, row in enumerate(requests))
        return LatestBeforeSelectionResult(
            status, message, size, points, tuple(times), states, checks, resolved,
        )

    unavailable = [row for row in resolved.requirements if row.status not in (
        'ready', 'already_satisfied', 'deadline_elapsed',
        'no_feasible_candidate', 'insufficient_candidates',
    )]
    if unavailable:
        return result('unavailable', 'Timing requirements cannot be resolved: '
                      + ', '.join(row.label for row in unavailable) + '.')
    impossible = [row for row in resolved.requirements if row.status not in ('ready', 'already_satisfied')]
    if impossible:
        return result('infeasible', 'Individual timing requirements cannot be satisfied: '
                      + ', '.join(row.label for row in impossible) + '.')

    windows = _windows_only(resolved)
    completion_cache = {}

    def complete(selected):
        nonlocal states, checks
        key = tuple(sorted(selected))
        if key not in completion_cache:
            if states >= budget:
                return None
            completion = check_schedule_feasibility(
                windows, constraints, size, required_times=selected,
                max_search_states=budget - states,
            )
            checks += 1
            states += completion.states_visited
            if completion.status == 'search_limit':
                return None
            completion_cache[key] = completion
        return completion_cache[key]

    initial = complete(())
    if initial is None:
        return result('search_limit', 'The joint selection limit was reached; latest points are not established.')
    if initial.status != 'feasible':
        return result(initial.status, initial.message)
    if not requests:
        return result('feasible', 'No latest-before points were requested; a feasible schedule exists.', times=initial.times)

    # Descending candidate visits establish the lexicographically latest
    # assignment. A memoized suffix only depends on the already reserved set.
    stack = [()]
    visited = set()
    while stack:
        selected = stack.pop()
        depth = len(selected)
        key = (depth, tuple(sorted(selected)))
        if key in visited:
            continue
        if states >= budget:
            return result('search_limit', 'The joint selection limit was reached; latest points are not established.')
        visited.add(key)
        states += 1
        remaining_pools = [tuple(
            value for value in pools[index]
            if value not in selected
            and all(_compatible(value, other, constraints.minimum_spacing) for other in selected)
        ) for index in order[depth:]]
        if _matching_count(remaining_pools) < len(remaining_pools):
            continue
        fixed_pools = [tuple(value for value in pool if _contains(constraints.fixed_times, value))
                       for pool in remaining_pools]
        free_slots = size - len(np.union1d(constraints.fixed_times, selected))
        if len(remaining_pools) - _matching_count(fixed_pools) > free_slots:
            continue
        completion = complete(selected)
        if completion is None:
            return result('search_limit', 'The joint selection limit was reached; latest points are not established.')
        if completion.status != 'feasible':
            continue
        if depth == len(order):
            return result('feasible', 'Distinct latest-before points and a complete feasible schedule were found.',
                          selected, completion.times)
        # Pools are ascending: the stack visits the largest time first.
        for value in remaining_pools[0]:
            stack.append(selected + (value,))
    return result('infeasible', 'No distinct latest-before assignment permits the requested complete schedule.')
