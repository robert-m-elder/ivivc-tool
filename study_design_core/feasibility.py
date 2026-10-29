"""Discrete completion checks for schedules with sampling-window counts.

Feasibility is separate from information optimization. A witness establishes
that a schedule exists; an exhausted finite search establishes infeasibility.
Reaching the state budget establishes neither and is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .requirements import (
    LatestBefore,
    ResolvedTimingRequirements,
    _contains,
    _qualifying_times,
    _signature,
)
from .types import DesignConstraints


@dataclass(frozen=True)
class ScheduleFeasibilityResult:
    """An exact completion result, or an explicitly bounded search result."""

    status: str
    times: tuple[float, ...] = ()
    states_visited: int = 0
    message: str = ''


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def _compatible(left: float, right: float, spacing: float) -> bool:
    return bool(
        not np.isclose(left, right, rtol=0.0, atol=1e-12)
        and abs(left - right) >= spacing
    )


def check_schedule_feasibility(
    resolved: ResolvedTimingRequirements,
    constraints: DesignConstraints,
    number_of_times: int,
    *,
    required_times=(),
    max_search_states: int = 100000,
) -> ScheduleFeasibilityResult:
    """Find a complete schedule containing all fixed and provisional times.

    ``number_of_times`` is the total number of distinct target-study times,
    including baseline and completed times. ``required_times`` adds temporary
    commitments without changing the input constraints or requirement scopes.
    Only candidates strictly after the resolved planning start can be added.

    The memoized, iterative search respects spacing and all window counts.
    Its state budget counts distinct states examined, not candidate tests.
    ``search_limit`` must not be interpreted as proof of infeasibility.
    Latest-before requests require the separate anchor scheduler and are
    explicitly rejected rather than treated as simple coverage windows.
    """
    if not isinstance(resolved, ResolvedTimingRequirements):
        raise TypeError('resolved must be ResolvedTimingRequirements.')
    if _signature(constraints) != resolved.constraints_signature:
        raise ValueError('Use the same design constraints that resolved the requirements.')
    if any(isinstance(row.request, LatestBefore) for row in resolved.requirements):
        raise ValueError('LatestBefore selection is not supported by the window scheduler.')
    size = _positive_integer(number_of_times, 'number_of_times')
    budget = _positive_integer(max_search_states, 'max_search_states')
    if not constraints.min_number_of_times <= size <= constraints.max_number_of_times:
        raise ValueError('number_of_times must lie within the design-size limits.')

    supplied = np.asarray(required_times, dtype=float)
    if supplied.ndim != 1 or np.any(~np.isfinite(supplied)):
        raise ValueError('required_times must be a finite one-dimensional time sequence.')
    all_times = constraints.all_available_times
    if np.any(np.isclose(all_times[:-1], all_times[1:], rtol=0.0, atol=1e-12)):
        raise ValueError('Fixed and candidate grid times must be distinct within the time-matching tolerance.')
    extra = []
    for value in supplied:
        matches = np.flatnonzero(np.isclose(all_times, value, rtol=0.0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError('Each required time must identify one fixed or candidate time.')
        canonical = float(all_times[matches[0]])
        if canonical <= resolved.context.planning_start and not _contains(
            constraints.fixed_times, canonical,
        ):
            raise ValueError('New required times must lie strictly after planning_start.')
        extra.append(canonical)
    mandatory = np.union1d(constraints.fixed_times, extra)
    if len(mandatory) > size:
        return ScheduleFeasibilityResult(
            'infeasible', message='Fixed and required times exceed the requested schedule size.',
        )
    if any(
        not _compatible(left, right, constraints.minimum_spacing)
        for left, right in zip(mandatory[:-1], mandatory[1:])
    ):
        return ScheduleFeasibilityResult(
            'infeasible', message='Fixed and required times violate spacing or distinctness.',
        )

    # Missing reference targets and malformed windows cannot supply a numerical
    # scheduling constraint. Other individual failures already prove that no
    # complete schedule can satisfy the request.
    unavailable = [
        row for row in resolved.requirements
        if row.status not in (
            'ready', 'already_satisfied', 'deadline_elapsed',
            'no_feasible_candidate', 'insufficient_candidates',
        )
    ]
    if unavailable:
        labels = ', '.join(row.label for row in unavailable)
        return ScheduleFeasibilityResult(
            'unavailable', message=f'Timing requirements cannot be resolved: {labels}.',
        )
    individually_impossible = [
        row for row in resolved.requirements
        if row.status in ('deadline_elapsed', 'no_feasible_candidate', 'insufficient_candidates')
    ]
    if individually_impossible:
        labels = ', '.join(row.label for row in individually_impossible)
        return ScheduleFeasibilityResult(
            'infeasible', message=f'Individual timing requirements cannot be satisfied: {labels}.',
        )

    candidates = np.asarray([
        value for value in resolved.candidate_times
        if not _contains(mandatory, value)
        and all(_compatible(value, fixed, constraints.minimum_spacing) for fixed in mandatory)
    ], dtype=float)
    slots = size - len(mandatory)
    deficits = tuple(max(
        0, row.minimum_count - len(_qualifying_times(
            mandatory, resolved.context, row.effective_lower_bound,
            row.upper_bound, row.count_scope,
        )),
    ) for row in resolved.requirements)
    if slots > len(candidates):
        return ScheduleFeasibilityResult(
            'infeasible', message='Too few compatible future candidates remain for this schedule size.',
        )

    count = len(candidates)
    memberships = np.asarray([
        [row.effective_lower_bound < value <= row.upper_bound for row in resolved.requirements]
        for value in candidates
    ], dtype=int).reshape(count, len(deficits))
    # After choosing i, only this suffix can follow it. Checking isclose as
    # well as spacing agrees with the existing candidate-matching convention.
    next_indices = np.empty(count, dtype=int)
    for index, value in enumerate(candidates):
        following = max(index + 1, int(np.searchsorted(
            candidates, value + constraints.minimum_spacing, side='left',
        )))
        while following < count and not _compatible(
            value, candidates[following], constraints.minimum_spacing,
        ):
            following += 1
        next_indices[index] = following

    # Maximum packing counts give safe pruning bounds, including when spacing
    # rules out many individually available candidates. Window bounds are
    # independent upper bounds; overlapping requirements may share a point.
    capacity = np.zeros(count + 1, dtype=int)
    window_capacity = np.zeros((count + 1, len(deficits)), dtype=int)
    for index in range(count - 1, -1, -1):
        following = next_indices[index]
        capacity[index] = max(capacity[index + 1], 1 + capacity[following])
        window_capacity[index] = np.maximum(
            window_capacity[index + 1],
            memberships[index] + window_capacity[following],
        )

    # State is (remaining suffix, slots, capped quota deficits). The selected
    # prefix is retained only to reconstruct a witness. An explicit stack
    # avoids Python recursion limits on large candidate grids.
    stack = [(0, slots, deficits, ())]
    visited: set[tuple] = set()
    states = 0
    while stack:
        start, remaining, needs, selected = stack.pop()
        state = (start, remaining, needs)
        if state in visited:
            continue
        if states >= budget:
            return ScheduleFeasibilityResult(
                'search_limit', states_visited=states,
                message='The feasibility state limit was reached; infeasibility has not been established.',
            )
        visited.add(state)
        states += 1
        if remaining == 0:
            if not any(needs):
                times = tuple(float(value) for value in np.union1d(mandatory, selected))
                return ScheduleFeasibilityResult(
                    'feasible', times, states,
                    'A complete schedule satisfying spacing and all window counts was found.',
                )
            continue
        if capacity[start] < remaining or any(
            need > min(remaining, int(window_capacity[start, column]))
            for column, need in enumerate(needs)
        ):
            continue
        # Reverse insertion visits earlier candidates first deterministically.
        for index in range(count - 1, start - 1, -1):
            following = int(next_indices[index])
            if capacity[following] < remaining - 1:
                continue
            updated = tuple(max(0, need - int(member)) for need, member in zip(
                needs, memberships[index],
            ))
            if any(
                need > min(remaining - 1, int(window_capacity[following, column]))
                for column, need in enumerate(updated)
            ):
                continue
            stack.append((following, remaining - 1, updated, selected + (candidates[index],)))
    return ScheduleFeasibilityResult(
        'infeasible', states_visited=states,
        message='The finite completion search exhausted all spacing-compatible schedules.',
    )
