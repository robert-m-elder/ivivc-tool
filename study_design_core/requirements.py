"""Resolve and inspect timing requirements without selecting a schedule.

This layer deliberately makes no joint-feasibility or information-optimality
claim. It separates target-study timing roles from the source of the reference
trajectories. Existing one-point policies and optimization remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .timing import reference_crossing_time_summary
from .types import CrossingTimeSummary, DesignConstraints, ReferenceTrajectoryEnsemble


def _finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number.')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number.') from exc
    if not np.isfinite(number):
        raise ValueError(f'{name} must be a finite number.')
    return number


def _times(values, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or np.any(~np.isfinite(array)):
        raise ValueError(f'{name} must be a finite one-dimensional time sequence.')
    return tuple(float(value) for value in np.unique(array))


def _contains(values, value: float) -> bool:
    return bool(np.any(np.isclose(values, value, rtol=0.0, atol=1e-12)))


def _label(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('label must be a string.')
    return value.strip()


@dataclass(frozen=True)
class CrossingTimeTarget:
    """A relative degradation milestone and reference uncertainty quantile."""

    degraded_fraction: float = 0.5
    quantile: float = 0.1
    baseline_time: float = 0.0

    def __post_init__(self) -> None:
        for name in ('degraded_fraction', 'quantile'):
            value = _finite(getattr(self, name), name)
            if not 0 < value < 1:
                raise ValueError(f'{name} must be strictly between zero and one.')
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'baseline_time', _finite(self.baseline_time, 'baseline_time'))


TimeBound = float | CrossingTimeTarget


def _bound(value, name: str) -> TimeBound:
    return value if isinstance(value, CrossingTimeTarget) else _finite(value, name)


@dataclass(frozen=True)
class SamplingWindow:
    """Require distinct nonbaseline times within ``(lower_bound, upper_bound]``.

    ``new`` excludes completed observations but includes fixed future times.
    ``all`` also counts completed nonbaseline target-study observations.
    """

    lower_bound: TimeBound
    upper_bound: TimeBound
    minimum_count: int = 1
    count_scope: Literal['new', 'all'] = 'new'
    label: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'lower_bound', _bound(self.lower_bound, 'lower_bound'))
        object.__setattr__(self, 'upper_bound', _bound(self.upper_bound, 'upper_bound'))
        if (
            isinstance(self.minimum_count, (bool, np.bool_))
            or not isinstance(self.minimum_count, (int, np.integer))
            or self.minimum_count < 1
        ):
            raise ValueError('minimum_count must be a positive integer.')
        object.__setattr__(self, 'minimum_count', int(self.minimum_count))
        if self.count_scope not in ('new', 'all'):
            raise ValueError("count_scope must be 'new' or 'all'.")
        object.__setattr__(self, 'label', _label(self.label))


@dataclass(frozen=True)
class LatestBefore:
    """Request a future sampling time as late as feasible before a deadline.

    Resolution and inspection only identify eligible times. Selecting the
    latest time jointly with other requests belongs to the later scheduler.
    """

    deadline: TimeBound
    lower_bound: TimeBound | None = None
    label: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'deadline', _bound(self.deadline, 'deadline'))
        if self.lower_bound is not None:
            object.__setattr__(self, 'lower_bound', _bound(self.lower_bound, 'lower_bound'))
        object.__setattr__(self, 'label', _label(self.label))


TimingRequirement = SamplingWindow | LatestBefore


@dataclass(frozen=True)
class PlanningContext:
    """Explicit target-study roles, independent of historical reference data.

    Baseline and completed times must be fixed in the design constraints.
    ``planning_start`` defaults to the latest completed time, or the design
    minimum for a prospective study. Other fixed times must lie after it.
    """

    baseline_times: Sequence[float]
    completed_times: Sequence[float] = ()
    planning_start: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'baseline_times', _times(self.baseline_times, 'baseline_times'))
        object.__setattr__(self, 'completed_times', _times(self.completed_times, 'completed_times'))
        if self.planning_start is not None:
            object.__setattr__(self, 'planning_start', _finite(self.planning_start, 'planning_start'))


@dataclass(frozen=True)
class ResolvedTimingRequirement:
    """Resolved bounds and individual capacity diagnostics, not feasibility."""

    request: TimingRequirement
    label: str
    lower_bound: float | None
    upper_bound: float | None
    effective_lower_bound: float | None
    lower_summary: CrossingTimeSummary | None
    upper_summary: CrossingTimeSummary | None
    status: str
    message: str
    fixed_qualifying_times: tuple[float, ...] = ()
    eligible_times: tuple[float, ...] = ()
    candidate_count_upper_bound: int = 0

    @property
    def kind(self) -> str:
        return 'window' if isinstance(self.request, SamplingWindow) else 'latest_before'

    @property
    def minimum_count(self) -> int:
        return self.request.minimum_count if isinstance(self.request, SamplingWindow) else 1

    @property
    def count_scope(self) -> str:
        return self.request.count_scope if isinstance(self.request, SamplingWindow) else 'new'


@dataclass(frozen=True)
class ResolvedTimingRequirements:
    """A resolution result; ``ready`` does not imply joint feasibility."""

    requirements: tuple[ResolvedTimingRequirement, ...]
    context: PlanningContext
    candidate_times: tuple[float, ...]
    status: str
    message: str
    constraints_signature: tuple
    supported_time_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class RequirementEvaluation:
    """Counts and deadline slack for one concrete schedule requirement."""

    label: str
    kind: str
    qualifying_times: tuple[float, ...]
    achieved_count: int
    minimum_count: int
    status: str
    message: str
    deadline_slack: float | None = None


@dataclass(frozen=True)
class TimingRequirementsEvaluation:
    """Inspection of a complete schedule, without verifying latestness."""

    times: tuple[float, ...]
    requirements: tuple[RequirementEvaluation, ...]
    status: str
    message: str


def _signature(constraints: DesignConstraints) -> tuple:
    return (
        tuple(constraints.candidate_times), tuple(constraints.fixed_times),
        constraints.min_time, constraints.max_time, constraints.minimum_spacing,
        constraints.min_number_of_times, constraints.max_number_of_times,
        constraints.replicates_per_time,
    )


def _planning_context(
    context: PlanningContext,
    constraints: DesignConstraints,
    reference: ReferenceTrajectoryEnsemble,
) -> PlanningContext:
    if not isinstance(context, PlanningContext):
        raise TypeError('context must be a PlanningContext.')
    lower, upper = reference.supported_time_range
    if constraints.min_time < lower or constraints.max_time > upper:
        raise ValueError('Design constraints extend beyond the supported reference domain.')
    for values, name in (
        (context.baseline_times, 'baseline_times'),
        (context.completed_times, 'completed_times'),
    ):
        if not all(_contains(constraints.fixed_times, value) for value in values):
            raise ValueError(f'{name} must be a subset of fixed_times.')
    start = context.planning_start
    if start is None:
        start = max(context.completed_times) if context.completed_times else constraints.min_time
    if not constraints.min_time <= start <= constraints.max_time:
        raise ValueError('planning_start must lie within the design and reference domain.')
    if context.completed_times and start < max(context.completed_times):
        raise ValueError('planning_start cannot precede a completed observation.')
    if any(value > start for value in context.baseline_times):
        raise ValueError('baseline_times cannot lie after planning_start.')
    for value in constraints.fixed_times:
        if (
            value <= start
            and not _contains(context.completed_times, value)
            and not _contains(context.baseline_times, value)
        ):
            raise ValueError('A fixed time at or before planning_start must be baseline or completed.')
    return PlanningContext(context.baseline_times, context.completed_times, start)


def _qualifying_times(times, context: PlanningContext, lower: float, upper: float, scope: str) -> tuple:
    return tuple(
        float(value) for value in times
        if lower < value <= upper
        and not _contains(context.baseline_times, value)
        and (scope == 'all' or not _contains(context.completed_times, value))
        and (scope == 'all' or value > context.planning_start)
    )


def resolve_timing_requirements(
    requirements: Sequence[TimingRequirement],
    reference: ReferenceTrajectoryEnsemble,
    constraints: DesignConstraints,
    *,
    context: PlanningContext,
) -> ResolvedTimingRequirements:
    """Resolve every request and retain individual diagnostic results.

    Unique crossing targets are calculated once from the original reference.
    Capacity is an upper bound: it accounts for fixed points and total budget,
    but not spacing between optional candidates or competition among requests.
    Consequently ``ready`` never asserts that a complete schedule exists.
    """
    context = _planning_context(context, constraints, reference)
    available = constraints.available_candidates()
    available = available[available > context.planning_start]
    remaining_budget = constraints.max_number_of_times - len(constraints.fixed_times)
    summaries: dict[CrossingTimeTarget, CrossingTimeSummary] = {}
    rows: list[ResolvedTimingRequirement] = []
    labels: set[str] = set()

    def resolve_bound(bound: TimeBound) -> tuple[float | None, CrossingTimeSummary | None]:
        if isinstance(bound, CrossingTimeTarget):
            if bound not in summaries:
                summaries[bound] = reference_crossing_time_summary(
                    reference, degraded_fraction=bound.degraded_fraction,
                    quantile=bound.quantile, baseline_time=bound.baseline_time,
                )
            summary = summaries[bound]
            return summary.deadline, summary
        return float(bound), None

    for index, request in enumerate(requirements, start=1):
        if not isinstance(request, (SamplingWindow, LatestBefore)):
            raise TypeError('Each timing requirement must be a SamplingWindow or LatestBefore.')
        kind = 'window' if isinstance(request, SamplingWindow) else 'latest_before'
        label = request.label or f'{kind}_{index}'
        if label in labels:
            raise ValueError('Timing requirement labels must be unique.')
        labels.add(label)
        lower_request = request.lower_bound
        if lower_request is None:
            lower_request = context.planning_start
        upper_request = request.upper_bound if isinstance(request, SamplingWindow) else request.deadline
        lower, lower_summary = resolve_bound(lower_request)
        upper, upper_summary = resolve_bound(upper_request)
        scope = request.count_scope if isinstance(request, SamplingWindow) else 'new'
        minimum = request.minimum_count if isinstance(request, SamplingWindow) else 1
        effective_lower = lower if scope == 'all' or lower is None else max(lower, context.planning_start)
        failed = next((
            summary for summary in (lower_summary, upper_summary)
            if summary is not None and summary.status != 'available'
        ), None)
        fixed: tuple[float, ...] = ()
        eligible: tuple[float, ...] = ()
        capacity = 0
        if failed is not None:
            status, message = failed.status, failed.message
        elif lower >= upper and not (
            isinstance(request, LatestBefore)
            and request.lower_bound is None
            and upper <= context.planning_start
        ):
            status, message = 'invalid_window', 'The lower bound must be strictly before the upper bound.'
        elif scope == 'new' and upper <= context.planning_start:
            status, message = 'deadline_elapsed', 'The deadline is at or before planning_start.'
        else:
            fixed = _qualifying_times(constraints.fixed_times, context, effective_lower, upper, scope)
            eligible = _qualifying_times(available, context, effective_lower, upper, scope)
            capacity = len(fixed) + min(len(eligible), remaining_budget)
            if len(fixed) >= minimum:
                status, message = 'already_satisfied', 'Fixed target-study times already meet this individual count.'
            elif not eligible:
                status, message = 'no_feasible_candidate', 'No eligible future candidate can complete this individual requirement.'
            elif capacity < minimum:
                status, message = 'insufficient_candidates', 'Even the individual candidate-count upper bound is below the requested count.'
            else:
                status, message = 'ready', 'Bounds resolved; individual capacity checks passed, without a joint-feasibility claim.'
        rows.append(ResolvedTimingRequirement(
            request, label, lower, upper, effective_lower, lower_summary, upper_summary,
            status, message, fixed, eligible, capacity,
        ))
    ready = all(row.status in ('ready', 'already_satisfied') for row in rows)
    return ResolvedTimingRequirements(
        tuple(rows), context, tuple(float(value) for value in available),
        'ready' if ready else 'unavailable',
        ('Requirements resolved; joint feasibility has not been checked.' if ready else
         'One or more requirements are individually unavailable; no schedule has been selected.'),
        _signature(constraints),
        reference.supported_time_range,
    )


def evaluate_timing_requirements(
    times,
    resolved: ResolvedTimingRequirements,
    constraints: DesignConstraints,
) -> TimingRequirementsEvaluation:
    """Validate a complete schedule and inspect its timing requirements.

    Window counts may be satisfied. Latest-before requests are reported only
    as ``not_past_deadline`` when an eligible point exists; this does not
    verify latestness or distinct assignment among multiple such requests.
    """
    if _signature(constraints) != resolved.constraints_signature:
        raise ValueError('Use the same design constraints that resolved the requirements.')
    supplied = np.asarray(times, dtype=float)
    if supplied.ndim != 1:
        raise ValueError('Schedule times must be one-dimensional.')
    schedule = constraints.validate_design(supplied)
    context = resolved.context
    if any(
        value <= context.planning_start and not _contains(constraints.fixed_times, value)
        for value in schedule
    ):
        raise ValueError('New schedule times must lie strictly after planning_start.')
    rows: list[RequirementEvaluation] = []
    for requirement in resolved.requirements:
        unavailable = requirement.status not in ('ready', 'already_satisfied')
        qualifying = ()
        if (
            requirement.effective_lower_bound is not None
            and requirement.upper_bound is not None
            and requirement.effective_lower_bound < requirement.upper_bound
        ):
            qualifying = _qualifying_times(
                schedule, context, requirement.effective_lower_bound,
                requirement.upper_bound, requirement.count_scope,
            )
        slack = requirement.upper_bound - max(qualifying) if qualifying else None
        if unavailable:
            status, message = requirement.status, requirement.message
        else:
            enough = len(qualifying) >= requirement.minimum_count
            if enough and requirement.kind == 'latest_before':
                status = 'not_past_deadline'
                message = 'An eligible future point is present; latestness and distinct assignment are not verified.'
            elif enough:
                status, message = 'satisfied', 'The requested distinct-time count is satisfied.'
            else:
                status, message = 'unsatisfied', 'The schedule does not contain enough qualifying times.'
        rows.append(RequirementEvaluation(
            requirement.label, requirement.kind, qualifying, len(qualifying),
            requirement.minimum_count, status, message, slack,
        ))
    if resolved.status != 'ready':
        status, message = 'unavailable', 'At least one timing requirement could not be resolved or completed individually.'
    elif any(row.status == 'unsatisfied' for row in rows):
        status, message = 'unsatisfied', 'At least one timing requirement is not met by this schedule.'
    elif any(row.kind == 'latest_before' for row in rows):
        status, message = 'not_past_deadline', 'Window counts and deadline eligibility pass; latestness and distinct assignment are not verified.'
    else:
        status, message = 'satisfied', 'All requested window counts are satisfied.'
    return TimingRequirementsEvaluation(tuple(float(value) for value in schedule), tuple(rows), status, message)
