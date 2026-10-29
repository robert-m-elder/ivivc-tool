"""Reference crossing-time deadlines and one-additional-point planning.

These policies use uncertainty in complete reference trajectories to constrain
sampling dates. They do not change the information criterion or observation
model, and do not treat reference weights as population event frequencies.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .optimize import optimize_local_i_optimal_designs
from .types import (
    CrossingTimeSummary,
    DesignConstraints,
    NextTimeRecommendation,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
)


def reference_crossing_time_summary(
    reference: ReferenceTrajectoryEnsemble,
    *,
    degraded_fraction: float = 0.5,
    quantile: float = 0.1,
    baseline_time: float = 0.0,
) -> CrossingTimeSummary:
    """Summarize first times reaching a relative degradation milestone.

    The milestone for each draw is ``(1 - degraded_fraction)`` times that
    draw's response at ``baseline_time``. Crossing times are linearly
    interpolated between reference-grid points, including the baseline and
    supported-horizon endpoints. The deadline is the weighted empirical
    inverse-CDF quantile, without interpolation between draws.

    Noncrossing trajectories retain their source weights at positive infinity;
    they are not discarded when calculating the quantile. A nonpositive or
    nonfinite baseline in any positive-weight draw returns invalid_reference.
    """
    degraded_fraction = float(degraded_fraction)
    quantile = float(quantile)
    for value, name in ((degraded_fraction, 'degraded_fraction'), (quantile, 'quantile')):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'{name} must be finite and strictly between zero and one.')
    baseline_time = float(baseline_time)
    lower, upper = reference.supported_time_range
    if not np.isfinite(baseline_time) or not lower <= baseline_time <= upper:
        raise ValueError('baseline_time must lie within the supported reference time range.')

    weights = reference.draw_weights.copy()
    weights[~reference.valid_draws] = 0.0
    weights /= np.sum(weights)
    active = np.flatnonzero(weights > 0)
    crossing_times = np.full(reference.n_draws, np.nan, dtype=float)
    grid = np.union1d(
        [baseline_time, upper],
        reference.time_grid[
            (reference.time_grid > baseline_time) & (reference.time_grid < upper)
        ],
    )
    curves = reference.evaluate(grid, draw_indices=active)
    for source_index, curve in zip(active, curves):
        baseline = float(curve[0])
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        # Normalizing first makes the crossing calculation insensitive to the
        # response scale, including small positive baseline responses.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = curve / baseline
        if np.any(~np.isfinite(normalized)):
            continue
        # Compare degradation itself so very small positive fractions do not
        # round their remaining-response threshold back up to one.
        degradation = 1.0 - normalized
        crossed = np.flatnonzero(degradation >= degraded_fraction)
        if len(crossed) == 0:
            crossing_times[source_index] = np.inf
            continue
        index = int(crossed[0])
        y0, y1 = degradation[index - 1:index + 1]
        scale = max(abs(float(y0)), abs(float(y1)), degraded_fraction)
        fraction = (degraded_fraction / scale - y0 / scale) / (y1 / scale - y0 / scale)
        crossing_times[source_index] = (
            (1.0 - fraction) * grid[index - 1] + fraction * grid[index]
        )

    if np.any(np.isnan(crossing_times[active])):
        status = 'invalid_reference'
        deadline = None
        message = 'A positive-weight trajectory has an invalid baseline or relative response.'
    else:
        order = np.argsort(crossing_times[active], kind='stable')
        cumulative = np.cumsum(weights[active][order])
        cumulative[-1] = 1.0
        index = int(np.searchsorted(cumulative, quantile, side='left'))
        candidate = float(crossing_times[active][order[index]])
        if np.isfinite(candidate):
            status = 'available'
            deadline = candidate
            message = 'A finite reference crossing-time deadline is available.'
        else:
            status = 'beyond_horizon'
            deadline = None
            message = 'Insufficient reference weight reaches the milestone within the supported horizon.'

    return CrossingTimeSummary(
        degraded_fraction=degraded_fraction,
        quantile=quantile,
        baseline_time=baseline_time,
        supported_time_range=reference.supported_time_range,
        crossing_times=crossing_times,
        draw_weights=weights,
        deadline=deadline,
        status=status,
        message=message,
    )


def _eligible_next_times(
    summary: CrossingTimeSummary,
    constraints: DesignConstraints,
    latest_completed_time: float,
) -> tuple[np.ndarray, str, str]:
    latest = float(latest_completed_time)
    lower, upper = summary.supported_time_range
    if (
        not np.isfinite(latest)
        or not summary.baseline_time <= latest <= upper
        or not constraints.min_time <= latest <= constraints.max_time
    ):
        raise ValueError('latest_completed_time must lie within the baseline, reference, and design range.')
    if np.any(constraints.fixed_times > latest):
        raise ValueError('Every fixed time must be completed by latest_completed_time.')
    if constraints.min_time < lower or constraints.max_time > upper:
        raise ValueError('Design constraints extend beyond the timing reference domain.')
    size = len(constraints.fixed_times) + 1
    if not constraints.min_number_of_times <= size <= constraints.max_number_of_times:
        raise ValueError('Design constraints must permit exactly one additional time.')
    if summary.status != 'available':
        return np.array([], dtype=float), summary.status, summary.message
    if summary.deadline <= latest:
        return (
            np.array([], dtype=float), 'deadline_elapsed',
            'The deadline is at or before the latest completed time.',
        )
    available = constraints.available_candidates(constraints.fixed_times)
    eligible = available[(available > latest) & (available <= summary.deadline)]
    if len(eligible) == 0:
        return (
            eligible, 'no_feasible_candidate',
            'No future candidate meets the deadline and existing design constraints.',
        )
    return eligible, 'available', 'A feasible future time is available before the deadline.'


def latest_time_before_deadline(
    summary: CrossingTimeSummary,
    constraints: DesignConstraints,
    *,
    latest_completed_time: float,
) -> NextTimeRecommendation:
    """Choose the latest legal future candidate at or before the deadline.

    Exactly one time is added to the fixed, completed schedule. The returned
    time is never rounded forward past the deadline. Unavailable results have
    an explicit status and no selected schedule.
    """
    eligible, status, message = _eligible_next_times(
        summary, constraints, latest_completed_time,
    )
    if status != 'available':
        return NextTimeRecommendation('direct_quantile', summary, status, message)
    new_time = float(eligible[-1])
    times = constraints.validate_design(np.union1d(constraints.fixed_times, [new_time]))
    return NextTimeRecommendation(
        'direct_quantile', summary, status, message,
        times=times,
        new_time=new_time,
        eligible_times=eligible,
        metadata={'selection_rule': 'latest_candidate_at_or_before_deadline'},
    )


def optimize_next_time_before_deadline(
    summary: CrossingTimeSummary,
    reference_model: ReferenceModelEnsemble,
    observation_model: ObservationModel,
    constraints: DesignConstraints,
    *,
    latest_completed_time: float,
    evaluation_times=None,
    evaluation_weights=None,
    relative_step: float = 1e-5,
    absolute_step: float = 1e-8,
    rank_tolerance: float | None = None,
) -> NextTimeRecommendation:
    """Minimize central I-optimality for one new time before the deadline.

    Only selectable times are restricted. The original reference domain,
    evaluation grid, fixed times, observation model, and rank fallback are
    retained. With one added time, the existing greedy search evaluates every
    eligible candidate; coordinate exchange would repeat that same search.
    """
    if not np.allclose(
        reference_model.supported_time_range, summary.supported_time_range,
        rtol=0.0, atol=1e-12,
    ):
        raise ValueError('The analysis model and timing summary must have the same supported domain.')
    eligible, status, message = _eligible_next_times(
        summary, constraints, latest_completed_time,
    )
    if status != 'available':
        return NextTimeRecommendation('deadline_i_optimal', summary, status, message)
    size = len(constraints.fixed_times) + 1
    restricted = replace(
        constraints,
        candidate_times=eligible,
        min_number_of_times=size,
        max_number_of_times=size,
    )
    optimization = optimize_local_i_optimal_designs(
        reference_model,
        observation_model,
        restricted,
        evaluation_times=evaluation_times,
        evaluation_weights=evaluation_weights,
        relative_step=relative_step,
        absolute_step=absolute_step,
        rank_tolerance=rank_tolerance,
        refine=False,
    )
    step = optimization.design_for_n(size)
    return NextTimeRecommendation(
        'deadline_i_optimal', summary, status, message,
        times=step.times,
        new_time=step.added_time,
        eligible_times=eligible,
        score=step.score,
        metadata={**optimization.metadata, **step.metadata},
    )
