from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

import study_design_core.requirements as requirements_module
from study_design_core import DesignConstraints, ReferenceTrajectoryEnsemble
from study_design_core.requirements import (
    CrossingTimeTarget,
    LatestBefore,
    PlanningContext,
    SamplingWindow,
    evaluate_timing_requirements,
    resolve_timing_requirements,
)


def _reference(crossing_times=(4.0,), weights=None):
    time = np.arange(0.0, 11.0)
    curves = np.vstack([
        np.ones_like(time) if np.isinf(crossing) else 1.0 - time / (2.0 * crossing)
        for crossing in crossing_times
    ])
    return ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=curves,
        valid_draws=np.ones(len(curves), dtype=bool),
        draw_weights=weights,
        interpolation='linear',
    )


def _constraints(*, fixed=(0.0, 2.0), candidates=None, spacing=1.0, max_n=7):
    return DesignConstraints(
        candidate_times=np.arange(11.0) if candidates is None else np.asarray(candidates),
        fixed_times=np.asarray(fixed),
        min_time=0.0,
        max_time=10.0,
        minimum_spacing=spacing,
        min_number_of_times=max(1, len(fixed)),
        max_number_of_times=max_n,
        replicates_per_time=3,
    )


def _resolve(requests, *, reference=None, constraints=None, context=None):
    return resolve_timing_requirements(
        requests,
        _reference() if reference is None else reference,
        _constraints() if constraints is None else constraints,
        context=(
            PlanningContext(baseline_times=(0.0,), completed_times=(0.0, 2.0))
            if context is None else context
        ),
    )


def test_planning_context_copies_and_canonicalizes_input_times():
    baseline = np.array([0.0])
    completed = np.array([2.0, 0.0, 2.0])
    context = PlanningContext(baseline_times=baseline, completed_times=completed)
    baseline[0] = 8.0
    completed[:] = 9.0
    assert context.baseline_times == (0.0,)
    assert context.completed_times == (0.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        context.planning_start = 4.0
    with pytest.raises(FrozenInstanceError):
        SamplingWindow(2.0, 5.0).minimum_count = 2


@pytest.mark.parametrize('field,value', [
    ('degraded_fraction', 0.0),
    ('degraded_fraction', 1.0),
    ('degraded_fraction', np.nan),
    ('quantile', 0.0),
    ('quantile', 1.0),
    ('quantile', np.inf),
    ('baseline_time', np.nan),
])
def test_crossing_targets_validate_inputs(field, value):
    with pytest.raises(ValueError):
        CrossingTimeTarget(**{field: value})


@pytest.mark.parametrize('count', [True, 0, -1, 1.5, np.nan])
def test_window_count_requires_a_positive_integer(count):
    with pytest.raises((TypeError, ValueError)):
        SamplingWindow(2.0, 5.0, minimum_count=count)


@pytest.mark.parametrize('make_requirement', [
    lambda: SamplingWindow(np.nan, 5.0),
    lambda: SamplingWindow(2.0, np.inf),
    lambda: SamplingWindow(2.0, 5.0, count_scope='reference'),
    lambda: LatestBefore(np.inf),
])
def test_malformed_requirement_inputs_are_rejected(make_requirement):
    with pytest.raises((TypeError, ValueError)):
        make_requirement()


@pytest.mark.parametrize('context', [
    PlanningContext(baseline_times=(1.0,), completed_times=(0.0, 2.0)),
    PlanningContext(baseline_times=(0.0,), completed_times=(0.0, 3.0)),
    PlanningContext(baseline_times=(0.0,), completed_times=(0.0, 2.0), planning_start=1.0),
    PlanningContext(baseline_times=(0.0,), completed_times=(0.0, 2.0), planning_start=11.0),
])
def test_context_roles_and_planning_start_must_match_the_design(context):
    with pytest.raises(ValueError):
        _resolve([SamplingWindow(2.0, 5.0)], context=context)


def test_prospective_start_does_not_invent_completed_observations():
    constraints = _constraints(fixed=(0.0,))
    resolved = _resolve(
        [SamplingWindow(0.0, 4.0)], constraints=constraints,
        context=PlanningContext(baseline_times=(0.0,)),
    )
    assert resolved.context.completed_times == ()
    assert resolved.context.planning_start == 0.0
    assert resolved.requirements[0].fixed_qualifying_times == ()
    assert resolved.requirements[0].status == 'ready'
    assert np.all(np.asarray(resolved.candidate_times) > 0.0)


def test_explicit_planning_start_filters_candidates_strictly():
    resolved = _resolve(
        [SamplingWindow(0.0, 6.0)],
        context=PlanningContext(
            baseline_times=(0.0,), completed_times=(0.0, 2.0), planning_start=4.0,
        ),
    )
    assert resolved.context.planning_start == 4.0
    np.testing.assert_array_equal(resolved.requirements[0].eligible_times, [5.0, 6.0])
    assert resolved.requirements[0].lower_bound == 0.0
    assert resolved.requirements[0].effective_lower_bound == 4.0


def test_unclassified_fixed_past_time_is_not_treated_as_completed():
    with pytest.raises(ValueError):
        _resolve(
            [SamplingWindow(4.0, 8.0)],
            constraints=_constraints(fixed=(0.0, 2.0, 3.0)),
            context=PlanningContext(
                baseline_times=(0.0,), completed_times=(0.0, 2.0), planning_start=4.0,
            ),
        )


def test_windows_use_an_open_lower_and_closed_upper_bound():
    constraints = _constraints()
    resolved = _resolve([SamplingWindow(3.0, 5.0)], constraints=constraints)
    np.testing.assert_array_equal(resolved.requirements[0].eligible_times, [4.0, 5.0])
    evaluated = evaluate_timing_requirements([0.0, 2.0, 3.0, 5.0], resolved, constraints)
    row = evaluated.requirements[0]
    assert evaluated.status == 'satisfied'
    assert row.qualifying_times == (5.0,)
    assert row.achieved_count == 1


def test_new_and_whole_study_scopes_count_different_target_study_times():
    constraints = _constraints()
    resolved = _resolve([
        SamplingWindow(0.0, 4.0, count_scope='new', label='additional'),
        SamplingWindow(0.0, 4.0, count_scope='all', label='coverage'),
    ], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0], resolved, constraints)
    assert evaluated.requirements[0].qualifying_times == (4.0,)
    assert evaluated.requirements[1].qualifying_times == (2.0, 4.0)
    assert resolved.requirements[0].fixed_qualifying_times == ()
    assert resolved.requirements[1].fixed_qualifying_times == (2.0,)


def test_baseline_is_excluded_even_from_whole_study_coverage():
    constraints = _constraints()
    resolved = _resolve(
        [SamplingWindow(0.0, 4.0, count_scope='all')], constraints=constraints,
        context=PlanningContext(baseline_times=(0.0, 2.0), completed_times=(0.0, 2.0)),
    )
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0], resolved, constraints)
    assert evaluated.requirements[0].qualifying_times == (4.0,)
    assert resolved.requirements[0].fixed_qualifying_times == ()


def test_fixed_future_points_count_as_new_sampling():
    constraints = _constraints(fixed=(0.0, 2.0, 6.0))
    resolved = _resolve([SamplingWindow(2.0, 6.0)], constraints=constraints)
    row = resolved.requirements[0]
    assert row.fixed_qualifying_times == (6.0,)
    assert row.status == 'already_satisfied'
    evaluated = evaluate_timing_requirements([0.0, 2.0, 6.0], resolved, constraints)
    assert evaluated.requirements[0].qualifying_times == (6.0,)


def test_historical_reference_grid_points_do_not_count_as_observations():
    constraints = _constraints()
    resolved = _resolve([SamplingWindow(2.0, 8.0, minimum_count=2)], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0], resolved, constraints)
    assert evaluated.status == 'unsatisfied'
    assert evaluated.requirements[0].achieved_count == 1
    assert evaluated.requirements[0].qualifying_times == (4.0,)


def test_crossing_bounds_use_original_weights_and_keep_noncrossers():
    reference = _reference((4.0, 8.0, np.inf), weights=[0.2, 0.3, 0.5])
    target = CrossingTimeTarget(quantile=0.5)
    resolved = _resolve([SamplingWindow(2.0, target)], reference=reference)
    row = resolved.requirements[0]
    assert row.upper_bound == 8.0
    np.testing.assert_allclose(row.upper_summary.draw_weights, [0.2, 0.3, 0.5])
    assert np.isposinf(row.upper_summary.crossing_times[-1])
    assert row.upper_summary.noncrossing_weight == 0.5
    unavailable = _resolve(
        [LatestBefore(CrossingTimeTarget(quantile=0.51))], reference=reference,
    )
    assert unavailable.status == 'unavailable'
    assert unavailable.requirements[0].status == 'beyond_horizon'
    assert unavailable.requirements[0].upper_bound is None


def test_equal_crossing_targets_are_resolved_once_per_request():
    with patch.object(
        requirements_module, 'reference_crossing_time_summary',
        wraps=requirements_module.reference_crossing_time_summary,
    ) as summarize:
        resolved = _resolve([
            SamplingWindow(0.0, CrossingTimeTarget(quantile=0.5)),
            SamplingWindow(CrossingTimeTarget(quantile=0.5), 9.0),
            LatestBefore(CrossingTimeTarget(quantile=0.5)),
        ])
    assert summarize.call_count == 1
    assert resolved.requirements[0].upper_summary is resolved.requirements[1].lower_summary
    assert resolved.requirements[0].upper_summary is resolved.requirements[2].upper_summary


def test_invalid_positive_weight_reference_baseline_is_retained_as_a_diagnostic():
    reference = _reference((4.0, 8.0), weights=[0.9, 0.1])
    reference.value_draws[1] = 0.0
    resolved = _resolve([LatestBefore(CrossingTimeTarget())], reference=reference)
    assert resolved.status == 'unavailable'
    assert resolved.requirements[0].status == 'invalid_reference'
    assert resolved.requirements[0].upper_summary.invalid_weight == pytest.approx(0.1)


@pytest.mark.parametrize('lower,upper', [(5.0, 4.0), (4.0, 4.0)])
def test_reversed_or_empty_resolved_windows_remain_visible(lower, upper):
    resolved = _resolve([
        SamplingWindow(lower, upper, label='invalid'),
        SamplingWindow(2.0, 6.0, label='valid'),
    ])
    assert resolved.status == 'unavailable'
    assert len(resolved.requirements) == 2
    assert resolved.requirements[0].status == 'invalid_window'
    assert resolved.requirements[1].status == 'ready'


def test_elapsed_deadline_can_be_covered_but_cannot_request_new_sampling():
    resolved = _resolve([
        SamplingWindow(0.0, 2.0, count_scope='all', label='covered'),
        SamplingWindow(0.0, 2.0, count_scope='new', label='additional'),
        LatestBefore(2.0, label='late'),
    ])
    assert resolved.requirements[0].status == 'already_satisfied'
    assert resolved.requirements[1].status == 'deadline_elapsed'
    assert resolved.requirements[2].status == 'deadline_elapsed'
    assert resolved.status == 'unavailable'


def test_explicit_reversed_past_window_is_invalid_rather_than_elapsed():
    resolved = _resolve([
        SamplingWindow(2.0, 1.0, label='reversed window'),
        LatestBefore(1.0, lower_bound=2.0, label='reversed latest window'),
        LatestBefore(1.0, label='elapsed implicit window'),
    ])
    assert [row.status for row in resolved.requirements] == [
        'invalid_window', 'invalid_window', 'deadline_elapsed',
    ]


def test_grid_gaps_and_insufficient_capacity_have_distinct_diagnostics():
    constraints = _constraints(candidates=[0.0, 2.0, 5.0, 8.0, 10.0], max_n=5)
    resolved = _resolve([
        SamplingWindow(2.0, 4.0, label='gap'),
        SamplingWindow(2.0, 5.0, minimum_count=2, label='short'),
    ], constraints=constraints)
    assert resolved.requirements[0].status == 'no_feasible_candidate'
    assert resolved.requirements[1].status == 'insufficient_candidates'
    assert resolved.requirements[1].candidate_count_upper_bound == 1


def test_fixed_time_spacing_is_applied_to_eligible_candidates():
    constraints = _constraints(fixed=(0.0, 2.0, 6.0), spacing=2.0)
    resolved = _resolve([SamplingWindow(2.0, 8.0)], constraints=constraints)
    np.testing.assert_array_equal(resolved.requirements[0].eligible_times, [4.0, 8.0])
    assert resolved.requirements[0].fixed_qualifying_times == (6.0,)


def test_candidate_capacity_respects_the_remaining_schedule_budget():
    resolved = _resolve(
        [SamplingWindow(2.0, 9.0, minimum_count=2)],
        constraints=_constraints(max_n=3),
    )
    assert resolved.requirements[0].candidate_count_upper_bound == 1
    assert resolved.requirements[0].status == 'insufficient_candidates'


def test_unavailable_capacity_diagnostic_still_reports_observed_coverage():
    constraints = _constraints(fixed=(0.0, 2.0, 6.0), max_n=4)
    resolved = _resolve(
        [SamplingWindow(2.0, 8.0, minimum_count=3)], constraints=constraints,
    )
    assert resolved.requirements[0].candidate_count_upper_bound == 2
    assert resolved.requirements[0].status == 'insufficient_candidates'
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0, 6.0], resolved, constraints)
    assert evaluated.status == 'unavailable'
    assert evaluated.requirements[0].qualifying_times == (4.0, 6.0)
    assert evaluated.requirements[0].achieved_count == 2


def test_individually_ready_requirements_do_not_claim_joint_feasibility():
    resolved = _resolve([
        SamplingWindow(2.0, 4.0, label='early'),
        SamplingWindow(6.0, 8.0, label='late'),
    ], constraints=_constraints(max_n=3))
    assert resolved.status == 'ready'
    assert [row.status for row in resolved.requirements] == ['ready', 'ready']
    # Only one additional time is allowed, so these two disjoint requests
    # cannot jointly be satisfied. Patch 1 resolves requirements, not schedules.
    assert all(row.candidate_count_upper_bound == 1 for row in resolved.requirements)


def test_labels_are_stable_and_duplicates_are_rejected():
    resolved = _resolve([SamplingWindow(2.0, 5.0), LatestBefore(8.0)])
    assert [row.label for row in resolved.requirements] == ['window_1', 'latest_before_2']
    with pytest.raises(ValueError):
        _resolve([SamplingWindow(2.0, 5.0, label='same'), LatestBefore(8.0, label='same')])


def test_calendar_bounds_may_extend_beyond_the_reference_without_extrapolating():
    resolved = _resolve([
        SamplingWindow(-1.0, 12.0, label='wide'),
        LatestBefore(12.0, label='later'),
        LatestBefore(-1.0, label='elapsed'),
    ])
    assert resolved.requirements[0].lower_bound == -1.0
    assert resolved.requirements[0].upper_bound == 12.0
    assert resolved.requirements[1].upper_bound == 12.0
    assert max(resolved.requirements[1].eligible_times) == 10.0
    assert resolved.requirements[2].status == 'deadline_elapsed'
    assert all(0.0 <= time <= 10.0 for time in resolved.candidate_times)


def test_crossing_target_baseline_must_still_lie_in_the_reference_domain():
    with pytest.raises(ValueError):
        _resolve([LatestBefore(CrossingTimeTarget(baseline_time=-1.0))])


def test_schedule_inspection_rejects_changed_design_constraints():
    constraints = _constraints()
    resolved = _resolve([SamplingWindow(2.0, 6.0)], constraints=constraints)
    constraints.minimum_spacing = 2.0
    with pytest.raises(ValueError):
        evaluate_timing_requirements([0.0, 2.0, 4.0], resolved, constraints)


def test_schedule_evaluation_requires_future_eligibility_and_design_validity():
    constraints = _constraints()
    resolved = _resolve(
        [SamplingWindow(4.0, 8.0)], constraints=constraints,
        context=PlanningContext(
            baseline_times=(0.0,), completed_times=(0.0, 2.0), planning_start=4.0,
        ),
    )
    with pytest.raises(ValueError):
        evaluate_timing_requirements([0.0, 2.0, 3.0, 5.0], resolved, constraints)
    with pytest.raises(ValueError):
        evaluate_timing_requirements([0.0, 5.0], resolved, constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 5.0], resolved, constraints)
    assert evaluated.status == 'satisfied'


def test_latest_before_evaluation_checks_timing_without_claiming_latestness():
    constraints = _constraints()
    resolved = _resolve([LatestBefore(8.0)], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0, 10.0], resolved, constraints)
    assert evaluated.status == 'not_past_deadline'
    row = evaluated.requirements[0]
    assert row.status == 'not_past_deadline'
    assert row.qualifying_times == (4.0,)
    assert row.deadline_slack == 4.0
    # Time 8 is eligible, so accepting time 4 cannot establish latestness.
    assert 8.0 in resolved.requirements[0].eligible_times


def test_overlapping_latest_requests_do_not_claim_distinct_anchor_assignments():
    constraints = _constraints()
    resolved = _resolve([LatestBefore(4.0), LatestBefore(5.0)], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 3.0], resolved, constraints)
    assert evaluated.status == 'not_past_deadline'
    assert [row.qualifying_times for row in evaluated.requirements] == [(3.0,), (3.0,)]


def test_counts_are_distinct_time_points_and_do_not_scale_with_replicates():
    constraints = _constraints()
    resolved = _resolve([SamplingWindow(2.0, 6.0, minimum_count=2)], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0, 4.0], resolved, constraints)
    assert evaluated.requirements[0].achieved_count == 1
    assert evaluated.status == 'unsatisfied'
    assert constraints.replicates_per_time == 3


def test_empty_requirements_and_input_constraints_remain_unchanged():
    constraints = _constraints()
    original_candidates = constraints.candidate_times.copy()
    original_fixed = constraints.fixed_times.copy()
    resolved = _resolve([], constraints=constraints)
    evaluated = evaluate_timing_requirements([0.0, 2.0, 4.0], resolved, constraints)
    assert resolved.status == 'ready'
    assert resolved.requirements == ()
    assert evaluated.status == 'satisfied'
    np.testing.assert_array_equal(constraints.candidate_times, original_candidates)
    np.testing.assert_array_equal(constraints.fixed_times, original_fixed)
