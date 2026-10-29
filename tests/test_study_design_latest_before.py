from __future__ import annotations

from itertools import combinations, permutations
from unittest.mock import patch

import numpy as np
import pytest

from study_design_core import (
    CrossingTimeTarget,
    DesignConstraints,
    LatestBefore,
    ObservationModel,
    PlanningContext,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
    SamplingWindow,
    evaluate_local_i_optimal_design,
    evaluate_timing_requirements,
    latest_time_before_deadline,
    optimize_timing_constrained_designs,
    optimize_window_constrained_designs,
    reference_crossing_time_summary,
    resolve_timing_requirements,
    select_latest_before_times,
)
import study_design_core.requirements as requirements_module


def _reference(upper=10.0, noncrossing=False):
    grid = np.arange(upper + 1.0)
    curve = np.ones_like(grid) if noncrossing else 1.0 - grid / (1.2 * upper)
    return ReferenceTrajectoryEnsemble(grid, curve[None, :], np.array([True]), interpolation='linear')


def _model(upper=10.0, exponential=False):
    parameters = [-0.2] if exponential else [1.0, -0.03]
    return ReferenceModelEnsemble(
        model_name='exponential_normalized' if exponential else 'linear',
        parameter_names=('b',) if exponential else ('a', 'b'),
        parameter_draws=np.array([parameters]), valid_draws=np.array([True]),
        draw_weights=np.array([1.0]), central_parameters=np.array(parameters),
        time_grid=np.linspace(0.0, upper, 21), supported_time_range=(0.0, upper),
    )


def _constraints(upper=10.0, *, fixed=(0.0,), candidates=None, spacing=1.0, min_n=2, max_n=5):
    return DesignConstraints(
        candidate_times=np.arange(upper + 1.0) if candidates is None else np.asarray(candidates),
        fixed_times=np.asarray(fixed), min_time=0.0, max_time=upper,
        minimum_spacing=spacing, min_number_of_times=min_n, max_number_of_times=max_n,
        replicates_per_time=3,
    )


def _resolved(requirements, constraints, completed=(), reference=None):
    return resolve_timing_requirements(
        requirements, _reference(constraints.max_time) if reference is None else reference, constraints,
        context=PlanningContext([0.0], completed),
    )


def _optimize(requirements, constraints, *, completed=(), **kwargs):
    resolved = _resolved(requirements, constraints, completed)
    return optimize_timing_constrained_designs(
        _model(constraints.max_time), ObservationModel('constant_sd', 0.05),
        constraints, resolved, **kwargs,
    )


def _best_assignment_by_enumeration(resolved, constraints, size):
    fixed = tuple(constraints.fixed_times)
    candidates = [time for time in constraints.candidate_times
                  if time > resolved.context.planning_start and time not in fixed]
    anchors = [row for row in resolved.requirements if row.kind == 'latest_before']
    priority = sorted(range(len(anchors)), key=lambda index: (-anchors[index].upper_bound, index))
    best, witnesses = None, []
    for added in combinations(candidates, size - len(fixed)):
        schedule = tuple(sorted(fixed + added))
        if any(right - left < constraints.minimum_spacing for left, right in zip(schedule, schedule[1:])):
            continue
        covered = True
        for row in resolved.requirements:
            if row.kind != 'window':
                continue
            qualifying = [time for time in schedule
                          if row.lower_bound < time <= row.upper_bound
                          and time not in resolved.context.baseline_times
                          and (row.count_scope == 'all' or time not in resolved.context.completed_times)]
            if len(qualifying) < row.minimum_count:
                covered = False
                break
        if not covered:
            continue
        future = [time for time in schedule if time > resolved.context.planning_start
                  and time not in resolved.context.baseline_times]
        for assigned in permutations(future, len(anchors)):
            if not all(row.effective_lower_bound < time <= row.upper_bound
                       for row, time in zip(anchors, assigned)):
                continue
            key = tuple(assigned[index] for index in priority)
            if best is None or key > best:
                best, witnesses = key, [schedule]
            elif key == best:
                witnesses.append(schedule)
    return best, witnesses


@pytest.mark.parametrize('spacing', [0.0, 1.0, 2.0])
@pytest.mark.parametrize('fixed,completed', [
    ((0.0,), ()),
    ((0.0, 2.0), (0.0, 2.0)),
    ((0.0, 2.0, 6.0), (0.0, 2.0)),
])
def test_joint_latest_selection_matches_exhaustive_assignment_search(spacing, fixed, completed):
    constraints = _constraints(
        candidates=[0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], fixed=fixed,
        spacing=spacing, min_n=max(2, len(fixed)), max_n=5,
    )
    for scope in ('new', 'all'):
        for deadlines in ((5.0, 8.0), (8.0, 5.0), (5.0, 5.0)):
            resolved = _resolved([
                LatestBefore(deadlines[0], label='first'),
                SamplingWindow(0.0, 6.0, count_scope=scope, label='coverage'),
                LatestBefore(deadlines[1], label='second'),
            ], constraints, completed)
            for size in range(constraints.min_number_of_times, 6):
                expected, witnesses = _best_assignment_by_enumeration(resolved, constraints, size)
                selected = select_latest_before_times(resolved, constraints, size)
                assert selected.status == ('infeasible' if expected is None else 'feasible')
                if expected is not None:
                    actual = tuple(next(point.selected_time for point in selected.points if point.label == label)
                                   for label in selected.priority_labels)
                    assert actual == expected
                    assert selected.times in witnesses
                    assert len(set(selected.selected_times)) == 2
                else:
                    assert selected.times == ()
                    assert selected.selected_times == ()


def test_interacting_deadlines_are_resolved_later_first_and_report_displacements():
    constraints = _constraints(30.0, spacing=4.0, min_n=3, max_n=3)
    resolved = _resolved([LatestBefore(24.0, label='early'), LatestBefore(25.0, label='late')], constraints)
    selected = select_latest_before_times(resolved, constraints, 3)
    assert selected.priority_labels == ('late', 'early')
    assert selected.times == (0.0, 21.0, 25.0)
    early, late = selected.points
    assert early.selected_time == 21.0
    assert early.latest_eligible_time == 24.0
    assert early.displacement == 3.0
    assert early.deadline_slack == 3.0
    assert late.selected_time == 25.0
    assert late.displacement == 0.0


def test_window_counts_can_force_an_anchor_earlier_than_spacing_alone_requires():
    constraints = _constraints(spacing=2.0, min_n=4, max_n=4)
    resolved = _resolved([
        LatestBefore(9.0, label='early'), LatestBefore(10.0, label='late'),
        SamplingWindow(3.0, 7.0, minimum_count=2, label='two middle observations'),
    ], constraints)
    selected = select_latest_before_times(resolved, constraints, 4)
    assert selected.selected_times == (7.0, 10.0)
    assert len([time for time in selected.times if 3.0 < time <= 7.0]) == 2


def test_equal_deadlines_use_request_order_and_distinct_times():
    constraints = _constraints(spacing=2.0, min_n=3, max_n=3)
    resolved = _resolved([LatestBefore(8.0, label='A'), LatestBefore(8.0, label='B')], constraints)
    selected = select_latest_before_times(resolved, constraints, 3)
    assert selected.priority_labels == ('A', 'B')
    assert [point.selected_time for point in selected.points] == [8.0, 6.0]


def test_three_equal_deadlines_reserve_three_distinct_points():
    constraints = _constraints(spacing=2.0, min_n=4, max_n=4)
    resolved = _resolved([LatestBefore(8.0, label=label) for label in ('A', 'B', 'C')], constraints)
    selected = select_latest_before_times(resolved, constraints, 4)
    assert [point.selected_time for point in selected.points] == [8.0, 6.0, 4.0]
    expected, witnesses = _best_assignment_by_enumeration(resolved, constraints, 4)
    assert expected == (8.0, 6.0, 4.0)
    assert selected.times in witnesses


def test_lower_bounds_and_fractional_deadlines_do_not_round_points_forward():
    constraints = _constraints(spacing=2.0, min_n=4, max_n=4)
    resolved = _resolved([
        LatestBefore(3.5, label='early'),
        LatestBefore(7.5, lower_bound=4.0, label='middle'),
        LatestBefore(10.0, lower_bound=7.0, label='late'),
    ], constraints)
    selected = select_latest_before_times(resolved, constraints, 4)
    assert [point.selected_time for point in selected.points] == [3.0, 7.0, 10.0]
    for point in selected.points:
        assert point.lower_bound < point.selected_time <= point.deadline


def test_anchors_can_also_satisfy_overlapping_windows_without_extra_points():
    constraints = _constraints(min_n=2, max_n=2)
    result = _optimize([
        LatestBefore(8.0, label='milestone'),
        SamplingWindow(5.0, 9.0, label='first window'),
        SamplingWindow(7.0, 10.0, label='second window'),
    ], constraints).design_for_n(2)
    assert result.times == (0.0, 8.0)
    assert result.requirement_evaluation.status == 'satisfied'
    assert [row.qualifying_times for row in result.requirement_evaluation.requirements] == [(8.0,)] * 3


def test_single_selection_matches_the_existing_direct_rule():
    constraints = _constraints(fixed=(0.0, 2.0), min_n=3, max_n=3)
    reference = _reference()
    target = CrossingTimeTarget()
    summary = reference_crossing_time_summary(reference)
    old = latest_time_before_deadline(summary, constraints, latest_completed_time=2.0)
    resolved = _resolved([LatestBefore(target)], constraints, (0.0, 2.0), reference)
    selected = select_latest_before_times(resolved, constraints, 3)
    assert selected.selected_times == (old.new_time,)
    np.testing.assert_array_equal(selected.times, old.times)


def test_requested_full_size_can_change_the_latest_feasible_anchor():
    constraints = _constraints(candidates=[0.0, 3.0, 5.0, 7.0], spacing=3.0, min_n=2, max_n=3)
    result = _optimize([LatestBefore(6.0)], constraints)
    assert result.design_for_n(2).times == (0.0, 5.0)
    assert result.design_for_n(3).times == (0.0, 3.0, 7.0)
    assert result.design_for_n(3).latest_before_selection.selected_times == (3.0,)


def test_existing_future_commitments_can_fill_distinct_requests_once_each():
    constraints = _constraints(fixed=(0.0, 2.0, 7.0, 9.0), min_n=4, max_n=4)
    result = _optimize([
        LatestBefore(8.0), LatestBefore(10.0), SamplingWindow(2.0, 10.0, minimum_count=2),
    ], constraints, completed=(0.0, 2.0)).design_for_n(4)
    assert result.times == (0.0, 2.0, 7.0, 9.0)
    assert result.latest_before_selection.selected_times == (7.0, 9.0)
    assert len(result.times) == len(constraints.fixed_times)
    assert result.requirement_evaluation.requirements[-1].achieved_count == 2


def test_completed_times_cannot_fill_an_expired_latest_request():
    constraints = _constraints(fixed=(0.0, 2.0), min_n=3, max_n=3)
    resolved = _resolved([
        SamplingWindow(0.0, 2.0, count_scope='all'),
        LatestBefore(2.0, label='expired'), LatestBefore(8.0, label='future'),
    ], constraints, (0.0, 2.0))
    selected = select_latest_before_times(resolved, constraints, 3)
    assert selected.status == 'infeasible'
    assert selected.selected_times == ()
    assert len(selected.points) == 2
    assert selected.points[0].status == 'deadline_elapsed'
    assert selected.points[1].status == 'unresolved'


def test_missing_crossing_target_remains_visible_without_partial_selection():
    constraints = _constraints()
    resolved = _resolved([
        LatestBefore(CrossingTimeTarget(), label='unreachable'), LatestBefore(8.0, label='calendar'),
    ], constraints, reference=_reference(noncrossing=True))
    result = optimize_timing_constrained_designs(
        _model(), ObservationModel('constant_sd', 0.05), constraints, resolved, number_of_times=3,
    )
    row = result.result_for_n(3)
    assert row.status == 'unavailable'
    assert row.times == () and row.score is None
    assert row.latest_before_selection.points[0].status == 'beyond_horizon'
    assert row.latest_before_selection.points[1].selected_time is None
    assert result.resolved_requirements is resolved


def test_search_limit_does_not_certify_unestablished_latest_times():
    constraints = _constraints(min_n=3, max_n=3)
    resolved = _resolved([LatestBefore(8.0), LatestBefore(9.0)], constraints)
    selected = select_latest_before_times(resolved, constraints, 3, max_search_states=1)
    assert selected.status == 'search_limit'
    assert selected.states_visited <= 1
    assert selected.times == () and selected.selected_times == ()
    result = optimize_timing_constrained_designs(
        _model(), ObservationModel('constant_sd', 0.05), constraints, resolved,
        max_anchor_search_states=1,
    ).result_for_n(3)
    assert result.status == 'search_limit'
    assert result.score is None
    assert select_latest_before_times(resolved, constraints, 3).status == 'feasible'


def test_information_search_limit_retains_the_joint_selection_witness():
    constraints = _constraints(min_n=4, max_n=4)
    result = _optimize([LatestBefore(8.0)], constraints, max_search_states=1).design_for_n(4)
    assert result.status == 'feasible'
    assert result.optimization_status == 'search_limit'
    assert result.times == result.latest_before_selection.times
    assert result.requirement_evaluation.status == 'satisfied'
    assert result.metadata['latest_before_locations_verified'] is True
    independent = evaluate_local_i_optimal_design(
        result.times, _model(), ObservationModel('constant_sd', 0.05), constraints,
    )
    assert result.score.criterion == pytest.approx(independent.criterion)


def test_limits_at_different_assignment_depths_never_certify_partial_choices():
    constraints = _constraints(spacing=2.0, min_n=3, max_n=3)
    resolved = _resolved([LatestBefore(5.0), LatestBefore(8.0)], constraints)
    unrestricted = select_latest_before_times(resolved, constraints, 3)
    limited = 0
    for budget in range(1, unrestricted.states_visited + 2):
        result = select_latest_before_times(resolved, constraints, 3, max_search_states=budget)
        assert result.states_visited <= budget
        if result.status == 'search_limit':
            limited += 1
            assert result.times == () and result.selected_times == ()
        else:
            assert result.status == 'feasible'
            assert result.selected_times == unrestricted.selected_times
    assert limited > 1


def test_window_only_calls_preserve_the_existing_optimizer_results():
    constraints = _constraints(min_n=3, max_n=4)
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    model, observation = _model(), ObservationModel('constant_sd', 0.05)
    old = optimize_window_constrained_designs(model, observation, constraints, resolved)
    new = optimize_timing_constrained_designs(model, observation, constraints, resolved)
    for before, after in zip(old.designs, new.designs):
        assert before.times == after.times
        assert before.exchanges == after.exchanges
        assert before.score.criterion == pytest.approx(after.score.criterion)
        assert after.latest_before_selection is None
    assert old.metadata == new.metadata


def test_refinement_locks_anchors_and_preserves_original_scoring_domain():
    constraints = _constraints(min_n=5, max_n=5)
    resolved = _resolved([LatestBefore(4.0), SamplingWindow(5.0, 8.0)], constraints)
    observation = ObservationModel('constant_cv', 0.1)
    grid, weights = np.array([0.0, 2.0, 5.0, 10.0]), np.array([2.0, 1.0, 4.0, 3.0])
    optimized = optimize_timing_constrained_designs(
        _model(), observation, constraints, resolved,
        evaluation_times=grid, evaluation_weights=weights,
    )
    result = optimized.design_for_n(5)
    anchor = result.latest_before_selection.selected_times[0]
    assert anchor == 4.0
    assert all(old != anchor for old, new in result.exchanges)
    np.testing.assert_array_equal(optimized.evaluation_times, grid)
    independent = evaluate_local_i_optimal_design(
        result.times, _model(), observation, constraints, evaluation_times=grid, evaluation_weights=weights,
    )
    assert result.score.criterion == pytest.approx(independent.criterion)
    assert result.requirement_evaluation.status == 'satisfied'
    # Generic inspection has no selection proof; the scheduler supplies it.
    assert evaluate_timing_requirements(result.times, resolved, constraints).status == 'not_past_deadline'


def test_latest_timing_has_priority_over_unrestricted_information_optimality():
    constraints = _constraints(min_n=2, max_n=2)
    resolved = _resolved([LatestBefore(9.0)], constraints)
    result = optimize_timing_constrained_designs(
        _model(exponential=True), ObservationModel('constant_sd', 0.05), constraints, resolved,
    ).design_for_n(2)
    assert result.times == (0.0, 9.0)
    earlier = evaluate_local_i_optimal_design(
        [0.0, 5.0], _model(exponential=True), ObservationModel('constant_sd', 0.05), constraints,
    )
    assert earlier.criterion < result.score.criterion


def test_mixed_feasible_and_infeasible_sizes_are_kept_separately():
    constraints = _constraints(min_n=2, max_n=3)
    result = _optimize([LatestBefore(5.0), LatestBefore(8.0)], constraints)
    assert result.result_for_n(2).status == 'infeasible'
    assert result.result_for_n(2).latest_before_selection.selected_times == ()
    assert result.design_for_n(3).latest_before_selection.selected_times == (5.0, 8.0)
    with pytest.raises(KeyError):
        result.design_for_n(2)


def test_crossing_targets_are_not_recomputed_or_reweighted_when_anchors_are_locked():
    constraints = _constraints(min_n=3, max_n=3)
    grid = np.arange(11.0)
    reference = ReferenceTrajectoryEnsemble(
        grid, np.vstack([1.0 - grid / 6.0, 1.0 - grid / 12.0]), np.array([True, True]),
        draw_weights=np.array([0.4, 0.6]), interpolation='linear',
    )
    with patch.object(requirements_module, 'reference_crossing_time_summary',
                      wraps=requirements_module.reference_crossing_time_summary) as summarize:
        resolved = _resolved([
            LatestBefore(CrossingTimeTarget(quantile=0.5)), SamplingWindow(0.0, 9.0),
        ], constraints, reference=reference)
        result = optimize_timing_constrained_designs(
            _model(), ObservationModel('constant_sd', 0.05), constraints, resolved,
        ).design_for_n(3)
        assert summarize.call_count == 1
    assert result.latest_before_selection.selected_times == (6.0,)
    np.testing.assert_array_equal(resolved.requirements[0].upper_summary.draw_weights, [0.4, 0.6])


@pytest.mark.parametrize('value', [True, 0, -1, 1.5])
def test_joint_search_rejects_invalid_limits(value):
    constraints = _constraints()
    resolved = _resolved([LatestBefore(8.0)], constraints)
    with pytest.raises(ValueError):
        select_latest_before_times(resolved, constraints, 3, max_search_states=value)
    with pytest.raises(ValueError):
        optimize_timing_constrained_designs(
            _model(), ObservationModel('constant_sd', 0.05), constraints, resolved,
            max_anchor_search_states=value,
        )


def test_stale_constraints_and_mismatched_domains_are_rejected():
    constraints = _constraints()
    resolved = _resolved([LatestBefore(8.0)], constraints)
    with pytest.raises(ValueError, match='domain'):
        optimize_timing_constrained_designs(
            _model(20.0), ObservationModel('constant_sd', 0.05), constraints, resolved,
        )
    constraints.minimum_spacing = 2.0
    with pytest.raises(ValueError):
        select_latest_before_times(resolved, constraints, 3)


def test_inputs_and_original_fixed_time_accounting_are_not_mutated():
    constraints = _constraints(fixed=(0.0, 2.0), min_n=4, max_n=4)
    resolved = _resolved([LatestBefore(6.0), SamplingWindow(2.0, 6.0)], constraints, (0.0, 2.0))
    before = constraints.fixed_times.copy(), constraints.candidate_times.copy()
    result = optimize_timing_constrained_designs(
        _model(), ObservationModel('constant_sd', 0.05), constraints, resolved,
    ).design_for_n(4)
    np.testing.assert_array_equal(constraints.fixed_times, before[0])
    np.testing.assert_array_equal(constraints.candidate_times, before[1])
    assert resolved.requirements[1].fixed_qualifying_times == ()
    assert 6.0 in result.requirement_evaluation.requirements[1].qualifying_times
