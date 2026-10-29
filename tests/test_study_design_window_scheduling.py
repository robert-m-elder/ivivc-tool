from __future__ import annotations

from itertools import combinations
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
    optimize_local_i_optimal_designs,
    optimize_next_time_before_deadline,
    reference_crossing_time_summary,
    resolve_timing_requirements,
)
from study_design_core.feasibility import (
    ScheduleFeasibilityResult,
    check_schedule_feasibility,
)
import study_design_core.scheduling as scheduling
from study_design_core.scheduling import optimize_window_constrained_designs


def _reference(*, noncrossing=False):
    time = np.arange(11.0)
    curve = np.ones_like(time) if noncrossing else 1.0 - time / 12.0
    return ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=curve[None, :],
        valid_draws=np.array([True]),
        interpolation='linear',
    )


def _model(*, normalized_exponential=False):
    if normalized_exponential:
        name, parameters, names = 'exponential_normalized', [-0.2], ('b',)
    else:
        name, parameters, names = 'linear', [1.0, -0.05], ('a', 'b')
    return ReferenceModelEnsemble(
        model_name=name,
        parameter_names=names,
        parameter_draws=np.array([parameters]),
        valid_draws=np.array([True]),
        draw_weights=np.array([1.0]),
        central_parameters=np.array(parameters),
        time_grid=np.linspace(0.0, 10.0, 21),
        supported_time_range=(0.0, 10.0),
    )


def _constraints(*, candidates=None, fixed=(0.0, 2.0), spacing=1.0, min_n=3, max_n=5):
    return DesignConstraints(
        candidate_times=np.arange(11.0) if candidates is None else np.asarray(candidates),
        fixed_times=np.asarray(fixed),
        min_time=0.0,
        max_time=10.0,
        minimum_spacing=spacing,
        min_number_of_times=min_n,
        max_number_of_times=max_n,
        replicates_per_time=3,
    )


def _resolved(requirements, constraints, *, completed=(0.0, 2.0), reference=None):
    return resolve_timing_requirements(
        requirements,
        _reference() if reference is None else reference,
        constraints,
        context=PlanningContext(baseline_times=(0.0,), completed_times=completed),
    )


def _optimize(requirements, constraints=None, *, completed=(0.0, 2.0), **kwargs):
    constraints = _constraints() if constraints is None else constraints
    resolved = _resolved(requirements, constraints, completed=completed)
    return optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1),
        constraints, resolved, **kwargs,
    )


def _brute_force_schedules(constraints, resolved, size, required=()):
    fixed = set(map(float, constraints.fixed_times))
    available = [
        float(value) for value in constraints.candidate_times
        if value > resolved.context.planning_start and value not in fixed
    ]
    number_new = size - len(fixed)
    if number_new < 0:
        return []
    schedules = []
    for added in combinations(available, number_new):
        schedule = tuple(sorted(fixed.union(added)))
        if not set(required).issubset(schedule):
            continue
        if any(right - left < constraints.minimum_spacing for left, right in zip(schedule, schedule[1:])):
            continue
        meets_all = True
        for row in resolved.requirements:
            if row.lower_bound is None or row.upper_bound is None or row.lower_bound >= row.upper_bound:
                meets_all = False
                break
            qualifying = [
                value for value in schedule
                if row.lower_bound < value <= row.upper_bound
                and value not in resolved.context.baseline_times
                and (row.count_scope == 'all' or value not in resolved.context.completed_times)
            ]
            if len(qualifying) < row.minimum_count:
                meets_all = False
                break
        if meets_all:
            schedules.append(schedule)
    return schedules


@pytest.mark.parametrize('spacing', [0.0, 1.0, 2.0])
@pytest.mark.parametrize('fixed,completed', [
    ((0.0,), ()),
    ((0.0, 2.0), (0.0, 2.0)),
    ((0.0, 2.0, 6.0), (0.0, 2.0)),
])
def test_feasibility_matches_brute_force_on_small_grids(spacing, fixed, completed):
    constraints = _constraints(
        candidates=[0.0, 2.0, 3.0, 4.0, 6.0, 8.0], fixed=fixed,
        spacing=spacing, min_n=max(1, len(fixed)), max_n=5,
    )
    for scope in ('new', 'all'):
        for quota in (1, 2):
            requirements = [
                SamplingWindow(0.0, 4.0, quota, scope, label='early'),
                SamplingWindow(2.0, 8.0, 1, scope, label='overlap'),
            ]
            resolved = _resolved(requirements, constraints, completed=completed)
            for size in range(constraints.min_number_of_times, 6):
                expected = _brute_force_schedules(constraints, resolved, size)
                result = check_schedule_feasibility(resolved, constraints, size)
                assert result.status == ('feasible' if expected else 'infeasible')
                if expected:
                    assert result.times in expected
                    assert len(result.times) == size
                else:
                    assert result.times == ()


def test_individually_ready_windows_can_be_jointly_infeasible():
    constraints = _constraints(max_n=3)
    resolved = _resolved([
        SamplingWindow(2.0, 4.0, label='early'),
        SamplingWindow(7.0, 9.0, label='late'),
    ], constraints)
    assert resolved.status == 'ready'
    result = check_schedule_feasibility(resolved, constraints, 3)
    assert result.status == 'infeasible'
    assert result.times == ()


def test_feasibility_requires_exact_size_even_after_window_is_covered():
    constraints = _constraints(candidates=[0.0, 2.0, 4.0, 6.0], spacing=3.0,
                               fixed=(0.0,), min_n=2, max_n=4)
    resolved = _resolved([SamplingWindow(0.0, 6.0)], constraints, completed=())
    assert check_schedule_feasibility(resolved, constraints, 2).status == 'feasible'
    assert check_schedule_feasibility(resolved, constraints, 4).status == 'infeasible'


def test_required_partial_points_are_retained_in_the_feasible_completion():
    constraints = _constraints()
    resolved = _resolved([SamplingWindow(6.0, 8.0)], constraints)
    result = check_schedule_feasibility(resolved, constraints, 4, required_times=(4.0,))
    assert result.status == 'feasible'
    assert 4.0 in result.times
    assert result.times in _brute_force_schedules(constraints, resolved, 4, required=(4.0,))


def test_search_limit_does_not_report_unproven_infeasibility():
    constraints = _constraints()
    resolved = _resolved([SamplingWindow(2.0, 9.0, minimum_count=2)], constraints)
    result = check_schedule_feasibility(resolved, constraints, 5, max_search_states=1)
    assert result.status == 'search_limit'
    assert result.times == ()
    assert check_schedule_feasibility(resolved, constraints, 5).status == 'feasible'


@pytest.mark.parametrize('size', [True, 3.5, 0, 99])
def test_feasibility_rejects_invalid_final_sizes(size):
    constraints = _constraints()
    resolved = _resolved([], constraints)
    with pytest.raises((TypeError, ValueError)):
        check_schedule_feasibility(resolved, constraints, size)


@pytest.mark.parametrize('limit', [True, 0, -1, 1.5])
def test_feasibility_rejects_invalid_search_limits(limit):
    constraints = _constraints()
    resolved = _resolved([], constraints)
    with pytest.raises((TypeError, ValueError)):
        check_schedule_feasibility(resolved, constraints, 3, max_search_states=limit)


def test_feasibility_rejects_unknown_or_elapsed_required_times():
    constraints = _constraints()
    resolved = _resolved([], constraints)
    for required in ((3.25,), (1.0,), (np.nan,)):
        with pytest.raises(ValueError):
            check_schedule_feasibility(resolved, constraints, 3, required_times=required)


def test_changed_constraints_cannot_reuse_stale_resolution():
    constraints = _constraints()
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    constraints.minimum_spacing = 2.0
    with pytest.raises(ValueError):
        check_schedule_feasibility(resolved, constraints, 3)
    with pytest.raises(ValueError):
        optimize_window_constrained_designs(
            _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
        )


def test_ambiguous_candidate_grid_is_rejected_before_information_matching():
    constraints = _constraints(candidates=[0.0, 2.0, 3.0, 3.0 + 5e-13, 4.0, 6.0])
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    with pytest.raises(ValueError, match='tolerance|ambiguous'):
        check_schedule_feasibility(resolved, constraints, 3)
    with pytest.raises(ValueError, match='tolerance|ambiguous'):
        optimize_window_constrained_designs(
            _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
        )


def test_optimizer_preserves_the_reference_domain_even_for_a_smaller_candidate_range():
    constraints = DesignConstraints(
        candidate_times=np.arange(9.0), fixed_times=np.array([0.0, 2.0]),
        min_time=0.0, max_time=8.0, min_number_of_times=3, max_number_of_times=3,
    )
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    model = _model()
    model.supported_time_range = (0.0, 8.0)
    with pytest.raises(ValueError, match='domain'):
        optimize_window_constrained_designs(
            model, ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
        )


def test_latest_before_requests_are_rejected_until_anchor_scheduling_is_available():
    constraints = _constraints()
    resolved = _resolved([LatestBefore(8.0)], constraints)
    with pytest.raises(ValueError, match='LatestBefore|latest|anchor'):
        check_schedule_feasibility(resolved, constraints, 3)
    with pytest.raises(ValueError, match='LatestBefore|latest|anchor'):
        optimize_window_constrained_designs(
            _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
        )


def test_unavailable_crossing_target_is_not_silently_dropped():
    constraints = _constraints()
    resolved = _resolved(
        [SamplingWindow(2.0, CrossingTimeTarget())], constraints,
        reference=_reference(noncrossing=True),
    )
    feasible = check_schedule_feasibility(resolved, constraints, 3)
    assert feasible.status == 'unavailable'
    optimized = optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1),
        constraints, resolved, number_of_times=3,
    )
    assert optimized.resolved_requirements is resolved
    row = optimized.result_for_n(3)
    assert row.status == 'unavailable'
    assert row.times == ()
    assert row.score is None


def test_completion_checks_avoid_a_greedy_choice_that_blocks_required_coverage():
    constraints = _constraints(candidates=[0.0, 3.5, 7.5, 10.0], fixed=(0.0,),
                               spacing=3.5, min_n=3, max_n=3)
    legacy = optimize_local_i_optimal_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, refine=False,
    ).design_for_n(3)
    assert 10.0 in legacy.times
    result = _optimize(
        [SamplingWindow(7.0, 8.0)], constraints, completed=(), refine=False,
    ).design_for_n(3)
    assert result.status == 'feasible'
    assert result.times == (0.0, 3.5, 7.5)
    assert result.requirement_evaluation.status == 'satisfied'


def test_prospective_sizes_are_optimized_independently_and_can_be_nonnested():
    constraints = _constraints(candidates=[0.0, 3.0, 5.0, 7.0], fixed=(0.0,),
                               spacing=3.0, min_n=2, max_n=3)
    resolved = _resolved([SamplingWindow(0.0, 10.0)], constraints, completed=())
    result = optimize_window_constrained_designs(
        _model(normalized_exponential=True), ObservationModel(kind='constant_sd', value=0.1),
        constraints, resolved, number_of_times=(2, 3),
    )
    assert result.design_for_n(2).times == (0.0, 5.0)
    assert result.design_for_n(3).times == (0.0, 3.0, 7.0)


def test_mixed_feasible_and_infeasible_sizes_remain_visible():
    result = _optimize([SamplingWindow(2.0, 4.0, minimum_count=2)], number_of_times=(3, 4))
    assert result.result_for_n(3).status == 'infeasible'
    assert result.result_for_n(3).score is None
    assert result.result_for_n(3).times == ()
    assert result.design_for_n(4).status == 'feasible'
    with pytest.raises(KeyError):
        result.design_for_n(3)


def test_adaptive_multiple_points_preserve_completed_and_fixed_future_times():
    constraints = _constraints(fixed=(0.0, 2.0, 7.0), min_n=5, max_n=5)
    resolved = _resolved([
        SamplingWindow(2.0, 5.0, label='early'),
        SamplingWindow(5.0, 8.0, label='fixed future'),
    ], constraints)
    result = optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
    ).design_for_n(5)
    assert set(constraints.fixed_times).issubset(result.times)
    assert len(result.times) == 5
    assert all(time > 2.0 for time in result.times if time not in constraints.fixed_times)
    assert result.requirement_evaluation.requirements[1].qualifying_times.count(7.0) == 1


def test_completed_coverage_can_satisfy_an_elapsed_window_without_new_past_points():
    constraints = _constraints(min_n=4, max_n=4)
    result = _optimize([
        SamplingWindow(0.0, 2.0, count_scope='all'),
    ], constraints).design_for_n(4)
    assert result.requirement_evaluation.requirements[0].qualifying_times == (2.0,)
    assert len(result.times) == 4
    assert all(time > 2.0 for time in result.times if time not in constraints.fixed_times)
    extra_required = _optimize([
        SamplingWindow(0.0, 2.0, count_scope='new'),
    ], constraints).result_for_n(4)
    assert extra_required.status == 'infeasible'
    assert extra_required.times == ()


def test_one_new_point_matches_the_existing_deadline_constrained_optimizer():
    constraints = _constraints(min_n=3, max_n=3)
    reference = _reference()
    summary = reference_crossing_time_summary(reference, degraded_fraction=0.5, quantile=0.1)
    old = optimize_next_time_before_deadline(
        summary, _model(), ObservationModel(kind='constant_sd', value=0.1),
        constraints, latest_completed_time=2.0,
    )
    resolved = _resolved([SamplingWindow(2.0, CrossingTimeTarget())], constraints, reference=reference)
    new = optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1),
        constraints, resolved, refine=False,
    ).design_for_n(3)
    np.testing.assert_array_equal(new.times, old.times)
    assert new.score.criterion == pytest.approx(old.score.criterion)


def test_information_horizon_and_custom_evaluation_weights_are_preserved():
    constraints = _constraints(min_n=3, max_n=3)
    resolved = _resolved([SamplingWindow(2.0, 4.0)], constraints)
    evaluation_times = np.array([0.0, 2.0, 5.0, 10.0])
    weights = np.array([2.0, 1.0, 4.0, 3.0])
    result = optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1),
        constraints, resolved, evaluation_times=evaluation_times, evaluation_weights=weights,
    )
    np.testing.assert_array_equal(result.evaluation_times, evaluation_times)
    scores = [
        evaluate_local_i_optimal_design(
            [0.0, 2.0, candidate], _model(), ObservationModel(kind='constant_sd', value=0.1),
            constraints, evaluation_times=evaluation_times, evaluation_weights=weights,
        ).criterion
        for candidate in (3.0, 4.0)
    ]
    assert result.design_for_n(3).score.criterion == pytest.approx(min(scores))
    assert result.evaluation_times[-1] > 4.0


def test_refinement_preserves_coverage_and_does_not_worsen_information_score():
    constraints = _constraints(min_n=5, max_n=5)
    resolved = _resolved([
        SamplingWindow(2.0, 5.0, minimum_count=2, label='early'),
        SamplingWindow(6.0, 9.0, label='late'),
    ], constraints)
    result = optimize_window_constrained_designs(
        _model(), ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
    ).design_for_n(5)
    initial_score = evaluate_local_i_optimal_design(
        result.initial_times, _model(), ObservationModel(kind='constant_sd', value=0.1), constraints,
    )
    assert result.score.criterion <= initial_score.criterion + 1e-12
    current = set(result.initial_times)
    for old_time, new_time in result.exchanges:
        current.remove(old_time)
        current.add(new_time)
        assert evaluate_timing_requirements(sorted(current), resolved, constraints).status == 'satisfied'
    assert tuple(sorted(current)) == result.times
    assert result.requirement_evaluation.status == 'satisfied'


@pytest.mark.parametrize('normalized_exponential', [False, True])
def test_unrefined_result_is_no_worse_than_the_known_complete_feasible_witness(normalized_exponential):
    constraints = _constraints(min_n=4, max_n=5)
    observation = ObservationModel(kind='constant_sd', value=0.1)
    model = _model(normalized_exponential=normalized_exponential)
    for requirements in (
        [SamplingWindow(2.0, 5.0, minimum_count=2)],
        [SamplingWindow(2.0, 5.0, label='early'), SamplingWindow(6.0, 10.0, label='late')],
    ):
        resolved = _resolved(requirements, constraints)
        optimized = optimize_window_constrained_designs(
            model, observation, constraints, resolved, refine=False,
        )
        for size in (4, 5):
            witness = check_schedule_feasibility(resolved, constraints, size)
            assert witness.status == 'feasible'
            witness_score = evaluate_local_i_optimal_design(
                witness.times, model, observation, constraints,
            )
            row = optimized.design_for_n(size)
            assert row.score.is_full_rank
            assert row.score.criterion <= witness_score.criterion + 1e-12


def test_full_schedule_score_can_prefer_the_feasibility_witness_over_greedy_additions():
    constraints = _constraints(
        candidates=[0.0, 4.0, 5.0, 6.0, 7.0], fixed=(0.0,),
        spacing=2.0, min_n=3, max_n=3,
    )
    resolved = _resolved([SamplingWindow(0.0, 10.0, minimum_count=2)], constraints, completed=())
    model = _model(normalized_exponential=True)
    observation = ObservationModel(kind='constant_sd', value=0.1)
    row = optimize_window_constrained_designs(
        model, observation, constraints, resolved, refine=False,
    ).design_for_n(3)
    greedy_score = evaluate_local_i_optimal_design([0.0, 5.0, 7.0], model, observation, constraints)
    assert row.times == (0.0, 4.0, 6.0)
    assert row.score.criterion < greedy_score.criterion
    assert row.metadata['seed_selection'] == 'initial_feasibility_witness'


def test_fixed_only_rank_deficient_schedule_retains_rank_fallback():
    constraints = _constraints(fixed=(0.0,), min_n=1, max_n=1)
    result = _optimize([], constraints, completed=()).design_for_n(1)
    assert result.status == 'feasible'
    assert result.times == (0.0,)
    assert not result.score.is_full_rank
    assert np.isinf(result.score.criterion)


def test_empty_requirements_preserve_the_single_point_central_optimum():
    constraints = _constraints(min_n=3, max_n=3)
    observation = ObservationModel(kind='constant_sd', value=0.1)
    old = optimize_local_i_optimal_designs(_model(), observation, constraints, refine=False).design_for_n(3)
    new = _optimize([], constraints, refine=False).design_for_n(3)
    np.testing.assert_array_equal(new.times, old.times)
    assert new.score.criterion == pytest.approx(old.score.criterion)


def test_search_limit_after_a_witness_preserves_a_feasible_schedule():
    constraints = _constraints(min_n=4, max_n=4)
    resolved = _resolved([SamplingWindow(2.0, 8.0)], constraints)
    original = scheduling.check_schedule_feasibility
    calls = 0

    def limited_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            return ScheduleFeasibilityResult(
                status='search_limit', times=(), states_visited=1, message='Test search budget reached.',
            )
        return original(*args, **kwargs)

    with patch.object(scheduling, 'check_schedule_feasibility', side_effect=limited_after_first):
        row = optimize_window_constrained_designs(
            _model(), ObservationModel(kind='constant_sd', value=0.1),
            constraints, resolved, refine=False,
        ).design_for_n(4)
    assert row.status == 'feasible'
    assert row.optimization_status == 'search_limit'
    assert row.requirement_evaluation.status == 'satisfied'


def test_optimizer_initial_search_limit_returns_no_invented_schedule():
    result = _optimize(
        [SamplingWindow(2.0, 9.0, minimum_count=2)], number_of_times=5, max_search_states=1,
    ).result_for_n(5)
    assert result.status == 'search_limit'
    assert result.times == ()
    assert result.score is None


@pytest.mark.parametrize('sizes', [True, 2.5, (), (0,), (99,)])
def test_optimizer_rejects_invalid_requested_sizes(sizes):
    with pytest.raises((TypeError, ValueError)):
        _optimize([], number_of_times=sizes)


def test_optimization_does_not_mutate_inputs_or_source_parameter_draws():
    constraints = _constraints(min_n=4, max_n=4)
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    model = _model()
    before = (constraints.candidate_times.copy(), constraints.fixed_times.copy(), model.parameter_draws.copy())
    optimize_window_constrained_designs(
        model, ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
    )
    np.testing.assert_array_equal(constraints.candidate_times, before[0])
    np.testing.assert_array_equal(constraints.fixed_times, before[1])
    np.testing.assert_array_equal(model.parameter_draws, before[2])


def test_local_information_uses_central_parameters_instead_of_ensemble_loss_scoring():
    constraints = _constraints(min_n=4, max_n=4)
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    model = _model()
    central = optimize_window_constrained_designs(
        model, ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
    ).design_for_n(4)
    model.parameter_draws = np.array([[1.0, -0.05], [10.0, 20.0]])
    model.valid_draws = np.array([True, True])
    model.draw_weights = np.array([0.01, 0.99])
    changed_ensemble = optimize_window_constrained_designs(
        model, ObservationModel(kind='constant_sd', value=0.1), constraints, resolved,
    ).design_for_n(4)
    assert changed_ensemble.times == central.times
    assert changed_ensemble.score.criterion == pytest.approx(central.score.criterion)


def test_uniform_observation_sd_rescales_information_without_changing_the_schedule():
    constraints = _constraints(min_n=4, max_n=4)
    resolved = _resolved([SamplingWindow(2.0, 6.0)], constraints)
    rows = [
        optimize_window_constrained_designs(
            _model(), ObservationModel(kind='constant_sd', value=sd), constraints, resolved,
        ).design_for_n(4)
        for sd in (0.1, 0.2)
    ]
    assert rows[0].times == rows[1].times
    assert rows[1].score.criterion == pytest.approx(4.0 * rows[0].score.criterion)
