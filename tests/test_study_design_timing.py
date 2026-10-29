from __future__ import annotations

import numpy as np
import pytest

from study_design_core import (
    DesignConstraints,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
    evaluate_local_i_optimal_design,
    latest_time_before_deadline,
    optimize_next_time_before_deadline,
    reference_crossing_time_summary,
)


def _reference(crossing_times, weights=None) -> ReferenceTrajectoryEnsemble:
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


def _constraints(*, minimum_spacing=1.0) -> DesignConstraints:
    return DesignConstraints(
        candidate_times=np.arange(0.0, 11.0),
        fixed_times=np.array([0.0, 2.0]),
        min_time=0.0,
        max_time=10.0,
        minimum_spacing=minimum_spacing,
        min_number_of_times=3,
        max_number_of_times=5,
        replicates_per_time=3,
    )


def _linear_model() -> ReferenceModelEnsemble:
    return ReferenceModelEnsemble(
        model_name='linear',
        parameter_names=('a', 'b'),
        parameter_draws=np.array([[1.0, -0.05]]),
        valid_draws=np.array([True]),
        draw_weights=np.array([1.0]),
        central_parameters=np.array([1.0, -0.05]),
        time_grid=np.linspace(0.0, 10.0, 21),
        supported_time_range=(0.0, 10.0),
    )


def test_crossing_quantile_keeps_noncrossers_and_uses_inverse_cdf():
    reference = _reference([2.0, 4.0, np.inf], weights=[0.2, 0.3, 0.5])
    early = reference_crossing_time_summary(reference, quantile=0.2)
    boundary = reference_crossing_time_summary(reference, quantile=0.5)
    unavailable = reference_crossing_time_summary(reference, quantile=0.51)

    np.testing.assert_allclose(boundary.crossing_times[:2], [2.0, 4.0])
    assert np.isposinf(boundary.crossing_times[2])
    np.testing.assert_allclose(boundary.draw_weights, [0.2, 0.3, 0.5])
    assert early.deadline == 2.0
    assert boundary.deadline == 4.0
    assert boundary.reached_weight == 0.5
    assert boundary.noncrossing_weight == 0.5
    assert unavailable.status == 'beyond_horizon'
    assert unavailable.deadline is None


def test_weighted_quantile_ties_and_crossing_at_horizon_are_retained():
    reference = _reference([3.0, 3.0, 10.0], weights=[0.1, 0.4, 0.5])
    tied = reference_crossing_time_summary(reference, quantile=0.5)
    later = reference_crossing_time_summary(reference, quantile=0.75)
    assert tied.deadline == 3.0
    assert later.deadline == 10.0
    assert later.reached_weight == 1.0
    assert tied.crossed_weight_before(tied.deadline) == 0.0
    np.testing.assert_allclose(tied.crossed_weight_before(tied.deadline + 1e-8), 0.5)


def test_original_invalid_and_zero_weight_draws_do_not_affect_planning():
    time = np.arange(0.0, 11.0)
    reference = ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=np.vstack([1.0 - time / 8.0, np.zeros_like(time), np.full_like(time, np.nan)]),
        valid_draws=np.array([True, True, False]),
        draw_weights=np.array([1.0, 0.0, 100.0]),
        central_curve=1.0 - time / 8.0,
        interpolation='linear',
    )
    result = reference_crossing_time_summary(reference)
    assert result.status == 'available'
    assert result.deadline == 4.0
    assert result.invalid_weight == 0.0
    np.testing.assert_array_equal(result.draw_weights, [1.0, 0.0, 0.0])
    assert np.isnan(result.crossing_times[1:]).all()


def test_nonpositive_positive_weight_baseline_is_not_silently_dropped():
    time = np.arange(0.0, 11.0)
    reference = ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=np.vstack([1.0 - time / 8.0, np.zeros_like(time)]),
        valid_draws=np.ones(2, dtype=bool),
        draw_weights=np.array([0.8, 0.2]),
        interpolation='linear',
    )
    result = reference_crossing_time_summary(reference)
    assert result.status == 'invalid_reference'
    assert result.deadline is None
    np.testing.assert_allclose(result.draw_weights, [0.8, 0.2])
    np.testing.assert_allclose(result.invalid_weight, 0.2)
    policy = latest_time_before_deadline(result, _constraints(), latest_completed_time=2.0)
    assert policy.status == 'invalid_reference'
    assert policy.new_time is None


def test_relative_milestone_uses_explicit_interpolated_baseline():
    time = np.arange(0.0, 11.0)
    reference = ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=np.array([1.2 - 0.1 * time]),
        valid_draws=np.array([True]),
        supported_time_range=(2.0, 8.0),
        interpolation='linear',
    )
    result = reference_crossing_time_summary(reference, baseline_time=3.25)
    assert result.baseline_time == 3.25
    np.testing.assert_allclose(result.deadline, 7.625)
    with pytest.raises(ValueError, match='baseline'):
        reference_crossing_time_summary(reference, baseline_time=0.0)


@pytest.mark.parametrize('scale', [1e-12, 1.0, 1e12])
def test_relative_crossing_time_is_invariant_to_response_scale(scale):
    reference = _reference([5.5])
    scaled = ReferenceTrajectoryEnsemble(
        time_grid=reference.time_grid,
        value_draws=scale * reference.value_draws,
        valid_draws=reference.valid_draws,
        interpolation='linear',
    )
    result = reference_crossing_time_summary(scaled)
    np.testing.assert_allclose(result.deadline, 5.5)


def test_milestone_fraction_and_quantile_are_independent_inputs():
    reference = _reference([4.0])
    halfway = reference_crossing_time_summary(reference, degraded_fraction=0.5)
    quarter = reference_crossing_time_summary(reference, degraded_fraction=0.25)
    assert halfway.deadline == 4.0
    assert quarter.deadline == 2.0


def test_tiny_positive_degradation_fraction_has_a_positive_crossing():
    result = reference_crossing_time_summary(_reference([4.0]), degraded_fraction=1e-20)
    assert result.status == 'available'
    assert result.deadline > 0.0
    np.testing.assert_allclose(result.deadline, 8e-20, rtol=1e-12, atol=0.0)


def test_first_crossing_is_used_when_a_trajectory_recrosses():
    reference = ReferenceTrajectoryEnsemble(
        time_grid=np.array([0.0, 2.0, 4.0, 6.0]),
        value_draws=np.array([[1.0, 0.4, 0.8, 0.3]]),
        valid_draws=np.array([True]),
        interpolation='linear',
    )
    result = reference_crossing_time_summary(reference)
    np.testing.assert_allclose(result.deadline, 5.0 / 3.0)


@pytest.mark.parametrize('field', ['degraded_fraction', 'quantile'])
@pytest.mark.parametrize('value', [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_fractional_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        reference_crossing_time_summary(_reference([4.0]), **{field: value})


def test_direct_timing_floors_to_legal_grid_and_preserves_fixed_times():
    summary = reference_crossing_time_summary(_reference([5.5]))
    constraints = _constraints()
    result = latest_time_before_deadline(summary, constraints, latest_completed_time=2.0)
    assert result.status == 'available'
    assert result.new_time == 5.0
    assert result.new_time <= summary.deadline
    np.testing.assert_array_equal(result.eligible_times, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(result.times, [0.0, 2.0, 5.0])
    constraints.validate_design(result.times)
    assert result.score is None
    # The original design window and allowed schedule sizes remain unchanged.
    np.testing.assert_array_equal(constraints.candidate_times, np.arange(11.0))
    assert constraints.max_time == 10.0
    assert constraints.max_number_of_times == 5


@pytest.mark.parametrize('deadline, spacing, expected', [
    (2.0, 1.0, 'deadline_elapsed'),
    (2.5, 1.0, 'no_feasible_candidate'),
    (3.5, 2.0, 'no_feasible_candidate'),
])
def test_unavailable_deadline_is_not_moved_forward(deadline, spacing, expected):
    summary = reference_crossing_time_summary(_reference([deadline]))
    constraints = _constraints(minimum_spacing=spacing)
    for result in (
        latest_time_before_deadline(summary, constraints, latest_completed_time=2.0),
        optimize_next_time_before_deadline(
            summary, _linear_model(), ObservationModel(kind='constant_sd', value=0.1),
            constraints, latest_completed_time=2.0,
        ),
    ):
        assert result.status == expected
        assert result.new_time is None
        assert len(result.times) == 0
        assert result.score is None
        assert result.message


def test_beyond_horizon_summary_remains_an_unavailable_policy():
    summary = reference_crossing_time_summary(_reference([np.inf]))
    result = latest_time_before_deadline(summary, _constraints(), latest_completed_time=2.0)
    assert result.status == 'beyond_horizon'
    assert result.new_time is None
    assert len(result.times) == 0


def test_constrained_optimizer_uses_original_horizon_and_requested_weights():
    summary = reference_crossing_time_summary(_reference([5.5]))
    constraints = _constraints()
    model = _linear_model()
    observation = ObservationModel(kind='constant_sd', value=0.2)
    evaluation_times = np.array([0.0, 2.0, 5.0, 10.0])
    evaluation_weights = np.array([2.0, 1.0, 4.0, 3.0])
    result = optimize_next_time_before_deadline(
        summary, model, observation, constraints, latest_completed_time=2.0,
        evaluation_times=evaluation_times, evaluation_weights=evaluation_weights,
    )
    assert result.status == 'available'
    assert result.new_time <= summary.deadline
    scores = [
        evaluate_local_i_optimal_design(
            np.union1d(constraints.fixed_times, [time]), model, observation, constraints,
            evaluation_times=evaluation_times, evaluation_weights=evaluation_weights,
        ).criterion
        for time in result.eligible_times
    ]
    np.testing.assert_allclose(result.score.criterion, min(scores), rtol=1e-12)
    assert result.metadata['selection_rule'] == 'i_optimality'
    constraints.validate_design(result.times)
    assert model.supported_time_range == (0.0, 10.0)
    assert constraints.max_time == 10.0


def test_tightening_quantile_never_moves_one_point_recommendation_later():
    reference = _reference([3.0, 5.0, 8.0])
    constraints = _constraints()
    model = _linear_model()
    observation = ObservationModel(kind='constant_sd', value=0.1)
    direct_times = []
    optimized_times = []
    for quantile in (0.1, 0.5, 0.9):
        summary = reference_crossing_time_summary(reference, quantile=quantile)
        direct_times.append(latest_time_before_deadline(
            summary, constraints, latest_completed_time=2.0,
        ).new_time)
        optimized_times.append(optimize_next_time_before_deadline(
            summary, model, observation, constraints, latest_completed_time=2.0,
        ).new_time)
    assert np.all(np.diff(direct_times) >= 0)
    assert np.all(np.diff(optimized_times) >= 0)
    assert direct_times[0] < direct_times[-1]


def test_constrained_optimizer_preserves_rank_fallback():
    summary = reference_crossing_time_summary(_reference([5.5]))
    result = optimize_next_time_before_deadline(
        summary, _linear_model(), ObservationModel(kind='constant_sd', value=0.1),
        _constraints(), latest_completed_time=2.0, rank_tolerance=1e6,
    )
    assert result.status == 'available'
    assert np.isinf(result.score.criterion)
    assert result.metadata['selection_rule'] == 'rank_pseudodeterminant_fallback'


@pytest.mark.parametrize('latest', [np.nan, -1.0, 11.0, 1.0])
def test_invalid_latest_completed_time_is_rejected(latest):
    summary = reference_crossing_time_summary(_reference([5.5]))
    with pytest.raises(ValueError):
        latest_time_before_deadline(summary, _constraints(), latest_completed_time=latest)


def test_next_time_cannot_precede_the_milestone_baseline():
    summary = reference_crossing_time_summary(_reference([5.5]), baseline_time=3.25)
    with pytest.raises(ValueError, match='baseline'):
        latest_time_before_deadline(summary, _constraints(), latest_completed_time=2.0)
    with pytest.raises(ValueError, match='baseline'):
        optimize_next_time_before_deadline(
            summary, _linear_model(), ObservationModel(kind='constant_sd', value=0.1),
            _constraints(), latest_completed_time=2.0,
        )


def test_one_point_selector_rejects_constraints_requiring_two_new_points():
    summary = reference_crossing_time_summary(_reference([5.5]))
    constraints = DesignConstraints(
        candidate_times=np.arange(11.0), fixed_times=np.array([0.0, 2.0]),
        min_number_of_times=4, max_number_of_times=5,
    )
    with pytest.raises(ValueError, match='one|additional|number'):
        latest_time_before_deadline(summary, constraints, latest_completed_time=2.0)
