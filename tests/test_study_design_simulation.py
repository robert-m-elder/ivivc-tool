from __future__ import annotations

import numpy as np
import pytest

from study_design_core import (
    DesignConstraints,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
    compare_designs_by_simulation,
    create_simulation_plan,
    equal_response_spaced_design,
    equally_spaced_design,
    integrated_root_mean_square_error,
    log_spaced_design,
    maximum_absolute_error,
    simulate_design,
    summarize_simulation,
    threshold_crossing_time,
)


def _linear_reference() -> ReferenceTrajectoryEnsemble:
    time = np.linspace(0.0, 4.0, 41)
    draws = np.vstack([
        1.0 - 0.20 * time,
        1.1 - 0.22 * time,
    ])
    return ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=draws,
        valid_draws=np.array([True, True]),
        draw_weights=np.array([0.4, 0.6]),
        central_curve=0.4 * draws[0] + 0.6 * draws[1],
        supported_time_range=(0.0, 4.0),
        interpolation='linear',
    )


def _constraints(*, min_times=2, max_times=4, replicates=2) -> DesignConstraints:
    return DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=min_times,
        max_number_of_times=max_times,
        replicates_per_time=replicates,
    )


def test_curve_metrics_and_threshold_crossing_are_exact():
    true = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    predicted = np.array([[1.0, 0.4, 0.0], [0.8, 0.5, 0.2]])
    irmse = integrated_root_mean_square_error(
        true,
        predicted,
        weights=[0.25, 0.5, 0.25],
    )
    np.testing.assert_allclose(
        irmse,
        [np.sqrt(0.005), np.sqrt(0.02)],
    )
    np.testing.assert_allclose(
        maximum_absolute_error(true, predicted),
        [0.1, 0.2],
    )
    np.testing.assert_allclose(
        threshold_crossing_time([0.0, 1.0, 2.0], true, 0.25),
        [1.5, 1.5],
    )


def test_common_random_plan_is_reproducible_and_subsets_by_time():
    reference = _linear_reference()
    observation = ObservationModel(
        kind='constant_sd',
        value=0.1,
        replicates_per_time=3,
    )
    first = create_simulation_plan(
        reference,
        observation,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        n_simulations=10,
        random_state=123,
    )
    second = create_simulation_plan(
        reference,
        observation,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        n_simulations=10,
        random_state=123,
    )

    np.testing.assert_array_equal(
        first.trajectory_draw_indices,
        second.trajectory_draw_indices,
    )
    np.testing.assert_allclose(
        first.standard_normal_errors,
        second.standard_normal_errors,
    )
    np.testing.assert_array_equal(first.indices_for_times([0.0, 4.0]), [0, 4])
    with pytest.raises(ValueError, match='not uniquely represented'):
        first.indices_for_times([2.5])


def test_simulation_recovers_linear_reference_and_reports_performance():
    reference = _linear_reference()
    constraints = _constraints(replicates=3)
    observation = ObservationModel(
        kind='constant_sd',
        value=0.03,
        replicates_per_time=3,
    )
    plan = create_simulation_plan(
        reference,
        observation,
        constraints.all_available_times,
        n_simulations=80,
        random_state=44,
    )
    result = simulate_design(
        reference,
        [0.0, 2.0, 4.0],
        observation,
        model_name='linear',
        constraints=constraints,
        simulation_plan=plan,
        target_response=0.5,
    )
    summary = summarize_simulation(result)

    assert result.n_simulations == 80
    assert result.n_successful == 80
    assert result.expected_values.shape == (80, 3)
    assert result.observed_means.shape == (80, 3)
    assert result.fitted_curves.shape == (80, len(reference.time_grid))
    assert summary.failure_rate == 0.0
    assert summary.median_irmse < 0.03
    assert summary.p90_irmse < 0.05
    assert np.isfinite(summary.median_absolute_target_time_error)
    assert summary.parameter_bound_hit_rate == 0.0


def test_failed_sparse_fits_receive_penalty_and_remain_visible():
    reference = _linear_reference()
    constraints = _constraints(min_times=1, max_times=1, replicates=1)
    observation = ObservationModel(
        kind='constant_sd',
        value=0.05,
        replicates_per_time=1,
    )
    result = simulate_design(
        reference,
        [2.0],
        observation,
        model_name='linear',
        constraints=constraints,
        n_simulations=12,
        random_state=8,
        fallback_grid_search=False,
        failure_penalty=3.0,
    )
    summary = summarize_simulation(result)

    assert result.n_successful == 0
    assert summary.failure_rate == 1.0
    np.testing.assert_allclose(result.penalized_integrated_rmse, 3.0)
    assert summary.penalized_mean_irmse == 3.0
    assert np.isnan(summary.median_irmse)


def test_design_comparison_reuses_identical_random_draws():
    reference = _linear_reference()
    constraints = _constraints(replicates=2)
    observation = ObservationModel(
        kind='constant_sd',
        value=0.08,
        replicates_per_time=2,
    )
    comparison = compare_designs_by_simulation(
        reference,
        {
            'first': [0.0, 4.0],
            'duplicate': [0.0, 4.0],
            'interior': [1.0, 3.0],
        },
        observation,
        model_name='linear',
        constraints=constraints,
        n_simulations=150,
        random_state=17,
    )

    np.testing.assert_allclose(
        comparison.results['first'].observed_means,
        comparison.results['duplicate'].observed_means,
    )
    np.testing.assert_allclose(
        comparison.results['first'].integrated_rmse,
        comparison.results['duplicate'].integrated_rmse,
    )
    assert (
        comparison.summaries['first'].penalized_mean_irmse
        < comparison.summaries['interior'].penalized_mean_irmse
    )


def test_simple_baseline_designs_are_feasible_and_deterministic():
    reference = _linear_reference()
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        fixed_times=np.array([0.0]),
        minimum_spacing=1.0,
        min_number_of_times=3,
        max_number_of_times=4,
    )

    equal_time = equally_spaced_design(constraints, 3)
    log_time = log_spaced_design(constraints, 3)
    equal_response = equal_response_spaced_design(reference, constraints, 3)

    np.testing.assert_allclose(equal_time, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(equal_response, [0.0, 2.0, 4.0])
    constraints.validate_design(log_time)
    assert len(log_time) == 3
    assert 0.0 in log_time


def test_equal_response_baseline_rejects_nonmonotonic_reference():
    time = np.linspace(0.0, 4.0, 41)
    reference = ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=np.sin(time).reshape(1, -1),
        valid_draws=np.array([True]),
        supported_time_range=(0.0, 4.0),
    )
    with pytest.raises(ValueError, match='monotonic'):
        equal_response_spaced_design(reference, _constraints(), 3)


def test_simulation_uses_data_driven_fit_initialization_by_default(monkeypatch):
    import study_design_core.simulation as simulation_module

    reference = _linear_reference()
    constraints = _constraints(replicates=2)
    observation = ObservationModel(
        kind='constant_sd',
        value=0.03,
        replicates_per_time=2,
    )
    starts = []
    original = simulation_module.fit_model_once

    def recording_fit(*args, **kwargs):
        starts.append(kwargs.get('initial_parameters'))
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_module, 'fit_model_once', recording_fit)
    result = simulation_module.simulate_design(
        reference,
        [0.0, 2.0, 4.0],
        observation,
        model_name='linear',
        constraints=constraints,
        n_simulations=6,
        random_state=91,
    )

    assert starts
    assert all(start is None for start in starts)
    assert result.metadata['fit_initialization'] == 'data_driven'
