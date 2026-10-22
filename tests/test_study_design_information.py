from __future__ import annotations

import numpy as np

from study_design_core import (
    DesignConstraints,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
    coordinate_exchange_i_optimal_design,
    coordinate_exchange_local_i_optimal_design,
    ensemble_i_optimality,
    evaluate_local_i_optimal_design,
    finite_difference_sensitivities,
    fit_reference_model_ensemble,
    greedy_local_i_optimal_designs,
    greedy_i_optimal_designs,
    information_matrix,
    local_i_optimality,
    optimize_local_i_optimal_designs,
    optimize_i_optimal_designs,
    trapezoidal_integration_weights,
)


def _linear_parameter_ensemble() -> ReferenceModelEnsemble:
    return ReferenceModelEnsemble(
        model_name='linear',
        parameter_names=('a', 'b'),
        parameter_draws=np.array([
            [1.0, -0.20],
            [1.1, -0.22],
        ]),
        valid_draws=np.array([True, True]),
        draw_weights=np.array([0.4, 0.6]),
        central_parameters=np.array([1.06, -0.212]),
        time_grid=np.linspace(0.0, 4.0, 9),
        supported_time_range=(0.0, 4.0),
    )


def test_finite_difference_sensitivities_and_information_matrix_are_exact_for_linear_model():
    times = np.array([0.0, 1.0, 2.0])
    sensitivities = finite_difference_sensitivities(
        'linear',
        times,
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        sensitivities,
        np.column_stack([np.ones(3), times]),
        atol=1e-8,
    )

    observation = ObservationModel(kind='constant_sd', value=1.0)
    matrix = information_matrix(
        'linear',
        [0.0, 1.0],
        [1.0, 2.0],
        observation,
        replicates_per_time=2,
    )
    np.testing.assert_allclose(matrix, [[4.0, 2.0], [2.0, 2.0]], atol=1e-7)


def test_i_optimality_is_infinite_until_the_model_is_identifiable():
    observation = ObservationModel(kind='constant_sd', value=0.1)
    evaluation_times = np.linspace(0.0, 4.0, 21)

    deficient = local_i_optimality(
        'linear',
        [2.0],
        [1.0, -0.2],
        observation,
        evaluation_times=evaluation_times,
    )
    full_rank = local_i_optimality(
        'linear',
        [0.0, 4.0],
        [1.0, -0.2],
        observation,
        evaluation_times=evaluation_times,
    )

    assert np.isinf(deficient)
    assert np.isfinite(full_rank)
    assert full_rank > 0
    np.testing.assert_allclose(
        trapezoidal_integration_weights([0.0, 1.0, 2.0]),
        [0.25, 0.5, 0.25],
    )


def test_reference_trajectory_draws_can_be_fitted_to_a_common_analysis_model():
    time = np.linspace(0.0, 4.0, 9)
    reference = ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=np.vstack([
            1.0 - 0.20 * time,
            1.1 - 0.22 * time,
        ]),
        valid_draws=np.array([True, True]),
        draw_weights=np.array([0.4, 0.6]),
        central_curve=1.06 - 0.212 * time,
        supported_time_range=(0.0, 4.0),
        interpolation='linear',
    )

    fitted = fit_reference_model_ensemble(reference, model_name='linear')

    assert fitted.model_name == 'linear'
    assert fitted.n_valid_draws == 2
    np.testing.assert_allclose(fitted.parameter_draws[0], [1.0, -0.20], atol=1e-8)
    np.testing.assert_allclose(fitted.parameter_draws[1], [1.1, -0.22], atol=1e-8)
    np.testing.assert_allclose(fitted.central_parameters, [1.06, -0.212], atol=1e-8)
    np.testing.assert_allclose(fitted.draw_weights, [0.4, 0.6])


def test_ensemble_information_score_retains_per_draw_rank_and_weights():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=0.1)

    deficient = ensemble_i_optimality(
        ensemble,
        [2.0],
        observation,
    )
    full_rank = ensemble_i_optimality(
        ensemble,
        [0.0, 4.0],
        observation,
    )

    assert deficient.min_rank == 1
    assert deficient.full_rank_fraction == 0.0
    assert np.isinf(deficient.criterion)
    assert full_rank.min_rank == 2
    assert full_rank.is_full_rank
    assert np.isfinite(full_rank.criterion)
    np.testing.assert_allclose(full_rank.draw_weights, [0.4, 0.6])


def test_greedy_optimizer_selects_boundary_times_for_a_linear_curve():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=2,
        max_number_of_times=3,
    )

    steps = greedy_i_optimal_designs(
        ensemble,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
    )

    assert [step.number_of_times for step in steps] == [2, 3]
    np.testing.assert_allclose(steps[0].times, [0.0, 4.0])
    assert steps[0].score.is_full_rank
    assert set(steps[0].times).issubset(set(steps[1].times))


def test_greedy_optimizer_respects_fixed_times_and_minimum_spacing():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        fixed_times=np.array([0.0]),
        minimum_spacing=2.0,
        min_number_of_times=2,
        max_number_of_times=2,
    )

    steps = greedy_i_optimal_designs(ensemble, observation, constraints)

    assert len(steps) == 1
    np.testing.assert_allclose(steps[0].times, [0.0, 4.0])


def test_coordinate_exchange_improves_a_suboptimal_fixed_size_design():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=2,
        max_number_of_times=2,
    )

    result = coordinate_exchange_i_optimal_design(
        [1.0, 3.0],
        ensemble,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
    )

    np.testing.assert_allclose(result.times, [0.0, 4.0])
    assert result.score.criterion < ensemble_i_optimality(
        ensemble,
        [1.0, 3.0],
        observation,
        evaluation_times=np.linspace(0.0, 4.0, 41),
    ).criterion
    assert len(result.exchanges) >= 1


def test_high_level_optimizer_returns_nested_and_refined_designs():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=2,
        max_number_of_times=4,
    )

    result = optimize_i_optimal_designs(
        ensemble,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
        refine=True,
    )

    assert [step.number_of_times for step in result.nested_steps] == [2, 3, 4]
    assert [step.number_of_times for step in result.refined_steps] == [2, 3, 4]
    np.testing.assert_allclose(result.design_for_n(2).times, [0.0, 4.0])
    assert result.design_for_n(2, refined=True).score.criterion <= (
        result.design_for_n(2).score.criterion + 1e-12
    )
    assert result.metadata['criterion'] == 'uncertainty_averaged_i_optimality'


def test_local_optimizer_uses_only_the_central_reference_parameters():
    ensemble = _linear_parameter_ensemble()
    altered_draws = ReferenceModelEnsemble(
        model_name=ensemble.model_name,
        parameter_names=ensemble.parameter_names,
        parameter_draws=np.array([
            [8.0, 2.0],
            [-4.0, -3.0],
        ]),
        valid_draws=np.array([True, True]),
        draw_weights=np.array([0.99, 0.01]),
        central_parameters=ensemble.central_parameters,
        time_grid=ensemble.time_grid,
        supported_time_range=ensemble.supported_time_range,
    )
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=2,
        max_number_of_times=3,
    )

    first = optimize_local_i_optimal_designs(
        ensemble,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
        refine=True,
    )
    second = optimize_local_i_optimal_designs(
        altered_draws,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
        refine=True,
    )

    for left, right in zip(first.nested_steps, second.nested_steps):
        np.testing.assert_allclose(left.times, right.times)
        np.testing.assert_allclose(left.score.criterion, right.score.criterion)
    for left, right in zip(first.refined_steps, second.refined_steps):
        np.testing.assert_allclose(left.times, right.times)
        np.testing.assert_allclose(left.score.criterion, right.score.criterion)
    assert first.metadata['criterion'] == 'local_i_optimality'
    assert first.metadata['parameter_source'] == 'central_reference_fit'
    assert first.metadata['reference_uncertainty_used_in_optimization'] is False
    assert first.metadata['reference_uncertainty_evaluated_by_monte_carlo'] is True


def test_local_optimizer_reports_rank_fallback_before_i_optimality():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=1,
        max_number_of_times=2,
    )

    steps = greedy_local_i_optimal_designs(
        ensemble,
        observation,
        constraints,
        evaluation_times=np.linspace(0.0, 4.0, 41),
    )

    assert [step.score.min_rank for step in steps] == [1, 2]
    assert steps[0].metadata['selection_rule'] == 'rank_pseudodeterminant_fallback'
    assert steps[1].metadata['selection_rule'] == 'i_optimality'
    assert np.isinf(steps[0].score.criterion)
    assert np.isfinite(steps[1].score.criterion)


def test_local_design_evaluation_and_exchange_use_central_reference():
    ensemble = _linear_parameter_ensemble()
    observation = ObservationModel(kind='constant_sd', value=1.0)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 5.0),
        min_number_of_times=2,
        max_number_of_times=2,
    )
    evaluation_times = np.linspace(0.0, 4.0, 41)

    initial = evaluate_local_i_optimal_design(
        [1.0, 3.0],
        ensemble,
        observation,
        constraints,
        evaluation_times=evaluation_times,
    )
    result = coordinate_exchange_local_i_optimal_design(
        [1.0, 3.0],
        ensemble,
        observation,
        constraints,
        evaluation_times=evaluation_times,
    )

    np.testing.assert_allclose(result.times, [0.0, 4.0])
    assert result.score.criterion < initial.criterion
