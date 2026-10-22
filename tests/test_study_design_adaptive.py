from __future__ import annotations

import numpy as np

from examples.study_design_proof_of_concept import (
    ProofOfConceptSettings,
    build_adaptive_in_vivo_only_scenario,
    build_adaptive_ivivc_scenario,
    build_measured_in_vivo_scenario,
    run_scenario,
)
from study_design_core import (
    DesignConstraints,
    ExistingObservations,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
    reweight_reference_ensemble,
    simulate_design,
)


def _linear_reference() -> ReferenceTrajectoryEnsemble:
    time = np.linspace(0.0, 4.0, 41)
    draws = np.vstack([
        1.0 - 0.20 * time,
        1.0 - 0.10 * time,
    ])
    return ReferenceTrajectoryEnsemble(
        time_grid=time,
        value_draws=draws,
        valid_draws=np.array([True, True]),
        draw_weights=np.array([0.5, 0.5]),
        interpolation='linear',
    )


def _small_settings() -> ProofOfConceptSettings:
    return ProofOfConceptSettings(
        n_reference_draws=10,
        n_simulations=10,
        random_seed=2468,
        refine=False,
        write_html=False,
        max_design_sizes=1,
    )


def test_reference_reweighting_favors_trajectories_consistent_with_completed_data():
    reference = _linear_reference()
    existing = ExistingObservations(
        time=np.array([2.0, 2.0, 4.0, 4.0]),
        response=np.array([0.61, 0.59, 0.21, 0.19]),
    )
    observation = ObservationModel(kind='constant_sd', value=0.03)

    update = reweight_reference_ensemble(reference, existing, observation)

    assert update.posterior_weights[0] > 0.999
    assert update.posterior_weights[1] < 0.001
    assert 1.0 <= update.effective_sample_size <= 2.0
    np.testing.assert_allclose(
        update.reference.central_curve,
        np.average(
            reference.value_draws,
            axis=0,
            weights=update.posterior_weights,
        ),
    )


def test_adaptive_simulation_keeps_completed_values_fixed_across_monte_carlo_runs():
    reference = _linear_reference()
    existing = ExistingObservations(
        time=np.array([0.0, 0.0]),
        response=np.array([1.03, 0.97]),
    )
    constraints = DesignConstraints(
        candidate_times=np.array([2.0, 4.0]),
        fixed_times=np.array([0.0]),
        min_time=0.0,
        max_time=4.0,
        min_number_of_times=3,
        max_number_of_times=3,
        replicates_per_time=2,
    )
    observation = ObservationModel(
        kind='constant_sd',
        value=0.05,
        replicates_per_time=2,
    )

    result = simulate_design(
        reference,
        [0.0, 2.0, 4.0],
        observation,
        model_name='linear',
        constraints=constraints,
        existing_observations=existing,
        n_simulations=20,
        random_state=99,
    )

    np.testing.assert_allclose(result.observed_means[:, 0], 1.0)
    assert result.metadata['existing_observation_count'] == 2
    assert result.metadata['existing_time_point_count'] == 1
    np.testing.assert_allclose(result.metadata['future_design_times'], [2.0, 4.0])


def test_both_adaptive_scenarios_use_only_future_candidate_times():
    settings = _small_settings()
    in_vivo_only = build_adaptive_in_vivo_only_scenario(settings)
    ivivc_informed = build_adaptive_ivivc_scenario(settings)

    for scenario in (in_vivo_only, ivivc_informed):
        assert scenario.existing_observations is not None
        latest = scenario.existing_observations.latest_time
        assert np.all(scenario.constraints.candidate_times > latest)
        np.testing.assert_allclose(
            scenario.constraints.fixed_times,
            scenario.existing_observations.unique_times,
        )

    assert in_vivo_only.metadata['adaptive_mode'] == 'in_vivo_only'
    assert ivivc_informed.metadata['adaptive_mode'] == 'ivivc_informed'
    assert ivivc_informed.metadata['effective_sample_size'] > 0
    assert {
        'IVIVC calibration data',
        'target in vitro prediction data',
        'completed target-study in vivo data',
    } <= set(ivivc_informed.observed_data['role'])


def test_adaptive_runner_recommends_only_times_after_completed_study():
    settings = _small_settings()
    for builder in (build_adaptive_in_vivo_only_scenario, build_adaptive_ivivc_scenario):
        definition = builder(settings)
        result = run_scenario(definition, settings, random_seed=123)
        latest = definition.existing_observations.latest_time
        selected = result.selected_times_table
        additions = selected.loc[~selected['fixed'], 'time'].to_numpy(dtype=float)
        assert len(additions) > 0
        assert np.all(additions > latest)


def test_displayed_synthetic_ground_truth_does_not_change_design_results():
    settings = _small_settings()
    with_truth = build_measured_in_vivo_scenario(settings)
    without_truth = build_measured_in_vivo_scenario(settings)
    without_truth.ground_truth = None

    first = run_scenario(with_truth, settings, random_seed=777)
    second = run_scenario(without_truth, settings, random_seed=777)

    np.testing.assert_allclose(
        first.selected_times_table['time'],
        second.selected_times_table['time'],
    )
    np.testing.assert_allclose(
        first.performance_table['penalized_mean_normalized_irmse'],
        second.performance_table['penalized_mean_normalized_irmse'],
        equal_nan=True,
    )
