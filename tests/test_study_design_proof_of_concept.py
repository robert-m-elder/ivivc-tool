from __future__ import annotations

import json

import numpy as np
import pandas as pd

from examples.study_design_proof_of_concept import (
    DESIGN_LABELS,
    ProofOfConceptSettings,
    build_adaptive_refinement_scenario,
    build_adaptive_in_vivo_only_scenario,
    build_adaptive_ivivc_scenario,
    build_ivivc_prediction_scenario,
    build_measured_in_vivo_scenario,
    run_proof_of_concept,
    write_proof_of_concept_outputs,
)


def _small_settings(*, write_html: bool = False) -> ProofOfConceptSettings:
    return ProofOfConceptSettings(
        n_reference_draws=8,
        n_simulations=8,
        random_seed=1234,
        refine=True,
        write_html=write_html,
        max_design_sizes=1,
        analysis_model='mass_loss_one_phase_normalized',
    )


def test_scenario_builders_cover_measured_ivivc_and_adaptive_inputs():
    settings = _small_settings()
    measured = build_measured_in_vivo_scenario(settings)
    ivivc = build_ivivc_prediction_scenario(settings)
    adaptive = build_adaptive_refinement_scenario(settings)
    adaptive_ivivc = build_adaptive_ivivc_scenario(settings)

    assert measured.metadata['reference_source'] == 'measured_in_vivo_replicates'
    assert measured.reference.n_valid_draws == settings.n_reference_draws
    assert measured.ground_truth is not None
    assert measured.ground_truth.label == 'synthetic_ground_truth'
    assert measured.analysis_model == settings.analysis_model
    assert measured.observation_model.kind == 'constant_sd'
    assert measured.observation_model.value == settings.observation_sd
    assert measured.metadata['reference_data_sd'] == settings.observation_sd
    assert {'time', 'response', 'replicate_id'}.issubset(measured.observed_data)

    assert ivivc.metadata['reference_source'] == 'ivivc_approach1_prediction'
    assert ivivc.reference.n_valid_draws > 1
    assert ivivc.ground_truth is not None
    assert ivivc.analysis_model == settings.analysis_model
    assert ivivc.observation_model.value == settings.observation_sd
    assert {'in_vitro_time', 'in_vivo_time'}.issubset(ivivc.observed_data)
    assert ivivc.preprocessed_data is not None
    assert ivivc.metadata['ivivc_preprocessing_interpolation'] == 'default'
    assert ivivc.metadata['ivivc_calibration_response_sd'] == (
        settings.ivivc_calibration_response_sd
    )

    assert adaptive.metadata['reference_source'] == 'partial_target_in_vivo_study'
    assert adaptive.analysis_model == settings.analysis_model
    assert adaptive.ground_truth is not None
    assert adaptive.observation_model.value == settings.observation_sd
    assert adaptive.metadata['reference_data_sd'] == settings.observation_sd
    np.testing.assert_allclose(
        adaptive.constraints.fixed_times,
        adaptive.metadata['completed_times'],
    )
    assert adaptive.constraints.min_number_of_times == len(
        adaptive.constraints.fixed_times
    ) + 1
    assert adaptive.name == 'adaptive_in_vivo_only'
    assert adaptive.existing_observations is not None
    assert np.all(
        adaptive.constraints.candidate_times > adaptive.existing_observations.latest_time
    )

    assert adaptive_ivivc.name == 'adaptive_ivivc_prediction'
    assert adaptive_ivivc.existing_observations is not None
    assert adaptive_ivivc.metadata['effective_sample_size'] > 0
    assert adaptive_ivivc.preprocessed_data is not None
    assert np.all(
        adaptive_ivivc.constraints.candidate_times
        > adaptive_ivivc.existing_observations.latest_time
    )


def test_proof_of_concept_uses_common_optimizer_and_simulation_pipeline():
    run = run_proof_of_concept(
        ('measured_in_vivo', 'adaptive_refinement'),
        settings=_small_settings(),
    )

    assert set(run.scenarios) == {'measured_in_vivo', 'adaptive_refinement'}
    for result in run.scenarios.values():
        assert result.optimization.metadata['criterion'] == 'local_i_optimality'
        assert result.optimization.metadata['parameter_source'] == 'central_reference_fit'
        assert result.optimization.metadata['reference_uncertainty_used_in_optimization'] is False
        assert result.optimization.metadata['reference_uncertainty_evaluated_by_monte_carlo'] is True
        assert len(result.comparisons_by_n) == 1
        labels = set(result.performance_table['design'])
        assert {'optimized_nested', 'optimized_refined', 'equal_time'} <= labels
        assert {
            DESIGN_LABELS['optimized_nested'],
            DESIGN_LABELS['optimized_refined'],
        } <= set(result.performance_table['design_label'])
        optimized = result.performance_table[
            result.performance_table['design'].isin(['optimized_nested', 'optimized_refined'])
        ]
        assert np.all(
            optimized['information_parameter_count'].to_numpy(dtype=float)
            == len(result.reference_model.parameter_names)
        )
        assert np.all(
            optimized['information_rank'].to_numpy(dtype=float)
            <= optimized['information_parameter_count'].to_numpy(dtype=float)
        )
        assert set(optimized['information_selection_rule'].dropna()) <= {
            'i_optimality',
            'rank_pseudodeterminant_fallback',
        }
        assert np.all(np.isfinite(result.performance_table['failure_rate']))
        assert np.all(
            np.isfinite(
                result.performance_table['penalized_mean_normalized_irmse']
            )
        )

    adaptive = run.scenarios['adaptive_refinement']
    fixed = adaptive.definition.constraints.fixed_times
    optimized = adaptive.selected_times_table[
        adaptive.selected_times_table['design'] == 'optimized_refined'
    ]['time'].to_numpy(dtype=float)
    for value in fixed:
        assert np.any(np.isclose(optimized, value, rtol=0.0, atol=1e-12))


def test_output_writer_creates_transparent_machine_readable_bundle(tmp_path):
    run = run_proof_of_concept(
        ('ivivc_prediction',),
        settings=_small_settings(write_html=False),
    )
    written = write_proof_of_concept_outputs(run, tmp_path)

    assert written.output_directory == tmp_path
    scenario_dir = tmp_path / 'ivivc_prediction'
    expected = {
        'observed_or_calibration_data.csv',
        'ivivc_preprocessed_pairs.csv',
        'reference_trajectory.csv',
        'reference_endpoints.csv',
        'design_performance.csv',
        'selected_times.csv',
        'metadata.json',
    }
    assert expected == {path.name for path in scenario_dir.iterdir()}
    assert (tmp_path / 'combined_design_performance.csv').exists()
    assert (tmp_path / 'combined_selected_times.csv').exists()
    assert (tmp_path / 'manifest.json').exists()
    assert not (tmp_path / 'index.html').exists()

    performance = pd.read_csv(scenario_dir / 'design_performance.csv')
    assert {
        'design',
        'design_label',
        'number_of_times',
        'complete_schedule',
        'new_times',
        'penalized_mean_irmse',
    } <= set(performance.columns)
    reference = pd.read_csv(scenario_dir / 'reference_trajectory.csv')
    assert 'ground_truth_response' in reference.columns
    assert np.all(np.isfinite(reference['ground_truth_response']))
    preprocessed = pd.read_csv(scenario_dir / 'ivivc_preprocessed_pairs.csv')
    assert {
        'response_level',
        'in_vitro_time_at_response',
        'in_vivo_time_at_response',
        'used_for_approach1_fit',
    } <= set(preprocessed.columns)
    assert preprocessed['used_for_approach1_fit'].any()

    metadata = json.loads((scenario_dir / 'metadata.json').read_text())
    assert metadata['name'] == 'ivivc_prediction'
    assert metadata['constraints']['fixed_times'] == [0.0]
    assert metadata['ground_truth']['available'] is True
    assert metadata['output_interpretation']['primary_evaluation'] == (
        'monte_carlo_curve_recovery'
    )
    assert metadata['output_interpretation']['design_labels']['optimized_nested'] == (
        DESIGN_LABELS['optimized_nested']
    )



def test_adaptive_outputs_separate_completed_and_new_future_times():
    run = run_proof_of_concept(
        ('adaptive_refinement',),
        settings=_small_settings(),
    )
    result = run.scenarios['adaptive_refinement']
    assert result.definition.existing_observations is not None

    performance = result.performance_table
    optimized = performance[performance['design'] == 'optimized_refined'].iloc[0]
    assert optimized['number_of_existing_times'] == (
        result.definition.existing_observations.n_time_points
    )
    assert optimized['number_of_new_times'] == (
        optimized['number_of_times'] - optimized['number_of_existing_times']
    )
    new_times = np.array(
        [float(value) for value in optimized['new_times'].split('; ')],
        dtype=float,
    )
    assert np.all(new_times > result.definition.existing_observations.latest_time)

    selected = result.selected_times_table[
        result.selected_times_table['design'] == 'optimized_refined'
    ]
    assert 'completed observation' in set(selected['time_role'])
    assert 'recommended future time' in set(selected['time_role'])
    assert np.all(
        selected.loc[selected['time_role'] == 'recommended future time', 'time']
        > result.definition.existing_observations.latest_time
    )

def test_runner_model_and_absolute_noise_are_configurable():
    settings = ProofOfConceptSettings(
        n_reference_draws=8,
        n_simulations=8,
        random_seed=4321,
        refine=False,
        write_html=False,
        max_design_sizes=1,
        analysis_model='exponential',
        observation_sd=0.07,
        fit_initialization='central_reference',
    )
    scenario = build_measured_in_vivo_scenario(settings)

    assert scenario.analysis_model == 'exponential'
    assert scenario.metadata['generating_model'] == 'exponential'
    assert scenario.observation_model.kind == 'constant_sd'
    assert scenario.observation_model.value == 0.07
    assert scenario.metadata['reference_data_sd'] == 0.07
    assert settings.fit_initialization == 'central_reference'


def test_html_summary_includes_ground_truth_and_explains_design_labels(tmp_path):
    run = run_proof_of_concept(
        ('measured_in_vivo',),
        settings=_small_settings(write_html=True),
    )
    write_proof_of_concept_outputs(run, tmp_path)

    html = (tmp_path / 'measured_in_vivo' / 'summary.html').read_text()
    assert 'Synthetic ground truth' in html
    assert DESIGN_LABELS['optimized_nested'] in html
    assert DESIGN_LABELS['optimized_refined'] in html
    assert 'Both schedules are information-based' in html
    assert 'Monte Carlo performance summary' in html
    assert 'Optimization diagnostics (supporting)' in html
