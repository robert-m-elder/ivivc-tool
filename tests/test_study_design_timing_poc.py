from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import examples.study_design_proof_of_concept as poc
from study_design_core import ReferenceTrajectoryEnsemble, reference_crossing_time_summary


POLICIES = {'central_next_time', 'direct_quantile', 'deadline_i_optimal'}


def _settings(**changes) -> poc.ProofOfConceptSettings:
    settings = poc.ProofOfConceptSettings(
        n_reference_draws=30,
        n_simulations=8,
        random_seed=20261021,
        refine=True,
        write_html=False,
        max_design_sizes=3,
        analysis_model='mass_loss_one_phase_normalized',
    )
    return replace(settings, **changes)


@pytest.fixture(scope='module')
def timing_run():
    return poc.run_proof_of_concept(
        ('adaptive_timing_comparison',), settings=_settings(),
    )


def test_timing_example_is_one_future_point_from_partial_in_vivo_data():
    settings = _settings(adaptive_candidate_policy='any_unobserved')
    scenario = poc.build_adaptive_timing_comparison_scenario(settings)
    original = poc.build_adaptive_in_vivo_only_scenario(settings)

    assert scenario.name == 'adaptive_timing_comparison'
    assert scenario.metadata['reference_source'] == 'partial_target_in_vivo_study'
    assert scenario.metadata['adaptive_candidate_policy'] == 'future_only'
    assert original.metadata['adaptive_candidate_policy'] == 'any_unobserved'
    existing = scenario.existing_observations
    assert existing is not None
    np.testing.assert_array_equal(scenario.constraints.fixed_times, existing.unique_times)
    assert np.all(scenario.constraints.candidate_times > existing.latest_time)
    assert scenario.constraints.min_number_of_times == existing.n_time_points + 1
    assert scenario.constraints.max_number_of_times == existing.n_time_points + 1


def test_timing_example_compares_one_schedule_per_policy(timing_run):
    result = timing_run.scenarios['adaptive_timing_comparison']
    summary = result.timing_summary
    policies = result.timing_policy_table.set_index('design')
    assert summary is not None and summary.status == 'available'
    assert set(policies.index) == POLICIES
    assert set(policies['status']) == {'available'}
    assert policies.index.is_unique
    assert result.optimization.refined_steps == ()
    assert len(result.comparisons_by_n) == 1

    existing = result.definition.existing_observations
    size = existing.n_time_points + 1
    comparison = result.comparisons_by_n[size]
    assert POLICIES <= set(comparison.results)
    assert len(result.performance_table) == len(comparison.results)
    assert set(result.performance_table['number_of_times']) == {size}
    assert not any('refined' in code for code in comparison.results)
    assert (result.performance_table['design'] == 'equal_time').sum() <= 1
    assert summary.degraded_fraction == timing_run.settings.timing_degraded_fraction
    assert summary.quantile == timing_run.settings.timing_quantile

    for code in POLICIES:
        simulation = comparison.results[code]
        result.definition.constraints.validate_design(simulation.design_times)
        assert len(simulation.design_times) == size
        assert all(time in simulation.design_times for time in existing.unique_times)
        selected = float(policies.loc[code, 'new_time'])
        assert selected > existing.latest_time
        if code != 'central_next_time':
            assert selected <= summary.deadline
            np.testing.assert_allclose(
                policies.loc[code, 'deadline_slack'], summary.deadline - selected,
            )
        np.testing.assert_allclose(
            policies.loc[code, 'crossed_weight_before'],
            summary.crossed_weight_before(selected),
        )
        assert np.isfinite(policies.loc[code, 'information_criterion'])


def test_timing_policies_share_simulated_trajectories_and_completed_values(timing_run):
    result = timing_run.scenarios['adaptive_timing_comparison']
    comparison = next(iter(result.comparisons_by_n.values()))
    plan = comparison.simulation_plan
    assert plan.metadata['common_random_numbers'] is True
    first = next(iter(comparison.results.values()))
    existing = result.definition.existing_observations
    completed_times, completed_means = existing.mean_response_by_time()
    for simulation in comparison.results.values():
        np.testing.assert_array_equal(
            simulation.trajectory_draw_indices, plan.trajectory_draw_indices,
        )
        np.testing.assert_allclose(simulation.true_curves, first.true_curves)
        columns = np.searchsorted(simulation.design_times, completed_times)
        np.testing.assert_allclose(
            simulation.observed_means[:, columns],
            np.broadcast_to(completed_means, (plan.n_simulations, len(completed_times))),
        )
        shared = np.intersect1d(first.design_times, simulation.design_times)
        np.testing.assert_allclose(
            first.observed_means[:, np.searchsorted(first.design_times, shared)],
            simulation.observed_means[:, np.searchsorted(simulation.design_times, shared)],
        )


def test_timing_settings_change_the_actual_deadline_and_recommendations(timing_run):
    early = timing_run.scenarios['adaptive_timing_comparison']
    later_run = poc.run_proof_of_concept(
        ('adaptive_timing_comparison',), settings=_settings(timing_quantile=0.9),
    )
    later = later_run.scenarios['adaptive_timing_comparison']
    assert early.timing_summary.deadline < later.timing_summary.deadline
    early_policies = early.timing_policy_table.set_index('design')
    later_policies = later.timing_policy_table.set_index('design')
    for code in ('direct_quantile', 'deadline_i_optimal'):
        assert early_policies.loc[code, 'new_time'] <= later_policies.loc[code, 'new_time']
    assert early_policies.loc['direct_quantile', 'new_time'] < (
        later_policies.loc['direct_quantile', 'new_time']
    )


def test_stretched_exponential_uses_the_same_timing_policy():
    settings = _settings(analysis_model='stretched_exponential', n_simulations=4)
    run = poc.run_proof_of_concept(('adaptive_timing_comparison',), settings=settings)
    result = run.scenarios['adaptive_timing_comparison']
    summary = result.timing_summary
    assert summary.status == 'available'
    assert summary.degraded_fraction == settings.timing_degraded_fraction
    assert summary.quantile == settings.timing_quantile
    table = result.timing_policy_table.set_index('design')
    for code in ('direct_quantile', 'deadline_i_optimal'):
        assert table.loc[code, 'status'] == 'available'
        assert result.definition.existing_observations.latest_time < (
            table.loc[code, 'new_time']
        ) <= summary.deadline


def test_elapsed_milestone_is_reported_without_fake_performance(tmp_path):
    run = poc.run_proof_of_concept(
        ('adaptive_timing_comparison',),
        settings=_settings(timing_degraded_fraction=0.1),
    )
    result = run.scenarios['adaptive_timing_comparison']
    assert result.timing_summary.deadline < result.definition.existing_observations.latest_time
    table = result.timing_policy_table.set_index('design')
    assert table.loc['central_next_time', 'status'] == 'available'
    for code in ('direct_quantile', 'deadline_i_optimal'):
        assert table.loc[code, 'status'] == 'deadline_elapsed'
        assert pd.isna(table.loc[code, 'new_time'])
        assert table.loc[code, 'message']
        assert code not in set(result.performance_table['design'])
        assert code not in set(result.selected_times_table['design'])
    poc.write_proof_of_concept_outputs(run, tmp_path)
    metadata_text = (tmp_path / 'adaptive_timing_comparison' / 'metadata.json').read_text()

    def reject_nonfinite(token):
        raise AssertionError(f'Timing metadata contains nonstandard JSON token {token}')

    metadata = json.loads(metadata_text, parse_constant=reject_nonfinite)
    unavailable = {
        row['design']: row
        for row in metadata['timing_policies']
        if row['status'] != 'available'
    }
    assert set(unavailable) == {'direct_quantile', 'deadline_i_optimal'}
    assert all(row['new_time'] is None for row in unavailable.values())


@pytest.mark.parametrize('status', ['beyond_horizon', 'invalid_reference'])
def test_unavailable_reference_timing_preserves_policy_status(monkeypatch, status):
    settings = _settings()
    scenario = poc.build_adaptive_timing_comparison_scenario(settings)
    reference = scenario.reference
    values = np.ones((1, len(reference.time_grid)))
    if status == 'invalid_reference':
        values[:] = 0.0
    unavailable = ReferenceTrajectoryEnsemble(
        time_grid=reference.time_grid,
        value_draws=values,
        valid_draws=np.array([True]),
        supported_time_range=reference.supported_time_range,
        interpolation='linear',
    )
    summary = reference_crossing_time_summary(unavailable)
    assert summary.status == status
    monkeypatch.setattr(poc, 'reference_crossing_time_summary', lambda *args, **kwargs: summary)
    result = poc.run_scenario(scenario, settings)

    table = result.timing_policy_table.set_index('design')
    assert set(table.index) == POLICIES
    assert table.loc['central_next_time', 'status'] == 'available'
    for code in ('direct_quantile', 'deadline_i_optimal'):
        assert table.loc[code, 'status'] == status
        assert pd.isna(table.loc[code, 'new_time'])
        assert code not in set(result.performance_table['design'])
    assert 'central_next_time' in set(result.performance_table['design'])


def test_timing_csv_and_metadata_record_policy_provenance(timing_run, tmp_path):
    poc.write_proof_of_concept_outputs(timing_run, tmp_path)
    directory = tmp_path / 'adaptive_timing_comparison'
    table = pd.read_csv(directory / 'timing_policies.csv')
    assert set(table['design']) == POLICIES
    assert set(table['quantile']) == {timing_run.settings.timing_quantile}
    assert set(table['degraded_fraction']) == {timing_run.settings.timing_degraded_fraction}
    assert {'status', 'message', 'deadline', 'new_time', 'crossed_weight_before'} <= set(table)
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['timing_summary']['quantile'] == timing_run.settings.timing_quantile
    assert metadata['timing_summary']['degraded_fraction'] == (
        timing_run.settings.timing_degraded_fraction
    )
    assert {row['design'] for row in metadata['timing_policies']} == POLICIES
    assert metadata['timing_summary']['status'] == 'available'
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['settings']['timing_quantile'] == timing_run.settings.timing_quantile
    assert manifest['settings']['timing_degraded_fraction'] == (
        timing_run.settings.timing_degraded_fraction
    )


def test_timing_html_shows_deadline_and_all_policy_labels(timing_run, tmp_path):
    pytest.importorskip('plotly')
    run = replace(timing_run, settings=replace(timing_run.settings, write_html=True))
    poc.write_proof_of_concept_outputs(run, tmp_path)
    html = (tmp_path / 'adaptive_timing_comparison' / 'summary.html').read_text()
    for code in POLICIES:
        assert poc.TIMING_POLICY_LABELS[code] in html
    assert 'deadline' in html.lower()
    assert '10th percentile' in html or '10%' in html
    assert '50%' in html


def test_existing_scenarios_do_not_receive_timing_outputs(tmp_path):
    run = poc.run_proof_of_concept(('adaptive_in_vivo_only',), settings=_settings())
    result = run.scenarios['adaptive_in_vivo_only']
    assert result.timing_summary is None
    assert result.timing_policy_table is None
    assert not POLICIES.intersection(result.performance_table['design'])
    assert {'optimized_nested', 'optimized_refined'} <= set(result.performance_table['design'])
    poc.write_proof_of_concept_outputs(run, tmp_path)
    assert not (tmp_path / 'adaptive_in_vivo_only' / 'timing_policies.csv').exists()


@pytest.mark.parametrize('field', ['timing_degraded_fraction', 'timing_quantile'])
@pytest.mark.parametrize('value', [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_timing_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _settings(**{field: value})


def test_timing_scenario_and_settings_are_available_from_cli():
    defaults = poc._parse_args([])
    assert defaults.timing_degraded_fraction == 0.5
    assert defaults.timing_quantile == 0.1
    parsed = poc._parse_args([
        '--scenario', 'adaptive_timing_comparison',
        '--timing-degraded-fraction', '0.2', '--timing-quantile', '0.3',
    ])
    assert parsed.scenario == 'adaptive_timing_comparison'
    assert parsed.timing_degraded_fraction == 0.2
    assert parsed.timing_quantile == 0.3
