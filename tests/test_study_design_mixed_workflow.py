from __future__ import annotations

from dataclasses import replace
import json
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import examples.study_design_proof_of_concept as poc
from examples.study_design_mixed_workflow import (
    PLANNING_INPUT_MODES,
    SchedulePlanningSpec,
    planning_json_safe,
    planning_summary_html,
)
from ivivc_core import TimeCourse
from study_design_core import CrossingTimeTarget, LatestBefore, PlanningContext, SamplingWindow


def _settings(**changes):
    return replace(poc.ProofOfConceptSettings(
        n_reference_draws=30, n_simulations=4, max_design_sizes=1,
        analysis_model='mass_loss_one_phase_normalized', write_html=False,
    ), **changes)


@pytest.fixture(scope='module')
def mode_runs():
    return {mode: poc.run_proof_of_concept(
        ('mixed_timing_requirements',), settings=_settings(planning_input_mode=mode),
    ) for mode in PLANNING_INPUT_MODES}


def _result(run):
    return run.scenarios['mixed_timing_requirements']


@pytest.mark.parametrize('mode', PLANNING_INPUT_MODES)
def test_one_example_supports_all_four_input_modes(mode_runs, mode):
    result = _result(mode_runs[mode])
    definition = result.definition
    assert definition.planning.input_mode == mode
    assert set(result.schedule_status_table['status']) == {'feasible'}
    assert set(result.performance_table['design']) == {'central_schedule', 'timing_schedule'}
    assert set(result.requirement_table['status']) == {'satisfied'}
    assert len(result.requirement_table) == 3
    assert len(result.anchor_table) == 2
    assert set(result.anchor_table['status']) == {'selected'}
    assert result.planning_optimization.metadata['independent_final_sizes'] is True
    if mode.startswith('prospective'):
        assert definition.existing_observations is None
        assert definition.planning.context.completed_times == ()
        assert set(result.performance_table['number_of_existing_times']) == {0}
        assert set(result.performance_table['number_of_times']) == {4}
    else:
        assert definition.existing_observations is not None
        assert set(result.performance_table['number_of_existing_times']) == {3}
        assert set(result.performance_table['number_of_new_times']) == {3}
        assert set(result.performance_table['number_of_times']) == {6}
    if 'ivivc' in mode:
        assert definition.preprocessed_data is not None
    if mode == 'adaptive-ivivc':
        assert definition.reference.metadata['adaptive_reweighting'] is True


@pytest.mark.parametrize('mode', PLANNING_INPUT_MODES)
def test_reference_weights_and_completed_target_data_are_preserved(mode_runs, mode):
    result = _result(mode_runs[mode])
    reference = result.definition.reference
    resolved = result.planning_optimization.resolved_requirements
    for row in resolved.requirements:
        np.testing.assert_allclose(row.upper_summary.draw_weights, reference.draw_weights)
        assert len(row.upper_summary.crossing_times) == reference.n_draws
    for comparison in result.comparisons_by_n.values():
        first, second = (comparison.results[code] for code in ('central_schedule', 'timing_schedule'))
        np.testing.assert_array_equal(first.trajectory_draw_indices, second.trajectory_draw_indices)
        np.testing.assert_allclose(first.true_curves, second.true_curves)
        shared = np.intersect1d(first.design_times, second.design_times)
        np.testing.assert_allclose(
            first.observed_means[:, np.searchsorted(first.design_times, shared)],
            second.observed_means[:, np.searchsorted(second.design_times, shared)],
        )
        existing = result.definition.existing_observations
        if existing is not None:
            times, means = existing.mean_response_by_time()
            for simulation in comparison.results.values():
                np.testing.assert_allclose(
                    simulation.observed_means[:, np.searchsorted(simulation.design_times, times)],
                    np.broadcast_to(means, (simulation.n_simulations, len(times))),
                )


def test_time_roles_distinguish_planned_baseline_completed_and_reserved_points(mode_runs):
    prospective = _result(mode_runs['prospective-measured'])
    baseline = prospective.selected_times_table.query('baseline_time')
    assert set(baseline['time_role']) == {'planned baseline'}
    assert baseline['new_study_time'].all()
    assert not baseline['existing_observation'].any()
    adaptive = _result(mode_runs['adaptive-in-vivo'])
    selected = adaptive.selected_times_table
    completed = selected[selected['existing_observation']]
    assert set(completed['time']) == {0.0, 10.0, 20.0}
    assert not completed['new_study_time'].any()
    anchors = selected[selected['latest_before_point']]
    assert set(anchors['design']) == {'timing_schedule'}
    assert anchors['fixed'].all() and anchors['new_study_time'].all()
    assert not anchors['original_fixed'].any()
    assert not selected.loc[selected['design'] == 'central_schedule', 'latest_before_point'].any()


def test_multiple_study_sizes_share_one_simulation_plan():
    run = poc.run_proof_of_concept(('mixed_timing_requirements',), settings=_settings(max_design_sizes=2))
    result = _result(run)
    assert list(result.comparisons_by_n) == [6, 7]
    comparisons = list(result.comparisons_by_n.values())
    assert comparisons[0].simulation_plan is comparisons[1].simulation_plan
    assert set(result.schedule_status_table['number_of_times']) == {6, 7}
    assert len(result.anchor_table) == 4


def test_planning_specification_drives_the_run_without_a_special_scenario_name():
    settings = _settings(n_reference_draws=8)
    original = poc.build_measured_in_vivo_scenario(settings)
    definition = replace(original, planning=SchedulePlanningSpec(
        PlanningContext([0.0]), (SamplingWindow(0.0, 30.0), LatestBefore(30.0)),
        'prospective-measured',
    ))
    result = poc.run_scenario(definition, settings)
    assert result.definition.name == 'measured_in_vivo'
    assert result.planning_optimization is not None
    assert set(result.performance_table['design']) == {'central_schedule', 'timing_schedule'}


def test_historical_reference_times_cannot_be_declared_as_completed_target_data():
    original = poc.build_measured_in_vivo_scenario(_settings(n_reference_draws=8))
    with pytest.raises(ValueError, match='completed_times'):
        replace(original, planning=SchedulePlanningSpec(
            PlanningContext([0.0], [0.0, 10.0]), (SamplingWindow(10.0, 30.0),),
            'prospective-measured',
        ))


def test_fixed_future_and_latest_points_count_toward_new_sampling_once():
    settings = _settings()
    definition = poc.build_mixed_timing_requirements_scenario(settings)
    definition = replace(definition,
        constraints=replace(definition.constraints, fixed_times=np.array([0., 10., 20., 35.])),
        planning=replace(definition.planning, requirements=(
            SamplingWindow(20.0, 36.0), LatestBefore(35.0),
        )),
    )
    result = poc.run_scenario(definition, settings)
    committed = result.selected_times_table.query('fixed_future_commitment')
    assert set(committed['time']) == {35.0}
    assert committed['new_study_time'].all()
    assert len(result.selected_times_table.query('design == "timing_schedule"')) == 6
    assert result.anchor_table.iloc[0]['selected_time'] == 35.0
    assert result.requirement_table.iloc[0]['achieved_count'] >= 1


def test_fully_committed_future_schedule_needs_no_free_candidates(tmp_path):
    settings = _settings()
    definition = poc.build_mixed_timing_requirements_scenario(settings)
    times = np.array([0.0, 10.0, 20.0, 25.0, 35.0])
    definition = replace(definition,
        constraints=replace(definition.constraints, candidate_times=times, fixed_times=times,
                            min_number_of_times=5, max_number_of_times=5),
        planning=replace(definition.planning, requirements=(SamplingWindow(20.0, 30.0), LatestBefore(35.0))),
    )
    result = poc.run_scenario(definition, settings)
    assert set(result.schedule_status_table['status']) == {'feasible'}
    assert set(result.performance_table['number_of_new_times']) == {2}
    run = poc.ProofOfConceptRun(settings, {definition.name: result})
    poc.write_proof_of_concept_outputs(run, tmp_path)
    metadata = json.loads((tmp_path / definition.name / 'metadata.json').read_text())
    assert metadata['adaptive_summary']['earliest_candidate_time'] is None


@pytest.mark.parametrize('fraction,quantile,expected', [
    (0.10, 0.10, 'deadline_elapsed'),
    (0.99, 0.99, 'beyond_horizon'),
])
def test_unavailable_requirements_remain_visible_without_fake_performance(fraction, quantile, expected):
    run = poc.run_proof_of_concept(('mixed_timing_requirements',), settings=_settings(
        timing_degraded_fraction=fraction, timing_quantile=quantile,
    ))
    result = _result(run)
    assert expected in set(result.requirement_table['status'])
    assert set(result.performance_table['design']) == {'central_schedule'}
    assert not (result.selected_times_table['design'] == 'timing_schedule').any()
    assert len(result.schedule_status_table) == 2
    assert len(result.anchor_table) == 2
    assert result.anchor_table['selected_time'].isna().all()


def test_search_limits_can_leave_all_schedules_unselected_without_output_failure(tmp_path):
    run = poc.run_proof_of_concept(('mixed_timing_requirements',), settings=_settings(planning_search_limit=1))
    result = _result(run)
    assert set(result.schedule_status_table['status']) == {'search_limit'}
    assert result.performance_table.empty
    assert result.selected_times_table.empty
    assert result.comparisons_by_n == {}
    poc.write_proof_of_concept_outputs(run, tmp_path)
    directory = tmp_path / 'mixed_timing_requirements'
    assert pd.read_csv(directory / 'design_performance.csv').empty
    assert pd.read_csv(directory / 'selected_times.csv').empty
    assert len(pd.read_csv(directory / 'schedule_status.csv')) == 2
    metadata = json.loads((directory / 'metadata.json').read_text(),
                          parse_constant=lambda value: pytest.fail('Nonstandard JSON: ' + value))
    assert metadata['planning']['schedule_statuses'][0]['information_criterion'] is None


def test_machine_readable_outputs_include_definitions_counts_and_anchor_priority(mode_runs, tmp_path):
    run = mode_runs['adaptive-in-vivo']
    poc.write_proof_of_concept_outputs(run, tmp_path)
    directory = tmp_path / 'mixed_timing_requirements'
    for filename in ('schedule_status.csv', 'timing_requirements.csv', 'latest_before_points.csv'):
        assert (directory / filename).exists()
    metadata = json.loads((directory / 'metadata.json').read_text(),
                          parse_constant=lambda value: pytest.fail('Nonstandard JSON: ' + value))
    planning = metadata['planning']
    assert planning['input_mode'] == 'adaptive-in-vivo'
    assert planning['completed_times'] == [0.0, 10.0, 20.0]
    assert planning['requirements'][0]['upper_target']['degraded_fraction'] == 0.5
    assert planning['requirements'][0]['upper_crossing_summary']['quantile'] == 0.1
    assert {row['priority'] for row in planning['latest_before_results']} == {1, 2}
    assert planning['central_benchmark_applies_timing_requirements'] is False
    assert set(planning['central_analysis_parameters']) == {'t50', 's'}
    assert planning['information_evaluation_times'][0] == 0.0
    assert planning['information_evaluation_times'][-1] == 52.0
    assert planning['monte_carlo']['shared_simulation_plan'] is True
    assert metadata['output_interpretation']['reference_uncertainty_averaged_into_information_criterion'] is False
    assert metadata['ground_truth']['used_for_optimization'] is False
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['output_interpretation']['planning_input_modes'] == {'mixed_timing_requirements': 'adaptive-in-vivo'}


def test_switching_input_mode_removes_stale_calibration_outputs(mode_runs, tmp_path):
    directory = tmp_path / 'mixed_timing_requirements'
    poc.write_proof_of_concept_outputs(mode_runs['prospective-ivivc'], tmp_path)
    assert (directory / 'ivivc_preprocessed_pairs.csv').exists()
    (directory / 'summary.html').write_text('stale output')
    poc.write_proof_of_concept_outputs(mode_runs['adaptive-in-vivo'], tmp_path)
    assert not (directory / 'ivivc_preprocessed_pairs.csv').exists()
    assert not (directory / 'summary.html').exists()


def test_json_sanitization_handles_nested_numpy_unavailable_values():
    value = {'values': np.array([np.inf, np.nan, 2.0]), 'nested': (np.float64(-np.inf),)}
    encoded = json.dumps(planning_json_safe(value), allow_nan=False)
    assert json.loads(encoded) == {'values': [None, None, 2.0], 'nested': [None]}


def test_quantile_settings_change_the_resolved_bounds_used_by_optimization(mode_runs):
    early = _result(mode_runs['prospective-measured'])
    later = _result(poc.run_proof_of_concept(('mixed_timing_requirements',), settings=_settings(
        planning_input_mode='prospective-measured', timing_quantile=0.9,
    )))
    assert later.planning_optimization.resolved_requirements.requirements[0].upper_bound > (
        early.planning_optimization.resolved_requirements.requirements[0].upper_bound
    )
    assert later.anchor_table.iloc[-1]['deadline'] > early.anchor_table.iloc[-1]['deadline']


def test_other_models_use_the_same_planning_specification():
    for changes in (
        {'analysis_model': 'stretched_exponential'},
        {'analysis_model': 'exponential', 'timing_degraded_fraction': 0.9,
         'timing_quantile': 0.5, 'planning_latest_fractions': (0.93, 0.97)},
    ):
        result = _result(poc.run_proof_of_concept(('mixed_timing_requirements',), settings=_settings(**changes)))
        assert result.definition.analysis_model == changes['analysis_model']
        assert result.planning_optimization.design_for_n(6).requirement_evaluation.status == 'satisfied'


def test_synthetic_ground_truth_does_not_select_the_schedule():
    settings = _settings(n_simulations=2)
    definition = poc.build_mixed_timing_requirements_scenario(settings)
    before = poc.run_scenario(definition, settings)
    altered = replace(definition, ground_truth=TimeCourse(
        definition.ground_truth.time, 1000.0 * definition.ground_truth.value,
        label='changed evaluation curve',
    ))
    after = poc.run_scenario(altered, settings)
    assert before.planning_optimization.design_for_n(6).times == after.planning_optimization.design_for_n(6).times


@pytest.mark.parametrize('changes', [
    {'planning_input_mode': 'unknown'}, {'planning_search_limit': 0},
    {'planning_search_limit': True}, {'planning_latest_fractions': ()},
    {'planning_latest_fractions': (0.0,)}, {'planning_latest_fractions': (np.nan,)},
])
def test_invalid_planning_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        _settings(**changes)


def test_cli_exposes_one_example_with_selectable_mode_and_multiple_milestones():
    args = poc._parse_args([
        '--scenario', 'mixed_timing_requirements', '--planning-input-mode', 'adaptive-ivivc',
        '--planning-latest-fraction', '0.8', '--planning-latest-fraction', '0.95',
    ])
    assert args.planning_input_mode == 'adaptive-ivivc'
    assert args.planning_latest_fraction == [0.8, 0.95]
    assert poc._parse_args([]).planning_latest_fraction is None


class _Figure:
    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def add_vrect(self, **kwargs):
        assert kwargs['x0'] < kwargs['x1']

    def add_vline(self, **kwargs):
        pass

    def update_layout(self, **kwargs):
        pass


def _template_html(result):
    go = SimpleNamespace(Figure=_Figure, Scatter=lambda **kwargs: kwargs)
    def to_html(figure, **kwargs):
        return '<div>' + '; '.join(trace.get('name', '') for trace in figure.traces) + '</div>'
    return planning_summary_html(result, poc._reference_table(
        result.definition.reference, result.definition.ground_truth,
    ), go, to_html)


def test_html_template_displays_requirements_and_roles(mode_runs):
    for run in mode_runs.values():
        result = _result(run)
        html = _template_html(result)
        assert 'Latest-before selections' in html
        assert 'Synthetic ground truth' in html
        assert result.definition.planning.input_mode in html
        assert 'Early curve coverage' in html


def test_html_handles_no_simulated_schedules_and_a_missing_lower_bound(mode_runs):
    settings = _settings(planning_search_limit=1)
    limited = _result(poc.run_proof_of_concept(('mixed_timing_requirements',), settings=settings))
    assert 'No feasible schedules were simulated' in _template_html(limited)
    original = _result(mode_runs['adaptive-in-vivo']).definition
    unavailable = replace(original, planning=replace(original.planning, requirements=(
        SamplingWindow(CrossingTimeTarget(0.99, 0.99), 52.0, label='unavailable lower bound'),
        LatestBefore(40.0),
    )))
    result = poc.run_scenario(unavailable, _settings())
    assert result.planning_optimization.resolved_requirements.requirements[0].lower_bound is None
    assert 'beyond horizon' in _template_html(result)


def test_actual_plotly_summary_contains_planning_tables(mode_runs):
    pytest.importorskip('plotly')
    html = poc._plotly_summary_html(_result(mode_runs['adaptive-in-vivo']))
    assert 'Latest-before selections' in html
    assert 'Schedule status' in html
    assert 'Central I-optimal schedule' in html


def test_html_writer_dispatch_and_index_with_plotly_stand_in(mode_runs, tmp_path):
    package, graph, io = (ModuleType(name) for name in ('plotly', 'plotly.graph_objects', 'plotly.io'))
    package.__path__ = []
    graph.Figure = _Figure
    graph.Scatter = lambda **kwargs: kwargs
    io.to_html = lambda figure, **kwargs: '<div>Figure placeholder</div>'
    run = mode_runs['adaptive-in-vivo']
    run = replace(run, settings=replace(run.settings, write_html=True))
    with patch.dict('sys.modules', {'plotly': package, 'plotly.graph_objects': graph, 'plotly.io': io}):
        poc.write_proof_of_concept_outputs(run, tmp_path)
    html = (tmp_path / 'mixed_timing_requirements' / 'summary.html').read_text()
    assert 'Latest-before selections' in html
    assert 'Milestone 1' in html
    assert 'joint latest-before' in (tmp_path / 'index.html').read_text()


def test_original_examples_do_not_receive_planning_fields_or_output_tables(tmp_path):
    run = poc.run_proof_of_concept(('adaptive_timing_comparison',), settings=_settings())
    result = run.scenarios['adaptive_timing_comparison']
    assert result.definition.planning is None and result.planning_optimization is None
    poc.write_proof_of_concept_outputs(run, tmp_path)
    directory = tmp_path / 'adaptive_timing_comparison'
    assert not (directory / 'schedule_status.csv').exists()
    assert 'planning' not in json.loads((directory / 'metadata.json').read_text())
