from __future__ import annotations

from dataclasses import replace
import json
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import examples.study_design_proof_of_concept as poc
from examples.study_design_time_scaling_poc import (
    load_time_scale_distribution,
    time_scaling_summary_html,
    time_scaling_tables,
)


SCENARIO = 'empirical_time_scaling'


def _settings(**changes):
    values = dict(
        n_reference_draws=8, n_simulations=4, max_design_sizes=1,
        random_seed=1234, refine=False, write_html=False,
        analysis_model='mass_loss_one_phase_normalized',
        ivivc_calibration_response_sd=.005,
    )
    values.update(changes)
    return poc.ProofOfConceptSettings(**values)


@pytest.fixture(scope='module')
def mode_runs():
    return {
        mode: poc.run_proof_of_concept((SCENARIO,), settings=_settings(time_scaling_mode=mode))
        for mode in ('prospective', 'adaptive')
    }


@pytest.fixture(scope='module')
def mixed_runs():
    return {
        mode: poc.run_proof_of_concept(
            ('mixed_timing_requirements',), settings=_settings(planning_input_mode=mode),
        )
        for mode in ('prospective-time-scaling', 'adaptive-time-scaling')
    }


def test_one_new_example_preserves_existing_seed_order_and_exposes_cli():
    assert poc.SCENARIO_NAMES[:-1] == (
        'measured_in_vivo', 'ivivc_prediction', 'adaptive_in_vivo_only',
        'adaptive_ivivc_prediction', 'adaptive_gap_filling', 'adaptive_timing_comparison',
        'mixed_timing_requirements', 'design_size_evolution',
    )
    assert poc.SCENARIO_NAMES[-1] == SCENARIO
    settings = _settings()
    assert settings.time_scaling_mode == 'prospective'
    assert settings.time_scaling_parameter_draws == 0
    assert settings.resolved_time_scaling_model == settings.analysis_model
    changed = replace(settings, analysis_model='exponential')
    assert changed.resolved_time_scaling_model == 'exponential'
    explicit = replace(changed, time_scaling_model='exponential_normalized')
    assert explicit.analysis_model == 'exponential'
    assert explicit.resolved_time_scaling_model == 'exponential_normalized'
    args = poc._parse_args([
        '--scenario', SCENARIO, '--time-scaling-mode', 'adaptive',
        '--time-scaling-ratios-csv', 'ratios.csv',
        '--time-scaling-model', 'exponential_normalized',
        '--time-scaling-parameter-draws', '3', '--time-scaling-allow-extrapolation',
    ])
    assert args.time_scaling_mode == 'adaptive'
    assert str(args.time_scaling_ratios_csv) == 'ratios.csv'
    assert args.time_scaling_model == 'exponential_normalized'
    assert args.time_scaling_parameter_draws == 3
    assert args.time_scaling_allow_extrapolation


@pytest.mark.parametrize('mode', ['prospective', 'adaptive'])
def test_empirical_example_uses_existing_planning_and_mc_pipeline(mode_runs, mode):
    result = mode_runs[mode].scenarios[SCENARIO]
    definition = result.definition
    prior = definition.prior_reference
    assert prior is not None
    assert definition.preprocessed_data is None
    assert prior.n_draws == 5  # Ratio support, not --reference-draws, sets the count.
    np.testing.assert_allclose(prior.metadata['time_scale_ratios'], [1.6, 1.8, 2., 2.2, 2.4])
    np.testing.assert_allclose(prior.draw_weights, np.full(5, .2))
    assert definition.reference.supported_time_range == (0., 52.)
    assert definition.metadata['time_scaling_weight_summary']['input_mode'] == f'{mode}-time-scaling'
    assert not result.performance_table.empty
    assert np.isfinite(result.performance_table['penalized_mean_normalized_irmse']).all()
    source = definition.observed_data.query("role == 'target in vitro reference data'")
    assert not source.empty
    assert 'IVIVC calibration data' not in set(definition.observed_data['role'])
    assert source['time'].min() == 0. and source['time'].max() == 40.
    assert definition.constraints.fixed_times[0] == 0.
    if mode == 'prospective':
        assert definition.existing_observations is None
        np.testing.assert_allclose(definition.reference.draw_weights, prior.draw_weights)
        assert definition.metadata['time_scaling_observation_agreement'] is None
    else:
        assert definition.reference.metadata['adaptive_reweighting'] is True
        assert definition.metadata['time_scaling_observation_agreement']['observation_count'] == 9
        existing = definition.existing_observations
        assert existing.unique_times.tolist() == [0., 10., 20.]
        assert np.all(definition.constraints.candidate_times > existing.latest_time)
        for comparison in result.comparisons_by_n.values():
            for simulation in comparison.results.values():
                times, means = existing.mean_response_by_time()
                np.testing.assert_allclose(
                    simulation.observed_means[:, np.searchsorted(simulation.design_times, times)],
                    np.broadcast_to(means, (simulation.n_simulations, len(times))),
                )


def test_prospective_and_adaptive_examples_share_the_original_prior(mode_runs):
    prior = mode_runs['prospective'].scenarios[SCENARIO].definition.prior_reference
    adaptive = mode_runs['adaptive'].scenarios[SCENARIO].definition
    np.testing.assert_array_equal(prior.value_draws, adaptive.prior_reference.value_draws)
    np.testing.assert_array_equal(prior.draw_weights, adaptive.prior_reference.draw_weights)
    np.testing.assert_array_equal(prior.value_draws, adaptive.reference.value_draws)
    assert not np.allclose(prior.draw_weights, adaptive.reference.draw_weights)


def test_joint_parameter_draw_count_keeps_the_empirical_ratio_marginal():
    definition = poc.build_empirical_time_scaling_scenario(_settings(time_scaling_parameter_draws=3))
    prior = definition.prior_reference
    assert prior.n_draws == 15
    marginal = np.bincount(prior.metadata['draw_ratio_indices'], weights=prior.draw_weights, minlength=5)
    np.testing.assert_allclose(marginal, np.full(5, .2))


def test_csv_weights_and_source_groups_remain_traceable(tmp_path):
    path = tmp_path / 'scales.csv'
    path.write_text('ratio,weight,source_id,group\n2,3,A,polymer-X\n3,1,B,polymer-Y\n2,0,excluded,polymer-Z\n')
    distribution, table = load_time_scale_distribution(path)
    np.testing.assert_allclose(distribution.ratios, [2., 3., 2.])
    np.testing.assert_allclose(distribution.weights, [.75, .25, 0.])
    assert tuple(distribution.source_ids) == ('A', 'B', 'excluded')
    assert table['group'].tolist() == ['polymer-X', 'polymer-Y', 'polymer-Z']
    definition = poc.build_empirical_time_scaling_scenario(_settings(time_scaling_ratios_csv=path))
    assert definition.prior_reference.n_draws == 2
    outputs = time_scaling_tables(definition)
    weights = outputs['time_scale_weights.csv']
    np.testing.assert_allclose(weights['prior_weight'], [.75, .25, 0.])
    np.testing.assert_allclose(weights['current_weight'], [.75, .25, 0.])
    assert weights['source_id'].tolist() == ['A', 'B', 'excluded']
    assert weights['group'].tolist() == table['group'].tolist()


def test_ratio_only_csv_uses_equal_record_weights(tmp_path):
    path = tmp_path / 'ratios.csv'
    path.write_text('ratio\n2\n3\n')
    distribution, table = load_time_scale_distribution(path)
    np.testing.assert_allclose(distribution.weights, [.5, .5])
    assert table['weight'].tolist() == [1., 1.]
    assert table['source_id'].str.len().gt(0).all()


@pytest.mark.parametrize('contents', [
    'weight\n1\n', 'ratio\n0\n', 'ratio\nNaN\n', 'ratio\n-2\n',
    'ratio,weight\n2,\n', 'ratio,weight\n2,-1\n', 'ratio,weight\n2,0\n',
    'ratio,source_id\n2,\n', 'ratio,ratio\n2,3\n', 'ratio\n',
    'ratio,weight\n2,1,unexpected\n', 'ratio,weight\n2\n',
])
def test_bad_csv_records_are_not_dropped_or_replaced_with_illustrative_data(tmp_path, contents):
    path = tmp_path / 'bad.csv'
    path.write_text(contents)
    with pytest.raises((ValueError, TypeError)):
        load_time_scale_distribution(path)


def test_missing_csv_and_unsupported_horizon_are_explicit(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        load_time_scale_distribution(tmp_path / 'missing.csv')
    path = tmp_path / 'fast.csv'
    path.write_text('ratio\n0.5\n2\n')
    settings = _settings(time_scaling_ratios_csv=path)
    with pytest.raises(ValueError, match='extrapolation'):
        poc.build_empirical_time_scaling_scenario(settings)
    extrapolated = poc.build_empirical_time_scaling_scenario(replace(
        settings, time_scaling_allow_extrapolation=True,
    ))
    assert np.any(extrapolated.reference.metadata['extrapolated_points_by_ratio'])


@pytest.mark.parametrize('changes', [
    {'time_scaling_mode': 'unknown'}, {'time_scaling_parameter_draws': -1},
    {'time_scaling_parameter_draws': 1.5}, {'time_scaling_parameter_draws': True},
    {'time_scaling_model': 'linear_zero_intercept'},
])
def test_invalid_time_scaling_settings_are_rejected(changes):
    with pytest.raises((ValueError, KeyError)):
        _settings(**changes)


@pytest.mark.parametrize('mode', ['prospective-time-scaling', 'adaptive-time-scaling'])
def test_empirical_input_modes_work_with_mixed_requirements(mixed_runs, mode):
    result = mixed_runs[mode].scenarios['mixed_timing_requirements']
    assert result.definition.planning.input_mode == mode
    assert result.definition.preprocessed_data is None
    assert set(result.schedule_status_table['status']) == {'feasible'}
    assert set(result.requirement_table['status']) == {'satisfied'}
    assert len(result.anchor_table) == 2
    assert set(result.anchor_table['status']) == {'selected'}
    for comparison in result.comparisons_by_n.values():
        first, second = tuple(comparison.results.values())
        np.testing.assert_array_equal(first.trajectory_draw_indices, second.trajectory_draw_indices)


def test_output_writer_retains_input_weights_and_removes_stale_empirical_files(mode_runs, tmp_path):
    adaptive = mode_runs['adaptive']
    poc.write_proof_of_concept_outputs(adaptive, tmp_path)
    directory = tmp_path / SCENARIO
    expected = {'time_scale_input.csv', 'time_scale_weights.csv', 'time_scale_ecdf.csv', 'observation_agreement.csv'}
    assert expected.issubset({path.name for path in directory.iterdir()})
    metadata = json.loads((directory / 'metadata.json').read_text(),
                          parse_constant=lambda value: pytest.fail(f'Non-JSON value {value}'))
    assert metadata['scenario_metadata']['time_scaling_weight_summary']['input_mode'] == 'adaptive-time-scaling'
    csv_weights = pd.read_csv(directory / 'time_scale_weights.csv')
    np.testing.assert_allclose(csv_weights['current_weight'], adaptive.scenarios[SCENARIO].definition.metadata[
        'time_scaling_weight_summary']['current_weights'])
    poc.write_proof_of_concept_outputs(mode_runs['prospective'], tmp_path)
    assert not (directory / 'observation_agreement.csv').exists()
    result = mode_runs['prospective'].scenarios[SCENARIO]
    plain = replace(result, definition=replace(
        result.definition, reference=replace(result.definition.reference, metadata={'reference_source': 'external'}),
        prior_reference=None, metadata={},
    ))
    poc.write_proof_of_concept_outputs(replace(mode_runs['prospective'], scenarios={SCENARIO: plain}), tmp_path)
    assert not expected.intersection({path.name for path in directory.iterdir()})


class _Figure:
    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        pass

    def add_vrect(self, **kwargs):
        pass

    def add_vline(self, **kwargs):
        pass


@pytest.mark.parametrize('planning', [False, True])
def test_html_contains_ecdf_prior_current_and_weight_diagnostics(mode_runs, mixed_runs, planning):
    result = (mixed_runs['adaptive-time-scaling'].scenarios['mixed_timing_requirements'] if planning
              else mode_runs['adaptive'].scenarios[SCENARIO])
    package, graph, io = (ModuleType(name) for name in ('plotly', 'plotly.graph_objects', 'plotly.io'))
    package.__path__ = []
    graph.Figure = _Figure
    graph.Scatter = lambda **kwargs: kwargs
    io.to_html = lambda figure, **kwargs: '<div>Figure placeholder</div>'
    with patch.dict('sys.modules', {'plotly': package, 'plotly.graph_objects': graph, 'plotly.io': io}):
        html = poc._plotly_summary_html(result)
    assert 'Empirical time-scale reference' in html
    assert 'ECDF' in html and 'prior' in html.lower() and 'current' in html.lower()
    assert 'ESS' in html
    assert 'illustrative' in html.lower()


def test_csv_provenance_is_escaped_in_html(tmp_path):
    path = tmp_path / 'labels.csv'
    path.write_text('ratio,source_id,group\n2,<script>alert(1)</script>,<b>polymer</b>\n')
    definition = poc.build_empirical_time_scaling_scenario(_settings(time_scaling_ratios_csv=path))
    graph = ModuleType('graph')
    graph.Figure = _Figure
    graph.Scatter = lambda **kwargs: kwargs
    html = time_scaling_summary_html(definition, graph, lambda *args, **kwargs: '<figure/>', poc._reference_table)
    assert '<script>' not in html and '<b>polymer</b>' not in html
    assert '&lt;script&gt;' in html and '&lt;b&gt;polymer&lt;/b&gt;' in html


def test_report_plots_use_the_prior_and_updated_reference_data(mode_runs):
    definition = mode_runs['adaptive'].scenarios[SCENARIO].definition
    graph = ModuleType('graph')
    graph.Figure = _Figure
    graph.Scatter = lambda **kwargs: kwargs
    figures = []
    time_scaling_summary_html(
        definition, graph,
        lambda figure, **kwargs: figures.append(figure) or '<figure/>',
        poc._reference_table,
    )
    assert len(figures) == 2
    for trace in figures[0].traces:
        probabilities = np.asarray(trace['y'])
        assert probabilities[0] == 0. and probabilities[-1] == 1.
        assert np.all(np.diff(probabilities) >= 0.)
    for label, reference in (('Prior', definition.prior_reference), ('Current', definition.reference)):
        trace = next(item for item in figures[1].traces if item.get('name') == f'{label} central reference')
        np.testing.assert_allclose(trace['y'], poc._reference_table(reference)['central_response'])
