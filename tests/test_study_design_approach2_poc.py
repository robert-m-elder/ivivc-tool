from __future__ import annotations

from dataclasses import replace
import json
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import examples.study_design_proof_of_concept as poc


METHODS = ('value_ratio', 'tau_ratio', 'observed_response_inverse')


def _settings(**changes):
    values = dict(
        n_reference_draws=8, n_simulations=4, max_design_sizes=1,
        random_seed=1234, refine=False, write_html=False,
        analysis_model='exponential', ivivc_approach='approach2',
        ivivc_model='exponential_normalized',
        ivivc_calibration_response_sd=.005,
    )
    values.update(changes)
    return poc.ProofOfConceptSettings(**values)


@pytest.fixture(scope='module')
def approach2_runs():
    return {
        (method, scenario): poc.run_proof_of_concept(
            (scenario,), settings=_settings(ivivc_method=method),
        )
        for method in METHODS
        for scenario in ('ivivc_prediction', 'adaptive_ivivc_prediction')
    }


@pytest.fixture(scope='module')
def mixed_runs():
    return {
        mode: poc.run_proof_of_concept(
            ('mixed_timing_requirements',),
            settings=_settings(
                ivivc_method='tau_ratio', planning_input_mode=mode,
                timing_degraded_fraction=.85, planning_latest_fractions=(.90, .95),
            ),
        )
        for mode in ('prospective-ivivc', 'adaptive-ivivc')
    }


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('scenario', ['ivivc_prediction', 'adaptive_ivivc_prediction'])
def test_each_approach2_method_uses_the_existing_poc_pipeline(approach2_runs, method, scenario):
    run = approach2_runs[method, scenario]
    result = run.scenarios[scenario]
    definition = result.definition
    metadata = definition.metadata
    assert definition.analysis_model == 'exponential'
    assert metadata['ivivc_selection'] == 'approach2:exponential_normalized'
    assert metadata['ivivc_approach'] == 'approach2'
    assert metadata['ivivc_calibration_model'] == 'exponential_normalized'
    assert metadata['prediction_method'] == method
    assert definition.preprocessed_data is None
    assert 'ivivc_fitted_time_mapping_parameters' not in metadata
    assert 'ivivc_preprocessed_pair_count' not in metadata
    fits = metadata['ivivc_calibration_fits']
    assert set(fits) == {'in_vitro', 'in_vivo'}
    for fit in fits.values():
        assert set(fit['parameters']) == {'b'}
        assert np.isfinite(fit['rmse'])
    reference = definition.reference
    assert reference.supported_time_range[0] == 0.
    assert reference.n_draws == run.settings.n_reference_draws
    assert reference.n_valid_draws > 0
    assert reference.metadata['ivivc_prediction_method'] == method
    assert 'ivivc_prediction_auxiliary' in reference.metadata
    assert result.reference_model.n_parameters == 2
    assert not result.performance_table.empty
    assert np.isfinite(result.performance_table['penalized_mean_normalized_irmse']).all()
    assert result.performance_table['failure_rate'].between(0., 1.).all()
    if scenario.startswith('adaptive'):
        assert reference.metadata['adaptive_reweighting'] is True
        assert metadata['effective_sample_size'] > 0
        existing = definition.existing_observations
        times, means = existing.mean_response_by_time()
        assert set(definition.constraints.fixed_times) == set(times)
        for comparison in result.comparisons_by_n.values():
            for simulation in comparison.results.values():
                np.testing.assert_allclose(
                    simulation.observed_means[:, np.searchsorted(simulation.design_times, times)],
                    np.broadcast_to(means, (simulation.n_simulations, len(times))),
                )


@pytest.mark.parametrize('mode', ['prospective-ivivc', 'adaptive-ivivc'])
def test_approach2_supports_mixed_timing_requirements(mixed_runs, mode):
    result = mixed_runs[mode].scenarios['mixed_timing_requirements']
    assert result.definition.planning.input_mode == mode
    assert result.definition.metadata['prediction_method'] == 'tau_ratio'
    assert set(result.schedule_status_table['status']) == {'feasible'}
    assert set(result.requirement_table['status']) == {'satisfied'}
    assert len(result.anchor_table) == 2
    assert set(result.anchor_table['status']) == {'selected'}
    assert set(result.performance_table['design']) == {'central_schedule', 'timing_schedule'}
    for comparison in result.comparisons_by_n.values():
        first, second = tuple(comparison.results.values())
        np.testing.assert_array_equal(first.trajectory_draw_indices, second.trajectory_draw_indices)
        np.testing.assert_allclose(first.true_curves, second.true_curves)


def test_value_ratio_reference_covers_completed_gap_filling_time():
    settings = _settings(ivivc_method='value_ratio')
    definition = poc.build_adaptive_gap_filling_scenario(settings)
    assert definition.metadata['prediction_method'] == 'value_ratio'
    assert definition.reference.supported_time_range[1] >= 40.
    assert definition.existing_observations.latest_time == 40.
    candidates = definition.constraints.candidate_times
    assert np.any((10. < candidates) & (candidates < 40.))
    auxiliary = definition.reference.metadata['ivivc_prediction_auxiliary']
    assert np.any(auxiliary['extrapolated'])


def test_ivivc_defaults_and_model_selection_are_independent():
    original = poc.ProofOfConceptSettings()
    assert original.ivivc_approach == 'approach1'
    assert original.resolved_ivivc_model == 'linear_zero_intercept'
    assert original.resolved_ivivc_method == 'time_mapping'
    default_a2 = replace(original, ivivc_approach='approach2')
    assert default_a2.resolved_ivivc_model == original.analysis_model
    assert default_a2.resolved_ivivc_method == 'value_ratio'
    changed = replace(default_a2, analysis_model='exponential')
    assert changed.resolved_ivivc_model == 'exponential'
    explicit = replace(changed, ivivc_model='exponential_normalized')
    assert explicit.analysis_model == 'exponential'
    assert explicit.resolved_ivivc_model == 'exponential_normalized'
    args = poc._parse_args([])
    assert args.ivivc_approach == 'approach1'
    assert args.ivivc_model is None and args.ivivc_method is None


def test_cli_exposes_the_selected_approach_model_and_method():
    args = poc._parse_args([
        '--scenario', 'ivivc_prediction', '--ivivc-approach', 'approach2',
        '--ivivc-model', 'exponential_normalized', '--ivivc-method', 'tau_ratio',
        '--analysis-model', 'exponential',
    ])
    assert args.ivivc_approach == 'approach2'
    assert args.ivivc_model == 'exponential_normalized'
    assert args.ivivc_method == 'tau_ratio'
    assert args.analysis_model == 'exponential'


@pytest.mark.parametrize('changes', [
    {'ivivc_approach': 'approach3'},
    {'ivivc_approach': 'approach1', 'ivivc_model': None, 'ivivc_method': 'tau_ratio'},
    {'ivivc_approach': 'approach2', 'ivivc_method': 'time_mapping'},
    {'ivivc_approach': 'approach2', 'ivivc_model': 'linear_zero_intercept'},
    {'ivivc_approach': 'approach2', 'ivivc_model': 'missing_model'},
])
def test_incompatible_ivivc_selections_are_rejected(changes):
    with pytest.raises((ValueError, KeyError)):
        _settings(**changes)


def test_unsupported_tau_is_explicit_in_the_poc():
    with pytest.raises(ValueError, match='unavailable|not supported'):
        poc.build_ivivc_prediction_scenario(_settings(
            analysis_model='mass_loss_one_phase_normalized',
            ivivc_model='mass_loss_one_phase_normalized', ivivc_method='tau_ratio',
        ))


def test_inverse_does_not_move_an_unsupported_time_zero_baseline():
    with pytest.raises(ValueError, match='required_times'):
        poc.build_ivivc_prediction_scenario(_settings(
            ivivc_model='exponential', ivivc_method='observed_response_inverse',
        ))


def test_output_metadata_identifies_approach2_and_removes_stale_approach1_pairs(approach2_runs, tmp_path):
    original = poc.run_proof_of_concept(
        ('ivivc_prediction',),
        settings=_settings(ivivc_approach='approach1', ivivc_model=None, ivivc_method=None),
    )
    poc.write_proof_of_concept_outputs(original, tmp_path)
    directory = tmp_path / 'ivivc_prediction'
    assert (directory / 'ivivc_preprocessed_pairs.csv').exists()
    (directory / 'summary.html').write_text('Previous Approach 1 summary')
    (tmp_path / 'index.html').write_text('Previous run index')
    run = approach2_runs['tau_ratio', 'ivivc_prediction']
    poc.write_proof_of_concept_outputs(run, tmp_path)
    assert not (directory / 'ivivc_preprocessed_pairs.csv').exists()
    assert not (directory / 'summary.html').exists()
    assert not (tmp_path / 'index.html').exists()
    metadata = json.loads(
        (directory / 'metadata.json').read_text(),
        parse_constant=lambda value: pytest.fail(f'Non-JSON value: {value}'),
    )
    selected = metadata['scenario_metadata']
    assert selected['ivivc_selection'] == 'approach2:exponential_normalized'
    assert selected['prediction_method'] == 'tau_ratio'
    assert set(selected['ivivc_calibration_fits']) == {'in_vitro', 'in_vivo'}
    assert metadata['analysis_model'] == 'exponential'
    assert metadata['reference']['draws'] == 8
    assert metadata['reference']['supported_time_range'][0] == 0.
    assert 'ivivc_prediction_auxiliary' in metadata['reference']['metadata']
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['settings']['ivivc_approach'] == 'approach2'
    assert manifest['settings']['ivivc_model'] == 'exponential_normalized'
    assert manifest['settings']['ivivc_method'] == 'tau_ratio'
    data = pd.read_csv(directory / 'observed_or_calibration_data.csv')
    assert {'IVIVC calibration data', 'target in vitro prediction data'} <= set(data['role'])


class _Figure:
    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def add_vrect(self, **kwargs):
        pass

    def add_vline(self, **kwargs):
        pass

    def update_layout(self, **kwargs):
        pass


@pytest.mark.parametrize('planning', [False, True])
def test_html_names_the_selected_method_without_claiming_approach1_fitting(
    approach2_runs, mixed_runs, planning,
):
    result = (mixed_runs['prospective-ivivc'].scenarios['mixed_timing_requirements']
              if planning else approach2_runs['tau_ratio', 'ivivc_prediction'].scenarios['ivivc_prediction'])
    package, graph, io = (ModuleType(name) for name in ('plotly', 'plotly.graph_objects', 'plotly.io'))
    package.__path__ = []
    graph.Figure = _Figure
    graph.Scatter = lambda **kwargs: kwargs
    io.to_html = lambda figure, **kwargs: '<div>Figure placeholder</div>'
    with patch.dict('sys.modules', {'plotly': package, 'plotly.graph_objects': graph, 'plotly.io': io}):
        html = poc._plotly_summary_html(result)
    assert 'exponential_normalized' in html and 'tau_ratio' in html
    assert 'calibration profiles are fitted directly' in html
    assert 'IVIVC preprocessing:</strong> response-aligned' not in html
    assert 'usable time pairs' not in html
    assert 'Reference' in html
