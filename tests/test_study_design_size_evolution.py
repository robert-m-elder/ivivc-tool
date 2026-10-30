from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples import study_design_proof_of_concept as poc


def _settings(**overrides):
    values = dict(n_reference_draws=8, n_simulations=4, random_seed=1234,
                  analysis_model='mass_loss_one_phase', write_html=False)
    values.update(overrides)
    return poc.ProofOfConceptSettings(**values)


@pytest.fixture(scope='module')
def evolution_run():
    return poc.run_proof_of_concept(('design_size_evolution',), settings=_settings())


def test_evolution_reuses_reference_and_requires_time_zero():
    settings = _settings()
    original = poc.build_measured_in_vivo_scenario(settings)
    evolution = poc.build_design_size_evolution_scenario(settings)
    assert original.constraints.fixed_times.tolist() == [0.0]
    assert original.constraints.min_number_of_times == 3
    assert original.constraints.max_number_of_times == 7
    assert evolution.constraints.fixed_times.tolist() == [0.0]
    assert evolution.constraints.min_number_of_times == 1
    assert evolution.constraints.max_number_of_times == 5
    assert 0.0 in evolution.constraints.candidate_times
    assert evolution.existing_observations is None
    assert evolution.metadata['requested_design_sizes'] == [1, 2, 3, 4, 5]
    assert evolution.metadata['required_sampling_times'] == [0.0]
    assert evolution.metadata['underdetermined_fit_statistics_retained']
    pd.testing.assert_frame_equal(original.observed_data, evolution.observed_data)
    np.testing.assert_array_equal(original.reference.value_draws, evolution.reference.value_draws)
    np.testing.assert_array_equal(original.reference.central_curve, evolution.reference.central_curve)


def test_evolution_is_separate_and_cli_accepts_it():
    assert poc.SCENARIO_NAMES[0] == 'measured_in_vivo'
    assert poc.SCENARIO_NAMES[-1] == 'design_size_evolution'
    assert poc.SCENARIO_BUILDERS['measured_in_vivo'] is poc.build_measured_in_vivo_scenario
    assert poc._parse_args(['--scenario', 'design_size_evolution']).scenario == 'design_size_evolution'
    short = poc.build_design_size_evolution_scenario(_settings(max_design_sizes=2))
    assert short.constraints.max_number_of_times == 2


def test_all_sizes_and_actual_optimizer_choices_are_retained(evolution_run):
    result = evolution_run.scenarios['design_size_evolution']
    assert list(result.comparisons_by_n) == [1, 2, 3, 4, 5]
    assert result.reference_model.n_parameters == 4
    for comparison in result.comparisons_by_n.values():
        for simulation in comparison.results.values():
            assert 0.0 in simulation.design_times
    for simulation in result.comparisons_by_n[1].results.values():
        np.testing.assert_array_equal(simulation.design_times, [0.0])
    for design, steps in (('optimized_nested', result.optimization.nested_steps),
                          ('optimized_refined', result.optimization.refined_steps)):
        for step in steps:
            simulation = result.comparisons_by_n[step.number_of_times].results[design]
            np.testing.assert_array_equal(simulation.design_times, step.times)
            assert simulation.n_simulations == 4
    low_n = result.performance_table.query('number_of_times < 4')
    assert len(low_n) == 9
    assert np.all(np.isfinite(low_n['penalized_mean_irmse']))
    optimized = low_n[low_n['design'].str.startswith('optimized')]
    assert set(optimized['information_selection_rule']) == {'rank_pseudodeterminant_fallback'}
    assert np.all(np.isinf(optimized['information_criterion']))
    tables = poc._t50_recovery_table(result)
    assert len(tables) == 15
    assert set(tables['number_of_times']) == {1, 2, 3, 4, 5}
    assert tables.loc[tables.number_of_times < 4, 'fewer_times_than_parameters'].all()
    assert not tables.loc[tables.number_of_times >= 4, 'fewer_times_than_parameters'].any()


def _synthetic_recovery(fitted_t50, *, success=None):
    time = np.arange(1001, dtype=float)
    fitted_t50 = np.asarray(fitted_t50, dtype=float)
    count = len(fitted_t50)
    fitted_curves = np.maximum(1.0 - time[None, :] / (2.0 * fitted_t50[:, None]), 0.0)
    simulation = SimpleNamespace(
        evaluation_times=time,
        true_curves=np.tile(np.maximum(1.0 - time / 10.0, 0.0), (count, 1)),
        fitted_curves=fitted_curves,
        fit_success=np.ones(count, dtype=bool) if success is None else np.asarray(success),
        n_simulations=count,
    )
    result = SimpleNamespace(
        reference_model=SimpleNamespace(n_parameters=4),
        comparisons_by_n={1: SimpleNamespace(results={'optimized_nested': simulation})},
    )
    return result, simulation


def test_underdetermined_endpoint_outliers_are_not_clipped():
    result, _ = _synthetic_recovery([5.0] * 99 + [900.0])
    row = poc._t50_recovery_table(result).iloc[0]
    assert row['number_of_times'] == 1
    assert row['n_paired_t50'] == 100
    assert row['mean_t50_error'] == pytest.approx(8.95)
    # A mean outside the central range is valid; the plot must not recenter it.
    assert row['lower_95_t50_error'] == pytest.approx(0.0)
    assert row['upper_95_t50_error'] == pytest.approx(0.0)


def test_unavailable_endpoints_are_counted_without_fabricating_values():
    result, simulation = _synthetic_recovery([3.0, 4.0, 5.0], success=[True, False, True])
    simulation.fitted_curves[2] = 1.0
    row = poc._t50_recovery_table(result).iloc[0]
    assert row['n_successful_fits'] == 2
    assert row['n_true_t50_reachable'] == 3
    assert row['n_paired_t50'] == 1
    assert row['mean_t50_error'] == pytest.approx(-2.0)
    simulation.fit_success[:] = False
    row = poc._t50_recovery_table(result).iloc[0]
    assert row['n_paired_t50'] == 0
    assert np.isnan(row['mean_t50_error'])
    assert np.isnan(row['lower_95_t50_error'])
    assert np.isnan(row['upper_95_t50_error'])


def test_endpoint_csv_and_metadata_include_underdetermined_sizes(evolution_run, tmp_path):
    poc.write_proof_of_concept_outputs(evolution_run, tmp_path)
    directory = tmp_path / 'design_size_evolution'
    endpoints = pd.read_csv(directory / 't50_recovery.csv')
    pd.testing.assert_frame_equal(
        endpoints, poc._t50_recovery_table(evolution_run.scenarios['design_size_evolution']),
        check_dtype=False,
    )
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['constraints']['fixed_times'] == [0.0]
    assert metadata['scenario_metadata']['required_sampling_times'] == [0.0]
    assert metadata['constraints']['min_number_of_times'] == 1
    assert metadata['constraints']['max_number_of_times'] == 5
    assert metadata['scenario_metadata']['underdetermined_fit_statistics_retained'] is True


def test_four_panel_html_uses_returned_data_and_unclipped_axes(evolution_run, monkeypatch):
    pytest.importorskip('plotly')
    import plotly.io

    captured = []
    monkeypatch.setattr(plotly.io, 'to_html', lambda figure, **kwargs: captured.append(figure) or '<figure/>')
    result = evolution_run.scenarios['design_size_evolution']
    html = poc._plotly_summary_html(result)
    figure = captured[0]
    titles = [item.text for item in figure.layout.annotations]
    assert titles[0] == 'a. Reference and schedules (N=4)'
    assert titles[1].startswith('b.')
    assert titles[2].startswith('c.')
    assert titles[3].startswith('d.')
    assert 'N&lt;4' in html
    assert 'without a rank filter or outlier clipping' in html
    assert 'not a confidence interval for the mean' in html
    assert 'Time zero is a required baseline' in html
    for axis in ('yaxis', 'yaxis2', 'yaxis3', 'yaxis4'):
        assert getattr(figure.layout, axis).range is None
    # N=1 remains visible even when it has no returned T50.
    assert figure.layout.xaxis4.range[0] < 1
    assert figure.layout.xaxis4.range[1] > 5
    for design, group in result.performance_table.groupby('design', sort=False):
        trace = next(t for t in figure.data if t.xaxis == 'x3' and t.name == poc._design_label(design))
        np.testing.assert_array_equal(trace.y, group.sort_values('number_of_times')['penalized_mean_irmse'])
    recovery = poc._t50_recovery_table(result)
    for design, group in recovery.groupby('design', sort=False):
        trace = next(t for t in figure.data if t.xaxis == 'x4' and t.name == poc._design_label(design)
                     and t.mode == 'lines+markers')
        np.testing.assert_allclose(trace.y, group.sort_values('number_of_times')['mean_t50_error'], equal_nan=True)


def test_short_run_uses_available_size_in_panel_a(monkeypatch):
    pytest.importorskip('plotly')
    import plotly.io

    captured = []
    monkeypatch.setattr(plotly.io, 'to_html', lambda figure, **kwargs: captured.append(figure) or '<figure/>')
    run = poc.run_proof_of_concept(('design_size_evolution',), settings=_settings(max_design_sizes=1))
    poc._plotly_summary_html(run.scenarios['design_size_evolution'])
    assert captured[0].layout.annotations[0].text == 'a. Reference and schedules (N=1)'


def test_index_lists_new_example_first_without_reordering_runs(monkeypatch, tmp_path):
    settings = _settings(max_design_sizes=1, write_html=True)
    run = poc.run_proof_of_concept(('measured_in_vivo', 'design_size_evolution'), settings=settings)
    assert list(run.scenarios) == ['measured_in_vivo', 'design_size_evolution']
    monkeypatch.setattr(poc, '_plotly_summary_html', lambda result: '<p>summary</p>')
    poc.write_proof_of_concept_outputs(run, tmp_path)
    html = (tmp_path / 'index.html').read_text()
    assert html.index('design_size_evolution/summary.html') < html.index('measured_in_vivo/summary.html')
    assert not (tmp_path / 'measured_in_vivo' / 't50_recovery.csv').exists()
    assert (tmp_path / 'design_size_evolution' / 't50_recovery.csv').exists()
