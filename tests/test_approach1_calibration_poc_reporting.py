"""Approach 1 POC reports distinguish empirical timing errors from fit bands."""
from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ivivc_core import AnalysisError, ObservationUncertainty, PredictedTrajectory, prepare_time_course
from study_design_core import (
    CalibrationEvaluation, CalibrationSimulationResult, IVIVCCalibrationAssessment,
    ReferenceTrajectoryEnsemble, summarize_calibration_simulation,
)
from examples.study_design_calibration_poc import run_calibration_poc
from examples.study_design_calibration_reporting import (
    calibration_poc_tables, write_calibration_poc_outputs,
)
from examples import study_design_calibration_reporting as reporting


LABEL = 'Candidate <one>'


def fixture(*, interval_span=1.):
    """Three experiments: complete, point-only, and failed, with explicit truth."""
    course, _ = prepare_time_course([0., 2.], [1., .7],
        observation_uncertainty=ObservationUncertainty(.03, response_kind='mean', replicate_counts=4))
    trajectory = PredictedTrajectory(
        'approach1', 'time_mapping', np.array([1., 2.]), np.array([.8, .6]),
        np.array([2., 3.]), np.array([.8, .6]), status=['predicted', 'predicted'],
        auxiliary={'interval_status': ['available', 'available'],
                   'interval_valid_samples': [24, 24], 'interval_requested_samples': 24,
                   'extrapolated': [False, True],
                   'extrapolation_status': ['within calibration range', 'extrapolation'],
                   'extrapolation_reason': ['', 'mapped time outside paired in vivo fitting range'],
                   'interval_extrapolated': [False, True],
                   'interval_extrapolation_status': ['within calibration range', 'interval extends beyond calibration range'],
                   'extrapolation_warnings': ['one point outside calibration']})
    first = CalibrationEvaluation('in_vivo_time', [1., 2.], [2., 3.],
                                  np.array([2., 3.])-interval_span,
                                  np.array([2., 3.])+interval_span, trajectory)
    second_trajectory = replace(
        trajectory, in_vivo_time=np.array([4., np.nan]), status=['predicted', 'unavailable'],
        auxiliary=dict(trajectory.auxiliary, interval_status=['unavailable', 'unavailable'],
                       interval_valid_samples=[0, 0]))
    second = CalibrationEvaluation('in_vivo_time', [1., 2.], [4., np.nan],
                                   None, None, second_trajectory)
    fitted = SimpleNamespace(model_results={'approach1:linear': object()})
    metadata = {'selection': 'approach1:linear',
                'interval_interpretation': 'mapping_regression_parameter_uncertainty',
                'calibration_uncertainty_used_in_fit': False}
    assessments = [
        IVIVCCalibrationAssessment(analysis=fitted, relationship=first, metadata=metadata),
        IVIVCCalibrationAssessment(analysis=fitted, relationship=second, metadata=metadata),
        IVIVCCalibrationAssessment(errors={'fit': AnalysisError('deliberate failure', 'RuntimeError')})]
    result = CalibrationSimulationResult(
        assessments, [course]*3, [course, course, None],
        truth={'relationship': np.tile([1., 2.], (3, 1))},
        metadata={'approach_id': 'approach1', 'propagation_samples': 24, 'confidence_level': .8,
                  'interval_interpretation': 'mapping_regression_parameter_uncertainty',
                  'empirical_interpretation': 'across_simulated_studies_conditional_on_fixed_historical_data',
                  'evaluation_spec': {'relationship': {'quantity': 'in_vivo_time', 'coordinate': [1., 2.]}}})
    return {
        'name': 'study', 'title': 'Approach 1 <example>',
        'description': 'Ordinary time mapping & repeated simulated studies',
        'assumptions': {'historical_data_fixed': True, 'reference_truth': 'illustrative',
                        'fixed_data_uncertainty': 'historical SD/SEM is not propagated by the ordinary mapping fit'},
        'designs': {LABEL: {'in_vitro_times': [], 'in_vitro_replicates_per_time': 0,
                           'in_vitro_added_measurements': 0, 'in_vivo_times': [0., 2.],
                           'in_vivo_replicates_per_time': 3, 'in_vivo_added_measurements': 6}},
        'existing_data': {'in_vitro': course}, 'target': course,
        'references': {'in_vivo': ReferenceTrajectoryEnsemble([0., 2.], [[1., .7]], [True])},
        'calibration_results': {LABEL: result},
        'calibration_summaries': {LABEL: summarize_calibration_simulation(result)},
        'recovery_results': {}, 'recovery_summaries': {}, 'simulation_plans': {},
    }


def controlled_run(scenario):
    return {'settings': {'approach_id': 'approach1', 'n_simulations': 3,
                         'n_interval_samples': 24, 'time_units': 'arbitrary time units'},
            'scenarios': {'study': scenario}}


class Approach1CalibrationPOCReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poc_run = run_calibration_poc(n_simulations=3, n_interval_samples=24,
                                         random_seed=12345, approach_id='approach1')

    def test_design_and_pointwise_exports_match_backend_timing_summaries(self):
        for scenario in self.poc_run['scenarios'].values():
            tables = calibration_poc_tables(scenario)
            designs = tables['design_summary.csv'].set_index('design')
            points = tables['pointwise_performance.csv']
            for label, backend in scenario['calibration_summaries'].items():
                for output in ('relationship', 'prediction'):
                    summary = backend[output]
                    with self.subTest(scenario=scenario['name'], design=label, output=output):
                        self.assertEqual(designs.loc[label, output+'_quantity'], 'in_vivo_time')
                        self.assertAlmostEqual(designs.loc[label, output+'_bias'], summary['bias'])
                        self.assertAlmostEqual(designs.loc[label, output+'_rmse'], summary['rmse'])
                        rows = points[(points.design == label) & (points.output == output)]
                        np.testing.assert_array_equal(rows.coordinate, summary['coordinate'])
                        np.testing.assert_array_equal(rows.point_counts, summary['point_counts'])
                        np.testing.assert_array_equal(rows.error_counts, summary['error_counts'])
                        np.testing.assert_array_equal(rows.pointwise_bias, summary['pointwise_bias'])
                        for index, suffix in enumerate(('lower', 'median', 'upper')):
                            np.testing.assert_array_equal(rows['empirical_quantile_'+suffix+'_probability'],
                                                          summary['empirical_quantiles'][index])
                            np.testing.assert_array_equal(rows['pointwise_estimate_'+suffix],
                                                          summary['pointwise_estimate_quantiles'][index])
                            np.testing.assert_array_equal(rows['pointwise_error_'+suffix],
                                                          summary['pointwise_error_quantiles'][index])

    def test_missing_bounds_and_failed_fits_do_not_drop_timing_evaluations(self):
        scenario = fixture()
        tables = calibration_poc_tables(scenario)
        row = tables['design_summary.csv'].iloc[0]
        self.assertAlmostEqual(row.relationship_bias, 5/3)
        self.assertAlmostEqual(row.relationship_rmse, np.sqrt(11/3))
        self.assertAlmostEqual(row.calibration_fit_failure_rate, 1/3)
        evaluations = tables['evaluations.csv']
        self.assertEqual(len(evaluations), 6)
        self.assertEqual(set(evaluations.quantity), {'in_vivo_time'})
        point_only = evaluations[(evaluations.experiment == 1) & (evaluations.coordinate == 1.)].iloc[0]
        self.assertTrue(point_only.point_available)
        self.assertFalse(point_only.interval_available)
        self.assertEqual(point_only.estimate, 4.)
        self.assertEqual(point_only.error, 3.)
        self.assertTrue(np.isnan(point_only.lower))
        self.assertTrue(evaluations[evaluations.experiment == 2].estimate.isna().all())
        self.assertTrue(evaluations[evaluations.experiment == 2].error.isna().all())
        self.assertEqual(set(evaluations.interval_interpretation), {'mapping_regression_parameter_uncertainty'})
        self.assertTrue(evaluations.empirical_interpretation.str.contains('simulated_studies').all())
        self.assertIn('deliberate failure', tables['experiment_outcomes.csv'].iloc[2].calculation_errors)
        self.assertTrue(evaluations.iloc[1].extrapolated)
        self.assertIn('mapped time', evaluations.iloc[1].extrapolation_reason)
        pointwise = tables['pointwise_performance.csv']
        np.testing.assert_array_equal(pointwise.point_counts, [2, 1])
        np.testing.assert_array_equal(pointwise.error_counts, [2, 1])
        np.testing.assert_array_equal(pointwise.interval_counts, [1, 1])

    def test_empirical_error_ranges_are_separate_from_regression_interval_widths(self):
        narrow = calibration_poc_tables(fixture(interval_span=.1))
        wide = calibration_poc_tables(fixture(interval_span=100.))
        first, second = narrow['design_summary.csv'].iloc[0], wide['design_summary.csv'].iloc[0]
        self.assertLess(first.relationship_mean_interval_width, second.relationship_mean_interval_width)
        self.assertEqual(first.relationship_interval_interpretation,
                         'mapping_regression_parameter_uncertainty')
        self.assertIn('simulated_studies', first.relationship_empirical_interpretation)
        for name in ('pointwise_estimate_lower', 'pointwise_estimate_median', 'pointwise_estimate_upper',
                     'pointwise_error_lower', 'pointwise_error_median', 'pointwise_error_upper', 'pointwise_bias'):
            np.testing.assert_array_equal(narrow['pointwise_performance.csv'][name],
                                          wide['pointwise_performance.csv'][name])
        points = narrow['pointwise_performance.csv']
        self.assertEqual(set(points.interval_interpretation), {'mapping_regression_parameter_uncertainty'})
        self.assertTrue(points.empirical_interpretation.str.contains('simulated_studies').all())

    def test_unknown_truth_stays_missing_without_removing_estimate_quantiles(self):
        scenario = fixture()
        result = scenario['calibration_results'][LABEL]
        result.truth = {}
        scenario['calibration_summaries'][LABEL] = summarize_calibration_simulation(result)
        tables = calibration_poc_tables(scenario)
        points = tables['pointwise_performance.csv']
        np.testing.assert_array_equal(points.error_counts, [0, 0])
        self.assertTrue(points.pointwise_bias.isna().all())
        self.assertTrue(points.pointwise_error_lower.isna().all())
        self.assertTrue(points.pointwise_estimate_lower.notna().all())
        self.assertTrue(tables['evaluations.csv'].truth.isna().all())

    def test_fixed_input_exports_preserve_sem_and_counts_for_unweighted_mapping(self):
        for scenario in self.poc_run['scenarios'].values():
            fixed = calibration_poc_tables(scenario)['fixed_inputs.csv']
            historical = fixed[fixed.role == 'fixed_calibration']
            self.assertEqual(set(historical.uncertainty_kind), {'sem'})
            self.assertTrue((historical.replicate_count == 3).all())
            self.assertTrue((historical.standard_uncertainty > 0).all())
            for result in scenario['calibration_results'].values():
                self.assertEqual(result.metadata['interval_interpretation'],
                                 'mapping_regression_parameter_uncertainty')
                for assessment in result.assessments:
                    if assessment.fit_available:
                        self.assertFalse(assessment.metadata['calibration_uncertainty_used_in_fit'])

    def test_html_and_json_explain_time_mapping_and_empirical_ranges(self):
        scenario = fixture()
        with TemporaryDirectory() as temporary:
            root = write_calibration_poc_outputs(controlled_run(scenario), temporary)
            html = (root/'study/summary.html').read_text()
            self.assertIn('Approach 1 &lt;example&gt;', html)
            self.assertNotIn('<h1>Approach 1 <example>', html)
            lower = html.lower()
            for phrase in ('in vivo time', 'regression', 'empirical', 'extrapolation', 'unavailable'):
                self.assertIn(phrase, lower)
            self.assertIn('not optimized', lower)
            if not find_spec('plotly'):
                self.assertIn('require Plotly', html)
            metadata = json.loads((root/'study/metadata.json').read_text(),
                                  parse_constant=lambda item: self.fail('Nonfinite JSON: '+item))
            summary = metadata['calibration_summaries'][LABEL]['relationship']
            self.assertEqual(summary['interval_interpretation'], 'mapping_regression_parameter_uncertainty')
            self.assertAlmostEqual(summary['bias'], 5/3)
            self.assertEqual(metadata['illustrative_mapping_experiment'], 0)
            self.assertEqual(len(metadata['calculation_failures'][LABEL]), 1)

    def test_actual_approach1_run_exports_both_workflows_and_all_requested_trials(self):
        with TemporaryDirectory() as temporary:
            root = write_calibration_poc_outputs(self.poc_run, temporary, include_html=False)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(set(manifest['scenarios']), {'study', 'refinement'})
            self.assertEqual(manifest['settings']['approach_id'], 'approach1')
            for name, scenario in self.poc_run['scenarios'].items():
                summary = pd.read_csv(root/name/'design_summary.csv')
                outcomes = pd.read_csv(root/name/'experiment_outcomes.csv')
                self.assertEqual(len(summary), len(scenario['designs']))
                self.assertEqual(len(outcomes), 3*len(scenario['designs']))
                self.assertEqual(set(summary.relationship_quantity), {'in_vivo_time'})
                self.assertIn('relationship_bias', summary)
                for filename in manifest['files'][name]:
                    self.assertTrue((root/filename).exists(), filename)
                if name == 'study':
                    self.assertIn('curve_mean_irmse', summary)
                else:
                    self.assertNotIn('curve_mean_irmse', summary)

    def test_no_html_write_preserves_empirical_csv_and_removes_stale_html(self):
        run = controlled_run(fixture())
        with TemporaryDirectory() as temporary:
            root = write_calibration_poc_outputs(run, temporary, include_html=True)
            write_calibration_poc_outputs(run, temporary, include_html=False)
            self.assertFalse((root/'index.html').exists())
            self.assertFalse((root/'study/summary.html').exists())
            points = pd.read_csv(root/'study/pointwise_performance.csv')
            np.testing.assert_allclose(points.pointwise_bias, [2., 1.])
            np.testing.assert_array_equal(points.point_counts, [2, 1])
            evaluation = pd.read_csv(root/'study/evaluations.csv')
            self.assertTrue(pd.isna(evaluation[evaluation.experiment == 2].estimate).all())

    def test_mapping_plot_uses_recorded_experiment_zero_without_replacing_failed_fit(self):
        traces = []

        class Figure:
            def add_trace(self, trace, row=None, col=None):
                traces.append((trace, row, col))

            def update_layout(self, *args, **kwargs):
                return self

            update_xaxes = update_layout
            update_yaxes = update_layout

        plotly = ModuleType('plotly')
        graph = ModuleType('plotly.graph_objects')
        graph.Scatter = lambda **kwargs: SimpleNamespace(**kwargs)
        subplots = ModuleType('plotly.subplots')
        subplots.make_subplots = lambda **kwargs: Figure()
        io = ModuleType('plotly.io')
        io.to_html = lambda figure, **kwargs: '<div>recorded mapping plot</div>'
        plotly.graph_objects, plotly.subplots, plotly.io = graph, subplots, io
        modules = {'plotly': plotly, 'plotly.graph_objects': graph,
                   'plotly.subplots': subplots, 'plotly.io': io}

        def mapping_values():
            return [np.asarray(trace.y, dtype=float) for trace, _, column in traces
                    if column == 1 and hasattr(trace, 'y')]

        with patch.dict(sys.modules, modules):
            scenario = fixture()
            _, available = reporting._time_mapping_plot_html(calibration_poc_tables(scenario), scenario['title'])
            self.assertTrue(available)
            self.assertTrue(any(values.shape == (2,) and np.array_equal(values, [2., 3.])
                                for values in mapping_values()))
            timing = [trace for trace, _, column in traces if column == 2]
            self.assertEqual(len(timing), 1)
            expected = scenario['calibration_summaries'][LABEL]['relationship']['pointwise_error_quantiles']
            np.testing.assert_array_equal(timing[0].y, expected[1])
            np.testing.assert_array_equal(timing[0].error_y['array'], expected[2]-expected[1])
            np.testing.assert_array_equal(timing[0].error_y['arrayminus'], expected[1]-expected[0])
            traces.clear()
            result = scenario['calibration_results'][LABEL]
            complete = result.assessments[0]
            complete.relationship.estimate[:] = [777., 888.]
            result.assessments = (result.assessments[2], complete, result.assessments[1])
            scenario['calibration_summaries'][LABEL] = summarize_calibration_simulation(result)
            reporting._time_mapping_plot_html(calibration_poc_tables(scenario), scenario['title'])
            self.assertFalse(any(values.shape == (2,) and np.array_equal(values, [777., 888.])
                                 for values in mapping_values()))


if __name__ == '__main__':
    unittest.main()
