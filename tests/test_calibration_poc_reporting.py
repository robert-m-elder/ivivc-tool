"""Calibration POC reports preserve backend scores, failures, and provenance."""
from dataclasses import dataclass
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from ivivc_core import AnalysisError, ObservationUncertainty, PredictedTrajectory, prepare_time_course
from study_design_core import (
    CalibrationEvaluation, CalibrationSimulationResult, IVIVCCalibrationAssessment,
    ReferenceTrajectoryEnsemble, summarize_calibration_simulation,
)
from examples.study_design_calibration_reporting import (
    calibration_poc_tables, write_calibration_poc_outputs,
)


@dataclass
class RecoverySummary:
    mean_irmse: float = .1
    penalized_mean_irmse: float = .4
    p90_irmse: float = .2
    failure_rate: float = .5


def fixture():
    course, _ = prepare_time_course([0., 2.], [1., .7],
        observation_uncertainty=ObservationUncertainty(.03, response_kind='mean', replicate_counts=4))
    trajectory = PredictedTrajectory('approach2', 'value_ratio', np.array([1., 2.]), np.ones(2),
        np.array([1., 2.]), np.array([1.2, np.nan]), status=['predicted', 'unavailable'],
        auxiliary={'interval_status': ['available (conditional on successful draws)', 'unavailable'],
                   'interval_conditional': [True, False], 'interval_valid_samples': [20, 0],
                   'interval_requested_samples': 24, 'extrapolated': [False, True],
                   'extrapolation_status': ['within calibration', 'extrapolation'],
                   'extrapolation_reason': ['', 'time beyond measured range'],
                   'interval_extrapolated': [False, True],
                   'interval_extrapolation_status': ['within calibration', 'outside calibration'],
                   'extrapolation_warnings': ['one point outside calibration']})
    evaluation = CalibrationEvaluation('value_ratio', [1., 2.], [1.2, np.nan], [.8, np.nan], [1.4, np.nan], trajectory)
    good = IVIVCCalibrationAssessment(analysis=SimpleNamespace(model_results={'fit': object()}), relationship=evaluation)
    bad = IVIVCCalibrationAssessment(errors={'fit': AnalysisError('deliberate failure', 'RuntimeError')})
    result = CalibrationSimulationResult([good, bad], [course, course], [course, None],
        truth={'relationship': np.ones((2, 2))},
        metadata={'propagation_samples': 24, 'confidence_level': .95,
                  'evaluation_spec': {'relationship': {'quantity': 'value_ratio', 'coordinate': [1., 2.]}}})
    summary = summarize_calibration_simulation(result)
    label = 'Candidate <one>'
    recovery = RecoverySummary()
    return {
        'name': 'study', 'title': 'Study <example>', 'description': 'Fixed inputs & independent future measurements',
        'assumptions': {'historical_data_fixed': True, 'reference_truth': 'illustrative'},
        'designs': {label: {'in_vitro_times': [], 'in_vitro_replicates_per_time': 0,
                           'in_vitro_added_measurements': 0, 'in_vivo_times': [0., 2.],
                           'in_vivo_replicates_per_time': 3, 'in_vivo_added_measurements': 6}},
        'existing_data': {'in_vitro': course}, 'target': course,
        'references': {'in_vivo': ReferenceTrajectoryEnsemble([0., 2.], [[1., .7]], [True])},
        'calibration_results': {label: result}, 'calibration_summaries': {label: summary},
        'recovery_results': {label: SimpleNamespace(fit_success=np.array([True, False]),
                                                  fit_messages=[None, 'curve failure'])},
        'recovery_summaries': {label: recovery}, 'simulation_plans': {},
    }


class CalibrationPOCReportingTests(unittest.TestCase):
    def test_summary_and_point_tables_preserve_backend_denominators(self):
        scenario = fixture()
        original = scenario['calibration_summaries']['Candidate <one>']['relationship']
        tables = calibration_poc_tables(scenario)
        row = tables['design_summary.csv'].iloc[0]
        self.assertEqual(row['total_added_measurements'], 6)
        self.assertEqual(row['curve_penalized_mean_irmse'], .4)
        self.assertEqual(row['calibration_fit_failure_rate'], .5)
        self.assertAlmostEqual(row['relationship_coverage'], .25)
        self.assertEqual(row['relationship_conditional_coverage'], 1.)
        self.assertEqual(row['relationship_truth_points'], 4)
        self.assertEqual(row['relationship_interval_truth_points'], 1)
        pointwise = tables['pointwise_performance.csv']
        np.testing.assert_array_equal(pointwise['truth_counts'], original['truth_counts'])
        np.testing.assert_array_equal(pointwise['interval_counts'], [1, 0])
        self.assertTrue(np.isnan(pointwise.iloc[1]['pointwise_mean_interval_width']))

    def test_evaluation_exports_keep_unavailable_conditional_and_extrapolation_details(self):
        tables = calibration_poc_tables(fixture())
        evaluations = tables['evaluations.csv']
        self.assertEqual(len(evaluations), 4)  # Both requested points in both requested experiments.
        available = evaluations.iloc[0]
        self.assertTrue(available['interval_conditional'])
        self.assertEqual(available['interval_valid_samples'], 20)
        self.assertEqual(available['interval_requested_samples'], 24)
        self.assertTrue(evaluations.iloc[1]['extrapolated'])
        self.assertIn('beyond', evaluations.iloc[1]['extrapolation_reason'])
        self.assertFalse(evaluations.iloc[2]['point_available'])
        self.assertTrue(np.isnan(evaluations.iloc[2]['estimate']))
        outcomes = tables['experiment_outcomes.csv']
        self.assertEqual(len(outcomes), 2)
        self.assertIn('deliberate failure', outcomes.iloc[1]['calculation_errors'])
        self.assertFalse(outcomes.iloc[1]['curve_fit_available'])
        self.assertIn('outside calibration', outcomes.iloc[0]['relationship_extrapolation_warnings'])

    def test_input_tables_keep_sem_counts_and_roles(self):
        tables = calibration_poc_tables(fixture())
        fixed = tables['fixed_inputs.csv']
        self.assertEqual(set(fixed['role']), {'fixed_calibration', 'target_prediction'})
        np.testing.assert_allclose(fixed['standard_uncertainty'], .03)
        self.assertEqual(set(fixed['uncertainty_kind']), {'sem'})
        np.testing.assert_array_equal(fixed['replicate_count'], 4)
        generated = tables['simulated_calibration_data.csv']
        self.assertEqual(set(generated['environment']), {'in_vitro', 'in_vivo'})
        np.testing.assert_allclose(generated['standard_uncertainty'], .03)
        self.assertEqual(set(tables['references.csv']['weight']), {1.})

    def test_strict_json_csv_and_html_preserve_missing_values_and_escape_text(self):
        scenario = fixture()
        scenario['assumptions'].update(unavailable=np.inf, numpy_bool=np.bool_(True), scalar=np.array(np.nan))
        scenario['calibration_results']['Candidate <one>'].truth = {}
        scenario['calibration_summaries']['Candidate <one>'] = summarize_calibration_simulation(
            scenario['calibration_results']['Candidate <one>'])
        run = {'settings': {'n_simulations': 2, 'n_interval_samples': 24}, 'scenarios': {'study': scenario}}
        with TemporaryDirectory() as temporary:
            output = write_calibration_poc_outputs(run, temporary)
            self.assertEqual(output, Path(temporary))
            manifest = json.loads((output / 'manifest.json').read_text())
            metadata = json.loads((output / 'study/metadata.json').read_text())
            self.assertIsNone(metadata['assumptions']['unavailable'])
            self.assertIsNone(metadata['assumptions']['scalar'])
            self.assertTrue(metadata['assumptions']['numpy_bool'])
            self.assertEqual(len(metadata['calculation_failures']['Candidate <one>']), 1)
            for filename in manifest['files']['study']:
                self.assertTrue((output / filename).exists())
            html = (output / 'study/summary.html').read_text()
            self.assertIn('Study &lt;example&gt;', html)
            self.assertNotIn('<h1>Study <example>', html)
            self.assertIn('Unavailable', html)
            self.assertIn('pointwise', html)
            self.assertIn('extrapolation', html)
            if find_spec('plotly'):
                self.assertIn('plotly-graph-div', html)
            else:
                self.assertIn('require Plotly', html)
            evaluations = pd.read_csv(output / 'study/evaluations.csv')
            self.assertTrue(pd.isna(evaluations.iloc[2]['estimate']))

    def test_no_html_mode_removes_only_stale_html_outputs(self):
        run = {'settings': {}, 'scenarios': {'study': fixture()}}
        with TemporaryDirectory() as temporary:
            output = write_calibration_poc_outputs(run, temporary, include_html=True)
            write_calibration_poc_outputs(run, temporary, include_html=False)
            self.assertFalse((output / 'index.html').exists())
            self.assertFalse((output / 'study/summary.html').exists())
            self.assertTrue((output / 'study/design_summary.csv').exists())

    def test_unsafe_scenario_directory_is_rejected(self):
        with TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            write_calibration_poc_outputs({'settings': {}, 'scenarios': {'../other': fixture()}}, temporary)

    def test_actual_c2_demonstrations_export_requested_trials_and_both_workflows(self):
        from examples.study_design_calibration_poc import run_calibration_poc
        run = run_calibration_poc(n_simulations=2, n_interval_samples=24)
        with TemporaryDirectory() as temporary:
            root = write_calibration_poc_outputs(run, temporary, include_html=False)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(set(manifest['scenarios']), {'study', 'refinement'})
            for name, scenario in run['scenarios'].items():
                summary = pd.read_csv(root / name / 'design_summary.csv')
                outcomes = pd.read_csv(root / name / 'experiment_outcomes.csv')
                self.assertEqual(len(summary), len(scenario['designs']))
                self.assertEqual(len(outcomes), 2*len(scenario['designs']))
                self.assertTrue((summary['relationship_truth_points'] == 10).all())
                self.assertTrue((summary['prediction_truth_points'] == 10).all())
            self.assertIn('curve_penalized_mean_irmse', pd.read_csv(root / 'study/design_summary.csv'))
            self.assertIn('curve_penalized_integrated_rmse', pd.read_csv(root / 'study/experiment_outcomes.csv'))


if __name__ == '__main__':
    unittest.main()
