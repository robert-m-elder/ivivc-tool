"""Approach 1 POC retains the same studies while assessing mapped times."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from examples import study_design_calibration_poc as poc


class Approach1CalibrationPOCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_run = poc.run_calibration_poc(3, 24, approach_id='approach1')
        cls.value_run = poc.run_calibration_poc(3, 24)

    def test_model_roles_grids_and_candidate_counts_are_explicit(self):
        settings = self.time_run['settings']
        self.assertEqual(settings['approach_id'], 'approach1')
        self.assertEqual(settings['model_name'], 'linear')
        self.assertEqual(settings['generating_model_name'], poc.MODEL)
        self.assertEqual(settings['study_analysis_model_name'], poc.MODEL)
        self.assertEqual(settings['prediction_method'], 'time_mapping')
        self.assertEqual(settings['selected_interpolation'], ['default'])
        np.testing.assert_array_equal(settings['evaluation_times'], poc.APPROACH1_EVALUATION_TIMES)
        for name, scenario in self.time_run['scenarios'].items():
            expected = self.value_run['scenarios'][name]
            self.assertEqual(tuple(scenario['designs']), tuple(expected['designs']))
            for label, design in scenario['designs'].items():
                self.assertEqual(design['selection'], 'specified_candidate')
                for side in ('in_vitro', 'in_vivo'):
                    np.testing.assert_array_equal(design[side+'_times'], expected['designs'][label][side+'_times'])
                    self.assertEqual(design[side+'_added_measurements'],
                                     expected['designs'][label][side+'_added_measurements'])

    def test_both_approaches_use_identical_historical_and_simulated_observations(self):
        for name, scenario in self.time_run['scenarios'].items():
            other = self.value_run['scenarios'][name]
            for side, course in scenario['existing_data'].items():
                np.testing.assert_array_equal(course.value, other['existing_data'][side].value)
                np.testing.assert_array_equal(course.observation_uncertainty.sigma,
                                              other['existing_data'][side].observation_uncertainty.sigma)
            for label, result in scenario['calibration_results'].items():
                previous = other['calibration_results'][label]
                for side in ('in_vitro_data', 'in_vivo_data'):
                    for actual, expected in zip(getattr(result, side), getattr(previous, side)):
                        np.testing.assert_array_equal(actual.time, expected.time)
                        np.testing.assert_array_equal(actual.value, expected.value)
                        np.testing.assert_array_equal(actual.observation_uncertainty.sigma,
                                                      expected.observation_uncertainty.sigma)

    def test_approach_choice_preserves_curve_recovery_arrays(self):
        study = self.time_run['scenarios']['study']
        for label, recovery in study['recovery_results'].items():
            previous = self.value_run['scenarios']['study']['recovery_results'][label]
            for name in ('trajectory_draw_indices', 'expected_values', 'observed_means',
                         'true_curves', 'fitted_curves', 'parameter_estimates',
                         'fit_success', 'integrated_rmse', 'penalized_integrated_rmse'):
                np.testing.assert_array_equal(getattr(recovery, name), getattr(previous, name))
            self.assertEqual(recovery.model_name, poc.MODEL)
            np.testing.assert_array_equal(recovery.evaluation_times[[0, -1]], [0., 8.])

    def test_timing_truth_follows_each_selected_generating_curve(self):
        times = poc.APPROACH1_EVALUATION_TIMES
        for scenario in self.time_run['scenarios'].values():
            common_indices = None
            for result in scenario['calibration_results'].values():
                indices = result.metadata['in_vivo_draw_indices']
                if common_indices is None:
                    common_indices = indices
                np.testing.assert_array_equal(indices, common_indices)
                # The backend uses PCHIP representations of these exponential
                # generating curves, so allow its small interpolation error.
                expected = poc.VITRO_RATE / poc.VIVO_RATES[indices, None] * times
                expected_target = (poc.VITRO_RATE / poc.VIVO_RATES[indices, None]
                                   * poc.APPROACH1_TARGET_TIMES)
                np.testing.assert_allclose(result.truth['relationship'], expected, atol=1e-7)
                np.testing.assert_allclose(result.truth['prediction'], expected_target, atol=1e-7)
                self.assertTrue(np.isfinite(result.truth['relationship']).all())
                self.assertTrue((result.truth['relationship'] <= 12.).all())

    def test_targets_predict_time_and_carry_fixed_response_coordinates(self):
        for scenario in self.time_run['scenarios'].values():
            target = scenario['target']
            np.testing.assert_array_equal(target.time, poc.APPROACH1_TARGET_TIMES)
            for result in scenario['calibration_results'].values():
                self.assertEqual(result.metadata['approach_id'], 'approach1')
                for assessment in result.assessments:
                    self.assertFalse(assessment.errors)
                    self.assertEqual(assessment.relationship.quantity, 'in_vivo_time')
                    self.assertEqual(assessment.prediction.quantity, 'in_vivo_time')
                    np.testing.assert_array_equal(assessment.prediction.coordinate, target.time)
                    np.testing.assert_array_equal(assessment.prediction.trajectory.in_vivo_value, target.value)
                    self.assertFalse(assessment.metadata['calibration_uncertainty_used_in_fit'])
                    fitted = assessment.analysis['approach1:linear'].time_mapping_fit
                    self.assertIsNone(fitted.observation_uncertainty)

    def test_assumptions_distinguish_fixed_history_regression_bands_and_empirical_ranges(self):
        for scenario in self.time_run['scenarios'].values():
            assumptions = scenario['assumptions']
            self.assertIn('does not weight', assumptions['fixed_data_uncertainty'])
            self.assertIn('regression', assumptions['uncertainty_intervals'])
            self.assertIn('each error subtracts', assumptions['empirical_ranges'])
            self.assertIn('time mapping', assumptions['target_transfer'])
            np.testing.assert_array_equal(assumptions['target_evaluation_times'],
                                          poc.APPROACH1_TARGET_TIMES)
            self.assertIn('not optimized', assumptions['schedules'])
            for summary in scenario['calibration_summaries'].values():
                self.assertIn('pointwise_error_quantiles', summary['relationship'])
                self.assertIn('pointwise_bias', summary['prediction'])

    def test_repeated_runs_and_fixed_history_are_reproducible(self):
        again = poc.run_calibration_poc(3, 24, approach_id='approach1')
        larger = poc.run_calibration_poc(4, 24, scenario='refinement', approach_id='approach1')
        for name, scenario in self.time_run['scenarios'].items():
            for label, result in scenario['calibration_results'].items():
                repeated = again['scenarios'][name]['calibration_results'][label]
                for actual, expected in zip(result.assessments, repeated.assessments):
                    np.testing.assert_array_equal(actual.relationship.estimate, expected.relationship.estimate)
                    np.testing.assert_array_equal(actual.relationship.lower, expected.relationship.lower)
                    np.testing.assert_array_equal(actual.relationship.upper, expected.relationship.upper)
        for side, original in self.time_run['scenarios']['refinement']['existing_data'].items():
            np.testing.assert_array_equal(original.value,
                                          larger['scenarios']['refinement']['existing_data'][side].value)

    def test_invalid_approach_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'approach_id'):
            poc.run_calibration_poc(approach_id='approach3')

    def test_cli_selects_approach1_and_writes_strict_json(self):
        with TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                status = poc.main(['--approach', 'approach1', '--scenario', 'study',
                                   '--simulations', '2', '--interval-samples', '24',
                                   '--output-dir', directory, '--no-html'])
            self.assertEqual(status, 0)
            console = json.loads(output.getvalue())
            self.assertEqual(console['settings']['approach_id'], 'approach1')
            self.assertEqual(console['settings']['model_name'], 'linear')
            self.assertEqual(set(console['scenarios']), {'study'})
            root = Path(directory)
            metadata = json.loads((root/'study/metadata.json').read_text(),
                                  parse_constant=lambda value: self.fail('Nonfinite JSON: '+value))
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['settings']['approach_id'], 'approach1')
            self.assertEqual(metadata['name'], 'study')
            self.assertTrue((root/'study/design_summary.csv').exists())
            self.assertFalse((root/'index.html').exists())


if __name__ == '__main__':
    unittest.main()
