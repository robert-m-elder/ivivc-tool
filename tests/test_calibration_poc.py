"""The focused calibration POC uses the shared backend and explicit assumptions."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from examples import study_design_calibration_poc as poc


class CalibrationPOCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poc_run = poc.run_calibration_poc(n_simulations=2, n_interval_samples=24)

    def test_both_workflows_keep_candidate_counts_and_evaluation_domains_explicit(self):
        self.assertEqual(set(self.poc_run['scenarios']), {'study', 'refinement'})
        study = self.poc_run['scenarios']['study']
        refinement = self.poc_run['scenarios']['refinement']
        self.assertEqual(set(study['designs']),
                         {'early_times', 'spread_times', 'more_times', 'more_replicates'})
        self.assertEqual([item['in_vivo_added_measurements'] for item in study['designs'].values()],
                         [9, 9, 15, 18])
        self.assertEqual([item['in_vitro_added_measurements'] + item['in_vivo_added_measurements']
                          for item in refinement['designs'].values()], [0, 6, 6, 6, 12])
        for result in study['recovery_results'].values():
            np.testing.assert_array_equal(result.evaluation_times[[0, -1]], [0., 8.])
        self.assertFalse(refinement['recovery_results'])
        self.assertEqual(self.poc_run['settings']['model_name'], 'exponential_normalized')
        self.assertEqual(self.poc_run['settings']['confidence_level'], .95)

    def test_historical_means_use_sem_once_and_fixed_data_keep_fit_covariance(self):
        study = self.poc_run['scenarios']['study']
        course = study['existing_data']['in_vitro']
        np.testing.assert_allclose(course.observation_uncertainty.sigma, .03/np.sqrt(3))
        np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, 3)
        self.assertEqual(course.observation_uncertainty.response_kind, 'mean')
        for result in study['calibration_results'].values():
            for assembled, assessment in zip(result.in_vitro_data, result.assessments):
                np.testing.assert_array_equal(assembled.value, course.value)
                fit = assessment.analysis['approach2:exponential_normalized'].in_vitro_fit
                self.assertGreater(float(fit.covariance[0, 0]), 0.)
                self.assertEqual(fit.fit_diagnostics['covariance_scaling'], 'absolute')

    def test_optional_assessment_reuses_curve_recovery_measurements(self):
        scenario = self.poc_run['scenarios']['study']
        shared = scenario['simulation_plans']['in_vivo']
        for label, recovery in scenario['recovery_results'].items():
            calibration = scenario['calibration_results'][label]
            count = scenario['designs'][label]['in_vivo_replicates_per_time']
            times = recovery.design_times
            columns = shared.indices_for_times(times)
            true_values = scenario['references']['in_vivo'].evaluate(
                times, draw_indices=shared.trajectory_draw_indices)
            generated = (true_values[:, None, :]
                         + .05*shared.standard_normal_errors[:, :count, :][:, :, columns])
            np.testing.assert_allclose(recovery.observed_means, generated.mean(axis=1))
            for experiment, course in enumerate(calibration.in_vivo_data):
                order = np.argsort(np.tile(times, count), kind='stable')
                np.testing.assert_array_equal(course.value, generated[experiment].reshape(-1)[order])
                np.testing.assert_allclose(course.observation_uncertainty.sigma, .05)
            self.assertTrue(calibration.metadata['uses_same_study_measurements'])

    def test_refinement_keeps_old_sem_counts_and_independent_duplicate_means(self):
        scenario = self.poc_run['scenarios']['refinement']
        for side, sd in (('in_vitro', .06), ('in_vivo', .05)):
            old = scenario['existing_data'][side]
            np.testing.assert_allclose(old.observation_uncertainty.sigma, sd/np.sqrt(3))
            np.testing.assert_array_equal(old.observation_uncertainty.replicate_counts, 3)
            for result in scenario['calibration_results'].values():
                for course in getattr(result, side + '_data'):
                    for time, response, sem in zip(old.time, old.value, old.observation_uncertainty.sigma):
                        match = np.flatnonzero(course.time == time)[0]
                        self.assertEqual(course.value[match], response)
                        self.assertEqual(course.observation_uncertainty.sigma[match], sem)
        repeated = scenario['calibration_results']['repeat_vitro'].in_vitro_data[0]
        self.assertEqual(np.count_nonzero(repeated.time == .5), 2)
        self.assertEqual(np.count_nonzero(repeated.time == 1.5), 2)
        np.testing.assert_array_equal(repeated.observation_uncertainty.replicate_counts, 3)
        self.assertIn('independent new cohorts', scenario['assumptions']['cohorts'])

    def test_shared_reference_draws_define_relationship_and_target_truth(self):
        for scenario in self.poc_run['scenarios'].values():
            first = None
            for result in scenario['calibration_results'].values():
                indices = result.metadata['in_vivo_draw_indices']
                if first is None:
                    first = indices
                np.testing.assert_array_equal(indices, first)
                true_vivo = scenario['references']['in_vivo'].evaluate(
                    poc.EVALUATION_TIMES, draw_indices=indices)
                true_vitro = scenario['references']['in_vitro'].evaluate(
                    poc.EVALUATION_TIMES, draw_indices=np.zeros(len(indices), dtype=int))
                expected = true_vivo/true_vitro
                np.testing.assert_allclose(result.truth['relationship'], expected)
                np.testing.assert_allclose(result.truth['prediction'], expected*scenario['target'].value)
                self.assertFalse(result.metadata['future_observation_variability_included'])

    def test_reproducibility_and_fixed_history_do_not_depend_on_simulation_count(self):
        again = poc.run_calibration_poc(n_simulations=2, n_interval_samples=24)
        larger = poc.run_calibration_poc(n_simulations=3, n_interval_samples=24)
        for name, original in self.poc_run['scenarios'].items():
            for side, course in original['existing_data'].items():
                np.testing.assert_array_equal(course.value, larger['scenarios'][name]['existing_data'][side].value)
            for label, result in original['calibration_results'].items():
                other = again['scenarios'][name]['calibration_results'][label]
                for actual, expected in zip(result.assessments, other.assessments):
                    np.testing.assert_array_equal(actual.relationship.estimate, expected.relationship.estimate)
                    np.testing.assert_array_equal(actual.relationship.lower, expected.relationship.lower)
                    np.testing.assert_array_equal(actual.relationship.upper, expected.relationship.upper)

    def test_invalid_settings_are_rejected_before_work(self):
        for options in ({'n_simulations': 0}, {'n_simulations': True}, {'n_simulations': 1.5},
                        {'n_interval_samples': 1}, {'random_seed': -1}, {'scenario': 'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                poc.run_calibration_poc(**options)

    def test_cli_writes_strict_json_and_numerical_outputs_without_html(self):
        with TemporaryDirectory() as temporary:
            output = io.StringIO()
            with redirect_stdout(output):
                status = poc.main(['--scenario', 'study', '--simulations', '2',
                                   '--interval-samples', '24', '--seed', '41',
                                   '--output-dir', temporary, '--no-html'])
            console = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(console['settings']['random_seed'], 41)
            self.assertEqual(console['settings']['n_simulations'], 2)
            self.assertEqual(console['settings']['n_interval_samples'], 24)
            self.assertEqual(set(console['scenarios']), {'study'})
            self.assertIsNone(console['index_html'])
            root = Path(temporary)
            self.assertTrue((root/'manifest.json').exists())
            self.assertTrue((root/'study/design_summary.csv').exists())
            self.assertFalse((root/'index.html').exists())
            metadata = json.loads((root/'study/metadata.json').read_text(),
                                  parse_constant=lambda item: self.fail('Nonfinite JSON token: ' + item))
            self.assertEqual(metadata['name'], 'study')


if __name__ == '__main__':
    unittest.main()
