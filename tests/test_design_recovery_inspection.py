"""Numerical pairing and unavailable-data checks for compact inspection data."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from utilities.design_planning import run_measured_planning
from utilities.design_recovery_inspection import build_recovery_inspection


def retained_result(*, success=(True, True, True)):
    truth = np.array([[0., 100.], [100., 0.], [101., 101.]])
    fitted = np.array([[20., 101.], [0., 20.], [102., 0.]])
    simulated = SimpleNamespace(
        evaluation_times=np.array([0., 1.]), design_times=np.array([0., 1.]),
        trajectory_draw_indices=np.array([4, 8, 2]),
        true_curves=truth, fitted_curves=fitted,
        observed_means=np.array([[.8, .4], [.7, .3], [.9, .5]]),
        expected_values=truth.copy(), fit_success=np.asarray(success),
        fit_messages=(None, 'Fit failed', None), metadata={'replicates_per_time': 2},
    )
    return SimpleNamespace(
        comparison=SimpleNamespace(
            simulation_plan=SimpleNamespace(trajectory_draw_indices=np.array([4, 8, 2])),
            results={'First': simulated, 'Second': deepcopy(simulated)},
        ), schedules={'First': [0., 1.], 'Second': [0., 1.]},
        settings={'replicates_per_time': 2}, completed_observations=None,
    )


class RecoveryInspection(unittest.TestCase):
    def test_errors_are_paired_before_summarizing_not_subtracted_marginal_quantiles(self):
        result = retained_result()
        data = build_recovery_inspection(result)
        simulated = result.comparison.results['First']
        errors = simulated.fitted_curves - simulated.true_curves
        summary = data['schedules'][0]['recovery_error']
        np.testing.assert_allclose(summary['median'], np.median(errors, axis=0))
        np.testing.assert_allclose(summary['lower'], np.quantile(errors, .025, axis=0))
        np.testing.assert_allclose(summary['upper'], np.quantile(errors, .975, axis=0))
        self.assertFalse(np.allclose(summary['median'],
                         np.median(simulated.fitted_curves, axis=0) -
                         np.median(simulated.true_curves, axis=0)))

    def test_failed_fits_are_excluded_even_if_curve_values_are_retained(self):
        result = retained_result(success=(True, False, True))
        data = build_recovery_inspection(result)['schedules'][0]
        self.assertEqual((data['n_successful'], data['n_failed'], data['n_simulations']), (2, 1, 3))
        self.assertEqual(data['recovered_response']['available_counts'], [2, 2])
        np.testing.assert_allclose(data['recovered_response']['median'], [61., 50.5])

    def test_nonfinite_values_keep_gaps_and_pointwise_counts(self):
        result = retained_result()
        simulated = result.comparison.results['First']
        simulated.fitted_curves[:, 1] = [np.nan, np.inf, -np.inf]
        simulated.true_curves[0, 0] = np.nan
        data = build_recovery_inspection(result)
        first = data['schedules'][0]
        self.assertEqual(first['recovered_response']['available_counts'], [3, 0])
        self.assertEqual(first['recovery_error']['available_counts'], [2, 0])
        self.assertIsNone(first['recovered_response']['median'][1])
        self.assertIsNone(first['example']['true_response'][0])
        self.assertIsNone(first['example']['recovered_response'][1])
        json.dumps(data, allow_nan=False)

    def test_one_success_shows_median_without_zero_width_range(self):
        data = build_recovery_inspection(retained_result(success=(False, True, False)))
        first = data['schedules'][0]
        self.assertEqual(first['recovered_response']['median'], [0., 20.])
        self.assertEqual(first['recovered_response']['lower'], [None, None])
        self.assertEqual(first['recovery_error']['upper'], [None, None])

    def test_extreme_finite_curves_have_finite_quantiles_and_overflow_errors_are_unavailable(self):
        result = retained_result(success=(True, True, False))
        simulated = result.comparison.results['First']
        simulated.fitted_curves[:2] = [[-1e308, -1e308], [1e308, 1e308]]
        simulated.true_curves[:2] = [[0., 1e308], [0., -1e308]]
        data = build_recovery_inspection(result)
        first = data['schedules'][0]
        response = first['recovered_response']
        np.testing.assert_allclose(response['lower'], [-9.5e307, -9.5e307])
        np.testing.assert_allclose(response['median'], [0., 0.])
        np.testing.assert_allclose(response['upper'], [9.5e307, 9.5e307])
        errors = first['recovery_error']
        self.assertEqual(errors['available_counts'], [2, 0])
        self.assertEqual(errors['median'], [0., None])
        self.assertEqual(first['example']['recovery_error'], [-1e308, None])
        json.dumps(data, allow_nan=False)

    def test_zero_successes_leave_recovery_unavailable_and_keep_truth_and_observations(self):
        data = build_recovery_inspection(retained_result(success=(False, False, False)))
        first = data['schedules'][0]
        self.assertEqual(first['n_successful'], 0)
        self.assertEqual(first['recovered_response']['median'], [None, None])
        self.assertEqual(first['recovered_response']['available_counts'], [0, 0])
        self.assertEqual(first['example']['recovered_response'], [None, None])
        self.assertEqual(first['example']['true_response'], [0., 100.])
        self.assertEqual(len(first['example']['observed_time_means']), 2)

    def test_example_is_fixed_before_fit_success_is_known_and_shared_between_schedules(self):
        result = retained_result()
        first = result.comparison.results['First']
        first.fit_success[0] = False
        first.fit_messages = ('Specific failure', None, None)
        data = build_recovery_inspection(result)
        self.assertTrue(data['shared_example']['available'])
        for schedule in data['schedules']:
            self.assertEqual(schedule['example']['simulation_index'], 0)
            self.assertEqual(schedule['example']['study_number'], 1)
            self.assertEqual(schedule['example']['trajectory_draw_index'], 4)
        self.assertFalse(data['schedules'][0]['example']['fit_success'])
        self.assertEqual(data['schedules'][0]['example']['fit_message'], 'Specific failure')
        self.assertEqual(data['schedules'][0]['example']['recovered_response'], [None, None])
        self.assertTrue(data['schedules'][1]['example']['fit_success'])

    def test_shared_example_is_unavailable_if_retained_draws_do_not_match_shared_plan(self):
        result = retained_result()
        result.comparison.results['Second'].trajectory_draw_indices[0] = 99
        data = build_recovery_inspection(result)
        self.assertFalse(data['shared_example']['available'])
        self.assertTrue(all(row['example'] is None for row in data['schedules']))
        self.assertEqual(data['schedules'][0]['n_successful'], 3)

    def test_fit_intervals_are_explicitly_separate_and_not_invented(self):
        data = build_recovery_inspection(retained_result())
        self.assertFalse(data['fit_reported_intervals']['available'])
        self.assertIn('not retained', data['fit_reported_intervals']['reason'])

    def test_no_comparison_has_no_example_or_invented_recovery_arrays(self):
        result = retained_result()
        result.comparison = None
        result.schedules = {}
        data = build_recovery_inspection(result)
        self.assertEqual(data['schedules'], [])
        self.assertFalse(data['shared_example']['available'])
        json.dumps(data, allow_nan=False)

    def test_returns_summaries_and_one_example_without_mutating_retained_arrays(self):
        result = retained_result()
        before = deepcopy(result.comparison.results['First'].fitted_curves)
        data = build_recovery_inspection(result)
        self.assertEqual([row['id'] for row in data['schedules']], ['schedule-0', 'schedule-1'])
        self.assertNotIn('fitted_curves', json.dumps(data))
        np.testing.assert_array_equal(result.comparison.results['First'].fitted_curves, before)


class ActualPlanningRecoveryInspection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        completed = prepare_time_course(
            [0., 0., 1.], [1., .98, .8],
            observation_uncertainty=ObservationUncertainty(
                [.02, .03, .04], response_kind='mean', replicate_counts=[4, 9, 16]),
        )[0]
        cls.result = run_measured_planning(
            None,
            {'model_name': 'linear', 'first_time': '0', 'last_time': '4', 'time_step': '1',
             'min_times': '3', 'max_times': '4', 'fixed_times': '0',
             'reference_draws': '8', 'n_simulations': '4', 'random_seed': '2',
             'replicates_per_time': '2', 'include_even': 'on', 'has_completed': 'on',
             'future_policy': 'future_only'},
            completed=completed,
        )

    def test_actual_outputs_share_the_same_illustrative_truth_and_retained_means(self):
        data = build_recovery_inspection(self.result)
        self.assertTrue(data['shared_example']['available'])
        self.assertGreater(len(data['schedules']), 1)
        first_example = data['schedules'][0]['example']
        for row in data['schedules']:
            source = self.result.comparison.results[row['name']]
            np.testing.assert_allclose(row['example']['true_response'], source.true_curves[0])
            self.assertEqual(row['example']['true_response'], first_example['true_response'])
            np.testing.assert_allclose([value['response'] for value in row['example']['observed_time_means']],
                                       source.observed_means[0])
            self.assertEqual(row['n_successful'], source.n_successful)
        json.dumps(data, allow_nan=False)

    def test_original_completed_rows_and_precision_are_separate_from_future_time_means(self):
        data = build_recovery_inspection(self.result)
        completed = data['completed_observations']
        self.assertEqual([row['time'] for row in completed], [0., 0., 1.])
        self.assertEqual([row['response'] for row in completed], [1., .98, .8])
        self.assertEqual([row['observation_sigma'] for row in completed], [.02, .03, .04])
        self.assertEqual([row['reported_replicate_count'] for row in completed], [4, 9, 16])
        self.assertTrue(all(row['uncertainty_kind'] == 'SEM' for row in completed))
        for schedule in data['schedules']:
            for mean in schedule['example']['observed_time_means']:
                self.assertEqual(mean['role'], 'completed' if mean['time'] in (0., 1.) else 'new')
                self.assertEqual(mean['planned_replicate_count'], None if mean['role'] == 'completed' else 2)


if __name__ == '__main__':
    unittest.main()
