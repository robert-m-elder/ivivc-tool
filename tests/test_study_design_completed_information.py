"""Completed precision replaces planned information at completed times."""
import unittest

import numpy as np

from ivivc_core import ObservationUncertainty
from study_design_core import (
    DesignConstraints, ExistingObservations, ObservationModel, ReferenceModelEnsemble,
    coordinate_exchange_i_optimal_design, coordinate_exchange_local_i_optimal_design,
    ensemble_i_optimality, evaluate_i_optimal_design, evaluate_local_i_optimal_design,
    greedy_i_optimal_designs, greedy_local_i_optimal_designs, information_matrix,
    local_i_optimality, optimize_i_optimal_designs, optimize_local_i_optimal_designs,
    trapezoidal_integration_weights,
)


def reference_model():
    return ReferenceModelEnsemble('linear', ('a', 'b'), [[1., -.2], [1.1, -.21]],
        [True, True], [.4, .6], [1.06, -.206], np.linspace(0., 4., 9), (0., 4.))


def completed(times, sigma=None, counts=None):
    return ExistingObservations(times, 1. - .2*np.asarray(times),
        observation_uncertainty=(None if sigma is None else ObservationUncertainty(
            sigma, response_kind='mean', replicate_counts=counts)))


def analytic_information(times, sigma):
    matrix = np.column_stack([np.ones(len(times)), times])
    return matrix.T @ (matrix / np.asarray(sigma)[:, None]**2)


def analytic_criterion(information, grid):
    matrix = np.column_stack([np.ones(len(grid)), grid])
    return np.sum(trapezoidal_integration_weights(grid)
                  * np.einsum('ij,jk,ik->i', matrix, np.linalg.inv(information), matrix))


class CompletedInformationTests(unittest.TestCase):
    def setUp(self):
        self.model = reference_model()
        self.observation = ObservationModel('constant_sd', .2, replicates_per_time=3)
        self.constraints = DesignConstraints(candidate_times=[1., 2., 3.], fixed_times=[0., 4.],
            min_time=0., max_time=4., min_number_of_times=3, max_number_of_times=3,
            replicates_per_time=3)

    def information(self, design, existing, observation=None, **kwargs):
        return information_matrix('linear', design, [1., -.2],
            self.observation if observation is None else observation,
            existing_observations=existing, **kwargs)

    def test_completed_rows_replace_future_replicates_in_analytic_information(self):
        existing = completed([0., 0., 4.], [.1, .2, .4], [4, 9, 25])
        observed = self.information([0., 2., 4.], existing)
        expected = analytic_information([0., 0., 4., 2.], [.1, .2, .4, .2/np.sqrt(3)])
        np.testing.assert_allclose(observed, expected, rtol=1e-7)

    def test_reported_mean_counts_do_not_rescale_supplied_sem(self):
        first = self.information([0., 2., 4.], completed([0., 4.], [.1, .3], [4, 9]))
        second = self.information([0., 2., 4.], completed([0., 4.], [.1, .3], [100, 1]))
        np.testing.assert_array_equal(first, second)

    def test_completed_raw_replicates_and_equivalent_means_have_equal_information(self):
        times, sd = np.array([0., 4.]), np.array([.1, .3])
        raw = ExistingObservations(np.repeat(times, 3), np.repeat([1., .2], 3),
            observation_uncertainty=ObservationUncertainty(np.repeat(sd, 3), response_kind='individual'))
        means = completed(times, sd/np.sqrt(3), counts=3)
        np.testing.assert_allclose(self.information([0., 2., 4.], raw),
                                   self.information([0., 2., 4.], means), rtol=1e-12)

    def test_future_count_changes_only_new_measurement_information(self):
        existing = completed([0., 0., 4.])  # Individual-SD fallback, three actual rows.
        first = self.information([0., 2., 4.], existing, replicates_per_time=1)
        second = self.information([0., 2., 4.], existing, replicates_per_time=5)
        np.testing.assert_allclose(first,
            analytic_information([0., 0., 4., 2.], [.2, .2, .2, .2]), rtol=1e-7)
        np.testing.assert_allclose(second - first,
            4*analytic_information([2.], [.2]), rtol=1e-7)

    def test_fixed_future_commitments_still_receive_planned_replicates(self):
        existing = completed([0.], [.1], counts=4)
        score = evaluate_local_i_optimal_design([0., 2., 4.], self.model, self.observation,
            self.constraints, existing_observations=existing)
        expected = analytic_information([0., 2., 4.], [.1, .2/np.sqrt(3), .2/np.sqrt(3)])
        self.assertAlmostEqual(score.criterion, analytic_criterion(expected, self.model.time_grid), places=9)

    def test_known_completed_precision_does_not_require_future_sd_at_completed_times(self):
        observation = ObservationModel('time_profile_sd', replicates_per_time=3,
            profile_times=[0., 2., 4.], profile_sd=[0., .2, 0.])
        existing = completed([0., 4.], [.1, .3])
        expected = analytic_information([0., 2., 4.], [.1, .2/np.sqrt(3), .3])
        np.testing.assert_allclose(self.information([0., 2., 4.], existing, observation), expected, rtol=1e-7)

    def test_completed_only_information_does_not_use_the_future_noise_model(self):
        existing = completed([0., 4.], [.1, .3])
        actual = self.information([0., 4.], existing, ObservationModel('constant_sd', 0.))
        np.testing.assert_allclose(actual, analytic_information([0., 4.], [.1, .3]), rtol=1e-7)

    def test_completed_times_must_have_unambiguous_design_and_fixed_membership(self):
        existing = completed([0., 4.], [.1, .3])
        for design in ([2., 4.], [0., 5e-13, 4.]):
            with self.subTest(design=design), self.assertRaises(ValueError):
                self.information(design, existing)
        constraints = DesignConstraints(candidate_times=[0., 1., 2., 3.], fixed_times=[4.],
            min_time=0., max_time=4., min_number_of_times=3, max_number_of_times=3,
            replicates_per_time=3)
        with self.assertRaisesRegex(ValueError, 'fixed|completed|Completed'):
            optimize_local_i_optimal_designs(self.model, self.observation, constraints,
                                             existing_observations=existing)

    def test_all_optimizer_paths_forward_completed_precision_to_scores(self):
        existing = completed([0., 4.], [.01, .5])
        expected_times = np.array([0., 3., 4.])
        expected = analytic_criterion(analytic_information(
            [0., 3., 4.], [.01, .2/np.sqrt(3), .5]), self.model.time_grid)
        for evaluate in (evaluate_local_i_optimal_design, evaluate_i_optimal_design):
            with self.subTest(api=evaluate.__name__):
                score = evaluate(expected_times, self.model, self.observation, self.constraints,
                                 existing_observations=existing)
                self.assertAlmostEqual(score.criterion, expected, places=9)
        for greedy in (greedy_local_i_optimal_designs, greedy_i_optimal_designs):
            with self.subTest(api=greedy.__name__):
                step = greedy(self.model, self.observation, self.constraints,
                              existing_observations=existing)[0]
                np.testing.assert_array_equal(step.times, expected_times)
                self.assertAlmostEqual(step.score.criterion, expected, places=9)
        for exchange in (coordinate_exchange_local_i_optimal_design, coordinate_exchange_i_optimal_design):
            with self.subTest(api=exchange.__name__):
                result = exchange([0., 2., 4.], self.model, self.observation, self.constraints,
                                  existing_observations=existing)
                np.testing.assert_array_equal(result.times, expected_times)
                self.assertAlmostEqual(result.score.criterion, expected, places=9)
        for optimize in (optimize_local_i_optimal_designs, optimize_i_optimal_designs):
            with self.subTest(api=optimize.__name__):
                result = optimize(self.model, self.observation, self.constraints,
                                  existing_observations=existing)
                np.testing.assert_array_equal(result.nested_steps[0].times, expected_times)
                np.testing.assert_array_equal(result.refined_steps[0].times, expected_times)
                self.assertAlmostEqual(result.refined_steps[0].score.criterion, expected, places=9)

    def test_reversing_completed_precision_changes_the_selected_new_time(self):
        chosen = []
        for sigma in ([.01, .5], [.5, .01]):
            result = optimize_local_i_optimal_designs(self.model, self.observation, self.constraints,
                existing_observations=completed([0., 4.], sigma), refine=False)
            chosen.append(result.nested_steps[0].added_time)
        self.assertEqual(chosen, [3., 1.])

    def test_low_level_criteria_and_legacy_unspecified_inputs_remain_consistent(self):
        existing = completed([0., 4.], [.1, .3])
        expected = analytic_criterion(analytic_information(
            [0., 2., 4.], [.1, .2/np.sqrt(3), .3]), self.model.time_grid)
        local = local_i_optimality('linear', [0., 2., 4.], [1., -.2], self.observation,
            evaluation_times=self.model.time_grid, existing_observations=existing)
        ensemble = ensemble_i_optimality(self.model, [0., 2., 4.], self.observation,
            existing_observations=existing)
        self.assertAlmostEqual(local, expected, places=9)
        self.assertAlmostEqual(ensemble.criterion, expected, places=9)
        plain = information_matrix('linear', [0., 2., 4.], [1., -.2], self.observation)
        explicit = self.information([0., 2., 4.], None)
        np.testing.assert_array_equal(plain, explicit)


if __name__ == '__main__':
    unittest.main()
