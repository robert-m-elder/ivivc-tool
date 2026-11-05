"""Study refits use observation precision without counting replicates twice."""
import unittest
from unittest.mock import patch

import numpy as np

from ivivc_core import ObservationUncertainty
from study_design_core import (
    ExistingObservations, ObservationModel, ReferenceTrajectoryEnsemble,
    ReferenceModelEnsemble, SimulationPlan, fit_model_once, fit_reference_model_ensemble, simulate_design,
)
import study_design_core.simulation as simulation


def reference():
    time = np.linspace(0., 4., 9)
    draws = np.vstack([1. - .15*time, 1.4 - .15*time])
    return ReferenceTrajectoryEnsemble(
        time, draws, [True, True], draw_weights=[.5, .5],
        central_curve=np.mean(draws, axis=0), interpolation='linear')


def plan(times, replicates=2):
    errors = np.arange(2*replicates*len(times), dtype=float)
    errors = (errors.reshape(2, replicates, len(times)) - errors.mean()) / 10
    return SimulationPlan(times, [0, 1], errors)


def captured_simulation(ref, times, model, **kwargs):
    records = []
    original = simulation.fit_model_once

    def record(time, response, **options):
        result = original(time, response, **options)
        records.append((np.array(time), np.array(response), options, result))
        return result

    with patch.object(simulation, 'fit_model_once', side_effect=record):
        result = simulate_design(ref, times, model, model_name='linear',
                                 fallback_grid_search=False, **kwargs)
    return result, records


def analytic_fit(time, response, sigma):
    matrix = np.column_stack([np.ones(len(time)), time])
    weighted = matrix / np.asarray(sigma)[:, None]
    covariance = np.linalg.inv(weighted.T @ weighted)
    parameters = covariance @ (weighted.T @ (response / sigma))
    return parameters, covariance


class WeightedSimulationTests(unittest.TestCase):
    def test_mixed_completed_means_and_future_replicates_match_analytic_wls(self):
        ref = reference()
        model = ObservationModel('time_profile_sd', replicates_per_time=2,
            profile_times=[0., 2., 4.], profile_sd=[.1, .3, .5])
        completed = ExistingObservations([0., 1.], [.95, .92],
            observation_uncertainty=ObservationUncertainty(
                [.07, .09], response_kind='mean', replicate_counts=[4, 9]))
        shared = plan([0., 1., 2., 4.])
        result, records = captured_simulation(ref, shared.master_times, model,
            existing_observations=completed, simulation_plan=shared)
        expected_times = [0., 1., 2., 4., 2., 4.]
        expected_sigma = np.array([.07, .09, .3, .5, .3, .5])
        for index, (time, response, options, fit) in enumerate(records):
            future = (ref.evaluate([2., 4.], draw_indices=[index])[0]
                      + np.array([.3, .5]) * shared.standard_normal_errors[index, :, 2:])
            expected_response = np.r_[completed.response, future.reshape(-1)]
            np.testing.assert_array_equal(time, expected_times)
            np.testing.assert_array_equal(response, expected_response)
            np.testing.assert_array_equal(options['sigma'], expected_sigma)
            self.assertNotIn('observation_uncertainty', options)
            parameters, covariance = analytic_fit(time, response, expected_sigma)
            np.testing.assert_allclose(fit.parameter_vector, parameters, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-6)
            np.testing.assert_array_equal(fit.observation_sigma, expected_sigma)
            self.assertIsNone(fit.observation_uncertainty)
        self.assertTrue(result.fit_success.all())
        np.testing.assert_array_equal(result.observed_means[:, :2], [[.95, .92]]*2)
        np.testing.assert_array_equal(result.metadata['refit_observation_sigma'], expected_sigma)
        self.assertEqual(result.metadata['refit_covariance_scaling'], 'absolute')
        self.assertEqual(result.metadata['completed_observation_uncertainty']['uncertainty_kind'], 'sem')

    def test_completed_sem_counts_do_not_change_combined_refits(self):
        model = ObservationModel('constant_sd', .2, replicates_per_time=2)
        shared = plan([0., 1., 2., 4.])
        fits = []
        for counts in ([4, 9], [100, 1]):
            completed = ExistingObservations([0., 1.], [.95, .92],
                observation_uncertainty=ObservationUncertainty(
                    [.07, .09], response_kind='mean', replicate_counts=counts))
            result, records = captured_simulation(reference(), shared.master_times, model,
                existing_observations=completed, simulation_plan=shared)
            self.assertTrue(result.fit_success.all())
            fits.append(records)
        for first, second in zip(*fits):
            np.testing.assert_array_equal(first[3].parameter_vector, second[3].parameter_vector)
            np.testing.assert_array_equal(first[3].covariance, second[3].covariance)

    def test_completed_replicates_and_equivalent_means_have_same_absolute_covariance(self):
        times = np.array([0., 2., 4.])
        means = np.array([1.02, .65, .43])
        sd = np.array([.1, .2, .3])
        responses = means[:, None] + sd[:, None] * np.array([-.2, 0., .2])
        raw = ExistingObservations(np.repeat(times, 3), responses.reshape(-1),
            observation_uncertainty=ObservationUncertainty(
                np.repeat(sd, 3), response_kind='individual'))
        averaged = ExistingObservations(times, means,
            observation_uncertainty=ObservationUncertainty(
                sd/np.sqrt(3), response_kind='mean', replicate_counts=3))
        records = []
        # Future replicate allocation must not rescale completed data.
        model = ObservationModel('constant_sd', .9, replicates_per_time=7)
        for completed in (raw, averaged):
            result, calls = captured_simulation(reference(), times, model,
                existing_observations=completed, simulation_plan=plan(times, replicates=7))
            self.assertTrue(result.fit_success.all())
            self.assertEqual(len(result.metadata['additional_design_times']), 0)
            np.testing.assert_array_equal(result.parameter_estimates[0], result.parameter_estimates[1])
            records.append(calls[0][3])
        np.testing.assert_allclose(records[0].parameter_vector, records[1].parameter_vector, rtol=1e-7)
        np.testing.assert_allclose(records[0].covariance, records[1].covariance, rtol=1e-6)

    def test_cv_refit_weights_use_central_reference_but_noise_uses_selected_draw(self):
        ref = reference()
        times = np.array([0., 2., 4.])
        shared = plan(times)
        model = ObservationModel('constant_cv', .1, replicates_per_time=2)
        analysis = fit_reference_model_ensemble(ref, model_name='linear')
        result, records = captured_simulation(ref, times, model, simulation_plan=shared,
                                              reference_model=analysis)
        expected_sigma = np.tile(.1 * np.abs(analysis.central_parameters[0]
                                             + analysis.central_parameters[1]*times), 2)
        for index, (_, response, options, _) in enumerate(records):
            truth = ref.evaluate(times, draw_indices=[index])[0]
            expected = truth + .1*np.abs(truth)*shared.standard_normal_errors[index]
            np.testing.assert_array_equal(response, expected.reshape(-1))
            np.testing.assert_array_equal(options['sigma'], expected_sigma)
        self.assertTrue(result.fit_success.all())
        self.assertEqual(result.metadata['refit_cv_weighting'], 'fixed_at_central_reference')

    def test_all_zero_noise_preserves_unweighted_refitting(self):
        result, records = captured_simulation(reference(), [0., 2., 4.],
            ObservationModel('constant_sd', 0., replicates_per_time=2),
            simulation_plan=plan([0., 2., 4.]))
        self.assertTrue(result.fit_success.all())
        self.assertEqual(result.metadata['refit_objective'], 'unweighted_least_squares')
        self.assertIsNone(result.metadata['refit_observation_sigma'])
        for time, response, options, fit in records:
            self.assertNotIn('sigma', options)
            direct = fit_model_once(time, response, model_name='linear', fallback_grid_search=False)
            np.testing.assert_array_equal(fit.parameter_vector, direct.parameter_vector)

    def test_mixed_zero_and_positive_noise_is_rejected_without_dropping_rows(self):
        model = ObservationModel('time_profile_sd', profile_times=[0., 2., 4.],
                                 profile_sd=[0., .1, .2])
        with self.assertRaisesRegex(ValueError, 'positive|zero'):
            simulate_design(reference(), [0., 2., 4.], model, model_name='linear', n_simulations=2)
        completed = ExistingObservations([0.], [1.], observation_uncertainty=
            ObservationUncertainty(.1, response_kind='individual'))
        with self.assertRaisesRegex(ValueError, 'positive|zero'):
            simulate_design(reference(), [0., 2., 4.], ObservationModel('constant_sd', 0.),
                            model_name='linear', existing_observations=completed, n_simulations=2)

    def test_zero_central_cv_weights_do_not_imply_zero_measurement_noise(self):
        times = np.array([0., 2., 4.])
        ref = ReferenceTrajectoryEnsemble(
            times, [[1., 1., 1.], [-1., -1., -1.]], [True, True],
            central_curve=np.zeros(3), interpolation='linear')
        analysis = ReferenceModelEnsemble('linear', ('a', 'b'), [[1., 0.], [-1., 0.]],
            [True, True], [.5, .5], [0., 0.], times, (0., 4.))
        with self.assertRaisesRegex(ValueError, 'positive|zero'):
            simulate_design(ref, times, ObservationModel('constant_cv', .1),
                            model_name='linear', reference_model=analysis,
                            simulation_plan=plan(times, replicates=1))


if __name__ == '__main__':
    unittest.main()
