"""Reported precision and future individual variability have separate meanings."""
import json
import unittest

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, resolve_observation_uncertainty
from study_design_core import (
    DesignConstraints, ExistingObservations, ObservationModel, ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble, evaluate_local_i_optimal_design, information_matrix,
    reweight_reference_ensemble, simulate_design,
)
from study_design_core.observation import (
    completed_observation_standard_deviation, existing_observations_from_time_course,
    observation_model_metadata,
)
from study_design_core.time_scaling_planning import _observation_agreement


def profile(**changes):
    options = dict(kind='time_profile_sd', profile_times=[0., 2., 4.],
                   profile_sd=[.1, .2, .4], replicates_per_time=3)
    options.update(changes)
    return ObservationModel(**options)


def reference():
    time = np.linspace(0., 4., 9)
    return ReferenceTrajectoryEnsemble(
        time, np.vstack([1. - .2*time, 1. - .1*time]), [True, True],
        draw_weights=[.5, .5], interpolation='linear')


class ObservationUncertaintyTests(unittest.TestCase):
    def test_existing_constant_and_cv_conventions_remain_individual_sd(self):
        constant = ObservationModel('constant_sd', .3, 9)
        cv = ObservationModel('constant_cv', .1, 4, .05)
        np.testing.assert_array_equal(constant.standard_deviation([1., 2.]), [.3, .3])
        np.testing.assert_allclose(constant.variance_of_mean([1., 2.]), [.01, .01])
        np.testing.assert_array_equal(cv.standard_deviation([-.1, 2.]), [.05, .2])
        np.testing.assert_allclose(cv.variance_of_mean([-.1, 2.]), [.05**2/4, .2**2/4])

    def test_profile_lookup_uses_fixed_times_ties_and_endpoints(self):
        model = profile(minimum_sd=.15)
        expected = np.ones((2, 4))
        sd = model.standard_deviation(expected, times=[-1., 1., 3., 10.])
        np.testing.assert_array_equal(sd, [[.15, .15, .2, .4]] * 2)
        np.testing.assert_allclose(model.variance_of_mean(expected, times=[-1., 1., 3., 10.]), sd**2/3)

    def test_profile_copies_inputs_and_records_json_safe_assumptions(self):
        times, errors = np.array([0., 2.]), np.array([.1, .3])
        model = profile(profile_times=times, profile_sd=errors)
        times[:] = 9.
        errors[:] = 9.
        np.testing.assert_array_equal(model.standard_deviation([1., 1.], times=[0., 2.]), [.1, .3])
        self.assertFalse(model.profile_times.flags.writeable)
        self.assertFalse(model.profile_sd.flags.writeable)
        record = observation_model_metadata(model)
        self.assertEqual(record['observation_model_kind'], 'time_profile_sd')
        self.assertIsNone(record['observation_model_value'])
        self.assertEqual(record['observation_model_profile']['profile_sd'], [.1, .3])
        json.dumps(record, allow_nan=False)

    def test_profile_endpoint_selection_does_not_overflow(self):
        model = profile(profile_times=[-1e308, -9e307], profile_sd=[.1, .3])
        np.testing.assert_array_equal(model.standard_deviation([1., 1.], times=[-1.7e308, 1e308]), [.1, .3])

    def test_invalid_profiles_and_missing_lookup_times_are_rejected(self):
        invalid = [
            {'profile_times': [], 'profile_sd': []},
            {'profile_times': [0., 1.], 'profile_sd': [.1]},
            {'profile_times': [0., 0.]}, {'profile_times': [2., 1., 0.]},
            {'profile_times': [0., np.inf, 2.]}, {'profile_sd': [.1, -.1, .2]},
            {'profile_sd': [.1, np.nan, .2]}, {'value': .1},
            {'profile_sd': [True, .1, .2]},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                profile(**changes)
        for times in (None, [0., np.nan], [0., 1., 2.]):
            with self.subTest(times=times), self.assertRaises((TypeError, ValueError)):
                profile().standard_deviation([1., 1.], times=times)

    def test_future_replicates_have_individual_sd_and_mean_variance_once(self):
        model = profile()
        observations = model.simulate([1., .5], times=[0., 4.],
                                      replicates_per_time=30000, random_state=42)
        np.testing.assert_allclose(np.std(observations, axis=0), [.1, .4], rtol=.025)
        means = observations.reshape(-1, 3, 2).mean(axis=1)
        np.testing.assert_allclose(np.var(means, axis=0), np.array([.1, .4])**2/3, rtol=.04)
        self.assertLess(abs(np.corrcoef(observations.T)[0, 1]), .03)

    def test_completed_mean_sem_is_already_precision_of_each_reported_row(self):
        uncertainty = ObservationUncertainty([.1, .2], response_kind='mean', replicate_counts=[4, 25])
        completed = ExistingObservations([1., 3.], [.8, .4], observation_uncertainty=uncertainty)
        self.assertEqual(completed.response_kind, 'mean')
        self.assertEqual(completed.n_observations, 2)
        np.testing.assert_array_equal(completed.observation_uncertainty.sigma, [.1, .2])
        np.testing.assert_array_equal(completed.observation_uncertainty.replicate_counts, [4, 25])
        std = completed_observation_standard_deviation(
            completed, np.ones((2, 2)), ObservationModel('constant_sd', 20., 100))
        np.testing.assert_array_equal(std, [[.1, .2], [.1, .2]])

    def test_time_course_adapter_retains_individual_sd_and_label(self):
        uncertainty = resolve_observation_uncertainty(
            ObservationUncertainty(.12, response_kind='individual'), 2)
        course = TimeCourse([1., 3.], [.8, .4], label='Measured values',
                            observation_uncertainty=uncertainty)
        completed = existing_observations_from_time_course(course)
        self.assertEqual(completed.label, course.label)
        self.assertEqual(completed.response_kind, 'individual')
        np.testing.assert_array_equal(completed.observation_uncertainty.sigma, [.12, .12])
        renamed = existing_observations_from_time_course(course, label='Completed')
        self.assertEqual(renamed.label, 'Completed')

    def test_unreported_completed_precision_uses_legacy_future_model_fallback(self):
        completed = ExistingObservations([0., 4.], [1., .2])
        self.assertEqual(completed.response_kind, 'individual')
        predicted = np.ones((2, 2))
        np.testing.assert_array_equal(
            completed_observation_standard_deviation(completed, predicted, profile()),
            profile().standard_deviation(predicted, times=completed.time))
        record = observation_model_metadata(ObservationModel('constant_sd', .1))
        self.assertEqual(record, {'observation_model_kind': 'constant_sd', 'observation_model_value': .1})

    def test_completed_uncertainty_must_align_and_remain_positive(self):
        for errors in ([.1], [.1, 0.], [.1, np.nan]):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                ExistingObservations([1., 3.], [.8, .4], observation_uncertainty=
                    ObservationUncertainty(errors, response_kind='mean'))

    def test_adaptive_likelihood_and_agreement_use_completed_sem_not_future_sd(self):
        prior = reference()
        completed = ExistingObservations([1., 3.], [.8, .4], observation_uncertainty=
            ObservationUncertainty([.1, .2], response_kind='mean', replicate_counts=[25, 4]))
        predicted = prior.evaluate(completed.time)
        residual = completed.response - predicted
        sigma = np.array([.1, .2])
        expected_log = np.sum(-.5*(residual/sigma)**2 - np.log(sigma) - .5*np.log(2*np.pi), axis=1)
        expected_weights = np.exp(expected_log - np.max(expected_log))
        expected_weights /= expected_weights.sum()
        results = [reweight_reference_ensemble(prior, completed, ObservationModel('constant_sd', value))
                   for value in (.01, 10.)]
        for update in results:
            np.testing.assert_allclose(update.log_likelihood, expected_log)
            np.testing.assert_allclose(update.posterior_weights, expected_weights)
        agreement = _observation_agreement(prior, completed, profile(), results[0])
        np.testing.assert_allclose(agreement['standardized_residual_rms_by_draw'],
                                   np.sqrt(np.mean((residual/sigma)**2, axis=1)))
        self.assertEqual(agreement['residual_scale'], 'supplied_observation_sem')

    def test_profile_information_matches_weighted_linear_information(self):
        times = np.array([0., 2., 4.])
        matrix = information_matrix('linear', times, [1., -.2], profile())
        x = np.column_stack([np.ones(3), times])
        expected = x.T @ (x * (3/np.array([.1, .2, .4])**2)[:, None])
        np.testing.assert_allclose(matrix, expected, rtol=1e-7)
        flat = profile(profile_sd=[.1, .1, .1])
        np.testing.assert_allclose(information_matrix('linear', times, [1., -.2], flat),
                                   information_matrix('linear', times, [1., -.2], ObservationModel('constant_sd', .1, 3)))

    def test_adaptive_metadata_retains_current_assumptions_only(self):
        prior = reference()
        completed = ExistingObservations([1., 3.], [.8, .4], observation_uncertainty=
            ObservationUncertainty([.1, .2], response_kind='mean'))
        profiled = reweight_reference_ensemble(prior, completed, profile()).reference
        self.assertEqual(profiled.metadata['observation_model_profile'], profile().to_metadata())
        self.assertEqual(profiled.metadata['completed_observation_uncertainty'],
                         completed.observation_uncertainty.to_metadata())
        # Use the same original curves/prior weights with copied metadata to
        # check assumption replacement without applying data a second time.
        prior.metadata = dict(profiled.metadata)
        prior.metadata.pop('adaptive_reweighting')
        plain = reweight_reference_ensemble(prior, ExistingObservations([1., 3.], [.8, .4]),
                                            ObservationModel('constant_sd', .1)).reference
        self.assertNotIn('observation_model_profile', plain.metadata)
        self.assertNotIn('completed_observation_uncertainty', plain.metadata)

    def test_optimizer_forwards_candidate_times_to_profile(self):
        time = np.linspace(0., 4., 9)
        model = ReferenceModelEnsemble('linear', ('a', 'b'), [[1., -.2]], [True], [1.],
                                     [1., -.2], time, (0., 4.))
        constraints = DesignConstraints(candidate_times=[2., 4.], fixed_times=[0.],
            min_time=0., max_time=4., min_number_of_times=2, max_number_of_times=2, replicates_per_time=3)
        score = evaluate_local_i_optimal_design([0., 4.], model, profile(), constraints)
        self.assertTrue(np.isfinite(score.criterion))

    def test_completed_precision_is_used_for_simulated_study_refits(self):
        completed = ExistingObservations([0.], [1.], observation_uncertainty=
            ObservationUncertainty(.1, response_kind='individual'))
        result = simulate_design(reference(), [0., 2., 4.], profile(), model_name='linear',
                                 existing_observations=completed, n_simulations=2, random_state=1)
        self.assertTrue(result.fit_success.all())
        self.assertEqual(result.metadata['refit_objective'], 'weighted_least_squares')
        np.testing.assert_array_equal(result.metadata['refit_observation_sigma'],
                                      [.1, .2, .4, .2, .4, .2, .4])

    def test_simulation_accepts_future_profile_without_changing_reference_curves(self):
        prior = reference()
        original = prior.value_draws.copy()
        result = simulate_design(prior, [0., 2., 4.], profile(), model_name='linear',
                                 n_simulations=8, random_state=8, fallback_grid_search=False)
        self.assertEqual(result.observed_means.shape, (8, 3))
        self.assertTrue(result.fit_success.all())
        np.testing.assert_array_equal(prior.value_draws, original)


if __name__ == '__main__':
    unittest.main()
