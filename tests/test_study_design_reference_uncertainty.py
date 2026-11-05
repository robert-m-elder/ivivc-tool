"""Reference curves retain source precision without adding measurement noise."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, fit_time_course, prepare_time_course
from study_design_core import (
    EmpiricalTimeScaleDistribution, ExistingObservations, ObservationModel,
    fit_reference_profile, reference_from_profile_fit, reference_from_time_scaling,
    reweight_time_scaled_reference,
)


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)


class ReferenceUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(5., dtype=float)
        self.response = np.array([1.05, .75, .67, .31, .3])
        self.sigma = np.array([.05, .2, .08, .3, .1])

    def course(self, sigma=None, **uncertainty_options):
        return prepare_time_course(self.time, self.response, label='Measured source',
            observation_uncertainty=ObservationUncertainty(
                self.sigma if sigma is None else sigma,
                response_kind=uncertainty_options.pop('response_kind', 'individual'),
                **uncertainty_options))[0]

    def fit(self, source=None):
        return fit_reference_profile(self.course() if source is None else source,
                                     model_name='linear', analysis_config=CONFIG)

    def test_typed_source_fit_matches_absolute_weighted_linear_solution(self):
        fit = self.fit()
        matrix = np.column_stack([np.ones(5), self.time])
        information = matrix.T @ (matrix / self.sigma[:, None]**2)
        covariance = np.linalg.inv(information)
        expected = covariance @ (matrix.T @ (self.response / self.sigma**2))
        np.testing.assert_allclose(fit.parameter_vector, expected, rtol=1e-6)
        np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-5)
        np.testing.assert_array_equal(fit.observation_uncertainty.sigma, self.sigma)
        self.assertEqual(fit.fit_diagnostics['covariance_scaling'], 'absolute')
        self.assertEqual(fit.fit_diagnostics['source_label'], 'Measured source')

    def test_completed_means_use_sem_without_dividing_by_counts_again(self):
        individual = self.fit()
        for counts in ([1, 4, 9, 16, 25], [100, 100, 100, 100, 100]):
            source = ExistingObservations(self.time, self.response,
                observation_uncertainty=ObservationUncertainty(
                    self.sigma, response_kind='mean', replicate_counts=counts))
            fit = self.fit(source)
            np.testing.assert_allclose(fit.parameter_vector, individual.parameter_vector)
            np.testing.assert_allclose(fit.covariance, individual.covariance)
            np.testing.assert_array_equal(fit.observation_uncertainty.replicate_counts, counts)
            self.assertEqual(fit.observation_uncertainty.uncertainty_kind, 'sem')

    def test_prepared_normalization_transports_precision_only_once(self):
        source, _ = prepare_time_course(self.time, 10*self.response,
            selected_normalizations=['initial_value'],
            observation_uncertainty=ObservationUncertainty(
                10*self.sigma, response_kind='mean', replicate_counts=4))
        fit = self.fit(source)
        np.testing.assert_allclose(fit.response, self.response / self.response[0])
        np.testing.assert_allclose(fit.observation_uncertainty.sigma,
                                   self.sigma / self.response[0])
        reference = reference_from_profile_fit(fit, self.time, n_samples=8, random_state=1)
        np.testing.assert_allclose(
            reference.metadata['source_observation_uncertainty']['standard_uncertainty'],
            source.observation_uncertainty.sigma)
        self.assertEqual(reference.metadata['source_observation_uncertainty']['response_kind'], 'mean')

    def test_unspecified_precision_retains_shared_fitter_behavior(self):
        source = TimeCourse(self.time, self.response)
        direct = fit_time_course(self.time, self.response, model_names='linear',
                                 selected_metrics=['rmse'], analysis_config=CONFIG)['linear']
        fit = self.fit(source)
        np.testing.assert_array_equal(fit.parameter_vector, direct.parameter_vector)
        np.testing.assert_array_equal(fit.covariance, direct.covariance)
        self.assertIsNone(fit.observation_uncertainty)
        reference = reference_from_profile_fit(fit, self.time, n_samples=8, random_state=1)
        self.assertIsNone(reference.metadata['source_observation_uncertainty'])
        self.assertEqual(reference.metadata['source_covariance_scaling'], 'residual_scaled')

    def test_profile_draws_are_complete_joint_curves_without_another_noise_layer(self):
        fit = self.fit()
        grid = np.linspace(0., 4., 17)
        reference = reference_from_profile_fit(fit, grid, n_samples=64, random_state=7)
        parameters = reference.metadata['parameter_samples']
        expected = parameters[:, 0, None] + parameters[:, 1, None]*grid
        np.testing.assert_allclose(reference.value_draws, expected, atol=1e-14)
        np.testing.assert_allclose(reference.central_curve,
                                   fit.parameters['a'] + fit.parameters['b']*grid)
        self.assertTrue(reference.metadata['source_parameter_uncertainty_propagated'])
        self.assertFalse(reference.metadata['future_observation_variability_included'])
        self.assertEqual(reference.metadata['source_covariance_scaling'], 'absolute')
        reference.metadata['source_fit_diagnostics']['source_label'] = 'Modified output'
        self.assertEqual(fit.fit_diagnostics['source_label'], 'Measured source')
        reference.metadata['source_observation_uncertainty']['standard_uncertainty'][0] = 99.
        self.assertEqual(fit.observation_uncertainty.sigma[0], self.sigma[0])

    def test_scaled_draws_reuse_joint_parameters_and_preserve_ratio_weights(self):
        scales = EmpiricalTimeScaleDistribution([1., 2., 3.], weights=[.2, .3, .5])
        grid = np.linspace(0., 4., 9)
        references = [reference_from_time_scaling(self.fit(self.course(multiplier*self.sigma)),
            scales, grid, n_parameter_draws=48, random_state=17) for multiplier in (1., 2.)]
        for reference in references:
            metadata = reference.metadata
            parameters = metadata['parameter_samples']
            ratio_index = metadata['draw_ratio_indices']
            parameter_index = metadata['draw_parameter_indices']
            expected = (parameters[parameter_index, 0, None]
                        + parameters[parameter_index, 1, None]*grid/scales.ratios[ratio_index, None])
            np.testing.assert_allclose(reference.value_draws, expected, atol=1e-14)
            np.testing.assert_allclose(np.bincount(ratio_index, weights=reference.draw_weights),
                                       scales.weights)
            np.testing.assert_allclose(np.bincount(parameter_index, weights=reference.draw_weights),
                                       np.full(48, 1/48))
            self.assertTrue(metadata['source_parameter_uncertainty_propagated'])
            self.assertFalse(metadata['future_observation_variability_included'])
        # Identical data and random normals isolate the supplied absolute SD:
        # doubling it doubles each parameter perturbation, not ratio variability.
        nominal = self.fit().parameter_vector
        np.testing.assert_allclose(references[1].metadata['parameter_samples'] - nominal,
            2*(references[0].metadata['parameter_samples'] - nominal), atol=1e-7, rtol=1e-5)

    def test_fixed_source_mode_retains_weighted_fit_but_does_not_require_covariance(self):
        fit = self.fit()
        fixed = replace(fit, covariance=np.full((2, 2), np.nan))
        reference = reference_from_time_scaling(fixed, [1., 2.], self.time)
        expected = fit.parameters['a'] + fit.parameters['b']*self.time / np.array([1., 2.])[:, None]
        np.testing.assert_allclose(reference.value_draws, expected)
        self.assertFalse(reference.metadata['source_parameter_uncertainty_propagated'])
        self.assertIsNone(reference.metadata['source_covariance'])
        self.assertEqual(reference.metadata['source_covariance_scaling'], 'absolute')
        self.assertIsNotNone(reference.metadata['source_observation_uncertainty'])

    def test_uncertain_raw_time_course_requires_an_explicit_fitted_source(self):
        with self.assertRaisesRegex(ValueError, 'fit_reference_profile'):
            reference_from_time_scaling(self.course(), [1., 2.], self.time)
        raw = TimeCourse(self.time, self.response)
        reference = reference_from_time_scaling(raw, [1.], self.time)
        np.testing.assert_array_equal(reference.value_draws[0], self.response)
        self.assertFalse(reference.metadata['source_parameter_uncertainty_propagated'])

    def test_adaptive_scale_update_preserves_source_precision_and_curves(self):
        prior = reference_from_time_scaling(self.fit(), [1., 2.], self.time,
                                          n_parameter_draws=16, random_state=13)
        completed = ExistingObservations([1., 3.], [.9, .6],
            observation_uncertainty=ObservationUncertainty(
                [.08, .1], response_kind='mean', replicate_counts=[4, 25]))
        result = reweight_time_scaled_reference(prior, completed,
                                                ObservationModel('constant_sd', .5, 10))
        np.testing.assert_array_equal(result.reference.value_draws, prior.value_draws)
        self.assertEqual(result.reference.metadata['source_observation_uncertainty'],
                         prior.metadata['source_observation_uncertainty'])
        self.assertEqual(result.reference.metadata['completed_observation_uncertainty'],
                         completed.observation_uncertainty.to_metadata())

    def test_source_adapter_requires_a_typed_observation_container(self):
        with self.assertRaises(TypeError):
            self.fit((self.time, self.response))

    def test_existing_fit_like_objects_may_omit_optional_precision_metadata(self):
        fit = SimpleNamespace(model_name='linear', parameters={'a': 1., 'b': -.2},
                              covariance=np.zeros((2, 2)))
        reference = reference_from_profile_fit(fit, self.time, n_samples=4, random_state=1)
        np.testing.assert_allclose(reference.value_draws,
                                   np.tile(1. - .2*self.time, (4, 1)))
        self.assertIsNone(reference.metadata['source_observation_uncertainty'])
        self.assertEqual(reference.metadata['source_covariance_scaling'], 'unspecified')


if __name__ == '__main__':
    unittest.main()
