"""Weighted repeated fits use the precision of each reported observation."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from ivivc_core import ObservationUncertainty, get_model_spec, resolve_observation_uncertainty
from study_design_core import fit_model_once
from study_design_core import fitting


class WeightedStudyRefittingTests(unittest.TestCase):
    def setUp(self):
        self.time = np.array([0., 1., 2., 3., 4.])
        self.response = np.array([1.05, .75, .67, .31, .3])
        self.sigma = np.array([.05, .2, .08, .3, .1])

    def fit(self, uncertainty=None, **options):
        return fit_model_once(self.time, self.response, model_name='linear',
                              observation_uncertainty=uncertainty,
                              fallback_grid_search=False, **options)

    def test_weighted_linear_parameters_and_absolute_covariance_match_closed_form(self):
        design = np.column_stack([np.ones(len(self.time)), self.time])
        information = design.T @ (design / self.sigma[:, None]**2)
        covariance = np.linalg.inv(information)
        expected = covariance @ (design.T @ (self.response / self.sigma**2))
        result = self.fit(ObservationUncertainty(self.sigma, response_kind='individual'))
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.parameter_vector, expected, rtol=1e-6)
        np.testing.assert_allclose(result.covariance, covariance, rtol=1e-5)
        np.testing.assert_array_equal(result.observation_sigma, self.sigma)
        np.testing.assert_allclose(result.residuals, self.response - design @ expected, atol=1e-8)
        self.assertAlmostEqual(result.residual_sd, np.linalg.norm(result.residuals)/np.sqrt(3))

    def test_constant_sigma_retains_optimum_and_sets_absolute_covariance_scale(self):
        plain = self.fit()
        small = self.fit(ObservationUncertainty(.05, response_kind='individual'))
        large = self.fit(ObservationUncertainty(.1, response_kind='individual'))
        np.testing.assert_allclose(small.parameter_vector, plain.parameter_vector, atol=1e-7)
        np.testing.assert_allclose(large.parameter_vector, plain.parameter_vector, atol=1e-7)
        np.testing.assert_allclose(large.covariance, 4*small.covariance, rtol=1e-5)
        np.testing.assert_allclose(plain.covariance,
                                   small.covariance * plain.residual_sd**2/.05**2, rtol=1e-5)

    def test_sem_and_individual_sd_use_same_sigma_without_count_rescaling(self):
        individual = self.fit(ObservationUncertainty(self.sigma, response_kind='individual'))
        mean = self.fit(ObservationUncertainty(self.sigma, response_kind='mean',
                                               replicate_counts=[1, 4, 9, 16, 25]))
        resolved = resolve_observation_uncertainty(
            ObservationUncertainty(self.sigma, response_kind='mean'), len(self.time))
        already_resolved = self.fit(resolved)
        for result in (mean, already_resolved):
            np.testing.assert_allclose(result.parameter_vector, individual.parameter_vector)
            np.testing.assert_allclose(result.covariance, individual.covariance)
            self.assertEqual(result.observation_uncertainty.response_kind, 'mean')
        np.testing.assert_array_equal(mean.observation_uncertainty.replicate_counts, [1, 4, 9, 16, 25])

    def test_nonfinite_pairs_mask_uncertainty_and_counts_in_original_row_order(self):
        result = fit_model_once([0., 1., np.nan, 2., np.inf], [1., .8, 99., .6, 4.],
            model_name='linear', fallback_grid_search=False,
            observation_uncertainty=ObservationUncertainty([.1, .2, np.nan, .3, np.nan],
                response_kind='mean', replicate_counts=[4, 9, np.nan, 16, np.nan]))
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.time, [0., 1., 2.])
        np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.1, .2, .3])
        np.testing.assert_array_equal(result.observation_uncertainty.replicate_counts, [4, 9, 16])

    def test_invalid_retained_uncertainty_and_original_length_are_rejected(self):
        for sigma in (0., -.1, np.inf, [.1, .2], [.1, .2, 0., .1, .2]):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                self.fit(ObservationUncertainty(sigma, response_kind='individual'))
        with self.assertRaisesRegex(ValueError, 'original'):
            fit_model_once([0., np.nan, 2.], [1., .8, .6], model_name='linear',
                observation_uncertainty=ObservationUncertainty([.1, .2], response_kind='individual'))

    def test_candidate_selection_uses_same_weighted_objective_as_final_fit(self):
        spec = replace(get_model_spec('linear'),
                       initial_guess=lambda t, y: [[0., 0.], [10., 0.]])
        with patch.object(fitting, 'get_model_spec', return_value=spec):
            plain = fit_model_once([0., 1., 2.], [0., 0., 10.], model_name='linear',
                                   fallback_grid_search=False)
            weighted = fit_model_once([0., 1., 2.], [0., 0., 10.], model_name='linear',
                fallback_grid_search=False,
                observation_uncertainty=ObservationUncertainty([10., 10., .1], response_kind='individual'))
        np.testing.assert_array_equal(plain.initial_parameters, [0., 0.])
        np.testing.assert_array_equal(weighted.initial_parameters, [10., 0.])

    def test_requested_grid_search_and_final_fit_receive_absolute_sigma(self):
        original_fit = fitting.sp.optimize.curve_fit
        with patch.object(fitting, 'auto_grid_search', return_value=[1., -.2]) as grid, \
                patch.object(fitting.sp.optimize, 'curve_fit', wraps=original_fit) as curve_fit:
            result = self.fit(ObservationUncertainty(self.sigma, response_kind='individual'),
                              perform_grid_search=True)
        self.assertTrue(result.success)
        self.assertTrue(result.used_grid_search)
        np.testing.assert_array_equal(grid.call_args.kwargs['sigma'], self.sigma)
        np.testing.assert_array_equal(curve_fit.call_args.kwargs['sigma'], self.sigma)
        self.assertIs(curve_fit.call_args.kwargs['absolute_sigma'], True)

    def test_fallback_grid_and_retry_preserve_sigma(self):
        original_fit = fitting.sp.optimize.curve_fit
        calls = []
        def fail_first(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError('deliberate first-fit failure')
            return original_fit(*args, **kwargs)
        with patch.object(fitting, 'auto_grid_search', return_value=[1., -.2]) as grid, \
                patch.object(fitting.sp.optimize, 'curve_fit', side_effect=fail_first):
            result = fit_model_once(self.time, self.response, model_name='linear',
                observation_uncertainty=ObservationUncertainty(self.sigma, response_kind='individual'))
        self.assertTrue(result.success)
        self.assertTrue(result.used_grid_search)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(grid.call_args.kwargs['sigma'], self.sigma)
        for kwargs in calls:
            np.testing.assert_array_equal(kwargs['sigma'], self.sigma)
            self.assertIs(kwargs['absolute_sigma'], True)

    def test_failure_results_retain_resolved_precision(self):
        with patch.object(fitting.sp.optimize, 'curve_fit', side_effect=RuntimeError('failure')):
            result = self.fit(ObservationUncertainty(self.sigma, response_kind='individual'))
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.observation_uncertainty.sigma, self.sigma)
        self.assertTrue(np.isnan(result.covariance).all())
        self.assertEqual(result.exception_type, 'RuntimeError')

    def test_registered_fit_options_cannot_override_supplied_uncertainty(self):
        for overrides in ({'sigma': .1}, {'absolute_sigma': False}):
            spec = replace(get_model_spec('linear'), fit_kwargs=overrides)
            with self.subTest(overrides=overrides), \
                    patch.object(fitting, 'get_model_spec', return_value=spec), \
                    self.assertRaisesRegex(ValueError, 'override'):
                self.fit(ObservationUncertainty(.1, response_kind='individual'))

    def test_unspecified_uncertainty_keeps_legacy_optimizer_call_contract(self):
        original_fit = fitting.sp.optimize.curve_fit
        with patch.object(fitting, 'auto_grid_search', return_value=[1., -.2]) as grid, \
                patch.object(fitting.sp.optimize, 'curve_fit', wraps=original_fit) as curve_fit:
            result = self.fit(perform_grid_search=True)
        self.assertTrue(result.success)
        self.assertIsNone(result.observation_uncertainty)
        self.assertNotIn('sigma', grid.call_args.kwargs)
        self.assertNotIn('sigma', curve_fit.call_args.kwargs)
        self.assertNotIn('absolute_sigma', curve_fit.call_args.kwargs)

    def test_numeric_sigma_for_mixed_rows_matches_shared_precision_calculation(self):
        # The numeric interface can combine existing-mean SEM and future raw SD.
        numeric = self.fit(sigma=self.sigma)
        typed = self.fit(ObservationUncertainty(self.sigma, response_kind='individual'))
        self.assertTrue(numeric.success)
        np.testing.assert_allclose(numeric.parameter_vector, typed.parameter_vector)
        np.testing.assert_allclose(numeric.covariance, typed.covariance)
        np.testing.assert_array_equal(numeric.observation_sigma, self.sigma)
        self.assertIsNone(numeric.observation_uncertainty)

    def test_numeric_sigma_and_typed_uncertainty_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self.fit(ObservationUncertainty(self.sigma, response_kind='mean'), sigma=self.sigma)

    def test_numeric_sigma_follows_original_row_mask_and_validates_retained_values(self):
        time, response = [0., np.nan, 1., 2.], [1., 9., .8, .6]
        result = fit_model_once(time, response, model_name='linear', fallback_grid_search=False,
                                sigma=[.1, np.nan, .2, .3])
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.observation_sigma, [.1, .2, .3])
        for invalid in ([.1, np.nan, 0., .3], [.1, .2, .3]):
            with self.subTest(sigma=invalid), self.assertRaises(ValueError):
                fit_model_once(time, response, model_name='linear', sigma=invalid)


if __name__ == '__main__':
    unittest.main()
