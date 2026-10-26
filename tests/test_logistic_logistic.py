"""Numerical contracts for the two-transition logistic response models."""
from __future__ import annotations

import unittest

import numpy as np

from models import approaches
from models.logistic_logistic import model_function as standard
from models.logistic_logistic_normalized import model_function as normalized
from ivivc_core.fitting import fit_profile
from ivivc_core.predictions import predict_inverse_response
from ivivc_core.registry import get_model_spec
from ivivc_core.types import TimeCourse
from utilities.time_constants import evaluate_tau


VARIANTS = ('logistic_logistic', 'logistic_logistic_normalized')
PARAMETERS = (0.5, 5.0, 15.0, 1.0, 2.0)


class LogisticLogisticTests(unittest.TestCase):
    def test_registry_pairs_variants_with_only_their_free_parameters(self):
        family = next(row for row in approaches['approach2']['model_families']
                      if row['id'] == VARIANTS[0])
        self.assertEqual((family['standard'], family['normalized']), VARIANTS)
        for name, names in zip(VARIANTS, (('A', 'w', 't1', 't2', 'k1', 'k2'),
                                         ('w', 't1', 't2', 'k1', 'k2'))):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                self.assertEqual(spec.parameter_names, names)
                self.assertEqual(spec.approaches, ('approach2',))
                self.assertTrue(spec.supports_tau)

    def test_matches_supplied_equation_and_exact_initial_response(self):
        time = np.linspace(0.0, 20.0, 100)
        logistic = lambda t, k, midpoint: 1.0 / (1.0 + np.exp(k * (t - midpoint)))
        expected = (0.5 * logistic(time, 1.0, 5.0) / logistic(0.0, 1.0, 5.0)
                    + 0.5 * logistic(time, 2.0, 15.0) / logistic(0.0, 2.0, 15.0))
        np.testing.assert_allclose(standard(time, 1.7, *PARAMETERS), 1.7 * expected)
        np.testing.assert_allclose(normalized(time, *PARAMETERS), expected)
        self.assertEqual(standard(0.0, 1.7, *PARAMETERS), 1.7)
        self.assertEqual(normalized(0.0, *PARAMETERS), 1.0)

    def test_component_labels_are_interchangeable_and_normalization_fixes_amplitude(self):
        time = np.linspace(0.0, 30.0, 101)
        original = normalized(time, 0.3, -2.0, 15.0, 0.7, 2.0)
        swapped = normalized(time, 0.7, 15.0, -2.0, 2.0, 0.7)
        np.testing.assert_allclose(original, swapped)
        np.testing.assert_allclose(standard(time, 1.0, 0.3, -2.0, 15.0, 0.7, 2.0), original)
        self.assertTrue(np.all(np.diff(original) <= 0.0))
        self.assertTrue(np.all((original >= 0.0) & (original <= 1.0)))

    def test_zero_and_one_weights_reduce_to_the_selected_component(self):
        time = np.array([0.0, 1.0, 5.0, 50.0])
        for weight, rate in ((0.0, 2.0), (1.0, 1.0)):
            with self.subTest(weight=weight):
                expected = 2.0 / (1.0 + np.exp(rate * time))
                np.testing.assert_allclose(normalized(time, weight, 0.0, 0.0, 1.0, 2.0), expected)

    def test_extreme_midpoints_remain_stable_and_allow_pre_zero_transitions(self):
        time = np.array([0.0, 0.1, 1.0, 10.0])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            before_zero = normalized(time, 0.3, -1e100, -1e100, 1.0, 2.0)
            after_zero = normalized(time, 0.3, 1e100, 1e100, 1.0, 2.0)
            far_tail = normalized(np.array([0.0, 1e6]), *PARAMETERS)
        np.testing.assert_allclose(before_zero, 0.3 * np.exp(-time) + 0.7 * np.exp(-2.0 * time))
        np.testing.assert_array_equal(after_zero, np.ones_like(time))
        np.testing.assert_array_equal(far_tail, [1.0, 0.0])

    def test_inactive_components_do_not_overflow_at_negative_analyzed_times(self):
        time = np.array([-1000.0, 0.0, 2.0])
        expected = (1.0 + np.exp(-0.1 * 15.0)) / (1.0 + np.exp(0.1 * (time - 15.0)))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            np.testing.assert_allclose(normalized(time, 0.0, -2000.0, 15.0, 1.0, 0.1), expected)
            np.testing.assert_allclose(normalized(time, 1.0, 15.0, -2000.0, 0.1, 1.0), expected)
            np.testing.assert_array_equal(standard(time, 0.0, 0.5, -2000.0, 15.0, 1.0, 0.1),
                                          np.zeros_like(time))
        with self.assertRaises(ValueError):
            standard(time, 0.0, 0.5, -2000.0, 15.0, -1.0, 0.1)

    def test_invalid_parameters_are_rejected_instead_of_changing_the_model(self):
        for index, value in ((0, -0.1), (0, 1.1), (1, np.nan), (2, np.inf),
                             (3, 0.0), (3, -1.0), (4, 0.0)):
            parameters = list(PARAMETERS)
            parameters[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                normalized([0.0, 1.0], *parameters)
        with self.assertRaises(ValueError):
            standard(0.0, -1.0, *PARAMETERS)
        with self.assertRaises(ValueError):
            normalized([0.0, np.nan], *PARAMETERS)

    def test_initial_guesses_remain_feasible_for_sparse_or_missing_inputs(self):
        for name in VARIANTS:
            spec = get_model_spec(name)
            lower, upper = np.asarray(spec.fit_kwargs['bounds'])
            for time, response in (([], []), ([2.0], [0.7]),
                                   ([0.0, 1.0, np.nan], [1.0, 1.0, 0.5])):
                with self.subTest(model=name, time=time):
                    starts = spec.initial_guess(time, response)
                    self.assertTrue(np.isfinite(starts).all())
                    self.assertTrue(np.all((starts >= lower) & (starts <= upper)))
                    for parameters in starts:
                        self.assertTrue(np.isfinite(spec.model_function(0.0, *parameters)))

    def test_generic_tau_matches_known_half_logistic_integrals(self):
        # For midpoint zero, L(t)/L(0) has integral 2*log(2)/k.
        expected = 2.0 * np.log(2.0) * (0.3 / 1.0 + 0.7 / 2.0)
        for function, parameters in ((standard, [1.7, 0.3, 0.0, 0.0, 1.0, 2.0]),
                                     (normalized, [0.3, 0.0, 0.0, 1.0, 2.0])):
            tau, reason = evaluate_tau(function, parameters)
            self.assertIsNone(reason)
            self.assertAlmostEqual(tau, expected, places=8)

    def test_existing_backend_fits_both_variants_from_data_based_starts(self):
        time = np.linspace(0.0, 25.0, 61)
        config = {'cv_scheme': 'none', 'grid_search_num_points': 2,
                  'grid_search_num_cores': 1, 'grid_search_param_min': -2.0,
                  'grid_search_param_max': 2.0, 'grid_search_random_state': 12345}
        for name, parameters in zip(VARIANTS, ((1.7, *PARAMETERS), PARAMETERS)):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                truth = spec.model_function(time, *parameters)
                response = truth + 0.0005 * np.sin(time)
                fit = fit_profile(time, response, model_name=name, selected_metrics=['rmse'],
                                  analysis_config=config, approach_id='approach2')
                self.assertEqual(fit.parameter_names, spec.parameter_names)
                self.assertEqual(fit.covariance.shape, (len(parameters), len(parameters)))
                self.assertTrue(np.isfinite(fit.covariance).all())
                np.testing.assert_allclose(fit.fitted_values, truth, atol=0.001)

    def test_existing_inverse_prediction_and_intervals_recover_known_times(self):
        known_time = np.array([2.3, 9.7, 16.1])
        for name, values in zip(VARIANTS, ((1.7, *PARAMETERS), PARAMETERS)):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                parameters = dict(zip(spec.parameter_names, values))
                response = spec.model_function(known_time, *values)
                prediction = predict_inverse_response(
                    TimeCourse(known_time, response), model_name=name, parameters=parameters,
                    covariance=np.zeros((len(values), len(values))), residuals=[],
                    calibration_time_range=(0.0, 20.0), n_samples=4)
                self.assertTrue(prediction.available)
                np.testing.assert_allclose(prediction.in_vivo_time, known_time, atol=1e-8)
                np.testing.assert_allclose(prediction.time_lower, known_time, atol=1e-8)
                np.testing.assert_allclose(prediction.time_upper, known_time, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
