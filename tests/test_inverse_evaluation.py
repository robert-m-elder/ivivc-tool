"""Regression checks for the inverse mapper's evaluation shortcuts."""
from __future__ import annotations

import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ivivc_core import inverse


class InverseEvaluationTests(unittest.TestCase):
    def test_exact_and_solved_roots_evaluate_the_center_once(self):
        for root in (0.5, 0.314159):
            with self.subTest(root=root):
                evaluated = []

                def model(time):
                    evaluated.append(np.asarray(time).copy())
                    return 1.0 - time

                with patch.object(inverse, 'brentq', return_value=root):
                    mapped, status, reason = inverse.inverse_response_times(
                        model, {}, [1.0 - root], (0.0, 1.0), grid_points=101,
                    )
                np.testing.assert_array_equal(mapped, [root])
                self.assertEqual(status.tolist(), ['mapped'])
                self.assertIsNone(reason)
                center_calls = sum(time.shape == (1,) and time[0] == root
                                   for time in evaluated)
                self.assertEqual(center_calls, 1)
                self.assertTrue(any(time.shape == (2,) for time in evaluated))

    def test_matching_shape_outputs_are_copied_before_the_next_evaluation(self):
        buffer = np.empty(101)

        def model(time):
            buffer[:time.size] = 1.0 - time
            return buffer[:time.size]

        mapped, status, reason = inverse.inverse_response_times(
            model, {}, [0.75, 0.25, 0.5], (0.0, 1.0), grid_points=101,
        )
        np.testing.assert_allclose(mapped, [0.25, 0.75, 0.5], atol=1e-12)
        self.assertEqual(status.tolist(), ['mapped'] * 3)
        self.assertIsNone(reason)

    def test_scalar_constant_output_keeps_its_unavailability_reason(self):
        mapped, status, reason = inverse.inverse_response_times(
            lambda time: 1.0, {}, [1.0, 0.5], (0.0, 1.0), grid_points=101,
        )
        self.assertTrue(np.isnan(mapped).all())
        self.assertEqual(status.tolist(), ['unavailable', 'unavailable'])
        self.assertIn('nonconstant', reason)

    def test_warning_and_error_settings_are_restored_on_all_exit_paths(self):
        for outcome in ('mapped', 'flat', 'nonfinite', 'exception'):
            with self.subTest(outcome=outcome):
                def model(time):
                    warnings.warn('model warning', RuntimeWarning)
                    if outcome == 'exception':
                        raise ValueError('model evaluation failed')
                    if outcome == 'flat':
                        return np.ones_like(time)
                    if outcome == 'nonfinite':
                        return np.full_like(time, np.nan)
                    return np.exp(-time)

                with warnings.catch_warnings(record=True) as caught, np.errstate(all='raise'):
                    warnings.simplefilter('always', RuntimeWarning)
                    filters = list(warnings.filters)
                    error_settings = np.geterr()
                    mapped, status, reason = inverse.inverse_response_times(
                        model, {}, [0.5], (0.0, 1.0), grid_points=101,
                    )
                    self.assertEqual(warnings.filters, filters)
                    self.assertEqual(np.geterr(), error_settings)
                    self.assertEqual(len(caught), 0)
                    warnings.warn('after inverse', RuntimeWarning)
                    self.assertEqual(len(caught), 1)
                if outcome == 'mapped':
                    self.assertIsNone(reason)
                    self.assertTrue(np.isfinite(mapped).all())
                else:
                    self.assertIsNotNone(reason)

    def test_root_response_tolerance_is_still_checked(self):
        with patch.object(inverse, 'brentq', return_value=0.4):
            mapped, status, reason = inverse.inverse_response_times(
                lambda time: 1.0 - time, {}, [0.555], (0.0, 1.0), grid_points=101,
            )
        self.assertTrue(np.isnan(mapped).all())
        self.assertEqual(status.tolist(), ['inverse root did not satisfy response tolerance'])
        self.assertIsNone(reason)

    def test_reused_root_response_still_rejects_a_flat_inverse(self):
        def model(time):
            return np.where(time < 0.4, 0.6 - time,
                            np.where(time <= 0.6, 0.2, 0.8 - time))

        with patch.object(inverse, 'brentq', return_value=0.4):
            mapped, status, reason = inverse.inverse_response_times(
                model, {}, [np.nextafter(0.2, np.inf)], (0.0, 1.0), grid_points=101,
            )
        self.assertTrue(np.isnan(mapped).all())
        self.assertEqual(status.tolist(), ['nonunique or numerically flat inverse'])
        self.assertIsNone(reason)


if __name__ == '__main__':
    unittest.main()
