"""Normalized model contracts and ordered fractional-response reversal."""
from __future__ import annotations

import unittest

import numpy as np

from models import approaches, models
from ivivc_core.preprocessing import prepare_ivivc_data, prepare_time_course
from ivivc_core.registry import get_model_spec
from ivivc_core.types import TimeCourse


class NormalizedModelTests(unittest.TestCase):
    def test_family_rows_preserve_all_existing_model_identifiers(self):
        families = approaches['approach2']['model_families']
        listed = [family[variant] for family in families
                  for variant in ('standard', 'normalized') if family[variant]]
        self.assertCountEqual(listed, approaches['approach2']['models'])
        self.assertEqual(len(listed), len(set(listed)))
        for family in families:
            if family['standard']:
                self.assertEqual(family['standard'], family['id'])
            if family['normalized']:
                self.assertEqual(family['normalized'], family['id'] + '_normalized')

    def test_exponential_variants_have_only_their_free_parameters(self):
        for name, parameters in (
            ('exponential_normalized', ('b',)),
            ('double_exponential_normalized', ('w', 'b', 'd')),
            ('stretched_exponential_normalized', ('b', 'c')),
        ):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                self.assertEqual(spec.parameter_names, parameters)
                self.assertEqual(spec.approaches, ('approach2',))
                self.assertNotIn(name, approaches['approach1']['models'])

    def test_normalized_formulas_and_time_zero_constraints(self):
        x = np.array([0.0, 1.0, 5.0])
        np.testing.assert_allclose(
            models['exponential_normalized']['model_function'](x, -0.2),
            np.exp(-0.2 * x),
        )
        np.testing.assert_allclose(
            models['stretched_exponential_normalized']['model_function'](x, -0.2, 1.5),
            np.exp(-0.2 * x ** 1.5),
        )
        double = models['double_exponential_normalized']['model_function'](x, 0.3, -0.2, -0.05)
        np.testing.assert_allclose(double, 0.3 * np.exp(-0.2 * x) + 0.7 * np.exp(-0.05 * x))
        self.assertEqual(double[0], 1.0)

    def test_normalized_models_fit_without_redundant_amplitude_parameters(self):
        x = np.linspace(0.1, 30.0, 30)
        for name, parameters, start in (
            ('exponential_normalized', [-0.2], [-0.1]),
            ('double_exponential_normalized', [0.4, -0.3, -0.03], [0.5, -0.25, -0.05]),
            ('stretched_exponential_normalized', [-0.1, 1.3], [-0.12, 1.1]),
        ):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                y = spec.model_function(x, *parameters)
                fit = spec.fit_function(x, y, p0=start)
                self.assertEqual(len(fit['params']), len(parameters))
                self.assertEqual(fit['pcov'].shape, (len(parameters), len(parameters)))
                self.assertTrue(np.isfinite(fit['pcov']).all())
                np.testing.assert_allclose(fit['predict'](x), y, rtol=1e-5, atol=1e-8)


class ResponseReversalTests(unittest.TestCase):
    def test_fraction_lost_can_be_reversed_without_other_normalization(self):
        values = np.array([0.0, 0.25, 1.0])
        course, metadata = prepare_time_course([0.0, 1.0, 2.0], values,
                                               selected_normalizations=['one_minus'])
        np.testing.assert_array_equal(course.value, [1.0, 0.75, 0.0])
        np.testing.assert_array_equal(values, [0.0, 0.25, 1.0])
        self.assertEqual(metadata.normalizations[0].in_vitro_parameters,
                         {'reference': 1.0, 'applied': True})

    def test_reversal_runs_once_after_normalization_regardless_of_selection_order(self):
        course, metadata = prepare_time_course([0.0, 1.0, 2.0], [0.0, 25.0, 100.0],
            selected_normalizations=['one_minus', 'min_max', 'one_minus'])
        np.testing.assert_allclose(course.value, [1.0, 0.75, 0.0])
        self.assertEqual(metadata.selected_normalizations, ('min_max', 'one_minus'))
        self.assertEqual(tuple(record.name for record in metadata.normalizations),
                         ('min_max', 'one_minus'))

    def test_both_calibration_profiles_are_reversed_before_interpolation(self):
        prepared = prepare_ivivc_data(
            TimeCourse([0.0, 1.0, 2.0], [0.0, 0.25, 0.75]),
            TimeCourse([0.0, 2.0, 4.0], [0.0, 0.5, 0.75]),
            selected_normalizations=['one_minus'], selected_interpolation=['default'],
        )
        np.testing.assert_allclose(prepared.in_vitro.value, [1.0, 0.75, 0.25])
        np.testing.assert_allclose(prepared.in_vivo.value, [1.0, 0.5, 0.25])
        np.testing.assert_allclose(prepared.in_vitro_time_at_response, [0.0, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(prepared.in_vivo_time_at_response, [0.0, 1.0, 2.0, 4.0])
        self.assertEqual(prepared.preprocessing.normalizations[0].in_vivo_parameters['reference'], 1.0)

    def test_reversal_precedes_logarithms_and_uses_existing_zero_fallback(self):
        course, metadata = prepare_time_course([0.0, 1.0], [0.25, 0.5],
            selected_normalizations=['one_minus'], selected_scalings=['log_y'])
        np.testing.assert_allclose(course.value, np.log10([0.75, 0.5]))
        self.assertEqual(metadata.applied_scalings, ('log_y',))
        course, metadata = prepare_time_course([0.0, 1.0], [0.0, 1.0],
            selected_normalizations=['one_minus'], selected_scalings=['log_y'],
            fallback_invalid_log_scaling=True)
        np.testing.assert_array_equal(course.value, [1.0, 0.0])
        self.assertEqual(metadata.applied_scalings, ())
        self.assertEqual(len(metadata.warnings), 1)

    def test_reversal_preserves_values_outside_fraction_range_without_clipping(self):
        course, _ = prepare_time_course([0.0, 1.0], [-0.2, 1.2],
                                       selected_normalizations=['one_minus'])
        np.testing.assert_allclose(course.value, [1.2, -0.2])


if __name__ == '__main__':
    unittest.main()
