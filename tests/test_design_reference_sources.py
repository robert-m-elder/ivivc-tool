"""Seeded checks of the shared reference-source adapters used by the browser."""
from copy import deepcopy
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import EmpiricalTimeScaleDistribution
from utilities.design_planning import PlanningInputError
from utilities.design_reference_sources import (
    build_ivivc_reference, build_scaling_reference, reference_options,
)


def settings(**changes):
    result = {'normalization': 'none', 'one_minus': False,
              'reference_draws': 8, 'random_seed': 12345}
    result.update(changes)
    return result


def courses(*, amplitude=1., sigma=None, counts=None):
    time = np.linspace(0., 8., 9)
    response = amplitude * np.exp(-.2 * time) * np.array([1., 1.002, .998, 1.001, 1., .999, 1.001, .998, 1.])
    uncertainty = (None if sigma is None else ResolvedObservationUncertainty(
        np.full(9, sigma), response_kind='mean', replicate_counts=counts))
    return (TimeCourse(time, response, observation_uncertainty=uncertainty),
            TimeCourse(2 * time, response, observation_uncertainty=uncertainty),
            TimeCourse(time, amplitude * np.exp(-.2 * time)))


class ReferenceSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = courses()

    def ivivc(self, form=None, *, inputs=None, grid=None, options=None):
        controls = {'ivivc_approach': 'approach2', 'ivivc_model': 'exponential_normalized',
                    'ivivc_method': 'value_ratio'}
        controls.update(form or {})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return build_ivivc_reference(
                *(inputs or self.data), controls, options or settings(),
                np.linspace(1., 4., 13) if grid is None else grid)

    def scaling(self, *, source=None, distribution=None, form=None, options=None, grid=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return build_scaling_reference(
                source or self.data[0],
                distribution or EmpiricalTimeScaleDistribution([2., 3.], weights=[1., 3.]),
                form or {'scaling_model': 'exponential_normalized'}, options or settings(),
                np.linspace(0., 4., 13) if grid is None else grid)

    def test_catalog_reuses_registered_models_and_actual_prediction_methods(self):
        options = reference_options()
        approaches = {item['id']: item for item in options['approaches']}
        self.assertEqual(set(approaches), {'approach1', 'approach2', 'approach3'})
        self.assertEqual(approaches['approach3']['models'], [])
        self.assertEqual({item['id'] for item in approaches['approach2']['methods']},
                         {'value_ratio', 'tau_ratio', 'observed_response_inverse'})
        self.assertIn('linear', {item['id'] for item in options['models']})

    def test_all_approach2_methods_generate_the_known_slowed_reference(self):
        for method in ('value_ratio', 'tau_ratio', 'observed_response_inverse'):
            with self.subTest(method=method):
                result = self.ivivc({'ivivc_method': method})
                np.testing.assert_allclose(result.reference.central_curve,
                                           np.exp(-.1 * result.reference.time_grid), atol=.002)
                self.assertEqual(result.reference.n_draws, 8)
                self.assertFalse(result.reference.metadata['future_observation_variability_included'])
                self.assertEqual(result.details['selection'], 'approach2:exponential_normalized')

    def test_approach1_uses_the_shared_time_mapping_and_fitted_uncertainty(self):
        result = self.ivivc({'ivivc_approach': 'approach1', 'ivivc_model': 'linear',
                             'ivivc_method': 'time_mapping'})
        self.assertIsNotNone(result.analysis.model_results['approach1:linear'].time_mapping_fit)
        np.testing.assert_allclose(result.reference.central_curve,
                                   np.exp(-.1 * result.reference.time_grid), atol=.002)
        self.assertTrue(any('Experimental response SD/SEM is not propagated' in text for text in result.warnings))

    def test_both_approach3_methods_are_explicitly_deterministic(self):
        for method in ('value_scaling', 'time_scaling'):
            with self.subTest(method=method):
                result = self.ivivc({'ivivc_approach': 'approach3', 'ivivc_model': 'inactive-junk',
                                     'ivivc_method': method})
                self.assertEqual(result.reference.n_draws, 1)
                self.assertTrue(result.details['deterministic'])
                self.assertTrue(any('one deterministic' in message for message in result.warnings))

    def test_method_model_and_interpolation_choices_are_validated(self):
        for form, field in [({'ivivc_approach': 'bad'}, 'ivivc_approach'),
                            ({'ivivc_model': 'linear'}, 'ivivc_model'),
                            ({'ivivc_method': 'bad'}, 'ivivc_method'),
                            ({'ivivc_approach': 'approach1', 'ivivc_model': 'linear',
                              'ivivc_method': 'time_mapping', 'ivivc_interpolation': 'bad'}, 'ivivc_interpolation')]:
            with self.subTest(form=form), self.assertRaises(PlanningInputError) as caught:
                self.ivivc(form)
            self.assertEqual(caught.exception.field_id, field)

    def test_inactive_interpolation_is_ignored_for_approach2(self):
        result = self.ivivc({'ivivc_interpolation': 'inactive-invalid'})
        self.assertEqual(result.preprocessing.interpolation, 'none')

    def test_mean_sem_with_counts_is_used_without_another_division(self):
        first = self.ivivc(inputs=courses(sigma=.03))
        second = self.ivivc(inputs=courses(sigma=.03, counts=np.full(9, 16)))
        for label in first.source_fits:
            np.testing.assert_allclose(first.source_fits[label].covariance,
                                       second.source_fits[label].covariance)
        np.testing.assert_allclose(second.source_fits['In vivo calibration'].observation_uncertainty.sigma, .03)

    def test_target_input_precision_propagates_without_future_measurement_noise(self):
        data = courses(sigma=.001)
        baseline = self.ivivc(inputs=data)
        target = TimeCourse(data[2].time, data[2].value,
                            observation_uncertainty=ResolvedObservationUncertainty(np.full(9, .1), response_kind='individual'))
        uncertain = self.ivivc(inputs=(data[0], data[1], target))
        self.assertGreater(float(np.mean(np.var(uncertain.reference.value_draws, axis=0))),
                           100 * float(np.mean(np.var(baseline.reference.value_draws, axis=0))))
        self.assertFalse(uncertain.reference.metadata['future_observation_variability_included'])

    def test_each_ivivc_dataset_is_normalized_once_and_sources_are_unchanged(self):
        original = courses(amplitude=10., sigma=.2, counts=np.full(9, 4))
        result = self.ivivc(inputs=original, options=settings(normalization='initial_value'))
        np.testing.assert_allclose(result.analysis.prepared_data.in_vitro.value[0], 1.)
        np.testing.assert_allclose(result.source_fits['In vitro calibration'].observation_uncertainty.sigma, .02)
        self.assertEqual(result.details['source_curves'][2]['value'][0], 1.)
        self.assertEqual(original[0].value[0], 10.)
        np.testing.assert_allclose(original[0].observation_uncertainty.sigma, .2)

    def test_precision_for_unsupported_approaches_is_not_silently_dropped(self):
        uncertain = courses(sigma=.03)
        for approach, method in (('approach1', 'time_mapping'), ('approach3', 'value_scaling')):
            form = {'ivivc_approach': approach, 'ivivc_method': method, 'ivivc_model': 'linear'}
            with self.subTest(approach=approach), self.assertRaises(PlanningInputError) as caught:
                self.ivivc(form, inputs=uncertain)
            self.assertEqual(caught.exception.field_id, 'ivivc-calibration-file')
            target = TimeCourse(self.data[2].time, self.data[2].value,
                                observation_uncertainty=uncertain[0].observation_uncertainty)
            with self.assertRaises(PlanningInputError) as caught:
                self.ivivc(form, inputs=(self.data[0], self.data[1], target))
            self.assertEqual(caught.exception.field_id, 'prediction-file')

    def test_unsupported_requested_horizon_is_not_shortened_or_moved(self):
        with self.assertRaises(PlanningInputError) as caught:
            self.ivivc(grid=np.linspace(0., 10., 21))
        self.assertEqual(caught.exception.field_id, 'last_time')
        self.assertIn('supports in vivo times', str(caught.exception))
        self.assertIn('No baseline or requested time was moved', str(caught.exception))

    def test_seeded_reference_draws_repeat_exactly(self):
        first, second = self.ivivc(), self.ivivc()
        np.testing.assert_array_equal(first.reference.value_draws, second.reference.value_draws)
        np.testing.assert_array_equal(first.reference.draw_weights, second.reference.draw_weights)

    def test_scaling_preserves_empirical_ratio_weights_in_the_parameter_product(self):
        distribution = EmpiricalTimeScaleDistribution([2., 2., 3., 999.], weights=[1., 1., 2., 0.],
                                                     source_ids=['A', 'A2', 'B', 'Ignored'])
        result = self.scaling(distribution=distribution)
        self.assertEqual(result.reference.n_draws, 3 * 8)
        metadata = result.reference.metadata
        mass = np.bincount(metadata['draw_ratio_indices'], weights=result.reference.draw_weights,
                           minlength=4)
        np.testing.assert_allclose(mass, [.25, .25, .5, 0.])
        self.assertTrue(metadata['source_fit_and_scale_independent'])
        self.assertFalse(metadata['future_observation_variability_included'])
        self.assertEqual(result.details['source_ids'], distribution.source_ids)

    def test_scaling_normalizes_source_precision_once_and_retains_mean_sem(self):
        source = courses(amplitude=10., sigma=.2, counts=np.full(9, 9))[0]
        result = self.scaling(source=source, options=settings(normalization='initial_value'))
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma, .02)
        self.assertEqual(result.fit.observation_uncertainty.response_kind, 'mean')
        np.testing.assert_allclose(source.value[0], 10.)

    def test_scaling_source_and_study_models_are_separate_selections(self):
        result = self.scaling(options=settings(model_name='linear'))
        self.assertEqual(result.fit.model_name, 'exponential_normalized')
        self.assertEqual(result.details['rows'][0][1], 'exponential_normalized')

    def test_scaling_requires_explicit_extrapolation_without_dropping_ratio_mass(self):
        distribution = EmpiricalTimeScaleDistribution([.5, 2.], weights=[1., 3.])
        with self.assertRaises(PlanningInputError) as caught:
            self.scaling(distribution=distribution, grid=np.linspace(0., 8., 21))
        self.assertEqual(caught.exception.field_id, 'scaling_allow_extrapolation')
        result = self.scaling(distribution=distribution, grid=np.linspace(0., 8., 21),
                              form={'scaling_model': 'exponential_normalized',
                                    'scaling_allow_extrapolation': 'on'})
        self.assertTrue(np.any(result.reference.metadata['extrapolated_points_by_ratio']))
        self.assertTrue(any('explicitly enabled' in text for text in result.warnings))
        np.testing.assert_allclose(np.bincount(result.reference.metadata['draw_ratio_indices'],
                                   weights=result.reference.draw_weights), [.25, .75])

    def test_scaling_workload_limit_is_checked_before_source_fitting(self):
        with patch('utilities.design_reference_sources.fit_reference_profile') as fit:
            with self.assertRaises(PlanningInputError) as caught:
                self.scaling(distribution=EmpiricalTimeScaleDistribution(np.ones(251)))
            self.assertEqual(caught.exception.field_id, 'reference_draws')
            fit.assert_not_called()

    def test_scaling_requires_usable_covariance_instead_of_a_fixed_curve_fallback(self):
        fit = deepcopy(self.scaling().fit)
        fit.covariance[:] = np.nan
        with patch('utilities.design_reference_sources.fit_reference_profile', return_value=fit):
            with self.assertRaises(PlanningInputError) as caught:
                self.scaling()
        self.assertEqual(caught.exception.field_id, 'scaling_model')
        self.assertIn('covariance is unavailable', str(caught.exception))

    def test_preview_details_include_fitted_curves_and_parameter_uncertainty(self):
        result = self.ivivc()
        self.assertEqual(len(result.details['source_curves']), 3)
        self.assertEqual(len(result.details['source_curves'][0]['fitted_time']), 101)
        self.assertEqual(result.details['source_curves'][0]['environment'], 'in_vitro')
        self.assertEqual(result.details['fit_rows'][0]['label'], 'In vitro calibration')
        self.assertIsNotNone(result.details['fit_rows'][0]['se'])


if __name__ == '__main__':
    unittest.main()
