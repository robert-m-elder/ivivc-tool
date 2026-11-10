"""Calibration assessment reuses the IVIVC calculations and preserves precision."""
from importlib.util import find_spec
from copy import deepcopy
import unittest
import warnings

import numpy as np

from ivivc_core import (
    ObservationUncertainty, TimeCourse, fit_ivivc, predict_in_vivo,
    prepare_ivivc_data, prepare_time_course,
)
from study_design_core import assess_ivivc_calibration


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)
GRID = np.array([.5, 1.5, 2.5])


def course(time, response, sigma=.025, *, kind='individual', counts=None):
    return prepare_time_course(
        time, response,
        observation_uncertainty=ObservationUncertainty(
            sigma, response_kind=kind, replicate_counts=counts),
    )[0]


def calibration_pair(*, kind='individual', counts=None):
    time = np.arange(5., dtype=float)
    sigma = np.array([.02, .03, .04, .03, .025])
    vitro = course(time, 1.02*np.exp(-.18*time) + [.0, .005, -.004, .006, -.003],
                   sigma, kind=kind, counts=counts)
    vivo = course(time, 1.05*np.exp(-.1*time) + [.0, -.004, .006, -.005, .003],
                  1.2*sigma, kind=kind, counts=counts)
    return vitro, vivo


def direct(vitro, vivo, prediction, *, method='value_ratio', normalizations=(),
           model='exponential', samples=64, seed=734):
    prepared = prepare_ivivc_data(vitro, vivo,
        selected_interpolation=['none'], selected_normalizations=normalizations)
    analysis = fit_ivivc(prepared, selections='approach2:' + model,
        selected_metrics=['rmse'], analysis_config=CONFIG, fail_fast=True)
    trajectory = predict_in_vivo(analysis, prediction,
        methods={'approach2': method}, n_samples=samples, random_state=seed,
        fail_fast=True)['approach2:' + model + ':' + method]
    return analysis, trajectory


class CalibrationAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.vitro, self.vivo = calibration_pair()
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def assess(self, *, vitro=None, vivo=None, **options):
        defaults = dict(model_name='exponential', evaluation_times=GRID,
                        analysis_config=CONFIG, n_samples=64, random_state=734)
        defaults.update(options)
        return assess_ivivc_calibration(
            self.vitro if vitro is None else vitro,
            self.vivo if vivo is None else vivo, **defaults)

    def test_weighted_fit_and_ratio_bounds_match_direct_ivivc_api(self):
        assessment = self.assess()
        expected_analysis, expected = direct(
            self.vitro, self.vivo, TimeCourse(GRID, np.ones(len(GRID))))
        self.assertFalse(assessment.errors)
        self.assertEqual(assessment.relationship.quantity, 'value_ratio')
        np.testing.assert_array_equal(assessment.relationship.coordinate, GRID)
        np.testing.assert_allclose(assessment.relationship.estimate, expected.in_vivo_value)
        np.testing.assert_array_equal(assessment.relationship.lower, expected.value_lower)
        np.testing.assert_array_equal(assessment.relationship.upper, expected.value_upper)
        self.assertTrue(assessment.relationship.interval_available.all())
        np.testing.assert_allclose(assessment.relationship.interval_width,
                                   expected.value_upper - expected.value_lower)
        actual_fit = assessment.analysis['approach2:exponential']
        expected_fit = expected_analysis['approach2:exponential']
        for field in ('in_vitro_fit', 'in_vivo_fit'):
            actual, reference = getattr(actual_fit, field), getattr(expected_fit, field)
            np.testing.assert_array_equal(actual.parameter_vector, reference.parameter_vector)
            np.testing.assert_array_equal(actual.covariance, reference.covariance)
        self.assertFalse(assessment.metadata['future_observation_variability_included'])

    def test_component_covariance_matches_absolute_jacobian_information(self):
        assessment = self.assess()
        fitted = assessment.analysis['approach2:exponential']
        for source, fit in ((self.vitro, fitted.in_vitro_fit),
                            (self.vivo, fitted.in_vivo_fit)):
            amplitude, rate = fit.parameter_vector
            exponential = np.exp(rate*source.time)
            jacobian = np.column_stack([exponential, amplitude*source.time*exponential])
            sigma = source.observation_uncertainty.sigma
            weighted = jacobian/sigma[:, None]
            covariance = np.linalg.inv(weighted.T @ weighted)
            np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-4)
            self.assertEqual(fit.fit_diagnostics['covariance_scaling'], 'absolute')

    def test_mean_counts_are_metadata_and_do_not_rescale_supplied_sem(self):
        first = calibration_pair(kind='mean', counts=4)
        second = calibration_pair(kind='mean', counts=[1, 9, 100, 4, 16])
        assessments = [self.assess(vitro=vitro, vivo=vivo) for vitro, vivo in (first, second)]
        for assessment in assessments:
            self.assertFalse(assessment.errors)
        np.testing.assert_array_equal(assessments[0].relationship.estimate,
                                      assessments[1].relationship.estimate)
        np.testing.assert_array_equal(assessments[0].relationship.lower,
                                      assessments[1].relationship.lower)
        np.testing.assert_array_equal(assessments[0].relationship.upper,
                                      assessments[1].relationship.upper)

    def test_optional_target_does_not_change_seeded_relationship_results(self):
        target = course([1., 2., 3.], [.85, .7, .55], .03, kind='mean', counts=4)
        without = self.assess()
        with_target = self.assess(prediction=target)
        self.assertFalse(with_target.errors)
        np.testing.assert_array_equal(without.relationship.estimate, with_target.relationship.estimate)
        np.testing.assert_array_equal(without.relationship.lower, with_target.relationship.lower)
        np.testing.assert_array_equal(without.relationship.upper, with_target.relationship.upper)
        self.assertIsNotNone(with_target.prediction)
        expected = predict_in_vivo(with_target.analysis, target,
            methods={'approach2': 'value_ratio'}, n_samples=64, random_state=734,
            fail_fast=True)['approach2:exponential:value_ratio']
        np.testing.assert_array_equal(with_target.prediction.estimate, expected.in_vivo_value)
        np.testing.assert_array_equal(with_target.prediction.lower, expected.value_lower)
        np.testing.assert_array_equal(with_target.prediction.upper, expected.value_upper)
        self.assertEqual(with_target.prediction.quantity, 'in_vivo_response')
        np.testing.assert_array_equal(with_target.prediction.coordinate, target.time)

    def test_target_input_precision_changes_bounds_without_changing_centers_or_calibration(self):
        time, values = np.array([1., 2., 3.]), np.array([.85, .7, .55])
        fixed = self.assess(prediction=TimeCourse(time, values))
        uncertain = self.assess(prediction=course(time, values, .1, kind='mean'))
        np.testing.assert_array_equal(fixed.prediction.estimate, uncertain.prediction.estimate)
        self.assertTrue(np.all(uncertain.prediction.interval_width > fixed.prediction.interval_width))
        np.testing.assert_array_equal(fixed.relationship.lower, uncertain.relationship.lower)
        np.testing.assert_array_equal(fixed.relationship.upper, uncertain.relationship.upper)

    def test_raw_calibration_and_target_are_normalized_once_without_mutation(self):
        time = np.arange(5., dtype=float)
        vitro = course(time, 10. - time, .2)
        vivo = course(time, 20. - 1.2*time, .6)
        target = course([0., 1., 2.], [50., 40., 25.], [.5, .7, .9],
                        kind='mean', counts=4)
        originals = [(s.time.copy(), s.value.copy(), s.observation_uncertainty.sigma.copy())
                     for s in (vitro, vivo, target)]
        assessment = self.assess(vitro=vitro, vivo=vivo, prediction=target,
                                selected_normalizations=['initial_value'])
        self.assertFalse(assessment.errors)
        fit = assessment.analysis['approach2:exponential']
        np.testing.assert_allclose(fit.in_vitro_fit.response, vitro.value/10.)
        np.testing.assert_allclose(fit.in_vivo_fit.response, vivo.value/20.)
        np.testing.assert_allclose(fit.in_vitro_fit.observation_uncertainty.sigma, .2/10.)
        np.testing.assert_allclose(fit.in_vivo_fit.observation_uncertainty.sigma, .6/20.)
        prepared_target, _ = prepare_time_course(target.time, target.value,
            selected_normalizations=['initial_value'],
            observation_uncertainty=target.observation_uncertainty)
        expected = predict_in_vivo(assessment.analysis, prepared_target,
            methods={'approach2': 'value_ratio'}, n_samples=64, random_state=734,
            fail_fast=True)['approach2:exponential:value_ratio']
        np.testing.assert_array_equal(assessment.prediction.estimate, expected.in_vivo_value)
        np.testing.assert_array_equal(assessment.prediction.lower, expected.value_lower)
        metadata = assessment.metadata['prediction_preprocessing']
        self.assertEqual(metadata.selected_normalizations, ('initial_value',))
        for source, (original_time, original_response, original_sigma) in zip(
                (vitro, vivo, target), originals):
            np.testing.assert_array_equal(source.time, original_time)
            np.testing.assert_array_equal(source.value, original_response)
            np.testing.assert_array_equal(source.observation_uncertainty.sigma, original_sigma)

    def test_inverse_relationship_matches_api_and_ignores_vitro_fit_precision(self):
        levels = np.array([.9, .8, .7])
        first = self.assess(method='observed_response_inverse', evaluation_times=None,
                            response_levels=levels)
        modified_vitro = course(self.vitro.time, self.vitro.value, .2)
        second = self.assess(vitro=modified_vitro,
            method='observed_response_inverse', evaluation_times=None, response_levels=levels)
        self.assertFalse(first.errors)
        self.assertEqual(first.relationship.quantity, 'in_vivo_time')
        np.testing.assert_array_equal(first.relationship.coordinate, levels)
        for attribute in ('estimate', 'lower', 'upper'):
            np.testing.assert_array_equal(getattr(first.relationship, attribute),
                                          getattr(second.relationship, attribute))
        _, expected = direct(self.vitro, self.vivo,
            TimeCourse(np.zeros(len(levels)), levels), method='observed_response_inverse')
        np.testing.assert_array_equal(first.relationship.estimate, expected.in_vivo_time)
        np.testing.assert_array_equal(first.relationship.lower, expected.time_lower)
        np.testing.assert_array_equal(first.relationship.upper, expected.time_upper)

    def test_unavailable_inverse_points_remain_distinct_from_fit_failure(self):
        assessment = self.assess(method='observed_response_inverse',
            evaluation_times=None, response_levels=[.8, 1.2])
        self.assertIsNotNone(assessment.analysis)
        self.assertIn('approach2:exponential', assessment.analysis.model_results)
        self.assertTrue(assessment.relationship.point_available[0])
        self.assertFalse(assessment.relationship.point_available[1])
        self.assertTrue(assessment.relationship.interval_available[0])
        self.assertFalse(assessment.relationship.interval_available[1])
        self.assertTrue(np.isnan(assessment.relationship.interval_width[1]))
        self.assertIn('interval_status', assessment.relationship.trajectory.auxiliary)

    def test_numerical_fit_failure_is_returned_without_a_substitute_relationship(self):
        assessment = self.assess(vitro=TimeCourse([0.], [1.]))
        self.assertIn('fit', assessment.errors)
        self.assertIsNone(assessment.relationship)
        self.assertIsNone(assessment.prediction)
        self.assertTrue(assessment.errors['fit'].message)

    def test_target_preprocessing_failure_does_not_discard_relationship(self):
        bad_target = course([0., 1.], [.8, .8], .02)
        assessment = self.assess(prediction=bad_target,
                                selected_normalizations=['min_max'])
        self.assertIsNotNone(assessment.analysis)
        self.assertIsNotNone(assessment.relationship)
        self.assertIsNone(assessment.prediction)
        self.assertIn('prediction', assessment.errors)
        self.assertNotIn('fit', assessment.errors)

    def test_calibration_preprocessing_failure_and_fail_fast(self):
        flat = course(np.arange(5.), np.ones(5), .02)
        result = self.assess(vitro=flat, selected_normalizations=['min_max'])
        self.assertIn('preprocessing', result.errors)
        self.assertIsNone(result.analysis)
        with self.assertRaises(ValueError):
            self.assess(vitro=flat, selected_normalizations=['min_max'], fail_fast=True)

    def test_target_only_assessment_does_not_require_a_relationship_grid(self):
        target = TimeCourse([1., 2.], [.85, .7])
        assessment = self.assess(evaluation_times=None, prediction=target)
        self.assertFalse(assessment.errors)
        self.assertIsNone(assessment.relationship)
        self.assertIsNotNone(assessment.prediction)

    def test_configuration_errors_are_rejected_before_assessment(self):
        invalid = [
            dict(method='not_a_method'),
            dict(model_name='linear'),
            dict(evaluation_times=None),
            dict(response_levels=[.8, .5]),
            dict(evaluation_times=[.5, np.nan]),
            dict(n_samples=1),
            dict(confidence_level=1.),
            dict(random_state='invalid'),
            dict(random_state=-1),
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                self.assess(**options)
        with self.assertRaises(TypeError):
            self.assess(vitro=(self.vitro.time, self.vitro.value))

    def test_calibration_reuse_and_invalid_inverse_settings_are_rejected_before_assessment(self):
        for target in (self.vitro, self.vivo):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'independent'):
                self.assess(prediction=target)
        invalid_inverse = [
            dict(inverse_interval_policy='not_a_policy'),
            dict(inverse_min_valid_fraction=0.),
            dict(inverse_min_valid_fraction=1.1),
        ]
        for options in invalid_inverse:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.assess(method='observed_response_inverse', evaluation_times=None,
                            response_levels=[.8], **options)
        with self.assertRaisesRegex(ValueError, 'Inverse interval settings'):
            self.assess(inverse_interval_policy='conditional')

    def test_conditional_and_strict_inverse_settings_preserve_success_counts_and_reasons(self):
        time = np.arange(5., dtype=float)
        vitro = course(time, 1.02*np.exp(-.2*time), .03)
        vivo = course(time, 1.05*np.exp(-.1*time), .03)
        options = dict(vitro=vitro, vivo=vivo, method='observed_response_inverse',
                       evaluation_times=None, response_levels=[1.04, .8],
                       inverse_min_valid_fraction=.1)
        conditional = self.assess(**options, inverse_interval_policy='conditional')
        strict = self.assess(**options, inverse_interval_policy='strict')
        self.assertFalse(conditional.errors)
        self.assertFalse(strict.errors)
        self.assertTrue(conditional.relationship.interval_available.all())
        self.assertFalse(strict.relationship.interval_available[0])
        self.assertTrue(strict.relationship.interval_available[1])
        np.testing.assert_array_equal(conditional.relationship.estimate, strict.relationship.estimate)
        conditional_metadata = conditional.relationship.trajectory.auxiliary
        strict_metadata = strict.relationship.trajectory.auxiliary
        valid = conditional_metadata['interval_valid_samples']
        self.assertGreater(valid[0], 0)
        self.assertLess(valid[0], 64)
        self.assertEqual(valid[1], 64)
        np.testing.assert_array_equal(valid, strict_metadata['interval_valid_samples'])
        np.testing.assert_array_equal(conditional_metadata['interval_conditional'], [True, False])
        self.assertEqual(conditional_metadata['interval_requested_samples'], 64)
        self.assertIn('conditional', conditional_metadata['interval_status'][0])
        self.assertIn('unavailable', strict_metadata['interval_status'][0])
        self.assertEqual(sum(conditional_metadata['inverse_failed_draw_reasons'][0].values()),
                         64-valid[0])
        self.assertEqual(conditional_metadata['inverse_failed_draw_reasons'],
                         strict_metadata['inverse_failed_draw_reasons'])

    def test_unspecified_precision_retains_direct_legacy_fit_and_covariance(self):
        vitro = TimeCourse(self.vitro.time, self.vitro.value)
        vivo = TimeCourse(self.vivo.time, self.vivo.value)
        actual = self.assess(vitro=vitro, vivo=vivo)
        expected, expected_trajectory = direct(vitro, vivo,
            TimeCourse(GRID, np.ones(len(GRID))))
        self.assertFalse(actual.errors)
        actual_fit = actual.analysis['approach2:exponential']
        expected_fit = expected['approach2:exponential']
        for field in ('in_vitro_fit', 'in_vivo_fit'):
            fitted, reference = getattr(actual_fit, field), getattr(expected_fit, field)
            self.assertIsNone(fitted.observation_uncertainty)
            self.assertEqual(fitted.fit_diagnostics['covariance_scaling'], 'residual_scaled')
            np.testing.assert_array_equal(fitted.parameter_vector, reference.parameter_vector)
            np.testing.assert_array_equal(fitted.covariance, reference.covariance)
        np.testing.assert_array_equal(actual.relationship.lower, expected_trajectory.value_lower)
        np.testing.assert_array_equal(actual.relationship.upper, expected_trajectory.value_upper)

    def test_supplied_generator_is_not_advanced_and_outputs_can_be_requested_independently(self):
        generator = np.random.default_rng(2134)
        state = deepcopy(generator.bit_generator.state)
        target = course([1., 2.], [.8, .6], .04, kind='mean')
        together = self.assess(prediction=target, random_state=generator)
        self.assertEqual(generator.bit_generator.state, state)
        separately = self.assess(prediction=target, evaluation_times=None,
                                 random_state=generator)
        self.assertEqual(generator.bit_generator.state, state)
        np.testing.assert_array_equal(together.prediction.lower, separately.prediction.lower)
        np.testing.assert_array_equal(together.prediction.upper, separately.prediction.upper)

    @unittest.skipUnless(find_spec('uncertainties'), 'requires declared uncertainties dependency')
    def test_tau_relationship_is_a_ratio_and_matches_direct_propagation(self):
        time = np.linspace(0., 5., 7)
        vitro = course(time, np.exp(-.3*time), .002)
        vivo = course(2*time, np.exp(-.3*time), .002)
        assessment = self.assess(vitro=vitro, vivo=vivo,
            model_name='exponential_normalized', method='tau_ratio', evaluation_times=None)
        self.assertFalse(assessment.errors)
        self.assertEqual(assessment.relationship.quantity, 'time_constant_ratio')
        np.testing.assert_array_equal(assessment.relationship.coordinate, [1.])
        np.testing.assert_allclose(assessment.relationship.estimate, [2.], rtol=1e-6)
        _, expected = direct(vitro, vivo, TimeCourse([1.], [1.]),
            method='tau_ratio', model='exponential_normalized')
        np.testing.assert_array_equal(assessment.relationship.lower, expected.time_lower)
        np.testing.assert_array_equal(assessment.relationship.upper, expected.time_upper)


if __name__ == '__main__':
    unittest.main()
