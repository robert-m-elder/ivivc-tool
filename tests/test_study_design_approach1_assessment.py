"""Approach 1 assessment follows the shared IVIVC numerical pipeline."""
from copy import deepcopy
import unittest
import warnings

import numpy as np

from ivivc_core import (
    ObservationUncertainty, TimeCourse, fit_ivivc, predict_in_vivo,
    prepare_ivivc_data, prepare_time_course,
)
from study_design_core import assess_ivivc_calibration
from study_design_core.calibration_performance import IVIVCAssessmentRequest
from study_design_core.calibration_simulation import compare_calibration_designs, CalibrationDesign


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)
GRID = np.array([.5, 1.5, 3., 5.])
KEY = 'approach1:linear'
SAMPLES = 64
SEED = 734


def numerical(course):
    return TimeCourse(course.time, course.value, label=course.label)


def noisy_pair():
    # Duplicate times, a local response reversal, and a response plateau are
    # deliberately left intact: their handling belongs to shared preprocessing.
    times = [0., 1., 1., 2., 3., 3., 4.]
    return (
        TimeCourse(times, [1., .84, .87, .69, .52, .52, .38], label='vitro'),
        TimeCourse(times, [1., .93, .90, .81, .79, .74, .64], label='vivo'),
    )


def with_precision(course, sigma, *, kind='individual', counts=None):
    return prepare_time_course(
        course.time, course.value, label=course.label,
        observation_uncertainty=ObservationUncertainty(
            sigma, response_kind=kind, replicate_counts=counts),
    )[0]


def direct(vitro, vivo, target, *, interpolation='default', normalizations=(),
           scalings=(), fallback=False, seed=SEED):
    """Use the same shared calls as an ordinary, unweighted IVIVC analysis."""
    prepared = prepare_ivivc_data(
        numerical(vitro), numerical(vivo),
        selected_interpolation=[interpolation],
        selected_normalizations=normalizations, selected_scalings=scalings,
        fallback_invalid_log_scaling=fallback,
    )
    analysis = fit_ivivc(
        prepared, selections=KEY, selected_metrics=['rmse'],
        analysis_config=CONFIG, fail_fast=True,
    )
    trajectory = predict_in_vivo(
        analysis, target, methods={'approach1': 'time_mapping'},
        n_samples=SAMPLES, random_state=deepcopy(seed), fail_fast=True,
    )[KEY + ':time_mapping']
    return analysis, trajectory


class Approach1AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.vitro, self.vivo = noisy_pair()
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(context.__exit__, None, None, None)

    def assess(self, *, vitro=None, vivo=None, **options):
        arguments = dict(approach_id='approach1', model_name='linear',
                         evaluation_times=GRID, n_samples=SAMPLES,
                         random_state=SEED, analysis_config=CONFIG)
        arguments.update(options)
        return assess_ivivc_calibration(
            self.vitro if vitro is None else vitro,
            self.vivo if vivo is None else vivo, **arguments)

    def assert_trajectory(self, actual, expected):
        for name in ('in_vitro_time', 'in_vitro_value', 'in_vivo_time',
                     'in_vivo_value', 'time_lower', 'time_upper'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        self.assertEqual(actual.available, expected.available)
        self.assertEqual(actual.source_method, expected.source_method)
        for name in ('extrapolated', 'extrapolation_status', 'extrapolation_reason',
                     'interval_extrapolated', 'interval_extrapolation_status',
                     'interval_status', 'interval_valid_samples'):
            np.testing.assert_array_equal(actual.auxiliary[name], expected.auxiliary[name])
        self.assertEqual(actual.auxiliary['calibration_ranges'],
                         expected.auxiliary['calibration_ranges'])
        self.assertEqual(actual.auxiliary['extrapolation_warnings'],
                         expected.auxiliary['extrapolation_warnings'])

    def assert_mapping_fit(self, actual, expected):
        fit = actual[KEY].time_mapping_fit
        reference = expected[KEY].time_mapping_fit
        for name in ('time', 'response', 'parameter_vector', 'covariance', 'residuals'):
            np.testing.assert_array_equal(getattr(fit, name), getattr(reference, name))

    def test_defaults_match_direct_shared_time_mapping(self):
        result = self.assess()
        expected_analysis, expected = direct(
            self.vitro, self.vivo, TimeCourse(GRID, np.ones(len(GRID))))
        self.assertFalse(result.errors)
        self.assertTrue(result.fit_available)
        self.assertEqual(result.metadata['selection'], KEY)
        self.assertEqual(result.metadata['method'], 'time_mapping')
        self.assertEqual(result.relationship.quantity, 'in_vivo_time')
        np.testing.assert_array_equal(result.relationship.coordinate, GRID)
        np.testing.assert_array_equal(result.relationship.estimate, expected.in_vivo_time)
        np.testing.assert_array_equal(result.relationship.lower, expected.time_lower)
        np.testing.assert_array_equal(result.relationship.upper, expected.time_upper)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.relationship.trajectory, expected)

    def test_alternative_interpolation_matches_shared_pipeline(self):
        result = self.assess(selected_interpolation=['alternative'])
        expected_analysis, expected = direct(
            self.vitro, self.vivo, TimeCourse(GRID, np.ones(len(GRID))),
            interpolation='alternative')
        self.assertFalse(result.errors)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.relationship.trajectory, expected)
        self.assertEqual(result.analysis.prepared_data.preprocessing.interpolation, 'alternative')

    def test_explicit_no_interpolation_preserves_exact_response_matching(self):
        vitro = TimeCourse([0., 1., 2., 3.], [1., .8, .6, .4])
        vivo = TimeCourse([0., 2., 4., 6.], [1., .8, .6, .4])
        result = self.assess(vitro=vitro, vivo=vivo, selected_interpolation=['none'])
        expected_analysis, expected = direct(
            vitro, vivo, TimeCourse(GRID, np.ones(len(GRID))), interpolation='none')
        self.assertFalse(result.errors)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.relationship.trajectory, expected)

    def test_noisy_duplicate_and_plateau_rows_are_not_pooled_or_filtered(self):
        result = self.assess()
        self.assertFalse(result.errors)
        prepared = result.analysis.prepared_data
        np.testing.assert_array_equal(prepared.in_vitro.time, self.vitro.time)
        np.testing.assert_array_equal(prepared.in_vitro.value, self.vitro.value)
        np.testing.assert_array_equal(prepared.in_vivo.time, self.vivo.time)
        np.testing.assert_array_equal(prepared.in_vivo.value, self.vivo.value)
        self.assertEqual(np.count_nonzero(prepared.in_vitro.time == 1.), 2)
        self.assertEqual(np.count_nonzero(prepared.in_vitro.value == .52), 2)

    def test_extrapolated_points_and_intervals_retain_shared_diagnostics(self):
        result = self.assess()
        _, expected = direct(self.vitro, self.vivo, TimeCourse(GRID, np.ones(len(GRID))))
        self.assert_trajectory(result.relationship.trajectory, expected)
        self.assertTrue(result.relationship.trajectory.auxiliary['extrapolated'][-1])
        self.assertTrue(result.relationship.point_available[-1])
        self.assertTrue(result.relationship.interval_available[-1])

    def test_calibration_sd_and_sem_do_not_change_ordinary_fit_or_inputs(self):
        plain = self.assess()
        for kind, counts in (('individual', None), ('mean', [2, 3, 4, 5, 6, 7, 8])):
            with self.subTest(kind=kind):
                vitro = with_precision(self.vitro, np.linspace(.01, .1, 7), kind=kind, counts=counts)
                vivo = with_precision(self.vivo, .2, kind=kind, counts=counts)
                originals = [(s.time.copy(), s.value.copy(), s.observation_uncertainty.sigma.copy())
                             for s in (vitro, vivo)]
                result = self.assess(vitro=vitro, vivo=vivo)
                self.assertFalse(result.errors)
                self.assert_mapping_fit(result.analysis, plain.analysis)
                self.assert_trajectory(result.relationship.trajectory, plain.relationship.trajectory)
                self.assertFalse(result.analysis.prepared_data.has_observation_uncertainty)
                self.assertFalse(result.metadata['calibration_uncertainty_used_in_fit'])
                self.assertEqual(result.metadata['interval_interpretation'],
                                 'mapping_regression_parameter_uncertainty')
                for name, source in (('in_vitro', vitro), ('in_vivo', vivo)):
                    recorded = result.metadata['calibration_inputs'][name]
                    self.assertIsNot(recorded, source)
                    np.testing.assert_array_equal(recorded.time, source.time)
                    np.testing.assert_array_equal(recorded.value, source.value)
                    np.testing.assert_array_equal(recorded.observation_uncertainty.sigma,
                                                  source.observation_uncertainty.sigma)
                    self.assertEqual(recorded.observation_uncertainty.response_kind, kind)
                    np.testing.assert_array_equal(recorded.observation_uncertainty.replicate_counts,
                                                  source.observation_uncertainty.replicate_counts)
                for source, (times, values, sigma) in zip((vitro, vivo), originals):
                    np.testing.assert_array_equal(source.time, times)
                    np.testing.assert_array_equal(source.value, values)
                    np.testing.assert_array_equal(source.observation_uncertainty.sigma, sigma)

    def test_shared_core_still_rejects_supplied_uncertainty_for_approach1(self):
        vitro = with_precision(self.vitro, .03)
        vivo = with_precision(self.vivo, .04)
        prepared = prepare_ivivc_data(vitro, vivo, selected_interpolation=['default'])
        with self.assertRaisesRegex(ValueError, 'supported only for Approach 2'):
            fit_ivivc(prepared, selections=KEY, selected_metrics=['rmse'],
                      analysis_config=CONFIG, fail_fast=True)

    def test_target_only_prediction_matches_core_and_preserves_responses(self):
        target = TimeCourse([.75, 2., 5.], [.9, .7, .4], label='separate target')
        result = self.assess(evaluation_times=None, prediction=target)
        _, expected = direct(self.vitro, self.vivo, target)
        self.assertFalse(result.errors)
        self.assertIsNone(result.relationship)
        self.assertEqual(result.prediction.quantity, 'in_vivo_time')
        np.testing.assert_array_equal(result.prediction.coordinate, target.time)
        np.testing.assert_array_equal(result.prediction.trajectory.in_vivo_value, target.value)
        self.assert_trajectory(result.prediction.trajectory, expected)

    def test_normalization_uses_shared_preparation_once_for_calibration_and_target(self):
        vitro = TimeCourse(self.vitro.time, 10*self.vitro.value)
        vivo = TimeCourse(self.vivo.time, 20*self.vivo.value)
        target = TimeCourse([.5, 1.5, 3.], [50., 35., 20.])
        result = self.assess(vitro=vitro, vivo=vivo, prediction=target,
                             selected_normalizations=['initial_value'])
        target_prepared, _ = prepare_time_course(
            target.time, target.value, selected_normalizations=['initial_value'])
        expected_analysis, expected = direct(
            vitro, vivo, target_prepared, normalizations=['initial_value'])
        self.assertFalse(result.errors)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.prediction.trajectory, expected)
        np.testing.assert_array_equal(target.value, [50., 35., 20.])

    def test_shared_log_scalings_match_in_analyzed_coordinates(self):
        vitro = TimeCourse(np.arange(1., 8.), self.vitro.value)
        vivo = TimeCourse(np.arange(1., 8.), self.vivo.value)
        target = TimeCourse([1.5, 3., 8.], [.9, .7, .4])
        scalings = ['log_x', 'log_y']
        result = self.assess(vitro=vitro, vivo=vivo, evaluation_times=None,
                             prediction=target, selected_scalings=scalings)
        target_prepared, _ = prepare_time_course(
            target.time, target.value, selected_scalings=scalings)
        expected_analysis, expected = direct(vitro, vivo, target_prepared, scalings=scalings)
        self.assertFalse(result.errors)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.prediction.trajectory, expected)
        np.testing.assert_array_equal(result.prediction.coordinate, np.log10(target.time))

    def test_shared_invalid_log_fallback_keeps_unscaled_time_and_warning(self):
        target = TimeCourse([1., 2., 5.], [.8, .6, .4])
        result = self.assess(prediction=target, selected_scalings=['log_x'],
                             fallback_invalid_log_scaling=True)
        expected_analysis, expected = direct(
            self.vitro, self.vivo, TimeCourse(GRID, np.ones(len(GRID))),
            scalings=['log_x'], fallback=True)
        self.assertFalse(result.errors)
        self.assert_mapping_fit(result.analysis, expected_analysis)
        self.assert_trajectory(result.relationship.trajectory, expected)
        actual_metadata = result.analysis.prepared_data.preprocessing
        expected_metadata = expected_analysis.prepared_data.preprocessing
        self.assertEqual(actual_metadata.warnings, expected_metadata.warnings)
        self.assertEqual(actual_metadata.applied_scalings, ())
        self.assertTrue(actual_metadata.warnings)
        # A valid target must use the calibration's actual scale after fallback,
        # rather than independently applying its requested log transform.
        _, target_expected = direct(self.vitro, self.vivo, target,
                                    scalings=['log_x'], fallback=True)
        self.assert_trajectory(result.prediction.trajectory, target_expected)
        np.testing.assert_array_equal(result.prediction.coordinate, target.time)

    def test_uncertain_target_is_rejected_without_losing_relationship(self):
        target = with_precision(TimeCourse([1., 2.], [.8, .6]), .04, kind='mean', counts=3)
        result = self.assess(prediction=target)
        self.assertTrue(result.fit_available)
        self.assertIsNotNone(result.relationship)
        self.assertIsNone(result.prediction)
        self.assertIn('prediction', result.errors)
        self.assertIn('prediction-response uncertainty', result.errors['prediction'].message)
        np.testing.assert_array_equal(target.observation_uncertainty.sigma, [.04, .04])

    def test_optional_target_and_generator_do_not_change_relationship_or_rng(self):
        target = TimeCourse([1., 2.], [.8, .6])
        state = np.random.default_rng(SEED)
        before = deepcopy(state.bit_generator.state)
        alone = self.assess(random_state=state)
        together = self.assess(prediction=target, random_state=state)
        target_only = self.assess(evaluation_times=None, prediction=target, random_state=state)
        self.assertEqual(state.bit_generator.state, before)
        self.assert_trajectory(alone.relationship.trajectory, together.relationship.trajectory)
        self.assert_trajectory(target_only.prediction.trajectory, together.prediction.trajectory)

    def test_shared_fit_failure_is_recorded_and_fail_fast_remains_available(self):
        # Exact matching leaves only one shared response, insufficient for the
        # two-parameter linear mapping, just as in the ordinary core pipeline.
        result = self.assess(selected_interpolation=['none'])
        self.assertFalse(result.fit_available)
        self.assertIn('fit', result.errors)
        self.assertIsNone(result.relationship)
        with self.assertRaises((ValueError, TypeError)):
            self.assess(selected_interpolation=['none'], fail_fast=True)

    def test_invalid_approach_method_and_grid_are_rejected_before_fitting(self):
        for options in (
            dict(approach_id='approach3'), dict(method='value_ratio'),
            dict(response_levels=[.8]), dict(evaluation_times=[np.nan]),
            dict(evaluation_times=None), dict(inverse_interval_policy='strict'),
            dict(selected_interpolation=['unknown']),
        ):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                self.assess(**options)

    def test_simulation_entry_points_reject_approach1_until_simulation_integration(self):
        # The new assessor option must not enter the current Approach 2 truth
        # and reporting route merely because signature binding now accepts it.
        options = dict(approach_id='approach1', model_name='exponential',
                       evaluation_times=GRID)
        with self.assertRaisesRegex(ValueError, 'simulations currently support Approach 2'):
            IVIVCAssessmentRequest(self.vitro, options)
        with self.assertRaisesRegex(ValueError, 'simulations currently support Approach 2'):
            compare_calibration_designs(
                {'fixed': CalibrationDesign()}, assessment_options=options,
                existing_in_vitro=self.vitro, existing_in_vivo=self.vivo)

    def test_approach2_simulations_reject_new_coordinate_changing_options_early(self):
        for changes in (dict(selected_interpolation=['default']),
                        dict(selected_scalings=['log_x'])):
            options = dict(model_name='exponential', evaluation_times=GRID, **changes)
            with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, 'simulations use no interpolation or axis scaling'):
                IVIVCAssessmentRequest(self.vitro, options)


if __name__ == '__main__':
    unittest.main()
