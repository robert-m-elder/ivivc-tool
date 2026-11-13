"""Approach 1 timing truth and across-study summaries use available points."""
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import AnalysisError, PredictedTrajectory, TimeCourse
from study_design_core.calibration_assessment import CalibrationEvaluation, IVIVCCalibrationAssessment
from study_design_core.calibration_performance import (
    CalibrationSimulationResult, _finish_calibration_simulation,
    _validate_assessment_options, summarize_calibration_simulation,
)
from study_design_core.types import ReferenceTrajectoryEnsemble


def reference(slopes=(.1,), *, interpolation='linear', supported_range=None):
    time = np.arange(5., dtype=float)
    values = 1. - np.asarray(slopes)[:, None]*time
    return ReferenceTrajectoryEnsemble(
        time, values, np.ones(len(slopes), dtype=bool),
        interpolation=interpolation, supported_time_range=supported_range)


def evaluation(estimates, lower=None, upper=None, *, coordinate=(1., 2.), available=True):
    coordinate = np.asarray(coordinate, dtype=float)
    estimates = np.asarray(estimates, dtype=float)
    trajectory = PredictedTrajectory(
        'approach1', 'time_mapping', coordinate, np.ones(len(coordinate)),
        estimates, np.ones(len(coordinate)), available=available)
    return CalibrationEvaluation('in_vivo_time', coordinate, estimates, lower, upper, trajectory)


def assessment(result=None, *, error=None):
    fitted = SimpleNamespace(model_results={'approach1:linear': object()})
    return IVIVCCalibrationAssessment(
        analysis=fitted if error is None else None, relationship=result,
        errors={} if error is None else {'fit': AnalysisError(error, 'RuntimeError')},
        metadata={'selection': 'approach1:linear',
                  'interval_interpretation': 'mapping_regression_parameter_uncertainty'})


def options(**changes):
    values = dict(approach_id='approach1', model_name='linear', evaluation_times=[0., .5, 1.])
    values.update(changes)
    return _validate_assessment_options(values)


def finish(*, count=2, settings=None, **kwargs):
    settings = options(**(settings or {}))
    failed = [assessment(error='controlled failed fit') for _ in range(count)]
    return _finish_calibration_simulation(
        failed, [None]*count, [None]*count, options=settings, **kwargs)


class Approach1CalibrationPerformanceTests(unittest.TestCase):
    def test_method_defaults_and_shared_interpolation_options(self):
        for selection in ('default', 'alternative', 'none'):
            for method in (None, 'time_mapping'):
                with self.subTest(interpolation=selection, method=method):
                    parsed = options(selected_interpolation=[selection], method=method)
                    self.assertEqual(parsed['method'], 'time_mapping')
                    self.assertEqual(parsed['approach_id'], 'approach1')
                    self.assertEqual(tuple(parsed['selected_interpolation']), (selection,))
        self.assertEqual(options()['method'], 'time_mapping')

    def test_configuration_rejects_cross_approach_methods_and_invalid_grids(self):
        for changes in (dict(method='value_ratio'), dict(method='tau_ratio'),
                        dict(approach_id='approach3'), dict(response_levels=[.8]),
                        dict(evaluation_times=[np.nan]),
                        dict(selected_interpolation=['unknown']),
                        dict(inverse_interval_policy='strict')):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                options(**changes)

    def test_known_reference_mapping_and_fixed_target_use_time_not_target_response(self):
        target = TimeCourse([.25, .75], [123., -456.])
        result = finish(settings={'prediction': target},
            in_vitro_reference=reference((.1,)), in_vivo_reference=reference((.05,)))
        np.testing.assert_allclose(result.truth['relationship'], [[0., 1., 2.]]*2, atol=1e-12)
        np.testing.assert_allclose(result.truth['prediction'], [[.5, 1.5]]*2, atol=1e-12)
        self.assertEqual(result.metadata['evaluation_spec']['relationship']['quantity'], 'in_vivo_time')
        np.testing.assert_array_equal(
            result.metadata['evaluation_spec']['prediction']['coordinate'], target.time)

    def test_reference_truth_follows_selected_environment_scenarios(self):
        result = finish(in_vitro_reference=reference((.1, .2)),
            in_vivo_reference=reference((.05, .1)),
            in_vitro_draw_indices=[1, 0], in_vivo_draw_indices=[0, 1])
        np.testing.assert_allclose(result.truth['relationship'],
                                   [[0., 2., 4.], [0., .5, 1.]], atol=1e-12)

    def test_log_time_truth_uses_the_analyzed_time_axes(self):
        physical_times = np.array([.5, 1., 2.])
        target = TimeCourse([.5, 1.], [.9, .8])
        result = finish(settings={'evaluation_times': np.log10(physical_times),
                                  'prediction': target, 'selected_scalings': ['log_x']},
            in_vitro_reference=reference((.1,)), in_vivo_reference=reference((.05,)))
        np.testing.assert_allclose(result.truth['relationship'],
                                   np.tile(np.log10(2*physical_times), (2, 1)), atol=1e-12)
        np.testing.assert_allclose(result.truth['prediction'],
                                   np.tile(np.log10(2*target.time), (2, 1)), atol=1e-12)
        with self.assertRaises(ValueError):
            options(selected_scalings=['log_x'], fallback_invalid_log_scaling=True)

    def test_common_response_reversal_does_not_change_true_time_mapping(self):
        result = finish(settings={'selected_normalizations': ['one_minus']},
            in_vitro_reference=reference((.1,)), in_vivo_reference=reference((.05,)))
        np.testing.assert_allclose(result.truth['relationship'], [[0., 1., 2.]]*2, atol=1e-12)

    def test_reference_domain_limits_only_truth_not_requested_evaluation_grid(self):
        result = finish(settings={'evaluation_times': [-.1, 1., 3.]},
            in_vitro_reference=reference((.1,)), in_vivo_reference=reference((.05,)))
        actual = result.truth['relationship']
        self.assertTrue(np.isnan(actual[:, 0]).all())
        np.testing.assert_allclose(actual[:, 1], [2., 2.], atol=1e-12)
        self.assertTrue(np.isnan(actual[:, 2]).all())
        summary = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(summary['coordinate'], [-.1, 1., 3.])
        np.testing.assert_array_equal(summary['truth_counts'], [0, 2, 0])

    def test_inverse_truth_honors_reference_interpolation(self):
        time = np.arange(5., dtype=float)
        vivo = ReferenceTrajectoryEnsemble(
            time, [[1., .85, .6, .3, .1]], [True], interpolation='pchip')
        level = vivo.evaluate([1.3], draw_indices=[0])[0, 0]
        vitro = ReferenceTrajectoryEnsemble(
            time, [[1., level, .5, .3, .1]], [True], interpolation='linear')
        result = finish(settings={'evaluation_times': [1.]},
                        in_vitro_reference=vitro, in_vivo_reference=vivo)
        np.testing.assert_allclose(result.truth['relationship'], [[1.3], [1.3]], atol=1e-10)

    def test_ambiguous_plateau_truth_is_unavailable_only_at_that_level(self):
        vivo = ReferenceTrajectoryEnsemble(
            np.arange(5.), [[1., .8, .8, .4, .2]], [True], interpolation='linear')
        vitro = ReferenceTrajectoryEnsemble(
            np.arange(5.), [[1., .8, .6, .4, .2]], [True], interpolation='linear')
        result = finish(settings={'evaluation_times': [0., 1., 4.]},
                        in_vitro_reference=vitro, in_vivo_reference=vivo)
        actual = result.truth['relationship']
        np.testing.assert_allclose(actual[:, [0, 2]], [[0., 4.], [0., 4.]], atol=1e-12)
        self.assertTrue(np.isnan(actual[:, 1]).all())

    def test_missing_generating_environment_keeps_truth_metrics_unavailable(self):
        for references in ({'in_vitro_reference': reference()},
                           {'in_vivo_reference': reference()}):
            with self.subTest(references=tuple(references)):
                result = finish(**references)
                summary = summarize_calibration_simulation(result)['relationship']
                np.testing.assert_array_equal(summary['truth_counts'], [0, 0, 0])
                self.assertTrue(np.isnan(summary['rmse']))

    def test_explicit_per_experiment_truth_overrides_reference_mapping(self):
        truth = np.array([[5., 6., 7.], [8., 9., 10.]])
        result = finish(truth={'relationship': truth},
            in_vitro_reference=reference((.1,)), in_vivo_reference=reference((.05,)))
        np.testing.assert_array_equal(result.truth['relationship'], truth)
        self.assertEqual(result.metadata['truth_sources']['relationship'], 'explicit_output_truth')

    def summary_result(self, outcomes, *, truth=None, confidence=.8):
        return CalibrationSimulationResult(
            outcomes, [None]*len(outcomes), [None]*len(outcomes),
            truth={} if truth is None else {'relationship': truth},
            metadata={'approach_id': 'approach1', 'confidence_level': confidence,
                      'interval_interpretation': 'mapping_regression_parameter_uncertainty',
                      'evaluation_spec': {'relationship': {'quantity': 'in_vivo_time',
                                                           'coordinate': np.array([1., 2.])}}})

    def test_empirical_bias_and_quantiles_use_points_even_when_intervals_are_missing(self):
        result = self.summary_result([
            assessment(evaluation([2., 3.], [1., 2.], [3., 4.])),
            assessment(evaluation([4., np.nan])), assessment(error='failed')],
            truth=np.tile([1., 2.], (3, 1)))
        summary = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(summary['point_counts'], [2, 1])
        np.testing.assert_array_equal(summary['error_counts'], [2, 1])
        np.testing.assert_array_equal(summary['interval_counts'], [1, 1])
        self.assertAlmostEqual(summary['bias'], 5/3)
        np.testing.assert_allclose(summary['pointwise_bias'], [2., 1.])
        self.assertAlmostEqual(summary['rmse'], np.sqrt(11/3))
        np.testing.assert_allclose(summary['empirical_quantiles'], [.1, .5, .9])
        np.testing.assert_allclose(summary['pointwise_estimate_quantiles'],
                                   [[2.2, 3.], [3., 3.], [3.8, 3.]])
        np.testing.assert_allclose(summary['pointwise_error_quantiles'],
                                   [[1.2, 1.], [2., 1.], [2.8, 1.]])
        self.assertAlmostEqual(summary['mean_interval_width'], 2.)

    def test_unknown_truth_preserves_estimate_spread_but_not_error_spread(self):
        result = self.summary_result([
            assessment(evaluation([2., 3.])), assessment(evaluation([4., 5.]))])
        summary = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(summary['error_counts'], [0, 0])
        self.assertTrue(np.isnan(summary['bias']))
        self.assertTrue(np.isnan(summary['pointwise_bias']).all())
        self.assertTrue(np.isnan(summary['pointwise_error_quantiles']).all())
        np.testing.assert_allclose(summary['pointwise_estimate_quantiles'],
                                   [[2.2, 3.2], [3., 4.], [3.8, 4.8]])

    def test_all_failed_studies_retain_nan_quantiles_and_zero_available_counts(self):
        result = self.summary_result([assessment(error='failed')]*3,
                                     truth=np.tile([1., 2.], (3, 1)))
        overall = summarize_calibration_simulation(result)
        summary = overall['relationship']
        self.assertEqual(overall['n_simulations'], 3)
        self.assertEqual(overall['fit_failure_rate'], 1.)
        np.testing.assert_array_equal(summary['point_counts'], [0, 0])
        np.testing.assert_array_equal(summary['error_counts'], [0, 0])
        self.assertTrue(np.isnan(summary['pointwise_estimate_quantiles']).all())
        self.assertTrue(np.isnan(summary['pointwise_error_quantiles']).all())
        self.assertTrue(np.isnan(summary['bias']))
        self.assertEqual(summary['coverage'], 0.)

    def test_unavailable_finite_estimates_do_not_enter_empirical_summaries(self):
        result = self.summary_result([
            assessment(evaluation([2., 3.])),
            assessment(evaluation([999., 888.], available=False))],
            truth=np.tile([1., 2.], (2, 1)))
        summary = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(summary['point_counts'], [1, 1])
        np.testing.assert_array_equal(summary['error_counts'], [1, 1])
        np.testing.assert_array_equal(summary['pointwise_estimate_quantiles'], [[2., 3.]]*3)
        np.testing.assert_array_equal(summary['pointwise_error_quantiles'], [[1., 1.]]*3)

    def test_existing_regression_interval_width_does_not_change_empirical_spread(self):
        def summary(span):
            return summarize_calibration_simulation(self.summary_result([
                assessment(evaluation([2., 3.], [2.-span, 3.-span], [2.+span, 3.+span])),
                assessment(evaluation([4., 5.], [4.-span, 5.-span], [4.+span, 5.+span]))],
                truth=np.tile([1., 2.], (2, 1))))['relationship']
        narrow, wide = summary(.1), summary(100.)
        for name in ('pointwise_estimate_quantiles', 'pointwise_error_quantiles', 'pointwise_bias'):
            np.testing.assert_array_equal(narrow[name], wide[name])
        self.assertGreater(wide['mean_interval_width'], narrow['mean_interval_width'])


if __name__ == '__main__':
    unittest.main()
