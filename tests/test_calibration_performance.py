"""Calibration summaries retain failed runs and score only stated truth."""
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import AnalysisError, ObservationUncertainty, PredictedTrajectory, TimeCourse, prepare_time_course
from study_design_core.calibration_assessment import CalibrationEvaluation, IVIVCCalibrationAssessment
from study_design_core.calibration_performance import (
    CalibrationSimulationResult, _finish_calibration_simulation,
    _validate_assessment_options, summarize_calibration_simulation,
)
from study_design_core.types import ReferenceTrajectoryEnsemble


def evaluation(estimate, lower, upper, *, coordinate=(1., 2.), quantity='value_ratio', available=True):
    coordinate = np.asarray(coordinate, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    trajectory = PredictedTrajectory(
        'approach2', 'value_ratio', coordinate, np.ones(len(coordinate)),
        coordinate, estimate, available=available,
    )
    return CalibrationEvaluation(quantity, coordinate, estimate, lower, upper, trajectory)


def assessment(result=None, *, error=None):
    # These are controlled numerical outcomes for summary arithmetic; the
    # simulation tests exercise real fits and the independent entry points.
    fitted = SimpleNamespace(model_results={'approach2:exponential': object()})
    return IVIVCCalibrationAssessment(
        analysis=fitted if error is None else None, relationship=result,
        errors={} if error is None else {'fit': AnalysisError(error, 'RuntimeError')},
    )


def reference(draws, *, interpolation='linear', supported_range=None):
    draws = np.atleast_2d(np.asarray(draws, dtype=float))
    return ReferenceTrajectoryEnsemble(
        [0., 1., 2.], draws, np.ones(len(draws), dtype=bool),
        interpolation=interpolation, supported_time_range=supported_range,
    )


def finish(*, count=2, options=None, **kwargs):
    settings = dict(model_name='exponential', method='value_ratio', evaluation_times=[0., 1., 2.])
    settings.update(options or {})
    settings = _validate_assessment_options(settings)
    failed = [assessment(error='deliberate unavailable fit') for _ in range(count)]
    return _finish_calibration_simulation(
        failed, [None]*count, [None]*count, options=settings, **kwargs,
    )


class CalibrationPerformanceTests(unittest.TestCase):
    def test_missing_predictions_count_against_overall_coverage(self):
        outcomes = [
            assessment(evaluation([1., 2.], [0., 1.], [2., 3.])),
            assessment(evaluation([3., np.nan], [2., np.nan], [4., np.nan])),
            assessment(error='fit failed'),
        ]
        result = CalibrationSimulationResult(
            outcomes, [None]*3, [None]*3,
            truth={'relationship': np.tile([1., 2.], (3, 1))},
        )
        summary = summarize_calibration_simulation(result)
        measured = summary['relationship']
        self.assertEqual(summary['n_simulations'], 3)
        self.assertEqual(summary['n_fit_successful'], 2)
        self.assertAlmostEqual(summary['fit_failure_rate'], 1/3)
        self.assertEqual(summary['error_counts']['fit'], 1)
        np.testing.assert_array_equal(result.fit_success, [True, True, False])
        np.testing.assert_array_equal(measured['point_counts'], [2, 1])
        np.testing.assert_array_equal(measured['interval_counts'], [2, 1])
        np.testing.assert_array_equal(measured['truth_counts'], [3, 3])
        np.testing.assert_array_equal(measured['interval_truth_counts'], [2, 1])
        np.testing.assert_allclose(measured['point_availability'], [2/3, 1/3])
        np.testing.assert_allclose(measured['pointwise_coverage'], [1/3, 1/3])
        np.testing.assert_allclose(measured['pointwise_conditional_coverage'], [.5, 1.])
        self.assertAlmostEqual(measured['coverage'], 1/3)
        self.assertAlmostEqual(measured['conditional_coverage'], 2/3)
        self.assertAlmostEqual(measured['mean_interval_width'], 2.)
        self.assertAlmostEqual(measured['rmse'], np.sqrt(4/3))
        np.testing.assert_allclose(measured['pointwise_rmse'], [np.sqrt(2.), 0.])

    def test_unknown_truth_does_not_remove_valid_precision_results(self):
        result = CalibrationSimulationResult(
            [assessment(evaluation([1., 2.], [.5, 1.5], [1.5, 2.5]))],
            [None], [None],
        )
        measured = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(measured['point_counts'], [1, 1])
        np.testing.assert_array_equal(measured['truth_counts'], [0, 0])
        self.assertAlmostEqual(measured['mean_interval_width'], 1.)
        self.assertTrue(np.isnan(measured['rmse']))
        self.assertTrue(np.isnan(measured['coverage']))

    def test_unknown_truth_and_missing_intervals_have_separate_denominators(self):
        result = CalibrationSimulationResult(
            [assessment(evaluation([1., 2.], [0., 1.], [2., 3.])),
             assessment(evaluation([1., 2.], [np.nan, 1.], [np.nan, 3.]))],
            [None]*2, [None]*2,
            truth={'relationship': [[np.nan, 2.], [1., np.nan]]},
        )
        measured = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(measured['truth_counts'], [1, 1])
        np.testing.assert_array_equal(measured['interval_truth_counts'], [0, 1])
        np.testing.assert_array_equal(measured['error_counts'], [1, 1])
        np.testing.assert_allclose(measured['pointwise_coverage'], [0., 1.])
        self.assertTrue(np.isnan(measured['pointwise_conditional_coverage'][0]))
        self.assertEqual(measured['pointwise_conditional_coverage'][1], 1.)

    def test_all_failed_runs_retain_the_requested_evaluation_grid(self):
        result = finish(truth={'relationship': 1.})
        summary = summarize_calibration_simulation(result)
        measured = summary['relationship']
        self.assertEqual(summary['n_simulations'], 2)
        self.assertEqual(summary['n_fit_successful'], 0)
        self.assertEqual(summary['fit_failure_rate'], 1.)
        np.testing.assert_array_equal(measured['coordinate'], [0., 1., 2.])
        np.testing.assert_array_equal(measured['point_counts'], [0, 0, 0])
        np.testing.assert_array_equal(measured['truth_counts'], [2, 2, 2])
        self.assertTrue(np.isnan(measured['mean_interval_width']))
        self.assertTrue(np.isnan(measured['rmse']))
        self.assertEqual(measured['coverage'], 0.)

    def test_auto_ratio_truth_follows_selected_complete_source_draws(self):
        vitro = reference([[2., 1., .5], [4., 2., 1.]])
        vivo = reference([[1., .8, .4], [2., 1., .25]])
        result = finish(
            in_vitro_reference=vitro, in_vivo_reference=vivo,
            in_vitro_draw_indices=[1, 0], in_vivo_draw_indices=[0, 1],
        )
        np.testing.assert_allclose(result.truth['relationship'], [[.25, .4, .4], [1., 1., .5]])

    def test_ratio_truth_reversal_is_applied_once_on_the_common_scale(self):
        result = finish(
            options={'selected_normalizations': ['one_minus']},
            in_vitro_reference=reference([[.8, .5, .2]]),
            in_vivo_reference=reference([[.9, .7, .4]]),
            in_vitro_draw_indices=[0, 0], in_vivo_draw_indices=[0, 0],
        )
        np.testing.assert_allclose(result.truth['relationship'], np.tile([.5, .6, .75], (2, 1)))

    def test_finite_reference_does_not_define_an_infinite_integral_tau(self):
        result = finish(
            options={'method': 'tau_ratio', 'evaluation_times': None},
            in_vitro_reference=reference([[1., .7, .5]]),
            in_vivo_reference=reference([[1., .8, .7]]),
            in_vitro_draw_indices=[0, 0], in_vivo_draw_indices=[0, 0],
        )
        measured = summarize_calibration_simulation(result)['relationship']
        np.testing.assert_array_equal(measured['truth_counts'], [0])
        self.assertTrue(np.isnan(measured['coverage']))

    def test_explicit_truth_has_unambiguous_point_and_experiment_shapes(self):
        vector = finish(count=3, truth={'relationship': [1., 2., 3.]})
        np.testing.assert_array_equal(vector.truth['relationship'], np.tile([1., 2., 3.], (3, 1)))
        per_run = finish(count=3,
            options={'method': 'tau_ratio', 'evaluation_times': None},
            truth={'relationship': np.array([[1.], [2.], [3.]])})
        np.testing.assert_array_equal(per_run.truth['relationship'], [[1.], [2.], [3.]])
        with self.assertRaises(ValueError):
            finish(count=3, options={'method': 'tau_ratio', 'evaluation_times': None},
                   truth={'relationship': [1., 2., 3.]})
        with self.assertRaises(ValueError):
            finish(truth={'prediction': 1.})

    def test_inverse_truth_honors_pchip_and_the_supported_time_domain(self):
        vivo = reference([[1., .4, .1]], interpolation='pchip', supported_range=(.25, 1.75))
        valid_level = vivo.evaluate([.6], draw_indices=[0])[0, 0]
        result = finish(
            options={'method': 'observed_response_inverse', 'evaluation_times': None,
                     'response_levels': [valid_level, 1.1]},
            in_vivo_reference=vivo, in_vivo_draw_indices=[0, 0],
        )
        np.testing.assert_allclose(result.truth['relationship'][:, 0], [.6, .6], atol=1e-8)
        self.assertTrue(np.isnan(result.truth['relationship'][:, 1]).all())

    def test_uncertain_target_requires_explicit_response_truth(self):
        target = prepare_time_course(
            [0., 1., 2.], [.8, .6, .4],
            observation_uncertainty=ObservationUncertainty(.05, response_kind='mean'),
        )[0]
        options = {'prediction': target}
        refs = dict(in_vitro_reference=reference([[1., .5, .25]]),
                    in_vivo_reference=reference([[1., .75, .5]]),
                    in_vitro_draw_indices=[0, 0], in_vivo_draw_indices=[0, 0])
        unspecified = finish(options=options, **refs)
        measured = summarize_calibration_simulation(unspecified)['prediction']
        np.testing.assert_array_equal(measured['truth_counts'], [0, 0, 0])
        explicit = finish(options=options, truth={'prediction': [.75, .85, .95]}, **refs)
        np.testing.assert_allclose(explicit.truth['prediction'], np.tile([.75, .85, .95], (2, 1)))

    def test_fixed_target_truth_and_tau_time_do_not_add_response_noise(self):
        target = TimeCourse([0., 1., 2.], [.8, .6, .4])
        result = finish(options={'prediction': target},
            in_vitro_reference=reference([[1., .5, .25]]),
            in_vivo_reference=reference([[1., .75, .5]]),
            in_vitro_draw_indices=[0, 0], in_vivo_draw_indices=[0, 0])
        np.testing.assert_allclose(result.truth['prediction'], np.tile([.8, .9, .8], (2, 1)))
        uncertain_target = prepare_time_course(target.time, target.value,
            observation_uncertainty=ObservationUncertainty(.1, response_kind='mean'))[0]
        tau = finish(options={'method': 'tau_ratio', 'evaluation_times': None,
                              'prediction': uncertain_target},
                     truth={'relationship': 3.})
        np.testing.assert_allclose(tau.truth['prediction'], np.tile([0., 3., 6.], (2, 1)))

    def test_comparisons_require_a_common_preprocessing_scale(self):
        for name in ('initial_value', 'min_max'):
            with self.subTest(normalization=name), self.assertRaises(ValueError):
                _validate_assessment_options(dict(model_name='exponential',
                    evaluation_times=[0., 1.], selected_normalizations=[name]))
        _validate_assessment_options(dict(model_name='exponential',
            evaluation_times=[0., 1.], selected_normalizations=['one_minus']))
        with self.assertRaises((TypeError, ValueError)):
            _validate_assessment_options(dict(model_name='exponential',
                evaluation_times=[0., 1.], fail_fast=True))

    def test_empty_optional_target_does_not_discard_a_usable_relationship(self):
        self._check_unusable_optional_target(TimeCourse([], []), empty_grid=True)

    def test_all_missing_optional_target_retains_its_requested_grid(self):
        self._check_unusable_optional_target(
            TimeCourse([1., 2.], [np.nan, np.nan]), empty_grid=False)

    def _check_unusable_optional_target(self, target, *, empty_grid):
        measured = evaluation([1., 1.1, 1.2], [.9, 1., 1.1], [1.1, 1.2, 1.3],
                              coordinate=[0., 1., 2.])
        outcomes = [assessment(measured), assessment(measured)]
        for item in outcomes:
            item.errors['prediction'] = AnalysisError('target unavailable', 'ValueError')
        options = _validate_assessment_options(dict(
            model_name='exponential', evaluation_times=[0., 1., 2.], prediction=target))
        result = _finish_calibration_simulation(
            outcomes, [None]*2, [None]*2, options=options)
        summary = summarize_calibration_simulation(result)
        self.assertEqual(summary['n_fit_successful'], 2)
        self.assertEqual(summary['error_counts']['prediction'], 2)
        np.testing.assert_array_equal(summary['relationship']['point_counts'], [2, 2, 2])
        self.assertAlmostEqual(summary['relationship']['mean_interval_width'], .2)
        predicted = summary['prediction']
        np.testing.assert_array_equal(predicted['coordinate'], target.time)
        np.testing.assert_array_equal(predicted['point_counts'], np.zeros(len(target.time)))
        np.testing.assert_array_equal(predicted['truth_counts'], np.zeros(len(target.time)))
        self.assertTrue(np.isnan(predicted['coverage']))
        if empty_grid:
            self.assertTrue(np.isnan(predicted['complete_point_rate']))
            self.assertTrue(np.isnan(predicted['complete_interval_rate']))
        else:
            self.assertEqual(predicted['complete_point_rate'], 0.)
            self.assertEqual(predicted['complete_interval_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
