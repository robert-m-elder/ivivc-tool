"""Approach 1 assessment uses the study's actual calibration observations."""
from collections.abc import Mapping
from dataclasses import fields, replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import (
    ExistingObservations, ObservationModel, ReferenceTrajectoryEnsemble,
    SimulationPlan, simulate_design,
)
from study_design_core.calibration_assessment import assess_ivivc_calibration
from study_design_core.calibration_performance import (
    IVIVCAssessmentRequest, summarize_calibration_simulation,
)
from study_design_core.calibration_simulation import (
    CalibrationDesign, compare_calibration_designs,
)
import study_design_core.simulation as simulation


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)
TIMES = np.array([0., 2., 4.])
GRID = np.array([.5, 1., 1.5])


def reference(rates=(.2, .25)):
    time = np.linspace(0., 8., 129)
    return ReferenceTrajectoryEnsemble(
        time, np.exp(-np.asarray(rates)[:, None]*time), [True]*len(rates),
        interpolation='linear')


def companion():
    time = np.arange(5.)
    return prepare_time_course(
        time, np.exp(-.4*time), label='fixed in vitro measurements',
        observation_uncertainty=ObservationUncertainty(
            .02, response_kind='mean', replicate_counts=4),
    )[0]


def options(**changes):
    result = dict(approach_id='approach1', model_name='linear',
                  evaluation_times=GRID, n_samples=32, random_state=321,
                  analysis_config=CONFIG)
    result.update(changes)
    return result


def plan(replicates=3, indices=(0, 1)):
    error = np.linspace(-1., 1., len(indices)*replicates*len(TIMES))
    return SimulationPlan(
        TIMES, indices, error.reshape(len(indices), replicates, len(TIMES)),
        metadata={'common_random_numbers': True})


class Approach1OptionalStudyTests(unittest.TestCase):
    def setUp(self):
        self.reference = reference()
        self.observation = ObservationModel('constant_sd', .04, replicates_per_time=3)
        self.plan = plan()
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(context.__exit__, None, None, None)

    def request(self, **changes):
        arguments = dict(in_vitro=companion(), options=options())
        arguments.update(changes)
        return IVIVCAssessmentRequest(**arguments)

    def run_study(self, request=None, **changes):
        arguments = dict(model_name='exponential_normalized',
                         simulation_plan=self.plan, fallback_grid_search=False,
                         ivivc_assessment=request)
        arguments.update(changes)
        return simulate_design(self.reference, TIMES, self.observation, **arguments)

    def assert_identical(self, first, second):
        if isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, Mapping):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_identical(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for old, new in zip(first, second):
                self.assert_identical(old, new)
        else:
            self.assertEqual(first, second)

    def assert_same_recovery(self, original, enabled):
        for item in fields(original):
            if item.name != 'ivivc_assessment':
                with self.subTest(field=item.name):
                    self.assert_identical(getattr(original, item.name),
                                          getattr(enabled, item.name))

    def test_assessment_preserves_every_curve_recovery_field_and_metadata(self):
        original = self.run_study()
        enabled = self.run_study(self.request())
        self.assert_same_recovery(original, enabled)
        self.assertIsNone(original.ivivc_assessment)
        self.assertTrue(enabled.ivivc_assessment.fit_success.all())
        self.assertTrue(enabled.ivivc_assessment.metadata['uses_same_study_measurements'])

    def test_actual_raw_replicates_match_manual_shared_assessment(self):
        request = self.request(options=options(selected_interpolation=['alternative']))
        result = self.run_study(request).ivivc_assessment
        raw = (self.reference.evaluate(TIMES, draw_indices=[0, 1])[:, None, :]
               + .04*self.plan.standard_normal_errors)
        order = np.argsort(np.tile(TIMES, 3), kind='stable')
        for index, actual in enumerate(result.assessments):
            vivo = result.in_vivo_data[index]
            np.testing.assert_array_equal(vivo.time, np.tile(TIMES, 3)[order])
            np.testing.assert_array_equal(vivo.value, raw[index].reshape(-1)[order])
            np.testing.assert_array_equal(vivo.observation_uncertainty.sigma, np.full(9, .04))
            expected = assess_ivivc_calibration(request.in_vitro, vivo, **request.options)
            self.assertFalse(actual.errors)
            for name in ('estimate', 'lower', 'upper'):
                np.testing.assert_array_equal(getattr(actual.relationship, name),
                                              getattr(expected.relationship, name))
            fitted = actual.analysis['approach1:linear'].time_mapping_fit
            self.assertIsNone(fitted.observation_uncertainty)
            self.assertFalse(actual.metadata['calibration_uncertainty_used_in_fit'])

    def test_completed_means_stay_fixed_and_new_means_use_actual_replicates(self):
        completed = ExistingObservations(
            [0.], [1.01], observation_uncertainty=ObservationUncertainty(
                .03, response_kind='mean', replicate_counts=4))
        result = self.run_study(self.request(), existing_observations=completed).ivivc_assessment
        for index, vivo in enumerate(result.in_vivo_data):
            future = (self.reference.evaluate([2., 4.], draw_indices=[index])[0]
                      + .04*self.plan.standard_normal_errors[index, :, 1:])
            np.testing.assert_array_equal(vivo.time, TIMES)
            np.testing.assert_allclose(vivo.value, np.r_[1.01, future.mean(axis=0)])
            np.testing.assert_allclose(vivo.observation_uncertainty.sigma,
                                       [.03, .04/np.sqrt(3), .04/np.sqrt(3)])
            np.testing.assert_array_equal(vivo.observation_uncertainty.replicate_counts,
                                          [4, 3, 3])
        np.testing.assert_array_equal(completed.response, [1.01])
        np.testing.assert_array_equal(completed.observation_uncertainty.sigma, [.03])

    def test_two_entry_points_match_when_their_observations_match(self):
        old_vivo = prepare_time_course(
            [0.], [1.01], observation_uncertainty=ObservationUncertainty(
                .03, response_kind='mean', replicate_counts=4))[0]
        completed = ExistingObservations(
            old_vivo.time, old_vivo.value,
            observation_uncertainty=old_vivo.observation_uncertainty)
        request = self.request()
        study = self.run_study(request, existing_observations=completed).ivivc_assessment
        compared = compare_calibration_designs(
            {'candidate': CalibrationDesign(
                in_vivo_times=[2., 4.], in_vivo_observation_model=self.observation)},
            assessment_options=request.options, existing_in_vitro=request.in_vitro,
            existing_in_vivo=old_vivo, in_vivo_reference=self.reference,
            simulation_plans={'in_vivo': self.plan}).results['candidate']
        for index, (first, second) in enumerate(zip(study.assessments, compared.assessments)):
            self.assertFalse(first.errors)
            self.assertFalse(second.errors)
            for side in ('in_vitro_data', 'in_vivo_data'):
                left, right = getattr(study, side)[index], getattr(compared, side)[index]
                np.testing.assert_array_equal(left.time, right.time)
                np.testing.assert_allclose(left.value, right.value)
                np.testing.assert_array_equal(left.observation_uncertainty.sigma,
                                              right.observation_uncertainty.sigma)
            for name in ('estimate', 'lower', 'upper'):
                np.testing.assert_allclose(getattr(first.relationship, name),
                                           getattr(second.relationship, name))

    def test_known_reference_mapping_supplies_time_truth(self):
        self.reference = reference((.2,))
        self.plan = plan(indices=(0, 0))
        request = self.request(in_vitro_reference=reference((.4,)))
        result = self.run_study(request).ivivc_assessment
        np.testing.assert_allclose(result.truth['relationship'], np.tile(2*GRID, (2, 1)),
                                   rtol=1e-10, atol=1e-10)
        summary = summarize_calibration_simulation(result)['relationship']
        expected = np.stack([trial.relationship.estimate for trial in result.assessments])
        self.assertEqual(summary['quantity'], 'in_vivo_time')
        self.assertAlmostEqual(summary['rmse'],
                               np.sqrt(np.mean((expected - result.truth['relationship'])**2)))
        np.testing.assert_array_equal(summary['truth_counts'], [2, 2, 2])
        error = expected - result.truth['relationship']
        np.testing.assert_allclose(summary['pointwise_bias'], error.mean(axis=0))
        np.testing.assert_allclose(summary['pointwise_error_quantiles'],
                                   np.quantile(error, [.025, .5, .975], axis=0))
        np.testing.assert_allclose(summary['pointwise_estimate_quantiles'],
                                   np.quantile(expected, [.025, .5, .975], axis=0))

    def test_explicit_none_method_resolves_to_shared_time_mapping(self):
        implicit = self.run_study(self.request()).ivivc_assessment
        explicit = self.run_study(self.request(options=options(method=None))).ivivc_assessment
        for first, second in zip(implicit.assessments, explicit.assessments):
            self.assertEqual(first.metadata['method'], 'time_mapping')
            np.testing.assert_array_equal(first.relationship.estimate,
                                          second.relationship.estimate)

    def test_curve_fit_failure_does_not_suppress_approach1_assessment(self):
        original = simulation.fit_model_once
        def fail(*args, **kwargs):
            return replace(original(*args, **kwargs), success=False,
                           message='forced curve-fit failure')
        with patch.object(simulation, 'fit_model_once', side_effect=fail):
            result = self.run_study(self.request())
        self.assertFalse(result.fit_success.any())
        self.assertTrue(result.ivivc_assessment.fit_success.all())

    def test_mapping_failure_does_not_change_successful_curve_recovery(self):
        original = self.run_study()
        enabled = self.run_study(self.request(in_vitro=TimeCourse([0.], [1.])))
        self.assert_same_recovery(original, enabled)
        self.assertTrue(enabled.fit_success.all())
        self.assertFalse(enabled.ivivc_assessment.fit_success.any())
        self.assertTrue(all('fit' in trial.errors
                            for trial in enabled.ivivc_assessment.assessments))

    def test_assessment_does_not_advance_measurement_rng(self):
        plain_rng = np.random.default_rng(905)
        assessment_rng = np.random.default_rng(905)
        original = self.run_study(simulation_plan=None, n_simulations=3,
                                   random_state=plain_rng)
        enabled = self.run_study(
            self.request(options=options(random_state=assessment_rng)),
            simulation_plan=None, n_simulations=3, random_state=assessment_rng)
        self.assert_same_recovery(original, enabled)
        self.assertEqual(plain_rng.bit_generator.state, assessment_rng.bit_generator.state)

    def test_target_times_are_assessed_with_same_fixed_response_coordinates(self):
        target = TimeCourse([.5, 1., 1.5], [.85, .7, .5])
        request = self.request(options=options(prediction=target),
                               in_vitro_reference=reference((.4,)))
        result = self.run_study(request).ivivc_assessment
        for trial in result.assessments:
            self.assertEqual(trial.prediction.quantity, 'in_vivo_time')
            np.testing.assert_array_equal(trial.prediction.coordinate, target.time)
            np.testing.assert_array_equal(trial.prediction.trajectory.in_vivo_value, target.value)
        # A target's response columns are carried through, not inverted again.
        np.testing.assert_allclose(result.truth['prediction'], result.truth['relationship'])


if __name__ == '__main__':
    unittest.main()
