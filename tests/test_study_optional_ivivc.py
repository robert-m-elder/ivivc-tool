"""Optional IVIVC assessment shares the study observations and leaves recovery unchanged."""
from dataclasses import fields, replace
from collections.abc import Mapping
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import (
    DesignConstraints, ExistingObservations, ObservationModel,
    ReferenceTrajectoryEnsemble, SimulationPlan, compare_designs_by_simulation,
    fit_reference_model_ensemble, simulate_design,
)
from study_design_core.calibration_assessment import assess_ivivc_calibration
from study_design_core.calibration_performance import IVIVCAssessmentRequest
from study_design_core.calibration_simulation import CalibrationDesign, compare_calibration_designs
import study_design_core.simulation as simulation


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)
TIMES = np.array([0., 2., 4.])


def reference(rates=(.2, .25), weights=(.5, .5)):
    grid = np.linspace(0., 4., 33)
    return ReferenceTrajectoryEnsemble(
        grid, np.exp(-np.asarray(rates)[:, None]*grid), [True]*len(rates),
        draw_weights=weights, interpolation='linear')


def companion():
    time = np.arange(5.)
    return prepare_time_course(time, np.exp(-.4*time),
        observation_uncertainty=ObservationUncertainty(
            .02, response_kind='mean', replicate_counts=4))[0]


def options(**updates):
    result = dict(model_name='exponential_normalized', method='value_ratio',
                  evaluation_times=[1., 2., 3.], n_samples=32,
                  random_state=321, analysis_config=CONFIG)
    result.update(updates)
    return result


def plan(replicates=3):
    errors = np.linspace(-1., 1., 2*replicates*len(TIMES)).reshape(2, replicates, len(TIMES))
    return SimulationPlan(TIMES, [0, 1], errors, metadata={'common_random_numbers': True})


class OptionalIVIVCTests(unittest.TestCase):
    def setUp(self):
        self.ref = reference()
        self.model = ObservationModel('constant_sd', .04, replicates_per_time=3)
        self.shared = plan()
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def request(self, **updates):
        values = dict(in_vitro=companion(), options=options())
        values.update(updates)
        return IVIVCAssessmentRequest(**values)

    def run_study(self, request=None, **updates):
        kwargs = dict(model_name='exponential_normalized', simulation_plan=self.shared,
                      fallback_grid_search=False, ivivc_assessment=request)
        kwargs.update(updates)
        return simulate_design(self.ref, TIMES, self.model, **kwargs)

    def assert_identical(self, first, second):
        if isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, Mapping):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_identical(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_identical(a, b)
        else:
            self.assertEqual(first, second)

    def assert_same_recovery(self, first, second):
        for item in fields(first):
            if item.name != 'ivivc_assessment':
                with self.subTest(field=item.name):
                    self.assert_identical(getattr(first, item.name), getattr(second, item.name))

    def test_enabling_assessment_leaves_all_recovery_fields_and_metadata_identical(self):
        original = self.run_study()
        enabled = self.run_study(self.request())
        self.assert_same_recovery(original, enabled)
        self.assertIsNone(original.ivivc_assessment)
        self.assertEqual(enabled.ivivc_assessment.n_simulations, 2)
        self.assertTrue(enabled.ivivc_assessment.fit_success.all())
        self.assertTrue(enabled.ivivc_assessment.metadata['uses_same_study_measurements'])

    def test_actual_replicate_rows_and_intervals_match_manual_c1_assessment(self):
        request = self.request()
        result = self.run_study(request)
        raw = (self.ref.evaluate(TIMES, draw_indices=[0, 1])[:, None, :]
               + .04*self.shared.standard_normal_errors)
        order = np.argsort(np.tile(TIMES, 3), kind='stable')
        for index, actual in enumerate(result.ivivc_assessment.assessments):
            vivo = result.ivivc_assessment.in_vivo_data[index]
            np.testing.assert_array_equal(vivo.value, raw[index].reshape(-1)[order])
            np.testing.assert_array_equal(vivo.observation_uncertainty.sigma, np.full(9, .04))
            manual_course, _ = prepare_time_course(
                np.tile(TIMES, 3)[order], raw[index].reshape(-1)[order],
                observation_uncertainty=ObservationUncertainty(.04, response_kind='individual'))
            expected = assess_ivivc_calibration(request.in_vitro, manual_course, **request.options)
            np.testing.assert_array_equal(actual.relationship.estimate, expected.relationship.estimate)
            np.testing.assert_array_equal(actual.relationship.lower, expected.relationship.lower)
            np.testing.assert_array_equal(actual.relationship.upper, expected.relationship.upper)
            self.assertEqual(actual.analysis['approach2:exponential_normalized']
                             .in_vitro_fit.fit_diagnostics['covariance_scaling'], 'absolute')

    def test_completed_means_keep_sem_and_new_batch_sem_uses_actual_future_count(self):
        completed = ExistingObservations([0.], [1.01],
            observation_uncertainty=ObservationUncertainty(.03, response_kind='mean', replicate_counts=4))
        result = self.run_study(self.request(), existing_observations=completed)
        for index, vivo in enumerate(result.ivivc_assessment.in_vivo_data):
            raw = (self.ref.evaluate([2., 4.], draw_indices=[index])[0]
                   + .04*self.shared.standard_normal_errors[index, :, 1:])
            np.testing.assert_array_equal(vivo.time, TIMES)
            np.testing.assert_allclose(vivo.value, np.r_[1.01, raw.mean(axis=0)])
            np.testing.assert_allclose(vivo.observation_uncertainty.sigma,
                                       [.03, .04/np.sqrt(3), .04/np.sqrt(3)])
            np.testing.assert_array_equal(vivo.observation_uncertainty.replicate_counts, [4, 3, 3])
        np.testing.assert_array_equal(completed.observation_uncertainty.sigma, [.03])

    def test_both_entry_points_match_with_identical_mean_batches_and_noise_plan(self):
        old_vivo, _ = prepare_time_course([0.], [1.01],
            observation_uncertainty=ObservationUncertainty(
                .03, response_kind='mean', replicate_counts=4))
        completed = ExistingObservations(old_vivo.time, old_vivo.value,
            observation_uncertainty=old_vivo.observation_uncertainty)
        request = self.request()
        study = self.run_study(request, existing_observations=completed).ivivc_assessment
        comparison = compare_calibration_designs(
            {'candidate': CalibrationDesign(in_vivo_times=[2., 4.],
                                           in_vivo_observation_model=self.model)},
            assessment_options=request.options,
            existing_in_vitro=request.in_vitro, existing_in_vivo=old_vivo,
            in_vivo_reference=self.ref, simulation_plans={'in_vivo': self.shared})
        calibration = comparison.results['candidate']
        self.assertEqual(study.n_simulations, calibration.n_simulations)
        self.assertNotIn('relationship', study.truth)
        self.assertNotIn('relationship', calibration.truth)
        for index, (study_assessment, calibration_assessment) in enumerate(
                zip(study.assessments, calibration.assessments)):
            for side in ('in_vitro_data', 'in_vivo_data'):
                first, second = getattr(study, side)[index], getattr(calibration, side)[index]
                np.testing.assert_array_equal(first.time, second.time)
                np.testing.assert_allclose(first.value, second.value)
                np.testing.assert_array_equal(first.observation_uncertainty.sigma,
                                              second.observation_uncertainty.sigma)
                np.testing.assert_array_equal(first.observation_uncertainty.replicate_counts,
                                              second.observation_uncertainty.replicate_counts)
            self.assertFalse(study_assessment.errors)
            self.assertFalse(calibration_assessment.errors)
            for attribute in ('estimate', 'lower', 'upper'):
                np.testing.assert_allclose(getattr(study_assessment.relationship, attribute),
                                           getattr(calibration_assessment.relationship, attribute))
            first = study_assessment.analysis['approach2:exponential_normalized']
            second = calibration_assessment.analysis['approach2:exponential_normalized']
            for component in ('in_vitro_fit', 'in_vivo_fit'):
                a, b = getattr(first, component), getattr(second, component)
                np.testing.assert_allclose(a.parameter_vector, b.parameter_vector)
                np.testing.assert_allclose(a.covariance, b.covariance)
            self.assertGreater(first.in_vitro_fit.covariance[0, 0], 0.)

    def test_completed_fallback_precision_matches_existing_curve_refit_assumption(self):
        completed = ExistingObservations([0.], [1.01])
        result = self.run_study(self.request(), existing_observations=completed)
        for vivo in result.ivivc_assessment.in_vivo_data:
            np.testing.assert_array_equal(vivo.observation_uncertainty.sigma, np.full(7, .04))
            self.assertIn('study_refit_observation_model_fallback', vivo.observation_uncertainty.source)
        self.assertIsNone(completed.observation_uncertainty)

    def test_cv_calibration_precision_uses_same_fixed_central_curve_sigma(self):
        self.model = ObservationModel('constant_cv', .1, replicates_per_time=3)
        reference_model = fit_reference_model_ensemble(self.ref, model_name='exponential_normalized')
        result = self.run_study(self.request(), reference_model=reference_model)
        order = np.argsort(np.tile(TIMES, 3), kind='stable')
        expected_sigma = result.metadata['refit_observation_sigma'][order]
        for index, vivo in enumerate(result.ivivc_assessment.in_vivo_data):
            actual_truth = self.ref.evaluate(TIMES, draw_indices=[index])[0]
            raw = actual_truth + .1*np.abs(actual_truth)*self.shared.standard_normal_errors[index]
            np.testing.assert_array_equal(vivo.value, raw.reshape(-1)[order])
            np.testing.assert_array_equal(vivo.observation_uncertainty.sigma, expected_sigma)

    def test_curve_failure_does_not_suppress_ivivc_assessment(self):
        original = simulation.fit_model_once
        def failed(*args, **kwargs):
            return replace(original(*args, **kwargs), success=False, message='forced curve-fit failure')
        with patch.object(simulation, 'fit_model_once', side_effect=failed):
            result = self.run_study(self.request())
        self.assertFalse(result.fit_success.any())
        self.assertTrue(result.ivivc_assessment.fit_success.all())
        self.assertTrue(all(item.relationship.point_available.all()
                            for item in result.ivivc_assessment.assessments))

    def test_ivivc_fit_failure_does_not_change_successful_curve_recovery(self):
        poor_vitro = prepare_time_course([0.], [1.], observation_uncertainty=
            ObservationUncertainty(.02, response_kind='individual'))[0]
        original = self.run_study()
        enabled = self.run_study(self.request(in_vitro=poor_vitro,
                                              options=options(model_name='exponential')))
        self.assert_same_recovery(original, enabled)
        self.assertTrue(enabled.fit_success.all())
        self.assertFalse(enabled.ivivc_assessment.fit_success.any())
        self.assertTrue(all('fit' in item.errors for item in enabled.ivivc_assessment.assessments))

    def test_measurement_generator_state_is_unchanged_by_optional_assessment(self):
        plain_rng = np.random.default_rng(905)
        enabled_rng = np.random.default_rng(905)
        original = self.run_study(simulation_plan=None, n_simulations=3, random_state=plain_rng)
        request = self.request(options=options(random_state=enabled_rng))
        enabled = self.run_study(request, simulation_plan=None, n_simulations=3,
                                 random_state=enabled_rng)
        self.assert_same_recovery(original, enabled)
        self.assertEqual(plain_rng.bit_generator.state, enabled_rng.bit_generator.state)

    def test_pairing_keeps_joint_reference_draws_and_rejects_mismatched_weights(self):
        vitro_reference = reference(rates=(.4, .5))
        result = self.run_study(self.request(in_vitro_reference=vitro_reference,
                                             paired_reference_draws=True))
        np.testing.assert_array_equal(result.ivivc_assessment.metadata['in_vitro_draw_indices'], [0, 1])
        np.testing.assert_array_equal(result.ivivc_assessment.metadata['in_vivo_draw_indices'], [0, 1])
        self.assertIn('relationship', result.ivivc_assessment.truth)
        with self.assertRaisesRegex(ValueError, 'matching draw'):
            self.run_study(self.request(in_vitro_reference=reference(rates=(.4, .5), weights=(.2, .8)),
                                       paired_reference_draws=True))

    def test_comparison_wrapper_forwards_optional_assessment(self):
        constraints = DesignConstraints(candidate_times=TIMES, replicates_per_time=3)
        result = compare_designs_by_simulation(
            self.ref, {'candidate': TIMES}, self.model,
            model_name='exponential_normalized', constraints=constraints,
            simulation_plan=self.shared, ivivc_assessment=self.request(),
            fallback_grid_search=False)
        self.assertEqual(result.results['candidate'].ivivc_assessment.n_simulations, 2)

    def test_independent_truth_sampling_does_not_pair_sources_using_the_same_seed(self):
        request = self.request(in_vitro_reference=reference(rates=(.4, .5)),
                               options=options(random_state=12345))
        result = self.run_study(request, simulation_plan=None, n_simulations=24,
                                 random_state=12345)
        indices = result.ivivc_assessment.metadata
        self.assertFalse(np.array_equal(indices['in_vitro_draw_indices'],
                                        indices['in_vivo_draw_indices']))

    def test_optional_comparison_rejects_run_specific_normalization(self):
        with self.assertRaisesRegex(ValueError, 'normaliz|scale'):
            self.run_study(self.request(options=options(selected_normalizations=['min_max'])))

    def test_companion_target_identity_is_rejected_before_calibration_copy(self):
        vitro = companion()
        with self.assertRaisesRegex(ValueError, 'independent'):
            self.run_study(self.request(in_vitro=vitro, options=options(prediction=vitro)))


if __name__ == '__main__':
    unittest.main()
