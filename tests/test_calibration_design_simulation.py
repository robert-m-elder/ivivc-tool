"""Calibration candidate comparisons preserve data, precision, and paired draws."""
from dataclasses import replace
import unittest
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core.calibration_simulation import (
    CalibrationDesign, compare_calibration_designs, create_calibration_simulation_plans,
)
from study_design_core.types import ObservationModel, ReferenceTrajectoryEnsemble, SimulationPlan


OPTIONS = dict(model_name='exponential_normalized', method='value_ratio',
               evaluation_times=[1., 3., 5.], n_samples=32, random_state=123,
               analysis_config=dict(cv_scheme='none', grid_search_num_points=3,
                                    grid_search_num_cores=1, grid_search_random_state=123))


def reference(rates=(.2,), weights=None):
    time = np.linspace(0., 8., 81)
    return ReferenceTrajectoryEnsemble(
        time, np.exp(-np.asarray(rates)[:, None]*time), np.ones(len(rates), dtype=bool),
        draw_weights=weights, interpolation='linear')


def measured(rate=.3, *, mean=True, sigma=.015):
    time = np.array([0., 2., 4.])
    return prepare_time_course(time, np.exp(-rate*time),
        observation_uncertainty=None if sigma is None else ObservationUncertainty(
            sigma, response_kind='mean' if mean else 'individual',
            replicate_counts=4 if mean else None))[0]


def vivo_design(times=(0., 2., 4., 6.), *, count=3, sd=.03):
    return CalibrationDesign(in_vivo_times=times,
        in_vivo_observation_model=ObservationModel('constant_sd', sd, count))


class CalibrationDesignSimulationTests(unittest.TestCase):
    def setUp(self):
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def compare(self, designs=None, **options):
        settings = dict(assessment_options=OPTIONS, existing_in_vitro=measured(),
                        in_vivo_reference=reference(), n_simulations=6, random_state=341)
        settings.update(options)
        return compare_calibration_designs({'study': vivo_design()} if designs is None else designs, **settings)

    def test_shared_plan_reproducibility_and_independent_environment_streams(self):
        refs = dict(in_vitro_reference=reference((.3, .35)),
                    in_vivo_reference=reference((.15, .2)))
        first_design = CalibrationDesign(
            in_vitro_times=[0., 4.], in_vitro_observation_model=ObservationModel('constant_sd', .02, 1),
            in_vivo_times=[0., 2., 6.], in_vivo_observation_model=ObservationModel('constant_sd', .03, 4))
        first = create_calibration_simulation_plans({'x': first_design}, **refs,
                                                   n_simulations=12, random_state=45)
        again = create_calibration_simulation_plans({'x': first_design}, **refs,
                                                   n_simulations=12, random_state=45)
        more_vitro = replace(first_design, in_vitro_observation_model=ObservationModel('constant_sd', .02, 9))
        changed = create_calibration_simulation_plans({'x': more_vitro}, **refs,
                                                     n_simulations=12, random_state=45)
        for side in first:
            np.testing.assert_array_equal(first[side].trajectory_draw_indices, again[side].trajectory_draw_indices)
            np.testing.assert_array_equal(first[side].standard_normal_errors, again[side].standard_normal_errors)
        np.testing.assert_array_equal(first['in_vivo'].standard_normal_errors,
                                      changed['in_vivo'].standard_normal_errors)
        self.assertEqual(first['in_vivo'].standard_normal_errors.shape, (12, 4, 3))

    def test_prospective_counts_use_shared_measurements_and_sem_once(self):
        ref = reference()
        designs = {'one': vivo_design(count=1), 'four': vivo_design(count=4)}
        comparison = self.compare(designs, in_vivo_reference=ref)
        plan = comparison.simulation_plans['in_vivo']
        for label, count in (('one', 1), ('four', 4)):
            result = comparison.results[label]
            self.assertEqual(result.n_simulations, 6)
            self.assertTrue(result.fit_success.all())
            for i, actual in enumerate(result.in_vivo_data):
                expected = ref.evaluate(actual.time, draw_indices=[plan.trajectory_draw_indices[i]])[0]
                expected += .03*np.mean(plan.standard_normal_errors[i, :count], axis=0)
                np.testing.assert_allclose(actual.value, expected)
                np.testing.assert_allclose(actual.observation_uncertainty.sigma, .03/np.sqrt(count))
                np.testing.assert_array_equal(actual.observation_uncertainty.replicate_counts,
                                              np.full(4, count))
            self.assertEqual(result.metadata['in_vivo_additional_measurements'], 4*count)

    def test_more_replicates_scale_absolute_covariance_without_shrinking_fixed_vitro(self):
        designs = {'one': vivo_design(count=1), 'four': vivo_design(count=4)}
        plan = SimulationPlan([0., 2., 4., 6.], [0, 0], np.zeros((2, 4, 4)))
        comparison = self.compare(designs, simulation_plans={'in_vivo': plan})
        first = comparison.results['one'].assessments[0].analysis['approach2:exponential_normalized']
        second = comparison.results['four'].assessments[0].analysis['approach2:exponential_normalized']
        np.testing.assert_allclose(second.in_vivo_fit.covariance, first.in_vivo_fit.covariance/4, rtol=1e-5)
        np.testing.assert_array_equal(second.in_vitro_fit.covariance, first.in_vitro_fit.covariance)
        self.assertGreater(first.in_vitro_fit.covariance[0, 0], 0.)

    def test_existing_mean_and_added_replicates_remain_independent_rows(self):
        old = measured(.2, sigma=.02)
        result = self.compare({'added': vivo_design((2., 6.), count=9)},
                              existing_in_vivo=old).results['added']
        expected_times = [0., 2., 2., 4., 6.]
        for actual in result.in_vivo_data:
            np.testing.assert_array_equal(actual.time, expected_times)
            np.testing.assert_array_equal(actual.value[[0, 1, 3]], old.value)
            np.testing.assert_allclose(actual.observation_uncertainty.sigma, [.02, .02, .01, .02, .01])
            np.testing.assert_array_equal(actual.observation_uncertainty.replicate_counts, [4, 4, 9, 4, 9])
        np.testing.assert_array_equal(old.time, [0., 2., 4.])
        np.testing.assert_array_equal(old.observation_uncertainty.sigma, [.02]*3)

    def test_existing_individuals_receive_raw_new_rows_with_individual_sd(self):
        old = measured(.2, mean=False, sigma=.02)
        result = self.compare({'added': vivo_design((2., 6.), count=3)},
                              existing_in_vivo=old).results['added']
        for actual in result.in_vivo_data:
            self.assertEqual(len(actual.time), 9)
            self.assertEqual(actual.observation_uncertainty.response_kind, 'individual')
            self.assertIsNone(actual.observation_uncertainty.replicate_counts)
            np.testing.assert_allclose(actual.observation_uncertainty.sigma,
                                       [.02, .02, .03, .03, .03, .02, .03, .03, .03])

    def test_either_environment_or_both_can_receive_additions(self):
        model = ObservationModel('constant_sd', .02, 2)
        designs = {
            'vitro': CalibrationDesign(in_vitro_times=[2., 6.], in_vitro_observation_model=model),
            'both': CalibrationDesign(in_vitro_times=[2., 6.], in_vitro_observation_model=model,
                                       in_vivo_times=[2., 6.], in_vivo_observation_model=model),
        }
        comparison = self.compare(designs, existing_in_vivo=measured(.2),
                                   in_vitro_reference=reference((.3,)), n_simulations=3)
        for label, result in comparison.results.items():
            self.assertTrue(result.fit_success.all())
            self.assertEqual(len(result.in_vitro_data[0].time), 5)
            self.assertEqual(len(result.in_vivo_data[0].time), 3 if label == 'vitro' else 5)
        np.testing.assert_array_equal(comparison.results['vitro'].in_vitro_data[0].value,
                                      comparison.results['both'].in_vitro_data[0].value)

    def test_no_addition_baseline_and_fixed_dataset_without_reference(self):
        old_vivo = measured(.2)
        comparison = self.compare({'baseline': CalibrationDesign()},
            existing_in_vivo=old_vivo, in_vivo_reference=None, n_simulations=3)
        result = comparison.results['baseline']
        self.assertEqual(comparison.simulation_plans, {})
        self.assertEqual(result.n_simulations, 3)
        self.assertTrue(result.fit_success.all())
        for dataset in result.in_vivo_data:
            np.testing.assert_array_equal(dataset.value, old_vivo.value)
        self.assertTrue(np.isnan(comparison.summaries['baseline']['relationship']['rmse']))

    def test_paired_references_keep_joint_scenarios_and_truth(self):
        refs = dict(in_vitro_reference=reference((.3, .4), [.4, .6]),
                    in_vivo_reference=reference((.15, .2), [.4, .6]))
        designs = {'vivo': vivo_design()}
        plans = create_calibration_simulation_plans(designs, **refs,
            n_simulations=12, random_state=93, paired_reference_draws=True)
        np.testing.assert_array_equal(plans['in_vitro'].trajectory_draw_indices,
                                      plans['in_vivo'].trajectory_draw_indices)
        comparison = self.compare(designs, **refs, simulation_plans=plans,
                                   paired_reference_draws=True)
        self.assertEqual(comparison.metadata['reference_draws'], 'paired')
        result = comparison.results['vivo']
        grid = np.asarray(OPTIONS['evaluation_times'])
        draw = plans['in_vivo'].trajectory_draw_indices
        expected = refs['in_vivo_reference'].evaluate(grid, draw_indices=draw)/refs['in_vitro_reference'].evaluate(grid, draw_indices=draw)
        np.testing.assert_allclose(result.truth['relationship'], expected)

    def test_reference_draws_are_independent_unless_pairing_is_explicit(self):
        plans = create_calibration_simulation_plans({'vivo': vivo_design()},
            in_vitro_reference=reference((.3, .4)), in_vivo_reference=reference((.15, .2)),
            n_simulations=40, random_state=93)
        self.assertTrue(np.any(plans['in_vitro'].trajectory_draw_indices !=
                               plans['in_vivo'].trajectory_draw_indices))

    def test_incompatible_pairs_and_supplied_plan_indices_are_rejected(self):
        refs = dict(in_vitro_reference=reference((.3, .4), [.4, .6]),
                    in_vivo_reference=reference((.15, .2), [.4, .6]))
        designs = {'vivo': vivo_design()}
        with self.assertRaisesRegex(ValueError, 'matching joint'):
            create_calibration_simulation_plans(designs, in_vitro_reference=reference((.3, .4)),
                in_vivo_reference=refs['in_vivo_reference'], paired_reference_draws=True)
        plans = create_calibration_simulation_plans(designs, **refs,
            n_simulations=4, random_state=4, paired_reference_draws=True)
        bad_vivo = SimulationPlan(plans['in_vivo'].master_times,
            1-plans['in_vivo'].trajectory_draw_indices, plans['in_vivo'].standard_normal_errors)
        with self.assertRaisesRegex(ValueError, 'matching joint scenario'):
            self.compare(designs, **refs, simulation_plans={**plans, 'in_vivo': bad_vivo},
                         paired_reference_draws=True)
        invalid = SimulationPlan([0., 2., 4., 6.], [999], np.zeros((1, 3, 4)))
        with self.assertRaisesRegex(ValueError, 'invalid generating'):
            self.compare(simulation_plans={'in_vivo': invalid})

    def test_configuration_rejects_missing_references_and_unspecified_historical_precision(self):
        with self.assertRaisesRegex(ValueError, 'explicit generating reference'):
            self.compare(in_vivo_reference=None)
        with self.assertRaisesRegex(ValueError, 'historical uncertainty'):
            self.compare(existing_in_vivo=measured(.2, sigma=None))
        with self.assertRaisesRegex(ValueError, 'existing or proposed in_vitro'):
            self.compare(existing_in_vitro=None)
        with self.assertRaises(ValueError):
            self.compare(assessment_options={**OPTIONS, 'selected_normalizations': ['min_max']})
        old = measured()
        with self.assertRaisesRegex(ValueError, 'independent'):
            self.compare(existing_in_vitro=old, assessment_options={**OPTIONS, 'prediction': old})
        for invalid_count in (0, 2.5, True):
            with self.subTest(n_simulations=invalid_count), self.assertRaises(ValueError):
                self.compare({'baseline': CalibrationDesign()}, existing_in_vivo=measured(.2),
                             in_vivo_reference=None, simulation_plans={}, n_simulations=invalid_count)

    def test_candidate_configuration_rejects_silent_time_or_count_changes(self):
        for times in ([0., 0.], [0., np.nan], [[0., 1.]]):
            with self.assertRaises(ValueError):
                vivo_design(times)
        with self.assertRaises(TypeError):
            CalibrationDesign(in_vivo_times=[0., 1.])
        with self.assertRaises(ValueError):
            CalibrationDesign(in_vivo_observation_model=ObservationModel('constant_sd', .1))
        with self.assertRaises(ValueError):
            create_calibration_simulation_plans({'study': vivo_design()},
                in_vivo_reference=reference(), n_simulations=0)

    def test_failed_sparse_fits_remain_in_experiment_denominator(self):
        options = {**OPTIONS, 'model_name': 'exponential'}
        comparison = self.compare({'sparse': vivo_design((1.,), count=1)},
                                   assessment_options=options, n_simulations=4)
        result = comparison.results['sparse']
        self.assertEqual(result.n_simulations, 4)
        self.assertEqual(result.failure_rate, 1.)
        self.assertTrue(all('fit' in item.errors for item in result.assessments))
        summary = comparison.summaries['sparse']
        self.assertEqual(summary['n_simulations'], 4)
        self.assertEqual(summary['n_fit_successful'], 0)

    def test_overflowed_measurements_are_retained_as_generation_failures(self):
        plan = SimulationPlan([0., 2., 4., 6.], [0, 0], np.full((2, 1, 4), 2.))
        comparison = self.compare({'overflow': vivo_design(count=1, sd=1.e308)},
                                   simulation_plans={'in_vivo': plan})
        result = comparison.results['overflow']
        self.assertEqual(result.n_simulations, 2)
        self.assertEqual(result.failure_rate, 1.)
        self.assertTrue(all('generation' in item.errors for item in result.assessments))


if __name__ == '__main__':
    unittest.main()
