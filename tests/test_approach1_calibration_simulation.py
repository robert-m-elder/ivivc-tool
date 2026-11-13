"""Approach 1 calibration comparisons reuse simulated rows and ordinary fits."""
import unittest
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import ObservationModel, ReferenceTrajectoryEnsemble, SimulationPlan
from study_design_core.calibration_assessment import assess_ivivc_calibration
from study_design_core.calibration_simulation import (
    CalibrationDesign, compare_calibration_designs,
)


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)
OPTIONS = dict(approach_id='approach1', model_name='linear',
               evaluation_times=[.5, 1., 2., 3.], n_samples=32,
               random_state=123, analysis_config=CONFIG)


def reference(rate=.1):
    times = np.linspace(0., 12., 121)
    return ReferenceTrajectoryEnsemble(
        times, np.exp(-rate*times)[None, :], [True], interpolation='linear')


def measured(rate=.2, times=(0., 1., 2., 3., 4.), *, sigma=.015, mean=True):
    times = np.asarray(times)
    return prepare_time_course(
        times, np.exp(-rate*times),
        observation_uncertainty=(None if sigma is None else ObservationUncertainty(
            sigma, response_kind='mean' if mean else 'individual',
            replicate_counts=4 if mean else None)),
    )[0]


def vivo_design(times=(0., 2., 4., 6., 8.), *, count=3, sd=.01):
    return CalibrationDesign(in_vivo_times=times,
        in_vivo_observation_model=ObservationModel('constant_sd', sd, count))


class Approach1CalibrationSimulationTests(unittest.TestCase):
    def setUp(self):
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        self.addCleanup(context.__exit__, None, None, None)

    def compare(self, designs=None, **changes):
        settings = dict(assessment_options=OPTIONS, existing_in_vitro=measured(),
                        in_vivo_reference=reference(), n_simulations=4,
                        random_state=4321)
        settings.update(changes)
        return compare_calibration_designs(
            {'study': vivo_design()} if designs is None else designs, **settings)

    def test_prospective_measurements_match_manual_ordinary_assessment(self):
        comparison = self.compare()
        result = comparison.results['study']
        self.assertTrue(result.fit_success.all())
        self.assertEqual(result.metadata['weighting'], 'unweighted')
        self.assertEqual(result.metadata['calibration_uncertainty_usage'], 'simulation_only')
        plan = comparison.simulation_plans['in_vivo']
        for experiment, actual in enumerate(result.assessments):
            vitro = result.in_vitro_data[experiment]
            vivo = result.in_vivo_data[experiment]
            raw = (reference().evaluate(vivo.time, draw_indices=[0])[0]
                   + .01*plan.standard_normal_errors[experiment, :3])
            np.testing.assert_allclose(vivo.value, raw.mean(axis=0))
            np.testing.assert_allclose(vivo.observation_uncertainty.sigma,
                                       .01/np.sqrt(3))
            manual = assess_ivivc_calibration(vitro, vivo, **OPTIONS)
            self.assertFalse(actual.errors)
            for name in ('estimate', 'lower', 'upper'):
                np.testing.assert_array_equal(getattr(actual.relationship, name),
                                              getattr(manual.relationship, name))
            self.assertFalse(actual.analysis.prepared_data.has_observation_uncertainty)

    def test_additions_to_unweighted_history_keep_raw_rows_and_precision_sources(self):
        old = measured(.1, (0., 2., 4.), sigma=None)
        design = vivo_design((4., 6., 8.), count=2)
        comparison = self.compare({'added': design}, existing_in_vivo=old)
        result = comparison.results['added']
        plan = comparison.simulation_plans['in_vivo']
        self.assertTrue(result.fit_success.all())
        for experiment, vivo in enumerate(result.in_vivo_data):
            np.testing.assert_array_equal(vivo.time, [0., 2., 4., 4., 4., 6., 6., 8., 8.])
            np.testing.assert_array_equal(vivo.value[:3], old.value)
            self.assertIsNone(vivo.observation_uncertainty)
            source = result.assessments[experiment].metadata['calibration_batches']['in_vivo']
            self.assertIsNot(source['existing'], old)
            self.assertIsNone(source['existing'].observation_uncertainty)
            added = source['additions']
            np.testing.assert_array_equal(added.time, [4., 6., 8., 4., 6., 8.])
            expected = (reference().evaluate([4., 6., 8.], draw_indices=[0])[0]
                        + .01*plan.standard_normal_errors[experiment, :2])
            np.testing.assert_allclose(added.value, expected.reshape(-1))
            np.testing.assert_array_equal(added.observation_uncertainty.sigma,
                                          np.full(6, .01))
        self.assertIsNone(old.observation_uncertainty)
        np.testing.assert_array_equal(old.time, [0., 2., 4.])

    def test_mean_batches_keep_old_sem_and_add_independent_means_at_same_time(self):
        old = measured(.1, (0., 2., 4.), sigma=.025)
        result = self.compare({'added': vivo_design((4., 6., 8.), count=9)},
                              existing_in_vivo=old).results['added']
        self.assertTrue(result.fit_success.all())
        for experiment, vivo in enumerate(result.in_vivo_data):
            np.testing.assert_array_equal(vivo.time, [0., 2., 4., 4., 6., 8.])
            np.testing.assert_array_equal(vivo.value[:3], old.value)
            np.testing.assert_allclose(vivo.observation_uncertainty.sigma,
                                       [.025, .025, .025, .01/3, .01/3, .01/3])
            np.testing.assert_array_equal(vivo.observation_uncertainty.replicate_counts,
                                          [4, 4, 4, 9, 9, 9])
            batches = result.assessments[experiment].metadata['calibration_batches']['in_vivo']
            np.testing.assert_array_equal(batches['existing'].observation_uncertainty.sigma,
                                          np.full(3, .025))
            self.assertEqual(batches['additions'].observation_uncertainty.response_kind, 'mean')

    def test_mixed_deterministic_additions_preserve_known_historical_precision(self):
        old = measured(.1, (0., 2., 4.), sigma=.02)
        result = self.compare({'added': vivo_design((6., 8.), sd=0)},
                              existing_in_vivo=old).results['added']
        self.assertTrue(result.fit_success.all())
        for experiment, vivo in enumerate(result.in_vivo_data):
            self.assertIsNone(vivo.observation_uncertainty)
            batches = result.assessments[experiment].metadata['calibration_batches']['in_vivo']
            np.testing.assert_array_equal(batches['existing'].observation_uncertainty.sigma,
                                          np.full(3, .02))
            self.assertIsNone(batches['additions'].observation_uncertainty)
            np.testing.assert_array_equal(vivo.value[:3], old.value)

    def test_raw_known_batches_remain_individual_rows_without_pooling(self):
        old = measured(.1, (0., 2., 4.), sigma=.02, mean=False)
        result = self.compare({'added': vivo_design((4., 6.), count=3)},
                              existing_in_vivo=old).results['added']
        for vivo in result.in_vivo_data:
            np.testing.assert_array_equal(vivo.time, [0., 2., 4., 4., 4., 4., 6., 6., 6.])
            np.testing.assert_array_equal(vivo.value[:3], old.value)
            self.assertEqual(vivo.observation_uncertainty.response_kind, 'individual')
            np.testing.assert_allclose(vivo.observation_uncertainty.sigma,
                                       [.02, .02, .02, .01, .01, .01, .01, .01, .01])

    def test_candidates_share_prefix_replicates_and_reproduce_measurements(self):
        designs = {'one': vivo_design(count=1), 'four': vivo_design(count=4)}
        first = self.compare(designs)
        again = self.compare(designs)
        shared = first.simulation_plans['in_vivo']
        np.testing.assert_array_equal(shared.standard_normal_errors,
                                      again.simulation_plans['in_vivo'].standard_normal_errors)
        for name, count in (('one', 1), ('four', 4)):
            for experiment, actual in enumerate(first.results[name].in_vivo_data):
                expected = (reference().evaluate(actual.time, draw_indices=[0])[0]
                            + .01*shared.standard_normal_errors[experiment, :count].mean(axis=0))
                np.testing.assert_allclose(actual.value, expected)
                np.testing.assert_array_equal(actual.value,
                                              again.results[name].in_vivo_data[experiment].value)

    def test_either_or_both_environments_can_vary_with_same_fixed_history(self):
        observation = ObservationModel('constant_sd', .005, 3)
        old_vitro = measured(.2, (0., 1., 2.))
        old_vivo = measured(.1, (0., 2., 4.))
        designs = {
            'vitro': CalibrationDesign(in_vitro_times=[3., 4.],
                                       in_vitro_observation_model=observation),
            'vivo': CalibrationDesign(in_vivo_times=[6., 8.],
                                      in_vivo_observation_model=observation),
            'both': CalibrationDesign(in_vitro_times=[3., 4.], in_vivo_times=[6., 8.],
                                      in_vitro_observation_model=observation,
                                      in_vivo_observation_model=observation),
        }
        comparison = self.compare(designs, existing_in_vitro=old_vitro,
                                  existing_in_vivo=old_vivo,
                                  in_vitro_reference=reference(.2))
        for name, result in comparison.results.items():
            self.assertTrue(result.fit_success.all(), name)
            for data, old in ((result.in_vitro_data, old_vitro),
                              (result.in_vivo_data, old_vivo)):
                for actual in data:
                    np.testing.assert_array_equal(actual.value[:3], old.value)
        for side in ('in_vitro_data', 'in_vivo_data'):
            varied = 'vitro' if side == 'in_vitro_data' else 'vivo'
            for first, both in zip(getattr(comparison.results[varied], side),
                                   getattr(comparison.results['both'], side)):
                np.testing.assert_array_equal(first.value, both.value)

    def test_existing_data_only_baseline_does_not_generate_measurements(self):
        old_vitro, old_vivo = measured(), measured(.1, (0., 2., 4., 6., 8.))
        result = self.compare({'baseline': CalibrationDesign()},
                              existing_in_vitro=old_vitro,
                              existing_in_vivo=old_vivo).results['baseline']
        self.assertFalse(result.metadata['calibration_measurement_noise_simulated'])
        self.assertTrue(result.fit_success.all())
        for actual in result.assessments:
            np.testing.assert_array_equal(actual.relationship.estimate,
                                          result.assessments[0].relationship.estimate)
        for actual in result.in_vivo_data:
            np.testing.assert_array_equal(actual.value, old_vivo.value)

    def test_exact_time_scaling_is_recovered_and_scored_against_reference_truth(self):
        options = dict(OPTIONS, prediction=TimeCourse([.5, 1.5, 3.], [.9, .8, .5]))
        design = CalibrationDesign(
            in_vitro_times=[0., 1., 2., 3., 4.],
            in_vivo_times=[0., 2., 4., 6., 8.],
            in_vitro_observation_model=ObservationModel('constant_sd', 0., 1),
            in_vivo_observation_model=ObservationModel('constant_sd', 0., 1))
        comparison = compare_calibration_designs(
            {'exact': design}, assessment_options=options,
            in_vitro_reference=reference(.2), in_vivo_reference=reference(.1),
            n_simulations=3, random_state=321)
        result = comparison.results['exact']
        self.assertTrue(result.fit_success.all())
        np.testing.assert_allclose(result.truth['relationship'],
                                   np.tile(2*np.asarray(OPTIONS['evaluation_times']), (3, 1)))
        np.testing.assert_allclose(result.truth['prediction'], np.tile([1., 3., 6.], (3, 1)))
        for actual in result.assessments:
            np.testing.assert_allclose(actual.relationship.estimate,
                                       2*np.asarray(OPTIONS['evaluation_times']), atol=1e-10)
        self.assertLess(comparison.summaries['exact']['relationship']['rmse'], 1e-7)

    def test_approach2_retains_historical_precision_requirement(self):
        options = dict(OPTIONS, approach_id='approach2',
                       model_name='exponential_normalized', method='value_ratio')
        with self.assertRaisesRegex(ValueError, 'compatible historical uncertainty'):
            self.compare({'added': vivo_design((6., 8.))},
                         assessment_options=options,
                         existing_in_vivo=measured(.1, (0., 2., 4.), sigma=None))

    def test_logged_time_comparison_keeps_target_and_truth_on_analyzed_axes(self):
        target = TimeCourse([1.5, 3.], [.9, .8])
        settings = dict(OPTIONS, evaluation_times=np.log10([.5, 1., 2., 3.]),
                        selected_scalings=['log_x'], prediction=target)
        comparison = self.compare(
            {'logged': vivo_design((1., 2., 4., 6., 8.))},
            assessment_options=settings,
            existing_in_vitro=measured(.2, (1., 2., 3., 4., 5.)),
            in_vitro_reference=reference(.2))
        result = comparison.results['logged']
        self.assertTrue(result.fit_success.all())
        np.testing.assert_allclose(result.truth['relationship'],
                                   np.tile(np.log10([1., 2., 4., 6.]), (4, 1)), atol=1e-10)
        np.testing.assert_allclose(result.truth['prediction'],
                                   np.tile(np.log10([3., 6.]), (4, 1)), atol=1e-10)
        for actual in result.assessments:
            self.assertFalse(actual.errors)
            np.testing.assert_array_equal(actual.prediction.coordinate, np.log10(target.time))

    def test_response_log_fallback_preserves_target_and_true_time_coordinates(self):
        vitro = TimeCourse([0., 1., 2.], [1., .5, 0.])
        vivo = TimeCourse([0., 1., 2., 3., 4.], [1., .75, .5, .25, 0.])
        refs = dict(in_vitro_reference=ReferenceTrajectoryEnsemble(
                        vitro.time, [vitro.value], [True], interpolation='linear'),
                    in_vivo_reference=ReferenceTrajectoryEnsemble(
                        vivo.time, [vivo.value], [True], interpolation='linear'))
        target = TimeCourse([.5, 1.], [-1., -2.])
        settings = dict(OPTIONS, evaluation_times=[.5, 1.], prediction=target,
                        selected_scalings=['log_y'], fallback_invalid_log_scaling=True)
        comparison = self.compare(
            {'baseline': CalibrationDesign()}, assessment_options=settings,
            existing_in_vitro=vitro, existing_in_vivo=vivo, **refs)
        result = comparison.results['baseline']
        np.testing.assert_allclose(result.truth['prediction'], [[1., 2.]]*4)
        for actual in result.assessments:
            self.assertFalse(actual.errors)
            self.assertEqual(actual.analysis.prepared_data.preprocessing.applied_scalings, ())
            self.assertTrue(actual.analysis.prepared_data.preprocessing.warnings)
            np.testing.assert_array_equal(actual.prediction.coordinate, target.time)
            np.testing.assert_array_equal(actual.prediction.trajectory.in_vivo_value, target.value)


if __name__ == '__main__':
    unittest.main()
