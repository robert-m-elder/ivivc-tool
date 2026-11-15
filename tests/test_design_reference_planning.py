"""Small seeded end-to-end planning checks for the additional source paths."""

import unittest
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import EmpiricalTimeScaleDistribution, ObservationModel
from utilities.design_planning import run_reference_planning


def form(source, **changes):
    settings = dict(
        task='study', reference_source=source, model_name='linear',
        normalization='none', first_time='0', last_time='4', time_step='1',
        min_times='3', max_times='4', fixed_times='0', reference_draws='8',
        n_simulations='4', random_seed='12345', observation_kind='constant_sd',
        observation_value='.05', replicates_per_time='2', include_even='on',
        ivivc_approach='approach2', ivivc_model='exponential_normalized',
        ivivc_method='value_ratio', scaling_model='exponential_normalized',
    )
    settings.update(changes)
    return settings


def inputs(*, scale=1., means=False):
    time = np.linspace(0., 8., 9)
    perturbation = np.array([1., 1.001, .999, 1.002, 1., .999, 1.001, 1., .999])
    response = scale * np.exp(-.2 * time) * perturbation
    uncertainty = (ObservationUncertainty(.02 * scale, response_kind='mean',
                                          replicate_counts=9) if means else None)
    vitro = prepare_time_course(time, response, observation_uncertainty=uncertainty)[0]
    vivo = prepare_time_course(2 * time, response, observation_uncertainty=uncertainty)[0]
    target = prepare_time_course(time, scale * np.exp(-.2 * time),
                                 observation_uncertainty=uncertainty)[0]
    return vitro, vivo, target


def run(settings, **options):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        return run_reference_planning(settings, **options)


class AdditionalReferencePlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vitro, cls.vivo, cls.target = inputs()
        cls.distribution = EmpiricalTimeScaleDistribution([2., 3., 999.], weights=[1., 3., 0.])
        cls.ivivc = run(form('ivivc', manual_schedules='Named: 0,2,4'),
                        in_vitro=cls.vitro, in_vivo=cls.vivo, prediction=cls.target)
        cls.scaling = run(form('scaling', manual_schedules='Named: 0,2,4'),
                          scaling_source=cls.vitro, distribution=cls.distribution)

    def test_both_sources_complete_the_shared_schedule_and_simulation_workflow(self):
        for result in (self.ivivc, self.scaling):
            with self.subTest(source=result.settings['reference_source']):
                self.assertEqual(len(result.schedules), 5)
                self.assertEqual(result.reference_model.model_name, 'linear')
                self.assertTrue(result.comparison.metadata['common_random_numbers'])
                self.assertFalse(result.reference.metadata['future_observation_variability_included'])
                for summary in result.comparison.summaries.values():
                    self.assertEqual(summary.n_simulations, 4)
                    self.assertEqual(summary.n_successful, 4)
                    self.assertTrue(np.isfinite(summary.mean_irmse))
                    self.assertIn(0., summary.design_times)
                # Identical named and generated schedules receive identical errors.
                even = result.comparison.results['Evenly spaced · 3 times']
                manual = result.comparison.results['Manual · Named']
                np.testing.assert_array_equal(even.design_times, manual.design_times)
                np.testing.assert_array_equal(even.observed_means, manual.observed_means)
                np.testing.assert_array_equal(even.integrated_rmse, manual.integrated_rmse)

    def test_source_and_study_analysis_models_are_independent(self):
        self.assertEqual(self.ivivc.source_details['selection'], 'approach2:exponential_normalized')
        self.assertEqual(self.ivivc.reference_model.model_name, 'linear')
        self.assertEqual(self.scaling.fit.model_name, 'exponential_normalized')
        self.assertEqual(self.scaling.reference_model.model_name, 'linear')
        # The IVIVC reference remains the shared prediction, not the study model fit.
        np.testing.assert_allclose(self.ivivc.reference.central_curve,
                                   np.exp(-.1 * self.ivivc.reference.time_grid), rtol=.003)

    def test_future_simulation_uses_a_separate_random_stream_from_source_sampling(self):
        for result in (self.ivivc, self.scaling):
            plan = result.comparison.simulation_plan
            self.assertNotEqual(plan.metadata['random_state'], result.settings['random_seed'])
            repeated = np.random.default_rng(plan.metadata['random_state'])
            active = np.flatnonzero(result.reference.valid_draws & (result.reference.draw_weights > 0))
            weights = result.reference.draw_weights[active]
            repeated.choice(active, size=plan.n_simulations, p=weights / weights.sum())
            np.testing.assert_array_equal(
                repeated.standard_normal(plan.standard_normal_errors.shape),
                plan.standard_normal_errors,
            )

    def test_empirical_reference_retains_zero_weight_records_and_exact_ratio_masses(self):
        reference = self.scaling.reference
        self.assertEqual(reference.n_draws, 16)
        ratio_indices = reference.metadata['draw_ratio_indices']
        self.assertAlmostEqual(reference.draw_weights[ratio_indices == 0].sum(), .25)
        self.assertAlmostEqual(reference.draw_weights[ratio_indices == 1].sum(), .75)
        self.assertNotIn(2, ratio_indices)
        np.testing.assert_array_equal(reference.metadata['time_scale_ratios'], [2., 3., 999.])
        for simulated in self.scaling.comparison.results.values():
            expected = reference.value_draws[simulated.trajectory_draw_indices]
            np.testing.assert_allclose(simulated.true_curves, expected)

    def test_identical_seed_reproduces_reference_schedules_and_measurements(self):
        for kind, original, options in (
            ('ivivc', self.ivivc, dict(in_vitro=self.vitro, in_vivo=self.vivo, prediction=self.target)),
            ('scaling', self.scaling, dict(scaling_source=self.vitro, distribution=self.distribution)),
        ):
            with self.subTest(source=kind):
                repeated = run(form(kind, manual_schedules='Named: 0,2,4'), **options)
                np.testing.assert_array_equal(original.reference.value_draws, repeated.reference.value_draws)
                for name, simulated in original.comparison.results.items():
                    other = repeated.comparison.results[name]
                    np.testing.assert_array_equal(simulated.design_times, other.design_times)
                    np.testing.assert_array_equal(simulated.observed_means, other.observed_means)
                    np.testing.assert_array_equal(simulated.integrated_rmse, other.integrated_rmse)

    def test_future_noise_changes_measurements_without_changing_source_uncertainty(self):
        for kind, options in (
            ('ivivc', dict(in_vitro=self.vitro, in_vivo=self.vivo, prediction=self.target)),
            ('scaling', dict(scaling_source=self.vitro, distribution=self.distribution)),
        ):
            with self.subTest(source=kind):
                low = run(form(kind, observation_value='.025'), **options)
                high = run(form(kind, observation_value='.1'), **options)
                np.testing.assert_array_equal(low.reference.value_draws, high.reference.value_draws)
                for name, result in low.comparison.results.items():
                    other = high.comparison.results[name]
                    np.testing.assert_array_equal(result.true_curves, other.true_curves)
                    np.testing.assert_allclose(other.observed_means - other.expected_values,
                                               4 * (result.observed_means - result.expected_values),
                                               atol=1e-15)

    def test_scaling_normalization_transports_reported_sem_once_and_preserves_counts(self):
        vitro, _, _ = inputs(scale=100., means=True)
        result = run(form('scaling', normalization='initial_value', observation_value='.04'),
                     scaling_source=vitro, distribution=self.distribution)
        np.testing.assert_allclose(result.fit.response, vitro.value / 100.)
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma, .02)
        np.testing.assert_array_equal(result.fit.observation_uncertainty.replicate_counts, [9] * 9)
        self.assertEqual(result.observation_model.value, .04)
        np.testing.assert_allclose(vitro.observation_uncertainty.sigma, 2.)
        self.assertEqual(result.reference.metadata['source_observation_uncertainty']['uncertainty_kind'], 'sem')

    def test_future_sd_profile_controls_new_reference_study_refits(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0., 2., 4.],
                                   profile_sd=[.02, .04, .08])
        result = run(form('ivivc', observation_kind='time_profile_sd'), in_vitro=self.vitro,
                     in_vivo=self.vivo, prediction=self.target, profile=profile)
        self.assertEqual(result.observation_model.replicates_per_time, 2)
        for simulation in result.comparison.results.values():
            self.assertEqual(simulation.metadata['refit_objective'], 'weighted_least_squares')
            expected = result.observation_model.standard_deviation(
                np.ones(len(simulation.design_times)), times=simulation.design_times)
            np.testing.assert_array_equal(simulation.metadata['refit_observation_sigma'],
                                           np.tile(expected, 2))
        np.testing.assert_array_equal(result.reference.value_draws, self.ivivc.reference.value_draws)


if __name__ == '__main__':
    unittest.main()
