"""Small seeded end-to-end checks for measured-reference browser planning."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import ObservationModel
from utilities.design_planning import PlanningInputError, model_options, run_measured_planning


def settings(**changes):
    result = dict(model_name='linear', first_time='0', last_time='4', time_step='.5',
                  min_times='3', max_times='4', fixed_times='0', replicates_per_time='3',
                  reference_draws='8', n_simulations='6', random_seed='12345',
                  observation_kind='constant_sd', observation_value='.05', include_even='on')
    result.update(changes)
    return result


def source(*, scale=1., error=True, kind='individual', counts=None):
    uncertainty = None if not error else ObservationUncertainty(
        scale*np.array([.05, .2, .08, .1, .06]), response_kind=kind, replicate_counts=counts)
    return prepare_time_course(np.arange(5.), scale*np.array([1.05, .78, .72, .5, .44]),
                               observation_uncertainty=uncertainty)[0]


class MeasuredPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The shared fitter emits an unrelated empty-CV mean warning with CV off.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            cls.result = run_measured_planning(source(), settings(manual_schedules='Named: 0, 2, 4'))

    def run_plan(self, data=None, **changes):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            return run_measured_planning(source() if data is None else data, settings(**changes))

    def test_registry_includes_linear_and_registered_kinetic_models(self):
        options = model_options()
        self.assertIn('linear', {item['id'] for item in options})
        self.assertIn('mass_loss_one_phase_normalized', {item['id'] for item in options})
        self.assertTrue(all(item['parameter_count'] >= 1 and item['label'] for item in options))

    def test_default_model_does_not_assume_responses_are_already_normalized(self):
        data = TimeCourse(np.arange(5.), 100 * np.exp(-np.arange(5.) / 3))
        form = settings(observation_value='2')
        form.pop('model_name')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            result = run_measured_planning(data, form)
        np.testing.assert_allclose(result.reference.central_curve[0], 100., rtol=1e-5)
        np.testing.assert_array_equal(result.analyzed_source.value, data.value)

    def test_complete_workflow_compares_counts_fixed_even_and_manual_schedules(self):
        result = self.result
        self.assertEqual(len(result.schedules), 5)
        self.assertEqual(result.settings['candidate_time_count'], 9)
        self.assertTrue(result.comparison.metadata['common_random_numbers'])
        for label, summary in result.comparison.summaries.items():
            self.assertIn(0., summary.design_times)
            self.assertIn(len(summary.design_times), (3, 4))
            self.assertEqual(summary.n_simulations, 6)
            self.assertEqual(summary.n_successful, 6)
            self.assertTrue(np.isfinite(summary.mean_irmse), label)
            self.assertEqual(summary.failure_rate, 0.)
        even = result.comparison.results['Evenly spaced · 3 times']
        manual = result.comparison.results['Manual · Named']
        np.testing.assert_array_equal(even.integrated_rmse, manual.integrated_rmse)

    def test_reference_fit_uses_absolute_supplied_observation_variances(self):
        data = source()
        matrix = np.column_stack([np.ones(len(data.time)), data.time])
        precision = 1/data.observation_uncertainty.sigma**2
        covariance = np.linalg.inv(matrix.T @ (matrix * precision[:, None]))
        estimate = covariance @ (matrix.T @ (data.value * precision))
        np.testing.assert_allclose(self.result.fit.parameter_vector, estimate, atol=1e-7)
        np.testing.assert_allclose(self.result.fit.covariance, covariance, rtol=1e-5)
        self.assertFalse(self.result.reference.metadata['future_observation_variability_included'])

    def test_supplied_sem_counts_are_not_divided_again(self):
        individual = self.result
        means = self.run_plan(source(kind='mean', counts=16))
        np.testing.assert_allclose(means.fit.covariance, individual.fit.covariance)
        np.testing.assert_allclose(means.fit.parameter_vector, individual.fit.parameter_vector)
        np.testing.assert_array_equal(means.analyzed_source.observation_uncertainty.replicate_counts, np.full(5, 16))
        self.assertEqual(means.analyzed_source.observation_uncertainty.uncertainty_kind, 'sem')

    def test_initial_normalization_transports_sd_once_and_future_sd_uses_analyzed_scale(self):
        data = source(scale=10.)
        result = self.run_plan(data, normalization='initial_value', observation_value='.02')
        np.testing.assert_allclose(result.analyzed_source.value, data.value / data.value[0])
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma,
                                   data.observation_uncertainty.sigma / data.value[0])
        self.assertEqual(result.observation_model.value, .02)
        self.assertTrue(any('constants are treated as fixed' in text for text in result.warnings))

    def test_minmax_and_reversal_reuse_shared_preprocessing(self):
        data = source()
        result = self.run_plan(data, normalization='min_max', one_minus='on')
        span = np.ptp(data.value)
        np.testing.assert_allclose(result.analyzed_source.value, 1-(data.value-data.value.min())/span)
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma, data.observation_uncertainty.sigma/span)
        self.assertEqual(result.preprocessing.selected_normalizations, ('min_max', 'one_minus'))

    def test_zero_initial_response_reports_the_shared_normalization_noop(self):
        data, _ = prepare_time_course(
            np.arange(5.), [0., .2, .41, .59, .8],
            observation_uncertainty=ObservationUncertainty(.03, response_kind='individual'))
        original_values = data.value.copy()
        result = self.run_plan(data, normalization='initial_value')
        np.testing.assert_array_equal(result.analyzed_source.value, original_values)
        np.testing.assert_array_equal(data.value, original_values)
        np.testing.assert_array_equal(result.analyzed_source.observation_uncertainty.sigma,
                                      data.observation_uncertainty.sigma)
        self.assertIn('Initial response was zero; initial-value normalization was not applied.', result.warnings)

    def test_unspecified_historical_precision_remains_unspecified(self):
        result = self.run_plan(source(error=False))
        self.assertIsNone(result.fit.observation_uncertainty)
        self.assertEqual(result.fit.fit_diagnostics['covariance_scaling'], 'residual_scaled')
        self.assertIsNone(result.reference.metadata['source_observation_uncertainty'])

    def test_historical_and_future_uncertainty_are_separate(self):
        changed = self.run_plan(observation_value='.1')
        np.testing.assert_array_equal(changed.fit.parameter_vector, self.result.fit.parameter_vector)
        np.testing.assert_array_equal(changed.reference.value_draws, self.result.reference.value_draws)
        self.assertNotEqual(changed.comparison.summaries['Suggested · 3 times'].mean_irmse,
                            self.result.comparison.summaries['Suggested · 3 times'].mean_irmse)

    def test_replicates_are_applied_by_simulation_without_changing_reference(self):
        changed = self.run_plan(replicates_per_time='1')
        self.assertEqual(changed.observation_model.replicates_per_time, 1)
        self.assertEqual(changed.constraints.replicates_per_time, 1)
        np.testing.assert_array_equal(changed.reference.value_draws, self.result.reference.value_draws)
        self.assertEqual(changed.constraints.total_experimental_units(3), 3)

    def test_same_seed_reproduces_schedule_reference_and_simulation(self):
        repeated = self.run_plan()
        np.testing.assert_array_equal(repeated.reference.value_draws, self.result.reference.value_draws)
        for label, result in repeated.comparison.results.items():
            np.testing.assert_array_equal(result.integrated_rmse,
                                          self.result.comparison.results[label].integrated_rmse)

    def test_profile_reuses_typed_nearest_time_model_with_planned_counts(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0., 2., 4.],
                                   profile_sd=[.01, .05, .2], replicates_per_time=1)
        result = run_measured_planning(source(), settings(observation_kind='time_profile_sd',
                                      observation_value='inactive', minimum_sd='inactive'), profile=profile)
        self.assertEqual(result.observation_model.replicates_per_time, 3)
        np.testing.assert_array_equal(result.observation_model.profile_sd, profile.profile_sd)
        self.assertEqual(profile.replicates_per_time, 1)
        self.assertEqual(result.observation_model.minimum_sd, 0.)

    def test_small_absolute_sd_is_not_silently_replaced_by_a_hidden_floor(self):
        result = self.run_plan(observation_value='.00001', minimum_sd='not active')
        self.assertEqual(result.observation_model.value, .00001)
        self.assertEqual(result.observation_model.minimum_sd, 0.)
        np.testing.assert_array_equal(result.observation_model.standard_deviation([0., 1.]), [1e-5, 1e-5])

    def test_constant_cv_retains_explicit_floor(self):
        result = self.run_plan(observation_kind='constant_cv', observation_value='.1', minimum_sd='.02')
        self.assertEqual(result.observation_model.kind, 'constant_cv')
        np.testing.assert_array_equal(result.observation_model.standard_deviation([0., 1.]), [.02, .1])

    def test_planning_beyond_measured_times_is_allowed_with_an_explicit_caution(self):
        result = self.run_plan(last_time='6')
        self.assertEqual(result.reference.supported_time_range, (0., 6.))
        self.assertTrue(any('extrapolation' in text for text in result.warnings))

    def test_non_grid_last_bound_is_not_invented_as_candidate(self):
        result = self.run_plan(last_time='4.2', time_step='1')
        np.testing.assert_array_equal(result.constraints.candidate_times, np.arange(5.))
        self.assertEqual(result.reference.supported_time_range, (0., 4.2))

    def test_invalid_inputs_have_actionable_field_ids_before_any_fit(self):
        cases = [
            ({'task': 'calibration'}, 'task'), ({'reference_source': 'ivivc'}, 'reference_source'),
            ({'has_completed': 'on'}, 'has_completed'), ({'model_name': 'unregistered'}, 'model_name'),
            ({'normalization': 'log_y'}, 'normalization'), ({'observation_kind': 'unknown'}, 'observation_kind'),
            ({'observation_value': '0'}, 'observation_value'), ({'first_time': '-1'}, 'first_time'),
            ({'last_time': '0'}, 'last_time'), ({'time_step': '0'}, 'time_step'),
            ({'time_step': '.0001'}, 'time_step'), ({'min_times': '4.5'}, 'min_times'),
            ({'max_times': '2'}, 'max_times'), ({'max_times': '20'}, 'max_times'),
            ({'replicates_per_time': '0'}, 'replicates_per_time'), ({'n_simulations': 'NaN'}, 'n_simulations'),
            ({'n_simulations': '1001'}, 'n_simulations'), ({'reference_draws': '501'}, 'reference_draws'),
            ({'random_seed': '-1'}, 'random_seed'), ({'fixed_times': '0,,2'}, 'fixed_times'),
            ({'fixed_times': '0,0'}, 'fixed_times'), ({'fixed_times': '0,6'}, 'fixed_times'),
            ({'fixed_times': '0,1,2,3'}, 'min_times'),
            ({'manual_schedules': 'Missing label separator'}, 'manual_schedules'),
            ({'manual_schedules': 'Missing fixed: 1, 2, 4'}, 'manual_schedules'),
            ({'manual_schedules': 'Not feasible: 0, .2, 4'}, 'manual_schedules'),
            ({'manual_schedules': 'Repeated: 0,2,4\nRepeated: 0,1,4'}, 'manual_schedules'),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes), patch('utilities.design_planning.fit_reference_profile') as fit:
                with self.assertRaises(PlanningInputError) as caught:
                    self.run_plan(**changes)
                self.assertEqual(caught.exception.field_id, expected)
                fit.assert_not_called()

    def test_workload_limits_are_checked_before_fitting(self):
        with patch('utilities.design_planning.fit_reference_profile') as fit:
            with self.assertRaises(PlanningInputError) as caught:
                self.run_plan(first_time='0', last_time='29', time_step='1', min_times='1',
                              max_times='30', n_simulations='1000')
            self.assertEqual(caught.exception.field_id, 'n_simulations')
            fit.assert_not_called()

    def test_zero_profile_sd_at_selectable_time_is_not_given_an_unrequested_floor(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0., 4.], profile_sd=[0., .1])
        with self.assertRaises(PlanningInputError) as caught:
            run_measured_planning(source(), settings(observation_kind='time_profile_sd'), profile=profile)
        self.assertEqual(caught.exception.field_id, 'future-sd-profile-file')

    def test_missing_profile_and_too_few_reference_observations_have_linked_errors(self):
        for data, changes, field in [
            (source(), {'observation_kind': 'time_profile_sd'}, 'future-sd-profile-file'),
            (TimeCourse([0., 1.], [1., .5]), {}, 'model_name'),
            (TimeCourse([1., 1., 1.], [1., .5, .7]), {}, 'reference-file'),
        ]:
            with self.subTest(field=field), self.assertRaises(PlanningInputError) as caught:
                self.run_plan(data, **changes)
            self.assertEqual(caught.exception.field_id, field)

    def test_unavailable_parameter_covariance_does_not_silently_drop_uncertainty(self):
        fit = self.result.fit
        with patch('utilities.design_planning.fit_reference_profile', return_value=fit), \
             patch.object(fit, 'covariance', np.full((2, 2), np.nan)):
            with self.assertRaises(PlanningInputError) as caught:
                self.run_plan()
        self.assertIn('covariance is unavailable', str(caught.exception))
        self.assertEqual(caught.exception.field_id, 'model_name')

    def test_even_spacing_failure_is_a_linked_planning_error(self):
        with patch('utilities.design_planning.equally_spaced_design', side_effect=ValueError('Unavailable comparison')):
            with self.assertRaises(PlanningInputError) as caught:
                self.run_plan()
        self.assertEqual(caught.exception.field_id, 'include_even')


if __name__ == '__main__':
    unittest.main()
