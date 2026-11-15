"""Seeded adaptive planning checks using the shared numerical pipeline."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import EmpiricalTimeScaleDistribution, ObservationModel
from utilities.design_planning import PlanningInputError, run_measured_planning, run_reference_planning


def options(**changes):
    form = dict(task='study', reference_source='measured', has_completed='on',
                model_name='linear', first_time='0', last_time='6', time_step='1',
                min_times='3', max_times='5', fixed_times='0', reference_draws='12',
                n_simulations='5', random_seed='123', observation_kind='constant_sd',
                observation_value='.05', replicates_per_time='3', include_even='on')
    form.update(changes)
    return form


def completed(*, scale=1., precision=True, times=(0., 1., 2.)):
    time = np.asarray(times)
    values = scale * (1. - .13*time + np.array([0., .01, -.01]))
    error = (ResolvedObservationUncertainty(np.full(3, .03*scale), response_kind='mean',
                                            replicate_counts=np.full(3, 9)) if precision else None)
    return TimeCourse(time, values, observation_uncertainty=error)


def prior():
    x = np.arange(7.)
    return TimeCourse(x, 1. - .13*x + np.array([0., .02, -.01, .01, -.02, .01, 0.]),
                      observation_uncertainty=ResolvedObservationUncertainty(
                          np.full(7, .08), response_kind='individual'))


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        return function(*args, **kwargs)


class AdaptiveMeasuredPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = completed()
        cls.result = quiet(run_measured_planning, None, options(), completed=cls.data)

    def test_completed_fit_is_not_reweighted_or_used_as_an_extra_likelihood(self):
        with patch('utilities.design_planning.reweight_reference_ensemble') as update:
            result = quiet(run_measured_planning, None, options(), completed=self.data)
        update.assert_not_called()
        self.assertIsNone(result.prior_reference)
        self.assertIsNone(result.adaptive_update)
        self.assertEqual(result.source_details['adaptive']['mode'], 'refit_completed')
        self.assertFalse(result.reference.metadata.get('adaptive_reweighting', False))

    def test_completed_rows_and_sem_remain_fixed_and_are_not_resimulated(self):
        result = self.result
        np.testing.assert_array_equal(result.completed_observations.response, self.data.value)
        np.testing.assert_array_equal(result.completed_observations.observation_uncertainty.sigma, [.03]*3)
        np.testing.assert_array_equal(result.completed_observations.observation_uncertainty.replicate_counts, [9]*3)
        for simulated in result.comparison.results.values():
            for t, y in zip(self.data.time, self.data.value):
                i = np.flatnonzero(simulated.design_times == t)[0]
                np.testing.assert_array_equal(simulated.observed_means[:, i], [y]*5)
        self.assertTrue(all(t > 2 for t in result.constraints.candidate_times))

    def test_zero_addition_schedule_is_available_as_a_current_study_baseline(self):
        summary = self.result.comparison.results['Suggested · 3 times']
        np.testing.assert_array_equal(summary.design_times, self.data.time)
        self.assertEqual(summary.n_successful, 5)
        np.testing.assert_allclose(summary.parameter_estimates,
                                   np.repeat(summary.parameter_estimates[[0]], 5, axis=0))

    def test_only_completed_times_work_when_no_future_opportunities_remain(self):
        result = quiet(run_measured_planning, None,
                       options(last_time='2', max_times='3'), completed=self.data)
        self.assertEqual(result.settings['candidate_time_count'], 0)
        np.testing.assert_array_equal(result.schedules['Suggested · 3 times'], [0., 1., 2.])

    def test_completed_normalization_happens_once_for_values_and_sem(self):
        data = completed(scale=10.)
        result = quiet(run_measured_planning, None,
                       options(normalization='initial_value'), completed=data)
        np.testing.assert_allclose(result.completed_observations.response, data.value/10.)
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma, [.03]*3)
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma, [.03]*3)
        np.testing.assert_array_equal(data.value, 10*self.data.value)

    def test_missing_historical_precision_requires_an_explicit_assumption(self):
        with self.assertRaises(PlanningInputError) as caught:
            quiet(run_measured_planning, None, options(), completed=completed(precision=False))
        self.assertEqual(caught.exception.field_id, 'completed_error_mode')

    def test_explicit_cv_assumption_is_resolved_once_at_completed_responses(self):
        data = completed(precision=False)
        result = quiet(run_measured_planning, None,
                       options(completed_error_mode='future_model', observation_kind='constant_cv',
                               observation_value='.1', minimum_sd='.01'), completed=data)
        expected = .1*np.abs(data.value)
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma, expected)
        np.testing.assert_allclose(result.fit.observation_uncertainty.sigma, expected)
        self.assertEqual(result.completed_observations.observation_uncertainty.source, 'future_model_assumption')

    def test_supplied_historical_precision_wins_over_an_enabled_fallback(self):
        result = quiet(run_measured_planning, None,
                       options(completed_error_mode='future_model', observation_value='.4'), completed=self.data)
        np.testing.assert_array_equal(result.completed_observations.observation_uncertainty.sigma, [.03]*3)

    def test_separate_prior_is_reweighted_once_with_absolute_completed_sem(self):
        form = options(measured_adaptive_mode='reweight_prior')
        result = quiet(run_measured_planning, prior(), form, completed=self.data)
        raw = result.prior_reference
        predicted = raw.evaluate(self.data.time)
        logw = np.log(raw.draw_weights) -.5*np.sum(((self.data.value-predicted)/.03)**2, axis=1)
        expected = np.exp(logw-np.max(logw)); expected /= expected.sum()
        np.testing.assert_allclose(result.reference.draw_weights, expected)
        np.testing.assert_array_equal(result.reference.value_draws, raw.value_draws)
        again = quiet(run_measured_planning, prior(), form, completed=self.data)
        np.testing.assert_array_equal(result.reference.draw_weights, again.reference.draw_weights)

    def test_prior_update_does_not_renormalize_partial_completed_data(self):
        result = quiet(run_measured_planning, TimeCourse(prior().time, 10*prior().value),
                       options(measured_adaptive_mode='reweight_prior', normalization='initial_value'),
                       completed=self.data)
        np.testing.assert_array_equal(result.completed_observations.response, self.data.value)
        np.testing.assert_array_equal(result.completed_observations.observation_uncertainty.sigma, [.03]*3)

    def test_latest_completed_grid_anchor_retains_the_baseline_and_whole_horizon(self):
        result = quiet(run_measured_planning, None,
                       options(first_time_mode='latest_completed', first_time='inactive-invalid'),
                       completed=self.data)
        self.assertEqual(result.settings['first_time'], 2.)
        self.assertEqual(result.reference.supported_time_range, (0., 6.))
        np.testing.assert_array_equal(result.constraints.candidate_times, [3., 4., 5., 6.])

    def test_gap_fill_allows_earlier_unmeasured_times_without_repeating_completed_times(self):
        data = completed(times=(0., 2., 4.))
        result = quiet(run_measured_planning, None,
                       options(future_policy='gap_fill'), completed=data)
        self.assertIn(1., result.constraints.candidate_times)
        self.assertIn(3., result.constraints.candidate_times)
        self.assertFalse(any(t in result.constraints.candidate_times for t in data.time))
        self.assertTrue(all(t > 4 for t in quiet(
            run_measured_planning, None, options(), completed=data).constraints.candidate_times))

    def test_future_sd_is_not_required_at_completed_times_with_supplied_sem(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0., 2., 3., 6.],
                                   profile_sd=[0., 0., .05, .05])
        result = quiet(run_measured_planning, None,
                       options(observation_kind='time_profile_sd'), completed=self.data, profile=profile)
        self.assertTrue(all(s.n_successful == 5 for s in result.comparison.summaries.values()))

    def test_exclusions_restrict_new_times_without_removing_completed_measurements(self):
        result = quiet(run_measured_planning, None,
                       options(excluded_times='1,3,4'), completed=self.data)
        np.testing.assert_array_equal(result.constraints.candidate_times, [5., 6.])
        np.testing.assert_array_equal(result.constraints.fixed_times, [0., 1., 2.])
        with self.assertRaises(PlanningInputError) as caught:
            quiet(run_measured_planning, None, options(excluded_times='5', fixed_times='0,5',
                                                      min_times='4'), completed=self.data)
        self.assertEqual(caught.exception.field_id, 'excluded_times')

    def test_invalid_adaptive_roles_and_count_bounds_have_linked_errors(self):
        cases = [({'completed_error_mode':'bad'}, 'completed_error_mode'),
                 ({'future_policy':'bad'}, 'future_policy'),
                 ({'first_time_mode':'bad'}, 'first_time_mode'),
                 ({'measured_adaptive_mode':'bad'}, 'measured_adaptive_mode'),
                 ({'min_times':'2'}, 'min_times'),
                 ({'fixed_times':'0,.5'}, 'fixed_times')]
        for changes, field in cases:
            with self.subTest(field=field), self.assertRaises(PlanningInputError) as caught:
                quiet(run_measured_planning, None, options(**changes), completed=self.data)
            self.assertEqual(caught.exception.field_id, field)

    def test_mixed_timing_keeps_infeasible_sizes_separate_from_feasible_suggestions(self):
        result = quiet(run_measured_planning, None, options(
            min_times='4', timing_0_kind='window', timing_0_lower_time='2',
            timing_0_upper_time='4', timing_0_label='Early window',
            timing_1_kind='latest_before', timing_1_upper_time='5',
            timing_1_label='Late point', timing_2_kind='latest_before',
            timing_2_upper_time='4', timing_2_label='Earlier distinct point'), completed=self.data)
        outcomes = {row['number_of_times']: row for row in result.timing_details['outcomes']}
        self.assertEqual(outcomes[4]['status'], 'infeasible')
        self.assertEqual(outcomes[5]['status'], 'feasible')
        self.assertNotIn('Suggested · 4 times', result.schedules)
        self.assertIn(5., result.schedules['Suggested · 5 times'])
        self.assertEqual(result.timing_details['evaluations']['Evenly spaced · 4 times'].status,
                         'unsatisfied')

    def test_unresolved_timing_can_return_statuses_without_simulations(self):
        result = quiet(run_measured_planning, None, options(
            include_even='off', timing_0_kind='latest_before',
            timing_0_upper_mode='milestone', timing_0_upper_fraction='.99',
            timing_0_upper_quantile='.99'), completed=self.data)
        self.assertIsNone(result.comparison)
        self.assertEqual(result.schedules, {})
        self.assertTrue(all(row['status'] != 'feasible' for row in result.timing_details['outcomes']))

    def test_gap_filling_with_timing_is_explicitly_rejected(self):
        with self.assertRaises(PlanningInputError) as caught:
            quiet(run_measured_planning, None, options(
                future_policy='gap_fill', timing_0_kind='latest_before',
                timing_0_upper_time='5'), completed=self.data)
        self.assertEqual(caught.exception.field_id, 'future_policy')


class AdaptiveExternalReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.linspace(0., 8., 9)
        y = np.exp(-.2*t)*[1.,1.001,.999,1.002,1.,.999,1.001,1.,.999]
        cls.vitro, cls.vivo, cls.target = TimeCourse(t, y), TimeCourse(2*t, y), TimeCourse(t, np.exp(-.2*t))
        cls.completed = TimeCourse([0., 1., 2.], [1., .89, .79], observation_uncertainty=
                                   ResolvedObservationUncertainty([.03]*3, response_kind='mean', replicate_counts=[9]*3))
        cls.distribution = EmpiricalTimeScaleDistribution([1.5, 2., 3.], weights=[1., 2., 1.])

    def run_external(self, source):
        form = options(reference_source=source, ivivc_approach='approach2',
                       ivivc_model='exponential_normalized', ivivc_method='value_ratio',
                       scaling_model='exponential_normalized')
        kwargs = (dict(in_vitro=self.vitro, in_vivo=self.vivo, prediction=self.target)
                  if source == 'ivivc' else dict(scaling_source=self.vitro, distribution=self.distribution))
        return quiet(run_reference_planning, form, completed=self.completed, **kwargs)

    def test_ivivc_and_scaling_reweight_original_priors_and_hold_completed_rows(self):
        for source in ('ivivc', 'scaling'):
            with self.subTest(source=source):
                result = self.run_external(source)
                self.assertIsNotNone(result.adaptive_update)
                self.assertIsNotNone(result.prior_reference)
                self.assertTrue(result.reference.metadata['adaptive_reweighting'])
                np.testing.assert_array_equal(result.reference.value_draws, result.prior_reference.value_draws)
                for simulated in result.comparison.results.values():
                    np.testing.assert_allclose(simulated.observed_means[:, :3], [[1., .89, .79]]*5)
                self.assertTrue(all(t > 2. for t in result.constraints.candidate_times))

    def test_empirical_scaling_retains_scale_marginals_and_update_diagnostics(self):
        result = self.run_external('scaling')
        self.assertIn('time_scaling_weight_summary', result.adaptive_update.metadata)
        np.testing.assert_array_equal(result.reference.metadata['time_scale_ratios'], self.distribution.ratios)
        weights = result.adaptive_update.metadata['time_scaling_weight_summary']
        np.testing.assert_allclose(weights['current_weights'].sum(), 1.)
        self.assertIsNotNone(result.source_details['adaptive']['best_standardized_residual_rms'])

    def test_reference_support_errors_identify_completed_data_or_baseline_not_the_future_start(self):
        target = TimeCourse(self.target.time, .8 * self.target.value)
        for times, baseline, field in (([3., 4.], '0', 'baseline_time'),
                                       ([1., 4.], '3', 'completed-target-file')):
            historical = TimeCourse(times, [.8, .7], observation_uncertainty=
                ResolvedObservationUncertainty([.03, .03], response_kind='individual'))
            with self.subTest(field=field), self.assertRaises(PlanningInputError) as caught:
                quiet(run_reference_planning, options(
                    reference_source='ivivc', ivivc_approach='approach2',
                    ivivc_model='exponential_normalized', ivivc_method='observed_response_inverse',
                    first_time_mode='latest_completed', last_time='8', fixed_times='',
                    baseline_time=baseline), in_vitro=self.vitro, in_vivo=self.vivo,
                    prediction=target, completed=historical)
            self.assertEqual(caught.exception.field_id, field)
            self.assertIn('No baseline or requested time was moved', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
