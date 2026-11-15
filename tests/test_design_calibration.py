"""Seeded calibration comparisons through the shared simulation and IVIVC APIs."""
from copy import deepcopy
import importlib.util
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import ObservationModel
from utilities.design_calibration import calibration_options, run_calibration_comparison
from utilities.design_planning import PlanningInputError


def settings(**changes):
    form = dict(task='calibration', varied_environment='in_vivo', cal_approach='approach2',
        cal_model='exponential_normalized', cal_method='value_ratio', cal_relationship='on',
        cal_evaluation_times='0,2,4', reference_draws='4', n_simulations='2', cal_interval_draws='8',
        random_seed='123', cal_in_vitro_model='exponential_normalized',
        cal_in_vivo_model='exponential_normalized', cal_in_vivo_source='file',
        cal_candidate_0_name='Candidate A', cal_candidate_0_in_vivo_kind='manual',
        cal_candidate_0_in_vivo_times='0,4,8,12', cal_candidate_0_in_vivo_replicates='3')
    form.update(changes)
    return form


def course(*, in_vivo=False, scale=1., mean=True, precision=True, short=False):
    t = np.arange(0., 9., 2.)
    y = scale * np.exp(-.2*t) * np.array([1., 1.001, .999, 1.002, .999])
    if in_vivo:
        t = 2*t
    if short:
        t, y = t[:3], y[:3]
    error = None if not precision else ResolvedObservationUncertainty(
        np.full(len(t), .025*scale), response_kind='mean' if mean else 'individual',
        replicate_counts=np.full(len(t), 4) if mean else None)
    return TimeCourse(t, y, observation_uncertainty=error)


def run(form=None, **arguments):
    defaults = dict(existing_in_vitro=course(), in_vivo_source=course(in_vivo=True))
    defaults.update(arguments)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        return run_calibration_comparison(settings() if form is None else form, **defaults)


class CalibrationFrontendAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run()

    def test_catalog_limits_calibration_assessment_to_supported_approaches(self):
        catalog = calibration_options()
        self.assertEqual({row['id'] for row in catalog['approaches']}, {'approach1', 'approach2'})
        self.assertIn('linear', {row['id'] for row in catalog['models']})

    def test_prospective_in_vivo_and_fixed_in_vitro_use_the_shared_pipeline(self):
        result = self.result
        summary = result.comparison.summaries['Candidate A']
        self.assertEqual(summary['n_fit_successful'], 2)
        self.assertEqual(result.source_details['in_vitro']['mode'], 'fixed')
        self.assertTrue(result.comparison.metadata['common_random_numbers'])
        for data in result.comparison.results['Candidate A'].in_vitro_data:
            np.testing.assert_array_equal(data.value, course().value)
            np.testing.assert_array_equal(data.observation_uncertainty.sigma, [.025]*5)
        for data in result.comparison.results['Candidate A'].in_vivo_data:
            np.testing.assert_allclose(data.observation_uncertainty.sigma, .05/np.sqrt(3))
            np.testing.assert_array_equal(data.observation_uncertainty.replicate_counts, [3]*4)
        self.assertTrue(np.isfinite(summary['relationship']['mean_interval_width']))

    def test_replicate_candidates_share_errors_and_convert_means_once(self):
        result = run(settings(cal_candidate_0_in_vivo_replicates='1',
            cal_candidate_1_name='Four replicates', cal_candidate_1_in_vivo_times='0,4,8,12',
            cal_candidate_1_in_vivo_replicates='4'))
        plan = result.comparison.simulation_plans['in_vivo']
        for label, count in (('Candidate A', 1), ('Four replicates', 4)):
            for index, data in enumerate(result.comparison.results[label].in_vivo_data):
                truth = result.references['in_vivo'].evaluate(data.time,
                    draw_indices=[plan.trajectory_draw_indices[index]])[0]
                expected = truth + .05*plan.standard_normal_errors[index, :count].mean(axis=0)
                np.testing.assert_allclose(data.value, expected)
                np.testing.assert_allclose(data.observation_uncertainty.sigma, .05/np.sqrt(count))

    def test_existing_means_and_added_replicates_preserve_separate_rows(self):
        old = course(in_vivo=True, short=True)
        before = old.value.copy()
        result = run(settings(cal_in_vivo_mode='additions',
                              cal_candidate_0_in_vivo_times='4,12'), existing_in_vivo=old)
        for data in result.comparison.results['Candidate A'].in_vivo_data:
            np.testing.assert_array_equal(data.time, [0, 4, 4, 8, 12])
            np.testing.assert_allclose(data.value[[0, 1, 3]], old.value)
            np.testing.assert_allclose(data.observation_uncertainty.sigma,
                                      [.025, .025, .05/np.sqrt(3), .025, .05/np.sqrt(3)])
            np.testing.assert_array_equal(data.observation_uncertainty.replicate_counts, [4, 4, 3, 4, 3])
        np.testing.assert_array_equal(old.value, before)

    def test_no_additions_candidate_preserves_all_existing_measurements(self):
        old = course(in_vivo=True, short=True, precision=False)
        result = run(settings(cal_in_vivo_mode='additions', cal_candidate_0_in_vivo_times=''),
                     existing_in_vivo=old)
        design = result.designs['Candidate A']
        self.assertEqual(len(design.in_vivo_times), 0)
        self.assertIsNone(design.in_vivo_observation_model)
        self.assertEqual(result.comparison.results['Candidate A'].metadata['in_vivo_additional_measurements'], 0)
        for data in result.comparison.results['Candidate A'].in_vivo_data:
            np.testing.assert_array_equal(data.value, old.value)
            self.assertIsNone(data.observation_uncertainty)

    def test_missing_historical_precision_requires_explicit_approach2_assumption(self):
        old = course(in_vivo=True, short=True, precision=False)
        form = settings(cal_in_vivo_mode='additions')
        with self.assertRaises(PlanningInputError) as caught:
            run(form, existing_in_vivo=old)
        self.assertEqual(caught.exception.field_id, 'cal_in_vivo_historical_error')
        result = run({**form, 'cal_in_vivo_historical_error':'future_model'}, existing_in_vivo=old)
        precision = result.analyzed_existing['in_vivo'].observation_uncertainty
        np.testing.assert_array_equal(precision.sigma, [.05]*3)
        self.assertEqual(precision.response_kind, 'individual')
        self.assertIsNone(precision.replicate_counts)

    def test_approach1_uses_noise_for_simulation_and_shared_unweighted_mapping(self):
        result = run(settings(cal_approach='approach1', cal_model='linear', cal_method='time_mapping',
                              cal_in_vivo_mode='additions', cal_candidate_0_in_vivo_times='4,12'),
                     existing_in_vivo=course(in_vivo=True, short=True, precision=False))
        simulated = result.comparison.results['Candidate A']
        self.assertEqual(simulated.metadata['calibration_uncertainty_usage'], 'simulation_only')
        self.assertEqual(simulated.metadata['weighting'], 'unweighted')
        self.assertTrue(all(item.analysis is not None for item in simulated.assessments))
        self.assertTrue(any('ordinary shared' in text for text in result.warnings))

    def test_generated_prospective_schedule_uses_shared_curve_recovery_optimizer(self):
        result = run(settings(cal_candidate_0_in_vivo_kind='generated',
            cal_candidate_0_in_vivo_times='inactive junk', cal_candidate_0_in_vivo_number_of_times='3',
            cal_in_vivo_first_time='0', cal_in_vivo_last_time='16', cal_in_vivo_time_step='2'))
        times = result.designs['Candidate A'].in_vivo_times
        self.assertEqual(len(times), 3)
        self.assertTrue(np.isin(times, np.arange(0, 17, 2)).all())
        self.assertIn('not optimized for IVIVC', result.generation_outcomes[0]['message'])

    def test_generated_additions_use_completed_precision_without_resampling_existing_times(self):
        old = course(in_vivo=True, short=True)
        result = run(settings(cal_in_vivo_mode='additions', cal_candidate_0_in_vivo_kind='generated',
            cal_candidate_0_in_vivo_number_of_times='2', cal_in_vivo_first_time='0',
            cal_in_vivo_last_time='16', cal_in_vivo_time_step='2'), existing_in_vivo=old)
        times = result.designs['Candidate A'].in_vivo_times
        self.assertEqual(len(times), 2)
        self.assertFalse(np.isin(times, old.time).any())
        np.testing.assert_array_equal(result.analyzed_existing['in_vivo'].observation_uncertainty.sigma, [.025]*3)

    def test_both_environments_can_vary_with_independent_shared_error_streams(self):
        result = run(settings(varied_environment='both', cal_in_vitro_mode='prospective',
            cal_in_vitro_source='file', cal_candidate_0_in_vitro_times='0,2,4,8'),
            existing_in_vitro=None, in_vitro_source=course())
        plans = result.comparison.simulation_plans
        self.assertEqual(set(plans), {'in_vitro', 'in_vivo'})
        self.assertFalse(np.array_equal(plans['in_vitro'].standard_normal_errors,
                                       plans['in_vivo'].standard_normal_errors))
        self.assertTrue(result.comparison.results['Candidate A'].fit_success.all())

    def test_in_vitro_alone_can_vary_while_in_vivo_observations_remain_fixed(self):
        old = course(in_vivo=True)
        result = run(settings(varied_environment='in_vitro', cal_in_vitro_source='file',
            cal_in_vivo_source='existing',cal_candidate_0_in_vitro_times='0,2,4,8',
            cal_candidate_0_in_vivo_times='inactive junk'),
            existing_in_vivo=old,in_vitro_source=course())
        self.assertEqual(len(result.designs['Candidate A'].in_vivo_times),0)
        for data in result.comparison.results['Candidate A'].in_vivo_data:
            np.testing.assert_array_equal(data.value,old.value)
            np.testing.assert_array_equal(data.observation_uncertainty.sigma,old.observation_uncertainty.sigma)

    def test_other_approach2_methods_use_their_own_relationship_coordinates(self):
        for method,quantity in (('tau_ratio','time_constant_ratio'),
                                ('observed_response_inverse','in_vivo_time')):
            with self.subTest(method=method):
                result = run(settings(cal_method=method,cal_evaluation_times='inactive junk',
                                      cal_response_levels='.5,.9'))
                summary = result.comparison.summaries['Candidate A']['relationship']
                self.assertEqual(summary['quantity'],quantity)
                if method == 'tau_ratio' and importlib.util.find_spec('uncertainties') is None:
                    # The application's ordinary requirements supply this
                    # dependency. Keep its actual stage failure visible in a
                    # reduced execution environment; do not substitute a method.
                    self.assertTrue(np.all(summary['point_counts'] == 0))
                    self.assertTrue(all('uncertainties' in item.errors['relationship'].message
                        for item in result.comparison.results['Candidate A'].assessments))
                else:
                    self.assertTrue(np.all(summary['point_counts'] > 0))

    def test_fixed_side_can_omit_truth_without_inventing_error_or_coverage(self):
        result = run(settings(cal_in_vitro_source='none'))
        self.assertIsNone(result.references['in_vitro'])
        summary = result.comparison.summaries['Candidate A']['relationship']
        self.assertEqual(summary['truth_counts'].sum(), 0)
        self.assertTrue(np.isnan(summary['rmse']))
        self.assertTrue(np.isfinite(summary['mean_interval_width']))

    def test_fixed_source_normalization_transports_historical_sem_once(self):
        old, src = course(scale=100.), course(in_vivo=True, scale=100.)
        result = run(settings(cal_normalization='initial_value'), existing_in_vitro=old, in_vivo_source=src)
        analyzed = result.analyzed_existing['in_vitro']
        np.testing.assert_allclose(analyzed.value, old.value/100)
        np.testing.assert_allclose(analyzed.observation_uncertainty.sigma, .025)
        np.testing.assert_array_equal(analyzed.observation_uncertainty.replicate_counts, [4]*5)
        self.assertAlmostEqual(result.source_details['in_vitro']['normalization']['slope'], .01)
        for assessment in result.comparison.results['Candidate A'].assessments:
            self.assertEqual(assessment.analysis.prepared_data.preprocessing.selected_normalizations, ())
        self.assertEqual(result.designs['Candidate A'].in_vivo_observation_model.value, .05)
        np.testing.assert_array_equal(old.value, course(scale=100.).value)

    def test_source_extrapolation_requires_explicit_opt_in(self):
        with self.assertRaises(PlanningInputError) as caught:
            run(settings(cal_candidate_0_in_vivo_times='0,8,20'))
        self.assertEqual(caught.exception.field_id, 'cal_in_vivo_allow_extrapolation')
        result = run(settings(cal_candidate_0_in_vivo_times='0,8,20', cal_in_vivo_allow_extrapolation='on'))
        self.assertEqual(result.references['in_vivo'].supported_time_range[1], 20.)
        self.assertTrue(any('explicitly allowed' in row for row in result.warnings))

    def test_minmax_and_one_minus_transport_the_same_fixed_constants_to_old_data(self):
        time = np.arange(5.)
        raw = TimeCourse(time, 10 - 2*time, observation_uncertainty=
            ResolvedObservationUncertainty(np.full(5, .25), response_kind='mean', replicate_counts=np.full(5,4)))
        generating = TimeCourse(2*time, raw.value, observation_uncertainty=raw.observation_uncertainty)
        result = run(settings(cal_approach='approach1', cal_model='linear', cal_method='time_mapping',
            cal_in_vitro_model='linear',cal_in_vivo_model='linear',cal_normalization='min_max',
            cal_one_minus='on',cal_candidate_0_in_vivo_times='0,2,4,8'),
            existing_in_vitro=raw,in_vivo_source=generating)
        np.testing.assert_allclose(result.analyzed_existing['in_vitro'].value, time/4)
        np.testing.assert_allclose(result.analyzed_existing['in_vitro'].observation_uncertainty.sigma, .25/8)
        np.testing.assert_array_equal(raw.value,10-2*time)

    def test_zero_initial_value_reports_the_shared_normalization_noop(self):
        time = np.arange(5.)
        raw = TimeCourse(time, .2*time, observation_uncertainty=
            ResolvedObservationUncertainty(np.full(5,.025),response_kind='individual'))
        generating = TimeCourse(2*time,raw.value,observation_uncertainty=raw.observation_uncertainty)
        result = run(settings(cal_approach='approach1',cal_model='linear',cal_method='time_mapping',
            cal_in_vitro_model='linear',cal_in_vivo_model='linear',cal_normalization='initial_value',
            cal_candidate_0_in_vivo_times='0,2,4,8'),existing_in_vitro=raw,in_vivo_source=generating)
        np.testing.assert_array_equal(result.analyzed_existing['in_vitro'].value,raw.value)
        self.assertTrue(any('initial source response was zero' in row for row in result.warnings))

    def test_future_sd_profiles_retain_pointwise_individual_noise(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0,8,16], profile_sd=[.02,.04,.08])
        result = run(settings(cal_in_vivo_observation_kind='time_profile_sd',
                              cal_in_vivo_observation_value='inactive junk'), in_vivo_profile=profile)
        model = result.designs['Candidate A'].in_vivo_observation_model
        self.assertEqual(model.replicates_per_time, 3)
        for data in result.comparison.results['Candidate A'].in_vivo_data:
            np.testing.assert_allclose(data.observation_uncertainty.sigma,
                                       profile.standard_deviation(data.value, times=data.time)/np.sqrt(3))

    def test_same_seed_reproduces_sources_measurements_and_summary_values(self):
        again = run()
        for side in ('in_vitro','in_vivo'):
            np.testing.assert_array_equal(again.references[side].value_draws,
                                          self.result.references[side].value_draws)
        for a,b in zip(again.comparison.results['Candidate A'].in_vivo_data,
                       self.result.comparison.results['Candidate A'].in_vivo_data):
            np.testing.assert_array_equal(a.value,b.value)

    def test_target_predictions_are_optional_and_share_no_additional_future_noise(self):
        target = course(precision=True)
        result = run(settings(cal_relationship='off', cal_assess_target='on', cal_evaluation_times='inactive junk'),
                     prediction=target)
        self.assertNotIn('relationship', result.comparison.summaries['Candidate A'])
        self.assertIn('prediction', result.comparison.summaries['Candidate A'])
        with self.assertRaises(PlanningInputError) as caught:
            run(settings(cal_approach='approach1', cal_model='linear', cal_method='time_mapping',
                         cal_assess_target='on'), prediction=target)
        self.assertEqual(caught.exception.field_id, 'cal-target-file')

    def test_direct_target_alias_to_calibration_is_rejected(self):
        existing = course()
        with self.assertRaises(PlanningInputError) as caught:
            run(settings(cal_assess_target='on'), existing_in_vitro=existing, prediction=existing)
        self.assertEqual(caught.exception.field_id, 'cal-target-file')

    def test_target_and_point_draw_budgets_fail_before_source_fitting(self):
        for size,form,field in (
            (201,settings(cal_assess_target='on'),'cal-target-file'),
            (101,settings(cal_assess_target='on',n_simulations='200',cal_interval_draws='250',
                          cal_candidate_1_name='B',cal_candidate_1_in_vivo_times='0,4,8'),
             'cal_interval_draws')):
            target = TimeCourse(np.linspace(0,8,size),np.exp(-.2*np.linspace(0,8,size)))
            with self.subTest(field=field), patch('utilities.design_calibration.fit_reference_profile') as fit:
                with self.assertRaises(PlanningInputError) as caught:
                    run(form,prediction=target)
                self.assertEqual(caught.exception.field_id,field)
                fit.assert_not_called()

    def test_inactive_fixed_environment_fields_are_ignored(self):
        result = run(settings(cal_in_vitro_mode='inactive junk',
                              cal_in_vitro_observation_value='inactive junk',
                              cal_interpolation='inactive junk'))
        self.assertTrue(result.comparison.results['Candidate A'].fit_success.all())

    def test_configuration_and_workload_errors_link_to_active_inputs(self):
        cases = [({'cal_approach':'approach3'},'cal_approach'),
                 ({'cal_relationship':'off'},'cal_relationship'),
                 ({'cal_model':'unknown'},'cal_model'),
                 ({'cal_candidate_0_in_vivo_replicates':'1.5'},'cal_candidate_0_in_vivo_replicates'),
                 ({'cal_candidate_0_in_vivo_times':'0,0,4'},'cal_candidate_0_in_vivo_times'),
                 ({'n_simulations':'10000'},'n_simulations'),
                 ({'cal_interval_draws':'10000'},'cal_interval_draws')]
        for changes, field in cases:
            with self.subTest(field=field), self.assertRaises(PlanningInputError) as caught:
                run(settings(**changes))
            self.assertEqual(caught.exception.field_id, field)
        with patch('utilities.design_calibration.fit_reference_profile') as fit:
            with self.assertRaises(PlanningInputError):
                run(settings(cal_candidate_1_name='B',cal_candidate_1_in_vivo_times='0,4,8',
                             n_simulations='200',cal_interval_draws='500'))
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
