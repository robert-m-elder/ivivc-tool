"""HTTP integration for completed-study updates and timing requirements.

Requests use current uploads and run the real small planning calculation.
Historical observations must remain fixed; importing a completed mean never
divides its supplied SEM by the replicate count a second time.
"""

import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from design_app import create_app


def curve_csv(times, values):
    return 'time,response\n' + ''.join(
        f'{time:.17g},{value:.17g}\n' for time, value in zip(times, values))


TIMES = np.arange(5.)
COMPLETED_VALUES = np.exp(-.1 * TIMES) + np.array([0., .003, -.002, .001, -.001])
COMPLETED = 'time,response,sem,n\n' + ''.join(
    f'{time:.17g},{value:.17g},.03,4\n'
    for time, value in zip(TIMES, COMPLETED_VALUES))
PRIOR = curve_csv(np.arange(0., 9., 2.), np.exp(-.1 * np.arange(0., 9., 2.)))
TARGET = curve_csv(np.arange(13.), np.exp(-.2 * np.arange(13.)))
CALIBRATION = 't_iv,y_iv,t_vivo,y_vivo\n' + ''.join(
    f'{time:.17g},{value:.17g},{2*time:.17g},{value:.17g}\n'
    for time, value in zip(
        np.arange(0., 13., 2.),
        np.exp(-.2 * np.arange(0., 13., 2.)) *
        np.array([1., 1.001, .999, 1.002, 1., .999, 1.001]),
    ))
RATIOS = 'ratio,weight\n2,1\n3,3\n'


class DesignAdaptiveRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    @staticmethod
    def upload(content, filename):
        if isinstance(content, str):
            content = content.encode('utf-8')
        return io.BytesIO(content), filename

    @staticmethod
    def completed_options(**changes):
        return {'completed-target-file__profile__' + key: value
                for key, value in changes.items()}

    def calculate(self, reference='measured', *, completed=COMPLETED,
                  completed_filename='completed.csv', prior=PRIOR, profile=None,
                  precision=True, **changes):
        form = {
            'task': 'study', 'reference_source': reference, 'has_completed': 'on',
            'measured_adaptive_mode': 'refit_completed',
            'completed_error_mode': 'supplied', 'future_policy': 'future_only',
            'first_time_mode': 'specified', 'baseline_time': '0',
            'model_name': 'linear', 'normalization': 'none',
            'observation_kind': 'constant_sd', 'observation_value': '.03',
            'replicates_per_time': '2', 'first_time': '0', 'last_time': '8',
            'time_step': '1', 'min_times': '6', 'max_times': '7', 'fixed_times': '0',
            'reference_draws': '8', 'n_simulations': '4', 'random_seed': '123',
            'ivivc_approach': 'approach2', 'ivivc_model': 'exponential_normalized',
            'ivivc_method': 'value_ratio', 'scaling_model': 'exponential_normalized',
        }
        if completed is not None:
            form['completed_target_file'] = self.upload(completed, completed_filename)
        if precision:
            form.update(self.completed_options(
                uncertainty='mean_sem', error_source='column', error_column='sem',
                count_source='column', count_column='n'))
        if prior is not None:
            form['reference_file'] = self.upload(prior, 'prior.csv')
        if reference == 'ivivc':
            form['ivivc_calibration_file'] = self.upload(CALIBRATION, 'calibration.csv')
            form['prediction_file'] = self.upload(TARGET, 'target.csv')
        elif reference == 'scaling':
            form['scaling_source_file'] = self.upload(TARGET, 'source.csv')
            form['time_scale_file'] = self.upload(RATIOS, 'ratios.csv')
        if profile is not None:
            form['future_sd_profile_file'] = self.upload(profile, 'future-sd.csv')
        form.update(changes)
        return self.client.post('/calculate', data=form, content_type='multipart/form-data')

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertIsInstance(payload['html'], str)
        self.assertNotRegex(payload['html'], r'>\s*(?:nan|[-+]?inf(?:inity)?)\s*<')
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        return payload

    def error(self, response, field_id, status=400):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertTrue(any(error.get('field_id') == field_id for error in payload['errors']), payload)
        return payload

    def capture(self, **options):
        from utilities.design_planning_results import present_planning_result

        with patch('utilities.design_planning_results.present_planning_result',
                   wraps=present_planning_result) as present:
            payload = self.success(self.calculate(**options))
        return payload, present.call_args.args[0]

    def test_completed_only_refit_needs_no_prior_reference_upload(self):
        payload, result = self.capture(prior=None)
        self.assertIn('completed.csv', payload['html'])
        self.assertIsNone(result.adaptive_update)
        self.assertEqual(result.completed_observations.n_time_points, 5)
        self.assertIsNone(result.prior_reference)

    def test_refit_mode_ignores_an_unrelated_or_malformed_prior_file(self):
        self.success(self.calculate(prior='not a usable source file'))

    def test_all_three_sources_support_completed_observations(self):
        for reference in ('measured', 'ivivc', 'scaling'):
            with self.subTest(reference=reference):
                payload, result = self.capture(reference=reference)
                self.assertIn('completed.csv', payload['html'])
                self.assertEqual(result.completed_observations.n_observations, 5)
                if reference != 'measured':
                    self.assertIsNotNone(result.adaptive_update)

    def test_separate_measured_prior_is_updated_without_pooling_studies(self):
        from utilities.design_planning import run_measured_planning

        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            payload, result = self.capture(measured_adaptive_mode='reweight_prior')
        source = calculate.call_args.args[0]
        np.testing.assert_array_equal(source.time, [0, 2, 4, 6, 8])
        self.assertIsNotNone(result.adaptive_update)
        self.assertIn('prior.csv', payload['html'])
        self.assertIn('completed.csv', payload['html'])

    def test_current_completed_means_keep_sem_and_counts_unchanged(self):
        _, result = self.capture()
        precision = result.completed_observations.observation_uncertainty
        self.assertEqual(precision.response_kind, 'mean')
        np.testing.assert_allclose(precision.sigma, [.03] * 5)
        np.testing.assert_array_equal(precision.replicate_counts, [4] * 5)

    def test_completed_sd_and_counts_are_converted_once_before_backend_dispatch(self):
        from utilities.design_planning import run_measured_planning

        content = COMPLETED.replace('sem,n', 'sd,n')
        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            self.success(self.calculate(completed=content, **self.completed_options(
                uncertainty='mean_sd', error_source='column', error_column='sd',
                count_source='column', count_column='n')))
        completed = calculate.call_args.kwargs['completed']
        np.testing.assert_allclose(completed.observation_uncertainty.sigma, [.015] * 5)

    def test_completed_observations_are_not_resimulated_or_given_the_future_count(self):
        _, result = self.capture(replicates_per_time='7')
        for simulated in result.comparison.results.values():
            for time, response in zip(TIMES, COMPLETED_VALUES):
                column = np.flatnonzero(np.isclose(simulated.design_times, time))[0]
                np.testing.assert_allclose(simulated.observed_means[:, column], response)
        np.testing.assert_array_equal(
            result.completed_observations.observation_uncertainty.replicate_counts, [4] * 5)

    def test_missing_completed_precision_requires_an_explicit_error_assumption(self):
        self.error(self.calculate(precision=False), 'completed_error_mode')
        _, result = self.capture(precision=False, completed_error_mode='future_model')
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma,
                                   [.03] * 5)

    def test_reported_completed_precision_wins_over_the_optional_future_assumption(self):
        _, result = self.capture(completed_error_mode='future_model', observation_value='.2')
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma,
                                   [.03] * 5)

    def test_completed_only_normalization_is_applied_once_to_values_and_sem(self):
        completed = 'time,response,sem,n\n' + ''.join(
            f'{time:.17g},{100*value:.17g},3,4\n'
            for time, value in zip(TIMES, COMPLETED_VALUES))
        _, result = self.capture(completed=completed, normalization='initial_value')
        np.testing.assert_allclose(result.completed_observations.response, COMPLETED_VALUES)
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma,
                                   [.03] * 5)

    def test_future_only_and_latest_completed_shortcut_preserve_the_evaluation_baseline(self):
        _, result = self.capture(first_time_mode='latest_completed', first_time='inactive-invalid')
        self.assertLessEqual(result.reference.time_grid[0], 0.)
        for times in result.schedules.values():
            new = times[~np.isin(times, TIMES)]
            self.assertTrue(np.all(new > 4.))

    def test_gap_filling_keeps_missing_earlier_elapsed_times_available(self):
        completed = 'time,response,sem,n\n0,1,.03,4\n2,.82,.03,4\n4,.67,.03,4\n'
        _, result = self.capture(completed=completed, future_policy='gap_fill',
                                 min_times='4', max_times='5')
        self.assertIn(1., result.constraints.candidate_times)
        self.assertIn(3., result.constraints.candidate_times)

    def test_current_completed_mapping_and_file_are_reimported(self):
        self.success(self.calculate())
        self.error(self.calculate(completed=None), 'completed-target-file')
        self.error(self.calculate(completed='time,response\n0,1\n2,\n', precision=False),
                   'completed-target-file__profile__response_column')
        self.error(self.calculate(**self.completed_options(response_column='missing')),
                   'completed-target-file__profile__response_column')

    def test_completed_workbook_uses_its_selected_worksheet(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame({'unused': [1]}).to_excel(writer, sheet_name='Other', index=False)
            pd.read_csv(io.StringIO(COMPLETED)).to_excel(writer, sheet_name='Current', index=False)
        self.error(self.calculate(completed=output.getvalue(), completed_filename='completed.xlsx'),
                   'completed-target-file__sheet')
        payload = self.success(self.calculate(
            completed=output.getvalue(), completed_filename='completed.xlsx',
            **{'completed-target-file__sheet': 'Current'}))
        self.assertIn('Current', payload['html'])

    def test_completed_data_are_ignored_when_the_active_request_is_prospective(self):
        self.success(self.calculate(has_completed='', completed='malformed inactive file',
                                    min_times='3', max_times='4'))

    def test_future_sd_profiles_remain_separate_from_completed_sem(self):
        _, result = self.capture(reference='scaling', observation_kind='time_profile_sd',
                                 profile='time,sd\n0,.05\n8,.1\n')
        np.testing.assert_allclose(result.completed_observations.observation_uncertainty.sigma,
                                   [.03] * 5)
        self.assertEqual(result.observation_model.kind, 'time_profile_sd')

    def test_numeric_window_and_latest_before_requests_are_resolved_and_reported(self):
        options = {
            'timing_0_kind': 'window', 'timing_0_label': 'Early future point',
            'timing_0_lower_mode': 'numeric', 'timing_0_lower_time': '4',
            'timing_0_upper_mode': 'numeric', 'timing_0_upper_time': '7',
            'timing_0_minimum_count': '1', 'timing_0_count_scope': 'new',
            'timing_1_kind': 'latest_before', 'timing_1_label': 'Latest before week seven',
            'timing_1_lower_mode': 'numeric', 'timing_1_lower_time': '4',
            'timing_1_upper_mode': 'numeric', 'timing_1_upper_time': '7',
        }
        payload, result = self.capture(**options)
        self.assertIn('Latest before week seven', payload['html'])
        self.assertTrue(result.timing_details['requirements'])
        for label, times in result.schedules.items():
            if label.startswith('Suggested'):
                self.assertIn(7., times)

    def test_unreached_milestone_produces_a_reviewable_status_without_fake_recovery(self):
        payload, result = self.capture(**{
            'timing_0_kind': 'window', 'timing_0_lower_mode': 'numeric',
            'timing_0_lower_time': '4', 'timing_0_upper_mode': 'milestone',
            'timing_0_upper_fraction': '.9', 'timing_0_upper_quantile': '.25',
            'timing_0_minimum_count': '1', 'timing_0_count_scope': 'new',
        })
        self.assertRegex(payload['html'], r'(?i)horizon|unavailable|unresolved')
        self.assertIsNone(result.comparison)

    def test_invalid_active_timing_field_has_an_actionable_error(self):
        self.error(self.calculate(**{
            'timing_0_kind': 'window', 'timing_0_lower_mode': 'numeric',
            'timing_0_lower_time': '4', 'timing_0_upper_mode': 'milestone',
            'timing_0_upper_fraction': '.5', 'timing_0_upper_quantile': 'nan',
        }), 'timing_0_upper_quantile')

    def test_completed_filename_and_timing_labels_remain_escaped(self):
        payload = self.success(self.calculate(
            completed_filename='<script>alert(1)</script>.csv', **{
                'timing_0_kind': 'window', 'timing_0_label': '<b>Window</b>',
                'timing_0_lower_mode': 'numeric', 'timing_0_lower_time': '4',
                'timing_0_upper_mode': 'numeric', 'timing_0_upper_time': '8',
            }))
        self.assertIn('&lt;script&gt;', payload['html'])
        self.assertIn('&lt;b&gt;Window&lt;/b&gt;', payload['html'])
        self.assertNotIn('<script>', payload['html'])
        self.assertNotIn('<b>Window</b>', payload['html'])


if __name__ == '__main__':
    unittest.main()
