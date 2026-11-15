"""Request integration for explicit calibration-sampling comparisons.

Every successful request uses current uploads and the real small simulation
pipeline. Preview requests and hidden server-side state are not prerequisites.
"""

import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from design_app import create_app


def curve_csv(times, rate):
    time = np.asarray(times, dtype=float)
    perturbation = np.resize([1., 1.002, .999, 1.001, .998], len(time))
    return 'time,response\n' + ''.join(
        f'{t:.17g},{y:.17g}\n' for t, y in zip(time, np.exp(-rate*time)*perturbation))


VITRO = curve_csv(np.arange(0., 9.), .2)
VIVO = curve_csv(np.arange(0., 17., 2.), .1)
TARGET = curve_csv(np.arange(0., 7.), .2)


class DesignCalibrationRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    @staticmethod
    def upload(content, filename):
        if isinstance(content, str):
            content = content.encode('utf-8')
        return io.BytesIO(content), filename

    @staticmethod
    def mapped(role, **options):
        return {role + '__profile__' + key: value for key, value in options.items()}

    def calculate(self, *, existing_vitro=VITRO, existing_vivo=None,
                  source_vitro=None, source_vivo=VIVO, target=None,
                  profile_vitro=None, profile_vivo=None,
                  existing_vitro_filename='fixed-vitro.csv',
                  source_vivo_filename='expected-vivo.csv', **changes):
        form = {
            'task': 'calibration', 'varied_environment': 'in_vivo',
            'cal_in_vitro_source': 'existing', 'cal_in_vitro_model': 'exponential_normalized',
            'cal_in_vivo_mode': 'prospective', 'cal_in_vivo_source': 'file',
            'cal_in_vivo_model': 'exponential_normalized',
            'cal_in_vivo_observation_kind': 'constant_sd',
            'cal_in_vivo_observation_value': '.02',
            'cal_in_vitro_observation_kind': 'constant_sd',
            'cal_in_vitro_observation_value': '.02',
            'cal_approach': 'approach2', 'cal_model': 'exponential_normalized',
            'cal_method': 'value_ratio', 'cal_relationship': 'on',
            'cal_evaluation_times': '1,3,5', 'cal_normalization': 'none',
            'cal_candidate_0_name': 'Base', 'cal_candidate_0_in_vivo_kind': 'manual',
            'cal_candidate_0_in_vivo_times': '0,4,8,12',
            'cal_candidate_0_in_vivo_replicates': '3',
            'reference_draws': '4', 'cal_interval_draws': '8',
            'n_simulations': '2', 'random_seed': '123',
        }
        uploads = (
            ('existing_in_vitro_file', existing_vitro, existing_vitro_filename),
            ('existing_in_vivo_file', existing_vivo, 'existing-vivo.csv'),
            ('cal_in_vitro_source_file', source_vitro, 'expected-vitro.csv'),
            ('cal_in_vivo_source_file', source_vivo, source_vivo_filename),
            ('cal_target_file', target, 'independent-target.csv'),
            ('calibration_in_vitro_sd_profile_file', profile_vitro, 'future-vitro-sd.csv'),
            ('calibration_in_vivo_sd_profile_file', profile_vivo, 'future-vivo-sd.csv'),
        )
        for field, content, filename in uploads:
            if content is not None:
                form[field] = self.upload(content, filename)
        form.update(changes)
        return self.client.post('/calculate', data=form, content_type='multipart/form-data')

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertIn('Calibration comparison', payload['html'])
        self.assertNotRegex(payload['html'], r'>\s*(?:nan|[-+]?inf(?:inity)?)\s*<')
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        return payload

    def error(self, response, field_id, status=400):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertTrue(any(item.get('field_id') == field_id for item in payload['errors']), payload)
        return payload

    def test_prospective_in_vivo_runs_with_fixed_in_vitro_without_preview(self):
        payload = self.success(self.calculate())
        for text in ('Base', 'fixed-vitro.csv', 'expected-vivo.csv', 'exponential_normalized'):
            self.assertIn(text, payload['html'])

    def test_same_seed_files_and_candidates_reproduce_the_same_result(self):
        self.assertEqual(self.success(self.calculate())['html'],
                         self.success(self.calculate())['html'])

    def test_both_environments_can_receive_prospective_measurements(self):
        self.success(self.calculate(
            existing_vitro=None, source_vitro=VITRO,
            varied_environment='both', cal_in_vitro_mode='prospective',
            cal_in_vitro_source='file', cal_candidate_0_in_vitro_kind='manual',
            cal_candidate_0_in_vitro_times='0,2,4,6',
            cal_candidate_0_in_vitro_replicates='2'))

    def test_in_vitro_only_changes_keep_the_in_vivo_environment_fixed(self):
        self.success(self.calculate(
            existing_vitro=None, existing_vivo=VIVO, source_vitro=VITRO,
            varied_environment='in_vitro', cal_in_vitro_mode='prospective',
            cal_in_vitro_source='file', cal_in_vivo_source='existing',
            cal_candidate_0_in_vitro_kind='manual',
            cal_candidate_0_in_vitro_times='0,2,4,6',
            cal_candidate_0_in_vitro_replicates='2'))

    def test_additions_preserve_completed_means_sem_and_independent_counts(self):
        from utilities.design_calibration import run_calibration_comparison

        results = []
        def record(*args, **kwargs):
            result = run_calibration_comparison(*args, **kwargs)
            results.append(result)
            return result
        with patch('utilities.design_calibration.run_calibration_comparison', side_effect=record) as calculate:
            self.success(self.calculate(
                existing_vivo=VIVO, cal_in_vivo_mode='additions',
                cal_candidate_0_in_vivo_times='4,12', **self.mapped(
                    'existing-in-vivo-file', uncertainty='mean_sem', error_source='constant',
                    error_value='.03', count_source='constant', count_value='9')))
        old = calculate.call_args.kwargs['existing_in_vivo']
        np.testing.assert_allclose(old.observation_uncertainty.sigma, .03)
        np.testing.assert_array_equal(old.observation_uncertainty.replicate_counts, [9]*9)
        for data in results[0].comparison.results['Base'].in_vivo_data:
            # Independent new means remain separate from the existing rows.
            self.assertEqual(len(data.time), len(old.time) + 2)
            for time, response in zip(old.time, old.value):
                self.assertTrue(np.any((data.time == time) & (data.value == response)))

    def test_sd_and_counts_are_resolved_once_before_calibration_dispatch(self):
        from utilities.design_calibration import run_calibration_comparison

        with patch('utilities.design_calibration.run_calibration_comparison',
                   wraps=run_calibration_comparison) as calculate:
            self.success(self.calculate(**self.mapped(
                'existing-in-vitro-file', uncertainty='mean_sd', error_source='constant',
                error_value='.06', count_source='constant', count_value='4')))
        old = calculate.call_args.kwargs['existing_in_vitro']
        np.testing.assert_allclose(old.observation_uncertainty.sigma, .03)
        np.testing.assert_array_equal(old.observation_uncertainty.replicate_counts, [4]*9)

    def test_fixed_data_are_not_replaced_by_simulated_measurements(self):
        from utilities.design_calibration import run_calibration_comparison

        results = []
        def record(*args, **kwargs):
            result = run_calibration_comparison(*args, **kwargs)
            results.append(result)
            return result
        with patch('utilities.design_calibration.run_calibration_comparison', side_effect=record) as calculate:
            self.success(self.calculate())
        old = calculate.call_args.kwargs['existing_in_vitro']
        for data in results[0].comparison.results['Base'].in_vitro_data:
            np.testing.assert_array_equal(data.time, old.time)
            np.testing.assert_array_equal(data.value, old.value)

    def test_generated_candidate_is_assessed_alongside_a_manual_schedule(self):
        payload = self.success(self.calculate(
            cal_in_vivo_first_time='0', cal_in_vivo_last_time='12',
            cal_in_vivo_time_step='2', cal_candidate_1_name='Generated',
            cal_candidate_1_in_vivo_kind='generated',
            cal_candidate_1_in_vivo_number_of_times='4',
            cal_candidate_1_in_vivo_replicates='2'))
        self.assertIn('Base', payload['html'])
        self.assertIn('Generated', payload['html'])

    def test_multiple_manual_candidates_can_compare_replicate_counts(self):
        payload = self.success(self.calculate(
            cal_candidate_1_name='More replicates', cal_candidate_1_in_vivo_kind='manual',
            cal_candidate_1_in_vivo_times='0,4,8,12',
            cal_candidate_1_in_vivo_replicates='6'))
        self.assertIn('More replicates', payload['html'])

    def test_approach1_uses_its_shared_time_mapping_assessment(self):
        payload = self.success(self.calculate(
            cal_approach='approach1', cal_model='linear', cal_method='time_mapping',
            cal_interpolation='default'))
        self.assertIn('linear', payload['html'])

    def test_target_prediction_is_optional_and_has_a_separate_file_role(self):
        self.success(self.calculate())
        payload = self.success(self.calculate(target=TARGET, cal_assess_target='on'))
        self.assertIn('independent-target.csv', payload['html'])
        self.error(self.calculate(cal_assess_target='on'), 'cal-target-file')

    def test_target_only_assessment_ignores_inactive_relationship_coordinates(self):
        payload = self.success(self.calculate(
            target=TARGET, cal_assess_target='on', cal_relationship='off',
            cal_evaluation_times='inactive invalid values'))
        self.assertIn('independent-target.csv', payload['html'])

    def test_target_sem_is_retained_without_a_second_count_division(self):
        from utilities.design_calibration import run_calibration_comparison

        with patch('utilities.design_calibration.run_calibration_comparison',
                   wraps=run_calibration_comparison) as calculate:
            self.success(self.calculate(target=TARGET, cal_assess_target='on', **self.mapped(
                'cal-target-file', uncertainty='mean_sem', error_source='constant',
                error_value='.04', count_source='constant', count_value='16')))
        target = calculate.call_args.kwargs['prediction']
        np.testing.assert_allclose(target.observation_uncertainty.sigma, .04)
        np.testing.assert_array_equal(target.observation_uncertainty.replicate_counts, [16]*7)

    def test_profiles_are_imported_only_for_a_varied_environment_using_them(self):
        self.success(self.calculate(profile_vitro='not a profile', profile_vivo='not a profile'))
        self.success(self.calculate(
            profile_vivo='time,sd\n0,.02\n12,.04\n',
            cal_in_vivo_observation_kind='time_profile_sd'))
        self.error(self.calculate(cal_in_vivo_observation_kind='time_profile_sd'),
                   'calibration-in-vivo-sd-profile-file')

    def test_fixed_environment_can_omit_a_truth_reference_for_precision_only(self):
        self.success(self.calculate(cal_in_vitro_source='none'))

    def test_current_columns_and_workbook_worksheet_are_used(self):
        stream = io.BytesIO()
        frame = pd.read_csv(io.StringIO(VIVO)).rename(columns={'time': 'elapsed', 'response': 'mass'})
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            pd.DataFrame({'unused': [1]}).to_excel(writer, sheet_name='Other', index=False)
            frame.to_excel(writer, sheet_name='Expected', index=False)
        with patch('utilities.design_calibration.run_calibration_comparison') as calculate:
            # Invalid mappings fail before the numerical function is called.
            self.error(self.calculate(**self.mapped(
                'cal-in-vivo-source-file', response_column='missing')),
                       'cal-in-vivo-source-file__profile__response_column')
            calculate.assert_not_called()
        payload = self.success(self.calculate(
            source_vivo=stream.getvalue(), source_vivo_filename='expected.xlsx',
            **{'cal-in-vivo-source-file__sheet': 'Expected'},
            **self.mapped('cal-in-vivo-source-file', time_column='elapsed', response_column='mass')))
        self.assertIn('Expected', payload['html'])

    def test_missing_required_active_files_have_role_linked_errors(self):
        self.error(self.calculate(existing_vitro=None), 'existing-in-vitro-file')
        self.error(self.calculate(source_vivo=None), 'cal-in-vivo-source-file')
        self.error(self.calculate(existing_vivo=None, cal_in_vivo_mode='additions'),
                   'existing-in-vivo-file')

    def test_inactive_study_files_and_existing_data_are_ignored(self):
        self.success(self.calculate(
            existing_vivo='not valid', source_vitro='not valid', target='not valid',
            reference_file=self.upload('invalid', 'inactive-study.csv'), has_completed='on',
            completed_target_file=self.upload('invalid', 'inactive-completed.csv'),
            measured_adaptive_mode='inactive-invalid'))

    def test_invalid_environment_roles_and_assessment_choices_have_linked_errors(self):
        for options, field in (
            ({'varied_environment': 'invalid'}, 'varied_environment'),
            ({'cal_in_vivo_mode': 'fixed'}, 'cal_in_vivo_mode'),
            ({'cal_in_vivo_source': 'none'}, 'cal_in_vivo_source'),
            ({'cal_model': 'missing'}, 'cal_model'),
            ({'cal_candidate_0_in_vivo_replicates': '0'}, 'cal_candidate_0_in_vivo_replicates'),
        ):
            with self.subTest(field=field):
                self.error(self.calculate(**options), field)

    def test_historical_precision_is_not_invented_when_adding_approach2_data(self):
        self.error(self.calculate(existing_vivo=VIVO, cal_in_vivo_mode='additions'),
                   'cal_in_vivo_historical_error')
        self.success(self.calculate(existing_vivo=VIVO, cal_in_vivo_mode='additions',
                                    cal_in_vivo_historical_error='future_model'))

    def test_uploaded_names_and_candidate_labels_are_escaped(self):
        payload = self.success(self.calculate(
            source_vivo_filename='<script>alert(1)</script>.csv',
            cal_candidate_0_name='<b>Candidate</b>'))
        self.assertIn('&lt;script&gt;', payload['html'])
        self.assertIn('&lt;b&gt;Candidate&lt;/b&gt;', payload['html'])
        self.assertNotIn('<script>', payload['html'])
        self.assertNotIn('<b>Candidate</b>', payload['html'])


if __name__ == '__main__':
    unittest.main()
