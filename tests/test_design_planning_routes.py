"""HTTP checks for the first complete, measured-reference planning workflow.

These requests deliberately calculate without an earlier preview: every
calculation must reinterpret the uploaded measurements and current mappings.
"""

import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from design_app import create_app


SOURCE = 'age,mass\n0,1.01\n1,.84\n2,.71\n3,.54\n4,.41\n'


class DesignPlanningRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    def calculate(self, content=SOURCE, filename='prior-measurements.csv',
                  profile=None, profile_filename='future-sd.csv', **changes):
        form = {
            'task': 'study', 'reference_source': 'measured',
            'model_name': 'linear', 'normalization': 'none',
            'observation_kind': 'constant_sd', 'observation_value': '.03',
            'minimum_sd': '.001', 'replicates_per_time': '2',
            'first_time': '0', 'last_time': '4', 'time_step': '1',
            'min_times': '3', 'max_times': '4', 'fixed_times': '0',
            'reference_draws': '8', 'n_simulations': '4', 'random_seed': '123',
        }
        if content is not None:
            if isinstance(content, str):
                content = content.encode('utf-8')
            form['reference_file'] = (io.BytesIO(content), filename)
        if profile is not None:
            if isinstance(profile, str):
                profile = profile.encode('utf-8')
            form['future_sd_profile_file'] = (io.BytesIO(profile), profile_filename)
        form.update(changes)
        return self.client.post('/calculate', data=form,
                                content_type='multipart/form-data')

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertIsInstance(payload['html'], str)
        self.assertNotRegex(payload['html'], r'>\s*(?:nan|[-+]?inf(?:inity)?)\s*<')
        self.assertNotIn(': NaN', response.get_data(as_text=True))
        self.assertNotIn(': Infinity', response.get_data(as_text=True))
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        return payload

    def error(self, response, field_id, status=400):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertTrue(any(item.get('field_id') == field_id
                            for item in payload['errors']), payload)
        return payload

    @staticmethod
    def mapped(**options):
        return {'reference-file__profile__' + key: value for key, value in options.items()}

    def test_calculation_runs_the_measured_workflow_without_a_preview_request(self):
        payload = self.success(self.calculate())
        self.assertIn('prior-measurements.csv', payload['html'])
        self.assertRegex(payload['html'], r'(?i)linear')
        self.assertIn('<table', payload['html'])
        self.assertRegex(payload['html'], r'(?i)IRMSE')

    def test_the_same_seed_and_current_upload_produce_the_same_comparison(self):
        first = self.success(self.calculate())
        second = self.success(self.calculate())
        self.assertEqual(first['html'], second['html'])

    def test_calculation_reimports_current_columns_and_preserves_source_values(self):
        from utilities.design_planning import run_measured_planning

        data = 'wrong,age,mass\n999,0,1.01\n999,1,.84\n999,2,.71\n999,3,.54\n999,4,.41\n'
        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            self.success(self.calculate(data, **self.mapped(time_column='age', response_column='mass')))
        source = calculate.call_args.args[0]
        np.testing.assert_array_equal(source.time, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(source.value, [1.01, .84, .71, .54, .41])
        self.assertIsNone(source.observation_uncertainty)

    def test_source_sd_and_counts_convert_to_sem_once_before_planning(self):
        from utilities.design_planning import run_measured_planning

        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            self.success(self.calculate(**self.mapped(
                uncertainty='mean_sd', error_source='constant', error_value='.06',
                count_source='constant', count_value='4')))
        uncertainty = calculate.call_args.args[0].observation_uncertainty
        self.assertEqual(uncertainty.response_kind, 'mean')
        np.testing.assert_allclose(uncertainty.sigma, [.03] * 5)
        np.testing.assert_array_equal(uncertainty.replicate_counts, [4] * 5)

    def test_supplied_sem_with_count_is_not_divided_again(self):
        from utilities.design_planning import run_measured_planning

        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            self.success(self.calculate(**self.mapped(
                uncertainty='mean_sem', error_source='constant', error_value='.06',
                count_source='constant', count_value='9')))
        uncertainty = calculate.call_args.args[0].observation_uncertainty
        np.testing.assert_allclose(uncertainty.sigma, [.06] * 5)
        np.testing.assert_array_equal(uncertainty.replicate_counts, [9] * 5)

    def test_missing_file_and_malformed_current_file_are_not_a_cached_preview(self):
        self.success(self.calculate())
        self.error(self.calculate(content=None), 'reference-file')
        self.error(self.calculate('age,mass\n0,1\n1,\n'),
                   'reference-file__profile__response_column')

    def test_explicitly_invalid_mapping_and_precision_have_linked_errors(self):
        self.error(self.calculate(**self.mapped(response_column='missing')),
                   'reference-file__profile__response_column')
        self.error(self.calculate(**self.mapped(
            uncertainty='mean_sd', error_source='constant', error_value='.05')),
                   'reference-file__profile__count_source')
        self.error(self.calculate(**self.mapped(
            uncertainty='individual_sd', error_source='constant', error_value='0')),
                   'reference-file__profile__error_value')

    def test_calibration_task_uses_its_own_required_files(self):
        self.error(self.calculate(content=None, task='calibration'),
                   'existing-in-vitro-file')

    def test_invalid_settings_and_oversized_work_are_linked_to_their_fields(self):
        for options, field in (
            ({'model_name': 'not-a-model'}, 'model_name'),
            ({'time_step': '0'}, 'time_step'),
            ({'max_times': '2'}, 'max_times'),
            ({'observation_value': '-.05'}, 'observation_value'),
            ({'replicates_per_time': '2.5'}, 'replicates_per_time'),
            ({'n_simulations': '1000000000'}, 'n_simulations'),
            ({'reference_draws': '1000000000'}, 'reference_draws'),
        ):
            with self.subTest(options=options):
                self.error(self.calculate(**options), field)

    def test_manual_and_evenly_spaced_comparisons_are_reported(self):
        payload = self.success(self.calculate(
            include_even='on', manual_schedules='My comparison: 0,2,4'))
        self.assertIn('My comparison', payload['html'])
        self.assertRegex(payload['html'], r'(?i)evenly')

    def test_time_profile_is_imported_as_future_individual_variability(self):
        from utilities.design_planning import run_measured_planning

        with patch('utilities.design_planning.run_measured_planning',
                   wraps=run_measured_planning) as calculate:
            self.success(self.calculate(observation_kind='time_profile_sd',
                                        profile='week,error\n0,.02\n2,.04\n4,.08\n'))
        profile = calculate.call_args.kwargs['profile']
        self.assertEqual(profile.kind, 'time_profile_sd')
        np.testing.assert_allclose(profile.profile_times, [0, 2, 4])
        np.testing.assert_allclose(profile.profile_sd, [.02, .04, .08])
        self.assertIsNone(calculate.call_args.args[0].observation_uncertainty)

    def test_time_profile_missing_file_and_invalid_order_have_linked_errors(self):
        self.error(self.calculate(observation_kind='time_profile_sd'),
                   'future-sd-profile-file')
        self.error(self.calculate(observation_kind='time_profile_sd',
                                  profile='time,sd\n2,.02\n1,.04\n'),
                   'future-sd-profile-file__time_column')

    def test_unused_profile_does_not_override_constant_future_variability(self):
        self.success(self.calculate(observation_kind='constant_sd',
                                    profile='this is not a valid time profile'))

    def test_calculation_uses_the_explicitly_selected_workbook_sheet(self):
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            pd.DataFrame({'incorrect': [1]}).to_excel(writer, sheet_name='Other', index=False)
            pd.read_csv(io.StringIO(SOURCE)).to_excel(writer, sheet_name='Measured', index=False)
        self.error(self.calculate(stream.getvalue(), filename='source.xlsx'),
                   'reference-file__sheet')
        payload = self.success(self.calculate(stream.getvalue(), filename='source.xlsx',
                                              **{'reference-file__sheet': 'Measured'}))
        self.assertIn('Measured', payload['html'])

    def test_source_filenames_and_manual_labels_are_escaped(self):
        payload = self.success(self.calculate(
            filename='<script>alert(1)</script>.csv',
            manual_schedules='<b>My schedule</b>: 0,2,4'))
        self.assertNotIn('<script>', payload['html'])
        self.assertNotIn('<b>My schedule</b>', payload['html'])
        self.assertIn('&lt;script&gt;', payload['html'])
        self.assertIn('&lt;b&gt;My schedule&lt;/b&gt;', payload['html'])

    def test_calculation_remains_a_request_scoped_operation(self):
        payload = self.success(self.calculate())
        new_workspace = self.client.get('/')
        self.assertFalse(new_workspace.headers.getlist('Set-Cookie'))
        self.assertNotIn(payload['html'], new_workspace.get_data(as_text=True))
        self.assertEqual(self.client.get('/calculate').status_code, 405)

    def test_upload_limit_applies_to_calculation(self):
        client = create_app({'TESTING': True, 'MAX_CONTENT_LENGTH': 100}).test_client()
        response = client.post('/calculate', data={
            'task': 'study', 'reference_source': 'measured',
            'reference_file': (io.BytesIO(b'x' * 200), 'source.csv'),
        })
        self.error(response, None, status=413)


if __name__ == '__main__':
    unittest.main()
