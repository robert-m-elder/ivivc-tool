"""Stateless generation requests use current uploads without running IVIVC simulations."""
import io
import unittest
from unittest.mock import patch

import numpy as np

from design_app import create_app


def source_csv(rate=.1):
    time = np.arange(0., 17., 2.)
    response = np.exp(-rate * time) * np.resize([1., 1.001, .999], len(time))
    return 'time,response\n' + ''.join(f'{t:.17g},{y:.17g}\n' for t, y in zip(time, response))


class CandidateGenerationHTTP(unittest.TestCase):
    def setUp(self):
        self.client = create_app({'TESTING': True}).test_client()

    @staticmethod
    def upload(content, name='source.csv'):
        return io.BytesIO(content.encode()), name

    def generate(self, source=None, **changes):
        form = dict(task='calibration', varied_environment='in_vivo',
            cal_in_vivo_mode='prospective', cal_in_vivo_source='file',
            cal_in_vivo_model='exponential_normalized',
            cal_in_vivo_observation_kind='constant_sd', cal_in_vivo_observation_value='.02',
            cal_in_vivo_first_time='0', cal_in_vivo_last_time='12', cal_in_vivo_time_step='2',
            cal_candidate_0_name='Generated vivo', cal_candidate_0_in_vivo_kind='generated',
            cal_candidate_0_in_vivo_number_of_times='4', cal_candidate_0_in_vivo_replicates='3',
            reference_draws='4', random_seed='123')
        if source is not False:
            form['cal_in_vivo_source_file'] = self.upload(source_csv() if source is None else source)
        form.update(changes)
        return self.client.post('/generate-candidates', data=form, content_type='multipart/form-data')

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertTrue(payload['html'])
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        for schedule in payload['schedules']:
            self.assertTrue(np.isfinite(schedule['times']).all())
        return payload

    def error(self, response, field):
        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertIn(field, [item['field_id'] for item in payload['errors']])

    def test_generation_needs_no_fixed_environment_or_assessment_uploads(self):
        with patch('utilities.design_calibration.compare_calibration_designs') as compare:
            payload = self.success(self.generate(cal_assess_target='on',
                n_simulations='not an integer', cal_interval_draws='invalid',
                cal_evaluation_times='not times', cal_target_file=self.upload('invalid target'),
                existing_in_vitro_file=self.upload('invalid fixed data')))
        compare.assert_not_called()
        self.assertEqual(len(payload['schedules']), 1)
        row = payload['schedules'][0]
        self.assertEqual((row['candidate_index'], row['side']), (0, 'in_vivo'))
        self.assertEqual(len(row['times']), 4)
        self.assertIn('data-copy-generated-candidate="0"', payload['html'])

    def test_generation_is_reproducible_without_preview_or_server_state(self):
        first = self.success(self.generate())
        second = self.success(self.generate())
        self.assertEqual(first, second)
        # A new request must provide its own file; a previous upload is not retained.
        self.error(self.generate(source=False), 'cal-in-vivo-source-file')

    def test_current_source_mapping_is_honored_and_linked(self):
        self.error(self.generate(**{
            'cal-in-vivo-source-file__profile__response_column': 'missing',
        }), 'cal-in-vivo-source-file__profile__response_column')

    def test_both_environments_can_generate_without_existing_data(self):
        payload = self.success(self.generate(varied_environment='both',
            cal_in_vitro_mode='prospective', cal_in_vitro_source='file',
            cal_in_vitro_model='exponential_normalized',
            cal_in_vitro_source_file=self.upload(source_csv(.2), 'vitro.csv'),
            cal_in_vitro_observation_kind='constant_sd', cal_in_vitro_observation_value='.03',
            cal_in_vitro_first_time='0', cal_in_vitro_last_time='12', cal_in_vitro_time_step='2',
            cal_candidate_0_in_vitro_kind='generated',
            cal_candidate_0_in_vitro_number_of_times='3', cal_candidate_0_in_vitro_replicates='2'))
        self.assertEqual({row['side'] for row in payload['schedules']}, {'in_vitro', 'in_vivo'})

    def test_manual_candidates_are_not_returned_as_generated_snapshots(self):
        payload = self.success(self.generate(cal_candidate_1_name='Manual comparison',
            cal_candidate_1_in_vivo_kind='manual', cal_candidate_1_in_vivo_times='0,4,8,12',
            cal_candidate_1_in_vivo_replicates='2'))
        self.assertEqual([row['candidate_index'] for row in payload['schedules']], [0])

    def test_additions_keep_existing_precision_and_avoid_completed_times(self):
        from utilities.design_calibration import run_calibration_generation

        old = 'time,response\n0,1\n4,.67\n8,.45\n'
        with patch('utilities.design_calibration.run_calibration_generation',
                   wraps=run_calibration_generation) as generate:
            payload = self.success(self.generate(cal_in_vivo_mode='additions',
                existing_in_vivo_file=self.upload(old, 'completed.csv'), **{
                    'existing-in-vivo-file__profile__uncertainty': 'mean_sem',
                    'existing-in-vivo-file__profile__error_source': 'constant',
                    'existing-in-vivo-file__profile__error_value': '.03',
                    'existing-in-vivo-file__profile__count_source': 'constant',
                    'existing-in-vivo-file__profile__count_value': '9',
                }))
        precision = generate.call_args.kwargs['existing_in_vivo'].observation_uncertainty
        np.testing.assert_allclose(precision.sigma, .03)
        np.testing.assert_array_equal(precision.replicate_counts, [9, 9, 9])
        self.assertEqual(len(payload['schedules'][0]['times']), 4)
        self.assertFalse(set(payload['schedules'][0]['times']) & {0., 4., 8.})

    def test_missing_active_profile_is_linked_to_its_upload(self):
        self.error(self.generate(cal_in_vivo_observation_kind='time_profile_sd'),
                   'calibration-in-vivo-sd-profile-file')

    def test_invalid_active_schedule_has_exact_field_error(self):
        self.error(self.generate(cal_candidate_0_in_vivo_number_of_times='20'),
                   'cal_candidate_0_in_vivo_number_of_times')

    def test_extrapolation_requires_explicit_source_permission(self):
        self.error(self.generate(cal_in_vivo_last_time='20'),
                   'cal_in_vivo_allow_extrapolation')

    def test_task_and_environment_are_validated_before_import(self):
        self.error(self.generate(source=False, task='study'), 'task')
        self.error(self.generate(source=False, varied_environment='unknown'), 'varied_environment')

    def test_generation_does_not_change_calculate_target_requirements(self):
        response = self.client.post('/calculate', data=dict(task='calibration',
            varied_environment='in_vivo', cal_in_vitro_source='none',
            existing_in_vitro_file=self.upload(source_csv(.2)),
            cal_in_vivo_source='file', cal_in_vivo_source_file=self.upload(source_csv()),
            cal_assess_target='on'), content_type='multipart/form-data')
        self.error(response, 'cal-target-file')


if __name__ == '__main__':
    unittest.main()
