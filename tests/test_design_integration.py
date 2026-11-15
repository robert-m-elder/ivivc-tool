"""Small real workflows across import, schedule generation, and assessment.

Each request supplies its own files. Spies retain results for assertions while
the source fitting, optimization, simulation, and presentation run normally.
"""
import io
import unittest
from unittest.mock import patch

import numpy as np

from design_app import create_app


def course_csv(times, rate=.1):
    times = np.asarray(times)
    values = np.exp(-rate * times) * np.resize([1., 1.001, .999], len(times))
    return 'time,response,sd,n\n' + ''.join(
        f'{time:.17g},{value:.17g},.04,4\n' for time, value in zip(times, values))


class DesignWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        self.client = create_app({'TESTING': True}).test_client()

    @staticmethod
    def upload(content, filename):
        return io.BytesIO(content.encode()), filename

    @staticmethod
    def precision(role, mode='mean_sd'):
        return {role + '__profile__' + key: value for key, value in dict(
            uncertainty=mode, error_source='column', error_column='sd',
            count_source='column', count_column='n').items()}

    def post(self, path, form):
        response = self.client.post(path, data=form, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        self.assertNotRegex(payload['html'], r'>\s*(?:nan|[-+]?inf(?:inity)?)\s*<')
        return payload

    @staticmethod
    def study_settings(**changes):
        form = dict(task='study', reference_source='measured', model_name='linear',
            normalization='none', observation_kind='constant_sd', observation_value='.03',
            replicates_per_time='2', first_time='0', last_time='4', time_step='1',
            min_times='3', max_times='3', fixed_times='0',
            reference_draws='4', n_simulations='2', random_seed='123')
        form.update(changes)
        return form

    def test_import_preview_generated_snapshot_and_calibration_comparison_agree(self):
        from utilities.design_calibration import run_calibration_generation
        from utilities.design_calibration_results import present_calibration_result

        source = course_csv(np.arange(0., 17., 2.))
        role = 'cal-in-vivo-source-file'
        interpreted = self.precision(role)
        inspected = self.post('/data/inspect', dict(dataset_id=role,
            file=self.upload(source, 'expected-vivo.csv')))
        self.assertEqual(inspected['columns'], ['time', 'response', 'sd', 'n'])
        preview = self.post('/data/preview', dict(dataset_id=role,
            file=self.upload(source, 'expected-vivo.csv'), **interpreted))
        self.assertIn('Observation SEM', preview['html'])

        settings = dict(task='calibration', varied_environment='in_vivo',
            cal_in_vivo_mode='prospective', cal_in_vivo_source='file',
            cal_in_vivo_model='exponential_normalized',
            cal_in_vivo_observation_kind='constant_sd', cal_in_vivo_observation_value='.02',
            cal_in_vivo_first_time='0', cal_in_vivo_last_time='12', cal_in_vivo_time_step='2',
            cal_candidate_0_name='Reviewed schedule', cal_candidate_0_in_vivo_kind='generated',
            cal_candidate_0_in_vivo_number_of_times='4', cal_candidate_0_in_vivo_replicates='3',
            reference_draws='4', random_seed='123', **interpreted)
        with patch('utilities.design_calibration.run_calibration_generation',
                   wraps=run_calibration_generation) as generate:
            generated = self.post('/generate-candidates', dict(settings,
                cal_in_vivo_source_file=self.upload(source, 'expected-vivo.csv')))
        source_course = generate.call_args.kwargs['in_vivo_source']
        np.testing.assert_allclose(source_course.observation_uncertainty.sigma, .02)
        times = generated['schedules'][0]['times']

        # Copying the browser preview creates an explicit schedule. The server
        # reimports current files and receives times, not a stored preview ID.
        calculation = dict(settings, cal_candidate_0_in_vivo_kind='manual',
            cal_candidate_0_in_vivo_times=','.join(map(str, times)),
            cal_approach='approach2', cal_model='exponential_normalized',
            cal_method='value_ratio', cal_relationship='on', cal_evaluation_times='1,3',
            cal_in_vitro_source='existing', cal_in_vitro_model='exponential_normalized',
            n_simulations='2', cal_interval_draws='4',
            cal_in_vivo_source_file=self.upload(source, 'expected-vivo.csv'),
            existing_in_vitro_file=self.upload(course_csv(np.arange(9.), .2), 'fixed-vitro.csv'))
        with patch('utilities.design_calibration_results.present_calibration_result',
                   wraps=present_calibration_result) as present:
            assessed = self.post('/calculate', calculation)
        result = present.call_args.args[0]
        design = result.designs['Reviewed schedule']
        np.testing.assert_array_equal(design.in_vivo_times, times)
        self.assertEqual(design.in_vivo_observation_model.replicates_per_time, 3)
        self.assertEqual(result.generation_outcomes[0]['kind'], 'manual')
        np.testing.assert_allclose(
            result.source_details['in_vivo']['analyzed_source'].observation_uncertainty.sigma, .02)
        self.assertIn('Reviewed schedule', assessed['html'])
        self.assertIn('fixed-vitro.csv', assessed['html'])

    def test_prospective_to_adaptive_rerun_keeps_completed_sem_and_responses_fixed(self):
        from utilities.design_planning_results import present_planning_result

        observations = course_csv(np.arange(5.))
        prospective = self.study_settings(**self.precision('reference-file', 'mean_sem'))
        first = self.post('/calculate', dict(prospective,
            reference_file=self.upload(observations, 'prior.csv')))
        adaptive = self.study_settings(has_completed='on',
            measured_adaptive_mode='refit_completed', completed_error_mode='supplied',
            future_policy='future_only', first_time_mode='latest_completed', baseline_time='0',
            last_time='8', min_times='6', max_times='6', replicates_per_time='7',
            **self.precision('completed-target-file', 'mean_sem'))
        with patch('utilities.design_planning_results.present_planning_result',
                   wraps=present_planning_result) as present:
            updated = self.post('/calculate', dict(adaptive,
                completed_target_file=self.upload(observations, 'completed.csv')))
        result = present.call_args.args[0]
        old = result.completed_observations
        np.testing.assert_allclose(old.observation_uncertainty.sigma, .04)
        np.testing.assert_array_equal(old.observation_uncertainty.replicate_counts, [4] * 5)
        for simulation in result.comparison.results.values():
            for time, response in zip(old.time, old.response):
                column = np.flatnonzero(np.isclose(simulation.design_times, time))[0]
                np.testing.assert_allclose(simulation.observed_means[:, column], response)
        self.assertIn('completed.csv', updated['html'])
        self.assertNotIn('prior.csv', updated['html'])
        self.assertNotEqual(first['html'], updated['html'])

    def test_switching_reference_workflows_does_not_retain_an_earlier_upload(self):
        source = course_csv(np.arange(9.), .2)
        measured = self.study_settings()
        first = self.post('/calculate', dict(measured,
            reference_file=self.upload(source, 'measured.csv')))
        common = self.study_settings(model_name='exponential_normalized',
            scaling_model='exponential_normalized', ivivc_approach='approach2',
            ivivc_model='exponential_normalized', ivivc_method='value_ratio')
        scaled = self.post('/calculate', dict(common, reference_source='scaling',
            scaling_source_file=self.upload(source, 'scaling-source.csv'),
            time_scale_file=self.upload('ratio,weight\n2,1\n3,3\n', 'ratios.csv')))
        self.assertIn('ratios.csv', scaled['html'])
        self.assertNotIn('measured.csv', scaled['html'])
        missing = self.client.post('/calculate', data=dict(common, reference_source='ivivc'))
        self.assertEqual(missing.status_code, 400)
        self.assertIn('ivivc-calibration-file',
            [row['field_id'] for row in missing.get_json()['errors']])
        # The earlier import remains unavailable unless the user uploads it
        # again; a return to its original inputs reproduces its result.
        missing = self.client.post('/calculate', data=measured)
        self.assertEqual(missing.status_code, 400)
        self.assertIn('reference-file',
            [row['field_id'] for row in missing.get_json()['errors']])
        repeated = self.post('/calculate', dict(measured,
            reference_file=self.upload(source, 'measured.csv')))
        self.assertEqual(first, repeated)


if __name__ == '__main__':
    unittest.main()
