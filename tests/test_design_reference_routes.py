"""Request integration for IVIVC and empirical time-scale planning references.

Every successful request runs the real small planning pipeline. Current files,
worksheets, mappings, and precision must be reinterpreted without a preview or
an earlier request supplying hidden state.
"""

import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from design_app import create_app


def calibration_csv():
    times = np.arange(0., 10., 2.)
    values = np.exp(-.2 * times) * np.array([1., 1.003, .998, 1.002, .999])
    records = ['t_iv,y_iv,t_vivo,y_vivo']
    records.extend(f'{time:.17g},{value:.17g},{2*time:.17g},{value:.17g}'
                   for time, value in zip(times, values))
    return '\n'.join(records) + '\n'


def target_csv():
    return 'time,response\n' + ''.join(
        f'{time:.17g},{np.exp(-.2*time):.17g}\n' for time in np.arange(9.))


CALIBRATION = calibration_csv()
TARGET = target_csv()
RATIOS = 'ratio,weight,source_id\n2,1,A\n3,3,B\n9,0,Unused\n'


class DesignReferenceRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    @staticmethod
    def upload(content, filename):
        if isinstance(content, str):
            content = content.encode('utf-8')
        return io.BytesIO(content), filename

    def calculate(self, source='ivivc', *, calibration=CALIBRATION, target=TARGET,
                  scaling=TARGET, ratios=RATIOS, profile=None,
                  calibration_filename='calibration.csv',
                  target_filename='target.csv', scaling_filename='in-vitro.csv',
                  ratios_filename='time-scales.csv', **changes):
        form = {
            'task': 'study', 'reference_source': source,
            'model_name': 'exponential_normalized', 'normalization': 'none',
            'ivivc_approach': 'approach2', 'ivivc_model': 'exponential_normalized',
            'ivivc_method': 'value_ratio', 'ivivc_interpolation': 'default',
            'scaling_model': 'exponential_normalized',
            'observation_kind': 'constant_sd', 'observation_value': '.03',
            'replicates_per_time': '2', 'first_time': '0', 'last_time': '4',
            'time_step': '1', 'min_times': '3', 'max_times': '4', 'fixed_times': '0',
            'reference_draws': '8', 'n_simulations': '4', 'random_seed': '123',
        }
        if source == 'ivivc':
            if calibration is not None:
                form['ivivc_calibration_file'] = self.upload(calibration, calibration_filename)
            if target is not None:
                form['prediction_file'] = self.upload(target, target_filename)
        elif source == 'scaling':
            if scaling is not None:
                form['scaling_source_file'] = self.upload(scaling, scaling_filename)
            if ratios is not None:
                form['time_scale_file'] = self.upload(ratios, ratios_filename)
        if profile is not None:
            form['future_sd_profile_file'] = self.upload(profile, 'future-errors.csv')
        form.update(changes)
        return self.client.post('/calculate', data=form, content_type='multipart/form-data')

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertIsInstance(payload['html'], str)
        self.assertIn('IRMSE', payload['html'])
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

    def test_ivivc_reference_runs_without_a_preview_and_reports_all_file_roles(self):
        payload = self.success(self.calculate())
        self.assertIn('calibration.csv', payload['html'])
        self.assertIn('target.csv', payload['html'])
        self.assertIn('Response ratio', payload['html'])

    def test_ivivc_and_scaling_requests_are_seeded_and_independent(self):
        before = self.success(self.calculate())
        self.success(self.calculate('scaling'))
        after = self.success(self.calculate())
        self.assertEqual(before['html'], after['html'])

    def test_scaling_reference_reports_source_and_empirical_distribution(self):
        payload = self.success(self.calculate('scaling'))
        self.assertIn('in-vitro.csv', payload['html'])
        self.assertIn('time-scales.csv', payload['html'])
        self.assertRegex(payload['html'], r'(?i)empirical')

    def test_ivivc_uses_current_independent_mappings_and_padded_sides(self):
        from utilities.design_planning import run_reference_planning

        frame = pd.read_csv(io.StringIO(CALIBRATION))
        frame.loc[4, ['t_vivo', 'y_vivo']] = np.nan
        frame.insert(0, 'unrelated', 999)
        content = frame.to_csv(index=False)
        mapping = {
            'ivivc-calibration-file__in_vitro__time_column': 't_iv',
            'ivivc-calibration-file__in_vitro__response_column': 'y_iv',
            'ivivc-calibration-file__in_vivo__time_column': 't_vivo',
            'ivivc-calibration-file__in_vivo__response_column': 'y_vivo',
        }
        with patch('utilities.design_planning.run_reference_planning',
                   wraps=run_reference_planning) as calculate:
            self.success(self.calculate(calibration=content, **mapping))
        np.testing.assert_array_equal(calculate.call_args.kwargs['in_vitro'].time,
                                      [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(calculate.call_args.kwargs['in_vivo'].time,
                                      [0, 4, 8, 12])
        self.assertEqual(calculate.call_args.kwargs['prediction'].label,
                         'Target in vitro prediction data')

    def test_reported_calibration_sd_and_counts_are_resolved_once(self):
        from utilities.design_planning import run_reference_planning

        options = {
            'ivivc-calibration-file__in_vivo__uncertainty': 'mean_sd',
            'ivivc-calibration-file__in_vivo__error_source': 'constant',
            'ivivc-calibration-file__in_vivo__error_value': '.06',
            'ivivc-calibration-file__in_vivo__count_source': 'constant',
            'ivivc-calibration-file__in_vivo__count_value': '4',
        }
        with patch('utilities.design_planning.run_reference_planning',
                   wraps=run_reference_planning) as calculate:
            self.success(self.calculate(**options))
        uncertainty = calculate.call_args.kwargs['in_vivo'].observation_uncertainty
        np.testing.assert_allclose(uncertainty.sigma, np.full(5, .03))
        np.testing.assert_array_equal(uncertainty.replicate_counts, np.full(5, 4))

    def test_prediction_sem_remains_sem_when_counts_are_supplied(self):
        from utilities.design_planning import run_reference_planning

        options = {
            'prediction-file__profile__uncertainty': 'mean_sem',
            'prediction-file__profile__error_source': 'constant',
            'prediction-file__profile__error_value': '.01',
            'prediction-file__profile__count_source': 'constant',
            'prediction-file__profile__count_value': '9',
        }
        with patch('utilities.design_planning.run_reference_planning',
                   wraps=run_reference_planning) as calculate:
            self.success(self.calculate(**options))
        uncertainty = calculate.call_args.kwargs['prediction'].observation_uncertainty
        np.testing.assert_allclose(uncertainty.sigma, np.full(9, .01))
        self.assertEqual(uncertainty.response_kind, 'mean')

    def test_scaling_preserves_current_weight_records_and_source_precision(self):
        from utilities.design_planning import run_reference_planning

        options = {
            'scaling-source-file__profile__uncertainty': 'individual_sd',
            'scaling-source-file__profile__error_source': 'constant',
            'scaling-source-file__profile__error_value': '.02',
        }
        with patch('utilities.design_planning.run_reference_planning',
                   wraps=run_reference_planning) as calculate:
            self.success(self.calculate('scaling', **options))
        distribution = calculate.call_args.kwargs['distribution']
        np.testing.assert_array_equal(distribution.ratios, [2, 3, 9])
        np.testing.assert_allclose(distribution.weights, [.25, .75, 0])
        self.assertEqual(distribution.source_ids, ('A', 'B', 'Unused'))
        np.testing.assert_allclose(
            calculate.call_args.kwargs['scaling_source'].observation_uncertainty.sigma,
            np.full(9, .02))

    def test_each_source_requires_its_current_files(self):
        self.error(self.calculate(calibration=None), 'ivivc-calibration-file')
        self.error(self.calculate(target=None), 'prediction-file')
        self.error(self.calculate('scaling', scaling=None), 'scaling-source-file')
        self.error(self.calculate('scaling', ratios=None), 'time-scale-file')

    def test_current_invalid_mapping_is_not_replaced_by_an_earlier_import(self):
        self.success(self.calculate())
        self.error(self.calculate(**{'ivivc-calibration-file__in_vivo__response_column': ''}),
                   'ivivc-calibration-file__in_vivo__response_column')
        self.error(self.calculate('scaling', **{'time-scale-file__ratio_column': 'missing'}),
                   'time-scale-file__ratio_column')

    def test_missing_empirical_weights_and_partial_calibration_pairs_are_errors(self):
        self.error(self.calculate('scaling', ratios='ratio,weight\n2,\n3,1\n'),
                   'time-scale-file__weight_column')
        broken = CALIBRATION.replace('0,1,0,1', '0,1,0,')
        self.error(self.calculate(calibration=broken),
                   'ivivc-calibration-file__in_vivo__response_column')

    def test_unselected_source_files_and_future_profile_are_ignored(self):
        self.success(self.calculate(
            reference_file=self.upload('bad', 'unselected.csv'),
            scaling_source_file=self.upload('bad', 'unselected-scaling.csv'),
            time_scale_file=self.upload('bad', 'unselected-ratios.csv'),
            profile='bad'))
        self.success(self.calculate('scaling',
                                    ivivc_calibration_file=self.upload('bad', 'inactive.csv'),
                                    prediction_file=self.upload('bad', 'inactive-target.csv')))

    def test_future_sd_profile_is_separate_and_available_for_both_new_sources(self):
        for source in ('ivivc', 'scaling'):
            with self.subTest(source=source):
                self.success(self.calculate(source, observation_kind='time_profile_sd',
                                            profile='time,sd\n0,.02\n4,.08\n'))
                self.error(self.calculate(source, observation_kind='time_profile_sd'),
                           'future-sd-profile-file')

    def test_requested_ivivc_method_must_be_supported(self):
        self.error(self.calculate(ivivc_method='not-a-method'), 'ivivc_method')

    def test_other_ivivc_approaches_use_their_explicit_prediction_methods(self):
        for approach, method, label in (
            ('approach1', 'time_mapping', 'Map in vitro time to in vivo time'),
            ('approach3', 'value_scaling', 'Response scaling'),
        ):
            with self.subTest(approach=approach):
                payload = self.success(self.calculate(
                    ivivc_approach=approach, ivivc_model='linear', ivivc_method=method,
                    first_time='1', fixed_times='1'))
                self.assertIn(label, payload['html'])

    def test_unavailable_selected_method_is_not_replaced_by_another_method(self):
        # At t=1 this fixture's interpolated time ratio is undefined. The
        # shared Approach 3 pipeline cannot supply a complete trajectory.
        self.error(self.calculate(ivivc_approach='approach3',
                                  ivivc_method='time_scaling'), 'ivivc_method')

    def test_precision_is_not_silently_removed_for_approaches_without_propagation(self):
        precision = {'uncertainty': 'individual_sd', 'error_source': 'constant',
                     'error_value': '.02'}
        for prefix, role in (('ivivc-calibration-file__in_vivo__', 'ivivc-calibration-file'),
                             ('prediction-file__profile__', 'prediction-file')):
            with self.subTest(role=role):
                self.error(self.calculate(
                    ivivc_approach='approach1', ivivc_model='linear',
                    ivivc_method='time_mapping', first_time='1', fixed_times='1',
                    **{prefix + key: value for key, value in precision.items()}), role)

    def test_scaling_extrapolation_is_an_explicit_current_request_option(self):
        # Source observations stop at 8; a scale of 1 needs source time 12.
        options = dict(ratios='ratio\n1\n', last_time='12', time_step='3',
                       max_times='4')
        denied = self.calculate('scaling', **options)
        self.assertEqual(denied.status_code, 400, denied.get_data(as_text=True))
        self.assertRegex(str(denied.get_json()), r'(?i)extrapolat')
        self.success(self.calculate('scaling', scaling_allow_extrapolation='on', **options))

    def test_ivivc_requested_horizon_is_not_silently_clipped(self):
        response = self.calculate(last_time='20', time_step='5')
        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        self.assertRegex(str(response.get_json()), r'(?i)(support|range|horizon)')

    def test_explicitly_selected_calibration_and_ratio_worksheets_are_used(self):
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            pd.DataFrame({'unrelated': [1]}).to_excel(writer, sheet_name='Other', index=False)
            pd.read_csv(io.StringIO(CALIBRATION)).to_excel(writer, sheet_name='Calibration', index=False)
        content = stream.getvalue()
        self.error(self.calculate(calibration=content, calibration_filename='paired.xlsx'),
                   'ivivc-calibration-file__sheet')
        self.success(self.calculate(calibration=content, calibration_filename='paired.xlsx',
                                    **{'ivivc-calibration-file__sheet': 'Calibration'}))
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            pd.DataFrame({'unrelated': [1]}).to_excel(writer, sheet_name='Other', index=False)
            pd.read_csv(io.StringIO(RATIOS)).to_excel(writer, sheet_name='Ratios', index=False)
        self.success(self.calculate('scaling', ratios=stream.getvalue(), ratios_filename='ratios.xlsx',
                                    **{'time-scale-file__sheet': 'Ratios'}))

    def test_uploaded_names_remain_escaped_in_other_reference_results(self):
        payload = self.success(self.calculate(
            calibration_filename='<script>calibration</script>.csv',
            target_filename='<b>target</b>.csv'))
        self.assertIn('&lt;script&gt;', payload['html'])
        self.assertIn('&lt;b&gt;', payload['html'])
        self.assertNotIn('<script>', payload['html'])
        self.assertNotIn('<b>target</b>', payload['html'])


if __name__ == '__main__':
    unittest.main()
