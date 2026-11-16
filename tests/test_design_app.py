"""Requests for the separate study-design Flask application.

F1 reviews a setup; it does not read measurement data or run an analysis.
"""

from html.parser import HTMLParser
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from design_app import create_app


class PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class DesignAppRequests(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    def review(self, **changes):
        form = {'task': 'study'}
        form.update(changes)
        return self.client.post('/review', data=form)

    def assert_error(self, response, field_id, status=400):
        self.assertEqual(response.status_code, status)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertTrue(any(error.get('field_id') == field_id
                            for error in payload['errors']), payload)
        return payload

    def test_index_contains_the_two_tasks_and_independent_input_result_regions(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith('text/html'))
        page = PageElements()
        page.feed(response.get_data(as_text=True))
        ids = [attrs['id'] for _, attrs in page.elements if 'id' in attrs]
        self.assertEqual(len(ids), len(set(ids)), 'Page IDs must be unique.')
        self.assertTrue({'design-workspace', 'design-inputs', 'design-results',
                         'design-setup-form', 'design-errors'} <= set(ids))
        self.assertTrue(any(tag == 'select' and attrs.get('name') == 'task'
                            for tag, attrs in page.elements))
        task_options = {attrs.get('value') for tag, attrs in page.elements
                        if tag == 'option'}
        self.assertTrue({'study', 'calibration'} <= task_options)
        self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_static_assets_are_served_by_the_separate_app(self):
        for path in ('/static/css/design.css', '/static/js/design.js',
                     '/static/js/design-data.js', '/static/js/design-plots.js'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 0)

    def test_plotly_is_loaded_once_before_the_shared_renderer_and_workspace_controller(self):
        page = PageElements()
        page.feed(self.client.get('/').get_data(as_text=True))
        scripts = [attrs for tag, attrs in page.elements if tag == 'script']
        urls = [attrs.get('src') for attrs in scripts]
        cdn = 'https://cdn.plot.ly/plotly-3.0.1.min.js'
        self.assertEqual(urls.count(cdn), 1)
        self.assertLess(urls.index(cdn), urls.index('/static/js/design-plots.js'))
        self.assertLess(urls.index('/static/js/design-plots.js'), urls.index('/static/js/design.js'))
        self.assertTrue(all('defer' in attrs for attrs in scripts if attrs.get('src') in (
            cdn, '/static/js/design-plots.js', '/static/js/design.js')))

    def test_study_setup_defaults_are_returned_without_numerical_results(self):
        response = self.review()
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        settings = payload['settings']
        self.assertEqual(settings['task'], 'study')
        self.assertEqual(settings['reference_source'], 'measured')
        self.assertEqual(settings['n_simulations'], 100)
        self.assertEqual(settings['random_seed'], 12345)
        self.assertIsInstance(payload['html'], str)
        self.assertIn('Setup review', payload['html'])
        self.assertNotIn('100/100', payload['html'])
        self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_each_supported_reference_source_can_be_reviewed(self):
        for source in ('measured', 'ivivc', 'scaling'):
            with self.subTest(source=source):
                response = self.review(reference_source=source)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()['settings']['reference_source'], source)

    def test_each_calibration_environment_can_be_reviewed(self):
        for environment in ('in_vitro', 'in_vivo', 'both'):
            with self.subTest(environment=environment):
                response = self.review(task='calibration', varied_environment=environment)
                self.assertEqual(response.status_code, 200)
                settings = response.get_json()['settings']
                self.assertEqual(settings['task'], 'calibration')
                self.assertEqual(settings['varied_environment'], environment)

    def test_calibration_defaults_to_varying_in_vivo(self):
        response = self.review(task='calibration')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['settings']['varied_environment'], 'in_vivo')

    def test_only_the_selected_task_branch_is_validated(self):
        study = self.review(varied_environment='inactive-invalid-value')
        calibration = self.review(task='calibration', reference_source='inactive-invalid-value')
        self.assertEqual(study.status_code, 200)
        self.assertEqual(calibration.status_code, 200)

    def test_missing_or_unknown_task_has_a_field_error(self):
        self.assert_error(self.client.post('/review', data={}), 'task')
        for task in ('', 'unknown', '<script>bad()</script>'):
            with self.subTest(task=task):
                self.assert_error(self.review(task=task), 'task')

    def test_invalid_active_task_options_have_field_errors(self):
        self.assert_error(self.review(reference_source='unknown'), 'reference_source')
        self.assert_error(self.review(task='calibration', varied_environment='unknown'),
                          'varied_environment')

    def test_blank_advanced_fields_use_the_defaults(self):
        response = self.review(n_simulations='', random_seed='')
        self.assertEqual(response.status_code, 200)
        settings = response.get_json()['settings']
        self.assertEqual(settings['n_simulations'], 100)
        self.assertEqual(settings['random_seed'], 12345)

    def test_advanced_integer_settings_are_returned_as_numbers(self):
        response = self.review(n_simulations='2', random_seed='0')
        self.assertEqual(response.status_code, 200)
        settings = response.get_json()['settings']
        self.assertEqual(settings['n_simulations'], 2)
        self.assertEqual(settings['random_seed'], 0)
        self.assertIsInstance(settings['n_simulations'], int)
        self.assertIsInstance(settings['random_seed'], int)

    def test_invalid_simulation_counts_have_a_field_error(self):
        for value in ('1', '0', '-1', '2.5', 'nan', 'inf', 'many'):
            with self.subTest(value=value):
                self.assert_error(self.review(n_simulations=value), 'n_simulations')

    def test_invalid_seeds_have_a_field_error(self):
        for value in ('-1', '1.5', 'nan', 'inf', 'random'):
            with self.subTest(value=value):
                self.assert_error(self.review(random_seed=value), 'random_seed')

    def test_validation_collects_independent_input_errors(self):
        response = self.review(reference_source='unknown', n_simulations='1', random_seed='-1')
        self.assertEqual(response.status_code, 400)
        self.assertEqual({error['field_id'] for error in response.get_json()['errors']},
                         {'reference_source', 'n_simulations', 'random_seed'})

    def test_review_does_not_treat_an_upload_as_parsed_measurement_data(self):
        response = self.review(reference_file=(io.BytesIO(b'not a dataset'), 'measurements.csv'))
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertNotIn('measurements', payload['settings'])
        self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_upload_limit_returns_a_json_error(self):
        small_app = create_app({'TESTING': True, 'MAX_CONTENT_LENGTH': 100})
        response = small_app.test_client().post('/review', data={
            'task': 'study',
            'reference_file': (io.BytesIO(b'x' * 200), 'large.csv'),
        })
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        self.assertIs(response.get_json()['ok'], False)
        self.assertNotIn('html', response.get_json())

    def test_wrong_review_method_returns_a_json_error(self):
        response = self.client.get('/review')
        self.assertEqual(response.status_code, 405)
        self.assertTrue(response.is_json)
        self.assertIs(response.get_json()['ok'], False)

    def test_unexpected_review_errors_are_logged_without_exposing_details(self):
        private_message = 'private template failure that must stay in the log'
        with patch.object(self.app.jinja_env, 'get_or_select_template',
                          side_effect=RuntimeError(private_message)):
            with self.assertLogs(self.app.logger, level='ERROR'):
                response = self.review()
        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.is_json)
        self.assertIs(response.get_json()['ok'], False)
        self.assertNotIn(private_message, response.get_data(as_text=True))
        self.assertNotIn('Traceback', response.get_data(as_text=True))
        self.assertNotIn('html', response.get_json())

    def test_factory_configuration_is_not_shared_between_apps(self):
        first = create_app({'TESTING': True, 'MAX_CONTENT_LENGTH': 1234})
        second = create_app({'TESTING': True})
        self.assertIsNot(first, second)
        self.assertEqual(first.config['MAX_CONTENT_LENGTH'], 1234)
        self.assertNotEqual(second.config['MAX_CONTENT_LENGTH'], 1234)

    def test_import_does_not_load_the_ivivc_app_or_numerical_workflow(self):
        script = (
            "import json, sys; import design_app; "
            "print(json.dumps([name for name in ('app', 'engine', 'study_design_core') "
            "if name in sys.modules]))"
        )
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == '__main__':
    unittest.main()
