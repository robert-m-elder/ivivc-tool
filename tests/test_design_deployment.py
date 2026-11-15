"""Deployment integration for the independent, request-scoped WSGI app."""
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.test import Client
from werkzeug.wrappers import Response

from design_app import create_app


ROOT = Path(__file__).resolve().parents[1]
PREFIX = '/design'


class PageURLs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        for name in ('src', 'href', 'action', 'formaction', 'data-inspect-url',
                     'data-preview-url', 'data-generation-url'):
            value = values.get(name, '')
            if value and not value.startswith('#'):
                self.urls.append(value)


class DesignDeployment(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        mounted = DispatcherMiddleware(Response('Outside the design app', status=404),
                                       {PREFIX: self.app})
        self.client = Client(mounted, Response)

    @staticmethod
    def upload(content=b'time,response\n0,1\n1,.9\n2,.81\n'):
        return io.BytesIO(content), 'measurements.csv'

    def test_wsgi_mount_keeps_all_form_import_generation_and_asset_urls_under_prefix(self):
        response = self.client.get(PREFIX + '/')
        self.assertEqual(response.status_code, 200)
        page = PageURLs()
        page.feed(response.get_data(as_text=True))
        expected = {PREFIX + path for path in (
            '/review', '/calculate', '/generate-candidates', '/data/inspect',
            '/data/preview', '/static/css/design.css', '/static/js/design.js',
            '/static/js/design-data.js')}
        self.assertEqual(set(page.urls), expected)
        for url in expected:
            if '/static/' in url:
                with self.subTest(url=url):
                    asset = self.client.get(url)
                    self.assertEqual(asset.status_code, 200)
                    self.assertGreater(len(asset.data), 0)
        self.assertEqual(self.client.get('/').status_code, 404)
        self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_mounted_import_workflow_reads_each_current_upload_without_cached_files(self):
        inspected = self.client.post(PREFIX + '/data/inspect', data={
            'dataset_id': 'reference-file', 'file': self.upload(),
        })
        self.assertEqual(inspected.status_code, 200)
        self.assertEqual(inspected.get_json()['columns'], ['time', 'response'])
        preview = self.client.post(PREFIX + '/data/preview', data={
            'dataset_id': 'reference-file', 'file': self.upload(),
        })
        self.assertEqual(preview.status_code, 200)
        self.assertIn('measurements.csv', preview.get_json()['html'])
        missing = self.client.post(PREFIX + '/data/preview', data={
            'dataset_id': 'reference-file',
        })
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.get_json()['errors'][0]['field_id'], 'reference-file')
        for response in (inspected, preview, missing):
            self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_mounted_generation_uses_generation_branch_and_needs_no_fixed_or_target_upload(self):
        source = b'time,response\n0,1\n1,.904\n2,.819\n3,.741\n4,.671\n'
        with patch('utilities.design_calibration.compare_calibration_designs') as compare:
            response = self.client.post(PREFIX + '/generate-candidates', data={
                'task': 'calibration', 'varied_environment': 'in_vivo',
                'cal_in_vivo_source': 'file', 'cal_in_vivo_model': 'exponential_normalized',
                'cal_in_vivo_source_file': self.upload(source),
                'cal_in_vivo_first_time': '0', 'cal_in_vivo_last_time': '4',
                'cal_in_vivo_time_step': '1',
                'cal_candidate_0_name': 'Mounted candidate',
                'cal_candidate_0_in_vivo_kind': 'generated',
                'cal_candidate_0_in_vivo_number_of_times': '3',
                'cal_candidate_0_in_vivo_replicates': '2',
                'cal_assess_target': 'on', 'n_simulations': 'unused',
                'cal_interval_draws': 'unused', 'reference_draws': '3',
            })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        compare.assert_not_called()
        schedule, = response.get_json()['schedules']
        self.assertEqual(schedule['side'], 'in_vivo')
        self.assertEqual(len(schedule['times']), 3)
        self.assertFalse(response.headers.getlist('Set-Cookie'))

    def test_upload_cap_applies_to_every_multipart_post_endpoint(self):
        app = create_app({'TESTING': True, 'MAX_CONTENT_LENGTH': 512})
        client = app.test_client()
        for path in ('/review', '/data/inspect', '/data/preview', '/calculate',
                     '/generate-candidates'):
            with self.subTest(path=path):
                response = client.post(path, data={
                    'task': 'calibration', 'dataset_id': 'reference-file',
                    'file': self.upload(b'x' * 1024),
                })
                self.assertEqual(response.status_code, 413)
                self.assertIs(response.get_json()['ok'], False)
                self.assertNotIn('html', response.get_json())
        self.assertEqual(create_app().config['MAX_CONTENT_LENGTH'], 25 * 1024 * 1024)

    def test_mounted_method_error_preserves_http_headers_and_json_contract(self):
        response = self.client.get(PREFIX + '/calculate')
        self.assertEqual(response.status_code, 405)
        self.assertIn('POST', response.headers['Allow'])
        self.assertTrue(response.is_json)
        self.assertIs(response.get_json()['ok'], False)
        self.assertNotIn('html', response.get_json())

    def test_wsgi_import_from_another_working_directory_does_not_start_server(self):
        script = '''
import json, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
with patch('flask.Flask.run', side_effect=AssertionError('WSGI import started a server')):
    import design_app
    before_request = [name for name in ('app', 'engine', 'study_design_core') if name in sys.modules]
    client = design_app.app.test_client()
    index = client.get('/')
    css = client.get('/static/css/design.css')
    print(json.dumps({'startup_imports': before_request, 'index': index.status_code,
                      'css': css.status_code, 'debug': design_app.app.debug}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', script, str(ROOT)], cwd=directory,
                                    check=True, capture_output=True, text=True)
        status = json.loads(result.stdout)
        self.assertEqual(status, {'startup_imports': [], 'index': 200, 'css': 200, 'debug': False})


if __name__ == '__main__':
    unittest.main()
