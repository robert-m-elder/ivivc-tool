"""Browser requests carry optional SD/SEM into the shared numerical workflow."""
from html.parser import HTMLParser
import importlib.util
import io
import json
import unittest
from unittest.mock import patch

import numpy as np


APP_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                       for name in ('flask', 'jinja2', 'plotly', 'uncertainties'))


class ReportJSON(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.fragments = False, []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'structured-report-sections':
            self.active = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.fragments.append(data)


@unittest.skipUnless(APP_DEPENDENCIES, 'Browser requests require the installed app requirements')
class BrowserUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as app_module
        cls.module = app_module
        cls.app = app_module.app
        cls.app.config.update(TESTING=True)

    def inputs(self, **changes):
        time = np.arange(6.)
        values = np.exp(-0.2 * time) + np.array([0., 0.001, -0.002, 0.001, -0.001, 0.001])
        rows = '\n'.join(f'{t},{y},{2*t},{y},0.04,4,extra' for t, y in zip(time, values))
        form = {
            'file': (io.BytesIO(('t1,y1,t2,y2,SEM,n,note\n' + rows).encode()), 'fit.csv'),
            'approaches': ['approach2'], 'models': ['approach2:exponential_normalized'],
            'metrics': ['rmse', 'aicc'], 'interpolation': ['none'],
            'cv_scheme': 'none', 'grid_search_num_points': '1', 'grid_search_num_cores': '1',
            'grid_search_param_min': '-1', 'grid_search_param_max': '0',
            'in_vitro_uncertainty_mode': 'constant', 'in_vitro_uncertainty_value': '0.04',
            'in_vitro_response_kind': 'mean', 'uncertainty_independence_confirmed': 'on',
        }
        form.update(changes)
        return form

    def post(self, **changes):
        return self.app.test_client().post('/', data=self.inputs(**changes),
                                           content_type='multipart/form-data')

    def test_setup_offers_the_three_optional_error_inputs_without_sd_to_sem_conversion(self):
        response = self.app.test_client().get('/')
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        for source in ('in_vitro', 'in_vivo', 'prediction'):
            self.assertIn(f'name="{source}_uncertainty_mode"', page)
            self.assertIn(f'name="{source}_uncertainty_column"', page)
            self.assertIn(f'name="{source}_response_kind"', page)
        self.assertIn('convert SD to SEM before entry', page)
        self.assertIn('uncertainty_independence_confirmed', page)

    def test_constant_sem_reaches_the_typed_profile_and_the_report(self):
        with patch.object(self.module, 'process_data', wraps=self.module.process_data) as fit:
            response = self.post()
        self.assertEqual(response.status_code, 200)
        prepared = fit.call_args.args[0]
        np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.04] * 6)
        self.assertEqual(prepared.in_vitro.observation_uncertainty.uncertainty_kind, 'sem')
        self.assertIsNone(prepared.in_vivo.observation_uncertainty)
        page = response.get_data(as_text=True)
        parser = ReportJSON()
        parser.feed(page)
        reports = json.loads(''.join(parser.fragments))
        self.assertTrue(reports)
        self.assertIn('SEM', json.dumps(reports))
        paragraphs = [section.get('text', '').lower()
                      for sections in reports.values() for section in sections
                      if section.get('type') == 'paragraph']
        self.assertTrue(any('future-observation variability' in paragraph and 'excluded' in paragraph
                            for paragraph in paragraphs),
                        'The report must explicitly exclude future-observation variability.')

    def test_error_and_count_columns_are_metadata_aligned_with_the_fitting_rows(self):
        with patch.object(self.module, 'process_data', wraps=self.module.process_data) as fit:
            response = self.post(in_vitro_uncertainty_mode='column',
                                 in_vitro_uncertainty_column='SEM', in_vitro_replicate_column='n')
        self.assertEqual(response.status_code, 200)
        uncertainty = fit.call_args.args[0].in_vitro.observation_uncertainty
        np.testing.assert_array_equal(uncertainty.sigma, [0.04] * 6)
        np.testing.assert_array_equal(uncertainty.replicate_counts, [4] * 6)

    def test_prediction_error_columns_reach_the_prediction_time_course(self):
        prediction = io.BytesIO(b't,y,error\n1,0.8,0.03\n2,0.6,0.05\n')
        with patch.object(self.module, 'process_predictions', wraps=self.module.process_predictions) as predict:
            response = self.post(prediction_file=(prediction, 'prediction.csv'),
                                 prediction_uncertainty_mode='column', prediction_uncertainty_column='error',
                                 prediction_response_kind='mean')
        self.assertEqual(response.status_code, 200)
        course = predict.call_args.args[1]
        np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.03, 0.05])
        self.assertEqual(course.observation_uncertainty.uncertainty_kind, 'sem')

    def test_uncertainty_confirmation_is_required_and_form_choices_survive_the_error(self):
        response = self.post(uncertainty_independence_confirmed='')
        self.assertEqual(response.status_code, 400)
        page = response.get_data(as_text=True)
        self.assertIn('Confirm', page)
        self.assertIn('value="constant" selected', page)
        self.assertIn('name="in_vitro_uncertainty_value" value="0.04"', page)
        self.assertIn('value="approach2" aria-controls="approach2-models" checked', page)
        self.assertIn('value="none" selected', page)

    def test_invalid_uncertainty_column_returns_a_linked_form_error(self):
        response = self.post(in_vitro_uncertainty_mode='column', in_vitro_uncertainty_column='missing')
        self.assertEqual(response.status_code, 400)
        page = response.get_data(as_text=True)
        self.assertIn('href="#uncertainty-group"', page)
        self.assertIn('name="in_vitro_uncertainty_column" value="missing"', page)

    def test_unsupported_approach_or_response_scaling_is_rejected_before_fitting(self):
        for changes in ({'approaches': ['approach1'], 'models': ['approach1:linear']},
                        {'scalings': ['log_y']}, {'prediction_uncertainty_mode': 'constant'}):
            with self.subTest(changes=changes), patch.object(self.module, 'process_data') as fit:
                response = self.post(**changes)
                self.assertEqual(response.status_code, 400)
                fit.assert_not_called()

    def test_nonpositive_constant_and_invalid_mean_counts_are_form_errors(self):
        invalid_counts = io.BytesIO(b't1,y1,t2,y2,n\n0,1,0,1,0\n1,.8,2,.8,4\n2,.6,4,.6,4\n')
        for changes in ({'in_vitro_uncertainty_value': '0'},
                        {'in_vitro_response_kind': 'individual', 'in_vitro_replicate_column': 'n'},
                        {'file': (invalid_counts, 'invalid-counts.csv'), 'in_vitro_replicate_column': 'n'}):
            with self.subTest(changes=changes):
                response = self.post(**changes)
                self.assertEqual(response.status_code, 400)

    def test_unweighted_upload_with_extra_columns_uses_the_legacy_array_path(self):
        with patch.object(self.module, 'process_data', wraps=self.module.process_data) as fit:
            response = self.post(in_vitro_uncertainty_mode='none')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(fit.call_args.args[0], (tuple, list))


if __name__ == '__main__':
    unittest.main()
