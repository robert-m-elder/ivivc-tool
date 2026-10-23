"""Request-to-report checks when the app requirements are installed."""
from html.parser import HTMLParser
import importlib.util
import io
import json
import unittest


APP_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                       for name in ('flask', 'jinja2', 'plotly', 'uncertainties'))


class ReportJSON(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'structured-report-sections':
            self.active = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.fragments.append(data)


@unittest.skipUnless(APP_DEPENDENCIES, 'Full request integration requires the app requirements')
class ReportRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        app.config.update(TESTING=True)
        cls.app = app

    def run_report(self, **changes):
        data = {
            'file': (io.BytesIO(b't1,y1,t2,y2\n0,1,0,1\n20,.9,40,.9\n40,.7,80,.7\n100,.01,200,.01\n'), 'calibration.csv'),
            'approaches': ['approach3'], 'metrics': ['rmse'], 'interpolation': ['default'],
            'cv_scheme': 'none', 'grid_search_num_points': '1', 'grid_search_num_cores': '1',
        }
        data.update(changes)
        response = self.app.test_client().post('/', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        parser = ReportJSON()
        parser.feed(response.get_data(as_text=True))
        reports = json.loads(''.join(parser.fragments))
        self.assertTrue(reports)
        return response, reports

    def test_direct_mapping_report_renders_complete_source_and_version(self):
        response, reports = self.run_report()
        sections = next(iter(reports.values()))
        identification = next(s for s in sections if s.get('title') == 'Analysis identification')
        self.assertTrue(any(row[0] == 'App version' for row in identification['rows']))
        self.assertTrue(any('calibration.csv' in row for row in identification['rows']))
        table = next(s for s in sections if s.get('title') == 'In vitro calibration source observations')
        self.assertEqual(len(table['rows']), 5)
        self.assertIn('Report details', response.get_data(as_text=True))
        self.assertNotIn('NaN', json.dumps(reports, allow_nan=False))

    def test_supplied_but_invalid_prediction_data_remain_identified(self):
        _, reports = self.run_report(
            scalings=['log_y'],
            prediction_file=(io.BytesIO(b't,y\n0,0\n1,.5\n'), 'prediction.csv'))
        sections = next(iter(reports.values()))
        paragraphs = '\n'.join(s.get('text', '') for s in sections)
        self.assertIn('prediction processing failed', paragraphs)
        self.assertNotIn('No prediction dataset was supplied', paragraphs)
        raw = next(s for s in sections if s.get('title') == 'Prediction source observations')
        self.assertEqual(raw['rows'][1], ['1', '0', '0'])

    def test_successful_prediction_methods_have_matching_numerical_rows(self):
        _, reports = self.run_report(prediction_file=(io.BytesIO(b't,y\n20,.8\n40,.6\n'), 'prediction.csv'))
        self.assertEqual(len(reports), 2)
        for sections in reports.values():
            table = next(s for s in sections if s.get('title') == 'Predictions and displayed uncertainty bounds')
            self.assertEqual(len(table['rows']), 3)
            self.assertEqual(table['rows'][1][1:3], ['20', '0.8'])


if __name__ == '__main__':
    unittest.main()
