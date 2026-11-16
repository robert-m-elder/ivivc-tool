"""Endpoint quantiles and the compact, inert schedule-inspection payload."""
from copy import copy
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ivivc_core import TimeCourse
from study_design_core import ReferenceTrajectoryEnsemble
from utilities.design_planning import run_measured_planning
from utilities.design_planning_results import _endpoint_presentation, present_planning_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class EndpointRanges(unittest.TestCase):
    @staticmethod
    def reference(weights=(.03, .57, .36, .04)):
        # T50 is 2, 4, 8, and beyond 10, respectively.
        return ReferenceTrajectoryEnsemble(
            [0., 10.], [[1., -1.5], [1., -.25], [1., .375], [1., .9]],
            [True] * 4, draw_weights=weights, interpolation='linear')

    def present(self, reference, baseline=0.):
        return _endpoint_presentation(reference, baseline, range_available=
            np.count_nonzero(reference.valid_draws & (reference.draw_weights > 0)) > 1)

    def test_weighted_bounds_keep_noncrossing_trajectories(self):
        rows, data, warnings = self.present(self.reference())
        t50 = rows[0]
        self.assertEqual(t50['median'], '4')
        self.assertEqual(t50['lower_95'], '2')
        self.assertEqual(t50['upper_95'], 'Beyond the assessment horizon')
        self.assertEqual(t50['reached'], '96.0%')
        self.assertEqual(t50['upper_status'], 'beyond_horizon')
        self.assertEqual(t50['range_95'], '2 to Beyond the assessment horizon')
        self.assertEqual([row['value'] for row in data[0]['quantiles']], [2., 4., None])
        self.assertEqual(warnings, [])
        json.dumps(data, allow_nan=False)

    def test_unresolved_median_and_both_unresolved_bounds_are_explicit(self):
        rows, _, _ = self.present(self.reference((.01, .01, .01, .97)))
        self.assertEqual(rows[0]['median'], 'Beyond the assessment horizon')
        self.assertEqual(rows[0]['lower_95'], '8')
        reference = ReferenceTrajectoryEnsemble(
            [0., 10.], [[1., .9], [1., .8]], [True, True])
        rows, _, _ = self.present(reference)
        self.assertEqual(rows[0]['lower_status'], 'beyond_horizon')
        self.assertEqual(rows[0]['upper_status'], 'beyond_horizon')
        self.assertNotIn('10', rows[0]['range_95'])

    def test_zero_weight_and_invalid_draws_cannot_supply_endpoint_bounds(self):
        reference = ReferenceTrajectoryEnsemble(
            [0., 10.], [[1., -1.5], [1., -.25], [np.nan, np.nan], [0., 0.]],
            [True, True, False, True], draw_weights=[.01, .99, 1., 0.], interpolation='linear')
        rows, _, warnings = self.present(reference)
        self.assertEqual(rows[0]['range_95'], '4 to 4')
        self.assertEqual(rows[0]['reached'], '100.0%')
        self.assertEqual(warnings, [])

    def test_positive_weight_invalid_baseline_is_unavailable(self):
        reference = ReferenceTrajectoryEnsemble(
            [0., 10.], [[1., -.25], [0., -.2]], [True, True])
        rows, data, warnings = self.present(reference)
        self.assertEqual(rows[0]['range_95'], '—')
        self.assertEqual(rows[0]['median'], '—')
        self.assertEqual(rows[0]['reached'], '—')
        self.assertTrue(all(row['status'] == 'invalid_reference' for row in data[0]['quantiles']))
        self.assertIsNone(data[0]['reached_weight'])
        self.assertIn('milestone baseline', warnings[0])
        json.dumps(data, allow_nan=False)

    def test_requested_baseline_defines_all_three_quantiles(self):
        reference = ReferenceTrajectoryEnsemble(
            [0., 2., 10.], [[1., .8, 0.], [1., .8, 0.]], [True, True],
            interpolation='linear')
        rows, data, _ = self.present(reference, baseline=2.)
        self.assertEqual(rows[0]['range_95'], '6 to 6')
        self.assertEqual(rows[0]['median'], '6')
        self.assertEqual(data[0]['baseline_time'], 2.)

    def test_single_reference_does_not_claim_zero_uncertainty(self):
        reference = ReferenceTrajectoryEnsemble([0., 10.], [[1., -.25]], [True],
                                               interpolation='linear')
        rows, data, _ = self.present(reference)
        self.assertEqual(rows[0]['median'], '4')
        self.assertFalse(rows[0]['range_available'])
        self.assertEqual(rows[0]['range_95'], 'Not available (one reference trajectory)')
        self.assertFalse(data[0]['sampled_range_available'])
        json.dumps(data, allow_nan=False)


class InspectionPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_measured_planning(
            TimeCourse(np.arange(5.), [1., .87, .67, .56, .39]),
            dict(model_name='linear', first_time='0', last_time='4', time_step='1',
                 min_times='3', max_times='4', fixed_times='0', include_even='on',
                 reference_draws='8', n_simulations='4', random_seed='2',
                 replicates_per_time='2',
                 manual_schedules='</script><script>alert(1)</script>: 0,2,4'))

    def test_inspection_ids_match_performance_rows_and_repeated_presentation_is_pure(self):
        with patch('utilities.design_planning.compare_designs_by_simulation',
                   side_effect=AssertionError('Presentation must not simulate.')):
            first = present_planning_result(self.result)
            second = present_planning_result(self.result)
        data = first['inspection']
        self.assertEqual(data['schema_version'], 1)
        expected = [(row['id'], row['name']) for row in first['rows']]
        self.assertEqual([(row['id'], row['name']) for row in data['schedule_data']['schedules']], expected)
        self.assertEqual([(row['id'], row['name']) for row in data['recovery']['schedules']], expected)
        np.testing.assert_array_equal(data['reference']['time'], self.result.reference.time_grid)
        np.testing.assert_array_equal(data['reference']['central_response'],
                                      self.result.reference.central_curve)
        self.assertEqual(data['reference']['lower'], first['reference_figure']['data'][0]['y'])
        self.assertEqual(data['reference']['upper'], first['reference_figure']['data'][1]['y'])
        self.assertEqual(first['inspection'], second['inspection'])
        json.dumps(data, allow_nan=False)

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
    def test_template_retains_an_inert_html_safe_payload_and_range_column(self):
        context = present_planning_result(self.result)
        environment = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
                                  autoescape=True, undefined=StrictUndefined)
        html = environment.get_template('design/_planning_results.html').render(**context)
        self.assertIn('Reference central 95% range', html)
        self.assertIn('weighted 2.5th to 97.5th percentile', html)
        self.assertNotIn('</script><script>alert(1)</script>', html)

        class PayloadParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.active = False
                self.parts = []

            def handle_starttag(self, tag, attrs):
                if tag == 'script' and 'data-planning-inspection' in dict(attrs):
                    self.active = True
                    self_type = dict(attrs).get('type')
                    if self_type != 'application/json':
                        raise AssertionError('Inspection data must be inert JSON.')

            def handle_data(self, value):
                if self.active:
                    self.parts.append(value)

            def handle_endtag(self, tag):
                if tag == 'script':
                    self.active = False

        parser = PayloadParser()
        parser.feed(html)
        self.assertEqual(json.loads(''.join(parser.parts)), context['inspection'])

    def test_unavailable_comparison_preserves_endpoint_data_without_inventing_schedules(self):
        result = copy(self.result)
        result.comparison = None
        result.schedules = {}
        result.optimization = None
        context = present_planning_result(result)
        self.assertEqual(context['rows'], [])
        self.assertEqual(context['inspection']['schedule_data']['schedules'], [])
        self.assertEqual(context['inspection']['recovery']['schedules'], [])
        self.assertEqual(len(context['inspection']['endpoints']), 2)
        json.dumps(context['inspection'], allow_nan=False)


@unittest.skipIf(importlib.util.find_spec('flask') is None, 'Flask is supplied by the app environment.')
class InspectionRoute(unittest.TestCase):
    def test_calculation_returns_ranges_and_inspection_data_with_the_same_result(self):
        from design_app import create_app
        client = create_app({'TESTING': True}).test_client()
        response = client.post('/calculate', data={
            'task': 'study', 'reference_source': 'measured', 'model_name': 'linear',
            'first_time': '0', 'last_time': '4', 'time_step': '1',
            'min_times': '3', 'max_times': '3', 'fixed_times': '0',
            'reference_draws': '4', 'n_simulations': '2', 'random_seed': '2',
            'reference_file': (io.BytesIO(b'time,response\n0,1\n1,.87\n2,.67\n3,.56\n4,.39\n'),
                               'reference.csv'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.get_json()['ok'])
        html = response.get_json()['html']
        self.assertIn('Reference central 95% range', html)
        self.assertIn('data-planning-inspection', html)
        self.assertIn('"shared_example"', html)
        self.assertNotIn(': NaN', html)
        self.assertNotIn(': Infinity', html)
        self.assertFalse(response.headers.getlist('Set-Cookie'))


if __name__ == '__main__':
    unittest.main()
