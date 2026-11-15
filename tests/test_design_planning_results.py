"""Presentation checks using actual small Monte Carlo planning results."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from ivivc_core import TimeCourse
from study_design_core import ReferenceTrajectoryEnsemble
from utilities.design_planning import run_measured_planning
from utilities.design_planning_results import _reference_band, present_planning_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class PlanningResultPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_measured_planning(
            TimeCourse(np.arange(5.), np.array([1., .87, .67, .56, .39])),
            {'model_name': 'linear', 'first_time': '0', 'last_time': '4', 'time_step': '1',
             'min_times': '3', 'max_times': '4', 'fixed_times': '0',
             'reference_draws': '8', 'n_simulations': '4', 'random_seed': '2',
             'replicates_per_time': '2', 'include_even': 'on',
             'manual_schedules': 'My design: 0, 2, 4'},
        )

    def present(self, result=None, **options):
        return present_planning_result(result or self.result,
                                       source_filename=options.pop('filename', 'reference.csv'),
                                       **options)

    def test_schedule_counts_and_performance_match_actual_results(self):
        context = self.present()
        self.assertEqual(len(context['rows']), 5)
        for row in context['rows']:
            summary = self.result.comparison.summaries[row['name']]
            self.assertEqual(row['experimental_units'], len(summary.design_times) * 2)
            self.assertEqual(row['n_successful'], summary.n_successful)
            self.assertEqual(row['n_simulations'], 4)
            self.assertAlmostEqual(float(row['mean_irmse']), summary.mean_irmse, places=5)
        self.assertEqual(context['rows'][-1]['kind'], 'Manual')

    def test_nonfinite_errors_are_unavailable_and_failures_are_explicit(self):
        result = deepcopy(self.result)
        summary = next(iter(result.comparison.summaries.values()))
        summary.n_successful, summary.failure_rate = 0, 1.
        summary.mean_irmse, summary.p90_irmse = np.nan, np.inf
        summary.mean_curve_interval_width = np.nan
        context = self.present(result)
        row = context['rows'][0]
        self.assertEqual(row['n_successful'], 0)
        self.assertEqual(row['failure_rate'], '100.0%')
        self.assertEqual(row['mean_irmse'], '—')
        self.assertEqual(row['p90_irmse'], '—')
        self.assertEqual(row['mean_curve_interval_width'], '—')
        self.assertTrue(any('failed' in warning for warning in context['warnings']))
        for line in context['performance_plot']['series']:
            self.assertNotIn('nan', line['path'].lower())
            self.assertNotIn('inf', line['path'].lower())

    def test_reference_band_respects_weights_and_excludes_invalid_rows(self):
        reference = ReferenceTrajectoryEnsemble(
            [0., 1.], [[1., .8], [3., 2.], [np.nan, 999.]],
            [True, True, False], draw_weights=[.99, .01, 0.],
        )
        lower, upper = _reference_band(reference)
        np.testing.assert_array_equal(lower, [1., .8])
        np.testing.assert_array_equal(upper, [1., .8])

    def test_partial_interval_availability_is_reported_separately_from_fit_success(self):
        result = deepcopy(self.result)
        simulated = next(iter(result.comparison.results.values()))
        available_before = np.count_nonzero(simulated.fit_success & np.isfinite(simulated.mean_curve_interval_width))
        first = np.flatnonzero(simulated.fit_success & np.isfinite(simulated.mean_curve_interval_width))[0]
        simulated.mean_curve_interval_width[first] = np.nan
        context = self.present(result)
        self.assertEqual(context['rows'][0]['interval_count'], available_before - 1)
        self.assertEqual(context['rows'][0]['n_successful'], simulated.n_successful)
        self.assertTrue(any('number of usable fits' in warning for warning in context['warnings']))

    def test_reported_information_rank_uses_the_displayed_refined_schedules(self):
        context = self.present()
        for row in context['ranks']:
            step = self.result.optimization.design_for_n(row['number_of_times'], refined=True)
            self.assertEqual(row['rank'], step.score.min_rank)
            self.assertEqual(row['parameters'], step.score.parameter_count)

    def test_provenance_and_numerical_settings_remain_in_context(self):
        summary = {'worksheet': 'Week data', 'mapping': {'Time': 'week'},
                   'interpretation': 'Means with supplied SEM; no further division.',
                   'retained_rows': 5, 'omitted_rows': 0}
        context = self.present(source_summary=summary)
        self.assertEqual(context['source_summary'], summary)
        settings = dict(context['settings_rows'])
        self.assertEqual(settings['Model'], 'linear')
        self.assertEqual(settings['Independent replicates at each time'], '2')
        self.assertEqual(settings['Random seed'], '2')
        self.assertEqual(settings['Requested reference draws'], '8')
        self.assertEqual(settings['Monte Carlo studies per schedule'], '4')

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask application environment.')
    def test_template_escapes_uploaded_names_and_does_not_render_nonfinite_numbers(self):
        result = deepcopy(self.result)
        summary = next(iter(result.comparison.summaries.values()))
        summary.mean_irmse, summary.p90_irmse = np.nan, np.inf
        summary.mean_curve_interval_width = np.nan
        context = self.present(result, filename='<script>alert(1)</script>.csv',
                               source_summary={'mapping': {'Response': '<img src=x>'}})
        environment = Environment(
            loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
            autoescape=True, undefined=StrictUndefined,
        )
        html = environment.get_template('design/_planning_results.html').render(**context)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('&lt;img src=x&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x>', html)
        self.assertNotIn('NaN', html)
        self.assertNotIn('Infinity', html)
        self.assertIn('successful simulated fits', html)
        self.assertIn('first planning time', html)
        self.assertIn('excludes future measurement variability', html)
        self.assertEqual(html.count('<svg '), 2)


if __name__ == '__main__':
    unittest.main()
