"""Reference-specific presentation checks using complete planning calculations."""
from copy import deepcopy
from pathlib import Path
import unittest
import warnings

import numpy as np

from ivivc_core import TimeCourse
from study_design_core import EmpiricalTimeScaleDistribution
from utilities.design_planning import run_reference_planning
from utilities.design_planning_results import present_planning_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class OtherReferencePresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.linspace(0., 8., 9)
        values = np.exp(-.2 * x)
        noisy = values * np.array([1., 1.002, .999, 1.001, 1., .998, 1.001, 1., 1.])
        vitro, vivo, target = TimeCourse(x, noisy), TimeCourse(2 * x, noisy), TimeCourse(x, values)
        form = {
            'task': 'study', 'reference_source': 'ivivc', 'model_name': 'linear',
            'first_time': '0', 'last_time': '4', 'time_step': '1',
            'min_times': '3', 'max_times': '3', 'fixed_times': '0',
            'reference_draws': '4', 'n_simulations': '3', 'random_seed': '2',
            'ivivc_model': 'exponential_normalized',
        }
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            cls.ivivc = run_reference_planning(
                form, in_vitro=vitro, in_vivo=vivo, prediction=target)
            cls.scaling = run_reference_planning(
                dict(form, reference_source='scaling', scaling_model='exponential_normalized',
                     last_time='12', time_step='2'),
                scaling_source=vitro,
                distribution=EmpiricalTimeScaleDistribution([2., 3.], weights=[3., 1.]),
            )
            cls.deterministic = run_reference_planning(
                dict(form, ivivc_approach='approach3', ivivc_method='value_scaling'),
                in_vitro=vitro, in_vivo=vivo, prediction=target,
            )

    def test_ivivc_reference_does_not_label_source_points_as_in_vivo_reference_measurements(self):
        context = present_planning_result(self.ivivc)
        self.assertEqual(context['source_kind'], 'ivivc')
        self.assertEqual(context['reference_plot']['observations'], [])
        self.assertEqual(len(context['source_plots']), 3)
        self.assertEqual([plot['x_label'] for plot in context['source_plots']],
                         ['In vitro time', 'In vivo time', 'In vitro time'])
        self.assertNotIn('Measured time range', dict(context['settings_rows']))
        self.assertTrue(any('approach2:exponential_normalized' == value
                            for _, value in context['source_details_rows']))

    def test_source_fit_preview_keeps_the_fitted_model_and_its_parameters(self):
        context = present_planning_result(self.ivivc)
        self.assertTrue(context['source_plots'][0]['plot']['series'])
        self.assertTrue(context['source_plots'][1]['plot']['series'])
        self.assertFalse(context['source_plots'][2]['plot']['series'])
        self.assertTrue(context['source_fit_rows'])
        self.assertTrue(all(row['parameter'] and row['value'] != '—'
                            for row in context['source_fit_rows']))

    def test_time_scaling_reports_parameter_draws_per_ratio_and_total_reference_curves(self):
        context = present_planning_result(self.scaling)
        settings = dict(context['settings_rows'])
        self.assertEqual(context['source_kind'], 'scaling')
        self.assertEqual(settings['Requested parameter draws per active ratio'], '4')
        self.assertEqual(settings['Usable reference draws'], '8 of 8 generated')
        self.assertEqual(context['reference_plot']['observations'], [])
        self.assertTrue(context['reference_range_available'])
        self.assertEqual([row['weight'] for row in context['scale_records']], ['0.75', '0.25'])
        self.assertEqual(context['scale_record_count'], 2)
        np.testing.assert_allclose(np.sum(self.scaling.reference.draw_weights[:4]), .75)
        self.assertFalse(any('planning horizon extends beyond the measured time range' in text.lower()
                             for text in context['warnings']))

    def test_deterministic_reference_does_not_show_a_zero_width_uncertainty_band(self):
        context = present_planning_result(self.deterministic)
        self.assertEqual(self.deterministic.reference.n_valid_draws, 1)
        self.assertFalse(context['reference_range_available'])
        self.assertEqual(context['reference_plot']['band'], '')
        self.assertTrue(any('no sampled reference-uncertainty range' in warning
                            for warning in context['warnings']))
        self.assertGreater(float(context['rows'][0]['mean_irmse']), 0.)

    def test_nonfinite_source_fit_statistics_display_as_unavailable(self):
        result = deepcopy(self.ivivc)
        result.source_details['rows'].append(('Unavailable source diagnostic', np.nan))
        result.source_details['fit_rows'][0]['se'] = np.inf
        context = present_planning_result(result)
        self.assertIn(('Unavailable source diagnostic', '—'), context['source_details_rows'])
        self.assertEqual(context['source_fit_rows'][0]['se'], '—')

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask application environment.')
    def test_multiple_source_files_mappings_and_selection_are_escaped_in_the_template(self):
        result = deepcopy(self.ivivc)
        result.source_details['rows'].append(('Source note', '<script>bad()</script>'))
        entries = [
            {'role': 'ivivc-calibration-file', 'label': 'In vitro calibration',
             'filename': '<calibration>.csv', 'worksheet': 'Measured',
             'summary': {'mapping': {'Response': '<img src=x>'},
                         'retained_rows': 9, 'omitted_rows': 0}},
            {'role': 'prediction-file', 'label': 'Target in vitro',
             'filename': 'target.csv', 'worksheet': None,
             'summary': {'interpretation': 'Supplied target SEM; no further division.',
                         'mapping': {'Time': 'age'}, 'retained_rows': 9, 'omitted_rows': 0}},
        ]
        context = present_planning_result(result, source_summaries=entries)
        environment = Environment(
            loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
            autoescape=True, undefined=StrictUndefined,
        )
        html = environment.get_template('design/_planning_results.html').render(**context)
        self.assertIn('&lt;calibration&gt;.csv', html)
        self.assertIn('&lt;img src=x&gt;', html)
        self.assertIn('&lt;script&gt;bad()&lt;/script&gt;', html)
        self.assertIn('target.csv', html)
        self.assertIn('Measured', html)
        self.assertIn('no further division', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('Dark points are the supplied reference observations', html)
        self.assertIn('separate choice from the IVIVC model', html)
        self.assertEqual(html.count('<svg '), 5)

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask application environment.')
    def test_deterministic_template_explains_future_variability_still_enters_simulation(self):
        environment = Environment(
            loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
            autoescape=True, undefined=StrictUndefined,
        )
        html = environment.get_template('design/_planning_results.html').render(
            **present_planning_result(self.deterministic))
        self.assertIn('No sampled reference-uncertainty band', html)
        self.assertIn('Future measurement variability still enters', html)
        self.assertIn('deterministic', html)


if __name__ == '__main__':
    unittest.main()
