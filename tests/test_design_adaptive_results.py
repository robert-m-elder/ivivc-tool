"""Adaptive and timing presentation checks using small real planning runs."""
from pathlib import Path
import unittest
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from study_design_core import EmpiricalTimeScaleDistribution
from utilities.design_planning import run_measured_planning, run_reference_planning
from utilities.design_planning_results import present_planning_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


def course(time, response, *, sigma=.03, kind='individual', counts=None):
    return prepare_time_course(time, response, observation_uncertainty=ObservationUncertainty(
        sigma, response_kind=kind, replicate_counts=counts))[0]


def form(**changes):
    values = dict(model_name='linear', first_time='0', last_time='4', time_step='1',
                  min_times='4', max_times='5', fixed_times='0', reference_draws='8',
                  n_simulations='4', random_seed='2', replicates_per_time='2',
                  include_even='on', has_completed='on', future_policy='future_only')
    values.update(changes)
    return values


def run(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        return function(*args, **kwargs)


class AdaptiveResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.completed = course([0., 1., 2.], [1., .84, .64], kind='mean', counts=[4, 9, 16])
        cls.refitted = run(run_measured_planning, None, form(), completed=cls.completed)
        prior = course(np.arange(5.), [1.02, .84, .63, .42, .24], sigma=.08)
        cls.updated = run(run_measured_planning, prior,
                          form(measured_adaptive_mode='reweight_prior'), completed=cls.completed)

    def test_new_time_and_unit_counts_exclude_completed_measurements(self):
        context = present_planning_result(self.refitted)
        self.assertEqual(context['performance_x_label'], 'New sampling times')
        self.assertEqual(dict(context['adaptive']['rows'])['Completed independent units reported'], '29')
        for row in context['rows']:
            self.assertEqual(row['number_of_new_times'], row['number_of_times'] - 3)
            self.assertEqual(row['experimental_units'], row['number_of_new_times'] * 2)
        suggested = context['performance_plot']['series'][0]
        self.assertEqual([value[0] for value in suggested['values']], [1, 2])

    def test_completed_means_and_counts_are_reported_without_reducing_sem_again(self):
        context = present_planning_result(self.refitted)
        adaptive = context['adaptive']
        self.assertFalse(adaptive['has_update'])
        self.assertIn('not applied again', adaptive['method'])
        self.assertIn('counts do not divide it again', adaptive['interpretation'])
        self.assertEqual([float(row['uncertainty']) for row in adaptive['preview']], [.03] * 3)
        self.assertEqual([int(row['count']) for row in adaptive['preview']], [4, 9, 16])
        self.assertEqual(len(context['reference_plot']['observations']), 3)

    def test_counts_for_reported_means_are_not_invented_when_metadata_is_missing(self):
        result = run(run_measured_planning, None, form(), completed=course(
            [0., 1., 2.], [1., .84, .64], kind='mean'))
        adaptive = present_planning_result(result)['adaptive']
        self.assertEqual(dict(adaptive['rows'])['Completed independent units reported'],
                         'Not supplied for the reported means')

    def test_explicit_future_error_assumption_is_distinguished_from_supplied_sd(self):
        result = run(run_measured_planning, None, form(completed_error_mode='future_model'),
                     completed=TimeCourse([0., 1., 2.], [1., .84, .64]))
        context = present_planning_result(result)
        self.assertIn('assumed from the future measurement model', context['adaptive']['interpretation'])
        self.assertEqual([float(row['uncertainty']) for row in context['adaptive']['preview']], [.05] * 3)

    def test_prior_and_updated_reference_and_weight_concentration_are_distinct(self):
        context = present_planning_result(self.updated)
        self.assertTrue(context['adaptive']['has_update'])
        self.assertIn('updated once', context['adaptive']['method'])
        labels = [line['label'] for line in context['reference_plot']['series']]
        self.assertEqual(labels, ['Original prior reference', 'Updated reference'])
        rows = dict(context['adaptive']['rows'])
        self.assertAlmostEqual(float(rows['Effective number of reference trajectories after updating']),
                               self.updated.adaptive_update.effective_sample_size, places=4)

    def test_scaling_weights_show_original_and_updated_record_marginals(self):
        time = np.linspace(0., 8., 9)
        source = course(time, np.exp(-.2 * time), sigma=.01)
        result = run(run_reference_planning,
                     form(reference_source='scaling', scaling_model='exponential_normalized'),
                     scaling_source=source, distribution=EmpiricalTimeScaleDistribution([2., 3.]),
                     completed=self.completed)
        context = present_planning_result(result)
        self.assertTrue(context['has_updated_scale_weights'])
        np.testing.assert_allclose([float(row['weight']) for row in context['scale_records']], [.5, .5])
        expected = result.adaptive_update.metadata['time_scaling_weight_summary']['current_weights']
        np.testing.assert_allclose([float(row['updated_weight']) for row in context['scale_records']],
                                   expected, rtol=1e-4, atol=1e-5)

    def test_milestone_presentation_uses_the_requested_baseline(self):
        source = course(np.arange(5.), 1 - .2 * np.arange(5.), sigma=.001)
        result = run(run_measured_planning, source,
                     form(has_completed='', min_times='3', max_times='3', baseline_time='1'))
        context = present_planning_result(result)
        self.assertEqual(context['milestone_baseline_time'], '1')
        self.assertAlmostEqual(float(context['endpoints'][0]['median']), 3., delta=.03)

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
    def test_adaptive_template_reports_fixed_data_and_new_burden(self):
        context = present_planning_result(self.updated, source_filename='<prior>.csv')
        environment = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
                                  autoescape=True, undefined=StrictUndefined)
        html = environment.get_template('design/_planning_results.html').render(**context)
        self.assertIn('&lt;prior&gt;.csv', html)
        self.assertIn('Completed observations and reference update', html)
        self.assertIn('New independent units', html)
        self.assertIn('not simulated again', html)
        self.assertIn('not counts of independent animals', html)
        self.assertNotIn('NaN', html)


class TimingResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = course(np.arange(5.), 1 - .1 * np.arange(5.), sigma=.001)
        cls.unresolved = run(run_measured_planning, source, form(
            has_completed='', min_times='3', max_times='4', include_even='',
            timing_0_kind='latest_before', timing_0_label='<T90>',
            timing_0_upper_mode='milestone', timing_0_upper_fraction='.9',
            timing_0_upper_quantile='.95'))
        cls.resolved = run(run_measured_planning, source, form(
            has_completed='', min_times='3', max_times='4',
            timing_0_kind='latest_before', timing_0_label='Early sample',
            timing_0_upper_mode='numeric', timing_0_upper_time='1',
            manual_schedules='Late comparison: 0,3,4'))

    def test_unresolved_requirement_retains_status_without_simulated_error(self):
        self.assertIsNone(self.unresolved.comparison)
        context = present_planning_result(self.unresolved)
        self.assertFalse(context['has_comparison'])
        self.assertEqual(context['rows'], [])
        self.assertIsNone(context['performance_plot'])
        self.assertTrue(context['timing']['requirements'])
        self.assertTrue(all(row['status'] != 'feasible' for row in context['timing']['outcomes']))
        self.assertIn('90% degradation', context['timing']['requirements'][0]['upper_description'])

    def test_latest_selection_and_comparison_inspection_remain_separate(self):
        context = present_planning_result(self.resolved)
        anchors = [anchor for outcome in context['timing']['outcomes'] for anchor in outcome['anchors']]
        self.assertTrue(anchors)
        self.assertTrue(all(float(anchor['selected_time']) == 1 for anchor in anchors))
        late = next(row for row in context['rows'] if row['name'] == 'Manual · Late comparison')
        self.assertIsNotNone(late['timing'])
        self.assertNotEqual(late['timing']['status'], 'satisfied')

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
    def test_unavailable_template_is_reviewable_and_does_not_invent_recovery_metrics(self):
        environment = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
                                  autoescape=True, undefined=StrictUndefined)
        template = environment.get_template('design/_planning_results.html')
        html = template.render(**present_planning_result(self.unresolved))
        self.assertIn('&lt;T90&gt;', html)
        self.assertIn('No schedule was available for simulation', html)
        self.assertIn('not evidence of infeasibility', html)
        self.assertNotIn('Mean IRMSE</th>', html)
        self.assertNotIn('NaN', html)
        compared = template.render(**present_planning_result(self.resolved))
        self.assertIn('does not verify latestness', compared)
        self.assertIn('data-schedule-timing', compared)
        self.assertIn('Latest-before selection retained', compared)
        self.assertNotIn('Latest-before selections for', compared)


if __name__ == '__main__':
    unittest.main()
