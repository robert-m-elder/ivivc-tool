"""Interactive reference specifications preserve numerical data and uncertainty."""
from copy import copy, deepcopy
import json
import unittest

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import ReferenceTrajectoryEnsemble
from utilities.design_planning import run_measured_planning
from utilities.design_planning_results import present_planning_result
from utilities.design_plotly import make_figure


def figure(series=(), **options):
    return make_figure(series, title='Reference', x_label='Time',
                       y_label='Response', identifier='example', **options)


class FigureCoordinates(unittest.TestCase):
    def test_lines_use_original_values_and_keep_independent_marker_series(self):
        value = figure([
            dict(label='Prior', color='#123456', dash='6 4', join=True,
                 values=[(10., .8), (20., .4)]),
            dict(label='Manual', color='#654321', join=False, values=[(15., .6)]),
        ])
        prior, manual = value['data']
        self.assertEqual(prior['x'], [10., 20.])
        self.assertEqual(prior['y'], [.8, .4])
        self.assertEqual(prior['line']['dash'], 'dash')
        self.assertEqual(prior['mode'], 'lines')
        self.assertEqual(manual['mode'], 'markers')
        self.assertEqual(value['layout']['uirevision'], 'example')

    def test_nonfinite_line_coordinates_are_json_null_gaps(self):
        value = figure([dict(label='Curve', color='#123456', values=[
            (0., 1.), (1., np.nan), (np.inf, .6), (3., .3)])])
        trace, = value['data']
        self.assertEqual(trace['x'], [0., None, None, 3.])
        self.assertEqual(trace['y'], [1., None, None, .3])
        self.assertIs(trace['connectgaps'], False)
        json.dumps(value, allow_nan=False)

    def test_pointwise_band_segments_do_not_fill_across_missing_or_reversed_bounds(self):
        value = figure(band=[(0., .8, 1.), (1., .6, .9), (2., np.nan, .8),
                             (3., .4, .7), (4., .3, .6), (5., .7, .2),
                             (6., .1, .3), (7., 0., .2)])
        self.assertEqual([trace['x'] for trace in value['data']],
                         [[0., 1.], [0., 1.], [3., 4.], [3., 4.], [6., 7.], [6., 7.]])
        self.assertEqual([trace['showlegend'] for trace in value['data']],
                         [False, True, False, False, False, False])
        self.assertTrue(all(trace['legendgroup'] == 'range' for trace in value['data']))
        json.dumps(value, allow_nan=False)

    def test_invalid_observations_are_omitted_and_unavailable_figures_stay_unavailable(self):
        value = figure(observations=[(0., 1.), (1., np.nan), (2., .4)])
        points, = value['data']
        self.assertEqual(points['x'], [0., 2.])
        self.assertEqual(points['y'], [1., .4])
        self.assertIsNone(figure([dict(label='Missing', color='#123456',
                                      values=[(0., np.nan), (np.inf, .1)])],
                                band=[(0., np.nan, .5)], observations=[(np.inf, .1)]))

    def test_labels_are_escaped_before_plotly_interprets_display_text(self):
        value = make_figure([dict(label='<b>Source</b>', color='#123456', values=[(0., 1.)])],
                            title='<script>title</script>', x_label='Time <unit>',
                            y_label='Response & value', identifier='safe')
        self.assertEqual(value['data'][0]['name'], '&lt;b&gt;Source&lt;/b&gt;')
        self.assertEqual(value['layout']['title']['text'], '&lt;script&gt;title&lt;/script&gt;')
        self.assertIn('Time &lt;unit&gt;', value['data'][0]['hovertemplate'])
        self.assertIn('Response &amp; value', value['data'][0]['hovertemplate'])


class ReferenceSpecifications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TimeCourse(np.arange(5.), [1., .87, .67, .56, .39],
            observation_uncertainty=ResolvedObservationUncertainty(
                np.full(5, .03), response_kind='mean', replicate_counts=np.full(5, 4)))
        settings = dict(model_name='linear', first_time='0', last_time='4', time_step='1',
                        min_times='3', max_times='3', fixed_times='0',
                        reference_draws='4', n_simulations='2', random_seed='2',
                        replicates_per_time='2')
        cls.measured = run_measured_planning(cls.source, settings)
        cls.completed = TimeCourse([0., 1., 2.], [1., .84, .64],
            observation_uncertainty=ResolvedObservationUncertainty(
                [.03] * 3, response_kind='mean', replicate_counts=[4, 9, 16]))
        cls.adaptive = run_measured_planning(cls.source,
            dict(settings, has_completed='on', measured_adaptive_mode='reweight_prior',
                 future_policy='future_only', min_times='4', max_times='4'),
            completed=cls.completed)

    def test_presenter_preserves_weighted_reference_band_and_original_observations(self):
        result = copy(self.measured)
        result.reference = ReferenceTrajectoryEnsemble(
            np.arange(5.), [[1., .8, .6, .4, .2], [2., 1.8, 1.6, 1.4, 1.2],
                           [np.nan] * 5], [True, True, False], draw_weights=[.99, .01, 0.])
        context = present_planning_result(result)
        value = context['reference_figure']
        lower, upper = value['data'][:2]
        np.testing.assert_array_equal(lower['y'], result.reference.value_draws[0])
        np.testing.assert_array_equal(upper['y'], result.reference.value_draws[0])
        self.assertEqual(lower['x'], list(result.reference.time_grid))
        points = value['data'][-1]
        self.assertEqual(points['name'], 'Source observations')
        np.testing.assert_array_equal(points['x'], self.source.time)
        np.testing.assert_array_equal(points['y'], self.source.value)
        np.testing.assert_array_equal(self.source.observation_uncertainty.sigma, [.03] * 5)
        self.assertIsNotNone(context['reference_plot'])
        json.dumps(value, allow_nan=False)

    def test_adaptive_reference_retains_prior_updated_curve_and_fixed_completed_means(self):
        value = present_planning_result(self.adaptive)['reference_figure']
        traces = {trace['name']: trace for trace in value['data']}
        np.testing.assert_array_equal(traces['Original prior reference']['y'],
                                      self.adaptive.prior_reference.central_curve)
        np.testing.assert_array_equal(traces['Updated reference']['y'],
                                      self.adaptive.reference.central_curve)
        self.assertEqual(traces['Original prior reference']['line']['dash'], 'dash')
        np.testing.assert_array_equal(traces['Completed observations']['x'], self.completed.time)
        np.testing.assert_array_equal(traces['Completed observations']['y'], self.completed.value)
        np.testing.assert_array_equal(self.completed.observation_uncertainty.sigma, [.03] * 3)

    def test_single_reference_has_no_invented_band_or_additional_sampling(self):
        result = copy(self.measured)
        result.reference = ReferenceTrajectoryEnsemble(np.arange(5.),
            [[1., .8, .6, .4, .2]], [True])
        one = present_planning_result(result)['reference_figure']
        two = present_planning_result(result)['reference_figure']
        self.assertFalse(any(trace.get('fill') for trace in one['data']))
        self.assertEqual(one, two)
        self.assertEqual(len(one['data']), 2)


class PerformanceSpecifications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = TimeCourse(np.arange(5.), [1., .87, .67, .56, .39])
        settings = dict(model_name='linear', first_time='0', last_time='4', time_step='1',
                        min_times='3', max_times='4', fixed_times='0', include_even='on',
                        reference_draws='8', n_simulations='4', random_seed='2',
                        replicates_per_time='2', manual_schedules='Comparison: 0,2,4')
        cls.result = run_measured_planning(source, settings)
        completed = TimeCourse([0., 1., 2.], [1., .84, .64],
            observation_uncertainty=ResolvedObservationUncertainty(
                [.03] * 3, response_kind='mean', replicate_counts=[4, 9, 16]))
        cls.adaptive = run_measured_planning(source,
            dict(settings, has_completed='on', measured_adaptive_mode='reweight_prior',
                 future_policy='future_only', min_times='4', max_times='4',
                 manual_schedules=''), completed=completed)

    def test_interactive_series_preserve_the_existing_static_plot_values(self):
        context = present_planning_result(self.result)
        value = context['performance_figure']
        expected = context['performance_plot']['series']
        self.assertEqual(len(value['data']), len(expected))
        for trace, series in zip(value['data'], expected):
            self.assertEqual(trace['name'], series['label'])
            self.assertEqual(trace['line']['color'], series['color'])
            np.testing.assert_array_equal(trace['x'], [point[0] for point in series['values']])
            np.testing.assert_array_equal(trace['y'], [point[1] for point in series['values']])
            self.assertIn('%{fullData.name}', trace['hovertemplate'])
        self.assertEqual(value['layout']['xaxis']['dtick'], 1)
        self.assertEqual(value['layout']['xaxis']['title']['text'], context['performance_x_label'])
        self.assertEqual(value['layout']['uirevision'], 'planning-performance-plot')
        self.assertEqual(value['layout']['height'], 460)
        self.assertEqual(value['layout']['margin']['b'], 160)
        self.assertEqual(value['layout']['legend']['yanchor'], 'top')
        json.dumps(value, allow_nan=False)

    def test_mean_and_percentile_markers_identify_manual_comparisons_without_joining_them(self):
        value = present_planning_result(self.result)['performance_figure']
        manual = [trace for trace in value['data'] if trace['name'].startswith('Manual:')]
        self.assertEqual([trace['name'] for trace in manual], ['Manual: Mean', 'Manual: 90th percentile'])
        self.assertTrue(all(trace['mode'] == 'markers' for trace in manual))
        self.assertEqual([trace['marker']['symbol'] for trace in manual], ['circle', 'diamond'])
        suggested = [trace for trace in value['data'] if trace['name'].startswith('Suggested:')]
        self.assertEqual([trace['line']['dash'] for trace in suggested], ['solid', 'dash'])
        self.assertTrue(all(trace['mode'] == 'lines+markers' for trace in suggested))

    def test_singleton_adaptive_study_size_remains_visible_and_counts_only_new_times(self):
        context = present_planning_result(self.adaptive)
        value = context['performance_figure']
        self.assertEqual(value['layout']['xaxis']['title']['text'], 'New sampling times')
        for trace in value['data']:
            self.assertEqual(trace['x'], [1.])
            self.assertEqual(trace['mode'], 'lines+markers')
        self.assertEqual(context['rows'][0]['number_of_times'], 4)
        self.assertEqual(context['rows'][0]['number_of_new_times'], 1)

    def test_nonfinite_metrics_stay_json_gaps_and_do_not_join_adjacent_results(self):
        result = deepcopy(self.result)
        summary = next(iter(result.comparison.summaries.values()))
        summary.mean_irmse, summary.p90_irmse = np.nan, np.inf
        value = present_planning_result(result)['performance_figure']
        for trace in value['data'][:2]:
            self.assertIsNone(trace['x'][0])
            self.assertIsNone(trace['y'][0])
            self.assertIsNotNone(trace['y'][1])
            self.assertIs(trace['connectgaps'], False)
        json.dumps(value, allow_nan=False)

    def test_unavailable_performance_or_comparison_does_not_get_a_synthetic_figure(self):
        result = deepcopy(self.result)
        for summary in result.comparison.summaries.values():
            summary.mean_irmse, summary.p90_irmse = np.nan, np.nan
        self.assertIsNone(present_planning_result(result)['performance_figure'])
        result.comparison, result.optimization, result.schedules = None, None, {}
        context = present_planning_result(result)
        self.assertIsNone(context['performance_figure'])
        self.assertIsNone(context['performance_plot'])


if __name__ == '__main__':
    unittest.main()
