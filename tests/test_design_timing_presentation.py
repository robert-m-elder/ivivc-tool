"""Consolidated timing summaries preserve the retained outcome distinctions."""
from copy import deepcopy
import unittest

import numpy as np

from ivivc_core import TimeCourse
from utilities.design_planning import run_measured_planning
from utilities.design_planning_results import (
    _timing_flag, _timing_presentation, present_planning_result,
)


def requirement(label='Early window', kind='window', status='ready', message='Resolved.'):
    return dict(label=label, kind=kind, status=status, message=message,
                lower=0., upper=4., effective_lower=0., fixed_times=(),
                minimum_count=1, count_scope='new')


def outcome(size, status='feasible', optimization='complete', message='Retained outcome.'):
    return dict(number_of_times=size, status=status, optimization_status=optimization,
                message=message, times=(0., 2., 4.) if status == 'feasible' else (), anchors=[])


class TimingOverview(unittest.TestCase):
    def test_no_requirements_has_no_timing_overview_or_flag(self):
        result = _timing_presentation({})
        self.assertIsNone(result['overview'])
        self.assertEqual(result['outcome_groups'], [])
        self.assertIsNone(_timing_flag(None, has_requirements=False))

    def test_available_sizes_are_compact_without_filling_missing_sizes(self):
        details = dict(requirements=[requirement(), requirement('By deadline', 'latest_before')],
                       outcomes=[outcome(6), outcome(3), outcome(4)])
        original = deepcopy(details)
        result = _timing_presentation(details)
        self.assertEqual(result['overview']['requirements_text'], '1 window and 1 deadline requested.')
        self.assertEqual(result['overview']['available_sizes'], '3–4, 6')
        self.assertIn('3–4, 6 total sampling times', result['overview']['availability_text'])
        self.assertEqual(result['overview']['exceptions'], [])
        self.assertEqual(result['outcome_groups'], [dict(
            sizes='3–4, 6', status='feasible', optimization_status='complete', messages=['Retained outcome.'])])
        self.assertEqual(details, original)

    def test_infeasible_unknown_and_interrupted_feasible_sizes_stay_distinct(self):
        details = dict(requirements=[requirement()], outcomes=[
            outcome(2, 'infeasible', 'not_run', 'Too few times.'),
            outcome(3, 'infeasible', 'not_run', 'Too few times.'),
            outcome(4, 'search_limit', 'not_run', 'Budget exhausted.'),
            outcome(5, 'unavailable', 'not_run', 'Bounds unresolved.'),
            outcome(6, optimization='search_limit'),
            outcome(7, optimization='exchange_limit'),
        ])
        result = _timing_presentation(details)
        overview = result['overview']
        self.assertEqual(overview['available_sizes'], '6–7')
        labels = [item['label'] for item in overview['exceptions']]
        self.assertEqual(len(labels), 5)
        self.assertIn('Infeasible study sizes: ', labels[0].replace(' — ', ': '))
        self.assertIn('2–3 total times', labels[0])
        self.assertIn('Availability undetermined: search limit', labels[1])
        self.assertIn('Schedule availability unavailable', labels[2])
        self.assertIn('Feasible; optimization search limit reached', labels[3])
        self.assertIn('Feasible; exchange limit reached', labels[4])
        self.assertNotIn('infeasible', overview['exceptions'][1]['message'].lower())
        self.assertEqual(len(result['outcomes']), 6, 'Full retained outcomes remain available.')

    def test_requirement_failures_are_grouped_without_losing_labels_or_reason(self):
        details = dict(requirements=[
            requirement('First milestone', status='beyond_horizon', message='Crossing beyond horizon.'),
            requirement('Second milestone', status='beyond_horizon', message='Crossing beyond horizon.'),
            requirement('Bad bounds', status='invalid_window', message='Lower bound follows upper.'),
        ], outcomes=[outcome(3, 'unavailable', 'not_run')])
        result = _timing_presentation(details)
        exceptions = result['overview']['exceptions']
        self.assertEqual(len(exceptions), 3)
        self.assertEqual(exceptions[0]['label'], 'Unresolved milestone: First milestone, Second milestone')
        self.assertEqual(exceptions[0]['message'], 'Crossing beyond horizon.')
        self.assertEqual(exceptions[1]['status'], 'invalid_window')
        self.assertEqual(exceptions[2]['status'], 'unavailable')
        self.assertEqual(result['requirements'][0]['status'], 'beyond_horizon')

    def test_missing_outcomes_are_unassessed_instead_of_infeasible(self):
        result = _timing_presentation(dict(requirements=[requirement()]))
        self.assertEqual(result['overview']['availability_text'], 'Suggested-schedule availability has not been assessed.')
        self.assertEqual(result['overview']['exceptions'], [])

    def test_grouped_diagnostics_deduplicate_messages_but_keep_distinct_search_statuses(self):
        result = _timing_presentation(dict(requirements=[requirement()], outcomes=[
            outcome(3, 'infeasible', 'not_run', 'Count conflict.'),
            outcome(4, 'infeasible', 'not_run', 'Spacing conflict.'),
            outcome(5, 'infeasible', 'not_run', 'Count conflict.'),
            outcome(6), outcome(7, optimization='exchange_limit'),
        ]))
        self.assertEqual(result['outcome_groups'][0], dict(
            sizes='3–5', status='infeasible', optimization_status='not_run',
            messages=['Count conflict.', 'Spacing conflict.']))
        self.assertEqual(len(result['outcome_groups']), 3)

    def test_schedule_flags_distinguish_failure_unavailability_and_deadline_eligibility(self):
        self.assertIsNone(_timing_flag(dict(status='satisfied'), has_requirements=True))
        self.assertEqual(_timing_flag(None, has_requirements=True)['label'], 'Timing not assessed')
        for status, label in [('unsatisfied', 'Timing not met'), ('unavailable', 'Timing unavailable'),
                              ('not_past_deadline', 'Latestness unverified')]:
            flag = _timing_flag(dict(status=status, message='Retained assessment.'), has_requirements=True)
            self.assertEqual(flag, dict(label=label, title='Retained assessment.'))
        self.assertIsNone(_timing_flag(dict(status='not_past_deadline'), has_requirements=True, certified=True))

    def test_unrecognized_outcomes_remain_explicit(self):
        result = _timing_presentation(dict(requirements=[requirement()], outcomes=[
            outcome(3, 'future_status', 'not_run', 'An additional retained outcome.')]))
        exception, = result['overview']['exceptions']
        self.assertEqual(exception['status'], 'future_status')
        self.assertIn('future status', exception['label'])
        self.assertEqual(exception['message'], 'An additional retained outcome.')


class TimingSelectionFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_measured_planning(
            TimeCourse(np.arange(5.), np.array([1., .89, .75, .65, .52])),
            dict(model_name='linear', first_time='0', last_time='4', time_step='1',
                 min_times='3', max_times='3', fixed_times='0', reference_draws='3',
                 n_simulations='2', random_seed='2', include_even='on',
                 manual_schedules='Early comparison: 0, 1, 2',
                 timing_0_kind='latest_before', timing_0_label='By three',
                 timing_0_lower_time='0', timing_0_upper_time='3',
                 timing_1_kind='latest_before', timing_1_label='By four',
                 timing_1_lower_time='0', timing_1_upper_time='4'),
        )

    def test_only_suggested_schedule_with_all_selected_anchors_is_certified(self):
        rows = present_planning_result(self.result)['rows']
        suggested = next(row for row in rows if row['kind'] == 'Suggested')
        self.assertEqual(suggested['timing']['status'], 'not_past_deadline')
        self.assertIsNone(suggested['timing_flag'])
        for row in rows:
            if row['kind'] != 'Suggested':
                self.assertEqual(row['timing_flag']['label'], 'Latestness unverified')

    def test_missing_nonfinite_or_unselected_anchor_cannot_certify_latestness(self):
        for change in ('missing', 'nonfinite', 'none', 'unselected', 'wrong_label'):
            with self.subTest(change=change):
                result = deepcopy(self.result)
                anchors = result.timing_details['outcomes'][0]['anchors']
                if change == 'missing':
                    anchors.pop()
                elif change == 'nonfinite':
                    anchors[0]['selected_time'] = np.nan
                elif change == 'none':
                    anchors[0]['selected_time'] = None
                elif change == 'unselected':
                    anchors[0]['status'] = 'unavailable'
                else:
                    anchors[0]['label'] = 'Another requirement'
                row = next(row for row in present_planning_result(result)['rows'] if row['kind'] == 'Suggested')
                self.assertEqual(row['timing_flag']['label'], 'Latestness unverified')

    def test_reordered_retained_times_keep_selector_and_detail_certification_aligned(self):
        result = deepcopy(self.result)
        for outcome in result.timing_details['outcomes']:
            outcome['times'] = np.asarray(outcome['times'])[::-1].copy()
        context = present_planning_result(result)
        suggested = next(row for row in context['rows'] if row['kind'] == 'Suggested')
        self.assertIsNone(suggested['timing_flag'])
        details = next(row for row in context['inspection']['schedule_data']['schedules']
                       if row['id'] == suggested['id'])
        latest = [row for row in details['timing_requirements'] if row['kind'] == 'latest_before']
        self.assertEqual(len(latest), 2)
        self.assertTrue(all(row['latest_selected'] for row in latest))


if __name__ == '__main__':
    unittest.main()
