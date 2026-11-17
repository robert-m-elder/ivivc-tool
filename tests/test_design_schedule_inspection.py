"""Point attribution from retained scheduling decisions and evaluations."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from study_design_core import (
    DesignConstraints, ExistingObservations, ObservationModel,
    ReferenceModelEnsemble, ReferenceTrajectoryEnsemble,
)
from utilities.design_schedule_inspection import build_schedule_inspection
from utilities.design_timing import assess_timing, inspect_comparison


def timing_row(index, **changes):
    values = dict(kind='window', label=f'Window {index}', lower_mode='numeric',
                  lower_time='0', upper_mode='numeric', upper_time='6',
                  minimum_count='1', count_scope='new')
    values.update(changes)
    return {f'timing_{index}_{key}': value for key, value in values.items()}


def reference():
    grid = np.arange(11.)
    curves = np.array([1. - .1 * grid, 1. - .05 * grid])
    return ReferenceTrajectoryEnsemble(grid, curves, [True, True], draw_weights=[.25, .75])


def model():
    return ReferenceModelEnsemble(
        model_name='linear', parameter_names=('a', 'b'),
        parameter_draws=[[1., -.1], [1., -.05]], valid_draws=[True, True],
        draw_weights=[.25, .75], central_parameters=[1., -.0625],
        time_grid=np.arange(11.), supported_time_range=(0., 10.),
    )


def timed_result(form, *, completed=None, fixed=(0.,), requested_fixed=None,
                 minimum=4, maximum=4):
    constraints = DesignConstraints(
        np.arange(11.), fixed_times=fixed, min_time=0., max_time=10.,
        min_number_of_times=minimum, max_number_of_times=maximum,
        replicates_per_time=3,
    )
    settings = dict(min_times=minimum, max_times=maximum, baseline_time=0.,
                    first_time=0., last_time=10., future_policy='future_only',
                    replicates_per_time=3,
                    requested_fixed_times=list(fixed if requested_fixed is None else requested_fixed))
    optimization, schedules, details = assess_timing(
        reference(), model(), ObservationModel('constant_sd', .1), constraints,
        settings, form, existing_observations=completed)
    for name, times in list(schedules.items()):
        schedules['Evenly spaced · comparison'] = times.copy()
        schedules['Manual · identical times'] = times.copy()
        break
    details['evaluations'] = {
        name: inspect_comparison(times, details, constraints) for name, times in schedules.items()
    }
    return SimpleNamespace(settings=settings, constraints=constraints,
                           completed_observations=completed,
                           timing_details=details, schedules=schedules)


class ScheduleInspection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = timed_result(
            {**timing_row(0, label='Future coverage'),
             **timing_row(1, label='All coverage', count_scope='all'),
             **timing_row(2, kind='latest_before', label='Latest by four', upper_time='4')},
            completed=ExistingObservations([0., 2., 2.], [1., .8, .79]),
            fixed=(0., 2., 4.), requested_fixed=[0., 4.],
        )
        cls.data = build_schedule_inspection(cls.result)

    def test_overlapping_fixed_anchor_and_window_roles_are_preserved(self):
        schedule = self.data['schedules'][0]
        point = next(point for point in schedule['points'] if point['time'] == 4.)
        self.assertEqual(point['roles'], ['new', 'user_fixed', 'latest_before'])
        self.assertEqual(point['latest_before'], ['Latest by four'])
        self.assertEqual(point['contributes_to'], ['Future coverage', 'All coverage'])
        self.assertEqual(point['new_replicates'], 3)
        self.assertTrue(any('Contributes to window count' in text for text in point['explanations']))
        self.assertFalse(any('mandated' in text for text in point['explanations']))

    def test_completed_rows_use_all_scope_only_and_do_not_acquire_future_counts(self):
        point = next(point for point in self.data['schedules'][0]['points'] if point['time'] == 2.)
        self.assertEqual(point['origin'], 'completed')
        self.assertEqual(point['contributes_to'], ['All coverage'])
        self.assertEqual(point['completed_rows'], 2)
        self.assertEqual(point['new_replicates'], 0)
        self.assertFalse(point['user_fixed'])
        self.assertEqual(self.data['planning_start'], 2.)

    def test_sampled_baseline_is_completed_and_excluded_from_window_counts(self):
        for schedule in self.data['schedules']:
            point = next(point for point in schedule['points'] if point['time'] == 0.)
            self.assertEqual(point['roles'], ['completed', 'baseline'])
            self.assertTrue(point['timing_baseline'])
            self.assertTrue(any('excluded from timing requirement counts' in text
                                for text in point['explanations']))
            self.assertEqual(point['contributes_to'], [])
            self.assertEqual(point['deadline_eligibility'], [])
            self.assertIn(schedule['id'], self.data['baseline_sampled_by'])

    def test_identical_comparison_times_do_not_inherit_certified_anchor_attribution(self):
        for schedule in self.data['schedules'][1:]:
            point = next(point for point in schedule['points'] if point['time'] == 4.)
            self.assertEqual(point['latest_before'], [])
            self.assertEqual(point['deadline_eligibility'], ['Latest by four'])
            self.assertNotIn('latest_before', point['roles'])
            self.assertTrue(any('latestness is not established' in text for text in point['explanations']))

    def test_requirement_rows_retain_actual_window_counts_and_scope(self):
        for name, schedule in zip(self.result.schedules, self.data['schedules']):
            evaluations = self.result.timing_details['evaluations'][name].requirements
            by_label = {row['label']: row for row in schedule['timing_requirements']}
            self.assertEqual(len(by_label), len(evaluations))
            for evaluation in evaluations:
                row = by_label[evaluation.label]
                self.assertEqual(row['kind'], evaluation.kind)
                self.assertEqual(row['status'], evaluation.status)
                self.assertEqual(row['message'], evaluation.message)
                self.assertEqual(row['achieved_count'], evaluation.achieved_count)
                self.assertEqual(row['minimum_count'], evaluation.minimum_count)
                self.assertEqual(row['qualifying_times'], list(evaluation.qualifying_times))
            self.assertEqual(by_label['Future coverage']['count_scope'], 'new')
            self.assertEqual(by_label['All coverage']['count_scope'], 'all')
            self.assertEqual(by_label['Future coverage']['upper'], 6.)
            self.assertEqual(by_label['Future coverage']['lower'], 0.)

    def test_deadline_rows_separate_certified_anchors_from_comparison_eligibility(self):
        suggested = next(row for row in self.data['schedules'][0]['timing_requirements']
                         if row['kind'] == 'latest_before')
        anchor = self.result.timing_details['outcomes'][0]['anchors'][0]
        self.assertTrue(suggested['latest_selected'])
        self.assertEqual(suggested['selected_time'], anchor['selected_time'])
        self.assertEqual(suggested['deadline'], anchor['deadline'])
        self.assertEqual(suggested['deadline_slack'], anchor['deadline_slack'])
        # The ordinary inspection remains eligibility-only even for a suggested
        # schedule; the independent retained anchor supplies selection evidence.
        self.assertEqual(suggested['status'], 'not_past_deadline')
        for schedule in self.data['schedules'][1:]:
            comparison = next(row for row in schedule['timing_requirements']
                              if row['kind'] == 'latest_before')
            self.assertEqual(comparison['status'], 'not_past_deadline')
            self.assertFalse(comparison['latest_selected'])
            self.assertIsNone(comparison['selected_time'])
            self.assertEqual(comparison['qualifying_times'], suggested['qualifying_times'])
            self.assertEqual(comparison['deadline'], 4.)

    def test_unavailable_rule_on_a_manual_schedule_has_no_invented_bound_or_anchor(self):
        result = timed_result(timing_row(
            0, kind='latest_before', upper_mode='milestone', upper_fraction='.9',
            upper_quantile='.75'))
        result.schedules = {'Manual · unresolved deadline': np.array([0., 2., 4., 8.])}
        result.timing_details['evaluations'] = {
            name: inspect_comparison(times, result.timing_details, result.constraints)
            for name, times in result.schedules.items()
        }
        data = build_schedule_inspection(result)
        row, = data['schedules'][0]['timing_requirements']
        self.assertEqual(row['status'], 'beyond_horizon')
        self.assertIsNone(row['deadline'])
        self.assertIsNone(row['selected_time'])
        self.assertFalse(row['latest_selected'])
        json.dumps(data, allow_nan=False)

    def test_unsampled_baseline_remains_a_guide_and_adds_no_point(self):
        result = deepcopy(self.result)
        result.settings['baseline_time'] = .5
        result.timing_details = {}
        data = build_schedule_inspection(result)
        self.assertEqual(data['baseline_time'], .5)
        self.assertEqual(data['baseline_sampled_by'], [])
        self.assertEqual(data['timing_baseline_times'], [])
        for row in data['schedules']:
            self.assertNotIn(.5, row['times'])
            self.assertTrue(all(not point['baseline'] for point in row['points']))

    def test_unresolved_requirements_and_unavailable_sizes_remain_explicit(self):
        result = timed_result(timing_row(
            0, upper_mode='milestone', upper_fraction='.9', upper_quantile='.75'))
        data = build_schedule_inspection(result)
        self.assertEqual(data['schedules'], [])
        self.assertEqual(data['requirements'][0]['status'], 'beyond_horizon')
        self.assertIsNone(data['requirements'][0]['upper'])
        self.assertTrue(data['outcomes'])
        self.assertTrue(all(row['status'] == 'unavailable' for row in data['outcomes']))
        json.dumps(data, allow_nan=False)

    def test_stable_order_ids_and_actual_times_are_json_safe_without_mutation(self):
        expected = [(name, values.copy()) for name, values in self.result.schedules.items()]
        data = build_schedule_inspection(self.result)
        self.assertEqual([row['id'] for row in data['schedules']], ['schedule-0', 'schedule-1', 'schedule-2'])
        self.assertEqual([row['name'] for row in data['schedules']], [name for name, _ in expected])
        for row, (name, times) in zip(data['schedules'], expected):
            np.testing.assert_array_equal(row['times'], times)
            np.testing.assert_array_equal(self.result.schedules[name], times)
        json.dumps(data, allow_nan=False)

    def test_gap_filling_keeps_latest_completed_separate_from_first_opportunity(self):
        result = deepcopy(self.result)
        result.settings.update(future_policy='gap_fill', first_time=0.)
        result.timing_details = {}
        result.schedules = {'Manual · gap fill': np.array([0., 1., 2., 4.])}
        data = build_schedule_inspection(result)
        self.assertIsNone(data['planning_start'])
        self.assertEqual(data['latest_completed_time'], 2.)
        self.assertEqual(data['first_opportunity_time'], 0.)
        self.assertEqual(data['evaluation_time_range'], [0., 10.])
        self.assertEqual(data['future_policy'], 'gap_fill')
        earlier = next(point for point in data['schedules'][0]['points'] if point['time'] == 1.)
        self.assertEqual(earlier['roles'], ['new'])
        self.assertEqual(earlier['new_replicates'], 3)
        json.dumps(data, allow_nan=False)

    def test_unfixed_milestone_baseline_can_contribute_to_a_timing_window(self):
        result = timed_result(timing_row(0, upper_time='2'), fixed=())
        result.settings['baseline_time'] = 1.
        # Re-resolve with the actual nonfixed milestone baseline, retaining the
        # shared scheduler's distinction between a guide and a fixed baseline.
        _, _, details = assess_timing(
            reference(), model(), ObservationModel('constant_sd', .1),
            result.constraints, result.settings, timing_row(0, upper_time='2'))
        result.schedules = {'Manual · baseline guide': np.array([1., 3., 5., 10.])}
        details['evaluations'] = {
            name: inspect_comparison(times, details, result.constraints)
            for name, times in result.schedules.items()
        }
        result.timing_details = details
        data = build_schedule_inspection(result)
        point = data['schedules'][0]['points'][0]
        self.assertTrue(point['baseline'])
        self.assertFalse(point['timing_baseline'])
        self.assertEqual(data['timing_baseline_times'], [])
        self.assertEqual(point['contributes_to'], ['Window 0'])
        self.assertFalse(any('excluded from timing' in text for text in point['explanations']))
        json.dumps(data, allow_nan=False)

    def test_correspondence_requires_actual_retained_suggested_schedule(self):
        result = deepcopy(self.result)
        name = next(iter(result.schedules))
        result.schedules[name] = np.array([0., 2., 4., 9.])
        # A similar size with different times must not acquire another outcome's anchors.
        data = build_schedule_inspection(result)
        self.assertTrue(all(not point['latest_before'] for point in data['schedules'][0]['points']))
        self.assertTrue(all(not row['latest_selected'] for row in data['schedules'][0]['timing_requirements']))


if __name__ == '__main__':
    unittest.main()
