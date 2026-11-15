"""Timing-form integration using real bounded feasibility and scheduling APIs."""
import unittest
from unittest.mock import patch

import numpy as np

from study_design_core import (
    DesignConstraints, ExistingObservations, ObservationModel, ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
)
from utilities.design_planning import PlanningInputError
from utilities.design_timing import (
    assess_timing, inspect_comparison, parse_timing_requirements,
)


def row(index=0, **changes):
    values = dict(kind='window', label=f'Window {index + 1}',
                  lower_mode='numeric', lower_time='0', upper_mode='numeric', upper_time='4',
                  minimum_count='1', count_scope='new')
    values.update(changes)
    return {f'timing_{index}_{key}': value for key, value in values.items()}


def reference():
    grid = np.linspace(0., 10., 21)
    curves = np.array([1. - .1 * grid, 1. - .05 * grid])
    return ReferenceTrajectoryEnsemble(
        grid, curves, [True, True], draw_weights=[.25, .75],
        central_curve=.25 * curves[0] + .75 * curves[1], interpolation='linear',
    )


def model():
    return ReferenceModelEnsemble(
        model_name='linear', parameter_names=('a', 'b'),
        parameter_draws=[[1., -.1], [1., -.05]], valid_draws=[True, True],
        draw_weights=[.25, .75], central_parameters=[1., -.0625],
        time_grid=np.linspace(0., 10., 21), supported_time_range=(0., 10.),
    )


class TimingFormScheduling(unittest.TestCase):
    def assess(self, form, *, completed=None, fixed=None, minimum=3, maximum=5,
               policy='future_only'):
        fixed = [0.] if fixed is None else fixed
        constraints = DesignConstraints(
            np.arange(11.), fixed_times=fixed, min_time=0., max_time=10.,
            min_number_of_times=minimum, max_number_of_times=maximum,
            replicates_per_time=2,
        )
        settings = dict(min_times=minimum, max_times=maximum,
                        baseline_time=0., first_time=0., last_time=10., future_policy=policy)
        result = assess_timing(reference(), model(), ObservationModel('constant_sd', .1),
                               constraints, settings, form, existing_observations=completed)
        return result, constraints

    def test_no_active_rows_uses_the_ordinary_schedule_path(self):
        result, _ = self.assess({'timing_0_upper_time': 'inactive junk'})
        self.assertIsNone(result)

    def test_windows_produce_complete_sizes_and_keep_requirement_bounds(self):
        (optimization, schedules, details), constraints = self.assess(row(upper_time='3'))
        self.assertEqual(len(schedules), 3)
        self.assertEqual(details['requirements'][0]['upper'], 3.)
        for outcome in details['outcomes']:
            self.assertEqual(outcome['status'], 'feasible')
            step = optimization.design_for_n(outcome['number_of_times'], refined=True)
            self.assertEqual(len(step.times), outcome['number_of_times'])
            assessment = inspect_comparison(step.times, details, constraints)
            self.assertEqual(assessment.status, 'satisfied')
            self.assertTrue(any(0 < t <= 3 for t in step.times))

    def test_informative_window_is_not_the_same_as_reserved_latest_point(self):
        form = row(kind='latest_before', upper_time='4')
        (optimization, schedules, details), constraints = self.assess(form, minimum=3, maximum=3)
        self.assertIn(4., optimization.design_for_n(3, refined=True).times)
        anchor = details['outcomes'][0]['anchors'][0]
        self.assertEqual(anchor['selected_time'], 4.)
        self.assertEqual(anchor['deadline_slack'], 0.)
        self.assertEqual(details['outcomes'][0]['evaluation'].status, 'satisfied')
        comparison = inspect_comparison([0., 2., 3.], details, constraints)
        self.assertEqual(comparison.status, 'not_past_deadline')
        self.assertIn('not verified', comparison.requirements[0].message)

    def test_mixed_latest_points_remain_distinct_and_sizes_can_be_infeasible(self):
        form = {**row(0, kind='latest_before', lower_time='4', upper_time='6', label='Before six'),
                **row(1, kind='latest_before', lower_time='6', upper_time='8', label='Before eight'),
                **row(2, upper_time='4', label='Early coverage')}
        (optimization, schedules, details), _ = self.assess(form, minimum=3, maximum=4)
        self.assertEqual(details['outcomes'][0]['status'], 'infeasible')
        self.assertNotIn('Suggested · 3 times', schedules)
        self.assertEqual(details['outcomes'][1]['status'], 'feasible')
        self.assertEqual({p['selected_time'] for p in details['outcomes'][1]['anchors']}, {6., 8.})
        self.assertEqual(optimization.design_for_n(4, refined=True).number_of_times, 4)

    def test_weighted_milestone_quantile_uses_the_shared_complete_trajectories(self):
        form = row(upper_mode='milestone', upper_fraction='.5', upper_quantile='.25',
                   upper_time='inactive junk')
        (_, _, details), _ = self.assess(form)
        request = details['requirements'][0]
        self.assertEqual(request['upper'], 5.)
        self.assertIn('25th percentile', request['upper_description'])
        self.assertIn('50% degradation', request['upper_description'])

    def test_unreached_milestone_is_unavailable_not_declared_infeasible(self):
        form = row(upper_mode='milestone', upper_fraction='.9', upper_quantile='.75')
        (optimization, schedules, details), _ = self.assess(form)
        self.assertEqual(details['requirements'][0]['status'], 'beyond_horizon')
        self.assertTrue(all(outcome['status'] == 'unavailable' for outcome in details['outcomes']))
        self.assertFalse(schedules)
        self.assertFalse(optimization.refined_steps)

    def test_state_budget_is_reported_as_search_limit(self):
        with patch('utilities.design_timing.MAX_TIMING_SEARCH_STATES', 1):
            (optimization, schedules, details), _ = self.assess(row(), minimum=4, maximum=4)
        self.assertEqual(details['outcomes'][0]['status'], 'search_limit')
        self.assertFalse(schedules)
        self.assertFalse(optimization.refined_steps)

    def test_completed_window_scope_counts_existing_nonbaseline_time_only(self):
        completed = ExistingObservations([0., 2.], [1., .8])
        (_, _, details), _ = self.assess(
            row(upper_time='2', count_scope='all'), completed=completed,
            fixed=[0., 2.], minimum=2, maximum=3,
        )
        requirement = details['requirements'][0]
        self.assertEqual(requirement['fixed_times'], (2.,))
        self.assertEqual(requirement['status'], 'already_satisfied')
        (_, schedules, details), _ = self.assess(
            row(upper_time='2', count_scope='new'), completed=completed,
            fixed=[0., 2.], minimum=2, maximum=3,
        )
        self.assertEqual(details['requirements'][0]['status'], 'deadline_elapsed')
        self.assertFalse(schedules)

    def test_gap_filling_and_timing_are_not_silently_reinterpreted(self):
        with self.assertRaises(PlanningInputError) as caught:
            self.assess(row(), completed=ExistingObservations([0., 2.], [1., .8]),
                        fixed=[0., 2.], policy='gap_fill')
        self.assertEqual(caught.exception.field_id, 'future_policy')

    def test_malformed_active_rows_link_errors_to_the_exact_control(self):
        settings = {'baseline_time': 0.}
        cases = [
            (dict(kind='unknown'), 'kind'),
            (dict(upper_time='nan'), 'upper_time'),
            (dict(upper_mode='unknown'), 'upper_mode'),
            (dict(upper_mode='milestone', upper_fraction='1'), 'upper_fraction'),
            (dict(upper_mode='milestone', upper_quantile='0'), 'upper_quantile'),
            (dict(minimum_count='1.5'), 'minimum_count'),
            (dict(count_scope='individuals'), 'count_scope'),
        ]
        for changes, field in cases:
            with self.subTest(field=field), self.assertRaises(PlanningInputError) as caught:
                parse_timing_requirements(row(**changes), settings)
            self.assertEqual(caught.exception.field_id, 'timing_0_' + field)

    def test_active_labels_are_unique_and_number_of_slots_is_bounded(self):
        with self.assertRaises(PlanningInputError) as caught:
            parse_timing_requirements({**row(0, label='Same'), **row(1, label='Same')}, {})
        self.assertEqual(caught.exception.field_id, 'timing_1_label')
        with self.assertRaises(PlanningInputError) as caught:
            parse_timing_requirements({'timing_8_kind': 'window'}, {})
        self.assertEqual(caught.exception.field_id, 'timing-requirements')

    def test_only_selected_controls_are_validated(self):
        request = parse_timing_requirements(
            row(kind='latest_before', lower_time='', minimum_count='junk',
                count_scope='junk', upper_fraction='junk', upper_quantile='junk'),
            {'baseline_time': 0.},
        )[0]
        self.assertIsNone(request.lower_bound)
        self.assertEqual(request.deadline, 4.)


if __name__ == '__main__':
    unittest.main()
