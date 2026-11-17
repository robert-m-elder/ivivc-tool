"""Endpoint semantics and availability using small retained curve fixtures."""
from types import SimpleNamespace
import json
import unittest

import numpy as np

from study_design_core import ReferenceTrajectoryEnsemble
from utilities.design_endpoints import (
    build_endpoint_inspection, endpoint_definitions, reference_endpoint_summary,
)


T50 = dict(kind='relative_decrease', value=50., direction='decreasing', label='T50')
ABS50 = dict(kind='absolute_threshold', value=50., direction='decreasing', label='Response ≤ 50')


def retained_result(truth, recovered, *, success=None, definitions=None):
    truth, recovered = np.asarray(truth, dtype=float), np.asarray(recovered, dtype=float)
    reference = ReferenceTrajectoryEnsemble([0., 10.], truth, [True] * len(truth),
                                            interpolation='linear')
    simulated = SimpleNamespace(
        evaluation_times=np.array([0., 10.]), true_curves=truth, fitted_curves=recovered,
        trajectory_draw_indices=np.arange(len(truth)), model_name='linear',
        parameter_estimates=np.column_stack((recovered[:, 0],
                                            (recovered[:, 1] - recovered[:, 0]) / 10.)),
        fit_success=np.ones(len(truth), dtype=bool) if success is None else np.asarray(success),
    )
    return SimpleNamespace(
        reference=reference, schedules={'A': [0., 10.]},
        settings={'baseline_time': 0., 'recovery_endpoints': definitions or [T50, ABS50]},
        comparison=SimpleNamespace(results={'A': simulated}),
    )


class RecoveryEndpoints(unittest.TestCase):
    def test_relative_uses_own_baseline_and_absolute_uses_same_level(self):
        result = retained_result([[100., 0.]], [[80., 0.]])
        data = build_endpoint_inspection(result)
        relative, absolute = data['schedules'][0]['endpoints']
        self.assertEqual(relative['truth']['quantiles'][1]['value'], 5.)
        self.assertEqual(relative['recovered']['quantiles'][1]['value'], 5.)
        self.assertEqual(relative['paired']['median_absolute_error'], 0.)
        self.assertEqual(absolute['recovered']['quantiles'][1]['value'], 3.75)
        self.assertEqual(absolute['paired']['median_signed_error'], -1.25)
        self.assertFalse(relative['recovered']['range_available'])
        json.dumps(data, allow_nan=False)

    def test_nonreach_invalid_baselines_and_failed_fits_remain_distinct(self):
        result = retained_result(
            [[100., 0.], [100., 80.], [100., 0.], [100., 80.], [100., 0.]],
            [[100., 80.], [100., 0.], [0., 0.], [100., 80.], [100., 0.]],
            success=[True, True, True, True, False], definitions=[T50])
        endpoint = build_endpoint_inspection(result)['schedules'][0]['endpoints'][0]
        recovered, paired = endpoint['recovered'], endpoint['paired']
        self.assertEqual((recovered['n_valid'], recovered['n_invalid'], recovered['n_failed']),
                         (3, 1, 1))
        self.assertEqual((recovered['n_reached'], recovered['n_beyond']), (1, 2))
        self.assertEqual(recovered['quantiles'][1]['status'], 'beyond_horizon')
        self.assertEqual((paired['true_only'], paired['recovered_only'], paired['neither']),
                         (1, 1, 1))
        self.assertEqual(paired['n_finite'], 0)
        self.assertIsNone(paired['median_absolute_error'])
        json.dumps(endpoint, allow_nan=False)

    def test_timing_errors_pair_each_study_with_its_own_truth(self):
        result = retained_result([[100., 0.], [100., 50.]],
                                 [[100., 50.], [100., 0.]], definitions=[T50])
        endpoint = build_endpoint_inspection(result)['schedules'][0]['endpoints'][0]
        self.assertEqual(endpoint['paired']['n_finite'], 2)
        self.assertEqual(endpoint['paired']['median_absolute_error'], 5.)
        self.assertEqual(endpoint['paired']['median_signed_error'], 0.)

    def test_baseline_between_grid_points_uses_exact_retained_fit_parameters(self):
        grid = np.array([0., 2., 4.])
        recovered = 100. * np.exp(-.5 * grid)
        reference = ReferenceTrajectoryEnsemble(grid, [recovered], [True], interpolation='linear')
        simulation = SimpleNamespace(
            evaluation_times=grid, true_curves=recovered[None, :],
            fitted_curves=recovered[None, :], trajectory_draw_indices=np.array([0]),
            model_name='exponential', parameter_estimates=np.array([[100., -.5]]),
            fit_success=np.array([True]),
        )
        result = SimpleNamespace(reference=reference, schedules={'A': grid},
            settings={'baseline_time': 1., 'recovery_endpoints': [T50]},
            comparison=SimpleNamespace(results={'A': simulation}))
        endpoint = build_endpoint_inspection(result)['schedules'][0]['endpoints'][0]
        target = .5 * 100. * np.exp(-.5)
        expected = 2. + 2. * (target - recovered[1]) / (recovered[2] - recovered[1])
        self.assertAlmostEqual(endpoint['recovered']['quantiles'][1]['value'], expected)
        self.assertNotEqual(endpoint['truth']['quantiles'][1]['value'], expected)
        self.assertLessEqual(expected, 4.)

    def test_absolute_direction_and_already_reached_threshold(self):
        increasing = dict(ABS50, direction='increasing')
        result = retained_result([[0., 100.], [60., 100.]], [[0., 80.], [60., 100.]],
                                 definitions=[increasing])
        endpoint = build_endpoint_inspection(result)['schedules'][0]['endpoints'][0]
        self.assertEqual(endpoint['truth']['n_reached'], 2)
        self.assertEqual(endpoint['truth']['quantiles'][0]['value'], 0.)
        self.assertEqual(endpoint['recovered']['quantiles'][2]['value'], 6.25)

    def test_weighted_reference_keeps_nonreach_mass_and_reports_invalid_reference(self):
        reference = ReferenceTrajectoryEnsemble(
            [0., 10.], [[100., 0.], [100., 80.]], [True, True],
            draw_weights=[.6, .4], interpolation='linear')
        summary = reference_endpoint_summary(reference, T50, 0.)
        self.assertEqual(summary['reached_weight'], .6)
        self.assertEqual(summary['quantiles'][1]['value'], 5.)
        self.assertEqual(summary['quantiles'][2]['status'], 'beyond_horizon')
        invalid = ReferenceTrajectoryEnsemble([0., 10.], [[100., 0.], [0., 0.]], [True, True])
        summary = reference_endpoint_summary(invalid, T50, 0.)
        self.assertIsNone(summary['reached_weight'])
        self.assertEqual(summary['invalid_weight'], .5)
        self.assertTrue(all(q['status'] == 'invalid_reference' for q in summary['quantiles']))
        json.dumps(summary, allow_nan=False)

    def test_absent_settings_default_but_empty_selection_stays_empty(self):
        self.assertEqual([d['label'] for d in endpoint_definitions({})], ['T50', 'T90'])
        self.assertEqual(endpoint_definitions({'recovery_endpoints': []}), [])
        result = retained_result([[100., 0.]], [[100., 0.]])
        result.settings['recovery_endpoints'] = []
        self.assertEqual(build_endpoint_inspection(result)['schedules'], [])


if __name__ == '__main__':
    unittest.main()
