"""The uncertainty demonstration compares schedules under one assessment model."""
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from examples import study_design_uncertainty_demo as demo


class UncertaintyDemoTests(unittest.TestCase):
    def test_changed_times_and_recovery_use_a_common_unequal_error_assessment(self):
        captured = []
        original = demo.compare_designs_by_simulation
        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append((args, kwargs, result))
            return result
        with patch.object(demo, 'compare_designs_by_simulation', side_effect=record):
            report = demo.run_uncertainty_comparison(n_simulations=128)
        rows = {row['planning_assumption']: row for row in report['designs']}
        self.assertNotEqual(rows['constant_sd']['selected_times'], rows['unequal_sd']['selected_times'])
        for row in rows.values():
            self.assertEqual(len(set(row['selected_times'])), 3)
            self.assertEqual(row['selected_times'][0], 0.)
            self.assertTrue(set(row['selected_times'][1:]).issubset(report['candidate_times']))
            self.assertEqual(row['failure_rate'], 0.)
        self.assertLess(rows['unequal_sd']['assessment_information_criterion'],
                        rows['constant_sd']['assessment_information_criterion'])
        self.assertLess(rows['unequal_sd']['mean_irmse'], .6*rows['constant_sd']['mean_irmse'])
        args, kwargs, comparison = captured[0]
        self.assertEqual(args[2].kind, 'time_profile_sd')
        self.assertIs(comparison.simulation_plan, kwargs['simulation_plan'])
        first, second = comparison.results.values()
        np.testing.assert_array_equal(first.trajectory_draw_indices, second.trajectory_draw_indices)
        np.testing.assert_array_equal(first.observed_means[:, 0], second.observed_means[:, 0])
        self.assertEqual(report['assessment_observation_model']['profile_sd'], [.03, .03, .05, .2, .3])
        self.assertFalse(report['reference']['future_measurement_noise_included'])
        self.assertTrue(report['common_random_numbers'])
        self.assertEqual(report['refit_objective'], 'weighted_least_squares')
        json.dumps(report, allow_nan=False)

    def test_cli_prints_json_and_honors_replicate_and_seed_options(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = demo.main(['--simulations', '4', '--seed', '41', '--replicates-per-time', '2'])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report['n_simulations'], 4)
        self.assertEqual(report['random_seed'], 41)
        self.assertEqual(report['replicates_per_time'], 2)
        self.assertEqual(report['assessment_observation_model']['replicates_per_time'], 2)

    def test_invalid_counts_are_rejected(self):
        for options in ({'n_simulations': 0}, {'n_simulations': 1.5},
                        {'replicates_per_time': True}, {'random_seed': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                demo.run_uncertainty_comparison(**options)


if __name__ == '__main__':
    unittest.main()
