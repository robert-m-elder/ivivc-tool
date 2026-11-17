"""Display adapter checks using retained, already summarized study outcomes."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from utilities.design_performance_inspection import build_performance_inspection


DEFINITION = dict(kind='relative_decrease', value=50., direction='decreasing', label='T50')


def fixture():
    simulated = SimpleNamespace(fit_success=np.array([True, True, True, False]),
                                integrated_rmse=np.array([.1, .3, np.nan, .9]))
    summary = SimpleNamespace(mean_irmse=.2, p90_irmse=.28,
                              mean_curve_interval_width=np.inf)
    result = SimpleNamespace(settings={'recovery_endpoints': [DEFINITION]},
        completed_observations=None,
        comparison=SimpleNamespace(results={'Suggested 3': simulated},
                                   summaries={'Suggested 3': summary}))
    rows = [dict(id='schedule-4', name='Suggested 3', kind='Suggested',
                 number_of_times=3, number_of_new_times=2,
                 n_simulations=4, n_successful=3, interval_count=2)]
    endpoint = {
        'definition': DEFINITION, 'n_simulations': 4,
        'recovered': dict(n_valid=3, n_reached=2, n_beyond=1, n_invalid=0, n_failed=1,
            quantiles=[dict(value=2., status='resolved'), dict(value=4., status='resolved'),
                       dict(value=None, status='beyond_horizon')]),
        'paired': dict(n_finite=2, median_absolute_error=.75, p90_absolute_error=1.2),
    }
    recovery = {'schedules': [dict(id='schedule-4', name='Suggested 3', endpoints=[endpoint])]}
    return result, rows, recovery


class PerformanceInspection(unittest.TestCase):
    def test_curve_and_endpoint_measures_reuse_retained_values_and_ids(self):
        result, rows, recovery = fixture()
        original = deepcopy(recovery)
        data = build_performance_inspection(result, rows, recovery, x_label='Total times')
        self.assertEqual([measure['id'] for measure in data['measures']], ['irmse', 'endpoint-0'])
        schedule = data['schedules'][0]
        self.assertEqual((schedule['id'], schedule['x']), ('schedule-4', 3))
        self.assertEqual(schedule['measures']['irmse']['error'],
                         dict(primary=.2, secondary=.28, n_pairs=2))
        self.assertIsNone(schedule['curve_interval_width'])
        endpoint = schedule['measures']['endpoint-0']
        self.assertEqual(endpoint['error']['primary'], .75)
        self.assertEqual(endpoint['error']['secondary'], 1.2)
        self.assertEqual(endpoint['error']['n_pairs'], 2)
        self.assertEqual((endpoint['error']['n_reached'], endpoint['error']['n_failed']), (2, 1))
        self.assertEqual(endpoint['time']['primary'], 4.)
        self.assertEqual(endpoint['time']['lower'], 2.)
        self.assertIsNone(endpoint['time']['upper'])
        self.assertEqual(endpoint['time']['upper_status'], 'beyond_horizon')
        self.assertEqual(recovery, original)
        json.dumps(data, allow_nan=False)

    def test_beyond_horizon_median_stays_unresolved_and_adaptive_size_uses_new_times(self):
        result, rows, recovery = fixture()
        result.completed_observations = SimpleNamespace()
        quantiles = recovery['schedules'][0]['endpoints'][0]['recovered']['quantiles']
        quantiles[1] = dict(value=None, status='beyond_horizon')
        data = build_performance_inspection(result, rows, recovery, x_label='New times')
        schedule = data['schedules'][0]
        self.assertEqual(schedule['x'], 2)
        time = schedule['measures']['endpoint-0']['time']
        self.assertIsNone(time['primary'])
        self.assertEqual(time['primary_status'], 'beyond_horizon')
        self.assertEqual(time['lower'], 2.)
        json.dumps(data, allow_nan=False)

    def test_one_valid_endpoint_gives_a_median_without_a_claimed_range(self):
        result, rows, recovery = fixture()
        recovered = recovery['schedules'][0]['endpoints'][0]['recovered']
        recovered.update(n_valid=1, n_reached=1, n_beyond=0, n_invalid=2)
        recovered['quantiles'] = [dict(value=4., status='resolved')] * 3
        time = build_performance_inspection(result, rows, recovery, x_label='Times')[
            'schedules'][0]['measures']['endpoint-0']['time']
        self.assertEqual(time['primary'], 4.)
        self.assertIsNone(time['lower'])
        self.assertIsNone(time['upper'])
        self.assertEqual((time['lower_status'], time['upper_status']),
                         ('unavailable', 'unavailable'))

    def test_empty_selection_and_absent_comparison_need_no_invented_values(self):
        result, rows, recovery = fixture()
        result.settings['recovery_endpoints'] = []
        data = build_performance_inspection(result, rows, {'schedules': []}, x_label='Times')
        self.assertEqual([measure['id'] for measure in data['measures']], ['irmse'])
        self.assertEqual(list(data['schedules'][0]['measures']), ['irmse'])
        result.comparison = None
        self.assertEqual(build_performance_inspection(result, [], recovery,
                                                     x_label='Times')['schedules'], [])


if __name__ == '__main__':
    unittest.main()
