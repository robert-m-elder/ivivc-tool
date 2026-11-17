"""Recovered calibration profiles use retained fits and one shared study."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import TimeCourse
from study_design_core import ReferenceTrajectoryEnsemble
from utilities.design_calibration_recovery import build_calibration_recovery


def assessment(a=100., b=-2., *, success=True):
    profile = lambda a, b: SimpleNamespace(model_name='linear', parameters={'a': a, 'b': b})
    model = SimpleNamespace(in_vitro_fit=profile(50., -1.), in_vivo_fit=profile(a, b))
    return SimpleNamespace(
        fit_available=success, analysis=SimpleNamespace(model_results={'approach2:linear': model}),
        errors={} if success else {'fit': SimpleNamespace(message='Deliberate failed first fit')},
    )


def retained_result():
    vitro = TimeCourse([0., 10.], [50., 40.])
    vivo = TimeCourse([0., 4.], [100., 91.])
    assembled = TimeCourse([0., 4., 4., 10.], [100., 91., 92., 80.])
    simulated = SimpleNamespace(
        assessments=[assessment(), assessment(120., -3.), assessment(999., 999., success=False)],
        in_vitro_data=[vitro] * 3, in_vivo_data=[assembled] * 3,
        metadata={'in_vitro_draw_indices': np.array([0, 1, 0]),
                  'in_vivo_draw_indices': np.array([1, 0, 1])},
    )
    return SimpleNamespace(
        settings={'approach_id': 'approach2', 'model_name': 'linear', 'varied_environment': 'in_vivo'},
        references={
            'in_vitro': ReferenceTrajectoryEnsemble([0., 10.], [[50., 40.], [55., 44.]], [True, True]),
            'in_vivo': ReferenceTrajectoryEnsemble([0., 10.], [[1000., 900.], [2000., 1800.]], [True, True]),
        },
        analyzed_existing={'in_vitro': vitro, 'in_vivo': vivo},
        source_details={'in_vitro': {'mode': 'fixed'}, 'in_vivo': {'mode': 'additions'}},
        comparison=SimpleNamespace(results={'First': simulated, 'Second': deepcopy(simulated)}),
    )


class CalibrationRecovery(unittest.TestCase):
    def test_retained_profiles_supply_quantiles_and_failed_values_do_not_enter(self):
        result = retained_result()
        data = build_calibration_recovery(result)
        self.assertEqual(data['default_environment'], 'in_vivo')
        self.assertEqual([row['id'] for row in data['candidates']],
                         ['calibration-design-0', 'calibration-design-1'])
        first = data['candidates'][0]
        self.assertEqual((first['n_fit_successful'], first['n_simulations']), (2, 3))
        vivo = first['sides']['in_vivo']
        self.assertEqual(vivo['available_counts'], [2, 2])
        np.testing.assert_allclose(vivo['median'], [110., 85.])
        np.testing.assert_allclose(vivo['lower'], np.quantile([[100., 80.], [120., 90.]], .025, axis=0))
        self.assertEqual(vivo['n_curves'], 2)
        json.dumps(data, allow_nan=False)

    def test_shared_example_uses_matching_draw_and_keeps_failed_first_study(self):
        result = retained_result()
        result.comparison.results['First'].assessments[0] = assessment(success=False)
        data = build_calibration_recovery(result)
        self.assertTrue(data['example']['available'])
        first = data['candidates'][0]['sides']['in_vivo']['example']
        self.assertFalse(first['fit_available'])
        self.assertEqual(first['curve'], [None, None])
        self.assertIn('failed first fit', first['message'])
        self.assertEqual(first['truth'], [2000., 1800.])
        second = data['candidates'][1]['sides']['in_vivo']['example']
        self.assertTrue(second['fit_available'])
        self.assertEqual(second['truth'], first['truth'])
        result.comparison.results['Second'].metadata['in_vivo_draw_indices'][0] = 0
        mismatched = build_calibration_recovery(result)
        self.assertFalse(mismatched['example']['available'])
        self.assertIsNone(mismatched['candidates'][0]['sides']['in_vivo']['example']['truth'])

    def test_fixed_environment_is_deduplicated_and_existing_rows_are_not_new_points(self):
        data = build_calibration_recovery(retained_result())
        fixed = data['environments']['in_vitro']
        self.assertTrue(fixed['fixed'])
        self.assertTrue(fixed['fixed_curve_shared'])
        self.assertEqual(fixed['fixed_curve'], [50., 40.])
        self.assertEqual(len(fixed['existing_points']), 2)
        self.assertEqual(data['candidates'][0]['sides']['in_vitro']['example']['new_points'], [])
        self.assertEqual(data['candidates'][0]['sides']['in_vivo']['example']['new_points'],
                         [{'time': 4., 'response': 92.}, {'time': 10., 'response': 80.}])

    def test_missing_reference_and_single_usable_fit_keep_counts_without_inventing_ranges(self):
        result = retained_result()
        result.references['in_vivo'] = None
        for simulated in result.comparison.results.values():
            simulated.assessments[1] = assessment(success=False)
        data = build_calibration_recovery(result)
        side = data['candidates'][0]['sides']['in_vivo']
        self.assertEqual(len(side['median']), 101)
        self.assertEqual(side['available_counts'], [1] * 101)
        self.assertEqual(side['lower'], [None] * 101)
        self.assertIsNone(side['example']['truth'])
        self.assertIsNone(data['environments']['in_vivo']['reference'])
        json.dumps(data, allow_nan=False)

    def test_all_unusable_curves_and_approach1_are_explicit_and_json_safe(self):
        result = retained_result()
        for simulated in result.comparison.results.values():
            simulated.assessments = [assessment(success=False)] * 3
        data = build_calibration_recovery(result)
        self.assertEqual(data['candidates'][0]['sides']['in_vivo']['median'], [None, None])
        self.assertIsNone(data['environments']['in_vitro']['fixed_curve'])
        json.dumps(data, allow_nan=False)
        result.settings['approach_id'] = 'approach1'
        data = build_calibration_recovery(result)
        self.assertFalse(data['supported'])
        self.assertIn('time-to-time mapping', data['message'])
        self.assertEqual(data['environments'], {})
        self.assertEqual([row['id'] for row in data['candidates']],
                         ['calibration-design-0', 'calibration-design-1'])
        self.assertFalse(data['example']['available'])


if __name__ == '__main__':
    unittest.main()
