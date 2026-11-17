"""Retained IVIVC/target output axes, ranges, and failure denominators."""
from types import SimpleNamespace
import json
import unittest

import numpy as np

from ivivc_core import PredictedTrajectory
from study_design_core import CalibrationEvaluation
from utilities.design_calibration_outputs import build_calibration_outputs


def evaluation(estimate, *, quantity='value_ratio', coordinate=(1., 3.),
               tx=None, ty=None, available=True, lower=None, upper=None):
    estimate, coordinate = np.asarray(estimate, dtype=float), np.asarray(coordinate, dtype=float)
    tx = np.asarray(coordinate if tx is None else tx, dtype=float)
    ty = np.asarray(estimate if ty is None else ty, dtype=float)
    trajectory = PredictedTrajectory('approach2', 'value_ratio', coordinate, ty,
                                     tx, ty, available=available)
    return CalibrationEvaluation(quantity, coordinate, estimate,
        estimate - .1 if lower is None else lower,
        estimate + .1 if upper is None else upper, trajectory)


def result(evaluations, *, output='relationship', method='value_ratio', truth=None, approach='approach2'):
    assessments = [SimpleNamespace(fit_available=item is not None,
        relationship=item if output == 'relationship' else None,
        prediction=item if output == 'prediction' else None,
        errors={} if item is not None else {'fit': SimpleNamespace(message='Deliberate failure')})
        for item in evaluations]
    first = next(item for item in evaluations if item is not None)
    simulated = SimpleNamespace(assessments=assessments,
        truth={} if truth is None else {output: np.asarray(truth, dtype=float)}, metadata={})
    return SimpleNamespace(settings={'approach_id': approach, 'method': method,
        'relationship': output == 'relationship', 'assess_target': output == 'prediction'},
        comparison=SimpleNamespace(results={'First': simulated}, summaries={'First': {
            output: {'quantity': first.quantity, 'coordinate': first.coordinate}}}))


class CalibrationOutputs(unittest.TestCase):
    def test_pointwise_ranges_keep_availability_and_work_for_both_approaches(self):
        for approach in ('approach1', 'approach2'):
            with self.subTest(approach=approach):
                inputs = result([evaluation([1., 2.]), evaluation([3., 4.]),
                                 evaluation([999., 999.], available=False)], approach=approach)
                data = build_calibration_outputs(inputs)
                self.assertEqual(list(data), ['relationship'])
                candidate = data['relationship']['candidates'][0]
                self.assertEqual(candidate['id'], 'calibration-design-0')
                native = candidate['views']['native']
                self.assertEqual(native['x'], [1., 3.])
                self.assertEqual(native['y'], [2., 3.])
                self.assertEqual(native['counts'], [2, 2])
                np.testing.assert_allclose(native['lower'], np.quantile([[1., 2.], [3., 4.]], .025, axis=0))
                self.assertIsNone(native['truth'])
                json.dumps(data, allow_nan=False)

    def test_value_prediction_keeps_time_fixed_and_response_uncertainty_vertical(self):
        inputs = result([evaluation([4., 2.], quantity='in_vivo_response'),
                         evaluation([6., 3.], quantity='in_vivo_response')],
                        output='prediction', truth=[[10., 20.], [12., 22.]])
        output = build_calibration_outputs(inputs)['prediction']
        self.assertEqual(output['views']['response']['range_axis'], 'y')
        response = output['candidates'][0]['views']['response']
        self.assertEqual(response['x'], [1., 3.])
        self.assertEqual(response['y'], [5., 2.5])
        self.assertEqual(response['truth']['y'], [11., 21.])
        self.assertEqual(response['example']['truth']['y'], [10., 20.])
        self.assertEqual(response['example']['y'], [4., 2.])

    def test_time_prediction_keeps_target_response_fixed_and_time_ranges_horizontal(self):
        for method in ('time_mapping', 'tau_ratio', 'observed_response_inverse'):
            with self.subTest(method=method):
                coordinate = [.8, .4] if method == 'observed_response_inverse' else [1., 3.]
                inputs = result([
                    evaluation([2., 6.], quantity='in_vivo_time', coordinate=coordinate, tx=[2., 6.], ty=[.8, .4]),
                    evaluation([4., 8.], quantity='in_vivo_time', coordinate=coordinate, tx=[4., 8.], ty=[.8, .4])],
                    output='prediction', method=method, truth=[[20., 60.], [40., 80.]])
                output = build_calibration_outputs(inputs)['prediction']
                self.assertEqual(output['views']['response']['range_axis'], 'x')
                response = output['candidates'][0]['views']['response']
                self.assertEqual(response['x'], [3., 7.])
                self.assertEqual(response['y'], [.8, .4])
                self.assertEqual(response['truth']['x'], [30., 70.])
                self.assertEqual(response['truth']['y'], [.8, .4])
                self.assertEqual(response['example']['truth']['x'], [20., 60.])
                self.assertEqual(response['example']['x'], [2., 6.])

    def test_time_constant_ratio_is_a_dimensionless_scalar(self):
        inputs = result([evaluation([2.], quantity='time_constant_ratio', coordinate=[1.]),
                         evaluation([4.], quantity='time_constant_ratio', coordinate=[1.])], method='tau_ratio')
        output = build_calibration_outputs(inputs)['relationship']
        self.assertTrue(output['scalar'])
        self.assertEqual(output['views']['native']['x_label'], 'Design')
        self.assertIn('dimensionless', output['quantity_label'])
        self.assertEqual(output['candidates'][0]['views']['native']['y'], [3.])

    def test_failed_first_study_stays_failed_while_its_own_truth_is_retained(self):
        inputs = result([None, evaluation([4., 2.], quantity='in_vivo_response')],
                        output='prediction', truth=[[5., 6.], [7., 8.]])
        candidate = build_calibration_outputs(inputs)['prediction']['candidates'][0]
        self.assertEqual((candidate['n_simulations'], candidate['n_fit_successful']), (2, 1))
        native = candidate['views']['native']
        self.assertEqual(native['counts'], [1, 1])
        self.assertEqual(native['lower'], [None, None])
        self.assertEqual(native['example']['y'], [None, None])
        self.assertEqual(native['example']['point_counts'], [0, 0])
        self.assertEqual(native['example']['truth']['y'], [5., 6.])
        self.assertIn('no replacement', native['example']['message'])
        self.assertIn('Deliberate failure', native['example']['message'])

    def test_nonmonotonic_target_order_is_preserved_and_varying_both_axes_is_not_summarized(self):
        first = evaluation([2., 4., 6.], quantity='in_vivo_time', coordinate=[1., 2., 3.],
                           tx=[2., 4., 6.], ty=[.8, .2, .6])
        second = evaluation([3., 5., 7.], quantity='in_vivo_time', coordinate=[1., 2., 3.],
                            tx=[3., 5., 7.], ty=[.8, .2, .6])
        inputs = result([first, second], output='prediction', method='time_mapping')
        output = build_calibration_outputs(inputs)['prediction']
        self.assertFalse(output['views']['response']['connect_points'])
        self.assertEqual(output['candidates'][0]['views']['response']['y'], [.8, .2, .6])
        second.trajectory.in_vivo_value[-1] = .7
        data = build_calibration_outputs(inputs)
        output = data['prediction']
        self.assertFalse(output['views']['response']['available'])
        self.assertEqual(output['candidates'][0]['views']['response']['counts'], [0, 0, 0])
        self.assertEqual(output['candidates'][0]['views']['native']['counts'], [2, 2, 2])
        self.assertEqual(output['candidates'][0]['views']['response']['example']['y'], [.8, .2, .6])
        json.dumps(data, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
