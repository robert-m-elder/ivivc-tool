"""Report metadata describes existing calculations without recomputing them."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

from utilities import evaluation
from preprocessing import preprocessing_options

try:
    import engine
    FULL_ENGINE_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name not in {'plotly', 'uncertainties'}:
        raise
    # The pure metadata/preprocessing functions remain testable when the local
    # verification runtime lacks the optional app visualization dependencies.
    # This executes the actual function bodies; it does not replace any fits.
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'engine.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'preprocess_data', '_prediction_records_for_report'}]
    namespace = {'np': np, 'pd': pd, 'preprocessing_options': preprocessing_options}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'engine.py', 'exec'), namespace)
    engine = SimpleNamespace(**namespace)
    FULL_ENGINE_AVAILABLE = False


def proportional(x, a):
    return a * np.asarray(x)


class CrossValidationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(1., 7.)
        self.y = 2. * self.x

    def test_scores_and_legacy_return_are_unchanged(self):
        args = (self.x, self.y, proportional, LeaveOneOut(), ['rmse', 'r_squared'])
        old_shape = evaluation.cross_validation_curve_fit(*args, p0=[1.])
        result, diagnostics = evaluation.cross_validation_curve_fit(*args, p0=[1.], return_diagnostics=True)
        self.assertIsInstance(old_shape, dict)
        for metric in old_shape:
            np.testing.assert_equal(result[metric], old_shape[metric])
        self.assertEqual(diagnostics['total_splits'], 6)
        self.assertEqual(diagnostics['successful_fits'], 6)
        self.assertEqual(diagnostics['fitting_failure_count'], 0)
        self.assertEqual(diagnostics['finite_scores']['rmse'], 6)
        self.assertEqual(diagnostics['finite_scores']['r_squared'], 0)
        self.assertEqual(diagnostics['min_test_observations'], 1)
        self.assertEqual(diagnostics['max_training_observations'], 5)

    def test_none_and_fit_params_preserve_existing_shapes(self):
        (scores, parameters), diagnostics = evaluation.cross_validation_curve_fit(
            self.x, self.y, proportional, None, ['rmse'], return_fit_params=True, return_diagnostics=True)
        self.assertTrue(np.isnan(scores['rmse']).all())
        self.assertEqual(parameters, [])
        self.assertFalse(diagnostics['enabled'])
        self.assertEqual(diagnostics['total_splits'], 0)

    def test_failed_fit_is_counted_without_discarding_other_scores(self):
        original = evaluation.sp.optimize.curve_fit
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('Deliberate failed fold')
            return original(*args, **kwargs)

        with patch.object(evaluation.sp.optimize, 'curve_fit', side_effect=fail_once):
            (scores, params), diagnostics = evaluation.cross_validation_curve_fit(
                self.x, self.y, proportional, LeaveOneOut(), ['rmse'], p0=[1.],
                return_fit_params=True, return_diagnostics=True)
        self.assertEqual(diagnostics['successful_fits'], 5)
        self.assertEqual(diagnostics['fitting_failure_count'], 1)
        self.assertEqual(diagnostics['finite_scores']['rmse'], 5)
        self.assertTrue(np.isnan(scores['rmse'][1]))
        self.assertIsNone(params[1])


class PreprocessingMetadataTests(unittest.TestCase):
    def test_reference_values_and_fallback_match_applied_data(self):
        t = np.array([0., 1., 2., 3.])
        vitro = np.array([10., 8., np.nan, 2.])
        vivo = np.array([20., 16., 12., 4.])
        kwargs = dict(selected_normalizations=['initial_value', 'min_max'], selected_scalings=['log_x'],
                      selected_interpolation=['default'], fallback_invalid_log_scaling=True)
        old_shape = engine.preprocess_data(t, vitro, t, vivo, **kwargs)
        data, metadata = engine.preprocess_data(t, vitro, t, vivo, return_metadata=True, **kwargs)
        for before, after in zip(old_shape, data):
            np.testing.assert_equal(before, after)
        self.assertEqual(metadata['requested_scalings'], ['log_x'])
        self.assertEqual(metadata['applied_scalings'], [])
        self.assertEqual(len(metadata['warnings']), 1)
        refs = metadata['normalization_records']
        self.assertEqual([r['dataset'] for r in refs], ['in_vitro', 'in_vivo', 'in_vitro', 'in_vivo'])
        self.assertEqual(refs[0]['reference'], 10.)
        self.assertEqual(refs[1]['reference'], 20.)
        self.assertAlmostEqual(refs[2]['minimum'], .2)
        self.assertEqual(refs[2]['maximum'], 1.)

    def test_zero_reference_is_identified_as_not_applied(self):
        data, metadata = engine.preprocess_data(np.array([0., 1.]), np.array([0., .5]), None, None,
                                               selected_normalizations=['initial_value'], return_metadata=True)
        np.testing.assert_equal(data[1], [0., .5])
        self.assertEqual(metadata['normalization_records'][0]['dataset'], 'prediction')
        self.assertFalse(metadata['normalization_records'][0]['applied'])


class PredictionMetadataHelperTests(unittest.TestCase):
    def test_displayed_bounds_and_nonfinite_values_are_retained_without_sampling(self):
        trace = SimpleNamespace(name='In Vivo (Predicted)', x=np.array([3., 5., np.nan]), y=np.array([.5, .2, .1]),
                                error_x=SimpleNamespace(array=np.array([5., 8., np.nan]),
                                                        arrayminus=np.array([3., 5., np.nan])))
        rows = engine._prediction_records_for_report(SimpleNamespace(data=[trace]), [1., 2., 3.], [.5, .2, .1], 'time')
        self.assertEqual(rows[0]['predicted_in_vivo_time'], 3.)
        self.assertEqual(rows[0]['lower_95'], 0.)
        self.assertEqual(rows[0]['upper_95'], 8.)
        self.assertEqual(rows[1]['upper_95'], 13.)
        self.assertIsNone(rows[2]['predicted_in_vivo_time'])
        self.assertIsNone(rows[2]['upper_95'])

    def test_missing_intervals_are_not_replaced_by_zero(self):
        rows = engine._prediction_records_for_report(None, [0., 1.], [.9, .8], 'value',
                                                      predictions=[.7, np.nan], interval_applicable=False)
        self.assertEqual(rows[0]['predicted_in_vivo_value'], .7)
        self.assertIsNone(rows[0]['lower_95'])
        self.assertEqual(rows[0]['interval_status'], 'not generated for this method')
        self.assertIsNone(rows[1]['predicted_in_vivo_value'])


@unittest.skipUnless(FULL_ENGINE_AVAILABLE, 'Full app integration requires installed plotly and uncertainties')
class PredictionRecordTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([1., 2.])
        self.values = np.array([.5, .2])
        self.data = [self.times, self.values, self.times, self.values,
                     self.values, self.times, self.times, self.times, self.values, self.values]

    def test_time_bounds_use_displayed_center_and_one_existing_band_call(self):
        band = {'mean': np.array([101., 102.]), 'lower': np.array([98., 97.]), 'upper': np.array([106., 110.])}
        with patch.object(engine, 'generate_prediction_bands', return_value=band) as calculate:
            result = engine.create_prediction_plot_a1(self.data, self.times, self.values, np.array([3., 5.]),
                {'model_name': 'Example', 'model_function': proportional, 'params': {'a': 2.},
                 'pcov': np.eye(1), 'residuals': np.array([.1, -.1])})
        self.assertEqual(calculate.call_count, 1)
        self.assertEqual(result['prediction_records'][0]['predicted_in_vivo_time'], 3.)
        self.assertEqual(result['prediction_records'][0]['lower_95'], 0.)
        self.assertEqual(result['prediction_records'][0]['upper_95'], 8.)
        self.assertEqual(result['prediction_records'][1]['upper_95'], 13.)

    def test_value_bounds_use_existing_scaled_bars(self):
        band = {'mean': np.array([10., 10.]), 'lower': np.array([9., 8.]), 'upper': np.array([12., 13.])}
        with patch.object(engine, 'generate_ratio_prediction_bands', return_value=band) as calculate:
            result = engine.create_prediction_plot_a2(self.data, self.times, self.values, self.values,
                self.times, self.values, self.values,
                {'model_name': 'Example', 'model_function': proportional, 'params': [{'a': 2.}, {'a': 3.}],
                 'pcov': [np.eye(1), np.eye(1)], 'residuals': [np.array([.1, -.1]), np.array([.1, -.1])]})
        self.assertEqual(calculate.call_count, 1)
        self.assertAlmostEqual(result['prediction_records'][0]['lower_95'], 0.)
        self.assertAlmostEqual(result['prediction_records'][0]['upper_95'], 1.5)
        self.assertAlmostEqual(result['prediction_records'][1]['lower_95'], -.2)
        self.assertAlmostEqual(result['prediction_records'][1]['upper_95'], .8)

    def test_inverse_missing_bars_and_direct_mapping_are_explicit(self):
        result = engine.create_prediction_plot_a2(self.data, self.times, self.values, np.array([2., 5.]),
            self.times, self.values, self.values, {'model_name': 'Example'},
            t_pred_vivo_err={'lower': np.array([.2, np.nan]), 'upper': np.array([.4, np.nan])},
            prediction_mode='observed_response_inverse')
        self.assertAlmostEqual(result['prediction_records'][0]['lower_95'], 1.8)
        self.assertAlmostEqual(result['prediction_records'][0]['upper_95'], 2.4)
        self.assertIsNone(result['prediction_records'][1]['lower_95'])
        direct = engine.create_prediction_plot_a3(self.data, self.times, self.values, np.array([.3, np.nan]))
        self.assertEqual(direct['prediction_records'][0]['interval_status'], 'not generated for this method')
        self.assertIsNone(direct['prediction_records'][1]['predicted_in_vivo_value'])


if __name__ == '__main__':
    unittest.main()
