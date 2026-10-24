"""Regression checks for the analysis record exported to reviewers."""

import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from utilities.report_content import REPORT_DETAIL_FIELDS, build_report_sections


def fixture(approach='approach2', method=None):
    key = approach + ':example' if approach != 'approach3' else 'approach3'
    x = np.array([0., 20., 40., 100.])
    y = np.array([1., .9, .7, .01])
    stats = pd.DataFrame([{'rmse': .015, 'CV rmse': .025, 'aicc': 2.5, 'CV aicc': 99.}])
    metadata = {'n_observations': 4, 'n_parameters': 2, 'residual_df': 2, 'ci_multiplier_95': 4.3}
    cv = {'total_splits': 4, 'successful_fits': 3, 'fitting_failure_count': 1,
          'finite_scores': {'rmse': 3, 'aicc': 2}}
    params = {'a': 1.23456789e-8, 'b': .8}
    uparams = {key: SimpleNamespace(n=value, s=abs(value) * .1) for key, value in params.items()}
    result = {'params': params, 'uparams': uparams, 'fit_metadata': metadata,
              'cv_diagnostics': cv, 'residual_x': x, 'residual_y': y, 'stats': stats,
              'fit_cv_ratios': {'rmse': .6}, 'cv_comparison': {'rmse': 'CV moderately worse'},
              'function': 'y = a + b x', 'plot': {'uncertainty_warnings': []},
              'parameter_diagnostics_html': '<div><span>Large parameter uncertainty</span><p>Review this fit.</p></div>'}
    if approach == 'approach2':
        result = {key: [value, copy.deepcopy(value)] if key not in ('function', 'plot') else value
                  for key, value in result.items()}
    elif approach == 'approach3':
        result = {'plot_v': {}, 'plot_t': {}}
    raw = {'time': [0., 20., 40., 100., None], 'response': [1., .9, .7, .01, None]}
    option = {'id': 'final-1', 'model_key': key, 'approach_id': approach,
              'model_name': 'example' if approach != 'approach3' else None,
              'model_dom_id': key.replace(':', '-'), 'label': approach + ' Example',
              'method_id': method, 'method_display': method,
              'fit_plot_sources': [{'source_id': 'fit-source', 'title': 'Selected fit'}],
              'prediction_plot_sources': []}
    context = {'app_info': {'version': '0.9.0', 'revision': 'abc123'}, 'analysis_timestamp': '2026-10-23T18:00:00Z',
               'fitting_source': {'filename': 'calibration.csv', 'worksheet': None, 'columns': ['t1', 'y1', 't2', 'y2']},
               'raw_fitting_data': {'in_vitro': raw, 'in_vivo': copy.deepcopy(raw)},
               'processed_data': (x, y, x, y, y, x, x, x, y, y), 'prediction_source': None,
               'raw_prediction_data': None, 'processed_prediction_data': None,
               'selected_models': [key], 'selected_metrics': ['rmse', 'aicc'],
               'selected_normalizations': [], 'selected_scalings': [], 'selected_interpolation': ['default'],
               'preprocessing_metadata': {'warnings': [], 'normalization_records': [], 'requested_scalings': []},
               'analysis_config': {'cv_scheme': 'leave_one_timepoint_out', 'grid_search_num_points': 10,
                                   'grid_search_param_min': -1000., 'grid_search_param_max': 1000., 'grid_search_random_state': 42},
               'results': {key: result}, 'prediction_results': {}, 'evidence_tables': {},
               'models_registry': {'example': {'display_name': 'Example', 'latex_equation': 'y = a + b x',
                                             'fit_model': lambda *a, **k: (_ for _ in ()).throw(AssertionError('No refitting'))}}}
    return context, option


def table(sections, title):
    return next(item for item in sections if item['type'] == 'table' and item['title'] == title)


def text(sections):
    return '\n'.join(item.get('text', '') for item in sections)


class ReportContentTests(unittest.TestCase):
    def test_parameter_constraints_preserve_small_bounds_and_unbounded_values(self):
        context, option = fixture()
        context['models_registry']['example']['fit_kwargs'] = {'bounds': ([0., 1e-12], np.inf)}
        sections = build_report_sections(context, option)
        self.assertEqual(table(sections, 'Parameter fitting constraints')['rows'][1:],
                         [['a', '0', 'Unbounded'], ['b', '1e-12', 'Unbounded']])

    def test_preserves_source_and_fitting_rows_and_precision(self):
        context, option = fixture()
        raw = context['raw_fitting_data']['in_vitro']
        raw['time'], raw['response'] = list(range(18)), [i / 20 for i in range(18)]
        sections = build_report_sections(context, option)
        self.assertEqual(len(table(sections, 'In vitro calibration source observations')['rows']), 19)
        self.assertEqual(table(sections, 'In vitro data actually fitted')['rows'][1], ['1', '0', '1'])
        self.assertEqual(table(sections, 'In vitro parameter estimates')['rows'][1][1], '1.23457e-08')
        self.assertEqual(table(sections, 'In vitro cross-validation outcomes')['rows'][2], ['Successful fits and finite test predictions', '3'])
        self.assertEqual(table(sections, 'In vitro performance metrics')['rows'][2][2], 'Not applicable')
        self.assertEqual(table(sections, 'In vitro performance metrics')['rows'][1][-2:], ['0.6', 'CV moderately worse'])
        self.assertEqual(table(sections, 'In vitro performance metrics')['rows'][2][-2:], ['Not meaningful', 'Not meaningful'])
        headings = [item['text'] for item in sections if item['type'] == 'heading']
        self.assertLess(headings.index('Candidate model context'), headings.index('Fit and diagnostic results'))
        self.assertIn('omits one row at a time', text(sections))
        self.assertIn('Large parameter uncertainty', text(sections))
        self.assertEqual(sum(bool(item.get('detail_key')) for item in sections), len(REPORT_DETAIL_FIELDS))
        json.dumps(sections, allow_nan=False)

    def test_relevant_interpolation_and_method_text_only(self):
        context, option = fixture(method='value_ratio')
        context['prediction_source'] = {'filename': 'prediction.csv', 'columns': ['t', 'y']}
        context['raw_prediction_data'] = {'time': [60], 'response': [.3]}
        context['prediction_results'] = {'approach2': {'example': {'prediction_tables': {'value_ratio': [
            {'in_vitro_time': 60, 'in_vitro_value': .3, 'predicted_in_vivo_value': .25,
             'lower_95': .1, 'upper_95': .4, 'status': 'predicted', 'interval_status': 'available'}]}}}}
        sections = build_report_sections(context, option)
        self.assertFalse(any('interpolation-' in item.get('source_id', '') for item in sections))
        self.assertNotIn('surrogate', text(sections))
        self.assertNotIn('Tau is the integral', text(sections))
        self.assertEqual(table(sections, 'Predictions and displayed uncertainty bounds')['rows'][1][3:6], ['0.25', '0.1', '0.4'])

    def test_ratio_parameter_intervals_and_diagnostics_are_not_recentered(self):
        context, option = fixture(method='value_ratio')
        context['prediction_source'] = {'filename': 'prediction.csv'}
        context['prediction_results'] = {'approach2': {'example': {
            'plot_value_ratio_note': 'One interval is unavailable.',
            'prediction_tables': {'value_ratio': [
                {'in_vitro_time': 1., 'in_vitro_value': .2, 'predicted_in_vivo_value': .8,
                 'lower_95': .9, 'upper_95': 1.1, 'status': 'predicted', 'interval_status': 'available',
                 'interval_valid_samples': 1000, 'interval_total_samples': 1000,
                 'interval_requested_samples': 1000, 'denominator_lower_95': .1,
                 'denominator_upper_95': .3, 'interval_random_seed': 12345},
                {'in_vitro_time': 2., 'in_vitro_value': .1, 'predicted_in_vivo_value': .4,
                 'lower_95': None, 'upper_95': None, 'status': 'predicted',
                 'interval_status': 'unavailable (denominator interval includes zero)',
                 'interval_valid_samples': 1000, 'interval_total_samples': 1000,
                 'interval_requested_samples': 1000, 'denominator_lower_95': -.1,
                 'denominator_upper_95': .1, 'interval_random_seed': 12345},
            ]},
        }}}
        sections = build_report_sections(context, option)
        rows = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
        self.assertEqual(rows[1][3:6], ['0.8', '0.9', '1.1'])
        self.assertEqual(rows[2][3:6], ['0.4', 'N/A', 'N/A'])
        self.assertIn('denominator interval includes zero', rows[2][-1])
        diagnostic = table(sections, 'Response-ratio uncertainty diagnostics')['rows'][1]
        self.assertEqual(diagnostic, ['1', '1000', '1000', '1000', '0.1', '0.3', '12345'])
        self.assertIn('not prediction intervals for future measurements', text(sections))
        self.assertIn('One interval is unavailable.', text(sections))
        self.assertNotIn('residual-based ratio uncertainty', text(sections))
        json.dumps(sections, allow_nan=False)

    def test_inverse_status_and_suppressed_bounds_are_retained(self):
        context, option = fixture(method='observed_response_inverse')
        context['prediction_source'] = {'filename': 'prediction.csv'}
        context['prediction_results'] = {'approach2': {'example': {
            'plot_inverse_response_note': 'Some intervals were suppressed.',
            'prediction_tables': {'observed_response_inverse': [
                {'in_vitro_time': 0, 'in_vitro_value': 1.02, 'predicted_in_vivo_time': 0.,
                 'lower_95': np.nan, 'upper_95': np.nan, 'status': 'mapped to calibration endpoint',
                 'interval_status': 'percentile interval excluded the point', 'calculated_lower_95': 1.,
                 'calculated_upper_95': 4., 'interval_valid_samples': 500, 'total_samples': 1000}]}}}}
        sections = build_report_sections(context, option)
        row = table(sections, 'Predictions and displayed uncertainty bounds')['rows'][1]
        self.assertEqual(row[4:7], ['N/A', 'N/A', 'mapped to calibration endpoint'])
        self.assertEqual(table(sections, 'Calculated inverse interval diagnostics')['rows'][1], ['1', '1', '4', '500', '1000'])
        self.assertIn('Some intervals were suppressed.', text(sections))
        json.dumps(sections, allow_nan=False)

    def test_supplied_failed_prediction_is_not_described_as_absent(self):
        for predictions in ({'error': 'file could not be read'}, {'approach2': {'example': {'error': 'prediction calculation failed'}}}):
            context, option = fixture()
            context['prediction_source'] = {'filename': 'bad.xlsx', 'worksheet': 'Input'}
            context['prediction_results'] = predictions
            report = text(build_report_sections(context, option))
            self.assertIn('prediction processing failed', report)
            self.assertNotIn('No prediction dataset was supplied', report)

    def test_invalid_prediction_cells_and_normalization_references_survive(self):
        context, option = fixture()
        context['prediction_source'] = {'filename': 'bad.csv'}
        context['raw_prediction_data'] = {'time': [1., 2.], 'response': ['not-a-number', 3.]}
        context['prediction_results'] = {'error': 'Prediction values must be numeric'}
        context['prediction_preprocessing_metadata'] = {'normalization_records': [
            {'dataset': 'prediction', 'operation': 'initial_value', 'reference': 12., 'minimum': None,
             'maximum': None, 'applied': True, 'note': ''}]}
        sections = build_report_sections(context, option)
        self.assertEqual(table(sections, 'Prediction source observations')['rows'][1], ['1', '1', 'not-a-number'])
        self.assertEqual(table(sections, 'Normalization reference values in application order')['rows'][1][:3],
                         ['prediction', 'initial_value', '12'])
        self.assertEqual(table(sections, 'Source data coverage')['rows'][-1][1:3], ['2', '1'])

    def test_approach3_retains_provenance_without_parametric_methods(self):
        context, option = fixture('approach3')
        sections = build_report_sections(context, option)
        identification = table(sections, 'Analysis identification')['rows']
        self.assertIn(['App version', '0.9.0'], identification)
        self.assertIn(['Source revision', 'abc123'], identification)
        self.assertFalse(any(item.get('title') == 'Settings used' for item in sections))
        self.assertEqual(len(table(sections, 'Time-mapping lookup data')['rows']), 5)
        self.assertIn('No parametric model', text(sections))

    def test_approach1_describes_derived_pairs(self):
        context, option = fixture('approach1')
        sections = build_report_sections(context, option)
        self.assertIn('derived paired-time observations', text(sections))
        self.assertEqual(table(sections, 'Derived paired-time data actually fitted')['rows'][0],
                         ['Fitting row', 'Analyzed in vitro time', 'Analyzed in vivo time'])
        source_ids = [item.get('source_id') for item in sections]
        self.assertIn('plot-processing-interpolation-value', source_ids)
        self.assertNotIn('plot-processing-interpolation-time', source_ids)


if __name__ == '__main__':
    unittest.main()
