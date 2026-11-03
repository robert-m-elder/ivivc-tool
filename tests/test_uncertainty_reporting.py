"""The report and export record keeps entered and analyzed uncertainty explicit."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from utilities.report_content import build_report_sections, build_uncertainty_sections


def context(method=None):
    time, value = np.array([0., 1., 3.]), np.array([1., .8, .4])
    metadata = {'n_observations': 3, 'n_parameters': 1, 'residual_df': 2,
                'ci_multiplier_95': 1.96, 'covariance_scaling': 'absolute'}
    result = {'params': [{'b': -.2}, {'b': -.1}],
              'uparams': [{'b': SimpleNamespace(n=-.2, s=.01)}, {'b': SimpleNamespace(n=-.1, s=.01)}],
              'fit_metadata': [metadata, dict(metadata, covariance_scaling='residual_scaled')],
              'residual_x': [time, 2 * time], 'residual_y': [value, value],
              'stats': [pd.DataFrame([{'rmse': .01, 'CV rmse': .02, 'aicc': 3.}])] * 2,
              'cv_diagnostics': [{}, {}], 'function': 'y = e^{-b x}', 'plot': {}}
    option = {'id': 'selected', 'model_key': 'approach2:example', 'approach_id': 'approach2',
              'model_name': 'example', 'method_id': method, 'label': 'Example',
              'fit_plot_sources': [], 'prediction_plot_sources': []}
    record = {'mode': 'column', 'response_kind': 'mean', 'uncertainty_kind': 'sem',
              'column': 'SEM', 'replicate_column': 'n', 'constant': None,
              'standard_uncertainty': [.5, .8, None, .4], 'replicate_counts': [4, 9, None, 16],
              'retained_rows': [0, 1, 3], 'source_row_count': 4,
              'analyzed': {'standard_uncertainty': [.05, .08, .04], 'replicate_counts': [4, 9, 16],
                           'response_kind': 'mean', 'uncertainty_kind': 'sem', 'response_units': 'dimensionless'}}
    data = {'results': {'approach2:example': result}, 'selected_models': ['approach2:example'],
            'selected_metrics': ['rmse', 'aicc'], 'selected_normalizations': ['initial_value'],
            'selected_scalings': [], 'selected_interpolation': [], 'analysis_config': {'cv_scheme': 'none'},
            'raw_fitting_data': {'in_vitro': {'time': [0, 1, None, 3], 'response': [10, 8, None, 4]},
                                 'in_vivo': {'time': [0, 2, 6], 'response': [1, .8, .4]}},
            'processed_data': [time, value, 2 * time, value],
            'fitting_source': {'filename': 'fit.csv', 'columns': ['t1', 'y1', 't2', 'y2', 'SEM', 'n'],
                               'column_roles': {'SEM': 'In vitro SEM', 'n': 'In vitro replicate count'}},
            'models_registry': {'example': {'display_name': 'Example'}},
            'observation_uncertainty': {'in_vitro': record}, 'uncertainty_independence_confirmed': True}
    if method:
        prediction_record = {'mode': 'column', 'response_kind': 'mean', 'uncertainty_kind': 'sem',
                             'column': 'SEM', 'replicate_column': 'n',
                             'standard_uncertainty': [.3, .5], 'replicate_counts': [4, 9],
                             'retained_rows': [0, 1], 'source_row_count': 2,
                             'analyzed': {'standard_uncertainty': [.3 / 7, .5 / 7], 'replicate_counts': [4, 9],
                                          'response_kind': 'mean', 'uncertainty_kind': 'sem', 'response_units': 'dimensionless'}}
        data['observation_uncertainty']['prediction'] = prediction_record
        data['prediction_source'] = {'filename': 'prediction.csv', 'columns': ['t', 'y', 'SEM', 'n'],
                                      'column_roles': {'SEM': 'Prediction SEM', 'n': 'Prediction replicate count'}}
        data['raw_prediction_data'] = {'time': [1, 2], 'response': [7, 5]}
        data['processed_prediction_data'] = [np.array([1., 2.]), np.array([1., 5 / 7])]
        rows = []
        for index, sigma in enumerate(prediction_record['analyzed']['standard_uncertainty']):
            rows.append({'in_vitro_time': index + 1, 'in_vitro_value': (1., 5 / 7)[index],
                         'predicted_in_vivo_value': (.8, .6)[index], 'predicted_in_vivo_time': (2., 4.)[index],
                         'lower_95': (.7, None)[index], 'upper_95': (.9, None)[index],
                         'status': 'predicted', 'interval_status': ('available', 'unavailable (failed draw)')[index],
                         'interval_kind': 'parameter_uncertainty' if method == 'tau_ratio' else 'parameter_and_input_uncertainty',
                         'input_standard_uncertainty': sigma, 'input_uncertainty_kind': 'sem', 'input_replicate_count': (4, 9)[index],
                         'response_lower_95': (.9, .6)[index], 'response_upper_95': (1.1, .8)[index],
                         'value_interval_kind': 'prediction_input_uncertainty', 'value_interval_status': 'available'})
        data['prediction_results'] = {'approach2': {'example': {'prediction_tables': {method: rows}}}}
        if method == 'tau_ratio':
            result['tau'] = [SimpleNamespace(n=5., s=.1), SimpleNamespace(n=10., s=.2)]
    return data, option


def table(sections, title):
    return next(item for item in sections if item.get('title') == title)


def text(sections):
    return '\n'.join(item.get('text', '') for item in sections)


class UncertaintyReportingTests(unittest.TestCase):
    def test_source_and_analyzed_errors_keep_row_alignment_counts_and_interpretation(self):
        data, option = context()
        sections = build_report_sections(data, option)
        supplied = table(sections, 'In vitro calibration uncertainty values')
        self.assertEqual(supplied['rows'][0], ['Source row', 'Entered SEM', 'Analyzed SEM', 'Replicate count (metadata)', 'Retained observation'])
        self.assertEqual(supplied['rows'][1], ['1', '0.5', '0.05', '4', 'Yes'])
        self.assertEqual(supplied['rows'][3], ['3', 'N/A', 'N/A', 'N/A', 'No'])
        self.assertEqual(supplied['rows'][4], ['4', '0.4', '0.04', '16', 'Yes'])
        self.assertIn('dimensionless', supplied['description'])
        raw = table(sections, 'In vitro calibration source observations')
        self.assertEqual(raw['rows'][1], ['1', '0', '10', '0.5', '4'])
        fitted = table(sections, 'In vitro data actually fitted')
        self.assertEqual(fitted['rows'][-1], ['3', '3', '0.4', '4', '0.04', '16'])
        self.assertIn('SEM is not divided by counts again', text(sections))
        self.assertIn('normalization constants are held fixed', text(sections))
        self.assertIn('normal multiplier', text(sections))
        self.assertIn('AICc retains the existing conventional correction', text(sections))
        json.dumps(sections, allow_nan=False)

    def test_constant_uncertainty_preserves_original_removed_rows(self):
        data, _ = context()
        record = data['observation_uncertainty']['in_vitro']
        record.update(mode='constant', constant=.5, column=None, standard_uncertainty=.5,
                      replicate_counts=None, replicate_column=None)
        record['analyzed']['standard_uncertainty'] = [.05] * 3
        sections = build_uncertainty_sections(data)
        values = table(sections, 'In vitro calibration uncertainty values')['rows']
        self.assertEqual(len(values), 5)
        self.assertEqual(values[3], ['3', '0.5', 'N/A', 'No'])
        self.assertEqual(table(sections, 'Supplied uncertainty interpretation')['rows'][1][1], 'Constant 0.5')

    def test_error_columns_have_their_recorded_source_roles(self):
        data, option = context('value_ratio')
        data['fitting_source']['columns'][4] = ' SEM '
        rows = table(build_report_sections(data, option), 'Analysis identification')['rows']
        values = ' '.join(cell for row in rows for cell in row)
        self.assertIn(' SEM  — In vitro SEM', values)
        self.assertIn('n — In vitro replicate count', values)
        self.assertIn('SEM — Prediction SEM', values)

    def test_prediction_methods_report_their_input_contribution_and_marginal_bounds(self):
        for method in ('value_ratio', 'tau_ratio', 'observed_response_inverse'):
            with self.subTest(method=method):
                data, option = context(method)
                sections = build_report_sections(data, option)
                prose = text(sections)
                self.assertNotIn('No response perturbations', prose)
                self.assertNotIn('prediction inputs held fixed', prose)
                self.assertIn('not a joint 95%', prose)
                values = table(sections, 'Prediction input and response uncertainty')['rows']
                self.assertEqual(values[1][1:4], [repr(.3 / 7), 'SEM', '4'])
                if method != 'value_ratio':
                    self.assertEqual(values[1][4:], ['0.9', '1.1', 'available'])
                primary = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
                self.assertEqual(primary[2][-1], 'unavailable (failed draw)')
                self.assertEqual(primary[2][4:6], ['N/A', 'N/A'])
                self.assertEqual(table(sections, 'Prediction data used for calculations')['rows'][1][3], repr(.3 / 7))
                json.dumps(sections, allow_nan=False)

    def test_confirmation_and_prediction_failures_are_retained(self):
        data, option = context('value_ratio')
        data['uncertainty_independence_confirmed'] = False
        data['prediction_results'] = {'error': 'Invalid prediction uncertainty.'}
        sections = build_report_sections(data, option)
        self.assertIn('User confirmation was not recorded', text(sections))
        self.assertIn('Prediction status: Invalid prediction uncertainty', text(sections))
        self.assertEqual(len(table(sections, 'Prediction source observations')['rows']), 3)

    def test_unspecified_uncertainty_keeps_legacy_report_tables_and_text(self):
        data, option = context()
        data.pop('observation_uncertainty')
        self.assertEqual(build_uncertainty_sections(data), [])
        sections = build_report_sections(data, option)
        self.assertNotIn('Experimental uncertainty', text(sections))
        self.assertIn('prediction inputs held fixed', text(sections))
        self.assertEqual(table(sections, 'In vitro calibration source observations')['rows'][0], ['Source row', 'Time', 'Response'])
        self.assertEqual(table(sections, 'In vitro data actually fitted')['rows'][0], ['Fitting row', 'Analyzed time', 'Analyzed response'])

    def test_word_and_excel_builders_keep_uncertainty_values_without_new_calculations(self):
        from docx import Document
        from openpyxl import load_workbook
        from utilities.excel_export import build_accessible_excel
        from utilities.word_report import build_word_report

        data, option = context('observed_response_inverse')
        sections = build_report_sections(data, option)
        document = Document(build_word_report({'report_label': 'Uncertainty record', 'sections': sections}))
        word_text = '\n'.join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn('Experimental uncertainty', word_text)
        self.assertIn('marginal intervals', word_text)
        self.assertTrue(any('Analyzed SEM' in cell.text for tab in document.tables for row in tab.rows for cell in row.cells))
        values = table(sections, 'In vitro calibration uncertainty values')
        stream, _ = build_accessible_excel({'title': values['title'], 'headers': values['rows'][0], 'rows': values['rows'][1:]})
        sheet = load_workbook(stream).active
        self.assertEqual(sheet.cell(2, 2).value, .5)
        self.assertEqual(sheet.cell(2, 3).value, .05)
        self.assertEqual(sheet.cell(4, 3).value, 'N/A')
        self.assertEqual(sheet.cell(4, 5).value, 'No')


if __name__ == '__main__':
    unittest.main()
