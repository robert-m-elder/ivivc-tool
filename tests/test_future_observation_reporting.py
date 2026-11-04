"""Future-response assumptions and bounds survive reports and document exports."""
import json
import unittest

from test_uncertainty_reporting import context, table, text
from utilities.report_content import build_report_sections, build_uncertainty_sections


def future_report(method='observed_response_inverse', *, kind='individual', source='supplied'):
    data, option = context(method)
    data['future_observation_input'] = {
        'mode': 'constant' if source == 'supplied' else source,
        'response_kind': kind, 'entered_standard_uncertainty': .3,
        'analyzed_standard_uncertainty': .03, 'normalization_constants_fixed': True,
    }
    records = data['prediction_results']['approach2']['example']['prediction_tables'][method]
    for index, row in enumerate(records):
        conditional = method == 'observed_response_inverse' and index == 0
        status = 'available (conditional on successful draws)' if conditional else 'available'
        row.update(future_observation_variability_included=True,
                   future_observation_uncertainty_source=source,
                   future_standard_uncertainty=(.03, .04)[index],
                   future_uncertainty_kind='sem' if kind == 'mean' else 'sd', future_response_kind=kind,
                   future_replicate_count=(4, 9)[index] if kind == 'mean' else None,
                   future_calibration_time=(2., 6.)[index], future_lookup_extrapolated=index == 1,
                   interval_status=status, interval_policy='conditional', interval_min_valid_fraction=.5,
                   interval_conditional=conditional, interval_valid_samples=(79, 100)[index],
                   interval_total_samples=100, interval_requested_samples=100,
                   value_interval_kind='future_observation_prediction', value_interval_status=status,
                   value_interval_valid_samples=(79, 100)[index], value_interval_requested_samples=100,
                   value_interval_conditional=conditional,
                   response_lower_95=(.65, .4)[index], response_upper_95=(1.35, 1.02)[index],
                   propagated_response_lower_95=(.9, .6)[index], propagated_response_upper_95=(1.1, .8)[index],
                   propagated_response_interval_kind='prediction_input_uncertainty',
                   propagated_response_interval_status='available')
        row.update(lower_95=(.65, .4)[index] if method == 'value_ratio' else (1.7, 3.6)[index],
                   upper_95=(1.35, 1.02)[index] if method == 'value_ratio' else (2.4, 4.3)[index])
    return data, option, records


class FutureObservationReportingTests(unittest.TestCase):
    def test_future_only_constant_records_original_and_analyzed_values(self):
        data, option, _ = future_report('value_ratio')
        data.pop('observation_uncertainty')
        sections = build_report_sections(data, option)
        settings = table(sections, 'Future measurement variability')['rows']
        self.assertIn(['Entered uncertainty (original in vivo response units)', '0.3'], settings)
        self.assertIn(['Analyzed uncertainty', '0.03'], settings)
        self.assertIn('one future individual measurement', text(sections))
        self.assertIn('Future in vivo measurement errors are assumed independent', text(sections))
        self.assertNotIn('these are not prediction intervals for future measurements', text(sections).lower())
        self.assertNotIn('Future in vivo measurement variability is excluded', text(sections))
        self.assertTrue(build_uncertainty_sections(data))
        json.dumps(sections, allow_nan=False)

    def test_calibration_reuse_reports_future_mean_and_endpoint_assumption(self):
        data, option, _ = future_report('tau_ratio', kind='mean', source='calibration')
        sections = build_report_sections(data, option)
        assumptions = table(sections, 'Future measurement assumptions')['rows']
        self.assertIn('Source replicate count (metadata)', assumptions[0])
        self.assertEqual(assumptions[1][2:6], ['0.03', 'SEM', 'mean', '4'])
        self.assertEqual(assumptions[2][-2:], ['6.0', 'True'])
        self.assertIn('a future replicate mean', text(sections))
        self.assertIn('nearest endpoint error is reused as an explicit assumption', text(sections))
        self.assertIn('held fixed during propagation', text(sections))
        self.assertNotIn('Future in vivo measurement variability is not included', text(sections))

    def test_inverse_retains_separate_time_and_future_response_bounds(self):
        data, option, records = future_report()
        records[0]['response_lower_95'] = 1.23456789012345
        sections = build_report_sections(data, option)
        primary = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
        future = table(sections, 'Future response prediction intervals')['rows']
        original = table(sections, 'Propagated response bounds before future variability')['rows']
        self.assertEqual(primary[1][4:6], ['1.7', '2.4'])
        self.assertEqual(future[1][1], repr(records[0]['response_lower_95']))
        self.assertEqual(future[1][3:], [records[0]['value_interval_status'], '79', '100'])
        self.assertEqual(original[1][1:3], ['0.9', '1.1'])
        self.assertIn('minimum usable fraction = 0.5', text(sections))
        self.assertIn('Conditional future-response intervals describe successful inverse mappings only', text(sections))
        self.assertIn('not a joint 95% region', text(sections))

    def test_word_and_excel_preserve_future_interpretation_and_exact_bounds(self):
        from docx import Document
        from openpyxl import load_workbook
        from utilities.excel_export import build_accessible_excel
        from utilities.word_report import build_word_report

        data, option, records = future_report()
        records[0]['response_lower_95'] = 1.23456789012345
        sections = build_report_sections(data, option)
        document = Document(build_word_report({'report_label': 'Future response', 'sections': sections}))
        self.assertIn('one future individual measurement', '\n'.join(p.text for p in document.paragraphs))
        self.assertTrue(any(repr(records[0]['response_lower_95']) == cell.text
                            for tab in document.tables for row in tab.rows for cell in row.cells))
        values = table(sections, 'Future response prediction intervals')
        stream, _ = build_accessible_excel({'title': values['title'], 'headers': values['rows'][0],
                                            'rows': values['rows'][1:]})
        sheet = load_workbook(stream).active
        self.assertAlmostEqual(sheet.cell(2, 2).value, records[0]['response_lower_95'], places=14)
        self.assertEqual(sheet.cell(2, 4).value, records[0]['value_interval_status'])


if __name__ == '__main__':
    unittest.main()
