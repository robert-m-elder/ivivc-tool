"""Conditional inverse bounds keep their meaning in reports and downloads."""
import json
import unittest

from test_uncertainty_reporting import context, table, text
from utilities.report_content import build_report_sections


def conditional_report():
    data, option = context('observed_response_inverse')
    rows = data['prediction_results']['approach2']['example']['prediction_tables']['observed_response_inverse']
    for index, row in enumerate(rows):
        row.update(lower_95=(1.7, 3.6)[index], upper_95=(2.4, 4.3)[index],
                   interval_status=('available (conditional on successful draws)', 'available')[index],
                   interval_policy='conditional', interval_min_valid_fraction=.95,
                   interval_conditional=index == 0, interval_valid_samples=(97, 100)[index],
                   interval_total_samples=100, interval_requested_samples=100,
                   interval_random_seed=12345, inverse_extrapolated_samples=0,
                   inverse_failed_draw_reasons=('response not reached within allowed time domain: 3', '')[index])
    return data, option, rows


class InverseConditionalOutputTests(unittest.TestCase):
    def test_report_preserves_bounds_status_counts_and_failure_reasons(self):
        data, option, rows = conditional_report()
        sections = build_report_sections(data, option)
        values = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
        self.assertEqual(values[1][4:6], ['1.7', '2.4'])
        self.assertEqual(values[1][-1], rows[0]['interval_status'])
        self.assertEqual(values[2][-1], 'available')
        counts = table(sections, 'Propagated-uncertainty diagnostics')['rows']
        self.assertEqual(counts[1][1:4], ['97', '100', '100'])
        failures = table(sections, 'Inverse search diagnostics')['rows']
        self.assertEqual(failures[1][-1], rows[0]['inverse_failed_draw_reasons'])
        self.assertIn('Inverse interval policy', text(sections))
        self.assertIn('0.95', text(sections))
        self.assertIn('successful mappings only', text(sections))
        self.assertIn('do not describe all requested outcomes', text(sections))
        json.dumps(sections, allow_nan=False)

    def test_strict_policy_retains_unavailable_bounds_and_no_conditional_claim(self):
        data, option, rows = conditional_report()
        for row in rows:
            row.update(interval_policy='strict', interval_min_valid_fraction=1., interval_conditional=False)
        rows[0].update(lower_95=None, upper_95=None,
                       interval_status='unavailable (nonfinite or missing Monte Carlo draws)')
        sections = build_report_sections(data, option)
        values = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
        self.assertEqual(values[1][4:6], ['N/A', 'N/A'])
        self.assertEqual(values[1][-1], rows[0]['interval_status'])
        policy = next(item['text'] for item in sections
                      if item.get('text', '').startswith('Inverse interval policy'))
        self.assertIn('strict', policy)
        self.assertIn('1', policy)
        self.assertFalse(any(item.get('text', '').startswith('Conditional inverse intervals')
                             for item in sections))

    def test_complete_draws_record_policy_without_conditional_result_caution(self):
        data, option, rows = conditional_report()
        rows[0].update(interval_conditional=False, interval_valid_samples=100,
                       interval_status='available', inverse_failed_draw_reasons='')
        sections = build_report_sections(data, option)
        self.assertIn('Inverse interval policy: conditional', text(sections))
        self.assertFalse(any(item.get('text', '').startswith('Conditional inverse intervals')
                             for item in sections))
        values = table(sections, 'Predictions and displayed uncertainty bounds')['rows']
        self.assertEqual(values[1][-1], 'available')

    def test_word_and_excel_keep_conditional_status_and_numerical_bounds(self):
        from docx import Document
        from openpyxl import load_workbook
        from utilities.excel_export import build_accessible_excel
        from utilities.word_report import build_word_report

        data, option, rows = conditional_report()
        sections = build_report_sections(data, option)
        document = Document(build_word_report({'report_label': 'Conditional inverse', 'sections': sections}))
        self.assertTrue(any(rows[0]['interval_status'] == cell.text
                            for tab in document.tables for row in tab.rows for cell in row.cells))
        self.assertIn('Inverse interval policy', '\n'.join(p.text for p in document.paragraphs))
        values = table(sections, 'Predictions and displayed uncertainty bounds')
        stream, _ = build_accessible_excel({'title': values['title'], 'headers': values['rows'][0],
                                            'rows': values['rows'][1:]})
        sheet = load_workbook(stream).active
        self.assertEqual(sheet.cell(2, 5).value, 1.7)
        self.assertEqual(sheet.cell(2, 6).value, 2.4)
        self.assertEqual(sheet.cell(2, 8).value, rows[0]['interval_status'])


if __name__ == '__main__':
    unittest.main()
