"""HTTP integration for request-scoped study-design data imports."""

from html.parser import HTMLParser
import io
import unittest

import pandas as pd

from design_app import create_app


class PreviewTables(HTMLParser):
    """Read rendered preview values without depending on numeric formatting."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('th', 'td') and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            if self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None

    @classmethod
    def records(cls, html):
        parser = cls()
        parser.feed(html)
        return [[dict(zip(table[0], row)) for row in table[1:]]
                for table in parser.tables]


def workbook_bytes(**sheets):
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return stream.getvalue()


class DesignDataRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    def post(self, action, content='time,response\n0,1\n4,.5\n',
             dataset_id='reference-file', filename='measurements.csv', **options):
        if isinstance(content, str):
            content = content.encode('utf-8')
        data = {'dataset_id': dataset_id, 'file': (io.BytesIO(content), filename)}
        data.update(options)
        return self.client.post('/data/' + action, data=data,
                                content_type='multipart/form-data')

    def preview(self, content='time,response\n0,1\n4,.5\n', **options):
        return self.post('preview', content=content, **options)

    def payload(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIs(payload['ok'], True)
        self.assertFalse(response.headers.getlist('Set-Cookie'))
        return payload

    def error(self, response, field_id, status=400):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIs(payload['ok'], False)
        self.assertNotIn('html', payload)
        self.assertTrue(any(error.get('field_id') == field_id
                            for error in payload['errors']), payload)
        return payload

    @staticmethod
    def course_options(**options):
        return {'reference-file__profile__' + name: value for name, value in options.items()}

    def test_csv_inspection_returns_columns_and_role_specific_mapping(self):
        payload = self.payload(self.post('inspect', 'age,mass,error\n0,1,.06\n'))
        self.assertEqual(payload['dataset_id'], 'reference-file')
        self.assertEqual(payload['columns'], ['age', 'mass', 'error'])
        self.assertEqual(payload['sheets'], [])
        self.assertFalse(payload['requires_sheet'])
        self.assertIn('reference-file__profile__time_column', payload['html'])
        self.assertIn('reference-file__profile__uncertainty', payload['html'])

    def test_missing_or_unknown_dataset_and_missing_upload_have_clear_errors(self):
        self.error(self.post('inspect', dataset_id='not-a-role'), None)
        self.error(self.client.post('/data/preview', data={}), None)
        for action in ('inspect', 'preview'):
            with self.subTest(action=action):
                response = self.client.post('/data/' + action,
                                            data={'dataset_id': 'reference-file'})
                self.error(response, 'reference-file')

    def test_malformed_files_fail_without_returning_partial_preview_html(self):
        cases = (
            ('a,a\n1,2\n', 'measurements.csv'),
            ('a,b\n1,2,extra\n', 'measurements.csv'),
            ('a,b\n', 'measurements.csv'),
            ('', 'measurements.csv'),
            ('a,b\n1,2\n', 'measurements.txt'),
        )
        for content, filename in cases:
            with self.subTest(content=content, filename=filename):
                self.error(self.post('inspect', content, filename=filename), 'reference-file')

    def test_excel_requires_a_sheet_choice_when_multiple_sheets_exist(self):
        content = workbook_bytes(
            Earlier=pd.DataFrame({'early_t': [0, 1], 'early_y': [1., .8]}),
            Later=pd.DataFrame({'later_t': [2, 4], 'later_y': [.6, .3]}),
        )
        initial = self.payload(self.post('inspect', content, filename='curves.xlsx'))
        self.assertTrue(initial['requires_sheet'])
        self.assertEqual(initial['sheets'], ['Earlier', 'Later'])
        self.assertEqual(initial['columns'], [])
        self.assertIn('reference-file__sheet', initial['html'])
        self.assertNotIn('data-import-action="preview"', initial['html'])

        chosen = self.payload(self.post('inspect', content, filename='curves.xlsx', sheet='Later'))
        self.assertEqual(chosen['selected_sheet'], 'Later')
        self.assertEqual(chosen['columns'], ['later_t', 'later_y'])
        preview = self.payload(self.preview(content, filename='curves.xlsx', sheet='Later'))
        rows = PreviewTables.records(preview['html'])[0]
        self.assertEqual([float(row['Time']) for row in rows], [2., 4.])
        self.assertIn('Later', preview['html'])

    def test_missing_or_invalid_excel_sheet_errors_target_the_sheet_control(self):
        frame = pd.DataFrame({'t': [0], 'y': [1]})
        content = workbook_bytes(A=frame, B=frame)
        self.error(self.preview(content, filename='curves.xlsx'), 'reference-file__sheet')
        self.error(self.post('inspect', content, filename='curves.xlsx', sheet='Missing'),
                   'reference-file__sheet')

    def test_single_sheet_excel_can_be_read_without_an_arbitrary_sheet_choice(self):
        content = workbook_bytes(Only=pd.DataFrame({'time': [0], 'response': [1.]}))
        payload = self.payload(self.post('inspect', content, filename='curve.xlsx'))
        self.assertFalse(payload['requires_sheet'])
        self.assertEqual(payload['selected_sheet'], 'Only')

    def test_means_with_column_sd_and_counts_are_converted_once(self):
        payload = self.payload(self.preview(
            't,y,sd,n\n0,1,.06,4\n4,.5,.08,16\n',
            **self.course_options(uncertainty='mean_sd', error_source='column',
                                  error_column='sd', count_source='column', count_column='n')))
        rows = PreviewTables.records(payload['html'])[0]
        self.assertEqual([float(row['Supplied SD']) for row in rows], [.06, .08])
        self.assertEqual([float(row['Observation SEM']) for row in rows], [.03, .02])
        self.assertEqual([int(row['Independent count']) for row in rows], [4, 16])

    def test_supplied_sem_is_not_divided_again_when_a_count_is_present(self):
        payload = self.payload(self.preview(**self.course_options(
            uncertainty='mean_sem', error_source='constant', error_value='.06',
            count_source='constant', count_value='9')))
        rows = PreviewTables.records(payload['html'])[0]
        self.assertEqual([float(row['Observation SEM']) for row in rows], [.06, .06])
        self.assertIn('no further division', payload['html'])

    def test_individual_sd_and_unspecified_precision_remain_distinct(self):
        supplied = self.payload(self.preview(**self.course_options(
            uncertainty='individual_sd', error_source='constant', error_value='.05')))
        rows = PreviewTables.records(supplied['html'])[0]
        self.assertEqual(float(rows[0]['Observation SD']), .05)
        unspecified = self.payload(self.preview())
        rows = PreviewTables.records(unspecified['html'])[0]
        self.assertNotIn('Observation SD', rows[0])
        self.assertIn('unspecified', unspecified['html'])

    def test_invalid_precision_and_mapping_errors_link_to_the_selected_options(self):
        cases = (
            ({'time_column': 'Missing'}, 'time_column'),
            ({'response_column': ''}, 'response_column'),
            ({'uncertainty': 'mean_sd', 'error_source': 'constant', 'error_value': '.05'},
             'count_source'),
            ({'uncertainty': 'mean_sem', 'error_source': 'constant', 'error_value': '0'},
             'error_value'),
        )
        for options, field in cases:
            with self.subTest(options=options):
                self.error(self.preview(**self.course_options(**options)),
                           'reference-file__profile__' + field)

    def test_paired_defaults_use_four_columns_and_report_each_sides_padding(self):
        payload = self.payload(self.preview(
            't_iv,y_iv,t_vivo,y_vivo\n0,1,0,1\n1,.7,4,.6\n2,.4,,\n',
            dataset_id='ivivc-calibration-file'))
        vitro, vivo = PreviewTables.records(payload['html'])
        self.assertEqual([float(row['Time']) for row in vitro], [0., 1., 2.])
        self.assertEqual([float(row['Time']) for row in vivo], [0., 4.])
        self.assertEqual([float(row['Response']) for row in vivo], [1., .6])
        self.assertIn('Omitted rows: 4', payload['html'])

    def test_partial_rows_fail_instead_of_being_silently_removed(self):
        self.error(self.preview('t,y\n0,1\n4,\n'),
                   'reference-file__profile__response_column')

    def test_future_sd_profiles_have_individual_sd_semantics_and_validate_order(self):
        payload = self.payload(self.preview('time,sd\n0,.02\n10,.08\n',
                                            dataset_id='future-sd-profile-file'))
        rows = PreviewTables.records(payload['html'])[0]
        self.assertEqual([float(row['Individual SD']) for row in rows], [.02, .08])
        self.assertIn('nearest-time', payload['html'])
        self.assertNotIn('Observation SEM', payload['html'])
        self.error(self.preview('time,sd\n2,.02\n1,.08\n',
                                dataset_id='future-sd-profile-file'),
                   'future-sd-profile-file__time_column')

    def test_empirical_ratios_preserve_records_and_normalize_only_weights(self):
        weighted = self.payload(self.preview('ratio,weight,source_id\n2,1,A\n3,3,B\n4,0,A\n',
                                             dataset_id='time-scale-file'))
        rows = PreviewTables.records(weighted['html'])[0]
        self.assertEqual([float(row['Ratio']) for row in rows], [2., 3., 4.])
        self.assertEqual([float(row['Normalized weight']) for row in rows], [.25, .75, 0.])
        equal = self.payload(self.preview('ratio\n2\n3\n', dataset_id='time-scale-file'))
        rows = PreviewTables.records(equal['html'])[0]
        self.assertEqual([float(row['Normalized weight']) for row in rows], [.5, .5])
        self.error(self.preview('ratio,weight\n2,\n3,1\n', dataset_id='time-scale-file'),
                   'time-scale-file__weight_column')

    def test_uploaded_names_and_headers_are_escaped_in_server_rendered_html(self):
        header = '<img src=x onerror=alert(1)>'
        filename = '<script>alert(1)</script>.csv'
        content = f'time,{header}\n0,1\n'
        inspected = self.payload(self.post('inspect', content, filename=filename))
        self.assertIn('&lt;img', inspected['html'])
        self.assertNotIn('<img ', inspected['html'])
        preview = self.payload(self.preview(content, filename=filename,
                                           **self.course_options(response_column=header)))
        self.assertIn('&lt;script&gt;', preview['html'])
        self.assertNotIn('<script>', preview['html'])

    def test_previews_are_request_scoped_and_do_not_set_session_cookies(self):
        first = self.payload(self.preview('t,y\n0,111.125\n'))
        second = self.payload(self.preview('t,y\n0,222.25\n'))
        self.assertEqual(float(PreviewTables.records(first['html'])[0][0]['Response']), 111.125)
        self.assertEqual(float(PreviewTables.records(second['html'])[0][0]['Response']), 222.25)
        self.assertNotIn('111.125', second['html'])

    def test_upload_limit_applies_to_both_import_endpoints(self):
        client = create_app({'TESTING': True, 'MAX_CONTENT_LENGTH': 100}).test_client()
        for action in ('inspect', 'preview'):
            with self.subTest(action=action):
                response = client.post('/data/' + action, data={
                    'dataset_id': 'reference-file',
                    'file': (io.BytesIO(b'x' * 200), 'large.csv'),
                })
                self.error(response, None, status=413)


if __name__ == '__main__':
    unittest.main()
