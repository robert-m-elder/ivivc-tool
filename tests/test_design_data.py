"""File interpretation preserves observation precision and dataset roles."""

from datetime import datetime
from io import BytesIO
import json
import unittest

import numpy as np
import pandas as pd

from utilities.design_data import (
    DataImportError, interpret_sd_profile, interpret_time_course,
    interpret_time_scales, read_upload,
)


class NamedUpload(BytesIO):
    def __init__(self, content, filename='measurements.csv'):
        super().__init__(content.encode('utf-8') if isinstance(content, str) else content)
        self.filename = filename


def time_options(**changes):
    result = {'time_column': 'time', 'response_column': 'response'}
    result.update(changes)
    return result


class UploadReaderTests(unittest.TestCase):
    def test_csv_preserves_headers_and_values_and_supports_utf8_bom(self):
        upload = NamedUpload('\ufeff time , response ,SD\n0,1,.1\n1,.5,.2\n')
        result = read_upload(upload)
        self.assertEqual(result.filename, 'measurements.csv')
        self.assertEqual(list(result.frame.columns), ['time', 'response', 'SD'])
        self.assertEqual(result.frame.shape, (2, 3))
        self.assertFalse(upload.closed)

    def test_csv_rejects_duplicate_empty_and_whitespace_equivalent_headers(self):
        for text in ('time,response,response\n0,1,.1\n', 'time,,SD\n0,1,.1\n',
                     ' time,time \n0,1\n', '\n', ''):
            with self.subTest(text=text), self.assertRaises(DataImportError):
                read_upload(NamedUpload(text))

    def test_csv_rejects_short_long_and_malformed_quoted_records(self):
        for text in ('time,response\n0\n', 'time,response\n0,1,.1\n',
                     'time,response\n0,"unfinished\n'):
            with self.subTest(text=text), self.assertRaises(DataImportError):
                read_upload(NamedUpload(text))

    def test_unknown_file_extensions_and_invalid_workbooks_are_rejected(self):
        for filename, content in (('input.txt', b'time,response\n0,1\n'),
                                  ('input.xlsx', b'not an Excel workbook')):
            with self.subTest(filename=filename), self.assertRaises(DataImportError):
                read_upload(NamedUpload(content, filename))

    def workbook(self, sheets):
        data = BytesIO()
        with pd.ExcelWriter(data, engine='openpyxl') as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        return NamedUpload(data.getvalue(), 'data.xlsx')

    def test_excel_returns_worksheet_choices_before_reading_a_multi_sheet_file(self):
        first = pd.DataFrame({'time': [0, 1], 'response': [1, .8]})
        second = pd.DataFrame({'elapsed': [0, 2], 'value': [1, .4]})
        upload = self.workbook({'First': first, 'Second': second})
        choice = read_upload(upload)
        self.assertEqual(choice.sheets, ['First', 'Second'])
        self.assertIsNone(choice.sheet)
        self.assertIsNone(choice.frame)
        upload.seek(0)
        loaded = read_upload(upload, sheet='Second')
        self.assertEqual(loaded.sheet, 'Second')
        self.assertEqual(list(loaded.frame.columns), ['elapsed', 'value'])
        upload.seek(0)
        with self.assertRaises(DataImportError) as caught:
            read_upload(upload, sheet='Absent')
        self.assertEqual(caught.exception.field, 'sheet')

    def test_single_excel_sheet_can_be_selected_without_an_extra_round_trip(self):
        upload = self.workbook({'Only': pd.DataFrame({'time': [0], 'response': [1]})})
        loaded = read_upload(upload)
        self.assertEqual(loaded.sheet, 'Only')
        self.assertEqual(loaded.frame.shape, (1, 2))

    def test_excel_duplicate_headers_are_not_silently_renamed(self):
        frame = pd.DataFrame([[0, 1, .1]], columns=['time', 'response', 'response'])
        with self.assertRaises(DataImportError):
            read_upload(self.workbook({'Sheet': frame}))


class TimeCourseInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'time': [0, 1], 'response': [1., .5],
                                   'SD': [.2, .3], 'n': [4, 9]})

    def test_individual_sd_is_the_uncertainty_of_each_supplied_value(self):
        result = interpret_time_course(self.frame, time_options(
            uncertainty='individual_sd', error_source='column', error_column='SD'))
        np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.2, .3])
        self.assertEqual(result.value.observation_uncertainty.response_kind, 'individual')
        self.assertIsNone(result.value.observation_uncertainty.replicate_counts)

    def test_mean_sd_and_independent_counts_convert_to_sem_exactly_once(self):
        result = interpret_time_course(self.frame, time_options(
            uncertainty='mean_sd', error_source='column', error_column='SD',
            count_source='column', count_column='n'))
        np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.1, .1])
        np.testing.assert_array_equal(result.value.observation_uncertainty.replicate_counts, [4, 9])
        self.assertEqual(result.value.observation_uncertainty.response_kind, 'mean')

    def test_reported_sem_is_not_divided_again_even_when_counts_are_supplied(self):
        for count_options in ({}, {'count_source': 'constant', 'count_value': '16'},
                              {'count_source': 'column', 'count_column': 'n'}):
            with self.subTest(count_options=count_options):
                result = interpret_time_course(self.frame, time_options(
                    uncertainty='mean_sem', error_source='column', error_column='SD',
                    **count_options))
                np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.2, .3])

    def test_constant_error_and_variable_counts_keep_pointwise_mean_precision(self):
        result = interpret_time_course(self.frame, time_options(
            uncertainty='mean_sd', error_source='constant', error_value='.3',
            count_source='column', count_column='n'))
        np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.15, .1])

    def test_constant_mean_count_one_retains_supplied_sd(self):
        result = interpret_time_course(self.frame, time_options(
            uncertainty='mean_sd', error_source='constant', error_value='.07',
            count_source='constant', count_value='1'))
        np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.07, .07])

    def test_unspecified_precision_stays_unspecified(self):
        result = interpret_time_course(self.frame, time_options())
        self.assertIsNone(result.value.observation_uncertainty)
        np.testing.assert_array_equal(result.value.time, [0, 1])

    def test_counts_are_required_for_mean_sd_and_must_be_positive_integers(self):
        options = time_options(uncertainty='mean_sd', error_source='constant', error_value='.1')
        with self.assertRaises(DataImportError):
            interpret_time_course(self.frame, options)
        for count in ('0', '-1', '2.5', 'nan', 'inf', str(2**64)):
            with self.subTest(count=count), self.assertRaises(DataImportError):
                interpret_time_course(self.frame, {**options, 'count_source': 'constant',
                                                  'count_value': count})

    def test_missing_pairs_are_reported_without_inspecting_their_unused_precision(self):
        frame = pd.DataFrame({'time': [0, None, 1], 'response': [1, None, .5],
                              'SD': [.1, 'not precision', .2], 'n': [4, 'not a count', 16]})
        result = interpret_time_course(frame, time_options(
            uncertainty='mean_sd', error_source='column', error_column='SD',
            count_source='column', count_column='n'))
        np.testing.assert_array_equal(result.value.time, [0, 1])
        np.testing.assert_allclose(result.value.observation_uncertainty.sigma, [.05, .05])
        self.assertEqual(result.omitted_rows, [3])

    def test_partial_rows_and_nonfinite_mapped_values_do_not_disappear(self):
        for time, response in ((None, .5), (1, None), ('invalid', .5),
                               (np.inf, .5), (1, np.nan)):
            with self.subTest(time=time, response=response), self.assertRaises(DataImportError):
                interpret_time_course(pd.DataFrame({'time': [time], 'response': [response]}),
                                      time_options())

    def test_excel_booleans_dates_and_complex_values_are_not_measurements(self):
        for value in (True, np.bool_(False), datetime(2026, 1, 1), 1+2j):
            with self.subTest(value=value), self.assertRaises(DataImportError):
                interpret_time_course(pd.DataFrame({'time': [0], 'response': [value]}, dtype=object),
                                      time_options())

    def test_invalid_errors_and_unknown_mapping_choices_are_rejected(self):
        for value in ('0', '-.1', 'nan', 'inf', 'not an SD'):
            with self.subTest(value=value), self.assertRaises(DataImportError):
                interpret_time_course(self.frame, time_options(
                    uncertainty='individual_sd', error_source='constant', error_value=value))
        for options in ({'time_column': 'absent'}, {'time_column': ''},
                        {'response_column': 'time'}, {'uncertainty': 'relative'},
                        {'uncertainty': 'individual_sd', 'error_source': 'column',
                         'error_column': 'missing'}):
            with self.subTest(options=options), self.assertRaises(DataImportError):
                interpret_time_course(self.frame, time_options(**options))

    def test_order_and_repeated_times_are_preserved_and_preview_is_bounded(self):
        frame = pd.DataFrame({'time': [2, 0, 2] + list(range(3, 20)),
                              'response': np.linspace(1, 0, 20)})
        result = interpret_time_course(frame, time_options())
        np.testing.assert_array_equal(result.value.time[:3], [2, 0, 2])
        self.assertEqual(len(result.value.time), 20)
        self.assertLessEqual(len(result.rows), 10)
        json.dumps(result.summary, allow_nan=False)
        json.dumps(result.rows, allow_nan=False)


class DistributionAndProfileTests(unittest.TestCase):
    def test_profile_uses_individual_sd_with_nearest_time_endpoint_extension(self):
        result = interpret_sd_profile(pd.DataFrame({'time': [0, 10], 'sd': [.02, .1]}), {})
        model = result.value
        self.assertEqual(model.kind, 'time_profile_sd')
        np.testing.assert_allclose(model.standard_deviation([1, 1, 1], times=[-1, 8, 12]),
                                   [.02, .1, .1])

    def test_zero_future_sd_is_allowed_but_bad_order_and_negative_sd_are_not(self):
        result = interpret_sd_profile(pd.DataFrame({'time': [0, 1], 'sd': [0., .1]}), {})
        np.testing.assert_array_equal(result.value.profile_sd, [0., .1])
        for times, sd in (([0, 0], [.1, .2]), ([1, 0], [.1, .2]),
                          ([0, 1], [.1, -.2]), ([0, 1], [.1, np.nan])):
            with self.subTest(times=times, sd=sd), self.assertRaises(DataImportError):
                interpret_sd_profile(pd.DataFrame({'time': times, 'sd': sd}), {})

    def test_empirical_ratios_keep_duplicate_records_and_normalize_given_weights(self):
        frame = pd.DataFrame({'ratio': [2., 1., 2.], 'weight': [0., 1., 3.],
                              'source': ['same', 'other', 'same']})
        result = interpret_time_scales(frame, {'ratio_column': 'ratio',
            'weight_column': 'weight', 'source_column': 'source'})
        np.testing.assert_array_equal(result.value.ratios, [2, 1, 2])
        np.testing.assert_allclose(result.value.weights, [0, .25, .75])
        self.assertEqual(tuple(result.value.source_ids), ('same', 'other', 'same'))

    def test_empirical_missing_weight_mapping_means_equal_weights(self):
        result = interpret_time_scales(pd.DataFrame({'ratio': [1., 3.]}), {})
        np.testing.assert_allclose(result.value.weights, [.5, .5])

    def test_empirical_bad_ratios_or_supplied_weights_are_not_removed(self):
        for ratio, weights in (([0, 1], [1, 1]), ([-1, 1], [1, 1]),
                               ([1, np.inf], [1, 1]), ([1, 2], [0, 0]),
                               ([1, 2], [1, -1]), ([1, 2], [1, None])):
            with self.subTest(ratio=ratio, weights=weights), self.assertRaises(DataImportError):
                interpret_time_scales(pd.DataFrame({'ratio': ratio, 'weight': weights}),
                                      {'weight_column': 'weight'})


if __name__ == '__main__':
    unittest.main()
