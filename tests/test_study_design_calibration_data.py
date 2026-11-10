"""Calibration additions retain independent rows and their original precision."""
import unittest

import numpy as np

from ivivc_core import (
    ObservationUncertainty, TimeCourse, fit_ivivc, fit_time_course, prepare_ivivc_data,
    prepare_time_course,
)
from study_design_core.calibration_data import append_calibration_observations


CONFIG = dict(cv_scheme='none', grid_search_num_points=3,
              grid_search_num_cores=1, grid_search_random_state=12345)


def course(time, value, sigma=None, *, kind='mean', counts=None, source='historical', units=None):
    uncertainty = None if sigma is None else ObservationUncertainty(
        sigma, response_kind=kind, replicate_counts=counts,
        source=source, response_units=units,
    )
    return prepare_time_course(time, value, label=source,
                               observation_uncertainty=uncertainty)[0]


class CalibrationDataTests(unittest.TestCase):
    def test_added_mean_at_existing_time_retains_old_sem_and_independent_row(self):
        old = course([0., 2.], [1., .6], [.05, .08])
        new = course([2., 4.], [.5, .2], [.03, .04], counts=[9, 4], source='proposed')
        result = append_calibration_observations(old, new)
        np.testing.assert_array_equal(result.time, [0., 2., 2., 4.])
        np.testing.assert_array_equal(result.value, [1., .6, .5, .2])
        np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.05, .08, .03, .04])
        self.assertEqual(result.observation_uncertainty.response_kind, 'mean')
        self.assertIsNone(result.observation_uncertainty.replicate_counts)
        np.testing.assert_array_equal(new.observation_uncertainty.replicate_counts, [9, 4])
        np.testing.assert_array_equal(old.observation_uncertainty.sigma, [.05, .08])
        self.assertIn('existing: historical', result.observation_uncertainty.source)
        self.assertIn('additions: proposed', result.observation_uncertainty.source)

    def test_stable_time_order_keeps_values_errors_and_counts_aligned(self):
        old = course([2., 0.], [.6, 1.], [.08, .05], counts=[4, 9], units='fraction')
        new = course([2., 1.], [.5, .8], [.03, .02], counts=[16, 25], units='fraction')
        result = append_calibration_observations(old, new, label='in vivo calibration')
        np.testing.assert_array_equal(result.time, [0., 1., 2., 2.])
        np.testing.assert_array_equal(result.value, [1., .8, .6, .5])
        np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.05, .02, .08, .03])
        np.testing.assert_array_equal(result.observation_uncertainty.replicate_counts, [9, 25, 4, 16])
        self.assertEqual(result.observation_uncertainty.response_units, 'fraction')
        self.assertEqual(result.label, 'in vivo calibration')
        np.testing.assert_array_equal(old.time, [2., 0.])

    def test_individual_replicates_are_not_collapsed_or_divided(self):
        old = course([0., 1.], [1., .8], .1, kind='individual')
        new = course([1., 1.], [.75, .85], [.2, .3], kind='individual')
        result = append_calibration_observations(old, new)
        np.testing.assert_array_equal(result.time, [0., 1., 1., 1.])
        np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.1, .1, .2, .3])
        self.assertEqual(result.observation_uncertainty.response_kind, 'individual')

    def test_single_input_is_independent_sorted_copy_with_aligned_precision(self):
        old = course([2., 0.], [.6, 1.], [.08, .05], counts=[4, 9], units='fraction')
        for existing, additions in ((old, None), (None, old)):
            with self.subTest(existing=existing is not None):
                result = append_calibration_observations(existing, additions)
                np.testing.assert_array_equal(result.time, [0., 2.])
                np.testing.assert_array_equal(result.value, [1., .6])
                np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.05, .08])
                np.testing.assert_array_equal(result.observation_uncertainty.replicate_counts, [9, 4])
                for name in ('response_kind', 'input_form', 'source', 'response_units'):
                    self.assertEqual(getattr(result.observation_uncertainty, name),
                                     getattr(old.observation_uncertainty, name))
                self.assertIsNot(result, old)
                self.assertFalse(np.shares_memory(result.time, old.time))
                self.assertFalse(np.shares_memory(result.value, old.value))
                self.assertFalse(np.shares_memory(result.observation_uncertainty.sigma,
                                                 old.observation_uncertainty.sigma))
                result.value[0] = 99.
                self.assertEqual(old.value[0], .6)
                np.testing.assert_array_equal(old.time, [2., 0.])
                np.testing.assert_array_equal(old.observation_uncertainty.sigma, [.08, .05])
                np.testing.assert_array_equal(old.observation_uncertainty.replicate_counts, [4, 9])

    def test_no_additions_and_added_rows_use_same_initial_normalization_reference(self):
        old = course([2., 0., 4.], [.6, 1., .2], [.08, .05, .04], counts=[4, 9, 16])
        additions = course([3.], [.4], .03, counts=4)
        prepared = []
        for new in (None, additions):
            rows = append_calibration_observations(old, new)
            prepared.append(prepare_ivivc_data(
                rows, rows, selected_interpolation=['none'],
                selected_normalizations=['initial_value']))
        for result in prepared:
            self.assertEqual(result.preprocessing.normalizations[0]
                             .in_vivo_parameters['initial_value'], 1.)
            self.assertEqual(result.in_vivo.time[0], 0.)
            self.assertEqual(result.in_vivo.value[0], 1.)

    def test_unspecified_precision_remains_unspecified(self):
        result = append_calibration_observations(
            course([2., 0.], [.6, 1.]), course([1.], [.8]))
        self.assertIsNone(result.observation_uncertainty)
        np.testing.assert_array_equal(result.time, [0., 1., 2.])

    def test_missing_unit_labels_do_not_claim_common_known_units(self):
        result = append_calibration_observations(
            course([0.], [1.], .1, units='fraction'),
            course([1.], [.8], .1))
        self.assertIsNone(result.observation_uncertainty.response_units)

    def test_unknown_counts_on_either_side_do_not_invent_metadata(self):
        for old_counts, new_counts in ((None, [4]), ([9], None), (None, None)):
            with self.subTest(old_counts=old_counts, new_counts=new_counts):
                result = append_calibration_observations(
                    course([0.], [1.], .1, counts=old_counts),
                    course([1.], [.8], .05, counts=new_counts))
                self.assertIsNone(result.observation_uncertainty.replicate_counts)
                np.testing.assert_array_equal(result.observation_uncertainty.sigma, [.1, .05])

    def test_mixed_precision_or_response_interpretation_is_rejected(self):
        known = course([0.], [1.], .1)
        unknown = course([1.], [.8])
        for old, new in ((known, unknown), (unknown, known)):
            with self.assertRaisesRegex(ValueError, 'historical uncertainty explicitly'):
                append_calibration_observations(old, new)
        with self.assertRaisesRegex(ValueError, 'same individual/mean'):
            append_calibration_observations(known, course([1.], [.8], .1, kind='individual'))

    def test_conflicting_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'compatible response units'):
            append_calibration_observations(
                course([0.], [1.], .1, units='fraction'),
                course([1.], [80.], 10., units='percent'))

    def test_absent_empty_or_nonfinite_datasets_are_rejected_without_removal(self):
        with self.assertRaises(ValueError):
            append_calibration_observations(None, None)
        for invalid in (TimeCourse([], []), TimeCourse([0., np.nan], [1., .8]),
                        TimeCourse([0., 1.], [1., np.inf])):
            with self.subTest(time=invalid.time):
                with self.assertRaisesRegex(ValueError, 'finite time/response rows'):
                    append_calibration_observations(None, invalid)
        with self.assertRaises(TypeError):
            append_calibration_observations(([0.], [1.]), None)

    def test_same_dataset_is_not_appended_as_an_independent_batch(self):
        old = course([0.], [1.], .1)
        with self.assertRaisesRegex(ValueError, 'same object'):
            append_calibration_observations(old, old)

    def test_duplicate_mean_rows_reach_weighted_fit_and_match_analytic_solution(self):
        combined = append_calibration_observations(
            course([0., 2.], [1.02, .61], [.05, .08], counts=[4, 9]),
            course([2., 4.], [.53, .24], [.03, .04], counts=[16, 4]))
        prepared = prepare_ivivc_data(combined, combined,
                                     selected_interpolation=['none'])
        fit = fit_time_course(
            prepared.in_vivo.time, prepared.in_vivo.value, model_names='linear',
            selected_metrics=['rmse'], analysis_config=CONFIG,
            observation_uncertainty=prepared.in_vivo.observation_uncertainty,
            fail_fast=True)['linear']
        sigma = combined.observation_uncertainty.sigma
        matrix = np.column_stack([np.ones(len(combined.time)), combined.time])
        information = matrix.T @ (matrix / sigma[:, None]**2)
        covariance = np.linalg.inv(information)
        estimate = covariance @ (matrix.T @ (combined.value / sigma**2))
        np.testing.assert_allclose(fit.parameter_vector, estimate, rtol=1e-7)
        np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-6)
        np.testing.assert_array_equal(fit.time, [0., 2., 2., 4.])
        np.testing.assert_array_equal(fit.observation_uncertainty.sigma, sigma)

    def test_approach2_preserves_added_independent_means_at_the_same_time(self):
        combined = append_calibration_observations(
            course([0., 2., 4.], [1., .67, .45], .03, counts=4),
            course([2., 6.], [.66, .3], .02, counts=9))
        prepared = prepare_ivivc_data(combined, combined,
                                     selected_interpolation=['none'])
        analysis = fit_ivivc(prepared, selections='approach2:exponential',
                            selected_metrics=['rmse'], analysis_config=CONFIG,
                            fail_fast=True)
        fit = analysis['approach2:exponential'].in_vivo_fit
        np.testing.assert_array_equal(fit.time, [0., 2., 2., 4., 6.])
        np.testing.assert_array_equal(fit.observation_uncertainty.sigma,
                                      [.03, .03, .02, .03, .02])
        np.testing.assert_array_equal(fit.observation_uncertainty.replicate_counts,
                                      [4, 4, 9, 4, 9])


if __name__ == '__main__':
    unittest.main()
