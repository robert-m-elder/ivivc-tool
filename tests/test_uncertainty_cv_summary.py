"""Compact held-out summaries respect the supplied absolute error scales."""
import unittest

import numpy as np
from sklearn.model_selection import LeaveOneOut

from utilities.evaluation import cross_validation_curve_fit


def slope(time, rate):
    return np.asarray(time) * rate


def line(time, intercept, rate):
    return intercept + np.asarray(time) * rate


class Splits:
    def __init__(self, values):
        self.values = values

    def split(self, _):
        for train, test in self.values:
            yield np.asarray(train), np.asarray(test)


class UncertaintyCVSummaryTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(1., 6.)
        self.response = np.array([1.1, 2., 2.8, 4.2, 10.])
        self.sigma = np.array([0.1, 0.2, 0.15, 0.2, 3.])

    def run_cv(self, cv, **changes):
        options = dict(p0=[1.], sigma=self.sigma, absolute_sigma=True,
                       return_diagnostics=True)
        options.update(changes)
        return cross_validation_curve_fit(
            self.time, self.response, slope, cv, ['rmse', 'aic', 'aicc', 'bic'], **options,
        )

    def expected(self, splits):
        chi = nll = 0.
        count = 0
        for train, test in splits:
            train, test = np.asarray(train), np.asarray(test)
            rate = np.sum(self.time[train] * self.response[train] / self.sigma[train] ** 2)
            rate /= np.sum(self.time[train] ** 2 / self.sigma[train] ** 2)
            z = (self.response[test] - rate * self.time[test]) / self.sigma[test]
            chi += np.sum(z ** 2)
            nll += .5 * np.sum(z ** 2 + np.log(2. * np.pi) + 2. * np.log(self.sigma[test]))
            count += len(test)
        return np.sqrt(chi / count), nll, count

    def test_loo_uses_training_weights_and_held_out_sigma(self):
        scores, diagnostics = self.run_cv(LeaveOneOut())
        splits = [(np.flatnonzero(np.arange(5) != i), [i]) for i in range(5)]
        rmse, nll, count = self.expected(splits)
        summary = diagnostics['observation_error']
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['successful_splits'], 5)
        self.assertEqual(summary['n_test'], count)
        np.testing.assert_allclose(summary['standardized_rmse'], rmse, rtol=1.e-6)
        np.testing.assert_allclose(summary['negative_log_likelihood'], nll, rtol=1.e-6)
        np.testing.assert_allclose(summary['mean_negative_log_likelihood'], nll / count, rtol=1.e-6)
        self.assertTrue(np.isfinite(scores['rmse']).all())
        for name in ('aic', 'aicc', 'bic'):
            self.assertTrue(np.isnan(scores[name]).all())

    def test_unequal_folds_and_repeated_test_rows_pool_occurrences(self):
        splits = [([0, 1, 2], [3]), ([0, 3, 4], [1, 2]), ([0, 1, 2], [3])]
        _, diagnostics = self.run_cv(Splits(splits))
        rmse, nll, count = self.expected(splits)
        summary = diagnostics['observation_error']
        self.assertEqual(summary['n_test'], 4)
        np.testing.assert_allclose(summary['standardized_rmse'], rmse, rtol=1.e-6)
        np.testing.assert_allclose(summary['negative_log_likelihood'], nll, rtol=1.e-6)
        np.testing.assert_allclose(summary['mean_negative_log_likelihood'], nll / count, rtol=1.e-6)

    def test_failed_fold_withholds_the_pooled_comparison(self):
        _, diagnostics = cross_validation_curve_fit(
            self.time, self.response, line, Splits([([0], [1, 2]), ([0, 1, 2], [3, 4])]),
            ['rmse'], p0=[0., 1.], sigma=self.sigma, absolute_sigma=True, return_diagnostics=True,
        )
        summary = diagnostics['observation_error']
        self.assertEqual(diagnostics['fitting_failure_count'], 1)
        self.assertEqual(summary['successful_splits'], 1)
        self.assertEqual(summary['n_test'], 4)
        self.assertFalse(summary['complete'])
        self.assertTrue(np.isnan(summary['standardized_rmse']))
        self.assertTrue(np.isnan(summary['negative_log_likelihood']))

    def test_disabled_cv_does_not_invent_test_scores(self):
        _, diagnostics = self.run_cv(None)
        summary = diagnostics['observation_error']
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['n_test'], 0)
        self.assertEqual(summary['successful_splits'], 0)
        self.assertTrue(np.isnan(summary['negative_log_likelihood']))

    def test_effective_absolute_options_and_unspecified_or_relative_errors(self):
        _, direct = self.run_cv(LeaveOneOut())
        _, legacy = self.run_cv(LeaveOneOut(), sigma=None, absolute_sigma=False,
                                kwargs={'sigma': self.sigma, 'absolute_sigma': True})
        self.assertEqual(direct['observation_error'], legacy['observation_error'])
        for sigma in (None, self.sigma):
            with self.subTest(sigma=sigma):
                _, diagnostics = self.run_cv(LeaveOneOut(), sigma=sigma, absolute_sigma=False)
                self.assertNotIn('observation_error', diagnostics)

    def test_nonfinite_test_likelihood_is_separate_from_fitting_failure(self):
        _, diagnostics = cross_validation_curve_fit(
            np.array([1., 2., 3.]), np.array([1., 2., 1.e100]), slope,
            Splits([([0, 1], [2])]), ['rmse'], p0=[1.],
            sigma=np.array([1., 1., 1.e-300]), absolute_sigma=True, return_diagnostics=True,
        )
        self.assertEqual(diagnostics['successful_fits'], 1)
        self.assertEqual(diagnostics['fitting_failure_count'], 0)
        summary = diagnostics['observation_error']
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['successful_splits'], 0)
        self.assertTrue(np.isnan(summary['mean_negative_log_likelihood']))


if __name__ == '__main__':
    unittest.main()
