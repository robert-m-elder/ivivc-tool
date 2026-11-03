"""Supplied-error criteria feed the existing source-wise evidence tables."""
from __future__ import annotations

import unittest

import numpy as np

from utilities.model_evidence import build_relative_evidence_tables, ic_from_n_rss_k


def supplied(log_likelihood=-2., *, n=6, k=1):
    aic = -2. * log_likelihood + 2. * k
    defined = n > k + 1
    return {
        'n': n, 'n_params': k, 'rss': 10000.,
        'information_criteria': {
            'n': n, 'k': k, 'rss': np.nan, 'log_likelihood': log_likelihood,
            'aic': aic,
            'aicc': aic + 2. * k * (k + 1) / (n - k - 1) if defined else np.nan,
            'bic': -2. * log_likelihood + k * np.log(n), 'aicc_defined': defined,
        },
    }


def panels(candidates):
    results = {'approach2:' + name: {'evidence_data': {'in_vitro': pair[0], 'in_vivo': pair[1]}}
               for name, pair in candidates.items()}
    tables, _ = build_relative_evidence_tables(results, {}, ['approach2'], ['aic', 'aicc', 'bic'])
    return tables['approach2']


class UncertaintyModelEvidenceTests(unittest.TestCase):
    def test_supplied_likelihood_and_conventional_aicc_are_used_instead_of_rss(self):
        residuals, sigma = np.array([0.02, -0.01, 0.04, 0.]), np.array([0.1, 0.2, 0.3, 0.4])
        log_likelihood = -.5 * np.sum((residuals / sigma) ** 2 + np.log(2. * np.pi) + 2. * np.log(sigma))
        evidence = panels({
            'first': (supplied(log_likelihood, n=4), supplied(-2.)),
            'second': (supplied(log_likelihood - 1., n=4), supplied(-3.)),
        })
        first = evidence['in_vitro']['rows'][0]
        self.assertEqual(first['aic'], -2. * log_likelihood + 2.)
        self.assertEqual(first['aicc'], first['aic'] + 2.)
        self.assertEqual(first['bic'], -2. * log_likelihood + np.log(4.))
        self.assertTrue(first['aicc_defined'])
        self.assertTrue(np.isnan(first['rss']))
        self.assertNotIn('Comparison group', evidence['in_vitro']['html'])

    def test_combined_criteria_keep_the_existing_sum_of_independent_sources(self):
        candidates = {
            'first': (supplied(-2., n=4), supplied(-4., n=8, k=2)),
            'second': (supplied(-3., n=4), supplied(-5., n=8, k=2)),
        }
        evidence = panels(candidates)
        for index, pair in enumerate(candidates.values()):
            row = evidence['combined']['rows'][index]
            self.assertEqual(row['n'], 12)
            self.assertEqual(row['k'], 3)
            for criterion in ('aic', 'aicc', 'bic'):
                self.assertAlmostEqual(row[criterion], sum(source['information_criteria'][criterion] for source in pair))
                self.assertTrue(np.isfinite(row[criterion + '_weight']))
        for criterion in ('aic', 'aicc', 'bic'):
            self.assertAlmostEqual(sum(row[criterion + '_weight'] for row in evidence['combined']['rows']), 1.)

    def test_mixed_supplied_and_residual_sources_retain_combined_relative_evidence(self):
        residual_sources = [{'n': 7, 'rss': 0.2, 'n_params': 2}, {'n': 7, 'rss': 0.4, 'n_params': 2}]
        weighted_sources = [supplied(-2.), supplied(-3.)]
        evidence = panels({
            'first': (weighted_sources[0], residual_sources[0]),
            'second': (weighted_sources[1], residual_sources[1]),
        })
        for index, row in enumerate(evidence['combined']['rows']):
            legacy = ic_from_n_rss_k(7, residual_sources[index]['rss'], 2)
            for criterion in ('aic', 'aicc', 'bic'):
                self.assertAlmostEqual(row[criterion], weighted_sources[index]['information_criteria'][criterion] + legacy[criterion])
                self.assertTrue(np.isfinite(row['delta_' + criterion]))
                self.assertTrue(np.isfinite(row[criterion + '_weight']))

    def test_missing_explicit_criteria_never_fall_back_to_the_rss_value(self):
        for payload in (None, {}, []):
            with self.subTest(payload=payload):
                broken = supplied()
                broken['information_criteria'] = payload
                evidence = panels({'broken': (broken, supplied()), 'valid': (supplied(-3.), supplied(-4.))})
                for source in ('in_vitro', 'combined'):
                    row = evidence[source]['rows'][0]
                    for criterion in ('aic', 'aicc', 'bic'):
                        self.assertTrue(np.isnan(row[criterion]))

    def test_aicc_keeps_the_existing_sample_size_cutoff_and_ignores_replicate_metadata(self):
        first = supplied(n=3, k=2)
        first['replicate_counts'] = [100] * 3
        first['information_criteria'].update(aicc=1., aicc_defined=True)
        evidence = panels({'first': (first, supplied()), 'second': (supplied(n=3, k=2), supplied(-3.))})
        row = evidence['in_vitro']['rows'][0]
        self.assertEqual(row['n'], 3)
        self.assertFalse(row['aicc_defined'])
        self.assertTrue(np.isnan(row['aicc']))
        self.assertTrue(np.isfinite(row['aic']))
        self.assertTrue(np.isfinite(row['bic']))

    def test_legacy_evidence_preserves_its_rss_calculations_and_row_shape(self):
        first, second = {'n': 6, 'rss': 0.2, 'n_params': 1}, {'n': 6, 'rss': 0.4, 'n_params': 1}
        evidence = panels({'first': (first, first), 'second': (second, second)})
        row = evidence['in_vitro']['rows'][0]
        expected = ic_from_n_rss_k(6, 0.2, 1)
        for criterion in ('aic', 'aicc', 'bic'):
            self.assertEqual(row[criterion], expected[criterion])
        self.assertNotIn('information_criteria', row)
        self.assertNotIn('comparison_group', row)
        self.assertNotIn('evidence_family', row)
        self.assertNotIn('notes', evidence['in_vitro'])


if __name__ == '__main__':
    unittest.main()
