"""Focused checks for supplied-error fit metrics and conventional AICc."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from ivivc_core import ObservationUncertainty, fit_ivivc, fit_profile, prepare_data
from metrics.likelihood import gaussian_information_criteria
from utilities.model_evidence import build_relative_evidence_tables


def test_supplied_error_criteria_match_the_normal_likelihood_and_conventional_aicc():
    residuals = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.15])
    sigma = np.array([0.2, 0.3, 0.1, 0.5, 0.2, 0.4])
    result = gaussian_information_criteria(residuals, sigma, 2)
    log_likelihood = np.sum(norm.logpdf(residuals, loc=0., scale=sigma))
    aic = -2. * log_likelihood + 2. * 2
    np.testing.assert_allclose(result['log_likelihood'], log_likelihood)
    np.testing.assert_allclose(result['aic'], aic)
    np.testing.assert_allclose(result['bic'], -2. * log_likelihood + 2 * np.log(6))
    np.testing.assert_allclose(result['aicc'], aic + 2. * 2 * 3 / (6 - 2 - 1))
    assert result['aicc_defined']


@pytest.mark.parametrize('n,k,defined', [(2, 1, False), (3, 1, True), (3, 2, False), (4, 2, True)])
def test_aicc_retains_its_existing_sample_size_threshold(n, k, defined):
    result = gaussian_information_criteria(np.zeros(n), 0.2, k)
    assert np.isfinite(result['aic']) and np.isfinite(result['bic'])
    assert result['aicc_defined'] is defined
    assert np.isfinite(result['aicc']) == defined


@pytest.mark.parametrize('sigma', [1.e-200, 0.1, 1.e200])
def test_perfect_fits_and_extreme_errors_need_no_rss_floor_or_sigma_squared(sigma):
    result = gaussian_information_criteria(np.zeros(4), sigma, 1)
    expected = np.sum(norm.logpdf(np.zeros(4), scale=sigma))
    np.testing.assert_allclose(result['log_likelihood'], expected)
    assert np.all(np.isfinite([result['aic'], result['aicc'], result['bic']]))


def test_nonfinite_residual_is_not_removed_to_make_information_criteria_finite():
    result = gaussian_information_criteria([0.1, np.inf, 0.2], 0.1, 1)
    assert result['n'] == 3
    for criterion in ('aic', 'aicc', 'bic'):
        assert np.isnan(result[criterion])


def test_weighted_profile_retains_raw_metrics_and_records_the_resolved_error_metrics(fast_analysis_config):
    time = np.arange(1., 7.)
    response = np.array([1.1, 2., 2.8, 4.2, 5.1, 6.])
    sigma = np.array([0.1, 0.2, 0.1, 0.4, 0.2, 0.3])
    fit = fit_profile(
        time, response, model_name='linear_zero_intercept',
        selected_metrics=['rmse', 'aic', 'aicc', 'bic'], analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty(sigma, response_kind='mean'),
    )
    np.testing.assert_allclose(fit.fit_diagnostics['standardized_residuals'], fit.residuals / sigma)
    np.testing.assert_allclose(fit.fit_metrics['rmse'], np.sqrt(np.mean(fit.residuals ** 2)))
    expected = gaussian_information_criteria(fit.residuals, sigma, 1)
    for criterion in ('aic', 'aicc', 'bic'):
        np.testing.assert_allclose(fit.fit_metrics[criterion], expected[criterion])
        assert np.isnan(fit.cv_means[criterion])
    assert fit.fit_diagnostics['unavailable_metrics'] == []


def test_sem_counts_and_constant_vs_column_errors_do_not_change_information_criteria(fast_analysis_config):
    time = np.arange(1., 6.)
    response = np.array([1.1, 2., 2.8, 4.2, 5.1])
    specifications = [ObservationUncertainty(0.2, response_kind='individual'),
                      ObservationUncertainty([0.2] * 5, response_kind='mean'),
                      ObservationUncertainty(0.2, response_kind='mean', replicate_counts=[2, 4, 8, 16, 32])]
    fits = [fit_profile(time, response, model_name='linear_zero_intercept',
                       selected_metrics=['aic', 'aicc', 'bic'], analysis_config=fast_analysis_config,
                       observation_uncertainty=uncertainty) for uncertainty in specifications]
    for fit in fits[1:]:
        assert fit.fit_diagnostics['information_criteria']['n'] == 5
        assert fit.fit_diagnostics['information_criteria']['k'] == 1
        for criterion in ('aic', 'aicc', 'bic'):
            np.testing.assert_array_equal(fit.fit_metrics[criterion], fits[0].fit_metrics[criterion])


@pytest.mark.parametrize('vitro,vivo', [(True, True), (True, False), (False, True)])
def test_actual_calibrations_keep_combined_aicc_with_both_or_one_uncertain_source(fast_analysis_config, vitro, vivo):
    time = np.arange(6.)
    response = np.exp(-0.2 * time) + np.array([0., 0.002, -0.001, 0.001, -0.002, 0.001])
    uncertainty = ObservationUncertainty(0.04, response_kind='mean')
    prepared = prepare_data(time, response, time * 2., response,
                            in_vitro_uncertainty=uncertainty if vitro else None,
                            in_vivo_uncertainty=uncertainty if vivo else None)
    analysis = fit_ivivc(prepared,
        selections=['approach2:exponential_normalized', 'approach2:exponential'],
        selected_metrics=['aic', 'aicc', 'bic'], analysis_config=fast_analysis_config)
    assert not analysis.errors
    results = {}
    for key, fitted in analysis.model_results.items():
        sources = {}
        for name, fit in (('in_vitro', fitted.in_vitro_fit), ('in_vivo', fitted.in_vivo_fit)):
            source = dict(n=len(fit.response), n_params=fit.n_parameters,
                          rss=float(np.sum(fit.residuals ** 2)))
            if fit.observation_uncertainty is not None:
                source.update(rss=np.nan, information_criteria=fit.fit_diagnostics['information_criteria'])
            sources[name] = source
        results[key] = {'evidence_data': sources}
    tables, _ = build_relative_evidence_tables(results, {}, ['approach2'], ['aic', 'aicc', 'bic'])
    rows = tables['approach2']['combined']['rows']
    assert len(rows) == 2
    for row in rows:
        fitted = analysis[row['model_key']]
        for criterion in ('aic', 'aicc', 'bic'):
            expected = fitted.in_vitro_fit.fit_metrics[criterion] + fitted.in_vivo_fit.fit_metrics[criterion]
            np.testing.assert_allclose(row[criterion], expected)
            assert np.isfinite(row[criterion + '_weight'])
    for criterion in ('aic', 'aicc', 'bic'):
        np.testing.assert_allclose(sum(row[criterion + '_weight'] for row in rows), 1.)
