"""App-facing coefficient/tau interval conventions consume the core fit scale."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import norm, t

from ivivc_core import ObservationUncertainty, fit_profile


def _fit(config, *, weighted=True, model_name='exponential_normalized'):
    time = np.linspace(0.0, 5.0, 8)
    response = np.exp(-0.3 * time) + np.array([0.0, 0.008, -0.01, 0.004, -0.005, 0.006, -0.003, 0.004])
    if model_name == 'linear':
        response = 1.0 + 2.0 * time + np.array([0.0, 0.008, -0.01, 0.004, -0.005, 0.006, -0.003, 0.004])
    return fit_profile(
        time, response, model_name=model_name, selected_metrics=['rmse'],
        analysis_config=config, calculate_tau=model_name != 'linear',
        observation_uncertainty=ObservationUncertainty(0.04, response_kind='mean') if weighted else None,
    )


def test_weighted_app_parameter_and_tau_tables_use_absolute_covariance_multiplier(fast_analysis_config):
    from engine import _profile_fit_to_legacy_components

    fit = _fit(fast_analysis_config)
    result = _profile_fit_to_legacy_components(fit, include_tau=True)
    multiplier = norm.ppf(0.975)
    assert result['fit_metadata']['ci_multiplier_95'] == multiplier
    assert result['fit_metadata']['parameter_interval_method'] == 'normal_wald'
    expected = multiplier * np.sqrt(np.diag(fit.covariance))[0]
    assert result['parameter_intervals'][0]['half_width'] == f'{expected:.4f}'
    assert result['tau_interval']['half_width'] == f'{multiplier * fit.tau_sd:.4f}'
    assert np.isnan(result['evidence_data']['rss'])
    assert result['fit_metadata']['information_criteria_available'] is False


def test_legacy_app_interval_metadata_retains_its_existing_schema(fast_analysis_config):
    from engine import _profile_fit_to_legacy_components

    fit = _fit(fast_analysis_config, weighted=False)
    result = _profile_fit_to_legacy_components(fit, include_tau=True)
    multiplier = t.ppf(0.975, fit.residual_dof)
    assert result['fit_metadata'] == {
        'n_observations': len(fit.response), 'n_parameters': fit.n_parameters,
        'residual_df': fit.residual_dof, 'ci_multiplier_95': multiplier,
    }
    assert result['evidence_data'] == {
        'n': len(fit.response), 'rss': float(np.sum(fit.residuals ** 2)),
        'n_params': fit.n_parameters,
    }


def test_weighted_invalid_joint_covariance_keeps_estimates_but_withholds_table_intervals(fast_analysis_config):
    from engine import _profile_fit_to_legacy_components

    fit = _fit(fast_analysis_config, model_name='linear')
    invalid = replace(fit, covariance=np.array([[1.0, 2.0], [2.0, 1.0]]))
    result = _profile_fit_to_legacy_components(invalid)
    for row in result['parameter_intervals']:
        assert row['estimate'] != 'N/A'
        assert row['is_available'] is False
        assert row['half_width'] == 'N/A'
    assert result['fit_metadata']['parameter_interval_covariance_status'] != 'available'
    assert 'Parameter uncertainty unavailable' in result['parameter_diagnostics_html']
    assert 'not a valid joint uncertainty matrix' in result['parameter_diagnostics_html']
