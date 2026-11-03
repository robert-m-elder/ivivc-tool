"""Reusable numerical fitting for one IVIVC time-course profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np

from metrics import calculate_metric
from utilities.evaluation import (
    auto_grid_search,
    calculate_tau_with_uncertainty,
    cross_validation_curve_fit,
    make_cross_validator,
)

from .registry import get_model_spec
from .observations import ObservationUncertainty, resolve_observation_uncertainty
from .types import ProfileFit


_DEFAULT_ANALYSIS_CONFIG = {
    'cv_scheme': 'shuffle_split',
    'cv_n_splits': 20,
    'cv_test_size': 0.25,
    'cv_random_state': 12345,
    'grid_search_num_points': 200,
    'grid_search_num_cores': None,
    'grid_search_param_min': -1e6,
    'grid_search_param_max': 1e6,
    'grid_search_random_state': 12345,
    'include_cv_fit_variability_plot': False,
}


def _resolved_config(
    analysis_config: Mapping[str, object] | None,
    defaults: Mapping[str, object] | None,
) -> dict[str, object]:
    config = dict(_DEFAULT_ANALYSIS_CONFIG)
    if defaults:
        config.update(defaults)
    if analysis_config:
        config.update(analysis_config)
    return config


def _finite_cv_mean(values: np.ndarray) -> float:
    """Mirror the legacy finite-value mean, including NaN for no valid folds."""
    values = np.asarray(values)
    return float(np.nanmean(values[np.isfinite(values)]))


def fit_profile(
    time,
    response,
    *,
    model_name: str,
    selected_metrics: Sequence[str],
    analysis_config: Mapping[str, object] | None = None,
    approach_id: str | None = None,
    calculate_tau: bool = False,
    config_defaults: Mapping[str, object] | None = None,
    observation_uncertainty: ObservationUncertainty | None = None,
) -> ProfileFit:
    """Fit one response-versus-time profile using the current IVIVC machinery.

    This function contains numerical fitting, cross-validation, metrics, and
    optional time-constant calculation only. It intentionally does not create
    plots, HTML tables, or legacy result dictionaries.

    Missing-value handling mirrors the existing app: paired rows containing a
    NaN in either the time or response are removed, while other nonfinite values
    continue to reach the fitting routines and are handled as before.

    Supplied uncertainty is absolute SD for individuals or SEM for means, on
    the same response scale as these arrays. It controls initialization, CV
    training fits, and the final fit. No residual rescaling is applied to its
    covariance. The optional approach identifier must be absent or Approach 2
    when uncertainty is supplied. High-level preprocessing and app integration
    are separate operations; this function does not transform uncertainty.

    Descriptive metrics retain their existing unweighted definitions. The
    legacy RSS-based information criteria are withheld for weighted fits until
    a likelihood-consistent implementation is available.
    """
    config = _resolved_config(analysis_config, config_defaults)
    spec = get_model_spec(model_name)

    x = np.asarray(time)
    y = np.asarray(response)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Profile time and response arrays must be one-dimensional.')
    if len(x) != len(y):
        raise ValueError('Profile time and response arrays must have the same length.')

    if observation_uncertainty is not None and approach_id not in (None, 'approach2'):
        raise ValueError('Supplied experimental uncertainty is supported only for Approach 2 profile fitting.')

    mask = ~np.isnan(x) & ~np.isnan(y)
    resolved_uncertainty = resolve_observation_uncertainty(
        observation_uncertainty, len(y), row_mask=mask,
    )
    weighted = resolved_uncertainty is not None
    sigma = None if resolved_uncertainty is None else resolved_uncertainty.sigma
    unavailable_metrics = [
        metric for metric in selected_metrics
        if weighted and metric in ('aic', 'aicc', 'bic')
    ]
    x = x[mask]
    y = y[mask]

    grid_search_num_points = int(config.get('grid_search_num_points', 200))
    grid_search_num_cores = config.get('grid_search_num_cores', None)
    grid_search_param_min = float(config.get('grid_search_param_min', -1e6))
    grid_search_param_max = float(config.get('grid_search_param_max', 1e6))
    grid_search_random_state = config.get('grid_search_random_state', 12345)
    include_cv_fit_variability = bool(
        config.get('include_cv_fit_variability_plot', False)
    )

    cv = make_cross_validator(config, approach_id)

    initial_points = None
    if spec.initial_guess is not None:
        initial_points = spec.initial_guess(x, y)

    # Keep the no-uncertainty call contract unchanged for existing adapters.
    search_options = {} if not weighted else {'sigma': sigma}
    fit_options = {} if not weighted else {'sigma': sigma, 'absolute_sigma': True}
    if weighted and any(key in spec.fit_kwargs for key in ('sigma', 'absolute_sigma')):
        raise ValueError('Registered fit options must not override the supplied observation uncertainty.')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        p0 = auto_grid_search(
            spec.model_function,
            x,
            y,
            param_min=grid_search_param_min,
            param_max=grid_search_param_max,
            num_points=grid_search_num_points,
            num_cores=grid_search_num_cores,
            random_state=grid_search_random_state,
            initial_points=initial_points,
            **search_options,
        )

    cv_output, cv_diagnostics = cross_validation_curve_fit(
        x,
        y,
        spec.model_function,
        cv,
        selected_metrics,
        p0=p0,
        kwargs=spec.fit_kwargs,
        return_fit_params=include_cv_fit_variability,
        return_diagnostics=True,
        **fit_options,
    )
    if include_cv_fit_variability:
        cv_scores, cv_parameter_sets = cv_output
    else:
        cv_scores, cv_parameter_sets = cv_output, []
    cv_means = {
        metric: np.nan if metric in unavailable_metrics else _finite_cv_mean(values)
        for metric, values in cv_scores.items()
    }

    model_result = spec.fit_function(x, y, p0=p0, **fit_options)
    fitted_values = np.asarray(model_result['predict'](x))
    parameters = dict(model_result['params'])
    covariance = np.asarray(model_result['pcov'])
    residuals = y - fitted_values

    fit_metrics = {
        metric: np.nan if metric in unavailable_metrics else calculate_metric(metric, y, fitted_values, len(p0))
        for metric in selected_metrics
    }

    fit_diagnostics = {
        'objective': 'weighted_least_squares' if weighted else 'unweighted_least_squares',
        'covariance_scaling': 'absolute' if weighted else 'residual_scaled',
        'absolute_sigma': weighted,
        'covariance_finite': bool(np.all(np.isfinite(covariance))),
        'n_observations': int(len(y)),
        'n_parameters': len(parameters),
        'metric_weighting': 'unweighted',
        'unavailable_metrics': unavailable_metrics,
    }
    if weighted:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            chi_squared = float(np.sum((residuals / sigma) ** 2))
        fit_diagnostics.update({
            'chi_squared': chi_squared,
            'reduced_chi_squared': chi_squared / (len(y) - len(parameters))
            if len(y) > len(parameters) else np.nan,
            'information_criteria_note': (
                'AIC, AICc, and BIC are unavailable for supplied-uncertainty fits '
                'until likelihood-aware information criteria are implemented.'
            ),
        })

    tau_estimate = None
    tau_sd = None
    tau_diagnostics = {}
    if calculate_tau:
        (tau_estimate, tau_sd), tau_diagnostics = calculate_tau_with_uncertainty(
            spec.model_function,
            [parameters[name] for name in spec.parameter_names],
            covariance,
            random_state=12345,
            return_diagnostics=True,
        )

    return ProfileFit(
        model_name=model_name,
        time=x,
        response=y,
        parameter_names=spec.parameter_names,
        parameters=parameters,
        covariance=covariance,
        initial_parameters=p0,
        fitted_values=fitted_values,
        residuals=residuals,
        residual_dof=int(len(y) - len(parameters)),
        fit_metrics=fit_metrics,
        cv_scores=cv_scores,
        cv_means=cv_means,
        cv_diagnostics=cv_diagnostics,
        cv_parameter_sets=cv_parameter_sets,
        tau_estimate=tau_estimate,
        tau_sd=tau_sd,
        tau_diagnostics=tau_diagnostics,
        observation_uncertainty=resolved_uncertainty,
        fit_diagnostics=fit_diagnostics,
    )
