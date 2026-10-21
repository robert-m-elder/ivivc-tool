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
) -> ProfileFit:
    """Fit one response-versus-time profile using the current IVIVC machinery.

    This function contains numerical fitting, cross-validation, metrics, and
    optional time-constant calculation only. It intentionally does not create
    plots, HTML tables, or legacy result dictionaries.

    Missing-value handling mirrors the existing app: paired rows containing a
    NaN in either the time or response are removed, while other nonfinite values
    continue to reach the fitting routines and are handled as before.
    """
    config = _resolved_config(analysis_config, config_defaults)
    spec = get_model_spec(model_name)

    x = np.asarray(time)
    y = np.asarray(response)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Profile time and response arrays must be one-dimensional.')
    if len(x) != len(y):
        raise ValueError('Profile time and response arrays must have the same length.')

    mask = ~np.isnan(x) & ~np.isnan(y)
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
        )

    cv_output = cross_validation_curve_fit(
        x,
        y,
        spec.model_function,
        cv,
        selected_metrics,
        p0=p0,
        kwargs=spec.fit_kwargs,
        return_fit_params=include_cv_fit_variability,
    )
    if include_cv_fit_variability:
        cv_scores, cv_parameter_sets = cv_output
    else:
        cv_scores, cv_parameter_sets = cv_output, []
    cv_means = {
        metric: _finite_cv_mean(values)
        for metric, values in cv_scores.items()
    }

    model_result = spec.fit_function(x, y, p0=p0)
    fitted_values = np.asarray(model_result['predict'](x))
    parameters = dict(model_result['params'])
    covariance = np.asarray(model_result['pcov'])
    residuals = y - fitted_values

    fit_metrics = {
        metric: calculate_metric(metric, y, fitted_values, len(p0))
        for metric in selected_metrics
    }

    tau_estimate = None
    tau_sd = None
    if calculate_tau:
        tau_estimate, tau_sd = calculate_tau_with_uncertainty(
            spec.model_function,
            list(parameters.values()),
            covariance,
        )
        # Preserve the established legacy sentinel for undefined/integral-failure
        # cases so downstream reporting and tau-ratio skip reasons are unchanged.
        if not (tau_estimate > 0 and np.isfinite(tau_estimate)):
            tau_estimate, tau_sd = 0.0, 1.0

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
        cv_parameter_sets=cv_parameter_sets,
        tau_estimate=tau_estimate,
        tau_sd=tau_sd,
    )
