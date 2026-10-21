"""Lightweight nonlinear fitting for repeated study-design simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np

from ivivc_core import get_model_spec
from utilities.evaluation import auto_grid_search

from .types import ModelFitResult


_DEFAULT_GRID_SEARCH = {
    'param_min': -1e6,
    'param_max': 1e6,
    'num_points': 50,
    'num_cores': 1,
    'random_state': 12345,
}


def _coerce_initial_parameters(
    initial_parameters: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    parameter_names: tuple[str, ...],
) -> np.ndarray | None:
    if initial_parameters is None:
        return None
    if isinstance(initial_parameters, Mapping):
        missing = [name for name in parameter_names if name not in initial_parameters]
        if missing:
            raise ValueError(
                'Initial parameter mapping is missing: ' + ', '.join(missing)
            )
        values = np.asarray(
            [initial_parameters[name] for name in parameter_names],
            dtype=float,
        )
    else:
        values = np.asarray(initial_parameters, dtype=float)
    if values.ndim != 1 or len(values) != len(parameter_names):
        raise ValueError('Initial parameters must contain one value per model parameter.')
    if np.any(~np.isfinite(values)):
        raise ValueError('Initial parameters must be finite.')
    return values.copy()


def _model_initial_candidates(spec, time: np.ndarray, response: np.ndarray) -> np.ndarray | None:
    if spec.initial_guess is None:
        return None
    candidates = np.atleast_2d(np.asarray(spec.initial_guess(time, response), dtype=float))
    if candidates.shape[1] != len(spec.parameter_names):
        raise ValueError('Model-specific initial points have the wrong number of parameters.')
    candidates = candidates[np.all(np.isfinite(candidates), axis=1)]
    return candidates if len(candidates) else None


def _best_candidate(
    model_function,
    time: np.ndarray,
    response: np.ndarray,
    candidates: np.ndarray | None,
) -> np.ndarray | None:
    if candidates is None or len(candidates) == 0:
        return None
    best = None
    best_error = np.inf
    for candidate in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                predicted = np.asarray(model_function(time, *candidate), dtype=float)
            if predicted.shape != response.shape or np.any(~np.isfinite(predicted)):
                continue
            error = float(np.sum((response - predicted) ** 2))
        except Exception:
            continue
        if error < best_error:
            best_error = error
            best = np.asarray(candidate, dtype=float)
    return None if best is None else best.copy()


def _grid_search_start(
    spec,
    time: np.ndarray,
    response: np.ndarray,
    *,
    initial_parameters: np.ndarray | None,
    model_candidates: np.ndarray | None,
    grid_search_config: Mapping[str, object] | None,
) -> np.ndarray:
    config = dict(_DEFAULT_GRID_SEARCH)
    if grid_search_config:
        config.update(grid_search_config)

    initial_points = model_candidates
    if initial_parameters is not None:
        initial_row = initial_parameters.reshape(1, -1)
        initial_points = (
            initial_row
            if initial_points is None
            else np.vstack([initial_row, initial_points])
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.asarray(
            auto_grid_search(
                spec.model_function,
                time,
                response,
                param_min=float(config['param_min']),
                param_max=float(config['param_max']),
                num_points=int(config['num_points']),
                num_cores=config['num_cores'],
                random_state=config['random_state'],
                initial_points=initial_points,
            ),
            dtype=float,
        )


def _failed_result(
    *,
    model_name: str,
    time: np.ndarray,
    response: np.ndarray,
    parameter_names: tuple[str, ...],
    initial_parameters: np.ndarray | None,
    used_grid_search: bool,
    exception: BaseException,
) -> ModelFitResult:
    parameter_count = len(parameter_names)
    nan_values = np.full(len(time), np.nan, dtype=float)
    return ModelFitResult(
        model_name=model_name,
        success=False,
        time=time,
        response=response,
        parameter_names=parameter_names,
        parameters={},
        covariance=np.full((parameter_count, parameter_count), np.nan, dtype=float),
        initial_parameters=initial_parameters,
        fitted_values=nan_values,
        residuals=nan_values,
        residual_dof=int(len(time) - parameter_count),
        residual_sd=np.nan,
        used_grid_search=used_grid_search,
        message=str(exception),
        exception_type=type(exception).__name__,
    )


def fit_model_once(
    time,
    response,
    *,
    model_name: str,
    initial_parameters: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    perform_grid_search: bool = False,
    fallback_grid_search: bool = True,
    grid_search_config: Mapping[str, object] | None = None,
    raise_on_failure: bool = False,
) -> ModelFitResult:
    """Fit one registered model without CV, metrics, plots, or HTML.

    This function is intended for repeated fitting of simulated sparse studies.
    A supplied parameter vector is used directly.  Otherwise, the best
    model-specific candidate is used when available.  If the first fit fails,
    a generic grid search can be attempted once as a fallback.
    """
    spec = get_model_spec(model_name)
    time = np.asarray(time, dtype=float)
    response = np.asarray(response, dtype=float)
    if time.ndim != 1 or response.ndim != 1:
        raise ValueError('Fit time and response arrays must be one-dimensional.')
    if len(time) != len(response):
        raise ValueError('Fit time and response arrays must have matching lengths.')

    finite = np.isfinite(time) & np.isfinite(response)
    time = time[finite]
    response = response[finite]
    if len(time) == 0:
        raise ValueError('At least one finite time-response pair is required.')

    supplied_initial = _coerce_initial_parameters(
        initial_parameters,
        spec.parameter_names,
    )
    try:
        model_candidates = _model_initial_candidates(spec, time, response)
    except Exception:
        # A model-specific starting heuristic is advisory. Sparse simulated
        # data should still be allowed to reach the registered fitter or the
        # generic fallback search.
        model_candidates = None
    start = supplied_initial
    if start is None:
        start = _best_candidate(
            spec.model_function,
            time,
            response,
            model_candidates,
        )

    used_grid_search = bool(perform_grid_search)
    if perform_grid_search:
        try:
            start = _grid_search_start(
                spec,
                time,
                response,
                initial_parameters=supplied_initial,
                model_candidates=model_candidates,
                grid_search_config=grid_search_config,
            )
        except Exception as exception:
            if raise_on_failure:
                raise
            return _failed_result(
                model_name=model_name,
                time=time,
                response=response,
                parameter_names=spec.parameter_names,
                initial_parameters=start,
                used_grid_search=True,
                exception=exception,
            )

    def run_fit(p0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            result = spec.fit_function(time, response, p0=p0)
        parameters = dict(result['params'])
        parameter_vector = np.asarray(
            [parameters[name] for name in spec.parameter_names],
            dtype=float,
        )
        if np.any(~np.isfinite(parameter_vector)):
            raise ValueError('The fitted parameter vector contains nonfinite values.')
        fitted_values = np.asarray(result['predict'](time), dtype=float)
        if fitted_values.shape != response.shape or np.any(~np.isfinite(fitted_values)):
            raise ValueError('The fitted model returned invalid values.')
        covariance = np.asarray(result['pcov'], dtype=float)
        expected_shape = (len(spec.parameter_names), len(spec.parameter_names))
        if covariance.shape != expected_shape:
            raise ValueError('The fitted covariance matrix has an unexpected shape.')
        return parameters, covariance, fitted_values

    try:
        parameters, covariance, fitted_values = run_fit(start)
    except Exception as first_exception:
        if fallback_grid_search and not perform_grid_search:
            try:
                start = _grid_search_start(
                    spec,
                    time,
                    response,
                    initial_parameters=supplied_initial,
                    model_candidates=model_candidates,
                    grid_search_config=grid_search_config,
                )
                used_grid_search = True
                parameters, covariance, fitted_values = run_fit(start)
            except Exception as final_exception:
                if raise_on_failure:
                    raise final_exception from first_exception
                return _failed_result(
                    model_name=model_name,
                    time=time,
                    response=response,
                    parameter_names=spec.parameter_names,
                    initial_parameters=start,
                    used_grid_search=used_grid_search,
                    exception=final_exception,
                )
        else:
            if raise_on_failure:
                raise
            return _failed_result(
                model_name=model_name,
                time=time,
                response=response,
                parameter_names=spec.parameter_names,
                initial_parameters=start,
                used_grid_search=used_grid_search,
                exception=first_exception,
            )

    residuals = response - fitted_values
    residual_dof = int(len(response) - len(spec.parameter_names))
    residual_sd = (
        float(np.sqrt(np.sum(residuals ** 2) / residual_dof))
        if residual_dof > 0
        else np.nan
    )
    return ModelFitResult(
        model_name=model_name,
        success=True,
        time=time,
        response=response,
        parameter_names=spec.parameter_names,
        parameters=parameters,
        covariance=covariance,
        initial_parameters=start,
        fitted_values=fitted_values,
        residuals=residuals,
        residual_dof=residual_dof,
        residual_sd=residual_sd,
        used_grid_search=used_grid_search,
    )
