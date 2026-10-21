"""Reusable numerical prediction methods for the current IVIVC approaches.

The functions in this module return structured numerical results only. Plotly
figures, HTML tables, and mutation of the legacy fit dictionaries remain in the
``engine.py`` compatibility layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy as sp

from utilities.prediction_validity import describe_tau_prediction_skip_reason

from .registry import get_model_spec
from .types import PredictedTrajectory, TimeCourse


def safe_ratio(numerator, denominator):
    """Return an elementwise ratio with zero-denominator values as NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def inverse_response_times(
    model_function,
    params,
    target_responses,
    time_range,
    observed_responses=None,
    grid_points=1001,
):
    """Map response values to unique fitted times over a calibration range."""
    target_responses = np.asarray(target_responses, dtype=float)
    result = np.full(target_responses.shape, np.nan, dtype=float)
    status = np.full(target_responses.shape, 'unavailable', dtype=object)

    t_min, t_max = [float(value) for value in time_range]
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return result, status, 'Observed-response inverse prediction requires at least two distinct in vivo calibration times.'

    t_grid = np.linspace(t_min, t_max, max(int(grid_points), 101))
    try:
        y_grid = np.asarray(model_function(t_grid, **params), dtype=float)
    except Exception as exc:
        return result, status, f'Observed-response inverse prediction was not performed because the fitted in vivo curve could not be evaluated: {exc}'

    finite = np.isfinite(t_grid) & np.isfinite(y_grid)
    if np.count_nonzero(finite) < 2:
        return result, status, 'Observed-response inverse prediction was not performed because the fitted in vivo curve is not finite over the calibration range.'
    t_grid = t_grid[finite]
    y_grid = y_grid[finite]

    y_span = float(np.max(y_grid) - np.min(y_grid))
    monotonic_tol = max(1e-10, y_span * 1e-8)
    dy = np.diff(y_grid)
    nondecreasing = bool(np.all(dy >= -monotonic_tol))
    nonincreasing = bool(np.all(dy <= monotonic_tol))
    if not (nondecreasing or nonincreasing) or y_span <= monotonic_tol:
        return result, status, 'Observed-response inverse prediction was not performed because the fitted in vivo curve is not one-to-one over the observed in vivo time range.'

    y_min = float(np.min(y_grid))
    y_max = float(np.max(y_grid))
    response_tol = max(1e-10, y_span * 1e-8)

    observed = np.asarray(observed_responses, dtype=float) if observed_responses is not None else y_grid
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        observed = y_grid
    observed_min = float(np.min(observed))
    observed_max = float(np.max(observed))
    observed_span = float(observed_max - observed_min)
    observed_tol = max(1e-10, observed_span * 1e-8)

    high_response_time = t_grid[0] if y_grid[0] >= y_grid[-1] else t_grid[-1]
    low_response_time = t_grid[-1] if y_grid[0] >= y_grid[-1] else t_grid[0]

    def root_function(time_value, target):
        value = np.asarray(
            model_function(np.asarray([time_value], dtype=float), **params),
            dtype=float,
        ).reshape(-1)[0]
        return float(value - target)

    for index, target in np.ndenumerate(target_responses):
        if not np.isfinite(target):
            status[index] = 'nonfinite input response'
            continue
        if target < observed_min - observed_tol or target > observed_max + observed_tol:
            status[index] = 'outside observed in vivo response range'
            continue
        if target > y_max + response_tol:
            result[index] = high_response_time
            status[index] = 'mapped to calibration endpoint'
            continue
        if target < y_min - response_tol:
            result[index] = low_response_time
            status[index] = 'mapped to calibration endpoint'
            continue
        if np.isclose(target, y_grid[0], atol=response_tol, rtol=0):
            result[index] = t_grid[0]
            status[index] = 'mapped'
            continue
        if np.isclose(target, y_grid[-1], atol=response_tol, rtol=0):
            result[index] = t_grid[-1]
            status[index] = 'mapped'
            continue
        try:
            result[index] = sp.optimize.brentq(
                root_function,
                t_grid[0],
                t_grid[-1],
                args=(float(target),),
                xtol=1e-10,
                rtol=1e-10,
                maxiter=200,
            )
            status[index] = 'mapped'
        except Exception:
            status[index] = 'no unique root in calibration range'

    return result, status, None


def inverse_response_prediction_intervals(
    model_function,
    params,
    pcov,
    residuals,
    target_responses,
    time_range,
    observed_responses=None,
    confidence_level=0.95,
    n_samples=1000,
    grid_points=501,
    random_state=None,
):
    """Approximate inverse-prediction intervals by Monte Carlo propagation."""
    targets = np.asarray(target_responses, dtype=float)
    lower = np.full(targets.shape, np.nan, dtype=float)
    upper = np.full(targets.shape, np.nan, dtype=float)
    valid_counts = np.zeros(targets.shape, dtype=int)

    if pcov is None:
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': np.nan,
        }, 'Approximate inverse-prediction intervals were unavailable because the fitted in vivo parameter covariance matrix was unavailable.'

    param_names = list(params.keys())
    param_values = np.asarray([params[name] for name in param_names], dtype=float)
    covariance = np.asarray(pcov, dtype=float)
    if covariance.shape != (len(param_values), len(param_values)) or not np.all(np.isfinite(covariance)):
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': np.nan,
        }, 'Approximate inverse-prediction intervals were unavailable because the fitted in vivo parameter covariance matrix was invalid.'

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    dof = int(len(residuals) - len(param_values))
    if dof <= 0:
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': np.nan,
        }, 'Approximate inverse-prediction intervals were unavailable because the fitted in vivo model had no positive residual degrees of freedom.'

    residual_std = float(np.sqrt(np.sum(residuals ** 2) / dof))
    if not np.isfinite(residual_std):
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': residual_std,
        }, 'Approximate inverse-prediction intervals were unavailable because the in vivo residual standard error could not be calculated.'

    t_min, t_max = [float(value) for value in time_range]
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': residual_std,
        }, 'Approximate inverse-prediction intervals require at least two distinct in vivo calibration times.'

    observed = np.asarray(observed_responses, dtype=float) if observed_responses is not None else np.asarray([], dtype=float)
    observed = observed[np.isfinite(observed)]
    observed_min = float(np.min(observed)) if observed.size else -np.inf
    observed_max = float(np.max(observed)) if observed.size else np.inf
    observed_span = float(observed_max - observed_min) if observed.size else 0.0
    observed_tol = max(1e-10, observed_span * 1e-8) if observed.size else 0.0

    from .uncertainty import sample_parameter_sets

    rng = np.random.default_rng(random_state)
    try:
        sampled_parameters = sample_parameter_sets(
            params,
            covariance,
            n_samples=int(n_samples),
            random_state=rng,
        )
    except Exception as exc:
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': residual_std,
        }, f'Approximate inverse-prediction intervals were unavailable because parameter sampling failed: {exc}'

    parameter_samples = sampled_parameters.values

    t_grid = np.linspace(t_min, t_max, max(int(grid_points), 101))
    try:
        reference_y_grid = np.asarray(model_function(t_grid, **params), dtype=float)
    except Exception:
        reference_y_grid = np.full(t_grid.shape, np.nan, dtype=float)
    if reference_y_grid.shape != t_grid.shape or not np.all(np.isfinite(reference_y_grid)):
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': residual_std,
        }, 'Approximate inverse-prediction intervals were unavailable because the fitted in vivo curve could not be evaluated over the calibration range.'
    reference_span = float(np.max(reference_y_grid) - np.min(reference_y_grid))
    reference_tol = max(1e-10, reference_span * 1e-8)
    reference_dy = np.diff(reference_y_grid)
    reference_nondecreasing = bool(np.all(reference_dy >= -reference_tol))
    reference_nonincreasing = bool(np.all(reference_dy <= reference_tol))
    if not (reference_nondecreasing or reference_nonincreasing) or reference_span <= reference_tol:
        return {
            'lower': lower,
            'upper': upper,
            'valid_counts': valid_counts,
            'n_samples': int(n_samples),
            'residual_std': residual_std,
        }, 'Approximate inverse-prediction intervals were unavailable because the fitted in vivo curve is not one-to-one over the calibration range.'

    time_samples = np.full((len(parameter_samples), targets.size), np.nan, dtype=float)
    response_samples = targets.reshape(1, -1) + rng.normal(
        loc=0.0,
        scale=residual_std,
        size=(len(parameter_samples), targets.size),
    )

    for sample_index, sample in enumerate(parameter_samples):
        sample_params = dict(zip(param_names, sample))
        try:
            y_grid = np.asarray(model_function(t_grid, **sample_params), dtype=float)
        except Exception:
            continue
        if y_grid.shape != t_grid.shape or not np.all(np.isfinite(y_grid)):
            continue

        y_span = float(np.max(y_grid) - np.min(y_grid))
        monotonic_tol = max(1e-10, y_span * 1e-8)
        dy = np.diff(y_grid)
        nondecreasing = bool(np.all(dy >= -monotonic_tol))
        nonincreasing = bool(np.all(dy <= monotonic_tol))
        if not (nondecreasing or nonincreasing) or y_span <= monotonic_tol:
            continue
        if reference_nondecreasing and not nondecreasing:
            continue
        if reference_nonincreasing and not nonincreasing:
            continue

        if nondecreasing:
            response_grid = y_grid
            time_for_response = t_grid
        else:
            response_grid = y_grid[::-1]
            time_for_response = t_grid[::-1]

        sampled_targets = response_samples[sample_index]
        observed_supported = (
            np.isfinite(sampled_targets)
            & (sampled_targets >= observed_min - observed_tol)
            & (sampled_targets <= observed_max + observed_tol)
        )
        if not np.any(observed_supported):
            continue
        clipped_targets = np.clip(
            sampled_targets[observed_supported],
            response_grid[0],
            response_grid[-1],
        )
        time_samples[sample_index, observed_supported] = np.interp(
            clipped_targets,
            response_grid,
            time_for_response,
        )

    alpha = 1.0 - float(confidence_level)
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    min_valid_samples = max(50, int(0.10 * int(n_samples)))

    for flat_index in range(targets.size):
        samples = time_samples[:, flat_index]
        samples = samples[np.isfinite(samples)]
        valid_counts.flat[flat_index] = len(samples)
        if len(samples) < min_valid_samples:
            continue
        lower.flat[flat_index], upper.flat[flat_index] = np.percentile(
            samples,
            [lower_percentile, upper_percentile],
        )

    result = {
        'lower': lower,
        'upper': upper,
        'valid_counts': valid_counts,
        'n_samples': int(n_samples),
        'residual_std': residual_std,
    }

    mapped_target_count = int(np.count_nonzero(np.isfinite(targets)))
    interval_count = int(np.count_nonzero(np.isfinite(lower) & np.isfinite(upper)))
    if interval_count == 0:
        return result, (
            'Approximate inverse-prediction intervals could not be estimated reliably for any mapped response. '
            'Many Monte Carlo draws may have fallen outside the invertible calibration range or produced non-invertible sampled curves.'
        )
    if interval_count < mapped_target_count:
        return result, (
            f'Approximate 95% inverse-prediction intervals were available for {interval_count} of {mapped_target_count} mapped responses; '
            'other intervals had too few valid Monte Carlo inverse solutions.'
        )
    return result, None


def predict_time_mapping(
    prediction: TimeCourse,
    *,
    model_name: str,
    parameters: Mapping[str, float],
) -> PredictedTrajectory:
    """Predict in vivo times from in vitro times using an Approach 1 fit."""
    model_function = get_model_spec(model_name).model_function
    in_vivo_time = np.asarray(model_function(prediction.time, **parameters))
    return PredictedTrajectory(
        source_approach='approach1',
        source_method='time_mapping',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
    )


def predict_value_ratio(
    prediction: TimeCourse,
    *,
    model_name: str,
    in_vitro_parameters: Mapping[str, float],
    in_vivo_parameters: Mapping[str, float],
    minimum_plot_points: int = 50,
) -> PredictedTrajectory:
    """Predict in vivo responses by the fitted response-value ratio."""
    model_function = get_model_spec(model_name).model_function
    in_vitro_model_value = np.asarray(
        model_function(prediction.time, **in_vitro_parameters)
    )
    in_vivo_model_value = np.asarray(
        model_function(prediction.time, **in_vivo_parameters)
    )
    ratio = in_vivo_model_value / in_vitro_model_value
    in_vivo_value = prediction.value * ratio

    plot_time = np.linspace(
        min(prediction.time),
        max(prediction.time),
        max(len(prediction.time), int(minimum_plot_points)),
    )
    in_vitro_model_plot = np.asarray(
        model_function(plot_time, **in_vitro_parameters)
    )
    in_vivo_model_plot = np.asarray(
        model_function(plot_time, **in_vivo_parameters)
    )

    return PredictedTrajectory(
        source_approach='approach2',
        source_method='value_ratio',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=prediction.time,
        in_vivo_value=in_vivo_value,
        scaling_factor=ratio,
        auxiliary={
            'in_vitro_model_value': in_vitro_model_value,
            'in_vivo_model_value': in_vivo_model_value,
            'plot_time': plot_time,
            'in_vitro_model_plot': in_vitro_model_plot,
            'in_vivo_model_plot': in_vivo_model_plot,
        },
    )


def predict_tau_ratio(
    prediction: TimeCourse,
    *,
    model_name: str,
    tau_values: Sequence[Any] | None,
    residual_dof: Sequence[int | None] = (),
    confidence_level: float = 0.95,
) -> PredictedTrajectory:
    """Predict in vivo times using the ratio of fitted time constants."""
    skip_reason = describe_tau_prediction_skip_reason(tau_values)
    if skip_reason is not None:
        return PredictedTrajectory(
            source_approach='approach2',
            source_method='tau_ratio',
            model_name=model_name,
            in_vitro_time=prediction.time,
            in_vitro_value=prediction.value,
            in_vivo_time=np.full(prediction.time.shape, np.nan, dtype=float),
            in_vivo_value=prediction.value,
            available=False,
            skip_reason=skip_reason,
        )

    tau1, tau2 = tau_values
    time_ratio = tau2 / tau1
    in_vivo_time_uncertain = prediction.time * time_ratio
    in_vivo_time = np.array([value.n for value in in_vivo_time_uncertain])
    in_vivo_time_sd = np.array([value.s for value in in_vivo_time_uncertain])

    valid_dof = [
        int(value)
        for value in residual_dof
        if value is not None and int(value) > 0
    ]
    dof = min(valid_dof) if valid_dof else 0
    alpha = 1.0 - float(confidence_level)
    t_value = sp.stats.t.ppf(1.0 - alpha / 2.0, dof) if dof > 0 else 1.96
    in_vivo_time_error = t_value * in_vivo_time_sd

    return PredictedTrajectory(
        source_approach='approach2',
        source_method='tau_ratio',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
        time_error_lower=in_vivo_time_error,
        time_error_upper=in_vivo_time_error,
        scaling_factor=np.full(prediction.time.shape, float(time_ratio.n)),
        auxiliary={
            'time_ratio': time_ratio,
            'time_standard_uncertainty': in_vivo_time_sd,
            'residual_dof': dof,
            'critical_value': t_value,
        },
    )


def predict_inverse_response(
    prediction: TimeCourse,
    *,
    model_name: str,
    parameters: Mapping[str, float],
    covariance,
    residuals,
    calibration_time_range,
    observed_responses=None,
    confidence_level: float = 0.95,
    n_samples: int = 1000,
    inverse_mapper: Callable[..., tuple[np.ndarray, np.ndarray, str | None]] = inverse_response_times,
    interval_estimator: Callable[..., tuple[dict[str, Any], str | None]] = inverse_response_prediction_intervals,
) -> PredictedTrajectory:
    """Map supplied responses to fitted in vivo times and optional intervals."""
    model_function = get_model_spec(model_name).model_function
    in_vivo_time, status, skip_reason = inverse_mapper(
        model_function,
        parameters,
        prediction.value,
        calibration_time_range,
        observed_responses=observed_responses,
    )
    in_vivo_time = np.asarray(in_vivo_time, dtype=float)
    valid_inverse = np.isfinite(in_vivo_time)

    interval_result = {
        'lower': np.full(prediction.value.shape, np.nan, dtype=float),
        'upper': np.full(prediction.value.shape, np.nan, dtype=float),
        'valid_counts': np.zeros(prediction.value.shape, dtype=int),
        'n_samples': int(n_samples),
        'residual_std': np.nan,
    }
    interval_note = None
    interval_displayable = np.zeros(prediction.value.shape, dtype=bool)
    time_error_lower = np.full(prediction.value.shape, np.nan, dtype=float)
    time_error_upper = np.full(prediction.value.shape, np.nan, dtype=float)
    notes: list[str] = []

    if skip_reason is None and np.any(valid_inverse):
        interval_result, interval_note = interval_estimator(
            model_function,
            parameters,
            covariance,
            residuals,
            np.where(valid_inverse, prediction.value, np.nan),
            calibration_time_range,
            observed_responses=observed_responses,
            confidence_level=confidence_level,
            n_samples=n_samples,
        )

        lower = np.asarray(interval_result['lower'], dtype=float)
        upper = np.asarray(interval_result['upper'], dtype=float)
        interval_displayable = (
            valid_inverse
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (lower <= in_vivo_time)
            & (upper >= in_vivo_time)
        )
        time_error_lower[interval_displayable] = (
            in_vivo_time[interval_displayable] - lower[interval_displayable]
        )
        time_error_upper[interval_displayable] = (
            upper[interval_displayable] - in_vivo_time[interval_displayable]
        )

        unavailable_count = int(np.size(valid_inverse) - np.count_nonzero(valid_inverse))
        if unavailable_count:
            notes.append(
                f'{unavailable_count} of {np.size(valid_inverse)} prediction response values could not be mapped '
                'because they were outside the observed in vivo calibration response range.'
            )
        if interval_note:
            notes.append(interval_note)
        elif np.count_nonzero(interval_displayable) < np.count_nonzero(valid_inverse):
            notes.append(
                'Some approximate 95% inverse-prediction intervals were not displayed because the percentile interval '
                'did not contain the point estimate or too few valid Monte Carlo inverse solutions were available.'
            )
    elif skip_reason is None:
        skip_reason = (
            'Observed-response inverse prediction was not performed because none of the prediction response values '
            'fall within the observed in vivo calibration response range.'
        )

    return PredictedTrajectory(
        source_approach='approach2',
        source_method='observed_response_inverse',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
        available=bool(skip_reason is None and np.any(valid_inverse)),
        skip_reason=skip_reason,
        note=' '.join(notes) if notes else None,
        status=status,
        time_lower=interval_result['lower'],
        time_upper=interval_result['upper'],
        time_error_lower=time_error_lower,
        time_error_upper=time_error_upper,
        auxiliary={
            'interval_valid_samples': np.asarray(interval_result['valid_counts']),
            'interval_total_samples': interval_result['n_samples'],
            'assumed_response_sd': interval_result['residual_std'],
            'interval_displayable': interval_displayable,
            'interval_note': interval_note,
        },
    )


def predict_direct_value_scaling(
    prediction: TimeCourse,
    *,
    time_grid,
    in_vitro_response_on_time_grid,
    in_vivo_response_on_time_grid,
) -> PredictedTrajectory:
    """Predict in vivo responses using the Approach 3 response ratio."""
    mapping_ratio = safe_ratio(
        in_vivo_response_on_time_grid,
        in_vitro_response_on_time_grid,
    )
    scaling_factor = np.interp(
        prediction.time,
        time_grid,
        mapping_ratio,
        left=np.nan,
        right=np.nan,
    )
    in_vivo_value = prediction.value * scaling_factor
    return PredictedTrajectory(
        source_approach='approach3',
        source_method='value_scaling',
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=prediction.time,
        in_vivo_value=in_vivo_value,
        scaling_factor=scaling_factor,
    )


def predict_direct_time_scaling(
    prediction: TimeCourse,
    *,
    response_grid,
    in_vitro_time_at_response,
    in_vivo_time_at_response,
) -> PredictedTrajectory:
    """Predict in vivo times using the Approach 3 time ratio."""
    order = np.argsort(response_grid)
    mapping_ratio = safe_ratio(in_vivo_time_at_response, in_vitro_time_at_response)
    scaling_factor = np.interp(
        prediction.value,
        np.asarray(response_grid)[order],
        mapping_ratio[order],
        left=np.nan,
        right=np.nan,
    )
    in_vivo_time = prediction.time * scaling_factor
    in_vivo_time[np.isfinite(prediction.time) & (prediction.time == 0)] = 0.0
    return PredictedTrajectory(
        source_approach='approach3',
        source_method='time_scaling',
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
        scaling_factor=scaling_factor,
    )
