"""Performance metrics for simulation-based study-design assessment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from ivivc_core import get_model_spec

from .information import finite_difference_sensitivities, trapezoidal_integration_weights


def _coerce_curve_arrays(true_values, predicted_values) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(true_values, dtype=float)
    predicted_array = np.asarray(predicted_values, dtype=float)
    if true_array.shape != predicted_array.shape:
        raise ValueError('True and predicted curve arrays must have matching shapes.')
    if true_array.ndim == 1:
        true_array = true_array.reshape(1, -1)
        predicted_array = predicted_array.reshape(1, -1)
    if true_array.ndim != 2 or true_array.shape[1] < 2:
        raise ValueError('Curve arrays must contain at least two evaluation points.')
    return true_array, predicted_array


def _normalized_weights(point_count: int, *, times=None, weights=None) -> np.ndarray:
    if weights is not None:
        normalized = np.asarray(weights, dtype=float)
        if normalized.ndim != 1 or len(normalized) != point_count:
            raise ValueError('Metric weights must contain one value per evaluation point.')
        if np.any(~np.isfinite(normalized)) or np.any(normalized < 0):
            raise ValueError('Metric weights must be finite and nonnegative.')
        total = float(np.sum(normalized))
        if not total > 0:
            raise ValueError('Metric weights must contain positive total weight.')
        return normalized / total
    if times is None:
        return np.full(point_count, 1.0 / point_count, dtype=float)
    normalized = trapezoidal_integration_weights(times)
    if len(normalized) != point_count:
        raise ValueError('Evaluation times must match the curve length.')
    return normalized


def integrated_root_mean_square_error(
    true_values,
    predicted_values,
    *,
    times=None,
    weights=None,
) -> np.ndarray:
    """Return integrated RMSE for each pair of complete curves."""
    true_array, predicted_array = _coerce_curve_arrays(true_values, predicted_values)
    point_weights = _normalized_weights(
        true_array.shape[1],
        times=times,
        weights=weights,
    )
    finite = np.all(np.isfinite(true_array) & np.isfinite(predicted_array), axis=1)
    result = np.full(true_array.shape[0], np.nan, dtype=float)
    if np.any(finite):
        squared_error = (predicted_array[finite] - true_array[finite]) ** 2
        result[finite] = np.sqrt(np.sum(squared_error * point_weights, axis=1))
    return result


def maximum_absolute_error(true_values, predicted_values) -> np.ndarray:
    """Return maximum absolute error for each pair of complete curves."""
    true_array, predicted_array = _coerce_curve_arrays(true_values, predicted_values)
    finite = np.all(np.isfinite(true_array) & np.isfinite(predicted_array), axis=1)
    result = np.full(true_array.shape[0], np.nan, dtype=float)
    if np.any(finite):
        result[finite] = np.max(
            np.abs(predicted_array[finite] - true_array[finite]),
            axis=1,
        )
    return result


def reference_response_scale(curves) -> float:
    """Return a robust global response scale for normalized error reporting."""
    values = np.asarray(curves, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError('At least one finite reference response is required.')
    scale = float(np.max(finite) - np.min(finite))
    if not scale > np.finfo(float).eps:
        scale = float(np.max(np.abs(finite)))
    if not scale > np.finfo(float).eps:
        scale = 1.0
    return scale


def _crossing_time_one(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str,
) -> float:
    if np.any(~np.isfinite(values)):
        return np.nan
    if direction == 'auto':
        resolved = 'decreasing' if values[-1] < values[0] else 'increasing'
    else:
        resolved = direction
    for index in range(len(times) - 1):
        y0 = values[index]
        y1 = values[index + 1]
        if resolved == 'decreasing':
            crossed = y0 >= threshold >= y1
        else:
            crossed = y0 <= threshold <= y1
        if not crossed:
            continue
        if np.isclose(y1, y0):
            return float(times[index])
        fraction = (threshold - y0) / (y1 - y0)
        return float(times[index] + fraction * (times[index + 1] - times[index]))
    return np.nan


def threshold_crossing_time(
    times,
    values,
    threshold: float,
    *,
    direction: str = 'auto',
) -> np.ndarray:
    """Return the first linearly interpolated threshold crossing for each curve."""
    times = np.asarray(times, dtype=float)
    curves = np.asarray(values, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(~np.isfinite(times)):
        raise ValueError('Crossing times require a finite one-dimensional time grid.')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Crossing-time grid must be strictly increasing.')
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.ndim != 2 or curves.shape[1] != len(times):
        raise ValueError('Each curve must contain one value per crossing-time point.')
    if direction not in ('auto', 'increasing', 'decreasing'):
        raise ValueError("direction must be 'auto', 'increasing', or 'decreasing'.")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite.')
    return np.asarray([
        _crossing_time_one(times, row, threshold, direction)
        for row in curves
    ])


def approximate_mean_curve_interval_width(
    model_name: str,
    parameters: Sequence[float],
    covariance,
    evaluation_times,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Return average first-order confidence-interval width for a fitted mean curve."""
    covariance = np.asarray(covariance, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    times = np.asarray(evaluation_times, dtype=float)
    if covariance.shape != (len(parameters), len(parameters)):
        return np.nan
    if np.any(~np.isfinite(covariance)):
        return np.nan
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie strictly between zero and one.')
    try:
        sensitivities = finite_difference_sensitivities(
            model_name,
            times,
            parameters,
        )
    except Exception:
        return np.nan
    variances = np.einsum('ij,jk,ik->i', sensitivities, covariance, sensitivities)
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(variances))))
    if np.any(variances < -tolerance):
        return np.nan
    variances = np.maximum(variances, 0.0)
    z_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    return float(np.mean(2.0 * z_value * np.sqrt(variances)))


def parameter_near_bounds(
    model_name: str,
    parameters: Sequence[float],
    *,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-8,
) -> np.ndarray:
    """Flag fitted parameters that lie numerically near registered fit bounds."""
    parameters = np.asarray(parameters, dtype=float)
    spec = get_model_spec(model_name)
    if len(parameters) != len(spec.parameter_names):
        raise ValueError('Parameter vector length does not match the registered model.')
    bounds = spec.parameter_bounds
    if bounds is None:
        bounds = spec.fit_kwargs.get('bounds')
    if bounds is None:
        return np.zeros(len(parameters), dtype=bool)
    lower, upper = bounds
    lower = np.broadcast_to(np.asarray(lower, dtype=float), parameters.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), parameters.shape)
    scale = np.maximum(1.0, np.abs(parameters))
    tolerance = absolute_tolerance + relative_tolerance * scale
    near_lower = np.isfinite(lower) & (parameters - lower <= tolerance)
    near_upper = np.isfinite(upper) & (upper - parameters <= tolerance)
    return near_lower | near_upper
