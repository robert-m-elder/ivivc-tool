"""Shared uncertainty sampling and complete trajectory ensembles.

The existing IVIVC application primarily displays pointwise uncertainty
summaries.  The future study-design engine also needs the complete joint draws
that generated those summaries so temporal dependence is retained.  This
module provides the common numerical primitives without producing plots or
HTML.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings
from typing import Any, Callable

import numpy as np
import scipy as sp

from .inverse import DEFAULT_INVERSE_EXPANSIONS
from .registry import get_model_spec
from .types import (
    CurveDraws,
    IntervalSummary,
    ParameterSamples,
    PredictedTrajectory,
    ProfileFit,
    TimeCourse,
    TrajectoryEnsemble,
)


def _parameter_input(
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
    parameter_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(parameters, Mapping):
        names = tuple(parameters.keys())
        values = np.asarray([parameters[name] for name in names], dtype=float)
        if parameter_names is not None and tuple(parameter_names) != names:
            raise ValueError(
                'parameter_names must match the order of the parameter mapping.'
            )
        return names, values

    values = np.asarray(parameters, dtype=float)
    if values.ndim != 1:
        raise ValueError('Parameter values must be one-dimensional.')
    if parameter_names is None:
        names = tuple(f'p{index}' for index in range(len(values)))
    else:
        names = tuple(parameter_names)
        if len(names) != len(values):
            raise ValueError(
                'The number of parameter names must match the parameter values.'
            )
    return names, values


def _split_random_state(random_state, count: int) -> list[Any]:
    """Return independent deterministic streams for an integer seed.

    A supplied Generator or RandomState is consumed sequentially by the
    sampling calls.  ``None`` retains SciPy's established use of NumPy's global
    random state in compatibility wrappers.
    """
    if count < 1:
        return []
    if isinstance(random_state, (int, np.integer)):
        seed_sequence = np.random.SeedSequence(int(random_state))
        return [np.random.default_rng(seed) for seed in seed_sequence.spawn(count)]
    return [random_state] * count


def _reshape_sampled_values(values, parameter_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if parameter_count == 1:
            values = values.reshape(-1, 1)
        else:
            values = values.reshape(1, -1)
    return values


def _registered_parameter_constraints(
    model_name: str | None,
    parameter_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, Callable[[Mapping[str, float]], bool] | None]:
    count = len(parameter_names)
    lower = np.full(count, -np.inf, dtype=float)
    upper = np.full(count, np.inf, dtype=float)
    validator = None
    if model_name is None:
        return lower, upper, validator

    spec = get_model_spec(model_name)
    if tuple(spec.parameter_names) != tuple(parameter_names):
        raise ValueError(
            'The sampled parameter order does not match the registered model.'
        )
    bounds = spec.parameter_bounds
    if bounds is None:
        bounds = spec.fit_kwargs.get('bounds')
    if bounds is not None:
        if len(bounds) != 2:
            raise ValueError('Registered model bounds must contain lower and upper values.')
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (count,)).copy()
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (count,)).copy()
    validator = spec.parameter_validator
    return lower, upper, validator


def _validate_parameter_rows(
    values: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    validator: Callable[[Mapping[str, float]], bool] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    finite = np.all(np.isfinite(values), axis=1)
    within_bounds = np.all((values >= lower) & (values <= upper), axis=1)
    valid = finite & within_bounds
    counts = {
        'nonfinite': int(np.count_nonzero(~finite)),
        'outside_registered_bounds': int(np.count_nonzero(finite & ~within_bounds)),
    }
    if validator is not None:
        validator_valid = np.zeros(len(values), dtype=bool)
        for index in np.flatnonzero(valid):
            mapping = dict(zip(parameter_names, values[index]))
            try:
                validator_valid[index] = bool(validator(mapping))
            except Exception:
                validator_valid[index] = False
        counts['model_validator'] = int(np.count_nonzero(valid & ~validator_valid))
        valid &= validator_valid
    return valid, {key: value for key, value in counts.items() if value > 0}


def sample_parameter_sets(
    parameters: Mapping[str, float] | Sequence[float] | np.ndarray,
    covariance,
    *,
    parameter_names: Sequence[str] | None = None,
    n_samples: int = 1000,
    random_state=None,
    allow_singular: bool = True,
    model_name: str | None = None,
    max_resample_batches: int = 50,
) -> ParameterSamples:
    """Draw complete parameter vectors from a multivariate normal model.

    When ``model_name`` is supplied, registered model bounds and an optional
    model-level ``parameter_validator`` are applied. Rejected vectors are
    resampled so the returned object contains the requested number of valid
    joint draws whenever feasible.
    """
    names, mean = _parameter_input(parameters, parameter_names)
    covariance = np.asarray(covariance, dtype=float)
    n_samples = int(n_samples)
    max_resample_batches = int(max_resample_batches)
    if n_samples < 1:
        raise ValueError('n_samples must be at least one.')
    if max_resample_batches < 1:
        raise ValueError('max_resample_batches must be at least one.')
    expected_shape = (len(mean), len(mean))
    if covariance.shape != expected_shape:
        raise ValueError(
            f'Parameter covariance must have shape {expected_shape}, not {covariance.shape}.'
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError('Parameter values and covariance must be finite.')

    lower, upper, validator = _registered_parameter_constraints(model_name, names)
    if np.any(mean < lower) or np.any(mean > upper):
        raise ValueError('The fitted parameter vector violates registered model bounds.')
    if validator is not None:
        try:
            mean_valid = bool(validator(dict(zip(names, mean))))
        except Exception:
            mean_valid = False
        if not mean_valid:
            raise ValueError('The fitted parameter vector fails the model parameter validator.')

    distribution = sp.stats.multivariate_normal(
        mean=mean,
        cov=covariance,
        allow_singular=allow_singular,
    )

    constraints_active = (
        np.any(np.isfinite(lower))
        or np.any(np.isfinite(upper))
        or validator is not None
    )
    if not constraints_active:
        values = distribution.rvs(size=n_samples, random_state=random_state)
        values = _reshape_sampled_values(values, len(mean))
        return ParameterSamples(
            parameter_names=names,
            values=values,
            requested_samples=n_samples,
            attempted_samples=n_samples,
        )

    rng = (
        np.random.default_rng(int(random_state))
        if isinstance(random_state, (int, np.integer))
        else random_state
    )
    accepted: list[np.ndarray] = []
    attempted = 0
    rejection_counts: dict[str, int] = {}
    remaining = n_samples
    for _ in range(max_resample_batches):
        batch_size = remaining
        sampled = distribution.rvs(size=batch_size, random_state=rng)
        sampled = _reshape_sampled_values(sampled, len(mean))
        attempted += len(sampled)
        valid, counts = _validate_parameter_rows(
            sampled,
            parameter_names=names,
            lower=lower,
            upper=upper,
            validator=validator,
        )
        for reason, count in counts.items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count
        if np.any(valid):
            accepted.append(sampled[valid])
            accepted_count = sum(len(batch) for batch in accepted)
            remaining = n_samples - accepted_count
            if remaining <= 0:
                break

    if not accepted or sum(len(batch) for batch in accepted) < n_samples:
        accepted_count = sum(len(batch) for batch in accepted)
        raise ValueError(
            f'Only {accepted_count} of {n_samples} valid parameter draws could be '
            f'generated after {attempted} attempts.'
        )
    values = np.vstack(accepted)[:n_samples]
    return ParameterSamples(
        parameter_names=names,
        values=values,
        requested_samples=n_samples,
        attempted_samples=attempted,
        rejection_counts=rejection_counts,
    )


def evaluate_curve_draws(
    model_function,
    x_values,
    parameter_samples: ParameterSamples,
    *,
    require_finite: bool = True,
) -> CurveDraws:
    """Evaluate a model for each complete sampled parameter vector.

    With ``require_finite=False``, retain finite values at individual points
    even if another point fails. ``valid_draws`` still identifies complete,
    finite rows; pass ``values`` to ``summarize_draws`` for pointwise use.
    """
    x_values = np.asarray(x_values, dtype=float)
    if x_values.ndim != 1:
        raise ValueError('x_values must be one-dimensional.')

    values = np.full(
        (parameter_samples.n_samples, len(x_values)),
        np.nan,
        dtype=float,
    )
    valid = np.zeros(parameter_samples.n_samples, dtype=bool)
    for index, sample in enumerate(parameter_samples.values):
        sample_parameters = dict(zip(parameter_samples.parameter_names, sample))
        try:
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                prediction = np.asarray(
                    model_function(x_values, **sample_parameters), dtype=float,
                ).reshape(-1)
            if len(prediction) != len(x_values):
                continue
        except Exception:
            if require_finite:
                continue
            # A vector evaluation can raise because of only one requested time.
            prediction = np.full(len(x_values), np.nan, dtype=float)
            for point_index, x in enumerate(x_values):
                try:
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        point = np.asarray(
                            model_function(np.asarray([x]), **sample_parameters),
                            dtype=float,
                        ).reshape(-1)
                    if len(point) == 1:
                        prediction[point_index] = point[0]
                except Exception:
                    continue
        finite = bool(np.all(np.isfinite(prediction)))
        if require_finite and not finite:
            continue
        values[index] = prediction
        valid[index] = finite

    return CurveDraws(x=x_values, values=values, valid_draws=valid)


def summarize_draws(
    draws: CurveDraws | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> IntervalSummary:
    """Calculate pointwise summaries while retaining complete draws upstream."""
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError('confidence_level must be between zero and one.')

    if isinstance(draws, CurveDraws):
        values = draws.values[draws.valid_draws]
        n_points = len(draws.x)
    else:
        values = np.asarray(draws, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError('Draw values must be a two-dimensional array.')
        n_points = values.shape[1]

    lower = np.full(n_points, np.nan, dtype=float)
    upper = np.full(n_points, np.nan, dtype=float)
    mean = np.full(n_points, np.nan, dtype=float)
    std = np.full(n_points, np.nan, dtype=float)
    valid_counts = np.zeros(n_points, dtype=int)

    alpha = 1.0 - confidence_level
    percentiles = (100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0))
    for point_index in range(n_points):
        point_values = values[:, point_index] if values.size else np.asarray([])
        point_values = point_values[np.isfinite(point_values)]
        valid_counts[point_index] = len(point_values)
        if not len(point_values):
            continue
        lower[point_index], upper[point_index] = np.percentile(
            point_values,
            percentiles,
        )
        mean[point_index] = np.mean(point_values)
        std[point_index] = np.std(point_values)

    return IntervalSummary(
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
        valid_counts=valid_counts,
        confidence_level=confidence_level,
    )


def deterministic_trajectory_ensemble(
    trajectory: PredictedTrajectory,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryEnsemble:
    """Wrap one central trajectory as a one-draw ensemble."""
    valid = bool(
        trajectory.available
        and np.all(np.isfinite(trajectory.in_vivo_time))
        and np.all(np.isfinite(trajectory.in_vivo_value))
    )
    return TrajectoryEnsemble(
        central=trajectory,
        time_draws=trajectory.in_vivo_time.reshape(1, -1),
        value_draws=trajectory.in_vivo_value.reshape(1, -1),
        valid_draws=np.asarray([valid]),
        metadata={} if metadata is None else metadata,
    )


def sample_profile_trajectory(
    fit: ProfileFit,
    time_grid,
    *,
    n_samples: int = 1000,
    random_state=None,
) -> TrajectoryEnsemble:
    """Generate complete response-trajectory draws from one fitted profile."""
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1:
        raise ValueError('time_grid must be one-dimensional.')
    spec = get_model_spec(fit.model_name)
    parameter_samples = sample_parameter_sets(
        fit.parameters,
        fit.covariance,
        n_samples=n_samples,
        random_state=random_state,
        model_name=fit.model_name,
    )
    curve_draws = evaluate_curve_draws(
        spec.model_function,
        time_grid,
        parameter_samples,
    )
    central_value = np.asarray(
        spec.model_function(time_grid, **fit.parameters),
        dtype=float,
    )
    central = PredictedTrajectory(
        source_approach='profile',
        source_method='fitted_profile',
        model_name=fit.model_name,
        in_vitro_time=time_grid,
        in_vitro_value=central_value,
        in_vivo_time=time_grid,
        in_vivo_value=central_value,
    )
    return TrajectoryEnsemble(
        central=central,
        time_draws=np.broadcast_to(time_grid, curve_draws.values.shape).copy(),
        value_draws=curve_draws.values,
        valid_draws=curve_draws.valid_draws,
        metadata={
            'parameter_names': parameter_samples.parameter_names,
            'parameter_samples': parameter_samples.values,
            'requested_samples': parameter_samples.requested_samples,
            'attempted_samples': parameter_samples.attempted_samples,
            'rejected_parameter_draws': dict(parameter_samples.rejection_counts),
        },
    )


def sample_time_mapping_trajectories(
    prediction: TimeCourse,
    *,
    model_name: str,
    parameters: Mapping[str, float],
    covariance,
    n_samples: int = 1000,
    random_state=None,
) -> TrajectoryEnsemble:
    """Generate complete Approach 1 time-mapping trajectory draws."""
    from .predictions import predict_time_mapping

    spec = get_model_spec(model_name)
    central = predict_time_mapping(
        prediction,
        model_name=model_name,
        parameters=parameters,
    )
    parameter_samples = sample_parameter_sets(
        parameters,
        covariance,
        n_samples=n_samples,
        random_state=random_state,
        model_name=model_name,
    )
    time_draws = evaluate_curve_draws(
        spec.model_function,
        prediction.time,
        parameter_samples,
    )
    return TrajectoryEnsemble(
        central=central,
        time_draws=time_draws.values,
        value_draws=np.broadcast_to(
            prediction.value,
            time_draws.values.shape,
        ).copy(),
        valid_draws=time_draws.valid_draws,
        metadata={
            'parameter_names': parameter_samples.parameter_names,
            'parameter_samples': parameter_samples.values,
            'requested_samples': parameter_samples.requested_samples,
            'attempted_samples': parameter_samples.attempted_samples,
            'rejected_parameter_draws': dict(parameter_samples.rejection_counts),
        },
    )


def sample_value_ratio_trajectories(
    prediction: TimeCourse,
    *,
    model_name: str,
    in_vitro_parameters: Mapping[str, float],
    in_vitro_covariance,
    in_vivo_parameters: Mapping[str, float],
    in_vivo_covariance,
    n_samples: int = 1000,
    random_state=None,
) -> TrajectoryEnsemble:
    """Generate complete Approach 2 response-value-ratio trajectory draws."""
    from .predictions import predict_value_ratio

    spec = get_model_spec(model_name)
    random_states = _split_random_state(random_state, 2)
    in_vitro_samples = sample_parameter_sets(
        in_vitro_parameters,
        in_vitro_covariance,
        n_samples=n_samples,
        random_state=random_states[0],
        model_name=model_name,
    )
    in_vivo_samples = sample_parameter_sets(
        in_vivo_parameters,
        in_vivo_covariance,
        n_samples=n_samples,
        random_state=random_states[1],
        model_name=model_name,
    )
    in_vitro_draws = evaluate_curve_draws(
        spec.model_function,
        prediction.time,
        in_vitro_samples,
        require_finite=False,
    )
    in_vivo_draws = evaluate_curve_draws(
        spec.model_function,
        prediction.time,
        in_vivo_samples,
        require_finite=False,
    )

    denominator = in_vitro_draws.values
    numerator = in_vivo_draws.values
    ratio_draws = np.full(denominator.shape, np.nan, dtype=float)
    supported = np.isfinite(denominator) & np.isfinite(numerator) & (denominator != 0)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        np.divide(numerator, denominator, out=ratio_draws, where=supported)
        value_draws = prediction.value.reshape(1, -1) * ratio_draws
    # Keep partially valid rows for pointwise intervals. Design callers still
    # use valid_draws to select complete finite trajectories.
    valid = np.all(np.isfinite(value_draws), axis=1)
    valid &= bool(np.all(np.isfinite(prediction.time)))
    central = predict_value_ratio(
        prediction,
        model_name=model_name,
        in_vitro_parameters=in_vitro_parameters,
        in_vivo_parameters=in_vivo_parameters,
    )
    return TrajectoryEnsemble(
        central=central,
        time_draws=np.broadcast_to(
            prediction.time,
            value_draws.shape,
        ).copy(),
        value_draws=value_draws,
        valid_draws=valid,
        metadata={
            'in_vitro_model_draws': in_vitro_draws.values,
            'in_vitro_parameter_names': in_vitro_samples.parameter_names,
            'in_vitro_parameter_samples': in_vitro_samples.values,
            'in_vivo_parameter_names': in_vivo_samples.parameter_names,
            'in_vivo_parameter_samples': in_vivo_samples.values,
            'ratio_draws': ratio_draws,
            'requested_samples': int(n_samples),
            'in_vitro_attempted_samples': in_vitro_samples.attempted_samples,
            'in_vivo_attempted_samples': in_vivo_samples.attempted_samples,
            'in_vitro_rejected_parameter_draws': dict(in_vitro_samples.rejection_counts),
            'in_vivo_rejected_parameter_draws': dict(in_vivo_samples.rejection_counts),
        },
    )


def _tau_from_parameter_vector(model_function, parameters) -> float:
    # Share the app's convergence checks; do not accept warned-about integrals
    # in trajectory ensembles when the same calculation is unavailable in a fit.
    from utilities.time_constants import evaluate_tau
    return evaluate_tau(model_function, parameters)[0]


def sample_tau_ratio_trajectories(
    prediction: TimeCourse,
    *,
    model_name: str,
    in_vitro_parameters: Mapping[str, float],
    in_vitro_covariance,
    in_vivo_parameters: Mapping[str, float],
    in_vivo_covariance,
    n_samples: int = 1000,
    random_state=None,
) -> TrajectoryEnsemble:
    """Generate complete time-scaling draws from sampled fitted time constants."""
    spec = get_model_spec(model_name)
    random_states = _split_random_state(random_state, 2)
    in_vitro_samples = sample_parameter_sets(
        in_vitro_parameters,
        in_vitro_covariance,
        n_samples=n_samples,
        random_state=random_states[0],
        model_name=model_name,
    )
    in_vivo_samples = sample_parameter_sets(
        in_vivo_parameters,
        in_vivo_covariance,
        n_samples=n_samples,
        random_state=random_states[1],
        model_name=model_name,
    )

    in_vitro_tau = np.asarray(
        [
            _tau_from_parameter_vector(spec.model_function, sample)
            for sample in in_vitro_samples.values
        ],
        dtype=float,
    )
    in_vivo_tau = np.asarray(
        [
            _tau_from_parameter_vector(spec.model_function, sample)
            for sample in in_vivo_samples.values
        ],
        dtype=float,
    )
    valid = (
        np.isfinite(in_vitro_tau)
        & np.isfinite(in_vivo_tau)
        & (in_vitro_tau > 0)
        & (in_vivo_tau > 0)
    )
    ratio_draws = np.full(int(n_samples), np.nan, dtype=float)
    ratio_draws[valid] = in_vivo_tau[valid] / in_vitro_tau[valid]
    time_draws = prediction.time.reshape(1, -1) * ratio_draws.reshape(-1, 1)
    value_draws = np.broadcast_to(prediction.value, time_draws.shape).copy()
    valid &= np.all(np.isfinite(time_draws), axis=1)

    nominal_in_vitro_tau = _tau_from_parameter_vector(
        spec.model_function,
        [in_vitro_parameters[name] for name in spec.parameter_names],
    )
    nominal_in_vivo_tau = _tau_from_parameter_vector(
        spec.model_function,
        [in_vivo_parameters[name] for name in spec.parameter_names],
    )
    nominal_valid = (
        np.isfinite(nominal_in_vitro_tau)
        and np.isfinite(nominal_in_vivo_tau)
        and nominal_in_vitro_tau > 0
        and nominal_in_vivo_tau > 0
    )
    if nominal_valid:
        nominal_ratio = nominal_in_vivo_tau / nominal_in_vitro_tau
        central_time = prediction.time * nominal_ratio
        central = PredictedTrajectory(
            source_approach='approach2',
            source_method='tau_ratio',
            model_name=model_name,
            in_vitro_time=prediction.time,
            in_vitro_value=prediction.value,
            in_vivo_time=central_time,
            in_vivo_value=prediction.value,
            scaling_factor=np.full(prediction.time.shape, nominal_ratio),
            auxiliary={
                'in_vitro_tau': nominal_in_vitro_tau,
                'in_vivo_tau': nominal_in_vivo_tau,
                'time_ratio': nominal_ratio,
            },
        )
    else:
        central = PredictedTrajectory(
            source_approach='approach2',
            source_method='tau_ratio',
            model_name=model_name,
            in_vitro_time=prediction.time,
            in_vitro_value=prediction.value,
            in_vivo_time=np.full(prediction.time.shape, np.nan),
            in_vivo_value=prediction.value,
            available=False,
            skip_reason='Time-constant ratio prediction is unavailable for the fitted parameter values.',
        )

    return TrajectoryEnsemble(
        central=central,
        time_draws=time_draws,
        value_draws=value_draws,
        valid_draws=valid,
        metadata={
            'in_vitro_parameter_samples': in_vitro_samples.values,
            'in_vivo_parameter_samples': in_vivo_samples.values,
            'in_vitro_tau_draws': in_vitro_tau,
            'in_vivo_tau_draws': in_vivo_tau,
            'ratio_draws': ratio_draws,
            'requested_samples': int(n_samples),
            'in_vitro_attempted_samples': in_vitro_samples.attempted_samples,
            'in_vivo_attempted_samples': in_vivo_samples.attempted_samples,
            'in_vitro_rejected_parameter_draws': dict(in_vitro_samples.rejection_counts),
            'in_vivo_rejected_parameter_draws': dict(in_vivo_samples.rejection_counts),
        },
    )


def sample_inverse_response_trajectories(
    prediction: TimeCourse,
    *,
    model_name: str,
    parameters: Mapping[str, float],
    covariance,
    calibration_time_range,
    observed_responses=None,
    n_samples: int = 1000,
    random_state=None,
    grid_points: int = 1001,
    minimum_time: float | None = 0.0,
    allow_extrapolation: bool = True,
    max_expansions: int = DEFAULT_INVERSE_EXPANSIONS,
) -> TrajectoryEnsemble:
    """Generate complete inverse response-to-time mapping draws."""
    from .predictions import inverse_response_times

    spec = get_model_spec(model_name)
    central_time, status, skip_reason = inverse_response_times(
        spec.model_function,
        parameters,
        prediction.value,
        calibration_time_range,
        observed_responses=observed_responses,
        grid_points=grid_points, minimum_time=minimum_time,
        allow_extrapolation=allow_extrapolation, max_expansions=max_expansions,
    )
    central_time = np.asarray(central_time, dtype=float)
    central_supported = np.isfinite(central_time)
    central = PredictedTrajectory(
        source_approach='approach2',
        source_method='observed_response_inverse',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=central_time,
        in_vivo_value=prediction.value,
        available=bool(skip_reason is None and np.any(central_supported)),
        skip_reason=skip_reason,
        status=status,
    )

    from .extrapolation import annotate_prediction_domain
    annotate_prediction_domain(central, in_vivo_time=calibration_time_range,
                               in_vivo_response=observed_responses)
    parameter_samples = sample_parameter_sets(
        parameters,
        covariance,
        n_samples=n_samples,
        random_state=random_state,
        model_name=model_name,
    )
    time_draws = np.full(
        (parameter_samples.n_samples, len(prediction.time)),
        np.nan,
        dtype=float,
    )
    valid = np.zeros(parameter_samples.n_samples, dtype=bool)
    for draw_index, sample in enumerate(parameter_samples.values):
        sample_parameters = dict(zip(parameter_samples.parameter_names, sample))
        mapped_time, _, draw_reason = inverse_response_times(
            spec.model_function,
            sample_parameters,
            prediction.value,
            calibration_time_range,
            observed_responses=observed_responses,
            grid_points=grid_points, minimum_time=minimum_time,
            allow_extrapolation=allow_extrapolation, max_expansions=max_expansions,
        )
        mapped_time = np.asarray(mapped_time, dtype=float)
        time_draws[draw_index] = mapped_time
        if draw_reason is not None or not np.any(central_supported):
            continue
        valid[draw_index] = bool(np.all(np.isfinite(mapped_time[central_supported])))

    return TrajectoryEnsemble(
        central=central,
        time_draws=time_draws,
        value_draws=np.broadcast_to(
            prediction.value,
            time_draws.shape,
        ).copy(),
        valid_draws=valid,
        metadata={
            'inverse_minimum_time': minimum_time,
            'inverse_max_expansions': max_expansions,
            'inverse_allow_extrapolation': allow_extrapolation,
            'parameter_names': parameter_samples.parameter_names,
            'parameter_samples': parameter_samples.values,
            'central_supported': central_supported,
            'requested_samples': parameter_samples.requested_samples,
            'attempted_samples': parameter_samples.attempted_samples,
            'rejected_parameter_draws': dict(parameter_samples.rejection_counts),
        },
    )
