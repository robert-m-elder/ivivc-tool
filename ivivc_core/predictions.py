"""Reusable numerical prediction methods for the current IVIVC approaches.

The functions in this module return structured numerical results only. Plotly
figures, HTML tables, and mutation of the legacy fit dictionaries remain in the
``engine.py`` compatibility layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from utilities.prediction_validity import (
    describe_tau_prediction_skip_reason,
)

from .inverse import inverse_response_times, DEFAULT_INVERSE_EXPANSIONS
from .extrapolation import annotate_prediction_domain
from .registry import get_model_spec
from .types import PredictedTrajectory, TimeCourse
from .intervals import (
    DEFAULT_INTERVAL_SAMPLES, DEFAULT_INTERVAL_SEED,
    validate_interval_settings, parameter_interval_summary,
    curve_parameter_intervals, attach_parameter_interval,
)


def safe_ratio(numerator, denominator):
    """Return an elementwise ratio with zero-denominator values as NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def inverse_response_prediction_intervals(
    model_function, params, pcov, residuals, target_responses, time_range,
    observed_responses=None, confidence_level=0.95, n_samples=100,
    grid_points=101, random_state=DEFAULT_INTERVAL_SEED,
    inverse_mapper=inverse_response_times, *, minimum_time=0.0,
    allow_extrapolation=True, max_expansions=DEFAULT_INVERSE_EXPANSIONS,
):
    """Parameter-only intervals for fixed responses under the current mapper.

    ``residuals`` is retained for compatibility but is not used. Central and
    sampled calculations use the same monotonic-branch search and time domain.
    Outside responses are never clipped to calibration endpoints.
    No response noise is generated and no failed draws are silently discarded.
    """
    from .uncertainty import sample_parameter_sets

    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    targets = np.asarray(target_responses, dtype=float)
    mapper_options = dict(minimum_time=minimum_time, allow_extrapolation=allow_extrapolation,
                          max_expansions=max_expansions)
    failed_reasons = [{} for _ in range(targets.size)]
    extrapolated_counts = np.zeros(targets.size, dtype=int)
    central, _, reason = inverse_mapper(
        model_function, params, targets, time_range,
        observed_responses=observed_responses, grid_points=grid_points, **mapper_options,
    )
    central = np.asarray(central, dtype=float).reshape(-1)
    sampling_error = None
    draws = np.empty((0, targets.size))
    if reason is None and np.any(np.isfinite(central)):
        try:
            samples = sample_parameter_sets(params, pcov, n_samples=n_samples,
                                            random_state=random_state)
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
            sampling_error = str(exc)
        else:
            draws = np.full((samples.n_samples, targets.size), np.nan)
            for index, sample in enumerate(samples.values):
                try:
                    mapped, draw_status, draw_reason = inverse_mapper(
                        model_function, dict(zip(samples.parameter_names, sample)),
                        targets, time_range, observed_responses=observed_responses,
                        grid_points=grid_points, **mapper_options,
                    )
                    mapped = np.asarray(mapped, dtype=float).reshape(-1)
                    if draw_reason is None:
                        draws[index] = mapped
                        extrapolated_counts += (np.isfinite(mapped) &
                            ((mapped < time_range[0]) | (mapped > time_range[1])))
                    for column in np.flatnonzero(~np.isfinite(draws[index])):
                        message = draw_reason or str(np.asarray(draw_status).reshape(-1)[column])
                        failed_reasons[column][message] = failed_reasons[column].get(message, 0) + 1
                except (ValueError, TypeError, RuntimeError, OverflowError, FloatingPointError) as exc:
                    for reasons in failed_reasons:
                        message = 'inverse evaluation failed: ' + str(exc)
                        reasons[message] = reasons.get(message, 0) + 1
    interval = parameter_interval_summary(
        draws, central, confidence_level=confidence_level, n_samples=n_samples,
        random_state=random_state, sampling_error=sampling_error,
    )
    # Legacy field now explicitly indicates that no response perturbation is used.
    interval['residual_std'] = 0.0
    interval['inverse_failed_draw_reasons'] = failed_reasons
    interval['inverse_extrapolated_samples'] = extrapolated_counts.reshape(targets.shape)
    for key in ('lower', 'upper', 'valid_counts', 'interval_status', 'interval_valid_samples'):
        interval[key] = interval[key].reshape(targets.shape)
    eligible_for_interval = np.isfinite(targets) & np.isfinite(central)
    unavailable = int(np.count_nonzero(
        (interval['interval_status'] != 'available') & eligible_for_interval
    ))
    note = reason
    if unavailable and note is None:
        if sampling_error:
            note = (
                'Uncertainty intervals could not be calculated because parameter '
                'uncertainty could not be evaluated.'
            )
        elif unavailable == 1:
            note = (
                f'An uncertainty interval could not be calculated for 1 of {targets.size} '
                'predictions because some parameter estimates did not produce a unique '
                'inverse mapping.'
            )
        else:
            note = (
                f'Uncertainty intervals could not be calculated for {unavailable} of '
                f'{targets.size} predictions because some parameter estimates did not '
                'produce unique inverse mappings.'
            )
    return interval, note

def predict_time_mapping(
    prediction: TimeCourse,
    *,
    model_name: str,
    parameters: Mapping[str, float],
    covariance=None,
    include_uncertainty: bool = False,
    confidence_level: float = .95,
    n_samples: int = DEFAULT_INTERVAL_SAMPLES,
    random_state=DEFAULT_INTERVAL_SEED,
) -> PredictedTrajectory:
    """Predict in vivo times from in vitro times using an Approach 1 fit."""
    model_function = get_model_spec(model_name).model_function
    in_vivo_time = np.asarray(model_function(prediction.time, **parameters))
    trajectory = PredictedTrajectory(
        source_approach='approach1',
        source_method='time_mapping',
        model_name=model_name,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
    )

    if include_uncertainty:
        interval = curve_parameter_intervals(
            get_model_spec(model_name).model_function, prediction.time, parameters, covariance,
            confidence_level=confidence_level, n_samples=n_samples, random_state=random_state,
        )
        attach_parameter_interval(trajectory, interval, output_kind='time')
    return trajectory


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
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        ratio = safe_ratio(in_vivo_model_value, in_vitro_model_value)
        in_vivo_value = prediction.value * ratio
    finite_prediction = np.isfinite(in_vivo_value) & np.isfinite(prediction.time)
    in_vivo_value = np.where(finite_prediction, in_vivo_value, np.nan)
    status = np.full(len(prediction.time), 'predicted', dtype=object)
    status[~finite_prediction] = 'unavailable (nonfinite prediction)'
    bad_denominator = ~np.isfinite(in_vitro_model_value) | (in_vitro_model_value == 0)
    status[bad_denominator] = 'unavailable (zero or nonfinite fitted in vitro response)'

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
        available=bool(np.any(finite_prediction)),
        status=status,
        scaling_factor=ratio,
        auxiliary={
            'in_vitro_model_value': in_vitro_model_value,
            'in_vivo_model_value': in_vivo_model_value,
            'plot_time': plot_time,
            'in_vitro_model_plot': in_vitro_model_plot,
            'in_vivo_model_plot': in_vivo_model_plot,
        },
    )


def predict_value_ratio_with_uncertainty(
    prediction: TimeCourse,
    *,
    model_name: str,
    in_vitro_parameters: Mapping[str, float],
    in_vitro_covariance,
    in_vivo_parameters: Mapping[str, float],
    in_vivo_covariance,
    confidence_level: float = 0.95,
    n_samples: int = 1000,
    random_state=DEFAULT_INTERVAL_SEED,
) -> PredictedTrajectory:
    """Add pointwise parameter-uncertainty intervals to response-ratio predictions.

    Both calibration parameter vectors are sampled independently, preserving
    covariance within each vector. Supplied prediction times/responses are
    fixed. No residual noise, observation error, or model discrepancy is added.
    The fitted-parameter prediction, not the sampled mean, remains the center.

    A bounded interval is withheld at a point if any requested draw is unusable
    there or the denominator's equal-tailed interval includes zero. The latter
    is a conservative display safeguard, not a formal ratio-confidence method.
    Nothing is clipped or silently conditioned on successful evaluations.
    """
    from .uncertainty import sample_value_ratio_trajectories, summarize_draws

    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError('confidence_level must be between zero and one.')
    if int(n_samples) != n_samples or n_samples < 2:
        raise ValueError('n_samples must be an integer of at least two.')
    n_samples = int(n_samples)
    count = len(prediction.time)
    lower = np.full(count, np.nan)
    upper = np.full(count, np.nan)
    denominator_lower = np.full(count, np.nan)
    denominator_upper = np.full(count, np.nan)
    valid_counts = np.zeros(count, dtype=int)
    total_samples = 0
    sampling_error = None
    interval_status = np.full(count, 'unavailable (parameter sampling failed)', dtype=object)

    try:
        ensemble = sample_value_ratio_trajectories(
            prediction,
            model_name=model_name,
            in_vitro_parameters=in_vitro_parameters,
            in_vitro_covariance=in_vitro_covariance,
            in_vivo_parameters=in_vivo_parameters,
            in_vivo_covariance=in_vivo_covariance,
            n_samples=n_samples,
            random_state=random_state,
        )
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        # Invalid covariance must not discard a usable central prediction or
        # prevent the other Approach 2 prediction methods from being evaluated.
        trajectory = predict_value_ratio(
            prediction,
            model_name=model_name,
            in_vitro_parameters=in_vitro_parameters,
            in_vivo_parameters=in_vivo_parameters,
        )
        sampling_error = str(exc)
    else:
        trajectory = ensemble.central
        total_samples = ensemble.n_draws
        # Pass the full matrix, NOT only rows flagged valid across all times.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            summary = summarize_draws(ensemble.value_draws, confidence_level=confidence_level)
            denominator_summary = summarize_draws(
                ensemble.metadata['in_vitro_model_draws'], confidence_level=confidence_level,
            )
        valid_counts = summary.valid_counts
        denominator_lower = denominator_summary.lower
        denominator_upper = denominator_summary.upper
        complete = (valid_counts == n_samples) & (total_samples == n_samples)
        complete &= denominator_summary.valid_counts == n_samples
        crosses_zero = (denominator_lower <= 0) & (denominator_upper >= 0)
        finite_bounds = np.isfinite(summary.lower) & np.isfinite(summary.upper)
        supported = complete & finite_bounds & ~crosses_zero & np.isfinite(trajectory.in_vivo_value)
        lower[supported], upper[supported] = summary.lower[supported], summary.upper[supported]
        interval_status[:] = 'unavailable (nonfinite or missing Monte Carlo draws)'
        interval_status[complete & ~finite_bounds] = 'unavailable (nonfinite percentile bounds)'
        interval_status[complete & crosses_zero] = 'unavailable (denominator interval includes zero)'
        interval_status[supported] = 'available'

    interval_status[~np.isfinite(trajectory.in_vivo_value)] = 'unavailable (point prediction unavailable)'
    trajectory.value_lower = lower
    trajectory.value_upper = upper
    trajectory.auxiliary.update({
        'interval_kind': 'parameter_uncertainty',
        'interval_confidence_level': confidence_level,
        'interval_status': interval_status,
        'interval_valid_samples': valid_counts,
        'interval_total_samples': total_samples,
        'interval_requested_samples': n_samples,
        'interval_random_seed': int(random_state) if isinstance(random_state, (int, np.integer)) else None,
        'interval_sampling_error': sampling_error,
        'denominator_lower': denominator_lower,
        'denominator_upper': denominator_upper,
    })
    unavailable = int(np.count_nonzero(interval_status != 'available'))
    if unavailable:
        trajectory.note = (
            f'{unavailable} of {count} response-ratio parameter-uncertainty intervals '
            'are unavailable. See the prediction data table for pointwise reasons.'
        )
        if sampling_error is not None:
            trajectory.note += f' Parameter sampling failed: {sampling_error}'
    return trajectory


def predict_tau_ratio(
    prediction: TimeCourse,
    *,
    model_name: str,
    tau_values: Sequence[Any] | None,
    residual_dof: Sequence[int | None] = (),
    confidence_level: float = 0.95,
    tau_diagnostics: Sequence[Mapping[str, Any]] = (),
    in_vitro_parameters=None,
    in_vitro_covariance=None,
    in_vivo_parameters=None,
    in_vivo_covariance=None,
    n_samples: int = DEFAULT_INTERVAL_SAMPLES,
    random_state=DEFAULT_INTERVAL_SEED,
) -> PredictedTrajectory:
    """Predict in vivo times using the ratio of fitted time constants."""
    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    skip_reason = describe_tau_prediction_skip_reason(tau_values)
    if skip_reason is not None:
        unsupported = [
            diagnostics.get('reason', '')
            for diagnostics in tau_diagnostics
            if diagnostics.get('reason')
        ]
        if unsupported and all('not supported for this model' in reason.lower() for reason in unsupported):
            skip_reason = 'Time-constant rescaling is not available for this model.'
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
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        nominal_ratio = float(np.float64(tau2.n) / np.float64(tau1.n))
        in_vivo_time = prediction.time * nominal_ratio
    if not np.isfinite(nominal_ratio) or nominal_ratio <= 0:
        return PredictedTrajectory(
            source_approach='approach2', source_method='tau_ratio', model_name=model_name,
            in_vitro_time=prediction.time, in_vitro_value=prediction.value,
            in_vivo_time=np.full(prediction.time.shape, np.nan),
            in_vivo_value=prediction.value, available=False,
            skip_reason='Time-constant rescaling was not performed because a valid time-constant ratio could not be calculated.',
        )
    trajectory = PredictedTrajectory(
        source_approach='approach2', source_method='tau_ratio', model_name=model_name,
        in_vitro_time=prediction.time, in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time, in_vivo_value=prediction.value,
        scaling_factor=np.full(prediction.time.shape, nominal_ratio),
        auxiliary={'time_ratio': nominal_ratio},
    )
    # The legacy tau SD and residual_dof are not a parameter distribution for
    # the ratio. Use the fitted vectors/covariances or report missing intervals.
    from .uncertainty import sample_tau_ratio_trajectories
    sampling_error = None
    draws = np.empty((0, len(prediction.time)))
    if in_vitro_parameters is None or in_vivo_parameters is None:
        sampling_error = 'Fitted parameter vectors and covariance matrices are required for tau-ratio intervals.'
    else:
        try:
            ensemble = sample_tau_ratio_trajectories(
                prediction, model_name=model_name,
                in_vitro_parameters=in_vitro_parameters, in_vitro_covariance=in_vitro_covariance,
                in_vivo_parameters=in_vivo_parameters, in_vivo_covariance=in_vivo_covariance,
                n_samples=n_samples, random_state=random_state,
            )
            draws = ensemble.time_draws
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
            sampling_error = str(exc)
    interval = parameter_interval_summary(
        draws, in_vivo_time, confidence_level=confidence_level, n_samples=n_samples,
        random_state=random_state, sampling_error=sampling_error,
    )
    return attach_parameter_interval(trajectory, interval, output_kind='time')


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
    n_samples: int = 100,
    random_state=DEFAULT_INTERVAL_SEED,
    inverse_mapper: Callable[..., tuple[np.ndarray, np.ndarray, str | None]] = inverse_response_times,
    interval_estimator: Callable[..., tuple[dict[str, Any], str | None]] = inverse_response_prediction_intervals,
    minimum_time: float | None = 0.0,
    allow_extrapolation: bool = True,
    max_expansions: int = DEFAULT_INVERSE_EXPANSIONS,
) -> PredictedTrajectory:
    """Map fixed responses and propagate only fitted in vivo parameter uncertainty."""
    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    function = get_model_spec(model_name).model_function
    mapper_options = dict(minimum_time=minimum_time, allow_extrapolation=allow_extrapolation,
                          max_expansions=max_expansions)
    times, status, skip_reason = inverse_mapper(
        function, parameters, prediction.value, calibration_time_range,
        observed_responses=observed_responses, **mapper_options,
    )
    times = np.asarray(times, dtype=float)
    supported = np.isfinite(times)
    interval = parameter_interval_summary(
        np.empty((0, len(times))), times, confidence_level=confidence_level,
        n_samples=n_samples, random_state=random_state,
    )
    interval_note = None
    if skip_reason is None and np.any(supported):
        interval, interval_note = interval_estimator(
            function, parameters, covariance, residuals,
            np.where(supported, prediction.value, np.nan), calibration_time_range,
            observed_responses=observed_responses, confidence_level=confidence_level,
            n_samples=n_samples, random_state=random_state, inverse_mapper=inverse_mapper,
            **mapper_options,
        )
    elif skip_reason is None:
        skip_reason = (
            'Observed-response inverse mapping could not be calculated because none of '
            'the responses have a unique inverse mapping to in vivo time.'
        )
    trajectory = PredictedTrajectory(
        source_approach='approach2', source_method='observed_response_inverse', model_name=model_name,
        in_vitro_time=prediction.time, in_vitro_value=prediction.value,
        in_vivo_time=times, in_vivo_value=prediction.value,
        available=bool(skip_reason is None and np.any(supported)),
        skip_reason=skip_reason, status=status,
    )
    attach_parameter_interval(trajectory, interval, output_kind='time')
    # Inverse mapping has more specific user-facing messages below; keep the
    # generic interval diagnostics in auxiliary metadata rather than duplicating them.
    trajectory.note = None
    trajectory.auxiliary.update({
        'assumed_response_sd': 0.0,
        'interval_note': interval_note,
        'inverse_minimum_time': minimum_time,
        'inverse_max_expansions': max_expansions,
        'inverse_allow_extrapolation': bool(allow_extrapolation),
        'inverse_failed_draw_reasons': interval.get('inverse_failed_draw_reasons', [{} for _ in times]),
        'inverse_extrapolated_samples': interval.get('inverse_extrapolated_samples', np.zeros(len(times), dtype=int)),
    })
    if interval_note:
        trajectory.note = interval_note
    missing = int(np.count_nonzero(~supported))
    if missing:
        if missing == 1:
            message = (
                f'1 of {len(times)} predictions could not be calculated because the response '
                'does not have a unique inverse mapping to in vivo time.'
            )
        else:
            message = (
                f'{missing} of {len(times)} predictions could not be calculated because the '
                'responses do not have unique inverse mappings to in vivo time.'
            )
        trajectory.note = ' '.join(part for part in (trajectory.note, message) if part)
    return annotate_prediction_domain(
        trajectory, in_vivo_time=calibration_time_range, in_vivo_response=observed_responses,
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
    status = np.where(np.isfinite(in_vivo_value), 'predicted', 'unavailable (outside or undefined response-ratio lookup)')
    return PredictedTrajectory(
        source_approach='approach3',
        source_method='value_scaling',
        status=status,
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
    status = np.where(np.isfinite(in_vivo_time), 'predicted', 'unavailable (outside or undefined time-ratio lookup)')
    retained_zero = (prediction.time == 0) & ~np.isfinite(scaling_factor)
    status[retained_zero] = 'time zero retained without supported response lookup'
    return PredictedTrajectory(
        source_approach='approach3',
        source_method='time_scaling',
        status=status,
        in_vitro_time=prediction.time,
        in_vitro_value=prediction.value,
        in_vivo_time=in_vivo_time,
        in_vivo_value=prediction.value,
        scaling_factor=scaling_factor,
    )
