"""Conversion and evaluation of common-grid reference trajectory ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from ivivc_core import (
    PredictedTrajectory,
    ProfileFit,
    TimeCourse,
    TrajectoryEnsemble,
    deterministic_trajectory_ensemble,
    sample_profile_trajectory,
)

from .types import InterpolationMethod, ReferenceTrajectoryEnsemble


DuplicateTimePolicy = Literal['mean', 'first', 'error']


def _prepare_time_value_pairs(
    time,
    value,
    *,
    duplicate_time_policy: DuplicateTimePolicy,
) -> tuple[np.ndarray, np.ndarray, int]:
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    finite = np.isfinite(time) & np.isfinite(value)
    time = time[finite]
    value = value[finite]
    if len(time) < 2:
        raise ValueError('A trajectory requires at least two finite time-response pairs.')

    order = np.argsort(time, kind='mergesort')
    time = time[order]
    value = value[order]
    unique_time, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    if duplicate_count:
        if duplicate_time_policy == 'error':
            raise ValueError('A trajectory contains duplicate in vivo times.')
        if duplicate_time_policy == 'first':
            first_indices = np.unique(time, return_index=True)[1]
            value = value[first_indices]
        elif duplicate_time_policy == 'mean':
            sums = np.bincount(inverse, weights=value)
            value = sums / counts
        else:
            raise ValueError(
                "duplicate_time_policy must be 'mean', 'first', or 'error'."
            )
        time = unique_time

    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('A trajectory must contain at least two distinct increasing times.')
    return time, value, duplicate_count


def _interpolate_values(
    source_time: np.ndarray,
    source_value: np.ndarray,
    target_time: np.ndarray,
    *,
    method: InterpolationMethod,
    extrapolate: bool,
) -> np.ndarray:
    if method == 'pchip':
        interpolator = PchipInterpolator(
            source_time,
            source_value,
            extrapolate=extrapolate,
        )
        return np.asarray(interpolator(target_time), dtype=float)
    if method == 'linear':
        fill_value = 'extrapolate' if extrapolate else np.nan
        interpolator = interp1d(
            source_time,
            source_value,
            kind='linear',
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )
        return np.asarray(interpolator(target_time), dtype=float)
    raise ValueError("Interpolation method must be 'linear' or 'pchip'.")


def _validate_target_grid(
    time_grid,
    *,
    supported_range: tuple[float, float],
) -> np.ndarray:
    grid = np.asarray(time_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError('The common time grid must contain at least two points.')
    if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError('The common time grid must be finite and strictly increasing.')
    tolerance = 1e-12 * max(1.0, abs(supported_range[0]), abs(supported_range[1]))
    if grid[0] < supported_range[0] - tolerance or grid[-1] > supported_range[1] + tolerance:
        raise ValueError('The common time grid extends beyond the shared supported range.')
    return grid.copy()


def reference_from_trajectory_ensemble(
    ensemble: TrajectoryEnsemble,
    *,
    time_grid=None,
    grid_size: int | None = None,
    interpolation: InterpolationMethod = 'pchip',
    duplicate_time_policy: DuplicateTimePolicy = 'mean',
    draw_weights=None,
) -> ReferenceTrajectoryEnsemble:
    """Convert IVIVC trajectory draws to one common in vivo time grid.

    The supported domain is the intersection of the finite time domains of all
    usable draws.  Extrapolation is deliberately excluded from this conversion.
    Draws that cannot supply two distinct finite time-response pairs are retained
    as invalid rows for traceability.
    """
    if interpolation not in ('linear', 'pchip'):
        raise ValueError("interpolation must be either 'linear' or 'pchip'.")
    if duplicate_time_policy not in ('mean', 'first', 'error'):
        raise ValueError(
            "duplicate_time_policy must be 'mean', 'first', or 'error'."
        )

    prepared: list[tuple[np.ndarray, np.ndarray] | None] = []
    input_valid = np.asarray(ensemble.valid_draws, dtype=bool)
    duplicate_counts = np.zeros(ensemble.n_draws, dtype=int)
    for index, (time_row, value_row) in enumerate(
        zip(ensemble.time_draws, ensemble.value_draws)
    ):
        if not input_valid[index]:
            prepared.append(None)
            continue
        try:
            draw_time, draw_value, duplicate_count = _prepare_time_value_pairs(
                time_row,
                value_row,
                duplicate_time_policy=duplicate_time_policy,
            )
        except ValueError:
            prepared.append(None)
            continue
        prepared.append((draw_time, draw_value))
        duplicate_counts[index] = duplicate_count

    usable = [item for item in prepared if item is not None]
    if not usable:
        raise ValueError('No trajectory draw has enough finite data for interpolation.')

    lower = max(float(item[0][0]) for item in usable)
    upper = min(float(item[0][-1]) for item in usable)
    if not lower < upper:
        raise ValueError('The valid trajectory draws do not share an overlapping time domain.')
    supported_range = (lower, upper)

    if time_grid is None:
        same_grid = all(
            len(item[0]) == len(usable[0][0])
            and np.allclose(item[0], usable[0][0], rtol=1e-10, atol=1e-12)
            for item in usable[1:]
        )
        if same_grid:
            candidate_grid = usable[0][0]
            candidate_grid = candidate_grid[
                (candidate_grid >= lower) & (candidate_grid <= upper)
            ]
            candidate_grid = np.unique(
                np.concatenate(([lower], candidate_grid, [upper]))
            )
        else:
            if grid_size is None:
                grid_size = max(100, len(ensemble.central.in_vivo_time))
            grid_size = int(grid_size)
            if grid_size < 2:
                raise ValueError('grid_size must be at least two.')
            candidate_grid = np.linspace(lower, upper, grid_size)
        common_grid = _validate_target_grid(
            candidate_grid,
            supported_range=supported_range,
        )
    else:
        common_grid = _validate_target_grid(
            time_grid,
            supported_range=supported_range,
        )

    value_draws = np.full((ensemble.n_draws, len(common_grid)), np.nan, dtype=float)
    valid_draws = np.zeros(ensemble.n_draws, dtype=bool)
    for index, item in enumerate(prepared):
        if item is None:
            continue
        draw_time, draw_value = item
        values = _interpolate_values(
            draw_time,
            draw_value,
            common_grid,
            method=interpolation,
            extrapolate=False,
        )
        if np.all(np.isfinite(values)):
            value_draws[index] = values
            valid_draws[index] = True

    if not np.any(valid_draws):
        raise ValueError('No trajectory draw covers the requested common time grid.')

    try:
        central_time, central_value, central_duplicate_count = _prepare_time_value_pairs(
            ensemble.central.in_vivo_time,
            ensemble.central.in_vivo_value,
            duplicate_time_policy=duplicate_time_policy,
        )
        central_curve = _interpolate_values(
            central_time,
            central_value,
            common_grid,
            method=interpolation,
            extrapolate=False,
        )
        if not np.all(np.isfinite(central_curve)):
            raise ValueError('Central trajectory does not cover the common grid.')
        central_source = 'input_central_trajectory'
    except ValueError:
        central_duplicate_count = 0
        if draw_weights is None:
            weights = valid_draws.astype(float)
        else:
            weights = np.asarray(draw_weights, dtype=float).copy()
            if weights.ndim != 1 or len(weights) != ensemble.n_draws:
                raise ValueError('draw_weights must contain one value per trajectory draw.')
            weights[~valid_draws] = 0.0
        if not np.sum(weights) > 0:
            weights = valid_draws.astype(float)
        central_curve = np.average(
            value_draws[valid_draws],
            axis=0,
            weights=weights[valid_draws],
        )
        central_source = 'weighted_draw_mean'

    metadata = dict(ensemble.metadata)
    metadata.update({
        'source_approach': ensemble.central.source_approach,
        'source_method': ensemble.central.source_method,
        'source_model_name': ensemble.central.model_name,
        'source_draw_count': ensemble.n_draws,
        'source_valid_draw_count': ensemble.n_valid_draws,
        'common_valid_draw_count': int(np.count_nonzero(valid_draws)),
        'common_supported_time_range': supported_range,
        'common_grid_interpolation': interpolation,
        'duplicate_time_policy': duplicate_time_policy,
        'duplicate_time_counts': duplicate_counts,
        'central_duplicate_time_count': central_duplicate_count,
        'central_curve_source': central_source,
    })

    return ReferenceTrajectoryEnsemble(
        time_grid=common_grid,
        value_draws=value_draws,
        valid_draws=valid_draws,
        draw_weights=draw_weights,
        central_curve=central_curve,
        supported_time_range=supported_range,
        interpolation=interpolation,
        metadata=metadata,
    )


def reference_from_time_course(
    time,
    value,
    *,
    interpolation: InterpolationMethod = 'pchip',
    duplicate_time_policy: DuplicateTimePolicy = 'mean',
    label: str | None = None,
) -> ReferenceTrajectoryEnsemble:
    """Create a deterministic reference ensemble from an external trajectory."""
    prepared_time, prepared_value, duplicate_count = _prepare_time_value_pairs(
        time,
        value,
        duplicate_time_policy=duplicate_time_policy,
    )
    central = PredictedTrajectory(
        source_approach='external',
        source_method='time_course',
        in_vitro_time=prepared_time,
        in_vitro_value=prepared_value,
        in_vivo_time=prepared_time,
        in_vivo_value=prepared_value,
        auxiliary={'label': label} if label is not None else {},
    )
    ensemble = deterministic_trajectory_ensemble(
        central,
        metadata={
            'uncertainty_source': 'none',
            'duplicate_time_count': duplicate_count,
            'label': label,
        },
    )
    return reference_from_trajectory_ensemble(
        ensemble,
        time_grid=prepared_time,
        interpolation=interpolation,
        duplicate_time_policy=duplicate_time_policy,
    )


def reference_from_profile_fit(
    fit: ProfileFit,
    time_grid,
    *,
    n_samples: int = 1000,
    random_state=None,
    interpolation: InterpolationMethod = 'pchip',
) -> ReferenceTrajectoryEnsemble:
    """Create a common-grid reference from one fitted in vivo profile."""
    ensemble = sample_profile_trajectory(
        fit,
        time_grid,
        n_samples=n_samples,
        random_state=random_state,
    )
    return reference_from_trajectory_ensemble(
        ensemble,
        time_grid=np.asarray(time_grid, dtype=float),
        interpolation=interpolation,
    )


def _evaluation_times(
    reference: ReferenceTrajectoryEnsemble,
    times,
    *,
    allow_extrapolation: bool,
) -> np.ndarray:
    target = np.atleast_1d(np.asarray(times, dtype=float))
    if target.ndim != 1 or np.any(~np.isfinite(target)):
        raise ValueError('Evaluation times must be a finite one-dimensional array.')
    if not allow_extrapolation:
        lower, upper = reference.supported_time_range
        tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
        if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
            raise ValueError('Evaluation times extend beyond the supported reference domain.')
    return target


def evaluate_reference_trajectories(
    reference: ReferenceTrajectoryEnsemble,
    times,
    *,
    draw_indices: Sequence[int] | None = None,
    method: InterpolationMethod | None = None,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Evaluate complete reference draws without breaking temporal dependence."""
    target = _evaluation_times(
        reference,
        times,
        allow_extrapolation=allow_extrapolation,
    )
    interpolation = reference.interpolation if method is None else method
    if interpolation not in ('linear', 'pchip'):
        raise ValueError("Interpolation method must be 'linear' or 'pchip'.")

    if draw_indices is None:
        indices = np.arange(reference.n_draws)
    else:
        indices = np.asarray(draw_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError('draw_indices must be one-dimensional.')
        if np.any(indices < 0) or np.any(indices >= reference.n_draws):
            raise IndexError('draw_indices contains an out-of-range index.')

    evaluated = np.full((len(indices), len(target)), np.nan, dtype=float)
    for output_index, draw_index in enumerate(indices):
        if not reference.valid_draws[draw_index]:
            continue
        evaluated[output_index] = _interpolate_values(
            reference.time_grid,
            reference.value_draws[draw_index],
            target,
            method=interpolation,
            extrapolate=allow_extrapolation,
        )
    return evaluated


def evaluate_central_trajectory(
    reference: ReferenceTrajectoryEnsemble,
    times,
    *,
    method: InterpolationMethod | None = None,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Evaluate the central reference trajectory at requested times."""
    target = _evaluation_times(
        reference,
        times,
        allow_extrapolation=allow_extrapolation,
    )
    interpolation = reference.interpolation if method is None else method
    return _interpolate_values(
        reference.time_grid,
        reference.central_curve,
        target,
        method=interpolation,
        extrapolate=allow_extrapolation,
    )
