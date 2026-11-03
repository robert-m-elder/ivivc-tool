"""Empirical IVV/IV time-scale distributions and complete reference curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from ivivc_core import (
    ParameterSamples,
    ProfileFit,
    TimeCourse,
    evaluate_curve_draws,
    get_model_spec,
    sample_parameter_sets,
)

from .reference import DuplicateTimePolicy, _interpolate_values, _prepare_time_value_pairs
from .types import InterpolationMethod, ReferenceTrajectoryEnsemble


@dataclass(frozen=True)
class EmpiricalTimeScaleDistribution:
    """A weighted empirical distribution of positive ``tau_IVV / tau_IV`` ratios.

    Records retain their input order and optional source identifiers, including
    duplicates and zero-weight records. ECDF operations use only positive mass.
    Identifiers are provenance labels; they do not group or reweight records.
    Metadata can describe material groups, units, and the historical tau
    definition. Both time constants must have compatible definitions/units.
    """

    ratios: np.ndarray
    weights: np.ndarray | None = None
    source_ids: Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ratios = np.asarray(self.ratios, dtype=float).copy()
        if ratios.ndim != 1 or len(ratios) == 0:
            raise ValueError('ratios must be a nonempty one-dimensional array.')
        if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
            raise ValueError('ratios must be finite and strictly positive.')
        weights = (np.ones(len(ratios), dtype=float) if self.weights is None
                   else np.asarray(self.weights, dtype=float).copy())
        if weights.ndim != 1 or len(weights) != len(ratios):
            raise ValueError('weights must contain one value per ratio.')
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError('weights must be finite, nonnegative, and have positive total weight.')
        positive = weights > 0
        # Normalize without overflowing the sum of large finite input weights.
        weights /= np.max(weights)
        weights /= np.sum(weights)
        if np.any(positive & (weights == 0)):
            raise ValueError('The weight range is too large to retain every positive weight.')
        identifiers = self.source_ids
        if identifiers is not None:
            if isinstance(identifiers, str):
                raise ValueError('source_ids must contain one string per ratio.')
            identifiers = tuple(identifiers)
            if len(identifiers) != len(ratios) or any(
                not isinstance(item, str) or not item.strip() for item in identifiers
            ):
                raise ValueError('source_ids must contain one nonempty string per ratio.')
        ratios.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, 'ratios', ratios)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'source_ids', identifiers)
        object.__setattr__(self, 'metadata', deepcopy(dict(self.metadata)))

    def ecdf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return positive-mass support and cumulative probabilities, without smoothing."""
        active = self.weights > 0
        support, inverse = np.unique(self.ratios[active], return_inverse=True)
        mass = np.bincount(inverse, weights=self.weights[active])
        cumulative = np.minimum(np.cumsum(mass), 1.0)
        cumulative[-1] = 1.0
        return support, cumulative

    def cdf(self, values):
        """Return ``P(ratio <= value)`` for scalar or array inputs."""
        values = np.asarray(values, dtype=float)
        if np.any(np.isnan(values)):
            raise ValueError('CDF values must not contain NaN.')
        support, cumulative = self.ecdf()
        positions = np.searchsorted(support, values, side='right')
        result = np.concatenate(([0.0], cumulative))[positions]
        return float(result) if values.ndim == 0 else result

    def quantile(self, probabilities):
        """Return inverse weighted-ECDF quantiles; endpoints use active extrema."""
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any(~np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError('Quantile probabilities must be finite and between zero and one.')
        support, cumulative = self.ecdf()
        positions = np.searchsorted(cumulative, probabilities, side='left')
        positions = np.where(probabilities == 1, len(support) - 1, positions)
        result = support[positions]
        return float(result) if probabilities.ndim == 0 else result


def _profile_parameters(fit: ProfileFit, spec) -> dict[str, float]:
    """Check the nominal fit independently of whether covariance is propagated."""
    if tuple(fit.parameter_names) != spec.parameter_names or set(fit.parameters) != set(spec.parameter_names):
        raise ValueError('ProfileFit parameters must match the registered model and parameter order.')
    parameters = {name: float(fit.parameters[name]) for name in spec.parameter_names}
    vector = np.asarray(list(parameters.values()))
    if np.any(~np.isfinite(vector)):
        raise ValueError('The source fitted parameters must be finite.')
    bounds = spec.parameter_bounds
    if bounds is None:
        bounds = spec.fit_kwargs.get('bounds')
    if bounds is not None:
        lower, upper = (np.broadcast_to(np.asarray(bound, dtype=float), vector.shape) for bound in bounds)
        if np.any(vector < lower) or np.any(vector > upper):
            raise ValueError('The source fitted parameters violate registered model bounds.')
    if spec.parameter_validator is not None and not spec.parameter_validator(parameters):
        raise ValueError('The source fitted parameters fail the registered model validator.')
    return parameters


def reference_from_time_scaling(
    in_vitro: ProfileFit | TimeCourse,
    distribution: EmpiricalTimeScaleDistribution | Sequence[float],
    time_grid,
    *,
    n_parameter_draws: int = 0,
    random_state=None,
    allow_extrapolation: bool = False,
    interpolation: InterpolationMethod = 'pchip',
    duplicate_time_policy: DuplicateTimePolicy = 'mean',
    required_times=None,
) -> ReferenceTrajectoryEnsemble:
    """Evaluate complete in vivo curves as ``f_IV(time_grid / ratio)``.

    A plain ratio sequence is equally weighted. By default the supplied fitted
    curve is fixed and its covariance is not required. With a positive
    ``n_parameter_draws=K``, reuse K joint fitted-parameter draws for every
    active ratio, with each complete trajectory weighted ``ratio_weight / K``.
    This Cartesian product assumes independent source-fit and scale uncertainty
    and preserves the empirical ratio marginal instead of resampling ratios.

    Requested times must be supported by every positive-weight record. Explicit
    extrapolation permits fitted-model evaluations beyond observed in vitro
    times; external TimeCourse data always require interpolation inside their
    supplied domain. Invalid positive-weight curves cause an error rather than
    dropping ratio mass. Responses are neither clipped nor renormalized, and
    no target-curve tau is calculated. Source and study times must use compatible
    units and the same elapsed-time origin on unlogged axes.
    """
    if not isinstance(in_vitro, (ProfileFit, TimeCourse)):
        raise TypeError('in_vitro must be a ProfileFit or TimeCourse.')
    if not isinstance(distribution, EmpiricalTimeScaleDistribution):
        distribution = EmpiricalTimeScaleDistribution(distribution)
    if isinstance(n_parameter_draws, bool) or not isinstance(n_parameter_draws, Integral) or n_parameter_draws < 0:
        raise ValueError('n_parameter_draws must be a nonnegative integer.')
    n_parameter_draws = int(n_parameter_draws)
    if not isinstance(allow_extrapolation, (bool, np.bool_)):
        raise ValueError('allow_extrapolation must be a boolean.')
    if interpolation not in ('linear', 'pchip'):
        raise ValueError("interpolation must be 'linear' or 'pchip'.")
    if duplicate_time_policy not in ('mean', 'first', 'error'):
        raise ValueError("duplicate_time_policy must be 'mean', 'first', or 'error'.")
    grid = np.asarray(time_grid, dtype=float).copy()
    if grid.ndim != 1 or len(grid) < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError('time_grid must contain at least two finite strictly increasing times.')
    if required_times is not None:
        required = np.asarray(required_times, dtype=float)
        if (required.ndim != 1 or np.any(~np.isfinite(required))
                or np.any(required < grid[0]) or np.any(required > grid[-1])):
            raise ValueError('required_times must be finite and lie within the reference time grid.')

    parameters = None
    covariance = None
    samples = None
    duplicate_count = 0
    if isinstance(in_vitro, ProfileFit):
        source_time = np.asarray(in_vitro.time, dtype=float)
        if (source_time.ndim != 1 or np.any(~np.isfinite(source_time))
                or len(np.unique(source_time)) < 2):
            raise ValueError('The source fit must contain at least two distinct finite observed times.')
        spec = get_model_spec(in_vitro.model_name)
        parameters = _profile_parameters(in_vitro, spec)
        if not n_parameter_draws:
            samples = ParameterSamples(spec.parameter_names, np.asarray([list(parameters.values())]))
    else:
        if n_parameter_draws:
            raise ValueError('Parameter uncertainty requires a ProfileFit, not a TimeCourse.')
        source_time, source_value, duplicate_count = _prepare_time_value_pairs(
            in_vitro.time, in_vitro.value, duplicate_time_policy=duplicate_time_policy,
        )

    source_range = (float(np.min(source_time)), float(np.max(source_time)))
    active = np.flatnonzero(distribution.weights > 0)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        mapped = grid[None, :] / distribution.ratios[active, None]
    if np.any(~np.isfinite(mapped)):
        raise ValueError('Time scaling requires nonfinite in vitro times for a positive-weight ratio.')
    tolerance = 16 * np.finfo(float).eps * max(
        abs(source_range[0]), abs(source_range[1]), np.finfo(float).tiny,
    )
    outside = (mapped < source_range[0] - tolerance) | (mapped > source_range[1] + tolerance)
    if np.any(outside) and (not allow_extrapolation or isinstance(in_vitro, TimeCourse)):
        bad = active[np.any(outside, axis=1)]
        raise ValueError(
            'The requested horizon requires extrapolation beyond the in vitro source '
            f'time range {source_range!r} for positive-weight ratio records {bad.tolist()}. '
            'Only a ProfileFit permits explicit allow_extrapolation=True.'
        )
    if not allow_extrapolation or isinstance(in_vitro, TimeCourse):
        # Roundoff at a declared boundary must not trigger interpolator NaNs.
        evaluation_times = np.clip(mapped, *source_range)
    else:
        evaluation_times = mapped

    # Validate the requested domain before consuming the caller's random stream.
    if n_parameter_draws:
        covariance = np.asarray(in_vitro.covariance, dtype=float).copy()
        samples = sample_parameter_sets(
            parameters, covariance, model_name=in_vitro.model_name,
            n_samples=n_parameter_draws, random_state=random_state,
        )
    count = samples.n_samples if samples is not None else 1
    draw_ratio_indices = np.repeat(active, count)
    draw_weights = np.repeat(distribution.weights[active] / count, count)
    if np.any(draw_weights == 0):
        raise ValueError('Parameter expansion cannot retain every positive ratio weight.')
    values = np.empty((len(draw_weights), len(grid)), dtype=float)
    for position, source_index in enumerate(active):
        if samples is not None:
            curves = evaluate_curve_draws(spec.model_function, evaluation_times[position], samples)
            if not np.all(curves.valid_draws):
                raise ValueError(
                    f'A nonfinite or invalid source curve occurred for ratio record {source_index}; '
                    'positive-weight trajectories were not discarded.'
                )
            value_rows = curves.values
        else:
            value_rows = _interpolate_values(
                source_time, source_value, evaluation_times[position],
                method=interpolation, extrapolate=False,
            )[None, :]
        if np.any(~np.isfinite(value_rows)):
            raise ValueError(f'A nonfinite source curve occurred for ratio record {source_index}.')
        values[position * count:(position + 1) * count] = value_rows
    central = np.average(values, axis=0, weights=draw_weights)
    if np.any(~np.isfinite(central)):
        raise ValueError('The weighted central reference curve is not finite.')

    metadata = {
        'reference_source': 'empirical_time_scaling',
        'ratio_definition': 'tau_in_vivo / tau_in_vitro',
        'source_model_name': in_vitro.model_name if isinstance(in_vitro, ProfileFit) else None,
        'source_label': in_vitro.label if isinstance(in_vitro, TimeCourse) else None,
        'source_time_range': source_range,
        'source_parameters': parameters,
        'source_covariance': covariance,
        'source_duplicate_time_count': duplicate_count,
        'time_scale_ratios': distribution.ratios.copy(),
        'time_scale_weights': distribution.weights.copy(),
        'time_scale_source_ids': distribution.source_ids,
        'time_scale_metadata': deepcopy(distribution.metadata),
        'active_ratio_indices': active.copy(),
        'draw_ratio_indices': draw_ratio_indices,
        'draw_parameter_indices': np.tile(np.arange(count), len(active)),
        'parameter_samples': None if samples is None else samples.values.copy(),
        'n_parameter_draws': n_parameter_draws,
        'parameter_sampling_attempts': 0 if not n_parameter_draws else samples.attempted_samples,
        'parameter_rejection_counts': {} if not n_parameter_draws else dict(samples.rejection_counts),
        'parameter_sampling_seed': (int(random_state) if n_parameter_draws
                                    and isinstance(random_state, Integral) else None),
        'source_fit_and_scale_independent': True,
        'allow_extrapolation': bool(allow_extrapolation),
        'required_in_vitro_time_ranges': mapped[:, [0, -1]].copy(),
        'extrapolated_points_by_ratio': outside.copy(),
        'central_curve_source': 'weighted_draw_mean',
    }
    return ReferenceTrajectoryEnsemble(
        time_grid=grid, value_draws=values, valid_draws=np.ones(len(values), dtype=bool),
        draw_weights=draw_weights, central_curve=central,
        interpolation=interpolation, metadata=metadata,
    )
