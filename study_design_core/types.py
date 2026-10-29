"""Structured numerical types for study-design calculations.

These objects are intentionally independent of Flask, Plotly, and the legacy
IVIVC result dictionaries.  They form the boundary between the existing
``ivivc_core`` calculations and the future time-point optimization engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np


InterpolationMethod = Literal['linear', 'pchip']
ObservationModelKind = Literal['constant_sd', 'constant_cv']


@dataclass
class ReferenceTrajectoryEnsemble:
    """Plausible in vivo trajectories evaluated on one common time grid.

    Rows of ``value_draws`` are complete plausible true trajectories.  Keeping
    each row intact preserves dependence across time.  Invalid rows can remain
    in the object for traceability but receive zero weight.
    """

    time_grid: np.ndarray
    value_draws: np.ndarray
    valid_draws: np.ndarray
    draw_weights: np.ndarray | None = None
    central_curve: np.ndarray | None = None
    supported_time_range: tuple[float, float] | None = None
    interpolation: InterpolationMethod = 'pchip'
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_grid = np.asarray(self.time_grid, dtype=float).copy()
        self.value_draws = np.asarray(self.value_draws, dtype=float).copy()
        self.valid_draws = np.asarray(self.valid_draws, dtype=bool).copy()

        if self.time_grid.ndim != 1 or len(self.time_grid) < 2:
            raise ValueError('Reference time_grid must be one-dimensional with at least two points.')
        if not np.all(np.isfinite(self.time_grid)):
            raise ValueError('Reference time_grid must contain only finite values.')
        if np.any(np.diff(self.time_grid) <= 0):
            raise ValueError('Reference time_grid must be strictly increasing.')

        if self.value_draws.ndim == 1:
            self.value_draws = self.value_draws.reshape(1, -1)
        if self.value_draws.ndim != 2:
            raise ValueError('Reference value_draws must be a two-dimensional array.')
        if self.value_draws.shape[1] != len(self.time_grid):
            raise ValueError('Each reference trajectory must contain one value per time-grid point.')
        if self.valid_draws.ndim != 1 or len(self.valid_draws) != self.value_draws.shape[0]:
            raise ValueError('valid_draws must contain one flag per reference trajectory.')

        finite_rows = np.all(np.isfinite(self.value_draws), axis=1)
        self.valid_draws &= finite_rows
        if not np.any(self.valid_draws):
            raise ValueError('At least one finite, valid reference trajectory is required.')

        if self.draw_weights is None:
            weights = self.valid_draws.astype(float)
        else:
            weights = np.asarray(self.draw_weights, dtype=float).copy()
            if weights.ndim != 1 or len(weights) != self.value_draws.shape[0]:
                raise ValueError('draw_weights must contain one value per trajectory draw.')
            if np.any(~np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError('draw_weights must be finite and nonnegative.')
            weights[~self.valid_draws] = 0.0
        total_weight = float(np.sum(weights))
        if not total_weight > 0:
            raise ValueError('At least one valid reference trajectory must have positive weight.')
        self.draw_weights = weights / total_weight

        if self.central_curve is None:
            self.central_curve = np.average(
                self.value_draws[self.valid_draws],
                axis=0,
                weights=self.draw_weights[self.valid_draws],
            )
        else:
            self.central_curve = np.asarray(self.central_curve, dtype=float).copy()
            if self.central_curve.ndim != 1 or len(self.central_curve) != len(self.time_grid):
                raise ValueError('central_curve must contain one value per time-grid point.')
            if not np.all(np.isfinite(self.central_curve)):
                raise ValueError('central_curve must contain only finite values.')

        if self.supported_time_range is None:
            supported_range = (float(self.time_grid[0]), float(self.time_grid[-1]))
        else:
            if len(self.supported_time_range) != 2:
                raise ValueError('supported_time_range must contain exactly two values.')
            supported_range = tuple(float(value) for value in self.supported_time_range)
        if not all(np.isfinite(supported_range)) or supported_range[0] >= supported_range[1]:
            raise ValueError('supported_time_range must be finite and increasing.')
        if (
            supported_range[0] < self.time_grid[0]
            or supported_range[1] > self.time_grid[-1]
        ):
            raise ValueError('supported_time_range must lie within the common time grid.')
        self.supported_time_range = supported_range

        if self.interpolation not in ('linear', 'pchip'):
            raise ValueError("interpolation must be either 'linear' or 'pchip'.")

        copied_metadata: dict[str, Any] = {}
        for key, value in dict(self.metadata).items():
            copied_metadata[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.metadata = copied_metadata

    @property
    def n_draws(self) -> int:
        return int(self.value_draws.shape[0])

    @property
    def n_valid_draws(self) -> int:
        return int(np.count_nonzero(self.valid_draws))

    @property
    def valid_weights(self) -> np.ndarray:
        return self.draw_weights[self.valid_draws].copy()

    def evaluate(
        self,
        times,
        *,
        draw_indices: Sequence[int] | None = None,
        method: InterpolationMethod | None = None,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        """Evaluate selected complete trajectories at arbitrary in-range times."""
        from .reference import evaluate_reference_trajectories

        return evaluate_reference_trajectories(
            self,
            times,
            draw_indices=draw_indices,
            method=method,
            allow_extrapolation=allow_extrapolation,
        )

    def evaluate_central(
        self,
        times,
        *,
        method: InterpolationMethod | None = None,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        """Evaluate the central reference curve at arbitrary in-range times."""
        from .reference import evaluate_central_trajectory

        return evaluate_central_trajectory(
            self,
            times,
            method=method,
            allow_extrapolation=allow_extrapolation,
        )


@dataclass(frozen=True)
class ObservationModel:
    """Future-study variability around one assumed true trajectory."""

    kind: ObservationModelKind
    value: float
    replicates_per_time: int = 1
    minimum_sd: float = 0.0
    source: str = 'user'

    def __post_init__(self) -> None:
        if self.kind not in ('constant_sd', 'constant_cv'):
            raise ValueError("Observation-model kind must be 'constant_sd' or 'constant_cv'.")
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError('Observation-model value must be finite and nonnegative.')
        if int(self.replicates_per_time) != self.replicates_per_time or self.replicates_per_time < 1:
            raise ValueError('replicates_per_time must be a positive integer.')
        if not np.isfinite(self.minimum_sd) or self.minimum_sd < 0:
            raise ValueError('minimum_sd must be finite and nonnegative.')

    def standard_deviation(self, expected_value) -> np.ndarray:
        """Return the observation SD at each expected response value."""
        expected = np.asarray(expected_value, dtype=float)
        if self.kind == 'constant_sd':
            sd = np.full(expected.shape, self.value, dtype=float)
        else:
            sd = self.value * np.abs(expected)
        return np.maximum(sd, self.minimum_sd)

    def variance_of_mean(self, expected_value) -> np.ndarray:
        """Return the variance of a mean across the configured replicates."""
        sd = self.standard_deviation(expected_value)
        return sd ** 2 / int(self.replicates_per_time)

    def simulate(
        self,
        expected_value,
        *,
        random_state=None,
        replicates_per_time: int | None = None,
    ) -> np.ndarray:
        """Generate independent replicate observations around expected values.

        The returned shape is ``(replicates, *expected_value.shape)``.
        """
        expected = np.asarray(expected_value, dtype=float)
        replicates = (
            int(self.replicates_per_time)
            if replicates_per_time is None
            else int(replicates_per_time)
        )
        if replicates < 1:
            raise ValueError('replicates_per_time must be at least one.')
        rng = np.random.default_rng(random_state)
        loc = np.broadcast_to(expected, (replicates,) + expected.shape)
        scale = np.broadcast_to(
            self.standard_deviation(expected),
            (replicates,) + expected.shape,
        )
        return rng.normal(loc=loc, scale=scale)


@dataclass
class ExistingObservations:
    """Completed observations that remain fixed during adaptive design.

    Repeated times are allowed so replicate-level observations can be retained
    exactly as measured.  These values are never resimulated by the Monte
    Carlo study-design evaluator.
    """

    time: np.ndarray
    response: np.ndarray
    label: str = 'completed in vivo observations'

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).copy()
        self.response = np.asarray(self.response, dtype=float).copy()
        if self.time.ndim != 1 or self.response.ndim != 1:
            raise ValueError('Existing observation time and response must be one-dimensional.')
        if len(self.time) == 0 or len(self.time) != len(self.response):
            raise ValueError('Existing observations require equally sized nonempty time and response arrays.')
        if np.any(~np.isfinite(self.time)) or np.any(~np.isfinite(self.response)):
            raise ValueError('Existing observations must contain only finite values.')
        self.label = str(self.label)

    @property
    def unique_times(self) -> np.ndarray:
        return np.unique(self.time)

    @property
    def latest_time(self) -> float:
        return float(np.max(self.time))

    @property
    def n_observations(self) -> int:
        return int(len(self.time))

    @property
    def n_time_points(self) -> int:
        return int(len(self.unique_times))

    def mean_response_by_time(self) -> tuple[np.ndarray, np.ndarray]:
        times = self.unique_times
        means = np.asarray([np.mean(self.response[self.time == value]) for value in times])
        return times, means


@dataclass
class ReferenceUpdateResult:
    """Likelihood reweighting of a reference ensemble by completed data."""

    reference: ReferenceTrajectoryEnsemble
    prior_weights: np.ndarray
    posterior_weights: np.ndarray
    log_likelihood: np.ndarray
    effective_sample_size: float
    log_normalizing_constant: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prior_weights = np.asarray(self.prior_weights, dtype=float).copy()
        self.posterior_weights = np.asarray(self.posterior_weights, dtype=float).copy()
        self.log_likelihood = np.asarray(self.log_likelihood, dtype=float).copy()
        expected = self.reference.n_draws
        for name, values in (
            ('prior_weights', self.prior_weights),
            ('posterior_weights', self.posterior_weights),
            ('log_likelihood', self.log_likelihood),
        ):
            if values.ndim != 1 or len(values) != expected:
                raise ValueError(f'{name} must contain one value per reference draw.')
        if np.any(~np.isfinite(self.prior_weights)) or np.any(self.prior_weights < 0):
            raise ValueError('prior_weights must be finite and nonnegative.')
        if np.any(~np.isfinite(self.posterior_weights)) or np.any(self.posterior_weights < 0):
            raise ValueError('posterior_weights must be finite and nonnegative.')
        if not np.isclose(np.sum(self.posterior_weights), 1.0):
            raise ValueError('posterior_weights must sum to one.')
        if not np.isfinite(self.effective_sample_size) or self.effective_sample_size <= 0:
            raise ValueError('effective_sample_size must be finite and positive.')
        self.effective_sample_size = float(self.effective_sample_size)
        self.log_normalizing_constant = float(self.log_normalizing_constant)
        self.metadata = dict(self.metadata)


@dataclass
class DesignConstraints:
    """User-defined numerical constraints for future time-point selection.

    These fields are not a regulatory rules engine.  They simply encode the
    design limits the user chooses to impose for one optimization run.
    """

    candidate_times: np.ndarray
    fixed_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    min_time: float | None = None
    max_time: float | None = None
    minimum_spacing: float = 0.0
    min_number_of_times: int = 1
    max_number_of_times: int | None = None
    replicates_per_time: int = 1

    def __post_init__(self) -> None:
        candidates = np.asarray(self.candidate_times, dtype=float)
        fixed = np.asarray(self.fixed_times, dtype=float)
        if candidates.ndim != 1 or fixed.ndim != 1:
            raise ValueError('candidate_times and fixed_times must be one-dimensional.')
        if len(candidates) == 0:
            raise ValueError('At least one candidate time is required.')
        if np.any(~np.isfinite(candidates)) or np.any(~np.isfinite(fixed)):
            raise ValueError('Design times must be finite.')
        self.candidate_times = np.unique(candidates)
        self.fixed_times = np.unique(fixed)

        combined = np.union1d(self.candidate_times, self.fixed_times)
        min_time = float(np.min(combined)) if self.min_time is None else float(self.min_time)
        max_time = float(np.max(combined)) if self.max_time is None else float(self.max_time)
        if not np.isfinite(min_time) or not np.isfinite(max_time) or min_time >= max_time:
            raise ValueError('min_time and max_time must be finite and increasing.')
        if np.any(self.candidate_times < min_time) or np.any(self.candidate_times > max_time):
            raise ValueError('All candidate times must lie within min_time and max_time.')
        if np.any(self.fixed_times < min_time) or np.any(self.fixed_times > max_time):
            raise ValueError('All fixed times must lie within min_time and max_time.')
        self.min_time = min_time
        self.max_time = max_time

        self.minimum_spacing = float(self.minimum_spacing)
        if not np.isfinite(self.minimum_spacing) or self.minimum_spacing < 0:
            raise ValueError('minimum_spacing must be finite and nonnegative.')
        if len(self.fixed_times) > 1 and np.any(np.diff(self.fixed_times) < self.minimum_spacing):
            raise ValueError('Fixed times violate the requested minimum spacing.')

        self.min_number_of_times = int(self.min_number_of_times)
        if self.min_number_of_times < 1:
            raise ValueError('min_number_of_times must be at least one.')
        maximum_available = len(combined)
        if self.max_number_of_times is None:
            self.max_number_of_times = maximum_available
        else:
            self.max_number_of_times = int(self.max_number_of_times)
        if self.max_number_of_times < self.min_number_of_times:
            raise ValueError('max_number_of_times cannot be smaller than min_number_of_times.')
        if self.max_number_of_times > maximum_available:
            raise ValueError('max_number_of_times exceeds the number of available unique times.')
        if self.max_number_of_times < len(self.fixed_times):
            raise ValueError('max_number_of_times cannot be smaller than the number of fixed times.')
        if self.min_number_of_times < len(self.fixed_times):
            self.min_number_of_times = len(self.fixed_times)

        self.replicates_per_time = int(self.replicates_per_time)
        if self.replicates_per_time < 1:
            raise ValueError('replicates_per_time must be at least one.')

    @property
    def all_available_times(self) -> np.ndarray:
        return np.union1d(self.candidate_times, self.fixed_times)

    def available_candidates(self, selected_times=()) -> np.ndarray:
        """Return candidate times compatible with selected and fixed times."""
        selected = np.union1d(np.asarray(selected_times, dtype=float), self.fixed_times)
        available = []
        for candidate in self.candidate_times:
            if np.any(np.isclose(selected, candidate, rtol=0.0, atol=1e-12)):
                continue
            if len(selected) and np.any(np.abs(selected - candidate) < self.minimum_spacing):
                continue
            available.append(candidate)
        return np.asarray(available, dtype=float)

    def validate_design(self, times) -> np.ndarray:
        """Validate and return a sorted unique candidate design."""
        design = np.unique(np.asarray(times, dtype=float))
        if design.ndim != 1 or np.any(~np.isfinite(design)):
            raise ValueError('Design times must be a finite one-dimensional array.')
        if len(design) < self.min_number_of_times or len(design) > self.max_number_of_times:
            raise ValueError('Design contains an invalid number of distinct time points.')
        if np.any(design < self.min_time) or np.any(design > self.max_time):
            raise ValueError('Design times lie outside the allowed time range.')
        allowed = self.all_available_times
        if not all(np.any(np.isclose(allowed, value, rtol=0.0, atol=1e-12)) for value in design):
            raise ValueError('Design includes a time that is not fixed or selectable.')
        if not all(np.any(np.isclose(design, value, rtol=0.0, atol=1e-12)) for value in self.fixed_times):
            raise ValueError('Design omits one or more fixed times.')
        if len(design) > 1 and np.any(np.diff(design) < self.minimum_spacing):
            raise ValueError('Design violates the requested minimum spacing.')
        return design

    def total_experimental_units(self, number_of_times: int) -> int:
        number_of_times = int(number_of_times)
        if number_of_times < 0:
            raise ValueError('number_of_times must be nonnegative.')
        return number_of_times * self.replicates_per_time


@dataclass
class ModelFitResult:
    """Lightweight result from one repeated nonlinear model fit."""

    model_name: str
    success: bool
    time: np.ndarray
    response: np.ndarray
    parameter_names: tuple[str, ...]
    parameters: Mapping[str, float]
    covariance: np.ndarray
    initial_parameters: np.ndarray | None
    fitted_values: np.ndarray
    residuals: np.ndarray
    residual_dof: int
    residual_sd: float
    used_grid_search: bool = False
    message: str | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).copy()
        self.response = np.asarray(self.response, dtype=float).copy()
        self.parameter_names = tuple(self.parameter_names)
        self.parameters = dict(self.parameters)
        self.covariance = np.asarray(self.covariance, dtype=float).copy()
        self.initial_parameters = (
            None
            if self.initial_parameters is None
            else np.asarray(self.initial_parameters, dtype=float).copy()
        )
        self.fitted_values = np.asarray(self.fitted_values, dtype=float).copy()
        self.residuals = np.asarray(self.residuals, dtype=float).copy()
        if len(self.time) != len(self.response):
            raise ValueError('Fit time and response arrays must have matching lengths.')
        if len(self.fitted_values) != len(self.time) or len(self.residuals) != len(self.time):
            raise ValueError('Fitted values and residuals must match the fitted data length.')
        expected_covariance_shape = (len(self.parameter_names), len(self.parameter_names))
        if self.covariance.shape != expected_covariance_shape:
            raise ValueError('Fit covariance has the wrong shape for the model parameters.')
        self.residual_dof = int(self.residual_dof)
        self.residual_sd = float(self.residual_sd)

    @property
    def parameter_vector(self) -> np.ndarray:
        if not self.success:
            return np.full(len(self.parameter_names), np.nan, dtype=float)
        return np.asarray([self.parameters[name] for name in self.parameter_names], dtype=float)

    def predict(self, time) -> np.ndarray:
        """Evaluate the fitted model at new times."""
        if not self.success:
            raise RuntimeError(self.message or 'The model fit was unsuccessful.')
        from ivivc_core import get_model_spec

        spec = get_model_spec(self.model_name)
        return np.asarray(spec.model_function(np.asarray(time, dtype=float), **self.parameters))


@dataclass
class ReferenceModelEnsemble:
    """Analysis-model parameter draws fitted to plausible reference curves.

    Each valid row of ``parameter_draws`` corresponds to one complete reference
    trajectory.  The rows and weights retain the same uncertainty structure as
    the source trajectory ensemble after unsuccessful model fits are excluded.
    """

    model_name: str
    parameter_names: tuple[str, ...]
    parameter_draws: np.ndarray
    valid_draws: np.ndarray
    draw_weights: np.ndarray
    central_parameters: np.ndarray
    time_grid: np.ndarray
    supported_time_range: tuple[float, float]
    fit_messages: Sequence[str | None] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameter_names = tuple(self.parameter_names)
        self.parameter_draws = np.asarray(self.parameter_draws, dtype=float).copy()
        self.valid_draws = np.asarray(self.valid_draws, dtype=bool).copy()
        self.draw_weights = np.asarray(self.draw_weights, dtype=float).copy()
        self.central_parameters = np.asarray(self.central_parameters, dtype=float).copy()
        self.time_grid = np.asarray(self.time_grid, dtype=float).copy()

        if self.parameter_draws.ndim == 1:
            self.parameter_draws = self.parameter_draws.reshape(1, -1)
        if self.parameter_draws.ndim != 2:
            raise ValueError('parameter_draws must be a two-dimensional array.')
        if self.parameter_draws.shape[1] != len(self.parameter_names):
            raise ValueError('parameter_draws has the wrong number of model parameters.')
        if self.valid_draws.ndim != 1 or len(self.valid_draws) != self.parameter_draws.shape[0]:
            raise ValueError('valid_draws must contain one flag per parameter draw.')
        if self.draw_weights.ndim != 1 or len(self.draw_weights) != self.parameter_draws.shape[0]:
            raise ValueError('draw_weights must contain one value per parameter draw.')

        finite_rows = np.all(np.isfinite(self.parameter_draws), axis=1)
        self.valid_draws &= finite_rows
        if not np.any(self.valid_draws):
            raise ValueError('At least one finite, valid parameter draw is required.')
        if np.any(~np.isfinite(self.draw_weights)) or np.any(self.draw_weights < 0):
            raise ValueError('draw_weights must be finite and nonnegative.')
        self.draw_weights[~self.valid_draws] = 0.0
        weight_total = float(np.sum(self.draw_weights))
        if not weight_total > 0:
            raise ValueError('At least one valid parameter draw must have positive weight.')
        self.draw_weights /= weight_total

        if self.central_parameters.ndim != 1 or len(self.central_parameters) != len(self.parameter_names):
            raise ValueError('central_parameters must contain one value per model parameter.')
        if np.any(~np.isfinite(self.central_parameters)):
            raise ValueError('central_parameters must be finite.')

        if self.time_grid.ndim != 1 or len(self.time_grid) < 2:
            raise ValueError('time_grid must contain at least two points.')
        if np.any(~np.isfinite(self.time_grid)) or np.any(np.diff(self.time_grid) <= 0):
            raise ValueError('time_grid must be finite and strictly increasing.')

        if len(self.supported_time_range) != 2:
            raise ValueError('supported_time_range must contain exactly two values.')
        supported = tuple(float(value) for value in self.supported_time_range)
        if not all(np.isfinite(supported)) or supported[0] >= supported[1]:
            raise ValueError('supported_time_range must be finite and increasing.')
        if supported[0] < self.time_grid[0] or supported[1] > self.time_grid[-1]:
            raise ValueError('supported_time_range must lie within time_grid.')
        self.supported_time_range = supported

        messages = tuple(self.fit_messages)
        if messages and len(messages) != self.parameter_draws.shape[0]:
            raise ValueError('fit_messages must contain one entry per parameter draw.')
        if not messages:
            messages = tuple(None for _ in range(self.parameter_draws.shape[0]))
        self.fit_messages = messages

        copied_metadata: dict[str, Any] = {}
        for key, value in dict(self.metadata).items():
            copied_metadata[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.metadata = copied_metadata

    @property
    def n_draws(self) -> int:
        return int(self.parameter_draws.shape[0])

    @property
    def n_valid_draws(self) -> int:
        return int(np.count_nonzero(self.valid_draws))

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def valid_weights(self) -> np.ndarray:
        return self.draw_weights[self.valid_draws].copy()

    @property
    def central_parameter_mapping(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, self.central_parameters))


@dataclass
class DesignInformationScore:
    """Information diagnostics for one candidate design.

    Ensemble-based scoring stores one row per reference-parameter draw. Local
    scoring uses a single row corresponding to the fitted central reference.
    """

    criterion: float
    parameter_count: int
    per_draw_criteria: np.ndarray
    per_draw_ranks: np.ndarray
    per_draw_log_pseudodeterminants: np.ndarray
    valid_draws: np.ndarray
    draw_weights: np.ndarray

    def __post_init__(self) -> None:
        self.criterion = float(self.criterion)
        self.parameter_count = int(self.parameter_count)
        self.per_draw_criteria = np.asarray(self.per_draw_criteria, dtype=float).copy()
        self.per_draw_ranks = np.asarray(self.per_draw_ranks, dtype=int).copy()
        self.per_draw_log_pseudodeterminants = np.asarray(
            self.per_draw_log_pseudodeterminants,
            dtype=float,
        ).copy()
        self.valid_draws = np.asarray(self.valid_draws, dtype=bool).copy()
        self.draw_weights = np.asarray(self.draw_weights, dtype=float).copy()

        count = len(self.valid_draws)
        for values, label in (
            (self.per_draw_criteria, 'per_draw_criteria'),
            (self.per_draw_ranks, 'per_draw_ranks'),
            (self.per_draw_log_pseudodeterminants, 'per_draw_log_pseudodeterminants'),
            (self.draw_weights, 'draw_weights'),
        ):
            if values.ndim != 1 or len(values) != count:
                raise ValueError(f'{label} must contain one value per draw.')
        if self.parameter_count < 1:
            raise ValueError('parameter_count must be at least one.')
        if np.any(self.per_draw_ranks < 0) or np.any(self.per_draw_ranks > self.parameter_count):
            raise ValueError('Per-draw information ranks are outside the valid range.')
        if np.any(~np.isfinite(self.draw_weights)) or np.any(self.draw_weights < 0):
            raise ValueError('draw_weights must be finite and nonnegative.')
        self.draw_weights[~self.valid_draws] = 0.0
        total = float(np.sum(self.draw_weights))
        if not total > 0:
            raise ValueError('At least one valid draw must have positive weight.')
        self.draw_weights /= total

    @property
    def full_rank_fraction(self) -> float:
        mask = self.valid_draws & (self.per_draw_ranks == self.parameter_count)
        return float(np.sum(self.draw_weights[mask]))

    @property
    def min_rank(self) -> int:
        effective = self.valid_draws & (self.draw_weights > 0)
        return int(np.min(self.per_draw_ranks[effective]))

    @property
    def mean_rank(self) -> float:
        return float(np.sum(self.draw_weights * self.per_draw_ranks))

    @property
    def mean_log_pseudodeterminant(self) -> float:
        values = self.per_draw_log_pseudodeterminants
        mask = self.valid_draws & (self.draw_weights > 0) & np.isfinite(values)
        if not np.any(mask):
            return float('-inf')
        weights = self.draw_weights[mask]
        weights = weights / np.sum(weights)
        return float(np.sum(weights * values[mask]))

    @property
    def is_full_rank(self) -> bool:
        return bool(np.isclose(self.full_rank_fraction, 1.0, rtol=0.0, atol=1e-12))


@dataclass
class DesignStep:
    """One optimized design at a particular number of distinct time points."""

    times: np.ndarray
    score: DesignInformationScore
    method: str
    added_time: float | None = None
    n_exchanges: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.unique(np.asarray(self.times, dtype=float))
        if self.times.ndim != 1 or len(self.times) == 0 or np.any(~np.isfinite(self.times)):
            raise ValueError('DesignStep times must be a nonempty finite one-dimensional array.')
        if self.added_time is not None:
            self.added_time = float(self.added_time)
        self.n_exchanges = int(self.n_exchanges)
        if self.n_exchanges < 0:
            raise ValueError('n_exchanges must be nonnegative.')
        self.metadata = dict(self.metadata)

    @property
    def number_of_times(self) -> int:
        return int(len(self.times))


@dataclass
class CoordinateExchangeResult:
    """Result of single-coordinate exchange refinement at fixed design size."""

    initial_times: np.ndarray
    times: np.ndarray
    score: DesignInformationScore
    exchanges: Sequence[tuple[float, float]] = field(default_factory=tuple)
    converged: bool = True

    def __post_init__(self) -> None:
        self.initial_times = np.unique(np.asarray(self.initial_times, dtype=float))
        self.times = np.unique(np.asarray(self.times, dtype=float))
        if len(self.initial_times) != len(self.times):
            raise ValueError('Coordinate exchange must preserve the number of time points.')
        self.exchanges = tuple((float(old), float(new)) for old, new in self.exchanges)
        self.converged = bool(self.converged)


@dataclass
class DesignOptimizationResult:
    """Nested greedy designs and optional coordinate-exchange refinements."""

    model_name: str
    nested_steps: Sequence[DesignStep]
    refined_steps: Sequence[DesignStep]
    evaluation_times: np.ndarray
    evaluation_weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nested_steps = tuple(self.nested_steps)
        self.refined_steps = tuple(self.refined_steps)
        self.evaluation_times = np.asarray(self.evaluation_times, dtype=float).copy()
        self.evaluation_weights = np.asarray(self.evaluation_weights, dtype=float).copy()
        if self.evaluation_times.ndim != 1 or len(self.evaluation_times) < 2:
            raise ValueError('evaluation_times must contain at least two points.')
        if np.any(~np.isfinite(self.evaluation_times)) or np.any(np.diff(self.evaluation_times) <= 0):
            raise ValueError('evaluation_times must be finite and strictly increasing.')
        if self.evaluation_weights.shape != self.evaluation_times.shape:
            raise ValueError('evaluation_weights must match evaluation_times.')
        if np.any(~np.isfinite(self.evaluation_weights)) or np.any(self.evaluation_weights < 0):
            raise ValueError('evaluation_weights must be finite and nonnegative.')
        if not np.sum(self.evaluation_weights) > 0:
            raise ValueError('evaluation_weights must contain positive total weight.')
        self.evaluation_weights /= np.sum(self.evaluation_weights)
        self.metadata = dict(self.metadata)

    def design_for_n(self, number_of_times: int, *, refined: bool = False) -> DesignStep:
        """Return the stored design with the requested number of time points."""
        steps = self.refined_steps if refined else self.nested_steps
        for step in steps:
            if step.number_of_times == int(number_of_times):
                return step
        label = 'refined' if refined else 'nested'
        raise KeyError(f'No {label} design is available for N={int(number_of_times)}.')


@dataclass
class CrossingTimeSummary:
    """Reference-ensemble timing of a relative degradation milestone.

    Finite ``crossing_times`` are first crossings after ``baseline_time``.
    Positive infinity marks a valid trajectory that has not crossed within the
    supported horizon. NaN marks an ignored draw or an invalid timing input;
    positive-weight invalid inputs make the whole summary unavailable.
    """

    degraded_fraction: float
    quantile: float
    baseline_time: float
    supported_time_range: tuple[float, float]
    crossing_times: np.ndarray
    draw_weights: np.ndarray
    deadline: float | None
    status: str
    message: str = ''

    def __post_init__(self) -> None:
        self.degraded_fraction = float(self.degraded_fraction)
        self.quantile = float(self.quantile)
        for value, name in (
            (self.degraded_fraction, 'degraded_fraction'),
            (self.quantile, 'quantile'),
        ):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be finite and strictly between zero and one.')
        self.supported_time_range = tuple(float(value) for value in self.supported_time_range)
        if (
            len(self.supported_time_range) != 2
            or not all(np.isfinite(self.supported_time_range))
            or self.supported_time_range[0] >= self.supported_time_range[1]
        ):
            raise ValueError('supported_time_range must be finite and increasing.')
        self.baseline_time = float(self.baseline_time)
        lower, upper = self.supported_time_range
        if not np.isfinite(self.baseline_time) or not lower <= self.baseline_time <= upper:
            raise ValueError('baseline_time must lie within the supported time range.')
        self.crossing_times = np.asarray(self.crossing_times, dtype=float).copy()
        self.draw_weights = np.asarray(self.draw_weights, dtype=float).copy()
        if (
            self.crossing_times.ndim != 1
            or self.draw_weights.shape != self.crossing_times.shape
            or len(self.crossing_times) == 0
        ):
            raise ValueError('crossing_times and draw_weights must be matching nonempty vectors.')
        if np.any(~np.isfinite(self.draw_weights)) or np.any(self.draw_weights < 0):
            raise ValueError('draw_weights must be finite and nonnegative.')
        total = float(np.sum(self.draw_weights))
        if not np.isfinite(total) or not total > 0:
            raise ValueError('At least one reference draw must have positive weight.')
        self.draw_weights /= total
        if np.any(np.isneginf(self.crossing_times)):
            raise ValueError('Crossing times cannot contain negative infinity.')
        finite = np.isfinite(self.crossing_times)
        if np.any(self.crossing_times[finite] < self.baseline_time) or np.any(
            self.crossing_times[finite] > upper
        ):
            raise ValueError('Finite crossing times must lie between baseline and horizon.')
        if self.status not in {'available', 'beyond_horizon', 'invalid_reference'}:
            raise ValueError('Unknown crossing-time summary status.')
        if self.deadline is not None:
            self.deadline = float(self.deadline)
            if not np.isfinite(self.deadline) or not self.baseline_time <= self.deadline <= upper:
                raise ValueError('A finite deadline must lie between baseline and horizon.')
        if (self.status == 'available') != (self.deadline is not None):
            raise ValueError('Only an available crossing-time summary can have a deadline.')
        if self.status != 'invalid_reference' and self.invalid_weight > 0:
            raise ValueError('Positive-weight invalid timing inputs require invalid_reference status.')

    @property
    def reached_weight(self) -> float:
        return float(np.sum(self.draw_weights[np.isfinite(self.crossing_times)]))

    @property
    def noncrossing_weight(self) -> float:
        return float(np.sum(self.draw_weights[np.isposinf(self.crossing_times)]))

    @property
    def invalid_weight(self) -> float:
        return float(np.sum(self.draw_weights[np.isnan(self.crossing_times)]))

    def crossed_weight_before(self, time: float) -> float:
        """Return reference weight crossing strictly before an in-range time."""
        time = float(time)
        lower, upper = self.supported_time_range
        if not np.isfinite(time) or not lower <= time <= upper:
            raise ValueError('time must lie within the supported time range.')
        if self.status == 'invalid_reference':
            raise ValueError('Crossed weight is unavailable for an invalid reference.')
        return float(np.sum(self.draw_weights[self.crossing_times < time]))


@dataclass
class NextTimeRecommendation:
    """One additional future time, or an explicit unavailable-policy result."""

    policy: str
    summary: CrossingTimeSummary
    status: str
    message: str
    times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    new_time: float | None = None
    eligible_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    score: DesignInformationScore | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in {'direct_quantile', 'deadline_i_optimal'}:
            raise ValueError('Unknown next-time policy.')
        if self.status not in {
            'available', 'beyond_horizon', 'invalid_reference',
            'deadline_elapsed', 'no_feasible_candidate',
        }:
            raise ValueError('Unknown next-time recommendation status.')
        self.times = np.asarray(self.times, dtype=float).copy()
        self.eligible_times = np.asarray(self.eligible_times, dtype=float).copy()
        for values, name in ((self.times, 'times'), (self.eligible_times, 'eligible_times')):
            if values.ndim != 1 or np.any(~np.isfinite(values)):
                raise ValueError(f'{name} must be a finite one-dimensional array.')
            if np.any(np.diff(values) <= 0):
                raise ValueError(f'{name} must be strictly increasing.')
        if self.status == 'available':
            if self.new_time is None or self.summary.deadline is None:
                raise ValueError('An available recommendation requires a new time and deadline.')
            self.new_time = float(self.new_time)
            if (
                not np.isfinite(self.new_time)
                or self.new_time > self.summary.deadline
                or not np.any(self.times == self.new_time)
                or not np.any(self.eligible_times == self.new_time)
            ):
                raise ValueError('The new time must be an eligible schedule time before the deadline.')
        elif self.new_time is not None or len(self.times) or self.score is not None:
            raise ValueError('Unavailable recommendations cannot contain a selected schedule or score.')
        self.metadata = dict(self.metadata)

    @property
    def deadline_slack(self) -> float | None:
        if self.new_time is None:
            return None
        return float(self.summary.deadline - self.new_time)

    @property
    def crossed_weight_before(self) -> float | None:
        if self.new_time is None:
            return None
        return self.summary.crossed_weight_before(self.new_time)


@dataclass
class SimulationPlan:
    """Reusable trajectory selections and standard-normal errors.

    Designs evaluated against the same plan share both the selected reference
    trajectories and the error draws at common candidate times.
    """

    master_times: np.ndarray
    trajectory_draw_indices: np.ndarray
    standard_normal_errors: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.master_times = np.asarray(self.master_times, dtype=float).copy()
        self.trajectory_draw_indices = np.asarray(
            self.trajectory_draw_indices,
            dtype=int,
        ).copy()
        self.standard_normal_errors = np.asarray(
            self.standard_normal_errors,
            dtype=float,
        ).copy()
        if self.master_times.ndim != 1 or len(self.master_times) == 0:
            raise ValueError('master_times must be a nonempty one-dimensional array.')
        if np.any(~np.isfinite(self.master_times)) or np.any(np.diff(self.master_times) <= 0):
            raise ValueError('master_times must be finite and strictly increasing.')
        if self.trajectory_draw_indices.ndim != 1 or len(self.trajectory_draw_indices) == 0:
            raise ValueError('trajectory_draw_indices must be a nonempty one-dimensional array.')
        if np.any(self.trajectory_draw_indices < 0):
            raise ValueError('trajectory_draw_indices cannot be negative.')
        if self.standard_normal_errors.ndim != 3:
            raise ValueError(
                'standard_normal_errors must have simulation, replicate, and time dimensions.'
            )
        expected_shape = (len(self.trajectory_draw_indices), len(self.master_times))
        if (
            self.standard_normal_errors.shape[0] != expected_shape[0]
            or self.standard_normal_errors.shape[2] != expected_shape[1]
        ):
            raise ValueError(
                'standard_normal_errors must match the simulation count and master time grid.'
            )
        if self.standard_normal_errors.shape[1] < 1:
            raise ValueError('Simulation plans require at least one replicate error draw.')
        if np.any(~np.isfinite(self.standard_normal_errors)):
            raise ValueError('standard_normal_errors must be finite.')
        self.metadata = dict(self.metadata)

    @property
    def n_simulations(self) -> int:
        return int(len(self.trajectory_draw_indices))

    @property
    def n_replicates(self) -> int:
        return int(self.standard_normal_errors.shape[1])

    def indices_for_times(self, times) -> np.ndarray:
        """Return master-grid positions for the requested design times."""
        requested = np.asarray(times, dtype=float)
        if requested.ndim != 1 or np.any(~np.isfinite(requested)):
            raise ValueError('Requested plan times must be finite and one-dimensional.')
        indices = []
        for value in requested:
            matches = np.flatnonzero(
                np.isclose(self.master_times, value, rtol=0.0, atol=1e-12)
            )
            if len(matches) != 1:
                raise ValueError(
                    f'Time {float(value):g} is not uniquely represented in the simulation plan.'
                )
            indices.append(int(matches[0]))
        return np.asarray(indices, dtype=int)


@dataclass
class DesignSimulationResult:
    """Complete Monte Carlo recovery results for one proposed design."""

    design_times: np.ndarray
    model_name: str
    evaluation_times: np.ndarray
    evaluation_weights: np.ndarray
    trajectory_draw_indices: np.ndarray
    expected_values: np.ndarray
    observed_means: np.ndarray
    true_curves: np.ndarray
    fitted_curves: np.ndarray
    parameter_estimates: np.ndarray
    parameter_bound_hits: np.ndarray
    fit_success: np.ndarray
    fit_messages: Sequence[str | None]
    integrated_rmse: np.ndarray
    normalized_integrated_rmse: np.ndarray
    maximum_absolute_error: np.ndarray
    mean_curve_interval_width: np.ndarray
    true_target_times: np.ndarray
    fitted_target_times: np.ndarray
    target_time_errors: np.ndarray
    response_scale: float
    failure_penalty: float
    penalized_integrated_rmse: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.design_times = np.unique(np.asarray(self.design_times, dtype=float))
        self.evaluation_times = np.asarray(self.evaluation_times, dtype=float).copy()
        self.evaluation_weights = np.asarray(self.evaluation_weights, dtype=float).copy()
        self.trajectory_draw_indices = np.asarray(
            self.trajectory_draw_indices,
            dtype=int,
        ).copy()
        self.expected_values = np.asarray(self.expected_values, dtype=float).copy()
        self.observed_means = np.asarray(self.observed_means, dtype=float).copy()
        self.true_curves = np.asarray(self.true_curves, dtype=float).copy()
        self.fitted_curves = np.asarray(self.fitted_curves, dtype=float).copy()
        self.parameter_estimates = np.asarray(self.parameter_estimates, dtype=float).copy()
        self.parameter_bound_hits = np.asarray(self.parameter_bound_hits, dtype=bool).copy()
        self.fit_success = np.asarray(self.fit_success, dtype=bool).copy()
        self.fit_messages = tuple(self.fit_messages)
        self.integrated_rmse = np.asarray(self.integrated_rmse, dtype=float).copy()
        self.normalized_integrated_rmse = np.asarray(
            self.normalized_integrated_rmse,
            dtype=float,
        ).copy()
        self.maximum_absolute_error = np.asarray(
            self.maximum_absolute_error,
            dtype=float,
        ).copy()
        self.mean_curve_interval_width = np.asarray(
            self.mean_curve_interval_width,
            dtype=float,
        ).copy()
        self.true_target_times = np.asarray(self.true_target_times, dtype=float).copy()
        self.fitted_target_times = np.asarray(self.fitted_target_times, dtype=float).copy()
        self.target_time_errors = np.asarray(self.target_time_errors, dtype=float).copy()
        self.penalized_integrated_rmse = np.asarray(
            self.penalized_integrated_rmse,
            dtype=float,
        ).copy()

        if self.design_times.ndim != 1 or len(self.design_times) == 0:
            raise ValueError('design_times must be a nonempty one-dimensional array.')
        if self.evaluation_times.ndim != 1 or len(self.evaluation_times) < 2:
            raise ValueError('evaluation_times must contain at least two points.')
        if np.any(np.diff(self.evaluation_times) <= 0):
            raise ValueError('evaluation_times must be strictly increasing.')
        if self.evaluation_weights.shape != self.evaluation_times.shape:
            raise ValueError('evaluation_weights must match evaluation_times.')
        if np.any(~np.isfinite(self.evaluation_weights)) or np.any(self.evaluation_weights < 0):
            raise ValueError('evaluation_weights must be finite and nonnegative.')
        if not np.sum(self.evaluation_weights) > 0:
            raise ValueError('evaluation_weights must contain positive total weight.')
        self.evaluation_weights /= np.sum(self.evaluation_weights)

        simulation_count = len(self.trajectory_draw_indices)
        design_shape = (simulation_count, len(self.design_times))
        curve_shape = (simulation_count, len(self.evaluation_times))
        if self.expected_values.shape != design_shape or self.observed_means.shape != design_shape:
            raise ValueError('Expected and observed mean arrays have an invalid design shape.')
        if self.true_curves.shape != curve_shape or self.fitted_curves.shape != curve_shape:
            raise ValueError('True and fitted curve arrays have an invalid evaluation shape.')
        if self.parameter_estimates.ndim != 2 or self.parameter_estimates.shape[0] != simulation_count:
            raise ValueError('parameter_estimates must contain one row per simulation.')
        if self.parameter_bound_hits.shape != self.parameter_estimates.shape:
            raise ValueError('parameter_bound_hits must match parameter_estimates.')
        per_simulation = (
            self.fit_success,
            self.integrated_rmse,
            self.normalized_integrated_rmse,
            self.maximum_absolute_error,
            self.mean_curve_interval_width,
            self.true_target_times,
            self.fitted_target_times,
            self.target_time_errors,
            self.penalized_integrated_rmse,
        )
        if any(values.ndim != 1 or len(values) != simulation_count for values in per_simulation):
            raise ValueError('Per-simulation outputs must match the simulation count.')
        if len(self.fit_messages) != simulation_count:
            raise ValueError('fit_messages must contain one entry per simulation.')
        self.response_scale = float(self.response_scale)
        self.failure_penalty = float(self.failure_penalty)
        if not self.response_scale > 0 or not np.isfinite(self.response_scale):
            raise ValueError('response_scale must be finite and positive.')
        if self.failure_penalty < 0 or not np.isfinite(self.failure_penalty):
            raise ValueError('failure_penalty must be finite and nonnegative.')
        self.metadata = dict(self.metadata)

    @property
    def n_simulations(self) -> int:
        return int(len(self.fit_success))

    @property
    def n_successful(self) -> int:
        return int(np.count_nonzero(self.fit_success))

    @property
    def failure_rate(self) -> float:
        return float(1.0 - self.n_successful / self.n_simulations)


@dataclass
class DesignPerformanceSummary:
    """Compact user-facing performance summary for one design."""

    design_times: np.ndarray
    n_simulations: int
    n_successful: int
    failure_rate: float
    response_scale: float
    failure_penalty: float
    mean_irmse: float
    median_irmse: float
    p90_irmse: float
    p95_irmse: float
    mean_normalized_irmse: float
    median_normalized_irmse: float
    p90_normalized_irmse: float
    penalized_mean_irmse: float
    penalized_mean_normalized_irmse: float
    mean_maximum_absolute_error: float
    p90_maximum_absolute_error: float
    mean_curve_interval_width: float
    median_curve_interval_width: float
    median_absolute_target_time_error: float
    p90_absolute_target_time_error: float
    parameter_bound_hit_rate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.design_times = np.unique(np.asarray(self.design_times, dtype=float))
        self.n_simulations = int(self.n_simulations)
        self.n_successful = int(self.n_successful)
        self.failure_rate = float(self.failure_rate)
        self.response_scale = float(self.response_scale)
        self.failure_penalty = float(self.failure_penalty)
        for name in (
            'mean_irmse',
            'median_irmse',
            'p90_irmse',
            'p95_irmse',
            'mean_normalized_irmse',
            'median_normalized_irmse',
            'p90_normalized_irmse',
            'penalized_mean_irmse',
            'penalized_mean_normalized_irmse',
            'mean_maximum_absolute_error',
            'p90_maximum_absolute_error',
            'mean_curve_interval_width',
            'median_curve_interval_width',
            'median_absolute_target_time_error',
            'p90_absolute_target_time_error',
            'parameter_bound_hit_rate',
        ):
            setattr(self, name, float(getattr(self, name)))
        if self.n_simulations < 1 or not 0 <= self.n_successful <= self.n_simulations:
            raise ValueError('Simulation summary counts are invalid.')
        self.metadata = dict(self.metadata)


@dataclass
class DesignSimulationComparison:
    """Named design results evaluated using one shared simulation plan."""

    simulation_plan: SimulationPlan
    results: Mapping[str, DesignSimulationResult]
    summaries: Mapping[str, DesignPerformanceSummary]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.results = dict(self.results)
        self.summaries = dict(self.summaries)
        if not self.results:
            raise ValueError('At least one design result is required.')
        if set(self.results) != set(self.summaries):
            raise ValueError('Simulation result and summary labels must match.')
        for result in self.results.values():
            if result.n_simulations != self.simulation_plan.n_simulations:
                raise ValueError('All results must use the stored simulation plan.')
        self.metadata = dict(self.metadata)
