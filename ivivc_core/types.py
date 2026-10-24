"""Structured backend data types for reusable IVIVC calculations.

These types intentionally contain numerical data and metadata only.  The
existing application continues to consume legacy lists, tuples, and
dictionaries through adapters so that backend refactoring does not alter the
current browser or CLI contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Mapping, TypeVar

import numpy as np


@dataclass
class TimeCourse:
    """A one-dimensional time/response dataset.

    The class performs only structural validation. Missing values and domain
    restrictions remain the responsibility of the preprocessing functions so
    existing IVIVC behavior is preserved.
    """

    time: np.ndarray
    value: np.ndarray
    label: str | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        value = np.asarray(self.value)
        if time.ndim != 1 or value.ndim != 1:
            raise ValueError('Time-course time and response arrays must be one-dimensional.')
        if len(time) != len(value):
            raise ValueError('Time-course time and response arrays must have the same length.')
        self.time = time.copy()
        self.value = value.copy()


@dataclass(frozen=True)
class PreprocessingWarning:
    """A structured preprocessing warning with a legacy dictionary adapter."""

    option: str
    message: str

    def to_legacy_dict(self) -> dict[str, str]:
        return {'option': self.option, 'message': self.message}


@dataclass(frozen=True)
class NormalizationRecord:
    """Parameters used for one normalization step.

    Values are recorded separately for the in vitro and in vivo datasets so a
    future caller can reconstruct or reverse the transformation where
    possible. The record does not change the normalization implementation used
    by the existing application.
    """

    name: str
    in_vitro_parameters: Mapping[str, float | bool]
    in_vivo_parameters: Mapping[str, float | bool] | None = None


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Structured description of preprocessing applied to one analysis."""

    selected_interpolation: tuple[str, ...] = ()
    interpolation: str | None = None
    selected_scalings: tuple[str, ...] = ()
    applied_scalings: tuple[str, ...] = ()
    selected_normalizations: tuple[str, ...] = ()
    normalizations: tuple[NormalizationRecord, ...] = ()
    warnings: tuple[PreprocessingWarning, ...] = ()

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the exact metadata shape currently used by the IVIVC app."""
        return {
            'applied_scalings': list(self.applied_scalings),
            'warnings': [warning.to_legacy_dict() for warning in self.warnings],
        }

@dataclass
class PreparedIVIVCData:
    """Preprocessed data used by the three existing IVIVC approaches."""

    in_vitro: TimeCourse
    in_vivo: TimeCourse | None
    response_grid: np.ndarray | None = None
    time_grid: np.ndarray | None = None
    in_vitro_time_at_response: np.ndarray | None = None
    in_vivo_time_at_response: np.ndarray | None = None
    in_vitro_response_on_time_grid: np.ndarray | None = None
    in_vivo_response_on_time_grid: np.ndarray | None = None
    preprocessing: PreprocessingMetadata = field(default_factory=PreprocessingMetadata)

    def __post_init__(self) -> None:
        array_fields = (
            'response_grid',
            'time_grid',
            'in_vitro_time_at_response',
            'in_vivo_time_at_response',
            'in_vitro_response_on_time_grid',
            'in_vivo_response_on_time_grid',
        )
        for field_name in array_fields:
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, np.asarray(value).copy())

    @property
    def has_in_vivo(self) -> bool:
        return self.in_vivo is not None

    def to_legacy_data(self) -> list[np.ndarray] | tuple[np.ndarray, ...]:
        """Return the positional data structure expected by ``engine.py``.

        A single time course historically returns a two-item list. Paired data
        historically return the ten-item tuple produced by the interpolation
        functions. Both distinctions are retained for compatibility.
        """
        if self.in_vivo is None:
            return [self.in_vitro.time.copy(), self.in_vitro.value.copy()]

        return (
            self.in_vitro.time.copy(),
            self.in_vitro.value.copy(),
            self.in_vivo.time.copy(),
            self.in_vivo.value.copy(),
            self.response_grid.copy(),
            self.time_grid.copy(),
            self.in_vitro_time_at_response.copy(),
            self.in_vivo_time_at_response.copy(),
            self.in_vitro_response_on_time_grid.copy(),
            self.in_vivo_response_on_time_grid.copy(),
        )


@dataclass
class ProfileFit:
    """Numerical result from fitting one response-versus-time profile.

    The object deliberately excludes HTML and Plotly output so it can be reused
    by the current IVIVC workflows and by future study-design calculations.
    Parameter order is preserved by ``parameters`` and ``parameter_names``.
    """

    model_name: str
    time: np.ndarray
    response: np.ndarray
    parameter_names: tuple[str, ...]
    parameters: Mapping[str, float]
    covariance: np.ndarray
    initial_parameters: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    residual_dof: int
    fit_metrics: Mapping[str, float]
    cv_scores: Mapping[str, np.ndarray]
    cv_means: Mapping[str, float]
    cv_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cv_parameter_sets: list[np.ndarray | None] = field(default_factory=list)
    tau_estimate: float | None = None
    tau_sd: float | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time).copy()
        self.response = np.asarray(self.response).copy()
        self.covariance = np.asarray(self.covariance).copy()
        self.initial_parameters = np.asarray(self.initial_parameters).copy()
        self.fitted_values = np.asarray(self.fitted_values).copy()
        self.residuals = np.asarray(self.residuals).copy()
        self.parameter_names = tuple(self.parameter_names)
        self.parameters = dict(self.parameters)
        self.fit_metrics = dict(self.fit_metrics)
        self.cv_scores = {
            metric: np.asarray(values).copy()
            for metric, values in self.cv_scores.items()
        }
        self.cv_means = dict(self.cv_means)
        self.cv_diagnostics = dict(self.cv_diagnostics)
        self.cv_parameter_sets = [
            None if values is None else np.asarray(values).copy()
            for values in self.cv_parameter_sets
        ]

    @property
    def parameter_vector(self) -> np.ndarray:
        """Return fitted parameters in the model function's declared order."""
        return np.asarray([self.parameters[name] for name in self.parameter_names], dtype=float)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)


@dataclass
class PredictedTrajectory:
    """Numerical in vivo trajectory produced by one prediction method.

    All prediction approaches are represented on the same paired axes. Methods
    that map time retain the supplied response as ``in_vivo_value``; methods
    that map response retain the supplied time as ``in_vivo_time``. Optional
    bounds are kept separate from the central trajectory so presentation code
    can decide whether and how to display them.
    """

    source_approach: str
    source_method: str
    in_vitro_time: np.ndarray
    in_vitro_value: np.ndarray
    in_vivo_time: np.ndarray
    in_vivo_value: np.ndarray
    model_name: str | None = None
    available: bool = True
    skip_reason: str | None = None
    note: str | None = None
    status: np.ndarray | None = None
    time_lower: np.ndarray | None = None
    time_upper: np.ndarray | None = None
    time_error_lower: np.ndarray | None = None
    time_error_upper: np.ndarray | None = None
    value_lower: np.ndarray | None = None
    value_upper: np.ndarray | None = None
    scaling_factor: np.ndarray | None = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array_fields = (
            'in_vitro_time',
            'in_vitro_value',
            'in_vivo_time',
            'in_vivo_value',
            'status',
            'time_lower',
            'time_upper',
            'time_error_lower',
            'time_error_upper',
            'value_lower',
            'value_upper',
            'scaling_factor',
        )
        for field_name in array_fields:
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, np.asarray(value).copy())

        expected_length = len(self.in_vitro_time)
        required_fields = (
            'in_vitro_value',
            'in_vivo_time',
            'in_vivo_value',
        )
        optional_fields = (
            'status',
            'time_lower',
            'time_upper',
            'time_error_lower',
            'time_error_upper',
            'value_lower',
            'value_upper',
            'scaling_factor',
        )
        for field_name in required_fields:
            if len(getattr(self, field_name)) != expected_length:
                raise ValueError(
                    f'Predicted-trajectory field {field_name!r} must match the input length.'
                )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None and len(value) != expected_length:
                raise ValueError(
                    f'Predicted-trajectory field {field_name!r} must match the input length.'
                )

        copied_auxiliary: dict[str, Any] = {}
        for key, value in dict(self.auxiliary).items():
            copied_auxiliary[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.auxiliary = copied_auxiliary


@dataclass
class ParameterSamples:
    """Joint samples from a fitted parameter distribution.

    Rows represent complete parameter vectors. Keeping the vectors together is
    important because parameter covariance determines how complete fitted
    trajectories vary across time. When model constraints are supplied, invalid
    draws are rejected and resampled before this object is constructed.
    """

    parameter_names: tuple[str, ...]
    values: np.ndarray
    requested_samples: int | None = None
    attempted_samples: int | None = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameter_names = tuple(self.parameter_names)
        self.values = np.asarray(self.values, dtype=float).copy()
        if self.values.ndim == 0:
            self.values = self.values.reshape(1, 1)
        elif self.values.ndim == 1:
            self.values = self.values.reshape(1, -1)
        if self.values.ndim != 2:
            raise ValueError('Parameter samples must be a two-dimensional array.')
        if self.values.shape[1] != len(self.parameter_names):
            raise ValueError(
                'The number of sampled parameter columns must match parameter_names.'
            )
        if self.requested_samples is None:
            self.requested_samples = int(self.values.shape[0])
        else:
            self.requested_samples = int(self.requested_samples)
            if self.requested_samples < self.values.shape[0]:
                raise ValueError(
                    'requested_samples cannot be smaller than the returned sample count.'
                )
        if self.attempted_samples is None:
            self.attempted_samples = int(self.values.shape[0])
        else:
            self.attempted_samples = int(self.attempted_samples)
            if self.attempted_samples < self.values.shape[0]:
                raise ValueError(
                    'attempted_samples cannot be smaller than the returned sample count.'
                )
        self.rejection_counts = {
            str(reason): int(count)
            for reason, count in dict(self.rejection_counts).items()
            if int(count) > 0
        }

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_parameters(self) -> int:
        return int(self.values.shape[1])


@dataclass
class CurveDraws:
    """Model evaluations for complete joint parameter draws.

    Invalid model evaluations are identified by ``valid_draws``. When
    pointwise evaluation is requested, partially finite rows may be retained;
    callers can inspect individual values without accepting an incomplete row
    as a valid full trajectory.
    """

    x: np.ndarray
    values: np.ndarray
    valid_draws: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).copy()
        self.values = np.asarray(self.values, dtype=float).copy()
        self.valid_draws = np.asarray(self.valid_draws, dtype=bool).copy()
        if self.x.ndim != 1:
            raise ValueError('Curve-draw x values must be one-dimensional.')
        if self.values.ndim == 1:
            self.values = self.values.reshape(1, -1)
        if self.values.ndim != 2:
            raise ValueError('Curve draws must be a two-dimensional array.')
        if self.values.shape[1] != len(self.x):
            raise ValueError('Each curve draw must contain one value per x value.')
        if self.valid_draws.ndim != 1 or len(self.valid_draws) != self.values.shape[0]:
            raise ValueError('valid_draws must contain one flag per curve draw.')

    @property
    def n_draws(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_valid_draws(self) -> int:
        return int(np.count_nonzero(self.valid_draws))


@dataclass
class IntervalSummary:
    """Pointwise summary of a set of complete trajectory draws."""

    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    valid_counts: np.ndarray
    confidence_level: float

    def __post_init__(self) -> None:
        array_fields = ('lower', 'upper', 'mean', 'std', 'valid_counts')
        for field_name in array_fields:
            value = np.asarray(getattr(self, field_name)).copy()
            if value.ndim != 1:
                raise ValueError('Interval summary arrays must be one-dimensional.')
            setattr(self, field_name, value)
        lengths = {len(getattr(self, field_name)) for field_name in array_fields}
        if len(lengths) != 1:
            raise ValueError('Interval summary arrays must have the same length.')
        self.confidence_level = float(self.confidence_level)


@dataclass
class TrajectoryEnsemble:
    """A central predicted trajectory and complete plausible trajectory draws.

    ``time_draws`` and ``value_draws`` have shape ``(draws, points)``. Keeping
    complete rows intact preserves temporal dependence and allows the future
    design engine to sample one plausible true trajectory at a time.
    """

    central: PredictedTrajectory
    time_draws: np.ndarray
    value_draws: np.ndarray
    valid_draws: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_draws = np.asarray(self.time_draws, dtype=float).copy()
        self.value_draws = np.asarray(self.value_draws, dtype=float).copy()
        self.valid_draws = np.asarray(self.valid_draws, dtype=bool).copy()
        if self.time_draws.ndim == 1:
            self.time_draws = self.time_draws.reshape(1, -1)
        if self.value_draws.ndim == 1:
            self.value_draws = self.value_draws.reshape(1, -1)
        if self.time_draws.ndim != 2 or self.value_draws.ndim != 2:
            raise ValueError('Trajectory draws must be two-dimensional arrays.')
        if self.time_draws.shape != self.value_draws.shape:
            raise ValueError('Time and response trajectory draws must have matching shapes.')
        expected_points = len(self.central.in_vivo_time)
        if self.time_draws.shape[1] != expected_points:
            raise ValueError(
                'Each trajectory draw must match the central trajectory length.'
            )
        if self.valid_draws.ndim != 1 or len(self.valid_draws) != self.time_draws.shape[0]:
            raise ValueError('valid_draws must contain one flag per trajectory draw.')

        copied_metadata: dict[str, Any] = {}
        for key, value in dict(self.metadata).items():
            copied_metadata[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.metadata = copied_metadata

    @property
    def n_draws(self) -> int:
        return int(self.time_draws.shape[0])

    @property
    def n_valid_draws(self) -> int:
        return int(np.count_nonzero(self.valid_draws))


T = TypeVar('T')


@dataclass(frozen=True)
class AnalysisError:
    """Structured error captured by a batch-oriented public API call.

    Numerical primitives such as :func:`fit_profile` continue to raise their
    original exceptions.  The higher-level API can instead collect independent
    model or prediction failures so one unsuccessful candidate does not discard
    successful results from other candidates.
    """

    message: str
    exception_type: str

    @classmethod
    def from_exception(cls, exception: BaseException) -> 'AnalysisError':
        """Create an error record without retaining the exception object."""
        return cls(
            message=str(exception),
            exception_type=type(exception).__name__,
        )


@dataclass
class ResultBatch(Generic[T]):
    """Successful keyed results and independently captured failures."""

    results: Mapping[str, T]
    errors: Mapping[str, AnalysisError] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.results = dict(self.results)
        self.errors = dict(self.errors)

    def __getitem__(self, key: str) -> T:
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def successful_keys(self) -> tuple[str, ...]:
        return tuple(self.results)

    @property
    def failed_keys(self) -> tuple[str, ...]:
        return tuple(self.errors)


@dataclass
class IVIVCModelFit:
    """Structured fit result for one IVIVC approach/model selection.

    Approach 1 stores a single fitted time-to-time mapping. Approach 2 stores
    independent in vitro and in vivo profile fits. Approach 3 has no parametric
    fit and therefore stores only its approach identifier.
    """

    approach_id: str
    model_name: str | None = None
    time_mapping_fit: ProfileFit | None = None
    in_vitro_fit: ProfileFit | None = None
    in_vivo_fit: ProfileFit | None = None

    def __post_init__(self) -> None:
        if self.approach_id == 'approach1':
            if self.model_name is None or self.time_mapping_fit is None:
                raise ValueError(
                    'Approach 1 results require a model name and time-mapping fit.'
                )
            if self.in_vitro_fit is not None or self.in_vivo_fit is not None:
                raise ValueError('Approach 1 results cannot contain profile fits.')
        elif self.approach_id == 'approach2':
            if (
                self.model_name is None
                or self.in_vitro_fit is None
                or self.in_vivo_fit is None
            ):
                raise ValueError(
                    'Approach 2 results require a model name and both profile fits.'
                )
            if self.time_mapping_fit is not None:
                raise ValueError('Approach 2 results cannot contain a time-mapping fit.')
        elif self.approach_id == 'approach3':
            if self.model_name is not None:
                raise ValueError('Approach 3 does not have a model name.')
            if any(
                fit is not None
                for fit in (
                    self.time_mapping_fit,
                    self.in_vitro_fit,
                    self.in_vivo_fit,
                )
            ):
                raise ValueError('Approach 3 does not contain parametric fits.')
        else:
            raise ValueError(f'Unknown IVIVC approach: {self.approach_id!r}')

    @property
    def key(self) -> str:
        if self.model_name is None:
            return self.approach_id
        return f'{self.approach_id}:{self.model_name}'

    @property
    def profile_fits(self) -> Mapping[str, ProfileFit]:
        """Return the component fits using stable, descriptive keys."""
        if self.approach_id == 'approach1':
            assert self.time_mapping_fit is not None
            return {'time_mapping': self.time_mapping_fit}
        if self.approach_id == 'approach2':
            assert self.in_vitro_fit is not None
            assert self.in_vivo_fit is not None
            return {
                'in_vitro': self.in_vitro_fit,
                'in_vivo': self.in_vivo_fit,
            }
        return {}


@dataclass
class IVIVCAnalysis:
    """Reusable numerical result from fitting selected IVIVC approaches."""

    prepared_data: PreparedIVIVCData
    model_results: Mapping[str, IVIVCModelFit]
    errors: Mapping[str, AnalysisError] = field(default_factory=dict)
    selected_metrics: tuple[str, ...] = ()
    analysis_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_results = dict(self.model_results)
        self.errors = dict(self.errors)
        self.selected_metrics = tuple(self.selected_metrics)
        self.analysis_config = dict(self.analysis_config)

    def __getitem__(self, key: str) -> IVIVCModelFit:
        return self.model_results[key]

    @property
    def successful_keys(self) -> tuple[str, ...]:
        return tuple(self.model_results)

    @property
    def failed_keys(self) -> tuple[str, ...]:
        return tuple(self.errors)
