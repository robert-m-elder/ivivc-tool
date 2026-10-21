"""Structured backend data types for reusable IVIVC calculations.

These types intentionally contain numerical data and metadata only.  The
existing application continues to consume legacy lists, tuples, and
dictionaries through adapters so that backend refactoring does not alter the
current browser or CLI contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
