"""Assess one assembled calibration dataset, without simulating a study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ivivc_core import (
    AnalysisError, DEFAULT_PREDICTION_METHODS, IVIVCAnalysis, PredictedTrajectory,
    TimeCourse, fit_ivivc, get_model_spec, predict_in_vivo,
    prepare_ivivc_data, prepare_time_course,
)
from ivivc_core.intervals import validate_interval_settings
from ivivc_core.predictions import _inverse_interval_settings


@dataclass
class CalibrationEvaluation:
    """One quantity on an explicit evaluation grid, retaining core diagnostics.

    Bounds may be missing even when the point estimate is available. The
    trajectory retains point statuses, extrapolation information, conditional
    inverse-interval counts, and any sampling failure. Neither mask requires
    the interval to contain the nominal estimate.
    """

    quantity: str
    coordinate: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    trajectory: PredictedTrajectory

    def __post_init__(self) -> None:
        self.coordinate = np.asarray(self.coordinate, dtype=float).copy()
        self.estimate = np.asarray(self.estimate, dtype=float).copy()
        if self.coordinate.ndim != 1 or self.estimate.shape != self.coordinate.shape:
            raise ValueError('Evaluation coordinates and estimates must be matching vectors.')
        for name in ('lower', 'upper'):
            value = getattr(self, name)
            value = (np.full(self.estimate.shape, np.nan) if value is None
                     else np.asarray(value, dtype=float).copy())
            if value.shape != self.estimate.shape:
                raise ValueError('Evaluation bounds must match the estimates.')
            setattr(self, name, value)

    @property
    def point_available(self) -> np.ndarray:
        return (bool(self.trajectory.available) & np.isfinite(self.coordinate)
                & np.isfinite(self.estimate))

    @property
    def interval_available(self) -> np.ndarray:
        return (self.point_available & np.isfinite(self.lower)
                & np.isfinite(self.upper) & (self.lower <= self.upper))

    @property
    def interval_width(self) -> np.ndarray:
        with np.errstate(over='ignore', invalid='ignore'):
            return np.where(self.interval_available, self.upper - self.lower, np.nan)


@dataclass
class IVIVCCalibrationAssessment:
    """Fit and independent evaluation outcomes for one calibration dataset.

    ``errors`` records exceptions by stage: preprocessing, fit, relationship,
    or prediction. An unavailable core prediction is retained as an evaluation
    with availability masks and its original diagnostics, not discarded.
    """

    analysis: IVIVCAnalysis | None = None
    relationship: CalibrationEvaluation | None = None
    prediction: CalibrationEvaluation | None = None
    errors: dict[str, AnalysisError] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def fit_available(self) -> bool:
        return self.analysis is not None and bool(self.analysis.model_results)


def _evaluation_grid(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a nonempty finite one-dimensional array.')
    return values.copy()


def _evaluation(trajectory, *, quantity, coordinate, value_axis):
    return CalibrationEvaluation(
        quantity, coordinate,
        trajectory.in_vivo_value if value_axis else trajectory.in_vivo_time,
        trajectory.value_lower if value_axis else trajectory.time_lower,
        trajectory.value_upper if value_axis else trajectory.time_upper,
        trajectory,
    )


def assess_ivivc_calibration(
    in_vitro: TimeCourse,
    in_vivo: TimeCourse,
    *,
    model_name: str,
    method: str = 'value_ratio',
    evaluation_times=None,
    response_levels=None,
    prediction: TimeCourse | None = None,
    selected_normalizations: Sequence[str] | None = (),
    analysis_config: Mapping[str, object] | None = None,
    confidence_level: float = .95,
    n_samples: int = 1000,
    random_state=12345,
    inverse_interval_policy: str | None = None,
    inverse_min_valid_fraction: float | None = None,
    fail_fast: bool = False,
) -> IVIVCCalibrationAssessment:
    """Preprocess, fit Approach 2, and assess one assembled calibration pair.

    Inputs, including an optional independent target ``prediction``, are on
    their original response scale. Each dataset is normalized separately with
    the same selected operations, as in the IVIVC browser. Calibration uses
    observed rows only, without interpolation or logarithmic axis scaling.
    Supplied individual SD or mean SEM is transported by the existing APIs.

    Relationship assessment does not require experimental target data:

    * ``value_ratio`` uses unit responses at ``evaluation_times`` to evaluate
      the fitted in vivo/in vitro response ratio.
    * ``tau_ratio`` uses a unit-time probe to report one dimensionless
      ``time_constant_ratio``. Its coordinate is [1], a computational probe,
      rather than an experimental sampling time.
    * ``observed_response_inverse`` evaluates in vivo time at ``response_levels``
      on the analyzed in vivo response scale. It does not use the in vitro fit.

    For value ratio and inverse mapping, omit the grid to request target-only
    assessment. A grid or a target is required. An optional target is evaluated
    with the same model and method, independently of relationship evaluation.

    Estimates and bounds use analyzed coordinates; normalization records are
    retained for callers comparing different datasets. In particular, widths
    on different min-max scales must not be compared as physical-unit widths.
    No measurements or future in vivo measurement noise are generated here.
    Propagation clones a supplied RNG so it cannot advance a study's noise
    stream, and adding a target does not alter relationship intervals.

    Invalid configuration raises. Data/preprocessing, fitting, and evaluation
    failures are recorded by stage unless ``fail_fast`` is true. A failure in
    target preprocessing or prediction does not discard a usable relationship.
    """
    if not isinstance(in_vitro, TimeCourse) or not isinstance(in_vivo, TimeCourse):
        raise TypeError('Calibration inputs must be TimeCourse objects.')
    if prediction is not None and not isinstance(prediction, TimeCourse):
        raise TypeError('prediction must be a TimeCourse or None.')
    if prediction is in_vitro or prediction is in_vivo:
        raise ValueError('Target prediction measurements must be independent of calibration measurements.')
    if method not in DEFAULT_PREDICTION_METHODS['approach2']:
        raise ValueError('Choose a supported Approach 2 prediction method.')
    if 'approach2' not in get_model_spec(model_name).approaches:
        raise ValueError('The selected model must support Approach 2.')
    confidence_level, n_samples = validate_interval_settings(confidence_level, n_samples)
    if isinstance(random_state, (bool, np.bool_)):
        raise TypeError('random_state must be an integer, Generator, RandomState, or None.')
    if isinstance(random_state, (int, np.integer)):
        np.random.SeedSequence(int(random_state))  # Validate without advancing an RNG.
    elif random_state is not None and not isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        raise TypeError('random_state must be an integer, Generator, RandomState, or None.')
    normalizations = ((selected_normalizations,) if isinstance(selected_normalizations, str)
                      else tuple(() if selected_normalizations is None else selected_normalizations))
    if any(name not in ('initial_value', 'min_max', 'one_minus') for name in normalizations):
        raise ValueError('Use initial_value, min_max, one_minus, or no normalization.')
    if evaluation_times is not None and method != 'value_ratio':
        raise ValueError('evaluation_times is used only for value_ratio relationships.')
    if response_levels is not None and method != 'observed_response_inverse':
        raise ValueError('response_levels is used only for observed_response_inverse relationships.')
    if method == 'observed_response_inverse':
        # Reuse the core validation and call-time defaults without freezing or
        # redefining the conditional-interval policy in the design layer.
        _inverse_interval_settings(inverse_interval_policy, inverse_min_valid_fraction)
    elif inverse_interval_policy is not None or inverse_min_valid_fraction is not None:
        raise ValueError('Inverse interval settings require observed_response_inverse.')

    probe = None
    quantity = None
    coordinate = None
    if method == 'value_ratio' and evaluation_times is not None:
        coordinate = _evaluation_grid(evaluation_times, 'evaluation_times')
        probe = TimeCourse(coordinate, np.ones(len(coordinate)))
        quantity = 'value_ratio'
    elif method == 'tau_ratio':
        coordinate = np.array([1.])
        probe = TimeCourse(coordinate, np.ones(1))
        quantity = 'time_constant_ratio'
    elif response_levels is not None:
        coordinate = _evaluation_grid(response_levels, 'response_levels')
        probe = TimeCourse(np.zeros(len(coordinate)), coordinate)
        quantity = 'in_vivo_time'
    if probe is None and prediction is None:
        raise ValueError('Supply a relationship evaluation grid or a target prediction dataset.')

    # Avoid default CV work and process pools in a repeated-study caller.
    config = {'cv_scheme': 'none', 'grid_search_num_cores': 1}
    config.update(analysis_config or {})
    selection = f'approach2:{model_name}'
    result = IVIVCCalibrationAssessment(metadata={
        'selection': selection, 'method': method,
        'evaluation_scale': 'analyzed',
        'relationship_quantity': quantity,
        'relationship_coordinate': ('unit_time' if method == 'tau_ratio'
                                    else 'response' if method == 'observed_response_inverse' else 'time'),
        'relationship_is_computational_probe': probe is not None,
        'prediction_requested': prediction is not None,
        'future_observation_variability_included': False,
        'confidence_level': confidence_level, 'n_samples': n_samples,
    })

    def record(stage, exception):
        if fail_fast:
            raise exception
        result.errors[stage] = AnalysisError.from_exception(exception)

    try:
        prepared = prepare_ivivc_data(
            in_vitro, in_vivo, selected_interpolation=['none'],
            selected_normalizations=normalizations,
        )
    except Exception as exception:
        record('preprocessing', exception)
        return result
    try:
        result.analysis = fit_ivivc(
            prepared, selections=selection, selected_metrics=['rmse'],
            analysis_config=config, fail_fast=fail_fast,
        )
        if selection in result.analysis.errors:
            result.errors['fit'] = result.analysis.errors[selection]
            return result
    except Exception as exception:
        record('fit', exception)
        return result

    # Each evaluation gets the same independent propagation stream. Calling
    # either output alone therefore gives the same bounds as calling both.
    state = np.random.default_rng() if random_state is None else random_state

    def evaluate(course):
        batch = predict_in_vivo(
            result.analysis, course, methods={'approach2': method},
            confidence_level=confidence_level, n_samples=n_samples,
            random_state=deepcopy(state), fail_fast=True,
            inverse_interval_policy=inverse_interval_policy,
            inverse_min_valid_fraction=inverse_min_valid_fraction,
        )
        return batch[f'{selection}:{method}']

    if probe is not None:
        try:
            result.relationship = _evaluation(
                evaluate(probe), quantity=quantity, coordinate=coordinate,
                value_axis=method == 'value_ratio',
            )
        except Exception as exception:
            record('relationship', exception)
    if prediction is not None:
        try:
            target, target_preprocessing = prepare_time_course(
                prediction.time, prediction.value, label=prediction.label,
                selected_normalizations=normalizations,
                observation_uncertainty=prediction.observation_uncertainty,
            )
            result.metadata['prediction_preprocessing'] = target_preprocessing
            result.prediction = _evaluation(
                evaluate(target),
                quantity='in_vivo_response' if method == 'value_ratio' else 'in_vivo_time',
                coordinate=(target.value if method == 'observed_response_inverse' else target.time),
                value_axis=method == 'value_ratio',
            )
        except Exception as exception:
            record('prediction', exception)
    return result
