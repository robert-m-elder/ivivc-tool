"""Common outcomes and truth-based summaries for calibration simulations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import signature

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from ivivc_core import DEFAULT_PREDICTION_METHODS, TimeCourse, get_model_spec, prepare_time_course
from ivivc_core.intervals import validate_interval_settings
from ivivc_core.predictions import _inverse_interval_settings

from .calibration_assessment import IVIVCCalibrationAssessment, assess_ivivc_calibration
from .types import ReferenceTrajectoryEnsemble


@dataclass
class IVIVCAssessmentRequest:
    """Optional IVIVC analysis of the same observations used for curve recovery.

    ``in_vitro`` is the companion calibration dataset for this study's system.
    ``options`` contains C1 assessor keyword arguments, including model_name.
    Optional source references/explicit output truth enable error and coverage
    calculations. Holding measured in vitro data fixed retains fit covariance.
    """

    in_vitro: TimeCourse
    options: Mapping[str, object]
    in_vitro_reference: ReferenceTrajectoryEnsemble | None = None
    truth: Mapping[str, object] | None = None
    paired_reference_draws: bool = False

    def __post_init__(self):
        if not isinstance(self.in_vitro, TimeCourse):
            raise TypeError('The companion in vitro calibration data must be a TimeCourse.')
        if isinstance(self.options, Mapping) and self.options.get('prediction') is self.in_vitro:
            raise ValueError('Target prediction measurements must be independent of calibration measurements.')
        if self.in_vitro_reference is not None and not isinstance(self.in_vitro_reference, ReferenceTrajectoryEnsemble):
            raise TypeError('in_vitro_reference must be a ReferenceTrajectoryEnsemble or None.')
        if self.truth is not None and not isinstance(self.truth, Mapping):
            raise TypeError('truth must be a mapping of evaluation names to known values.')
        self.options = _validate_assessment_options(self.options)
        self.truth = deepcopy(self.truth)


@dataclass
class CalibrationSimulationResult:
    """Every requested experiment, including failed fits and missing intervals."""

    assessments: Sequence[IVIVCCalibrationAssessment]
    in_vitro_data: Sequence[TimeCourse | None]
    in_vivo_data: Sequence[TimeCourse | None]
    truth: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.assessments = tuple(self.assessments)
        self.in_vitro_data = tuple(self.in_vitro_data)
        self.in_vivo_data = tuple(self.in_vivo_data)
        if (not self.assessments or len(self.in_vitro_data) != len(self.assessments)
                or len(self.in_vivo_data) != len(self.assessments)):
            raise ValueError('Calibration datasets and assessments must match every requested experiment.')
        self.truth = {key: np.asarray(value, dtype=float).copy() for key, value in self.truth.items()}
        self.metadata = deepcopy(dict(self.metadata))
        for key, values in self.truth.items():
            if key not in ('relationship', 'prediction') or values.ndim != 2 or values.shape[0] != self.n_simulations:
                raise ValueError('Truth must contain one evaluation vector per requested experiment.')

    @property
    def n_simulations(self):
        return len(self.assessments)

    @property
    def fit_success(self):
        return np.array([item.fit_available for item in self.assessments], dtype=bool)

    @property
    def failure_rate(self):
        return float(1. - np.mean(self.fit_success))


def _validate_assessment_options(options):
    """Validate configuration without fitting a dummy calibration dataset."""
    if not isinstance(options, Mapping):
        raise TypeError('assessment_options must be a mapping of C1 keyword arguments.')
    options = deepcopy(dict(options))
    if 'in_vitro' in options or 'in_vivo' in options:
        raise ValueError('Calibration datasets are supplied separately from assessment_options.')
    signature(assess_ivivc_calibration).bind(None, None, **options)
    # Single-dataset Approach 1 assessment is available, but simulation
    # output/truth conventions still require Approach 2.
    if options.get('approach_id', 'approach2') != 'approach2':
        raise ValueError('Calibration simulations currently support Approach 2 only; '
                         'use assess_ivivc_calibration for single-dataset Approach 1 assessment.')
    interpolation = options.get('selected_interpolation') or ('none',)
    interpolation = (interpolation,) if isinstance(interpolation, str) else tuple(interpolation)
    if interpolation != ('none',) or options.get('selected_scalings'):
        raise ValueError('Calibration simulations use no interpolation or axis scaling.')
    method = options.get('method', 'value_ratio')
    if method not in DEFAULT_PREDICTION_METHODS['approach2']:
        raise ValueError('Choose a supported Approach 2 prediction method.')
    if 'approach2' not in get_model_spec(options['model_name']).approaches:
        raise ValueError('The calibration model must support Approach 2.')
    validate_interval_settings(options.get('confidence_level', .95), options.get('n_samples', 1000))
    norms = options.get('selected_normalizations') or ()
    norms = (norms,) if isinstance(norms, str) else tuple(norms)
    if any(name != 'one_minus' for name in norms):
        raise ValueError('Calibration comparisons require a common response scale: use no normalization or one_minus. '
                         'Apply any fixed scaling to data, references, and SD together before comparison; '
                         'data-dependent initial_value and min_max remain available in the C1 assessor.')
    options['selected_normalizations'] = norms
    if options.get('fail_fast', False):
        raise ValueError('Calibration simulations retain calculation failures; fail_fast must be false.')
    for key, allowed in (('evaluation_times', 'value_ratio'), ('response_levels', 'observed_response_inverse')):
        grid = options.get(key)
        if grid is not None:
            array = np.asarray(grid, dtype=float)
            if method != allowed or array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError(f'{key} must be a finite vector for {allowed}.')
            options[key] = array.copy()
    if method == 'observed_response_inverse':
        _inverse_interval_settings(options.get('inverse_interval_policy'), options.get('inverse_min_valid_fraction'))
    elif options.get('inverse_interval_policy') is not None or options.get('inverse_min_valid_fraction') is not None:
        raise ValueError('Inverse interval settings require observed_response_inverse.')
    prediction = options.get('prediction')
    if prediction is not None and not isinstance(prediction, TimeCourse):
        raise TypeError('prediction must be a TimeCourse.')
    if method != 'tau_ratio' and options.get('evaluation_times') is None and options.get('response_levels') is None and prediction is None:
        raise ValueError('Supply a relationship evaluation grid or a target dataset.')
    state = options.get('random_state', 12345)
    if isinstance(state, (bool, np.bool_)):
        raise TypeError('The propagation random_state must be an integer, Generator, RandomState, or None.')
    if isinstance(state, (int, np.integer)):
        np.random.SeedSequence(int(state))
    elif state is not None and not isinstance(state, (np.random.Generator, np.random.RandomState)):
        raise TypeError('The propagation random_state must be an integer, Generator, RandomState, or None.')
    if state is None:
        # Draw entropy once, then C1 clones this same stream for every candidate.
        options['random_state'] = np.random.default_rng()
    return options


def _evaluation_spec(options):
    method = options.get('method', 'value_ratio')
    spec = {}
    if method == 'tau_ratio':
        spec['relationship'] = {'quantity': 'time_constant_ratio', 'coordinate': np.array([1.])}
    elif method == 'value_ratio' and options.get('evaluation_times') is not None:
        spec['relationship'] = {'quantity': 'value_ratio', 'coordinate': np.asarray(options['evaluation_times'])}
    elif method == 'observed_response_inverse' and options.get('response_levels') is not None:
        spec['relationship'] = {'quantity': 'in_vivo_time', 'coordinate': np.asarray(options['response_levels'])}
    target = None
    if options.get('prediction') is not None:
        raw = options['prediction']
        try:
            target, _ = prepare_time_course(raw.time, raw.value, selected_normalizations=options['selected_normalizations'],
                                          observation_uncertainty=raw.observation_uncertainty)
            if not len(target.time):
                target = None
        except Exception:
            # C1 records the actual target failure. Keep its requested grid so
            # an unusable optional target cannot discard the relationship.
            target = None
        source = raw if target is None else target
        try:
            coordinate = np.asarray(source.value if method == 'observed_response_inverse' else source.time, dtype=float)
        except (ValueError, TypeError):
            coordinate = np.full(len(raw.time), np.nan)
        spec['prediction'] = {
            'quantity': 'in_vivo_response' if method == 'value_ratio' else 'in_vivo_time',
            'coordinate': coordinate.copy(),
        }
    return spec, target


def _reference_indices(reference, indices, count):
    if indices is None:
        if reference.n_draws != 1:
            raise ValueError('Truth for an ensemble requires its selected draw indices.')
        indices = np.zeros(count, dtype=int)
    indices = np.asarray(indices, dtype=int)
    if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= reference.n_draws):
        raise ValueError('Reference draw indices must match the requested experiments.')
    if not np.all(reference.valid_draws[indices] & (reference.draw_weights[indices] > 0)):
        raise ValueError('Truth indices must select valid positive-weight reference draws.')
    return indices


def _reference_values(reference, indices, times, count, reverse):
    if reference is None:
        return None
    indices = _reference_indices(reference, indices, count)
    times = np.asarray(times, dtype=float)
    values = np.full((count, len(times)), np.nan)
    lo, hi = reference.supported_time_range
    inside = np.isfinite(times) & (times >= lo) & (times <= hi)
    if inside.any():
        values[:, inside] = reference.evaluate(times[inside], draw_indices=indices)
    return 1. - values if reverse else values


def _inverse_reference(reference, indices, levels, count, reverse):
    if reference is None:
        return None
    indices = _reference_indices(reference, indices, count)
    lo, hi = reference.supported_time_range
    times = np.unique(np.r_[lo, reference.time_grid[(reference.time_grid > lo) & (reference.time_grid < hi)], hi])
    result = np.full((count, len(levels)), np.nan)
    for row, source in enumerate(indices):
        values = reference.value_draws[source]
        values = 1. - values if reverse else values
        if reference.interpolation == 'pchip':
            function = PchipInterpolator(reference.time_grid, values, extrapolate=False)
        else:
            function = lambda t: np.interp(t, reference.time_grid, values)
        on_grid = function(times)
        differences = np.diff(on_grid)
        if not (np.all(differences >= 0) or np.all(differences <= 0)) or np.all(differences == 0):
            continue  # A reference with ambiguous crossings supplies no inverse truth.
        for column, level in enumerate(levels):
            if not min(on_grid[0], on_grid[-1]) <= level <= max(on_grid[0], on_grid[-1]):
                continue
            if np.count_nonzero(on_grid == level) > 1:
                continue  # A response plateau does not define a unique time.
            result[row, column] = brentq(lambda t: float(function(t)) - level, lo, hi)
    return result


def _expand_truth(values, count, width):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full((count, width), float(values))
    if values.shape == (width,):
        return np.broadcast_to(values, (count, width)).copy()
    if values.shape == (count, width):
        return values.copy()
    raise ValueError('Truth must be a scalar, a per-point vector, or an (experiments, points) array. '
                     'Use (experiments, 1) for per-experiment scalar truth.')


def _validate_output_truth(options, truth, count):
    """Reject incompatible truth before performing repeated calibration fits."""
    if truth is not None and not isinstance(truth, Mapping):
        raise TypeError('truth must be a mapping of requested evaluation names to known values.')
    spec, _ = _evaluation_spec(options)
    if set(truth or {}) - set(spec):
        raise ValueError('Explicit truth keys must name requested relationship or prediction outputs.')
    for key, values in (truth or {}).items():
        _expand_truth(values, count, len(spec[key]['coordinate']))


def _finish_calibration_simulation(assessments, in_vitro_data, in_vivo_data, *, options,
                                   in_vitro_reference=None, in_vivo_reference=None,
                                   in_vitro_draw_indices=None, in_vivo_draw_indices=None,
                                   truth=None, metadata=None):
    count = len(assessments)
    spec, target = _evaluation_spec(options)
    method = options.get('method', 'value_ratio')
    reverse = 'one_minus' in options['selected_normalizations']
    truths, sources = {}, {}
    supplied = dict(truth or {})
    if set(supplied) - set(spec):
        raise ValueError('Explicit truth keys must name requested relationship or prediction outputs.')
    for key, item in spec.items():
        if key in supplied:
            truths[key] = _expand_truth(supplied[key], count, len(item['coordinate']))
            sources[key] = 'explicit_output_truth'
    if 'relationship' in spec and 'relationship' not in truths:
        grid = spec['relationship']['coordinate']
        if method == 'value_ratio':
            first = _reference_values(in_vitro_reference, in_vitro_draw_indices, grid, count, reverse)
            second = _reference_values(in_vivo_reference, in_vivo_draw_indices, grid, count, reverse)
            if first is not None and second is not None:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    ratio = second / first
                truths['relationship'] = np.where(np.isfinite(ratio), ratio, np.nan)
        elif method == 'observed_response_inverse':
            values = _inverse_reference(in_vivo_reference, in_vivo_draw_indices, grid, count, reverse)
            if values is not None:
                truths['relationship'] = values
        # A finite reference grid does not define an integral from zero to infinity.
        if 'relationship' in truths:
            sources['relationship'] = 'supplied_reference_trajectories'
    if target is not None and 'prediction' not in truths:
        if method == 'tau_ratio' and 'relationship' in truths:
            truths['prediction'] = truths['relationship'] * target.time[None, :]
        elif target.observation_uncertainty is None:
            if method == 'value_ratio':
                first = _reference_values(in_vitro_reference, in_vitro_draw_indices, target.time, count, reverse)
                second = _reference_values(in_vivo_reference, in_vivo_draw_indices, target.time, count, reverse)
                if first is not None and second is not None:
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        values = target.value[None, :] * second / first
                    truths['prediction'] = np.where(np.isfinite(values), values, np.nan)
            elif method == 'observed_response_inverse':
                values = _inverse_reference(in_vivo_reference, in_vivo_draw_indices, target.value, count, reverse)
                if values is not None:
                    truths['prediction'] = values
        if 'prediction' in truths:
            sources['prediction'] = 'fixed_target_inputs_and_supplied_relationship_truth'
    record = dict(metadata or {})
    record.update(evaluation_spec=spec, truth_sources=sources,
                  evaluation_scale='common_input_scale_with_optional_one_minus',
                  confidence_level=float(options.get('confidence_level', .95)),
                  propagation_samples=int(options.get('n_samples', 1000)),
                  future_observation_variability_included=False)
    for key, values in (('in_vitro_draw_indices', in_vitro_draw_indices),
                        ('in_vivo_draw_indices', in_vivo_draw_indices)):
        if values is not None:
            record[key] = np.asarray(values, dtype=int).copy()
    return CalibrationSimulationResult(assessments, in_vitro_data, in_vivo_data, truths, record)


def _divide(values, counts):
    return np.divide(values, counts, out=np.full(np.shape(values), np.nan, dtype=float), where=np.asarray(counts) > 0)


def summarize_calibration_simulation(result: CalibrationSimulationResult) -> dict:
    """Summarize precision and reliability without dropping failed experiments.

    Widths and errors are conditional on the corresponding values being
    available, with their denominators reported. Coverage uses ALL experiments
    with known truth; unavailable intervals count as not covering. Conditional
    coverage among available intervals is reported separately. No ranking or
    sufficiency threshold is imposed.
    """
    count = result.n_simulations
    summary = {
        'n_simulations': count, 'n_fit_successful': int(result.fit_success.sum()),
        'fit_failure_rate': result.failure_rate,
        'error_counts': dict(Counter(stage for item in result.assessments for stage in item.errors)),
    }
    for key in ('relationship', 'prediction'):
        evaluations = [getattr(item, key) for item in result.assessments]
        first = next((item for item in evaluations if item is not None), None)
        spec = result.metadata.get('evaluation_spec', {}).get(key)
        if first is None and spec is None:
            continue
        quantity = first.quantity if first is not None else spec['quantity']
        coordinate = first.coordinate if first is not None else np.asarray(spec['coordinate'])
        width = len(coordinate)
        estimates = np.full((count, width), np.nan)
        widths = estimates.copy()
        points = np.zeros((count, width), dtype=bool)
        intervals = points.copy()
        covered = points.copy()
        actual = result.truth.get(key, np.full((count, width), np.nan))
        if actual.shape != (count, width):
            raise ValueError('Truth shape must match the evaluation grid.')
        for row, evaluation in enumerate(evaluations):
            if evaluation is None:
                continue
            if evaluation.quantity != quantity or not np.array_equal(evaluation.coordinate, coordinate, equal_nan=True):
                raise ValueError('Calibration comparisons require the same evaluation quantity and coordinates.')
            estimates[row] = evaluation.estimate
            widths[row] = evaluation.interval_width
            points[row] = evaluation.point_available
            intervals[row] = evaluation.interval_available
            covered[row] = intervals[row] & (evaluation.lower <= actual[row]) & (actual[row] <= evaluation.upper)
        known = np.isfinite(actual)
        paired = known & points
        errors = np.where(paired, estimates - actual, np.nan)
        interval_truth = known & intervals
        error_counts = paired.sum(axis=0)
        truth_counts = known.sum(axis=0)
        finite_width = intervals & np.isfinite(widths)
        width_counts = finite_width.sum(axis=0)
        with np.errstate(over='ignore', invalid='ignore'):
            squares = np.where(paired, errors**2, 0.)
        summary[key] = {
            'quantity': quantity, 'coordinate': coordinate.copy(),
            'coverage_scope': 'pooled_pointwise',
            'widths_conditional_on_availability': True,
            'point_counts': points.sum(axis=0), 'interval_counts': intervals.sum(axis=0),
            'truth_counts': truth_counts, 'interval_truth_counts': interval_truth.sum(axis=0),
            'error_counts': error_counts, 'width_counts': width_counts,
            'point_availability': points.mean(axis=0), 'interval_availability': intervals.mean(axis=0),
            'complete_point_rate': float(np.all(points, axis=1).mean()) if width else np.nan,
            'complete_interval_rate': float(np.all(intervals, axis=1).mean()) if width else np.nan,
            'mean_interval_width': float(_divide(np.where(finite_width, widths, 0.).sum(), finite_width.sum())),
            'pointwise_mean_interval_width': _divide(np.where(finite_width, widths, 0.).sum(axis=0), width_counts),
            'rmse': float(np.sqrt(_divide(squares.sum(), paired.sum()))),
            'pointwise_rmse': np.sqrt(_divide(squares.sum(axis=0), error_counts)),
            'coverage': float(_divide(covered.sum(), known.sum())),
            'conditional_coverage': float(_divide(covered.sum(), interval_truth.sum())),
            'pointwise_coverage': _divide(covered.sum(axis=0), truth_counts),
            'pointwise_conditional_coverage': _divide(covered.sum(axis=0), interval_truth.sum(axis=0)),
        }
    return summary
