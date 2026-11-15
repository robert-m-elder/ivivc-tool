"""Build planning references using the shared IVIVC and time-scaling APIs.

Inputs are already interpreted time courses. This adapter applies the selected
response preprocessing once and never adds future measurement noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ivivc_core import (
    DEFAULT_PREDICTION_METHODS, TimeCourse, fit_ivivc, get_model_spec,
    iter_approach_specs, iter_model_specs, prepare_ivivc_data, prepare_time_course,
)
from study_design_core import (
    EmpiricalTimeScaleDistribution, ReferenceTrajectoryEnsemble,
    fit_reference_profile, reference_from_ivivc, reference_from_time_scaling,
)

from .design_planning import PlanningInputError


@dataclass
class ReferenceInputs:
    reference: ReferenceTrajectoryEnsemble
    analyzed_source: TimeCourse | None = None
    preprocessing: Any = None
    fit: Any = None
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    analysis: Any = None
    source_fits: dict = field(default_factory=dict)


METHOD_LABELS = {
    'time_mapping': 'Map in vitro time to in vivo time',
    'value_ratio': 'Response ratio',
    'tau_ratio': 'Time-constant ratio',
    'observed_response_inverse': 'Inverse response mapping',
    'value_scaling': 'Response scaling',
    'time_scaling': 'Time scaling',
}


def reference_options():
    def models(specs):
        return [{'id': spec.key, 'label': spec.display_name} for spec in specs]
    return {
        'approaches': [
            {'id': approach.key, 'label': approach.display_name,
             'models': models(iter_model_specs(approach.key)),
             'methods': [{'id': method, 'label': METHOD_LABELS[method]}
                         for method in DEFAULT_PREDICTION_METHODS[approach.key]]}
            for approach in iter_approach_specs()
        ],
        'models': models(iter_model_specs()),
    }


def _normalizations(settings):
    result = ([] if settings['normalization'] == 'none' else [settings['normalization']])
    if settings.get('one_minus'):
        result.append('one_minus')
    return result


def _config(settings):
    return {'cv_scheme': 'none', 'grid_search_num_points': 25,
            'grid_search_num_cores': 1, 'grid_search_random_state': settings['random_seed']}


def _course(course, name, *, minimum_rows=2):
    if not isinstance(course, TimeCourse):
        raise TypeError(f'{name} must be an imported TimeCourse.')
    if len(course.time) > 5000:
        raise PlanningInputError('Use at most 5000 observations per source dataset.', name)
    if (len(course.time) < minimum_rows or len(np.unique(course.time)) < 2
            or not np.isfinite(course.time).all() or not np.isfinite(course.value).all()):
        raise PlanningInputError(
            f'This source needs at least {minimum_rows} finite observations and two distinct times.', name)


def _prepared(course, settings, name):
    try:
        analyzed, metadata = prepare_time_course(
            course.time, course.value, label=course.label,
            observation_uncertainty=course.observation_uncertainty,
            selected_normalizations=_normalizations(settings),
        )
    except (ValueError, KeyError, FloatingPointError) as error:
        raise PlanningInputError(f'Response preprocessing could not be applied: {error}', 'normalization') from error
    if not np.isfinite(analyzed.value).all():
        raise PlanningInputError('Response normalization produced nonfinite values.', 'normalization')
    return analyzed, metadata


def _fit_is_usable(fit, control):
    if not np.isfinite(fit.covariance).all():
        raise PlanningInputError(
            'Fitted parameter covariance is unavailable. Check the source data or select a simpler model.', control)


def _normalization_warnings(metadata):
    warnings = []
    for record in metadata.normalizations:
        for constants in (record.in_vitro_parameters, record.in_vivo_parameters):
            if (record.name == 'initial_value' and constants is not None
                    and not constants.get('applied', True)):
                warnings.append('An initial response was zero; initial-value normalization was not applied to that dataset.')
    return warnings


def _fit_rows(fits):
    rows = []
    for label, fit in fits.items():
        rows.append((f'{label}: model', fit.model_name))
        rows.append((f'{label}: RMSE', fit.fit_metrics.get('rmse', np.nan)))
        rows.append((f'{label}: parameters', ', '.join(f'{name}={value:.5g}' for name, value in fit.parameters.items())))
    return rows


def _fit_parameters(fits):
    rows = []
    for label, fit in fits.items():
        for index, (name, value) in enumerate(fit.parameters.items()):
            variance = fit.covariance[index, index]
            rows.append({'label': label, 'parameter': name, 'value': value,
                         'se': float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else None})
    return rows


def _curve(label, course, fit=None, *, environment=None):
    curve = {'label': label, 'time': course.time, 'value': course.value, 'environment': environment}
    if fit is not None:
        grid = np.linspace(np.min(fit.time), np.max(fit.time), 101)
        values = get_model_spec(fit.model_name).model_function(grid, *fit.parameter_vector)
        if np.isfinite(values).all():
            curve.update(fitted_time=grid, fitted_value=values)
    return curve


def build_ivivc_reference(in_vitro, in_vivo, prediction, form, settings, time_grid):
    """Fit one selected IVIVC and retain its joint prediction trajectories."""
    approach = form.get('ivivc_approach', 'approach2')
    catalog = {item['id']: item for item in reference_options()['approaches']}
    if approach not in catalog:
        raise PlanningInputError('Select a supported IVIVC approach.', 'ivivc_approach')
    default_methods = {'approach1': 'time_mapping', 'approach2': 'value_ratio', 'approach3': 'value_scaling'}
    method = form.get('ivivc_method', default_methods[approach])
    if method not in DEFAULT_PREDICTION_METHODS[approach]:
        raise PlanningInputError('Select a prediction method available for this IVIVC approach.', 'ivivc_method')
    model = None if approach == 'approach3' else form.get('ivivc_model', 'exponential')
    if model is not None and model not in {item['id'] for item in catalog[approach]['models']}:
        raise PlanningInputError('Select a model registered for this IVIVC approach.', 'ivivc_model')
    interpolation = 'none' if approach == 'approach2' else form.get('ivivc_interpolation', 'default')
    if interpolation not in ('default', 'alternative', 'none'):
        raise PlanningInputError('Select a supported interpolation method.', 'ivivc_interpolation')
    for course, name in ((in_vitro, 'ivivc-calibration-file'), (in_vivo, 'ivivc-calibration-file'),
                         (prediction, 'prediction-file')):
        minimum = (len(get_model_spec(model).parameter_names) + 1
                   if model is not None and approach == 'approach2' and course is not prediction else 2)
        _course(course, name, minimum_rows=minimum)
        if approach != 'approach2' and course.observation_uncertainty is not None:
            raise PlanningInputError(
                'Supplied response SD/SEM is supported only for Approach 2 IVIVC references. '
                'Choose Approach 2, or explicitly select unspecified uncertainty for this dataset.', name)
    if prediction is in_vitro and prediction.observation_uncertainty is not None:
        raise PlanningInputError('Target prediction measurements with SD/SEM must be independent of calibration measurements.', 'prediction-file')
    try:
        prepared = prepare_ivivc_data(
            in_vitro, in_vivo, selected_normalizations=_normalizations(settings),
            selected_interpolation=[interpolation],
        )
    except (ValueError, KeyError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'IVIVC preprocessing could not be applied: {error}', 'ivivc_interpolation') from error
    target, target_preprocessing = _prepared(prediction, settings, 'prediction-file')
    selection = approach if model is None else f'{approach}:{model}'
    try:
        analysis = fit_ivivc(prepared, selections=selection, selected_metrics=['rmse'],
                             analysis_config=_config(settings), fail_fast=True)
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'The selected IVIVC could not be fitted: {error}', 'ivivc_model' if model else 'ivivc_approach') from error
    selected = analysis.model_results[selection]
    fits = ({'In vitro calibration': selected.in_vitro_fit,
             'In vivo calibration': selected.in_vivo_fit} if approach == 'approach2'
            else {'Time mapping': selected.time_mapping_fit} if approach == 'approach1' else {})
    contributors = ([selected.in_vivo_fit] if approach == 'approach2' and method == 'observed_response_inverse'
                    else list(fits.values()))
    for fit in contributors:
        _fit_is_usable(fit, 'ivivc_model')
    try:
        natural = reference_from_ivivc(
            analysis, target, selection=selection, method=method,
            n_samples=settings['reference_draws'], random_state=settings['random_seed'],
        )
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'The selected IVIVC prediction could not supply a reference: {error}', 'ivivc_method') from error
    grid = np.asarray(time_grid, dtype=float)
    lower, upper = natural.supported_time_range
    tolerance = 1e-12 * max(1., abs(lower), abs(upper))
    if grid[0] < lower - tolerance or grid[-1] > upper + tolerance:
        field = 'first_time' if grid[0] < lower - tolerance else 'last_time'
        if grid[0] < lower - tolerance:
            completed_times = settings.get('completed_times', ())
            if len(completed_times) and min(completed_times) < lower - tolerance:
                field = 'completed-target-file'
            elif 'baseline_time' in settings and settings['baseline_time'] < lower - tolerance:
                field = 'baseline_time'
        raise PlanningInputError(
            f'The selected IVIVC reference supports in vivo times {lower:.6g} to {upper:.6g}. '
            'Set the planning range inside this domain or revise the prediction data/method. '
            'No baseline or requested time was moved.', field)
    # Keep the same complete draws and source diagnostics; do not sample again.
    try:
        reference = ReferenceTrajectoryEnsemble(
            grid, natural.evaluate(grid), natural.valid_draws,
            draw_weights=natural.draw_weights, central_curve=natural.evaluate_central(grid),
            interpolation=natural.interpolation,
            metadata={**natural.metadata, 'ivivc_available_time_range': (lower, upper)},
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(
            f'The selected IVIVC trajectories could not cover the requested planning grid: {error}',
            'ivivc_method') from error
    warnings = _normalization_warnings(prepared.preprocessing) + _normalization_warnings(target_preprocessing)
    if approach == 'approach1':
        warnings.append('Approach 1 uses the shared time-mapping fit and its fitted uncertainty. Experimental response SD/SEM is not propagated by that approach.')
    if approach == 'approach3':
        warnings.append('Approach 3 supplies one deterministic reference trajectory; its calibration and interpolation uncertainty are not quantified. Future measurement variability is still simulated separately.')
    if any(np.linalg.matrix_rank(fit.covariance) < fit.n_parameters for fit in contributors):
        warnings.append('An IVIVC fitted parameter covariance is rank deficient; reference draws may underrepresent uncertainty. Review the calibration data and model in the IVIVC app.')
    if not prepared.in_vivo.time.min() <= grid[0] or grid[-1] > prepared.in_vivo.time.max():
        warnings.append('The planning horizon extends beyond the in vivo calibration time range. Model or mapping extrapolation may contribute to this IVIVC reference.')
    if reference.n_valid_draws < reference.n_draws:
        warnings.append('Some IVIVC draws could not supply complete trajectories and have zero reference weight.')
    details = {
        'kind': 'ivivc', 'label': 'IVIVC prediction', 'selection': selection, 'method': method,
        'deterministic': approach == 'approach3', 'fit_rows': _fit_parameters(fits),
        'available_time_range': (lower, upper), 'prediction_preprocessing': target_preprocessing,
        'rows': [('IVIVC approach', catalog[approach]['label']), ('IVIVC selection', selection),
                 ('Prediction method', METHOD_LABELS[method]), ('Calibration interpolation', interpolation),
                 ('Available IVIVC in vivo time range', f'{lower:.6g}–{upper:.6g}'), *_fit_rows(fits)],
        'source_curves': [_curve('In vitro calibration', prepared.in_vitro,
                                selected.in_vitro_fit if approach == 'approach2' else None,
                                environment='in_vitro'),
                          _curve('In vivo calibration', prepared.in_vivo,
                                selected.in_vivo_fit if approach == 'approach2' else None,
                                environment='in_vivo'),
                          _curve('Target in vitro', target, environment='in_vitro')],
    }
    return ReferenceInputs(reference, None, prepared.preprocessing, None,
                           list(dict.fromkeys(warnings)), details, analysis, fits)


def build_scaling_reference(source, distribution, form, settings, time_grid):
    """Cross source-parameter draws with every positive-weight empirical ratio."""
    if not isinstance(distribution, EmpiricalTimeScaleDistribution):
        raise TypeError('distribution must be an interpreted empirical distribution.')
    if len(distribution.ratios) > 5000:
        raise PlanningInputError('Use at most 5000 empirical ratio records.', 'time-scale-file')
    active_count = int(np.count_nonzero(distribution.weights))
    if active_count * settings['reference_draws'] > 2000:
        raise PlanningInputError(
            'Limit empirical references to 2000 complete curves per request: reduce parameter draws '
            'or the number of positive-weight ratio records. All supplied positive weights are retained.', 'reference_draws')
    model = form.get('scaling_model', 'exponential')
    if model not in {spec.key for spec in iter_model_specs()}:
        raise PlanningInputError('Select a registered in vitro source model.', 'scaling_model')
    _course(source, 'scaling-source-file', minimum_rows=len(get_model_spec(model).parameter_names) + 1)
    analyzed, preprocessing = _prepared(source, settings, 'scaling-source-file')
    allow_extrapolation = str(form.get('scaling_allow_extrapolation', '')).lower() in ('1', 'true', 'on', 'yes')
    try:
        fit = fit_reference_profile(analyzed, model_name=model, analysis_config=_config(settings))
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'The in vitro source could not be fitted: {error}', 'scaling_model') from error
    _fit_is_usable(fit, 'scaling_model')
    try:
        reference = reference_from_time_scaling(
            fit, distribution, time_grid, n_parameter_draws=settings['reference_draws'],
            random_state=settings['random_seed'], allow_extrapolation=allow_extrapolation,
        )
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        field = 'scaling_allow_extrapolation' if 'extrapolation' in str(error) else 'scaling_model'
        raise PlanningInputError(f'Empirical time scaling could not supply the requested horizon: {error}', field) from error
    warnings = _normalization_warnings(preprocessing)
    warnings.append('Source-fit uncertainty and empirical time-scale uncertainty are assumed independent. Supplied empirical weights are preserved; they do not establish transferability to this target material.')
    if np.any(reference.metadata['extrapolated_points_by_ratio']):
        warnings.append('The selected empirical ratios require fitted in vitro source extrapolation. This was explicitly enabled and its uncertainty remains conditional on the selected source model.')
    if np.linalg.matrix_rank(fit.covariance) < fit.n_parameters:
        warnings.append('The source parameter covariance is rank deficient; reference draws may underrepresent uncertainty. Review the source and model in the IVIVC app.')
    details = {
        'kind': 'scaling', 'label': 'Empirical time scaling',
        'rows': [('In vitro source model', model), ('Time-scale ratio', 'in vivo / in vitro'),
                 ('Empirical ratio records', len(distribution.ratios)),
                 ('Positive-weight ratio records', active_count),
                 ('Ratio 2.5th / median / 97.5th percentiles', ' / '.join(f'{q:.5g}' for q in distribution.quantile([.025, .5, .975]))),
                 ('Source parameter draws per ratio', settings['reference_draws']),
                 ('Fitted source extrapolation allowed', 'Yes' if allow_extrapolation else 'No'), *_fit_rows({'In vitro source': fit})],
        'source_curves': [_curve('In vitro source', analyzed, fit, environment='in_vitro')],
        'fit_rows': _fit_parameters({'In vitro source': fit}),
        'ratios': distribution.ratios.copy(), 'weights': distribution.weights.copy(),
        'source_ids': distribution.source_ids,
    }
    return ReferenceInputs(reference, None, preprocessing, fit, warnings, details,
                           source_fits={'In vitro source': fit})
