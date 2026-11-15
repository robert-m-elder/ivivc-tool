"""Request-scoped calibration-design comparison using the shared IVIVC pipeline.

Source normalization constants are established once before simulation. Existing
measurements remain fixed, and source uncertainty is separate from new noise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import re

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse, iter_model_specs, prepare_time_course
from study_design_core import (
    CalibrationDesign, DesignConstraints, ObservationModel, compare_calibration_designs,
    existing_observations_from_time_course, fit_reference_model_ensemble,
    fit_reference_profile, optimize_local_i_optimal_designs, reference_from_profile_fit,
)
from .design_planning import PlanningInputError, _checked, _number, _times
from .design_reference_sources import reference_options


SIDES = ('in_vitro', 'in_vivo')
MAX_CANDIDATES = 6


@dataclass
class CalibrationComparisonResult:
    settings: dict
    comparison: object
    designs: dict
    references: dict
    source_fits: dict
    analyzed_existing: dict
    source_details: dict
    warnings: list[str]
    generation_outcomes: list[dict]


def calibration_options():
    catalog = reference_options()
    return {**catalog, 'approaches': [row for row in catalog['approaches']
                                    if row['id'] in ('approach1', 'approach2')]}


def _seed(seed, stream):
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def _settings(form, *, generation_only=False):
    if form.get('task', 'calibration') != 'calibration':
        raise PlanningInputError('Choose the calibration-comparison task.', 'task')
    varied = form.get('varied_environment', 'in_vivo')
    if varied not in (*SIDES, 'both'):
        raise PlanningInputError('Choose in vitro, in vivo, or both environments.', 'varied_environment')
    approach = form.get('cal_approach', 'approach2')
    catalog = {row['id']: row for row in calibration_options()['approaches']}
    if approach not in catalog:
        raise PlanningInputError('Choose Approach 1 or Approach 2.', 'cal_approach')
    normalization = form.get('cal_normalization', 'none')
    if normalization not in ('none', 'initial_value', 'min_max'):
        raise PlanningInputError('Choose a supported fixed response normalization.', 'cal_normalization')
    result = dict(varied_environment=varied, approach_id=approach, normalization=normalization,
                  one_minus=_checked(form, 'cal_one_minus'),
                  reference_draws=_number(form, 'reference_draws', 100, integer=True, minimum=2, maximum=200),
                  random_seed=_number(form, 'random_seed', 12345, integer=True, minimum=0, maximum=2**32-1))
    if generation_only:
        return result
    model = form.get('cal_model', 'linear' if approach == 'approach1' else 'exponential')
    if model not in {row['id'] for row in catalog[approach]['models']}:
        raise PlanningInputError('Choose a model registered for this approach.', 'cal_model')
    method = form.get('cal_method', 'time_mapping' if approach == 'approach1' else 'value_ratio')
    if method not in {row['id'] for row in catalog[approach]['methods']}:
        raise PlanningInputError('Choose a prediction method available for this approach.', 'cal_method')
    interpolation = form.get('cal_interpolation', 'default') if approach == 'approach1' else 'none'
    if interpolation not in ('none', 'default', 'alternative'):
        raise PlanningInputError('Choose a supported interpolation scheme.', 'cal_interpolation')
    result.update(model_name=model, method=method, interpolation=interpolation,
                  relationship=_checked(form, 'cal_relationship'),
                  assess_target=_checked(form, 'cal_assess_target'),
                  interval_draws=_number(form, 'cal_interval_draws', 100, integer=True, minimum=2, maximum=500),
                  n_simulations=_number(form, 'n_simulations', 100, integer=True, minimum=2, maximum=200))
    if not result['relationship'] and not result['assess_target']:
        raise PlanningInputError('Assess the relationship, target predictions, or both.', 'cal_relationship')
    if result['relationship'] and method in ('time_mapping', 'value_ratio'):
        result['evaluation_times'] = _times(form.get('cal_evaluation_times', ''), 'cal_evaluation_times', allow_empty=False)
        if len(result['evaluation_times']) > 100 or np.any(result['evaluation_times'] < 0):
            raise PlanningInputError('Use at most 100 nonnegative evaluation times.', 'cal_evaluation_times')
    if result['relationship'] and method == 'observed_response_inverse':
        result['response_levels'] = _times(form.get('cal_response_levels', ''), 'cal_response_levels', allow_empty=False)
        if len(result['response_levels']) > 100:
            raise PlanningInputError('Use at most 100 response levels.', 'cal_response_levels')
    return result


def _course(course, field):
    if not isinstance(course, TimeCourse) or not 1 <= len(course.time) <= 5000:
        raise PlanningInputError('Import 1 to 5000 source observations.', field)
    if not np.isfinite(course.time).all() or not np.isfinite(course.value).all() or np.any(course.time < 0):
        raise PlanningInputError('Source observations need finite responses and nonnegative finite times.', field)
    return course


def _prepare(course, settings):
    norms = [] if settings['normalization'] == 'none' else [settings['normalization']]
    if settings['one_minus']:
        norms.append('one_minus')
    try:
        analyzed, metadata = prepare_time_course(
            course.time, course.value, label=course.label,
            observation_uncertainty=course.observation_uncertainty,
            selected_normalizations=norms,
        )
    except (ValueError, KeyError, FloatingPointError) as error:
        raise PlanningInputError(f'Response preprocessing failed: {error}', 'cal_normalization') from error
    if not np.isfinite(analyzed.value).all():
        raise PlanningInputError('Normalization produced nonfinite values.', 'cal_normalization')
    slope, intercept = 1., 0.
    for record in metadata.normalizations:
        constants = record.in_vitro_parameters
        if record.name == 'initial_value' and constants.get('applied', True):
            divisor = constants['initial_value']
            slope, intercept = slope / divisor, intercept / divisor
        elif record.name == 'min_max':
            divisor = constants['range']
            if not np.isfinite(divisor) or divisor <= 0:
                raise PlanningInputError('Min–max normalization requires a positive source response range.', 'cal_normalization')
            slope, intercept = slope / divisor, (intercept - constants['minimum']) / divisor
        elif record.name == 'one_minus':
            slope, intercept = -slope, 1. - intercept
    return analyzed, dict(slope=slope, intercept=intercept,
                          records=[{'name': r.name, **dict(r.in_vitro_parameters)} for r in metadata.normalizations])


def _fixed_transform(course, transform):
    if course is None:
        return None
    precision = course.observation_uncertainty
    if precision is not None:
        precision = ResolvedObservationUncertainty(
            np.abs(transform['slope']) * precision.sigma,
            response_kind=precision.response_kind, input_form=precision.input_form,
            replicate_counts=precision.replicate_counts, source=precision.source,
            response_units=precision.response_units,
        )
    return TimeCourse(course.time, transform['slope'] * course.value + transform['intercept'],
                      label=course.label, observation_uncertainty=precision)


def _future_model(form, side, profile):
    prefix = 'cal_' + side + '_'
    kind = form.get(prefix + 'observation_kind', 'constant_sd')
    if kind not in ('constant_sd', 'constant_cv', 'time_profile_sd'):
        raise PlanningInputError('Choose a future measurement variability model.', prefix + 'observation_kind')
    if kind == 'time_profile_sd':
        if not isinstance(profile, ObservationModel) or profile.kind != kind:
            raise PlanningInputError('Import a future individual-SD profile.', 'calibration-' + side.replace('_', '-') + '-sd-profile-file')
        return replace(profile, replicates_per_time=1, minimum_sd=0.)
    value = _number(form, prefix + 'observation_value', .05, minimum=0.)
    floor = _number(form, prefix + 'minimum_sd', .001, minimum=0.) if kind == 'constant_cv' else 0.
    if value <= 0 and floor <= 0:
        raise PlanningInputError('Use positive future variability.', prefix + 'observation_value')
    return ObservationModel(kind, value, minimum_sd=floor)


def _candidate_requests(form, settings, *, generation_only=False):
    for key in form:
        match = re.fullmatch(r'cal_candidate_(\d+)_name', key)
        if match and int(match[1]) >= MAX_CANDIDATES:
            raise PlanningInputError('Use at most six calibration candidates.', 'calibration-candidates')
    candidates, names = [], set()
    for index in range(MAX_CANDIDATES):
        prefix = f'cal_candidate_{index}_'
        if prefix + 'name' not in form:
            continue
        name = str(form[prefix + 'name']).strip()
        if not name or len(name) > 80 or name in names:
            raise PlanningInputError('Use distinct candidate names of 1 to 80 characters.', prefix + 'name')
        names.add(name)
        row = dict(name=name, index=index, sides={})
        for side in SIDES:
            if settings['varied_environment'] not in (side, 'both'):
                continue
            field = prefix + side + '_'
            kind = form.get(field + 'kind', 'manual')
            if kind not in ('manual', 'generated'):
                raise PlanningInputError('Enter sampling times or generate a curve-recovery schedule.', field + 'kind')
            reps = _number(form, field + 'replicates', 3, integer=True, minimum=1, maximum=100)
            request = dict(kind=kind, replicates=reps, field=field)
            if kind == 'manual':
                request['times'] = _times(form.get(field + 'times', ''), field + 'times')
                if len(request['times']) > 20 or np.any(request['times'] < 0):
                    raise PlanningInputError('Use at most 20 nonnegative added times.', field + 'times')
            else:
                request['number_of_times'] = _number(form, field + 'number_of_times', 3, integer=True, minimum=1, maximum=20)
            row['sides'][side] = request
        candidates.append(row)
    if not candidates:
        raise PlanningInputError('Add at least one named calibration candidate.', 'calibration-candidates')
    if generation_only:
        return candidates
    if len(candidates) * settings['n_simulations'] > 1200:
        raise PlanningInputError('Use at most 1200 candidate simulations per request.', 'n_simulations')
    if len(candidates) * settings['n_simulations'] * settings['interval_draws'] > 100000:
        raise PlanningInputError('Reduce simulations, candidates, or interval draws to at most 100000 propagation draws per request.', 'cal_interval_draws')
    return candidates


def _grid(form, side):
    prefix = 'cal_' + side + '_'
    first = _number(form, prefix + 'first_time', 0., minimum=0.)
    last = _number(form, prefix + 'last_time', 52., minimum=0.)
    step = _number(form, prefix + 'time_step', 2., minimum=0.)
    if last <= first:
        raise PlanningInputError('Use an increasing opportunity range.', prefix + 'last_time')
    if step <= 0:
        raise PlanningInputError('Use a positive time step.', prefix + 'time_step')
    count = (last - first) / step
    if not np.isfinite(count) or count > 499:
        raise PlanningInputError('Use at most 500 candidate opportunities.', prefix + 'time_step')
    grid = first + np.arange(int(np.floor(count + 1e-10)) + 1) * step
    grid[np.isclose(grid, last, rtol=0., atol=1e-10 * max(1., last))] = last
    return grid


def _build_calibration_candidates(form, settings, candidates, *, existing_in_vitro=None,
                                  existing_in_vivo=None, in_vitro_source=None,
                                  in_vivo_source=None, in_vitro_profile=None,
                                  in_vivo_profile=None, generation_only=False):
    """Prepare fixed source scales and plans once for preview and simulation."""
    existing = dict(in_vitro=existing_in_vitro, in_vivo=existing_in_vivo)
    supplied_sources = dict(in_vitro=in_vitro_source, in_vivo=in_vivo_source)
    profiles = dict(in_vitro=in_vitro_profile, in_vivo=in_vivo_profile)
    references, fits, details, models, analyzed = {}, {}, {}, {}, {}
    warnings, outcomes = [], []
    generated_requests = {side: any(row['sides'].get(side, {}).get('kind') == 'generated' for row in candidates) for side in SIDES}
    added_requests = {side: any(row['sides'].get(side, {}).get('kind') == 'generated' or
                               len(row['sides'].get(side, {}).get('times', ()))
                               for row in candidates) for side in SIDES}
    grids = {side: _grid(form, side) for side in SIDES if generated_requests[side]}
    for side_index, side in enumerate(SIDES):
        prefix = 'cal_' + side + '_'
        varied = settings['varied_environment'] in (side, 'both')
        if generation_only and not varied:
            continue
        mode = form.get(prefix + 'mode', 'prospective') if varied else 'fixed'
        if mode not in ('fixed', 'prospective', 'additions') or (varied and mode == 'fixed'):
            raise PlanningInputError('Choose a prospective dataset or additions to existing observations.', prefix + 'mode')
        old = None if mode == 'prospective' else _course(existing[side], 'existing-' + side.replace('_', '-') + '-file')
        source_kind = form.get(prefix + 'source', 'existing' if old is not None else 'file')
        if source_kind not in ('existing', 'file', 'none') or (varied and source_kind == 'none'):
            raise PlanningInputError('Choose existing source observations or a separate source file.', prefix + 'source')
        if source_kind == 'existing' and old is None:
            raise PlanningInputError('Existing calibration data are required for this source option.', prefix + 'source')
        raw_source = old if source_kind in ('existing', 'none') else supplied_sources[side]
        source_field = ('existing-' + side.replace('_', '-') + '-file' if source_kind in ('existing', 'none')
                        else 'cal-' + side.replace('_', '-') + '-source-file')
        raw_source = _course(raw_source, source_field)
        source, transform = _prepare(raw_source, settings)
        if any(record['name'] == 'initial_value' and not record.get('applied', True)
               for record in transform['records']):
            warnings.append(f'{side}: the initial source response was zero; initial-value normalization was not applied.')
        old = _fixed_transform(old, transform)
        future = _future_model(form, side, profiles[side]) if varied else None
        if old is not None and old.observation_uncertainty is None and varied:
            historical = form.get(prefix + 'historical_error', 'supplied')
            if historical not in ('supplied', 'future_model'):
                raise PlanningInputError('Choose a historical precision assumption.', prefix + 'historical_error')
            if historical == 'future_model':
                sd = future.standard_deviation(old.value, times=old.time)
                if not np.isfinite(sd).all() or np.any(sd <= 0):
                    raise PlanningInputError('Assumed historical SD must be positive at all observed values.', prefix + 'historical_error')
                old = TimeCourse(old.time, old.value, label=old.label,
                    observation_uncertainty=ResolvedObservationUncertainty(sd, response_kind='individual', source='future_model_assumption'))
                warnings.append(f'{side}: unspecified historical precision uses the explicit future-model SD at observed responses once; no historical replicate reduction is assumed.')
                if source_kind == 'existing':
                    source = old
            elif (settings['approach_id'] == 'approach2' and added_requests[side]) or generated_requests[side]:
                raise PlanningInputError('Supply historical SD/SEM or explicitly reuse the future model for missing precision.', prefix + 'historical_error')
            elif settings['approach_id'] == 'approach1':
                warnings.append(f'{side}: Approach 1 retains historical observations without supplied precision; its ordinary mapping fit does not use response SD/SEM.')
        analyzed[side] = old
        details[side] = dict(mode=mode, source_kind=source_kind, normalization=transform,
                             observed_time_range=(float(source.time.min()), float(source.time.max())),
                             analyzed_source=source)
        if side in grids:
            details[side]['generation_grid'] = dict(first_time=float(grids[side][0]),
                last_time_bound=float(form.get(prefix + 'last_time', 52.)),
                time_step=float(form.get(prefix + 'time_step', 2.)),
                candidate_times=grids[side].copy())
        references[side] = None
        if source_kind == 'none':
            warnings.append(f'{side}: no generating reference was requested. Truth-dependent error or coverage may be unavailable; fit precision and calculation availability are still assessed.')
            continue
        model = form.get(prefix + 'model', 'exponential')
        specs = {spec.key: spec for spec in iter_model_specs()}
        if model not in specs:
            raise PlanningInputError('Choose a registered generating-curve model.', prefix + 'model')
        if len(source.time) <= len(specs[model].parameter_names) or len(np.unique(source.time)) < 2:
            raise PlanningInputError('The source model requires more observations than fitted parameters and at least two distinct times.', prefix + 'model')
        ranges = [source.time] + ([] if old is None else [old.time])
        for row in candidates:
            request = row['sides'].get(side)
            if request is not None and request['kind'] == 'manual' and len(request['times']):
                ranges.append(request['times'])
        if side in grids:
            ranges.append(grids[side])
            ranges.append(np.asarray([grids[side][0], float(form.get(prefix + 'last_time', 52.))]))
        requested = np.concatenate(ranges)
        lower, upper = float(requested.min()), float(requested.max())
        if not lower < upper:
            raise PlanningInputError('The generating source needs an increasing time range.', prefix + 'model')
        outside = lower < source.time.min() or upper > source.time.max()
        if outside and not _checked(form, prefix + 'allow_extrapolation'):
            raise PlanningInputError('Proposed opportunities require source-model extrapolation. Shorten the range or explicitly allow extrapolation.', prefix + 'allow_extrapolation')
        config = dict(cv_scheme='none', grid_search_num_points=25, grid_search_num_cores=1,
                      grid_search_random_state=_seed(settings['random_seed'], 10 + side_index))
        try:
            fitted = fit_reference_profile(source, model_name=model, analysis_config=config)
            if not np.isfinite(fitted.covariance).all():
                raise ValueError('Fitted covariance is unavailable; use a simpler source model or additional source data.')
            reference = reference_from_profile_fit(fitted, np.linspace(lower, upper, 101),
                n_samples=settings['reference_draws'], random_state=_seed(settings['random_seed'], 20 + side_index))
        except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
            raise PlanningInputError(f'Generating reference could not be fitted: {error}', prefix + 'model') from error
        fits[side], references[side] = fitted, reference
        details[side].update(model=model, reference_time_range=(lower, upper), source_parameter_draws=reference.n_draws)
        if outside:
            warnings.append(f'{side}: the explicitly allowed source-model extrapolation is conditional on that model and does not establish unobserved behavior.')
        if np.linalg.matrix_rank(fitted.covariance) < fitted.n_parameters:
            warnings.append(f'{side}: source parameter covariance is rank deficient; inspect source data and model choices.')
        if reference.n_draws < settings['reference_draws'] or reference.n_valid_draws < reference.n_draws:
            warnings.append(f'{side}: {reference.n_valid_draws} usable source curves were available from {settings["reference_draws"]} requested parameter draws.')
        if varied:
            models[side] = future
    designs, fitted_reference_models = {}, {}
    for row in candidates:
        kwargs = {}
        for side in SIDES:
            request = row['sides'].get(side)
            if request is None:
                continue
            field, old = request['field'], analyzed[side]
            observation = replace(models[side], replicates_per_time=request['replicates'])
            criterion = None
            if request['kind'] == 'manual':
                times = request['times']
            else:
                historical = (None if old is None else existing_observations_from_time_course(old))
                fixed = np.array([]) if historical is None else historical.unique_times
                available = grids[side][~np.any(np.isclose(grids[side][:, None], fixed, rtol=0., atol=1e-12), axis=1)]
                count = request['number_of_times']
                if count > len(available):
                    raise PlanningInputError('Requested added times exceed the available unmeasured opportunities.', field + 'number_of_times')
                reference = references[side]
                constraints = DesignConstraints(available, fixed_times=fixed,
                    min_time=min(reference.time_grid[0], float(fixed.min()) if len(fixed) else reference.time_grid[0]),
                    max_time=max(reference.time_grid[-1], float(fixed.max()) if len(fixed) else reference.time_grid[-1]),
                    min_number_of_times=len(fixed)+count, max_number_of_times=len(fixed)+count,
                    replicates_per_time=request['replicates'])
                try:
                    if side not in fitted_reference_models:
                        fitted_reference_models[side] = fit_reference_model_ensemble(
                            reference, model_name=fits[side].model_name,
                            initial_parameters=fits[side].parameter_vector, fallback_grid_search=False)
                    fitted_reference = fitted_reference_models[side]
                    optimized = optimize_local_i_optimal_designs(fitted_reference, observation, constraints,
                        existing_observations=historical, refine=True, max_exchange_iterations=10)
                    selected = optimized.design_for_n(len(fixed)+count, refined=True)
                    criterion = selected.score.criterion
                    if not np.isfinite(criterion):
                        warnings.append(f'{row["name"]} ({side}): local information is insufficient under the generating model; consider more distinct times and inspect the source model. The generated times remain available for comparison.')
                    times = selected.times[~np.any(np.isclose(selected.times[:, None], fixed, rtol=0., atol=1e-12), axis=1)]
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    raise PlanningInputError(f'Curve-recovery schedule generation failed: {error}', field + 'number_of_times') from error
            if not len(times) and old is None:
                raise PlanningInputError('A prospective dataset requires proposed sampling times.', field + 'times')
            if len(times):
                sigma = observation.standard_deviation(references[side].evaluate_central(times), times=times)
                if not np.isfinite(sigma).all() or np.any(sigma <= 0):
                    error_field = ('calibration-' + side.replace('_', '-') + '-sd-profile-file'
                                   if observation.kind == 'time_profile_sd' else
                                   'cal_' + side + '_minimum_sd' if observation.kind == 'constant_cv'
                                   else 'cal_' + side + '_observation_value')
                    raise PlanningInputError('Future individual SD must be positive at every proposed time.', error_field)
                kwargs[side + '_times'] = times
                kwargs[side + '_observation_model'] = observation
            outcomes.append(dict(candidate=row['name'], candidate_index=row['index'],
                side=side, kind=request['kind'], status='available',
                fixed_times=np.array([]) if old is None else np.unique(old.time),
                new_units=len(times) * request['replicates'], information_criterion=criterion,
                reference_time_range=details[side].get('reference_time_range'),
                times=times.copy(), replicates=request['replicates'],
                message=('Generated for curve recovery, not optimized for IVIVC performance.' if request['kind']=='generated' else 'Explicitly specified added observations.')))
        designs[row['name']] = CalibrationDesign(**kwargs)
    if settings['normalization'] != 'none' or settings['one_minus']:
        warnings.append('Response normalization constants are fixed from each source dataset and applied once to that environment. Future SD uses this analyzed scale; simulated datasets are not normalized again.')
    return CalibrationComparisonResult(settings, None, designs, references, fits, analyzed,
                                       details, list(dict.fromkeys(warnings)), outcomes)


def run_calibration_generation(form: Mapping, *, existing_in_vitro=None, existing_in_vivo=None,
                               in_vitro_source=None, in_vivo_source=None,
                               in_vitro_profile=None, in_vivo_profile=None):
    """Generate reviewable candidates without IVIVC fitting or measurement simulation.

    All active varied-environment candidates define the same source domains used
    by a full comparison. Held-fixed data, target predictions, and Monte Carlo
    settings are irrelevant to this request and are not required or validated.
    """
    settings = _settings(form, generation_only=True)
    candidates = _candidate_requests(form, settings, generation_only=True)
    if not any(request['kind'] == 'generated' for row in candidates
               for request in row['sides'].values()):
        raise PlanningInputError('Choose generation for at least one candidate environment.', 'calibration-candidates')
    return _build_calibration_candidates(form, settings, candidates,
        existing_in_vitro=existing_in_vitro, existing_in_vivo=existing_in_vivo,
        in_vitro_source=in_vitro_source, in_vivo_source=in_vivo_source,
        in_vitro_profile=in_vitro_profile, in_vivo_profile=in_vivo_profile,
        generation_only=True)


def run_calibration_comparison(form: Mapping, *, existing_in_vitro=None, existing_in_vivo=None,
                               in_vitro_source=None, in_vivo_source=None,
                               in_vitro_profile=None, in_vivo_profile=None, prediction=None):
    settings = _settings(form)
    candidates = _candidate_requests(form, settings)
    if settings['assess_target']:
        _course(prediction, 'cal-target-file')
        if len(prediction.time) > 200:
            raise PlanningInputError('Use at most 200 target-prediction observations per comparison.', 'cal-target-file')
        if settings['approach_id'] == 'approach1' and prediction.observation_uncertainty is not None:
            raise PlanningInputError('Approach 1 does not propagate target response SD/SEM. Choose Approach 2 or explicitly omit target precision.', 'cal-target-file')
    point_count = (len(settings.get('evaluation_times', settings.get('response_levels', ())))
                   + (1 if settings['method'] == 'tau_ratio' else 0)
                   + (len(prediction.time) if settings['assess_target'] else 0))
    if (len(candidates) * settings['n_simulations'] * settings['interval_draws'] * point_count
            > 10_000_000):
        raise PlanningInputError('Reduce evaluation points, candidates, simulations, or interval draws to at most 10000000 point draws per request.', 'cal_interval_draws')
    if settings['assess_target'] and prediction is not None and (
            prediction is existing_in_vitro or prediction is existing_in_vivo):
        raise PlanningInputError('Target observations must be independent of calibration measurements.', 'cal-target-file')
    prepared = _build_calibration_candidates(form, settings, candidates,
        existing_in_vitro=existing_in_vitro, existing_in_vivo=existing_in_vivo,
        in_vitro_source=in_vitro_source, in_vivo_source=in_vivo_source,
        in_vitro_profile=in_vitro_profile, in_vivo_profile=in_vivo_profile)
    designs, references, fits = prepared.designs, prepared.references, prepared.source_fits
    analyzed, details = prepared.analyzed_existing, prepared.source_details
    warnings, outcomes = prepared.warnings, prepared.generation_outcomes
    assessment = dict(approach_id=settings['approach_id'], model_name=settings['model_name'],
                      method=settings['method'], selected_normalizations=(), selected_scalings=(),
                      selected_interpolation=[settings['interpolation']], n_samples=settings['interval_draws'],
                      random_state=_seed(settings['random_seed'], 40),
                      analysis_config=dict(cv_scheme='none', grid_search_num_points=25,
                          grid_search_num_cores=1, grid_search_random_state=_seed(settings['random_seed'], 41)))
    for key in ('evaluation_times', 'response_levels'):
        if key in settings:
            assessment[key] = settings[key]
    if settings['assess_target']:
        target, target_transform = _prepare(prediction, settings)
        assessment['prediction'] = target
        settings['target_normalization'] = target_transform
        if target.observation_uncertainty is not None:
            warnings.append('Target-input precision is propagated, but reported uncertain target responses are not assumed to be latent generating truth. Target errors and coverage may therefore be unavailable even when reference curves exist.')
        if any(record['name'] == 'initial_value' and not record.get('applied', True)
               for record in target_transform['records']):
            warnings.append('The target initial response was zero; initial-value normalization was not applied to the target.')
    try:
        comparison = compare_calibration_designs(designs, assessment_options=assessment,
            existing_in_vitro=analyzed['in_vitro'], existing_in_vivo=analyzed['in_vivo'],
            in_vitro_reference=references['in_vitro'], in_vivo_reference=references['in_vivo'],
            n_simulations=settings['n_simulations'], random_state=_seed(settings['random_seed'], 30))
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'Calibration comparison could not be completed: {error}', 'calibration-candidates') from error
    warnings.append('Generating references in the two environments are assumed independent. Existing calibration rows remain fixed; only proposed independent observations are simulated.')
    if settings['approach_id'] == 'approach1':
        warnings.append('Approach 1 repeats the ordinary shared interpolation and time-mapping fit. Experimental response uncertainty drives simulated measurements, not weighting or propagation within that fit; across-study variation is reported separately from mapping-fit intervals.')
    for key, requested in (('relationship', settings['relationship']), ('prediction', settings['assess_target'])):
        if requested and all(np.sum(summary.get(key, {}).get('truth_counts', ())) == 0
                             for summary in comparison.summaries.values()):
            warnings.append(f'No generating truth was available for {key} evaluation. Precision and calculation availability remain assessable; error and coverage are unavailable, rather than zero.')
    return CalibrationComparisonResult(settings, comparison, designs, references, fits, analyzed,
                                       details, list(dict.fromkeys(warnings)), outcomes)
