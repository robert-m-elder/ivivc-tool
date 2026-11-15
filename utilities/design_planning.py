"""Request-independent adapters for prospective study-planning workflows.

This module assembles the existing numerical APIs. It does not store uploads,
choose an IVIVC, or define a second fitting, optimization, or simulation method.
Future SD inputs use the analyzed response scale after selected preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import re

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse, iter_model_specs, prepare_time_course
from study_design_core import (
    DesignConstraints, ObservationModel, compare_designs_by_simulation,
    equally_spaced_design, fit_reference_model_ensemble, fit_reference_profile,
    optimize_local_i_optimal_designs, reference_from_profile_fit,
    existing_observations_from_time_course, reweight_reference_ensemble,
    reweight_time_scaled_reference,
)


class PlanningInputError(ValueError):
    """A user-correctable setting or numerical failure linked to its control."""

    def __init__(self, message: str, field_id: str | None = None):
        super().__init__(message)
        self.field_id = field_id


@dataclass
class PlanningResult:
    settings: dict
    analyzed_source: TimeCourse
    preprocessing: object
    fit: object
    reference: object
    reference_model: object
    observation_model: ObservationModel
    constraints: DesignConstraints
    optimization: object
    comparison: object
    schedules: dict
    warnings: list[str]
    source_details: dict = field(default_factory=dict)
    completed_observations: object = None
    adaptive_update: object = None
    prior_reference: object = None
    timing_details: dict = field(default_factory=dict)


def model_options() -> list[dict]:
    """Expose the shared response-versus-time model registry in display order."""
    return [{'id': spec.key, 'label': spec.display_name,
             'parameter_count': len(spec.parameter_names)}
            for spec in iter_model_specs()]


def _checked(form, key):
    return str(form.get(key, '')).strip().lower() in ('1', 'on', 'true', 'yes')


def _number(form, key, default, *, integer=False, minimum=None, maximum=None):
    raw = form.get(key, default)
    try:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError
        if integer:
            # int() must not silently truncate fractional numeric form values.
            text = str(raw).strip()
            if not re.fullmatch(r'[+-]?\d+', text):
                raise ValueError
            value = int(text)
        else:
            value = float(raw)
        if not np.isfinite(value):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as error:
        kind = 'whole number' if integer else 'finite number'
        raise PlanningInputError(f'Enter a {kind}.', key) from error
    if minimum is not None and value < minimum:
        raise PlanningInputError(f'Use a value of at least {minimum}.', key)
    if maximum is not None and value > maximum:
        raise PlanningInputError(f'Use a value no greater than {maximum}.', key)
    return value


def _times(text, field, *, allow_empty=True):
    text = str(text).strip()
    if not text and allow_empty:
        return np.array([], dtype=float)
    parts = text.split(',')
    try:
        if not text or any(not part.strip() for part in parts):
            raise ValueError
        values = np.array([float(part) for part in parts], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError
    except (ValueError, TypeError, OverflowError) as error:
        raise PlanningInputError('Enter finite times separated by commas.', field) from error
    if len(np.unique(values)) != len(values):
        raise PlanningInputError('Enter each sampling time once; use the replicate count for repeated observations.', field)
    return np.sort(values)


def _settings(form):
    if form.get('task', 'study') != 'study':
        raise PlanningInputError('Numerical planning currently supports the in vivo study task.', 'task')
    reference_source = form.get('reference_source', 'measured')
    if reference_source not in ('measured', 'ivivc', 'scaling'):
        raise PlanningInputError('Select a supported source of expected in vivo behavior.', 'reference_source')
    models = {item['id'] for item in model_options()}
    model = form.get('model_name', 'exponential')
    if model not in models:
        raise PlanningInputError('Select a registered response-versus-time model.', 'model_name')
    normalization = form.get('normalization', 'none')
    if normalization not in ('none', 'initial_value', 'min_max'):
        raise PlanningInputError('Select a supported response normalization.', 'normalization')
    kind = form.get('observation_kind', 'constant_sd')
    if kind not in ('constant_sd', 'constant_cv', 'time_profile_sd'):
        raise PlanningInputError('Select a supported future measurement model.', 'observation_kind')
    settings = {
        'task': 'study', 'reference_source': reference_source, 'model_name': model,
        'normalization': normalization, 'one_minus': _checked(form, 'one_minus'),
        'observation_kind': kind, 'has_completed': _checked(form, 'has_completed'),
        'first_time': (0. if _checked(form, 'has_completed') and form.get('first_time_mode') == 'latest_completed'
                       else _number(form, 'first_time', 0., minimum=0.)),
        'last_time': _number(form, 'last_time', 52., minimum=0.),
        'time_step': _number(form, 'time_step', 1., minimum=0.),
        'min_times': _number(form, 'min_times', 3, integer=True, minimum=1, maximum=30),
        'max_times': _number(form, 'max_times', 6, integer=True, minimum=1, maximum=30),
        'replicates_per_time': _number(form, 'replicates_per_time', 3, integer=True, minimum=1, maximum=100),
        'minimum_sd': (_number(form, 'minimum_sd', .001, minimum=0.)
                       if kind == 'constant_cv' else 0.),
        'reference_draws': _number(form, 'reference_draws', 100, integer=True, minimum=2, maximum=500),
        'n_simulations': _number(form, 'n_simulations', 100, integer=True, minimum=2, maximum=1000),
        'random_seed': _number(form, 'random_seed', 12345, integer=True, minimum=0, maximum=2**32-1),
        'include_even': _checked(form, 'include_even'),
    }
    if kind != 'time_profile_sd':
        settings['observation_value'] = _number(form, 'observation_value', .05, minimum=0.)
        if kind == 'constant_sd' and settings['observation_value'] <= 0:
            raise PlanningInputError('Future individual SD must be positive.', 'observation_value')
    if settings['has_completed']:
        settings['measured_adaptive_mode'] = form.get('measured_adaptive_mode', 'refit_completed')
        if reference_source == 'measured' and settings['measured_adaptive_mode'] not in ('refit_completed', 'reweight_prior'):
            raise PlanningInputError('Select how the measured reference should be updated.', 'measured_adaptive_mode')
        settings['completed_error_mode'] = form.get('completed_error_mode', 'supplied')
        if settings['completed_error_mode'] not in ('supplied', 'future_model'):
            raise PlanningInputError('Select how missing completed-observation precision is handled.', 'completed_error_mode')
        settings['future_policy'] = form.get('future_policy', 'future_only')
        if settings['future_policy'] not in ('future_only', 'gap_fill'):
            raise PlanningInputError('Select future-only planning or earlier unmeasured times.', 'future_policy')
        settings['first_time_mode'] = form.get('first_time_mode', 'specified')
        if settings['first_time_mode'] not in ('specified', 'latest_completed'):
            raise PlanningInputError('Select a specified first time or the latest completed time.', 'first_time_mode')
    else:
        settings['future_policy'] = 'gap_fill'
        settings['first_time_mode'] = 'specified'
    if settings['last_time'] <= settings['first_time']:
        raise PlanningInputError('Last opportunity must be later than the first.', 'last_time')
    if settings['time_step'] <= 0:
        raise PlanningInputError('Time step must be positive.', 'time_step')
    if settings['max_times'] < settings['min_times']:
        raise PlanningInputError('Maximum times must be at least the minimum.', 'max_times')
    return settings


def _planning_constraints(settings, form, existing_observations=None):
    first, last, step = (settings[key] for key in ('first_time', 'last_time', 'time_step'))
    completed_times = (np.array([], dtype=float) if existing_observations is None
                       else existing_observations.unique_times)
    if settings['first_time_mode'] == 'latest_completed':
        first = existing_observations.latest_time
        settings['first_time'] = first
    default_baseline = float(completed_times[0]) if len(completed_times) else first
    baseline = _number(form, 'baseline_time', default_baseline, minimum=0.)
    if first > last or baseline >= last:
        raise PlanningInputError('The last time must be after the baseline and not before the first opportunity.', 'last_time')
    span_steps = (last - first) / step
    if not np.isfinite(span_steps) or span_steps > 999 + 1e-10:
        raise PlanningInputError('Use at most 1000 candidate times; increase the time step or shorten the range.', 'time_step')
    candidates = first + np.arange(int(np.floor(span_steps + 1e-10)) + 1) * step
    candidates = candidates[candidates <= last + 1e-10 * max(1., abs(last))]
    candidates[np.isclose(candidates, last, rtol=0, atol=1e-10*max(1., abs(last)))] = last
    requested_fixed = _times(form.get('fixed_times', ''), 'fixed_times')
    excluded = _times(form.get('excluded_times', ''), 'excluded_times')
    if len(excluded):
        candidates = candidates[~np.any(np.isclose(
            candidates[:, None], excluded, rtol=0, atol=1e-12), axis=1)]
        proposed_fixed = requested_fixed[~np.any(np.isclose(
            requested_fixed[:, None], completed_times, rtol=0, atol=1e-12), axis=1)]
        if np.any(np.isclose(proposed_fixed[:, None], excluded, rtol=0, atol=1e-12)):
            raise PlanningInputError('An excluded time is also fixed for a proposed measurement. Resolve the conflicting choices.', 'excluded_times')
    if np.any(requested_fixed < 0) or np.any(requested_fixed > last):
        raise PlanningInputError('Fixed times must be nonnegative and no later than the last time.', 'fixed_times')
    if len(completed_times):
        if completed_times[0] < 0 or completed_times[-1] > last:
            raise PlanningInputError('Completed observations must use nonnegative times within the study end time.', 'completed-target-file')
        if settings['future_policy'] == 'future_only':
            candidates = candidates[candidates > completed_times[-1] + 1e-12]
            unmeasured_fixed = requested_fixed[~np.any(np.isclose(
                requested_fixed[:, None], completed_times, rtol=0, atol=1e-12), axis=1)]
            if np.any(unmeasured_fixed <= completed_times[-1] + 1e-12):
                raise PlanningInputError('A fixed proposed measurement is at or before the latest completed time. Remove it or allow earlier unmeasured times.', 'fixed_times')
        candidates = candidates[~np.any(np.isclose(
            candidates[:, None], completed_times, rtol=0, atol=1e-12), axis=1)]
    fixed = np.union1d(requested_fixed, completed_times)
    candidate_count = len(candidates)
    if len(fixed) > settings['min_times']:
        raise PlanningInputError('Minimum total times must include every completed and fixed time.', 'min_times')
    if settings['max_times'] > len(np.union1d(candidates, fixed)):
        raise PlanningInputError('Maximum total times exceeds the available distinct opportunities, including completed times.', 'max_times')
    evaluation_start = min(first, baseline, float(fixed[0]) if len(fixed) else first)
    if not candidate_count:
        if not len(fixed):
            raise PlanningInputError('No sampling opportunities remain.', 'first_time')
        # The shared constraints require a candidate array. Already-fixed times
        # supply its domain without creating any additional sampling opportunity.
        candidates = fixed.copy()
    constraints = DesignConstraints(
        candidates, fixed_times=fixed, min_time=evaluation_start, max_time=last,
        min_number_of_times=settings['min_times'], max_number_of_times=settings['max_times'],
        replicates_per_time=settings['replicates_per_time'],
    )
    settings['evaluation_start'] = evaluation_start
    settings['baseline_time'] = baseline
    settings['completed_times'] = completed_times.tolist()
    settings['latest_completed_time'] = float(completed_times[-1]) if len(completed_times) else None
    settings['requested_fixed_times'] = requested_fixed.tolist()
    settings['excluded_times'] = excluded.tolist()
    manual = {}
    for line_number, line in enumerate(str(form.get('manual_schedules', '')).splitlines(), 1):
        if not line.strip():
            continue
        label, separator, times = line.partition(':')
        label = label.strip()
        if not separator or not label:
            raise PlanningInputError(f'Manual schedule line {line_number}: use Name: time, time, ...', 'manual_schedules')
        if len(label) > 80 or label in manual:
            raise PlanningInputError('Manual schedule names must be unique and no longer than 80 characters.', 'manual_schedules')
        values = _times(times, 'manual_schedules', allow_empty=False)
        try:
            manual[label] = constraints.validate_design(values)
        except ValueError as error:
            raise PlanningInputError(f'{label}: {error}', 'manual_schedules') from error
    if len(manual) > 10:
        raise PlanningInputError('Use at most 10 manual schedules per comparison.', 'manual_schedules')
    number_of_designs = (settings['max_times'] - settings['min_times'] + 1) * (2 if settings['include_even'] else 1) + len(manual)
    if number_of_designs * settings['n_simulations'] > 20000:
        raise PlanningInputError('Limit each request to 20000 simulated refits; reduce simulations, schedule sizes, or manual schedules.', 'n_simulations')
    if (len(candidates) + len(fixed)) * settings['replicates_per_time'] * settings['n_simulations'] > 2_000_000:
        raise PlanningInputError('The candidate grid, replicates, and simulations are too large for one synchronous request. Reduce one of these settings.', 'n_simulations')
    settings['fixed_times'] = fixed.tolist()
    settings['candidate_time_count'] = candidate_count
    settings['manual_schedules'] = {key: value.tolist() for key, value in manual.items()}
    return constraints, manual


def _future_observation_model(settings, profile):
    if settings['observation_kind'] == 'time_profile_sd':
        if not isinstance(profile, ObservationModel) or profile.kind != 'time_profile_sd':
            raise PlanningInputError('Import a future individual-SD profile for the selected variability model.', 'future-sd-profile-file')
        observation_model = replace(profile, replicates_per_time=settings['replicates_per_time'], minimum_sd=settings['minimum_sd'])
    else:
        observation_model = ObservationModel(
            settings['observation_kind'], settings['observation_value'],
            replicates_per_time=settings['replicates_per_time'], minimum_sd=settings['minimum_sd'],
        )
    return observation_model


def _prepare_completed(completed, settings, observation_model, *, normalize=False):
    if not settings['has_completed']:
        return None, None, []
    if not isinstance(completed, TimeCourse):
        raise PlanningInputError('Import the cumulative completed target-study observations.', 'completed-target-file')
    if (not len(completed.time) or len(completed.time) > 5000
            or not np.isfinite(completed.time).all() or not np.isfinite(completed.value).all()):
        raise PlanningInputError('Supply 1 to 5000 finite completed observations.', 'completed-target-file')
    if normalize:
        normalizations = ([] if settings['normalization'] == 'none' else [settings['normalization']])
        if settings['one_minus']:
            normalizations.append('one_minus')
        try:
            course, metadata = prepare_time_course(
                completed.time, completed.value, label=completed.label,
                observation_uncertainty=completed.observation_uncertainty,
                selected_normalizations=normalizations,
            )
        except (ValueError, KeyError, FloatingPointError) as error:
            raise PlanningInputError(f'Completed-data preprocessing failed: {error}', 'normalization') from error
    else:
        course = TimeCourse(completed.time, completed.value, label=completed.label,
                            observation_uncertainty=completed.observation_uncertainty)
        metadata = None
    warnings = []
    if course.observation_uncertainty is None:
        if settings['completed_error_mode'] != 'future_model':
            raise PlanningInputError('Supply completed-observation SD/SEM, or explicitly assume the future measurement model for missing historical precision.', 'completed_error_mode')
        sigma = observation_model.standard_deviation(course.value, times=course.time)
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise PlanningInputError('The assumed historical SD must be finite and positive at every completed observation.', 'completed_error_mode')
        course = TimeCourse(course.time, course.value, label=course.label,
                            observation_uncertainty=ResolvedObservationUncertainty(
                                sigma, response_kind='individual',
                                source='future_model_assumption'))
        warnings.append('Missing historical precision uses the explicitly selected future measurement model, evaluated once at each completed response and time. Those SDs are fixed for source fitting, adaptive weighting, information, and refitting; no historical replicate reduction is assumed.')
    if not np.isfinite(course.value).all():
        raise PlanningInputError('Completed-data preprocessing produced nonfinite responses.', 'normalization')
    return course, metadata, warnings


def _update_reference(reference, completed_observations, observation_model, *, source_kind):
    if completed_observations is None:
        return reference, None, None
    try:
        update = (reweight_time_scaled_reference(reference, completed_observations, observation_model)
                  if source_kind == 'scaling' else
                  reweight_reference_ensemble(reference, completed_observations, observation_model))
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The completed observations could not update the original reference: {error}', 'completed-target-file') from error
    # A continuous agreement diagnostic remains informative even for a
    # deterministic prior, whose single weight cannot change.
    predicted = reference.evaluate(completed_observations.time)
    active = reference.valid_draws & (reference.draw_weights > 0)
    sigma = completed_observations.observation_uncertainty.sigma
    standardized = (completed_observations.response - predicted[active]) / sigma
    with np.errstate(over='ignore', invalid='ignore'):
        rms = np.sqrt(np.mean(standardized**2, axis=1))
    finite = rms[np.isfinite(rms)]
    update.metadata['best_standardized_residual_rms'] = float(np.min(finite)) if len(finite) else None
    return update.reference, update, reference


def _adaptive_details(settings, observations, update, *, refit_completed=False):
    if observations is None:
        return {}
    return {
        'mode': 'refit_completed' if refit_completed else 'reweight_prior',
        'completed_rows': observations.n_observations,
        'completed_times': observations.unique_times.tolist(),
        'latest_completed_time': observations.latest_time,
        'completed_response_kind': observations.response_kind,
        'completed_error_mode': settings['completed_error_mode'],
        'completed_scale': 'normalized once with the fitted completed dataset' if refit_completed else 'supplied on the analyzed reference response scale',
        'effective_sample_size': None if update is None else update.effective_sample_size,
        'best_standardized_residual_rms': None if update is None else update.metadata.get('best_standardized_residual_rms'),
        'future_policy': settings['future_policy'],
    }


def _assess_reference(reference, settings, constraints, manual, observation_model, *, initial_parameters=None, existing_observations=None, form=None):
    """Use one optimization and simulation path for every reference source."""
    try:
        reference_model = fit_reference_model_ensemble(
            reference, model_name=settings['model_name'], initial_parameters=initial_parameters,
            fallback_grid_search=False,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The study analysis model could not represent the reference: {error}', 'model_name') from error
    future_times = constraints.all_available_times
    if existing_observations is not None:
        future_times = future_times[~np.any(np.isclose(
            future_times[:, None], existing_observations.unique_times,
            rtol=0, atol=1e-12), axis=1)]
    sigma = observation_model.standard_deviation(reference.evaluate_central(future_times), times=future_times)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        field = ('future-sd-profile-file' if observation_model.kind == 'time_profile_sd'
                 else 'minimum_sd' if observation_model.kind == 'constant_cv' else 'observation_value')
        raise PlanningInputError('Future SD must be positive at every selectable time. Check the selected variability model.', field)
    from .design_timing import assess_timing, inspect_comparison
    timed = assess_timing(reference, reference_model, observation_model, constraints,
                          settings, form or {}, existing_observations=existing_observations)
    if timed is None:
        try:
            optimization = optimize_local_i_optimal_designs(
                reference_model, observation_model, constraints,
                refine=True, max_exchange_iterations=10,
                existing_observations=existing_observations,
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            raise PlanningInputError(f'The requested schedules could not be generated: {error}', 'max_times') from error
        schedules = {f'Suggested · {count} times': optimization.design_for_n(count, refined=True).times.copy()
                     for count in range(settings['min_times'], settings['max_times'] + 1)}
        timing_details = {}
    else:
        optimization, schedules, timing_details = timed
    if settings['include_even']:
        for count in range(settings['min_times'], settings['max_times'] + 1):
            try:
                schedules[f'Evenly spaced · {count} times'] = equally_spaced_design(constraints, count)
            except ValueError as error:
                raise PlanningInputError(f'Evenly spaced comparison could not be generated: {error}', 'include_even') from error
    for label, values in manual.items():
        schedules['Manual · ' + label] = values.copy()
    if timing_details:
        timing_details['evaluations'] = {
            label: inspect_comparison(times, timing_details, constraints)
            for label, times in schedules.items()
        }
    if not schedules:
        return reference_model, optimization, None, schedules, timing_details
    try:
        # Source-parameter sampling uses the user seed. Give measurement
        # simulation a separate reproducible stream while sharing it across
        # all schedules in this comparison.
        simulation_seed = int(np.random.SeedSequence(
            [settings['random_seed'], 1]).generate_state(1)[0])
        comparison = compare_designs_by_simulation(
            reference, schedules, observation_model, model_name=settings['model_name'],
            constraints=constraints, reference_model=reference_model,
            n_simulations=settings['n_simulations'], random_state=simulation_seed,
            fallback_grid_search=False, existing_observations=existing_observations,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The simulation comparison could not be completed: {error}', 'n_simulations') from error
    return reference_model, optimization, comparison, schedules, timing_details


def _comparison_warnings(reference, reference_model, optimization, comparison, settings):
    warnings = []
    if reference.n_valid_draws < reference.n_draws:
        warnings.append(f'{reference.n_draws - reference.n_valid_draws} reference draws were unavailable over the planning range and excluded.')
    if reference_model.n_valid_draws < reference_model.n_draws:
        warnings.append(f'{reference_model.n_draws - reference_model.n_valid_draws} reference trajectories could not be fitted by the analysis model.')
    if comparison is not None and np.any([summary.failure_rate > 0 for summary in comparison.summaries.values()]):
        warnings.append('Some simulated studies could not be fitted. Recovery-error summaries use successful fits; compare their failure rates as well.')
    if any(not item.score.is_full_rank for item in optimization.refined_steps):
        warnings.append('At least one suggested schedule has rank-deficient information at the central reference fit. Its parameters may not be distinguishable from those sampling times.')
    if settings['normalization'] != 'none':
        warnings.append('Response normalization constants are treated as fixed. Future SD and minimum SD must be supplied on this analyzed response scale.')
    return warnings


def run_measured_planning(source: TimeCourse | None, form: Mapping, *, profile=None, completed=None) -> PlanningResult:
    """Fit completed data once, or update an independent measured reference."""
    settings = _settings(form)
    if settings['reference_source'] != 'measured':
        raise PlanningInputError('Use the corresponding reference-source workflow for these inputs.', 'reference_source')
    observation_model = _future_observation_model(settings, profile)
    refit_completed = settings['has_completed'] and settings['measured_adaptive_mode'] == 'refit_completed'
    completed_course, completed_preprocessing, historical_warnings = _prepare_completed(
        completed, settings, observation_model, normalize=refit_completed,
    )
    existing = (None if completed_course is None else
                existing_observations_from_time_course(completed_course))
    constraints, manual = _planning_constraints(settings, form, existing)
    source = completed_course if refit_completed else source
    role = 'completed-target-file' if refit_completed else 'reference-file'
    if not isinstance(source, TimeCourse):
        raise PlanningInputError('Import the measured source observations.', role)
    if len(source.time) > 5000:
        raise PlanningInputError('Use at most 5000 reference rows in this workflow.', role)
    if not np.isfinite(source.time).all() or not np.isfinite(source.value).all():
        raise PlanningInputError('Reference times and responses must be finite.', role)
    if len(np.unique(source.time)) < 2:
        raise PlanningInputError('Reference data need at least two distinct measurement times.', role)
    spec = next(item for item in iter_model_specs() if item.key == settings['model_name'])
    if len(source.time) <= len(spec.parameter_names):
        raise PlanningInputError(f'The selected model needs more than {len(spec.parameter_names)} reference observations to estimate fitted uncertainty.', 'model_name')
    if refit_completed:
        analyzed, preprocessing = source, completed_preprocessing
    else:
        normalizations = ([] if settings['normalization'] == 'none' else [settings['normalization']])
        if settings['one_minus']:
            normalizations.append('one_minus')
        try:
            analyzed, preprocessing = prepare_time_course(
                source.time, source.value, label=source.label,
                observation_uncertainty=source.observation_uncertainty,
                selected_normalizations=normalizations,
            )
        except (ValueError, KeyError) as error:
            raise PlanningInputError(f'Response preprocessing could not be applied: {error}', 'normalization') from error
    if not np.isfinite(analyzed.value).all():
        raise PlanningInputError('Normalization produced nonfinite values; check the initial value or response range.', 'normalization')
    try:
        fit = fit_reference_profile(analyzed, model_name=settings['model_name'], analysis_config={
            'cv_scheme': 'none', 'grid_search_num_points': 25, 'grid_search_num_cores': 1,
            'grid_search_random_state': settings['random_seed'],
        })
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'Reference fitting failed: {error}', 'model_name') from error
    if not np.isfinite(fit.covariance).all():
        raise PlanningInputError('Fitted parameter covariance is unavailable. Check the reference observations or try a simpler model.', 'model_name')
    evaluation_grid = np.linspace(settings['evaluation_start'], settings['last_time'], 101)
    try:
        reference = reference_from_profile_fit(
            fit, evaluation_grid, n_samples=settings['reference_draws'], random_state=settings['random_seed'],
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The reference ensemble could not be constructed over the planning range: {error}', 'model_name') from error
    update, prior = None, None
    if existing is not None and not refit_completed:
        reference, update, prior = _update_reference(reference, existing, observation_model, source_kind='measured')
    reference_model, optimization, comparison, schedules, timing_details = _assess_reference(
        reference, settings, constraints, manual, observation_model,
        initial_parameters=fit.parameter_vector, existing_observations=existing, form=form,
    )
    warnings = historical_warnings + _comparison_warnings(reference, reference_model, optimization, comparison, settings)
    if settings['evaluation_start'] < np.min(analyzed.time) or settings['last_time'] > np.max(analyzed.time):
        warnings.append('The planning range extends beyond the measured reference times. Results there depend on extrapolation of the selected fitted model.')
    if np.linalg.matrix_rank(fit.covariance) < fit.n_parameters:
        warnings.append('The fitted parameter covariance is rank deficient. Reference draws may underrepresent uncertainty; review the data and model in the IVIVC app.')
    if reference.n_draws < settings['reference_draws']:
        warnings.append(f'Only {reference.n_draws} of {settings["reference_draws"]} requested reference parameter draws were available.')
    if any(record.name == 'initial_value' and not record.in_vitro_parameters.get('applied', True)
           for record in preprocessing.normalizations):
        warnings.append('Initial response was zero; initial-value normalization was not applied.')
    details = {}
    if existing is not None:
        details['adaptive'] = _adaptive_details(settings, existing, update, refit_completed=refit_completed)
        if refit_completed:
            details.update(kind='measured', label='Reference fitted from completed target-study observations')
            warnings.append('The completed observations define the fitted reference and are not used again to reweight it. They remain fixed in every simulated study.')
        else:
            warnings.append('Cumulative completed observations update an independent original measured reference once. Completed responses and SD/SEM must already use its analyzed scale.')
    return PlanningResult(settings, analyzed, preprocessing, fit, reference, reference_model,
                          observation_model, constraints, optimization, comparison, schedules, warnings,
                          details, existing, update, prior, timing_details)


def run_reference_planning(form: Mapping, *, in_vitro=None, in_vivo=None, prediction=None,
                           scaling_source=None, distribution=None, profile=None, completed=None) -> PlanningResult:
    """Update an original IVIVC or empirical-scale prior with cumulative data."""
    from .design_reference_sources import build_ivivc_reference, build_scaling_reference

    settings = _settings(form)
    if settings['reference_source'] == 'measured':
        raise PlanningInputError('Supply measured data through the measured-reference workflow.', 'reference_source')
    observation_model = _future_observation_model(settings, profile)
    completed_course, _, historical_warnings = _prepare_completed(completed, settings, observation_model)
    existing = (None if completed_course is None else
                existing_observations_from_time_course(completed_course))
    constraints, manual = _planning_constraints(settings, form, existing)
    grid = np.linspace(settings['evaluation_start'], settings['last_time'], 101)
    if settings['reference_source'] == 'ivivc':
        inputs = build_ivivc_reference(in_vitro, in_vivo, prediction, form, settings, grid)
    else:
        inputs = build_scaling_reference(scaling_source, distribution, form, settings, grid)
    reference, update, prior = _update_reference(
        inputs.reference, existing, observation_model, source_kind=settings['reference_source'],
    )
    reference_model, optimization, comparison, schedules, timing_details = _assess_reference(
        reference, settings, constraints, manual, observation_model,
        existing_observations=existing, form=form,
    )
    warnings = historical_warnings + list(inputs.warnings) + _comparison_warnings(
        reference, reference_model, optimization, comparison, settings,
    )
    details = dict(inputs.details)
    if existing is not None:
        details['adaptive'] = _adaptive_details(settings, existing, update)
        warnings.append('Cumulative completed observations reweight the original reference once; its trajectory shapes are unchanged. Completed responses and SD/SEM must already use the analyzed reference scale and remain fixed in each simulated study.')
        if reference.n_draws == 1:
            warnings.append('A deterministic reference cannot learn a new trajectory shape through reweighting. Inspect its agreement with the completed data.')
    return PlanningResult(settings, inputs.analyzed_source, inputs.preprocessing, inputs.fit,
                          reference, reference_model, observation_model, constraints,
                          optimization, comparison, schedules, warnings, details,
                          existing, update, prior, timing_details)
