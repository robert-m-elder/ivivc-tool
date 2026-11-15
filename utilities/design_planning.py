"""Request-independent adapter for the first measured-reference planning workflow.

This module assembles the existing numerical APIs. It does not store uploads,
choose an IVIVC, or define a second fitting, optimization, or simulation method.
Future SD inputs use the analyzed response scale after selected preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import re

import numpy as np

from ivivc_core import TimeCourse, iter_model_specs, prepare_time_course
from study_design_core import (
    DesignConstraints, ObservationModel, compare_designs_by_simulation,
    equally_spaced_design, fit_reference_model_ensemble, fit_reference_profile,
    optimize_local_i_optimal_designs, reference_from_profile_fit,
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
    if form.get('reference_source', 'measured') != 'measured':
        raise PlanningInputError('Numerical planning currently requires measured in vivo reference data.', 'reference_source')
    if _checked(form, 'has_completed'):
        raise PlanningInputError('Completed target-study observations are not connected to planning yet. Use prior reference data for a prospective study.', 'has_completed')
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
        'task': 'study', 'reference_source': 'measured', 'model_name': model,
        'normalization': normalization, 'one_minus': _checked(form, 'one_minus'),
        'observation_kind': kind,
        'first_time': _number(form, 'first_time', 0., minimum=0.),
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
    if settings['last_time'] <= settings['first_time']:
        raise PlanningInputError('Last opportunity must be later than the first.', 'last_time')
    if settings['time_step'] <= 0:
        raise PlanningInputError('Time step must be positive.', 'time_step')
    if settings['max_times'] < settings['min_times']:
        raise PlanningInputError('Maximum times must be at least the minimum.', 'max_times')
    return settings


def run_measured_planning(source: TimeCourse, form: Mapping, *, profile=None) -> PlanningResult:
    """Reconstruct one prospective measured-reference comparison from current inputs.

    Reference uncertainty is sampled from the source fit. Optimized schedules
    use local I-optimality at its central fit; complete reference draws and new
    independent measurement errors enter the shared Monte Carlo comparison.
    Counts refer to all distinct planned times, including fixed times.
    """
    settings = _settings(form)
    first, last, step = (settings[key] for key in ('first_time', 'last_time', 'time_step'))
    span_steps = (last - first) / step
    if not np.isfinite(span_steps) or span_steps > 999 + 1e-10:
        raise PlanningInputError('Use at most 1000 candidate times; increase the time step or shorten the range.', 'time_step')
    # Last is a bound, not an extra off-grid sampling opportunity.
    candidates = first + np.arange(int(np.floor(span_steps + 1e-10)) + 1) * step
    candidates = candidates[candidates <= last + 1e-10 * max(1., abs(last))]
    candidates[np.isclose(candidates, last, rtol=0, atol=1e-10*max(1., abs(last)))] = last
    fixed = _times(form.get('fixed_times', ''), 'fixed_times')
    if len(fixed) > settings['min_times']:
        raise PlanningInputError('Minimum times must include every fixed time.', 'min_times')
    if np.any(fixed < first) or np.any(fixed > last):
        raise PlanningInputError('Fixed times must lie between the first and last opportunities.', 'fixed_times')
    if settings['max_times'] > len(np.union1d(candidates, fixed)):
        raise PlanningInputError('Maximum times exceeds the available distinct opportunities.', 'max_times')
    constraints = DesignConstraints(
        candidates, fixed_times=fixed, min_time=first, max_time=last,
        min_number_of_times=settings['min_times'], max_number_of_times=settings['max_times'],
        replicates_per_time=settings['replicates_per_time'],
    )
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
    if not isinstance(source, TimeCourse):
        raise TypeError('source must be an imported TimeCourse.')
    if len(source.time) > 5000:
        raise PlanningInputError('Use at most 5000 reference rows in this workflow.', 'reference-file')
    if not np.isfinite(source.time).all() or not np.isfinite(source.value).all():
        raise PlanningInputError('Reference times and responses must be finite.', 'reference-file')
    if len(np.unique(source.time)) < 2:
        raise PlanningInputError('Reference data need at least two distinct measurement times.', 'reference-file')
    spec = next(item for item in iter_model_specs() if item.key == settings['model_name'])
    if len(source.time) <= len(spec.parameter_names):
        raise PlanningInputError(f'The selected model needs more than {len(spec.parameter_names)} reference observations to estimate fitted uncertainty.', 'model_name')
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
    if settings['observation_kind'] == 'time_profile_sd':
        if not isinstance(profile, ObservationModel) or profile.kind != 'time_profile_sd':
            raise PlanningInputError('Import a future individual-SD profile for the selected variability model.', 'future-sd-profile-file')
        observation_model = replace(profile, replicates_per_time=settings['replicates_per_time'], minimum_sd=settings['minimum_sd'])
    else:
        observation_model = ObservationModel(
            settings['observation_kind'], settings['observation_value'],
            replicates_per_time=settings['replicates_per_time'], minimum_sd=settings['minimum_sd'],
        )
    settings['fixed_times'] = fixed.tolist()
    settings['candidate_time_count'] = len(candidates)
    settings['manual_schedules'] = {key: value.tolist() for key, value in manual.items()}
    try:
        fit = fit_reference_profile(analyzed, model_name=settings['model_name'], analysis_config={
            'cv_scheme': 'none', 'grid_search_num_points': 25, 'grid_search_num_cores': 1,
            'grid_search_random_state': settings['random_seed'],
        })
    except (ValueError, RuntimeError, TypeError, FloatingPointError) as error:
        raise PlanningInputError(f'Reference fitting failed: {error}', 'model_name') from error
    if not np.isfinite(fit.covariance).all():
        raise PlanningInputError('Fitted parameter covariance is unavailable. Check the reference observations or try a simpler model.', 'model_name')
    evaluation_grid = np.linspace(first, last, 101)
    try:
        reference = reference_from_profile_fit(
            fit, evaluation_grid, n_samples=settings['reference_draws'], random_state=settings['random_seed'],
        )
        reference_model = fit_reference_model_ensemble(
            reference, model_name=settings['model_name'], initial_parameters=fit.parameter_vector,
            fallback_grid_search=False,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The reference ensemble could not be constructed over the planning range: {error}', 'model_name') from error
    sigma = observation_model.standard_deviation(reference.evaluate_central(constraints.all_available_times), times=constraints.all_available_times)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        field = ('future-sd-profile-file' if observation_model.kind == 'time_profile_sd'
                 else 'minimum_sd' if observation_model.kind == 'constant_cv' else 'observation_value')
        raise PlanningInputError('Future SD must be positive at every selectable time. Check the selected variability model.', field)
    try:
        optimization = optimize_local_i_optimal_designs(
            reference_model, observation_model, constraints,
            refine=True, max_exchange_iterations=10,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The requested schedules could not be generated: {error}', 'max_times') from error
    schedules = {}
    for count in range(settings['min_times'], settings['max_times'] + 1):
        schedules[f'Suggested · {count} times'] = optimization.design_for_n(count, refined=True).times.copy()
        if settings['include_even']:
            try:
                schedules[f'Evenly spaced · {count} times'] = equally_spaced_design(constraints, count)
            except ValueError as error:
                raise PlanningInputError(f'Evenly spaced comparison could not be generated: {error}', 'include_even') from error
    for label, values in manual.items():
        schedules['Manual · ' + label] = values.copy()
    try:
        comparison = compare_designs_by_simulation(
            reference, schedules, observation_model, model_name=settings['model_name'],
            constraints=constraints, reference_model=reference_model,
            n_simulations=settings['n_simulations'], random_state=settings['random_seed'],
            fallback_grid_search=False,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'The simulation comparison could not be completed: {error}', 'n_simulations') from error
    warnings = []
    if first < np.min(analyzed.time) or last > np.max(analyzed.time):
        warnings.append('The planning range extends beyond the measured reference times. Results there depend on extrapolation of the selected fitted model.')
    covariance_rank = np.linalg.matrix_rank(fit.covariance)
    if covariance_rank < fit.n_parameters:
        warnings.append('The fitted parameter covariance is rank deficient. Reference draws may underrepresent uncertainty; review the data and model in the IVIVC app.')
    if reference.n_valid_draws < reference.n_draws:
        warnings.append(f'{reference.n_draws - reference.n_valid_draws} reference draws were unavailable over the planning range and excluded.')
    if reference.n_draws < settings['reference_draws']:
        warnings.append(f'Only {reference.n_draws} of {settings["reference_draws"]} requested reference parameter draws were available.')
    if reference_model.n_valid_draws < reference_model.n_draws:
        warnings.append(f'{reference_model.n_draws - reference_model.n_valid_draws} reference trajectories could not be fitted by the analysis model.')
    if np.any([summary.failure_rate > 0 for summary in comparison.summaries.values()]):
        warnings.append('Some simulated studies could not be fitted. Recovery-error summaries use successful fits; compare their failure rates as well.')
    if any(not item.score.is_full_rank for item in optimization.refined_steps):
        warnings.append('At least one suggested schedule has rank-deficient information at the central reference fit. Its parameters may not be distinguishable from those sampling times.')
    if settings['normalization'] != 'none':
        warnings.append('Response normalization constants are treated as fixed. Future SD and minimum SD must be supplied on this analyzed response scale.')
    if any(record.name == 'initial_value' and not record.in_vitro_parameters.get('applied', True)
           for record in preprocessing.normalizations):
        warnings.append('Initial response was zero; initial-value normalization was not applied.')
    return PlanningResult(settings, analyzed, preprocessing, fit, reference, reference_model,
                          observation_model, constraints, optimization, comparison, schedules, warnings)
