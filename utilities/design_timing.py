"""Translate the planning form's timing requests to shared scheduling APIs.

This adapter keeps bound resolution, schedule feasibility, and information
optimization separate. It does not replace the shared timing calculations.
"""
from __future__ import annotations

import re

import numpy as np

from study_design_core import (
    CrossingTimeTarget, DesignOptimizationResult, DesignStep, LatestBefore,
    PlanningContext, SamplingWindow, evaluate_timing_requirements,
    optimize_timing_constrained_designs, resolve_timing_requirements,
)

from .design_planning import PlanningInputError


MAX_TIMING_REQUIREMENTS = 8
MAX_TIMING_SEARCH_STATES = 50000


def _number(form, field, default=None, *, integer=False, positive_fraction=False):
    raw = form.get(field, default)
    try:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError
        if integer:
            if not re.fullmatch(r'[+]?\d+', str(raw).strip()):
                raise ValueError
            value = int(raw)
        else:
            value = float(raw)
        if not np.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise PlanningInputError(
            'Enter a whole number.' if integer else 'Enter a finite number.', field,
        ) from error
    if integer and not 1 <= value <= 30:
        raise PlanningInputError('Use a whole-number count from 1 to 30.', field)
    if positive_fraction and not 0 < value < 1:
        raise PlanningInputError('Use a fraction strictly between 0 and 1.', field)
    return value


def _bound(form, prefix, side, settings, *, optional=False):
    mode_field = prefix + side + '_mode'
    mode = form.get(mode_field, 'numeric')
    if mode not in ('numeric', 'milestone'):
        raise PlanningInputError('Choose a numerical time or a degradation milestone.', mode_field)
    if mode == 'numeric':
        field = prefix + side + '_time'
        default = (settings.get('baseline_time', settings.get('first_time', 0.))
                   if side == 'lower' and not optional else None)
        if optional and str(form.get(field, '')).strip() == '':
            return None
        return _number(form, field, default)
    return CrossingTimeTarget(
        degraded_fraction=_number(form, prefix + side + '_fraction', .5, positive_fraction=True),
        quantile=_number(form, prefix + side + '_quantile', .25, positive_fraction=True),
        baseline_time=float(settings.get('baseline_time', settings.get('first_time', 0.))),
    )


def _description(bound):
    if bound is None:
        return 'Planning start'
    if isinstance(bound, CrossingTimeTarget):
        return (f'{100 * bound.quantile:g}th percentile of the time to '
                f'{100 * bound.degraded_fraction:g}% degradation '
                f'(response baseline at time {bound.baseline_time:g})')
    return f'Time {bound:g}'


def _parse(form, settings):
    for key in form:
        match = re.fullmatch(r'timing_(\d+)_kind', key)
        if match is not None and int(match[1]) >= MAX_TIMING_REQUIREMENTS:
            raise PlanningInputError('Use at most eight timing requirements.', 'timing-requirements')
    parsed, labels = [], set()
    for index in range(MAX_TIMING_REQUIREMENTS):
        prefix = f'timing_{index}_'
        if prefix + 'kind' not in form:
            continue
        kind = form[prefix + 'kind']
        if kind not in ('window', 'latest_before'):
            raise PlanningInputError('Choose window coverage or a latest-before point.', prefix + 'kind')
        label = str(form.get(prefix + 'label', '')).strip() or f'Timing requirement {index + 1}'
        if len(label) > 80 or label in labels:
            raise PlanningInputError('Use a unique label of at most 80 characters.', prefix + 'label')
        labels.add(label)
        lower = _bound(form, prefix, 'lower', settings, optional=kind == 'latest_before')
        upper = _bound(form, prefix, 'upper', settings)
        if kind == 'window':
            count = _number(form, prefix + 'minimum_count', 1, integer=True)
            scope = form.get(prefix + 'count_scope', 'new')
            if scope not in ('new', 'all'):
                raise PlanningInputError('Count new times or all study times.', prefix + 'count_scope')
            request = SamplingWindow(lower, upper, count, scope, label)
        else:
            # Count/scope fields are inactive for latest-before requests.
            request = LatestBefore(upper, lower_bound=lower, label=label)
        parsed.append((index, request))
    return parsed


def parse_timing_requirements(form, settings):
    """Validate active form rows without fitting or resolving a reference."""
    return tuple(request for _, request in _parse(form, settings))


def assess_timing(reference, reference_model, observation_model, constraints,
                  settings, form, existing_observations=None):
    """Return constrained suggestions and full outcomes, or None without requests.

    Feasible outcomes are adapted to the ordinary optimization container so
    common simulation and reporting can reuse them. Unavailable sizes and
    search limits remain in ``details['outcomes']`` and the original result.
    """
    parsed = _parse(form, settings)
    if not parsed:
        return None
    if existing_observations is not None and settings.get('future_policy') == 'gap_fill':
        raise PlanningInputError(
            'Timing requirements currently apply to future-only additions. '
            'Choose future-only planning, or remove the timing requirements for gap filling.',
            'future_policy',
        )
    baseline = float(settings.get('baseline_time', constraints.min_time))
    baseline_times = tuple(float(t) for t in constraints.fixed_times
                           if np.isclose(t, baseline, rtol=0., atol=1e-12))
    completed_times = (() if existing_observations is None else
                       tuple(map(float, existing_observations.unique_times)))
    context = PlanningContext(
        baseline_times=baseline_times, completed_times=completed_times,
        planning_start=(constraints.min_time if existing_observations is None else
                        existing_observations.latest_time),
    )
    try:
        resolved = resolve_timing_requirements(
            tuple(request for _, request in parsed), reference, constraints, context=context,
        )
        constrained = optimize_timing_constrained_designs(
            reference_model, observation_model, constraints, resolved,
            number_of_times=tuple(range(settings['min_times'], settings['max_times'] + 1)),
            refine=True, max_exchange_iterations=10,
            max_search_states=MAX_TIMING_SEARCH_STATES,
            max_anchor_search_states=MAX_TIMING_SEARCH_STATES,
            **({} if existing_observations is None else
               {'existing_observations': existing_observations}),
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise PlanningInputError(f'Timing requirements could not be assessed: {error}',
                                 'timing-requirements') from error
    steps, schedules, outcomes = [], {}, []
    for outcome in constrained.designs:
        anchors = []
        if outcome.latest_before_selection is not None:
            anchors = [{
                'label': point.label, 'selected_time': point.selected_time,
                'deadline': point.deadline, 'deadline_slack': point.deadline_slack,
                'priority': point.priority, 'status': point.status, 'message': point.message,
            } for point in outcome.latest_before_selection.points]
        outcomes.append({
            'number_of_times': outcome.number_of_times, 'status': outcome.status,
            'message': outcome.message, 'optimization_status': outcome.optimization_status,
            'times': outcome.times, 'anchors': anchors,
            'evaluation': outcome.requirement_evaluation,
        })
        if outcome.status == 'feasible':
            step = DesignStep(np.asarray(outcome.times), outcome.score,
                              method='timing_constrained_local_i_optimality',
                              n_exchanges=len(outcome.exchanges), metadata=outcome.metadata)
            steps.append(step)
            schedules[f'Suggested · {outcome.number_of_times} times'] = step.times.copy()
    optimization = DesignOptimizationResult(
        constrained.model_name, tuple(steps), tuple(steps),
        np.asarray(constrained.evaluation_times), np.asarray(constrained.evaluation_weights),
        metadata={**constrained.metadata, 'timing_requirements': True},
    )
    requirements = []
    for (index, request), row in zip(parsed, resolved.requirements):
        upper_request = request.upper_bound if isinstance(request, SamplingWindow) else request.deadline
        requirements.append({
            'index': index, 'label': row.label, 'kind': row.kind,
            'lower': row.lower_bound, 'upper': row.upper_bound,
            'effective_lower': row.effective_lower_bound,
            'count_scope': row.count_scope, 'minimum_count': row.minimum_count,
            'status': row.status, 'message': row.message,
            'fixed_times': row.fixed_qualifying_times, 'eligible_count': len(row.eligible_times),
            'lower_description': _description(request.lower_bound),
            'upper_description': _description(upper_request),
        })
    return optimization, schedules, {
        'resolved': resolved, 'constrained': constrained,
        'requirements': requirements, 'outcomes': outcomes, 'evaluations': {},
    }


def inspect_comparison(times, details, constraints):
    """Inspect a comparison without claiming latest-before optimal assignment."""
    if not details or details.get('resolved') is None:
        return None
    return evaluate_timing_requirements(times, details['resolved'], constraints)
