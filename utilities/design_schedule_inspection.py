"""Compact, numerical schedule explanations retained for browser inspection.

This presentation layer describes completed calculations. It does not resolve
requirements, choose points, or certify a comparison schedule's latestness.
Window contributions come from the backend's retained schedule evaluations;
explicit latest-before attribution requires a retained certified selection.
"""
from __future__ import annotations

import numpy as np


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _contains(times, time):
    return bool(np.any(np.isclose(times, time, rtol=0., atol=1e-12)))


def _same_times(left, right):
    left, right = np.sort(left), np.sort(right)
    return bool(len(left) == len(right) and np.allclose(left, right, rtol=0., atol=1e-12))


def _kind(name):
    name = name.lower()
    return ('suggested' if name.startswith(('suggested', 'optimized')) else
            'evenly_spaced' if name.startswith(('evenly', 'equal')) else 'manual')


def _requirements(details):
    rows = []
    for row in details.get('requirements', ()):
        rows.append({
            'index': int(row['index']), 'label': str(row['label']), 'kind': str(row['kind']),
            'lower': _finite(row.get('lower')), 'upper': _finite(row.get('upper')),
            'effective_lower': _finite(row.get('effective_lower')),
            'count_scope': str(row['count_scope']), 'minimum_count': int(row['minimum_count']),
            'status': str(row['status']), 'message': str(row['message']),
            'lower_description': str(row.get('lower_description', '')),
            'upper_description': str(row.get('upper_description', '')),
        })
    return rows


def _outcomes(details):
    return [{
        'number_of_times': int(row['number_of_times']), 'status': str(row['status']),
        'message': str(row['message']), 'optimization_status': str(row['optimization_status']),
        'times': [float(time) for time in row.get('times', ())],
        'anchors': [{
            'label': str(anchor['label']), 'selected_time': _finite(anchor.get('selected_time')),
            'deadline': _finite(anchor.get('deadline')),
            'deadline_slack': _finite(anchor.get('deadline_slack')),
            'priority': int(anchor['priority']), 'status': str(anchor['status']),
            'message': str(anchor['message']),
        } for anchor in row.get('anchors', ())],
    } for row in details.get('outcomes', ())]


def _schedule_requirements(evaluations, requirements, anchors):
    """Join retained counts with rule bounds and certified anchor evidence.

    A comparison's eligible point is not a latest-before selection. ``anchors``
    is supplied only for the matching feasible suggested schedule below.
    """
    by_label = {row['label']: row for row in requirements}
    selected = {row['label']: row for row in anchors
                if row['status'] == 'selected' and row['selected_time'] is not None}
    rows = []
    for evaluation in evaluations:
        requirement = by_label.get(evaluation.label, {})
        anchor = selected.get(evaluation.label) if evaluation.kind == 'latest_before' else None
        rows.append({
            'label': str(evaluation.label), 'kind': str(evaluation.kind),
            'status': str(evaluation.status), 'message': str(evaluation.message),
            'lower': requirement.get('lower'), 'upper': requirement.get('upper'),
            'count_scope': requirement.get('count_scope', 'new'),
            'achieved_count': int(evaluation.achieved_count),
            'minimum_count': int(evaluation.minimum_count),
            'qualifying_times': [_finite(time) for time in evaluation.qualifying_times],
            'deadline': (None if evaluation.kind != 'latest_before' else
                         requirement.get('upper') if anchor is None else anchor['deadline']),
            'selected_time': None if anchor is None else anchor['selected_time'],
            'deadline_slack': (_finite(evaluation.deadline_slack) if anchor is None
                               else anchor['deadline_slack']),
            'latest_selected': anchor is not None,
        })
    return rows


def build_schedule_inspection(result):
    """Return JSON-safe point roles, guides, and retained timing diagnostics.

    Schedule IDs use result insertion order. ``baseline_time`` is a guide even
    when unsampled; it never creates an observation. A point may have several
    roles. ``contributes_to`` lists window-count contributions, without implying
    that an individual point was mandated. ``deadline_eligibility`` only reports
    backend eligibility. ``latest_before`` labels occur solely on selected
    anchors of the corresponding certified suggested schedule, including when
    a fixed proposed time also supplies that anchor. ``planning_start`` belongs
    only to resolved timing requirements. Latest-completed and first-opportunity
    guides remain separate, especially when gap filling permits earlier times.
    ``baseline`` marks coincidence with the milestone response baseline;
    ``timing_baseline`` marks the explicit fixed role excluded by the scheduler.
    """
    settings = result.settings
    details = getattr(result, 'timing_details', {}) or {}
    completed = getattr(result, 'completed_observations', None)
    completed_times = () if completed is None else completed.unique_times
    requested_fixed = settings.get('requested_fixed_times', ())
    baseline = float(settings.get('baseline_time', result.constraints.min_time))
    resolved = details.get('resolved')
    planning_start = (None if resolved is None else
                      _finite(resolved.context.planning_start))
    latest_completed = None if completed is None else float(completed.latest_time)
    timing_baselines = ([] if resolved is None else
                        [float(time) for time in resolved.context.baseline_times])
    requirements = _requirements(details)
    outcomes = _outcomes(details)
    schedules = []
    sampled_by = []
    origin_labels = {'suggested': 'Selected by curve-recovery information optimization.',
                     'evenly_spaced': 'Selected for the evenly spaced comparison.',
                     'manual': 'Specified in this manual comparison.'}
    for index, (name, values) in enumerate(result.schedules.items()):
        times = [float(time) for time in values]
        identifier = f'schedule-{index}'
        kind = _kind(name)
        evaluation = details.get('evaluations', {}).get(name)
        evaluations = () if evaluation is None else evaluation.requirements
        outcome = next((row for row in outcomes if kind == 'suggested'
                        and row['status'] == 'feasible'
                        and _same_times(row['times'], times)), None)
        anchors = [] if outcome is None else outcome['anchors']
        points = []
        for time in times:
            is_completed = _contains(completed_times, time)
            is_baseline = bool(np.isclose(time, baseline, rtol=0., atol=1e-12))
            is_timing_baseline = _contains(timing_baselines, time)
            is_fixed = not is_completed and _contains(requested_fixed, time)
            latest = [anchor['label'] for anchor in anchors
                      if anchor['status'] == 'selected' and anchor['selected_time'] is not None
                      and np.isclose(anchor['selected_time'], time, rtol=0., atol=1e-12)]
            contributions = [row.label for row in evaluations
                             if row.kind == 'window' and _contains(row.qualifying_times, time)]
            eligible = [row.label for row in evaluations
                        if row.kind == 'latest_before' and _contains(row.qualifying_times, time)]
            roles = ['completed' if is_completed else 'new']
            explanations = ['Completed observation; retained without simulating it again.'
                            if is_completed else 'Proposed new measurement.']
            if is_baseline:
                roles.append('baseline')
                explanations.append('Sampled at the milestone response baseline.')
            if is_timing_baseline:
                explanations.append('Fixed timing baseline; excluded from timing requirement counts.')
            if is_fixed:
                roles.append('user_fixed')
                explanations.append('Fixed by the user for a proposed measurement.')
            if latest:
                roles.append('latest_before')
                explanations.extend('Selected by the explicit latest-before rule: ' + label + '.'
                                    for label in latest)
            origin = ('completed' if is_completed else 'user_fixed' if is_fixed else
                      'latest_before' if latest else kind)
            if origin in origin_labels:
                explanations.append(origin_labels[origin])
            explanations.extend('Contributes to window count: ' + label + '.'
                                for label in contributions)
            explanations.extend('Eligible before deadline: ' + label +
                                '; latestness is not established by eligibility alone.'
                                for label in eligible if label not in latest)
            completed_rows = int(np.count_nonzero(np.isclose(
                completed.time, time, rtol=0., atol=1e-12))) if is_completed else 0
            points.append({
                'time': time, 'origin': origin, 'roles': roles, 'completed': is_completed,
                'baseline': is_baseline, 'timing_baseline': is_timing_baseline,
                'user_fixed': is_fixed,
                'latest_before': latest, 'contributes_to': contributions,
                'deadline_eligibility': eligible, 'completed_rows': completed_rows,
                'new_replicates': 0 if is_completed else int(settings['replicates_per_time']),
                'explanations': explanations,
            })
        if any(point['baseline'] for point in points):
            sampled_by.append(identifier)
        schedules.append({
            'id': identifier, 'name': str(name), 'kind': kind, 'times': times, 'points': points,
            'timing_status': None if evaluation is None else str(evaluation.status),
            'timing_message': None if evaluation is None else str(evaluation.message),
            'timing_requirements': _schedule_requirements(evaluations, requirements, anchors),
        })
    return {'baseline_time': baseline, 'planning_start': planning_start,
            'latest_completed_time': latest_completed,
            'timing_baseline_times': timing_baselines,
            'first_opportunity_time': _finite(settings.get('first_time')),
            'evaluation_time_range': [float(result.constraints.min_time),
                                      float(result.constraints.max_time)],
            'future_policy': str(settings.get('future_policy', 'gap_fill')),
            'baseline_sampled_by': sampled_by, 'requirements': requirements,
            'outcomes': outcomes, 'schedules': schedules}
