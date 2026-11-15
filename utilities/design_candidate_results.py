"""Presentation of generated calibration schedules, without simulation or refitting."""
from __future__ import annotations

import numpy as np

from .design_calibration_results import SIDE_LABELS, _existing_rows, _observation_description
from .design_planning_results import _number


def _times_text(times):
    return ', '.join(_number(value) for value in times) or 'None'


def present_generation_result(result, *, source_summaries=None, profile_summaries=None):
    """Format the shared optimizer's outcomes; do not calculate performance scores."""
    candidates = {}
    existing = {row['side']: row for row in _existing_rows(result.analyzed_existing)}
    for outcome in result.generation_outcomes:
        if outcome['kind'] != 'generated':
            continue
        name, side = outcome['candidate'], outcome['side']
        if name not in candidates:
            design = result.designs[name]
            total_units = 0
            for environment in SIDE_LABELS:
                observation = getattr(design, environment + '_observation_model')
                if observation is not None:
                    total_units += len(getattr(design, environment + '_times')) * observation.replicates_per_time
            candidates[name] = {
                'name': name, 'index': int(outcome['candidate_index']),
                'total_new_units': int(total_units), 'rows': [],
            }
        observation = getattr(result.designs[name], side + '_observation_model')
        description = _observation_description(observation)
        if observation.kind == 'constant_sd':
            description = 'Individual SD = ' + str(float(observation.value))
        elif observation.kind == 'constant_cv':
            description = ('Individual CV = ' + str(float(observation.value)) +
                           '; minimum SD = ' + str(float(observation.minimum_sd)))
        fixed = existing[SIDE_LABELS[side]]
        candidates[name]['rows'].append({
            'side': SIDE_LABELS[side], 'status': outcome['status'],
            'times': _times_text(outcome['times']), 'time_count': len(outcome['times']),
            'replicates': int(outcome['replicates']), 'new_units': int(outcome['new_units']),
            'fixed_times': _times_text(outcome['fixed_times']),
            'existing_rows': fixed['rows'], 'existing_units': fixed['units'],
            'existing_precision': fixed['precision'],
            'observation': description,
            'criterion': _number(outcome.get('information_criterion')),
            'range': '–'.join(_number(value) for value in outcome['reference_time_range']),
        })
    sources = []
    for side, detail in result.source_details.items():
        grid = detail.get('generation_grid')
        if grid is None:
            continue
        reference = result.references.get(side)
        constants = detail.get('normalization', {})
        opportunities = np.asarray(grid['candidate_times'])
        sources.append({
            'side': SIDE_LABELS[side], 'mode': detail['mode'],
            'source_kind': detail['source_kind'], 'model': detail.get('model', '—'),
            'first_time': _number(grid['first_time']),
            'last_time_bound': _number(grid['last_time_bound']),
            'time_step': _number(grid['time_step']),
            'last_opportunity': _number(opportunities[-1]) if len(opportunities) else '—',
            'opportunity_count': len(opportunities),
            'observed_range': '–'.join(_number(value) for value in detail['observed_time_range']),
            'reference_range': '–'.join(_number(value) for value in detail['reference_time_range']),
            'slope': _number(constants.get('slope', 1.)),
            'intercept': _number(constants.get('intercept', 0.)),
            'draws': ('—' if reference is None else
                      f'{reference.n_valid_draws} / {reference.n_draws}'),
        })
    return {
        'generation_candidates': list(candidates.values()),
        'generation_sources': sources,
        'generation_warnings': list(result.warnings),
        'generation_settings': {
            'random_seed': result.settings['random_seed'],
            'reference_draws': result.settings['reference_draws'],
        },
        'source_summaries': list(source_summaries or []),
        'profile_summaries': list(profile_summaries or []),
    }
