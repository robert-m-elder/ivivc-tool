"""Presentation values for request-scoped study-planning comparisons.

Plot geometry contains finite numbers only. Templates escape all uploaded names
and text, and render the small SVG plots without remote scripts or new packages.
"""
from __future__ import annotations

import numpy as np

from study_design_core import reference_crossing_time_summary


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return '—'
    return f'{value:.5g}' if np.isfinite(value) else '—'


def _percent(value):
    return f'{100 * float(value):.1f}%' if np.isfinite(value) else '—'


def _plot(series, *, band=None, observations=None):
    """Map finite coordinates to one shared SVG frame; retain gaps in lines."""
    finite = []
    for line in series:
        finite.extend((float(x), float(y)) for x, y in line['values']
                      if np.isfinite(x) and np.isfinite(y))
    if band is not None:
        finite.extend((float(x), float(y)) for x, lo, hi in band for y in (lo, hi)
                      if np.isfinite(x) and np.isfinite(y))
    if observations is not None:
        finite.extend((float(x), float(y)) for x, y in observations
                      if np.isfinite(x) and np.isfinite(y))
    if not finite:
        return None
    xs, ys = np.asarray(finite).T
    xmin, xmax, ymin, ymax = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    if xmin == xmax:
        xmin, xmax = xmin - .5, xmax + .5
    if ymin == ymax:
        pad = max(abs(ymin) * .1, .01)
        ymin, ymax = ymin - pad, ymax + pad
    else:
        pad = (ymax - ymin) * .05
        ymin, ymax = ymin - pad, ymax + pad
    def point(x, y):
        return (70 + (float(x) - xmin) / (xmax - xmin) * 650,
                235 - (float(y) - ymin) / (ymax - ymin) * 210)
    plotted = []
    for line in series:
        path, circles, connected = [], [], False
        for x, y in line['values']:
            if not np.isfinite(x) or not np.isfinite(y):
                connected = False
                continue
            px, py = point(x, y)
            path.append(f'{"L" if connected else "M"}{px:.2f},{py:.2f}')
            circles.append((round(px, 2), round(py, 2)))
            connected = True
        plotted.append({**line, 'path': ' '.join(path), 'circles': circles})
    polygon = ''
    if band is not None:
        valid = [(x, lo, hi) for x, lo, hi in band
                 if np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)]
        coordinates = [point(x, lo) for x, lo, hi in valid]
        coordinates += [point(x, hi) for x, lo, hi in reversed(valid)]
        polygon = ' '.join(f'{x:.2f},{y:.2f}' for x, y in coordinates)
    return {
        'series': plotted, 'band': polygon,
        'observations': [point(x, y) for x, y in observations or []
                         if np.isfinite(x) and np.isfinite(y)],
        'x_ticks': [(point(x, ymin)[0], _number(x)) for x in np.linspace(xmin, xmax, 5)],
        'y_ticks': [(point(xmin, y)[1], _number(y)) for y in np.linspace(ymin, ymax, 5)],
    }


def _reference_band(reference):
    """Pointwise inverse weighted-empirical-CDF percentiles of usable draws."""
    active = reference.valid_draws & (reference.draw_weights > 0)
    curves, weights = reference.value_draws[active], reference.draw_weights[active]
    order = np.argsort(curves, axis=0, kind='stable')
    sorted_curves = np.take_along_axis(curves, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    cumulative /= cumulative[-1]
    return [sorted_curves[np.argmax(cumulative >= q, axis=0), np.arange(curves.shape[1])]
            for q in (.025, .975)]


def _adaptive_presentation(result):
    """Describe actual completed rows without inventing historical unit counts."""
    completed = getattr(result, 'completed_observations', None)
    if completed is None:
        return None
    precision = completed.observation_uncertainty
    if completed.response_kind == 'individual':
        units = str(completed.n_observations)
    elif precision is not None and precision.replicate_counts is not None:
        units = str(sum(int(value) for value in precision.replicate_counts))
    else:
        units = 'Not supplied for the reported means'
    assumed = precision is not None and precision.source == 'future_model_assumption'
    interpretation = ('Completed observation uncertainty is unspecified.' if precision is None else
                      'Individual SD assumed from the future measurement model at the completed responses and times, then held fixed.'
                      if assumed else
                      'Supplied SEM describes each completed mean; counts do not divide it again.'
                      if completed.response_kind == 'mean' else
                      'Supplied individual SD describes each completed response.')
    update = getattr(result, 'adaptive_update', None)
    adaptive = (getattr(result, 'source_details', {}) or {}).get('adaptive', {})
    rows = [
        ('Completed rows', str(completed.n_observations)),
        ('Distinct completed times', str(completed.n_time_points)),
        ('Completed independent units reported', units),
        ('Latest completed elapsed time', _number(completed.latest_time)),
        ('Completed response scale', adaptive.get('completed_scale', 'Analyzed reference scale')),
    ]
    if update is None:
        method = 'The completed dataset was fitted once to construct the reference. It was not applied again as a likelihood update.'
    else:
        method = 'The original prior reference was updated once using the cumulative completed observations. Complete trajectory shapes were retained and only their weights changed.'
        rows.extend([
            ('Effective number of reference trajectories after updating', _number(update.effective_sample_size)),
            ('Largest updated trajectory weight', _percent(np.max(update.posterior_weights))),
        ])
        agreement = update.metadata.get('observation_agreement', {})
        best = agreement.get('best_standardized_residual_rms', adaptive.get('best_standardized_residual_rms'))
        if best is not None:
            rows.append(('Closest reference agreement: residual RMS divided by observation SD', _number(best)))
        weighted = agreement.get('posterior_weighted_standardized_residual_rms')
        if weighted is not None:
            rows.append(('Updated weighted residual RMS divided by observation SD', _number(weighted)))
    preview = []
    for index, (time, response) in enumerate(zip(completed.time[:10], completed.response[:10])):
        preview.append({'time': _number(time), 'response': _number(response),
                        'uncertainty': _number(precision.sigma[index]) if precision is not None else '—',
                        'count': (str(int(precision.replicate_counts[index]))
                                  if precision is not None and precision.replicate_counts is not None else '—')})
    return {'rows': rows, 'method': method, 'interpretation': interpretation,
            'preview': preview, 'row_count': completed.n_observations,
            'has_update': update is not None,
            'response_kind': completed.response_kind,
            'future_policy': result.settings.get('future_policy', 'future_only')}


def _timing_presentation(details):
    """Format retained feasibility and inspection outcomes, without new decisions."""
    requirements = []
    for row in details.get('requirements', ()):
        requirements.append({**row, 'lower': _number(row.get('lower')),
                             'upper': _number(row.get('upper')),
                             'effective_lower': _number(row.get('effective_lower')),
                             'fixed_times': ', '.join(_number(value) for value in row.get('fixed_times', ())) or 'None'})
    outcomes = []
    for row in details.get('outcomes', ()):
        anchors = [{**anchor, 'selected_time': _number(anchor.get('selected_time')),
                    'deadline': _number(anchor.get('deadline')),
                    'deadline_slack': _number(anchor.get('deadline_slack'))}
                   for anchor in row.get('anchors', ())]
        outcomes.append({**row, 'times': ', '.join(_number(value) for value in row.get('times', ())) or '—',
                         'anchors': anchors})
    evaluations = {}
    for label, evaluation in details.get('evaluations', {}).items():
        evaluations[label] = {'status': evaluation.status, 'message': evaluation.message,
                              'requirements': [{'label': row.label, 'status': row.status,
                                                'message': row.message,
                                                'achieved_count': row.achieved_count,
                                                'minimum_count': row.minimum_count,
                                                'deadline_slack': _number(row.deadline_slack)}
                                               for row in evaluation.requirements]}
    return {'requirements': requirements, 'outcomes': outcomes, 'evaluations': evaluations}


def present_planning_result(result, *, source_filename=None, source_summary=None,
                            profile_summary=None, source_summaries=None):
    """Return template context without changing the calculation or ranking designs."""
    settings = result.settings
    reference = result.reference
    source = result.analyzed_source
    details = getattr(result, 'source_details', {}) or {}
    source_kind = details.get('kind', settings.get('reference_source', 'measured'))
    source_label = details.get('label', {
        'measured': 'Measured in vivo reference', 'ivivc': 'IVIVC-derived in vivo reference',
        'scaling': 'Empirically time-scaled in vivo reference',
    }.get(source_kind, 'In vivo reference'))
    warnings = list(result.warnings)
    completed = getattr(result, 'completed_observations', None)
    adaptive = _adaptive_presentation(result)
    completed_times = np.array([]) if completed is None else completed.unique_times
    timing = _timing_presentation(getattr(result, 'timing_details', {}) or {})
    summaries = {} if result.comparison is None else result.comparison.summaries
    replicates = int(settings['replicates_per_time'])
    rows, performance = [], {}
    colors = {'Suggested': '#24535c', 'Evenly spaced': '#a75328', 'Manual': '#6f4a92'}
    for label, summary in summaries.items():
        simulated = result.comparison.results[label]
        interval_count = int(np.count_nonzero(
            simulated.fit_success & np.isfinite(simulated.mean_curve_interval_width)))
        kind = ('Suggested' if label.lower().startswith(('suggested', 'optimized')) else
                'Evenly spaced' if label.lower().startswith(('evenly', 'equal')) else 'Manual')
        count = len(summary.design_times)
        new_times = np.asarray([time for time in summary.design_times
                               if not np.any(np.isclose(completed_times, time, rtol=0., atol=1e-12))])
        row = {
            'name': label, 'kind': kind, 'number_of_times': count,
            'times': ', '.join(_number(value) for value in summary.design_times),
            'number_of_new_times': len(new_times),
            'new_times': ', '.join(_number(value) for value in new_times) or 'None',
            'experimental_units': len(new_times) * replicates,
            'timing': timing['evaluations'].get(label),
            'n_successful': summary.n_successful, 'n_simulations': summary.n_simulations,
            'failure_rate': _percent(summary.failure_rate),
            'mean_irmse': _number(summary.mean_irmse), 'p90_irmse': _number(summary.p90_irmse),
            'mean_curve_interval_width': _number(summary.mean_curve_interval_width),
            'interval_count': interval_count,
        }
        rows.append(row)
        performance.setdefault(kind, []).append((len(new_times) if adaptive else count,
                                                 summary.mean_irmse, summary.p90_irmse))
    lines = []
    for kind, values in performance.items():
        values.sort(key=lambda value: value[0])
        for index, metric in ((1, 'Mean'), (2, '90th percentile')):
            lines.append({'label': f'{kind}: {metric}', 'color': colors[kind],
                          'dash': '' if index == 1 else '6 4', 'join': kind != 'Manual',
                          'values': [(value[0], value[index]) for value in values]})
    available_draws = np.count_nonzero(reference.valid_draws & (reference.draw_weights > 0))
    reference_range_available = available_draws > 1
    if reference_range_available:
        lower, upper = _reference_band(reference)
        reference_band = list(zip(reference.time_grid, lower, upper))
    else:
        reference_band = None
        warnings.append('Only one usable reference trajectory is available. It supplies no '
                        'sampled reference-uncertainty range; this does not mean the true curve is known.')
    reference_lines = [{'label': ('Updated reference' if adaptive and adaptive['has_update'] else
                                  'Reference from completed data' if adaptive else 'Central reference'),
                        'color': '#24535c', 'dash': '', 'join': True,
                        'values': list(zip(reference.time_grid, reference.central_curve))}]
    prior = getattr(result, 'prior_reference', None)
    if prior is not None:
        reference_lines.insert(0, {'label': 'Original prior reference', 'color': '#a75328',
                                  'dash': '6 4', 'join': True,
                                  'values': list(zip(prior.time_grid, prior.central_curve))})
    reference_plot = _plot(
        reference_lines,
        band=reference_band,
        observations=(list(zip(completed.time, completed.response)) if completed is not None else
                      list(zip(source.time, source.value))
                      if source_kind == 'measured' and source is not None else None),
    )
    source_plots = []
    for index, curve in enumerate(details.get('source_curves', ())):
        series = []
        if curve.get('fitted_time') is not None and curve.get('fitted_value') is not None:
            series.append({'label': 'Fitted source curve', 'color': '#24535c', 'dash': '',
                           'join': True, 'values': list(zip(curve['fitted_time'], curve['fitted_value']))})
        environment = curve.get('environment', '')
        x_label = ('In vivo time' if environment == 'in_vivo' else
                   'In vitro time' if environment == 'in_vitro' else 'Source time')
        source_plots.append({
            'title': curve.get('label', 'Source data'), 'x_label': x_label,
            'identifier': f'planning-source-{index}',
            'plot': _plot(series, observations=list(zip(curve['time'], curve['value']))),
        })
    endpoints = []
    for label, degraded_fraction in (('T50', .5), ('T90', .9)):
        milestone = reference_crossing_time_summary(
            reference, degraded_fraction=degraded_fraction, quantile=.5,
            baseline_time=settings.get('baseline_time', reference.supported_time_range[0]),
        )
        if milestone.status == 'invalid_reference':
            warnings.append(f'{label} is unavailable: a positive, finite response at the '
                            'first planning time is needed to define relative degradation.')
        endpoints.append({'name': label,
                          'reached': _percent(milestone.reached_weight)
                          if milestone.status != 'invalid_reference' else '—',
                          'median': _number(milestone.deadline), 'status': milestone.message})
    source_range = ((float(np.min(source.time)), float(np.max(source.time)))
                    if source_kind == 'measured' and source is not None else None)
    if source_range is not None and (reference.time_grid[0] < source_range[0] or reference.time_grid[-1] > source_range[1]):
        warnings.append('The planning horizon extends beyond the measured time range. '
                        'Curve recovery there depends on the chosen model and its extrapolation.')
    if reference.n_valid_draws < reference.n_draws:
        warnings.append('The reference band and simulations use the valid reference draws; '
                        'unusable draws are excluded and the retained weights are normalized.')
    if any(summary.failure_rate > 0 for summary in summaries.values()):
        warnings.append('Some simulated fits failed. Read recovery errors together with the '
                        'success counts; lower error among successful fits can hide failures.')
    if any(not np.isfinite(summary.mean_curve_interval_width)
           for summary in summaries.values()):
        warnings.append('Some fitted curve interval widths are unavailable, even where a curve fit succeeded.')
    elif any(row['interval_count'] < row['n_successful'] for row in rows):
        warnings.append('Some successful fits have unavailable fitted curve intervals. '
                        'The width column gives the number of usable fits.')
    ranks = []
    for step in (() if result.optimization is None else result.optimization.refined_steps):
        if settings['min_times'] <= step.number_of_times <= settings['max_times']:
            ranks.append({'number_of_times': step.number_of_times,
                          'rank': step.score.min_rank, 'parameters': step.score.parameter_count})
            if not step.score.is_full_rank:
                warnings.append(f'The suggested {step.number_of_times}-time schedule has incomplete '
                                'local information rank. Parameter precision may be poorly constrained.')
    observation_kind = settings['observation_kind']
    if observation_kind == 'time_profile_sd':
        future = 'Time-dependent individual SD; nearest-time lookup with endpoint extension'
    elif observation_kind == 'constant_cv':
        future = f'Individual CV = {_number(settings["observation_value"])} (fraction of absolute response)'
    else:
        future = f'Constant individual SD = {_number(settings["observation_value"])}'
    normalizations = {'none': 'None', 'initial_value': 'Divide by the initial response',
                      'min_max': 'Min–max normalization'}
    normalization = normalizations.get(settings['normalization'], settings['normalization'])
    if settings.get('one_minus'):
        normalization += '; apply one minus response'
    requested = settings['reference_draws']
    settings_rows = [
        ('Model', settings['model_name']), ('Normalization', normalization),
        ('Future variability', future), ('Independent replicates at each time', str(replicates)),
        ('Minimum future individual SD', _number(settings['minimum_sd'])),
        ('Planning and recovery evaluation range', f'{_number(reference.time_grid[0])}–{_number(reference.time_grid[-1])}'),
    ]
    if source_range is not None:
        settings_rows.append(('Measured time range', f'{_number(source_range[0])}–{_number(source_range[1])}'))
    settings_rows.extend([
        ('Candidate-time grid', f'{_number(settings.get("candidate_first_time", settings["first_time"]))} to {_number(settings["last_time"])} in steps of {_number(settings["time_step"])}'),
        ('Fixed times, including completed observations' if adaptive else 'Fixed proposed times', ', '.join(_number(value) for value in result.constraints.fixed_times) or 'None'),
        ('Milestone baseline time', _number(settings.get('baseline_time', reference.supported_time_range[0]))),
        ('Requested parameter draws per active ratio' if source_kind == 'scaling' else
         'Requested reference draws', str(requested)),
        ('Usable reference draws', f'{reference.n_valid_draws} of {reference.n_draws} generated'),
        ('Usable analysis-model draws', f'{result.reference_model.n_valid_draws} of {result.reference_model.n_draws}'),
        ('Monte Carlo studies per schedule', str(settings['n_simulations'])),
        ('Random seed', str(settings['random_seed'])),
    ])
    if settings.get('excluded_times'):
        settings_rows.append(('Unavailable proposed times', ', '.join(
            _number(value) for value in settings['excluded_times'])))
    scale_records = []
    ratios, scale_weights = details.get('ratios', []), details.get('weights', [])
    scale_current = None
    if getattr(result, 'adaptive_update', None) is not None:
        scale_current = result.adaptive_update.metadata.get('time_scaling_weight_summary', {}).get('current_weights')
    identifiers = details.get('source_ids')
    for index, (ratio, weight) in enumerate(zip(ratios[:10], scale_weights[:10])):
        scale_records.append({'row': index + 1, 'ratio': _number(ratio),
                              'weight': _number(weight),
                              'updated_weight': _number(scale_current[index]) if scale_current is not None else None,
                              'source': identifiers[index] if identifiers is not None else '—'})
    return {
        'source_filename': source_filename, 'source_summary': source_summary or {},
        'source_summaries': source_summaries or [], 'source_kind': source_kind,
        'source_label': source_label, 'source_plots': source_plots,
        'scale_records': scale_records, 'scale_record_count': len(ratios),
        'source_details_rows': [(label, _number(value) if isinstance(value, (int, float, np.number))
                                else value) for label, value in details.get('rows', [])],
        'source_fit_rows': [{**row, 'value': _number(row.get('value')),
                             'se': _number(row.get('se'))} for row in details.get('fit_rows', [])],
        'reference_range_available': reference_range_available,
        'adaptive': adaptive, 'timing': timing, 'has_comparison': result.comparison is not None,
        'has_updated_scale_weights': scale_current is not None,
        'milestone_baseline_time': _number(settings.get('baseline_time', reference.supported_time_range[0])),
        'performance_x_label': 'New sampling times' if adaptive else 'Distinct planned times, including fixed times',
        'profile_summary': profile_summary or {}, 'rows': rows,
        'reference_plot': reference_plot, 'performance_plot': _plot(lines),
        'warnings': list(dict.fromkeys(warnings)), 'endpoints': endpoints, 'ranks': ranks,
        'settings_rows': settings_rows,
    }
