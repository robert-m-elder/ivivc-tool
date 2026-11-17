"""Presentation values for request-scoped study-planning comparisons.

Plot geometry contains finite numbers only. The reference also has an interactive
figure specification; its original numerical values are separate from SVG geometry.
Compact inspection values retain original coordinates for later browser views.
"""
from __future__ import annotations

import numpy as np

from study_design_core import reference_crossing_time_summary

from .design_plotly import make_figure
from .design_recovery_inspection import build_recovery_inspection
from .design_schedule_inspection import build_schedule_inspection


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return '—'
    return f'{value:.5g}' if np.isfinite(value) else '—'


def _percent(value):
    return f'{100 * float(value):.1f}%' if np.isfinite(value) else '—'


def _endpoint_presentation(reference, baseline_time, *, range_available):
    """Use shared weighted crossing quantiles, retaining noncrossing weight."""
    range_available = bool(range_available)
    rows, numerical, warnings = [], [], []
    for label, fraction in (('T50', .5), ('T90', .9)):
        summaries = [reference_crossing_time_summary(
            reference, degraded_fraction=fraction, quantile=quantile,
            baseline_time=baseline_time,
        ) for quantile in (.025, .5, .975)]
        lower, median, upper = summaries

        def display(summary):
            if summary.status == 'beyond_horizon':
                return 'Beyond the assessment horizon'
            return _number(summary.deadline)

        if median.status == 'invalid_reference':
            warnings.append(f'{label} is unavailable: a positive, finite response at the '
                            'milestone baseline and finite relative responses are needed '
                            'in every positive-weight reference trajectory.')
        interval = (f'{display(lower)} to {display(upper)}' if range_available else
                    'Not available (one reference trajectory)')
        if median.status == 'invalid_reference':
            interval = '—'
        rows.append({
            'name': label, 'reached': (_percent(median.reached_weight)
                                      if median.status != 'invalid_reference' else '—'),
            'median': display(median), 'status': median.message,
            'lower_95': display(lower), 'upper_95': display(upper), 'range_95': interval,
            'median_status': median.status, 'lower_status': lower.status,
            'upper_status': upper.status, 'range_available': range_available and
            median.status != 'invalid_reference',
        })
        numerical.append({
            'name': label, 'degraded_fraction': fraction,
            'baseline_time': float(baseline_time),
            'supported_time_range': list(reference.supported_time_range),
            'reached_weight': (float(median.reached_weight)
                               if median.status != 'invalid_reference' else None),
            'sampled_range_available': range_available and median.status != 'invalid_reference',
            'quantiles': [{'probability': row.quantile, 'value': row.deadline,
                           'status': row.status, 'message': row.message}
                          for row in summaries],
        })
    return rows, numerical, warnings


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


def _study_size_ranges(values):
    """Compact retained integer study sizes without inventing missing outcomes."""
    sizes = sorted({int(value) for value in values if value is not None})
    spans = []
    for size in sizes:
        if spans and size == spans[-1][-1] + 1:
            spans[-1].append(size)
        else:
            spans.append([size])
    return ', '.join(str(span[0]) if len(span) == 1 else f'{span[0]}–{span[-1]}'
                     for span in spans) or 'None'


def _timing_outcome_groups(outcomes):
    groups = {}
    for row in outcomes:
        key = (row.get('status', 'unavailable'), row.get('optimization_status', 'not_run'))
        group = groups.setdefault(key, {'status': key[0], 'optimization_status': key[1],
                                       'sizes': [], 'messages': []})
        group['sizes'].append(row.get('number_of_times'))
        message = str(row.get('message', ''))
        if message and message not in group['messages']:
            group['messages'].append(message)
    return [{**group, 'sizes': _study_size_ranges(group['sizes'])} for group in groups.values()]


def _timing_overview(requirements, outcomes, outcome_groups):
    if not requirements:
        return None
    counts = []
    for kind, label in (('window', 'window'), ('latest_before', 'deadline')):
        count = sum(row.get('kind') == kind for row in requirements)
        if count:
            counts.append(f'{count} {label}' + ('s' if count != 1 else ''))
    requirements_text = (' and '.join(counts) if counts else
                         f'{len(requirements)} timing requirements') + ' requested.'
    available = _study_size_ranges(row.get('number_of_times') for row in outcomes
                                   if row.get('status') == 'feasible')
    availability = (f'Suggested schedules available for {available} total sampling times.'
                    if available != 'None' else
                    'No suggested schedule was produced for the requested study sizes.' if outcomes else
                    'Suggested-schedule availability has not been assessed.')
    exceptions = []
    failed_requirements = {}
    for row in requirements:
        status = row.get('status', 'unavailable')
        if status in ('ready', 'already_satisfied'):
            continue
        message = str(row.get('message', ''))
        failed_requirements.setdefault((status, message), []).append(str(row.get('label', 'Requirement')))
    requirement_labels = {
        'beyond_horizon': 'Unresolved milestone', 'invalid_reference': 'Milestone unavailable',
        'invalid_window': 'Invalid timing bounds', 'deadline_elapsed': 'Elapsed deadline',
        'no_feasible_candidate': 'No eligible sampling opportunity',
        'insufficient_candidates': 'Insufficient sampling opportunities',
    }
    for (status, message), labels in failed_requirements.items():
        exceptions.append({'status': status, 'sizes': '',
                           'label': requirement_labels.get(status, 'Requirement unavailable') + ': ' + ', '.join(labels),
                           'message': message or 'This timing requirement could not be resolved or satisfied.'})
    outcome_labels = {
        'infeasible': ('Infeasible study sizes', 'No schedule satisfies the requested timing and design constraints at these sizes.'),
        'search_limit': ('Availability undetermined: search limit', 'The search limit was reached before a feasible schedule was established.'),
        'unavailable': ('Schedule availability unavailable', 'Timing requirements could not be resolved or evaluated for these sizes.'),
    }
    for group in outcome_groups:
        status, optimization = group['status'], group['optimization_status']
        if status == 'feasible':
            if optimization == 'search_limit':
                label, message = ('Feasible; optimization search limit reached',
                                  'A timing-compliant schedule is available, but its improvement search was interrupted.')
            elif optimization == 'exchange_limit':
                label, message = ('Feasible; exchange limit reached',
                                  'A timing-compliant schedule is available, but coordinate exchange reached its iteration limit.')
            else:
                continue
        else:
            label, message = outcome_labels.get(status, (
                'Schedule outcome: ' + str(status).replace('_', ' '),
                ' '.join(group['messages']) or 'See the retained outcome details.'))
        exceptions.append({'status': status, 'sizes': group['sizes'],
                           'label': label + ' — ' + group['sizes'] + ' total times', 'message': message})
    return {'requirements_text': requirements_text, 'availability_text': availability,
            'available_sizes': available, 'exceptions': exceptions}


def _timing_flag(evaluation, *, has_requirements, certified=False):
    """Flag exceptions without interpreting deadline eligibility as latestness."""
    if not has_requirements:
        return None
    if evaluation is None:
        return {'label': 'Timing not assessed', 'title': 'No timing assessment is available for this schedule.'}
    status = evaluation['status']
    if status == 'satisfied' or (status == 'not_past_deadline' and certified):
        return None
    label = {'unsatisfied': 'Timing not met', 'unavailable': 'Timing unavailable',
             'not_past_deadline': 'Latestness unverified'}.get(status, 'Timing: ' + str(status).replace('_', ' '))
    return {'label': label, 'title': evaluation.get('message') or label}


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
    outcome_groups = _timing_outcome_groups(outcomes)
    return {'requirements': requirements, 'outcomes': outcomes, 'evaluations': evaluations,
            'outcome_groups': outcome_groups,
            'overview': _timing_overview(requirements, outcomes, outcome_groups)}


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
    timing_details = getattr(result, 'timing_details', {}) or {}
    timing = _timing_presentation(timing_details)
    latest_labels = {item['label'] for item in timing['requirements']
                     if item.get('kind') == 'latest_before'}
    summaries = {} if result.comparison is None else result.comparison.summaries
    replicates = int(settings['replicates_per_time'])
    rows, performance = [], {}
    colors = {'Suggested': '#24535c', 'Evenly spaced': '#a75328', 'Manual': '#6f4a92'}
    for index, (label, summary) in enumerate(summaries.items()):
        simulated = result.comparison.results[label]
        interval_count = int(np.count_nonzero(
            simulated.fit_success & np.isfinite(simulated.mean_curve_interval_width)))
        kind = ('Suggested' if label.lower().startswith(('suggested', 'optimized')) else
                'Evenly spaced' if label.lower().startswith(('evenly', 'equal')) else 'Manual')
        count = len(summary.design_times)
        new_times = np.asarray([time for time in summary.design_times
                               if not np.any(np.isclose(completed_times, time, rtol=0., atol=1e-12))])
        certified = kind == 'Suggested' and bool(latest_labels) and any(
            outcome.get('status') == 'feasible' and
            len(outcome.get('times', ())) == count and
            np.allclose(np.sort(outcome['times']), np.sort(summary.design_times), rtol=0., atol=1e-12) and
            latest_labels <= {anchor['label'] for anchor in outcome.get('anchors', ())
                              if anchor.get('status') == 'selected' and
                              _number(anchor.get('selected_time')) != '—'}
            for outcome in timing_details.get('outcomes', ()))
        timing_evaluation = timing['evaluations'].get(label)
        row = {
            'id': f'schedule-{index}', 'name': label, 'kind': kind, 'number_of_times': count,
            'times': ', '.join(_number(value) for value in summary.design_times),
            'number_of_new_times': len(new_times),
            'new_times': ', '.join(_number(value) for value in new_times) or 'None',
            'experimental_units': len(new_times) * replicates,
            'timing': timing_evaluation,
            'timing_flag': _timing_flag(timing_evaluation, has_requirements=bool(timing['requirements']),
                                        certified=certified),
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
    performance_x_label = ('New sampling times' if adaptive else
                           'Distinct planned times, including fixed times')
    performance_figure = make_figure(
        lines, title='Simulated recovery error versus number of sampling times',
        x_label=performance_x_label, y_label='IRMSE', identifier='planning-performance-plot',
    )
    if performance_figure is not None:
        performance_layout = performance_figure['layout']
        performance_layout['xaxis'].update(tickmode='linear', tick0=0, dtick=1)
        # Six metric entries can wrap to several rows on a narrow screen.
        performance_layout['height'] = 460
        performance_layout['margin']['b'] = 160
        plot_height = 460 - performance_layout['margin']['t'] - 160
        performance_layout['legend'].update(y=-45 / plot_height, yanchor='top', font={'size': 11})
        for trace in performance_figure['data']:
            # Markers retain a visible result when only one study size is assessed.
            if trace['mode'] == 'lines':
                trace['mode'] = 'lines+markers'
            trace['marker']['symbol'] = ('diamond' if '90th percentile' in trace['name']
                                          else 'circle')
    available_draws = np.count_nonzero(reference.valid_draws & (reference.draw_weights > 0))
    reference_range_available = bool(available_draws > 1)
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
    reference_observations = (
        list(zip(completed.time, completed.response)) if completed is not None else
        list(zip(source.time, source.value)) if source_kind == 'measured' and source is not None else None
    )
    reference_plot = _plot(
        reference_lines,
        band=reference_band,
        observations=reference_observations,
    )
    reference_figure = make_figure(
        reference_lines, band=reference_band, observations=reference_observations,
        title=('Central reference and pointwise 2.5th–97.5th percentiles'
               if reference_range_available else 'Central reference; no sampled uncertainty range'),
        x_label='In vivo time', y_label='Analyzed response', identifier='planning-reference-plot',
        observation_label='Completed observations' if completed is not None else 'Source observations',
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
        title = curve.get('label', 'Source data')
        y_label = 'Analyzed source response'
        identifier = f'planning-source-{index}'
        observations = list(zip(curve['time'], curve['value']))
        figure = make_figure(
            series, observations=observations, title=title,
            x_label=x_label, y_label=y_label, identifier=identifier,
            observation_label='Supplied observations',
        )
        if figure is not None:
            # A calibration fit and its observations can require two legend
            # rows on narrow screens. Keep both labels below the time axis.
            figure['layout']['height'] = 420
            figure['layout']['margin']['b'] = 110
            plot_height = 420 - figure['layout']['margin']['t'] - 110
            figure['layout']['legend'].update(
                y=-45 / plot_height, yanchor='top', font={'size': 11},
            )
        source_plots.append({
            'title': title, 'x_label': x_label, 'y_label': y_label,
            'identifier': identifier, 'figure': figure,
            'plot': _plot(series, observations=observations),
        })
    endpoints, endpoint_data, endpoint_warnings = _endpoint_presentation(
        reference, settings.get('baseline_time', reference.supported_time_range[0]),
        range_available=reference_range_available,
    )
    warnings.extend(endpoint_warnings)
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
        ('Analysis task', 'Plan or extend an in vivo study'),
        ('Reference source', {
            'measured': 'Measured in vivo data', 'ivivc': 'IVIVC prediction',
            'scaling': 'Empirical time scaling',
        }.get(source_kind, source_label)),
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
        'performance_x_label': performance_x_label,
        'profile_summary': profile_summary or {}, 'rows': rows,
        'reference_plot': reference_plot, 'reference_figure': reference_figure,
        'performance_plot': _plot(lines),
        'performance_figure': performance_figure,
        'warnings': list(dict.fromkeys(warnings)), 'endpoints': endpoints, 'ranks': ranks,
        'inspection': {
            'schema_version': 1,
            'reference': {
                'time': reference.time_grid.tolist(),
                'central_response': reference.central_curve.tolist(),
                'lower': lower.tolist() if reference_range_available else None,
                'upper': upper.tolist() if reference_range_available else None,
                'range_available': reference_range_available,
                'range_interpretation': 'Pointwise weighted central 95% reference range; '
                                       'excludes future measurement variability.',
                'time_label': 'In vivo time', 'response_label': 'Analyzed response',
            },
            'schedule_data': build_schedule_inspection(result),
            'recovery': build_recovery_inspection(result),
            'endpoints': endpoint_data,
        },
        'settings_rows': settings_rows,
    }
