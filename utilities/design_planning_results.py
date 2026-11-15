"""Presentation values for the request-scoped measured-reference comparison.

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


def present_planning_result(result, *, source_filename, source_summary=None, profile_summary=None):
    """Return template context without changing the calculation or ranking designs."""
    settings = result.settings
    reference = result.reference
    source = result.analyzed_source
    warnings = list(result.warnings)
    replicates = int(settings['replicates_per_time'])
    rows, performance = [], {}
    colors = {'Suggested': '#24535c', 'Evenly spaced': '#a75328', 'Manual': '#6f4a92'}
    for label, summary in result.comparison.summaries.items():
        simulated = result.comparison.results[label]
        interval_count = int(np.count_nonzero(
            simulated.fit_success & np.isfinite(simulated.mean_curve_interval_width)))
        kind = ('Suggested' if label.lower().startswith(('suggested', 'optimized')) else
                'Evenly spaced' if label.lower().startswith(('evenly', 'equal')) else 'Manual')
        count = len(summary.design_times)
        row = {
            'name': label, 'kind': kind, 'number_of_times': count,
            'times': ', '.join(_number(value) for value in summary.design_times),
            'experimental_units': count * replicates,
            'n_successful': summary.n_successful, 'n_simulations': summary.n_simulations,
            'failure_rate': _percent(summary.failure_rate),
            'mean_irmse': _number(summary.mean_irmse), 'p90_irmse': _number(summary.p90_irmse),
            'mean_curve_interval_width': _number(summary.mean_curve_interval_width),
            'interval_count': interval_count,
        }
        rows.append(row)
        performance.setdefault(kind, []).append((count, summary.mean_irmse, summary.p90_irmse))
    lines = []
    for kind, values in performance.items():
        values.sort(key=lambda value: value[0])
        for index, metric in ((1, 'Mean'), (2, '90th percentile')):
            lines.append({'label': f'{kind}: {metric}', 'color': colors[kind],
                          'dash': '' if index == 1 else '6 4', 'join': kind != 'Manual',
                          'values': [(value[0], value[index]) for value in values]})
    lower, upper = _reference_band(reference)
    reference_plot = _plot(
        [{'label': 'Central reference', 'color': '#24535c', 'dash': '', 'join': True,
          'values': list(zip(reference.time_grid, reference.central_curve))}],
        band=list(zip(reference.time_grid, lower, upper)),
        observations=list(zip(source.time, source.value)),
    )
    endpoints = []
    for label, degraded_fraction in (('T50', .5), ('T90', .9)):
        milestone = reference_crossing_time_summary(
            reference, degraded_fraction=degraded_fraction, quantile=.5,
            baseline_time=reference.supported_time_range[0],
        )
        if milestone.status == 'invalid_reference':
            warnings.append(f'{label} is unavailable: a positive, finite response at the '
                            'first planning time is needed to define relative degradation.')
        endpoints.append({'name': label,
                          'reached': _percent(milestone.reached_weight)
                          if milestone.status != 'invalid_reference' else '—',
                          'median': _number(milestone.deadline), 'status': milestone.message})
    source_range = float(np.min(source.time)), float(np.max(source.time))
    if reference.time_grid[0] < source_range[0] or reference.time_grid[-1] > source_range[1]:
        warnings.append('The planning horizon extends beyond the measured time range. '
                        'Curve recovery there depends on the chosen model and its extrapolation.')
    if reference.n_valid_draws < reference.n_draws:
        warnings.append('The reference band and simulations use the valid reference draws; '
                        'unusable draws are excluded and the retained weights are normalized.')
    if any(summary.failure_rate > 0 for summary in result.comparison.summaries.values()):
        warnings.append('Some simulated fits failed. Read recovery errors together with the '
                        'success counts; lower error among successful fits can hide failures.')
    if any(not np.isfinite(summary.mean_curve_interval_width)
           for summary in result.comparison.summaries.values()):
        warnings.append('Some fitted curve interval widths are unavailable, even where a curve fit succeeded.')
    elif any(row['interval_count'] < row['n_successful'] for row in rows):
        warnings.append('Some successful fits have unavailable fitted curve intervals. '
                        'The width column gives the number of usable fits.')
    ranks = []
    for step in result.optimization.refined_steps:
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
    return {
        'source_filename': source_filename, 'source_summary': source_summary or {},
        'profile_summary': profile_summary or {}, 'rows': rows,
        'reference_plot': reference_plot, 'performance_plot': _plot(lines),
        'warnings': list(dict.fromkeys(warnings)), 'endpoints': endpoints, 'ranks': ranks,
        'settings_rows': [
            ('Model', settings['model_name']), ('Normalization', normalization),
            ('Future variability', future), ('Independent replicates at each time', str(replicates)),
            ('Minimum future individual SD', _number(settings['minimum_sd'])),
            ('Planning and recovery evaluation range', f'{_number(reference.time_grid[0])}–{_number(reference.time_grid[-1])}'),
            ('Measured time range', f'{_number(source_range[0])}–{_number(source_range[1])}'),
            ('Candidate-time grid', f'{_number(settings["first_time"])} to {_number(settings["last_time"])} in steps of {_number(settings["time_step"])}'),
            ('Fixed proposed times', ', '.join(_number(value) for value in result.constraints.fixed_times) or 'None'),
            ('Requested reference draws', str(requested)),
            ('Usable reference draws', f'{reference.n_valid_draws} of {reference.n_draws} generated'),
            ('Usable analysis-model draws', f'{result.reference_model.n_valid_draws} of {result.reference_model.n_draws}'),
            ('Monte Carlo studies per schedule', str(settings['n_simulations'])),
            ('Random seed', str(settings['random_seed'])),
        ],
    }
