"""Planning specifications and reporting for the mixed-requirements example."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Mapping

import numpy as np
import pandas as pd

from study_design_core import CrossingTimeTarget, LatestBefore, PlanningContext, SamplingWindow


PLANNING_INPUT_MODES = (
    'prospective-measured', 'prospective-ivivc', 'adaptive-in-vivo', 'adaptive-ivivc',
)
PLANNING_DESIGN_LABELS = {
    'central_schedule': 'Central I-optimal schedule (unrestricted benchmark)',
    'timing_schedule': 'I-optimal schedule with timing requirements',
}


@dataclass(frozen=True)
class SchedulePlanningSpec:
    """Source-independent timing requests for the POC's common run path."""

    context: PlanningContext
    requirements: tuple[SamplingWindow | LatestBefore, ...]
    input_mode: str
    search_limit: int = 100000

    def __post_init__(self):
        if not isinstance(self.context, PlanningContext):
            raise TypeError('context must be a PlanningContext.')
        requirements = tuple(self.requirements)
        if not all(isinstance(item, (SamplingWindow, LatestBefore)) for item in requirements):
            raise TypeError('requirements must contain SamplingWindow or LatestBefore requests.')
        if self.input_mode not in PLANNING_INPUT_MODES:
            raise ValueError('Unknown planning input mode.')
        if (isinstance(self.search_limit, bool) or not isinstance(self.search_limit, (int, np.integer))
                or self.search_limit < 1):
            raise ValueError('search_limit must be a positive integer.')
        object.__setattr__(self, 'requirements', requirements)
        object.__setattr__(self, 'search_limit', int(self.search_limit))


def _times(values):
    return '; '.join(f'{float(value):g}' for value in values)


def planning_json_safe(value):
    """Preserve unavailable values as JSON null, including nested arrays."""
    if isinstance(value, Mapping):
        return {str(key): planning_json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return planning_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return planning_json_safe(value.item())
    if isinstance(value, (tuple, list)):
        return [planning_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bound(bound):
    if bound is None:
        return {'kind': 'planning_start'}
    if isinstance(bound, CrossingTimeTarget):
        return {'kind': 'crossing_time', 'degraded_fraction': bound.degraded_fraction,
                'quantile': bound.quantile, 'baseline_time': bound.baseline_time}
    return {'kind': 'calendar_time', 'time': bound}


def _target_text(bound):
    if isinstance(bound, CrossingTimeTarget):
        return f'{100 * bound.degraded_fraction:g}% degraded; quantile {bound.quantile:g}; baseline {bound.baseline_time:g}'
    return 'planning start' if bound is None else f'time {bound:g}'


def planning_tables(definition, benchmark, constrained):
    """Retain every requested size and requirement, including unavailable ones."""
    status_rows, requirement_rows, anchor_rows = [], [], []
    resolved = constrained.resolved_requirements
    context = resolved.context
    for code, optimization in (('central_schedule', benchmark), ('timing_schedule', constrained)):
        for design in optimization.designs:
            feasible = design.status == 'feasible'
            new = tuple(time for time in design.times if time not in context.completed_times)
            selection = design.latest_before_selection
            status_rows.append({
                'design': code, 'design_label': PLANNING_DESIGN_LABELS[code],
                'number_of_times': design.number_of_times, 'status': design.status,
                'message': design.message, 'optimization_status': design.optimization_status,
                'complete_schedule': _times(design.times) if feasible else None,
                'baseline_times': _times(context.baseline_times),
                'completed_times': _times(context.completed_times),
                'fixed_future_times': _times(time for time in definition.constraints.fixed_times
                                           if time not in context.baseline_times
                                           and time not in context.completed_times),
                'new_times': _times(new) if feasible else None,
                'number_of_new_times': len(new) if feasible else None,
                'latest_before_times': _times(selection.selected_times) if selection is not None else None,
                'requested_total_experimental_units': definition.constraints.total_experimental_units(design.number_of_times),
                'new_experimental_units': len(new) * definition.constraints.replicates_per_time if feasible else None,
                'information_criterion': design.score.criterion if feasible else None,
                'full_rank_fraction': design.score.full_rank_fraction if feasible else None,
                'information_rank': design.score.min_rank if feasible else None,
                'information_parameter_count': design.score.parameter_count if feasible else None,
                'information_selection_rule': design.metadata.get('selection_rule'),
                'feasibility_states_visited': design.feasibility_states_visited,
                'feasibility_checks': design.feasibility_checks,
            })
            if code != 'timing_schedule':
                continue
            evaluations = ({} if design.requirement_evaluation is None else
                           {row.label: row for row in design.requirement_evaluation.requirements})
            points = {} if selection is None else {point.label: point for point in selection.points}
            for row in resolved.requirements:
                evaluated = evaluations.get(row.label)
                upper = row.request.deadline if row.kind == 'latest_before' else row.request.upper_bound
                summary = row.upper_summary
                requirement_rows.append({
                    'design': code, 'number_of_times': design.number_of_times,
                    'label': row.label, 'kind': row.kind, 'schedule_status': design.status,
                    'status': (evaluated.status if evaluated is not None else
                               row.status if row.status not in ('ready', 'already_satisfied') else 'unresolved'),
                    'message': evaluated.message if evaluated is not None else row.message + ' ' + design.message,
                    'lower_target': _target_text(row.request.lower_bound),
                    'upper_target': _target_text(upper),
                    'lower_bound': row.effective_lower_bound, 'upper_bound': row.upper_bound,
                    'count_scope': row.count_scope, 'minimum_count': row.minimum_count,
                    'qualifying_times': _times(evaluated.qualifying_times if evaluated is not None else row.fixed_qualifying_times),
                    'achieved_count': evaluated.achieved_count if evaluated is not None else len(row.fixed_qualifying_times),
                    'count_basis': 'selected_schedule' if evaluated is not None else 'fixed_times_only',
                    'deadline_status': summary.status if summary is not None else 'calendar_time',
                    'reached_weight': summary.reached_weight if summary is not None else None,
                    'noncrossing_weight': summary.noncrossing_weight if summary is not None else None,
                    'invalid_weight': summary.invalid_weight if summary is not None else None,
                })
                if row.kind != 'latest_before':
                    continue
                point = points.get(row.label)
                anchor_rows.append({
                    'design': code, 'number_of_times': design.number_of_times,
                    'label': row.label, 'schedule_status': design.status,
                    'status': point.status if point is not None else 'unresolved',
                    'message': point.message if point is not None else design.message,
                    'priority': point.priority if point is not None else None,
                    'lower_bound': row.effective_lower_bound, 'deadline': row.upper_bound,
                    'selected_time': point.selected_time if point is not None else None,
                    'latest_eligible_time': point.latest_eligible_time if point is not None else None,
                    'displacement': point.displacement if point is not None else None,
                    'deadline_slack': point.deadline_slack if point is not None else None,
                })
    anchor_columns = ('design number_of_times label schedule_status status message priority lower_bound '
                      'deadline selected_time latest_eligible_time displacement deadline_slack').split()
    requirement_columns = (
        'design number_of_times label kind schedule_status status message lower_target upper_target '
        'lower_bound upper_bound count_scope minimum_count qualifying_times achieved_count count_basis '
        'deadline_status reached_weight noncrossing_weight invalid_weight'
    ).split()
    return (pd.DataFrame(status_rows), pd.DataFrame(requirement_rows, columns=requirement_columns),
            pd.DataFrame(anchor_rows, columns=anchor_columns))


def decorate_planning_times(table, definition, constrained):
    table = table.copy()
    context = constrained.resolved_requirements.context
    selections = {row.number_of_times: row.latest_before_selection for row in constrained.designs}
    for index, row in table.iterrows():
        time = float(row['time'])
        baseline = time in context.baseline_times
        completed = time in context.completed_times
        original_fixed = time in definition.constraints.fixed_times
        selection = selections.get(row['number_of_times'])
        anchor = (row['design'] == 'timing_schedule' and selection is not None
                  and time in selection.selected_times)
        future_fixed = original_fixed and not baseline and not completed
        role = ('completed baseline observation' if completed and baseline else
                'completed observation' if completed else 'planned baseline' if baseline else
                'fixed future / latest-before point' if future_fixed and anchor else
                'fixed future commitment' if future_fixed else 'latest-before point' if anchor else
                'optimized new study time')
        table.loc[index, 'time_role'] = role
        table.loc[index, 'baseline_time'] = baseline
        table.loc[index, 'fixed_future_commitment'] = future_fixed
        table.loc[index, 'latest_before_point'] = anchor
        table.loc[index, 'original_fixed'] = original_fixed
        table.loc[index, 'fixed'] = original_fixed or anchor
    for column in ('baseline_time', 'fixed_future_commitment', 'latest_before_point', 'original_fixed'):
        if column not in table:
            table[column] = pd.Series(dtype=bool)
        table[column] = table[column].astype(bool)
    return table


def empty_performance_tables():
    performance = ("scenario design design_label number_of_times number_of_existing_times number_of_new_times "
                   "existing_times new_times latest_completed_time adaptive_candidate_policy earliest_candidate_time "
                   "earliest_future_candidate_time reference_effective_sample_size reference_effective_sample_size_fraction "
                   "complete_schedule times replicates_per_time total_experimental_units new_experimental_units n_simulations "
                   "n_successful failure_rate median_irmse p90_irmse p95_irmse median_normalized_irmse p90_normalized_irmse "
                   "penalized_mean_irmse penalized_mean_normalized_irmse median_curve_interval_width median_absolute_target_time_error "
                   "median_absolute_t50_error t50_recovery_rate t50_true_reachable_rate median_absolute_t90_error t90_recovery_rate "
                   "t90_true_reachable_rate parameter_bound_hit_rate information_criterion full_rank_fraction minimum_information_rank "
                   "information_rank information_parameter_count information_selection_rule information_selection_rule_label "
                   "mean_log_pseudodeterminant marginal_penalized_nirmse_improvement").split()
    times = ('scenario design design_label number_of_times order time time_role '
             'existing_observation new_study_time fixed').split()
    return pd.DataFrame(columns=performance), pd.DataFrame(columns=times)


def planning_metadata(result):
    spec = result.definition.planning
    resolved = result.planning_optimization.resolved_requirements
    requirements = []
    def summary_metadata(summary):
        if summary is None:
            return None
        return {key: getattr(summary, key) for key in (
            'degraded_fraction', 'quantile', 'baseline_time', 'deadline', 'status',
            'message', 'reached_weight', 'noncrossing_weight', 'invalid_weight',
        )}
    for row in resolved.requirements:
        upper = row.request.deadline if row.kind == 'latest_before' else row.request.upper_bound
        requirements.append({
            'label': row.label, 'kind': row.kind, 'lower_target': _bound(row.request.lower_bound),
            'upper_target': _bound(upper), 'lower_bound': row.effective_lower_bound,
            'upper_bound': row.upper_bound, 'count_scope': row.count_scope,
            'minimum_count': row.minimum_count, 'status': row.status, 'message': row.message,
            'lower_crossing_summary': summary_metadata(row.lower_summary),
            'upper_crossing_summary': summary_metadata(row.upper_summary),
        })
    comparisons = tuple(result.comparisons_by_n.values())
    return {
        'input_mode': spec.input_mode, 'planning_start': resolved.context.planning_start,
        'baseline_times': resolved.context.baseline_times,
        'completed_times': resolved.context.completed_times,
        'requirements': requirements, 'resolution_status': resolved.status,
        'max_anchor_search_states_per_size': spec.search_limit,
        'max_search_states_per_information_check': spec.search_limit,
        'latest_before_priority': 'later_deadlines_first_then_request_order',
        'independent_final_sizes': True,
        'global_information_optimum_verified': False,
        'central_benchmark_applies_timing_requirements': False,
        'central_analysis_parameters': dict(zip(result.reference_model.parameter_names,
                                                result.reference_model.central_parameters)),
        'information_evaluation_times': result.planning_optimization.evaluation_times,
        'information_evaluation_weights': result.planning_optimization.evaluation_weights,
        'monte_carlo': {
            'simulated_schedule_count': sum(len(comparison.results) for comparison in comparisons),
            'shared_simulation_plan': (all(comparison.simulation_plan is comparisons[0].simulation_plan
                                           for comparison in comparisons) if comparisons else None),
        },
        'schedule_statuses': result.schedule_status_table.to_dict(orient='records'),
        'requirement_results': result.requirement_table.to_dict(orient='records'),
        'latest_before_results': result.anchor_table.to_dict(orient='records'),
    }


def _html_table(table, columns=None):
    table = table.copy() if columns is None else table[columns].copy()
    if 'design' in table:
        table = table.drop(columns='design')
    for column in ('kind', 'status', 'schedule_status', 'optimization_status', 'count_basis', 'count_scope', 'deadline_status'):
        if column in table:
            table[column] = table[column].map(
                lambda value: value.replace('_', ' ') if isinstance(value, str) else value
            )
    table = table.rename(columns=lambda column: column.replace('_', ' ').capitalize())
    return '<div style="overflow-x:auto">' + table.fillna('—').to_html(
        index=False, escape=True, float_format=lambda value: f'{value:.5g}',
    ) + '</div>'


def planning_summary_html(result, reference_table, go, to_html, *, reference_summary_html=''):
    """Show timing requirements, complete schedules, and paired MC recovery."""
    definition = result.definition
    resolved = result.planning_optimization.resolved_requirements
    plot = go.Figure()
    x = reference_table['time']
    plot.add_trace(go.Scatter(x=x, y=reference_table['lower_95'], mode='lines', line={'width': 0}, showlegend=False))
    plot.add_trace(go.Scatter(x=x, y=reference_table['upper_95'], mode='lines', fill='tonexty',
                             fillcolor='rgba(80,120,200,0.18)', line={'width': 0}, name='95% reference envelope'))
    plot.add_trace(go.Scatter(x=x, y=reference_table['central_response'], mode='lines', name='Central reference'))
    if 'ground_truth_response' in reference_table:
        plot.add_trace(go.Scatter(x=x, y=reference_table['ground_truth_response'], mode='lines',
                                 line={'dash': 'dash'}, name='Synthetic ground truth — not used by optimizer'))
    existing = definition.existing_observations
    if existing is not None:
        times, means = existing.mean_response_by_time()
        plot.add_trace(go.Scatter(x=times, y=means, mode='markers', name='Completed target-study observations'))
    for row in resolved.requirements:
        if row.upper_bound is None:
            continue
        if row.kind == 'window' and row.effective_lower_bound is not None:
            left = max(row.effective_lower_bound, float(x.iloc[0]))
            right = min(row.upper_bound, float(x.iloc[-1]))
            if left < right:
                plot.add_vrect(x0=left, x1=right, fillcolor='green', opacity=0.07, line_width=0)
        elif row.kind == 'latest_before' and x.iloc[0] <= row.upper_bound <= x.iloc[-1]:
            plot.add_vline(x=row.upper_bound, line_dash='dot', line_color='orange')
    for (size, code), group in result.selected_times_table.groupby(['number_of_times', 'design'], sort=False):
        selected = group['time'].to_numpy(dtype=float)
        y = np.interp(selected, reference_table['time'], reference_table['central_response'])
        symbols = ['diamond' if value else 'circle' for value in group['latest_before_point']]
        plot.add_trace(go.Scatter(x=selected, y=y, mode='markers',
                                 marker={'symbol': symbols, 'size': 9},
                                 text=group['time_role'], name=f'{PLANNING_DESIGN_LABELS[code]}, N={size}'))
    plot.update_layout(title='Reference, coverage windows, deadlines, and selected study times',
                       xaxis_title='Time', yaxis_title='Response', template='plotly_white')
    recovery = go.Figure()
    for code, group in result.performance_table.groupby('design', sort=False):
        recovery.add_trace(go.Scatter(x=group['number_of_times'], y=group['penalized_mean_normalized_irmse'],
                                     mode='lines+markers', name=PLANNING_DESIGN_LABELS[code]))
    recovery.update_layout(title='Monte Carlo curve recovery by total study size', xaxis_title='Total distinct times',
                           yaxis_title='Penalized mean normalized IRMSE', template='plotly_white')
    first = to_html(plot, full_html=False, include_plotlyjs=True)
    second = (to_html(recovery, full_html=False, include_plotlyjs=False)
              if not result.performance_table.empty else '<p>No feasible schedules were simulated.</p>')
    statuses = _html_table(result.schedule_status_table)
    requirements = _html_table(result.requirement_table)
    anchors = _html_table(result.anchor_table)
    performance = _html_table(result.performance_table, [
        'design_label', 'number_of_times', 'new_times', 'penalized_mean_normalized_irmse',
        'p90_normalized_irmse', 'failure_rate', 'median_absolute_t50_error', 'median_absolute_t90_error',
    ])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{escape(definition.title)}</title></head>
<body style="font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto">
<h1>{escape(definition.title)}</h1><p>{escape(definition.description)}</p>
<p><strong>Input mode:</strong> {escape(definition.planning.input_mode)};
<strong>analysis model:</strong> {escape(definition.analysis_model)};
<strong>completed target-study times:</strong> {_times(resolved.context.completed_times) or 'none'};
<strong>baseline times:</strong> {_times(resolved.context.baseline_times)}.</p>
{reference_summary_html}
<p>Baseline does not count toward milestone coverage. Historical and calibration data provide the reference;
completed target-study observations remain fixed. New sampling includes fixed future commitments.</p>
<p>Green shading marks coverage windows; dotted lines mark latest-before deadlines; diamonds mark reserved points.
Later deadlines have priority, with request order breaking ties. Remaining times use central I-optimality over
the original evaluation horizon. Each final size is planned independently. Planned-time markers show the
central reference response; the information search is heuristic.</p>
<p>The unrestricted benchmark retains fixed times, spacing, and future eligibility but does not apply milestone
requirements. Reference uncertainty determines crossing-time
bounds and Monte Carlo trajectories; it is not averaged into the information criterion. Crossing weights describe
reference uncertainty, not population event rates. The envelope is pointwise; crossing targets use complete
trajectories. Synthetic ground truth is displayed for evaluation only.</p>
{first}<h2>Schedule status</h2>{statuses}<h2>Timing requirements</h2>
<p>Windows use (lower, upper]. Counts for unavailable schedules refer only to fixed times.
All requested sizes and requirements remain visible; only feasible schedules are simulated.</p>{requirements}
<h2>Latest-before selections</h2><p>Displacement is measured from the latest grid time eligible under the original
fixed times, before competing requirements and final study size are considered. Slack uses the actual deadline.</p>{anchors}
{second}<h2>Monte Carlo performance summary</h2>
<p>Feasible comparisons share reference draws and observation errors. No synthetic performance row is added for
an unavailable schedule.</p>{performance}</body></html>'''
