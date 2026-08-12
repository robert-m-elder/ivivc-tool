import os
import traceback
import warnings
from html import escape
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy as sp
import plotly.graph_objects as go
import plotly.io as pio
import uncertainties as unc

from metrics import calculate_metric, metrics
from models import approaches, models
from preprocessing import preprocessing_options
from utilities.evaluation import (
    auto_grid_search,
    calculate_tau_with_uncertainty,
    cross_validation_curve_fit,
    generate_prediction_bands,
    generate_ratio_prediction_bands,
    make_cross_validator,
)
from utilities.misc import hex_to_rgba
from utilities.parameter_diagnostics import (
    build_parameter_diagnostics_html,
    extract_parameter_diagnostic_badges_html,
)
from utilities.prediction_validity import describe_tau_prediction_skip_reason

# Determine environment
IS_PRODUCTION = 'PYTHONANYWHERE_DOMAIN' in os.environ
if IS_PRODUCTION:
    num_cores_for_grid_search = 1
    num_points_for_grid_search = 10
else:
    num_cores_for_grid_search = None
    num_points_for_grid_search = 200


def _model_initial_points(model_name, x, y):
    """Return optional model-specific/data-driven starting points."""
    initial_guess = models[model_name].get('initial_guess')
    if initial_guess is None:
        return None
    return initial_guess(x, y)

colors = {
    'in_vitro': '#2E5F8A',      # Darker blue
    'in_vivo': '#8B3A6B',       # Darker purple-red
    'in_vitro_2': '#006FA6',    # Accessible prediction blue
    'in_vivo_2': '#A52A6A'      # Accessible prediction purple-red
}


def _apply_accessible_plot_styles(fig):
    """Add non-color distinctions and visible outlines to Plotly series."""
    for trace in fig.data:
        name = str(getattr(trace, 'name', '') or '')
        name_lower = name.lower()
        mode = str(getattr(trace, 'mode', '') or '')

        if 'markers' in mode:
            if 'in vivo' in name_lower and 'predicted' in name_lower:
                symbol = 'triangle-up'
            elif 'in vitro' in name_lower and ('input' in name_lower or 'prediction' in name_lower):
                symbol = 'square'
            elif 'in vivo' in name_lower and 'interpolated' in name_lower:
                symbol = 'diamond-open'
            elif 'in vitro' in name_lower and 'interpolated' in name_lower:
                symbol = 'circle-open'
            elif 'in vivo' in name_lower:
                symbol = 'diamond'
            elif 'in vitro' in name_lower:
                symbol = 'circle'
            else:
                symbol = getattr(getattr(trace, 'marker', None), 'symbol', None) or 'circle'

            trace.update(
                marker_symbol=symbol,
                marker_line_color='#1f1f1f',
                marker_line_width=2.5 if 'interpolated' in name_lower else 1.5,
            )

        if 'lines' in mode:
            line = getattr(trace, 'line', None)
            line_width = getattr(line, 'width', None) if line is not None else None
            if line_width == 0:
                continue
            if 'in vitro' in name_lower and ('fit' in name_lower or 'fitted model' in name_lower):
                trace.update(line_dash='solid')
            elif 'in vivo' in name_lower and ('fit' in name_lower or 'fitted model' in name_lower):
                trace.update(line_dash='dash')


CV_COMPARISON_EXCLUDED_METRICS = {'adjusted_r_squared', 'aic', 'aicc', 'bic', 'nrmse', 'mnrmse'}


UNCERTAINTY_RANGE_MATERIAL_MARGIN_SPANS = 1.0
UNCERTAINTY_RANGE_FAR_MARGIN_SPANS = 5.0
UNCERTAINTY_RANGE_FRACTION_THRESHOLD = 0.20


def _uncertainty_unavailable_warning(series_label=None, reason='calculation_failed'):
    """Return user-facing metadata when a requested uncertainty band is unavailable."""
    prefix = f'{series_label}: ' if series_label else ''
    if reason == 'covariance_unavailable':
        detail = f'{prefix}the fitted parameter covariance matrix was unavailable.'
    else:
        detail = f'{prefix}the uncertainty-band calculation did not produce displayable bounds.'
    return {
        'code': 'invalid',
        'level': 'warning',
        'badge_label': 'Some uncertainty bands unavailable',
        'message': (
            'One or more uncertainty bands could not be calculated or displayed. '
            'Review the fitted parameter diagnostics.'
        ),
        'detail': detail,
        'reason': reason,
    }


def _assess_uncertainty_band_visibility(lower, upper, axis_range, series_label=None):
    """Identify nonfinite or materially off-scale uncertainty intervals.

    The initial fitting-plot range is intentionally based on the observed data and
    fitted curve rather than the uncertainty band. This helper flags intervals that
    may therefore be invisible at the initial scale without changing that scale.
    """
    lower_values = np.asarray(lower, dtype=float).reshape(-1)
    upper_values = np.asarray(upper, dtype=float).reshape(-1)
    n_total = min(lower_values.size, upper_values.size)
    if n_total == 0:
        return [_uncertainty_unavailable_warning(series_label)]

    lower_values = lower_values[:n_total]
    upper_values = upper_values[:n_total]
    finite_mask = np.isfinite(lower_values) & np.isfinite(upper_values)
    n_invalid = int(np.count_nonzero(~finite_mask))
    warnings_out = []
    prefix = f'{series_label}: ' if series_label else ''

    if n_invalid:
        warnings_out.append({
            'code': 'invalid',
            'level': 'warning',
            'badge_label': 'Some uncertainty bands unavailable',
            'message': (
                'One or more calculated uncertainty bounds were nonfinite and could not be displayed. '
                'Review the fitted parameter diagnostics.'
            ),
            'detail': f'{prefix}{n_invalid} of {n_total} intervals had nonfinite bounds.',
            'reason': 'nonfinite_bounds',
            'n_total': n_total,
            'n_invalid': n_invalid,
        })

    n_finite = int(np.count_nonzero(finite_mask))
    if n_finite == 0:
        return warnings_out

    try:
        axis_min, axis_max = (float(axis_range[0]), float(axis_range[1]))
    except (TypeError, ValueError, IndexError):
        return warnings_out

    axis_span = axis_max - axis_min
    if not np.isfinite(axis_span) or axis_span <= 0:
        return warnings_out

    finite_lower = lower_values[finite_mask]
    finite_upper = upper_values[finite_mask]
    material_margin = UNCERTAINTY_RANGE_MATERIAL_MARGIN_SPANS * axis_span
    far_margin = UNCERTAINTY_RANGE_FAR_MARGIN_SPANS * axis_span

    materially_outside = (
        (finite_lower < axis_min - material_margin)
        | (finite_upper > axis_max + material_margin)
    )
    far_outside = (
        (finite_lower < axis_min - far_margin)
        | (finite_upper > axis_max + far_margin)
    )
    n_outside = int(np.count_nonzero(materially_outside))
    outside_fraction = n_outside / n_finite

    if np.any(far_outside) or outside_fraction >= UNCERTAINTY_RANGE_FRACTION_THRESHOLD:
        warnings_out.append({
            'code': 'outside_initial_range',
            'level': 'info',
            'badge_label': 'Large uncertainty outside plot range',
            'message': (
                'Some uncertainty bands extend beyond the initial plot range and may not be visible. '
                'Zoom out or autoscale to view them. Review the fitted parameter diagnostics.'
            ),
            'detail': (
                f'{prefix}{n_outside} of {n_finite} finite intervals extend materially '
                'beyond the initial y-axis range.'
            ),
            'reason': 'outside_initial_range',
            'n_total': n_total,
            'n_finite': n_finite,
            'n_outside': n_outside,
        })

    return warnings_out


def _merge_uncertainty_warnings(warnings_in):
    """Combine repeated warning types while retaining dataset-specific details."""
    merged = []
    by_code = {}
    for warning in warnings_in:
        code = warning.get('code', 'unknown')
        if code not in by_code:
            item = dict(warning)
            item['_details'] = [warning.get('detail')] if warning.get('detail') else []
            by_code[code] = item
            merged.append(item)
            continue

        item = by_code[code]
        if warning.get('detail'):
            item['_details'].append(warning['detail'])
        for key in ('n_total', 'n_finite', 'n_invalid', 'n_outside'):
            if key in warning:
                item[key] = int(item.get(key, 0)) + int(warning[key])

    for item in merged:
        details = item.pop('_details', [])
        item['detail'] = ' '.join(details)
    return merged


def _not_meaningful_badge():
    return '<span class="evidence-badge evidence-badge-neutral">Not meaningful</span>'


def _goodness_cv_comparison_table(gof, cvs_mean):
    """Build the final-model vs CV table while preserving final fitted metrics.

    Adjusted R² and information criteria are shown for the final model, but their
    CV values and ratios are intentionally not shown because the final model and
    CV models are fitted to different data subsets.
    """
    displayed_metrics = [metric for metric in gof if metric in metrics]

    if not displayed_metrics:
        return '<p class="evidence-note">Goodness-of-fit results are not available for the selected metrics.</p>'

    final_row = {}
    cv_row = {}
    ratio_row = {}
    comparison_row = {}
    for metric in displayed_metrics:
        display_name = metrics[metric]['display_name']
        fit_value = gof[metric]
        final_row[display_name] = fit_value

        if metric in CV_COMPARISON_EXCLUDED_METRICS:
            cv_row[display_name] = 'Not shown'
            ratio_row[display_name] = 'Not shown'
            comparison_row[display_name] = _not_meaningful_badge()
            continue

        cv_value = cvs_mean.get(metric, np.nan)
        cv_row[display_name] = _finite_float(cv_value)
        ratio_row[display_name] = _fit_cv_ratio(fit_value, cv_value)
        comparison_row[display_name] = _cv_status_badge(metric, fit_value, cv_value)[0]

    stats_table = pd.DataFrame(
        [final_row, cv_row, ratio_row, comparison_row],
        index=['Final model', 'Cross-validation', 'Final/CV ratio', 'CV comparison']
    )

    return stats_table.to_html(
        classes='table table-striped',
        index=True,
        float_format=lambda x: f'{x:.4f}',
        na_rep='N/A',
        escape=False
    )



def _ci_multiplier_95(n_obs, n_params):
    """Return a two-sided 95% t multiplier for approximate parameter intervals."""
    try:
        dof = int(n_obs) - int(n_params)
    except (TypeError, ValueError):
        dof = 0
    if dof > 0:
        multiplier = sp.stats.t.ppf(1 - 0.05 / 2, dof)
        if np.isfinite(multiplier):
            return float(multiplier)
    return 1.96


def _format_interval_number(value):
    value = _finite_float(value)
    return f'{value:.4f}' if np.isfinite(value) else 'N/A'


def _interval_rows_from_uparams(uparams, multiplier, confidence_label='95% CI'):
    """Format uncertainty-aware values as estimate plus/minus CI value."""
    rows = []
    for name, value in uparams.items():
        estimate = _finite_float(getattr(value, 'n', np.nan))
        std_uncertainty = _finite_float(getattr(value, 's', np.nan))
        row = {
            'name': name,
            'estimate': _format_interval_number(estimate),
            'confidence_label': confidence_label,
            'is_available': bool(np.isfinite(estimate) and np.isfinite(std_uncertainty) and std_uncertainty >= 0),
        }
        if row['is_available']:
            half_width = multiplier * std_uncertainty
            row['half_width'] = _format_interval_number(half_width)
        else:
            row['half_width'] = 'N/A'
        rows.append(row)
    return rows


def _tau_interval_row(tau_value, multiplier, label='Tau'):
    """Format a propagated tau value as estimate plus/minus CI value."""
    estimate = _finite_float(getattr(tau_value, 'n', np.nan))
    std_uncertainty = _finite_float(getattr(tau_value, 's', np.nan))
    is_available = bool(estimate > 0 and np.isfinite(std_uncertainty) and std_uncertainty >= 0)
    row = {
        'label': label,
        'estimate': _format_interval_number(estimate),
        'confidence_label': '95% CI',
        'is_available': is_available,
    }
    if is_available:
        half_width = multiplier * std_uncertainty
        row['half_width'] = _format_interval_number(half_width)
    else:
        row['half_width'] = 'N/A'
    return row

def preprocess_data(t1, m1, t2, m2, selected_interpolation=None, selected_scalings=None, selected_normalizations=None):
    selected_interpolation = selected_interpolation or []
    selected_scalings = selected_scalings or []
    selected_normalizations = selected_normalizations or []

    if (t2 is not None) and (m2 is not None):
        two_datasets = True
    else:
        two_datasets = False
    # Apply basic data cleaning
    # set zero to small value to avoid division errors
    t1[t1==0] = 1e-2; 
    if two_datasets:
        t2[t2==0] = 1e-2
    # remove missing values due to unequal number of points in spreadsheet
    mask = ~pd.isna(m1); t1,m1 = t1[mask], m1[mask]
    if two_datasets:
        mask = ~pd.isna(m2); t2,m2 = t2[mask], m2[mask]

    # Apply normalizations
    for norm in selected_normalizations:
        m1 = preprocessing_options['normalization'][norm](m1)
        if two_datasets:
            m2 = preprocessing_options['normalization'][norm](m2)

    # Apply scalings
    for scale in selected_scalings:
        if scale == 'log_x':
            t1 = preprocessing_options['scaling'][scale](t1)
            if two_datasets:
                t2 = preprocessing_options['scaling'][scale](t2)
        elif scale == 'log_y':
            m1 = preprocessing_options['scaling'][scale](m1)
            if two_datasets:
                m2 = preprocessing_options['scaling'][scale](m2)

    # Apply interpolation/alignment. If no interpolation option is supplied,
    # use exact raw alignment rather than failing with an unset data variable.
    if two_datasets:
        interpolation_choices = selected_interpolation or ['none']
        for interp in interpolation_choices:
            data = preprocessing_options['interpolation'][interp](t1, m1, t2, m2)
    else:
        data = [t1, m1]

    return data

def process_data(data, selected_models, selected_approaches, selected_metrics, analysis_config=None):
    analysis_config = analysis_config or {}
    grid_search_num_points = int(analysis_config.get('grid_search_num_points', num_points_for_grid_search))
    grid_search_num_cores = analysis_config.get('grid_search_num_cores', num_cores_for_grid_search)
    grid_search_param_min = float(analysis_config.get('grid_search_param_min', -1e6))
    grid_search_param_max = float(analysis_config.get('grid_search_param_max', 1e6))
    grid_search_random_state = analysis_config.get('grid_search_random_state', 12345)
    include_cv_fit_variability = bool(analysis_config.get('include_cv_fit_variability_plot', False))
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    
    results = {}
    
    for model_key in selected_models:
        if len(model_key.split(':'))>1:
            approach_id, model_name = model_key.split(':')
        else:
            approach_id = model_key
        
        try:
            if approach_id == 'approach1':
                cv = make_cross_validator(analysis_config, approach_id)
                # process data for this approach
                m = ~np.isnan(ti1) & ~np.isnan(ti2)
                x,y = ti1[m], ti2[m]
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x, y, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state,
                                          initial_points=_model_initial_points(model_name, x, y))
                # cross-validation
                cv_output = cross_validation_curve_fit(
                    x, y, models[model_name]['model_function'], cv, selected_metrics,
                    p0=p0, kwargs=models[model_name]['fit_kwargs'],
                    return_fit_params=include_cv_fit_variability,
                )
                if include_cv_fit_variability:
                    cvs, cv_fit_params = cv_output
                else:
                    cvs, cv_fit_params = cv_output, []
                cvs_mean = {f'{metric}':np.nanmean(values[np.isfinite(values)]) for metric,values in cvs.items()}
                # fit final model
                model_result = models[model_name]['fit_model'](x, y, p0=p0)
                y_pred = model_result['predict'](x)
                # Evaluate goodness of fit
                gof = {metric: calculate_metric(metric, y, y_pred, len(p0)) for metric in selected_metrics}
                stats = pd.concat([pd.DataFrame(gof, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs_mean.items()}, index=[0])], axis=1)
                stats_table = _goodness_cv_comparison_table(gof, cvs_mean)
                popt, pcov = list(model_result['params'].values()), model_result['pcov']
                upopt = unc.correlated_values(popt, pcov)
                upopt = dict(zip(model_result['params'].keys(),upopt))
                parameter_ci_multiplier = _ci_multiplier_95(len(y), len(popt))
                parameter_intervals = _interval_rows_from_uparams(upopt, parameter_ci_multiplier)
                # outputs
                plot_image = create_plotly_a1(x, y, {
                    'model_name': models[model_name]['display_name'],
                    'model_function': models[model_name]['model_function'],
                    'params': model_result['params'],
                    'pcov': model_result['pcov'],
                    'residuals': y-y_pred,
                    'approach': approaches[approach_id]['display_name']
                })
                residuals = y - y_pred
                results[model_key] = {
                    'params': model_result['params'],
                    'uparams': upopt,
                    'parameter_intervals': parameter_intervals,
                    'pcov': model_result['pcov'],
                    'residuals': residuals,
                    'residual_x': x,
                    'residual_y': y,
                    'cv_fit_params': cv_fit_params,
                    'stats': stats,
                    'predictions': y_pred,
                    'evidence_data': {
                        'fit': {
                            'n': int(len(y)),
                            'rss': float(np.sum(residuals ** 2)),
                            'n_params': int(len(model_result['params']))
                        }
                    },
                    'stats_table': stats_table,
                    'parameter_diagnostics_html': build_parameter_diagnostics_html(model_result['params'], model_result['pcov']),
                    'plot': plot_image
                }
            elif approach_id == 'approach2':
                cv = make_cross_validator(analysis_config, approach_id)
                # Approach 2 fits the two preprocessed datasets independently.
                # No interpolation/alignment is needed because the datasets do
                # not need matched times or matched response values for separate
                # curve fitting. Using only observed/preprocessed points also
                # avoids treating deterministic interpolation estimates as
                # additional fitting observations.
                m = ~np.isnan(t1) & ~np.isnan(m1)
                x1,y1 = t1[m], m1[m]
                m = ~np.isnan(t2) & ~np.isnan(m2)
                x2,y2 = t2[m], m2[m]
                # Dataset 1
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x1, y1, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state,
                                          initial_points=_model_initial_points(model_name, x1, y1))
                # cross-validation
                cv_output1 = cross_validation_curve_fit(
                    x1, y1, models[model_name]['model_function'], cv, selected_metrics,
                    p0=p0, kwargs=models[model_name]['fit_kwargs'],
                    return_fit_params=include_cv_fit_variability,
                )
                if include_cv_fit_variability:
                    cvs1, cv_fit_params1 = cv_output1
                else:
                    cvs1, cv_fit_params1 = cv_output1, []
                cvs1_mean = {f'{metric}':np.nanmean(v[np.isfinite(v)]) for metric,v in cvs1.items()}
                # fit final model
                model_result1 = models[model_name]['fit_model'](x1, y1, p0=p0)
                y_pred1 = model_result1['predict'](x1)
                # Evaluate goodness of fit
                gof1 = {metric: calculate_metric(metric, y1, y_pred1, len(p0)) for metric in selected_metrics}
                stats1 = pd.concat([pd.DataFrame(gof1, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs1_mean.items()}, index=[0])], axis=1)
                stats_table1 = _goodness_cv_comparison_table(gof1, cvs1_mean)
                popt1, pcov1 = list(model_result1['params'].values()), model_result1['pcov']
                upopt1 = unc.correlated_values(popt1, pcov1)
                upopt1 = dict(zip(model_result1['params'].keys(),upopt1))
                parameter_ci_multiplier1 = _ci_multiplier_95(len(y1), len(popt1))
                parameter_intervals1 = _interval_rows_from_uparams(upopt1, parameter_ci_multiplier1)
                tau1 = calculate_tau_with_uncertainty(models[model_name]['model_function'], popt1, pcov1)
                # deal with integration failure
                if tau1[0]>0 and np.isfinite(tau1[0]):
                    tau1 = unc.ufloat(*tau1)
                else:
                    tau1 = unc.ufloat(0,1)
                tau_interval1 = _tau_interval_row(tau1, parameter_ci_multiplier1)
                # Dataset 2
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x2, y2, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state,
                                          initial_points=_model_initial_points(model_name, x2, y2))
                # cross-validation
                cv_output2 = cross_validation_curve_fit(
                    x2, y2, models[model_name]['model_function'], cv, selected_metrics,
                    p0=p0, kwargs=models[model_name]['fit_kwargs'],
                    return_fit_params=include_cv_fit_variability,
                )
                if include_cv_fit_variability:
                    cvs2, cv_fit_params2 = cv_output2
                else:
                    cvs2, cv_fit_params2 = cv_output2, []
                cvs2_mean = {f'{metric}':np.nanmean(v[np.isfinite(v)]) for metric,v in cvs2.items()}
                # fit final model
                model_result2 = models[model_name]['fit_model'](x2, y2, p0=p0)
                y_pred2 = model_result2['predict'](x2)
                # Evaluate goodness of fit
                gof2 = {metric: calculate_metric(metric, y2, y_pred2, len(p0)) for metric in selected_metrics}
                stats2 = pd.concat([pd.DataFrame(gof2, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs2_mean.items()}, index=[0])], axis=1)
                stats_table2 = _goodness_cv_comparison_table(gof2, cvs2_mean)
                popt2, pcov2 = list(model_result2['params'].values()), model_result2['pcov']
                upopt2 = unc.correlated_values(popt2, pcov2)
                upopt2 = dict(zip(model_result2['params'].keys(),upopt2))
                parameter_ci_multiplier2 = _ci_multiplier_95(len(y2), len(popt2))
                parameter_intervals2 = _interval_rows_from_uparams(upopt2, parameter_ci_multiplier2)
                tau2 = calculate_tau_with_uncertainty(models[model_name]['model_function'], popt2, pcov2)
                # deal with integration failure
                if tau2[0]>0 and np.isfinite(tau2[0]):
                    tau2 = unc.ufloat(*tau2)
                else:
                    tau2 = unc.ufloat(0,1)
                tau_interval2 = _tau_interval_row(tau2, parameter_ci_multiplier2)
                # outputs
                plot_image = create_plotly_a2(x1, y1, x2, y2, {
                    'model_name': models[model_name]['display_name'],
                    'model_function': models[model_name]['model_function'],
                    'params1': model_result1['params'],
                    'params2': model_result2['params'],
                    'pcov1': model_result1['pcov'],  # Add this line
                    'pcov2': model_result2['pcov'],  # Add this line
                    'residuals1': y1-y_pred1,
                    'residuals2': y2-y_pred2,
                    'approach': approaches[approach_id]['display_name']
                })
                residuals1 = y1 - y_pred1
                residuals2 = y2 - y_pred2
                results[model_key] = {
                    'params': [model_result1['params'],model_result2['params']],
                    'uparams': [upopt1,upopt2],
                    'parameter_intervals': [parameter_intervals1, parameter_intervals2],
                    'tau': [tau1,tau2],
                    'tau_intervals': [tau_interval1, tau_interval2],
                    'stats': [stats1,stats2],
                    'predictions': [y_pred1,y_pred2],
                    'pcov': [pcov1,pcov2],
                    'residuals': [residuals1,residuals2],
                    'residual_x': [x1, x2],
                    'residual_y': [y1, y2],
                    'cv_fit_params': [cv_fit_params1, cv_fit_params2],
                    'residual_dof': [
                        int(len(y1) - len(model_result1['params'])),
                        int(len(y2) - len(model_result2['params']))
                    ],
                    'evidence_data': {
                        'in_vitro': {
                            'n': int(len(y1)),
                            'rss': float(np.sum(residuals1 ** 2)),
                            'n_params': int(len(model_result1['params']))
                        },
                        'in_vivo': {
                            'n': int(len(y2)),
                            'rss': float(np.sum(residuals2 ** 2)),
                            'n_params': int(len(model_result2['params']))
                        }
                    },
                    'stats_table': [stats_table1, stats_table2],
                    'parameter_diagnostics_html': [
                        build_parameter_diagnostics_html(model_result1['params'], pcov1),
                        build_parameter_diagnostics_html(model_result2['params'], pcov2)
                    ],
                    'plot': plot_image
                }
            elif approach_id == 'approach3':
                plot_image_v = create_plotly_a3(tt,mi1,mi2,ptype='v')
                plot_image_t = create_plotly_a3(mm,ti1,ti2,ptype='t')
                results[model_key] = {'plot_v': plot_image_v, 'plot_t': plot_image_t}
        except Exception as e:
            # If an error occurs, store the error message
            results[model_key] = {'error': str(e)}
            traceback.print_exc()
    
    _attach_cv_fit_variability_plots(results, analysis_config=analysis_config)
    _attach_residual_plots(results, analysis_config=analysis_config)
    return results

def process_predictions(data, prediction_data, results, selected_approaches, interpolated_data=None):
    """Process predictions using fitted models on new dataset"""
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    t_pred, m_pred = prediction_data
    prediction_results = {}
    
    for approach in selected_approaches:
        approach_predictions = {}
        
        for model_key, model_results in results.items():
            if len(model_key.split(':')) > 1:
                approach_id, model_name = model_key.split(':')
            else:
                approach_id = model_key
                
            if approach_id == approach and 'error' not in model_results:
                try:
                    if approach == 'approach1':
                        # Approach 1: Model predicts in vivo time as function of in vitro time
                        # Apply model to scale prediction (in vitro) time to find corresponding in vivo time
                        params = model_results['params']
                        t_pred_vivo = models[model_name]['model_function'](t_pred, **params)
                        
                        # Create prediction plot
                        plot_image = create_prediction_plot_a1(data, t_pred, m_pred, t_pred_vivo, {
                            'model_name': models[model_name]['display_name'],
                            'approach': approaches[approach_id]['display_name'],
                            'model_function': models[model_name]['model_function'],
                            'params': model_results['params'],
                            'pcov': model_results['pcov'],
                            'residuals': model_results['residuals']

                        })
                        
                        approach_predictions[model_name] = {
                            'predictions': {'in_vitro_time': t_pred, 'in_vitro_value': m_pred, 'predicted_in_vivo_time': t_pred_vivo},
                            'plot': plot_image
                        }
                        
                    elif approach == 'approach2':
                        # Approach 2: Independent models fitted to dataset 1 (in vitro) and dataset 2 (in vivo)
                        # Predicted in vivo value = prediction in vitro value * (dataset2_model / dataset1_model)
                        params1, params2 = model_results['params']  # params1: in vitro, params2: in vivo
                        
                        if 1:
                            ## Rescale value by value ratio -- can produce some odd results, such as when both values are near zero
                            # Calculate model predictions at prediction times
                            model1_pred = models[model_name]['model_function'](t_pred, **params1)  # in vitro model
                            model2_pred = models[model_name]['model_function'](t_pred, **params2)  # in vivo model

                            # Calculate model predictions for plotting
                            t_pred_plot = np.linspace(min(t_pred), max(t_pred), max(len(t_pred),50))
                            model1_pred_plot = models[model_name]['model_function'](t_pred_plot, **params1)  # in vitro model
                            model2_pred_plot = models[model_name]['model_function'](t_pred_plot, **params2)  # in vitro model
                            
                            # Calculate ratio and apply to prediction values
                            ratio = model2_pred / model1_pred
                            m_pred_vivo = m_pred * ratio
                            
                            # Create prediction plot
                            plot_image = create_prediction_plot_a2(data, t_pred, m_pred, m_pred_vivo, t_pred_plot, model1_pred_plot, model2_pred_plot, {
                                'model_name': models[model_name]['display_name'],
                                'approach': approaches[approach_id]['display_name'],
                                'model_function': models[model_name]['model_function'],
                                'params': model_results['params'],
                                'pcov': model_results['pcov'],
                                'residuals': model_results['residuals']
                            })

                            ## Rescale by tau ratio, if both tau values are valid
                            tau_skip_reason = describe_tau_prediction_skip_reason(model_results.get('tau'))
                            model_results.setdefault('prediction_skip_reasons', {})['tau_ratio'] = tau_skip_reason
                            if tau_skip_reason is None:
                                try:
                                    tau1,tau2 = model_results['tau']
                                    time_ratio = tau2/tau1 # uncertainties handles error propagation
                                    t_pred_vivo_tau_unc = t_pred * time_ratio
                                    t_pred_vivo_tau = np.array([t.n for t in t_pred_vivo_tau_unc])
                                    t_pred_vivo_err_tau = np.array([t.s for t in t_pred_vivo_tau_unc])
                                    ### Convert to error bar
                                    residual_dof = model_results.get('residual_dof', [])
                                    valid_dof = [int(value) for value in residual_dof if value is not None and int(value) > 0]
                                    dof = min(valid_dof) if valid_dof else 0
                                    t_value = sp.stats.t.ppf(1-0.05/2, dof) if dof > 0 else 1.96
                                    t_pred_vivo_err_tau = t_value * t_pred_vivo_err_tau
                                    # Create prediction plot
                                    plot_image_tau = create_prediction_plot_a2(data, t_pred, m_pred, t_pred_vivo_tau, t_pred_plot, model1_pred_plot, model2_pred_plot, {
                                        'model_name': models[model_name]['display_name'],
                                        'approach': approaches[approach_id]['display_name'],
                                        'model_function': models[model_name]['model_function'],
                                    }, t_pred_vivo_err_tau)
                                except Exception as e:
                                    tau_skip_reason = f'Time-constant-ratio rescaling was not performed because plot generation failed: {e}'
                                    model_results['prediction_skip_reasons']['tau_ratio'] = tau_skip_reason
                                    print(e)
                                    t_pred_vivo_tau = None
                                    plot_image_tau = None
                            else:
                                t_pred_vivo_tau = None
                                plot_image_tau = None

                        approach_predictions[model_name] = {
                            'plot': plot_image,
                            'plot_tau': plot_image_tau,
                            'plot_tau_skip_reason': model_results.get('prediction_skip_reasons', {}).get('tau_ratio')
                        }
                        
                    elif approach == 'approach3' and interpolated_data is not None:
                        # Extract interpolated data
                        t1_scale, m1_scale, t2_scale, m2_scale, mm, tt, ti1, ti2, mi1, mi2 = interpolated_data
                        
                        # Method 1: Scale prediction values by (dataset2_value / dataset1_value)
                        # Interpolate scaling factors at prediction times
                        value_ratio = np.interp(t_pred, tt, mi2/mi1, left=np.nan, right=np.nan)  # dataset2_value / dataset1_value
                        m_pred_scaled = m_pred * value_ratio
                        
                        # Method 2: Scale prediction times by (dataset2_time / dataset1_time)
                        mask = np.argsort(mm)
                        time_ratio = np.interp(m_pred, mm[mask], (ti2/ti1)[mask], left=np.nan, right=np.nan)  # dataset2_time / dataset1_time
                        t_pred_scaled = t_pred * time_ratio
                        
                        # Create plots for both methods
                        plot_method1 = create_prediction_plot_a3(data, t_pred, m_pred, m_pred_scaled, ptype='v')
                        plot_method2 = create_prediction_plot_a3(data, t_pred, m_pred, t_pred_scaled, ptype='t')
                        
                        approach_predictions['method1_value_scaling'] = {
                            'predictions': {
                                'in_vitro_time': t_pred, 
                                'in_vitro_value': m_pred, 
                                'predicted_in_vivo_value': m_pred_scaled,
                                'scaling_factor': value_ratio
                            },
                            'plot': plot_method1,
                            'description': 'Scale prediction values by (dataset2_value / dataset1_value)'
                        }
                        
                        approach_predictions['method2_time_scaling'] = {
                            'predictions': {
                                'in_vitro_time': t_pred, 
                                'in_vitro_value': m_pred, 
                                'predicted_in_vivo_time': t_pred_scaled,
                                'scaling_factor': time_ratio
                            },
                            'plot': plot_method2,
                            'description': 'Scale prediction times by (dataset2_time / dataset1_time)'
                        }
                    
                    prediction_results[approach] = approach_predictions 
                except Exception as e:
                    print('ERROR:', str(e))
                    approach_predictions[model_name] = {'error': str(e)}
                    
        prediction_results[approach] = approach_predictions
    
    return prediction_results


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _fit_cv_ratio(fit_value, cv_value):
    fit_value = _finite_float(fit_value)
    cv_value = _finite_float(cv_value)
    if not np.isfinite(fit_value) or not np.isfinite(cv_value) or cv_value == 0:
        return np.nan
    return fit_value / cv_value


CV_RATIO_MODERATE_THRESHOLD = 1.25
CV_RATIO_SUBSTANTIAL_THRESHOLD = 2.0

CV_SUMMARY_EXCLUDED_METRICS = CV_COMPARISON_EXCLUDED_METRICS



def _finite_vector(values):
    """Return finite numeric values as a one-dimensional array."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]




def _concat_finite(arrays):
    """Concatenate finite values from multiple arrays; return empty if none."""
    pieces = []
    for values in arrays:
        finite = _finite_vector(values)
        if finite.size:
            pieces.append(finite)
    if not pieces:
        return np.array([], dtype=float)
    return np.concatenate(pieces)

def _residual_display_scale(residuals, response_values, minimum_fraction=0.025, padding=1.10):
    """Return an independent symmetric residual-axis limit in response units.

    The limit covers the largest absolute residual but is never smaller than a
    small fraction of the observed response range. This prevents near-zero
    numerical residuals from being expanded to fill the entire plot.
    """
    residuals = _finite_vector(residuals)
    response_values = _finite_vector(response_values)
    max_abs_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    response_range = float(np.ptp(response_values)) if response_values.size else 0.0
    response_scale = float(np.max(np.abs(response_values))) if response_values.size else 0.0
    minimum_half_range = max(
        minimum_fraction * response_range,
        1e-12 * max(1.0, response_scale),
    )
    half_range = max(max_abs_residual, minimum_half_range)
    if not np.isfinite(half_range) or half_range <= 0:
        half_range = 1.0
    return {
        'axis_limit': padding * half_range,
        'max_abs_residual': max_abs_residual,
        'minimum_controls_scale': minimum_half_range > max_abs_residual,
        'response_range': response_range,
    }


def _qq_residual_tolerance(response_values, relative_tolerance=1e-2):
    """Return a practical lower bound for residual variation in a Q-Q plot."""
    response_values = _finite_vector(response_values)
    response_range = float(np.ptp(response_values)) if response_values.size else 0.0
    response_scale = float(np.max(np.abs(response_values))) if response_values.size else 0.0
    return max(
        relative_tolerance * response_range,
        1e-12 * max(1.0, response_scale),
    )



def create_cv_fit_variability_plotly(
    x,
    y,
    model_function,
    final_params,
    cv_fit_params,
    title,
    x_axis_title='Time',
    y_axis_title='Response value',
    marker_color='black',
):
    """Create a descriptive mean +/- SD plot across CV training-set fits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_data = np.isfinite(x) & np.isfinite(y)
    x = x[finite_data]
    y = y[finite_data]
    total_splits = len(cv_fit_params or [])

    if x.size < 2:
        return {
            'available': False,
            'message': 'CV fit variability plot not available: fewer than two finite observations are available.',
            'plot': None,
            'table_html': '',
            'successful_fits': 0,
            'total_splits': total_splits,
        }

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_min == x_max:
        return {
            'available': False,
            'message': 'CV fit variability plot not available: the fitting x-values do not span a range.',
            'plot': None,
            'table_html': '',
            'successful_fits': 0,
            'total_splits': total_splits,
        }

    x_grid = np.linspace(x_min, x_max, 200)
    cv_curves = []
    for params in cv_fit_params or []:
        if params is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                curve = np.asarray(model_function(x_grid, *params), dtype=float)
            if curve.shape == x_grid.shape and np.all(np.isfinite(curve)):
                cv_curves.append(curve)
        except Exception:
            continue

    successful_fits = len(cv_curves)
    if successful_fits < 2:
        if total_splits == 0:
            message = 'CV fit variability plot not available because cross-validation was not run.'
        else:
            message = (
                'CV fit variability plot not available: fewer than two cross-validation fits '
                'produced finite curves over the displayed range.'
            )
        return {
            'available': False,
            'message': message,
            'plot': None,
            'table_html': '',
            'successful_fits': successful_fits,
            'total_splits': total_splits,
        }

    cv_curves = np.vstack(cv_curves)
    cv_mean = np.mean(cv_curves, axis=0)
    cv_sd = np.std(cv_curves, axis=0, ddof=1)
    cv_lower = cv_mean - cv_sd
    cv_upper = cv_mean + cv_sd

    try:
        final_curve = np.asarray(model_function(x_grid, **final_params), dtype=float)
    except TypeError:
        final_curve = np.asarray(model_function(x_grid, *list(final_params.values())), dtype=float)

    if final_curve.shape != x_grid.shape or not np.all(np.isfinite(final_curve)):
        return {
            'available': False,
            'message': 'CV fit variability plot not available because the final fitted curve could not be evaluated over the displayed range.',
            'plot': None,
            'table_html': '',
            'successful_fits': successful_fits,
            'total_splits': total_splits,
        }

    band_fill = hex_to_rgba(marker_color, 0.18) if str(marker_color).startswith('#') else 'rgba(128,128,128,0.18)'
    mean_color = marker_color if str(marker_color).startswith('#') else '#555555'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=cv_upper,
        mode='lines',
        line=dict(width=0),
        name='CV mean + 1 SD',
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=cv_lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=band_fill,
        name='CV mean +/- 1 SD',
        meta='CV mean - 1 SD',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=cv_mean,
        mode='lines',
        name='Mean CV fit',
        line=dict(color=mean_color, width=3, dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=final_curve,
        mode='lines',
        name='Final full-data fit',
        line=dict(color='black', width=3),
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Observed data',
        marker=dict(color=marker_color, size=13, line=dict(color='white', width=2)),
    ))

    split_note = f'Band based on {successful_fits} successful CV fits'
    if total_splits and successful_fits != total_splits:
        split_note += f' of {total_splits} attempted splits'
    split_note += '. Mean +/- SD describes variation among CV training-set fits; it is not an uncertainty interval.'
    fig.add_annotation(
        text=split_note,
        xref='paper',
        yref='paper',
        x=0,
        y=1.02,
        showarrow=False,
        xanchor='left',
        yanchor='bottom',
        align='left',
        font=dict(size=12, color='rgba(80, 80, 80, 0.95)'),
    )

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend=dict(font=dict(size=16)),
        margin=dict(l=30, r=30, t=80, b=30),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
    )
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    table_html = extract_plotly_data_for_table(fig)
    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {
        'available': True,
        'message': '',
        'plot': plot_html,
        'table_html': table_html,
        'successful_fits': successful_fits,
        'total_splits': total_splits,
    }


def _attach_cv_fit_variability_plots(results, analysis_config=None):
    """Attach optional CV training-fit variability plots to completed fits."""
    analysis_config = analysis_config or {}
    if not bool(analysis_config.get('include_cv_fit_variability_plot', False)):
        return

    for model_key, model_results in results.items():
        if not isinstance(model_results, dict) or 'error' in model_results or ':' not in model_key:
            continue
        approach_id, model_name = model_key.split(':', 1)
        if approach_id not in ('approach1', 'approach2'):
            continue
        model_function = models[model_name]['model_function']
        model_display = models[model_name]['display_name']

        if approach_id == 'approach1':
            model_results['cv_fit_variability_plot'] = create_cv_fit_variability_plotly(
                model_results.get('residual_x'),
                model_results.get('residual_y'),
                model_function,
                model_results.get('params', {}),
                model_results.get('cv_fit_params', []),
                f'Cross-Validation Fit Variability: {model_display}',
                x_axis_title='Time (In Vitro Data)',
                y_axis_title='Time (In Vivo Data)',
                marker_color='black',
            )
        else:
            residual_x = model_results.get('residual_x', [None, None])
            residual_y = model_results.get('residual_y', [None, None])
            final_params = model_results.get('params', [{}, {}])
            cv_fit_params = model_results.get('cv_fit_params', [[], []])
            model_results['cv_fit_variability_plot'] = [
                create_cv_fit_variability_plotly(
                    residual_x[0], residual_y[0], model_function, final_params[0], cv_fit_params[0],
                    f'In Vitro Cross-Validation Fit Variability: {model_display}',
                    x_axis_title='Time', y_axis_title='Response value', marker_color=colors['in_vitro'],
                ),
                create_cv_fit_variability_plotly(
                    residual_x[1], residual_y[1], model_function, final_params[1], cv_fit_params[1],
                    f'In Vivo Cross-Validation Fit Variability: {model_display}',
                    x_axis_title='Time', y_axis_title='Response value', marker_color=colors['in_vivo'],
                ),
            ]


def create_residual_plotly(
    x,
    residuals,
    response_values,
    title,
    x_axis_title='Fitting x-axis',
    marker_color='black',
):
    """Create an independently scaled raw residual-vs-time Plotly figure."""
    x = np.asarray(x, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    response_values = np.asarray(response_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(residuals)
    x = x[mask]
    residuals = residuals[mask]

    scale_info = _residual_display_scale(residuals, response_values)
    axis_limit = scale_info['axis_limit']
    max_abs_residual = scale_info['max_abs_residual']

    fig = go.Figure()
    if x.size > 0:
        order = np.argsort(x)
        x_plot = x[order]
        residual_plot = residuals[order]
        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        fig.add_shape(
            type='line',
            xref='x',
            yref='y',
            x0=x_min,
            x1=x_max,
            y0=0,
            y1=0,
            line=dict(color='black', dash='dash', width=2),
            layer='above',
        )
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=residual_plot,
            mode='markers',
            name='Residuals',
            marker=dict(color=marker_color, size=15, line=dict(color='white', width=2)),
        ))

    #note = f"Maximum absolute residual: {max_abs_residual:.4g}"
    note = ''
    if scale_info['minimum_controls_scale']:
        note += 'Residuals are very small relative to the observed response range; the y-axis has not been expanded to fill the plot.'
    fig.add_annotation(
        text=note,
        xref='paper',
        yref='paper',
        x=0,
        y=1.02,
        showarrow=False,
        xanchor='left',
        yanchor='bottom',
        align='left',
        font=dict(size=12, color='rgba(80, 80, 80, 0.95)'),
    )

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title='Residual (observed - predicted)',
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True,
            zeroline=False,
            range=[-float(axis_limit), float(axis_limit)],
        ),
    )
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    table_html = extract_plotly_data_for_table(fig)
    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot': plot_html, 'table_html': table_html, 'scale_info': scale_info}


def create_residual_qq_plotly(
    residuals,
    response_values,
    title,
    marker_color='black',
):
    """Create a normal Q-Q plot unless residual variation is negligible."""
    residuals = _finite_vector(residuals)
    n = residuals.size

    if n < 2:
        return {
            'available': False,
            'message': 'Q-Q plot not informative: fewer than two finite residuals are available.',
            'plot': None,
            'table_html': '',
        }

    residual_sd = float(np.std(residuals, ddof=1))
    tolerance = _qq_residual_tolerance(response_values)
    if not np.isfinite(residual_sd) or residual_sd <= tolerance:
        return {
            'available': False,
            'message': 'Q-Q plot not informative: residual variation is too small relative to the response scale to assess the distribution reliably.',
            'plot': None,
            'table_html': '',
        }

    standardized = (residuals - float(np.mean(residuals))) / residual_sd
    ordered_residuals = np.sort(standardized)
    probabilities = (np.arange(1, n + 1, dtype=float) - 0.5) / n
    theoretical_quantiles = sp.stats.norm.ppf(probabilities)
    finite_mask = np.isfinite(theoretical_quantiles) & np.isfinite(ordered_residuals)
    theoretical_quantiles = theoretical_quantiles[finite_mask]
    ordered_residuals = ordered_residuals[finite_mask]

    if theoretical_quantiles.size < 2:
        return {
            'available': False,
            'message': 'Q-Q plot not informative: too few finite quantiles are available.',
            'plot': None,
            'table_html': '',
        }

    combined = np.concatenate([theoretical_quantiles, ordered_residuals])
    axis_min = float(np.min(combined))
    axis_max = float(np.max(combined))
    if axis_min == axis_max:
        axis_min -= 0.5
        axis_max += 0.5
    padding = 0.08 * (axis_max - axis_min)
    axis_min -= padding
    axis_max += padding

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode='lines',
        name='Normal reference line',
        line=dict(color='rgba(80, 80, 80, 0.85)', dash='dash', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=ordered_residuals,
        mode='markers',
        name='Standardized residuals',
        marker=dict(color=marker_color, size=15, line=dict(color='white', width=2)),
    ))
    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=title,
        xaxis_title='Theoretical normal quantile',
        yaxis_title='Ordered standardized residual',
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True, zeroline=False, range=[axis_min, axis_max]),
        yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True, zeroline=False, scaleanchor='x', scaleratio=1, range=[axis_min, axis_max]),
    )
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    table_html = extract_plotly_data_for_table(fig)
    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'available': True, 'message': '', 'plot': plot_html, 'table_html': table_html}


def _collect_residual_contexts(results):
    """Group residual data by approach and dataset."""
    contexts = {
        ('approach1', 'fit'): [],
        ('approach2', 'in_vitro'): [],
        ('approach2', 'in_vivo'): [],
    }
    for model_key, model_results in results.items():
        if not isinstance(model_results, dict) or 'error' in model_results or ':' not in model_key:
            continue
        approach_id, model_name = model_key.split(':', 1)
        if approach_id == 'approach1' and 'residuals' in model_results:
            contexts[('approach1', 'fit')].append({
                'model_key': model_key,
                'model_name': model_name,
                'x': model_results.get('residual_x'),
                'y': model_results.get('residual_y'),
                'residuals': model_results.get('residuals'),
                'marker_color': 'black',
                'title_prefix': 'Residuals',
                'x_axis_title': 'Time (In Vitro Data)',
            })
        elif approach_id == 'approach2' and isinstance(model_results.get('residuals'), list):
            residual_x = model_results.get('residual_x', [None, None])
            residual_y = model_results.get('residual_y', [None, None])
            contexts[('approach2', 'in_vitro')].append({
                'model_key': model_key,
                'model_name': model_name,
                'x': residual_x[0],
                'y': residual_y[0],
                'residuals': model_results['residuals'][0],
                'marker_color': colors['in_vitro'],
                'title_prefix': 'In Vitro Residuals',
                'x_axis_title': 'Time',
            })
            contexts[('approach2', 'in_vivo')].append({
                'model_key': model_key,
                'model_name': model_name,
                'x': residual_x[1],
                'y': residual_y[1],
                'residuals': model_results['residuals'][1],
                'marker_color': colors['in_vivo'],
                'title_prefix': 'In Vivo Residuals',
                'x_axis_title': 'Time',
            })
    return contexts


def _attach_residual_plots(results, analysis_config=None):
    """Attach the selected residual diagnostics to completed model results."""
    analysis_config = analysis_config or {}
    include_raw = bool(analysis_config.get('include_raw_residual_plots', True))
    include_qq = bool(analysis_config.get('include_residual_qq_plots', True))
    if not include_raw and not include_qq:
        return

    for (approach_id, dataset_id), items in _collect_residual_contexts(results).items():
        for item in items:
            model_results = results[item['model_key']]
            display_name = models[item['model_name']]['display_name']
            raw_plot = None
            qq_plot = None
            if include_raw:
                raw_plot = create_residual_plotly(
                    item['x'],
                    item['residuals'],
                    item['y'],
                    f"{item['title_prefix']} for {display_name} Fit",
                    x_axis_title=item['x_axis_title'],
                    marker_color=item['marker_color'],
                )
            if include_qq:
                qq_plot = create_residual_qq_plotly(
                    item['residuals'],
                    item['y'],
                    f"{item['title_prefix']} Normal Q-Q Plot for {display_name} Fit",
                    marker_color=item['marker_color'],
                )

            if approach_id == 'approach1':
                if raw_plot is not None:
                    model_results['residual_plot'] = raw_plot
                if qq_plot is not None:
                    model_results['residual_qq_plot'] = qq_plot
            else:
                idx = 0 if dataset_id == 'in_vitro' else 1
                if raw_plot is not None:
                    model_results.setdefault('residual_plot', [None, None])[idx] = raw_plot
                if qq_plot is not None:
                    model_results.setdefault('residual_qq_plot', [None, None])[idx] = qq_plot


_CV_BADGE_CLASS_BY_LEVEL = {
    'ok': 'evidence-badge-comparable',
    'caution': 'evidence-badge-lower',
    'warning': 'evidence-badge-minimal',
    'unavailable': 'evidence-badge-lower',
}

_CV_STATUS_RANK = {
    'ok': 0,
    'unavailable': 1,
    'caution': 2,
    'warning': 3,
}


def _cv_badge(label, level):
    badge_class = _CV_BADGE_CLASS_BY_LEVEL.get(level, 'evidence-badge-lower')
    return (
        f'<span class="evidence-badge {badge_class}">'
        f'{escape(label)}'
        '</span>'
    )


def _cv_worse_factor(metric, fit_value, cv_value):
    """Return how much worse CV performance is than final-model performance.

    A value near 1 indicates similar or better CV performance. Values above 1
    indicate worse CV performance, with direction interpreted using the metric's
    better_direction metadata.
    """
    fit_value = _finite_float(fit_value)
    cv_value = _finite_float(cv_value)
    if not np.isfinite(fit_value) or not np.isfinite(cv_value):
        return np.nan

    direction = metrics.get(metric, {}).get('better_direction', 'lower')

    if direction == 'higher':
        if cv_value >= fit_value:
            return 1.0
        if fit_value > 0 and cv_value <= 0:
            return np.inf
        if fit_value > 0 and cv_value > 0:
            return fit_value / cv_value
        return np.nan

    # Default: lower values are better.
    if cv_value <= fit_value:
        return 1.0
    if fit_value == 0:
        return np.inf if cv_value > 0 else 1.0
    if fit_value > 0 and cv_value > 0:
        return cv_value / fit_value
    return np.nan


def _cv_status(metric, fit_value, cv_value):
    factor = _cv_worse_factor(metric, fit_value, cv_value)
    if not np.isfinite(factor):
        return 'Unavailable', 'unavailable'
    if factor <= CV_RATIO_MODERATE_THRESHOLD:
        return 'CV similar to final', 'ok'
    if factor <= CV_RATIO_SUBSTANTIAL_THRESHOLD:
        return 'CV moderately worse', 'caution'
    return 'CV substantially worse', 'warning'


def _cv_status_badge(metric, fit_value, cv_value):
    label, level = _cv_status(metric, fit_value, cv_value)
    return _cv_badge(label, level), level


def _overall_cv_status_badge(levels):
    if not levels:
        return _cv_badge('Unavailable', 'unavailable')

    worst_level = max(levels, key=lambda level: _CV_STATUS_RANK.get(level, 0))
    if worst_level == 'warning':
        return _cv_badge('One or more CV metrics substantially worse', 'warning')
    if worst_level == 'caution':
        return _cv_badge('One or more CV metrics moderately worse', 'caution')
    if worst_level == 'unavailable':
        return _cv_badge('Review raw CV values', 'unavailable')
    return _cv_badge('No substantial CV degradation identified', 'ok')


def _cross_validation_row(model_display_name, stats_row, selected_metrics, dataset_label=None):
    row = {'Model': model_display_name}
    if dataset_label is not None:
        row['Dataset'] = dataset_label

    has_cv_value = False
    cv_status_levels = []
    for metric in selected_metrics:
        if metric in CV_SUMMARY_EXCLUDED_METRICS or metric not in metrics:
            continue
        display_name = metrics[metric]['display_name']
        fit_value = stats_row.get(metric, np.nan)
        cv_value = stats_row.get(f'CV {metric}', np.nan)
        ratio = _fit_cv_ratio(fit_value, cv_value)
        cv_value = _finite_float(cv_value)
        if np.isfinite(cv_value):
            has_cv_value = True
            _, level = _cv_status_badge(metric, fit_value, cv_value)
            cv_status_levels.append(level)
        row[f'CV {display_name}'] = cv_value
        row[f'Ratio {display_name}'] = ratio

    if has_cv_value:
        row['CV comparison'] = _overall_cv_status_badge(cv_status_levels)
        return row
    return None


def _cross_validation_table(rows):
    if not rows:
        return '<p class="evidence-note">Cross-validation summary is not available because cross-validation was not run or did not produce finite values for the selected eligible metrics. Adjusted R² and information criteria (AIC, AICc, BIC) are excluded from this table.</p>'
    df = pd.DataFrame(rows)
    return df.to_html(classes='table table-striped cv-summary-table', index=False, float_format=lambda x: f'{x:.4f}', na_rep='N/A', escape=False)


def _parameter_diagnostic_table(rows):
    if not rows:
        return '<p class="evidence-note">Fitted parameter diagnostics are not available for this approach.</p>'
    df = pd.DataFrame(rows)
    return df.to_html(classes='table table-striped parameter-diagnostics-summary-table', index=False, escape=False)


def create_parameter_diagnostics_summary(results, approach):
    """Create a compact model-comparison table of parameter diagnostic badges."""
    if approach not in ('approach1', 'approach2'):
        return '<p class="evidence-note">Fitted parameter diagnostics are not applicable because no parametric model is used.</p>'

    rows = []
    for model_key, model_results in results.items():
        if len(model_key.split(':')) > 1:
            approach_id, model_name = model_key.split(':')
        else:
            approach_id = model_key
        if approach_id != approach:
            continue

        model_display_name = models[model_name]['display_name']
        if model_results.get('error'):
            unavailable = '<span class="evidence-badge evidence-badge-minimal">Parameter uncertainty unavailable</span>'
            if approach == 'approach1':
                rows.append({'Model': model_display_name, 'Fitted parameter diagnostics': unavailable})
            else:
                rows.append({'Model': model_display_name, 'Dataset': 'In Vitro', 'Fitted parameter diagnostics': unavailable})
                rows.append({'Model': model_display_name, 'Dataset': 'In Vivo', 'Fitted parameter diagnostics': unavailable})
            continue

        if approach == 'approach1':
            rows.append({
                'Model': model_display_name,
                'Fitted parameter diagnostics': extract_parameter_diagnostic_badges_html(model_results.get('parameter_diagnostics_html')),
            })
        elif approach == 'approach2':
            diagnostics = model_results.get('parameter_diagnostics_html') or []
            for i, dataset_label in enumerate(['In Vitro', 'In Vivo']):
                diagnostics_html = diagnostics[i] if i < len(diagnostics) else ''
                rows.append({
                    'Model': model_display_name,
                    'Dataset': dataset_label,
                    'Fitted parameter diagnostics': extract_parameter_diagnostic_badges_html(diagnostics_html),
                })

    return _parameter_diagnostic_table(rows)


def create_cross_validation_summary(results, approach, selected_metrics):
    """Create compact model-comparison tables for CV scores.

    The table reports mean cross-validation scores that were already calculated
    during fitting, plus a neutral fit/CV ratio for each selected metric.
    """
    if approach not in ('approach1', 'approach2'):
        return '<p class="evidence-note">Cross-validation summary is not applicable because no parametric model is used.</p>'

    rows = []
    for model_key, model_results in results.items():
        if len(model_key.split(':')) > 1:
            approach_id, model_name = model_key.split(':')
        else:
            approach_id = model_key
        if approach_id != approach or model_results.get('error'):
            continue

        model_display_name = models[model_name]['display_name']
        if approach == 'approach1':
            row = _cross_validation_row(
                model_display_name,
                model_results['stats'].iloc[0],
                selected_metrics,
            )
            if row:
                rows.append(row)
        elif approach == 'approach2':
            for i, dataset_label in enumerate(['In Vitro', 'In Vivo']):
                row = _cross_validation_row(
                    model_display_name,
                    model_results['stats'][i].iloc[0],
                    selected_metrics,
                    dataset_label=dataset_label,
                )
                if row:
                    rows.append(row)

    return _cross_validation_table(rows)

def create_comparison(results, approach):
    comparison_data = []
    model_keys, model_display_names = [],[]
    for model_key, model_results in results.items():
            if len(model_key.split(':'))>1:
                approach_id, model_name = model_key.split(':')
            else:
                approach_id = model_key
            if approach_id == approach:
                model_keys.append(model_key)
                model_display_names.append(models[model_name]['display_name'])
                if 'error' in model_results and model_results['error'] is not None:
                    comparison_data.append({
                        'Model': models[model_name]['display_name'],
                        'error': model_results['error']
                    })
                else:
                    if approach == 'approach1':
                        tmp_comparison_data = {'Model': models[model_name]['display_name']}
                        for k,v in model_results['stats'].iloc[0].items():
                            if k.startswith('CV '):
                                continue
                                #metric = k[3:]
                            else:
                                metric = k
                            tmp_comparison_data.update({f'{metrics[metric]["display_name"]}':v})
                        comparison_data.append(tmp_comparison_data)
                    if approach == 'approach2':
                        for i in [1,2]:
                            tmp_comparison_data = {'Model': models[model_name]['display_name'], 'Dataset':('In Vitro' if i == 1 else 'In Vivo')}
                            for k,v in model_results['stats'][i-1].iloc[0].items():
                                if k.startswith('CV '):
                                    continue
                                    #metric = k[3:]
                                else:
                                    metric = k
                                tmp_comparison_data.update({f'{metrics[metric]["display_name"]}':v})
                            comparison_data.append(tmp_comparison_data)

    df = pd.DataFrame(comparison_data)
    metrics_table = df.to_html(classes='table table-striped', index=False, float_format=lambda x: f'{x:.4f}')

    # Create pairwise comparison table using KS test
#    n_models = len(model_keys)
#    if approach == 'approach1':
#        ks_matrix = np.zeros((n_models, n_models))
#        np.fill_diagonal(ks_matrix, 1)
#        for i in range(n_models):
#            for j in range(i+1, n_models):
#                try:
#                    _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'], results[model_keys[j]]['predictions'])
#                except Exception as e:
#                    print(f"Error processing prediction: {e}")
#                    p_value = np.nan
#                ks_matrix[i, j] = ks_matrix[j, i] = p_value
#        df_ks = pd.DataFrame(ks_matrix, index=model_display_names, columns=model_display_names)
#        ks_table = df_ks.to_html(classes='table table-striped', float_format=lambda x: f'{x:.4f}')
#        ks_tables = [ks_table]
#    elif approach == 'approach2':
#        ks_tables = []
#        for r in [1,2]:
#            ks_matrix = np.zeros((n_models, n_models))
#            np.fill_diagonal(ks_matrix, 1)
#            for i in range(n_models):
#                for j in range(i+1, n_models):
#                    try:
#                        _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'][r-1], results[model_keys[j]]['predictions'][r-1])
#                    except Exception as e:
#                        print(f"Error processing prediction: {e}")
#                        p_value = np.nan
#                    ks_matrix[i, j] = ks_matrix[j, i] = p_value
#            df_ks = pd.DataFrame(ks_matrix, index=model_display_names, columns=model_display_names)
#            ks_table = df_ks.to_html(classes='table table-striped', float_format=lambda x: f'{x:.4f}')
#            ks_tables.append(ks_table)
#    elif approach == 'approach3':
#        ks_tables = [None]
    ks_tables = [None]

    return metrics_table, ks_tables

    
def create_initial_plotly(t1, m1, t2, m2):
    fig = go.Figure()
    # In Vitro Data (actual)
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (raw input)',
        marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
    ))
    # In Vivo Data (actual)
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (raw input)',
        marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
    ))
    fig.update_layout(
        template='plotly_white',
        autosize=True, 
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=18)),  # Slightly smaller than base font
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Base font size
            color="black"
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        )
    )
    # Update axes with relative font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)
    # Configure the plot for download options
    config = {'responsive': True,'displaylogo': False}
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'table_html': table_html}

def _create_interpolation_plotly(t1, m1, t2, m2, interp_x1, interp_y1, interp_x2, interp_y2):
    fig = go.Figure()
    # Draw the measured/scaled points first so the larger open interpolation
    # markers remain visible when both series share a coordinate.
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (preprocessed)',
        marker=dict(color=colors['in_vitro'], size=14, symbol='circle', line=dict(color='white', width=2))
    ))
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (preprocessed)',
        marker=dict(color=colors['in_vivo'], size=14, symbol='diamond', line=dict(color='white', width=2))
    ))
    # Filled/open variants retain a consistent shape for each dataset while
    # making interpolated points visible outside overlapping measured points.
    fig.add_trace(go.Scatter(
        x=interp_x1, y=interp_y1,
        mode='markers',
        name='In Vitro Data (interpolated)',
        marker=dict(color=colors['in_vitro'], size=22, symbol='circle-open', line=dict(color='#1f1f1f', width=2.5))
    ))
    fig.add_trace(go.Scatter(
        x=interp_x2, y=interp_y2,
        mode='markers',
        name='In Vivo Data (interpolated)',
        marker=dict(color=colors['in_vivo'], size=22, symbol='diamond-open', line=dict(color='#1f1f1f', width=2.5))
    ))
    fig.update_layout(
        template='plotly_white',
        autosize=True, 
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=13)),
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        )
    )
    fig.update_xaxes(
        title_font=dict(size=16),
        tickfont=dict(size=13)
    )
    fig.update_yaxes(
        title_font=dict(size=16),
        tickfont=dict(size=13)
    )
    table_html = extract_plotly_data_for_table(fig)
    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot': plot_html, 'table_html': table_html}

def create_interpolation_plotly(t1, m1, t2, m2, tt, mi1, mi2):
    """Plot response values estimated at a shared time grid."""
    return _create_interpolation_plotly(t1, m1, t2, m2, tt, mi1, tt, mi2)


def create_value_interpolation_plotly(t1, m1, t2, m2, mm, ti1, ti2):
    """Plot times estimated at a shared response-value grid."""
    return _create_interpolation_plotly(t1, m1, t2, m2, ti1, mm, ti2, mm)

def create_prediction_interpolation_plotly(t1, m1):
    fig = go.Figure()
    # In Vitro Data (actual)
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (scaled/normed)',
        marker=dict(color=colors['in_vitro_2'], size=15, line=dict(color='white', width=2))
    ))
    fig.update_layout(
        template='plotly_white',
        autosize=True, 
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=18)),  # Slightly smaller than base font
        showlegend=True,  # Force legend to show even with one trace
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Base font size
            color="black"
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        )
    )
    # Update axes with relative font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'table_html': table_html}

def write_plotly_data_to_excel(fig, filename, directory):
    full_path = os.path.join(directory, filename)
    # Extract axis titles
    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else 'x'
    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else 'y'
    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        for trace in fig.data:
            df = pd.DataFrame({
                x_title: trace.x,
                y_title: trace.y
            })
            df.to_excel(writer, sheet_name=trace.name[:31], index=False)
    return full_path

def _plot_table_axis_label(axis_title):
    """Return a concise, readable axis label for downloaded plot data."""
    label = str(axis_title or '').strip() or 'Value'
    exact_labels = {
        'Time (In Vitro Data)': 'In vitro time',
        'Time (In Vivo Data)': 'In vivo time',
        'Value': 'Response value',
    }
    if label in exact_labels:
        return exact_labels[label]
    if label.endswith(' (In Vitro Data)'):
        base = label[:-len(' (In Vitro Data)')].strip().lower()
        return f'In vitro {base}'
    if label.endswith(' (In Vivo Data)'):
        base = label[:-len(' (In Vivo Data)')].strip().lower()
        return f'In vivo {base}'
    return label


def _plot_table_series_label(trace_name):
    """Return a concise series label without changing the visible plot legend."""
    label = str(trace_name or '').strip()
    replacements = {
        'Data': 'observed data',
        '95% Prediction Band Upper Bound': '95% band upper',
        '95% Prediction Band Lower Bound': '95% band lower',
        '95% Prediction Band Upper Bound (In Vitro)': '95% band upper, in vitro',
        '95% Prediction Band Lower Bound (In Vitro)': '95% band lower, in vitro',
        '95% Prediction Band Upper Bound (In Vivo)': '95% band upper, in vivo',
        '95% Prediction Band Lower Bound (In Vivo)': '95% band lower, in vivo',
    }
    if label in replacements:
        return replacements[label]
    if not label:
        return 'series'
    if label.endswith(' (In Vitro Data)'):
        base = label[:-len(' (In Vitro Data)')].replace(' Fit', ' fit')
        return f'{base}, in vitro'
    if label.endswith(' (In Vivo Data)'):
        base = label[:-len(' (In Vivo Data)')].replace(' Fit', ' fit')
        return f'{base}, in vivo'
    return (
        label
        .replace('In Vitro', 'in vitro')
        .replace('In Vivo', 'in vivo')
        .replace(' Data', ' data')
        .replace(' Fit', ' fit')
    )


def _plot_table_column_label(axis_title, trace_name, bound=None):
    axis_label = _plot_table_axis_label(axis_title)
    series_label = _plot_table_series_label(trace_name)
    if bound:
        series_label = f'{series_label} {bound} bound'
    return f'{axis_label} ({series_label})'


def extract_plotly_data_for_table(fig):
    """Return an HTML table containing plotted series and uncertainty bounds."""
    _apply_accessible_plot_styles(fig)

    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else 'X'
    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else 'Y'

    max_length = 0
    trace_data = []
    name_counts = {}

    for trace in fig.data:
        if not (hasattr(trace, 'x') and hasattr(trace, 'y')):
            continue
        if trace.x is None or trace.y is None:
            continue

        base_name = str(getattr(trace, 'name', '') or '').strip()
        if not base_name:
            continue
        table_name = getattr(trace, 'meta', None)
        if not isinstance(table_name, str) or not table_name.strip():
            table_name = base_name
        else:
            table_name = table_name.strip()
        name_counts[table_name] = name_counts.get(table_name, 0) + 1
        trace_name = table_name if name_counts[table_name] == 1 else f'{table_name} ({name_counts[table_name]})'

        x_values = list(trace.x)
        y_values = list(trace.y)
        max_length = max(max_length, len(x_values), len(y_values))
        columns = {
            _plot_table_column_label(x_title, trace_name): x_values,
            _plot_table_column_label(y_title, trace_name): y_values,
        }

        for axis_name, values, axis_title in (
            ('error_x', x_values, x_title),
            ('error_y', y_values, y_title),
        ):
            error = getattr(trace, axis_name, None)
            if error is None or getattr(error, 'visible', True) is False:
                continue
            error_plus = getattr(error, 'array', None)
            if error_plus is None:
                continue
            error_plus = list(error_plus)
            error_minus_raw = getattr(error, 'arrayminus', None)
            error_minus = list(error_minus_raw) if error_minus_raw is not None else list(error_plus)
            n = min(len(values), len(error_plus), len(error_minus))
            if n == 0:
                continue
            try:
                center = np.asarray(values[:n], dtype=float)
                plus = np.asarray(error_plus[:n], dtype=float)
                minus = np.asarray(error_minus[:n], dtype=float)
                columns[_plot_table_column_label(axis_title, trace_name, 'lower')] = list(center - minus)
                columns[_plot_table_column_label(axis_title, trace_name, 'upper')] = list(center + plus)
            except (TypeError, ValueError):
                continue

        trace_data.append(columns)

    if not trace_data:
        return '<p>No data available</p>'

    table_dfs = []
    for columns in trace_data:
        padded = {}
        for column_name, values in columns.items():
            padded[column_name] = list(values) + [''] * (max_length - len(values))
        table_dfs.append(pd.DataFrame(padded))

    combined_df = pd.concat(table_dfs, axis=1)
    return combined_df.to_html(
        classes='table table-striped',
        table_id=f'plot-data-{uuid4().hex}',
        index=False,
        float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else '',
        escape=False,
        na_rep='',
    )

def create_plotly_a1(x, y, model_info, include_bands=True):
    # Create the scatter plot for the data
    trace_data = go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(color='black', size=15, line=dict(color='white', width=2)), zorder=10)

    # Create the smooth line for the model fit
    x_smooth = np.linspace(min(x), max(x), max(len(x),10))
    y_smooth = model_info['model_function'](x_smooth, **model_info['params'])
    trace_fit = go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f"{model_info['model_name']} Fit", line=dict(color='black', dash='dash', width=3), zorder=9)

    traces = [trace_data, trace_fit]
    uncertainty_warnings = []

    # manual axis range
    x_range_data = [min(np.concatenate([x, x_smooth])), max(np.concatenate([x, x_smooth]))]
    y_range_data = [min(np.concatenate([y, y_smooth])), max(np.concatenate([y, y_smooth]))]
    # Add some padding (5% on each side)
    x_padding = (x_range_data[1] - x_range_data[0]) * 0.1
    y_padding = (y_range_data[1] - y_range_data[0]) * 0.1
    x_range = [x_range_data[0] - x_padding, x_range_data[1] + x_padding]
    y_range = [y_range_data[0] - y_padding, y_range_data[1] + y_padding]
    
    # Add prediction bands if requested and covariance matrix is available
    if include_bands:
        if 'pcov' not in model_info or model_info['pcov'] is None:
            uncertainty_warnings.append(
                _uncertainty_unavailable_warning(reason='covariance_unavailable')
            )
        else:
            try:
                bands = generate_prediction_bands(
                    model_info['model_function'],
                    x_smooth,
                    model_info['params'],
                    model_info['pcov'],
                    model_info['residuals']
                )
                uncertainty_warnings.extend(
                    _assess_uncertainty_band_visibility(
                        bands['lower'], bands['upper'], y_range
                    )
                )

                finite_band_mask = np.isfinite(bands['lower']) & np.isfinite(bands['upper'])
                if np.any(finite_band_mask):
                    # Add confidence band. Nonfinite points remain gaps in the Plotly trace.
                    trace_upper = go.Scatter(
                        x=x_smooth, y=bands['upper'],
                        mode='lines',
                        line=dict(width=0),
                        name='95% Prediction Band Upper Bound',
                        showlegend=False,
                        hoverinfo='skip',
                        zorder=0
                    )

                    trace_lower = go.Scatter(
                        x=x_smooth, y=bands['lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.3)',
                        line=dict(width=0),
                        name='95% Prediction Band',
                        meta='95% Prediction Band Lower Bound',
                        hoverinfo='skip',
                        zorder=0
                    )

                    # Insert bands before the fit line for proper layering
                    traces = [trace_data, trace_upper, trace_lower, trace_fit]

            except Exception as e:
                print(f"Warning: Could not generate prediction bands: {e}")
                uncertainty_warnings.append(_uncertainty_unavailable_warning())

    # Create the figure and add the traces
    fig = go.Figure(data=traces)

    # Create the layout
    fig.update_layout(
        template='plotly_white',
        autosize=True, 
        title=f"Time Series Data with {model_info['model_name']} Fit",
        xaxis_title='Time (In Vitro Data)',
        yaxis_title='Time (In Vivo Data)',
        legend=dict(font=dict(size=18)),  # Slightly smaller than base font
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Base font size
            color="black"
        ),
        xaxis=dict(
            range=x_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            layer='above traces'
        ),
        yaxis=dict(
            range=y_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            layer='above traces'
        )
    )
    # Update axes with relative font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    # Draw an explicit border around the plotting area. Mirrored axis lines can
    # be obscured by filled prediction-band traces in some Plotly/browser
    # combinations, whereas a paper-referenced shape remains visible above all
    # traces and matches the boxed appearance of the diagnostic plots.
    fig.add_shape(
        type='rect',
        xref='paper',
        yref='paper',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color='#767676', width=2),
        fillcolor='rgba(0, 0, 0, 0)',
        layer='above'
    )
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {
        'plot': plot_html,
        'table_html': table_html,
        'uncertainty_warnings': _merge_uncertainty_warnings(uncertainty_warnings),
    }

def create_plotly_a2(x1, y1, x2, y2, model_info, include_bands=True):
    # Create traces for the datasets
    trace1 = go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='In Vitro Data',
        marker=dict(
            color=colors['in_vitro'],
            size=15,
            line=dict(
                color='white',
                width=2
            ),
            symbol='circle'
        ),
        zorder=10
    )

    trace2 = go.Scatter(
        x=x2, y=y2,
        mode='markers',
        name='In Vivo Data',
        marker=dict(
            color=colors['in_vivo'],
            size=15,
            line=dict(
                color='white',
                width=2
            ),
            symbol='circle'
        ),
        zorder=10
    )

    # Create smooth lines for model fits
    x_smooth = np.linspace(min(np.concatenate([x1,x2])), max(np.concatenate([x1,x2])), 200)
    y_smooth1 = model_info['model_function'](x_smooth, **model_info['params1'])
    y_smooth2 = model_info['model_function'](x_smooth, **model_info['params2'])

    # manual axis range
    x_range_data = [min(np.concatenate([x1, x2, x_smooth])), max(np.concatenate([x1, x2, x_smooth]))]
    y_range_data = [min(np.concatenate([y1, y2, y_smooth1, y_smooth2])), max(np.concatenate([y1, y2, y_smooth1, y_smooth2]))]
    # Add some padding (5% on each side)
    x_padding = (x_range_data[1] - x_range_data[0]) * 0.1
    y_padding = (y_range_data[1] - y_range_data[0]) * 0.1
    x_range = [x_range_data[0] - x_padding, x_range_data[1] + x_padding]
    y_range = [y_range_data[0] - y_padding, y_range_data[1] + y_padding]

    traces = [trace1, trace2]
    uncertainty_warnings = []
    
    # Add prediction bands for dataset 1 (In Vitro) if requested.
    if include_bands:
        if 'pcov1' not in model_info or model_info['pcov1'] is None:
            uncertainty_warnings.append(
                _uncertainty_unavailable_warning('In vitro', 'covariance_unavailable')
            )
        else:
            try:
                bands1 = generate_prediction_bands(
                    model_info['model_function'],
                    x_smooth,
                    model_info['params1'],
                    model_info['pcov1'],
                    model_info['residuals1']
                )
                uncertainty_warnings.extend(
                    _assess_uncertainty_band_visibility(
                        bands1['lower'], bands1['upper'], y_range, 'In vitro'
                    )
                )

                finite_band_mask1 = np.isfinite(bands1['lower']) & np.isfinite(bands1['upper'])
                if np.any(finite_band_mask1):
                    trace1_upper = go.Scatter(
                        x=x_smooth, y=bands1['upper'],
                        mode='lines',
                        line=dict(width=0),
                        name='95% Prediction Band Upper Bound (In Vitro)',
                        showlegend=False,
                        hoverinfo='skip',
                        zorder=0
                    )

                    trace1_lower = go.Scatter(
                        x=x_smooth, y=bands1['lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=hex_to_rgba(colors['in_vitro'], 0.2),
                        line=dict(width=0),
                        name='95% Prediction Band (In Vitro)',
                        meta='95% Prediction Band Lower Bound (In Vitro)',
                        hoverinfo='skip',
                        zorder=0
                    )

                    traces.extend([trace1_upper, trace1_lower])

            except Exception as e:
                print(f"Warning: Could not generate prediction bands for dataset 1: {e}")
                uncertainty_warnings.append(_uncertainty_unavailable_warning('In vitro'))

    # Add prediction bands for dataset 2 (In Vivo) if requested.
    if include_bands:
        if 'pcov2' not in model_info or model_info['pcov2'] is None:
            uncertainty_warnings.append(
                _uncertainty_unavailable_warning('In vivo', 'covariance_unavailable')
            )
        else:
            try:
                bands2 = generate_prediction_bands(
                    model_info['model_function'],
                    x_smooth,
                    model_info['params2'],
                    model_info['pcov2'],
                    model_info['residuals2']
                )
                uncertainty_warnings.extend(
                    _assess_uncertainty_band_visibility(
                        bands2['lower'], bands2['upper'], y_range, 'In vivo'
                    )
                )

                finite_band_mask2 = np.isfinite(bands2['lower']) & np.isfinite(bands2['upper'])
                if np.any(finite_band_mask2):
                    trace2_upper = go.Scatter(
                        x=x_smooth, y=bands2['upper'],
                        mode='lines',
                        line=dict(width=0),
                        name='95% Prediction Band Upper Bound (In Vivo)',
                        showlegend=False,
                        hoverinfo='skip',
                        zorder=0
                    )

                    trace2_lower = go.Scatter(
                        x=x_smooth, y=bands2['lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=hex_to_rgba(colors['in_vivo'], 0.2),
                        line=dict(width=0),
                        name='95% Prediction Band (In Vivo)',
                        meta='95% Prediction Band Lower Bound (In Vivo)',
                        hoverinfo='skip',
                        zorder=0
                    )

                    traces.extend([trace2_upper, trace2_lower])

            except Exception as e:
                print(f"Warning: Could not generate prediction bands for dataset 2: {e}")
                uncertainty_warnings.append(_uncertainty_unavailable_warning('In vivo'))


    trace3 = go.Scatter(
        x=x_smooth, y=y_smooth1,
        mode='lines',
        name=f"{model_info['model_name']} Fit (In Vitro Data)",
        line=dict(color=colors['in_vitro'], dash='dash', width=3),
        zorder=9
    )

    trace4 = go.Scatter(
        x=x_smooth, y=y_smooth2,
        mode='lines',
        name=f"{model_info['model_name']} Fit (In Vivo Data)",
        line=dict(color=colors['in_vivo'], dash='dash', width=3),
        zorder=9
    )

    traces.extend([trace3, trace4])

    # Create the layout
    layout = dict(
        template='plotly_white',
        autosize=True, 
        title=f"Time Series Data with {model_info['model_name']} Fit",
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=18)),  # Slightly smaller than base font
        #width=600,
        #height=600*0.5,
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Base font size
            color="black"
        ),
        xaxis=dict(
            range=x_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            layer='above traces'
        ),
        yaxis=dict(
            range=y_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            layer='above traces'
        )
    )

    # Create the figure and add the traces
    fig = go.Figure(data=traces, layout=layout)

    # Update axes with relative font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    # Draw an explicit border around the plotting area so the filled
    # prediction bands cannot hide the top or right edge of the plot box.
    fig.add_shape(
        type='rect',
        xref='paper',
        yref='paper',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color='#767676', width=2),
        fillcolor='rgba(0, 0, 0, 0)',
        layer='above'
    )
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {
        'plot': plot_html,
        'table_html': table_html,
        'uncertainty_warnings': _merge_uncertainty_warnings(uncertainty_warnings),
    }

def create_plotly_a3(x, y1, y2, ptype='v'):
    if ptype == 'v':
        y = y2 / y1
        x_label = 'Time'
        y_label = 'In Vivo Value / In Vitro Value'
        plot_name = 'Value vs. time'
    elif ptype == 't':
        y = y2 / y1
        x_label = 'Value'
        y_label = 'In Vivo Time / In Vitro Time'
        plot_name = 'Time vs. value'
    else:
        raise ValueError("ptype must be 'v' or 't'")

    # Create the scatter plot
    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name=plot_name,
        marker=dict(
            color='black',
            size=15,
            line=dict(
                color='white',
                width=2
            ),
            symbol='circle'
        )
    )

    # Create the layout
    layout = dict(
        template='plotly_white',
        autosize=True, 
        title=f"Direct mapping {plot_name}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(font=dict(size=18)),  # Slightly smaller than base font
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Base font size
            color="black"
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#767676',
            mirror=True
        )
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[trace], layout=layout)

    # Update axes with relative font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_font=dict(size=20),  # Slightly larger than base font
        tickfont=dict(size=18)
    )
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'table_html': table_html}

def create_prediction_plot_a1(data, t_pred, m_pred, t_pred_vivo, model_info, include_bands=True):
    """Create prediction plot for approach 1 - time scaling"""
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    fig = go.Figure()

    # Fitting in vitro data
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (Fitting)',
        marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
    ))

    # Fitting in vivo timeline
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (Fitting)',
        marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
    ))

    # Original in vitro data
    fig.add_trace(go.Scatter(
        x=t_pred, y=m_pred,
        mode='markers',
        name='In Vitro (Input)',
        marker=dict(color=colors['in_vitro_2'], size=15, line=dict(color='white', width=2))
    ))

    # Calculate prediction bands for the predicted in vivo times if available
    error_x_lower = None
    error_x_upper = None
    
    if include_bands and 'pcov' in model_info and model_info['pcov'] is not None and 'residuals' in model_info:
        try:
            # Generate prediction bands for the predicted times
            bands = generate_prediction_bands(
                model_info['model_function'], 
                t_pred,  # Use the actual prediction times
                model_info['params'], 
                model_info['pcov'],
                model_info['residuals']
            )
            
            # Calculate error bar lengths (distance from mean to upper/lower bounds)
            error_x_lower = bands['mean'] - bands['lower']  # Distance to lower bound
            error_x_upper = bands['upper'] - bands['mean']  # Distance to upper bound
            
        except Exception as e:
            print(f"Warning: Could not generate prediction bands for error bars: {e}")

    # Predicted in vivo timeline with error bars
    if error_x_lower is not None and error_x_upper is not None:
        fig.add_trace(go.Scatter(
            x=t_pred_vivo, y=m_pred,
            mode='markers',
            name='In Vivo (Predicted)',
            marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2)),
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_x_upper,      # Upper error
                arrayminus=error_x_lower, # Lower error
                visible=True,
                color=colors['in_vivo_2'],
                thickness=2,
                width=8
            )
        ))
    else:
        # Fallback: predicted points without error bars
        fig.add_trace(go.Scatter(
            x=t_pred_vivo, y=m_pred,
            mode='markers',
            name='In Vivo (Predicted)',
            marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
        ))

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=f"In Vitro to In Vivo Time Prediction using {model_info['model_name']} Fit",
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=18)),
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True)
    )

    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    return {'plot': plot_html, 'table_html': table_html}

def create_prediction_plot_a2(data, t_pred, m_pred, mt_pred_vivo, t_pred_plot, model1_pred_plot, model2_pred_plot, model_info, t_pred_vivo_err=None, include_bands=True):
    """Create prediction plot for approach 2 - value scaling using model ratio"""
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    fig = go.Figure()

    # Fitting in vitro data
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (Fitting)',
        marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
    ))

    # Fitting in vivo timeline
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (Fitting)',
        marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
    ))

    # Original in vitro data
    fig.add_trace(go.Scatter(
        x=t_pred, y=m_pred,
        mode='markers',
        name='In Vitro (Input)',
        marker=dict(color=colors['in_vitro_2'], size=15, line=dict(color='white', width=2))
    ))

    # Calculate prediction bands/error bars for the predicted in vivo values or times if available
    error_x_lower = None
    error_x_upper = None
    error_y_lower = None
    error_y_upper = None
    
    if t_pred_vivo_err is None:
        ## MASS SCALING WITH ERROR BARS
        title = f"In Vitro to In Vivo Value Prediction using value ratio rescaling with {model_info['model_name']}"
        if include_bands and 'pcov' in model_info and model_info['pcov'][0] is not None:
            try:
                bands = generate_ratio_prediction_bands(model_info['model_function'], model_info['model_function'], t_pred, model_info['params'][0], model_info['pcov'][0], model_info['params'][1], model_info['pcov'][1], residuals1=model_info['residuals'][0], residuals2=model_info['residuals'][1], confidence_level=0.95, n_samples=1000)
                
                # Calculate error bar lengths (distance from mean to upper/lower bounds)
                error_y_lower = bands['mean'] - bands['lower']  # Distance to lower bound
                error_y_upper = bands['upper'] - bands['mean']  # Distance to upper bound

                error_y_lower = mt_pred_vivo * error_y_lower
                error_y_upper = mt_pred_vivo * error_y_upper
                
            except Exception as e:
                print(f"Warning: Could not generate prediction bands for error bars: {e}")

        # Predicted in vivo timeline with error bars
        if error_y_lower is not None and error_y_upper is not None:
            fig.add_trace(go.Scatter(
                x=t_pred, y=mt_pred_vivo,
                mode='markers',
                name='In Vivo (Predicted)',
                marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2)),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_y_upper,      # Upper error
                    arrayminus=error_y_lower, # Lower error
                    visible=True,
                    color=colors['in_vivo_2'],
                    thickness=2,
                    width=8
                )
            ))
        else:
            # Fallback: predicted points without error bars
            fig.add_trace(go.Scatter(
                x=t_pred, y=mt_pred_vivo,
                mode='markers',
                name='In Vivo (Predicted)',
                marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
            ))

    else:
        ## TIME
        # Predicted in vivo timeline with error bars
        title = f"In Vitro to In Vivo Value Prediction using time constant ratio rescaling with {model_info['model_name']}"
        if t_pred_vivo_err is not None:
            fig.add_trace(go.Scatter(
                x=mt_pred_vivo, y=m_pred,
                mode='markers',
                name='In Vivo (Predicted)',
                marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2)),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=t_pred_vivo_err,      # Upper error
                    arrayminus=t_pred_vivo_err, # Lower error
                    visible=True,
                    color=colors['in_vivo_2'],
                    thickness=2,
                    width=8
                )
            ))
        else:
            # Fallback: predicted points without error bars
            fig.add_trace(go.Scatter(
                x=mt_pred_vivo, y=m_pred,
                mode='markers',
                name='In Vivo (Predicted)',
                marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
            ))
    if 0:
        ## MASS, no error bars
        # Predicted in vivo values
        fig.add_trace(go.Scatter(
            x=mt_pred_vivo, y=m_pred,
            mode='markers',
            name='In Vivo (Predicted)',
            marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
        ))

        # Connection lines to show time scaling
        for i in range(len(t_pred)):
            fig.add_trace(go.Scatter(
                x=[t_pred[i], t_pred_vivo[i]],
                y=[m_pred[i], m_pred[i]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Model predictions for reference
    fig.add_trace(go.Scatter(
        x=t_pred_plot, y=model1_pred_plot,
        mode='lines',
        name='In Vitro (Fitted Model)',
        line=dict(color=colors['in_vitro'], dash='dash', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=t_pred_plot, y=model2_pred_plot,
        mode='lines',
        name='In Vivo (Fitted Model)',
        line=dict(color=colors['in_vivo'], dash='dash', width=2)
    ))

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(font=dict(size=18)),
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True)
    )

    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    return {'plot': plot_html, 'table_html': table_html}

def create_prediction_plot_a3(data, t_pred, m_pred, var_pred_scaled, ptype='v'):
    """Create prediction plot for approach 3 - direct mapping"""
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    if ptype == 'v':
        plot_string = 'predicted value scaled by value ratio'
        fig = go.Figure()

        # Fitting in vitro data
        fig.add_trace(go.Scatter(
            x=t1, y=m1,
            mode='markers',
            name='In Vitro Data (Fitting)',
            marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
        ))

        # Fitting in vivo timeline
        fig.add_trace(go.Scatter(
            x=t2, y=m2,
            mode='markers',
            name='In Vivo Data (Fitting)',
            marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
        ))

        # Prediction data
        fig.add_trace(go.Scatter(
            x=t_pred, y=m_pred,
            mode='markers',
            name='In Vitro (Input)',
            marker=dict(color=colors['in_vitro_2'], size=15, line=dict(color='white', width=2))
        ))

        fig.add_trace(go.Scatter(
            x=t_pred, y=var_pred_scaled,
            mode='markers',
            name='In Vivo (Predicted)',
            marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
        ))

        # Connection lines to show time scaling
        for i in range(len(t_pred)):
            fig.add_trace(go.Scatter(
                x=[t_pred[i], t_pred[i]],
                y=[m_pred[i], var_pred_scaled[i]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            template='plotly_white',
            autosize=True,
            title=f"In Vitro Prediction Data ({plot_string})",
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(font=dict(size=18)),
            margin=dict(l=30, r=30, t=30, b=30),
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True)
        )
    elif ptype == 't':
        plot_string = 'predicted time scaled by time ratio'
        fig = go.Figure()

        # Fitting in vitro data
        fig.add_trace(go.Scatter(
            x=t1, y=m1,
            mode='markers',
            name='In Vitro Data (Fitting)',
            marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
        ))

        # Fitting in vivo timeline
        fig.add_trace(go.Scatter(
            x=t2, y=m2,
            mode='markers',
            name='In Vivo Data (Fitting)',
            marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
        ))

        # Prediction data
        fig.add_trace(go.Scatter(
            x=t_pred, y=m_pred, 
            mode='markers',
            name='In Vitro (Input)',
            marker=dict(color=colors['in_vitro_2'], size=15, line=dict(color='white', width=2))
        ))

        fig.add_trace(go.Scatter(
            x=var_pred_scaled, y=m_pred, 
            mode='markers',
            name='In Vivo (Predicted)',
            marker=dict(color=colors['in_vivo_2'], size=15, line=dict(color='white', width=2))
        ))

        # Connection lines to show time scaling
        for i in range(len(t_pred)):
            fig.add_trace(go.Scatter(
                x=[t_pred[i], var_pred_scaled[i]],
                y=[m_pred[i], m_pred[i]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            template='plotly_white',
            autosize=True,
            title=f"In Vitro Prediction Data ({plot_string})",
            yaxis_title='Value',
            xaxis_title='Time',
            legend=dict(font=dict(size=18)),
            margin=dict(l=30, r=30, t=30, b=30),
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            xaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='#767676', mirror=True)
        )

    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    return {'plot': plot_html, 'table_html': table_html}

