import os
import traceback
import warnings
from html import escape

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
    'in_vitro_2': '#7BB8E8',    # Light blue
    'in_vivo_2': '#D487B8'      # Light purple-red
}


CV_COMPARISON_EXCLUDED_METRICS = {'adjusted_r_squared', 'aic', 'aicc', 'bic', 'nrmse', 'mnrmse'}


def _not_meaningful_badge():
    return '<span class="evidence-badge evidence-badge-neutral">CV comparison not meaningful</span>'


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
                cvs = cross_validation_curve_fit(x, y, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
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
                    'pcov': model_result['pcov'],
                    'residuals': residuals,
                    'residual_x': x,
                    'residual_y': y,
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
                # process data for this approach
                m = ~np.isnan(mi1)
                x1,y1 = tt[m], mi1[m]
                m = ~np.isnan(mi2)
                x2,y2 = tt[m], mi2[m]
                # Dataset 1
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x1, y1, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state,
                                          initial_points=_model_initial_points(model_name, x1, y1))
                # cross-validation
                cvs1 = cross_validation_curve_fit(x1, y1, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
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
                tau1 = calculate_tau_with_uncertainty(models[model_name]['model_function'], popt1, pcov1)
                # deal with integration failure
                if tau1[0]>0 and np.isfinite(tau1[0]):
                    tau1 = unc.ufloat(*tau1)
                else:
                    tau1 = unc.ufloat(0,1)
                # Dataset 2
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x2, y2, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state,
                                          initial_points=_model_initial_points(model_name, x2, y2))
                # cross-validation
                cvs2 = cross_validation_curve_fit(x2, y2, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs2_mean = {f'{metric}':np.nanmean(v[np.isfinite(v)]) for metric,v in cvs2.items()}
                # fit final model
                model_result2 = models[model_name]['fit_model'](x2, y2, p0=p0)
                y_pred2 = model_result2['predict'](x2)
                # Evaluate goodness of fit
                gof2 = {metric: calculate_metric(metric, y2, y_pred2, len(p0)) for metric in selected_metrics}
                stats2 = pd.concat([pd.DataFrame(gof2, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs2_mean.items()}, index=[0])], axis=1)
                stats_table2 = _goodness_cv_comparison_table(gof2, cvs2_mean)
                #upopt2 = unc.correlated_values(popt2, pcov2)
                #tau2 = get_tau(modelname, upopt2)
                popt2, pcov2 = list(model_result2['params'].values()), model_result2['pcov']
                upopt2 = unc.correlated_values(popt2, pcov2)
                upopt2 = dict(zip(model_result2['params'].keys(),upopt2))
                tau2 = calculate_tau_with_uncertainty(models[model_name]['model_function'], popt2, pcov2)
                # deal with integration failure
                if tau2[0]>0 and np.isfinite(tau2[0]):
                    tau2 = unc.ufloat(*tau2)
                else:
                    tau2 = unc.ufloat(0,1)
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
                    'tau': [tau1,tau2],
                    'stats': [stats1,stats2],
                    'predictions': [y_pred1,y_pred2],
                    'pcov': [pcov1,pcov2],
                    'residuals': [residuals1,residuals2],
                    'residual_x': [x1, x2],
                    'residual_y': [y1, y2],
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
    
    _attach_residual_plots(results)
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
                                    dof = len(tt) - len(model_results['params'])
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

def _robust_residual_reference_scale(residual_values, percentile=95.0):
    """Return a robust common residual scale across comparable models."""
    vals = np.abs(_finite_vector(residual_values))
    if vals.size == 0:
        return np.nan
    scale = float(np.percentile(vals, percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(vals)) if vals.size else np.nan
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def _common_axis_limit(values, reference_scale=None, padding=1.20):
    """Return a symmetric y-axis limit that covers residuals and reference bands."""
    vals = _finite_vector(values)
    candidates = []
    if vals.size:
        candidates.append(float(np.max(np.abs(vals))))
    if reference_scale is not None and np.isfinite(reference_scale) and reference_scale > 0:
        candidates.append(float(reference_scale))
    max_value = max(candidates) if candidates else 1.0
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    return padding * max_value






def create_residual_plotly(
    x,
    residuals,
    title,
    x_axis_title='Fitting x-axis',
    marker_color='black',
    reference_scale=None,
    reference_label='observed response SD',
    y_axis_limit=None,
):
    """Create an original-unit residual-vs-x Plotly figure and source-data table.

    Reference bands are based on a robust residual scale computed across
    comparable models, not model-specific RMSE, so residual plots are easier
    to compare while staying focused on residual behavior.
    """
    x = np.asarray(x, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(residuals)
    x = x[mask]
    residuals = residuals[mask]

    y_axis_title = 'Residual (observed - predicted)'
    residual_values = residuals

    band_scale = None
    inner_band = None
    outer_band = None
    if reference_scale is not None and np.isfinite(reference_scale) and reference_scale > 0:
        band_scale = float(reference_scale)
        inner_band = 0.25 * band_scale
        outer_band = 0.50 * band_scale

    fig = go.Figure()
    if x.size > 0:
        order = np.argsort(x)
        x_plot = x[order]
        residual_plot = residual_values[order]
        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        if inner_band is not None and outer_band is not None and np.isfinite(inner_band) and np.isfinite(outer_band) and outer_band > 0:
            fig.add_shape(
                type='rect',
                xref='x',
                yref='y',
                x0=x_min,
                x1=x_max,
                y0=-inner_band,
                y1=inner_band,
                fillcolor='rgba(128, 128, 128, 0.18)',
                line=dict(width=0),
                layer='below',
            )
            fig.add_shape(
                type='line',
                xref='x',
                yref='y',
                x0=x_min,
                x1=x_max,
                y0=outer_band,
                y1=outer_band,
                line=dict(color='rgba(80, 80, 80, 0.8)', dash='dot', width=2),
                layer='below',
            )
            fig.add_shape(
                type='line',
                xref='x',
                yref='y',
                x0=x_min,
                x1=x_max,
                y0=-outer_band,
                y1=-outer_band,
                line=dict(color='rgba(80, 80, 80, 0.8)', dash='dot', width=2),
                layer='below',
            )
            fig.add_annotation(
                x=x_max,
                y=inner_band,
                xref='x',
                yref='y',
                text=f'+/- 0.25 {reference_label}',
                showarrow=False,
                xanchor='right',
                yanchor='bottom',
                font=dict(size=12, color='rgba(80, 80, 80, 0.9)'),
                bgcolor='rgba(255, 255, 255, 0.65)',
            )
            fig.add_annotation(
                x=x_max,
                y=outer_band,
                xref='x',
                yref='y',
                text=f'+/- 0.5 {reference_label}',
                showarrow=False,
                xanchor='right',
                yanchor='bottom',
                font=dict(size=12, color='rgba(80, 80, 80, 0.9)'),
                bgcolor='rgba(255, 255, 255, 0.65)',
            )

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

    yaxis_kwargs = dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True, zeroline=False)
    if y_axis_limit is not None and np.isfinite(y_axis_limit) and y_axis_limit > 0:
        yaxis_kwargs['range'] = [-float(y_axis_limit), float(y_axis_limit)]

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        xaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True),
        yaxis=yaxis_kwargs,
    )
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    table_html = extract_plotly_data_for_table(fig)
    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot': plot_html, 'table_html': table_html}


def _collect_residual_contexts(results):
    """Group residual data so plots can share scaling across comparable models."""
    contexts = {
        ('approach1', 'fit'): [],
        ('approach2', 'in_vitro'): [],
        ('approach2', 'in_vivo'): [],
    }
    for model_key, model_results in results.items():
        if not isinstance(model_results, dict) or 'error' in model_results:
            continue
        if ':' not in model_key:
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


def _attach_residual_plots(results):
    """Attach original-unit residual plots with common robust scaling."""
    for (approach_id, dataset_id), items in _collect_residual_contexts(results).items():
        if not items:
            continue

        all_residuals = _concat_finite(item['residuals'] for item in items)
        residual_reference_scale = _robust_residual_reference_scale(all_residuals)
        if all_residuals.size == 0:
            continue

        original_axis_limit = _common_axis_limit(all_residuals, reference_scale=residual_reference_scale)

        for item in items:
            model_key = item['model_key']
            model_results = results[model_key]
            display_name = models[item['model_name']]['display_name']
            original_plot = create_residual_plotly(
                item['x'],
                item['residuals'],
                f"{item['title_prefix']} for {display_name} Fit (Original Units, Common Scale)",
                x_axis_title=item['x_axis_title'],
                marker_color=item['marker_color'],
                reference_scale=residual_reference_scale,
                reference_label='residual scale',
                y_axis_limit=original_axis_limit,
            )
            if approach_id == 'approach1':
                model_results['residual_plot'] = original_plot
            elif approach_id == 'approach2':
                if 'residual_plot' not in model_results:
                    model_results['residual_plot'] = [None, None]
                idx = 0 if dataset_id == 'in_vitro' else 1
                model_results['residual_plot'][idx] = original_plot


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
        return 'CV comparison unavailable', 'unavailable'
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
        return _cv_badge('CV comparison unavailable', 'unavailable')

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
    """Create compact model-comparison tables for held-out CV scores.

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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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

def create_interpolation_plotly(t1, m1, t2, m2, tt, mi1, mi2):
    fig = go.Figure()
    # In Vitro Data (actual)
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (scaled/normed)',
        marker=dict(color=colors['in_vitro'], size=24, line=dict(color='white', width=2))
    ))
    # In Vitro Data (interpolated)
    fig.add_trace(go.Scatter(
        x=tt, y=mi1,
        mode='markers',
        name='In Vitro Data (interpolated)',
        marker=dict(color=colors['in_vitro'], size=12, symbol='square', line=dict(color='white', width=2))
    ))
    # In Vivo Data (actual)
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (scaled/normed)',
        marker=dict(color=colors['in_vivo'], size=24, line=dict(color='white', width=2))
    ))
    # In Vivo Data (interpolated)
    fig.add_trace(go.Scatter(
        x=tt, y=mi2,
        mode='markers',
        name='In Vivo Data (interpolated)',
        marker=dict(color=colors['in_vivo'], size=12, symbol='square', line=dict(color='white', width=2))
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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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

def create_interpolation_plotly_new(t1, m1, t2, m2, mm, ti1, ti2):
    fig = go.Figure()
    # In Vitro Data (actual)
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='In Vitro Data (actual)',
        marker=dict(color=colors['in_vitro'], size=15, line=dict(color='black', width=1))
    ))
    # In Vitro Data (interpolated)
    fig.add_trace(go.Scatter(
        x=ti1, y=mm,
        mode='markers',
        name='In Vitro Data (interpolated)',
        marker=dict(color=colors['in_vitro'], size=8, line=dict(color='black', width=1))
    ))
    # In Vivo Data (actual)
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='In Vivo Data (actual)',
        marker=dict(color=colors['in_vivo'], size=15, line=dict(color='black', width=1))
    ))
    # In Vivo Data (interpolated)
    fig.add_trace(go.Scatter(
        x=ti2, y=mm,
        mode='markers',
        name='In Vivo Data (interpolated)',
        marker=dict(color=colors['in_vivo'], size=8, line=dict(color='black', width=1))
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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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

def extract_plotly_data_for_table(fig):
    """Extract data from plotly figure and return as HTML table with each trace in separate columns"""

    # Extract axis titles
    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else 'X'
    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else 'Y'

    # Find the maximum length among all traces to determine table size
    max_length = 0
    trace_data = {}

    for trace in fig.data:
        if hasattr(trace, 'x') and hasattr(trace, 'y') and hasattr(trace, 'name') and trace.name is not None:
            trace_length = len(trace.x)
            max_length = max(max_length, trace_length)
            trace_data[trace.name] = {
                'x': list(trace.x),
                'y': list(trace.y)
            }

    if not trace_data:
        return "<p>No data available</p>"

    # Create a dictionary to hold all columns
    table_dfs = []

    # Add columns for each trace
    for trace_name, data in trace_data.items():
        # Pad shorter traces with NaN to match max_length
        x_data = data['x'] + [''] * (max_length - len(data['x']))
        y_data = data['y'] + [''] * (max_length - len(data['y']))

        # Create column names that include the trace name
        x_col_name = f"{trace_name} - {x_title}"
        y_col_name = f"{trace_name} - {y_title}"

        table_dfs.append(pd.DataFrame(data=np.array([x_data, y_data]).T, columns=[x_col_name, y_col_name]))

    # Create DataFrame with all traces as separate columns
    combined_df = pd.concat(table_dfs, axis=1)

    # Convert to HTML table with DataTables-compatible structure
    table_html = combined_df.to_html(
        classes='table table-striped',
        table_id='data-table',
        index=False,
        float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else '',
        escape=False,
        na_rep=''  # Display empty string for NaN values
    )

    return table_html


def create_plotly_a1(x, y, model_info, include_bands=True):
    # Create the scatter plot for the data
    trace_data = go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(color='black', size=15, line=dict(color='white', width=2)), zorder=10)

    # Create the smooth line for the model fit
    x_smooth = np.linspace(min(x), max(x), max(len(x),10))
    y_smooth = model_info['model_function'](x_smooth, **model_info['params'])
    trace_fit = go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f"{model_info['model_name']} Fit", line=dict(color='black', dash='dash', width=3), zorder=9)

    traces = [trace_data, trace_fit]

    # manual axis range
    x_range_data = [min(np.concatenate([x, x_smooth])), max(np.concatenate([x, x_smooth]))]
    y_range_data = [min(np.concatenate([y, y_smooth])), max(np.concatenate([y, y_smooth]))]
    # Add some padding (5% on each side)
    x_padding = (x_range_data[1] - x_range_data[0]) * 0.1
    y_padding = (y_range_data[1] - y_range_data[0]) * 0.1
    x_range = [x_range_data[0] - x_padding, x_range_data[1] + x_padding]
    y_range = [y_range_data[0] - y_padding, y_range_data[1] + y_padding]
    
    # Add prediction bands if requested and covariance matrix is available
    if include_bands and 'pcov' in model_info and model_info['pcov'] is not None:
        try:
            bands = generate_prediction_bands(
                model_info['model_function'], 
                x_smooth, 
                model_info['params'], 
                model_info['pcov'],
                model_info['residuals']
            )
            
            # Add confidence band
            trace_upper = go.Scatter(
                x=x_smooth, y=bands['upper'],
                mode='lines',
                line=dict(width=0),
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
                hoverinfo='skip',
                zorder=0
            )
            
            # Insert bands before the fit line for proper layering
            traces = [trace_data, trace_upper, trace_lower, trace_fit]
            
        except Exception as e:
            print(f"Warning: Could not generate prediction bands: {e}")

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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            range=y_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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
    # Write data to Excel file
    #output_filename = f'data_approach1_{model_info["model_name"]}.xlsx'
    #output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
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
    
    # Add prediction bands for dataset 1 (In Vitro) if requested and covariance matrix is available
    if include_bands and 'pcov1' in model_info and model_info['pcov1'] is not None:
        try:
            bands1 = generate_prediction_bands(
                model_info['model_function'], 
                x_smooth, 
                model_info['params1'], 
                model_info['pcov1'],
                model_info['residuals1']
            )
            
            # Add confidence band for dataset 1
            trace1_upper = go.Scatter(
                x=x_smooth, y=bands1['upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                zorder=0
            )
            
            trace1_lower = go.Scatter(
                x=x_smooth, y=bands1['lower'],
                mode='lines',
                fill='tonexty',
                fillcolor=hex_to_rgba(colors['in_vitro'],0.2),  # Blue with transparency
                #fillcolor='rgba(0,0,255,0.2)',  # Blue with transparency
                line=dict(width=0),
                name='95% Prediction Band (In Vitro)',
                hoverinfo='skip',
                zorder=0
            )
            
            traces.extend([trace1_upper, trace1_lower])
            
        except Exception as e:
            print(f"Warning: Could not generate prediction bands for dataset 1: {e}")

    # Add prediction bands for dataset 2 (In Vivo) if requested and covariance matrix is available
    if include_bands and 'pcov2' in model_info and model_info['pcov2'] is not None:
        try:
            bands2 = generate_prediction_bands(
                model_info['model_function'], 
                x_smooth, 
                model_info['params2'], 
                model_info['pcov2'],
                model_info['residuals2']
            )
            
            # Add confidence band for dataset 2
            trace2_upper = go.Scatter(
                x=x_smooth, y=bands2['upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                zorder=0
            )
            
            trace2_lower = go.Scatter(
                x=x_smooth, y=bands2['lower'],
                mode='lines',
                fill='tonexty',
                fillcolor=hex_to_rgba(colors['in_vivo'],0.2),  # Red with transparency
                line=dict(width=0),
                name='95% Prediction Band (In Vivo)',
                hoverinfo='skip',
                zorder=0
            )
            
            traces.extend([trace2_upper, trace2_lower])
            
        except Exception as e:
            print(f"Warning: Could not generate prediction bands for dataset 2: {e}")


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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            range=y_range,  # Manual range
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
            mirror=True
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
            linecolor='#EBF0F8',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#EBF0F8',
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
    # Write data to Excel file
    #output_filename = f'data_approach3_{ptype}.xlsx'
    #output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
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
        xaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True)
    )

    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    # Write data to Excel file
    #output_filename = f'prediction_approach1_{model_info["model_name"]}.xlsx'
    #output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    #output_path = ''
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
        xaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True)
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
            xaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True)
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
            xaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='#EBF0F8', mirror=True)
        )

    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))

    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    return {'plot': plot_html, 'table_html': table_html}

