from pprint import pprint
import io,re,os
import base64

import numpy as np
import scipy as sp
import pandas as pd

# for correct error propagation
import uncertainties as unc
import uncertainties.unumpy as unp

import traceback
import warnings
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import UndefinedMetricWarning

from models import models, approaches
from preprocessing import preprocessing_options
from utilities.evaluation import evaluate_goodness_of_fit, cross_validation_curve_fit, make_cross_validator, auto_grid_search, calculate_tau_with_uncertainty, generate_prediction_bands, generate_ratio_prediction_bands
from utilities.selection import create_model_selector_from_metrics, select_best_models
from utilities.prediction_validity import describe_tau_prediction_skip_reason
from utilities.model_display import get_human_readable_function
from utilities.misc import hex_to_rgba
from metrics import metrics, calculate_metric

import plotly.utils
import plotly.graph_objects as go
import plotly.io as pio
import json

# Determine environment
IS_PRODUCTION = 'PYTHONANYWHERE_DOMAIN' in os.environ
# Other configuration for PythonAnywhere
if IS_PRODUCTION:
    # PythonAnywhere-specific settings
    num_cores_for_grid_search = 1
    num_points_for_grid_search = 10
else:
    # Local development settings
    num_cores_for_grid_search = None
    num_points_for_grid_search = 200

#colors = {
#    'in_vitro': '#4A9FD1',      # Blue
#    'in_vivo': '#B85A9B',       # Purple-red
#    'in_vitro_2': '#6BB26E',  # Light sage green
#    'in_vivo_2':'#E8956A'   # Light peach-orange
#    }
colors = {
    'in_vitro': '#2E5F8A',      # Darker blue
    'in_vivo': '#8B3A6B',       # Darker purple-red
    'in_vitro_2': '#7BB8E8',    # Light blue
    'in_vivo_2': '#D487B8'      # Light purple-red
}

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
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state)
                # cross-validation
                cvs = cross_validation_curve_fit(x, y, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs_mean = {f'{metric}':np.nanmean(values[np.isfinite(values)]) for metric,values in cvs.items()}
                # fit final model
                model_result = models[model_name]['fit_model'](x, y, p0=p0)
                y_pred = model_result['predict'](x)
                # Evaluate goodness of fit
                #gof = evaluate_goodness_of_fit(y, y_pred)
                #gof = {metric: metrics[metric]['function'](y, y_pred) for metric in selected_metrics}
                gof = {metric: calculate_metric(metric, y, y_pred, len(p0)) for metric in selected_metrics}
                stats = pd.concat([pd.DataFrame(gof, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs_mean.items()}, index=[0])], axis=1)
                stats_table = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof.items()}, index=['Final model']), pd.DataFrame({metrics[metric]['display_name']:v for metric,v in cvs_mean.items()}, index=['Cross-validation'])], axis=0)
                #stats_table = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof.items()}, index=[0]), pd.DataFrame({'CV '+metrics[metric]['display_name']:v for metric,v in cvs_mean.items()}, index=[0])], axis=1)
                popt, pcov = list(model_result['params'].values()), model_result['pcov']
                upopt = unc.correlated_values(popt, pcov)
                upopt = dict(zip(model_result['params'].keys(),upopt))
                # outputs
                plot_image = create_plotly_a1(x, y, {
                    'model_name': models[model_name]['display_name'],
                    'model_function': models[model_name]['model_function'],
                    'params': model_result['params'],
                    'pcov': model_result['pcov'],  # Add this line
                    'residuals': y-y_pred,
                    'approach': approaches[approach_id]['display_name']
                })
                results[model_key] = {
                    'params': model_result['params'],
                    'uparams': upopt,
                    'pcov': model_result['pcov'],  # Add this line
                    'residuals': y-y_pred,
                    'stats': stats,
                    'predictions': y_pred,
                    'stats_table': stats_table.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}'),
                    'plot': plot_image
                }
            elif approach_id == 'approach2':
                cv = make_cross_validator(analysis_config, approach_id)
                # process data for this approach
                ## XXX
                if 0:
                    m = ~np.isnan(ti1)
                    x1,y1 = ti1[m], mm[m]
                    m = ~np.isnan(ti2)
                    x2,y2 = ti2[m], mm[m]
                if 1:
                    m = ~np.isnan(mi1)
                    x1,y1 = tt[m], mi1[m]
                    m = ~np.isnan(mi2)
                    x2,y2 = tt[m], mi2[m]
                ## dataset 1
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x1, y1, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state)
                # cross-validation
                cvs1 = cross_validation_curve_fit(x1, y1, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs1_mean = {f'{metric}':np.nanmean(v[np.isfinite(v)]) for metric,v in cvs1.items()}
                # fit final model
                model_result1 = models[model_name]['fit_model'](x1, y1, p0=p0)
                y_pred1 = model_result1['predict'](x1)
                # Evaluate goodness of fit
                #gof1 = {metric: metrics[metric]['function'](y1, y_pred1) for metric in selected_metrics}
                gof1 = {metric: calculate_metric(metric, y1, y_pred1, len(p0)) for metric in selected_metrics}
                stats1 = pd.concat([pd.DataFrame(gof1, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs1_mean.items()}, index=[0])], axis=1)
                stats_table1 = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof1.items()}, index=['Final model']), pd.DataFrame({metrics[metric]['display_name']:v for metric,v in cvs1_mean.items()}, index=['Cross-validation'])], axis=0)
                #stats_table1 = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof1.items()}, index=[0]), pd.DataFrame({'CV '+metrics[metric]['display_name']:v for metric,v in cvs1_mean.items()}, index=[0])], axis=1)
                #upopt1 = unc.correlated_values(popt1, pcov1)
                #tau1 = get_tau(modelname, upopt1)
                popt1, pcov1 = list(model_result1['params'].values()), model_result1['pcov']
                upopt1 = unc.correlated_values(popt1, pcov1)
                upopt1 = dict(zip(model_result1['params'].keys(),upopt1))
                tau1 = calculate_tau_with_uncertainty(models[model_name]['model_function'], popt1, pcov1)
                # deal with integration failure
                if tau1[0]>0 and np.isfinite(tau1[0]):
                    tau1 = unc.ufloat(*tau1)
                else:
                    tau1 = unc.ufloat(0,1)
                ## dataset 2
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x2, y2, param_min=grid_search_param_min, param_max=grid_search_param_max, 
                                          num_points=grid_search_num_points, num_cores=grid_search_num_cores, random_state=grid_search_random_state)
                # cross-validation
                cvs2 = cross_validation_curve_fit(x2, y2, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs2_mean = {f'{metric}':np.nanmean(v[np.isfinite(v)]) for metric,v in cvs2.items()}
                # fit final model
                model_result2 = models[model_name]['fit_model'](x2, y2, p0=p0)
                y_pred2 = model_result2['predict'](x2)
                # Evaluate goodness of fit
                #gof2 = {metric: metrics[metric]['function'](y2, y_pred2) for metric in selected_metrics}
                gof2 = {metric: calculate_metric(metric, y2, y_pred2, len(p0)) for metric in selected_metrics}
                stats2 = pd.concat([pd.DataFrame(gof2, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs2_mean.items()}, index=[0])], axis=1)
                stats_table2 = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof2.items()}, index=['Final model']), pd.DataFrame({metrics[metric]['display_name']:v for metric,v in cvs2_mean.items()}, index=['Cross-validation'])], axis=0)
                #stats_table2 = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof2.items()}, index=[0]), pd.DataFrame({'CV '+metrics[metric]['display_name']:v for metric,v in cvs2_mean.items()}, index=[0])], axis=1)
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
                results[model_key] = {
                    'params': [model_result1['params'],model_result2['params']],
                    'uparams': [upopt1,upopt2],
                    'tau': [tau1,tau2],
                    'stats': [stats1,stats2],
                    'predictions': [y_pred1,y_pred2],
                    'pcov': [pcov1,pcov2],
                    'residuals': [y1-y_pred1,y2-y_pred2],
                    'stats_table': [stats_table1.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}'), 
                                    stats_table2.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}')],
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
    
    return results

def invert_with_monte_carlo_parameters(func, params, param_cov, y_values, n_samples=1000, x_guess=None, full_output=False):
    """
    Use Monte Carlo sampling of parameters for error propagation
    """
    params = np.asarray(params)
    param_cov = np.asarray(param_cov)
    y_values = np.asarray(y_values)
    if x_guess is None:
        x_guess = np.ones_like(y_values)
    # Sample parameters from multivariate normal distribution
    param_samples = np.random.multivariate_normal(params, param_cov, n_samples)
    x_inverted = np.zeros_like(y_values)
    x_errors = np.zeros_like(y_values)
    if full_output:
        all_valid_samples = []
    for i, (y_val, x_init) in enumerate(zip(y_values, x_guess)):
        x_samples = np.zeros(n_samples)
        for j, param_sample in enumerate(param_samples):
            def equation(x):
                return func(x, *param_sample) - y_val
            try:
                x_samples[j] = sp.optimize.fsolve(equation, x_init)[0]
            except:
                x_samples[j] = np.nan
        # Remove failed inversions
        valid_samples = x_samples[~np.isnan(x_samples)]
        if len(valid_samples) > 0:
            #x_inverted[i] = np.mean(valid_samples)
            x_errors[i] = np.std(valid_samples)
        else:
            #x_inverted[i] = np.nan
            x_errors[i] = np.nan
        if full_output:
            all_valid_samples.append(valid_samples)
        # Get true inverted value rather than sampling
        def equation(x):
            return func(x, *params) - y_val
        try:
            x_inverted[i] = sp.optimize.fsolve(equation, x_init)[0]
        except:
            x_inverted[i] = np.nan
    if full_output:
        return x_inverted, x_errors, all_valid_samples
    else:
        return x_inverted, x_errors

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
#                            plot_image = create_prediction_plot_a2(data, t_pred, m_pred, m_pred_vivo, t_pred_plot, model1_pred_plot, model2_pred_plot, {
#                                'model_name': models[model_name]['display_name'],
#                                'approach': approaches[approach_id]['display_name']
#                            })
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
                                    #print(model_name,tau1,tau2,time_ratio,t_pred_vivo_tau,t_pred_vivo_err_tau)
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

                        if 0:
                            ## Rescale time by the ratio of times implied by each value, i.e., numerically solve for the time at each value
                            # Calculate model predictions for plotting
                            t_pred_plot = np.linspace(min(t_pred), max(t_pred), max(len(t_pred),50))
                            model1_pred_plot = models[model_name]['model_function'](t_pred_plot, **params1)  # in vitro model
                            model2_pred_plot = models[model_name]['model_function'](t_pred_plot, **params2)  # in vitro model
                            
                            # Determine times implied by values, including error propagation
                            t_pred_implied_vitro, t_pred_implied_vitro_err, all_valid_samples_vitro = invert_with_monte_carlo_parameters(models[model_name]['model_function'], np.array(list(params1.values())), 
                                                                                                                                         model_results['pcov'][0], m_pred, n_samples=1000, x_guess=None, full_output=True)
                            t_pred_implied_vivo, t_pred_implied_vivo_err, all_valid_samples_vivo = invert_with_monte_carlo_parameters(models[model_name]['model_function'], np.array(list(params2.values())), 
                                                                                                                                      model_results['pcov'][1], m_pred, n_samples=1000, x_guess=None, full_output=True)
                            ## XXX alternative error propagation through ratio...
#                            for sample_vitro, sample_vivo in zip(all_valid_samples_vitro, all_valid_samples_vivo):
#                                ratios = np.random.choice(sample_vivo,10000) / np.random.choice(sample_vitro,10000)
#                                ratios = ratios[np.isfinite(ratios)]
#                                print(np.nanmean(ratios), np.nanquantile(ratios, [0.05,0.95]))

                            # Calculate the time scaling ratio
                            time_ratio = t_pred_implied_vivo / t_pred_implied_vitro
                            time_ratio_err = ((t_pred_implied_vitro_err/t_pred_implied_vitro)**2 + (t_pred_implied_vivo_err/t_pred_implied_vivo)**2)**0.5 # relative error

#                            print('XXXXXX', model_name)
#                            print(m_pred)
#                            print(t_pred_implied_vitro, t_pred_implied_vivo, time_ratio)

                            # Apply time scaling to get predicted in vivo times
                            t_pred_vivo = t_pred * time_ratio
                            t_pred_vivo_err = t_pred_vivo * time_ratio_err

                            # Create prediction plot
                            plot_image = create_prediction_plot_a2(data, t_pred, m_pred, t_pred_vivo, t_pred_plot, model1_pred_plot, model2_pred_plot, t_pred_vivo_err, {
                                'model_name': models[model_name]['display_name'],
                                'approach': approaches[approach_id]['display_name'],
                                'model_function': models[model_name]['model_function'],
                                'params': model_results['params'],
                                'pcov': model_results['pcov'],
                                'residuals': model_results['residuals']
                            })

                            ## Rescale by tau ratio, if available
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
                                print(model_name,tau1,tau2,time_ratio,t_pred_vivo_tau,t_pred_vivo_err_tau)
                                # Create prediction plot
                                plot_image_tau = create_prediction_plot_a2(data, t_pred, m_pred, t_pred_vivo_tau, t_pred_plot, model1_pred_plot, model2_pred_plot, t_pred_vivo_err_tau, {
                                    'model_name': models[model_name]['display_name'],
                                    'approach': approaches[approach_id]['display_name'],
                                    'model_function': models[model_name]['model_function'],
                                })
                            except Exception as e:
                                print(e)
                                t_pred_vivo_tau = None
                                plot_image_tau = None

                        approach_predictions[model_name] = {
#                            'predictions': {
#                                'in_vitro_time': t_pred, 
#                                'in_vitro_value': m_pred, 
#                                'predicted_in_vivo_time': t_pred_vivo,
#                                'predicted_in_vivo_time_tau': t_pred_vivo_tau
#                            },
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
    n_models = len(model_keys)
    if approach == 'approach1':
        ks_matrix = np.zeros((n_models, n_models))
        np.fill_diagonal(ks_matrix, 1)
        for i in range(n_models):
            for j in range(i+1, n_models):
                try:
                    _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'], results[model_keys[j]]['predictions'])
                except Exception as e:
                    print(f"Error processing prediction: {e}")
                    p_value = np.nan
                ks_matrix[i, j] = ks_matrix[j, i] = p_value
        df_ks = pd.DataFrame(ks_matrix, index=model_display_names, columns=model_display_names)
        ks_table = df_ks.to_html(classes='table table-striped', float_format=lambda x: f'{x:.4f}')
        ks_tables = [ks_table]
    elif approach == 'approach2':
        ks_tables = []
        for r in [1,2]:
            ks_matrix = np.zeros((n_models, n_models))
            np.fill_diagonal(ks_matrix, 1)
            for i in range(n_models):
                for j in range(i+1, n_models):
                    try:
                        _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'][r-1], results[model_keys[j]]['predictions'][r-1])
                    except Exception as e:
                        print(f"Error processing prediction: {e}")
                        p_value = np.nan
                    ks_matrix[i, j] = ks_matrix[j, i] = p_value
            df_ks = pd.DataFrame(ks_matrix, index=model_display_names, columns=model_display_names)
            ks_table = df_ks.to_html(classes='table table-striped', float_format=lambda x: f'{x:.4f}')
            ks_tables.append(ks_table)
    elif approach == 'approach3':
        ks_tables = [None]

    return metrics_table, ks_tables

    
def create_plotly_table(df):
    numeric_cols = df.select_dtypes(include='number').columns
    formats = ['.4f' if c in numeric_cols else None for c in df.columns]
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#f0f0f0',  # Light gray for header
            align='left',
            font=dict(color='black', size=12),
            line_color='#d9d9d9',  # Slightly darker gray for borders
            height=40
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=['#ffffff', '#f9f9f9'],  # Alternating white and very light gray
            align='left',
            font=dict(color='black', size=11),
            line_color='#d9d9d9',  # Slightly darker gray for borders
            height=30,
            format=formats  # Assuming first column is string (model/approach name)
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        #width='100%'  # This makes the table responsive
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
    # Write data to Excel file
    #output_filename = 'data_interpolation.xlsx'
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

#def extract_plotly_data_for_table(fig):
#    """Extract data from plotly figure and return as HTML table using pandas"""
#    data_list = []
#
#    # Extract axis titles
#    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else 'X'
#    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else 'Y'
#
#    for trace in fig.data:
#        if hasattr(trace, 'x') and hasattr(trace, 'y') and hasattr(trace, 'name') and trace.name is not None:
#            # Create a DataFrame for this trace
#            trace_df = pd.DataFrame({
#                'Series': trace.name,
#                x_title: trace.x,
#                y_title: trace.y
#            })
#            data_list.append(trace_df)
#
#    # Combine all traces into one DataFrame
#    if data_list:
#        combined_df = pd.concat(data_list, ignore_index=True)
#
#        # Convert to HTML table with DataTables-compatible structure
#        table_html = combined_df.to_html(
#            classes='table table-striped',
#            table_id='data-table',
#            index=False,
#            float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else '',
#            escape=False
#        )
#
#        return table_html
#    else:
#        return "<p>No data available</p>"

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

        #table_dict[x_col_name] = x_data
        #table_dict[y_col_name] = y_data
        table_dfs.append(pd.DataFrame(data=np.array([x_data, y_data]).T, columns=[x_col_name, y_col_name]))

    # Create DataFrame with all traces as separate columns
    #combined_df = pd.DataFrame(table_dict)
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
        #width=600,
        #height=600*0.5,
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
    # Write data to Excel file
    #output_filename = f'data_approach2_{model_info["model_name"]}.xlsx'
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
        #width=600,
        #height=600*0.5,
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

    # Connection lines to show time scaling
#    for i in range(len(t_pred)):
#        fig.add_trace(go.Scatter(
#            x=[t_pred[i], t_pred_vivo[i]],
#            y=[m_pred[i], m_pred[i]],
#            mode='lines',
#            line=dict(color='gray', width=2, dash='dot'),
#            showlegend=False,
#            hoverinfo='skip'
#        ))

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
#def create_prediction_plot_a2(data, t_pred, m_pred, m_pred_vivo, t_pred_plot, model1_pred_plot, model2_pred_plot, model_info, include_bands=True):
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
                if 0:
                    # Generate prediction bands for the predicted times
                    bands1 = generate_prediction_bands(
                        model_info['model_function'], 
                        t_pred,  # Use the actual prediction times
                        model_info['params'][0], 
                        model_info['pcov'][0],
                        model_info['residuals'][0]
                    )
                    bands2 = generate_prediction_bands(
                        model_info['model_function'], 
                        t_pred,  # Use the actual prediction times
                        model_info['params'][1], 
                        model_info['pcov'][1],
                        model_info['residuals'][1]
                    )

                    # XXX
                    # Estimate relative error in value ratio and multiply by time value to get estimate of absolute time error
                    #relative_error = np.sqrt((bands1['std']/(bands1['mean'].max()-bands1['mean'].min()))**2 + (bands2['std']/(bands2['mean'].max()-bands2['mean'].min()))**2)
                    relative_error = np.sqrt((bands1['std']/bands1['mean'][0])**2 + (bands2['std']/bands2['mean'][0])**2)
                    #error = relative_error * (np.max(t_pred_vivo)-np.min(t_pred_vivo))
                    error = m_pred_vivo * relative_error
                    # For confidence intervals, assume t-distribution (more conservative)
                    # Use the smaller degrees of freedom if available, otherwise use normal approximation
                    t_value = 1.96  # 95% CI for normal distribution
                    margin_of_error = t_value * error
                    
                    bands = {
                        'mean': m_pred_vivo,
                        'std': error,
                        'lower': m_pred_vivo - margin_of_error,
                        'upper': m_pred_vivo + margin_of_error
                    }
                    #print(bands)
                    #print(bands1)
                    #print(bands2)
                    #print(m_pred_vivo)
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

    # Write data to Excel file
    #output_filename = f'prediction_approach2_{model_info["model_name"]}.xlsx'
    #output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
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

    # Write data to Excel file
    #output_filename = 'prediction_approach3.xlsx'
    #output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    # Extract data as HTML table
    table_html = extract_plotly_data_for_table(fig)

    config = {'responsive': True, 'displaylogo': False}
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    return {'plot': plot_html, 'table_html': table_html}

