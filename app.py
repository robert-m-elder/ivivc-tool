import io,re,os
import base64
from pprint import pprint

from flask import Flask, render_template, request, jsonify
from flask import send_from_directory, send_file

import numpy as np
import scipy as sp
import pandas as pd
#import sklearn
#import sklearn.model_selection
# for correct error propagation
import uncertainties as unc
import uncertainties.unumpy as unp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import traceback
import warnings
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import UndefinedMetricWarning

from models import models, approaches
from preprocessing import preprocessing_options
from utilities.evaluation import evaluate_goodness_of_fit, cross_validation_curve_fit, cross_validation_schemes, auto_grid_search, calculate_tau, calculate_tau_with_uncertainty
from utilities.model_display import get_human_readable_function
from metrics import metrics

import plotly.utils
import plotly.graph_objects as go
import plotly.io as pio
import json

app = Flask(__name__)

# Set the default output directory
DEFAULT_OUTPUT_DIR = os.path.join(app.root_path, 'output')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        selected_models = request.form.getlist('models')
        selected_approaches = request.form.getlist('approaches')
        selected_normalizations = request.form.getlist('normalizations')
        selected_scalings = request.form.getlist('scalings')
        selected_interpolation = request.form.getlist('interpolation')
        selected_metrics = request.form.getlist('metrics')

        # Get the output directory from the form, or use the default
        output_directory = request.form.get('output_directory', DEFAULT_OUTPUT_DIR)
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        human_readable_functions = {model_name:get_human_readable_function(models[model_name]['model_function']) for model_name in set([m.split(':')[-1] for m in selected_models])}

        # add placeholder for approach3
        if 'approach3' in selected_approaches:
            selected_models.append('approach3')

        if not file or not selected_models or not selected_approaches or not selected_metrics:
            return "Please upload a file, select at least one model, one approach, and one metric", 400

        print(selected_models)
        print(selected_approaches)
        print(selected_scalings)
        print(selected_normalizations)
        print(selected_interpolation)
        print(selected_metrics)

        # Read data
        filename = file.filename
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(file)
        else:
            sheet_name = request.form.get('sheet')
            print(sheet_name)
            if not sheet_name:
                return "Please select a sheet for Excel files", 400
            df = pd.read_excel(file, sheet_name=sheet_name)
        #df = pd.read_excel(file, sheet_name=sheet_name)
        t1,m1,t2,m2 = df.values.T
        
        # Apply preprocessing
        data = preprocess_data(t1, m1, t2, m2, selected_interpolation=selected_interpolation, selected_scalings=selected_scalings, selected_normalizations=selected_normalizations)
        t1_scale,m1_scale,t2_scale,m2_scale,mm,tt,ti1,ti2,mi1,mi2 = data
        ## TODO add before/after scaling/normalization plot
        #interpolation_plot = create_interpolation_plot(t1_scale,m1_scale,t2_scale,m2_scale,tt,mi1,mi2)
        interpolation_info = create_interpolation_plotly(t1_scale, m1_scale, t2_scale, m2_scale, tt, mi1, mi2, output_directory)

        # Apply models
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = process_data(data, selected_models, selected_approaches, selected_metrics, output_directory)
        for model_key in results:
            # special for approach3
            if len(model_key.split(':'))>1:
                model_name = model_key.split(':')[-1]
                results[model_key]['function'] = human_readable_functions[model_name]
        #pprint(results.keys())
        #print(results['approach3'])
#        for model_key in results:
#            if 'approach2' in model_key:
#                print(results[model_key])
#                pprint(results[model_key]['params'])
#                pprint(results[model_key]['uparams'])
#                pprint(results[model_key]['tau'])
        comparisons, ks_tables = {}, {}
        for approach in selected_approaches:
            comparisons[approach], ks_tables[approach] = create_comparison(results, approach)
        return render_template('results.html', results=results, models=models, interpolation_info=interpolation_info, comparisons=comparisons, ks_tables=ks_tables, 
                               approaches=approaches, selected_approaches=selected_approaches, 
                               selected_scalings=selected_scalings, selected_normalizations=selected_normalizations, selected_interpolation=selected_interpolation,
                               metrics=metrics, selected_metrics=selected_metrics, output_directory=output_directory)
    default_interpolation = 'default_interpolation'
    return render_template('index.html', models=models, approaches=approaches, preprocessing_options=preprocessing_options, 
                           default_interpolation=default_interpolation, metrics=metrics, output_directory=DEFAULT_OUTPUT_DIR)

@app.route('/get_sheets', methods=['POST'])
def get_sheets():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            xls = pd.ExcelFile(file)
            sheets = xls.sheet_names
            return jsonify({'sheets': sheets})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/download/<path:filename>')
def download_excel(filename):
    base_dir = os.path.abspath(os.sep)  # Root directory
    full_path = os.path.normpath(os.path.join(base_dir, filename))
    return send_file(full_path, as_attachment=True)

def preprocess_data(t1, m1, t2, m2, selected_interpolation=None, selected_scalings=None, selected_normalizations=None):
    # Apply basic data cleaning
    # set zero to small value to avoid division errors
    t1[t1==0] = 1e-2; t2[t2==0] = 1e-2
    # remove missing values due to unequal number of points in spreadsheet
    mask = ~pd.isna(m2); t2,m2 = t2[mask], m2[mask]
    mask = ~pd.isna(m1); t1,m1 = t1[mask], m1[mask]

    # Apply normalizations
    for norm in selected_normalizations:
        m1 = preprocessing_options['normalization'][norm](m1)
        m2 = preprocessing_options['normalization'][norm](m2)

    # Apply scalings
    for scale in selected_scalings:
        if scale == 'log_x':
            t1 = preprocessing_options['scaling'][scale](t1)
            t2 = preprocessing_options['scaling'][scale](t2)
        elif scale == 'log_y':
            m1 = preprocessing_options['scaling'][scale](m1)
            m2 = preprocessing_options['scaling'][scale](m2)

    # Apply interpolation
    for interp in selected_interpolation:
        data = preprocessing_options['interpolation'][interp](t1, m1, t2, m2, N_interp=10)

    return data

def process_data(data, selected_models, selected_approaches, selected_metrics, output_directory):
    t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2 = data
    
    results = {}
    
    for model_key in selected_models:
        if len(model_key.split(':'))>1:
            approach_id, model_name = model_key.split(':')
        else:
            approach_id = model_key
        
        try:
            if approach_id == 'approach1':
                cv = cross_validation_schemes[approach_id]
                # process data for this approach
                m = ~np.isnan(ti1) & ~np.isnan(ti2)
                x,y = ti1[m], ti2[m]
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x, y, param_min=1e-6, param_max=1e6)
                # cross-validation
                cvs = cross_validation_curve_fit(x, y, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs_mean = {f'{metric}':values.mean() for metric,values in cvs.items()}
                # fit final model
                model_result = models[model_name]['fit_model'](x, y, p0=p0)
                y_pred = model_result['predict'](x)
                # Evaluate goodness of fit
                #gof = evaluate_goodness_of_fit(y, y_pred)
                gof = {metric: metrics[metric]['function'](y, y_pred) for metric in selected_metrics}
                stats = pd.concat([pd.DataFrame(gof, index=[0]), pd.DataFrame({'CV '+metric:v for metric,v in cvs_mean.items()}, index=[0])], axis=1)
                stats_table = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof.items()}, index=['Final model']), pd.DataFrame({metrics[metric]['display_name']:v for metric,v in cvs_mean.items()}, index=['Cross-validation'])], axis=0)
                #stats_table = pd.concat([pd.DataFrame({metrics[metric]['display_name']:v for metric,v in gof.items()}, index=[0]), pd.DataFrame({'CV '+metrics[metric]['display_name']:v for metric,v in cvs_mean.items()}, index=[0])], axis=1)
                # outputs
                plot_image = create_plotly_a1(x, y, {
                    'model_name': models[model_name]['display_name'],
                    'model_function': models[model_name]['model_function'],
                    'params': model_result['params'],
                    'approach': approaches[approach_id]['display_name']
                }, output_directory)
                results[model_key] = {
                    'params': model_result['params'],
                    'stats': stats,
                    'predictions': y_pred,
                    'stats_table': stats_table.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}'),
                    'plot': plot_image
                }
            elif approach_id == 'approach2':
                cv = cross_validation_schemes[approach_id]
                # process data for this approach
                m = ~np.isnan(ti1)
                x1,y1 = ti1[m], mm[m]
                m = ~np.isnan(ti2)
                x2,y2 = ti2[m], mm[m]
                ## dataset 1
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x1, y1, param_min=1e-6, param_max=1e6)
                # cross-validation
                cvs1 = cross_validation_curve_fit(x1, y1, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs1_mean = {f'{metric}':v.mean() for metric,v in cvs1.items()}
                # fit final model
                model_result1 = models[model_name]['fit_model'](x1, y1, p0=p0)
                y_pred1 = model_result1['predict'](x1)
                # Evaluate goodness of fit
                gof1 = {metric: metrics[metric]['function'](y1, y_pred1) for metric in selected_metrics}
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
                if tau1[0]>0:
                    tau1 = unc.ufloat(*tau1)
                else:
                    tau1 = unc.ufloat(0,1)
                ## dataset 2
                # get rough initial estimate of parameters
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p0 = auto_grid_search(models[model_name]['model_function'], x2, y2, param_min=1e-6, param_max=1e6)
                # cross-validation
                cvs2 = cross_validation_curve_fit(x2, y2, models[model_name]['model_function'], cv, selected_metrics, p0=p0, kwargs=models[model_name]['fit_kwargs'])
                cvs2_mean = {f'{metric}':v.mean() for metric,v in cvs2.items()}
                # fit final model
                model_result2 = models[model_name]['fit_model'](x2, y2, p0=p0)
                y_pred2 = model_result2['predict'](x2)
                # Evaluate goodness of fit
                gof2 = {metric: metrics[metric]['function'](y2, y_pred2) for metric in selected_metrics}
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
                if tau2[0]>0:
                    tau2 = unc.ufloat(*tau2)
                else:
                    tau2 = unc.ufloat(0,1)
                # outputs
                plot_image = create_plotly_a2(x1, y1, x2, y2, {
                    'model_name': models[model_name]['display_name'],
                    'model_function': models[model_name]['model_function'],
                    'params1': model_result1['params'],
                    'params2': model_result2['params'],
                    'approach': approaches[approach_id]['display_name']
                }, output_directory)
                results[model_key] = {
                    'params': [model_result1['params'],model_result2['params']],
                    'uparams': [upopt1,upopt2],
                    'tau': [tau1,tau2],
                    'stats': [stats1,stats2],
                    'predictions': [y_pred1,y_pred2],
                    'stats_table': [stats_table1.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}'), 
                                    stats_table2.to_html(classes='table table-striped', index=True, float_format=lambda x: f'{x:.4f}')],
                    'plot': plot_image
                }
            elif approach_id == 'approach3':
                #plot_image_v = create_plot_a3(tt,mi1,mi2,ptype='v')
                #plot_image_t = create_plot_a3(mm,ti1,ti2,ptype='t')
                plot_image_v = create_plotly_a3(tt,mi1,mi2,output_directory,ptype='v')
                plot_image_t = create_plotly_a3(mm,ti1,ti2,output_directory,ptype='t')
                results[model_key] = {'plot_v': plot_image_v, 'plot_t': plot_image_t}
        except Exception as e:
            # If an error occurs, store the error message
            results[model_key] = {'error': str(e)}
            traceback.print_exc()
    
    return results

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
                                metric = k[3:]
                            else:
                                metric = k
                            tmp_comparison_data.update({f'{metrics[metric]["display_name"]}':v})
                        comparison_data.append(tmp_comparison_data)
                    if approach == 'approach2':
                        for i in [1,2]:
                            tmp_comparison_data = {'Model': models[model_name]['display_name'], 'Dataset':f'Dataset{i}'}
                            for k,v in model_results['stats'][i-1].iloc[0].items():
                                if k.startswith('CV '):
                                    metric = k[3:]
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
                _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'], results[model_keys[j]]['predictions'])
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
                    _, p_value = sp.stats.ks_2samp(results[model_keys[i]]['predictions'][r-1], results[model_keys[j]]['predictions'][r-1])
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

def create_interpolation_plot(t1,m1,t2,m2,tt,mi1,mi2):
    plt.figure(figsize=(6, 3.6))
    plt.plot(t1,m1,'ro',ms=15,mew=1,mec='k',label='Dataset 1 (actual)')
    plt.plot(tt,mi1,'ro',ms=8,mew=1,mec='k',mfc='r',label='Dataset 1 (interpolated)')
    plt.plot(t2,m2,'bo',ms=15,mew=1,mec='k',label='Dataset 2 (actual)')
    plt.plot(tt,mi2,'bo',ms=8,mew=1,mec='k',mfc='b',label='Dataset 2 (interpolated)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=12)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    interpolation_plot = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return interpolation_plot

def create_interpolation_plotly(t1, m1, t2, m2, tt, mi1, mi2, output_directory):
    fig = go.Figure()
    # Dataset 1 (actual)
    fig.add_trace(go.Scatter(
        x=t1, y=m1,
        mode='markers',
        name='Dataset 1 (actual)',
        marker=dict(color='red', size=15, line=dict(color='black', width=1))
    ))
    # Dataset 1 (interpolated)
    fig.add_trace(go.Scatter(
        x=tt, y=mi1,
        mode='markers',
        name='Dataset 1 (interpolated)',
        marker=dict(color='red', size=8, line=dict(color='black', width=1))
    ))
    # Dataset 2 (actual)
    fig.add_trace(go.Scatter(
        x=t2, y=m2,
        mode='markers',
        name='Dataset 2 (actual)',
        marker=dict(color='blue', size=15, line=dict(color='black', width=1))
    ))
    # Dataset 2 (interpolated)
    fig.add_trace(go.Scatter(
        x=tt, y=mi2,
        mode='markers',
        name='Dataset 2 (interpolated)',
        marker=dict(color='blue', size=8, line=dict(color='black', width=1))
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
    output_filename = 'data_interpolation.xlsx'
    output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'filename':output_path}

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

def create_plotly_a1(x, y, model_info, output_directory):
    # Create the scatter plot for the data
    trace_data = go.Scatter(x=x, y=y, mode='markers', name='Data', marker=dict(color='white', size=10, line=dict(color='gray', width=1)))

    # Create the smooth line for the model fit
    x_smooth = np.linspace(min(x), max(x), 200)
    y_smooth = model_info['model_function'](x_smooth, **model_info['params'])
    trace_fit = go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f"{model_info['model_name']} Fit", line=dict(color='black', dash='dash', width=3))

    # Create the figure and add the traces
    fig = go.Figure(data=[trace_data, trace_fit])

    # Create the layout
    fig.update_layout(
        template='plotly_white',
        autosize=True, 
        title=f"Time Series Data with {model_info['model_name']} Fit",
        xaxis_title='Time (Dataset 1)',
        yaxis_title='Time (Dataset 2)',
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
    output_filename = f'data_approach1_{model_info["model_name"]}.xlsx'
    output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'filename':output_path}

def create_plot_a1(x, y, model_info):
    plt.figure(figsize=(6, 3.6))
    plt.plot(x, y, 'ko', mec='0.5', mew=1, mfc='w', ms=10, label='Data')

    x_smooth = np.linspace(min(x), max(x), 200)
    y_smooth = model_info['model_function'](x_smooth, **model_info['params'])
    plt.plot(x_smooth, y_smooth, 'k--', lw=3, label=f"{model_info['model_name']} Fit")

    plt.title(f"Time Series Data with {model_info['model_name']} Fit")
    plt.xlabel('Time (Dataset 1)')
    plt.ylabel('Time (Dataset 2)')
    plt.legend()

    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)

    # Encode the image to base64
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close()  # Close the plot to free up memory

    return plot_base64

def create_plot_a2(x1, y1, x2, y2, model_info):
    plt.figure(figsize=(6, 3.6))
    plt.plot(x1, y1, 'bo', mec=(0.5,0.5,1.0), mew=1, mfc='w', ms=10, label='Dataset 1')
    plt.plot(x2, y2, 'ro', mec=(1.0,0.5,0.5), mew=1, mfc='w', ms=10, label='Dataset 2')
    x_smooth = np.linspace(min(np.concatenate([x1,x2])), max(np.concatenate([x1,x2])), 200)
    y_smooth1 = model_info['model_function'](x_smooth, **model_info['params1'])
    y_smooth2 = model_info['model_function'](x_smooth, **model_info['params2'])
    plt.plot(x_smooth, y_smooth1, 'b--', lw=3, label=f"{model_info['model_name']} Fit")
    plt.plot(x_smooth, y_smooth2, 'r--', lw=3, label=f"{model_info['model_name']} Fit")
    plt.title(f"Time Series Data with {model_info['model_name']} Fit")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)

    # Encode the image to base64
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close()  # Close the plot to free up memory

    return plot_base64

def create_plotly_a2(x1, y1, x2, y2, model_info, output_directory):
    # Create traces for the datasets
    trace1 = go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='Dataset 1',
        marker=dict(
            color='white',
            size=10,
            line=dict(
                color='rgba(0.5,0.5,1.0,1)',
                width=1
            ),
            symbol='circle'
        )
    )

    trace2 = go.Scatter(
        x=x2, y=y2,
        mode='markers',
        name='Dataset 2',
        marker=dict(
            color='white',
            size=10,
            line=dict(
                color='rgba(1.0,0.5,0.5,1)',
                width=1
            ),
            symbol='circle'
        )
    )

    # Create smooth lines for model fits
    x_smooth = np.linspace(min(np.concatenate([x1,x2])), max(np.concatenate([x1,x2])), 200)
    y_smooth1 = model_info['model_function'](x_smooth, **model_info['params1'])
    y_smooth2 = model_info['model_function'](x_smooth, **model_info['params2'])

    trace3 = go.Scatter(
        x=x_smooth, y=y_smooth1,
        mode='lines',
        name=f"{model_info['model_name']} Fit (Dataset 1)",
        line=dict(color='blue', dash='dash', width=3)
    )

    trace4 = go.Scatter(
        x=x_smooth, y=y_smooth2,
        mode='lines',
        name=f"{model_info['model_name']} Fit (Dataset 2)",
        line=dict(color='red', dash='dash', width=3)
    )

    # Create the layout
    layout = dict(
        template='plotly_white',
        autosize=True, 
        title=f"Time Series Data with {model_info['model_name']} Fit",
        xaxis_title='Time (Dataset 1)',
        yaxis_title='Time (Dataset 2)',
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
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

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
    output_filename = f'data_approach2_{model_info["model_name"]}.xlsx'
    output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'filename':output_path}

def create_plot_a3(x,y1,y2,ptype='v'):
    #def create_plot_a3(tt,mi1,mi2,ptype='v'):
    plt.figure(figsize=(6, 3.6))
    #plt.plot(tt,(mi1-mi2),'ko',mec='k',mew=1,mfc='w',ms=10); 
    if ptype == 'v':
        plt.plot(x,(y1-y2),'ko',mec='k',mew=1,mfc='w',ms=10); 
        plt.xlabel('Time')
        plt.ylabel('Dataset 1 Value - Dataset 2 Value')
    elif ptype == 't':
        plt.plot(x,(y1/y2),'ko',mec='k',mew=1,mfc='w',ms=10); 
        plt.xlabel('Value')
        plt.ylabel('Dataset 1 Time / Dataset 2 Time')
    #plt.legend(fontsize=12)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_base64

def create_plotly_a3(x, y1, y2, output_directory, ptype='v'):
    if ptype == 'v':
        y = y1 - y2
        x_label = 'Time'
        y_label = 'Dataset 1 Value - Dataset 2 Value'
        plot_name = 'Value vs. time'
    elif ptype == 't':
        y = y1 / y2
        x_label = 'Value'
        y_label = 'Dataset 1 Time / Dataset 2 Time'
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
            color='white',
            size=10,
            line=dict(
                color='black',
                width=1
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
    output_filename = f'data_approach3_{ptype}.xlsx'
    output_path = write_plotly_data_to_excel(fig, output_filename, output_directory)
    # Configure the plot for download options
    config = {
        'responsive': True,
        'displaylogo': False,
    }
    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    return {'plot':plot_html, 'filename':output_path}

if __name__ == '__main__':
    app.run(debug=True)

