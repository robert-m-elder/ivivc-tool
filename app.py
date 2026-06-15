import os
import traceback
import warnings

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import UndefinedMetricWarning

from models import approaches, models
from preprocessing import preprocessing_options
from utilities.descriptions import (
    APPROACH_DESCRIPTIONS,
    GRID_SEARCH_DESCRIPTION_HTML,
    TOOL_PURPOSE_HTML,
    METRIC_DESCRIPTIONS,
    PREPROCESSING_DESCRIPTIONS,
    format_analysis_config_rows,
    format_metric_description_rows,
    format_preprocessing_rows,
    format_selected_preprocessing_summary,
)
from utilities.reporting import build_final_model_options
from utilities.word_report import build_word_report, safe_report_filename, WORD_MIME_TYPE
from utilities.model_display import get_human_readable_function
from metrics import metrics
from engine import (
    create_comparison,
    create_initial_plotly,
    create_interpolation_plotly,
    create_prediction_interpolation_plotly,
    preprocess_data,
    process_data,
    process_predictions,
)

app = Flask(__name__)


@app.route('/user-guide')
def user_guide():
    """Serve the generated user guide PDF, if present."""
    return send_from_directory(
        os.path.join(app.root_path, 'docs'),
        'user_guide.pdf',
        as_attachment=False,
    )

# Determine environment
IS_PRODUCTION = 'PYTHONANYWHERE_DOMAIN' in os.environ
# Basic configuration
app.config['DEBUG'] = not IS_PRODUCTION
app.config['ENV'] = 'production' if IS_PRODUCTION else 'development'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MiB

def _form_int(name, default):
    value = request.form.get(name, '')
    return default if value == '' else int(value)

def _form_float(name, default):
    value = request.form.get(name, '')
    return default if value == '' else float(value)

def parse_analysis_config():
    default_grid_points = 10 if IS_PRODUCTION else 200
    default_grid_cores = 1 if IS_PRODUCTION else None
    grid_cores_raw = request.form.get('grid_search_num_cores', '')

    return {
        'cv_scheme': request.form.get('cv_scheme', 'shuffle_split'),
        'cv_n_splits': _form_int('cv_n_splits', 20),
        'cv_test_size': _form_float('cv_test_size', 0.25),
        'cv_random_state': _form_int('cv_random_state', 12345),
        'grid_search_num_points': _form_int('grid_search_num_points', default_grid_points),
        'grid_search_num_cores': default_grid_cores if grid_cores_raw == '' else int(grid_cores_raw),
        'grid_search_param_min': _form_float('grid_search_param_min', -1e6),
        'grid_search_param_max': _form_float('grid_search_param_max', 1e6),
        'grid_search_random_state': _form_int('grid_search_random_state', 12345),
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        prediction_file = request.files.get('prediction_file')
        selected_models = request.form.getlist('models')
        selected_approaches = request.form.getlist('approaches')
        selected_normalizations = request.form.getlist('normalizations')
        selected_scalings = request.form.getlist('scalings')
        selected_interpolation = request.form.getlist('interpolation')
        selected_metrics = request.form.getlist('metrics')
        analysis_config = parse_analysis_config()

        human_readable_functions = {model_name:get_human_readable_function(models[model_name]['model_function']) for model_name in set([m.split(':')[-1] for m in selected_models])}

        # add placeholder for approach3
        if 'approach3' in selected_approaches:
            selected_models.append('approach3')

        if not file or not selected_models or not selected_approaches or not selected_metrics:
            return "Please upload a file, select at least one model, one approach, and one metric", 400

        # Read data
        filename = file.filename
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(file)
        else:
            sheet_name = request.form.get('sheet')
            if not sheet_name:
                return "Please select a sheet for Excel files", 400
            df = pd.read_excel(file, sheet_name=sheet_name)
        t1,m1,t2,m2 = df.values.T
        
        # Apply preprocessing
        data = preprocess_data(t1, m1, t2, m2, selected_interpolation=selected_interpolation, selected_scalings=selected_scalings, selected_normalizations=selected_normalizations)
        t1_scale,m1_scale,t2_scale,m2_scale,mm,tt,ti1,ti2,mi1,mi2 = data

        raw_data_info = create_initial_plotly(t1, m1, t2, m2)
        interpolation_info = create_interpolation_plotly(t1_scale, m1_scale, t2_scale, m2_scale, tt, mi1, mi2)

        # Apply models
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = process_data(data, selected_models, selected_approaches, selected_metrics, analysis_config=analysis_config)
        for model_key in results:
            # special for approach3
            if len(model_key.split(':'))>1:
                model_name = model_key.split(':')[-1]
                results[model_key]['function'] = human_readable_functions[model_name]

        # Compare model performance
        comparisons, ks_tables = {}, {}
        for approach in selected_approaches:
            comparisons[approach], ks_tables[approach] = create_comparison(results, approach)

        analysis_config_rows = format_analysis_config_rows(analysis_config)
        preprocessing_rows = format_preprocessing_rows(
            selected_normalizations,
            selected_scalings,
            selected_interpolation,
        )
        preprocessing_summary = format_selected_preprocessing_summary(
            selected_normalizations,
            selected_scalings,
            selected_interpolation,
        )
        metric_description_rows = format_metric_description_rows(metrics, selected_metrics)

        # Process predictions if prediction file is provided
        prediction_results, prediction_interpolation_info = {}, {}
        has_prediction_dataset = bool(prediction_file and prediction_file.filename)
        if has_prediction_dataset:
            try:
                # Read prediction data
                pred_filename = prediction_file.filename
                pred_file_extension = pred_filename.rsplit('.', 1)[1].lower()
                
                if pred_file_extension == 'csv':
                    pred_df = pd.read_csv(prediction_file)
                else:
                    pred_sheet_name = request.form.get('prediction_sheet')
                    if not pred_sheet_name:
                        return "Please select a sheet for prediction Excel files", 400
                    pred_df = pd.read_excel(prediction_file, sheet_name=pred_sheet_name)
                
                # Assuming prediction file has 2 columns: time, value
                t_pred, m_pred = pred_df.values.T[:2]
                
                prediction_data = preprocess_data(t_pred, m_pred, None, None, selected_interpolation=selected_interpolation, selected_scalings=selected_scalings, selected_normalizations=selected_normalizations)

                prediction_interpolation_info = create_prediction_interpolation_plotly(*prediction_data)

                prediction_results = process_predictions(data, prediction_data, results, selected_approaches, interpolated_data=data)

            except Exception as e:
                print(f"Error processing prediction file: {e}")
                prediction_results = {'error': str(e)}

        include_prediction_methods = bool(prediction_results and prediction_results != {} and 'error' not in prediction_results)
        final_model_options = build_final_model_options(
            results,
            models,
            approaches,
            include_prediction_methods=include_prediction_methods
        )

        return render_template('results.html', results=results, models=models, raw_data_info=raw_data_info, interpolation_info=interpolation_info, prediction_interpolation_info=prediction_interpolation_info, 
                               comparisons=comparisons, ks_tables=ks_tables, 
                               approaches=approaches, selected_approaches=selected_approaches, 
                               selected_scalings=selected_scalings, selected_normalizations=selected_normalizations, selected_interpolation=selected_interpolation,
                               metrics=metrics, selected_metrics=selected_metrics, prediction_results=prediction_results,
                               analysis_config=analysis_config,
                               analysis_config_rows=analysis_config_rows, preprocessing_rows=preprocessing_rows,
                               preprocessing_summary=preprocessing_summary, metric_description_rows=metric_description_rows,
                               grid_search_description_html=GRID_SEARCH_DESCRIPTION_HTML, tool_purpose_html=TOOL_PURPOSE_HTML,
                               final_model_options=final_model_options,
                               include_prediction_methods=include_prediction_methods,
                               approach_descriptions=APPROACH_DESCRIPTIONS)
    return render_template('index.html', models=models, approaches=approaches, preprocessing_options=preprocessing_options, 
                           metrics=metrics,
                           preprocessing_descriptions=PREPROCESSING_DESCRIPTIONS, metric_descriptions=METRIC_DESCRIPTIONS,
                           metric_description_rows=format_metric_description_rows(metrics),
                           grid_search_description_html=GRID_SEARCH_DESCRIPTION_HTML,
                           tool_purpose_html=TOOL_PURPOSE_HTML,
                           default_grid_search_num_points=(10 if IS_PRODUCTION else 200),
                           default_grid_search_param_min='-1000000',
                           default_grid_search_param_max='1000000',
                           default_grid_search_random_state='12345')

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


@app.route('/download_word_report', methods=['POST'])
def download_word_report():
    payload = request.get_json(silent=True) or {}
    report_label = payload.get('report_label') or 'IVIVC_Report'

    try:
        document_io = build_word_report(payload)
    except Exception as e:
        print(f"Error generating Word report: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

    filename = f"{safe_report_filename(report_label)}.docx"
    return send_file(
        document_io,
        as_attachment=True,
        download_name=filename,
        mimetype=WORD_MIME_TYPE,
    )

@app.route('/download/<path:filename>')
def download_excel(filename):
    base_dir = os.path.abspath(os.sep)  # Root directory
    full_path = os.path.normpath(os.path.join(base_dir, filename))
    return send_file(full_path, as_attachment=True)

if not IS_PRODUCTION and __name__ == '__main__':
    app.run(debug=True)

