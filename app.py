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
    CROSS_VALIDATION_OVERVIEW_SUMMARY_HTML,
    CROSS_VALIDATION_SUMMARY_HTML,
    GOODNESS_VALIDATION_TABLE_HTML,
    RESIDUAL_DIAGNOSTICS_HTML,
    RELATIVE_MODEL_EVIDENCE_HTML,
    PARAMETER_DIAGNOSTICS_COMPARISON_HTML,
    TOOL_PURPOSE_HTML,
    METRIC_DESCRIPTIONS,
    MODAL_HELP,
    PREPROCESSING_DESCRIPTIONS,
    PREPROCESSING_OVERVIEW_SUMMARY_HTML,
    format_analysis_config_rows,
    format_metric_description_rows,
    format_preprocessing_rows,
    format_selected_preprocessing_summary,
)
from utilities.reporting import build_final_model_options
from utilities.model_evidence import build_relative_evidence_tables
from utilities.word_report import build_word_report, safe_report_filename, WORD_MIME_TYPE
from utilities.excel_export import build_accessible_excel, EXCEL_MIME_TYPE
from utilities.app_info import get_app_info
from utilities.model_display import get_human_readable_function
from metrics import metrics
from engine import (
    create_comparison,
    create_cross_validation_summary,
    create_parameter_diagnostics_summary,
    create_initial_plotly,
    create_interpolation_plotly,
    create_prediction_interpolation_plotly,
    preprocess_data,
    process_data,
    process_predictions,
)

app = Flask(__name__)


@app.context_processor
def inject_app_info():
    """Expose version and contact information to all browser templates."""
    return {'app_info': get_app_info()}


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


def _index_template_context(form_errors=None):
    errors = form_errors or []
    return {
        'models': models,
        'approaches': approaches,
        'preprocessing_options': preprocessing_options,
        'metrics': metrics,
        'preprocessing_descriptions': PREPROCESSING_DESCRIPTIONS,
        'metric_descriptions': METRIC_DESCRIPTIONS,
        'metric_description_rows': format_metric_description_rows(metrics),
        'grid_search_description_html': GRID_SEARCH_DESCRIPTION_HTML,
        'tool_purpose_html': TOOL_PURPOSE_HTML,
        'modal_help': MODAL_HELP,
        'default_grid_search_num_points': (10 if IS_PRODUCTION else 200),
        'default_grid_search_param_min': '-1000000',
        'default_grid_search_param_max': '1000000',
        'default_grid_search_random_state': '12345',
        'form_errors': errors,
        'error_field_ids': {error.get('field_id') for error in errors if error.get('field_id')},
    }


def _render_index(form_errors=None, status=200):
    return render_template('index.html', **_index_template_context(form_errors)), status

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
        'include_raw_residual_plots': request.form.get('include_raw_residual_plots') == 'on',
        'include_residual_qq_plots': request.form.get('include_residual_qq_plots') == 'on',
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        prediction_file = request.files.get('prediction_file')
        selected_models = request.form.getlist('models')
        selected_approaches = request.form.getlist('approaches')
        selected_normalizations = request.form.getlist('normalizations')
        selected_scalings = request.form.getlist('scalings')
        selected_interpolation = request.form.getlist('interpolation')
        selected_metrics = request.form.getlist('metrics')
        try:
            analysis_config = parse_analysis_config()
        except (TypeError, ValueError):
            return _render_index([
                {'message': 'Enter valid numeric values for the advanced analysis parameters.', 'field_id': 'advanced-analysis-group'}
            ], status=400)

        validation_errors = []
        if not file or not file.filename:
            validation_errors.append({'message': 'Upload a fitting dataset.', 'field_id': 'file'})
        if not selected_approaches:
            validation_errors.append({'message': 'Select at least one analysis approach.', 'field_id': 'approaches-group'})
        if any(approach_id != 'approach3' for approach_id in selected_approaches) and not selected_models:
            validation_errors.append({'message': 'Select at least one model for the selected parametric approach.', 'field_id': 'approaches-group'})
        if not selected_metrics:
            validation_errors.append({'message': 'Select at least one performance metric.', 'field_id': 'metrics-group'})

        filename = file.filename if file and file.filename else ''
        file_extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if filename and file_extension not in {'csv', 'xls', 'xlsx'}:
            validation_errors.append({'message': 'Upload the fitting dataset as a CSV or Excel file.', 'field_id': 'file'})
        if file_extension in {'xls', 'xlsx'} and not request.form.get('sheet'):
            validation_errors.append({'message': 'Re-upload the fitting Excel file and select a worksheet.', 'field_id': 'file'})

        prediction_filename = prediction_file.filename if prediction_file and prediction_file.filename else ''
        prediction_extension = prediction_filename.rsplit('.', 1)[-1].lower() if '.' in prediction_filename else ''
        if prediction_filename and prediction_extension not in {'csv', 'xls', 'xlsx'}:
            validation_errors.append({'message': 'Upload the prediction dataset as a CSV or Excel file.', 'field_id': 'prediction_file'})
        if prediction_extension in {'xls', 'xlsx'} and not request.form.get('prediction_sheet'):
            validation_errors.append({'message': 'Re-upload the prediction Excel file and select a worksheet.', 'field_id': 'prediction_file'})

        if validation_errors:
            return _render_index(validation_errors, status=400)

        human_readable_functions = {
            model_name: models[model_name].get('latex_equation') or get_human_readable_function(models[model_name]['model_function'])
            for model_name in set([m.split(':')[-1] for m in selected_models])
        }

        # add placeholder for approach3
        if 'approach3' in selected_approaches:
            selected_models.append('approach3')

        # Read data
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, sheet_name=request.form.get('sheet'))
        except Exception as exc:
            print(f"Error reading fitting dataset: {exc}")
            return _render_index([
                {'message': 'The fitting dataset could not be read. Check the file format and selected worksheet.', 'field_id': 'file'}
            ], status=400)
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
        comparisons, ks_tables, cv_summary_tables, parameter_diagnostic_tables = {}, {}, {}, {}
        for approach in selected_approaches:
            comparisons[approach], ks_tables[approach] = create_comparison(results, approach)
            cv_summary_tables[approach] = create_cross_validation_summary(results, approach, selected_metrics)
            parameter_diagnostic_tables[approach] = create_parameter_diagnostics_summary(results, approach)

        evidence_tables, evidence_lookup = build_relative_evidence_tables(
            results=results,
            models_registry=models,
            selected_approaches=selected_approaches,
            selected_metrics=selected_metrics,
        )

        analysis_config_rows = format_analysis_config_rows(
            analysis_config,
            app_info=get_app_info(),
        )
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
                               comparisons=comparisons, ks_tables=ks_tables, cv_summary_tables=cv_summary_tables,
                               approaches=approaches, selected_approaches=selected_approaches, 
                               selected_scalings=selected_scalings, selected_normalizations=selected_normalizations, selected_interpolation=selected_interpolation,
                               metrics=metrics, selected_metrics=selected_metrics, prediction_results=prediction_results,
                               analysis_config=analysis_config,
                               analysis_config_rows=analysis_config_rows, preprocessing_rows=preprocessing_rows,
                               preprocessing_summary=preprocessing_summary, metric_description_rows=metric_description_rows,
                               grid_search_description_html=GRID_SEARCH_DESCRIPTION_HTML, tool_purpose_html=TOOL_PURPOSE_HTML,
                               preprocessing_overview_summary_html=PREPROCESSING_OVERVIEW_SUMMARY_HTML,
                               cross_validation_overview_summary_html=CROSS_VALIDATION_OVERVIEW_SUMMARY_HTML,
                               relative_model_evidence_html=RELATIVE_MODEL_EVIDENCE_HTML,
                               cross_validation_summary_html=CROSS_VALIDATION_SUMMARY_HTML,
                               goodness_validation_table_html=GOODNESS_VALIDATION_TABLE_HTML,
                               residual_diagnostics_html=RESIDUAL_DIAGNOSTICS_HTML,
                               parameter_diagnostics_comparison_html=PARAMETER_DIAGNOSTICS_COMPARISON_HTML,
                               parameter_diagnostic_tables=parameter_diagnostic_tables,
                               modal_help=MODAL_HELP,
                               final_model_options=final_model_options,
                               evidence_tables=evidence_tables, evidence_lookup=evidence_lookup,
                               include_prediction_methods=include_prediction_methods,
                               approach_descriptions=APPROACH_DESCRIPTIONS)
    return _render_index()

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


@app.route('/download_table_excel', methods=['POST'])
def download_table_excel():
    """Generate an accessible Excel workbook for a browser data table."""
    payload = request.get_json(silent=True) or {}
    try:
        workbook_io, filename = build_accessible_excel(payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        print(f"Error generating Excel table download: {exc}")
        traceback.print_exc()
        return jsonify({'error': 'The Excel workbook could not be generated.'}), 500

    return send_file(
        workbook_io,
        as_attachment=True,
        download_name=filename,
        mimetype=EXCEL_MIME_TYPE,
    )


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

