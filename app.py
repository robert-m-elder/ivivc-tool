import os
import traceback
import warnings
from datetime import datetime, timezone

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
    CV_FIT_VARIABILITY_HTML,
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
from utilities.report_content import build_report_sections, build_uncertainty_sections
from utilities.model_evidence import build_relative_evidence_tables
from utilities.uncertainty_inputs import (
    future_uncertainty_from_options, make_uncertainty_record,
    uncertainty_from_options, validate_uncertainty_options,
)
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
    create_value_interpolation_plotly,
    create_prediction_interpolation_plotly,
    preprocess_data,
    process_data,
    process_predictions,
    data_for_display,
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
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MiB

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
        'include_cv_fit_variability_plot': request.form.get('include_cv_fit_variability_plot') == 'on',
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
        has_prediction_dataset = bool(prediction_file and prediction_file.filename)
        future_active = (request.form.get('future_uncertainty_mode', 'none') or 'none') != 'none'
        try:
            uncertainty_active = validate_uncertainty_options(
                request.form, selected_approaches, selected_scalings,
                has_prediction=has_prediction_dataset,
            )
        except ValueError as exc:
            uncertainty_active = False
            validation_errors.append({'message': str(exc), 'field_id': 'uncertainty-group'})
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
        try:
            if len(df.columns) < 4:
                raise ValueError('The fitting dataset requires at least four columns: Time1, Value1, Time2, Value2.')
            t1, m1, t2, m2 = [pd.to_numeric(df.iloc[:, index], errors='raise').to_numpy(dtype=float)
                              for index in range(4)]
        except (TypeError, ValueError) as exc:
            return _render_index([{'message': str(exc), 'field_id': 'file'}], status=400)

        source_uncertainties, raw_uncertainty, observation_uncertainty = {}, {}, {}
        future_uncertainty, future_uncertainty_record = None, None
        if uncertainty_active:
            try:
                for source in ('in_vitro', 'in_vivo'):
                    supplied, record = uncertainty_from_options(df, source, request.form)
                    if supplied is not None:
                        source_uncertainties[source + '_uncertainty'] = supplied
                        raw_uncertainty[source] = record
            except ValueError as exc:
                return _render_index([{'message': str(exc), 'field_id': 'uncertainty-group'}], status=400)
        
        # Apply preprocessing after the uploaded data have been read. If a
        # selected Log10 scaling is incompatible with the processed values,
        # continue with that axis unscaled and report the fallback in results.
        try:
            preparation_options = ({'return_prepared': True, **source_uncertainties}
                                   if source_uncertainties or future_active else {})
            calculation_data, preprocessing_metadata = preprocess_data(
                t1, m1, t2, m2,
                selected_interpolation=selected_interpolation,
                selected_scalings=selected_scalings,
                selected_normalizations=selected_normalizations,
                fallback_invalid_log_scaling=True,
                return_metadata=True,
                **preparation_options,
            )
            data = data_for_display(calculation_data) if source_uncertainties or future_active else calculation_data
            if future_active:
                future_uncertainty, future_uncertainty_record = future_uncertainty_from_options(
                    request.form, preprocessing_metadata,
                )
            if source_uncertainties:
                for source, record in raw_uncertainty.items():
                    observation_uncertainty[source] = make_uncertainty_record(
                        record, getattr(calculation_data, source), calculation_data.preprocessing,
                        source_name=source,
                    )
        except ValueError as exc:
            return _render_index([{'message': str(exc),
                                   'field_id': 'uncertainty-group' if source_uncertainties or future_active else None}], status=400)
        selected_scalings = preprocessing_metadata['applied_scalings']
        preprocessing_warnings = preprocessing_metadata['warnings']
        t1_scale,m1_scale,t2_scale,m2_scale,mm,tt,ti1,ti2,mi1,mi2 = data

        raw_data_info = create_initial_plotly(t1, m1, t2, m2)
        time_interpolation_info = create_interpolation_plotly(t1_scale, m1_scale, t2_scale, m2_scale, tt, mi1, mi2)
        value_interpolation_info = create_value_interpolation_plotly(t1_scale, m1_scale, t2_scale, m2_scale, mm, ti1, ti2)

        # Apply models
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = process_data(calculation_data, selected_models, selected_approaches, selected_metrics, analysis_config=analysis_config)
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
        if preprocessing_warnings and not selected_scalings:
            for row in preprocessing_rows:
                if row['category'] == 'Scaling' and row['name'] == 'None':
                    row['name'] = 'None (fallback)'
                    row['description'] = 'No scaling was applied because the selected Log10 scaling option could not be applied to the processed data; see the scaling fallback warning.'
                    break
        preprocessing_summary = format_selected_preprocessing_summary(
            selected_normalizations,
            selected_scalings,
            selected_interpolation,
        )
        metric_description_rows = format_metric_description_rows(metrics, selected_metrics)

        # Process predictions if prediction file is provided
        prediction_results, prediction_interpolation_info = {}, {}
        pred_df = None
        prediction_data = None
        prediction_preprocessing_metadata = None
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
                
                if len(pred_df.columns) < 2:
                    raise ValueError('The prediction dataset requires at least two columns: Time, Value.')
                t_pred, m_pred = [pd.to_numeric(pred_df.iloc[:, index], errors='raise').to_numpy(dtype=float)
                                  for index in range(2)]
                prediction_uncertainty, prediction_uncertainty_record = uncertainty_from_options(
                    pred_df, 'prediction', request.form,
                ) if uncertainty_active else (None, None)
                prediction_options = ({'return_prepared': True, 'in_vitro_uncertainty': prediction_uncertainty}
                                      if prediction_uncertainty is not None else {})
                
                calculation_prediction, prediction_preprocessing_metadata = preprocess_data(
                    t_pred, m_pred, None, None,
                    selected_interpolation=selected_interpolation,
                    selected_scalings=selected_scalings,
                    selected_normalizations=selected_normalizations,
                    return_metadata=True,
                    **prediction_options,
                )
                if prediction_uncertainty is not None:
                    prediction_data = data_for_display(calculation_prediction)
                    observation_uncertainty['prediction'] = make_uncertainty_record(
                        prediction_uncertainty_record, calculation_prediction.in_vitro,
                        calculation_prediction.preprocessing, source_name='profile',
                    )
                    calculation_prediction = calculation_prediction.in_vitro
                else:
                    prediction_data = calculation_prediction

                prediction_interpolation_info = create_prediction_interpolation_plotly(*prediction_data)

                future_options = ({'future_observation_uncertainty': future_uncertainty}
                                  if future_active else {})
                prediction_results = process_predictions(calculation_data, calculation_prediction, results, selected_approaches, interpolated_data=data, selected_scalings=selected_scalings, **future_options)

            except Exception as e:
                if request.form.get('prediction_uncertainty_mode', 'none') != 'none' or future_active:
                    return _render_index([{'message': str(e), 'field_id': 'uncertainty-group'}], status=400)
                print(f"Error processing prediction file: {e}")
                prediction_results = {'error': str(e)}

        include_prediction_methods = bool(prediction_results and prediction_results != {} and 'error' not in prediction_results)
        final_model_options = build_final_model_options(
            results,
            models,
            approaches,
            include_prediction_methods=include_prediction_methods,
            prediction_results=prediction_results,
        )

        analysis_timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')

        def source_record(upload_name, worksheet, frame):
            record = {
                'filename': upload_name.replace('\\', '/').rsplit('/', 1)[-1],
                'worksheet': worksheet or 'Not applicable (CSV)',
                'columns': [str(column) for column in frame.columns] if frame is not None else [],
            }
            roles = {}
            sources = ('prediction',) if frame is pred_df else ('in_vitro', 'in_vivo')
            labels = {'in_vitro': 'In vitro calibration', 'in_vivo': 'In vivo calibration',
                      'prediction': 'Prediction input'}
            for source in sources:
                source_uncertainty = observation_uncertainty.get(source, {})
                for key in ('column', 'replicate_column'):
                    if source_uncertainty.get(key):
                        column = source_uncertainty[key]
                        role = ('replicate count' if key == 'replicate_column'
                                else source_uncertainty['uncertainty_kind'].upper())
                        role = labels[source] + ' ' + role
                        roles[column] = roles[column] + '; ' + role if column in roles else role
            if roles:
                record['column_roles'] = roles
            return record

        # These records describe this calculation. Report export never refits a
        # model, repeats CV, or resamples the uncertainty calculations.
        report_context = {
            'app_info': get_app_info(),
            'analysis_timestamp': analysis_timestamp,
            'fitting_source': source_record(filename, request.form.get('sheet') if file_extension in {'xls', 'xlsx'} else None, df),
            'prediction_source': source_record(prediction_filename, request.form.get('prediction_sheet') if prediction_extension in {'xls', 'xlsx'} else None, pred_df)
                if has_prediction_dataset else None,
            'raw_fitting_data': {
                'in_vitro': {'time': t1, 'response': m1},
                'in_vivo': {'time': t2, 'response': m2},
            },
            'raw_prediction_data': {'time': pred_df.iloc[:, 0].to_numpy(), 'response': pred_df.iloc[:, 1].to_numpy()}
                if pred_df is not None and len(pred_df.columns) >= 2 else None,
            'processed_data': data,
            'processed_prediction_data': prediction_data,
            'preprocessing_metadata': preprocessing_metadata,
            'prediction_preprocessing_metadata': prediction_preprocessing_metadata,
            'selected_normalizations': selected_normalizations,
            'selected_scalings': selected_scalings,
            'selected_interpolation': selected_interpolation,
            'selected_models': selected_models,
            'selected_approaches': selected_approaches,
            'selected_metrics': selected_metrics,
            'analysis_config': analysis_config,
            'results': results,
            'prediction_results': prediction_results,
            'evidence_tables': evidence_tables,
            'models_registry': models,
        }
        if observation_uncertainty:
            report_context['observation_uncertainty'] = observation_uncertainty
        if future_uncertainty_record is not None:
            report_context['future_observation_input'] = future_uncertainty_record
        if observation_uncertainty or future_active:
            report_context['uncertainty_independence_confirmed'] = True
        final_report_sections = {
            option['id']: build_report_sections(report_context, option)
            for option in final_model_options
        }

        return render_template('results.html', results=results, models=models, raw_data_info=raw_data_info, time_interpolation_info=time_interpolation_info, value_interpolation_info=value_interpolation_info, prediction_interpolation_info=prediction_interpolation_info,
                               comparisons=comparisons, ks_tables=ks_tables, cv_summary_tables=cv_summary_tables,
                               approaches=approaches, selected_approaches=selected_approaches, 
                               selected_scalings=selected_scalings, selected_normalizations=selected_normalizations, selected_interpolation=selected_interpolation,
                               metrics=metrics, selected_metrics=selected_metrics, prediction_results=prediction_results,
                               analysis_config=analysis_config,
                               analysis_config_rows=analysis_config_rows, preprocessing_rows=preprocessing_rows,
                               preprocessing_warnings=preprocessing_warnings,
                               preprocessing_summary=preprocessing_summary, metric_description_rows=metric_description_rows,
                               grid_search_description_html=GRID_SEARCH_DESCRIPTION_HTML, tool_purpose_html=TOOL_PURPOSE_HTML,
                               preprocessing_overview_summary_html=PREPROCESSING_OVERVIEW_SUMMARY_HTML,
                               cross_validation_overview_summary_html=CROSS_VALIDATION_OVERVIEW_SUMMARY_HTML,
                               relative_model_evidence_html=RELATIVE_MODEL_EVIDENCE_HTML,
                               cross_validation_summary_html=CROSS_VALIDATION_SUMMARY_HTML,
                               goodness_validation_table_html=GOODNESS_VALIDATION_TABLE_HTML,
                               cv_fit_variability_html=CV_FIT_VARIABILITY_HTML,
                               residual_diagnostics_html=RESIDUAL_DIAGNOSTICS_HTML,
                               parameter_diagnostics_comparison_html=PARAMETER_DIAGNOSTICS_COMPARISON_HTML,
                               parameter_diagnostic_tables=parameter_diagnostic_tables,
                               modal_help=MODAL_HELP,
                               final_model_options=final_model_options,
                               final_report_sections=final_report_sections,
                               report_analysis_timestamp=analysis_timestamp,
                               has_prediction_dataset=has_prediction_dataset,
                               evidence_tables=evidence_tables, evidence_lookup=evidence_lookup,
                               include_prediction_methods=include_prediction_methods,
                               observation_uncertainty=observation_uncertainty,
                               uncertainty_sections=build_uncertainty_sections(report_context),
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

if not IS_PRODUCTION and __name__ == '__main__':
    app.run(debug=True)

