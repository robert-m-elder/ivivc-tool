"""Separate, request-scoped Flask frontend for study-design assessment.

Data imports and prospective planning complete within individual requests.
This module does not import the IVIVC web application.
"""

import json

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from utilities.design_uploads import COURSE_FIELDS, DATASETS, options_for


TASK_LABELS = {
    'study': 'Plan or extend an in vivo study',
    'calibration': 'Compare calibration sampling',
}
REFERENCE_LABELS = {
    'measured': 'Measured in vivo data',
    'ivivc': 'IVIVC prediction',
    'scaling': 'Empirical time scaling',
}
ENVIRONMENT_LABELS = {
    'in_vivo': 'In vivo only; keep in vitro data fixed',
    'in_vitro': 'In vitro only; keep in vivo data fixed',
    'both': 'Both environments',
}


def _default_settings():
    return {
        'task': 'study',
        'reference_source': 'measured',
        'varied_environment': 'in_vivo',
        'n_simulations': 100,
        'random_seed': 12345,
    }


def _review_settings(form):
    """Validate only the setup choices exposed by this initial frontend."""
    settings = _default_settings()
    errors = []

    def error(field_id, message):
        errors.append({'field_id': field_id, 'message': message})

    task = form.get('task', '').strip()
    if task not in TASK_LABELS:
        error('task', 'Select a supported analysis task.')
    else:
        settings['task'] = task

    if task == 'study':
        source = form.get('reference_source', 'measured').strip()
        if source not in REFERENCE_LABELS:
            error('reference_source', 'Select a supported source of expected in vivo behavior.')
        else:
            settings['reference_source'] = source
    elif task == 'calibration':
        environment = form.get('varied_environment', 'in_vivo').strip()
        if environment not in ENVIRONMENT_LABELS:
            error('varied_environment', 'Select which calibration environment will vary.')
        else:
            settings['varied_environment'] = environment

    for name, label, minimum in (
        ('n_simulations', 'Number of simulated studies', 2),
        ('random_seed', 'Random seed', 0),
    ):
        raw = form.get(name, '').strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            error(name, f'{label} must be a whole number of at least {minimum}.')
            continue
        if value < minimum:
            error(name, f'{label} must be at least {minimum}.')
        else:
            settings[name] = value

    return settings, errors


def create_app(config=None):
    """Create the independent local/WSGI app without a result or upload store."""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
    if config is not None:
        app.config.update(config)

    @app.get('/')
    def index():
        from utilities.design_planning import model_options
        from utilities.design_reference_sources import reference_options

        return render_template('design/index.html', settings=_default_settings(),
                               datasets=DATASETS, planning_models=model_options(),
                               reference_options=reference_options())

    def import_error(message, field_id=None):
        return jsonify(ok=False, errors=[{'field_id': field_id, 'message': message}]), 400

    def dataset_request():
        identifier = request.form.get('dataset_id', '')
        return DATASETS.get(identifier), request.files.get('file'), request.form.get('sheet') or None

    @app.post('/data/inspect')
    def inspect_data():
        # Lazy imports retain the independent, lightweight application entry point.
        from utilities.design_data import DataImportError, read_upload

        dataset, upload, sheet = dataset_request()
        if dataset is None:
            return import_error('Select a supported dataset role.')
        try:
            loaded = read_upload(upload, sheet=sheet)
        except DataImportError as error:
            field = dataset['id'] + '__sheet' if error.field == 'sheet' and sheet else dataset['id']
            return import_error(str(error), field)
        columns = None if loaded.frame is None else loaded.frame.columns.tolist()
        html = render_template(
            'design/_data_mapping.html', dataset=dataset,
            sheet_names=loaded.sheets, selected_sheet=loaded.sheet, columns=columns,
        )
        return jsonify(
            ok=True, html=html, dataset_id=dataset['id'], filename=loaded.filename,
            sheets=loaded.sheets, selected_sheet=loaded.sheet,
            requires_sheet=loaded.frame is None, columns=columns or [],
        )

    @app.post('/data/preview')
    def preview_data():
        from utilities.design_data import (
            DataImportError, interpret_sd_profile, interpret_time_course,
            interpret_time_scales, read_upload,
        )

        dataset, upload, sheet = dataset_request()
        if dataset is None:
            return import_error('Select a supported dataset role.')
        try:
            loaded = read_upload(upload, sheet=sheet)
        except DataImportError as error:
            field = dataset['id'] + '__sheet' if error.field == 'sheet' and sheet else dataset['id']
            return import_error(str(error), field)
        if loaded.frame is None:
            return import_error('Select a worksheet and read its columns first.', dataset['id'] + '__sheet')

        imports, errors = [], []
        if dataset['kind'] in ('course', 'pair'):
            for series in dataset['series']:
                prefix = dataset['id'] + '__' + series['key'] + '__'
                try:
                    options = options_for(request.form, prefix, COURSE_FIELDS)
                    for field, index in (('time_column', series['time_default']),
                                         ('response_column', series['response_default'])):
                        if field not in options:
                            options[field] = (loaded.frame.columns[index]
                                              if index < len(loaded.frame.columns) else '')
                    interpreted = interpret_time_course(
                        loaded.frame, options,
                        label=series['label'],
                    )
                except DataImportError as error:
                    field = dataset['id'] if error.field == 'file' else prefix + error.field
                    errors.append({'field_id': field, 'message': series['label'] + ': ' + str(error)})
                else:
                    imports.append({'label': series['label'], 'data': interpreted})
        else:
            prefix = dataset['id'] + '__'
            fields = (('ratio_column', 'weight_column', 'source_column')
                      if dataset['kind'] == 'ratios' else ('time_column', 'sd_column'))
            interpreter = interpret_time_scales if dataset['kind'] == 'ratios' else interpret_sd_profile
            try:
                interpreted = interpreter(loaded.frame, options_for(request.form, prefix, fields))
            except DataImportError as error:
                field = dataset['id'] if error.field == 'file' else prefix + error.field
                errors.append({'field_id': field, 'message': str(error)})
            else:
                imports.append({'label': dataset['label'], 'data': interpreted})
        if errors:
            return jsonify(ok=False, errors=errors), 400
        html = render_template(
            'design/_data_preview.html', dataset=dataset, filename=loaded.filename,
            selected_sheet=loaded.sheet, imports=imports,
        )
        return jsonify(ok=True, html=html, dataset_id=dataset['id'], filename=loaded.filename)

    @app.post('/calculate')
    def calculate_comparison():
        from utilities.design_data import (
            DataImportError, interpret_sd_profile, interpret_time_course,
            interpret_time_scales, read_upload,
        )
        from utilities.design_planning import (
            PlanningInputError, run_measured_planning, run_reference_planning,
        )
        from utilities.design_planning_results import present_planning_result

        settings, errors = _review_settings(request.form)
        if errors:
            return jsonify(ok=False, errors=errors), 400
        if settings['task'] != 'study':
            return import_error('Calibration comparisons are not connected to this frontend yet.', 'task')
        has_completed = str(request.form.get('has_completed', '')).strip().lower() in (
            '1', 'on', 'true', 'yes',
        )
        measured_mode = request.form.get('measured_adaptive_mode', 'refit_completed')
        if (has_completed and settings['reference_source'] == 'measured'
                and measured_mode not in ('refit_completed', 'reweight_prior')):
            return import_error('Select how completed data update the measured reference.',
                                'measured_adaptive_mode')

        def planning_input(identifier):
            """Rebuild a typed input from this request, without relying on a preview."""
            dataset = DATASETS[identifier]
            sheet = request.form.get(identifier + '__sheet') or None
            try:
                loaded = read_upload(request.files.get(dataset['file_field']), sheet=sheet)
            except DataImportError as error:
                field = identifier + '__sheet' if error.field == 'sheet' and sheet else identifier
                raise PlanningInputError(str(error), field) from error
            if loaded.frame is None:
                raise PlanningInputError('Select a worksheet and read its columns first.', identifier + '__sheet')
            values, summaries = {}, []
            if dataset['kind'] in ('course', 'pair'):
                for series in dataset['series']:
                    prefix = identifier + '__' + series['key'] + '__'
                    options = options_for(request.form, prefix, COURSE_FIELDS)
                    for field, index in (('time_column', series['time_default']),
                                         ('response_column', series['response_default'])):
                        if field not in options:
                            options[field] = (loaded.frame.columns[index]
                                              if index < len(loaded.frame.columns) else '')
                    try:
                        interpreted = interpret_time_course(loaded.frame, options, label=series['label'])
                    except DataImportError as error:
                        field = identifier if error.field == 'file' else prefix + error.field
                        raise PlanningInputError(str(error), field) from error
                    values[series['key']] = interpreted.value
                    summaries.append({
                        'role': identifier, 'label': series['label'],
                        'filename': loaded.filename, 'worksheet': loaded.sheet,
                        'summary': dict(interpreted.summary, filename=loaded.filename,
                                        worksheet=loaded.sheet),
                    })
                value = values if dataset['kind'] == 'pair' else values['profile']
            else:
                prefix = identifier + '__'
                fields = (('ratio_column', 'weight_column', 'source_column')
                          if dataset['kind'] == 'ratios' else ('time_column', 'sd_column'))
                interpreter = interpret_time_scales if dataset['kind'] == 'ratios' else interpret_sd_profile
                try:
                    interpreted = interpreter(loaded.frame, options_for(request.form, prefix, fields))
                except DataImportError as error:
                    field = identifier if error.field == 'file' else prefix + error.field
                    raise PlanningInputError(str(error), field) from error
                value = interpreted.value
                summaries.append({
                    'role': identifier, 'label': dataset['label'],
                    'filename': loaded.filename, 'worksheet': loaded.sheet,
                    'summary': dict(interpreted.summary, filename=loaded.filename,
                                    worksheet=loaded.sheet),
                })
            return value, summaries

        try:
            source_kind = settings['reference_source']
            arguments, source_summaries = {}, []
            completed, completed_summaries = None, []
            if has_completed:
                completed, completed_summaries = planning_input('completed-target-file')
            if source_kind == 'measured':
                if has_completed and measured_mode == 'refit_completed':
                    # These current-study rows are fitted once by the backend;
                    # an unrelated prior upload must not enter that fit.
                    source = None
                else:
                    source, source_summaries = planning_input('reference-file')
            elif source_kind == 'ivivc':
                calibration, source_summaries = planning_input('ivivc-calibration-file')
                prediction, prediction_summaries = planning_input('prediction-file')
                source_summaries.extend(prediction_summaries)
                arguments = dict(in_vitro=calibration['in_vitro'], in_vivo=calibration['in_vivo'],
                                 prediction=prediction)
            else:
                scaling_source, source_summaries = planning_input('scaling-source-file')
                distribution, distribution_summaries = planning_input('time-scale-file')
                source_summaries.extend(distribution_summaries)
                arguments = dict(scaling_source=scaling_source, distribution=distribution)
            source_summaries.extend(completed_summaries)
            profile, profile_summary = None, None
            if request.form.get('observation_kind') == 'time_profile_sd':
                profile, profile_summaries = planning_input('future-sd-profile-file')
                profile_summary = profile_summaries[0]['summary']
            result = (run_measured_planning(source, request.form, profile=profile,
                                           completed=completed)
                      if source_kind == 'measured' else
                      run_reference_planning(request.form, profile=profile,
                                             completed=completed, **arguments))
        except PlanningInputError as error:
            return import_error(str(error), error.field_id)
        source_summary = source_summaries[0]['summary']
        context = present_planning_result(
            result, source_filename=(source_summary['filename'] if source_kind == 'measured'
                                     else None),
            source_summary=source_summary, profile_summary=profile_summary,
            source_summaries=source_summaries,
        )
        html = render_template('design/_planning_results.html', **context)
        return jsonify(ok=True, html=html)

    @app.post('/review')
    def review_setup():
        settings, errors = _review_settings(request.form)
        if errors:
            return jsonify(ok=False, errors=errors), 400
        html = render_template(
            'design/_setup_review.html',
            settings=settings,
            task_labels=TASK_LABELS,
            reference_labels=REFERENCE_LABELS,
            environment_labels=ENVIRONMENT_LABELS,
        )
        return jsonify(ok=True, html=html, settings=settings)

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = ('The upload is too large. Use smaller files and try again.'
                   if isinstance(error, RequestEntityTooLarge) else error.description)
        # Retain HTTP headers such as Allow on method errors.
        response = error.get_response()
        response.set_data(json.dumps({
            'ok': False,
            'errors': [{'field_id': None, 'message': message}],
        }))
        response.content_type = 'application/json'
        return response

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception('Study-design request failed.')
        return jsonify(ok=False, errors=[{
            'field_id': None,
            'message': 'The request could not be completed. Try again; earlier displayed results are unchanged.',
        }]), 500

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
