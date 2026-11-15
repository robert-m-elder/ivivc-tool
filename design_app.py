"""Separate, request-scoped Flask frontend for study-design assessment.

F1 provides setup review and request handling. Numerical workflows are connected
in subsequent patches; importing this module does not import the IVIVC web app.
"""

import json

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


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
        return render_template('design/index.html', settings=_default_settings())

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
