"""Temporary, opt-in profiling launcher; normal app launches are unaffected."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextvars import ContextVar
import cProfile
import csv
from datetime import datetime, timezone
from functools import wraps
import importlib
import io
import json
from pathlib import Path
import platform
import pstats
import sys
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ContextVar('ivivc_profile_request', default=None)

# Only relatively coarse calls are timed here. cProfile supplies inner-loop
# call counts/timings when requested, without wrapping every integrand/root step.
TARGETS = {
    'engine': (
        'preprocess_data', 'process_data', 'process_predictions', 'fit_profile',
        'curve_parameter_intervals', 'predict_time_mapping',
        'predict_value_ratio_with_uncertainty', 'predict_tau_ratio',
        'predict_inverse_response', '_attach_cv_fit_variability_plots',
        '_attach_residual_plots', 'create_comparison',
        'create_cross_validation_summary', 'create_parameter_diagnostics_summary',
        'create_initial_plotly', 'create_interpolation_plotly',
        'create_value_interpolation_plotly', 'create_prediction_interpolation_plotly',
        'create_plotly_a1', 'create_plotly_a2', 'create_plotly_a3',
        'create_prediction_plot_a1', 'create_prediction_plot_a2',
        'create_prediction_plot_a3', 'extract_plotly_data_for_table',
    ),
    'utilities.evaluation': (
        'auto_grid_search', 'cross_validation_curve_fit',
        'calculate_tau_with_uncertainty',
    ),
    'utilities.report_content': ('build_report_sections',),
    'utilities.model_evidence': ('build_relative_evidence_tables',),
    'utilities.word_report': ('build_word_report',),
    'app': ('render_template',),
    'plotly.io': ('to_html',),
}


class Timings:
    def __init__(self):
        self.stack = []
        self.rows = defaultdict(lambda: dict(
            calls=0, inclusive_seconds=0.0, exclusive_seconds=0.0,
            max_seconds=0.0, exceptions=0,
        ))

    def records(self):
        return sorted((dict(path=path, model=model, **values)
                       for (path, model), values in self.rows.items()),
                      key=lambda row: row['inclusive_seconds'], reverse=True)


def _model_name(args, kwargs, parent):
    if kwargs.get('model_name'):
        return str(kwargs['model_name'])
    if parent and parent['model']:
        return parent['model']
    for arg in args:
        if isinstance(arg, dict) and 'model_function' in arg:
            return arg['model_function'].__module__.rsplit('.', 1)[-1]
        if callable(arg) and getattr(arg, '__module__', '').startswith('models.'):
            return arg.__module__.rsplit('.', 1)[-1]
    return ''


def timed(function, label):
    @wraps(function)
    def wrapper(*args, **kwargs):
        timings = ACTIVE.get()
        if timings is None:
            return function(*args, **kwargs)
        parent = timings.stack[-1] if timings.stack else None
        detail = kwargs.get('prediction_mode', kwargs.get('ptype'))
        stage = label + (f'[{detail}]' if detail is not None else '')
        path = (parent['path'] + ' > ' if parent else '') + stage
        model = _model_name(args, kwargs, parent)
        frame = dict(path=path, model=model, children=0.0)
        timings.stack.append(frame)
        failed = False
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = perf_counter() - started
            timings.stack.pop()
            if parent is not None:
                parent['children'] += elapsed
            row = timings.rows[path, model]
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += max(0.0, elapsed - frame['children'])
            row['max_seconds'] = max(row['max_seconds'], elapsed)
            row['exceptions'] += int(failed)
    return wrapper


def install_timers():
    """Wrap current bindings and their already-imported app/core aliases.

    Browser predictions explicitly pass the engine's inverse callbacks; those
    and all deeper numerical calls remain untouched and visible to cProfile.
    """
    replacements = {}
    for module_name, names in TARGETS.items():
        module = importlib.import_module(module_name)
        for name in names:
            function = getattr(module, name)
            replacements[id(function)] = timed(function, f'{module_name}.{name}')
    modules = [module for name, module in list(sys.modules.items())
               if module is not None and (name in ('app', 'engine', 'plotly.io')
               or name.startswith(('ivivc_core.', 'utilities.')))]
    for module in modules:
        for name, value in list(vars(module).items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                setattr(module, name, replacement)


def save_profile(directory, metadata, timings, profiler):
    prefix = directory / metadata['request_id']
    records = timings.records()
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'path', 'model', 'calls', 'inclusive_seconds', 'exclusive_seconds',
            'max_seconds', 'exceptions',
        ))
        writer.writeheader()
        writer.writerows(records)
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                          encoding='utf-8')
    if profiler is not None:
        profiler.dump_stats(str(prefix.with_suffix('.prof')))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stream.write('Top 40 functions by cumulative time (nested totals overlap)\n')
        stats.sort_stats('cumulative').print_stats(40)
        stream.write('\nTop 40 functions by self time\n')
        stats.sort_stats('tottime').print_stats(40)
        stream.write('\nSuspect functions and their callers\n')
        stats.print_callers(r'(evaluate_tau|inverse_response_times|evaluate_curve_draws|to_html)')
        prefix.with_suffix('.txt').write_text(stream.getvalue(), encoding='utf-8')
    print(f"\n[profile] {metadata['method']} {metadata['path']}: "
          f"{metadata['application_seconds']:.3f} s server, "
          f"{metadata['response_bytes'] / 1048576:.2f} MiB response, "
          f"status {metadata['status']}", flush=True)
    print('  Inclusive seconds overlap; exclusive excludes only other timed calls.')
    for row in records[:15]:
        print(f"  {row['inclusive_seconds']:9.3f} s  {row['calls']:4d} calls  "
              f"{row['model'] or '-':24s} {row['path'].rsplit(' > ', 1)[-1]}")
    print(f'  Saved: {prefix}.*\n', flush=True)


class ProfileRequests:
    """Measure buffered app responses, excluding transfer/browser rendering."""
    def __init__(self, application, directory, *, detailed=False, label=''):
        self.application = application
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.detailed = detailed
        self.label = label

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if (environ.get('REQUEST_METHOD') != 'POST' or
                path not in ('/', '/download_word_report', '/download_table_excel')):
            return self.application(environ, start_response)
        return self._profile(environ, start_response)

    def _profile(self, environ, start_response):
        request_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
        metadata = dict(request_id=request_id, label=self.label,
                        method=environ['REQUEST_METHOD'], path=environ['PATH_INFO'],
                        python=platform.python_version(), platform=platform.platform(),
                        cprofile=self.detailed, status='exception', response_bytes=0)
        timings = Timings()
        token = ACTIVE.set(timings)
        profiler = cProfile.Profile() if self.detailed else None
        response = None

        def capture_status(status, headers, exc_info=None):
            metadata['status'] = status
            # Visible in the browser's Network panel; this measures time up to
            # the response headers, not network transfer or browser rendering.
            seconds = perf_counter() - started
            return start_response(status, list(headers) + [
                ('Server-Timing', f'ivivc;dur={seconds * 1000:.3f}')
            ], exc_info)

        print(f"[profile] Started {request_id}: {metadata['method']} {metadata['path']}", flush=True)
        started = perf_counter()
        try:
            if profiler is not None:
                profiler.enable()
            try:
                response = self.application(environ, capture_status)
            finally:
                metadata['application_seconds'] = perf_counter() - started
                if profiler is not None:
                    profiler.disable()
                ACTIVE.reset(token)
            for chunk in response:
                metadata['response_bytes'] += len(chunk)
                yield chunk
        finally:
            try:
                if response is not None and hasattr(response, 'close'):
                    response.close()
            finally:
                try:
                    save_profile(self.directory, metadata, timings, profiler)
                except Exception as exc:
                    # Failure to save diagnostics must not replace the app response.
                    print(f'[profile] Could not save timings: {exc}', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--output', type=Path, default=ROOT / 'profiling-output')
    parser.add_argument('--label', default='', help='Scenario label stored in request metadata.')
    parser.add_argument('--cprofile', action='store_true',
                        help='Also save detailed call statistics; adds profiling overhead.')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from app import app
    install_timers()
    app.wsgi_app = ProfileRequests(app.wsgi_app, args.output,
                                   detailed=args.cprofile, label=args.label)
    print(f'Open http://127.0.0.1:{args.port}/ and use the app normally.')
    print(f'Timing output: {args.output.resolve()}')
    print('Keep the same analysis settings across comparisons; repeat to assess warm-up.')
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == '__main__':
    main()
