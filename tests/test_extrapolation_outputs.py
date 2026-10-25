"""Coverage diagnostics agree across the core, UI, plot downloads and reports."""
from io import StringIO, BytesIO
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from jinja2 import Environment, Template

import engine
from ivivc_core.predictions import predict_inverse_response
from ivivc_core.extrapolation import annotate_prediction_domain
from ivivc_core.types import TimeCourse, PredictedTrajectory
from utilities.report_content import build_report_sections


def inverse_result():
    return predict_inverse_response(TimeCourse(np.array([3., 6.]), np.array([.6, .2])),
        model_name='exponential', parameters={'a': 1., 'b': -.1}, covariance=np.zeros((2, 2)),
        residuals=[], calibration_time_range=(0., 10.), observed_responses=[1., np.exp(-1.)], n_samples=8)


def calibration():
    t = np.linspace(0., 10., 6)
    y = np.exp(-.1 * t)
    return (t, y, t, y, y, t, t, t, y, y)


def test_inverse_plot_download_report_and_annotations_reuse_numerical_results(monkeypatch):
    result = inverse_result()
    figures = []
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **k: figures.append(fig) or '<plot>')
    data = calibration()
    plot = engine.create_prediction_plot_a2(data, result.in_vitro_time, result.in_vitro_value,
        result.in_vivo_time, data[0], data[1], data[1],
        {'model_name': 'Exponential', 'model_function': lambda t, a, b: a * np.exp(b * t),
         'params': [{'a': 1., 'b': -.1}, {'a': 1., 'b': -.1}]},
        prediction_mode='observed_response_inverse', time_prediction=result)
    records = engine._parameter_prediction_records(result, 'time')
    assert plot['prediction_records'] == records
    assert records[0]['extrapolation_status'] == 'within calibration range'
    assert records[1]['extrapolated'] is True
    assert records[1]['inverse_extrapolated_samples'] == 8
    assert plot['extrapolation_warnings'][0]['label'] == 'Extrapolation'
    assert any('Extrapolation' in a.text for a in figures[0].layout.annotations)
    reference = next(trace for trace in figures[0].data if trace.name == 'In Vivo (Fitted Model)')
    assert max(reference.x) >= result.in_vivo_time[-1]
    table = pd.read_html(StringIO(plot['table_html']))[0]
    assert table[next(c for c in table if c.startswith('Calibration domain status'))][1] == 'extrapolation'
    # Reports use the exact stored bounds, not another random calculation.
    from test_report_content import fixture, table as report_table, text
    context, option = fixture(method='observed_response_inverse')
    context['prediction_source'] = {'filename': 'prediction.csv'}
    context['prediction_results'] = {'approach2': {'example': {'prediction_tables': {'observed_response_inverse': records}}}}
    sections = build_report_sections(context, option)
    domains = report_table(sections, 'Calibration-domain cautions')['rows']
    assert domains[2][1] == 'extrapolation'
    assert 'Extrapolation caution' in text(sections)
    assert 'maximum doubling extensions = 16' in text(sections)
    np.testing.assert_allclose([float(row[4]) for row in report_table(sections, 'Predictions and displayed uncertainty bounds')['rows'][1:]], result.time_lower, rtol=1e-5)
    json.dumps(records, allow_nan=False)
    json.dumps(sections, allow_nan=False)


@pytest.mark.parametrize('method', ['time_mapping', 'tau_ratio'])
def test_interval_only_caution_survives_plot_and_export(monkeypatch, method):
    result = PredictedTrajectory('approach1' if method == 'time_mapping' else 'approach2', method,
        np.array([4.]), np.array([.5]), np.array([9.]), np.array([.5]),
        time_lower=np.array([8.]), time_upper=np.array([12.]))
    annotate_prediction_domain(result, in_vitro_time=[0, 5], in_vivo_time=[0, 10],
                                paired_in_vitro_time=[0, 5], paired_in_vivo_time=[0, 10])
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **k: '<plot>')
    data = calibration()
    if method == 'time_mapping':
        plot = engine.create_prediction_plot_a1(data, result.in_vitro_time, result.in_vitro_value,
            result.in_vivo_time, {'model_name': 'Linear'}, trajectory=result)
    else:
        plot = engine.create_prediction_plot_a2(data, result.in_vitro_time, result.in_vitro_value,
            result.in_vivo_time, data[0], data[1], data[1], {'model_name': 'Exponential'},
            time_prediction=result, prediction_mode=method)
    assert plot['extrapolation_warnings'][0]['label'] == 'Interval extends beyond calibration'
    assert plot['prediction_records'][0]['extrapolated'] is False
    assert plot['prediction_records'][0]['upper_95'] == 12.


def test_badge_macro_is_text_labelled_and_escaped_and_template_is_valid():
    source = Path('templates/results.html').read_text()
    Environment().parse(source)
    macro = source[:source.index('{%- endmacro %}') + len('{%- endmacro %}')]
    template = Template(macro + '{{ render_prediction_cautions(items) }}', autoescape=True)
    rendered = template.render(items=[{'label': 'Extrapolation', 'message': '<outside>'}])
    assert 'role="note"' in rendered and 'Extrapolation' in rendered
    assert '&lt;outside&gt;' in rendered and '<outside>' not in rendered
    assert template.render(items=[]).strip() == ''


def result_dict():
    return {'approach2:exponential': {
        'params': [{'a': 1., 'b': -.1}, {'a': 1., 'b': -.1}],
        'pcov': [np.zeros((2, 2)), np.zeros((2, 2))], 'residuals': [np.zeros(6), np.zeros(6)],
        'tau': [SimpleNamespace(n=10., s=0.), SimpleNamespace(n=10., s=0.)],
        'residual_dof': [4, 4],
    }}


def test_app_adapter_keeps_failed_inverse_row_statuses_available_without_a_plot(monkeypatch):
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *a, **k: {'plot': '<plot>', 'table_html': ''})
    output = engine.process_predictions(calibration(), [np.array([1.]), np.array([0.])], result_dict(), ['approach2'])
    inverse = output['approach2']['exponential']
    assert inverse['plot_inverse_response'] is None
    assert 'flat' in inverse['prediction_tables']['observed_response_inverse'][0]['status']
    assert 'flat' in inverse['inverse_response_table_html']


def test_app_adapter_passes_applied_log_time_domain(monkeypatch):
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *a, **k: {'plot': '<plot>', 'table_html': ''})
    output = engine.process_predictions(calibration(), [np.array([1.]), np.array([2.])],
                                        result_dict(), ['approach2'], selected_scalings=['log_x'])
    prediction = output['approach2']['exponential']
    assert prediction['inverse_response_predictions']['predicted_in_vivo_time'][0] == pytest.approx(-10 * np.log(2.))
    assert prediction['prediction_tables']['observed_response_inverse'][0]['inverse_minimum_time'] is None


@pytest.mark.parametrize('method', ['method1_value_scaling', 'method2_time_scaling'])
def test_direct_mapping_export_has_domain_and_availability_status(monkeypatch, method):
    monkeypatch.setattr(engine.pio, 'to_html', lambda fig, **k: '<plot>')
    data = calibration()
    output = engine.process_predictions(data, [np.array([3., 20.]), np.array([.6, .5])],
                                        {'approach3': {'plot_v': {}, 'plot_t': {}}}, ['approach3'], interpolated_data=data)
    result = output['approach3'][method]
    records = next(iter(result['prediction_tables'].values()))
    assert 'extrapolation_status' in records[0]
    assert 'Calibration domain status' in result['plot']['table_html']
    if method == 'method1_value_scaling':
        assert records[1]['extrapolation_status'] == 'unavailable prediction'
    else:
        assert records[1]['extrapolated'] is True


def test_source_descriptions_no_longer_describe_bounded_or_clipped_inverse():
    from utilities.descriptions import APPROACH_DESCRIPTIONS
    text = APPROACH_DESCRIPTIONS['approach2']['prediction_error_html']
    assert 'never clips' in text
    for path in ('docs/user_guide.tex', 'docs/appendices/glossary.tex', 'ivivc_core/README.md'):
        source = Path(path).read_text()
        assert 'does not automatically identify' not in source
        assert 'mapped to that calibration-time endpoint' not in source
    assert 'restricted to the observed in vivo calibration range' not in text


@pytest.mark.skipif(importlib.util.find_spec('flask') is None, reason='requires full app dependencies')
def test_flask_request_includes_inverse_extrapolation_badge_and_report():
    from app import app
    from test_report_route import ReportJSON
    t = np.arange(6.)
    rows = '\n'.join(f'{a},{np.exp(-.2*a)},{2*a},{np.exp(-.2*a)}' for a in t)
    form = {'file': (BytesIO(('t1,y1,t2,y2\n' + rows).encode()), 'calibration.csv'),
        'prediction_file': (BytesIO(b't,y\n2,.2\n'), 'prediction.csv'),
        'approaches': ['approach2'], 'models': ['approach2:exponential'], 'metrics': ['rmse'],
        'interpolation': ['none'], 'cv_scheme': 'none', 'grid_search_num_points': '1',
        'grid_search_num_cores': '1'}
    app.config.update(TESTING=True)
    response = app.test_client().post('/', data=form, content_type='multipart/form-data')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'prediction-domain-badge' in html and 'Extrapolation' in html
    parser = ReportJSON(); parser.feed(html)
    reports = json.loads(''.join(parser.fragments))
    assert any(section.get('title') == 'Calibration-domain cautions' for sections in reports.values() for section in sections)
