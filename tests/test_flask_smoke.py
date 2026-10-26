from __future__ import annotations

import io
import json

import pytest

pytest.importorskip('flask')

import app as app_module


def test_setup_lists_each_model_variant_once_and_response_reversal():
    from models import approaches

    with app_module.app.test_client() as client:
        response = client.get('/')
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    for approach_id in ('approach1', 'approach2'):
        for model_name in approaches[approach_id]['models']:
            assert page.count(f'name="models" value="{approach_id}:{model_name}"') == 1
    assert page.count('class="model-family-row"') == len(approaches['approach2']['model_families'])
    assert 'aria-label="Exponential — Normalized"' in page
    assert 'value="one_minus"' in page


def test_fraction_lost_submission_fits_normalized_model_and_records_reversal():
    import numpy as np
    from test_report_route import ReportJSON

    time = np.arange(6.)
    loss = 1.0 - np.exp(-0.2 * time)
    rows = '\n'.join(f'{t},{y},{2*t},{y}' for t, y in zip(time, loss))
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = client.post('/', data={
            'file': (io.BytesIO(('t1,y1,t2,y2\n' + rows).encode()), 'loss.csv'),
            'approaches': ['approach2'], 'models': ['approach2:exponential_normalized'],
            'normalizations': ['one_minus'], 'metrics': ['rmse'], 'interpolation': ['none'],
            'cv_scheme': 'none', 'grid_search_num_points': '3', 'grid_search_num_cores': '1',
            'grid_search_param_min': '-2', 'grid_search_param_max': '2',
        }, content_type='multipart/form-data')
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Exponential (Normalized)' in page
    parser = ReportJSON()
    parser.feed(page)
    reports = json.loads(''.join(parser.fragments))
    assert reports
    for sections in reports.values():
        references = next(section for section in sections if
            section.get('title') == 'Normalization reference values in application order')
        assert any(row[:3] == ['in_vitro', 'one_minus', '1'] for row in references['rows'][1:])
        assert any(row[:3] == ['in_vivo', 'one_minus', '1'] for row in references['rows'][1:])


def test_browser_submission_smoke_returns_results_page(paired_degradation_data):
    t1, m1, t2, m2 = paired_degradation_data
    csv_text = 'time1,value1,time2,value2\n' + '\n'.join(
        f'{a},{b},{c},{d}' for a, b, c, d in zip(t1, m1, t2, m2)
    )

    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = client.post(
            '/',
            data={
                'file': (io.BytesIO(csv_text.encode('utf-8')), 'fit.csv'),
                'approaches': 'approach1',
                'models': 'approach1:linear_zero_intercept',
                'metrics': ['r_squared', 'rmse'],
                'interpolation': 'none',
                'cv_scheme': 'none',
                'grid_search_num_points': '3',
                'grid_search_num_cores': '1',
                'grid_search_param_min': '-2',
                'grid_search_param_max': '2',
                'grid_search_random_state': '12345',
            },
            content_type='multipart/form-data',
        )

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Linear (zero intercept)' in page
    assert 'Approach 1' in page
    assert 'Model Comparison' in page
