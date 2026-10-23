from __future__ import annotations

import io

import pytest

pytest.importorskip('flask')

import app as app_module


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
