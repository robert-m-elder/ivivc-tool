from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def paired_degradation_data():
    """Small deterministic paired dataset used across characterization tests."""
    t_in_vitro = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    response = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    t_in_vivo = 2.0 * t_in_vitro
    return t_in_vitro, response.copy(), t_in_vivo, response.copy()


@pytest.fixture
def fast_analysis_config():
    """Keep characterization tests deterministic and inexpensive."""
    return {
        'cv_scheme': 'none',
        'cv_n_splits': 2,
        'cv_test_size': 0.25,
        'cv_random_state': 12345,
        'grid_search_num_points': 3,
        'grid_search_num_cores': 1,
        'grid_search_param_min': -2.0,
        'grid_search_param_max': 2.0,
        'grid_search_random_state': 12345,
        'include_raw_residual_plots': False,
        'include_residual_qq_plots': False,
        'include_cv_fit_variability_plot': False,
    }


@pytest.fixture
def stub_fit_plots(monkeypatch):
    """Replace Plotly rendering while preserving the process_data contract."""
    import engine

    sentinel_a1 = {'plot': 'approach1-plot', 'table_html': 'approach1-table'}
    sentinel_a2 = {'plot': 'approach2-plot', 'table_html': 'approach2-table'}
    sentinel_a3_value = {'plot': 'approach3-value-plot', 'table_html': 'approach3-value-table'}
    sentinel_a3_time = {'plot': 'approach3-time-plot', 'table_html': 'approach3-time-table'}

    monkeypatch.setattr(engine, 'create_plotly_a1', lambda *args, **kwargs: sentinel_a1)
    monkeypatch.setattr(engine, 'create_plotly_a2', lambda *args, **kwargs: sentinel_a2)

    def fake_a3(*args, **kwargs):
        return sentinel_a3_time if kwargs.get('ptype') == 't' else sentinel_a3_value

    monkeypatch.setattr(engine, 'create_plotly_a3', fake_a3)
    return {
        'approach1': sentinel_a1,
        'approach2': sentinel_a2,
        'approach3_value': sentinel_a3_value,
        'approach3_time': sentinel_a3_time,
    }
