from __future__ import annotations

import numpy as np

from ivivc_core import (
    AnalysisError,
    IVIVCAnalysis,
    ResultBatch,
    TimeCourse,
    fit_ivivc,
    fit_time_course,
    generate_trajectory_ensembles,
    predict_in_vivo,
    prepare_data,
)


def _paired_exponential_analysis(fast_analysis_config) -> IVIVCAnalysis:
    in_vitro_time = np.linspace(0.0, 5.0, 6)
    perturbation = np.array([1.0, 0.99, 1.01, 0.995, 1.005, 1.0])
    in_vitro_response = np.exp(-0.2 * in_vitro_time) * perturbation
    in_vivo_time = 2.0 * in_vitro_time
    in_vivo_response = np.exp(-0.1 * in_vivo_time) * perturbation
    prepared = prepare_data(
        in_vitro_time,
        in_vitro_response,
        in_vivo_time,
        in_vivo_response,
        selected_interpolation=['none'],
    )
    analysis_config = dict(fast_analysis_config)
    analysis_config['grid_search_num_points'] = 5
    return fit_ivivc(
        prepared,
        selections=[
            'approach1:linear',
            'approach2:exponential',
            'approach3',
        ],
        selected_metrics=['rmse'],
        analysis_config=analysis_config,
    )


def test_fit_time_course_batches_successful_models_and_errors(fast_analysis_config):
    time = np.arange(5.0)
    response = 2.0 * time

    batch = fit_time_course(
        time,
        response,
        model_names=['linear_zero_intercept', 'not_a_registered_model'],
        selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        approach_id='approach1',
    )

    assert isinstance(batch, ResultBatch)
    assert batch.successful_keys == ('linear_zero_intercept',)
    assert batch.failed_keys == ('not_a_registered_model',)
    assert np.isclose(batch['linear_zero_intercept'].parameters['a'], 2.0)
    assert isinstance(batch.errors['not_a_registered_model'], AnalysisError)
    assert batch.errors['not_a_registered_model'].exception_type == 'KeyError'


def test_fit_ivivc_returns_structured_model_components(fast_analysis_config):
    analysis = _paired_exponential_analysis(fast_analysis_config)

    assert isinstance(analysis, IVIVCAnalysis)
    assert analysis.failed_keys == ()
    assert analysis.successful_keys == (
        'approach1:linear',
        'approach2:exponential',
        'approach3',
    )

    approach1 = analysis['approach1:linear']
    assert approach1.time_mapping_fit is not None
    assert set(approach1.profile_fits) == {'time_mapping'}
    assert np.isclose(approach1.time_mapping_fit.parameters['b'], 2.0, rtol=1e-4)

    approach2 = analysis['approach2:exponential']
    assert set(approach2.profile_fits) == {'in_vitro', 'in_vivo'}
    assert np.isclose(approach2.in_vitro_fit.parameters['b'], -0.2, rtol=0.05)
    assert np.isclose(approach2.in_vivo_fit.parameters['b'], -0.1, rtol=0.05)
    assert np.isclose(approach2.in_vitro_fit.tau_estimate, 5.0, rtol=0.10)
    assert np.isclose(approach2.in_vivo_fit.tau_estimate, 10.0, rtol=0.10)

    approach3 = analysis['approach3']
    assert approach3.model_name is None
    assert approach3.profile_fits == {}


def test_predict_in_vivo_uses_one_common_public_result_contract(fast_analysis_config):
    analysis = _paired_exponential_analysis(fast_analysis_config)
    prediction_time = np.array([0.0, 2.5, 5.0])
    prediction = TimeCourse(
        prediction_time,
        np.exp(-0.2 * prediction_time),
    )

    batch = predict_in_vivo(
        analysis,
        prediction,
        methods={
            'approach1': 'time_mapping',
            'approach2': (
                'value_ratio',
                'tau_ratio',
                'observed_response_inverse',
            ),
        },
        n_samples=60,
    )

    assert batch.errors == {}
    ratio = batch['approach2:exponential:value_ratio']
    assert ratio.value_lower is not None and ratio.value_upper is not None
    assert ratio.auxiliary['interval_kind'] == 'parameter_uncertainty'
    assert ratio.auxiliary['interval_requested_samples'] == 60
    assert set(batch.results) == {
        'approach1:linear:time_mapping',
        'approach2:exponential:value_ratio',
        'approach2:exponential:tau_ratio',
        'approach2:exponential:observed_response_inverse',
    }
    np.testing.assert_allclose(
        batch['approach1:linear:time_mapping'].in_vivo_time,
        2.0 * prediction_time,
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        batch['approach2:exponential:value_ratio'].in_vivo_value,
        np.exp(-0.1 * prediction_time),
        rtol=0.05,
    )
    np.testing.assert_allclose(
        batch['approach2:exponential:tau_ratio'].in_vivo_time,
        2.0 * prediction_time,
        rtol=0.05,
    )
    np.testing.assert_allclose(
        batch['approach2:exponential:observed_response_inverse'].in_vivo_time,
        2.0 * prediction_time,
        rtol=0.05,
        atol=1e-6,
    )


def test_generate_trajectory_ensembles_is_seeded_and_model_agnostic(
    fast_analysis_config,
):
    analysis = _paired_exponential_analysis(fast_analysis_config)
    prediction_time = np.array([0.0, 2.5, 5.0])
    prediction = TimeCourse(
        prediction_time,
        np.exp(-0.2 * prediction_time),
    )
    methods = {
        'approach1': 'time_mapping',
        'approach2': ('value_ratio', 'tau_ratio', 'observed_response_inverse'),
        'approach3': ('value_scaling', 'time_scaling'),
    }

    first = generate_trajectory_ensembles(
        analysis,
        prediction,
        methods=methods,
        n_samples=20,
        random_state=12345,
    )
    second = generate_trajectory_ensembles(
        analysis,
        prediction,
        methods=methods,
        n_samples=20,
        random_state=12345,
    )

    assert first.errors == {}
    assert set(first.results) == {
        'approach1:linear:time_mapping',
        'approach2:exponential:value_ratio',
        'approach2:exponential:tau_ratio',
        'approach2:exponential:observed_response_inverse',
        'approach3:value_scaling',
        'approach3:time_scaling',
    }

    for key in (
        'approach1:linear:time_mapping',
        'approach2:exponential:value_ratio',
        'approach2:exponential:tau_ratio',
        'approach2:exponential:observed_response_inverse',
    ):
        assert first[key].n_draws == 20
        assert first[key].n_valid_draws > 0
        np.testing.assert_allclose(first[key].time_draws, second[key].time_draws)
        np.testing.assert_allclose(first[key].value_draws, second[key].value_draws)

    for key in ('approach3:value_scaling', 'approach3:time_scaling'):
        assert first[key].n_draws == 1
        assert first[key].metadata['uncertainty_source'] == 'none'
