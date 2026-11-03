"""Approach 2 consumes calibration errors without adding observation noise."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.stats import norm, t

from ivivc_core import (
    ObservationUncertainty,
    TimeCourse,
    fit_ivivc,
    fit_profile,
    fit_time_course,
    generate_trajectory_ensembles,
    predict_in_vivo,
    prepare_data,
    profile_curve_intervals,
    profile_parameter_intervals,
    resolve_observation_uncertainty,
)
from ivivc_core.intervals import curve_parameter_intervals
from models import models


MODEL = 'exponential_normalized'
SELECTION = 'approach2:' + MODEL


def _data():
    time = np.linspace(0.0, 5.0, 8)
    response = np.exp(-0.3 * time) + np.array([0.0, 0.008, -0.01, 0.004, -0.005, 0.006, -0.003, 0.004])
    sigma = np.array([0.025, 0.03, 0.02, 0.04, 0.035, 0.025, 0.04, 0.03])
    return time, response, sigma


def _analysis(config, *, scale=1.0, vitro=True, vivo=True, normalize=False):
    time, response, sigma = _data()
    vitro_scale, vivo_scale = (10.0, 100.0) if normalize else (1.0, 1.0)
    prepared = prepare_data(
        time, vitro_scale * response, 2.0 * time, vivo_scale * response,
        selected_normalizations=['initial_value'] if normalize else [],
        in_vitro_uncertainty=ObservationUncertainty(
            scale * vitro_scale * sigma, response_kind='mean',
            replicate_counts=4, source='vitro SEM', response_units='mg' if normalize else 'fraction remaining',
        ) if vitro else None,
        in_vivo_uncertainty=ObservationUncertainty(
            scale * vivo_scale * sigma, response_kind='mean',
            replicate_counts=6, source='vivo SEM', response_units='mg' if normalize else 'fraction remaining',
        ) if vivo else None,
    )
    analysis = fit_ivivc(
        prepared, selections=[SELECTION], selected_metrics=['rmse'],
        analysis_config=config,
    )
    assert analysis.errors == {}
    return analysis


def _prediction():
    time = np.array([1.0, 2.0, 3.0])
    return TimeCourse(time, np.exp(-0.3 * time))


def test_fit_time_course_uses_absolute_uncertainty_and_preserves_independent_model_errors(fast_analysis_config):
    time, response, sigma = _data()
    supplied = ObservationUncertainty(sigma, response_kind='mean', source='reported SEM')
    batch = fit_time_course(
        time, response, model_names=[MODEL, 'not_registered'], selected_metrics=['rmse'],
        analysis_config=fast_analysis_config, approach_id='approach2',
        observation_uncertainty=supplied,
    )
    assert batch.successful_keys == (MODEL,)
    assert batch.failed_keys == ('not_registered',)
    fit = batch[MODEL]
    expected, covariance = curve_fit(
        models[MODEL]['model_function'], time, response, p0=[-0.3],
        sigma=sigma, absolute_sigma=True,
    )
    np.testing.assert_allclose(fit.parameter_vector, expected, rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-5)
    np.testing.assert_array_equal(fit.observation_uncertainty.sigma, sigma)
    assert fit.fit_diagnostics['covariance_scaling'] == 'absolute'


def test_fit_time_course_accepts_resolved_errors_and_honors_fail_fast(fast_analysis_config):
    time, response, sigma = _data()
    resolved = resolve_observation_uncertainty(ObservationUncertainty(sigma, response_kind='mean'), len(time))
    batch = fit_time_course(
        time, response, model_names=MODEL, selected_metrics=['rmse'],
        analysis_config=fast_analysis_config, observation_uncertainty=resolved,
    )
    np.testing.assert_array_equal(batch[MODEL].observation_uncertainty.sigma, sigma)
    with pytest.raises(KeyError):
        fit_time_course(
            time, response, model_names='not_registered', selected_metrics=['rmse'],
            observation_uncertainty=resolved, fail_fast=True,
        )


def test_paired_approach2_matches_the_weighted_standalone_source_fits(fast_analysis_config):
    analysis = _analysis(fast_analysis_config)
    result = analysis[SELECTION]
    for course, actual in (
        (analysis.prepared_data.in_vitro, result.in_vitro_fit),
        (analysis.prepared_data.in_vivo, result.in_vivo_fit),
    ):
        expected = fit_profile(
            course.time, course.value, model_name=MODEL, selected_metrics=['rmse'],
            analysis_config=fast_analysis_config, approach_id='approach2', calculate_tau=True,
            observation_uncertainty=course.observation_uncertainty,
        )
        np.testing.assert_array_equal(actual.parameter_vector, expected.parameter_vector)
        np.testing.assert_array_equal(actual.covariance, expected.covariance)
        np.testing.assert_array_equal(actual.observation_uncertainty.sigma, course.observation_uncertainty.sigma)
        assert actual.residual_dof == len(course.value) - 1
        assert np.isfinite(actual.tau_estimate)


@pytest.mark.parametrize('vitro,vivo', [(True, False), (False, True)])
def test_mixed_supplied_and_unspecified_calibration_scales_stay_distinct(fast_analysis_config, vitro, vivo):
    result = _analysis(fast_analysis_config, vitro=vitro, vivo=vivo)[SELECTION]
    for fit, supplied in ((result.in_vitro_fit, vitro), (result.in_vivo_fit, vivo)):
        assert (fit.observation_uncertainty is not None) is supplied
        summary = profile_parameter_intervals(fit)
        assert summary['covariance_scaling'] == ('absolute' if supplied else 'residual_scaled')
        assert summary['distribution'] == ('normal' if supplied else 'student_t')


def test_mixed_approach_selections_keep_approach2_and_report_unsupported_uncertainty(fast_analysis_config):
    base = _analysis(fast_analysis_config)
    analysis = fit_ivivc(
        base.prepared_data,
        selections=['approach1:linear', SELECTION, 'approach3'],
        selected_metrics=['rmse'], analysis_config=fast_analysis_config,
    )
    assert analysis.successful_keys == (SELECTION,)
    assert analysis.failed_keys == ('approach1:linear', 'approach3')
    for error in analysis.errors.values():
        assert error.exception_type == 'ValueError'
    with pytest.raises(ValueError, match='Approach 2'):
        fit_ivivc(
            base.prepared_data, selections=['approach1:linear', SELECTION],
            selected_metrics=['rmse'], analysis_config=fast_analysis_config, fail_fast=True,
        )


def test_uncertainty_present_only_in_provenance_cannot_be_silently_ignored(fast_analysis_config):
    prepared = _analysis(fast_analysis_config).prepared_data
    prepared.in_vitro = TimeCourse(prepared.in_vitro.time, prepared.in_vitro.value)
    with pytest.raises(ValueError, match='processed source profile'):
        fit_ivivc(prepared, selections=[SELECTION], selected_metrics=['rmse'], analysis_config=fast_analysis_config)


def test_fixed_normalization_and_counts_survive_approach2_calibration(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, normalize=True)
    _, _, sigma = _data()
    for fit, count in ((analysis[SELECTION].in_vitro_fit, 4), (analysis[SELECTION].in_vivo_fit, 6)):
        np.testing.assert_allclose(fit.observation_uncertainty.sigma, sigma)
        np.testing.assert_array_equal(fit.observation_uncertainty.replicate_counts, [count] * len(sigma))
        assert fit.observation_uncertainty.response_units == 'dimensionless'
        assert fit.residual_dof == len(sigma) - 1
    record = analysis.prepared_data.preprocessing.normalizations[0]
    assert record.in_vitro_parameters['constants_treated_as_fixed'] is True
    assert record.in_vivo_parameters['constants_treated_as_fixed'] is True
    trajectory = predict_in_vivo(
        analysis, _prediction(), methods={'approach2': ('value_ratio',)}, n_samples=30,
    )[SELECTION + ':value_ratio']
    normalization = trajectory.auxiliary['fixed_normalization_constants'][0]
    assert normalization['name'] == 'initial_value'
    assert normalization['in_vitro_parameters']['initial_value'] == 10.0
    assert normalization['in_vivo_parameters']['initial_value'] == 100.0
    assert normalization['in_vitro_parameters']['constants_treated_as_fixed'] is True
    assert normalization['in_vivo_parameters']['constants_treated_as_fixed'] is True


def test_absolute_parameter_intervals_use_normal_quantiles_and_residual_fits_use_t(fast_analysis_config):
    time, response, _ = _data()
    weighted = fit_time_course(
        time, response, model_names=MODEL, selected_metrics=['rmse'], analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty(0.04, response_kind='mean'),
    )[MODEL]
    unweighted = fit_time_course(
        time, response, model_names=MODEL, selected_metrics=['rmse'], analysis_config=fast_analysis_config,
    )[MODEL]
    absolute = profile_parameter_intervals(weighted)
    residual = profile_parameter_intervals(unweighted)
    assert absolute['distribution'] == 'normal'
    assert absolute['method'] == 'normal_wald'
    assert absolute['scale_source'] == 'supplied_observation_uncertainty'
    assert absolute['multiplier'] == norm.ppf(0.975)
    assert residual['distribution'] == 'student_t'
    assert residual['method'] == 'student_t_wald'
    assert residual['multiplier'] == t.ppf(0.975, unweighted.residual_dof)
    for fit, summary in ((weighted, absolute), (unweighted, residual)):
        standard_error = np.sqrt(np.diag(fit.covariance))
        np.testing.assert_array_equal(summary['estimate'], fit.parameter_vector)
        np.testing.assert_allclose(summary['standard_error'], standard_error)
        np.testing.assert_allclose(summary['lower'], fit.parameter_vector - summary['multiplier'] * standard_error)
        np.testing.assert_allclose(summary['upper'], fit.parameter_vector + summary['multiplier'] * standard_error)
        assert np.all(summary['available'])


def test_parameter_intervals_preserve_the_legacy_nonpositive_df_fallback(fast_analysis_config):
    fit = _analysis(fast_analysis_config)[SELECTION].in_vitro_fit
    fallback = replace(fit, observation_uncertainty=None, fit_diagnostics={}, residual_dof=0)
    summary = profile_parameter_intervals(fallback)
    assert summary['distribution'] == 'normal_fallback'
    assert summary['method'] == 'normal_wald_fallback'
    assert summary['multiplier'] == 1.96


@pytest.mark.parametrize('covariance', [np.array([[np.nan]]), np.array([[-0.1]])])
def test_invalid_parameter_variance_is_unavailable_instead_of_zero(fast_analysis_config, covariance):
    fit = _analysis(fast_analysis_config)[SELECTION].in_vitro_fit
    summary = profile_parameter_intervals(replace(fit, covariance=covariance))
    assert not np.any(summary['available'])
    assert np.all(np.isnan(summary['lower']))
    assert np.all(np.isnan(summary['upper']))


@pytest.mark.parametrize('confidence_level', [0.0, 1.0, np.nan])
def test_parameter_interval_confidence_level_is_validated(fast_analysis_config, confidence_level):
    fit = _analysis(fast_analysis_config)[SELECTION].in_vitro_fit
    with pytest.raises(ValueError):
        profile_parameter_intervals(fit, confidence_level=confidence_level)


def test_curve_intervals_reuse_joint_covariance_draws_and_expose_the_scale_source(fast_analysis_config):
    time = np.arange(5.0)
    fit = fit_profile(
        time, [1.0, 3.1, 4.9, 7.2, 9.0], model_name='linear', selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty([0.2, 0.3, 0.4, 0.2, 0.3], response_kind='mean'),
    )
    grid = np.array([0.0, 1.5, 4.0])
    actual = profile_curve_intervals(fit, grid, n_samples=100, random_state=17)
    expected = curve_parameter_intervals(
        models['linear']['model_function'], grid, fit.parameters, fit.covariance,
        n_samples=100, random_state=17,
    )
    assert fit.covariance[0, 1] != 0.0
    for key in ('lower', 'upper', 'std', 'valid_counts'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['interval_covariance_scaling'] == 'absolute'
    assert actual['interval_scale_source'] == 'supplied_observation_uncertainty'
    assert actual['interval_distribution'] == 'multivariate_normal'
    assert actual['observation_noise_included'] is False


@pytest.mark.parametrize('covariance', [
    np.array([[1.0, 2.0], [2.0, 1.0]]),
    np.array([[0.01, 0.008], [0.001, 0.01]]),
])
def test_invalid_joint_covariance_is_consistently_unavailable_for_parameters_and_curves(
    fast_analysis_config, covariance,
):
    time = np.arange(5.0)
    fit = fit_profile(
        time, 1.0 + 2.0 * time, model_name='linear', selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty(0.2, response_kind='mean'),
    )
    fit = replace(fit, covariance=covariance)
    parameters = profile_parameter_intervals(fit)
    rng = np.random.default_rng(19)
    state = dict(rng.bit_generator.state)
    curve = profile_curve_intervals(fit, time, n_samples=20, random_state=rng)
    assert parameters['covariance_status'] != 'available'
    assert not np.any(parameters['available'])
    assert np.all(np.isnan(curve['lower']))
    assert np.all(np.isnan(curve['upper']))
    assert curve['interval_covariance_status'] != 'available'
    assert curve['interval_sampling_error']
    assert rng.bit_generator.state == state


@pytest.mark.parametrize('covariance', [np.zeros((2, 2)), np.ones((2, 2))])
def test_singular_valid_covariance_remains_supported(fast_analysis_config, covariance):
    time = np.arange(5.0)
    fit = fit_profile(
        time, 1.0 + 2.0 * time, model_name='linear', selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        observation_uncertainty=ObservationUncertainty(0.2, response_kind='mean'),
    )
    fit = replace(fit, covariance=covariance)
    parameters = profile_parameter_intervals(fit)
    curve = profile_curve_intervals(fit, time, n_samples=20, random_state=19)
    assert parameters['covariance_status'] == 'available'
    assert np.all(parameters['available'])
    assert np.all(np.isfinite(curve['lower']))
    assert np.all(np.isfinite(curve['upper']))


def test_scaling_absolute_errors_changes_parameter_and_curve_uncertainty_not_the_fit(fast_analysis_config):
    base = _analysis(fast_analysis_config)[SELECTION].in_vitro_fit
    larger = _analysis(fast_analysis_config, scale=3.0)[SELECTION].in_vitro_fit
    np.testing.assert_allclose(base.parameter_vector, larger.parameter_vector, rtol=1e-6)
    np.testing.assert_allclose(larger.covariance, 9.0 * base.covariance, rtol=1e-5)
    small = profile_curve_intervals(base, [1.0, 2.0, 3.0], n_samples=100, random_state=17)
    large = profile_curve_intervals(larger, [1.0, 2.0, 3.0], n_samples=100, random_state=17)
    assert np.all(large['upper'] - large['lower'] > small['upper'] - small['lower'])


def test_calibration_error_changes_existing_value_and_inverse_prediction_intervals(fast_analysis_config):
    small = _analysis(fast_analysis_config)
    large = _analysis(fast_analysis_config, scale=3.0)
    methods = {'approach2': ('value_ratio', 'observed_response_inverse')}
    small_predictions = predict_in_vivo(small, _prediction(), methods=methods, n_samples=80, random_state=17)
    large_predictions = predict_in_vivo(large, _prediction(), methods=methods, n_samples=80, random_state=17)
    assert small_predictions.errors == {}
    assert large_predictions.errors == {}
    for method, prefix in (('value_ratio', 'value'), ('observed_response_inverse', 'time')):
        first = small_predictions[SELECTION + ':' + method]
        second = large_predictions[SELECTION + ':' + method]
        np.testing.assert_allclose(first.in_vivo_value, second.in_vivo_value, rtol=1e-6)
        np.testing.assert_allclose(first.in_vivo_time, second.in_vivo_time, rtol=1e-6)
        small_width = getattr(first, prefix + '_upper') - getattr(first, prefix + '_lower')
        large_width = getattr(second, prefix + '_upper') - getattr(second, prefix + '_lower')
        assert np.all(np.isfinite(small_width))
        assert np.all(large_width > small_width)
        assert first.auxiliary['prediction_inputs_fixed'] is True
        assert first.auxiliary['future_observation_variability_included'] is False


def test_complete_ensembles_retain_weighted_calibration_provenance_and_dependencies(fast_analysis_config):
    analysis = _analysis(fast_analysis_config)
    methods = {'approach2': ('value_ratio', 'tau_ratio', 'observed_response_inverse')}
    first = generate_trajectory_ensembles(analysis, _prediction(), methods=methods, n_samples=30, random_state=19)
    second = generate_trajectory_ensembles(analysis, _prediction(), methods=methods, n_samples=30, random_state=19)
    assert first.errors == {}
    assert set(first.results) == {SELECTION + ':' + name for name in methods['approach2']}
    for key, ensemble in first.results.items():
        assert ensemble.n_draws == 30
        assert ensemble.n_valid_draws == 30
        np.testing.assert_array_equal(ensemble.time_draws, second[key].time_draws)
        np.testing.assert_array_equal(ensemble.value_draws, second[key].value_draws)
        assert ensemble.metadata['prediction_inputs_fixed'] is True
        assert ensemble.metadata['future_observation_variability_included'] is False
        calibration = ensemble.metadata['calibration_uncertainty']
        for source in ('in_vitro', 'in_vivo'):
            assert calibration[source]['covariance_scaling'] == 'absolute'
            assert calibration[source]['observation_uncertainty']['uncertainty_kind'] == 'sem'
        if key.endswith(':value_ratio'):
            # For these normalized exponentials each draw has one rate
            # difference shared across all times, rather than independent
            # pointwise error bars sampled into an invented trajectory.
            rate_difference = np.log(ensemble.value_draws / _prediction().value) / _prediction().time
            np.testing.assert_allclose(rate_difference, np.repeat(rate_difference[:, :1], 3, axis=1),
                                       rtol=1e-11, atol=1e-12)
        else:
            time_ratio = ensemble.time_draws / _prediction().time
            np.testing.assert_allclose(time_ratio, np.repeat(time_ratio[:, :1], 3, axis=1),
                                       rtol=1e-11, atol=1e-12)


def test_output_metadata_keeps_supplied_and_residual_calibration_distinct(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, vivo=False)
    result = predict_in_vivo(
        analysis, _prediction(), methods={'approach2': ('value_ratio',)}, n_samples=30,
    )[SELECTION + ':value_ratio']
    calibration = result.auxiliary['calibration_uncertainty']
    assert calibration['in_vitro']['covariance_scaling'] == 'absolute'
    assert calibration['in_vivo']['covariance_scaling'] == 'residual_scaled'
    assert calibration['in_vivo']['observation_uncertainty'] is None


@pytest.mark.parametrize('legacy_source', ['in_vitro', 'in_vivo'])
def test_mixed_weighted_and_legacy_source_fits_preserve_predictions_and_metadata(
    fast_analysis_config, legacy_source,
):
    analysis = _analysis(
        fast_analysis_config,
        vitro=legacy_source != 'in_vitro', vivo=legacy_source != 'in_vivo',
    )
    methods = {'approach2': ('value_ratio', 'observed_response_inverse')}
    operations = (predict_in_vivo, generate_trajectory_ensembles)
    options = dict(methods=methods, n_samples=10, random_state=19)
    expected = [operation(analysis, _prediction(), **options) for operation in operations]
    model_fit = analysis[SELECTION]
    fit = getattr(model_fit, legacy_source + '_fit')
    setattr(model_fit, legacy_source + '_fit', SimpleNamespace(
        parameters=fit.parameters, covariance=fit.covariance, residuals=fit.residuals,
    ))

    for operation, before in zip(operations, expected):
        after = operation(analysis, _prediction(), **options)
        assert before.errors == after.errors == {}
        assert before.successful_keys == after.successful_keys
        for key, output in after.results.items():
            reference = before[key]
            if operation is generate_trajectory_ensembles:
                np.testing.assert_array_equal(output.time_draws, reference.time_draws)
                np.testing.assert_array_equal(output.value_draws, reference.value_draws)
                assert output.metadata['calibration_uncertainty'] == reference.metadata['calibration_uncertainty']
                output, reference = output.central, reference.central
            np.testing.assert_array_equal(output.in_vivo_time, reference.in_vivo_time)
            np.testing.assert_array_equal(output.in_vivo_value, reference.in_vivo_value)
            calibration = output.auxiliary['calibration_uncertainty']
            assert calibration == reference.auxiliary['calibration_uncertainty']
            assert calibration[legacy_source]['observation_uncertainty'] is None
            assert calibration[legacy_source]['covariance_scaling'] == 'residual_scaled'
            weighted_source = 'in_vivo' if legacy_source == 'in_vitro' else 'in_vitro'
            assert calibration[weighted_source]['covariance_scaling'] == 'absolute'
            assert calibration[weighted_source]['observation_uncertainty'] is not None


def test_legacy_predictions_do_not_gain_weighted_calibration_metadata(fast_analysis_config):
    analysis = _analysis(fast_analysis_config, vitro=False, vivo=False)
    result = predict_in_vivo(
        analysis, _prediction(), methods={'approach2': ('value_ratio',)}, n_samples=30,
    )[SELECTION + ':value_ratio']
    assert 'calibration_uncertainty' not in result.auxiliary
    ensemble = generate_trajectory_ensembles(
        analysis, _prediction(), methods={'approach2': ('value_ratio',)}, n_samples=10,
    )[SELECTION + ':value_ratio']
    assert 'calibration_uncertainty' not in ensemble.metadata


def test_prediction_input_uncertainty_remains_guarded_after_calibration_integration(fast_analysis_config):
    analysis = _analysis(fast_analysis_config)
    prediction = _prediction()
    prediction.observation_uncertainty = resolve_observation_uncertainty(
        ObservationUncertainty(0.02, response_kind='mean'), len(prediction.time),
    )
    for operation in (predict_in_vivo, generate_trajectory_ensembles):
        with pytest.raises(ValueError, match='prediction-response uncertainty'):
            operation(analysis, prediction, methods={'approach2': ('value_ratio',)})
