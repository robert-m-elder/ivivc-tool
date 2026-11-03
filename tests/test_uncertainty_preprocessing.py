"""Observation uncertainty remains aligned and explicit through preprocessing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from ivivc_core import (
    ObservationUncertainty,
    TimeCourse,
    fit_profile,
    prepare_data,
    prepare_ivivc_data,
    prepare_time_course,
    resolve_observation_uncertainty,
)


def _sem(values, counts=None):
    return ObservationUncertainty(
        values, response_kind='mean', replicate_counts=counts,
        source='reported SEM', response_units='fraction remaining',
    )


def test_single_course_cleans_time_response_and_uncertainty_with_one_row_mask():
    time = np.array([0.0, 1.0, np.nan, 3.0])
    value = np.array([1.0, np.nan, 0.6, 0.4])
    uncertainty = _sem([0.1, np.nan, np.nan, 0.2], [3, np.nan, np.nan, 5])
    course, metadata = prepare_time_course(
        time, value, observation_uncertainty=uncertainty, label='reference',
    )

    np.testing.assert_array_equal(course.time, [0.0, 3.0])
    np.testing.assert_array_equal(course.value, [1.0, 0.4])
    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.1, 0.2])
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [3, 5])
    source = metadata.sources['profile']
    np.testing.assert_array_equal(source.time, time)
    np.testing.assert_array_equal(source.value, value)
    np.testing.assert_array_equal(source.retained_rows, [0, 3])
    np.testing.assert_array_equal(source.observation_uncertainty.standard_uncertainty,
                                  uncertainty.standard_uncertainty)
    assert source.label == 'reference'


def test_constant_uncertainty_is_expanded_to_retained_observations_only():
    course, metadata = prepare_time_course(
        [0.0, np.nan, 2.0, 3.0], [1.0, 0.8, np.nan, 0.4],
        observation_uncertainty=_sem(0.04, 7),
    )

    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.04, 0.04])
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [7, 7])
    np.testing.assert_array_equal(metadata.sources['profile'].retained_rows, [0, 3])
    assert course.observation_uncertainty.input_form == 'constant'


def test_preprocessing_sources_are_defensive_read_only_snapshots():
    time = np.array([1.0, 10.0, 100.0])
    value = np.array([0.9, 0.5, 0.2])
    course, metadata = prepare_time_course(
        time, value, selected_scalings=['log_x'],
        selected_normalizations=['one_minus'], observation_uncertainty=_sem(0.05),
    )
    source = metadata.sources['profile']
    time[:] = 99.0
    value[:] = 99.0

    np.testing.assert_array_equal(source.time, [1.0, 10.0, 100.0])
    np.testing.assert_array_equal(source.value, [0.9, 0.5, 0.2])
    np.testing.assert_array_equal(course.time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(course.value, [0.1, 0.5, 0.8])
    with pytest.raises(ValueError):
        source.time[0] = 4.0
    with pytest.raises(ValueError):
        source.value[0] = 4.0
    with pytest.raises(ValueError):
        source.retained_rows[0] = 1
    with pytest.raises(TypeError):
        metadata.sources['other'] = source


def test_one_minus_preserves_absolute_sigma_semantics_counts_and_provenance():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [0.9, 0.5, 0.2],
        selected_normalizations=['one_minus', 'one_minus'],
        observation_uncertainty=_sem([0.03, 0.05, 0.07], [3, 4, 5]),
    )

    np.testing.assert_allclose(course.value, [0.1, 0.5, 0.8])
    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.03, 0.05, 0.07])
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [3, 4, 5])
    assert course.observation_uncertainty.uncertainty_kind == 'sem'
    assert course.observation_uncertainty.source == 'reported SEM'
    assert course.observation_uncertainty.response_units == 'fraction remaining'
    assert metadata.selected_normalizations == ('one_minus',)
    assert len(metadata.normalizations) == 1
    assert metadata.normalizations[0].in_vitro_parameters['reference'] == 1.0


def test_log_time_changes_time_without_changing_response_uncertainty():
    course, metadata = prepare_time_course(
        [1.0, 10.0, 100.0], [1.0, 0.5, 0.2],
        selected_scalings=['log_x'], observation_uncertainty=_sem([0.03, 0.04, 0.05]),
    )

    np.testing.assert_allclose(course.time, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(course.value, [1.0, 0.5, 0.2])
    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.03, 0.04, 0.05])
    np.testing.assert_array_equal(metadata.sources['profile'].time, [1.0, 10.0, 100.0])
    assert metadata.applied_scalings == ('log_x',)


@pytest.mark.parametrize('normalization', ['initial_value', 'min_max', 'custom_normalization'])
@pytest.mark.parametrize('fallback', [False, True])
def test_unsupported_response_normalization_is_rejected_with_supplied_uncertainty(
    normalization, fallback,
):
    with pytest.raises(ValueError):
        prepare_time_course(
            [0.0, 1.0, 2.0], [0.0, 0.5, 1.0],
            selected_normalizations=[normalization],
            fallback_invalid_log_scaling=fallback, observation_uncertainty=_sem(0.1),
        )


@pytest.mark.parametrize('scaling', ['log_y', 'custom_scaling'])
@pytest.mark.parametrize('fallback', [False, True])
def test_unsupported_response_scaling_is_not_silently_skipped_by_fallback(scaling, fallback):
    with pytest.raises(ValueError):
        prepare_time_course(
            [0.0, 1.0, 2.0], [1.0, 0.5, 0.0],
            selected_scalings=[scaling],
            fallback_invalid_log_scaling=fallback, observation_uncertainty=_sem(0.1),
        )


def test_paired_log_time_fallback_is_atomic_and_preserves_both_uncertainties():
    prepared = prepare_data(
        [1.0, 10.0, 100.0], [1.0, 0.7, 0.2],
        [0.0, 2.0, 4.0], [1.0, 0.6, 0.2],
        selected_scalings=['log_x'], fallback_invalid_log_scaling=True,
        in_vitro_uncertainty=_sem([0.03, 0.04, 0.05]),
        in_vivo_uncertainty=_sem(0.06),
    )

    np.testing.assert_array_equal(prepared.in_vitro.time, [1.0, 10.0, 100.0])
    np.testing.assert_array_equal(prepared.in_vivo.time, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.03, 0.04, 0.05])
    np.testing.assert_array_equal(prepared.in_vivo.observation_uncertainty.sigma, [0.06] * 3)
    assert prepared.preprocessing.applied_scalings == ()
    assert prepared.preprocessing.warnings[0].option == 'Log10 time scaling'


def test_invalid_log_time_remains_an_error_when_fallback_is_disabled():
    with pytest.raises(ValueError):
        prepare_time_course(
            [0.0, 1.0], [1.0, 0.5], selected_scalings=['log_x'],
            observation_uncertainty=_sem(0.03),
        )


def test_paired_profiles_use_independent_original_row_maps_with_mixed_uncertainty():
    prepared = prepare_data(
        [0.0, np.nan, 2.0, 3.0], [1.0, 0.8, np.nan, 0.2],
        [0.0, 2.0, np.nan, 6.0], [1.0, np.nan, 0.6, 0.2],
        selected_interpolation=['none'],
        in_vitro_uncertainty=_sem([0.1, np.nan, np.nan, 0.2], [2, np.nan, np.nan, 4]),
    )

    np.testing.assert_array_equal(prepared.in_vitro.time, [0.0, 3.0])
    np.testing.assert_array_equal(prepared.in_vivo.time, [0.0, 6.0])
    np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.1, 0.2])
    assert prepared.in_vivo.observation_uncertainty is None
    assert prepared.has_observation_uncertainty
    assert set(prepared.preprocessing.sources) == {'in_vitro', 'in_vivo'}
    np.testing.assert_array_equal(prepared.preprocessing.sources['in_vitro'].retained_rows, [0, 3])
    np.testing.assert_array_equal(prepared.preprocessing.sources['in_vivo'].retained_rows, [0, 3])
    assert prepared.preprocessing.sources['in_vivo'].observation_uncertainty is None


@pytest.mark.parametrize('interpolation', ['none', 'default', 'alternative'])
def test_interpolation_does_not_inflate_observation_rows_or_interpolate_sigma(interpolation):
    prepared = prepare_data(
        [0.0, 1.0, 1.0, 3.0], [1.0, 0.8, 0.6, 0.2],
        [0.0, 2.0, 3.0], [1.0, 0.5, 0.2],
        selected_interpolation=[interpolation],
        in_vitro_uncertainty=_sem([0.03, 0.04, 0.05, 0.06]),
        in_vivo_uncertainty=_sem(0.07),
    )

    np.testing.assert_array_equal(prepared.in_vitro.time, [0.0, 1.0, 1.0, 3.0])
    np.testing.assert_array_equal(prepared.in_vitro.value, [1.0, 0.8, 0.6, 0.2])
    np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.03, 0.04, 0.05, 0.06])
    assert prepared.in_vitro.observation_uncertainty.n_observations == 4
    assert prepared.in_vivo.observation_uncertainty.n_observations == 3
    if interpolation == 'alternative':
        assert len(prepared.time_grid) > len(prepared.in_vitro.time)


def test_resolved_uncertainty_is_accepted_and_subset_without_losing_provenance():
    original = resolve_observation_uncertainty(_sem(0.05, [3, 4, 5]), 3)
    course, metadata = prepare_time_course(
        [0.0, np.nan, 2.0], [1.0, 0.7, 0.2], observation_uncertainty=original,
    )

    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.05, 0.05])
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [3, 5])
    assert course.observation_uncertainty.input_form == 'constant'
    assert course.observation_uncertainty.source == original.source
    assert metadata.sources['profile'].observation_uncertainty.input_form == 'constant'
    again = resolve_observation_uncertainty(original, 3, row_mask=[True, False, True])
    np.testing.assert_array_equal(again.sigma, course.observation_uncertainty.sigma)
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(original, 2)


def test_time_course_accepts_only_resolved_row_aligned_uncertainty():
    resolved = resolve_observation_uncertainty(_sem(0.05), 2)
    course = TimeCourse([0.0, 1.0], [1.0, 0.5], observation_uncertainty=resolved)
    assert course.observation_uncertainty.n_observations == 2
    with pytest.raises(TypeError):
        TimeCourse([0.0, 1.0], [1.0, 0.5], observation_uncertainty=_sem(0.05))
    with pytest.raises(ValueError):
        TimeCourse([0.0], [1.0], observation_uncertainty=resolved)


def test_paired_preprocessing_consumes_attached_resolved_uncertainty_once():
    resolved = resolve_observation_uncertainty(_sem(0.05), 3)
    in_vitro = TimeCourse([0.0, 1.0, 2.0], [1.0, 0.7, 0.2],
                         observation_uncertainty=resolved)
    in_vivo = TimeCourse([0.0, 2.0, 4.0], [1.0, 0.6, 0.2])
    prepared = prepare_ivivc_data(in_vitro, in_vivo)

    np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.05] * 3)
    assert prepared.in_vivo.observation_uncertainty is None
    with pytest.raises(ValueError):
        prepare_ivivc_data(in_vitro, in_vivo, in_vitro_uncertainty=_sem(0.1))


def test_single_wrapper_transports_uncertainty_and_rejects_unmatched_in_vivo_error():
    prepared = prepare_data([0.0, 1.0], [1.0, 0.5], in_vitro_uncertainty=_sem(0.05))
    assert prepared.in_vivo is None
    assert prepared.has_observation_uncertainty
    np.testing.assert_array_equal(prepared.in_vitro.observation_uncertainty.sigma, [0.05, 0.05])
    with pytest.raises(ValueError):
        prepare_data([0.0, 1.0], [1.0, 0.5], in_vivo_uncertainty=_sem(0.05))


def test_prepared_uncertainty_cannot_be_silently_discarded_by_legacy_adapters():
    prepared = prepare_data([0.0, 1.0], [1.0, 0.5], in_vitro_uncertainty=_sem(0.05))

    with pytest.raises(ValueError):
        prepared.to_legacy_data()
    with pytest.raises(ValueError):
        prepared.preprocessing.to_legacy_dict()


def test_preprocessed_uncertainty_feeds_single_profile_fitting_without_reconversion(
    fast_analysis_config,
):
    course, _ = prepare_time_course(
        [1.0, 2.0, 3.0, 4.0], [1.1, 2.1, np.nan, 8.0],
        observation_uncertainty=_sem([0.1, 0.2, np.nan, 2.0], [2, 3, np.nan, 4]),
    )
    fit = fit_profile(
        course.time, course.value, model_name='linear_zero_intercept',
        selected_metrics=['rmse'], analysis_config=fast_analysis_config,
        observation_uncertainty=course.observation_uncertainty,
    )
    precision = 1.0 / course.observation_uncertainty.sigma ** 2
    information = np.sum(precision * course.time ** 2)
    expected = np.sum(precision * course.time * course.value) / information

    np.testing.assert_allclose(fit.parameter_vector, [expected], rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, [[1.0 / information]], rtol=1e-5)
    np.testing.assert_array_equal(fit.observation_uncertainty.replicate_counts, [2, 3, 4])
    assert fit.residual_dof == 2


def test_without_uncertainty_legacy_cleaning_and_metadata_shape_are_unchanged():
    course, metadata = prepare_time_course(
        [0.0, np.nan, 2.0], [1.0, 0.5, np.nan],
    )

    np.testing.assert_array_equal(course.time, [0.0, np.nan])
    np.testing.assert_array_equal(course.value, [1.0, 0.5])
    assert course.observation_uncertainty is None
    assert not metadata.sources
    assert metadata.to_legacy_dict() == {'applied_scalings': [], 'warnings': []}
    assert deepcopy(metadata).to_legacy_dict() == metadata.to_legacy_dict()
    assert asdict(metadata)['sources'] == {}


def test_without_uncertainty_existing_data_derived_transformations_remain_available():
    prepared = prepare_data(
        [1.0, 10.0, 100.0], [10.0, 5.0, 2.5],
        selected_normalizations=['initial_value'], selected_scalings=['log_y'],
    )

    np.testing.assert_allclose(prepared.in_vitro.value, np.log10([1.0, 0.5, 0.25]))
    assert prepared.in_vitro.observation_uncertainty is None
    assert not prepared.has_observation_uncertainty
    assert not prepared.preprocessing.sources
    assert len(prepared.to_legacy_data()) == 2


@pytest.mark.parametrize('sigma', [[0.1], [0.1, np.nan], [0.1, 0.0]])
def test_preprocessing_does_not_fill_or_drop_invalid_retained_uncertainty(sigma):
    with pytest.raises(ValueError):
        prepare_time_course([0.0, 1.0], [1.0, 0.5], observation_uncertainty=_sem(sigma))


@pytest.mark.parametrize('normalization', ['initial_value', 'min_max'])
def test_mixed_pair_cannot_normalize_supplied_errors_on_only_one_side(normalization):
    with pytest.raises(ValueError):
        prepare_data(
            [0.0, 1.0], [1.0, 0.5], [0.0, 2.0], [1.0, 0.5],
            selected_normalizations=[normalization], in_vivo_uncertainty=_sem(0.05),
        )


@pytest.mark.parametrize('method,options', [
    ('predict_time_mapping', {'model_name': 'linear_zero_intercept', 'parameters': {'a': 1.0}}),
    ('predict_value_ratio', {'model_name': 'exponential', 'in_vitro_parameters': {}, 'in_vivo_parameters': {}}),
    ('predict_value_ratio_with_uncertainty', {'model_name': 'exponential', 'in_vitro_parameters': {},
                                            'in_vivo_parameters': {}, 'in_vitro_covariance': None,
                                            'in_vivo_covariance': None}),
    ('predict_tau_ratio', {'model_name': 'exponential', 'tau_values': None}),
    ('predict_inverse_response', {'model_name': 'exponential', 'parameters': {}, 'covariance': None,
                                 'residuals': None, 'calibration_time_range': (0.0, 1.0)}),
    ('predict_direct_value_scaling', {'time_grid': [], 'in_vitro_response_on_time_grid': [],
                                    'in_vivo_response_on_time_grid': []}),
    ('predict_direct_time_scaling', {'response_grid': [], 'in_vitro_time_at_response': [],
                                   'in_vivo_time_at_response': []}),
])
def test_prediction_entry_points_reject_attached_uncertainty_until_propagation_is_implemented(method, options):
    from ivivc_core import predictions
    course, _ = prepare_time_course([0.0, 1.0], [1.0, 0.5], observation_uncertainty=_sem(0.05))
    with pytest.raises(ValueError, match='prediction-response uncertainty'):
        getattr(predictions, method)(course, **options)


def test_high_level_apis_reject_attached_uncertainty_before_dispatch():
    from ivivc_core import fit_ivivc, generate_trajectory_ensembles, predict_in_vivo

    prepared = prepare_data(
        [0.0, 1.0], [1.0, 0.5], [0.0, 2.0], [1.0, 0.5],
        in_vitro_uncertainty=_sem(0.05),
    )
    with pytest.raises(ValueError, match='supplied source-profile uncertainty'):
        fit_ivivc(prepared, selections=['approach2:exponential'], selected_metrics=['rmse'])
    prediction, _ = prepare_time_course(
        [0.0, 1.0], [1.0, 0.5], observation_uncertainty=_sem(0.05),
    )
    for operation in (predict_in_vivo, generate_trajectory_ensembles):
        with pytest.raises(ValueError, match='prediction-response uncertainty'):
            operation(None, prediction)


@pytest.mark.parametrize('method,options', [
    ('sample_time_mapping_trajectories', {'model_name': 'linear_zero_intercept',
                                        'parameters': {'a': 1.0}, 'covariance': None}),
    ('sample_value_ratio_trajectories', {'model_name': 'exponential',
                                       'in_vitro_parameters': {}, 'in_vivo_parameters': {},
                                       'in_vitro_covariance': None, 'in_vivo_covariance': None}),
    ('sample_tau_ratio_trajectories', {'model_name': 'exponential',
                                     'in_vitro_parameters': {}, 'in_vivo_parameters': {},
                                     'in_vitro_covariance': None, 'in_vivo_covariance': None}),
    ('sample_inverse_response_trajectories', {'model_name': 'exponential', 'parameters': {},
                                            'covariance': None, 'calibration_time_range': (0.0, 1.0)}),
])
def test_trajectory_samplers_reject_attached_uncertainty_before_sampling(method, options):
    from ivivc_core import uncertainty

    course, _ = prepare_time_course([0.0, 1.0], [1.0, 0.5], observation_uncertainty=_sem(0.05))
    rng = np.random.default_rng(19)
    state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match='prediction-response uncertainty'):
        getattr(uncertainty, method)(course, random_state=rng, **options)
    assert rng.bit_generator.state == state
