"""Fixed normalization references rescale absolute SD/SEM without re-estimation."""

from __future__ import annotations

import numpy as np
import pytest

from ivivc_core import ObservationUncertainty, fit_profile, prepare_data, prepare_time_course


def _sem(values, counts=None):
    return ObservationUncertainty(
        values, response_kind='mean', replicate_counts=counts,
        source='reported SEM', response_units='mg',
    )


def test_initial_value_rescales_sem_and_retains_uncertain_baseline_and_raw_inputs():
    raw_values = [10.0, 8.0, 4.0]
    raw_sem = [1.0, 0.8, 0.4]
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], raw_values, selected_normalizations=['initial_value'],
        observation_uncertainty=_sem(raw_sem, [3, 5, 7]),
    )

    np.testing.assert_allclose(course.value, [1.0, 0.8, 0.4])
    uncertainty = course.observation_uncertainty
    np.testing.assert_allclose(uncertainty.sigma, [0.1, 0.08, 0.04])
    assert uncertainty.sigma[0] > 0.0
    np.testing.assert_array_equal(uncertainty.replicate_counts, [3, 5, 7])
    assert uncertainty.uncertainty_kind == 'sem'
    assert uncertainty.source == 'reported SEM'
    assert uncertainty.response_units == 'dimensionless'
    record = metadata.normalizations[0].in_vitro_parameters
    assert record['initial_value'] == 10.0
    assert record['applied'] is True
    assert record['constants_treated_as_fixed'] is True
    source = metadata.sources['profile']
    np.testing.assert_array_equal(source.value, raw_values)
    np.testing.assert_array_equal(source.observation_uncertainty.standard_uncertainty, raw_sem)
    assert source.observation_uncertainty.response_units == 'mg'


def test_negative_initial_reference_uses_its_absolute_value_for_sd():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [-10.0, -5.0, -2.0],
        selected_normalizations=['initial_value'],
        observation_uncertainty=ObservationUncertainty(1.0, response_kind='individual', response_units='mg'),
    )

    np.testing.assert_allclose(course.value, [1.0, 0.5, 0.2])
    np.testing.assert_allclose(course.observation_uncertainty.sigma, [0.1] * 3)
    assert course.observation_uncertainty.uncertainty_kind == 'sd'
    assert course.observation_uncertainty.input_form == 'constant'
    assert metadata.normalizations[0].in_vitro_parameters['initial_value'] == -10.0


def test_zero_initial_reference_preserves_existing_noop_and_original_units():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [0.0, 2.0, 4.0], selected_normalizations=['initial_value'],
        observation_uncertainty=_sem(0.5),
    )

    np.testing.assert_array_equal(course.value, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(course.observation_uncertainty.sigma, [0.5] * 3)
    assert course.observation_uncertainty.response_units == 'mg'
    record = metadata.normalizations[0].in_vitro_parameters
    assert record['initial_value'] == 0.0
    assert record['applied'] is False
    assert record['constants_treated_as_fixed'] is True


def test_min_max_rescales_sem_without_assigning_zero_error_to_sample_extrema():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [12.0, 8.0, 4.0], selected_normalizations=['min_max'],
        observation_uncertainty=_sem([1.2, 0.8, 0.4]),
    )

    np.testing.assert_allclose(course.value, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(course.observation_uncertainty.sigma, [0.15, 0.1, 0.05])
    assert np.all(course.observation_uncertainty.sigma > 0.0)
    assert course.observation_uncertainty.response_units == 'dimensionless'
    record = metadata.normalizations[0].in_vitro_parameters
    assert record['minimum'] == 4.0
    assert record['maximum'] == 12.0
    assert record['range'] == 8.0
    assert record['constants_treated_as_fixed'] is True


@pytest.mark.parametrize('normalization', ['initial_value', 'min_max'])
def test_paired_profiles_use_their_own_fixed_constants(normalization):
    prepared = prepare_data(
        [0.0, 1.0, 2.0], [10.0, 8.0, 6.0],
        [0.0, 2.0, 4.0], [100.0, 60.0, 20.0],
        selected_normalizations=[normalization],
        in_vitro_uncertainty=_sem([1.0, 0.8, 0.6]),
        in_vivo_uncertainty=_sem([2.0, 3.0, 4.0]),
    )
    vitro_divisor = 10.0 if normalization == 'initial_value' else 4.0
    vivo_divisor = 100.0 if normalization == 'initial_value' else 80.0

    np.testing.assert_allclose(prepared.in_vitro.observation_uncertainty.sigma,
                               np.array([1.0, 0.8, 0.6]) / vitro_divisor)
    np.testing.assert_allclose(prepared.in_vivo.observation_uncertainty.sigma,
                               np.array([2.0, 3.0, 4.0]) / vivo_divisor)
    assert prepared.in_vitro.observation_uncertainty.response_units == 'dimensionless'
    assert prepared.in_vivo.observation_uncertainty.response_units == 'dimensionless'


def test_chained_fixed_normalizations_rescale_sequentially_and_reverse_once():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [20.0, 15.0, 10.0],
        selected_normalizations=['one_minus', 'initial_value', 'min_max', 'one_minus'],
        observation_uncertainty=_sem([2.0, 1.5, 1.0], [3, 4, 5]),
    )

    np.testing.assert_allclose(course.value, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(course.observation_uncertainty.sigma, [0.2, 0.15, 0.1])
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [3, 4, 5])
    assert metadata.selected_normalizations == ('initial_value', 'min_max', 'one_minus')
    assert metadata.normalizations[0].in_vitro_parameters['initial_value'] == 20.0
    assert metadata.normalizations[1].in_vitro_parameters['range'] == 0.5
    assert metadata.normalizations[0].in_vitro_parameters['constants_treated_as_fixed'] is True
    assert metadata.normalizations[1].in_vitro_parameters['constants_treated_as_fixed'] is True


def test_initial_reference_is_taken_after_missing_row_cleaning():
    course, metadata = prepare_time_course(
        [np.nan, 1.0, 2.0], [100.0, 10.0, 5.0],
        selected_normalizations=['initial_value'],
        observation_uncertainty=_sem([np.nan, 1.0, 0.5]),
    )

    np.testing.assert_allclose(course.value, [1.0, 0.5])
    np.testing.assert_allclose(course.observation_uncertainty.sigma, [0.1, 0.05])
    assert metadata.normalizations[0].in_vitro_parameters['initial_value'] == 10.0
    np.testing.assert_array_equal(metadata.sources['profile'].retained_rows, [1, 2])


@pytest.mark.parametrize('normalization,values', [
    ('initial_value', [np.inf, 2.0, 1.0]),
    ('initial_value', [-np.inf, 2.0, 1.0]),
    ('min_max', [5.0, 5.0, 5.0]),
    ('min_max', [np.inf, 2.0, 1.0]),
    ('min_max', [-np.inf, 2.0, 1.0]),
    ('min_max', [-1e308, 0.0, 1e308]),
])
def test_nonfinite_or_zero_range_normalization_is_rejected_with_supplied_uncertainty(normalization, values):
    with pytest.raises(ValueError):
        prepare_time_course(
            [0.0, 1.0, 2.0], values, selected_normalizations=[normalization],
            observation_uncertainty=_sem(0.1),
        )


def test_mixed_pair_checks_fixed_constants_even_on_the_side_without_supplied_uncertainty():
    with pytest.raises(ValueError):
        prepare_data(
            [0.0, 1.0, 2.0], [5.0, 5.0, 5.0],
            [0.0, 2.0, 4.0], [100.0, 60.0, 20.0],
            selected_normalizations=['min_max'], in_vivo_uncertainty=_sem(0.2),
        )


@pytest.mark.parametrize('normalization', ['initial_value', 'min_max'])
def test_normalization_rejects_an_empty_profile_after_cleaning(normalization):
    with pytest.raises(ValueError):
        prepare_time_course(
            [np.nan], [1.0], selected_normalizations=[normalization],
            observation_uncertainty=_sem([np.nan]),
        )


@pytest.mark.parametrize('reference,sigma', [(0.1, 1e308), (1e308, 1e-308)])
def test_unrepresentable_rescaled_uncertainties_are_not_replaced_by_zero_or_infinity(reference, sigma):
    with pytest.raises(ValueError):
        prepare_time_course(
            [0.0, 1.0], [reference, reference / 2],
            selected_normalizations=['initial_value'], observation_uncertainty=_sem(sigma),
        )


def test_normalized_linear_fit_matches_analytic_fixed_scale_parameters_and_covariance(fast_analysis_config):
    time = np.arange(5.0)
    value = np.array([10.0, 12.1, 13.9, 16.0, 18.2])
    sigma = np.array([0.5, 0.8, 0.7, 1.0, 1.2])
    course, _ = prepare_time_course(
        time, value, selected_normalizations=['initial_value'],
        observation_uncertainty=_sem(sigma),
    )
    fit = fit_profile(
        course.time, course.value, model_name='linear', selected_metrics=['rmse'],
        analysis_config=fast_analysis_config,
        observation_uncertainty=course.observation_uncertainty,
    )
    design = np.column_stack([np.ones(len(time)), time])
    precision = 1.0 / course.observation_uncertainty.sigma ** 2
    information = design.T @ (precision[:, None] * design)
    expected = np.linalg.solve(information, design.T @ (precision * course.value))

    np.testing.assert_allclose(fit.parameter_vector, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.covariance, np.linalg.inv(information), rtol=1e-5, atol=1e-10)
    assert fit.fit_diagnostics['covariance_scaling'] == 'absolute'


def test_no_uncertainty_retains_existing_normalization_records_and_degenerate_behavior():
    course, metadata = prepare_time_course(
        [0.0, 1.0, 2.0], [10.0, 8.0, 4.0], selected_normalizations=['initial_value'],
    )
    np.testing.assert_allclose(course.value, [1.0, 0.8, 0.4])
    assert metadata.normalizations[0].in_vitro_parameters == {'initial_value': 10.0, 'applied': True}
    with np.errstate(invalid='ignore', divide='ignore'):
        degenerate, metadata = prepare_time_course(
            [0.0, 1.0, 2.0], [5.0, 5.0, 5.0], selected_normalizations=['min_max'],
        )
    assert np.all(np.isnan(degenerate.value))
    assert 'constants_treated_as_fixed' not in metadata.normalizations[0].in_vitro_parameters
