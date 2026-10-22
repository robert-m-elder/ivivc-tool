from __future__ import annotations

import numpy as np
import pytest

from examples.study_design_poc_models import (
    generate_noisy_monotonic_responses,
    get_synthetic_model_preset,
    supported_synthetic_models,
)
from examples.study_design_proof_of_concept import (
    ProofOfConceptSettings,
    build_ivivc_prediction_scenario,
    build_measured_in_vivo_scenario,
)


def _small_settings(**overrides) -> ProofOfConceptSettings:
    values = {
        'n_reference_draws': 12,
        'n_simulations': 8,
        'random_seed': 9876,
        'refine': False,
        'write_html': False,
        'max_design_sizes': 1,
    }
    values.update(overrides)
    return ProofOfConceptSettings(**values)


def test_supported_synthetic_presets_are_finite_decreasing_and_registered():
    time = np.linspace(0.0, 48.0, 193)
    for model_name in supported_synthetic_models():
        preset = get_synthetic_model_preset(
            model_name,
            time_min=0.0,
            time_max=48.0,
        )
        values = preset.evaluate(time)

        assert preset.model_name == model_name
        assert tuple(preset.parameter_mapping) == preset.parameter_names
        assert np.all(np.isfinite(values))
        assert values[0] > 0.95
        assert values[-1] < 0.05
        assert np.all(np.diff(values) <= 1e-12)


def test_unsupported_registered_model_does_not_silently_use_one_phase():
    with pytest.raises(ValueError, match='No proof-of-concept synthetic preset'):
        ProofOfConceptSettings(
            n_reference_draws=8,
            n_simulations=8,
            analysis_model='mass_loss_two_phase',
        )


def test_noisy_response_generation_is_reproducible_monotonic_and_nonnegative():
    true_response = np.array([1.0, 0.85, 0.60, 0.30, 0.08])
    first = generate_noisy_monotonic_responses(
        true_response,
        absolute_sd=0.05,
        random_state=123,
        minimum_response=0.0,
    )
    second = generate_noisy_monotonic_responses(
        true_response,
        absolute_sd=0.05,
        random_state=123,
        minimum_response=0.0,
    )

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.all(np.diff(first[0]) < 0.0)
    assert np.all(first[0] >= 0.0)
    assert np.any(np.abs(first[1]) > 0.0)
    assert first[2] >= 1


def test_selected_model_controls_synthetic_measured_data_generation():
    exponential = build_measured_in_vivo_scenario(
        _small_settings(analysis_model='exponential')
    )
    one_phase = build_measured_in_vivo_scenario(
        _small_settings(analysis_model='mass_loss_one_phase')
    )

    assert exponential.analysis_model == 'exponential'
    assert exponential.metadata['generating_model'] == 'exponential'
    assert one_phase.metadata['generating_model'] == 'mass_loss_one_phase'
    assert set(exponential.metadata['generating_parameters']) == {'a', 'b'}
    assert set(one_phase.metadata['generating_parameters']) == {'y0', 'A', 't50', 's'}

    exp_means = exponential.observed_data.groupby('time')['response'].mean().to_numpy()
    one_phase_means = one_phase.observed_data.groupby('time')['response'].mean().to_numpy()
    assert not np.allclose(exp_means, one_phase_means)


def test_ivivc_response_noise_uses_fixed_times_preprocessing_and_broadens_uncertainty():
    low_noise = build_ivivc_prediction_scenario(
        _small_settings(ivivc_calibration_response_sd=0.005)
    )
    high_noise = build_ivivc_prediction_scenario(
        _small_settings(ivivc_calibration_response_sd=0.08)
    )

    assert low_noise.metadata['ivivc_calibration_response_sd'] == 0.005
    assert high_noise.metadata['ivivc_calibration_response_sd'] == 0.08
    assert high_noise.metadata['ivivc_preprocessing_interpolation'] == 'default'
    assert high_noise.preprocessed_data is not None
    assert high_noise.preprocessed_data['used_for_approach1_fit'].any()

    low_calibration = low_noise.observed_data.loc[
        low_noise.observed_data['role'] == 'IVIVC calibration data'
    ]
    high_calibration = high_noise.observed_data.loc[
        high_noise.observed_data['role'] == 'IVIVC calibration data'
    ]
    np.testing.assert_allclose(
        low_calibration['in_vitro_time'],
        high_calibration['in_vitro_time'],
    )
    np.testing.assert_allclose(
        low_calibration['in_vivo_time'],
        high_calibration['in_vivo_time'],
    )
    assert np.any(np.abs(high_calibration['in_vitro_response_noise']) > 0.0)
    assert np.any(np.abs(high_calibration['in_vivo_response_noise']) > 0.0)
    assert np.all(np.diff(high_calibration['in_vitro_response']) < 0.0)
    assert np.all(np.diff(high_calibration['in_vivo_response']) < 0.0)

    low_spread = np.nanmedian(np.nanstd(low_noise.reference.value_draws, axis=0))
    high_spread = np.nanmedian(np.nanstd(high_noise.reference.value_draws, axis=0))
    assert high_spread > low_spread


def test_zero_ivivc_response_noise_recovers_true_time_scale_closely():
    scenario = build_ivivc_prediction_scenario(
        _small_settings(ivivc_calibration_response_sd=0.0)
    )
    fitted = scenario.metadata['ivivc_fitted_time_mapping_parameters']['a']
    assert fitted == pytest.approx(scenario.metadata['ivivc_time_scale'], rel=0.02)


def test_time_scaled_preset_evaluation_matches_parameter_mapping():
    time = np.linspace(0.0, 80.0, 161)
    for model_name in supported_synthetic_models():
        preset = get_synthetic_model_preset(
            model_name,
            time_min=0.0,
            time_max=40.0,
        )
        scale = 1.75
        direct = preset.evaluate_time_scaled(time, scale)
        expected = preset.evaluate(time / scale)
        np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)
