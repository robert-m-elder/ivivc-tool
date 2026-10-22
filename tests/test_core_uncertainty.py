from __future__ import annotations

import numpy as np

from ivivc_core import (
    CurveDraws,
    ParameterSamples,
    PredictedTrajectory,
    ProfileFit,
    TimeCourse,
    TrajectoryEnsemble,
    deterministic_trajectory_ensemble,
    evaluate_curve_draws,
    sample_inverse_response_trajectories,
    sample_parameter_sets,
    sample_profile_trajectory,
    sample_tau_ratio_trajectories,
    sample_time_mapping_trajectories,
    sample_value_ratio_trajectories,
    summarize_draws,
)
from models.exponential import model_function as exponential_model
from utilities.evaluation import (
    generate_confidence_bands,
    generate_prediction_bands,
)


def _profile_fit() -> ProfileFit:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    response = np.exp(-0.2 * time)
    return ProfileFit(
        model_name='exponential',
        time=time,
        response=response,
        parameter_names=('a', 'b'),
        parameters={'a': 1.0, 'b': -0.2},
        covariance=np.diag([1e-4, 1e-4]),
        initial_parameters=np.array([1.0, -0.2]),
        fitted_values=response,
        residuals=np.zeros_like(response),
        residual_dof=2,
        fit_metrics={'R2': 1.0},
        cv_scores={'R2': np.array([1.0])},
        cv_means={'R2': 1.0},
    )


def test_seeded_parameter_and_curve_draws_preserve_complete_joint_rows():
    first = sample_parameter_sets(
        {'a': 1.0, 'b': -0.2},
        np.diag([1e-4, 1e-4]),
        n_samples=40,
        random_state=123,
    )
    second = sample_parameter_sets(
        {'a': 1.0, 'b': -0.2},
        np.diag([1e-4, 1e-4]),
        n_samples=40,
        random_state=123,
    )

    assert isinstance(first, ParameterSamples)
    assert first.parameter_names == ('a', 'b')
    assert first.values.shape == (40, 2)
    np.testing.assert_allclose(first.values, second.values)

    x = np.array([0.0, 1.0, 2.0])
    curves = evaluate_curve_draws(exponential_model, x, first)
    assert isinstance(curves, CurveDraws)
    assert curves.values.shape == (40, 3)
    assert curves.n_valid_draws == 40
    np.testing.assert_allclose(
        curves.values[0],
        exponential_model(x, *first.values[0]),
    )

    summary = summarize_draws(curves, confidence_level=0.90)
    np.testing.assert_array_equal(summary.valid_counts, 40)
    assert np.all(summary.lower <= summary.mean)
    assert np.all(summary.mean <= summary.upper)


def test_profile_trajectory_ensemble_retains_joint_parameter_draws():
    fit = _profile_fit()
    grid = np.linspace(0.0, 5.0, 11)
    ensemble = sample_profile_trajectory(
        fit,
        grid,
        n_samples=30,
        random_state=321,
    )

    assert isinstance(ensemble, TrajectoryEnsemble)
    assert ensemble.time_draws.shape == (30, 11)
    assert ensemble.value_draws.shape == (30, 11)
    assert ensemble.n_valid_draws == 30
    np.testing.assert_allclose(
        ensemble.time_draws,
        np.broadcast_to(grid, ensemble.time_draws.shape),
    )
    np.testing.assert_allclose(
        ensemble.central.in_vivo_value,
        exponential_model(grid, 1.0, -0.2),
    )
    assert ensemble.metadata['parameter_samples'].shape == (30, 2)
    assert not np.allclose(ensemble.value_draws[0], ensemble.value_draws[1])


def test_prediction_ensemble_generators_share_one_common_contract():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = TimeCourse(time, np.exp(-0.2 * time))
    covariance = np.diag([1e-5, 1e-5])

    time_mapping = sample_time_mapping_trajectories(
        prediction,
        model_name='linear',
        parameters={'a': 1.0, 'b': 2.0},
        covariance=covariance,
        n_samples=20,
        random_state=11,
    )
    value_ratio = sample_value_ratio_trajectories(
        prediction,
        model_name='exponential',
        in_vitro_parameters={'a': 1.0, 'b': -0.2},
        in_vitro_covariance=covariance,
        in_vivo_parameters={'a': 1.0, 'b': -0.1},
        in_vivo_covariance=covariance,
        n_samples=20,
        random_state=11,
    )
    tau_ratio = sample_tau_ratio_trajectories(
        prediction,
        model_name='exponential',
        in_vitro_parameters={'a': 1.0, 'b': -0.2},
        in_vitro_covariance=covariance,
        in_vivo_parameters={'a': 1.0, 'b': -0.1},
        in_vivo_covariance=covariance,
        n_samples=20,
        random_state=11,
    )
    inverse = sample_inverse_response_trajectories(
        prediction,
        model_name='exponential',
        parameters={'a': 1.0, 'b': -0.1},
        covariance=covariance,
        calibration_time_range=(0.0, 10.0),
        observed_responses=np.exp(-0.1 * np.linspace(0.0, 10.0, 11)),
        n_samples=20,
        random_state=11,
    )

    for ensemble in (time_mapping, value_ratio, tau_ratio, inverse):
        assert isinstance(ensemble, TrajectoryEnsemble)
        assert ensemble.time_draws.shape == (20, len(time))
        assert ensemble.value_draws.shape == (20, len(time))
        assert ensemble.valid_draws.shape == (20,)
        assert ensemble.n_valid_draws > 0

    np.testing.assert_allclose(time_mapping.central.in_vivo_time, 1.0 + 2.0 * time)
    np.testing.assert_allclose(
        value_ratio.time_draws,
        np.broadcast_to(time, value_ratio.time_draws.shape),
    )
    np.testing.assert_allclose(tau_ratio.central.in_vivo_time, 2.0 * time)
    np.testing.assert_allclose(inverse.central.in_vivo_time, 2.0 * time, atol=1e-7)


def test_deterministic_trajectory_is_a_one_draw_ensemble():
    central = PredictedTrajectory(
        source_approach='external',
        source_method='central_curve',
        in_vitro_time=np.array([0.0, 1.0]),
        in_vitro_value=np.array([1.0, 0.5]),
        in_vivo_time=np.array([0.0, 2.0]),
        in_vivo_value=np.array([1.0, 0.5]),
    )
    ensemble = deterministic_trajectory_ensemble(central)
    assert ensemble.n_draws == 1
    assert ensemble.n_valid_draws == 1
    np.testing.assert_array_equal(ensemble.time_draws[0], central.in_vivo_time)
    np.testing.assert_array_equal(ensemble.value_draws[0], central.in_vivo_value)


def test_legacy_band_wrappers_keep_contract_and_allow_reproducible_sampling():
    x = np.array([0.0, 1.0, 2.0])
    params = {'a': 1.0, 'b': -0.2}
    covariance = np.diag([1e-4, 1e-4])

    first = generate_confidence_bands(
        exponential_model,
        x,
        params,
        covariance,
        n_samples=100,
        random_state=99,
    )
    second = generate_confidence_bands(
        exponential_model,
        x,
        params,
        covariance,
        n_samples=100,
        random_state=99,
    )
    assert set(first) == {'lower', 'upper', 'mean'}
    for key in first:
        np.testing.assert_allclose(first[key], second[key])

    residuals = np.array([0.02, -0.01, 0.01, -0.02, 0.0, 0.01])
    prediction = generate_prediction_bands(
        exponential_model,
        x,
        params,
        covariance,
        residuals=residuals,
        n_samples=100,
        random_state=99,
    )
    assert set(prediction) == {'lower', 'upper', 'mean', 'std'}
    assert all(np.asarray(prediction[key]).shape == x.shape for key in prediction)


def test_registered_model_bounds_reject_and_resample_invalid_parameter_draws():
    samples = sample_parameter_sets(
        {'y0': 1.0, 'A': 1.0, 't50': 20.0, 's': 0.1},
        np.diag([1e-4, 1e-4, 1e-2, 1.0]),
        n_samples=100,
        random_state=2026,
        model_name='mass_loss_one_phase',
    )

    assert samples.values.shape == (100, 4)
    assert np.all(samples.values[:, 0] >= 0.0)
    assert np.all(samples.values[:, 1] >= 0.0)
    assert np.all(samples.values[:, 3] > 0.0)
    assert samples.attempted_samples > samples.requested_samples
    assert samples.rejection_counts['outside_registered_bounds'] > 0
