from __future__ import annotations

import numpy as np
import pytest

from ivivc_core import PredictedTrajectory, ProfileFit, TrajectoryEnsemble
from study_design_core import (
    DesignConstraints,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
    fit_model_once,
    observation_model_from_residuals,
    reference_from_profile_fit,
    reference_from_time_course,
    reference_from_trajectory_ensemble,
)


def _linear_profile_fit() -> ProfileFit:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    response = np.array([1.0, 0.82, 0.61, 0.39])
    fitted = np.array([1.01, 0.81, 0.61, 0.39])
    return ProfileFit(
        model_name='linear',
        time=time,
        response=response,
        parameter_names=('a', 'b'),
        parameters={'a': 1.01, 'b': -0.205},
        covariance=np.diag([1e-4, 1e-4]),
        initial_parameters=np.array([1.0, -0.2]),
        fitted_values=fitted,
        residuals=response - fitted,
        residual_dof=2,
        fit_metrics={'rmse': 0.01},
        cv_scores={'rmse': np.array([np.nan])},
        cv_means={'rmse': np.nan},
    )


def test_reference_conversion_uses_shared_domain_and_complete_draws():
    central = PredictedTrajectory(
        source_approach='approach1',
        source_method='time_mapping',
        model_name='linear',
        in_vitro_time=np.array([0.0, 1.0, 2.0]),
        in_vitro_value=np.array([1.0, 0.5, 0.0]),
        in_vivo_time=np.array([0.25, 1.25, 2.25]),
        in_vivo_value=np.array([1.0, 0.5, 0.0]),
    )
    ensemble = TrajectoryEnsemble(
        central=central,
        time_draws=np.array([
            [0.0, 1.0, 2.0],
            [0.5, 1.5, 2.5],
            [np.nan, np.nan, np.nan],
        ]),
        value_draws=np.array([
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [np.nan, np.nan, np.nan],
        ]),
        valid_draws=np.array([True, True, False]),
    )

    reference = reference_from_trajectory_ensemble(
        ensemble,
        time_grid=np.array([0.5, 1.0, 1.5, 2.0]),
        interpolation='linear',
    )

    assert isinstance(reference, ReferenceTrajectoryEnsemble)
    assert reference.supported_time_range == (0.5, 2.0)
    np.testing.assert_array_equal(reference.valid_draws, [True, True, False])
    np.testing.assert_allclose(reference.draw_weights, [0.5, 0.5, 0.0])
    assert reference.value_draws.shape == (3, 4)
    np.testing.assert_allclose(
        reference.evaluate([0.75, 1.75], draw_indices=[0, 1], method='linear'),
        np.array([[0.625, 0.125], [0.875, 0.375]]),
    )
    with pytest.raises(ValueError, match='supported reference domain'):
        reference.evaluate([0.25])


def test_deterministic_reference_sorts_and_averages_duplicate_times():
    reference = reference_from_time_course(
        time=np.array([2.0, 0.0, 1.0, 1.0]),
        value=np.array([0.0, 1.0, 0.4, 0.6]),
        interpolation='linear',
    )

    np.testing.assert_array_equal(reference.time_grid, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(reference.central_curve, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(reference.evaluate_central([0.5, 1.5]), [0.75, 0.25])
    assert reference.metadata['duplicate_time_count'] == 1


def test_profile_fit_can_be_transformed_to_common_reference_ensemble():
    fit = _linear_profile_fit()
    reference = reference_from_profile_fit(
        fit,
        np.linspace(0.0, 3.0, 7),
        n_samples=20,
        random_state=123,
    )

    assert reference.n_draws == 20
    assert reference.n_valid_draws == 20
    assert reference.value_draws.shape == (20, 7)
    np.testing.assert_allclose(reference.central_curve, 1.01 - 0.205 * reference.time_grid)


def test_observation_models_separate_future_noise_from_reference_uncertainty():
    constant_sd = ObservationModel(
        kind='constant_sd',
        value=0.2,
        replicates_per_time=3,
    )
    np.testing.assert_allclose(constant_sd.standard_deviation([1.0, 0.1]), [0.2, 0.2])
    np.testing.assert_allclose(constant_sd.variance_of_mean([1.0]), [0.04 / 3.0])
    first = constant_sd.simulate([1.0, 0.5], random_state=44)
    second = constant_sd.simulate([1.0, 0.5], random_state=44)
    assert first.shape == (3, 2)
    np.testing.assert_allclose(first, second)

    constant_cv = ObservationModel(
        kind='constant_cv',
        value=0.1,
        minimum_sd=0.02,
    )
    np.testing.assert_allclose(
        constant_cv.standard_deviation([2.0, 0.0]),
        [0.2, 0.02],
    )

    surrogate = observation_model_from_residuals(_linear_profile_fit())
    assert surrogate.kind == 'constant_sd'
    assert surrogate.source == 'reference_fit_residuals_surrogate'
    assert np.isclose(surrogate.value, 0.01)


def test_design_constraints_validate_fixed_times_spacing_and_cost():
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 11.0),
        fixed_times=np.array([0.0, 4.0]),
        minimum_spacing=2.0,
        min_number_of_times=3,
        max_number_of_times=5,
        replicates_per_time=4,
    )

    np.testing.assert_array_equal(
        constraints.available_candidates([8.0]),
        np.array([2.0, 6.0, 10.0]),
    )
    np.testing.assert_array_equal(
        constraints.validate_design([0.0, 4.0, 8.0]),
        np.array([0.0, 4.0, 8.0]),
    )
    assert constraints.total_experimental_units(5) == 20

    with pytest.raises(ValueError, match='omits'):
        constraints.validate_design([2.0, 4.0, 8.0])
    with pytest.raises(ValueError, match='minimum spacing'):
        constraints.validate_design([0.0, 4.0, 5.0])


def test_lightweight_fit_uses_supplied_start_without_cv_or_plotting():
    time = np.arange(6.0)
    response = 1.5 + 2.0 * time
    result = fit_model_once(
        time,
        response,
        model_name='linear',
        initial_parameters={'a': 1.0, 'b': 1.0},
        fallback_grid_search=False,
    )

    assert result.success
    assert not result.used_grid_search
    assert result.parameter_names == ('a', 'b')
    np.testing.assert_allclose(result.parameter_vector, [1.5, 2.0], atol=1e-8)
    np.testing.assert_allclose(result.fitted_values, response, atol=1e-8)
    assert result.residual_dof == 4
    assert result.residual_sd < 1e-8
    np.testing.assert_allclose(result.predict([6.0]), [13.5], atol=1e-8)


def test_lightweight_fit_returns_structured_failure_when_requested():
    result = fit_model_once(
        [0.0],
        [1.0],
        model_name='linear',
        initial_parameters=[1.0, 0.0],
        fallback_grid_search=False,
    )

    assert not result.success
    assert result.exception_type is not None
    assert np.isnan(result.parameter_vector).all()
    assert np.isnan(result.fitted_values).all()
