from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from ivivc_core import ProfileFit, TimeCourse, get_model_spec
from study_design_core import (
    DesignConstraints,
    EmpiricalTimeScaleDistribution,
    ObservationModel,
    fit_reference_model_ensemble,
    optimize_local_i_optimal_designs,
    reference_crossing_time_summary,
    reference_from_time_scaling,
    simulate_design,
)


def _profile(name='exponential', *, parameters=None, covariance=None, time=None):
    parameters = {'a': 1., 'b': -.2} if parameters is None else parameters
    time = np.linspace(0., 20., 21) if time is None else np.asarray(time, dtype=float)
    spec = get_model_spec(name)
    vector = np.array([parameters[key] for key in spec.parameter_names])
    values = np.asarray(spec.model_function(time, *vector))
    covariance = np.zeros((len(vector), len(vector))) if covariance is None else covariance
    return ProfileFit(
        model_name=name, time=time, response=values,
        parameter_names=spec.parameter_names, parameters=parameters,
        covariance=covariance, initial_parameters=vector,
        fitted_values=values, residuals=np.zeros_like(values),
        residual_dof=len(time) - len(vector), fit_metrics={}, cv_scores={}, cv_means={},
    )


def test_empirical_ecdf_keeps_records_and_aggregates_duplicate_positive_support():
    ratios = np.array([4., 2., 4., 16.])
    weights = np.array([1., 2., 3., 0.])
    provenance = {'material_group': ['polymer A']}
    distribution = EmpiricalTimeScaleDistribution(
        ratios, weights=weights, source_ids=['A', 'B', 'C', 'excluded'], metadata=provenance,
    )
    ratios[0] = 100.
    weights[:] = 1.
    provenance['material_group'].append('another group')
    np.testing.assert_array_equal(distribution.ratios, [4., 2., 4., 16.])
    np.testing.assert_allclose(distribution.weights, [1 / 6, 2 / 6, 3 / 6, 0.])
    assert tuple(distribution.source_ids) == ('A', 'B', 'C', 'excluded')
    assert distribution.metadata['material_group'] == ['polymer A']
    with pytest.raises(ValueError):
        distribution.ratios[0] = 1.
    with pytest.raises(ValueError):
        distribution.weights[0] = 1.
    support, cumulative = distribution.ecdf()
    np.testing.assert_array_equal(support, [2., 4.])
    np.testing.assert_allclose(cumulative, [1 / 3, 1.])
    np.testing.assert_allclose(
        distribution.cdf([-np.inf, 1., 2., 3., 4., np.inf]),
        [0., 0., 1 / 3, 1 / 3, 1., 1.],
    )
    np.testing.assert_array_equal(distribution.quantile([0., .1, 1 / 3, .34, 1.]), [2., 2., 2., 4., 4.])
    assert distribution.quantile(.5) == 4.
    assert distribution.cdf(3.) == pytest.approx(1 / 3)


def test_equal_weights_and_large_weights_have_the_same_empirical_distribution():
    ordinary = EmpiricalTimeScaleDistribution([2., 4.])
    large = EmpiricalTimeScaleDistribution([2., 4.], weights=[1e308, 1e308])
    np.testing.assert_allclose(ordinary.weights, [.5, .5])
    np.testing.assert_array_equal(ordinary.weights, large.weights)
    with pytest.raises(ValueError):
        EmpiricalTimeScaleDistribution([2., 4.], weights=[1e308, 1e-308])


def test_ecdf_remains_monotone_when_cumulative_roundoff_exceeds_one():
    distribution = EmpiricalTimeScaleDistribution(
        [1., 2., 3.], weights=[0.9990480294258779, 5.249824014444734e-05, 8.837042578890616e-17],
    )
    _, cumulative = distribution.ecdf()
    assert np.all((cumulative >= 0) & (cumulative <= 1))
    assert np.all(np.diff(cumulative) >= 0)
    assert distribution.quantile(1.) == 3.
    np.testing.assert_array_equal(distribution.cdf([2., 3., np.inf]), [1., 1., 1.])


@pytest.mark.parametrize('ratios', [[], 2., [[1., 2.]], [0., 2.], [-1., 2.], [np.nan], [np.inf]])
def test_invalid_ratio_records_are_rejected(ratios):
    with pytest.raises(ValueError):
        EmpiricalTimeScaleDistribution(ratios)


@pytest.mark.parametrize('weights', [[1.], [[1., 2.]], [0., 0.], [-1., 2.], [np.nan, 1.], [np.inf, 1.]])
def test_invalid_empirical_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        EmpiricalTimeScaleDistribution([1., 2.], weights=weights)


@pytest.mark.parametrize('source_ids', ['AB', ['A'], ['A', ''], ['A', 2]])
def test_source_identifiers_match_ratio_records(source_ids):
    with pytest.raises((TypeError, ValueError)):
        EmpiricalTimeScaleDistribution([1., 2.], source_ids=source_ids)


@pytest.mark.parametrize('probability', [-.1, 1.1, np.nan])
def test_quantile_probabilities_are_validated(probability):
    with pytest.raises(ValueError):
        EmpiricalTimeScaleDistribution([1., 2.]).quantile(probability)


def test_known_exponential_scaling_keeps_whole_curves_and_weighted_central_reference():
    grid = np.linspace(0., 20., 401)
    distribution = EmpiricalTimeScaleDistribution([1., 2., 4.], weights=[.2, .3, .5])
    reference = reference_from_time_scaling(_profile(), distribution, grid, required_times=[0., 20.])
    expected = np.exp(-.2 * grid[None, :] / np.array([1., 2., 4.])[:, None])
    np.testing.assert_allclose(reference.value_draws, expected)
    np.testing.assert_allclose(reference.central_curve, np.average(expected, axis=0, weights=distribution.weights))
    np.testing.assert_allclose(reference.draw_weights, distribution.weights)
    assert reference.n_draws == reference.n_valid_draws == 3
    assert reference.supported_time_range == (0., 20.)
    assert reference.metadata['reference_source'] == 'empirical_time_scaling'
    assert reference.metadata['central_curve_source'] == 'weighted_draw_mean'
    summary = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.5)
    assert summary.deadline == pytest.approx(2 * np.log(2) / .2, rel=1e-4)
    np.testing.assert_allclose(summary.crossing_times, np.array([1., 2., 4.]) * np.log(2) / .2, rtol=1e-4)


def test_zero_weight_extreme_is_traceable_but_does_not_restrict_the_horizon():
    distribution = EmpiricalTimeScaleDistribution(
        [1e-6, 2.], weights=[0., 1.], source_ids=['excluded', 'included'],
        metadata={'time_unit': 'days', 'ratio_definition': 'tau_IVV/tau_IV'},
    )
    reference = reference_from_time_scaling(_profile(time=[0., 10.]), distribution, [0., 20.])
    assert reference.n_draws == 1
    np.testing.assert_array_equal(reference.metadata['draw_ratio_indices'], [1])
    np.testing.assert_array_equal(reference.metadata['time_scale_ratios'], [1e-6, 2.])
    np.testing.assert_array_equal(reference.metadata['time_scale_weights'], [0., 1.])
    assert tuple(reference.metadata['time_scale_source_ids']) == ('excluded', 'included')
    assert reference.metadata['time_scale_metadata']['time_unit'] == 'days'
    np.testing.assert_allclose(reference.value_draws[0], [1., np.exp(-2.)])


def test_supplied_ratios_do_not_require_the_target_models_own_tau():
    fit = _profile('mass_loss_one_phase_normalized', parameters={'t50': 8., 's': 1.5})
    assert not get_model_spec(fit.model_name).supports_tau
    grid = np.linspace(0., 40., 81)
    with patch('utilities.time_constants.evaluate_tau', side_effect=AssertionError('Unexpected target tau calculation')):
        reference = reference_from_time_scaling(fit, [2.], grid)
    expected = get_model_spec(fit.model_name).model_function(grid / 2., **fit.parameters)
    np.testing.assert_allclose(reference.value_draws[0], expected)


def test_covariance_draws_cross_every_ratio_preserving_both_marginal_weights():
    covariance = np.array([[4e-4, -2e-5], [-2e-5, 4e-6]])
    fit = _profile(covariance=covariance)
    # Parameter mapping order is independent of the registered vector order.
    fit = replace(fit, parameters={'b': -.2, 'a': 1.})
    distribution = EmpiricalTimeScaleDistribution([2., 3.], weights=[.2, .8])
    grid = np.linspace(0., 20., 21)
    kwargs = dict(n_parameter_draws=40, random_state=123)
    first = reference_from_time_scaling(fit, distribution, grid, **kwargs)
    second = reference_from_time_scaling(fit, distribution, grid, **kwargs)
    np.testing.assert_array_equal(first.value_draws, second.value_draws)
    ratio_indices = np.asarray(first.metadata['draw_ratio_indices'])
    parameter_indices = np.asarray(first.metadata['draw_parameter_indices'])
    parameters = np.asarray(first.metadata['parameter_samples'])
    assert parameters.shape == (40, 2)
    assert first.n_draws == first.n_valid_draws == 80
    assert not np.allclose(parameters[0], parameters[1])
    np.testing.assert_allclose(np.bincount(ratio_indices, weights=first.draw_weights), distribution.weights)
    np.testing.assert_allclose(
        np.bincount(parameter_indices, weights=first.draw_weights), np.full(40, 1 / 40),
    )
    expected = parameters[parameter_indices, 0, None] * np.exp(
        parameters[parameter_indices, 1, None] * grid[None, :] / distribution.ratios[ratio_indices, None]
    )
    np.testing.assert_allclose(first.value_draws, expected)
    np.testing.assert_allclose(first.central_curve, np.average(expected, axis=0, weights=first.draw_weights))


def test_fixed_fitted_curve_does_not_need_a_usable_covariance():
    fit = _profile(covariance=np.full((2, 2), np.nan))
    reference = reference_from_time_scaling(fit, [2.], [0., 10.])
    np.testing.assert_allclose(reference.value_draws[0], [1., np.exp(-1.)])
    with pytest.raises(ValueError):
        reference_from_time_scaling(fit, [2.], [0., 10.], n_parameter_draws=3)


@pytest.mark.parametrize('count', [-1, 1.5, True])
def test_parameter_draw_count_requires_a_nonnegative_integer(count):
    with pytest.raises(ValueError):
        reference_from_time_scaling(_profile(), [2.], [0., 10.], n_parameter_draws=count)


def test_time_course_scaling_uses_the_selected_interpolation_and_duplicate_policy():
    source = TimeCourse(np.array([2., 0., 1., 1.]), np.array([0., 1., .4, .6]))
    reference = reference_from_time_scaling(source, [2.], np.arange(5.), interpolation='linear')
    np.testing.assert_allclose(reference.value_draws[0], [1., .75, .5, .25, 0.])
    with pytest.raises(ValueError, match='duplicate'):
        reference_from_time_scaling(source, [2.], np.arange(5.), duplicate_time_policy='error')
    with pytest.raises((TypeError, ValueError)):
        reference_from_time_scaling(source, [2.], np.arange(5.), n_parameter_draws=2)


def test_fitted_extrapolation_is_explicit_and_recorded():
    fit = _profile(time=[0., 10.])
    grid = np.arange(31.)
    with pytest.raises(ValueError):
        reference_from_time_scaling(fit, [2.], grid)
    reference = reference_from_time_scaling(fit, [2.], grid, allow_extrapolation=True)
    np.testing.assert_allclose(reference.value_draws[0], np.exp(-.1 * grid))
    extrapolated = np.asarray(reference.metadata['extrapolated_points_by_ratio'])
    np.testing.assert_array_equal(extrapolated[0], grid > 20.)
    np.testing.assert_allclose(reference.metadata['required_in_vitro_time_ranges'], [[0., 15.]])


@pytest.mark.parametrize('allow', [False, True])
def test_external_time_courses_are_not_extrapolated(allow):
    source = TimeCourse(np.array([0., 10.]), np.array([1., 0.]))
    with pytest.raises(ValueError):
        reference_from_time_scaling(source, [2.], [0., 30.], allow_extrapolation=allow)


def test_nonfinite_scaled_times_and_curves_do_not_remove_ratio_probability():
    with pytest.raises(ValueError):
        reference_from_time_scaling(_profile(), [1e-320], [0., 1.], allow_extrapolation=True)
    growing = _profile(parameters={'a': 1., 'b': 1.}, time=[0., 1.])
    # Ratio 2 remains finite, but failure of the positive-weight ratio 1 must
    # invalidate construction instead of renormalizing onto the surviving row.
    with pytest.raises(ValueError):
        reference_from_time_scaling(growing, [1., 2.], [0., 1000.], allow_extrapolation=True)


@pytest.mark.parametrize('grid', [[], [0.], [1., 0.], [0., 0.], [0., np.nan], [[0., 1.]]])
def test_reference_grid_must_be_finite_and_strictly_increasing(grid):
    with pytest.raises(ValueError):
        reference_from_time_scaling(_profile(), [2.], grid)


@pytest.mark.parametrize('required', [[-1.], [11.], [np.nan], [[0.]]])
def test_required_times_cannot_extend_the_requested_reference_grid(required):
    with pytest.raises(ValueError):
        reference_from_time_scaling(_profile(), [2.], [0., 10.], required_times=required)


def test_slow_non_crossing_curves_keep_their_empirical_weight():
    distribution = EmpiricalTimeScaleDistribution([1., 100.], weights=[.25, .75])
    reference = reference_from_time_scaling(_profile(), distribution, np.linspace(0., 10., 201))
    summary = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.5)
    assert summary.status == 'beyond_horizon'
    assert summary.reached_weight == pytest.approx(.25)
    assert summary.noncrossing_weight == pytest.approx(.75)
    np.testing.assert_allclose(summary.draw_weights, [.25, .75])
    early = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.1)
    assert early.deadline == pytest.approx(np.log(2) / .2, rel=1e-4)


def test_changing_the_common_time_unit_preserves_scaled_trajectories():
    grid = np.linspace(0., 20., 101)
    distribution = EmpiricalTimeScaleDistribution([1., 2.], weights=[.4, .6])
    days = reference_from_time_scaling(_profile(), distribution, grid)
    hours_fit = _profile(parameters={'a': 1., 'b': -.2 / 24}, time=np.linspace(0., 480., 21))
    hours = reference_from_time_scaling(hours_fit, distribution, 24 * grid)
    np.testing.assert_allclose(hours.value_draws, days.value_draws, rtol=1e-14)
    days_summary = reference_crossing_time_summary(days, degraded_fraction=.5, quantile=.5)
    hours_summary = reference_crossing_time_summary(hours, degraded_fraction=.5, quantile=.5)
    assert hours_summary.deadline == pytest.approx(24 * days_summary.deadline)


def test_empirical_reference_uses_the_existing_design_and_simulation_contract():
    reference = reference_from_time_scaling(_profile(), [1., 2.], np.linspace(0., 20., 81))
    model = fit_reference_model_ensemble(reference, model_name='exponential')
    observation = ObservationModel(kind='constant_sd', value=.01, replicates_per_time=2)
    constraints = DesignConstraints(
        candidate_times=np.arange(21.), fixed_times=np.array([0.]),
        min_number_of_times=3, max_number_of_times=3, replicates_per_time=2,
    )
    optimization = optimize_local_i_optimal_designs(model, observation, constraints, refine=False)
    design = optimization.nested_steps[-1]
    assert np.isfinite(design.score.criterion)
    simulation = simulate_design(
        reference, design.times, observation, model_name='exponential',
        constraints=constraints, reference_model=model, n_simulations=8, random_state=42,
    )
    assert simulation.fit_success.all()
    assert np.isfinite(simulation.integrated_rmse).all()
    np.testing.assert_allclose(
        simulation.true_curves, reference.value_draws[simulation.trajectory_draw_indices],
    )
