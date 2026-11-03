from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ivivc_core import ProfileFit
from study_design_core import (
    CrossingTimeTarget,
    DesignConstraints,
    EmpiricalTimeScaleDistribution,
    ExistingObservations,
    LatestBefore,
    ObservationModel,
    PlanningContext,
    SamplingWindow,
    compare_designs_by_simulation,
    create_simulation_plan,
    equally_spaced_design,
    fit_reference_model_ensemble,
    optimize_local_i_optimal_designs,
    optimize_next_time_before_deadline,
    optimize_timing_constrained_designs,
    reference_crossing_time_summary,
    reference_from_time_course,
    reference_from_time_scaling,
    resolve_timing_requirements,
    reweight_reference_ensemble,
    reweight_time_scaled_reference,
    time_scaling_weight_summary,
)


def _profile(*, covariance=None):
    time = np.linspace(0., 30., 31)
    values = np.exp(-.2 * time)
    return ProfileFit(
        model_name='exponential', time=time, response=values,
        parameter_names=('a', 'b'), parameters={'a': 1., 'b': -.2},
        covariance=np.zeros((2, 2)) if covariance is None else covariance,
        initial_parameters=np.array([1., -.2]), fitted_values=values,
        residuals=np.zeros_like(values), residual_dof=len(time) - 2,
        fit_metrics={}, cv_scores={}, cv_means={},
    )


def _prior(*, count=0, covariance=None):
    scales = EmpiricalTimeScaleDistribution(
        [1., 2., 2., 1e-6], weights=[.2, .3, .5, 0.],
        source_ids=['fast', 'medium-A', 'medium-B', 'excluded'],
    )
    return reference_from_time_scaling(
        _profile(covariance=covariance), scales, np.linspace(0., 30., 301),
        n_parameter_draws=count, random_state=123,
    )


def _completed(times=(0., 1.)):
    times = np.repeat(np.asarray(times, dtype=float), 2)
    perturbation = np.tile([.005, -.005], len(times) // 2)
    return ExistingObservations(times, np.exp(-.1 * times) + perturbation)


def test_prospective_summary_preserves_records_but_separates_unique_scale_weights():
    reference = _prior()
    summary = time_scaling_weight_summary(reference)
    assert summary['input_mode'] == 'prospective-time-scaling'
    assert summary['record_count'] == 4
    assert summary['positive_prior_record_count'] == 3
    assert summary['positive_current_record_count'] == 3
    assert tuple(summary['source_ids']) == ('fast', 'medium-A', 'medium-B', 'excluded')
    np.testing.assert_array_equal(summary['ratios'], [1., 2., 2., 1e-6])
    np.testing.assert_allclose(summary['prior_weights'], [.2, .3, .5, 0.])
    np.testing.assert_allclose(summary['current_weights'], summary['prior_weights'])
    np.testing.assert_array_equal(summary['unique_ratios'], [1e-6, 1., 2.])
    np.testing.assert_allclose(summary['prior_unique_weights'], [0., .2, .8])
    np.testing.assert_allclose(summary['current_unique_weights'], [0., .2, .8])
    assert summary['draw_effective_sample_size'] == pytest.approx(1 / (.2**2 + .3**2 + .5**2))
    assert summary['record_effective_sample_size'] == pytest.approx(summary['draw_effective_sample_size'])
    assert summary['unique_scale_effective_sample_size'] == pytest.approx(1 / (.2**2 + .8**2))
    # A returned report is not a mutable view into the original reference.
    summary['current_weights'][0] = 0.
    np.testing.assert_allclose(reference.draw_weights, [.2, .3, .5])
    assert 'adaptive_reweighting' not in reference.metadata


@pytest.mark.parametrize('count', [0, 5])
def test_adaptive_weights_and_agreement_match_the_replicate_likelihood(count):
    covariance = np.array([[1e-4, -2e-6], [-2e-6, 1e-6]])
    prior = _prior(count=count, covariance=covariance)
    original_weights = prior.draw_weights.copy()
    existing = _completed()
    observation = ObservationModel(kind='constant_sd', value=.03, replicates_per_time=2)
    predicted = prior.evaluate(existing.time)
    residual = existing.response[None, :] - predicted
    standardized = residual / .03
    log_weights = np.log(prior.draw_weights) - .5 * np.sum(standardized**2, axis=1)
    expected = np.exp(log_weights - np.max(log_weights))
    expected /= expected.sum()

    update = reweight_time_scaled_reference(prior, existing, observation)
    np.testing.assert_allclose(update.posterior_weights, expected)
    generic = reweight_reference_ensemble(prior, existing, observation)
    np.testing.assert_allclose(update.posterior_weights, generic.posterior_weights)
    np.testing.assert_array_equal(prior.draw_weights, original_weights)
    np.testing.assert_array_equal(update.reference.value_draws, prior.value_draws)
    np.testing.assert_allclose(
        update.reference.central_curve, np.average(prior.value_draws, axis=0, weights=expected),
    )
    summary = time_scaling_weight_summary(update.reference)
    assert summary['input_mode'] == 'adaptive-time-scaling'
    np.testing.assert_allclose(summary['prior_weights'], [.2, .3, .5, 0.])
    marginal = np.bincount(prior.metadata['draw_ratio_indices'], weights=expected, minlength=4)
    np.testing.assert_allclose(summary['current_weights'], marginal)
    assert marginal[3] == 0.
    np.testing.assert_allclose(
        update.metadata['time_scaling_weight_summary']['current_weights'], marginal,
    )
    np.testing.assert_allclose(
        update.reference.metadata['time_scaling_weight_summary']['current_weights'], marginal,
    )
    agreement = update.metadata['observation_agreement']
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    standardized_rms = np.sqrt(np.mean(standardized**2, axis=1))
    assert agreement['observation_count'] == existing.n_observations
    np.testing.assert_allclose(agreement['rmse_by_draw'], rmse)
    np.testing.assert_allclose(agreement['standardized_residual_rms_by_draw'], standardized_rms)
    assert agreement['best_standardized_residual_rms'] == pytest.approx(standardized_rms.min())
    assert agreement['prior_weighted_rmse'] == pytest.approx(np.sqrt(np.sum(original_weights * rmse**2)))
    assert agreement['posterior_weighted_rmse'] == pytest.approx(np.sqrt(np.sum(expected * rmse**2)))
    assert agreement['prior_weighted_standardized_residual_rms'] == pytest.approx(
        np.sqrt(np.sum(original_weights * standardized_rms**2)),
    )
    assert agreement['posterior_weighted_standardized_residual_rms'] == pytest.approx(
        np.sqrt(np.sum(expected * standardized_rms**2)),
    )
    prior_deadline = reference_crossing_time_summary(prior, degraded_fraction=.5, quantile=.1)
    updated_deadline = reference_crossing_time_summary(update.reference, degraded_fraction=.5, quantile=.1)
    assert updated_deadline.deadline > prior_deadline.deadline


def test_more_parameter_draws_do_not_increase_record_or_unique_scale_ess():
    observation = ObservationModel(kind='constant_sd', value=.03)
    ordinary = _prior()
    expanded = _prior(count=5)
    for first, second in (
        (ordinary, expanded),
        (reweight_time_scaled_reference(ordinary, _completed(), observation).reference,
         reweight_time_scaled_reference(expanded, _completed(), observation).reference),
    ):
        initial, repeated = time_scaling_weight_summary(first), time_scaling_weight_summary(second)
        np.testing.assert_allclose(initial['current_weights'], repeated['current_weights'])
        assert initial['record_effective_sample_size'] == pytest.approx(repeated['record_effective_sample_size'])
        assert initial['unique_scale_effective_sample_size'] == pytest.approx(repeated['unique_scale_effective_sample_size'])
        assert repeated['draw_effective_sample_size'] == pytest.approx(5 * initial['draw_effective_sample_size'])


@pytest.mark.parametrize('generic_first', [False, True])
def test_updates_require_the_original_prior_and_accept_cumulative_observations(generic_first):
    prior = _prior()
    observation = ObservationModel(kind='constant_sd', value=.04)
    first_data, cumulative = _completed((0., 1.)), _completed((0., 1., 2.))
    updater = reweight_reference_ensemble if generic_first else reweight_time_scaled_reference
    first_update = updater(prior, first_data, observation)
    with pytest.raises(ValueError):
        reweight_time_scaled_reference(first_update.reference, cumulative, observation)
    correct = reweight_time_scaled_reference(prior, cumulative, observation)
    repeated = reweight_time_scaled_reference(prior, cumulative, observation)
    expected = reweight_reference_ensemble(prior, cumulative, observation)
    np.testing.assert_allclose(correct.posterior_weights, expected.posterior_weights)
    np.testing.assert_array_equal(correct.posterior_weights, repeated.posterior_weights)
    assert correct.metadata['observation_agreement']['observation_count'] == cumulative.n_observations
    assert not prior.metadata.get('adaptive_reweighting', False)


def test_poor_agreement_is_visible_even_when_reweighting_favors_one_scale():
    prior = _prior()
    existing = ExistingObservations([1., 1.], [.1, .11])
    observation = ObservationModel(kind='constant_sd', value=.01)
    update = reweight_time_scaled_reference(prior, existing, observation)
    agreement = update.metadata['observation_agreement']
    assert agreement['best_standardized_residual_rms'] > 50.
    assert agreement['posterior_weighted_standardized_residual_rms'] > 50.
    np.testing.assert_array_equal(update.reference.value_draws, prior.value_draws)
    np.testing.assert_array_equal(update.reference.time_grid, prior.time_grid)
    np.testing.assert_array_equal(update.reference.metadata['time_scale_ratios'], prior.metadata['time_scale_ratios'])
    assert time_scaling_weight_summary(update.reference)['current_weights'][3] == 0.


@pytest.mark.parametrize('metadata_change', [
    {'reference_source': 'external'},
    {'draw_ratio_indices': [0, 1]},
    {'draw_ratio_indices': [0., 1.5, 2.]},
    {'draw_ratio_indices': [0, 1, 4]},
    {'draw_ratio_indices': [-1, 1, 2]},
    {'time_scale_weights': [.2, .3, .5]},
])
def test_invalid_empirical_provenance_is_not_silently_aggregated(metadata_change):
    original = _prior()
    invalid = replace(original, metadata={**original.metadata, **metadata_change})
    with pytest.raises((TypeError, ValueError)):
        time_scaling_weight_summary(invalid)
    with pytest.raises((TypeError, ValueError)):
        reweight_time_scaled_reference(invalid, _completed(), ObservationModel('constant_sd', .03))


def test_plain_external_reference_is_not_an_empirical_time_scaling_input():
    external = reference_from_time_course([0., 1., 2.], [1., .8, .6])
    with pytest.raises((TypeError, ValueError)):
        time_scaling_weight_summary(external)


@pytest.mark.parametrize('adaptive', [False, True])
def test_empirical_references_support_windows_multiple_anchors_and_shared_mc(adaptive):
    prior = _prior()
    observation = ObservationModel(kind='constant_sd', value=.02, replicates_per_time=2)
    existing = _completed() if adaptive else None
    reference = (reweight_time_scaled_reference(prior, existing, observation).reference
                 if adaptive else prior)
    completed = () if existing is None else tuple(existing.unique_times)
    fixed = np.array([0.]) if existing is None else existing.unique_times
    start = 0. if existing is None else existing.latest_time
    constraints = DesignConstraints(
        candidate_times=np.arange(start + 1., 31.), fixed_times=fixed,
        min_time=0., max_time=30., minimum_spacing=1.,
        min_number_of_times=4, max_number_of_times=5, replicates_per_time=2,
    )
    requirements = (
        SamplingWindow(0., CrossingTimeTarget(.5, .25), label='Early coverage'),
        LatestBefore(CrossingTimeTarget(.5, .25), label='T50'),
        LatestBefore(CrossingTimeTarget(.75, .25), label='T75'),
    )
    resolved = resolve_timing_requirements(
        requirements, reference, constraints, context=PlanningContext([0.], completed, start),
    )
    model = fit_reference_model_ensemble(reference, model_name='exponential')
    optimization = optimize_timing_constrained_designs(
        model, observation, constraints, resolved, number_of_times=[4, 5],
    )
    plan = create_simulation_plan(
        reference, observation, constraints.all_available_times, n_simulations=4, random_state=987,
    )
    for size in (4, 5):
        design = optimization.design_for_n(size)
        assert design.requirement_evaluation.status == 'satisfied'
        assert len(design.latest_before_selection.selected_times) == 2
        assert set(fixed).issubset(design.times)
        assert all(time > start for time in design.times if time not in fixed)
        comparison = compare_designs_by_simulation(
            reference, {'optimized': design.times, 'equal_time': equally_spaced_design(constraints, size)},
            observation, model_name='exponential', constraints=constraints,
            reference_model=model, simulation_plan=plan, existing_observations=existing,
        )
        first, second = tuple(comparison.results.values())
        np.testing.assert_array_equal(first.trajectory_draw_indices, second.trajectory_draw_indices)
        np.testing.assert_allclose(first.true_curves, second.true_curves)
        for simulation in comparison.results.values():
            assert simulation.fit_success.all()
            assert np.isfinite(simulation.integrated_rmse).all()
            if existing is not None:
                times, means = existing.mean_response_by_time()
                np.testing.assert_allclose(
                    simulation.observed_means[:, np.searchsorted(simulation.design_times, times)],
                    np.broadcast_to(means, (4, len(times))),
                )
    if adaptive:
        deadline = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.25)
        one_more = optimize_next_time_before_deadline(
            deadline, model, observation,
            replace(constraints, min_number_of_times=3, max_number_of_times=3),
            latest_completed_time=existing.latest_time,
        )
        assert one_more.status == 'available'
        assert existing.latest_time < one_more.new_time <= deadline.deadline
        assert set(fixed).issubset(one_more.times)


def test_ordinary_local_planning_can_add_an_earlier_unobserved_elapsed_time():
    prior = _prior()
    existing = _completed((0., 8.))
    observation = ObservationModel(kind='constant_sd', value=.02, replicates_per_time=2)
    reference = reweight_time_scaled_reference(prior, existing, observation).reference
    model = fit_reference_model_ensemble(reference, model_name='exponential')
    # The caller's candidate grid controls ordinary local eligibility. Timing
    # requirement APIs retain their separate future-only planning-start rules.
    constraints = DesignConstraints(
        candidate_times=np.array([2., 4., 6.]), fixed_times=existing.unique_times,
        min_time=0., max_time=30., min_number_of_times=3, max_number_of_times=3,
        replicates_per_time=2,
    )
    optimization = optimize_local_i_optimal_designs(model, observation, constraints, refine=False)
    selected = optimization.nested_steps[-1].times
    new = np.setdiff1d(selected, existing.unique_times)
    assert len(new) == 1 and 0. < new[0] < existing.latest_time
    assert set(existing.unique_times).issubset(selected)
