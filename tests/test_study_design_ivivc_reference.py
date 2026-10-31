from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from ivivc_core import (
    AnalysisError,
    IVIVCAnalysis,
    IVIVCModelFit,
    PredictedTrajectory,
    ProfileFit,
    ResultBatch,
    TimeCourse,
    TrajectoryEnsemble,
    fit_ivivc,
    get_model_spec,
    prepare_data,
)
from study_design_core import (
    CrossingTimeTarget,
    DesignConstraints,
    ExistingObservations,
    LatestBefore,
    ObservationModel,
    PlanningContext,
    SamplingWindow,
    compare_designs_by_simulation,
    create_simulation_plan,
    equally_spaced_design,
    fit_reference_model_ensemble,
    optimize_next_time_before_deadline,
    optimize_timing_constrained_designs,
    reference_crossing_time_summary,
    reference_from_ivivc,
    resolve_timing_requirements,
    reweight_reference_ensemble,
)


METHODS = ('value_ratio', 'tau_ratio', 'observed_response_inverse')
SELECTION = 'approach2:exponential_normalized'


@pytest.fixture(scope='module')
def approach2_analysis():
    """Fit genuine paired profiles; normalized amplitude keeps zero supported."""
    time = np.linspace(0.0, 25.0, 11)
    perturbation = np.array([1., 1.003, .998, 1.002, .999, 1.001, .998, 1.002, 1., .999, 1.])
    response = np.exp(-0.2 * time) * perturbation
    prepared = prepare_data(time, response, 2 * time, response, selected_interpolation=['none'])
    analysis = fit_ivivc(
        prepared, selections=SELECTION, selected_metrics=['rmse'],
        analysis_config={
            'cv_scheme': 'none', 'grid_search_num_points': 3,
            'grid_search_num_cores': 1, 'grid_search_random_state': 12345,
        },
    )
    assert analysis.errors == {}
    return analysis


def _prediction():
    time = np.linspace(0.0, 25.0, 101)
    return TimeCourse(time, np.exp(-0.2 * time), label='target in vitro')


def _reference(analysis, method, **kwargs):
    return reference_from_ivivc(
        analysis, _prediction(), selection=SELECTION, method=method,
        n_samples=12, random_state=321, **kwargs,
    )


def _profile(name, time, parameters):
    spec = get_model_spec(name)
    vector = np.array([parameters[key] for key in spec.parameter_names])
    values = np.asarray(spec.model_function(time, *vector))
    return ProfileFit(
        model_name=name, time=time, response=values,
        parameter_names=spec.parameter_names, parameters=parameters,
        covariance=np.zeros((len(vector), len(vector))), initial_parameters=vector,
        fitted_values=values, residuals=np.zeros_like(values),
        residual_dof=len(time) - len(vector), fit_metrics={}, cv_scores={}, cv_means={},
    )


@pytest.mark.parametrize('method', METHODS)
def test_approach2_reference_recovers_known_exponential_curve(approach2_analysis, method):
    grid = np.linspace(0.0, 15.0, 61)
    reference = _reference(approach2_analysis, method, time_grid=grid, required_times=[0., 15.])
    assert reference.n_draws == reference.n_valid_draws == 12
    assert reference.supported_time_range == (0., 15.)
    assert reference.metadata['common_supported_time_range'] == (0., 15.)
    assert reference.metadata['source_shared_supported_time_range'][1] > 15.
    with pytest.raises(ValueError, match='supported reference domain'):
        reference.evaluate([16.])
    np.testing.assert_allclose(reference.central_curve, np.exp(-.1 * grid), rtol=.01)
    np.testing.assert_allclose(reference.draw_weights, np.full(12, 1 / 12))
    assert reference.metadata['ivivc_selection'] == SELECTION
    assert reference.metadata['ivivc_calibration_model'] == 'exponential_normalized'
    assert reference.metadata['ivivc_prediction_method'] == method
    assert reference.metadata['ivivc_approach'] == 'approach2'
    summary = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.5)
    assert summary.status == 'available'
    assert summary.deadline == pytest.approx(np.log(2) / .1, rel=.01)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('adaptive', [False, True])
def test_approach2_supports_existing_design_and_simulation_workflows(
    approach2_analysis, method, adaptive,
):
    reference = _reference(approach2_analysis, method, time_grid=np.linspace(0., 15., 61))
    observation = ObservationModel(kind='constant_sd', value=.02, replicates_per_time=2)
    existing = None
    if adaptive:
        existing = ExistingObservations(
            time=np.array([0., 0., 2., 2.]),
            response=np.array([1.01, .99, np.exp(-.2) + .01, np.exp(-.2) - .01]),
        )
        update = reweight_reference_ensemble(reference, existing, observation)
        reference = update.reference
        assert 1 <= update.effective_sample_size <= reference.n_draws + 1e-12
        np.testing.assert_allclose(reference.draw_weights, update.posterior_weights)
        assert reference.metadata['ivivc_prediction_method'] == method
    completed = () if existing is None else tuple(existing.unique_times)
    fixed = np.array([0.]) if existing is None else existing.unique_times
    start = 0. if existing is None else existing.latest_time
    constraints = DesignConstraints(
        candidate_times=np.arange(start + 1, 16.), fixed_times=fixed,
        min_time=0., max_time=15., minimum_spacing=1.,
        min_number_of_times=4, max_number_of_times=5, replicates_per_time=2,
    )
    requirements = (
        SamplingWindow(0., CrossingTimeTarget(.5, .25), label='Early coverage'),
        LatestBefore(CrossingTimeTarget(.5, .25), label='T50'),
        LatestBefore(CrossingTimeTarget(.75, .25), label='T75'),
    )
    resolved = resolve_timing_requirements(
        requirements, reference, constraints,
        context=PlanningContext([0.], completed, start),
    )
    # Calibration used one parameter; the study independently estimates two.
    model = fit_reference_model_ensemble(reference, model_name='exponential')
    assert model.n_parameters == 2
    if adaptive:
        summary = reference_crossing_time_summary(reference, degraded_fraction=.5, quantile=.25)
        next_point = optimize_next_time_before_deadline(
            summary, model, observation,
            replace(constraints, min_number_of_times=3, max_number_of_times=3),
            latest_completed_time=existing.latest_time,
        )
        assert next_point.status == 'available'
        assert existing.latest_time < next_point.new_time <= summary.deadline
        assert len(next_point.times) == 3 and set(fixed).issubset(next_point.times)
    optimization = optimize_timing_constrained_designs(
        model, observation, constraints, resolved, number_of_times=[4, 5],
    )
    plan = create_simulation_plan(
        reference, observation, constraints.all_available_times,
        n_simulations=4, random_state=987,
    )
    for size in (4, 5):
        design = optimization.design_for_n(size)
        assert design.requirement_evaluation.status == 'satisfied'
        assert len(design.latest_before_selection.selected_times) == 2
        assert set(fixed).issubset(design.times)
        comparison = compare_designs_by_simulation(
            reference,
            {'optimized': design.times, 'equal_time': equally_spaced_design(constraints, size)},
            observation, model_name='exponential', constraints=constraints,
            reference_model=model, simulation_plan=plan, existing_observations=existing,
        )
        simulations = tuple(comparison.results.values())
        np.testing.assert_array_equal(
            simulations[0].trajectory_draw_indices, simulations[1].trajectory_draw_indices,
        )
        np.testing.assert_allclose(simulations[0].true_curves, simulations[1].true_curves)
        for simulation in simulations:
            assert simulation.fit_success.all()
            assert np.all(np.isfinite(simulation.integrated_rmse))
            if existing is not None:
                times, means = existing.mean_response_by_time()
                np.testing.assert_allclose(
                    simulation.observed_means[:, np.searchsorted(simulation.design_times, times)],
                    np.broadcast_to(means, (4, len(times))),
                )


def test_selection_isolation_preserves_seed_and_ignores_unrelated_failures(approach2_analysis):
    original = _reference(approach2_analysis, 'tau_ratio', grid_size=21)
    extra = replace(approach2_analysis[SELECTION], model_name='not_a_registered_model')
    unrelated_error = AnalysisError.from_exception(ValueError('Unrelated calibration failed.'))
    expanded = replace(
        approach2_analysis,
        model_results={extra.key: extra, **approach2_analysis.model_results},
        errors={'approach1:bad_fit': unrelated_error},
    )
    actual = _reference(expanded, 'tau_ratio', grid_size=21)
    np.testing.assert_allclose(actual.time_grid, original.time_grid)
    np.testing.assert_allclose(actual.value_draws, original.value_draws)
    assert set(expanded.model_results) == {extra.key, SELECTION}
    assert expanded.errors == {'approach1:bad_fit': unrelated_error}


def test_unknown_failed_and_incompatible_selections_are_explicit(approach2_analysis):
    error = AnalysisError.from_exception(ValueError('Calibration failed.'))
    failed = replace(approach2_analysis, errors={'approach2:failed': error})
    with pytest.raises(ValueError, match='not present'):
        reference_from_ivivc(failed, _prediction(), selection='missing', method='tau_ratio')
    with pytest.raises(ValueError, match='Calibration failed'):
        reference_from_ivivc(failed, _prediction(), selection='approach2:failed', method='tau_ratio')
    with pytest.raises(ValueError, match='not supported'):
        _reference(failed, 'time_mapping')


def test_failed_uncertainty_generation_is_not_replaced_with_a_deterministic_curve(approach2_analysis):
    fit = approach2_analysis[SELECTION]
    broken = replace(fit, in_vitro_fit=replace(fit.in_vitro_fit, covariance=np.array([[np.nan]])))
    analysis = replace(approach2_analysis, model_results={SELECTION: broken})
    with pytest.raises(ValueError, match='could not be generated'):
        _reference(analysis, 'value_ratio')


def test_existing_approach1_ensemble_path_remains_available(approach2_analysis):
    fit = IVIVCModelFit(
        'approach1', 'linear_zero_intercept',
        time_mapping_fit=_profile('linear_zero_intercept', np.arange(6.), {'a': 2.}),
    )
    analysis = replace(approach2_analysis, model_results={fit.key: fit})
    reference = reference_from_ivivc(
        analysis, _prediction(), selection=fit.key, method='time_mapping',
        time_grid=np.linspace(0., 15., 61), n_samples=4, random_state=321,
    )
    np.testing.assert_allclose(reference.central_curve, np.exp(-.1 * reference.time_grid), rtol=1e-4)
    assert reference.metadata['ivivc_approach'] == 'approach1'


@pytest.mark.parametrize('method', METHODS)
def test_requested_horizon_is_not_extrapolated_during_reference_conversion(approach2_analysis, method):
    with pytest.raises(ValueError, match='shared supported range'):
        _reference(approach2_analysis, method, time_grid=[0., 100.])


@pytest.mark.parametrize('required', [[np.nan], [[0.]], 0.])
def test_required_times_must_be_a_finite_sequence(approach2_analysis, required):
    with pytest.raises(ValueError, match='required_times'):
        _reference(approach2_analysis, 'tau_ratio', required_times=required)


@pytest.mark.parametrize('scaling', ['log_x', 'log_y'])
def test_logged_coordinates_are_not_silently_used_for_study_design(approach2_analysis, scaling):
    prepared = approach2_analysis.prepared_data
    logged = replace(approach2_analysis, prepared_data=replace(
        prepared, preprocessing=replace(prepared.preprocessing, applied_scalings=(scaling,)),
    ))
    with pytest.raises(ValueError, match='unlogged'):
        _reference(logged, 'tau_ratio')


def test_unsupported_tau_does_not_disable_other_approach2_methods():
    time = np.linspace(0., 10., 21)
    name = 'mass_loss_one_phase_normalized'
    iv = _profile(name, time, {'t50': 5., 's': 1.})
    ivv = _profile(name, 2 * time, {'t50': 10., 's': 2.})
    fit = IVIVCModelFit('approach2', name, in_vitro_fit=iv, in_vivo_fit=ivv)
    analysis = IVIVCAnalysis(
        prepare_data(time, iv.response, 2 * time, ivv.response, selected_interpolation=['none']),
        {fit.key: fit},
    )
    prediction = TimeCourse(time, iv.response)
    with pytest.raises(ValueError, match='unavailable'):
        reference_from_ivivc(
            analysis, prediction, selection=fit.key, method='tau_ratio', n_samples=2,
        )
    reference = reference_from_ivivc(
        analysis, prediction, selection=fit.key, method='value_ratio', n_samples=2,
    )
    assert reference.n_valid_draws == 2


@pytest.mark.parametrize('nominal_times,available,error_message', [
    ([0., 1.], False, 'Nominal time constant'),
    ([0., np.nan], True, 'at least two distinct finite'),
])
def test_unusable_nominal_prediction_is_not_replaced_with_a_draw_mean(
    approach2_analysis, nominal_times, available, error_message,
):
    prediction = TimeCourse(np.array([0., 1.]), np.array([1., .5]))
    central = PredictedTrajectory(
        source_approach='approach2', source_method='tau_ratio',
        in_vitro_time=prediction.time, in_vitro_value=prediction.value,
        in_vivo_time=np.array(nominal_times), in_vivo_value=prediction.value,
        available=available, skip_reason='Nominal time constant is unavailable.' if not available else None,
    )
    ensemble = TrajectoryEnsemble(
        central=central, time_draws=prediction.time[None, :],
        value_draws=prediction.value[None, :], valid_draws=np.array([True]),
    )
    batch = ResultBatch(results={f'{SELECTION}:tau_ratio': ensemble}, errors={})
    with patch('study_design_core.ivivc_reference.generate_trajectory_ensembles', return_value=batch):
        with pytest.raises(ValueError, match=error_message):
            reference_from_ivivc(
                approach2_analysis, prediction, selection=SELECTION, method='tau_ratio',
            )


def test_partial_inverse_support_preserves_diagnostics_and_rejects_required_zero():
    time = np.linspace(0., 25., 101)
    iv = _profile('exponential', time, {'a': 1., 'b': -.2})
    ivv = _profile('exponential', 2 * time, {'a': .8, 'b': -.1})
    fit = IVIVCModelFit('approach2', 'exponential', in_vitro_fit=iv, in_vivo_fit=ivv)
    analysis = IVIVCAnalysis(
        prepare_data(time, iv.response, 2 * time, ivv.response, selected_interpolation=['none']),
        {fit.key: fit},
    )
    kwargs = dict(selection=fit.key, method='observed_response_inverse', n_samples=3, random_state=19)
    reference = reference_from_ivivc(analysis, TimeCourse(time, iv.response), **kwargs)
    supported = reference.metadata['ivivc_prediction_supported_points']
    assert not supported[0] and supported[-1]
    assert 0 < np.count_nonzero(supported) < len(time)
    assert reference.supported_time_range[0] > 0
    assert reference.metadata['ivivc_prediction_status'][0] != 'mapped'
    assert 'extrapolation_status' in reference.metadata['ivivc_prediction_auxiliary']
    with pytest.raises(ValueError, match='required_times'):
        reference_from_ivivc(analysis, TimeCourse(time, iv.response), required_times=[0.], **kwargs)
