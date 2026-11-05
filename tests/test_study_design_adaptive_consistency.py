"""Adaptive selection and refitting use the same completed-row precision."""
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse
from study_design_core import (
    DesignConstraints, ExistingObservations, LatestBefore, ObservationModel,
    PlanningContext, ReferenceModelEnsemble, ReferenceTrajectoryEnsemble, SamplingWindow,
    evaluate_local_i_optimal_design, fit_reference_model_ensemble, fit_reference_profile,
    information_matrix, optimize_next_time_before_deadline,
    optimize_timing_constrained_designs, optimize_window_constrained_designs,
    reference_crossing_time_summary, reference_from_time_scaling,
    resolve_timing_requirements, reweight_reference_ensemble,
    reweight_time_scaled_reference, simulate_design,
)
import study_design_core.simulation as simulation


def reference():
    time = np.arange(11.)
    return ReferenceTrajectoryEnsemble(time, np.vstack([1-time/12, 1-time/15]),
        [True, True], draw_weights=[.4, .6], interpolation='linear')


def analysis():
    return ReferenceModelEnsemble('linear', ('a', 'b'), [[1., -.07]], [True], [1.],
                                  [1., -.07], np.arange(11.), (0., 10.))


def completed():
    return ExistingObservations([0., 2.], [1., .85], observation_uncertainty=
        ObservationUncertainty([.02, .12], response_kind='mean', replicate_counts=[4, 25]))


def constraints(fixed=(0., 2.), size=4):
    return DesignConstraints(np.arange(11.), fixed_times=fixed, min_time=0., max_time=10.,
        minimum_spacing=1., min_number_of_times=size, max_number_of_times=size,
        replicates_per_time=3)


class AdaptiveConsistencyTests(unittest.TestCase):
    def test_cv_optimizer_information_matches_refit_covariance_for_mismatched_source(self):
        time = np.linspace(0., 4., 9)
        curves = np.vstack([.9+.1*time+.05*time**2, 1.1+.2*time+.04*time**2])
        prior = ReferenceTrajectoryEnsemble(time, curves, [True, True], interpolation='linear')
        fitted = fit_reference_model_ensemble(prior, model_name='linear')
        measured = ExistingObservations([0.], [1.], observation_uncertainty=
            ObservationUncertainty(.07, response_kind='mean', replicate_counts=9))
        error = ObservationModel('constant_cv', .1, replicates_per_time=2)
        records = []
        original = simulation.fit_model_once
        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            records.append(result)
            return result
        with patch.object(simulation, 'fit_model_once', side_effect=record):
            result = simulate_design(prior, [0., 2., 4.], error, model_name='linear',
                reference_model=fitted, existing_observations=measured,
                n_simulations=2, random_state=4, fallback_grid_search=False)
        matrix = information_matrix('linear', [0., 2., 4.], fitted.central_parameters,
                                    error, existing_observations=measured)
        self.assertTrue(result.fit_success.all())
        for fit in records:
            np.testing.assert_allclose(fit.covariance, np.linalg.inv(matrix), rtol=1e-5)
        nominal = fitted.central_parameters[0] + fitted.central_parameters[1]*np.array([2., 4.])
        np.testing.assert_allclose(result.metadata['refit_observation_sigma'],
                                   np.r_[.07, .1*np.abs(np.tile(nominal, 2))])
        self.assertFalse(np.allclose(nominal, prior.evaluate_central([2., 4.])))

    def test_generic_cumulative_update_uses_sem_once_and_rejects_updated_prior(self):
        prior = reference()
        original_weights = prior.draw_weights.copy()
        measured = ExistingObservations([1., 3.], [.9, .76], observation_uncertainty=
            ObservationUncertainty([.1, .2], response_kind='mean', replicate_counts=[25, 4]))
        first = ExistingObservations([1.], [.9], observation_uncertainty=
            ObservationUncertainty(.1, response_kind='mean', replicate_counts=25))
        error = ObservationModel('constant_sd', 10., replicates_per_time=99)
        previous = reweight_reference_ensemble(prior, first, error)
        update = reweight_reference_ensemble(prior, measured, error)
        residual = measured.response - prior.evaluate(measured.time)
        expected_log = np.sum(-.5*(residual/np.array([.1, .2]))**2
                              - np.log([.1, .2]) - .5*np.log(2*np.pi), axis=1)
        weight = original_weights*np.exp(expected_log-expected_log.max())
        np.testing.assert_allclose(update.posterior_weights, weight/weight.sum())
        np.testing.assert_array_equal(prior.draw_weights, original_weights)
        with self.assertRaisesRegex(ValueError, 'original|cumulative'):
            reweight_reference_ensemble(previous.reference, measured, error)

    def test_empirical_cumulative_update_has_the_same_prior_guard(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            source = fit_reference_profile(TimeCourse([0., 2., 4.], [1., .7, .4]),
                model_name='linear', analysis_config={'cv_scheme': 'none',
                    'grid_search_num_points': 3, 'grid_search_num_cores': 1})
        prior = reference_from_time_scaling(source, [1., 2.], [0., 1., 2., 3., 4.])
        measured = ExistingObservations([1.], [.85], observation_uncertainty=
            ObservationUncertainty(.03, response_kind='mean', replicate_counts=16))
        error = ObservationModel('constant_sd', .1)
        update = reweight_time_scaled_reference(prior, measured, error)
        np.testing.assert_array_equal(update.reference.value_draws, prior.value_draws)
        with self.assertRaisesRegex(ValueError, 'original|cumulative'):
            reweight_time_scaled_reference(update.reference, measured, error)
        with self.assertRaisesRegex(ValueError, 'original|cumulative'):
            reweight_reference_ensemble(update.reference, measured, error)

    def test_window_and_mixed_schedules_use_actual_completed_precision(self):
        design = constraints(fixed=(0., 2., 8.))
        measured = completed()
        error = ObservationModel('constant_sd', .1, replicates_per_time=3)
        requests = ([SamplingWindow(2., 7.)], [SamplingWindow(2., 7.), LatestBefore(7.)])
        for optimizer, requirements in zip(
                (optimize_window_constrained_designs, optimize_timing_constrained_designs), requests):
            with self.subTest(optimizer=optimizer.__name__):
                resolved = resolve_timing_requirements(requirements, reference(), design,
                    context=PlanningContext([0.], [0., 2.]))
                result = optimizer(analysis(), error, design, resolved, number_of_times=4,
                                   existing_observations=measured, refine=True).result_for_n(4)
                self.assertEqual(result.status, 'feasible')
                self.assertIn(8., result.times)  # Fixed future commitment remains planned.
                expected = evaluate_local_i_optimal_design(result.times, analysis(), error, design,
                                                           existing_observations=measured)
                self.assertAlmostEqual(result.score.criterion, expected.criterion, places=12)

    def test_deadline_score_uses_actual_precision_and_latest_time(self):
        design = constraints(size=3)
        measured = completed()
        error = ObservationModel('constant_sd', .1, replicates_per_time=3)
        summary = reference_crossing_time_summary(reference(), quantile=.1)
        result = optimize_next_time_before_deadline(summary, analysis(), error, design,
            latest_completed_time=2., existing_observations=measured)
        self.assertEqual(result.status, 'available')
        expected = evaluate_local_i_optimal_design(result.times, analysis(), error, design,
                                                   existing_observations=measured)
        self.assertAlmostEqual(result.score.criterion, expected.criterion, places=12)
        with self.assertRaisesRegex(ValueError, 'latest_completed_time'):
            optimize_next_time_before_deadline(summary, analysis(), error, design,
                latest_completed_time=3., existing_observations=measured)

    def test_scheduling_rejects_completed_role_mismatch(self):
        design = constraints()
        resolved = resolve_timing_requirements((), reference(), design,
            context=PlanningContext([0.], [0.]))
        for optimizer in (optimize_window_constrained_designs, optimize_timing_constrained_designs):
            with self.subTest(optimizer=optimizer.__name__), self.assertRaisesRegex(ValueError, 'completed'):
                optimizer(analysis(), ObservationModel('constant_sd', .1, 3), design, resolved,
                          existing_observations=completed())

    def test_poc_forwards_actual_precision_through_all_adaptive_run_routes(self):
        import pandas as pd
        from examples import study_design_proof_of_concept as poc
        from examples.study_design_mixed_workflow import SchedulePlanningSpec

        time = np.linspace(0., 4., 9)
        prior = ReferenceTrajectoryEnsemble(time, [1-.15*time], [True], interpolation='linear')
        measured = ExistingObservations([0., 0.], [1.01, .99], observation_uncertainty=
            ObservationUncertainty([.04, .08], response_kind='mean', replicate_counts=[4, 9]))
        design = DesignConstraints([1., 2., 3., 4.], fixed_times=[0.], min_time=0., max_time=4.,
            min_number_of_times=2, max_number_of_times=2, replicates_per_time=3)
        error = ObservationModel('constant_sd', .2, replicates_per_time=3)
        settings = poc.ProofOfConceptSettings(n_simulations=2, n_reference_draws=4,
                                              max_design_sizes=1, refine=False, write_html=False)
        for route in ('adaptive_in_vivo_only', 'adaptive_timing_comparison', 'mixed_timing_requirements'):
            with self.subTest(route=route):
                planning = (SchedulePlanningSpec(PlanningContext([0.], [0.]), (), 'adaptive-in-vivo')
                            if route == 'mixed_timing_requirements' else None)
                scenario = poc.ScenarioDefinition(route, route, 'precision routing', prior, error,
                    design, 'linear', pd.DataFrame({'time': measured.time, 'response': measured.response}),
                    target_response=.5, existing_observations=measured, planning=planning,
                    metadata={'timing_degraded_fraction': .5, 'timing_quantile': .1,
                              'timing_baseline_time': 0.})
                result = poc.run_scenario(scenario, settings)
                outcomes = [result.optimization]
                if result.planning_optimization is not None:
                    outcomes.append(result.planning_optimization)
                for outcome in outcomes:
                    self.assertEqual(outcome.metadata['completed_precision_source'], 'supplied_sem')
                    self.assertEqual(outcome.metadata['completed_observation_count'], 2)
                    step = (outcome.result_for_n(2) if planning else outcome.nested_steps[0])
                    expected = evaluate_local_i_optimal_design(step.times, result.reference_model,
                        error, design, existing_observations=measured)
                    self.assertAlmostEqual(step.score.criterion, expected.criterion, places=12)
                if result.timing_policy_table is not None:
                    for row in result.timing_policy_table.to_dict(orient='records'):
                        self.assertEqual(row['status'], 'available')
                        expected = evaluate_local_i_optimal_design([0., row['new_time']],
                            result.reference_model, error, design, existing_observations=measured)
                        self.assertAlmostEqual(row['information_criterion'], expected.criterion, places=12)


if __name__ == '__main__':
    unittest.main()
