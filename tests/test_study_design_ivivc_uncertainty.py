"""Reference adapters retain shared uncertainty without adding future noise."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from ivivc_core import (
    ObservationUncertainty, fit_ivivc, generate_trajectory_ensembles,
    prepare_data, prepare_time_course,
)
from study_design_core import (
    ExistingObservations, reference_from_ivivc, reference_from_trajectory_ensemble,
)


CONFIG = {'cv_scheme': 'none', 'grid_search_num_points': 3,
          'grid_search_num_cores': 1, 'grid_search_random_state': 12345}
SELECTION = 'approach2:exponential_normalized'
METHODS = ('value_ratio', 'tau_ratio', 'observed_response_inverse')


class IVIVCReferenceUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        time = np.linspace(0., 25., 9)
        response = np.exp(-.2*time) * np.array([1., 1.002, .998, 1.001, 1., .999, 1.002, 1., 1.])
        precision = ObservationUncertainty(.002, response_kind='mean', replicate_counts=4)
        prepared = prepare_data(time, response, 2*time, response,
            selected_interpolation=['none'], in_vitro_uncertainty=precision,
            in_vivo_uncertainty=precision)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            cls.analysis = fit_ivivc(prepared, selections=SELECTION,
                selected_metrics=['rmse'], analysis_config=CONFIG, fail_fast=True)
        time = np.linspace(1., 20., 41)
        cls.prediction, _ = prepare_time_course(time, np.exp(-.2*time),
            observation_uncertainty=ObservationUncertainty(.0005, response_kind='mean'))

    def reference(self, prediction, method):
        return reference_from_ivivc(self.analysis, prediction, selection=SELECTION,
            method=method, n_samples=8, random_state=432, grid_size=21, interpolation='linear')

    def test_all_approach2_modes_preserve_shared_joint_draws_and_precision_metadata(self):
        for method in METHODS:
            with self.subTest(method=method):
                source = generate_trajectory_ensembles(self.analysis, self.prediction,
                    methods={'approach2': method}, n_samples=8, random_state=432,
                    fail_fast=True)[f'{SELECTION}:{method}']
                expected = reference_from_trajectory_ensemble(source,
                    grid_size=21, interpolation='linear')
                reference = self.reference(self.prediction, method)
                np.testing.assert_array_equal(reference.time_grid, expected.time_grid)
                np.testing.assert_array_equal(reference.value_draws, expected.value_draws)
                np.testing.assert_array_equal(reference.valid_draws, expected.valid_draws)
                self.assertEqual(reference.metadata['calibration_uncertainty'],
                                 source.metadata['calibration_uncertainty'])
                self.assertEqual(reference.metadata['prediction_input_uncertainty'],
                                 self.prediction.observation_uncertainty.to_metadata())
                self.assertFalse(reference.metadata['future_observation_variability_included'])
                self.assertTrue(reference.metadata['prediction_input_uncertainty_propagated'])
                self.assertEqual(reference.metadata['calibration_uncertainty']['in_vivo']
                                 ['covariance_scaling'], 'absolute')

    def test_prediction_sem_does_not_change_calibration_draws_or_get_applied_twice(self):
        larger, _ = prepare_time_course(self.prediction.time, self.prediction.value,
            observation_uncertainty=ObservationUncertainty(.001, response_kind='mean',
                                                          replicate_counts=100))
        for method in METHODS:
            with self.subTest(method=method):
                small, large = self.reference(self.prediction, method), self.reference(larger, method)
                parameter_key = ('parameter_samples' if method == 'observed_response_inverse'
                                 else 'in_vivo_parameter_samples')
                np.testing.assert_array_equal(small.metadata[parameter_key], large.metadata[parameter_key])
                center = self.prediction.value[None, :]
                np.testing.assert_allclose(large.metadata['prediction_response_draws'] - center,
                    2*(small.metadata['prediction_response_draws'] - center), atol=2e-16)

    def test_adaptive_poc_preserves_completed_sem_in_source_fit_and_scenario_copy(self):
        from examples import study_design_proof_of_concept as poc

        original = poc._generate_completed_target_observations
        supplied = []
        def with_precision(*args, **kwargs):
            completed, table = original(*args, **kwargs)
            completed = ExistingObservations(completed.time, completed.response,
                label=completed.label, observation_uncertainty=ObservationUncertainty(
                    np.linspace(.01, .03, completed.n_observations),
                    response_kind='mean', replicate_counts=4))
            supplied.append(completed)
            return completed, table

        settings = replace(poc.ProofOfConceptSettings(), analysis_model='exponential',
                           n_reference_draws=5, n_simulations=2, max_design_sizes=1)
        with patch.object(poc, '_generate_completed_target_observations', side_effect=with_precision):
            scenario = poc.build_adaptive_in_vivo_only_scenario(settings)
        expected = supplied[0].observation_uncertainty.to_metadata()
        self.assertEqual(scenario.existing_observations.observation_uncertainty.to_metadata(), expected)
        self.assertEqual(scenario.reference.metadata['source_observation_uncertainty'], expected)
        self.assertEqual(scenario.reference.metadata['source_covariance_scaling'], 'absolute')


if __name__ == '__main__':
    unittest.main()
