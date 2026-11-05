"""POC controls keep reported precision separate from future-study error."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples import study_design_proof_of_concept as poc
from examples.study_design_poc_uncertainty import load_future_sd_profile


def settings(**changes):
    options = dict(analysis_model='exponential', n_reference_draws=4, n_simulations=2,
                   max_design_sizes=1, random_seed=1234, refine=False, write_html=False,
                   observation_sd=.02, ivivc_calibration_response_sd=.003)
    options.update(changes)
    return poc.ProofOfConceptSettings(**options)


class POCUncertaintyControlsTests(unittest.TestCase):
    def test_defaults_keep_unreported_source_precision_and_legacy_future_assumption(self):
        configured = settings()
        self.assertFalse(configured.use_measurement_uncertainty)
        self.assertEqual(configured.source_response_kind, 'individual')
        scenario = poc.build_measured_in_vivo_scenario(configured)
        self.assertIsNone(scenario.reference.metadata['source_observation_uncertainty'])
        self.assertEqual(scenario.observation_model.kind, 'constant_sd')
        self.assertEqual(scenario.observation_model.value, configured.observation_sd)
        self.assertEqual(scenario.constraints.replicates_per_time, 3)
        self.assertGreaterEqual(scenario.observed_data['response'].min(), .002)

    def test_new_settings_validate_precision_and_positive_integer_counts(self):
        invalid = [
            {'source_response_kind': 'mean'}, {'source_response_kind': 'other'},
            {'future_observation_sd': 0.}, {'future_observation_sd': np.nan},
            {'future_sd_profile': ((0., .1), (0., .2))},
            {'future_sd_profile': ((False, .1), (1., .2))},
            {'future_sd_profile': ((0., .1),), 'future_observation_sd': .2},
        ]
        for field in ('reference_replicates_per_time', 'completed_replicates_per_time',
                      'future_replicates_per_time'):
            invalid.extend([{field: 0}, {field: 1.5}, {field: True}])
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                settings(**changes)
        with self.assertRaisesRegex(ValueError, 'positive'):
            poc.build_empirical_time_scaling_scenario(settings(
                use_measurement_uncertainty=True, ivivc_calibration_response_sd=0.))

    def test_profile_csv_rejects_malformed_records_instead_of_dropping_values(self):
        contents = ('time,sem\n0,.1\n', 'time,sd\n', 'time,sd\n0,0\n',
                    'time,sd\n0,.1\n0,.2\n', 'time,sd\n1,.1\n0,.2\n',
                    'time,sd\n0,nan\n', 'time,sd,sd\n0,.1,.2\n',
                    'time,sd\n0,.1,extra\n', 'time,sd\n0\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.csv'
            for content in contents:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_future_sd_profile(path)
            path.write_text('time,sd\n0,.02\n10,.08\n')
            self.assertEqual(load_future_sd_profile(path), ((0., .02), (10., .08)))

    def test_cli_forwards_source_counts_profile_and_prediction_sem(self):
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.csv'
            path.write_text('time,sd\n0,.02\n52,.08\n')
            arguments = ['--use-measurement-uncertainty', '--source-response-kind', 'mean',
                '--reference-replicates-per-time', '4', '--completed-replicates-per-time', '2',
                '--future-replicates-per-time', '5', '--future-sd-profile-csv', str(path),
                '--ivivc-approach', 'approach2', '--ivivc-prediction-sem', '.0005', '--no-html']
            def run(*args, **kwargs):
                captured.append(kwargs['settings'])
                return object()
            with patch.object(poc, 'run_proof_of_concept', side_effect=run), \
                    patch.object(poc, 'write_proof_of_concept_outputs'), \
                    patch.object(poc, '_print_console_summary'), redirect_stdout(StringIO()):
                self.assertEqual(poc.main(arguments), 0)
        configured = captured[0]
        self.assertTrue(configured.use_measurement_uncertainty)
        self.assertEqual(configured.source_response_kind, 'mean')
        self.assertEqual((configured.reference_replicates_per_time,
                          configured.completed_replicates_per_time,
                          configured.future_replicates_per_time), (4, 2, 5))
        self.assertEqual(configured.future_sd_profile, ((0., .02), (52., .08)))
        self.assertEqual(configured.ivivc_prediction_sem, .0005)
        self.assertEqual(poc._parse_args(['--future-observation-sd', '.17']).future_observation_sd, .17)

    def test_measured_means_keep_actual_counts_and_sem_once(self):
        configured = settings(use_measurement_uncertainty=True,
            reference_replicates_per_time=4, future_replicates_per_time=7)
        raw = poc.build_measured_in_vivo_scenario(configured)
        means = poc.build_measured_in_vivo_scenario(replace(configured, source_response_kind='mean'))
        expected = raw.observed_data.groupby('time', sort=True)['response'].mean().to_numpy()
        np.testing.assert_allclose(means.observed_data['response'], expected)
        precision = means.reference.metadata['source_observation_uncertainty']
        self.assertEqual(precision['uncertainty_kind'], 'sem')
        np.testing.assert_allclose(precision['standard_uncertainty'], configured.observation_sd/2)
        np.testing.assert_array_equal(precision['replicate_counts'], np.full(8, 4))
        self.assertEqual(means.reference.metadata['source_covariance_scaling'], 'absolute')
        self.assertEqual(means.constraints.replicates_per_time, 7)
        truth = poc._common_in_vivo_preset(configured).evaluate(np.unique(raw.observed_data['time']))
        expected_raw = truth[:, None] + np.random.default_rng(configured.random_seed+101).normal(
            0., configured.observation_sd, size=(8, 4))
        np.testing.assert_array_equal(raw.observed_data['response'], expected_raw.reshape(-1))

    def test_future_counts_do_not_change_sources_completed_sem_or_prior_draws(self):
        configured = settings(use_measurement_uncertainty=True, source_response_kind='mean',
            reference_replicates_per_time=4, completed_replicates_per_time=2,
            time_scaling_mode='adaptive', time_scaling_parameter_draws=4)
        builders = (poc.build_measured_in_vivo_scenario, poc.build_adaptive_in_vivo_only_scenario,
                    poc.build_empirical_time_scaling_scenario)
        for build in builders:
            with self.subTest(builder=build.__name__):
                first = build(configured)
                second = build(replace(configured, future_replicates_per_time=7))
                pd.testing.assert_frame_equal(first.observed_data, second.observed_data)
                np.testing.assert_array_equal(first.reference.value_draws, second.reference.value_draws)
                np.testing.assert_array_equal(first.reference.draw_weights, second.reference.draw_weights)
                self.assertEqual(second.constraints.replicates_per_time, 7)
                self.assertEqual(second.observation_model.replicates_per_time, 7)
                if first.existing_observations is not None:
                    precision = first.existing_observations.observation_uncertainty
                    np.testing.assert_allclose(precision.sigma, configured.observation_sd/np.sqrt(2))
                    np.testing.assert_array_equal(precision.replicate_counts,
                                                 np.full(first.existing_observations.n_observations, 2))
                    self.assertEqual(precision.to_metadata(),
                        second.existing_observations.observation_uncertainty.to_metadata())

    def test_future_profile_uses_in_vivo_times_and_leaves_historical_source_unchanged(self):
        configured = settings(use_measurement_uncertainty=True)
        plain = poc.build_measured_in_vivo_scenario(configured)
        profiled = poc.build_measured_in_vivo_scenario(replace(configured,
            future_sd_profile=((0., .01), (20., .08), (52., .03))))
        pd.testing.assert_frame_equal(plain.observed_data, profiled.observed_data)
        np.testing.assert_array_equal(plain.reference.value_draws, profiled.reference.value_draws)
        np.testing.assert_array_equal(profiled.observation_model.standard_deviation(
            [1., 1., 1.], times=[0., 25., 52.]), [.01, .08, .03])
        self.assertIsNone(profiled.metadata['future_observation_sd'])
        self.assertEqual(profiled.metadata['future_observation_model']['kind'], 'time_profile_sd')
        constant = poc.build_measured_in_vivo_scenario(replace(configured, future_observation_sd=.17))
        self.assertEqual(constant.observation_model.value, .17)
        pd.testing.assert_frame_equal(plain.observed_data, constant.observed_data)

    def test_future_overrides_preserve_legacy_completed_likelihood_and_source_fit(self):
        configured = settings(ivivc_approach='approach2', ivivc_model='exponential_normalized',
            ivivc_method='tau_ratio', time_scaling_mode='adaptive', time_scaling_parameter_draws=4)
        builders = (poc.build_adaptive_in_vivo_only_scenario,
                    poc.build_adaptive_ivivc_scenario, poc.build_empirical_time_scaling_scenario)
        overrides = ({'future_observation_sd': .02}, {'future_observation_sd': .2},
                     {'future_sd_profile': ((0., .01), (52., .2))})
        for build in builders:
            baseline = build(configured)
            for change in overrides:
                with self.subTest(builder=build.__name__, change=change):
                    scenario = build(replace(configured, **change))
                    np.testing.assert_array_equal(scenario.existing_observations.time,
                                                  baseline.existing_observations.time)
                    np.testing.assert_array_equal(scenario.existing_observations.response,
                                                  baseline.existing_observations.response)
                    np.testing.assert_array_equal(scenario.reference.value_draws, baseline.reference.value_draws)
                    np.testing.assert_array_equal(scenario.reference.draw_weights, baseline.reference.draw_weights)
                    precision = scenario.existing_observations.observation_uncertainty
                    np.testing.assert_allclose(precision.sigma, configured.observation_sd)
                    self.assertEqual(precision.source, 'historical_nominal_sd_assumption')
                    if build is poc.build_adaptive_in_vivo_only_scenario:
                        self.assertIsNone(scenario.reference.metadata['source_observation_uncertainty'])
                    if 'future_observation_sd' in change:
                        self.assertEqual(scenario.observation_model.value, change['future_observation_sd'])
                    else:
                        self.assertEqual(scenario.observation_model.kind, 'time_profile_sd')

    def test_accepted_numpy_control_scalars_are_normalized_for_json(self):
        configured = settings(use_measurement_uncertainty=np.bool_(True),
            future_observation_sd=np.float32(.17), future_replicates_per_time=np.int64(4),
            ivivc_approach='approach2', ivivc_prediction_sem=np.float32(.001))
        self.assertIs(type(configured.use_measurement_uncertainty), bool)
        self.assertIs(type(configured.future_observation_sd), float)
        self.assertIs(type(configured.ivivc_prediction_sem), float)
        self.assertIs(type(configured.future_replicates_per_time), int)
        json.dumps(asdict(configured), allow_nan=False)

    def test_approach2_calibration_means_and_prediction_sem_reach_reference(self):
        configured = settings(use_measurement_uncertainty=True, source_response_kind='mean',
            reference_replicates_per_time=4, ivivc_approach='approach2',
            ivivc_model='exponential_normalized', ivivc_prediction_sem=.0001)
        for method in ('value_ratio', 'tau_ratio'):
            with self.subTest(method=method):
                scenario = poc.build_ivivc_prediction_scenario(replace(configured, ivivc_method=method))
                metadata = scenario.reference.metadata
                for precision in metadata['calibration_uncertainty'].values():
                    self.assertEqual(precision['covariance_scaling'], 'absolute')
                    supplied = precision['observation_uncertainty']
                    self.assertEqual(supplied['uncertainty_kind'], 'sem')
                    np.testing.assert_allclose(supplied['standard_uncertainty'], .003/2)
                    np.testing.assert_array_equal(supplied['replicate_counts'], np.full(7, 4))
                self.assertEqual(metadata['prediction_input_uncertainty']['uncertainty_kind'], 'sem')
                np.testing.assert_allclose(
                    metadata['prediction_input_uncertainty']['standard_uncertainty'], .0001)
                self.assertFalse(metadata['future_observation_variability_included'])

    def test_approach1_keeps_its_scope_and_rejects_prediction_sem(self):
        with self.assertRaisesRegex(ValueError, 'Approach 2|approach2'):
            settings(ivivc_prediction_sem=.001)
        scenario = poc.build_ivivc_prediction_scenario(settings(use_measurement_uncertainty=True))
        self.assertEqual(scenario.reference.metadata['ivivc_approach'], 'approach1')
        self.assertNotIn('calibration_uncertainty', scenario.reference.metadata)

    def test_inverse_prediction_uncertainty_does_not_invent_zero_baseline_support(self):
        configured = settings(use_measurement_uncertainty=True, ivivc_approach='approach2',
            ivivc_model='exponential_normalized', ivivc_method='observed_response_inverse',
            ivivc_prediction_sem=.001, n_reference_draws=16)
        with self.assertRaisesRegex(ValueError, 'required_times|supported|reference.*generated'):
            poc.build_ivivc_prediction_scenario(configured)


if __name__ == '__main__':
    unittest.main()
