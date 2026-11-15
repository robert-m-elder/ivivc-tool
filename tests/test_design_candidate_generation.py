"""Reviewable candidate generation shares the actual calibration optimizer path."""
from copy import deepcopy
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import (
    DesignConstraints, ObservationModel, fit_reference_model_ensemble,
    optimize_local_i_optimal_designs,
)
from utilities.design_calibration import run_calibration_comparison, run_calibration_generation
from utilities.design_planning import PlanningInputError


def form(**changes):
    data = dict(task='calibration', varied_environment='in_vivo',
        cal_approach='approach2', cal_model='exponential_normalized',
        cal_relationship='on', cal_method='value_ratio', cal_evaluation_times='0,4,8',
        reference_draws='4', random_seed='123', n_simulations='2', cal_interval_draws='4',
        cal_in_vivo_source='file', cal_in_vivo_model='exponential_normalized',
        cal_in_vitro_model='exponential_normalized',
        cal_in_vivo_first_time='0', cal_in_vivo_last_time='16', cal_in_vivo_time_step='2',
        cal_candidate_0_name='Generated', cal_candidate_0_in_vivo_kind='generated',
        cal_candidate_0_in_vivo_number_of_times='3', cal_candidate_0_in_vivo_replicates='3')
    data.update(changes)
    return data


def course(*, scale_time=1., precision=True):
    time = scale_time * np.arange(0., 9., 2.)
    values = np.exp(-.2*time/scale_time) * [1., 1.001, .999, 1.002, .999]
    return TimeCourse(time, values, observation_uncertainty=(
        ResolvedObservationUncertainty(np.full(5, .025), response_kind='mean',
            replicate_counts=np.full(5, 4)) if precision else None))


def generate(data=None, **inputs):
    inputs.setdefault('in_vivo_source', course(scale_time=2.))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        return run_calibration_generation(form() if data is None else data, **inputs)


class CandidateGeneration(unittest.TestCase):
    def test_generation_requires_no_held_fixed_data_or_simulation_and_ignores_target_options(self):
        data = form(cal_model='invalid-unused', cal_method='invalid-unused',
                    cal_interpolation='invalid-unused', cal_relationship='',
                    cal_assess_target='on', cal_evaluation_times='not times',
                    n_simulations='not a count', cal_interval_draws='not a count',
                    cal_in_vitro_model='invalid-unused', cal_in_vitro_source='file')
        with patch('utilities.design_calibration.compare_calibration_designs',
                   side_effect=AssertionError('Preview must not simulate calibration studies')):
            result = generate(data)
        self.assertIsNone(result.comparison)
        self.assertEqual(set(result.references), {'in_vivo'})
        self.assertEqual(set(result.source_fits), {'in_vivo'})
        self.assertEqual(result.generation_outcomes[0]['new_units'], 9)

    def test_preview_matches_full_calculation_and_direct_shared_optimizer(self):
        data = form()
        preview = generate(data)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            compared = run_calibration_comparison(data, existing_in_vitro=course(),
                                                   in_vivo_source=course(scale_time=2.))
        np.testing.assert_array_equal(preview.designs['Generated'].in_vivo_times,
                                      compared.designs['Generated'].in_vivo_times)
        source_fit = preview.source_fits['in_vivo']
        model = fit_reference_model_ensemble(preview.references['in_vivo'],
            model_name=source_fit.model_name, initial_parameters=source_fit.parameter_vector,
            fallback_grid_search=False)
        constraints = DesignConstraints(np.arange(0., 17., 2.),
            min_time=0., max_time=16., min_number_of_times=3, max_number_of_times=3,
            replicates_per_time=3)
        result = optimize_local_i_optimal_designs(model, ObservationModel('constant_sd', .05,
            replicates_per_time=3), constraints, refine=True, max_exchange_iterations=10)
        selected = result.design_for_n(3, refined=True)
        np.testing.assert_array_equal(preview.designs['Generated'].in_vivo_times, selected.times)
        self.assertAlmostEqual(preview.generation_outcomes[0]['information_criterion'],
                               selected.score.criterion)

    def test_additions_keep_historical_precision_and_count_only_new_units(self):
        old = course(scale_time=2.)
        original = deepcopy(old)
        result = generate(form(cal_in_vivo_mode='additions',
                               cal_candidate_0_in_vivo_number_of_times='2'), existing_in_vivo=old)
        row = result.generation_outcomes[0]
        self.assertEqual(len(row['times']), 2)
        self.assertFalse(np.isin(row['times'], old.time).any())
        np.testing.assert_array_equal(row['fixed_times'], old.time)
        np.testing.assert_array_equal(result.analyzed_existing['in_vivo'].observation_uncertainty.sigma,
                                      original.observation_uncertainty.sigma)
        np.testing.assert_array_equal(old.value, original.value)
        self.assertEqual(row['new_units'], 6)
        self.assertEqual(row['replicates'], 3)
        self.assertEqual(row['candidate_index'], 0)
        self.assertEqual(row['reference_time_range'], (0., 16.))

    def test_both_environments_use_their_own_source_and_preserve_active_manual_candidates(self):
        data = form(varied_environment='both', cal_in_vitro_source='file',
            cal_in_vitro_model='exponential_normalized', cal_in_vitro_first_time='0',
            cal_in_vitro_last_time='8', cal_in_vitro_time_step='1',
            cal_candidate_0_in_vitro_kind='generated', cal_candidate_0_in_vitro_number_of_times='3',
            cal_candidate_1_name='Manual', cal_candidate_1_in_vivo_times='0,3,16',
            cal_candidate_1_in_vitro_times='0,1,8')
        result = generate(data, in_vitro_source=course())
        self.assertEqual(set(result.references), {'in_vitro', 'in_vivo'})
        self.assertEqual(len(result.generation_outcomes), 4)
        np.testing.assert_array_equal(result.designs['Manual'].in_vivo_times, [0., 3., 16.])
        np.testing.assert_array_equal(result.designs['Manual'].in_vitro_times, [0., 1., 8.])
        for side in ('in_vitro', 'in_vivo'):
            selected = getattr(result.designs['Generated'], side + '_times')
            self.assertEqual(len(selected), 3)
            self.assertTrue(np.isfinite(selected).all())

    def test_unequal_individual_sd_is_used_by_the_shared_generator(self):
        profile = ObservationModel('time_profile_sd', profile_times=[0., 4., 8., 12., 16.],
                                   profile_sd=[.02, .03, .05, .2, .3])
        uniform = generate()
        unequal = generate(form(cal_in_vivo_observation_kind='time_profile_sd'), in_vivo_profile=profile)
        self.assertFalse(np.array_equal(uniform.designs['Generated'].in_vivo_times,
                                        unequal.designs['Generated'].in_vivo_times))
        self.assertEqual(unequal.designs['Generated'].in_vivo_observation_model.replicates_per_time, 3)
        np.testing.assert_array_equal(unequal.designs['Generated'].in_vivo_observation_model.profile_sd,
                                      profile.profile_sd)

    def test_source_normalization_and_assumed_historical_sd_are_applied_once(self):
        old = course(scale_time=2., precision=False)
        old = TimeCourse(old.time, old.value*100.)
        result = generate(form(cal_in_vivo_mode='additions', cal_in_vivo_source='existing',
            cal_normalization='initial_value', cal_in_vivo_historical_error='future_model',
            cal_candidate_0_in_vivo_number_of_times='2'), existing_in_vivo=old)
        np.testing.assert_allclose(result.analyzed_existing['in_vivo'].value, old.value/100.)
        np.testing.assert_allclose(result.analyzed_existing['in_vivo'].observation_uncertainty.sigma, .05)
        self.assertIsNone(result.analyzed_existing['in_vivo'].observation_uncertainty.replicate_counts)
        self.assertTrue(any('once' in message for message in result.warnings))

    def test_same_seed_reproduces_reference_draws_and_generated_times(self):
        one, two = generate(), generate()
        np.testing.assert_array_equal(one.references['in_vivo'].value_draws,
                                      two.references['in_vivo'].value_draws)
        np.testing.assert_array_equal(one.designs['Generated'].in_vivo_times,
                                      two.designs['Generated'].in_vivo_times)

    def test_insufficient_local_information_is_reported_without_blocking_the_candidate(self):
        result = generate(form(cal_in_vivo_model='exponential',
                               cal_candidate_0_in_vivo_number_of_times='1'))
        row = result.generation_outcomes[0]
        self.assertEqual(len(row['times']), 1)
        self.assertFalse(np.isfinite(row['information_criterion']))
        self.assertTrue(any('Generated (in_vivo)' in message and 'local information is insufficient' in message
                            for message in result.warnings))

    def test_invalid_generation_inputs_link_to_the_relevant_controls(self):
        cases = [
            ({'cal_candidate_0_in_vivo_kind': 'manual', 'cal_candidate_0_in_vivo_times': '0,4,8'},
             {}, 'calibration-candidates'),
            ({'cal_candidate_0_in_vivo_number_of_times': '10'}, {},
             'cal_candidate_0_in_vivo_number_of_times'),
            ({'cal_in_vivo_time_step': '0'}, {}, 'cal_in_vivo_time_step'),
            ({'cal_in_vivo_last_time': '20'}, {}, 'cal_in_vivo_allow_extrapolation'),
            ({'cal_in_vivo_mode': 'additions'}, {'existing_in_vivo': course(scale_time=2., precision=False)},
             'cal_in_vivo_historical_error'),
            ({'cal_in_vivo_observation_kind': 'time_profile_sd'}, {},
             'calibration-in-vivo-sd-profile-file'),
            ({'reference_draws': 'bad'}, {}, 'reference_draws'),
            ({'random_seed': '-1'}, {}, 'random_seed'),
        ]
        for changes, inputs, expected in cases:
            with self.subTest(changes=changes), self.assertRaises(PlanningInputError) as caught:
                generate(form(**changes), **inputs)
            self.assertEqual(caught.exception.field_id, expected)


if __name__ == '__main__':
    unittest.main()
