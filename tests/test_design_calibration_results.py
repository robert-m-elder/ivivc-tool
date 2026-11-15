"""Calibration output presentation using actual small simulated comparisons."""
from copy import copy
from pathlib import Path
import unittest
import warnings

import numpy as np

from ivivc_core import AnalysisError, ResolvedObservationUncertainty, TimeCourse
from study_design_core import IVIVCCalibrationAssessment, ObservationModel, summarize_calibration_simulation
from utilities.design_calibration import run_calibration_comparison
from utilities.design_calibration_results import present_calibration_result, _observation_description, _quantity_labels

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


def form(**changes):
    values = dict(
        task='calibration', varied_environment='in_vivo', cal_approach='approach2',
        cal_model='exponential_normalized', cal_relationship='on', cal_method='value_ratio',
        cal_evaluation_times='1,3', cal_in_vitro_source='existing',
        cal_in_vitro_model='exponential_normalized', cal_in_vivo_mode='prospective',
        cal_in_vivo_source='file', cal_in_vivo_model='exponential_normalized',
        cal_candidate_0_name='One', cal_candidate_0_in_vivo_kind='manual',
        cal_candidate_0_in_vivo_times='0,4,8', cal_candidate_0_in_vivo_replicates='1',
        cal_candidate_1_name='Four', cal_candidate_1_in_vivo_kind='manual',
        cal_candidate_1_in_vivo_times='0,4,8', cal_candidate_1_in_vivo_replicates='4',
        reference_draws='4', n_simulations='3', cal_interval_draws='8', random_seed='3',
        cal_assess_target='on',
    )
    values.update(changes)
    return values


class CalibrationPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        time = np.arange(0., 9., 2.)
        values = np.exp(-.2*time)*[1., 1.001, .999, 1.001, .999]
        cls.existing = TimeCourse(time, values, observation_uncertainty=
                                 ResolvedObservationUncertainty(np.full(5, .02),
                                     response_kind='mean', replicate_counts=np.full(5, 4)))
        cls.source = TimeCourse(2*time, values)
        cls.target = TimeCourse([1., 3.], np.exp(-.2*np.array([1., 3.])))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            cls.result = run_calibration_comparison(
                form(), existing_in_vitro=cls.existing, in_vivo_source=cls.source,
                prediction=cls.target)
            cls.approach1 = run_calibration_comparison(
                form(cal_approach='approach1', cal_model='linear', cal_method='time_mapping'),
                existing_in_vitro=cls.existing, in_vivo_source=cls.source, prediction=cls.target)
            cls.without_truth = run_calibration_comparison(
                form(cal_in_vitro_source='none'), existing_in_vitro=cls.existing,
                in_vivo_source=cls.source, prediction=cls.target)

    def mutable_result(self):
        # Keep registry-backed immutable analyses in place; only clone the
        # result arrays that individual unavailable-result tests modify.
        result = copy(self.result)
        result.settings = dict(self.result.settings)
        result.analyzed_existing = dict(self.result.analyzed_existing)
        result.comparison = copy(self.result.comparison)
        result.comparison.summaries = dict(self.result.comparison.summaries)
        result.comparison.results = {}
        for label, original in self.result.comparison.results.items():
            simulated = copy(original)
            assessments = []
            for item in original.assessments:
                assessment = copy(item)
                for output in ('relationship', 'prediction'):
                    evaluation = getattr(item, output)
                    if evaluation is not None:
                        cloned = copy(evaluation)
                        cloned.lower, cloned.upper = evaluation.lower.copy(), evaluation.upper.copy()
                        setattr(assessment, output, cloned)
                assessments.append(assessment)
            simulated.assessments = tuple(assessments)
            result.comparison.results[label] = simulated
        return result

    def test_candidate_counts_and_width_denominators_match_the_backend(self):
        context = present_calibration_result(self.result)
        self.assertEqual([row['new_units'] for row in context['rows']], [3, 12])
        for row in context['rows']:
            summary = self.result.comparison.summaries[row['name']]
            self.assertEqual(row['n_fit_successful'], summary['n_fit_successful'])
            item = row['outputs']['relationship']
            self.assertEqual(item['width_count'], int(np.sum(summary['relationship']['width_counts'])))
            self.assertEqual(item['error_count'], int(np.sum(summary['relationship']['error_counts'])))
            self.assertAlmostEqual(float(item['width']), summary['relationship']['mean_interval_width'], places=5)
        self.assertEqual(context['existing_rows'][0]['units'], '20')
        self.assertEqual(context['rows'][0]['sides']['in_vitro']['units'], 0)

    def test_precision_only_comparison_does_not_invent_truth_or_zero_error(self):
        item = present_calibration_result(self.without_truth)['rows'][0]['outputs']['relationship']
        self.assertEqual(item['truth_count'], 0)
        self.assertEqual(item['error_count'], 0)
        self.assertEqual(item['rmse'], '—')
        self.assertEqual(item['coverage'], '—')
        self.assertGreater(item['width_count'], 0)

    def test_missing_intervals_keep_full_truth_and_availability_denominators(self):
        result = self.mutable_result()
        simulated = result.comparison.results['One']
        simulated.assessments[0].relationship.lower[:] = np.nan
        result.comparison.summaries['One'] = summarize_calibration_simulation(simulated)
        item = present_calibration_result(result)['rows'][0]['outputs']['relationship']
        self.assertEqual(item['point_count'], 6)
        self.assertEqual(item['interval_count'], 4)
        self.assertEqual(item['width_count'], 4)
        self.assertEqual(item['truth_count'], 6)
        self.assertEqual(item['interval_truth_count'], 4)
        self.assertEqual(item['complete_point_rate'], '100.0%')
        self.assertEqual(item['complete_interval_rate'], '66.7%')
        self.assertFalse(item['example']['points'][0]['interval_available'])

    def test_all_fit_failures_remain_visible_with_unavailable_widths_and_known_truth(self):
        result = self.mutable_result()
        simulated = result.comparison.results['One']
        simulated.assessments = tuple(IVIVCCalibrationAssessment(errors={
            'fit': AnalysisError.from_exception(RuntimeError('Controlled failure'))})
                                     for _ in simulated.assessments)
        result.comparison.summaries['One'] = summarize_calibration_simulation(simulated)
        row = present_calibration_result(result)['rows'][0]
        self.assertEqual(row['n_fit_successful'], 0)
        self.assertEqual(row['fit_failure_rate'], '100.0%')
        self.assertEqual(row['outputs']['relationship']['width'], '—')
        self.assertEqual(row['outputs']['relationship']['rmse'], '—')
        self.assertEqual(row['outputs']['relationship']['coverage'], '0.0%')
        self.assertEqual(row['outputs']['relationship']['conditional_coverage'], '—')
        self.assertIsNone(row['outputs']['relationship']['example'])
        self.assertEqual(row['stages'], [{'stage': 'fit', 'count': 3}])
        self.assertIn('Controlled failure', row['error_examples'][0]['message'])

    def test_approach1_empirical_ranges_remain_separate_from_fit_intervals(self):
        context = present_calibration_result(self.approach1)
        self.assertTrue(context['is_approach1'])
        item = context['rows'][0]['outputs']['relationship']
        self.assertTrue(item['empirical'])
        summary = self.approach1.comparison.summaries['One']['relationship']
        self.assertEqual(item['quantity_label'], 'Predicted in vivo time')
        self.assertEqual(item['coordinate_label'], 'In vitro time')
        self.assertAlmostEqual(float(item['points'][0]['estimate_median']),
                               summary['pointwise_estimate_quantiles'][1, 0], places=4)

    def test_output_labels_distinguish_time_response_and_dimensionless_quantities(self):
        self.assertIn('dimensionless', _quantity_labels('value_ratio', 'relationship', 'value_ratio')[0])
        self.assertIn('dimensionless', _quantity_labels('time_constant_ratio', 'relationship', 'tau_ratio')[0])
        self.assertEqual(_quantity_labels('in_vivo_time', 'prediction', 'observed_response_inverse')[1],
                         'In vivo response level (analyzed scale)')
        self.assertEqual(_quantity_labels('in_vivo_response', 'prediction', 'value_ratio')[0],
                         'Predicted in vivo response (analyzed scale)')

    def test_example_uses_one_existing_simulation_and_does_not_mix_pointwise_results(self):
        result = self.result.comparison.results['One']
        example = present_calibration_result(self.result)['rows'][0]['outputs']['relationship']['example']
        source = result.assessments[example['experiment']-1].relationship
        np.testing.assert_allclose([float(row['estimate']) for row in example['points']],
                                   source.estimate, rtol=1e-4)
        self.assertTrue(all(row['status'] for row in example['points']))

    def test_selected_output_scope_and_target_fixed_transform_remain_explicit(self):
        result = self.mutable_result()
        result.settings['relationship'] = False
        result.settings['target_normalization'] = {'slope': .01, 'intercept': 0.}
        context = present_calibration_result(result)
        self.assertEqual([section['key'] for section in context['output_sections']], ['prediction'])
        self.assertIn('0.01', dict(context['settings_rows'])['Fixed target response transformation'])
        self.assertIn('Calibration interpolation', dict(context['settings_rows']))

    def test_source_previews_and_future_noise_show_the_actual_assumptions(self):
        context = present_calibration_result(self.result)
        self.assertEqual([source['side'] for source in context['source_previews']], ['In vitro', 'In vivo'])
        self.assertEqual(context['source_previews'][0]['mode'], 'fixed')
        self.assertTrue(context['source_previews'][0]['parameters'])
        self.assertIn('0.05', context['rows'][0]['sides']['in_vivo']['observation'])
        self.assertEqual(context['source_previews'][1]['range'], '0–16')

    def test_reported_mean_units_are_not_inferred_from_new_replicates(self):
        result = self.mutable_result()
        old = self.existing
        result.analyzed_existing['in_vitro'] = TimeCourse(old.time, old.value,
            observation_uncertainty=ResolvedObservationUncertainty(np.full(5, .02), response_kind='mean'))
        context = present_calibration_result(result)
        self.assertEqual(context['existing_rows'][0]['units'], 'Not reported for means')

    def test_time_profile_variability_records_its_range_and_lookup_rule(self):
        description = _observation_description(ObservationModel(
            'time_profile_sd', profile_times=[0., 4., 8.], profile_sd=[.02, .05, .08]))
        self.assertIn('3 profile times, 0–8', description)
        self.assertIn('SD 0.02–0.08', description)
        self.assertIn('nearest-time lookup with endpoint extension', description)

    @unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
    def test_template_renders_escaped_provenance_and_distinguishes_calibration_from_target_noise(self):
        context = present_calibration_result(self.result, source_summaries=[{
            'label': 'Expected in vivo source', 'filename': '<script>source</script>.csv',
            'worksheet': 'Historical', 'summary': {'mapping': {'Response': '<img src=x>'}}}])
        environment = Environment(loader=FileSystemLoader(Path(__file__).parents[1]/'templates'),
                                  autoescape=True, undefined=StrictUndefined)
        html = environment.get_template('design/_calibration_results.html').render(**context)
        self.assertIn('Calibration comparison', html)
        self.assertIn('&lt;script&gt;source&lt;/script&gt;', html)
        self.assertIn('&lt;img src=x&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('NaN', html)
        self.assertNotIn('Infinity', html)
        self.assertIn('exponential_normalized', html)
        self.assertIn('exclude scatter of additional future individual in vivo target measurements', html)
        self.assertIn('not an average or a representative-study selection', html)
        self.assertIn('not optimization for IVIVC performance', html)


if __name__ == '__main__':
    unittest.main()
