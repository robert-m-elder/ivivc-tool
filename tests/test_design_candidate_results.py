"""Generated schedules retain optimizer values and describe their assumptions."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse
from study_design_core import CalibrationDesign, ObservationModel
from utilities.design_candidate_results import present_generation_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


def generated_result():
    name = 'Generated <candidate>'
    old = TimeCourse([0., 2.], [1., .7], observation_uncertainty=
                     ResolvedObservationUncertainty([.03, .04], response_kind='mean',
                                                    replicate_counts=[4, 5]))
    design = CalibrationDesign(
        in_vitro_times=[1., 3.], in_vitro_observation_model=ObservationModel('constant_cv', .1, replicates_per_time=2, minimum_sd=.01),
        in_vivo_times=[4., 6.], in_vivo_observation_model=ObservationModel('constant_sd', .05, replicates_per_time=3))
    return SimpleNamespace(
        settings={'random_seed': 123, 'reference_draws': 12},
        comparison=None,
        designs={name: design}, analyzed_existing={'in_vivo': old},
        references={'in_vivo': SimpleNamespace(n_valid_draws=10, n_draws=12)},
        generation_outcomes=[
            dict(candidate=name, candidate_index=2, side='in_vitro', kind='manual', status='available',
                 times=np.array([1., 3.]), replicates=2, fixed_times=np.array([]), new_units=4,
                 reference_time_range=(0., 8.), information_criterion=None),
            dict(candidate=name, candidate_index=2, side='in_vivo', kind='generated', status='available',
                 times=np.array([4., 6.]), replicates=3, fixed_times=np.array([0., 2.]), new_units=6,
                 reference_time_range=(0., 8.), information_criterion=.0012345),
        ],
        source_details={'in_vivo': {
            'mode': 'additions', 'source_kind': 'existing', 'model': 'linear',
            'normalization': {'slope': 2., 'intercept': -1.},
            'observed_time_range': (0., 2.), 'reference_time_range': (0., 8.),
            'generation_grid': {'first_time': 0., 'last_time_bound': 8., 'time_step': 3.,
                                'candidate_times': np.array([0., 3., 6.])},
        }},
        warnings=['Source-model extrapolation is conditional on that model.'],
    )


class CandidatePresentation(unittest.TestCase):
    def test_counts_separate_existing_units_and_new_means(self):
        context = present_generation_result(generated_result())
        candidate, = context['generation_candidates']
        row, = candidate['rows']
        self.assertEqual(candidate['index'], 2)
        self.assertEqual(candidate['total_new_units'], 10)
        self.assertEqual(row['time_count'], 2)
        self.assertEqual(row['replicates'], 3)
        self.assertEqual(row['new_units'], 6)
        self.assertEqual(row['existing_rows'], 2)
        self.assertEqual(row['existing_units'], '9')
        self.assertEqual(row['fixed_times'], '0, 2')
        self.assertEqual(row['observation'], 'Individual SD = 0.05')
        self.assertIn('without another division', row['existing_precision'])

    def test_manual_environment_is_not_misrepresented_as_generated(self):
        row, = present_generation_result(generated_result())['generation_candidates'][0]['rows']
        self.assertEqual(row['side'], 'In vivo')
        self.assertEqual(row['times'], '4, 6')
        self.assertEqual(row['criterion'], '0.0012345')

    def test_grid_reports_actual_last_opportunity_and_reference_domain(self):
        source, = present_generation_result(generated_result())['generation_sources']
        self.assertEqual(source['last_time_bound'], '8')
        self.assertEqual(source['last_opportunity'], '6')
        self.assertEqual(source['opportunity_count'], 3)
        self.assertEqual(source['observed_range'], '0–2')
        self.assertEqual(source['reference_range'], '0–8')
        self.assertEqual(source['slope'], '2')
        self.assertEqual(source['intercept'], '-1')
        self.assertEqual(source['draws'], '10 / 12')

    def test_nonfinite_score_is_unavailable_without_replacing_it_with_zero(self):
        result = generated_result()
        result.generation_outcomes[1]['information_criterion'] = np.inf
        row, = present_generation_result(result)['generation_candidates'][0]['rows']
        self.assertEqual(row['criterion'], '—')

    def test_supplied_individual_sd_is_not_rounded_in_the_assumptions(self):
        result = generated_result()
        design = result.designs['Generated <candidate>']
        result.designs['Generated <candidate>'] = replace(design, in_vivo_observation_model=
            replace(design.in_vivo_observation_model, value=.0512345678901234))
        row, = present_generation_result(result)['generation_candidates'][0]['rows']
        self.assertEqual(row['observation'], 'Individual SD = 0.0512345678901234')

    def test_unknown_historical_counts_stay_unknown(self):
        result = generated_result()
        result.analyzed_existing['in_vivo'] = TimeCourse([0., 2.], [1., .7],
            observation_uncertainty=ResolvedObservationUncertainty([.03, .04], response_kind='mean'))
        row, = present_generation_result(result)['generation_candidates'][0]['rows']
        self.assertEqual(row['existing_units'], 'Not reported for means')
        self.assertEqual(row['new_units'], 6)

    @unittest.skipIf(Environment is None, 'Jinja is not installed')
    def test_actual_template_renders_escaped_input_with_strict_variables(self):
        env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / 'templates'),
                          undefined=StrictUndefined, autoescape=True)
        imported = {'label': '<source>', 'filename': '<source>.csv', 'worksheet': '<sheet>',
                    'summary': {'interpretation': 'SEM is already uncertainty of a mean.',
                                'mapping': {'Response': '<column>'}}}
        context = present_generation_result(generated_result(), source_summaries=[imported])
        html = env.get_template('design/_candidate_generation.html').render(**context)
        self.assertIn('Generated &lt;candidate&gt;', html)
        self.assertIn('&lt;source&gt;.csv', html)
        self.assertIn('&lt;sheet&gt;', html)
        self.assertIn('&lt;column&gt;', html)
        self.assertIn('data-copy-generated-candidate="2"', html)
        self.assertIn('not optimized for IVIVC performance', html)
        self.assertIn('No calibration-study simulations', html)
        self.assertIn('last actual opportunity 6', html)
        self.assertIn('analyzed response = 2 × supplied response + -1', html)
        self.assertNotIn('NaN', html)
        self.assertNotIn('Infinity', html)
        self.assertNotIn('id=', html)

    @unittest.skipIf(Environment is None, 'Jinja is not installed')
    def test_two_generated_sides_share_one_copy_action(self):
        result = generated_result()
        result.generation_outcomes[0]['kind'] = 'generated'
        context = present_generation_result(result)
        candidate, = context['generation_candidates']
        self.assertEqual(len(candidate['rows']), 2)
        env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / 'templates'),
                          undefined=StrictUndefined, autoescape=True)
        html = env.get_template('design/_candidate_generation.html').render(**context)
        self.assertEqual(html.count('data-copy-generated-candidate='), 1)


if __name__ == '__main__':
    unittest.main()
