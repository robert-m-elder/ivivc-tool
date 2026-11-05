"""POC reporting keeps source uncertainty separate from future measurement SD."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from ivivc_core import ObservationUncertainty
from study_design_core import DesignConstraints, ExistingObservations, ObservationModel, reference_from_time_course
from examples.study_design_uncertainty_reporting import (
    future_variability_table, uncertainty_metadata, uncertainty_summary_html,
)


def definition(model=None, completed=None):
    return SimpleNamespace(
        reference=reference_from_time_course([0., 2., 4.], [1., .6, .2]),
        observation_model=model or ObservationModel('constant_sd', .3, 9),
        existing_observations=completed,
        constraints=DesignConstraints(candidate_times=[0., 1., 2., 3., 4.],
            fixed_times=[0., 4.], replicates_per_time=9),
    )


class UncertaintyReportingTests(unittest.TestCase):
    def test_future_table_uses_individual_sd_and_future_n_once(self):
        completed = ExistingObservations([0., 0.], [1., 1.],
            observation_uncertainty=ObservationUncertainty(.04, response_kind='mean', replicate_counts=25))
        table = future_variability_table(definition(completed=completed))
        np.testing.assert_array_equal(table['time'], [1., 2., 3., 4.])
        np.testing.assert_allclose(table['individual_sd'], .3)
        np.testing.assert_allclose(table['future_mean_sem'], .1)
        self.assertEqual(table.iloc[-1]['time_role'], 'fixed_future')
        self.assertTrue((table.iloc[:-1]['time_role'] == 'candidate').all())

    def test_profile_metadata_and_lookup_are_explicit_and_json_safe(self):
        model = ObservationModel('time_profile_sd', replicates_per_time=9,
            profile_times=[1., 3.], profile_sd=[.1, .4])
        item = definition(model)
        record = uncertainty_metadata(item)
        self.assertEqual(record['future']['profile_times'], [1., 3.])
        self.assertEqual(record['future']['profile_rule'], 'nearest_time_with_endpoint_extension')
        self.assertFalse(record['future_measurement_variability_in_reference_envelope'])
        json.dumps(record, allow_nan=False)
        np.testing.assert_array_equal(future_variability_table(item)['individual_sd'], [.1, .1, .1, .4, .4])
        html = uncertainty_summary_html(item)
        self.assertIn('nearest time', html)
        self.assertIn('endpoint SDs', html)
        self.assertIn('Supplied individual SD', html)

    def test_mean_sem_completed_precision_and_fixed_source_are_labeled(self):
        completed = ExistingObservations([0.], [1.],
            observation_uncertainty=ObservationUncertainty(.04, response_kind='mean', replicate_counts=25))
        item = definition(completed=completed)
        item.reference.metadata.update(source_observation_uncertainty=completed.observation_uncertainty.to_metadata(),
                                       source_parameter_uncertainty_propagated=False)
        html = uncertainty_summary_html(item)
        self.assertIn('SEM for reported means: 0.04', html)
        self.assertIn('reported n=25', html)
        self.assertIn('source curve is fixed', html)
        self.assertIn('count is not applied again', html)
        self.assertIn('not an individual-measurement prediction band', html)
        record = uncertainty_metadata(item)
        record['reference']['source_observation_uncertainty']['standard_uncertainty'][0] = 999.
        self.assertEqual(item.reference.metadata['source_observation_uncertainty']['standard_uncertainty'], [.04])

    def test_legacy_and_calibration_sources_keep_their_distinct_meanings(self):
        item = definition()
        self.assertIn('existing uncertainty scope', uncertainty_summary_html(item))
        precision = ExistingObservations([0.], [1.], observation_uncertainty=
            ObservationUncertainty(.06, response_kind='individual')).observation_uncertainty.to_metadata()
        item.reference.metadata['calibration_uncertainty'] = {'in_vitro': {'observation_uncertainty': precision}}
        self.assertIn('In vitro calibration — SD for individual values: 0.06', uncertainty_summary_html(item))
        item.existing_observations = ExistingObservations([0.], [1.])
        self.assertIn('individual SD is assumed', uncertainty_summary_html(item))
        item.existing_observations = ExistingObservations([0.], [1.],
            observation_uncertainty=ObservationUncertainty(.05, response_kind='individual',
                source='historical_nominal_sd_assumption'))
        self.assertIn('nominal historical SD assumption', uncertainty_summary_html(item))
        self.assertEqual(uncertainty_metadata(item)['completed']['precision_source'],
                         'historical_nominal_sd_assumption')

    def test_completed_only_study_has_an_empty_future_table(self):
        item = definition(completed=ExistingObservations([0., 1., 2., 3., 4.], [1., .8, .6, .4, .2]))
        table = future_variability_table(item)
        self.assertTrue(table.empty)
        self.assertEqual(list(table), ['time', 'individual_sd', 'replicates_per_time', 'future_mean_sem', 'time_role'])

    def test_poc_exports_keep_profile_and_completed_sem_separate(self):
        from examples import study_design_proof_of_concept as poc

        settings = poc.ProofOfConceptSettings(
            analysis_model='exponential', n_reference_draws=4, n_simulations=2,
            max_design_sizes=1, refine=False, write_html=False,
            use_measurement_uncertainty=True, source_response_kind='mean',
            completed_replicates_per_time=4, future_replicates_per_time=5,
            future_sd_profile=((0., .03), (20., .08), (52., .12)),
        )
        run = poc.run_proof_of_concept(('adaptive_in_vivo_only',), settings=settings)
        with TemporaryDirectory() as directory:
            poc.write_proof_of_concept_outputs(run, directory)
            root = Path(directory)
            metadata = json.loads((root / 'adaptive_in_vivo_only/metadata.json').read_text())
            manifest = json.loads((root / 'manifest.json').read_text())
            future = pd.read_csv(root / 'adaptive_in_vivo_only/future_measurement_variability.csv')
        self.assertIsNone(metadata['observation_model']['value'])
        self.assertEqual(metadata['observation_model']['profile_sd'], [.03, .08, .12])
        precision = metadata['uncertainty']['completed']['observation_uncertainty']
        self.assertEqual(precision['uncertainty_kind'], 'sem')
        self.assertEqual(precision['replicate_counts'], [4, 4, 4])
        self.assertEqual(manifest['settings']['future_replicates_per_time'], 5)
        self.assertIn('adaptive_in_vivo_only', manifest['files']['uncertainty_by_scenario'])
        self.assertFalse(np.isin(future['time'], metadata['existing_observations']['times']).any())
        np.testing.assert_allclose(future['future_mean_sem'], future['individual_sd']/np.sqrt(5))


if __name__ == '__main__':
    unittest.main()
