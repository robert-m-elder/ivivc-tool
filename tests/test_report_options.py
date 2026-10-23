"""Only successful method outputs may be offered for report export."""
from types import SimpleNamespace
import unittest

from utilities.reporting import build_final_model_options


class ReportOptionTests(unittest.TestCase):
    def setUp(self):
        self.models = {'example': {'display_name': 'Example'}, 'other': {'display_name': 'Other'}}
        self.approaches = {key: {'display_name': key} for key in ('approach1', 'approach2', 'approach3')}

    def options(self, results, predictions=None):
        return build_final_model_options(results, self.models, self.approaches,
                                         include_prediction_methods=True,
                                         prediction_results=predictions)

    def test_failed_predictions_retain_fit_only_report(self):
        results = {'approach1:example': {}, 'approach2:example': {}, 'approach3': {}}
        options = self.options(results, {'error': 'Prediction file could not be read'})
        self.assertEqual(len(options), 3)
        self.assertTrue(all(option['method_id'] is None for option in options))
        self.assertTrue(all(not option['prediction_plot_sources'] for option in options))

    def test_partially_successful_methods_match_their_real_source_plots(self):
        tau = [SimpleNamespace(n=5, s=.2), SimpleNamespace(n=10, s=.4)]
        options = self.options({'approach2:example': {'tau': tau}, 'approach2:other': {}},
                               {'approach2': {'example': {'plot': {}, 'plot_tau': {},
                                                        'plot_inverse_response': {}},
                                              'other': {'error': 'Prediction calculation failed'}}})
        self.assertEqual([option['method_id'] for option in options],
                         ['value_ratio', 'tau_ratio', 'observed_response_inverse', None])
        self.assertEqual(options[2]['prediction_plot_sources'][0]['source_id'],
                         'plot-prediction-approach2-example-inverse')
        self.assertEqual(sum(option['is_default'] for option in options), 1)

    def test_unavailable_tau_and_missing_direct_mapping_method_are_omitted(self):
        options = self.options({'approach2:example': {}, 'approach3': {}}, {
            'approach2': {'example': {'plot': {}, 'plot_tau': {}}},
            'approach3': {'method1_value_scaling': {'plot': {}},
                          'method2_time_scaling': {'error': 'Unavailable'}}})
        self.assertEqual([option['method_id'] for option in options], ['value_ratio', 'value_scaling'])
        self.assertEqual(options[1]['prediction_plot_sources'][0]['source_id'],
                         'plot-prediction-approach3-method1-value-scaling')


if __name__ == '__main__':
    unittest.main()
