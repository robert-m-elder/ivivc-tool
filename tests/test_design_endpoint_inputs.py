"""Focused validation of endpoint selections without running simulations."""
import unittest

from utilities.design_planning import PlanningInputError, _settings


class RecoveryEndpointInputs(unittest.TestCase):
    def endpoints(self, **fields):
        return _settings(fields)['recovery_endpoints']

    def test_defaults_and_explicit_empty_selection_are_distinct(self):
        self.assertEqual([row['value'] for row in self.endpoints()], [50., 90.])
        self.assertEqual(self.endpoints(recovery_endpoints_present='1'), [])

    def test_custom_and_absolute_values_preserve_their_definitions(self):
        result = self.endpoints(
            recovery_endpoints_present='1', recovery_t50='on',
            recovery_tx_custom='50, 25, 25', recovery_absolute_thresholds='0.5, -2',
            recovery_absolute_direction='increasing')
        self.assertEqual([row['value'] for row in result], [50., 25., .5, -2.])
        self.assertEqual([row['kind'] for row in result],
                         ['relative_decrease'] * 2 + ['absolute_threshold'] * 2)
        self.assertEqual(result[2]['direction'], 'increasing')
        self.assertEqual(result[0]['label'], 'T50')

    def test_invalid_values_are_linked_to_the_relevant_input(self):
        for field, value in [('recovery_tx_custom', '0'), ('recovery_tx_custom', '100'),
                             ('recovery_tx_custom', 'NaN'), ('recovery_tx_custom', '25,'),
                             ('recovery_absolute_thresholds', 'infinity'),
                             ('recovery_absolute_thresholds', ',1')]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(PlanningInputError) as caught:
                    self.endpoints(**{field: value})
                self.assertEqual(caught.exception.field_id, field)
        with self.assertRaises(PlanningInputError) as caught:
            self.endpoints(recovery_absolute_thresholds='.5',
                           recovery_absolute_direction='auto')
        self.assertEqual(caught.exception.field_id, 'recovery_absolute_direction')

    def test_limit_counts_unique_endpoints_and_ignores_unused_direction(self):
        self.assertEqual(len(self.endpoints(recovery_tx_custom=','.join(['50'] * 25))), 2)
        with self.assertRaises(PlanningInputError) as caught:
            self.endpoints(recovery_tx_custom=','.join(str(value) for value in range(1, 22)))
        self.assertEqual(caught.exception.field_id, 'recovery_tx_custom')
        self.assertEqual(len(self.endpoints(recovery_absolute_direction='unused')), 2)


if __name__ == '__main__':
    unittest.main()
