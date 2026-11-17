"""Production-render checks for the consolidated timing comparison layout.

The base result comes from a small real planning calculation. Exception cases
replace only retained presentation outcomes, rather than running another search.
"""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import unittest

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course
from utilities.design_planning import run_measured_planning
from utilities.design_planning_results import present_planning_result

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class Element:
    def __init__(self, tag, attributes=None, parent=None):
        self.tag = tag
        self.attributes = attributes or {}
        self.parent = parent
        self.parts = []

    def descendants(self):
        for part in self.parts:
            if isinstance(part, Element):
                yield part
                yield from part.descendants()

    @property
    def text(self):
        if self.tag in ('script', 'style'):
            return ''
        return ' '.join(' '.join(part.text if isinstance(part, Element) else part
                                 for part in self.parts).split())

    def ancestors(self):
        parent = self.parent
        while parent is not None:
            yield parent
            parent = parent.parent


class RenderedPage(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.root = Element('root')
        self.stack = [self.root]
        self.feed(markup)
        self.elements = list(self.root.descendants())

    def handle_starttag(self, tag, attributes):
        element = Element(tag, dict(attributes), self.stack[-1])
        self.stack[-1].parts.append(element)
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                       'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, text):
        self.stack[-1].parts.append(text)

    def by_id(self, identifier):
        found = [element for element in self.elements
                 if element.attributes.get('id') == identifier]
        if len(found) != 1:
            raise AssertionError(f'Expected one {identifier!r}, found {len(found)}.')
        return found[0]


@unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
class TimingComparisonLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        form = {
            'model_name': 'linear', 'first_time': '0', 'last_time': '4', 'time_step': '1',
            'min_times': '3', 'max_times': '5', 'fixed_times': '0',
            'reference_draws': '4', 'n_simulations': '3', 'random_seed': '2',
            'replicates_per_time': '2', 'include_even': 'on',
            'manual_schedules': 'Late comparison: 0, 3, 4',
        }
        source = TimeCourse(np.arange(5.), np.array([1., .87, .67, .56, .39]))
        cls.ordinary = run_measured_planning(source, form)
        cls.timed = run_measured_planning(source, dict(form, **{
            'timing_0_kind': 'window', 'timing_0_label': 'Early coverage',
            'timing_0_lower_mode': 'numeric', 'timing_0_lower_time': '0',
            'timing_0_upper_mode': 'numeric', 'timing_0_upper_time': '2',
            'timing_0_minimum_count': '1', 'timing_0_count_scope': 'new',
        }))
        completed, _ = prepare_time_course(
            [0., 1., 2.], [1., .87, .67],
            observation_uncertainty=ObservationUncertainty(.03, response_kind='individual'),
        )
        cls.adaptive = run_measured_planning(None, dict(form, **{
            'has_completed': 'on', 'min_times': '4',
            'manual_schedules': 'Late adaptive comparison: 0, 1, 2, 4',
            'timing_0_kind': 'window', 'timing_0_label': 'New coverage',
            'timing_0_lower_mode': 'numeric', 'timing_0_lower_time': '2',
            'timing_0_upper_mode': 'numeric', 'timing_0_upper_time': '3',
            'timing_0_minimum_count': '1', 'timing_0_count_scope': 'new',
        }), completed=completed)
        cls.environment = Environment(
            loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
            autoescape=True, undefined=StrictUndefined,
        )

    def render(self, result=None):
        context = present_planning_result(result or self.timed)
        markup = self.environment.get_template('design/_planning_results.html').render(**context)
        return context, RenderedPage(markup), markup

    def test_overview_precedes_results_and_shared_diagnostics_start_collapsed(self):
        _, page, _ = self.render()
        overview = page.by_id('planning-timing-summary')
        details = page.by_id('planning-timing-details')
        reference = page.by_id('planning-reference-result-heading')
        performance = page.by_id('planning-performance-heading')
        self.assertLess(page.elements.index(overview), page.elements.index(reference))
        self.assertLess(page.elements.index(overview), page.elements.index(performance))
        self.assertFalse(any(node.tag == 'details' for node in overview.ancestors()))
        self.assertEqual(details.tag, 'details')
        self.assertNotIn('open', details.attributes)
        self.assertIn('Early coverage', details.text)
        definitions = [node for node in page.elements if node.tag == 'table' and
                       {'Rule', 'Bounds', 'Required coverage', 'Resolution'} <=
                       {header.text for header in node.descendants() if header.tag == 'th'
                        and header.attributes.get('scope') == 'col'}]
        self.assertEqual(len(definitions), 1)
        self.assertIn(details, list(definitions[0].ancestors()))
        captions = [node.text for node in page.elements if node.tag == 'caption']
        self.assertFalse(any('Latest-before selections for' in value for value in captions))

    def test_compact_performance_keeps_metrics_and_controls_but_moves_time_lists(self):
        context, page, _ = self.render()
        rows = [node for node in page.elements if 'data-schedule-row' in node.attributes]
        self.assertEqual([row.attributes['data-schedule-row'] for row in rows],
                         [row['id'] for row in context['rows']])
        for row, expected in zip(rows, context['rows']):
            controls = [node for node in row.descendants() if 'data-schedule-select' in node.attributes]
            self.assertEqual(len(controls), 1)
            self.assertEqual(controls[0].attributes['data-schedule-select'], expected['id'])
            self.assertIn('disabled', controls[0].attributes)
            self.assertNotIn('name', controls[0].attributes)
            self.assertNotIn(expected['times'], row.text)
            values = [node.text for node in row.parts if isinstance(node, Element) and node.tag == 'td']
            for value in (expected['failure_rate'], expected['mean_irmse'], expected['p90_irmse'],
                          f"{expected['n_successful']} / {expected['n_simulations']}"):
                self.assertIn(value, values)
            self.assertIn(f"{expected['mean_curve_interval_width']} ({expected['interval_count']} fits)", values)
            if expected.get('timing_flag'):
                self.assertIn(expected['timing_flag']['label'], row.text)
        table = next(node for node in rows[0].ancestors() if node.tag == 'table')
        headers = [node.text for node in table.descendants() if node.tag == 'th' and node.attributes.get('scope') == 'col']
        self.assertNotIn('Sampling times', headers)
        self.assertNotIn('New sampling times', headers)
        self.assertNotIn('Timing inspection', headers)
        manual = next(row for row in context['rows'] if row['name'] == 'Manual · Late comparison')
        self.assertTrue(manual.get('timing_flag'), 'A comparator that misses the window must remain visibly flagged.')

    def test_selected_details_consolidate_compliance_and_remain_accessible_without_inspection_js(self):
        context, page, _ = self.render()
        details = next(node for node in page.elements if 'data-schedule-details' in node.attributes)
        self.assertEqual(details.tag, 'details')
        self.assertNotIn('open', details.attributes)
        self.assertNotIn('hidden', details.attributes)
        sections = [node for node in details.descendants() if 'data-schedule-detail' in node.attributes]
        self.assertEqual([node.attributes['data-schedule-detail'] for node in sections],
                         [row['id'] for row in context['rows']])
        for section, expected in zip(sections, context['inspection']['schedule_data']['schedules']):
            self.assertNotIn('hidden', section.attributes)
            self.assertEqual(sum('data-schedule-timing' in node.attributes for node in section.descendants()), 1)
            coordinates = []
            for node in section.descendants():
                if node.tag == 'th':
                    try:
                        coordinates.append(float(node.text))
                    except ValueError:
                        pass
            for point in expected['points']:
                # Exact coordinates stay in the per-time table, independently
                # of JavaScript-driven selection or Plotly availability.
                self.assertIn(point['time'], coordinates)

    def test_compact_adaptive_rows_keep_new_time_and_unit_counts(self):
        context, page, _ = self.render(self.adaptive)
        self.assertTrue(context['adaptive'])
        rows = [node for node in page.elements if 'data-schedule-row' in node.attributes]
        self.assertEqual(len(rows), len(context['rows']))
        table = next(node for node in rows[0].ancestors() if node.tag == 'table')
        headers = [node.text for node in table.descendants() if node.tag == 'th' and node.attributes.get('scope') == 'col']
        self.assertIn('New times', headers)
        self.assertIn('New independent units', headers)
        self.assertNotIn('Sampling times', headers)
        self.assertNotIn('New sampling times', headers)
        for row, expected in zip(rows, context['rows']):
            values = [node.text for node in row.parts if isinstance(node, Element) and node.tag == 'td']
            self.assertIn(str(expected['number_of_new_times']), values)
            self.assertIn(str(expected['experimental_units']), values)

    def test_missing_schedule_outcomes_are_summarized_without_requiring_a_selection(self):
        result = deepcopy(self.timed)
        result.comparison, result.optimization, result.schedules = None, None, {}
        template = result.timing_details['outcomes'][0]
        result.timing_details['outcomes'] = [dict(template, number_of_times=size, status=status,
                                               message=message, times=(), anchors=())
                                           for size, status, message in (
                                               (3, 'infeasible', 'Requested rules cannot fit this study size.'),
                                               (4, 'search_limit', 'Search budget ended without establishing availability.'),
                                               (5, 'unavailable', 'The requested milestone is beyond the horizon.'),
                                           )]
        result.timing_details['requirements'][0].update(
            status='beyond_horizon', message='The requested milestone is beyond the horizon.')
        context, page, _ = self.render(result)
        self.assertFalse(context['has_comparison'])
        self.assertFalse(any('data-schedule-select' in node.attributes for node in page.elements))
        overview = page.by_id('planning-timing-summary')
        words = overview.text.lower()
        self.assertIn('infeasible', words)
        self.assertIn('search limit', words)
        self.assertTrue(any(word in words for word in ('unavailable', 'unresolved', 'beyond')))
        self.assertFalse(any(node.tag == 'details' for node in overview.ancestors()))
        diagnostics = page.by_id('planning-timing-details')
        self.assertNotIn('open', diagnostics.attributes)
        self.assertIn('Early coverage', diagnostics.text)

    def test_comparisons_without_timing_rules_do_not_get_empty_timing_sections(self):
        context, page, _ = self.render(self.ordinary)
        identifiers = {node.attributes.get('id') for node in page.elements}
        self.assertNotIn('planning-timing-summary', identifiers)
        self.assertNotIn('planning-timing-details', identifiers)
        self.assertFalse(any(row.get('timing_flag') for row in context['rows']))
        self.assertFalse(any('data-schedule-timing' in node.attributes for node in page.elements))


if __name__ == '__main__':
    unittest.main()
