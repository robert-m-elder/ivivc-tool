"""Render the production shell to check form ownership and reachable navigation.

These checks complement the JS DOM fixtures; native submission, sticky geometry,
and assistive-technology announcements still need the browser smoke checks.
"""
from html.parser import HTMLParser
from pathlib import Path
import unittest

from utilities.design_uploads import DATASETS
from utilities.design_planning import model_options
from utilities.design_reference_sources import reference_options
from utilities.design_calibration import calibration_options

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class ShellElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.stack = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        self.elements.append((tag, attributes, tuple(self.stack)))
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                       'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append((tag, attributes))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


@unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
class DesignNavigationMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = []
        for prefix in ('', '/tools/design'):
            environment = Environment(
                loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
                autoescape=True, undefined=StrictUndefined,
            )
            routes = dict(review_setup='/review', calculate_comparison='/calculate',
                          inspect_data='/data/inspect', preview_data='/data/preview',
                          generate_candidates='/candidates/generate')
            def url_for(endpoint, filename=None, prefix=prefix):
                return prefix + ('/static/' + filename if endpoint == 'static' else routes[endpoint])
            environment.globals['url_for'] = url_for
            for task in ('study', 'calibration'):
                page = ShellElements()
                page.feed(environment.get_template('design/index.html').render(
                    settings=dict(task=task, reference_source='measured', varied_environment='in_vivo',
                                  n_simulations=2, random_seed=123),
                    datasets=DATASETS, planning_models=model_options(),
                    reference_options=reference_options(), cal_options=calibration_options(),
                ))
                cls.pages.append((prefix, task, page))

    def test_calculate_has_one_form_owner_and_remains_outside_hidden_views(self):
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                forms = [attrs for tag, attrs, _ in page.elements if tag == 'form']
                self.assertEqual(len(forms), 1)
                self.assertEqual(forms[0]['action'], prefix + '/review')
                self.assertEqual(forms[0]['method'], 'post')
                buttons = [(attrs, ancestors) for tag, attrs, ancestors in page.elements
                           if attrs.get('id') == 'calculate-comparison']
                self.assertEqual(len(buttons), 1)
                button, ancestors = buttons[0]
                self.assertEqual(button['form'], forms[0]['id'])
                self.assertEqual(button['type'], 'submit')
                self.assertEqual(button['formaction'], prefix + '/calculate')
                self.assertFalse(any(tag == 'form' for tag, _ in ancestors))
                self.assertIn('design-calculation-bar', [attrs.get('id') for _, attrs in ancestors])
                self.assertFalse(any('hidden' in attrs for _, attrs in ancestors))
                self.assertFalse({'design-inputs', 'design-results'} &
                                 {attrs.get('id') for _, attrs in ancestors})

    def test_section_buttons_reach_unique_focusable_headings_for_the_selected_task(self):
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                by_id = {attrs['id']: (tag, attrs) for tag, attrs, _ in page.elements if 'id' in attrs}
                self.assertEqual(len(by_id), sum('id' in attrs for _, attrs, _ in page.elements))
                visible = []
                for tag, attrs, _ in page.elements:
                    if 'data-design-section' not in attrs:
                        continue
                    self.assertEqual(tag, 'button')
                    self.assertEqual(attrs['type'], 'button')
                    target_tag, target = by_id[attrs['data-design-section']]
                    self.assertIn(target_tag, ('h2', 'h3'))
                    self.assertEqual(target['tabindex'], '-1')
                    if 'hidden' not in attrs:
                        visible.append(attrs['data-section-task'])
                self.assertEqual(visible.count(task), 4)
                self.assertEqual(visible.count('all'), 1)
                self.assertEqual(len(visible), 5)
                for _, attrs, _ in page.elements:
                    if 'data-design-view' in attrs:
                        self.assertIn(attrs['aria-controls'], by_id)

    def test_feedback_is_outside_the_busy_form_and_hidden_views(self):
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                for identifier in ('design-comparison-status', 'design-operation-feedback',
                                   'design-request-status'):
                    _, attrs, ancestors = next(row for row in page.elements if row[1].get('id') == identifier)
                    self.assertEqual(attrs['role'], 'status')
                    self.assertEqual(attrs['aria-live'], 'polite')
                    self.assertFalse(any(tag == 'form' for tag, _ in ancestors))
                    self.assertFalse(any('hidden' in attrs for _, attrs in ancestors))
                    self.assertIn('design-calculation-bar', [item.get('id') for _, item in ancestors])


    def test_sampling_options_are_collapsed_in_order_without_disabling_their_values(self):
        expected = ['sampling-alternatives', 'timing-requirements', 'sampling-time-options']
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                sections = [(tag, attrs) for tag, attrs, _ in page.elements
                            if attrs.get('id') in expected]
                self.assertEqual([attrs['id'] for _, attrs in sections], expected)
                self.assertTrue(all(tag == 'details' and 'open' not in attrs and 'hidden' not in attrs
                                    for tag, attrs in sections))
                for identifier, value in [('fixed_times', '0'), ('baseline_time', '0'), ('excluded_times', '')]:
                    _, attrs, ancestors = next(row for row in page.elements if row[1].get('id') == identifier)
                    self.assertEqual(attrs['value'], value)
                    self.assertNotIn('disabled', attrs)
                    self.assertIn('sampling-time-options', [item.get('id') for _, item in ancestors])
                _, even, ancestors = next(row for row in page.elements if row[1].get('id') == 'include_even')
                self.assertIn('checked', even)
                self.assertNotIn('disabled', even)
                self.assertIn('sampling-alternatives', [item.get('id') for _, item in ancestors])
                _, policy, ancestors = next(row for row in page.elements if row[1].get('id') == 'future_policy')
                self.assertIn('sampling-time-options', [item.get('id') for _, item in ancestors])

    def test_latest_completed_is_available_without_changing_the_default_grid_mode(self):
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                options = [attrs for tag, attrs, ancestors in page.elements if tag == 'option'
                           and any(item.get('id') == 'first_time_mode' for _, item in ancestors)]
                self.assertEqual({attrs['value'] for attrs in options}, {'specified', 'latest_completed'})
                self.assertEqual([attrs['value'] for attrs in options if 'selected' in attrs], ['specified'])

    def test_each_preview_toggle_controls_its_own_initially_hidden_preview(self):
        for prefix, task, page in self.pages:
            with self.subTest(prefix=prefix, task=task):
                by_id = {attrs['id']: (tag, attrs, ancestors) for tag, attrs, ancestors in page.elements if 'id' in attrs}
                toggles = [row for row in page.elements if 'data-import-preview-toggle' in row[1]]
                self.assertTrue(toggles)
                for tag, attrs, ancestors in toggles:
                    self.assertEqual(tag, 'button')
                    self.assertEqual(attrs['type'], 'button')
                    self.assertEqual(attrs['aria-expanded'], 'false')
                    self.assertIn('hidden', attrs)
                    _, target, target_ancestors = by_id[attrs['aria-controls']]
                    self.assertIn('data-import-preview', target)
                    self.assertIn('hidden', target)
                    dataset = next(item['data-design-dataset'] for _, item in ancestors if 'data-design-dataset' in item)
                    target_dataset = next(item['data-design-dataset'] for _, item in target_ancestors if 'data-design-dataset' in item)
                    self.assertEqual(dataset, target_dataset)
                for _, attrs, ancestors in page.elements:
                    if 'data-import-stale' in attrs:
                        self.assertFalse(any('data-import-preview' in item for _, item in ancestors))


if __name__ == '__main__':
    unittest.main()
