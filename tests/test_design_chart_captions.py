"""Caption punctuation for fitted curves, observation-only plots, and fallbacks."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    Environment = None


class ChartText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.captions, self.labels = [], []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'figcaption':
            self.current = []
        if 'data-design-plot-target' in attrs:
            self.labels.append(attrs['aria-label'])

    def handle_data(self, text):
        if self.current is not None:
            self.current.append(text)

    def handle_endtag(self, tag):
        if tag == 'figcaption':
            self.captions.append(''.join(self.current))
            self.current = None


@unittest.skipIf(Environment is None, 'Jinja is supplied by the Flask environment.')
class ChartCaptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = Environment(
            loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'),
            autoescape=True, undefined=StrictUndefined,
        )
        cls.macros = cls.environment.get_template('design/_charts.html').module

    @staticmethod
    def plot(*lines):
        return dict(x_ticks=[], y_ticks=[], band='', observations=[(100., 100.)],
                    series=[dict(label=label, color='#24535c', dash=dash,
                                 path='M100,100 L200,150', join=True, circles=[])
                            for label, dash in lines])

    def render(self, title, *lines, interactive=False):
        plot = self.plot(*lines)
        if interactive:
            html = str(self.macros.interactive_chart(
                {'data': [], 'layout': {}}, plot, title,
                'In vitro time', 'Analyzed response', 'source',
            ))
        else:
            html = str(self.macros.chart(
                plot, title, 'In vitro time', 'Analyzed response', 'source',
            ))
        text = ChartText()
        text.feed(html)
        return html, text

    def test_observation_only_target_has_one_caption_sentence(self):
        _, text = self.render('Target in vitro')
        self.assertEqual(text.captions, ['Target in vitro.'])

    def test_one_fitted_series_has_a_single_ending_period(self):
        _, text = self.render('In vitro calibration', ('Fitted source curve', ''))
        self.assertEqual(text.captions, ['In vitro calibration. Fitted source curve.'])

    def test_multiple_series_keep_separators_and_dashed_description(self):
        _, text = self.render('Reference', ('Prior', '6 4'), ('Updated', ''))
        self.assertEqual(text.captions, ['Reference. Prior (dashed); Updated.'])

    def test_interactive_fallback_and_aria_label_do_not_duplicate_title_punctuation(self):
        _, text = self.render('  Target in vitro.  ', interactive=True)
        self.assertEqual(text.captions, ['Target in vitro.'])
        self.assertEqual(text.labels, ['Target in vitro. Analyzed response versus In vitro time.'])

    def test_hostile_titles_and_series_labels_remain_escaped(self):
        html, text = self.render('<script>target</script>', ('<img src=x>', ''), interactive=True)
        self.assertNotIn('<script>target</script>', html)
        self.assertNotIn('<img src=x>', html)
        self.assertEqual(text.captions, ['<script>target</script>. <img src=x>.'])
        self.assertEqual(text.labels, ['<script>target</script>. Analyzed response versus In vitro time.'])

    def test_empty_title_does_not_manufacture_a_period_only_description(self):
        _, text = self.render('  ')
        self.assertEqual(text.captions, [''])


if __name__ == '__main__':
    unittest.main()
