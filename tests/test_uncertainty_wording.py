"""Keep user-facing interval explanations aligned without numerical refitting."""
from pathlib import Path

import pytest
from docx import Document
from jinja2 import Environment, FileSystemLoader

from utilities.descriptions import APPROACH_DESCRIPTIONS
from utilities.report_content import build_report_sections
from utilities.word_report import build_word_report
from test_report_content import fixture, text

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('approach', ['approach1', 'approach2'])
@pytest.mark.parametrize('field', ['fit_uncertainty_html', 'prediction_error_html'])
def test_help_distinguishes_parameter_uncertainty_from_observation_scatter(approach, field):
    value = APPROACH_DESCRIPTIONS[approach][field]
    assert '95% parameter-uncertainty intervals' in value
    assert 'not observation prediction bands' in value
    assert 'additional scatter of future observations' in value
    assert 'middle 95%' in value
    assert 'A missing interval does not mean zero uncertainty' in value
    assert 'unchanged' not in value
    assert 'retain their existing' not in value


@pytest.mark.parametrize('approach,method', [
    ('approach1', 'direct_time_correlation'),
    ('approach2', 'value_ratio'),
    ('approach2', 'tau_ratio'),
    ('approach2', 'observed_response_inverse'),
])
def test_browser_and_word_report_preserve_the_same_interval_explanation(approach, method):
    context, option = fixture(approach=approach, method=method)
    sections = build_report_sections(context, option)
    explanation = next(item['text'] for item in sections if item.get('text', '').startswith('Shading and error bars'))
    assert 'not prediction intervals for future observations' in explanation
    assert 'half-width is the amount added to or subtracted from the estimate' in explanation
    assert '1,000 parameter draws and uncertainty seed 12345' in text(sections)
    assert '2.5th and 97.5th percentiles' in text(sections)

    environment = Environment(loader=FileSystemLoader(ROOT / 'templates'), autoescape=True)
    html = environment.get_template('report_sections.html').render(
        final_report_sections={option['id']: sections}, option=option,
    )
    assert explanation in html
    # Use the actual Word renderer; no Flask or numerical uncertainty stubs.
    document = Document(build_word_report({'report_label': 'Interval check', 'sections': sections}))
    assert explanation in [p.text for p in document.paragraphs]


def test_tutorials_keep_parameter_tables_and_repeated_study_bands_distinct():
    comparison = (ROOT / 'docs/appendices/model_comparison_tutorial.tex').read_text()
    sampling = (ROOT / 'docs/appendices/sampling_design.tex').read_text()
    assert 'half-width of an approximate 95\\% confidence interval' in comparison
    assert 'SE / absolute estimate' in comparison
    assert 'curve figures show fitted lines without uncertainty bands' in comparison
    assert 'middle 90\\%' in sampling
    assert "not the app's 95\\%" in sampling
    assert "the app's parameter tables" in sampling
    assert 'same sampling principles apply to other model forms' in sampling
    assert 'stretched' not in sampling.lower()
