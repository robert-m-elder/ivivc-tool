"""Keep user-facing interval explanations aligned without numerical refitting."""
from html import unescape
from pathlib import Path
import re

import pytest
from docx import Document

from utilities.descriptions import (
    ANALYSIS_PARAMETERS, APPROACH_DESCRIPTIONS, CROSS_VALIDATION_SUMMARY_HTML,
    GOODNESS_VALIDATION_TABLE_HTML, METRIC_DESCRIPTIONS, MODAL_HELP,
    PREPROCESSING_DESCRIPTIONS, RELATIVE_MODEL_EVIDENCE_HTML, RESIDUAL_DIAGNOSTICS_HTML,
)
from utilities.report_content import build_report_sections
from utilities.word_report import build_word_report
from test_report_content import fixture, text
from test_uncertainty_reporting import context as supplied_error_fixture

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]+>', ' ', value))).strip()


@pytest.mark.parametrize('field', ['fit_uncertainty_html', 'prediction_error_html'])
def test_approach1_help_keeps_parameter_only_intervals(field):
    value = plain(APPROACH_DESCRIPTIONS['approach1'][field]).lower()
    assert '95% parameter-uncertainty intervals' in value
    assert 'prediction times and responses are treated as fixed' in value
    assert 'additional scatter of future observations' in value
    assert 'missing interval does not mean zero uncertainty' in value


def test_approach2_fit_help_distinguishes_observation_bars_from_fitted_intervals():
    value = plain(APPROACH_DESCRIPTIONS['approach2']['fit_uncertainty_html']).lower()
    assert 'standard deviation (sd)' in value and 'standard error of the mean (sem)' in value
    assert 'replicate counts are optional metadata' in value
    assert 'without dividing it again' in value
    assert 'smaller supplied errors' in value and 'more influence in fitting' in value
    assert 'with supplied errors' in value and 'normal multiplier' in value
    assert 'without supplied errors' in value and 'residual degrees of freedom' in value
    assert 'observed-data bars' in value and 'one supplied sd or sem' in value
    assert 'shading separately' in value and '95% parameter uncertainty' in value
    assert 'normalization constants are held fixed' in value
    assert 'independent across rows and times' in value
    assert 'future in vivo measurement variability' in value
    assert 'cv fit variability is a separate sensitivity diagnostic' in value


def test_approach2_prediction_help_describes_each_input_contribution():
    value = APPROACH_DESCRIPTIONS['approach2']['prediction_error_html']
    paragraphs = [plain(paragraph).lower() for paragraph in re.findall(r'<p>.*?</p>', value)]
    opening = paragraphs[0]
    assert 'when supplied' in opening and 'prediction responses' in opening
    assert 'without supplied prediction uncertainty' in opening and 'held fixed' in opening
    assert 'independent of calibration measurements' in opening
    ratio = next(paragraph for paragraph in paragraphs if paragraph.startswith('response-value ratio:'))
    assert 'vertical interval combines' in ratio
    assert 'calibration-parameter uncertainty' in ratio and 'prediction-response uncertainty' in ratio
    tau = next(paragraph for paragraph in paragraphs if paragraph.startswith('time-constant ratio:'))
    assert 'horizontal time intervals use calibration uncertainty' in tau
    assert 'shown vertically' in tau and 'does not change the mapped time or its interval' in tau
    inverse = next(paragraph for paragraph in paragraphs if paragraph.startswith('observed-response inverse:'))
    assert 'in vivo calibration uncertainty' in inverse and 'prediction-response uncertainty' in inverse
    assert 'dependence between each sampled response and mapped time' in inverse
    assert 'prediction time and fitted in vitro calibration curve are not used' in inverse
    all_text = ' '.join(paragraphs)
    assert 'separate marginal intervals' in all_text and 'not form a joint 95% region' in all_text
    assert 'independent normal errors' in all_text
    assert 'exclude future in vivo measurement variability' in all_text
    assert 'missing interval does not mean zero uncertainty' in all_text


def test_shared_preprocessing_and_metric_help_explains_supplied_error_cases():
    normalization = PREPROCESSING_DESCRIPTIONS['normalization']
    assert 'treated as fixed' in normalization['initial_value']['description']
    assert 'treated as fixed' in normalization['min_max']['description']
    assert 'unchanged by this reversal' in normalization['one_minus']['description']
    log_response = PREPROCESSING_DESCRIPTIONS['scaling']['log_y']['description']
    assert 'unavailable with supplied SD/SEM' in log_response
    assert 'Without supplied uncertainty' in log_response
    for key in ('rmse', 'r_squared'):
        assert 'unweighted descriptive score' in METRIC_DESCRIPTIONS[key]['description']
    for key in ('aic', 'bic'):
        assert 'normal likelihood' in METRIC_DESCRIPTIONS[key]['description']
        assert 'analyzed scale' in METRIC_DESCRIPTIONS[key]['description']
    assert 'conventional correction' in METRIC_DESCRIPTIONS['aicc']['description']
    evidence = plain(RELATIVE_MODEL_EVIDENCE_HTML).lower()
    assert 'uncertainty assumptions' in evidence
    assert 'source-specific and combined approach 2 evidence remain available' in evidence
    assert 'different from the observation weights' in evidence
    cv = plain(CROSS_VALIDATION_SUMMARY_HTML + GOODNESS_VALIDATION_TABLE_HTML).lower()
    assert 'each approach 2 training fit' in cv
    assert 'omitted observations' in cv and 'sd/sem' in cv
    assert 'does not add uncertainty from the training fit' in cv
    assert 'sensitivity and prediction-error diagnostics' in cv


def test_residual_help_keeps_output_units_and_standardizations_distinct():
    residual = plain(RESIDUAL_DIAGNOSTICS_HTML).lower()
    assert 'fitted output on the analyzed scale' in residual
    assert 'in vivo time for approach 1' in residual and 'response for approach 2' in residual
    assert 'sample standard deviation' in residual
    assert 'does not divide residuals by the supplied sd/sem' in residual
    for key in ('include_raw_residual_plots', 'include_residual_qq_plots'):
        assert 'optional plot' in ANALYSIS_PARAMETERS[key]['description']
        assert 'selected by default' not in ANALYSIS_PARAMETERS[key]['description']
    assert 'start off unselected' in MODAL_HELP['index']['advanced_analysis_overview_html']


@pytest.mark.parametrize('calibration,prediction', [(True, False), (False, True), (True, True)])
def test_supplied_error_reports_keep_their_run_specific_interval_explanation(calibration, prediction):
    context, option = supplied_error_fixture('observed_response_inverse')
    if not calibration:
        context['observation_uncertainty'].pop('in_vitro')
        context['results'][option['model_key']]['fit_metadata'][0]['covariance_scaling'] = 'residual_scaled'
    if not prediction:
        context['observation_uncertainty'].pop('prediction')
        rows = context['prediction_results']['approach2']['example']['prediction_tables']['observed_response_inverse']
        for row in rows:
            for key in list(row):
                if key.startswith(('input_', 'response_', 'value_interval_')):
                    row.pop(key)
            row['interval_kind'] = 'parameter_uncertainty'
    sections = build_report_sections(context, option)
    prose = text(sections).lower()
    assert 'future' in prose and ('excluded' in prose or 'not included' in prose)
    assert 'normalization constants are held fixed' in prose
    if calibration:
        assert 'normal multiplier' in prose
        assert 'one supplied sd/sem' in prose or 'one supplied sd or sem' in prose
    if prediction:
        assert 'independent normal errors' in prose
        assert 'marginal intervals' in prose
        assert 'with the supplied response held fixed' not in prose
    else:
        assert 'prediction inputs held fixed' in prose
    document = Document(build_word_report({'report_label': 'Supplied uncertainty', 'sections': sections}))
    word_prose = '\n'.join(paragraph.text for paragraph in document.paragraphs).lower()
    assert 'future' in word_prose and ('excluded' in word_prose or 'not included' in word_prose)


@pytest.mark.parametrize('approach,method', [
    ('approach1', 'direct_time_correlation'),
    ('approach2', 'value_ratio'),
    ('approach2', 'tau_ratio'),
    ('approach2', 'observed_response_inverse'),
])
def test_browser_and_word_report_preserve_the_same_interval_explanation(approach, method):
    from jinja2 import Environment, FileSystemLoader

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
