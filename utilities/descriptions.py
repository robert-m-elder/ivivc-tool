"""Reusable explanatory text for IVIVC approaches, preprocessing, metrics, and reports.

Routine wording edits should be made in utilities/descriptions.yaml. This module
loads that YAML source and exposes Python constants and formatting helpers used
by Flask templates, reports, and generated documentation.
"""

import re
from pathlib import Path

import yaml


_DESCRIPTION_PATH = Path(__file__).with_suffix('.yaml')


def _load_yaml_descriptions():
    """Load the human-editable description source file."""
    if not _DESCRIPTION_PATH.exists():
        raise FileNotFoundError(f"Description source not found: {_DESCRIPTION_PATH}")

    with _DESCRIPTION_PATH.open('r', encoding='utf-8') as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Description source must contain a YAML mapping: {_DESCRIPTION_PATH}")
    return content


def _required(content, key):
    """Return a required top-level YAML section."""
    value = content.get(key)
    if value in (None, '', [], {}):
        raise KeyError(f"Missing required description section: {key}")
    return value


_DESCRIPTION_CONTENT = _load_yaml_descriptions()

_tool_content = _required(_DESCRIPTION_CONTENT, 'tool')
TOOL_PURPOSE_HTML = _required(_tool_content, 'purpose_html')
TOOL_PURPOSE_TEXT = _required(_tool_content, 'purpose_text')
TOOL_CAPABILITY_SUMMARY = _required(_tool_content, 'capability_summary')

APPROACH_DESCRIPTIONS = _required(_DESCRIPTION_CONTENT, 'approaches')
PREPROCESSING_DESCRIPTIONS = _required(_DESCRIPTION_CONTENT, 'preprocessing')
METRIC_DESCRIPTIONS = _required(_DESCRIPTION_CONTENT, 'metrics')
RELATIVE_MODEL_EVIDENCE_HTML = _required(_DESCRIPTION_CONTENT, 'relative_model_evidence_html')
RELATIVE_MODEL_EVIDENCE_NOTE_HTML = RELATIVE_MODEL_EVIDENCE_HTML
CROSS_VALIDATION_SUMMARY_HTML = _required(_DESCRIPTION_CONTENT, 'cross_validation_summary_html')
GOODNESS_VALIDATION_TABLE_HTML = _required(_DESCRIPTION_CONTENT, 'goodness_validation_table_html')
RESIDUAL_DIAGNOSTICS_HTML = _required(_DESCRIPTION_CONTENT, 'residual_diagnostics_html')
PARAMETER_DIAGNOSTICS_COMPARISON_HTML = _required(_DESCRIPTION_CONTENT, 'parameter_diagnostics_comparison_html')
GRID_SEARCH_DESCRIPTION_HTML = _required(_DESCRIPTION_CONTENT, 'grid_search_description_html')
CV_SCHEMES = _required(_DESCRIPTION_CONTENT, 'cv_schemes')
ANALYSIS_PARAMETERS = _required(_DESCRIPTION_CONTENT, 'analysis_parameters')
MODAL_HELP = _required(_DESCRIPTION_CONTENT, 'modal_help')


def _first_paragraph_html(value):
    """Return the first HTML paragraph from reusable explanatory text."""
    match = re.search(r'<p\b[^>]*>.*?</p>', str(value or ''), flags=re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else str(value or '')


PREPROCESSING_OVERVIEW_SUMMARY_HTML = _first_paragraph_html(
    MODAL_HELP['index']['preprocessing_overview_html']
)
CROSS_VALIDATION_OVERVIEW_SUMMARY_HTML = _first_paragraph_html(
    MODAL_HELP['index']['cross_validation_html']
)

CV_SCHEME_LABELS = {
    key: value.get('display_name', key)
    for key, value in CV_SCHEMES.items()
}
CV_SCHEME_DESCRIPTIONS = {
    key: value.get('description', '')
    for key, value in CV_SCHEMES.items()
}
ANALYSIS_CONFIG_LABELS = {
    key: value.get('display_name', key)
    for key, value in ANALYSIS_PARAMETERS.items()
}
ANALYSIS_PARAMETER_DESCRIPTIONS = {
    key: value.get('description', '')
    for key, value in ANALYSIS_PARAMETERS.items()
}


def _preprocessing_label(category, key):
    return PREPROCESSING_DESCRIPTIONS.get(category, {}).get(key, {}).get('display_name', key)


def _preprocessing_description(category, key):
    return PREPROCESSING_DESCRIPTIONS.get(category, {}).get(key, {}).get('description', '')


def format_preprocessing_rows(selected_normalizations=None, selected_scalings=None, selected_interpolation=None):
    """Return display-ready preprocessing rows for templates and reports."""
    selections = [
        ('Normalization', 'normalization', selected_normalizations or []),
        ('Scaling', 'scaling', selected_scalings or []),
        ('Interpolation', 'interpolation', selected_interpolation or []),
    ]
    rows = []
    for category_label, category_key, values in selections:
        if values:
            for value in values:
                rows.append({
                    'category': category_label,
                    'name': _preprocessing_label(category_key, value),
                    'description': _preprocessing_description(category_key, value),
                })
        else:
            rows.append({
                'category': category_label,
                'name': 'None',
                'description': 'No option selected for this preprocessing category.',
            })
    return rows


def format_selected_preprocessing_summary(selected_normalizations=None, selected_scalings=None, selected_interpolation=None):
    """Return short human-readable summaries by preprocessing category."""
    return {
        'normalization': '; '.join(_preprocessing_label('normalization', v) for v in (selected_normalizations or [])) or 'None',
        'scaling': '; '.join(_preprocessing_label('scaling', v) for v in (selected_scalings or [])) or 'None',
        'interpolation': '; '.join(_preprocessing_label('interpolation', v) for v in (selected_interpolation or [])) or 'None',
    }


def format_metric_description_rows(metrics, selected_metrics=None):
    """Return display-ready metric description rows."""
    selected_metrics = selected_metrics or list(metrics.keys())
    rows = []
    for metric_key in selected_metrics:
        metric_info = metrics.get(metric_key, {})
        description = METRIC_DESCRIPTIONS.get(metric_key, {}).get('description', '')
        better_direction = metric_info.get('better_direction', '')
        if better_direction == 'higher':
            direction = 'Higher is better'
        elif better_direction == 'lower':
            direction = 'Lower is better'
        else:
            direction = ''
        rows.append({
            'key': metric_key,
            'name': metric_info.get('display_name', metric_key),
            'direction': direction,
            'description': description,
        })
    return rows


def format_analysis_config_rows(config, app_info=None):
    """Return display-ready analysis settings rows with section headers.

    Cross-validation rows are limited to settings relevant to the selected
    scheme. Grid-search settings are always shown because they control
    parametric model initialization. Application version information is
    appended when supplied so it is retained in browser and report settings.
    """
    scheme = config.get('cv_scheme', 'shuffle_split')
    rows = []

    def add_section(label):
        rows.append({'kind': 'section', 'label': label, 'value': ''})

    def add_setting(key, value=None):
        if value is None:
            value = config.get(key, '')
        if key == 'cv_scheme':
            value = CV_SCHEME_LABELS.get(value, value)
        rows.append({
            'kind': 'setting',
            'label': ANALYSIS_CONFIG_LABELS[key],
            'value': value,
        })

    add_section('Cross-validation settings')
    add_setting('cv_scheme', scheme)
    if scheme == 'shuffle_split':
        add_setting('cv_n_splits')
        add_setting('cv_test_size')
        add_setting('cv_random_state')
    elif scheme == 'leave_contiguous_block_out':
        add_setting('cv_n_splits')

    add_section('Grid-search settings')
    for key in [
        'grid_search_num_points',
        'grid_search_param_min',
        'grid_search_param_max',
        'grid_search_random_state',
    ]:
        add_setting(key)

    if app_info:
        add_section('Application information')
        rows.append({
            'kind': 'setting',
            'label': 'App version',
            'value': app_info.get('version', 'unknown'),
        })
        rows.append({
            'kind': 'setting',
            'label': 'Source revision',
            'value': app_info.get('revision', 'Not available'),
        })

    return rows
