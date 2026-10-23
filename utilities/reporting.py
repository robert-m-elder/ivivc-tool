"""Helpers for the final model/report tab."""

import re

from utilities.descriptions import APPROACH_DESCRIPTIONS
from utilities.prediction_validity import describe_tau_prediction_skip_reason


def _model_label(models, model_name):
    return models.get(model_name, {}).get('display_name', model_name)


def _approach_label(approaches, approach_id):
    return approaches.get(approach_id, {}).get('display_name', approach_id)


def _dom_id(value):
    """Return a conservative identifier fragment for DOM ids."""
    value = str(value or 'none')
    value = value.replace(':', '-')
    return re.sub(r'[^A-Za-z0-9_-]+', '-', value).strip('-') or 'none'


def _plot_source(source_id, title):
    return {'source_id': source_id, 'title': title}


def build_final_model_options(results, models, approaches, include_prediction_methods=False, prediction_results=None):
    """Build reportable final-model options from completed model results.

    When prediction data are not supplied, options are limited to fitted
    approach/model combinations. Prediction-specific methods are only shown
    when prediction outputs exist, because otherwise the report should describe
    model fitting rather than a hypothetical prediction workflow.

    The first available option is checked by default for UI convenience only. No
    automatic recommendation or ranking is applied here.
    """
    options = []
    prediction_results = prediction_results or {}

    def add_option(model_key, approach_id, model_name, method_id=None, method_display='',
                   fit_plot_sources=None, prediction_plot_sources=None):
        option_id = f"final-{len(options) + 1}"
        approach_description = APPROACH_DESCRIPTIONS.get(approach_id, {})
        model_dom_id = _dom_id(model_key)
        label_parts = [_approach_label(approaches, approach_id)]
        if model_name:
            label_parts.append(_model_label(models, model_name))
        else:
            label_parts.append('Direct mapping')
        if include_prediction_methods and method_display:
            label_parts.append(method_display)

        options.append({
            'id': option_id,
            'model_key': model_key,
            'model_dom_id': model_dom_id,
            'approach_id': approach_id,
            'model_name': model_name,
            'model_display': _model_label(models, model_name) if model_name else 'Direct mapping',
            'approach_display': _approach_label(approaches, approach_id),
            'method_id': method_id,
            'method_display': method_display if include_prediction_methods else '',
            'label': ' - '.join(label_parts),
            'is_default': len(options) == 0,
            'fit_plot_sources': fit_plot_sources or [],
            'prediction_plot_sources': prediction_plot_sources or [],
            'method_html': approach_description.get('method_html', ''),
            'fit_uncertainty_html': approach_description.get('fit_uncertainty_html', ''),
            'prediction_error_html': approach_description.get('prediction_error_html', '') if include_prediction_methods else '',
        })

    for model_key, model_results in results.items():
        if 'error' in model_results:
            continue
        if ':' in model_key:
            approach_id, model_name = model_key.split(':', 1)
        else:
            approach_id, model_name = model_key, None
        model_dom_id = _dom_id(model_key)
        first_option = len(options)
        if approach_id == 'approach1':
            fit_sources = [_plot_source(f"plot-fit-{model_dom_id}", 'Time-correlation fitting plot')]
            prediction = prediction_results.get(approach_id, {}).get(model_name, {})
            if include_prediction_methods and prediction.get('plot') is not None and 'error' not in prediction:
                add_option(model_key, approach_id, model_name,
                           method_id='direct_time_correlation', method_display='Direct time correlation',
                           fit_plot_sources=fit_sources,
                           prediction_plot_sources=[_plot_source(f"plot-prediction-{model_dom_id}", 'Prediction plot')])
        elif approach_id == 'approach2':
            fit_sources = [_plot_source(f"plot-fit-{model_dom_id}", 'In vitro and in vivo fitting plot')]
            prediction = prediction_results.get(approach_id, {}).get(model_name, {})
            if include_prediction_methods and 'error' not in prediction:
                methods = (
                    ('value_ratio', 'Value-ratio rescaling', 'plot', 'value', 'Response-value ratio prediction plot'),
                    ('tau_ratio', 'Time-constant-ratio rescaling', 'plot_tau', 'tau', 'Time-constant ratio prediction plot'),
                    ('observed_response_inverse', 'Observed-response inverse mapping', 'plot_inverse_response', 'inverse', 'Observed-response inverse prediction plot'),
                )
                for method_id, label, plot_key, suffix, plot_title in methods:
                    if prediction.get(plot_key) is None:
                        continue
                    if method_id == 'tau_ratio' and describe_tau_prediction_skip_reason(model_results.get('tau', [])) is not None:
                        continue
                    add_option(model_key, approach_id, model_name, method_id=method_id, method_display=label,
                               fit_plot_sources=fit_sources,
                               prediction_plot_sources=[_plot_source(f"plot-prediction-{model_dom_id}-{suffix}", plot_title)])
        elif approach_id == 'approach3':
            fit_sources = [_plot_source(f"plot-fit-{model_dom_id}-value", 'Value-ratio mapping plot'),
                           _plot_source(f"plot-fit-{model_dom_id}-time", 'Time-ratio mapping plot')]
            if include_prediction_methods:
                for method_id, label, prediction_key, suffix in (
                    ('value_scaling', 'Direct value-ratio mapping', 'method1_value_scaling', 'value'),
                    ('time_scaling', 'Direct time-ratio mapping', 'method2_time_scaling', 'time'),
                ):
                    prediction = prediction_results.get(approach_id, {}).get(prediction_key, {})
                    if prediction.get('plot') is None or 'error' in prediction:
                        continue
                    add_option(model_key, approach_id, model_name, method_id=method_id, method_display=label,
                               fit_plot_sources=[_plot_source(f"plot-fit-{model_dom_id}-{suffix}", label + ' plot')],
                               prediction_plot_sources=[_plot_source('plot-prediction-approach3-' + prediction_key.replace('_', '-'), label + ' prediction plot')])
        else:
            continue
        # A failed prediction must not hide a successful calibration result or
        # offer a method whose output was never produced.
        if len(options) == first_option:
            add_option(model_key, approach_id, model_name, fit_plot_sources=fit_sources)

    return options
