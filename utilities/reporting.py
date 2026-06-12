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


def build_final_model_options(results, models, approaches, include_prediction_methods=False):
    """Build reportable final-model options from completed model results.

    When prediction data are not supplied, options are limited to fitted
    approach/model combinations. Prediction-specific methods are only shown
    when prediction outputs exist, because otherwise the report should describe
    model fitting rather than a hypothetical prediction workflow.

    The first available option is checked by default for UI convenience only. No
    automatic recommendation or ranking is applied here.
    """
    options = []

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

        if approach_id == 'approach1':
            fit_sources = [_plot_source(f"plot-fit-{model_dom_id}", 'Model fitting plot')]
            prediction_sources = []
            if include_prediction_methods:
                prediction_sources = [_plot_source(f"plot-prediction-{model_dom_id}", 'Prediction plot')]
            add_option(
                model_key, approach_id, model_name,
                method_id='direct_time_correlation' if include_prediction_methods else None,
                method_display='Direct time correlation',
                fit_plot_sources=fit_sources,
                prediction_plot_sources=prediction_sources,
            )
        elif approach_id == 'approach2':
            fit_sources = [_plot_source(f"plot-fit-{model_dom_id}", 'Model fitting plot')]
            if include_prediction_methods:
                add_option(
                    model_key, approach_id, model_name,
                    method_id='value_ratio',
                    method_display='Value-ratio rescaling',
                    fit_plot_sources=fit_sources,
                    prediction_plot_sources=[_plot_source(f"plot-prediction-{model_dom_id}-value", 'Value-ratio prediction plot')],
                )
                tau_values = model_results.get('tau', [])
                if describe_tau_prediction_skip_reason(tau_values) is None:
                    add_option(
                        model_key, approach_id, model_name,
                        method_id='tau_ratio',
                        method_display='Time-constant-ratio rescaling',
                        fit_plot_sources=fit_sources,
                        prediction_plot_sources=[_plot_source(f"plot-prediction-{model_dom_id}-tau", 'Time-constant-ratio prediction plot')],
                    )
            else:
                add_option(model_key, approach_id, model_name, fit_plot_sources=fit_sources)
        elif approach_id == 'approach3':
            if include_prediction_methods:
                add_option(
                    model_key, approach_id, model_name,
                    method_id='value_scaling',
                    method_display='Direct value-ratio mapping',
                    fit_plot_sources=[_plot_source(f"plot-fit-{model_dom_id}-value", 'Value-ratio mapping plot')],
                    prediction_plot_sources=[_plot_source('plot-prediction-approach3-method1-value-scaling', 'Direct value-ratio prediction plot')],
                )
                add_option(
                    model_key, approach_id, model_name,
                    method_id='time_scaling',
                    method_display='Direct time-ratio mapping',
                    fit_plot_sources=[_plot_source(f"plot-fit-{model_dom_id}-time", 'Time-ratio mapping plot')],
                    prediction_plot_sources=[_plot_source('plot-prediction-approach3-method2-time-scaling', 'Direct time-ratio prediction plot')],
                )
            else:
                add_option(
                    model_key, approach_id, model_name,
                    fit_plot_sources=[
                        _plot_source(f"plot-fit-{model_dom_id}-value", 'Value-ratio mapping plot'),
                        _plot_source(f"plot-fit-{model_dom_id}-time", 'Time-ratio mapping plot'),
                    ],
                )

    return options
