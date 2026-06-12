"""Helpers for the final model/report tab."""

import math
import re

from utilities.descriptions import APPROACH_DESCRIPTIONS


def _model_label(models, model_name):
    return models.get(model_name, {}).get('display_name', model_name)


def _approach_label(approaches, approach_id):
    return approaches.get(approach_id, {}).get('display_name', approach_id)


def _dom_id(value):
    """Return a conservative identifier fragment for DOM ids."""
    value = str(value or 'none')
    value = value.replace(':', '-')
    return re.sub(r'[^A-Za-z0-9_-]+', '-', value).strip('-') or 'none'


def build_final_model_options(results, models, approaches):
    """Build reportable final-model/method options from completed model results.

    The first available option is checked by default for UI convenience only. No
    automatic recommendation or ranking is applied here.
    """
    options = []

    def add_option(model_key, approach_id, model_name, method_id, method_display):
        option_id = f"final-{len(options) + 1}"
        approach_description = APPROACH_DESCRIPTIONS.get(approach_id, {})
        model_dom_id = _dom_id(model_key)
        label_parts = [_approach_label(approaches, approach_id)]
        if model_name:
            label_parts.append(_model_label(models, model_name))
        if method_display:
            label_parts.append(method_display)

        fit_plot_source = f"plot-fit-{model_dom_id}"
        prediction_plot_source = None
        if approach_id == 'approach1':
            prediction_plot_source = f"plot-prediction-{model_dom_id}"
        elif approach_id == 'approach2':
            prediction_suffix = 'tau' if method_id == 'tau_ratio' else 'value'
            prediction_plot_source = f"plot-prediction-{model_dom_id}-{prediction_suffix}"
        elif approach_id == 'approach3':
            fit_suffix = 'value' if method_id == 'value_scaling' else 'time'
            prediction_suffix = 'method1-value-scaling' if method_id == 'value_scaling' else 'method2-time-scaling'
            fit_plot_source = f"plot-fit-{model_dom_id}-{fit_suffix}"
            prediction_plot_source = f"plot-prediction-{approach_id}-{prediction_suffix}"

        options.append({
            'id': option_id,
            'model_key': model_key,
            'model_dom_id': model_dom_id,
            'approach_id': approach_id,
            'model_name': model_name,
            'model_display': _model_label(models, model_name) if model_name else 'Direct mapping',
            'approach_display': _approach_label(approaches, approach_id),
            'method_id': method_id,
            'method_display': method_display,
            'label': ' - '.join(label_parts),
            'is_default': len(options) == 0,
            'fit_plot_source': fit_plot_source,
            'prediction_plot_source': prediction_plot_source,
            'method_html': approach_description.get('method_html', ''),
            'error_html': approach_description.get('prediction_error_html', ''),
        })

    for model_key, model_results in results.items():
        if 'error' in model_results:
            continue

        if ':' in model_key:
            approach_id, model_name = model_key.split(':', 1)
        else:
            approach_id, model_name = model_key, None

        if approach_id == 'approach1':
            add_option(model_key, approach_id, model_name, 'direct_time_correlation', 'Direct time correlation')
        elif approach_id == 'approach2':
            add_option(model_key, approach_id, model_name, 'value_ratio', 'Value-ratio rescaling')
            tau_values = model_results.get('tau', [])
            has_tau = (
                len(tau_values) == 2
                and all(getattr(tau, 'n', 0) > 0 for tau in tau_values)
                and all(math.isfinite(getattr(tau, 'n', float('nan'))) for tau in tau_values)
            )
            if has_tau:
                add_option(model_key, approach_id, model_name, 'tau_ratio', 'Time-constant-ratio rescaling')
        elif approach_id == 'approach3':
            add_option(model_key, approach_id, model_name, 'value_scaling', 'Direct value-ratio mapping')
            add_option(model_key, approach_id, model_name, 'time_scaling', 'Direct time-ratio mapping')

    return options
