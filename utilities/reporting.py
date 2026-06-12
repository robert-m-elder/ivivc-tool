"""Helpers for the final model/report tab."""

import math

from utilities.descriptions import APPROACH_DESCRIPTIONS


def _model_label(models, model_name):
    return models.get(model_name, {}).get('display_name', model_name)


def _approach_label(approaches, approach_id):
    return approaches.get(approach_id, {}).get('display_name', approach_id)


def build_final_model_options(results, models, approaches, best_models=None):
    """Build reportable final-model/method options from completed model results."""
    options = []
    recommended_keys = set()

    if best_models:
        for score in best_models:
            for model_key in results:
                if ':' in model_key:
                    approach_id, model_name = model_key.split(':', 1)
                else:
                    approach_id, model_name = model_key, model_key
                if model_name == score.model_name and _approach_label(approaches, approach_id) == score.approach:
                    recommended_keys.add(model_key)

    def add_option(model_key, approach_id, model_name, method_id, method_display):
        option_id = f"final-{len(options) + 1}"
        approach_description = APPROACH_DESCRIPTIONS.get(approach_id, {})
        label_parts = [_approach_label(approaches, approach_id)]
        if model_name:
            label_parts.append(_model_label(models, model_name))
        if method_display:
            label_parts.append(method_display)

        options.append({
            'id': option_id,
            'model_key': model_key,
            'approach_id': approach_id,
            'model_name': model_name,
            'model_display': _model_label(models, model_name) if model_name else 'Direct mapping',
            'approach_display': _approach_label(approaches, approach_id),
            'method_id': method_id,
            'method_display': method_display,
            'label': ' - '.join(label_parts),
            'is_recommended': model_key in recommended_keys,
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

    if options and not any(option['is_recommended'] for option in options):
        options[0]['is_recommended'] = True

    return options
