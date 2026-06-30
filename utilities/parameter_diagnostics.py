"""Plain-language fitted-parameter diagnostics for fitted models."""

from html import escape

import numpy as np


RSE_ELEVATED_THRESHOLD = 50.0
RSE_HIGH_THRESHOLD = 100.0
CORRELATION_ELEVATED_THRESHOLD = 0.95
CORRELATION_HIGH_THRESHOLD = 0.99
CONDITION_ELEVATED_THRESHOLD = 1e8
CONDITION_HIGH_THRESHOLD = 1e12


_BADGE_CLASS_BY_LEVEL = {
    'ok': 'evidence-badge-comparable',
    'caution': 'evidence-badge-lower',
    'warning': 'evidence-badge-minimal',
}


def _format_percent(value):
    if np.isinf(value):
        return 'infinite'
    if not np.isfinite(value):
        return 'N/A'
    return f'{value:.1f}%'


def _format_number(value):
    if np.isinf(value):
        return 'infinite'
    if not np.isfinite(value):
        return 'N/A'
    return f'{value:.3g}'


def _badge(label, level):
    badge_class = _BADGE_CLASS_BY_LEVEL.get(level, 'evidence-badge-lower')
    return (
        f'<span class="evidence-badge {badge_class}">'
        f'{escape(label)}'
        '</span>'
    )


def _diagnostic_unavailable(message):
    return (
        '<div class="parameter-diagnostics">'
        '<div class="parameter-diagnostic-badges">'
        f'{_badge("Parameter uncertainty unavailable", "warning")}'
        '</div>'
        f'<p>{escape(message)}</p>'
        '</div>'
    )


def _validate_covariance(params, pcov):
    param_names = list(params.keys())
    if pcov is None:
        return None, param_names, 'Parameter-uncertainty diagnostics are not available because uncertainty information was not returned for this fit.'

    try:
        covariance = np.asarray(pcov, dtype=float)
    except Exception:
        return None, param_names, 'Parameter-uncertainty diagnostics are not available because uncertainty information could not be converted to numeric values.'

    n_params = len(param_names)
    if covariance.ndim != 2 or covariance.shape != (n_params, n_params):
        return None, param_names, 'Parameter-uncertainty diagnostics are not available because uncertainty information does not match the number of fitted parameters.'

    if not np.all(np.isfinite(covariance)):
        return None, param_names, 'Parameter-uncertainty diagnostics are not available because uncertainty information contains non-finite values.'

    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        return None, param_names, 'Parameter-uncertainty diagnostics are not available because one or more uncertainty entries are not physically interpretable.'

    return covariance, param_names, None


def _relative_standard_errors(params, covariance, param_names):
    diagonal = np.diag(covariance)
    standard_errors = np.sqrt(diagonal)
    rse_values = []

    for i, param_name in enumerate(param_names):
        estimate = float(params[param_name])
        standard_error = float(standard_errors[i])
        denominator = abs(estimate)
        if denominator > np.finfo(float).eps:
            rse = 100.0 * standard_error / denominator
        elif standard_error > 0:
            rse = np.inf
        else:
            rse = 0.0
        rse_values.append((param_name, rse))

    return rse_values


def _largest_parameter_correlation(covariance, param_names):
    if len(param_names) < 2:
        return None

    standard_errors = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_errors, standard_errors)
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = np.divide(
            covariance,
            denominator,
            out=np.full_like(covariance, np.nan, dtype=float),
            where=denominator > 0,
        )

    candidates = []
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            value = correlation[i, j]
            if np.isfinite(value):
                candidates.append((abs(float(value)), float(value), param_names[i], param_names[j]))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _covariance_condition_number(covariance):
    try:
        return float(np.linalg.cond(covariance))
    except Exception:
        return np.inf


def build_parameter_diagnostics_html(params, pcov):
    """Return a compact HTML summary of fitted-parameter diagnostics.

    The diagnostics are intended as screening aids. They do not label a model as
    valid or invalid and should be interpreted together with plots, residuals,
    cross-validation behavior, and scientific plausibility.
    """
    covariance, param_names, error_message = _validate_covariance(params, pcov)
    if error_message:
        return _diagnostic_unavailable(error_message)

    badges = []
    details = []
    concern_details = []
    coupling_details = []

    rse_values = _relative_standard_errors(params, covariance, param_names)
    largest_rse = max(rse_values, key=lambda item: item[1]) if rse_values else None
    has_elevated_rse = bool(largest_rse and largest_rse[1] >= RSE_ELEVATED_THRESHOLD)
    has_high_rse = bool(largest_rse and largest_rse[1] >= RSE_HIGH_THRESHOLD)
    if has_high_rse:
        badges.append(_badge('Large parameter uncertainty', 'warning'))
        concern_details.append(
            f'Largest ± uncertainty relative to the fitted value: {escape(largest_rse[0])} = '
            f'{escape(_format_percent(largest_rse[1]))}.'
        )
    elif has_elevated_rse:
        badges.append(_badge('Large parameter uncertainty', 'caution'))
        concern_details.append(
            f'Largest ± uncertainty relative to the fitted value: {escape(largest_rse[0])} = '
            f'{escape(_format_percent(largest_rse[1]))}.'
        )

    condition_number = _covariance_condition_number(covariance)
    has_elevated_stability = bool(
        not np.isfinite(condition_number)
        or condition_number >= CONDITION_ELEVATED_THRESHOLD
    )
    has_high_stability = bool(
        not np.isfinite(condition_number)
        or condition_number >= CONDITION_HIGH_THRESHOLD
    )
    if has_high_stability:
        badges.append(_badge('Parameter stability caution', 'warning'))
        concern_details.append(
            'The fitted parameter estimates or their ± uncertainty may be sensitive '
            'to small changes in the data.'
        )
    elif has_elevated_stability:
        badges.append(_badge('Parameter stability caution', 'caution'))
        concern_details.append(
            'The fitted parameter estimates or their ± uncertainty may be sensitive '
            'to small changes in the data.'
        )

    largest_correlation = _largest_parameter_correlation(covariance, param_names)
    if largest_correlation:
        abs_corr, signed_corr, first_param, second_param = largest_correlation
        if abs_corr >= CORRELATION_ELEVATED_THRESHOLD:
            correlation_detail = (
                'Strongest parameter coupling: '
                f'|corr({escape(first_param)}, {escape(second_param)})| = {abs_corr:.3f} '
                f'(signed value {signed_corr:.3f}).'
            )
            if has_elevated_rse or has_elevated_stability:
                if abs_corr >= CORRELATION_HIGH_THRESHOLD or has_high_rse or has_high_stability:
                    badges.append(_badge('Parameter identifiability caution', 'warning'))
                else:
                    badges.append(_badge('Parameter identifiability caution', 'caution'))
                concern_details.append(correlation_detail)
            else:
                badges.append(_badge('Parameter coupling noted', 'ok'))
                coupling_details.append(correlation_detail)

    details.extend(concern_details)
    details.extend(coupling_details)

    if not badges:
        badges.append(_badge('No parameter uncertainty concerns identified', 'ok'))
        summary = (
            'Screening checks did not identify large parameter uncertainty, strong '
            'parameter coupling, or signs that parameter estimates may be sensitive '
            'to small changes in the data. This does not by itself validate the model.'
        )
    elif concern_details:
        if coupling_details:
            summary = (
                'Review these fitted-parameter diagnostics. Correlated parameter '
                'estimates can occur naturally in nonlinear models, especially when '
                'parameters control scale, shape, or rate, but correlation accompanied '
                'by large uncertainty or parameter-stability concerns may indicate '
                'weakly identified, redundant, or data-sensitive fitted parameters.'
            )
        else:
            summary = (
                'Review these fitted-parameter diagnostics. They may indicate '
                'large parameter uncertainty, sensitivity to the available data, or '
                'weakly identified fitted parameters.'
            )
    else:
        summary = (
            'Correlated parameter estimates were detected. This can occur naturally in '
            'nonlinear models, especially when parameters control scale, shape, or rate. '
            'Review whether large parameter uncertainty or parameter-stability concerns '
            'are also present before treating this as an identifiability concern.'
        )

    details_html = ''
    if details:
        details_html = '<ul>' + ''.join(f'<li>{detail}</li>' for detail in details) + '</ul>'

    return (
        '<div class="parameter-diagnostics">'
        '<div class="parameter-diagnostic-badges">'
        + ''.join(badges)
        + '</div>'
        + f'<p>{escape(summary)}</p>'
        + details_html
        + '</div>'
    )
