"""Validation helpers for optional prediction methods."""

import math


def _finite_number(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value)


def describe_tau_prediction_skip_reason(tau_values):
    """Return None when tau-ratio prediction is valid, otherwise a reason string.

    Tau-ratio rescaling requires both in vitro and in vivo tau values to have
    positive finite nominal values and finite, non-negative standard
    uncertainties. The returned text is suitable for display in the prediction
    tab and report.
    """
    if not tau_values or len(tau_values) != 2:
        return 'Time-constant-ratio rescaling was not performed because both in vitro and in vivo tau values were not available.'

    labels = ['In vitro tau', 'In vivo tau']
    reasons = []

    for label, tau in zip(labels, tau_values):
        nominal = getattr(tau, 'n', None)
        std_uncertainty = getattr(tau, 's', None)

        if not _finite_number(nominal):
            reasons.append(f'{label} is not finite')
        elif float(nominal) <= 0:
            reasons.append(f'{label} is not positive')

        if not _finite_number(std_uncertainty):
            reasons.append(f'{label} standard uncertainty is not finite')
        elif float(std_uncertainty) < 0:
            reasons.append(f'{label} standard uncertainty is negative')

    if reasons:
        return 'Time-constant-ratio rescaling was not performed because ' + '; '.join(reasons) + '.'

    return None


def tau_prediction_is_valid(tau_values):
    """Return True when tau-ratio prediction can be performed."""
    return describe_tau_prediction_skip_reason(tau_values) is None
