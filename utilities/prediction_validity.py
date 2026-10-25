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
    positive finite nominal values. Unavailable standard uncertainties suppress
    intervals, not central predictions. The returned text is suitable for
    display in the prediction tab and report.
    """
    if not tau_values or len(tau_values) != 2:
        return 'Time-constant-ratio rescaling was not performed because both in vitro and in vivo tau values were not available.'

    labels = ['In vitro tau', 'In vivo tau']
    reasons = []

    for label, tau in zip(labels, tau_values):
        nominal = getattr(tau, 'n', None)

        if not _finite_number(nominal):
            reasons.append(f'{label} is not finite')
        elif float(nominal) <= 0:
            reasons.append(f'{label} is not positive')

    if reasons:
        return 'Time-constant-ratio rescaling was not performed because ' + '; '.join(reasons) + '.'

    return None


def tau_prediction_is_valid(tau_values):
    """Return True when tau-ratio prediction can be performed."""
    return describe_tau_prediction_skip_reason(tau_values) is None


def tau_estimate_is_available(tau):
    """Treat missing values and legacy nonpositive sentinels as unavailable."""
    value = getattr(tau, 'n', None)
    return _finite_number(value) and float(value) > 0


def tau_uncertainty_is_available(tau):
    value = getattr(tau, 's', None)
    return tau_estimate_is_available(tau) and _finite_number(value) and float(value) >= 0
