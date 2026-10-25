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
        return (
            'Time-constant rescaling was not performed because valid time constants '
            'could not be calculated for the in vitro and in vivo fits.'
        )

    available = []
    for tau in tau_values:
        nominal = getattr(tau, 'n', None)
        available.append(_finite_number(nominal) and float(nominal) > 0)

    if all(available):
        return None
    if not any(available):
        return (
            'Time-constant rescaling was not performed because valid time constants '
            'could not be calculated for the in vitro and in vivo fits.'
        )
    missing_label = 'in vitro' if not available[0] else 'in vivo'
    return (
        f'Time-constant rescaling was not performed because the {missing_label} '
        'time constant could not be calculated.'
    )


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
