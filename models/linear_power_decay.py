"""Empirical remaining response with additive linear and power-law loss."""

import numpy as np


display_name = "Linear + Power Decay"
approaches = ['approach2']
fit_kwargs = {
    'bounds': ([0.0, 0.0, np.finfo(float).tiny, np.finfo(float).tiny], np.inf),
    'method': 'trf',
    'x_scale': 'jac',
    'maxfev': 20000,
    'ftol': 1e-8,
    'xtol': 1e-8,
}
latex_equation = r"y = y_0 \max\{1-kx-(x/t_f)^n,0\}"
description = (
    "Empirical remaining-response model combining linear and power-law loss, "
    "with the initial response fitted and negative remaining values set to zero. "
    "For n > 1, the power term can describe accelerating loss. "
    "Requires nonnegative time; k >= 0, tf > 0, n > 0, and y0 >= 0."
)


def _remaining_fraction(x, k, tf, n):
    """Evaluate the requested clipped form without overflowing its zero tail."""
    if not (np.isfinite(k) and k >= 0 and np.isfinite(tf) and tf > 0
            and np.isfinite(n) and n > 0):
        raise ValueError('Linear + Power Decay requires k >= 0, tf > 0, and n > 0.')

    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError('Linear + Power Decay requires finite, nonnegative time values.')

    remaining = np.zeros_like(x, dtype=float)
    # For valid parameters every x >= tf is already on the zero tail.
    # Below tf the power has a base below 1 and cannot overflow.
    active = x < tf
    with np.errstate(over='ignore'):
        remaining[active] = np.maximum(
            1.0 - k * x[active] - (x[active] / tf) ** n,
            0.0,
        )
#    with np.errstate(over='ignore'):
#        remaining = np.maximum(
#            1.0 - k * x - (x / tf) ** n,
#            0.0,
#        )
    return remaining


def model_function(x, y0, k, tf, n):
    """Return y0 * max(1 - k*x - (x/tf)**n, 0) for nonnegative time.

    y0 is the initial response, k the linear-loss rate, tf the power-term
    time scale, and n the power exponent. tf is not generally the zero time.
    """
    if not np.isfinite(y0) or y0 < 0:
        raise ValueError('Linear + Power Decay requires a finite, nonnegative y0.')
    return y0 * _remaining_fraction(x, k, tf, n)


def _initial_guess_candidates(x, y, normalized=False):
    """Supply feasible, non-flat starts to the existing unconstrained search."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x, y = x[valid], y[valid]

    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
        time_scale = float(np.max(x)) if np.max(x) > 0 else 1.0
        y0 = 1.0 if normalized else max(float(np.max(y)), np.finfo(float).eps)
        late_response = float(np.median(y[-min(3, len(y)):]))
        loss = float(np.clip(1.0 - late_response / y0, 0.05, 0.95))
    else:
        time_scale, y0, loss = 1.0, 1.0, 0.5

    candidates = []
    for exponent in (0.5, 2.0, 4.0):
        for linear_share in (0.1, 0.5, 0.9):
            k = linear_share * loss / time_scale
            tf = time_scale / ((1.0 - linear_share) * loss) ** (1.0 / exponent)
            parameters = [k, tf, exponent]
            if not normalized:
                parameters.insert(0, y0)
            candidates.append(parameters)
    return np.asarray(candidates, dtype=float)


def initial_guess(x, y):
    return _initial_guess_candidates(x, y)


# Match the existing finite-ending response models: no time-constant mapping.
model_function.supports_tau = False
