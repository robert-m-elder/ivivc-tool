"""Linear-plus-power remaining response with its initial value fixed at 1."""

import numpy as np

from .linear_power_decay import _initial_guess_candidates, _remaining_fraction


display_name = "Linear + Power Decay (Normalized)"
approaches = ['approach2']
fit_kwargs = {
    'bounds': ([0.0, np.finfo(float).tiny, np.finfo(float).tiny], np.inf),
    'method': 'trf',
    'x_scale': 'jac',
    'maxfev': 20000,
    'ftol': 1e-8,
    'xtol': 1e-8,
}
latex_equation = r"y = \max\{1-kx-(x/t_f)^n,0\}"
description = (
    "Linear-plus-power remaining-response model for data already on a 0-1 "
    "scale. The initial response is fixed at 1, leaving the linear rate, "
    "power-term time scale, and exponent to be fitted. "
    "Requires nonnegative time; k >= 0, tf > 0, and n > 0."
)


def model_function(x, k, tf, n):
    """Return max(1 - k*x - (x/tf)**n, 0) for nonnegative time."""
    return _remaining_fraction(x, k, tf, n)


def initial_guess(x, y):
    return _initial_guess_candidates(x, y, normalized=True)


model_function.supports_tau = False
