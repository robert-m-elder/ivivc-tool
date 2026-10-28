import numpy as np

# Geometry-inspired surface-erosion model with fitted shape exponent.
display_name = 'Finite-Time Power Decay'
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = A + (y_0-A) \max(1-kx,0)^n"
description = (
    'Response profile that reaches zero in finite time, with fitted amplitude, rate, and '
    'shape exponent. This surface-erosion form can also be used empirically for remaining '
    'mass, volume, molecular weight, or retained mechanical properties when the shape is '
    'appropriate.'
)


def _positive_or_epsilon(value):
    """Return a small positive value when a rate/shape parameter is nonpositive."""
    value = float(value)
    eps = np.finfo(float).eps
    if not np.isfinite(value) or value <= eps:
        return eps
    return value


def model_function(x, y0, k, n, A):
    """Surface-erosion model expressed as response over time.

    Parameters
    ----------
    y0 : initial mass/response level
    k : surface erosion rate parameter
    n : fitted geometry/shape exponent
    A : terminal/asymptotic mass/response level
    """
    x = np.asarray(x, dtype=float)
    k_safe = _positive_or_epsilon(k)
    n_safe = _positive_or_epsilon(n)

    remaining_fraction = np.maximum(1.0 - k_safe * x, 0.0) ** n_safe
    return A + (y0 - A) * remaining_fraction


def initial_guess(x, y):
    """Return a data-driven starting point for the surface-erosion model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([100.0, 0.0, 1.0, 1.0], dtype=float)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    y0 = float(y[0])
    k = 0.8 / span
    n = 1.0
    A = float(y[-1])

    return np.array([y0, k, n, A], dtype=float)


model_function.supports_tau = False
