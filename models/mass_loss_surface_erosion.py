import numpy as np

# Geometry-inspired surface-erosion model with fitted shape exponent.
display_name = "Mass Loss (Surface Erosion)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = y0 \cdot \max(1 - k \cdot x, 0)^n"
#latex_equation = r"y = yinf + (y0 - yinf) \cdot \max(1 - k \cdot x, 0)^n"
description = (
    "Mass-remaining model for surface-controlled erosion. The fitted exponent "
    "n allows the profile shape to adapt to slab-like, cylindrical, spherical, "
    "or empirical geometry-like erosion behavior."
)


def _positive_or_epsilon(value):
    """Return a small positive value when a rate/shape parameter is nonpositive."""
    value = float(value)
    eps = np.finfo(float).eps
    if not np.isfinite(value) or value <= eps:
        return eps
    return value


def model_function(x, y0, k, n):
#def model_function(x, y0, yinf, k, n):
    """Surface-erosion mass-loss model expressed as mass remaining over time.

    Parameters
    ----------
    y0 : initial mass/response level
    yinf : terminal/asymptotic mass/response level
    k : surface erosion rate parameter
    n : fitted geometry/shape exponent
    """
    yinf = 0
    x = np.asarray(x, dtype=float)
    k_safe = _positive_or_epsilon(k)
    n_safe = _positive_or_epsilon(n)

    remaining_fraction = np.maximum(1.0 - k_safe * x, 0.0) ** n_safe
    return yinf + (y0 - yinf) * remaining_fraction


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
    yinf = float(y[-1])
    k = 0.8 / span
    n = 1.0

    return np.array([y0, k, n], dtype=float)
    #return np.array([y0, yinf, k, n], dtype=float)


model_function.supports_tau = False
