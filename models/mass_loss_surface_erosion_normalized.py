import numpy as np

# Constrained surface-erosion model for responses already normalized to 0-1.
# The response at time zero is fixed at 1 and the terminal response remains fixed at 0.
display_name = "Mass Loss (Surface Erosion, Normalized)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = \max(1-kx,0)^n"
description = (
    "Less-flexible surface-erosion mass-remaining model for responses already "
    "normalized from 0 to 1. The response at time zero is fixed at 1, leaving the "
    "erosion-rate and geometry/shape parameters to be fitted."
)


def _positive_or_epsilon(value):
    """Return a small positive value when a rate/shape parameter is nonpositive."""
    value = float(value)
    eps = np.finfo(float).eps
    if not np.isfinite(value) or value <= eps:
        return eps
    return value


def model_function(x, k, n):
    """Normalized surface-erosion mass-loss model.

    Parameters
    ----------
    k : surface erosion rate parameter
    n : fitted geometry/shape exponent
    """
    x = np.asarray(x, dtype=float)
    k_safe = _positive_or_epsilon(k)
    n_safe = _positive_or_epsilon(n)
    return np.maximum(1.0 - k_safe * x, 0.0) ** n_safe


def initial_guess(x, y):
    """Return a data-driven starting point for the normalized surface model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([1.0, 1.0], dtype=float)

    x = x[mask]
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    k = 0.8 / span
    n = 1.0
    return np.array([k, n], dtype=float)


model_function.supports_tau = False
