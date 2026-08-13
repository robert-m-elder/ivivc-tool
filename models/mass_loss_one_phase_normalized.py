import numpy as np

# Constrained one-phase mass-loss model for responses already normalized to 0-1.
# The baseline response and total delayed-loss magnitude are both fixed at 1.
display_name = "Mass Loss (One-Phase, Normalized)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = 1 - 1/(1 + e^{-(x-t_{50})/s})"
description = (
    "Less-flexible one-phase mass-remaining model for responses already normalized "
    "from 0 to 1. The baseline response and delayed-loss magnitude are fixed at 1, "
    "leaving only the transition midpoint and width to be fitted. "
    "This is a shifted and inverted logistic (sigmoid) function."
)


def _safe_width(value):
    """Avoid division by zero while preserving the sign of the width parameter."""
    value = float(value)
    eps = np.finfo(float).eps
    if abs(value) < eps:
        return eps
    return value


def model_function(x, t50, s):
    """Normalized one-phase mass-loss model.

    Parameters
    ----------
    t50 : midpoint time of the delayed loss phase
    s : transition-width parameter for the delayed loss phase
    """
    x = np.asarray(x, dtype=float)
    s_safe = _safe_width(s)
    delayed_exponent = np.clip(-((x - t50) / s_safe), -700, 700)
    delayed_loss = 1.0 / (1.0 + np.exp(delayed_exponent))
    return 1.0 - delayed_loss


def initial_guess(x, y):
    """Return a data-driven starting point for the normalized one-phase model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([1.0, 1.0], dtype=float)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    midpoint_index = int(np.nanargmin(np.abs(y - 0.5)))
    midpoint = float(x[midpoint_index])
    transition_width = max(span * 0.1, np.finfo(float).eps)

    return np.array([midpoint, transition_width], dtype=float)


model_function.supports_tau = False
