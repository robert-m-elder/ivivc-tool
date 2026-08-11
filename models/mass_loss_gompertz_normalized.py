import numpy as np

# Constrained modified-Gompertz mass-loss model for responses already normalized
# to a 0-1 scale. The baseline response and ultimate loss magnitude are fixed at 1.
display_name = "Mass Loss (Gompertz, Normalized)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = 1 - \exp\{-\exp[e r_{max}(t_{lag}-x)+1]\}"
description = (
    "Less-flexible modified Gompertz mass-remaining model for responses already "
    "normalized from 0 to 1. The baseline response and ultimate loss magnitude "
    "are fixed at 1, leaving only the maximum loss-rate and lag-time parameters "
    "to be fitted."
)


def model_function(x, rmax, lag):
    """Normalized modified-Gompertz mass-loss model.

    Parameters
    ----------
    rmax : maximum mass-loss rate parameter on the normalized response scale
    lag : apparent lag time before rapid mass loss
    """
    x = np.asarray(x, dtype=float)
    inner = (np.e * rmax) * (lag - x) + 1.0
    inner = np.clip(inner, -700, 700)
    loss = np.exp(-np.exp(inner))
    return 1.0 - loss


def initial_guess(x, y):
    """Return a data-driven starting point for the normalized Gompertz model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([0.1, 1.0], dtype=float)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        valid_dx = np.abs(dx) > np.finfo(float).eps
        slopes = np.full_like(dy, np.nan, dtype=float)
        slopes[valid_dx] = dy[valid_dx] / dx[valid_dx]
        decreasing_rates = -slopes[np.isfinite(slopes)]
        positive_rates = decreasing_rates[decreasing_rates > 0]
        max_rate = float(np.nanmax(positive_rates)) if len(positive_rates) else 1.0 / span
    else:
        max_rate = 1.0 / span

    cumulative_loss = 1.0 - y
    above_threshold = np.where(cumulative_loss >= 0.05)[0]
    lag = float(x[above_threshold[0]]) if len(above_threshold) else x_min + 0.25 * span

    return np.array([max(max_rate, np.finfo(float).eps), lag], dtype=float)


model_function.supports_tau = False
