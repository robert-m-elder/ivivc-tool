import numpy as np

# Compact asymmetric sigmoidal mass-loss model with lag/plateau behavior.
display_name = "Mass Loss (Gompertz)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = y_0 - A \exp\{-\exp[(e r_{max}/A)(t_{lag}-x)+1]\}"
description = (
    "Mass-remaining model based on the modified Gompertz curve. It is intended "
    "for asymmetric sigmoidal mass-loss profiles with an apparent lag or plateau, "
    "a maximum loss-rate region, and a terminal loss extent."
)


def _safe_nonzero(value):
    """Avoid division by zero while preserving the sign of the parameter."""
    value = float(value)
    eps = np.finfo(float).eps
    if abs(value) < eps:
        return eps if value >= 0 else -eps
    return value


def model_function(x, y0, A, rmax, lag):
    """Modified Gompertz mass-loss model expressed as mass remaining over time.

    Parameters
    ----------
    y0 : initial mass/response level
    A : ultimate mass-loss magnitude
    rmax : maximum mass-loss rate parameter
    lag : apparent lag time before rapid mass loss
    """
    x = np.asarray(x, dtype=float)
    A_safe = _safe_nonzero(A)

    inner = (np.e * rmax / A_safe) * (lag - x) + 1.0
    inner = np.clip(inner, -700, 700)
    loss = A * np.exp(-np.exp(inner))
    return y0 - loss


def initial_guess(x, y):
    """Return a data-driven starting point for the Gompertz mass-loss model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([100.0, 10.0, 1.0, 1.0], dtype=float)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    y0 = float(y[0])
    y_min = float(np.nanmin(y))
    y_end = float(y[-1])
    response_span = max(abs(float(np.nanmax(y) - y_min)), np.finfo(float).eps)
    ultimate_loss = max(response_span, y0 - y_end, np.finfo(float).eps)

    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        valid_dx = np.abs(dx) > np.finfo(float).eps
        slopes = np.full_like(dy, np.nan, dtype=float)
        slopes[valid_dx] = dy[valid_dx] / dx[valid_dx]
        decreasing_rates = -slopes[np.isfinite(slopes)]
        positive_rates = decreasing_rates[decreasing_rates > 0]
        max_rate = float(np.nanmax(positive_rates)) if len(positive_rates) else ultimate_loss / span
    else:
        max_rate = ultimate_loss / span

    cumulative_loss = y0 - y
    threshold = 0.05 * ultimate_loss
    above_threshold = np.where(cumulative_loss >= threshold)[0]
    lag = float(x[above_threshold[0]]) if len(above_threshold) else x_min + 0.25 * span

    return np.array([y0, ultimate_loss, max(max_rate, np.finfo(float).eps), lag], dtype=float)


model_function.supports_tau = False
