import numpy as np

# Constrained two-phase mass-loss model for responses already normalized to 0-1.
# The baseline response is fixed at 1; the early and delayed loss magnitudes remain fitted.
display_name = "Mass Loss (Two-Phase, Normalized)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = 1 - B(1-e^{-k_b x}) - D/(1+e^{-(x-t_{50})/s})"
description = (
    "Less-flexible two-phase mass-remaining model for responses already normalized "
    "from 0 to 1. The baseline response is fixed at 1, while the early and delayed "
    "loss magnitudes (B and D) and their timing/rate parameters remain fitted."
)


def _safe_width(value):
    """Avoid division by zero while preserving the sign of the width parameter."""
    value = float(value)
    eps = np.finfo(float).eps
    if abs(value) < eps:
        return eps
    return value


def model_function(x, B, kb, D, t50, s):
    """Normalized two-phase mass-loss model.

    Parameters
    ----------
    B : magnitude of the early burst-loss phase
    kb : first-order rate constant for the early burst-loss phase
    D : magnitude of the delayed loss phase
    t50 : midpoint time of the delayed loss phase
    s : transition-width parameter for the delayed loss phase
    """
    x = np.asarray(x, dtype=float)
    s_safe = _safe_width(s)

    early_exponent = np.clip(-kb * x, -700, 700)
    delayed_exponent = np.clip(-((x - t50) / s_safe), -700, 700)

    early_loss = B * (1.0 - np.exp(early_exponent))
    delayed_loss = D / (1.0 + np.exp(delayed_exponent))
    return 1.0 - early_loss - delayed_loss


def initial_guess(x, y):
    """Return a data-driven starting point for the normalized two-phase model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([0.1, 1.0, 0.9, 1.0, 1.0], dtype=float)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    n_early = max(2, int(np.ceil(len(y) / 3))) if len(y) >= 2 else 1
    early_level = float(np.nanmedian(y[:n_early]))
    burst_loss = max(0.0, 1.0 - early_level)
    total_loss = max(0.0, 1.0 - float(y[-1]))
    delayed_loss = max(0.05, total_loss - burst_loss)

    burst_rate = 1.0 / max(span * 0.1, np.finfo(float).eps)
    midpoint = x_min + 0.75 * span
    transition_width = max(span * 0.1, np.finfo(float).eps)

    return np.array([burst_loss, burst_rate, delayed_loss, midpoint, transition_width], dtype=float)


model_function.supports_tau = False
