import numpy as np

# A parsimonious mass-remaining model for profiles with an optional early
# burst-loss phase, an intermediate plateau, and a delayed/bulk-loss phase.
display_name = "Mass Loss (Two-Phase)"
approaches = ['approach2']
fit_kwargs = {'maxfev': 20000, 'ftol': 1e-6, 'xtol': 1e-6}
latex_equation = r"y = y_0 - B(1-e^{-k_b x}) - D/(1+e^{-(x-t_{50})/s})"
description = (
    "Mass-remaining model with an optional early burst-loss phase followed by "
    "a delayed sigmoidal loss phase. It is intended for mass-loss profiles that "
    "may show an initial drop, an intermediate plateau, and a later drop."
)


def _safe_width(value):
    """Avoid division by zero while preserving the sign of the width parameter."""
    value = float(value)
    eps = np.finfo(float).eps
    if abs(value) < eps:
        return eps
    return value


def model_function(x, y0, B, kb, D, t50, s):
    """Two-phase mass-loss model expressed as mass remaining over time.

    Parameters
    ----------
    y0 : initial mass/response level
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

    early_loss = B * (1 - np.exp(early_exponent))
    delayed_loss = D / (1 + np.exp(delayed_exponent))
    return y0 - early_loss - delayed_loss


def initial_guess(x, y):
    """Return a data-driven starting point for the mass-loss model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([100.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, np.finfo(float).eps)

    y0 = float(y[0])
    y_end = float(y[-1])
    response_span = max(abs(float(np.nanmax(y) - np.nanmin(y))), np.finfo(float).eps)

    n_early = max(2, int(np.ceil(len(y) / 3))) if len(y) >= 2 else 1
    early_level = float(np.nanmedian(y[:n_early]))
    burst_loss = max(0.0, y0 - early_level)
    total_loss = max(response_span, y0 - y_end)
    delayed_loss = max(response_span * 0.25, total_loss - burst_loss)

    burst_rate = 1.0 / max(span * 0.1, np.finfo(float).eps)
    midpoint = x_min + 0.75 * span
    transition_width = max(span * 0.1, np.finfo(float).eps)

    return np.array([y0, burst_loss, burst_rate, delayed_loss, midpoint, transition_width], dtype=float)


model_function.supports_tau = False
