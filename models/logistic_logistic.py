"""Two decreasing logistic transitions with a fitted initial response."""

import numpy as np


display_name = 'Logistic + Logistic'
approaches = ['approach2']
fit_kwargs = {
    'bounds': ([0.0, 0.0, -np.inf, -np.inf, np.finfo(float).tiny, np.finfo(float).tiny],
               [np.inf, 1.0, np.inf, np.inf, np.inf, np.inf]),
    'method': 'trf',
    'x_scale': 'jac',
    'maxfev': 20000,
    'ftol': 1e-8,
    'xtol': 1e-8,
}
latex_equation = (
    r"y = A\left[w\frac{L(x;k_1,t_1)}{L(0;k_1,t_1)}"
    r"+(1-w)\frac{L(x;k_2,t_2)}{L(0;k_2,t_2)}\right],\quad "
    r"L(x;k,t)=\frac{1}{1+e^{k(x-t)}}"
)
description = (
    'Weighted sum of two decreasing logistic transitions, with the initial response '
    'A fitted and the terminal response fixed at zero. The weight w is the first '
    'component\'s share of the initial response; t1 and t2 locate the transitions, '
    'and positive k1 and k2 control their steepness. Requires A >= 0 and 0 <= w <= 1. '
    'Examples include two-stage changes in mass remaining or retained mechanical '
    'properties. The components are empirical and need not represent physical phases.'
)


def _logistic_ratio(x, rate, midpoint):
    """Evaluate L(x)/L(0) without exponentiating large positive arguments."""
    if midpoint < 0.0:
        # Factor out exp(-rate*x) to avoid subtracting two large logarithms
        # when both the initial and current unnormalized responses are tiny.
        return np.exp(
            -rate * x + np.logaddexp(0.0, rate * midpoint)
            - np.logaddexp(0.0, rate * (midpoint - x))
        )
    return np.exp(
        np.logaddexp(0.0, -rate * midpoint)
        - np.logaddexp(0.0, rate * (x - midpoint))
    )


def model_function(x, A, w, t1, t2, k1, k2):
    """Return the two-transition response, with y(0) = A and y(infinity) = 0.

    Component labels are interchangeable: t1 is not constrained to precede t2.
    Transition times may be negative, allowing a transition to start before zero.
    """
    if not np.isfinite(A) or A < 0.0:
        raise ValueError('Logistic + Logistic requires a finite, nonnegative A.')
    if not (np.all(np.isfinite([w, t1, t2, k1, k2]))
            and 0.0 <= w <= 1.0 and k1 > 0.0 and k2 > 0.0):
        raise ValueError('Logistic + Logistic requires finite parameters, 0 <= w <= 1, and positive rates.')
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError('Logistic + Logistic requires finite time values.')
    # Zero coefficients are valid fit boundaries. Skip their components so
    # an unused transition cannot produce 0 * infinity before time zero.
    if A == 0.0:
        return np.zeros_like(x)
    if w == 0.0:
        return A * _logistic_ratio(x, k2, t2)
    if w == 1.0:
        return A * _logistic_ratio(x, k1, t1)
    return A * (w * _logistic_ratio(x, k1, t1)
                + (1.0 - w) * _logistic_ratio(x, k2, t2))


def _initial_guess_candidates(x, y, normalized=False):
    """Provide three feasible starts spanning different first-component shares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
        span = float(np.ptp(x)) or 1.0
        amplitude = 1.0 if normalized else max(float(np.max(y)), np.finfo(float).eps)
        fraction = y / amplitude
    else:
        x, span, amplitude = np.array([0.0, 1.0]), 1.0, 1.0
        fraction = np.array([1.0, 0.0])

    candidates = []
    for weight in (0.25, 0.5, 0.75):
        first = float(x[np.argmin(np.abs(fraction - (1.0 - weight / 2.0)))])
        second = float(x[np.argmin(np.abs(fraction - (1.0 - weight) / 2.0))])
        if first >= second:
            first, second = float(x[0]) + 0.25 * span, float(x[0]) + 0.75 * span
        parameters = [weight, first, second, 8.0 / span, 8.0 / span]
        if not normalized:
            parameters.insert(0, amplitude)
        candidates.append(parameters)
    return np.asarray(candidates, dtype=float)


def initial_guess(x, y):
    return _initial_guess_candidates(x, y)
