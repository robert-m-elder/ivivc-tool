"""Two decreasing logistic transitions with the initial response fixed at one."""

import numpy as np

from .logistic_logistic import (
    _initial_guess_candidates,
    fit_kwargs as _standard_fit_kwargs,
    model_function as _standard_model,
)


display_name = 'Logistic + Logistic (Normalized)'
approaches = ['approach2']
fit_kwargs = {
    **_standard_fit_kwargs,
    'bounds': ([0.0, -np.inf, -np.inf, np.finfo(float).tiny, np.finfo(float).tiny],
               [1.0, np.inf, np.inf, np.inf, np.inf]),
}
latex_equation = (
    r"y = w\frac{L(x;k_1,t_1)}{L(0;k_1,t_1)}"
    r"+(1-w)\frac{L(x;k_2,t_2)}{L(0;k_2,t_2)},\quad "
    r"L(x;k,t)=\frac{1}{1+e^{k(x-t)}}"
)
description = (
    'Weighted sum of two decreasing logistic transitions with the initial response '
    'fixed at 1 and the terminal response fixed at zero. The weight w is the first '
    'component\'s initial contribution; t1 and t2 locate the transitions, and positive '
    'k1 and k2 control their steepness. Requires 0 <= w <= 1. Examples include '
    'fraction remaining or normalized retained properties with two transitions. '
    'The components are empirical and need not represent physical phases. '
    'Selecting this model does not normalize the input data.'
)


def model_function(x, w, t1, t2, k1, k2):
    """Return the normalized response with y(0) = 1 and y(infinity) = 0."""
    return _standard_model(x, 1.0, w, t1, t2, k1, k2)


def initial_guess(x, y):
    return _initial_guess_candidates(x, y, normalized=True)
