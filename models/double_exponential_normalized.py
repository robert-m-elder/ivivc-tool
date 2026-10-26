"""Two exponential components whose coefficients sum to one."""
from .double_exponential import model_function as _double_exponential, fit_kwargs

display_name = 'Double Exponential (Normalized)'
approaches = ['approach2']
latex_equation = r'y = w e^{b x} + (1-w)e^{d x}'
description = (
    'Two exponential components with coefficients w and 1-w, fixing the response '
    'at time zero to 1. Examples include fraction remaining or retained properties '
    'with two apparent rates. The coefficient and both rates remain fitted; '
    'selecting this model does not normalize data.'
)


def model_function(x, w, b, d):
    return _double_exponential(x, w, b, 1.0 - w, d)
