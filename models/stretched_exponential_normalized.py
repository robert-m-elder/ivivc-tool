"""Stretched exponential response with its amplitude fixed at one."""
from .stretched_exponential import model_function as _stretched_exponential, fit_kwargs

display_name = 'Stretched Exponential (Normalized)'
approaches = ['approach2']
latex_equation = r'y = e^{b x^{c}}'
description = (
    'Stretched exponential profile with amplitude fixed at 1 for responses already '
    'on a relative scale, such as fraction remaining or normalized molecular weight. '
    'The rate and stretching exponent remain fitted; selecting this model does '
    'not normalize data.'
)


def model_function(x, b, c):
    return _stretched_exponential(x, 1.0, b, c)
