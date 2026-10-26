"""Exponential response with its amplitude fixed at one."""
from .exponential import model_function as _exponential

display_name = 'Exponential (Normalized)'
approaches = ['approach2']
fit_kwargs = {}
latex_equation = r'y = e^{b x}'
description = (
    'Exponential profile with amplitude fixed at 1 for responses already on a '
    'relative scale, such as fraction remaining or retained mechanical properties. '
    'Only the rate parameter is fitted; selecting this model does not normalize data.'
)


def model_function(x, b):
    return _exponential(x, 1.0, b)
