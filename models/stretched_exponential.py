import numpy as np

display_name = "Stretched Exponential"
approaches = ['approach2']
fit_kwargs = {'maxfev':10000, 'ftol':1e-5, 'xtol':1e-5}
latex_equation = r"y = a e^{b x^{c}}"
description = (
    "Exponential relationship with a fitted stretching exponent. It can describe "
    "non-single-rate behavior, but the additional shape parameter can increase "
    "sensitivity when data are sparse."
)

def model_function(x, a, b, c):
    return a * np.exp(b * x**c)
