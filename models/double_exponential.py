import numpy as np

display_name = "Double Exponential"
approaches = ['approach1','approach2']
fit_kwargs = {'maxfev':10000, 'ftol':1e-5, 'xtol':1e-5}
latex_equation = r"y = a e^{b x} + c e^{d x}"
description = (
    "Sum of two exponential terms. It can represent profiles with two apparent "
    "rates, but the added flexibility may require more data to estimate stably."
)

def model_function(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)
