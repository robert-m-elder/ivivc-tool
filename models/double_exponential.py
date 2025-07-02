import numpy as np

display_name = "Double Exponential"
approaches = ['approach1','approach2']
fit_kwargs = {'maxfev':10000, 'ftol':1e-5, 'xtol':1e-5}

def model_function(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

