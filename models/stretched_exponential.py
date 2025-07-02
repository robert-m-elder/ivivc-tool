import inspect
import numpy as np
from scipy.optimize import curve_fit

display_name = "Stretched Exponential"
approaches = ['approach2']
fit_kwargs = {'maxfev':10000, 'ftol':1e-5, 'xtol':1e-5}

def model_function(x, a, b, c):
    return a * np.exp(b * x**c)

