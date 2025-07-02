import inspect
import numpy as np
from scipy.optimize import curve_fit

display_name = "Linear"
approaches = ['approach1']
fit_kwargs = {}

def model_function(x, a, b):
    return a + b * x

