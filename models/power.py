import inspect
import numpy as np
from scipy.optimize import curve_fit

display_name = "Power"
approaches = ['approach1','approach2']
fit_kwargs = {}

def model_function(x, a, b):
    return a * x**b

