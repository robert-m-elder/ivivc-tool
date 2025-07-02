import inspect
import numpy as np
from scipy.optimize import curve_fit

display_name = "Exponential"
approaches = ['approach1','approach2']
fit_kwargs = {}

def model_function(x, a, b):
    return a * np.exp(b * x)

#def fit_model(x, y, p0=None, kwargs=fit_kwargs):
#    param_names = list(inspect.signature(model_function).parameters.keys())[1:]
#    popt, pcov = curve_fit(model_function, x, y, p0, **kwargs)
#    
#    return {
#        'params': dict(zip(param_names, popt)),
#        'pcov': pcov,
#        'predict': lambda x_new: model_function(x_new, *popt)
#    }
