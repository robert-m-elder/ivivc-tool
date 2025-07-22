import numpy as np

display_name = "Stretched Exponential"
approaches = ['approach2']
fit_kwargs = {'maxfev':10000, 'ftol':1e-5, 'xtol':1e-5}

def model_function(x, a, b, c):
    return a * np.exp(b * x**c)
    #return a * np.exp(b * np.sign(x) * np.abs(x)**c)

