import numpy as np

display_name = "Exponential"
approaches = ['approach1','approach2']
fit_kwargs = {}

def model_function(x, a, b):
    return a * np.exp(b * x)

