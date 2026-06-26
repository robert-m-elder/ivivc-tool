import numpy as np

display_name = "Exponential"
approaches = ['approach1','approach2']
fit_kwargs = {}
latex_equation = r"y = a e^{b x}"
description = (
    "Two-parameter exponential relationship. It can describe monotonic change "
    "that accelerates or decelerates exponentially over the fitted range."
)

def model_function(x, a, b):
    return a * np.exp(b * x)
