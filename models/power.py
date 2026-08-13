display_name = "Power"
approaches = ['approach1']
fit_kwargs = {}
latex_equation = r"y = a x^{b}"
description = (
    "Power-law empirical relationship. It can describe nonlinear monotonic "
    "relationships where the response changes as a fitted power of time or the "
    "mapped predictor."
)

def model_function(x, a, b):
    return a * x**b
