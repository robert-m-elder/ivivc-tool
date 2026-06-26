display_name = "Linear"
approaches = ['approach1']
fit_kwargs = {}
latex_equation = r"y = a + b x"
description = (
    "Straight-line empirical relationship with an intercept and slope. It is "
    "appropriate when the in vitro/in vivo relationship is approximately linear "
    "over the fitted range."
)

def model_function(x, a, b):
    return a + b * x
