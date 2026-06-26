display_name = "Linear (zero intercept)"
approaches = ['approach1']
fit_kwargs = {}
latex_equation = r"y = a x"
description = (
    "Straight-line empirical relationship constrained to pass through zero. It "
    "may be useful when a zero in vitro value should correspond to a zero in vivo "
    "value under the selected preprocessing and response scale."
)

def model_function(x, a):
    return a * x
