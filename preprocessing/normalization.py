import numpy as np

def min_max_normalization(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def initial_value_normalization(x):
    """Normalize by the initial (first) value"""
    return x / x[0] if x[0] != 0 else x


def one_minus_response(x):
    """Reverse a fractional response after any selected normalizations."""
    return 1.0 - np.asarray(x, dtype=float)


normalizations = {
    'min_max': min_max_normalization,
    'initial_value': initial_value_normalization,
    'one_minus': one_minus_response,
}
