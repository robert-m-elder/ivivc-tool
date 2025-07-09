import numpy as np

def min_max_normalization(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def initial_value_normalization(x):
    """Normalize by the initial (first) value"""
    return x / x[0] if x[0] != 0 else x

normalizations = {
    'min_max': min_max_normalization,
    'initial_value': initial_value_normalization
}
