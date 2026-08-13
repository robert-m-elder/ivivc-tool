import numpy as np


def log_scaling(x):
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Log10 scaling requires all values being transformed to be greater than zero.')
    return np.log10(values)


scalings = {
    'log_x': log_scaling,
    'log_y': log_scaling
}
