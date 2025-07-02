import numpy as np

def log_scaling(x):
    return np.log10(x)

scalings = {
    'log_x': log_scaling,
    'log_y': log_scaling
}
