import numpy as np

def min_max_normalization(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def z_score_normalization(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

normalizations = {
    'min_max': min_max_normalization,
    'z_score': z_score_normalization
}
