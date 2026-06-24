import numpy as np

display_name = "BIC"
sort_order = 7
better_direction = 'lower'

def metric_function(y_true, y_pred, n_params=None):
    """
    Bayesian Information Criterion (BIC)
    
    BIC = n * ln(RSS/n) + k * ln(n)
    where n is sample size, RSS is residual sum of squares, k is number of parameters
    """
    if n_params is None:
        n_params = 2  # Default assumption: slope + intercept
    
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    
    if rss <= 0:
        rss = np.finfo(float).eps
    
    bic = n * np.log(rss / n) + n_params * np.log(n)
    
    return bic

