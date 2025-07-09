import numpy as np

display_name = "AIC"
sort_order = 5

def metric_function(y_true, y_pred, n_params=None):
    """
    Akaike Information Criterion (AIC)
    
    AIC = n * ln(RSS/n) + 2k
    where n is sample size, RSS is residual sum of squares, k is number of parameters
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like  
        Predicted values
    n_params : int, optional
        Number of parameters in the model (defaults to 2 if not provided)
    
    Returns:
    --------
    float
        AIC value (lower is better)
    """
    if n_params is None:
        n_params = 2  # Default assumption: slope + intercept
    
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    
    # Handle edge case where RSS is zero or very small
    if rss <= 0:
        rss = np.finfo(float).eps
    
    aic = n * np.log(rss / n) + 2 * n_params
    
    return aic

