import numpy as np

display_name = "Adjusted R²"
sort_order = 1
better_direction = 'higher'

def metric_function(y_true, y_pred, n_params=None):
    """
    Adjusted R-squared
    
    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k)]
    where n is sample size, k is number of parameters, R² is coefficient of determination
    
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
        Adjusted R-squared value (higher is better, max = 1.0)
    """
    if n_params is None:
        n_params = 2  # Default assumption: slope + intercept

    n = len(y_true)
    
    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    
    # Handle edge case where total variance is zero
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else float('-inf')
    
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate adjusted R-squared
    # Handle edge case where n <= n_params
    if n <= n_params:
        return float('-inf')
    
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params))
    
    return adj_r_squared

