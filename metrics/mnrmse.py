import numpy as np
from sklearn.metrics import root_mean_squared_error

display_name = "NRMSE (mean)"
sort_order = 4
better_direction = 'lower'

def metric_function(y_true, y_pred):
    """
    RMSE normalized by mean -- more suitable for small (or single point) sample sizes
    """
    return root_mean_squared_error(y_true, y_pred) / np.mean(y_true)

