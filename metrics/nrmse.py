import numpy as np
from sklearn.metrics import root_mean_squared_error

display_name = "NRMSE"
sort_order = 4

def metric_function(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / np.std(y_true)

