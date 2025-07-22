import numpy as np
from sklearn.metrics import root_mean_squared_error

display_name = "NRMSE (std. dev.)"
sort_order = 4
better_direction = 'lower'

def metric_function(y_true, y_pred):
    """
    RMSE normalized by standard deviation
    sd-based NRMSE represent the ratio between the variation not explained by the regression vs the overall variation in Y. 
    If the regression explains all of the variation in Y, nothing gets unexplained and the RMSE, and consequently NRMSE is zero. 
    If the regression explains some part and leaves some other unexplained, which is at a similar scale than the overall variation, the ratio will be around 1. 
    Anything beyond will indicate a much greater variation or noise than in the variable itself and consequently a low predictability.
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    """
    return root_mean_squared_error(y_true, y_pred) / np.std(y_true)

