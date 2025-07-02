from sklearn.metrics import mean_squared_error

display_name = "MSE"
sort_order = 2

def metric_function(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
