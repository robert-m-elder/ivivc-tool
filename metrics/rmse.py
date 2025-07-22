from sklearn.metrics import root_mean_squared_error

display_name = "RMSE"
sort_order = 3
better_direction = 'lower'

def metric_function(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

