from sklearn.metrics import mean_absolute_error

display_name = "MAE"
sort_order = 4
better_direction = 'lower'

def metric_function(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
