from sklearn.metrics import r2_score

display_name = "R²"
sort_order = 0
better_direction = 'higher'

def metric_function(y_true, y_pred):
    return r2_score(y_true, y_pred)
