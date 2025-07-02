from sklearn.metrics import r2_score

display_name = "R-squared"
sort_order = 1

def metric_function(y_true, y_pred):
    return r2_score(y_true, y_pred)
