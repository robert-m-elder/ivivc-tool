import os
import importlib
import inspect

metrics = {}

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through all .py files in the metrics directory
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'metrics.{module_name}')
        
        if hasattr(module, 'metric_function') and hasattr(module, 'display_name'):
            # Check if function accepts n_params parameter
            sig = inspect.signature(module.metric_function)
            accepts_n_params = 'n_params' in sig.parameters
            metrics[module_name] = {
                'function': module.metric_function,
                'display_name': module.display_name,
                'accepts_n_params': accepts_n_params,
                'sort_order': getattr(module, 'sort_order', float('inf'))  # Use infinity if sort_order is not specified
            }

# Sort metrics by sort_order first, then by display_name
metrics = dict(sorted(metrics.items(), key=lambda x: (x[1]['sort_order'], x[1]['display_name'])))

def calculate_metric(metric_name, y_true, y_pred, n_params=None):
    """
    Calculate a metric with optional parameter count support

    Parameters:
    -----------
    metric_name : str
        Name of the metric to calculate
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_params : int, optional
        Number of model parameters (for AIC, BIC, etc.)

    Returns:
    --------
    float
        Calculated metric value
    """
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric_info = metrics[metric_name]
    metric_func = metric_info['function']

    # Call function with n_params if it accepts it, otherwise without
    if metric_info['accepts_n_params'] and n_params is not None:
        return metric_func(y_true, y_pred, n_params=n_params)
    else:
        return metric_func(y_true, y_pred)

