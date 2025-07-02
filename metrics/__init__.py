import os
import importlib

metrics = {}

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through all .py files in the metrics directory
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'metrics.{module_name}')
        
        if hasattr(module, 'metric_function') and hasattr(module, 'display_name'):
            metrics[module_name] = {
                'function': module.metric_function,
                'display_name': module.display_name,
                'sort_order': getattr(module, 'sort_order', float('inf'))  # Use infinity if sort_order is not specified
            }

# Sort metrics by sort_order first, then by display_name
metrics = dict(sorted(metrics.items(), key=lambda x: (x[1]['sort_order'], x[1]['display_name'])))
