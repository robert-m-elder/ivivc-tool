# models/__init__.py
import os
import importlib
import numpy as np
from scipy.optimize import curve_fit
import inspect

models = {}
approaches = {
    'approach1': {'display_name': 'Approach 1', 'models': []},
    'approach2': {'display_name': 'Approach 2', 'models': []},
    'approach3': {'display_name': 'Approach 3', 'models': []}
}

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

def generic_fit_model(model_function, x, y, p0=None, kwargs={}):
    param_names = list(inspect.signature(model_function).parameters.keys())[1:]
    popt, pcov = curve_fit(model_function, x, y, p0, **kwargs)
    
    return {
        'params': dict(zip(param_names, popt)),
        'pcov': pcov,
        'predict': lambda x_new: model_function(x_new, *popt)
    }

# Iterate through all .py files in the models directory
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'models.{module_name}')
        
        # Check if the module has the required attributes
        if hasattr(module, 'model_function'):
            models[module_name] = {
                'model_function': module.model_function,
                'display_name': getattr(module, 'display_name', module_name.replace('_', ' ').title())
            }
            if hasattr(module, 'fit_kwargs'):
                models[module_name].update({'fit_kwargs':module.fit_kwargs})
            else:
                models[module_name].update({'fit_kwargs':{}})
            fit_model = lambda x, y, p0=None, model_function=module.model_function, kwargs=models[module_name]['fit_kwargs']: generic_fit_model(model_function, x, y, p0, kwargs)
            models[module_name].update({'fit_model': fit_model})
            
            # Assign models to approaches
            for approach in getattr(module, 'approaches', []):
                if approach in approaches:
                    approaches[approach]['models'].append(module_name)

# Sort models within each approach
for approach in approaches.values():
    approach['models'].sort(key=lambda x: models[x]['display_name'])

