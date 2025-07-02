# models/evaluation.py

import warnings
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import scipy as sp
import scipy.optimize
import inspect
from itertools import product
from metrics import metrics

cross_validation_schemes = {
                            'approach1':sklearn.model_selection.ShuffleSplit(n_splits=50, test_size=0.2, random_state=12345),
                            'approach2':sklearn.model_selection.ShuffleSplit(n_splits=50, test_size=0.2, random_state=12345),
                            #'approach2':sklearn.model_selection.TimeSeriesSplit(n_splits=5)
                           }

def calc_nrmse(y_true,y_pred):
    """
    RMSE normalized by standard deviation
    sd-based NRMSE represent the ratio between the variation not explained by the regression vs the overall variation in Y. 
    If the regression explains all of the variation in Y, nothing gets unexplained and the RMSE, and consequently NRMSE is zero. 
    If the regression explains some part and leaves some other unexplained, which is at a similar scale than the overall variation, the ratio will be around 1. 
    Anything beyond will indicate a much greater variation or noise than in the variable itself and consequently a low predictability.
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    """
    return root_mean_squared_error(y_true,y_pred) / np.std(y_true)

def evaluate_goodness_of_fit(y_true, y_pred):
    """
    Evaluate the goodness of fit for a model.
    
    Args:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    
    Returns:
    dict: A dictionary containing various goodness-of-fit metrics
    """
    r_squared = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = calc_nrmse(y_true,y_pred)
    
    return {
        'r_squared': r_squared,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse
    }

def auto_grid_search(func, x, y, param_min=1e-6, param_max=1e6, num_points=25):
    """
    Perform an automatic grid search for the best parameters of a given function,
    including both positive and negative log-scaled values.

    Parameters:
    func : callable
        The function to optimize. Should take x as the first argument,
        followed by the parameters to be optimized.
    x : array_like
        The x data.
    y : array_like
        The y data.
    num_points : int, optional
        Number of points to sample for each parameter (default is 20).
    range_scale : float, optional
        Scale factor for parameter ranges (default is 1, giving a range of -1e5 to 1e5).

    Returns:
    best_params : ndarray
        The best parameters found.
    """
    # Automatically determine the number of parameters
    num_params = len(inspect.signature(func).parameters) - 1  # subtract 1 for 'x'

    # Generate log-spaced ranges for each parameter, including negative values
    tmp_range = np.logspace(np.log10(param_min), np.log10(param_max), num=num_points)
    full_range = np.concatenate([-tmp_range[::-1], [0], tmp_range])
    allparams = np.array(list(product(*[full_range]*num_params)))

    # Define the objective function for scipy.optimize.brute
    def objective(params):
        # Convert params from log space, preserving signs
        #actual_params = np.sign(params) * 10**np.abs(params)
        return np.sum((y - func(x, *params))**2)

    # Perform the grid search
    results = np.apply_along_axis(objective, 1, allparams) 

    # Get best set of parameters, handling nan and inf
    results[~np.isfinite(results)] = np.inf
    best_params = allparams[np.nanargmin(results)]

    return best_params

def cross_validation_curve_fit(x_data, y_data, model_function, cv, selected_metrics, p0=None, kwargs={}):
    scores = {metric: [] for metric in selected_metrics}
    #scores = {metric: metrics[metric]['function'](y2, y_pred2) for metric in selected_metrics}
    #scores = {k:[] for k in evaluate_goodness_of_fit([0],[0])}
    for train_index, test_index in cv.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            popt, pcov = sp.optimize.curve_fit(model_function, x_train, y_train, p0, **kwargs)
        y_pred = model_function(x_test, *popt)
        tmp_scores = {metric: metrics[metric]['function'](y_test, y_pred) for metric in selected_metrics}
        #tmp_scores = evaluate_goodness_of_fit(y_test, y_pred)
        for k,v in tmp_scores.items():
            scores[k].append(v)
    scores = {k:np.array(scores[k]) for k in scores}
    return scores

## time constant, generic function
def calculate_tau(model_function, params):
    # Get the initial value
    f_0 = model_function(0, *params)
    # Integrate from 0 to infinity
    integral, _ = sp.integrate.quad(model_function, 0, np.inf, args=params)
    # Normalize by dividing by the initial value
    tau = integral / f_0
    return tau

## time constant
def get_tau(modelname,popt):
    if modelname == 'double exponential':
        # weighted average -- note form of multi exponential is a*exp(-b*x), not a*exp(-b/x)
        tau = (popt[0]/popt[1]+popt[2]/popt[3])/(popt[0]+popt[2])
    elif modelname == 'multi exponential':
        # weighted average over all exponential terms
        num = sum([popt[i]/popt[i+1] for i in range(0, len(popt), 2)])
        den = sum([popt[i] for i in range(0, len(popt), 2)])
        tau = num/den
    elif modelname == 'linear':
        ## NOTE questionable to get time constant from linear fit, but seems to work OK for relative tau/tau
        tau = np.abs(popt[0]/popt[1])
    elif modelname == 'stretched exponential':
        tau,beta = popt[1],popt[2]
        tau = (tau/beta*unc.wrap(sp.special.gamma)(1.0/beta))
    elif modelname == 'exponential':
        tau = 1/popt[1]
    else:
        tau = np.nan
    return tau

