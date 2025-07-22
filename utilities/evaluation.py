# models/evaluation.py

import warnings
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import scipy as sp
import scipy.optimize
import scipy.stats
import inspect
from itertools import product
from metrics import metrics, calculate_metric

from joblib import Parallel, delayed
import multiprocessing

cross_validation_schemes = {
                            #'approach1':sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=12345),
                            #'approach2':sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=12345),
                            #'approach1':sklearn.model_selection.LeaveOneOut(),
                            #'approach2':sklearn.model_selection.LeaveOneOut(),
                            'approach1':sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.25, random_state=12345),
                            'approach2':sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.25, random_state=12345),
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

def auto_grid_search(func, x, y, param_min=1e-6, param_max=1e6, num_points=100, num_cores=None):
    num_params = len(inspect.signature(func).parameters) - 1  # subtract 1 for 'x'
    
    # Define the objective function with warning filter
    def objective(params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            #return np.sum((y - func(x, *params))**2)
            try:
                pred = func(x, *params)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    return np.inf
                return np.sum((y - pred)**2)
            except:
                return np.inf
    
    # Generate random starting points (log-uniform distribution)
    log_min, log_max = np.log10(param_min), np.log10(param_max) 
    random_starts = np.power(10, np.random.uniform(log_min, log_max, size=(num_points, num_params)))
    
    # Add negative values
    random_starts *= np.random.choice([-1, 1], size=random_starts.shape)

    # Wrapper function for minimize to catch warnings
    def minimize_wrapper(x0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            #return sp.optimize.minimize(objective, x0, method='BFGS', tol=1e-4)
            return sp.optimize.minimize(objective, x0, method='Nelder-Mead')

    # Parallel optimization with warning filter
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = Parallel(n_jobs=num_cores)(
            delayed(minimize_wrapper)(x0) for x0 in random_starts
        )
    
    # Find the best result
    best_result = min(results, key=lambda x: x.fun)
    
    return best_result.x

def auto_grid_search_old(func, x, y, param_min=1e-6, param_max=1e6, num_points=25):
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
    
    if num_params>2: num_points = num_points // 2

    # Generate log-spaced ranges for each parameter, including negative values
    tmp_range = np.logspace(np.log10(param_min), np.log10(param_max), num=num_points)
    full_range = np.concatenate([-tmp_range[::-1], [0], tmp_range])
    allparams = np.array(list(product(*[full_range]*num_params)))

    # Define the objective function 
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
    num_params = len(inspect.signature(model_function).parameters) - 1  # subtract 1 for 'x'
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
        tmp_scores = {metric: calculate_metric(metric, y_test, y_pred, num_params) for metric in selected_metrics}
        #tmp_scores = {metric: metrics[metric]['function'](y_test, y_pred) for metric in selected_metrics}
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

def calculate_tau_with_uncertainty(model_function, params, pcov, num_samples=50):
    # Generate samples from the multivariate normal distribution
    param_samples = np.random.multivariate_normal(params, pcov, num_samples)
    tau_samples = []
    for sample in param_samples:
        # Get the initial value
        f_0 = model_function(0, *sample)
        # Integrate from 0 to infinity
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sp.integrate.IntegrationWarning)
            integral, _ = sp.integrate.quad(model_function, 0, np.inf, args=tuple(sample), limit=100)
        # Normalize by dividing by the initial value
        tau = integral / f_0
        tau_samples.append(tau)
    # Calculate mean and standard deviation of tau
    tau_mean = np.mean(tau_samples)
    tau_std = np.std(tau_samples)
    return tau_mean, tau_std

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

def generate_confidence_bands(model_function, x_values, params, pcov, confidence_level=0.95, n_samples=1000):
    """
    Generate prediction bands using Monte Carlo sampling of parameter uncertainty

    Parameters:
    -----------
    model_function : callable
        The model function to evaluate
    x_values : array-like
        X values for prediction
    params : dict
        Best-fit parameters
    pcov : array-like
        Parameter covariance matrix
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    dict with 'lower', 'upper', 'mean' prediction bands
    """
    #import numpy as np
    #from scipy.stats import multivariate_normal

    # Convert params to array in consistent order
    param_names = list(params.keys())
    param_values = np.array([params[name] for name in param_names])

    # Generate parameter samples from multivariate normal distribution
    param_samples = sp.stats.multivariate_normal.rvs(mean=param_values, cov=pcov, size=n_samples)

    # Ensure param_samples is 2D even for single parameter
    if param_samples.ndim == 1:
        param_samples = param_samples.reshape(-1, 1)

    # Generate predictions for each parameter sample
    predictions = []
    for sample in param_samples:
        sample_params = dict(zip(param_names, sample))
        try:
            y_pred = model_function(x_values, **sample_params)
            predictions.append(y_pred)
        except:
            # Skip invalid parameter combinations
            continue

    predictions = np.array(predictions)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    return {
        'lower': np.percentile(predictions, lower_percentile, axis=0),
        'upper': np.percentile(predictions, upper_percentile, axis=0),
        'mean': np.mean(predictions, axis=0)
    }

def generate_prediction_bands(model_function, x_values, params, pcov, residuals=None, confidence_level=0.95, n_samples=1000):
    """
    Generate prediction bands that include both parameter uncertainty and residual variance

    Parameters:
    -----------
    model_function : callable
        The model function to evaluate
    x_values : array-like
        X values for prediction
    params : dict
        Best-fit parameters
    pcov : array-like
        Parameter covariance matrix
    residuals : array-like, optional
        Residuals from the fit (y_true - y_pred). If None, will estimate from parameter uncertainty only
    confidence_level : float
        Confidence level (default 0.95 for 95% prediction interval)
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    dict with 'lower', 'upper', 'mean' prediction bands
    """
    #import numpy as np
    #from scipy.stats import multivariate_normal, t

    # Convert params to array in consistent order
    param_names = list(params.keys())
    param_values = np.array([params[name] for name in param_names])

    # Generate parameter samples from multivariate normal distribution
    param_samples = sp.stats.multivariate_normal.rvs(mean=param_values, cov=pcov, size=n_samples)

    # Ensure param_samples is 2D even for single parameter
    if param_samples.ndim == 1:
        param_samples = param_samples.reshape(-1, 1)

    # Generate predictions for each parameter sample
    predictions = []
    for sample in param_samples:
        sample_params = dict(zip(param_names, sample))
        try:
            y_pred = model_function(x_values, **sample_params)
            predictions.append(y_pred)
        except:
            # Skip invalid parameter combinations
            continue

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)

    # Estimate residual standard error
    if residuals is not None:
        # Use provided residuals
        residual_std = np.std(residuals, ddof=len(param_values))
    else:
        # Estimate from parameter uncertainty (this is less accurate)
        residual_std = np.std(predictions, axis=0).mean()
        print("Warning: No residuals provided. Using parameter uncertainty to estimate residual variance.")

    # For prediction intervals, we need to account for:
    # 1. Parameter uncertainty (captured in the Monte Carlo samples)
    # 2. Residual variance (the inherent scatter in the data)

    # Calculate prediction variance = model uncertainty + residual variance
    model_variance = np.var(predictions, axis=0)
    total_variance = model_variance + residual_std**2
    prediction_std = np.sqrt(total_variance)

    # Use t-distribution for small sample sizes
    # Degrees of freedom = number of data points - number of parameters
    if residuals is not None:
        dof = len(residuals) - len(param_values)
    else:
        dof = n_samples - len(param_values)  # Conservative estimate

    # Calculate prediction intervals using t-distribution
    alpha = 1 - confidence_level
    t_value = sp.stats.t.ppf(1 - alpha/2, dof) if dof > 0 else 1.96  # Fall back to normal if dof <= 0

    margin_of_error = t_value * prediction_std

    return {
        'lower': mean_prediction - margin_of_error,
        'upper': mean_prediction + margin_of_error,
        'mean': mean_prediction,
        'std': prediction_std
    }

