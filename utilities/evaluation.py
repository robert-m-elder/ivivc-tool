# models/evaluation.py

import warnings
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import scipy as sp
import inspect
from metrics import calculate_metric

from joblib import Parallel, delayed
import multiprocessing

def make_cross_validator(config, approach_id=None):
    """Create a scikit-learn cross-validator from user/app configuration."""
    scheme = config.get('cv_scheme', 'shuffle_split')
    n_splits = int(config.get('cv_n_splits', 20))
    random_state = config.get('cv_random_state', 12345)
    random_state = None if random_state in (None, '') else int(random_state)

    if scheme == 'shuffle_split':
        return sklearn.model_selection.ShuffleSplit(
            n_splits=n_splits,
            test_size=float(config.get('cv_test_size', 0.5)),
            random_state=random_state,
        )
    if scheme == 'kfold':
        return sklearn.model_selection.KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    if scheme == 'leave_one_out':
        return sklearn.model_selection.LeaveOneOut()
    if scheme == 'none':
        return None

    raise ValueError(f"Unknown cross-validation scheme: {scheme}")

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

def _sample_log_uniform_real_space(param_min, param_max, size, rng):
    """Sample real-space values with logarithmic spacing by magnitude.

    The user-supplied bounds are interpreted directly in real space. For
    ranges that cross zero, negative and positive values are sampled from the
    portions of the interval that are available. Zero itself cannot be sampled
    logarithmically, so an internal lower magnitude based on the span is used
    as the closest nonzero value.
    """
    param_min = float(param_min)
    param_max = float(param_max)
    if not np.isfinite(param_min) or not np.isfinite(param_max):
        raise ValueError('Grid-search parameter bounds must be finite numbers.')
    if param_min >= param_max:
        raise ValueError('Grid-search parameter minimum must be less than the maximum.')

    span = param_max - param_min
    magnitude_floor = max(abs(span) * 1e-12, np.finfo(float).tiny)

    def sample_magnitude(upper_magnitude, draw_size):
        upper_magnitude = float(abs(upper_magnitude))
        if upper_magnitude <= 0:
            return np.zeros(draw_size)
        lower_magnitude = min(magnitude_floor, upper_magnitude)
        if np.isclose(lower_magnitude, upper_magnitude):
            return np.full(draw_size, upper_magnitude)
        return np.exp(rng.uniform(np.log(lower_magnitude), np.log(upper_magnitude), size=draw_size))

    if param_min > 0:
        lower_magnitude = max(param_min, magnitude_floor)
        return np.exp(rng.uniform(np.log(lower_magnitude), np.log(param_max), size=size))

    if param_max < 0:
        magnitudes = sample_magnitude(abs(param_min), size)
        lower_allowed = abs(param_max)
        if lower_allowed > 0:
            magnitudes = np.maximum(magnitudes, lower_allowed)
        return -magnitudes

    # Range crosses zero. Select sign using the amount of log-magnitude space
    # available on each side, then sample the magnitude for that side.
    negative_max = abs(param_min)
    positive_max = abs(param_max)
    negative_weight = max(np.log(negative_max / magnitude_floor), 0.0) if negative_max > 0 else 0.0
    positive_weight = max(np.log(positive_max / magnitude_floor), 0.0) if positive_max > 0 else 0.0
    total_weight = negative_weight + positive_weight

    if total_weight == 0:
        return np.zeros(size)

    choose_negative = rng.random(size) < (negative_weight / total_weight)
    starts = np.empty(size, dtype=float)
    if np.any(choose_negative):
        starts[choose_negative] = -sample_magnitude(negative_max, np.count_nonzero(choose_negative))
    if np.any(~choose_negative):
        starts[~choose_negative] = sample_magnitude(positive_max, np.count_nonzero(~choose_negative))
    return starts


def auto_grid_search(func, x, y, param_min=-1e6, param_max=1e6, num_points=100, num_cores=None, random_state=12345, initial_points=None):
    num_params = len(inspect.signature(func).parameters) - 1  # subtract 1 for 'x'
    rng = np.random.default_rng(None if random_state in (None, '') else int(random_state))
    
    # Define the objective function with warning filter
    def objective(params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                pred = func(x, *params)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    return np.inf
                return np.sum((y - pred)**2)
            except Exception:
                return np.inf
    
    # Generate random starting points. Bounds are interpreted directly in real
    # space, while magnitudes are sampled logarithmically within those bounds.
    random_starts = _sample_log_uniform_real_space(param_min, param_max, (num_points, num_params), rng)

    # Include any model-specific/data-driven starting points alongside the
    # sampled points. These are useful for higher-parameter models where a fully
    # generic random search may otherwise waste many candidates in implausible
    # regions of parameter space.
    if initial_points is not None:
        initial_points = np.atleast_2d(np.asarray(initial_points, dtype=float))
        if initial_points.shape[1] != num_params:
            raise ValueError('Model-specific initial points have the wrong number of parameters.')
        initial_points = initial_points[np.all(np.isfinite(initial_points), axis=1)]
        if len(initial_points):
            random_starts = np.vstack([initial_points, random_starts])

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

def cross_validation_curve_fit(x_data, y_data, model_function, cv, selected_metrics, p0=None, kwargs={}):
    num_params = len(inspect.signature(model_function).parameters) - 1  # subtract 1 for 'x'
    scores = {metric: [] for metric in selected_metrics}
    if cv is None:
        return {metric: np.array([np.nan]) for metric in selected_metrics}
    for train_index, test_index in cv.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            popt, pcov = sp.optimize.curve_fit(model_function, x_train, y_train, p0, **kwargs)
        y_pred = model_function(x_test, *popt)
        tmp_scores = {metric: calculate_metric(metric, y_test, y_pred, num_params) for metric in selected_metrics}
        #tmp_scores = evaluate_goodness_of_fit(y_test, y_pred)
        for k,v in tmp_scores.items():
            scores[k].append(v)
    scores = {k:np.array(scores[k]) for k in scores}
    return scores

## time constant, generic function
def calculate_tau(model_function, params):
    if getattr(model_function, 'supports_tau', True) is False:
        return np.nan
    # Get the initial value
    f_0 = model_function(0, *params)
    # Integrate from 0 to infinity
    integral, _ = sp.integrate.quad(model_function, 0, np.inf, args=params)
    # Normalize by dividing by the initial value
    tau = integral / f_0
    return tau

def calculate_tau_with_uncertainty(model_function, params, pcov, num_samples=50):
    if getattr(model_function, 'supports_tau', True) is False:
        return np.nan, np.nan
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
        except Exception:
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
        except Exception:
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

def generate_ratio_prediction_bands(model1_function, model2_function, x_values,
                                  params1, pcov1, params2, pcov2,
                                  residuals1=None, residuals2=None,
                                  confidence_level=0.95, n_samples=1000):
    """
    Generate prediction bands for the ratio of two models (model2 / model1)

    Parameters:
    -----------
    model1_function : callable
        The first model function (denominator)
    model2_function : callable
        The second model function (numerator)
    x_values : array-like
        X values for prediction
    params1 : dict
        Best-fit parameters for model1
    pcov1 : array-like
        Parameter covariance matrix for model1
    params2 : dict
        Best-fit parameters for model2
    pcov2 : array-like
        Parameter covariance matrix for model2
    residuals1 : array-like, optional
        Residuals from model1 fit
    residuals2 : array-like, optional
        Residuals from model2 fit
    confidence_level : float
        Confidence level (default 0.95 for 95% prediction interval)
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    dict with 'lower', 'upper', 'mean' prediction bands for the ratio
    """

    # Convert params to arrays in consistent order
    param_names1 = list(params1.keys())
    param_values1 = np.array([params1[name] for name in param_names1])

    param_names2 = list(params2.keys())
    param_values2 = np.array([params2[name] for name in param_names2])

    # Generate parameter samples from multivariate normal distributions
    param_samples1 = sp.stats.multivariate_normal.rvs(mean=param_values1, cov=pcov1, size=n_samples)
    param_samples2 = sp.stats.multivariate_normal.rvs(mean=param_values2, cov=pcov2, size=n_samples)

    # Ensure param_samples are 2D even for single parameter
    if param_samples1.ndim == 1:
        param_samples1 = param_samples1.reshape(-1, 1)
    if param_samples2.ndim == 1:
        param_samples2 = param_samples2.reshape(-1, 1)

    # Generate predictions for each parameter sample
    ratios = []
    valid_samples = 0

    for i in range(n_samples):
        sample_params1 = dict(zip(param_names1, param_samples1[i]))
        sample_params2 = dict(zip(param_names2, param_samples2[i]))

        try:
            y_pred1 = model1_function(x_values, **sample_params1)
            y_pred2 = model2_function(x_values, **sample_params2)

            # Calculate ratio, avoiding division by zero
            # Add small epsilon to denominator to avoid numerical issues
            epsilon = 1e-4
            y_pred1[y_pred1<epsilon] = epsilon
            ratio = y_pred2 / (y_pred1 + epsilon)

            # Filter out unreasonable ratios (optional - adjust thresholds as needed)
            if np.all(np.isfinite(ratio)) and np.all(y_pred1 >= epsilon):
                ratios.append(ratio)
                valid_samples += 1

        except Exception as e:
            # Skip invalid parameter combinations
            print(e)
            continue

    if valid_samples == 0:
        raise ValueError("No valid parameter combinations found. Check your models and parameter ranges.")

    print(f"Used {valid_samples} out of {n_samples} samples ({100*valid_samples/n_samples:.1f}%)")

    ratios = np.array(ratios)
    mean_ratio = np.mean(ratios, axis=0)

    # Estimate residual standard error for the ratio
    # This is more complex for ratios - we need to propagate uncertainty
    if residuals1 is not None and residuals2 is not None:
        # Calculate residual standard errors for individual models
        residual_std1 = np.std(residuals1, ddof=len(param_values1))
        residual_std2 = np.std(residuals2, ddof=len(param_values2))

        # For ratio uncertainty propagation: if R = Y2/Y1, then
        # Var(R) ≈ R² * [(σ₁/Y1)² + (σ₂/Y2)²] for uncorrelated errors
        # We'll use the mean predictions to estimate this
        y_mean1 = np.mean([model1_function(x_values, **params1)])
        y_mean2 = np.mean([model2_function(x_values, **params2)])

        # Relative errors
        rel_error1 = residual_std1 / np.abs(y_mean1)
        rel_error2 = residual_std2 / np.abs(y_mean2)

        # Propagated relative error for ratio
        rel_error_ratio = np.sqrt(rel_error1**2 + rel_error2**2)
        residual_std_ratio = rel_error_ratio * np.abs(mean_ratio)

    else:
        # Estimate from Monte Carlo samples (less accurate)
        residual_std_ratio = np.std(ratios, axis=0).mean()
        print("Warning: No residuals provided. Using Monte Carlo uncertainty to estimate residual variance.")

    # Calculate prediction variance = model uncertainty + residual variance
    model_variance = np.var(ratios, axis=0)
    total_variance = model_variance + residual_std_ratio**2
    prediction_std = np.sqrt(total_variance)

    # Degrees of freedom calculation
    if residuals1 is not None and residuals2 is not None:
        # Use minimum of the two models' degrees of freedom (conservative)
        dof1 = len(residuals1) - len(param_values1)
        dof2 = len(residuals2) - len(param_values2)
        dof = min(dof1, dof2)
    else:
        dof = valid_samples - len(param_values1) - len(param_values2)  # Conservative estimate

    # Calculate prediction intervals
    alpha = 1 - confidence_level
    t_value = sp.stats.t.ppf(1 - alpha/2, dof) if dof > 0 else 1.96

    margin_of_error = t_value * prediction_std

    return {
        'lower': mean_ratio - margin_of_error,
        'upper': mean_ratio + margin_of_error,
        'mean': mean_ratio,
        'std': prediction_std,
        #'valid_samples': valid_samples,
        #'total_samples': n_samples
    }

