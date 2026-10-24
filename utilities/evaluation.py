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


class LeaveContiguousBlockOut:
    """Hold out contiguous blocks in x-axis order.

    The block size is interpreted as the number of adjacent observations to hold
    out. Splits are generated with stride 1, so a 10-point dataset and block size
    2 produces 9 train/test splits.
    """

    def __init__(self, block_size=2):
        self.block_size = max(1, int(block_size))

    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        n = len(X)
        if n == 0:
            return
        order = np.argsort(X)
        block_size = min(self.block_size, max(1, n - 1))
        all_indices = np.arange(n)
        for start in range(0, n - block_size + 1):
            test_index = order[start:start + block_size]
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_index] = False
            train_index = all_indices[train_mask]
            if len(train_index) == 0 or len(test_index) == 0:
                continue
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            return None
        n = len(X)
        block_size = min(self.block_size, max(1, n - 1))
        return max(n - block_size + 1, 0)

def make_cross_validator(config, approach_id=None):
    """Create a scikit-learn cross-validator from user/app configuration."""
    scheme = config.get('cv_scheme', 'shuffle_split')
    n_splits = int(config.get('cv_n_splits', 20))
    random_state = config.get('cv_random_state', 12345)
    random_state = None if random_state in (None, '') else int(random_state)

    if scheme == 'shuffle_split':
        return sklearn.model_selection.ShuffleSplit(
            n_splits=n_splits,
            test_size=float(config.get('cv_test_size', 0.25)),
            random_state=random_state,
        )
    if scheme in ('leave_one_timepoint_out', 'leave_one_out'):
        return sklearn.model_selection.LeaveOneOut()
    if scheme == 'leave_contiguous_block_out':
        return LeaveContiguousBlockOut(block_size=n_splits)
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

def cross_validation_curve_fit(
    x_data,
    y_data,
    model_function,
    cv,
    selected_metrics,
    p0=None,
    kwargs={},
    return_fit_params=False,
    return_diagnostics=False,
):
    """Run cross-validation and optionally retain each split's fitted parameters.

    The optional parameter vectors are used only for descriptive CV-fit
    variability plots. A ``None`` entry is retained for a split whose fit or
    test prediction failed so the plot can report how many CV fits contributed.
    When requested, diagnostics wrap the existing return value as
    ``(existing_result, diagnostics)`` without changing scores or split behavior.
    """
    num_params = len(inspect.signature(model_function).parameters) - 1  # subtract 1 for 'x'
    scores = {metric: [] for metric in selected_metrics}
    fit_params = []
    diagnostics = {
        'enabled': cv is not None,
        'total_splits': 0,
        'successful_fits': 0,
        'fitting_failure_count': 0,
        'finite_scores': {metric: 0 for metric in selected_metrics},
        'min_training_observations': None,
        'max_training_observations': None,
        'min_test_observations': None,
        'max_test_observations': None,
    }
    if cv is None:
        scores = {metric: np.array([np.nan]) for metric in selected_metrics}
        result = (scores, fit_params) if return_fit_params else scores
        return (result, diagnostics) if return_diagnostics else result
    for train_index, test_index in cv.split(x_data):
        diagnostics['total_splits'] += 1
        for label, indices in (('training', train_index), ('test', test_index)):
            count = int(len(indices))
            minimum, maximum = f'min_{label}_observations', f'max_{label}_observations'
            diagnostics[minimum] = count if diagnostics[minimum] is None else min(diagnostics[minimum], count)
            diagnostics[maximum] = count if diagnostics[maximum] is None else max(diagnostics[maximum], count)
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        popt = None
        fit_succeeded = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                popt, pcov = sp.optimize.curve_fit(model_function, x_train, y_train, p0, **kwargs)
            y_pred = model_function(x_test, *popt)
            if not np.all(np.isfinite(y_pred)):
                raise ValueError('Cross-validation prediction contains nonfinite values.')
            fit_succeeded = True
            tmp_scores = {metric: calculate_metric(metric, y_test, y_pred, num_params) for metric in selected_metrics}
        except Exception:
            popt = None
            tmp_scores = {metric: np.nan for metric in selected_metrics}
        diagnostics['successful_fits'] += int(fit_succeeded)
        diagnostics['fitting_failure_count'] += int(not fit_succeeded)
        for k,v in tmp_scores.items():
            scores[k].append(v)
            diagnostics['finite_scores'][k] += int(np.isfinite(v))
        if return_fit_params:
            fit_params.append(None if popt is None else np.asarray(popt, dtype=float))
    scores = {k:np.array(scores[k]) for k in scores}
    result = (scores, fit_params) if return_fit_params else scores
    return (result, diagnostics) if return_diagnostics else result

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

def calculate_tau_with_uncertainty(model_function, params, pcov, num_samples=50, random_state=None):
    if getattr(model_function, 'supports_tau', True) is False:
        return np.nan, np.nan
    # Import locally to avoid a module-import cycle: the reusable fitting core
    # depends on this compatibility module.
    from ivivc_core.uncertainty import sample_parameter_sets

    sampled_parameters = sample_parameter_sets(
        params,
        pcov,
        parameter_names=[f'p{index}' for index in range(len(params))],
        n_samples=num_samples,
        random_state=random_state,
    )
    tau_samples = []
    for sample in sampled_parameters.values:
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

def generate_confidence_bands(
    model_function,
    x_values,
    params,
    pcov,
    confidence_level=0.95,
    n_samples=1000,
    random_state=None,
):
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
    # Import locally to avoid a module-import cycle: the reusable fitting core
    # depends on this compatibility module.
    from ivivc_core.uncertainty import (
        evaluate_curve_draws,
        sample_parameter_sets,
        summarize_draws,
    )

    parameter_samples = sample_parameter_sets(
        params,
        pcov,
        n_samples=n_samples,
        random_state=random_state,
    )
    curve_draws = evaluate_curve_draws(
        model_function,
        x_values,
        parameter_samples,
    )
    summary = summarize_draws(
        curve_draws,
        confidence_level=confidence_level,
    )

    return {
        'lower': summary.lower,
        'upper': summary.upper,
        'mean': summary.mean,
    }

def generate_prediction_bands(
    model_function,
    x_values,
    params,
    pcov,
    residuals=None,
    confidence_level=0.95,
    n_samples=1000,
    random_state=None,
):
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
    from ivivc_core.uncertainty import (
        evaluate_curve_draws,
        sample_parameter_sets,
        summarize_draws,
    )

    parameter_samples = sample_parameter_sets(
        params,
        pcov,
        n_samples=n_samples,
        random_state=random_state,
    )
    curve_draws = evaluate_curve_draws(
        model_function,
        x_values,
        parameter_samples,
    )
    summary = summarize_draws(curve_draws, confidence_level=confidence_level)
    mean_prediction = summary.mean
    param_values = np.asarray([params[name] for name in params], dtype=float)

    # Estimate residual standard error
    if residuals is not None:
        # Use provided residuals
        residual_std = np.std(residuals, ddof=len(param_values))
    else:
        # Estimate from parameter uncertainty (this is less accurate)
        residual_std = np.nanmean(summary.std)
        print("Warning: No residuals provided. Using parameter uncertainty to estimate residual variance.")

    # For prediction intervals, we need to account for:
    # 1. Parameter uncertainty (captured in the Monte Carlo samples)
    # 2. Residual variance (the inherent scatter in the data)

    # Calculate prediction variance = model uncertainty + residual variance
    model_variance = summary.std ** 2
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
