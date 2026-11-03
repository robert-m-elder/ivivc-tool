"""Information criteria for fits using supplied absolute observation errors."""
from __future__ import annotations

import operator

import numpy as np


def gaussian_information_criteria(residuals, sigma, n_params):
    """Use the normal likelihood and the app's conventional AICc correction.

    Supplied SD/SEM is held fixed; k counts fitted curve parameters. AICc uses
    the existing 2*k*(k+1)/(n-k-1) convention when its denominator is positive.
    This retains the application convention without asserting exact small-sample
    calibration for every nonlinear model or variance assumption.
    """
    residuals = np.asarray(residuals, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    if errors.ndim == 0:
        errors = np.full(residuals.shape, errors.item())
    if residuals.ndim != 1 or errors.shape != residuals.shape:
        raise ValueError('Residuals and sigma must be aligned one-dimensional arrays.')
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError('Sigma must be finite and strictly positive.')
    try:
        k = operator.index(n_params)
    except TypeError as exc:
        raise ValueError('n_params must be a nonnegative integer.') from exc
    if isinstance(n_params, (bool, np.bool_)) or k < 0:
        raise ValueError('n_params must be a nonnegative integer.')
    n = len(residuals)
    values = dict(n=n, k=k, rss=np.nan, log_likelihood=np.nan,
                  aic=np.nan, aicc=np.nan, bic=np.nan, aicc_defined=False)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        # Log sigma directly so its square need not be representable.
        deviance = float(np.sum((residuals / errors) ** 2
                               + np.log(2. * np.pi) + 2. * np.log(errors)))
    if not n or not np.isfinite(deviance):
        return values
    aic = deviance + 2. * k
    values.update(log_likelihood=-.5 * deviance, aic=aic,
                  bic=deviance + k * np.log(n))
    if n > k + 1:
        aicc = aic + 2. * k * (k + 1) / (n - k - 1)
        values.update(aicc=aicc, aicc_defined=bool(np.isfinite(aicc)))
    return values
