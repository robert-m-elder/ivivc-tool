"""Small, model-independent checks for the integral time constant."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad


def evaluate_tau(model_function, parameters) -> tuple[float, str | None]:
    """Return the fitted-curve tau, or NaN and an unavailability reason.

    Keep QUADPACK's default tolerances and the existing subdivision limit.
    These are defensive numerical checks, not a proof of convergence for
    arbitrary model functions. No truncated area or model-specific formula is
    substituted when integration fails.
    """
    if getattr(model_function, 'supports_tau', True) is False:
        return np.nan, 'Time-constant calculation is not supported for this model.'
    try:
        parameters = tuple(np.asarray(parameters, dtype=float))
        if not np.all(np.isfinite(parameters)):
            return np.nan, 'Fitted parameters are not finite.'
        with warnings.catch_warnings(), np.errstate(over='raise', invalid='raise', divide='raise'):
            # Override an outer ignore filter: a finite returned area is not
            # sufficient when QUADPACK reports unreliable convergence.
            warnings.simplefilter('error', IntegrationWarning)
            initial_value = float(model_function(0.0, *parameters))
            if not np.isfinite(initial_value) or initial_value == 0:
                return np.nan, 'The fitted initial response is zero or nonfinite.'

            def integrand(time):
                value = float(model_function(time, *parameters))
                if not np.isfinite(value):
                    raise ValueError('The fitted curve has a nonfinite evaluation.')
                return value

            integral, error = quad(integrand, 0.0, np.inf, limit=100)
            if not np.isfinite(integral) or not np.isfinite(error) or error < 0:
                return np.nan, 'Numerical integration returned an invalid area or error estimate.'
            tau = float(integral / initial_value)
    except IntegrationWarning as exception:
        return np.nan, f'Numerical integration did not converge reliably: {exception}'
    except Exception as exception:
        return np.nan, f'Time-constant calculation failed: {exception}'
    if not np.isfinite(tau) or tau <= 0:
        return np.nan, 'The calculated time constant is not finite and positive.'
    return tau, None
