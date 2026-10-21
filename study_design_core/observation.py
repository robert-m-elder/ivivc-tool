"""Future-study observation models and residual-based surrogates."""

from __future__ import annotations

import numpy as np

from ivivc_core import ProfileFit

from .types import ObservationModel, ObservationModelKind


def observation_model_from_residuals(
    fit: ProfileFit,
    *,
    kind: ObservationModelKind = 'constant_sd',
    replicates_per_time: int = 1,
    minimum_sd: float = 0.0,
    relative_value_floor: float | None = None,
) -> ObservationModel:
    """Estimate a future observation model from one reference fit's residuals.

    The result is a residual-based surrogate that can combine measurement error
    and model lack of fit.  It should not be interpreted automatically as pure
    experimental variability.
    """
    if fit.residual_dof <= 0:
        raise ValueError('Residual degrees of freedom must be positive to estimate variability.')

    residuals = np.asarray(fit.residuals, dtype=float)
    fitted = np.asarray(fit.fitted_values, dtype=float)
    finite = np.isfinite(residuals) & np.isfinite(fitted)
    if not np.any(finite):
        raise ValueError('No finite residuals are available to estimate variability.')

    if kind == 'constant_sd':
        value = float(np.sqrt(np.sum(residuals[finite] ** 2) / fit.residual_dof))
    elif kind == 'constant_cv':
        if relative_value_floor is None:
            finite_fitted = np.abs(fitted[finite])
            scale = float(np.nanmax(finite_fitted)) if len(finite_fitted) else 0.0
            relative_value_floor = max(np.finfo(float).eps, scale * 1e-8)
        relative_value_floor = float(relative_value_floor)
        if not np.isfinite(relative_value_floor) or relative_value_floor <= 0:
            raise ValueError('relative_value_floor must be finite and positive.')
        usable = finite & (np.abs(fitted) >= relative_value_floor)
        if not np.any(usable):
            raise ValueError('No fitted values are large enough to estimate a constant CV.')
        relative_residuals = residuals[usable] / fitted[usable]
        denominator = max(1, fit.residual_dof)
        value = float(np.sqrt(np.sum(relative_residuals ** 2) / denominator))
    else:
        raise ValueError("kind must be either 'constant_sd' or 'constant_cv'.")

    return ObservationModel(
        kind=kind,
        value=value,
        replicates_per_time=replicates_per_time,
        minimum_sd=minimum_sd,
        source='reference_fit_residuals_surrogate',
    )
