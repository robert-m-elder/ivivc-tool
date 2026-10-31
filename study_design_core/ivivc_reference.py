"""Explicit IVIVC prediction inputs for prospective and adaptive study design."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from ivivc_core import (
    DEFAULT_PREDICTION_METHODS,
    IVIVCAnalysis,
    TimeCourse,
    generate_trajectory_ensembles,
)

from .reference import DuplicateTimePolicy, reference_from_trajectory_ensemble
from .types import InterpolationMethod, ReferenceTrajectoryEnsemble


def reference_from_ivivc(
    analysis: IVIVCAnalysis,
    prediction: TimeCourse,
    *,
    selection: str,
    method: str,
    n_samples: int = 1000,
    random_state=None,
    time_grid=None,
    grid_size: int | None = None,
    interpolation: InterpolationMethod = 'pchip',
    duplicate_time_policy: DuplicateTimePolicy = 'mean',
    required_times=None,
) -> ReferenceTrajectoryEnsemble:
    """Build a design reference from one explicitly selected fitted IVIVC.

    ``selection`` is a successful analysis key, such as
    ``'approach2:exponential'``. ``method`` selects an existing prediction
    method; Approach 2 supports ``'value_ratio'``, ``'tau_ratio'``, and
    ``'observed_response_inverse'``. Neither a failed fit nor an unavailable
    nominal prediction is replaced with another method.

    Prediction inputs must already use the analysis's response normalization
    and time units. Logged axes are excluded: this adapter does not reverse
    logarithmic preprocessing into study time/response coordinates.

    Conversion retains complete joint draws and uses their shared supported
    domain, without additional extrapolation. Existing method-specific model
    extrapolation is recorded in prediction diagnostics. ``time_grid`` can
    require a particular planning horizon; ``required_times`` can check fixed
    or completed observation times without dictating the reference grid.

    The returned reference supports the existing optimization, timing, and
    simulation APIs. Choose the study analysis model separately with
    ``fit_reference_model_ensemble``. For adaptive design, update this prior
    with ``reweight_reference_ensemble`` and cumulative completed observations.
    """
    if not isinstance(analysis, IVIVCAnalysis):
        raise TypeError('analysis must be an IVIVCAnalysis.')
    if not isinstance(prediction, TimeCourse):
        raise TypeError('prediction must be a TimeCourse.')
    if selection not in analysis.model_results:
        if selection in analysis.errors:
            raise ValueError(
                f'Selected IVIVC fit {selection!r} is unavailable: '
                f'{analysis.errors[selection].message}'
            )
        raise ValueError(f'Selected IVIVC fit {selection!r} is not present in the analysis.')

    fit = analysis.model_results[selection]
    allowed = DEFAULT_PREDICTION_METHODS[fit.approach_id]
    if method not in allowed:
        raise ValueError(
            f'Prediction method {method!r} is not supported for {fit.approach_id}; '
            f'choose one of {allowed!r}.'
        )
    preprocessing = analysis.prepared_data.preprocessing
    logged_axes = {'log_x', 'log_y'}.intersection(preprocessing.applied_scalings)
    if logged_axes:
        raise ValueError(
            'Study-design IVIVC references require unlogged time and response axes; '
            f'the analysis has applied {sorted(logged_axes)!r}. '
            'Fit the IVIVC without logarithmic axis preprocessing.'
        )
    required = None
    if required_times is not None:
        required = np.asarray(required_times, dtype=float)
        if required.ndim != 1 or np.any(~np.isfinite(required)):
            raise ValueError('required_times must be a finite one-dimensional array.')

    # Generate only the selected fit. Unrelated selections must not affect its
    # random stream, introduce extra work, or trigger unrelated fit failures.
    selected_analysis = replace(
        analysis, model_results={selection: fit}, errors={},
    )
    batch = generate_trajectory_ensembles(
        selected_analysis,
        prediction,
        methods={fit.approach_id: method},
        n_samples=n_samples,
        random_state=random_state,
    )
    key = f'{selection}:{method}'
    if key in batch.errors:
        raise ValueError(
            f'IVIVC reference {key!r} could not be generated: {batch.errors[key].message}'
        )
    ensemble = batch[key]
    central = ensemble.central
    if not central.available:
        raise ValueError(
            f'IVIVC reference {key!r} is unavailable: '
            f'{central.skip_reason or "The nominal prediction is unavailable."}'
        )
    central_supported = (
        np.isfinite(central.in_vivo_time) & np.isfinite(central.in_vivo_value)
    )
    if len(np.unique(central.in_vivo_time[central_supported])) < 2:
        raise ValueError(
            f'IVIVC reference {key!r} is unavailable: the nominal prediction '
            'requires at least two distinct finite in vivo times.'
        )

    reference = reference_from_trajectory_ensemble(
        ensemble,
        time_grid=time_grid,
        grid_size=grid_size,
        interpolation=interpolation,
        duplicate_time_policy=duplicate_time_policy,
    )
    lower, upper = reference.supported_time_range
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    if required is not None and (
        np.any(required < lower - tolerance) or np.any(required > upper + tolerance)
    ):
        raise ValueError(
            'required_times extend beyond the shared supported reference time range '
            f'[{lower:g}, {upper:g}]. No baseline or completed time was added or moved.'
        )

    reference.metadata = {
        **reference.metadata,
        'reference_source': 'ivivc_prediction',
        'ivivc_selection': selection,
        'ivivc_approach': fit.approach_id,
        'ivivc_calibration_model': fit.model_name,
        'ivivc_prediction_method': method,
        'ivivc_prediction_key': key,
        'ivivc_prediction_status': deepcopy(central.status),
        'ivivc_prediction_supported_points': central_supported.copy(),
        'ivivc_prediction_auxiliary': deepcopy(central.auxiliary),
        'ivivc_preprocessing': {
            'applied_scalings': tuple(preprocessing.applied_scalings),
            'selected_normalizations': tuple(preprocessing.selected_normalizations),
            'normalizations': [
                {
                    'name': record.name,
                    'in_vitro_parameters': dict(record.in_vitro_parameters),
                    'in_vivo_parameters': (
                        None if record.in_vivo_parameters is None
                        else dict(record.in_vivo_parameters)
                    ),
                }
                for record in preprocessing.normalizations
            ],
        },
        'ivivc_required_times': None if required is None else required.copy(),
    }
    return reference
