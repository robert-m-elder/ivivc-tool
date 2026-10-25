"""Method-specific calibration-range cautions, separate from numerical validity.

Ranges use preprocessed coordinates. A range check is not validation of transfer
between devices, a test of model adequacy, or evidence of observation coverage.
"""
from __future__ import annotations

import numpy as np


def finite_range(values):
    values = np.asarray([] if values is None else values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return (float(np.min(values)), float(np.max(values))) if values.size else None


def _outside(values, limits):
    values = np.asarray(values, dtype=float)
    if limits is None:
        return np.zeros(values.shape, dtype=bool)
    low, high = limits
    tolerance = 1e-9 * max(abs(low), abs(high), high - low, np.finfo(float).tiny)
    return np.isfinite(values) & ((values < low - tolerance) | (values > high + tolerance))


def annotate_prediction_domain(
    trajectory, *, in_vitro_time=None, in_vivo_time=None, in_vivo_response=None,
    paired_in_vitro_time=None, paired_in_vivo_time=None,
):
    """Attach point/interval range diagnostics to a calculated trajectory.

    Approach 1 uses the paired times actually fitted. Response ratios check the
    times at which both profiles are evaluated. Tau scaling checks input/output
    times. The inverse checks only in vivo response/time (input time is unused).
    Approach 3 retains bounded lookup support and warns about out-of-range
    resulting times; no new extrapolation rule for lookup ratios is invented.
    """
    count = len(trajectory.in_vitro_time)
    method = trajectory.source_method
    ranges = {
        'in_vitro_time': finite_range(in_vitro_time),
        'in_vivo_time': finite_range(in_vivo_time),
    }
    checks = []
    interval_limits = None
    if method == 'time_mapping':
        ranges = {
            'paired_in_vitro_time': finite_range(paired_in_vitro_time),
            'paired_in_vivo_time': finite_range(paired_in_vivo_time),
        }
        checks = [
            (trajectory.in_vitro_time, ranges['paired_in_vitro_time'], 'input time outside paired in vitro fitting range'),
            (trajectory.in_vivo_time, ranges['paired_in_vivo_time'], 'mapped time outside paired in vivo fitting range'),
        ]
        interval_limits = ranges['paired_in_vivo_time']
    elif method in ('value_ratio', 'value_scaling'):
        checks = [
            (trajectory.in_vitro_time, ranges['in_vitro_time'], 'evaluation time outside in vitro calibration range'),
            (trajectory.in_vitro_time, ranges['in_vivo_time'], 'evaluation time outside in vivo calibration range'),
        ]
    elif method in ('tau_ratio', 'time_scaling'):
        checks = [
            (trajectory.in_vitro_time, ranges['in_vitro_time'], 'input time outside in vitro calibration range'),
            (trajectory.in_vivo_time, ranges['in_vivo_time'], 'mapped time outside in vivo calibration range'),
        ]
        interval_limits = ranges['in_vivo_time']
    elif method == 'observed_response_inverse':
        # In vitro calibration and supplied prediction times are not operands.
        ranges = {'in_vivo_time': ranges['in_vivo_time'],
                  'in_vivo_response': finite_range(in_vivo_response)}
        checks = [
            (trajectory.in_vitro_value, ranges['in_vivo_response'], 'input response outside observed in vivo response range'),
            (trajectory.in_vivo_time, ranges['in_vivo_time'], 'mapped time outside in vivo calibration range'),
        ]
        interval_limits = ranges['in_vivo_time']
    else:
        return trajectory

    is_value = method in ('value_ratio', 'value_scaling')
    centers = trajectory.in_vivo_value if is_value else trajectory.in_vivo_time
    supported = np.isfinite(centers) & np.isfinite(trajectory.in_vivo_time) & np.isfinite(trajectory.in_vivo_value)
    reasons = [[] for _ in range(count)]
    for coordinates, limits, text in checks:
        for index in np.flatnonzero(_outside(coordinates, limits) & supported):
            reasons[index].append(text)
    extrapolated = np.array([bool(items) for items in reasons])
    assessed = all(limits is not None for _, limits, _ in checks)
    status = np.full(count, 'within calibration range' if assessed else 'calibration range unavailable', dtype=object)
    status[extrapolated] = 'extrapolation'
    status[~supported] = 'unavailable prediction'
    interval_status = np.full(count, 'interval unavailable', dtype=object)
    lower = trajectory.value_lower if is_value else trajectory.time_lower
    upper = trajectory.value_upper if is_value else trajectory.time_upper
    interval_outside = np.zeros(count, dtype=bool)
    if lower is not None and upper is not None:
        bounded = np.isfinite(lower) & np.isfinite(upper) & supported
        interval_status[bounded] = ('within calibration range' if assessed else 'calibration range unavailable')
        interval_status[bounded & extrapolated] = 'extrapolation (point outside calibration range)'
        if interval_limits is not None:
            interval_outside = bounded & (_outside(lower, interval_limits) | _outside(upper, interval_limits))
            interval_status[interval_outside] = 'interval extends beyond calibration range'
    if trajectory.source_approach == 'approach3':
        interval_status[:] = 'not calculated for this method'

    warnings = []
    n_points = int(extrapolated.sum())
    n_intervals = int((interval_outside & ~extrapolated).sum())
    if n_points:
        warnings.append({
            'label': 'Extrapolation',
            'message': (f'{n_points} of {count} calculated points are outside a relevant calibration range. '
                        'See the prediction data for ranges and reasons. Model-based intervals do not establish reliability outside calibration.'),
        })
    if n_intervals:
        warnings.append({
            'label': 'Interval extends beyond calibration',
            'message': (f'{n_intervals} parameter-uncertainty intervals extend beyond calibration although '
                        'their central points are within the checked ranges.'),
        })
    trajectory.auxiliary.update({
        'extrapolated': extrapolated,
        'extrapolation_status': status,
        'extrapolation_reason': np.array(['; '.join(items) for items in reasons], dtype=object),
        'interval_extrapolated': interval_outside,
        'interval_extrapolation_status': interval_status,
        'calibration_ranges': ranges,
        'extrapolation_warnings': warnings,
    })
    return trajectory
