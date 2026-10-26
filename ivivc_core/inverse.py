"""Generic, bounded-search inversion of fitted response curves.

Calibration limits trigger cautions, not endpoint clipping. Monotonicity and
uniqueness checks are numerical checks on the examined branch, not proofs about
an arbitrary function outside the finite search. No model-specific inverses.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import brentq

DEFAULT_INVERSE_EXPANSIONS = 3


def inverse_response_times(
    model_function, params, target_responses, time_range,
    observed_responses=None, grid_points=101, *,
    allow_extrapolation=True, minimum_time=0.0,
    max_expansions=DEFAULT_INVERSE_EXPANSIONS,
):
    """Return times, per-target statuses, and an optional overall skip reason.

    Start in the calibration interval and, for an outside target, double the
    search span up to ``max_expansions`` times. ``minimum_time=0`` excludes
    negative physical times; use ``None`` for log10-transformed time coordinates.
    The observed responses are retained as a compatibility argument; their range
    is for reporting and must not clip the mathematical inverse. All evaluations
    and returned times use the supplied (possibly preprocessed) coordinates.
    """
    targets = np.asarray(target_responses, dtype=float)
    times = np.full(targets.shape, np.nan)
    statuses = np.full(targets.shape, 'unavailable', dtype=object)
    lo, hi = map(float, time_range)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return times, statuses, 'Inverse mapping requires two distinct finite in vivo calibration times.'
    if minimum_time is not None:
        minimum_time = float(minimum_time)
        if not np.isfinite(minimum_time) or minimum_time > lo:
            raise ValueError('minimum_time must be finite and no greater than the calibration minimum.')
    if int(max_expansions) != max_expansions or max_expansions < 0:
        raise ValueError('max_expansions must be a nonnegative integer.')
    span = hi - lo
    if not np.isfinite(span):
        return times, statuses, 'Inverse mapping calibration span is not finite.'
    grid_points = max(int(grid_points), 101)

    # Share warning/error handling across this curve's evaluations.
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        finfo = np.finfo(float)
        grid_step = span / (grid_points - 1)
        root_xtol = max(span * 1e-12, finfo.tiny)

        def evaluate(x):
            x = np.asarray(x, dtype=float)
            y = np.asarray(model_function(x, **params), dtype=float)
            # A scalar constant is a legitimate (but noninvertible) model output.
            if y.shape != x.shape:
                y = np.broadcast_to(y, x.shape)
            return y.copy()

        def monotonic(y, direction):
            scale = max(float(np.max(np.abs(y))), finfo.tiny)
            tolerance = 64 * finfo.eps * scale
            return bool(np.all(direction * np.diff(y) >= -tolerance))

        grid = np.linspace(lo, hi, grid_points)
        try:
            values = evaluate(grid)
        except Exception as exc:
            return times, statuses, f'Inverse mapping could not evaluate the in vivo calibration curve: {exc}'
        if not np.all(np.isfinite(values)):
            return times, statuses, 'Inverse mapping requires a finite curve throughout the calibration interval.'
        direction = np.sign(values[-1] - values[0])
        if direction == 0 or not monotonic(values, direction):
            return times, statuses, 'Inverse mapping requires a numerically monotonic, nonconstant calibration curve.'

        def check_root(root, center=None):
            # Test local numerical uniqueness for *both* exact grid hits and solved
            # roots. Saturation can create a spurious sign change near a plateau.
            step = max(grid_step, abs(np.spacing(root)) * 8)
            probes = np.array([root - step, root + step])
            if minimum_time is not None:
                probes = probes[probes >= minimum_time]
            try:
                if center is None:
                    center = float(evaluate([root])[0])
                probe_values = evaluate(probes)
            except Exception:
                return np.nan, 'inverse uniqueness could not be checked'
            if np.any(np.isfinite(probe_values) & (probe_values == center)):
                return np.nan, 'nonunique or numerically flat inverse'
            return root, 'mapped'

        def solve(x, y, target):
            """Solve a supported target; refuse flat/rounded endpoints and jumps."""
            exact = np.flatnonzero(y == target)
            if len(exact) > 1:
                return np.nan, 'nonunique or numerically flat inverse'
            if len(exact):
                return check_root(float(x[exact[0]]))
            # Strict brackets avoid accepting a value merely close to an asymptote.
            crossing = np.flatnonzero(((y[:-1] < target) & (y[1:] > target)) |
                                      ((y[:-1] > target) & (y[1:] < target)))
            if len(crossing) != 1:
                return np.nan, 'no unique bracket on searched branch'
            i = int(crossing[0])
            a, b = sorted((float(x[i]), float(x[i + 1])))
            try:
                root = brentq(lambda t: float(evaluate([t])[0] - target), a, b,
                              xtol=root_xtol,
                              rtol=1e-12, maxiter=200)
                check = float(evaluate([root])[0])
                scale = max(abs(target), abs(float(y[i])), abs(float(y[i + 1])), finfo.tiny)
                if not np.isfinite(check) or abs(check - target) > 1e-8 * scale:
                    return np.nan, 'inverse root did not satisfy response tolerance'
                return check_root(root, center=check)
            except Exception:
                return np.nan, 'inverse root calculation failed'

        for index, target in np.ndenumerate(targets):
            if not np.isfinite(target):
                statuses[index] = 'nonfinite input response'
                continue
            if min(values[0], values[-1]) <= target <= max(values[0], values[-1]):
                times[index], statuses[index] = solve(grid, values, target)
                continue
            if not allow_extrapolation:
                statuses[index] = 'outside fitted calibration response range (extrapolation disabled)'
                continue
            # Search only the continuation indicated by the calibrated direction.
            right = direction * (target - values[-1]) > 0
            edge = hi if right else lo
            edge_value = values[-1] if right else values[0]
            search_direction = 1 if right else -1
            statuses[index] = 'inverse not resolved within search limit'
            for attempt in range(int(max_expansions)):
                endpoint = edge + search_direction * span * (2.0 ** attempt)
                if minimum_time is not None:
                    endpoint = max(endpoint, minimum_time)
                if not np.isfinite(endpoint):
                    statuses[index] = 'inverse search exceeded finite time coordinates'
                    break
                if endpoint == edge:
                    statuses[index] = 'response not reached within allowed time domain'
                    break
                extension = np.linspace(edge, endpoint, min(grid_points, 129))
                try:
                    extended_values = evaluate(extension)
                except Exception:
                    statuses[index] = 'model evaluation failed on inverse search branch'
                    break
                finite = np.isfinite(extended_values)
                # Never bridge a nonfinite part of the curve.
                stop = int(np.flatnonzero(~finite)[0]) if not finite.all() else len(extension)
                x, y = extension[:stop], extended_values[:stop]
                if len(x) < 2:
                    statuses[index] = 'nonfinite model on inverse search branch'
                    break
                if not monotonic(y, direction * search_direction):
                    statuses[index] = 'inverse continuation is not monotonic'
                    break
                if min(y[0], y[-1]) <= target <= max(y[0], y[-1]):
                    root, status = solve(x, y, target)
                    times[index] = root
                    statuses[index] = 'mapped (extrapolated time)' if np.isfinite(root) else status
                    break
                if stop < len(extension):
                    statuses[index] = 'response not resolved before nonfinite model domain'
                    break
                if y[-1] == edge_value:
                    statuses[index] = 'response not reached before numerically flat tail'
                    break
                edge, edge_value = endpoint, y[-1]
        return times, statuses, None
