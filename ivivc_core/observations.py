"""Shared absolute response uncertainty on the supplied data's current scale.

An individual response uses its SD; a reported mean uses its SEM. Both resolve
to ``sigma``, the standard uncertainty of the actual supplied value. Optional
counts describe means but never rescale an already supplied SEM. Original
experimental errors are assumed independent; covariance introduced by later
transformations must be handled by those transformations.

This module does not apply weights, transform data, generate measurement noise,
or specify variability for future observations. Those are separate operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import operator
from typing import Literal

import numpy as np


ResponseKind = Literal['individual', 'mean']
UncertaintyInputForm = Literal['constant', 'pointwise']


def _real_values(values, name: str) -> np.ndarray:
    """Inspect elements before NumPy can coerce mixed booleans into numbers."""
    try:
        if np.asarray(values).dtype.kind in ('b', 'c', 'M', 'm'):
            raise ValueError(f'{name} must contain real numbers, not booleans, dates, or complex values.')
        original = np.asarray(values, dtype=object)
        if original.ndim not in (0, 1):
            raise ValueError(f'{name} must be a scalar or one-dimensional array.')
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in original.flat):
            raise ValueError(f'{name} must contain real numbers, not booleans, dates, strings, or complex values.')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a real numeric scalar or one-dimensional array: {exc}') from exc
    return original


def _numeric_values(values, name: str) -> np.ndarray:
    try:
        result = np.array(_real_values(values, name), dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} cannot be represented as real numeric values: {exc}') from exc
    result.setflags(write=False)
    return result


def _count_values(values) -> np.ndarray:
    # Retain Python/NumPy integer values exactly, including above 2**53.
    result = _real_values(values, 'replicate_counts').copy()
    result.setflags(write=False)
    return result


def _metadata(response_kind, replicate_counts, source, response_units) -> None:
    if not isinstance(response_kind, str) or response_kind not in ('individual', 'mean'):
        raise ValueError("response_kind must be 'individual' or 'mean'.")
    if replicate_counts is not None and response_kind != 'mean':
        raise ValueError('replicate_counts are metadata for reported means only.')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('source must be a nonempty string.')
    if response_units is not None and (
        not isinstance(response_units, str) or not response_units.strip()
    ):
        raise ValueError('response_units must be a nonempty string or None.')


def _observation_count(value) -> int:
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError('n_observations must be a nonnegative integer.') from exc
    if count < 0:
        raise ValueError('n_observations must be a nonnegative integer.')
    return count


def _selection(row_mask, count: int) -> np.ndarray:
    if row_mask is None:
        return np.ones(count, dtype=bool)
    mask = np.asarray(row_mask)
    if mask.ndim != 1 or mask.dtype.kind != 'b' or len(mask) != count:
        raise ValueError('row_mask must be a one-dimensional boolean array matching the original observations.')
    return mask.copy()


def _validated_sigma(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Standard uncertainty must be finite and strictly positive for every retained observation.')
    return values


def _validated_counts(values) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    counts = np.empty(values.shape, dtype=np.int64)
    for index, value in np.ndenumerate(values):
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            count = int(value)
        elif isinstance(value, Real) and math.isfinite(value) and value == math.floor(value):
            count = int(value)
        else:
            raise ValueError('replicate_counts must be finite positive integers within the int64 range.')
        if count < 1 or count > np.iinfo(np.int64).max:
            raise ValueError('replicate_counts must be finite positive integers within the int64 range.')
        counts[index] = count
    return counts


def _selected_values(values, count, mask, name, validator) -> np.ndarray:
    if values.ndim == 0:
        # Invalid constants remain invalid even when no observations are kept.
        validator(np.asarray([values.item()]))
        return np.full(int(np.count_nonzero(mask)), values.item(), dtype=values.dtype)
    if len(values) != count:
        raise ValueError(f'{name} must have one entry per original observation before row masking.')
    selected = values[mask].copy()
    validator(selected)
    return selected


@dataclass(frozen=True, kw_only=True, eq=False)
class ObservationUncertainty:
    """Supplied SD (individual responses) or SEM (reported means).

    ``standard_uncertainty`` and optional ``replicate_counts`` can each be a
    scalar or a vector aligned to the original response rows. A constant SEM
    means every supplied mean has that SEM, even if their counts differ.

    Construction checks structure and metadata. Resolution checks numerical
    values on retained rows, allowing missing entries on explicitly removed
    rows. No missing row is removed automatically. Counts never convert an SD
    into an SEM here: users supplying means must already supply SEMs.
    """

    standard_uncertainty: float | np.ndarray
    response_kind: ResponseKind
    replicate_counts: int | np.ndarray | None = None
    source: str = 'user'
    response_units: str | None = None

    def __init__(
        self, standard_uncertainty, *, response_kind,
        replicate_counts=None, source='user', response_units=None,
    ) -> None:
        _metadata(response_kind, replicate_counts, source, response_units)
        object.__setattr__(self, 'standard_uncertainty', _numeric_values(
            standard_uncertainty, 'standard_uncertainty',
        ))
        object.__setattr__(self, 'response_kind', response_kind)
        object.__setattr__(self, 'replicate_counts', None if replicate_counts is None else
                           _count_values(replicate_counts))
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'response_units', response_units)


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedObservationUncertainty:
    """Validated absolute standard uncertainties aligned to retained responses.

    Arrays are copied and read-only to prevent accidental changes to shared
    uncertainty. ``input_form`` records the supplied format, not a different
    statistical model. Empty arrays are permitted after explicit row removal.
    """

    sigma: np.ndarray
    response_kind: ResponseKind
    input_form: UncertaintyInputForm = 'pointwise'
    replicate_counts: np.ndarray | None = None
    source: str = 'user'
    response_units: str | None = None

    def __init__(
        self, sigma, *, response_kind, input_form='pointwise',
        replicate_counts=None, source='user', response_units=None,
    ) -> None:
        _metadata(response_kind, replicate_counts, source, response_units)
        if not isinstance(input_form, str) or input_form not in ('constant', 'pointwise'):
            raise ValueError("input_form must be 'constant' or 'pointwise'.")
        sigma = _numeric_values(sigma, 'sigma')
        if sigma.ndim != 1:
            raise ValueError('Resolved sigma must be one-dimensional.')
        _validated_sigma(sigma)
        counts = None
        if replicate_counts is not None:
            counts = _count_values(replicate_counts)
            if counts.ndim != 1 or len(counts) != len(sigma):
                raise ValueError('Resolved replicate_counts must match the sigma array.')
            counts = _validated_counts(counts)
            counts.setflags(write=False)
        object.__setattr__(self, 'sigma', sigma)
        object.__setattr__(self, 'response_kind', response_kind)
        object.__setattr__(self, 'input_form', input_form)
        object.__setattr__(self, 'replicate_counts', counts)
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'response_units', response_units)

    @property
    def n_observations(self) -> int:
        return len(self.sigma)

    @property
    def uncertainty_kind(self) -> Literal['sd', 'sem']:
        return 'sem' if self.response_kind == 'mean' else 'sd'

    def subset(self, row_mask) -> ResolvedObservationUncertainty:
        """Apply the same explicit boolean row mask used for the responses."""
        mask = _selection(row_mask, self.n_observations)
        return ResolvedObservationUncertainty(
            self.sigma[mask], response_kind=self.response_kind,
            input_form=self.input_form,
            replicate_counts=None if self.replicate_counts is None else self.replicate_counts[mask],
            source=self.source, response_units=self.response_units,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return an independent, JSON-compatible record of values and meaning."""
        return {
            'standard_uncertainty': self.sigma.tolist(),
            'response_kind': self.response_kind,
            'uncertainty_kind': self.uncertainty_kind,
            'input_form': self.input_form,
            'scale': 'absolute',
            'independent_observations': True,
            'replicate_counts': None if self.replicate_counts is None else self.replicate_counts.tolist(),
            'source': self.source,
            'response_units': self.response_units,
        }


def resolve_observation_uncertainty(
    uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None,
    n_observations: int,
    *,
    row_mask=None,
) -> ResolvedObservationUncertainty | None:
    """Resolve supplied uncertainty against original, pre-cleaning row counts.

    ``None`` stays unspecified; it is never interpreted as zero or as known
    unit uncertainty. No residual-based uncertainty is estimated here.
    Pointwise arrays must match the original row count, even when a mask is
    supplied. Only retained values must be finite and positive; this permits
    missing entries on rows removed for missing time/response values.
    Already-resolved uncertainty is checked against the same original count
    and subset without changing its interpretation or supplied-format metadata.
    """
    count = _observation_count(n_observations)
    mask = _selection(row_mask, count)
    if uncertainty is None:
        return None
    if isinstance(uncertainty, ResolvedObservationUncertainty):
        if uncertainty.n_observations != count:
            raise ValueError('Resolved uncertainty must match the original observations before row masking.')
        return uncertainty.subset(mask)
    if not isinstance(uncertainty, ObservationUncertainty):
        raise TypeError('uncertainty must be supplied/resolved observation uncertainty or None.')
    sigma = _selected_values(
        uncertainty.standard_uncertainty, count, mask,
        'standard_uncertainty', _validated_sigma,
    )
    counts = None
    if uncertainty.replicate_counts is not None:
        counts = _selected_values(
            uncertainty.replicate_counts, count, mask,
            'replicate_counts', _validated_counts,
        )
    return ResolvedObservationUncertainty(
        sigma, response_kind=uncertainty.response_kind,
        input_form='constant' if uncertainty.standard_uncertainty.ndim == 0 else 'pointwise',
        replicate_counts=counts, source=uncertainty.source,
        response_units=uncertainty.response_units,
    )
