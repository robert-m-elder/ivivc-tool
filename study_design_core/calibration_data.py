"""Assemble independent calibration observations without pooling their errors."""

from __future__ import annotations

import numpy as np

from ivivc_core import ResolvedObservationUncertainty, TimeCourse


def _copy_course(course: TimeCourse, *, label: str | None) -> TimeCourse:
    if not isinstance(course, TimeCourse):
        raise TypeError('Calibration observations must be a TimeCourse or None.')
    time = np.asarray(course.time, dtype=float)
    value = np.asarray(course.value, dtype=float)
    if not len(time) or np.any(~np.isfinite(time)) or np.any(~np.isfinite(value)):
        raise ValueError('Calibration observations require nonempty finite time/response rows; use None for an absent dataset.')
    order = np.argsort(time, kind='stable')
    uncertainty = course.observation_uncertainty
    if uncertainty is not None:
        uncertainty = ResolvedObservationUncertainty(
            uncertainty.sigma[order], response_kind=uncertainty.response_kind,
            input_form=uncertainty.input_form,
            replicate_counts=(None if uncertainty.replicate_counts is None
                              else uncertainty.replicate_counts[order]),
            source=uncertainty.source, response_units=uncertainty.response_units,
        )
    return TimeCourse(time[order], value[order], label=course.label if label is None else label,
                      observation_uncertainty=uncertainty)


def append_calibration_observations(
    existing: TimeCourse | None,
    additions: TimeCourse | None,
    *,
    label: str | None = None,
) -> TimeCourse:
    """Copy and append independent calibration rows on one response/time scale.

    At least one dataset is required. All results are stably sorted by time,
    including a single dataset with no additions. This gives baseline and
    augmented fits the same chronological convention for initial-value
    normalization. Existing rows precede additions at ties. Duplicate times are
    retained, including old and new independent batch means. No existing value,
    SD, SEM, or count is updated or pooled; rows are only copied and reordered.

    Supplied uncertainty must cover both datasets and have the same individual
    or mean interpretation. SEM is already the error of each mean and is never
    divided by its count. If either mean batch lacks count metadata, the combined
    count field is omitted rather than inventing counts for unknown rows. Input
    datasets keep their original metadata. Response-unit labels must agree when
    both are supplied; a combined label is retained only when both are known.

    These are assembled measurement rows, not raw uploads: missing/nonfinite
    values must be handled explicitly before calling this helper. The caller is
    responsible for compatible scales and independence between the batches.
    """
    if existing is None and additions is None:
        raise ValueError('At least one calibration dataset is required.')
    if existing is not None and existing is additions:
        raise ValueError('Existing and additional calibration observations must be independent batches, not the same object.')
    if existing is None:
        return _copy_course(additions, label=label)
    if additions is None:
        return _copy_course(existing, label=label)

    old = _copy_course(existing, label=None)
    new = _copy_course(additions, label=None)
    first, second = old.observation_uncertainty, new.observation_uncertainty
    if (first is None) != (second is None):
        raise ValueError('Supplied SD/SEM must cover existing and additional calibration observations; specify historical uncertainty explicitly.')
    if first is not None and first.response_kind != second.response_kind:
        raise ValueError('Existing and additional calibration observations must use the same individual/mean interpretation.')
    if first is not None and (
        first.response_units is not None and second.response_units is not None
        and first.response_units != second.response_units
    ):
        raise ValueError('Existing and additional calibration uncertainty must use compatible response units.')

    time = np.concatenate([old.time, new.time])
    value = np.concatenate([old.value, new.value])
    order = np.argsort(time, kind='stable')
    uncertainty = None
    if first is not None:
        counts = None
        if first.replicate_counts is not None and second.replicate_counts is not None:
            counts = np.concatenate([first.replicate_counts, second.replicate_counts])[order]
        uncertainty = ResolvedObservationUncertainty(
            np.concatenate([first.sigma, second.sigma])[order],
            response_kind=first.response_kind, replicate_counts=counts,
            source=f'independent calibration batches (existing: {first.source}; additions: {second.source})',
            response_units=(first.response_units
                            if first.response_units == second.response_units else None),
        )
    return TimeCourse(
        time[order], value[order],
        label=label if label is not None else old.label or new.label,
        observation_uncertainty=uncertainty,
    )


def _new_calibration_batch(times, replicate_values, individual_sigma, *,
                           response_kind='mean', source='simulated_calibration'):
    """Represent already generated independent observations, without new noise."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(replicate_values, dtype=float)
    sigma = np.asarray(individual_sigma, dtype=float)
    if (times.ndim != 1 or not len(times) or values.ndim != 2
            or values.shape[1] != len(times) or not values.shape[0]
            or sigma.shape != times.shape):
        raise ValueError('Calibration replicates and individual SD must match the sampling times.')
    if not np.isfinite(values).all() or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError('Simulated observations and SD must be finite, with nonnegative SD.')
    count = values.shape[0]
    if response_kind == 'mean':
        response = values.mean(axis=0)
        errors = sigma / np.sqrt(count)
        counts = np.full(len(times), count)
    elif response_kind == 'individual':
        times = np.tile(times, count)
        response = values.reshape(-1)
        errors = np.tile(sigma, count)
        counts = None
    else:
        raise ValueError('Calibration batches require individual or mean responses.')
    # Preserve the existing deterministic, wholly unweighted study case.
    uncertainty = None if np.all(errors == 0) else ResolvedObservationUncertainty(
        errors, response_kind=response_kind, replicate_counts=counts, source=source,
    )
    return TimeCourse(times, response, label=source, observation_uncertainty=uncertainty)


def _completed_calibration_course(existing, completed_sigma):
    """Retain completed rows and the precision actually used by study refitting."""
    uncertainty = existing.observation_uncertainty
    if uncertainty is None and completed_sigma is not None:
        uncertainty = ResolvedObservationUncertainty(
            completed_sigma, response_kind='individual',
            source='study_refit_observation_model_fallback',
        )
    return TimeCourse(existing.time, existing.response, label=existing.label,
                      observation_uncertainty=uncertainty)
