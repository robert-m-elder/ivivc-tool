"""Small input adapters for the synthetic study-design examples."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ivivc_core import ObservationUncertainty, TimeCourse, resolve_observation_uncertainty
from study_design_core import ExistingObservations, ObservationModel


def load_future_sd_profile(path) -> tuple[tuple[float, float], ...]:
    """Read explicit future individual SD on the in vivo time coordinate."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or not {'time', 'sd'}.issubset(reader.fieldnames)
                or len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError('Future SD profile CSV requires time and sd columns.')
        try:
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Every future SD profile row must match the header columns.')
                rows.append((float(row['time']), float(row['sd'])))
            rows = tuple(rows)
        except (TypeError, ValueError) as exc:
            raise ValueError('Future SD profile CSV must contain numeric time and sd values.') from exc
    validate_future_sd_profile(rows)
    return rows


def validate_future_sd_profile(rows) -> tuple[tuple[float, float], ...]:
    try:
        values = np.asarray(rows, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError('future_sd_profile must contain time, SD pairs.') from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError('future_sd_profile must contain nonempty time, SD pairs.')
    model = ObservationModel('time_profile_sd', profile_times=values[:, 0], profile_sd=values[:, 1])
    if np.any(model.profile_sd <= 0):
        raise ValueError('Future SD profile values must be positive for optimization.')
    return tuple(zip(model.profile_times.tolist(), model.profile_sd.tolist()))


def future_observation_model(settings) -> ObservationModel:
    """Build one explicit future-error assumption, separate from source error."""
    if settings.future_sd_profile is not None:
        rows = np.asarray(settings.future_sd_profile, dtype=float)
        return ObservationModel('time_profile_sd', profile_times=rows[:, 0], profile_sd=rows[:, 1],
            replicates_per_time=settings.future_replicates_per_time, source='proof_of_concept_assumption')
    sd = settings.observation_sd if settings.future_observation_sd is None else settings.future_observation_sd
    return ObservationModel('constant_sd', sd, replicates_per_time=settings.future_replicates_per_time,
                            source='proof_of_concept_assumption')


def future_observation_metadata(settings) -> dict:
    model = future_observation_model(settings)
    return {'future_observation_model': model.to_metadata(),
            'future_observation_sd': model.value,
            'future_replicates_per_time': model.replicates_per_time}


def completed_with_historical_assumption(existing, settings):
    """Keep the legacy completed-data assumption independent of future overrides.

    This nominal SD is an explicit assumption for legacy clipped toy data,
    not a claim that those data follow an exact independent normal model.
    """
    if (existing.observation_uncertainty is not None or
            (settings.future_observation_sd is None and settings.future_sd_profile is None)):
        return existing
    return ExistingObservations(existing.time, existing.response, label=existing.label,
        observation_uncertainty=ObservationUncertainty(
            settings.observation_sd, response_kind='individual',
            source='historical_nominal_sd_assumption',
        ))


def source_observation_rows(times, observed, individual_sd, *, enabled, response_kind, label):
    """Keep individual rows or report means with known SEM exactly once.

    ``observed`` has shape (times, actual replicates). Its generating SD is
    an explicit synthetic assumption, not estimated from fitted residuals.
    """
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if response_kind not in ('individual', 'mean'):
        raise ValueError('Source response kind must be individual or mean.')
    if response_kind == 'mean' and not enabled:
        raise ValueError('Reported source means require explicit measurement uncertainty.')
    if observed.ndim != 2 or observed.shape[0] != len(times) or observed.shape[1] < 1:
        raise ValueError('Source observations must have time-by-replicate shape.')
    n = observed.shape[1]
    if enabled and (not np.isfinite(individual_sd) or individual_sd <= 0):
        raise ValueError('Supplied synthetic source SD must be positive when measurement uncertainty is enabled.')
    if response_kind == 'mean':
        response = observed.mean(axis=1)
        row_times = times
        sigma = individual_sd / np.sqrt(n)
        counts = np.full(len(times), n, dtype=int)
        identifiers = np.ones(len(times), dtype=int)
    else:
        response = observed.reshape(-1)
        row_times = np.repeat(times, n)
        sigma = individual_sd
        counts = np.ones(len(response), dtype=int)
        identifiers = np.tile(np.arange(1, n + 1), len(times))
    uncertainty = None
    if enabled:
        uncertainty = resolve_observation_uncertainty(ObservationUncertainty(
            sigma, response_kind=response_kind,
            replicate_counts=counts if response_kind == 'mean' else None,
            source='known_synthetic_measurement_sd'), len(response))
    return TimeCourse(row_times, response, label=label, observation_uncertainty=uncertainty), identifiers, counts


def source_observation_columns(course, counts) -> dict:
    uncertainty = course.observation_uncertainty
    return {'response_kind': ('individual' if uncertainty is None else uncertainty.response_kind),
            'replicate_count': counts,
            'standard_uncertainty': np.nan if uncertainty is None else uncertainty.sigma,
            'uncertainty_kind': None if uncertainty is None else uncertainty.uncertainty_kind}
