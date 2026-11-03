"""Reusable structured preprocessing for IVIVC time-course data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from preprocessing import preprocessing_options

from .observations import (
    ObservationUncertainty,
    ResolvedObservationUncertainty,
    resolve_observation_uncertainty,
)
from .types import (
    NormalizationRecord,
    PreparedIVIVCData,
    PreprocessingMetadata,
    PreprocessingSource,
    PreprocessingWarning,
    TimeCourse,
)


def _as_selection_list(selection: Iterable[str] | None) -> list[str]:
    # This deliberately mirrors the existing ``selection or []`` behavior.
    return list(selection or [])


def _normalization_order(selection: Iterable[str] | None) -> list[str]:
    """Apply response reversal once, after the other selected normalizations."""
    names = _as_selection_list(selection)
    return [name for name in names if name != 'one_minus'] + (
        ['one_minus'] if 'one_minus' in names else []
    )


def _course_uncertainty(course, supplied):
    if supplied is not None and course.observation_uncertainty is not None:
        raise ValueError('Supply uncertainty either on the TimeCourse or as a preprocessing keyword, not both.')
    return course.observation_uncertainty if supplied is None else supplied


def _check_uncertainty_transforms(normalizations, scalings, interpolations=()):
    """Require explicit support rather than guessing a transformed error model."""
    for name in normalizations:
        if name not in ('one_minus', 'initial_value', 'min_max'):
            raise ValueError(
                f'Normalization {name!r} is not supported with supplied uncertainty. '
                'Use initial-value, min-max, or one-minus normalization, '
                'or leave the response unnormalized.'
            )
    for name in scalings:
        if name != 'log_x':
            raise ValueError(
                f'Scaling {name!r} is not supported with supplied uncertainty. '
                'Log-response scaling needs a transformed observation-error model; '
                'use the unscaled response with matching SD/SEM.'
            )
    if any(name not in ('none', 'default', 'alternative') for name in interpolations):
        raise ValueError('This interpolation has not been verified to preserve uncertainty-bearing source rows.')


def _clean_time_course(
    course: TimeCourse, *, observation_uncertainty=None, clean_missing_time=False,
) -> tuple[TimeCourse, np.ndarray]:
    """Mask observations and uncertainty together; preserve legacy cleaning by default."""
    mask = ~pd.isna(course.value)
    if clean_missing_time:
        mask &= ~pd.isna(course.time)
    uncertainty = resolve_observation_uncertainty(
        observation_uncertainty, len(course.value), row_mask=mask,
    )
    return TimeCourse(
        course.time[mask], course.value[mask], label=course.label,
        observation_uncertainty=uncertainty,
    ), np.flatnonzero(mask)


def _source_record(course, uncertainty, retained_rows):
    return PreprocessingSource(
        course.time, course.value, retained_rows,
        observation_uncertainty=uncertainty, label=course.label,
    )


def _normalization_parameters(values: np.ndarray, name: str) -> dict[str, float | bool]:
    """Capture constants used by the existing normalization implementation."""
    if name == 'initial_value':
        initial_value = float(values[0])
        return {
            'initial_value': initial_value,
            'applied': bool(initial_value != 0),
        }
    if name == 'min_max':
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        return {
            'minimum': minimum,
            'maximum': maximum,
            'range': maximum - minimum,
            'applied': True,
        }
    if name == 'one_minus':
        return {'reference': 1.0, 'applied': True}
    return {'applied': True}


def _fixed_normalization_uncertainty(course, name, parameters):
    """Rescale SD/SEM conditional on the observed normalization constants.

    References are calculated once from the current cleaned values and held
    fixed. Their own uncertainty is excluded; this is not joint propagation
    through a random denominator or random extrema.
    """
    uncertainty = course.observation_uncertainty
    if name not in ('initial_value', 'min_max'):
        return uncertainty
    parameters['constants_treated_as_fixed'] = True
    if name == 'initial_value':
        divisor = parameters['initial_value']
        if not np.isfinite(divisor):
            raise ValueError('Initial-value normalization requires a finite initial value with supplied uncertainty.')
        if not parameters['applied']:
            # Preserve the existing zero-initial-value no-op and original units.
            return uncertainty
    else:
        divisor = parameters['range']
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError('Min-max normalization requires a finite, strictly positive response range with supplied uncertainty.')
    if uncertainty is None:
        return None
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        sigma = uncertainty.sigma / abs(divisor)
    return ResolvedObservationUncertainty(
        sigma, response_kind=uncertainty.response_kind,
        input_form=uncertainty.input_form,
        replicate_counts=uncertainty.replicate_counts,
        source=uncertainty.source, response_units='dimensionless',
    )


def _apply_normalizations(
    in_vitro: TimeCourse,
    in_vivo: TimeCourse | None,
    selected_normalizations: list[str],
) -> tuple[TimeCourse, TimeCourse | None, tuple[NormalizationRecord, ...]]:
    records: list[NormalizationRecord] = []
    current_in_vitro = in_vitro
    current_in_vivo = in_vivo
    fixed_reference_mode = (
        in_vitro.observation_uncertainty is not None
        or (in_vivo is not None and in_vivo.observation_uncertainty is not None)
    )

    for normalization_name in selected_normalizations:
        if fixed_reference_mode and normalization_name in ('initial_value', 'min_max'):
            if len(current_in_vitro.value) == 0 or (
                current_in_vivo is not None and len(current_in_vivo.value) == 0
            ):
                raise ValueError('Normalization with supplied uncertainty requires at least one retained response per profile.')
        in_vitro_parameters = _normalization_parameters(
            current_in_vitro.value,
            normalization_name,
        )
        in_vitro_uncertainty = current_in_vitro.observation_uncertainty
        if fixed_reference_mode:
            in_vitro_uncertainty = _fixed_normalization_uncertainty(
                current_in_vitro, normalization_name, in_vitro_parameters,
            )
        normalized_in_vitro = preprocessing_options['normalization'][normalization_name](
            current_in_vitro.value
        )

        in_vivo_parameters = None
        normalized_in_vivo = None
        in_vivo_uncertainty = None
        if current_in_vivo is not None:
            in_vivo_parameters = _normalization_parameters(
                current_in_vivo.value,
                normalization_name,
            )
            in_vivo_uncertainty = current_in_vivo.observation_uncertainty
            if fixed_reference_mode:
                in_vivo_uncertainty = _fixed_normalization_uncertainty(
                    current_in_vivo, normalization_name, in_vivo_parameters,
                )
            normalized_in_vivo = preprocessing_options['normalization'][normalization_name](
                current_in_vivo.value
            )

        current_in_vitro = TimeCourse(
            current_in_vitro.time,
            normalized_in_vitro,
            label=current_in_vitro.label,
            observation_uncertainty=in_vitro_uncertainty,
        )
        if current_in_vivo is not None:
            current_in_vivo = TimeCourse(
                current_in_vivo.time,
                normalized_in_vivo,
                label=current_in_vivo.label,
                observation_uncertainty=in_vivo_uncertainty,
            )

        records.append(
            NormalizationRecord(
                name=normalization_name,
                in_vitro_parameters=in_vitro_parameters,
                in_vivo_parameters=in_vivo_parameters,
            )
        )

    return current_in_vitro, current_in_vivo, tuple(records)


def _apply_scalings(
    in_vitro: TimeCourse,
    in_vivo: TimeCourse | None,
    selected_scalings: list[str],
    *,
    fallback_invalid_log_scaling: bool,
) -> tuple[TimeCourse, TimeCourse | None, tuple[str, ...], tuple[PreprocessingWarning, ...]]:
    """Apply selected scalings atomically across the applicable datasets."""
    current_in_vitro = in_vitro
    current_in_vivo = in_vivo
    applied_scalings: list[str] = []
    preprocessing_warnings: list[PreprocessingWarning] = []
    two_datasets = current_in_vivo is not None

    for scaling_name in selected_scalings:
        try:
            if scaling_name == 'log_x':
                scaled_t1 = preprocessing_options['scaling'][scaling_name](current_in_vitro.time)
                scaled_t2 = (
                    preprocessing_options['scaling'][scaling_name](current_in_vivo.time)
                    if two_datasets
                    else None
                )
                current_in_vitro = TimeCourse(
                    scaled_t1,
                    current_in_vitro.value,
                    label=current_in_vitro.label,
                    observation_uncertainty=current_in_vitro.observation_uncertainty,
                )
                if two_datasets:
                    current_in_vivo = TimeCourse(
                        scaled_t2,
                        current_in_vivo.value,
                        label=current_in_vivo.label,
                        observation_uncertainty=current_in_vivo.observation_uncertainty,
                    )
            elif scaling_name == 'log_y':
                scaled_m1 = preprocessing_options['scaling'][scaling_name](current_in_vitro.value)
                scaled_m2 = (
                    preprocessing_options['scaling'][scaling_name](current_in_vivo.value)
                    if two_datasets
                    else None
                )
                current_in_vitro = TimeCourse(
                    current_in_vitro.time,
                    scaled_m1,
                    label=current_in_vitro.label,
                )
                if two_datasets:
                    current_in_vivo = TimeCourse(
                        current_in_vivo.time,
                        scaled_m2,
                        label=current_in_vivo.label,
                    )
            # Preserve the legacy behavior for any caller-supplied unknown
            # scaling key: it is recorded but does not transform an axis.
            applied_scalings.append(scaling_name)
        except ValueError:
            if not fallback_invalid_log_scaling or scaling_name not in {'log_x', 'log_y'}:
                raise

            if scaling_name == 'log_x':
                values_to_check = (
                    (current_in_vitro.time, current_in_vivo.time)
                    if two_datasets
                    else (current_in_vitro.time,)
                )
            else:
                values_to_check = (
                    (current_in_vitro.value, current_in_vivo.value)
                    if two_datasets
                    else (current_in_vitro.value,)
                )
            if any(
                np.any(~np.isfinite(np.asarray(values, dtype=float)))
                for values in values_to_check
            ):
                raise

            if scaling_name == 'log_x':
                preprocessing_warnings.append(
                    PreprocessingWarning(
                        option='Log10 time scaling',
                        message=(
                            'Log10 time scaling was selected but was not applied because at least one '
                            'analyzed time value is zero or negative. The analysis continued using '
                            'unscaled time values.'
                        ),
                    )
                )
            else:
                preprocessing_warnings.append(
                    PreprocessingWarning(
                        option='Log10 response scaling',
                        message=(
                            'Log10 response scaling was selected but was not applied because at least '
                            'one analyzed response value is zero or negative after normalization. The '
                            'analysis continued using unscaled response values.'
                        ),
                    )
                )

    return (
        current_in_vitro,
        current_in_vivo,
        tuple(applied_scalings),
        tuple(preprocessing_warnings),
    )


def prepare_time_course(
    time,
    value,
    *,
    selected_scalings: Iterable[str] | None = None,
    selected_normalizations: Iterable[str] | None = None,
    fallback_invalid_log_scaling: bool = False,
    label: str | None = None,
    observation_uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None = None,
) -> tuple[TimeCourse, PreprocessingMetadata]:
    """Clean and transform one time course without performing interpolation."""
    scaling_names = _as_selection_list(selected_scalings)
    normalization_names = _normalization_order(selected_normalizations)

    has_uncertainty = observation_uncertainty is not None
    if has_uncertainty:
        _check_uncertainty_transforms(normalization_names, scaling_names)
    original = TimeCourse(time, value, label=label)
    cleaned, retained_rows = _clean_time_course(
        original, observation_uncertainty=observation_uncertainty,
        clean_missing_time=has_uncertainty,
    )
    normalized, _, normalization_records = _apply_normalizations(
        cleaned,
        None,
        normalization_names,
    )
    scaled, _, applied_scalings, preprocessing_warnings = _apply_scalings(
        normalized,
        None,
        scaling_names,
        fallback_invalid_log_scaling=fallback_invalid_log_scaling,
    )

    metadata = PreprocessingMetadata(
        selected_scalings=tuple(scaling_names),
        applied_scalings=applied_scalings,
        selected_normalizations=tuple(normalization_names),
        normalizations=normalization_records,
        warnings=preprocessing_warnings,
        sources={'profile': _source_record(original, observation_uncertainty, retained_rows)}
        if has_uncertainty else {},
    )
    return scaled, metadata


def prepare_ivivc_data(
    in_vitro: TimeCourse,
    in_vivo: TimeCourse,
    *,
    selected_interpolation: Iterable[str] | None = None,
    selected_scalings: Iterable[str] | None = None,
    selected_normalizations: Iterable[str] | None = None,
    fallback_invalid_log_scaling: bool = False,
    in_vitro_uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None = None,
    in_vivo_uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None = None,
) -> PreparedIVIVCData:
    """Prepare paired in vitro and in vivo time courses for IVIVC analysis."""
    interpolation_names = _as_selection_list(selected_interpolation)
    scaling_names = _as_selection_list(selected_scalings)
    normalization_names = _normalization_order(selected_normalizations)

    vitro_uncertainty = _course_uncertainty(in_vitro, in_vitro_uncertainty)
    vivo_uncertainty = _course_uncertainty(in_vivo, in_vivo_uncertainty)
    has_uncertainty = vitro_uncertainty is not None or vivo_uncertainty is not None
    if has_uncertainty:
        _check_uncertainty_transforms(normalization_names, scaling_names, interpolation_names)
    cleaned_in_vitro, vitro_rows = _clean_time_course(
        in_vitro, observation_uncertainty=vitro_uncertainty, clean_missing_time=has_uncertainty,
    )
    cleaned_in_vivo, vivo_rows = _clean_time_course(
        in_vivo, observation_uncertainty=vivo_uncertainty, clean_missing_time=has_uncertainty,
    )
    normalized_in_vitro, normalized_in_vivo, normalization_records = _apply_normalizations(
        cleaned_in_vitro,
        cleaned_in_vivo,
        normalization_names,
    )
    scaled_in_vitro, scaled_in_vivo, applied_scalings, preprocessing_warnings = _apply_scalings(
        normalized_in_vitro,
        normalized_in_vivo,
        scaling_names,
        fallback_invalid_log_scaling=fallback_invalid_log_scaling,
    )

    interpolation_choices = interpolation_names or ['none']
    data = None
    for interpolation_name in interpolation_choices:
        data = preprocessing_options['interpolation'][interpolation_name](
            scaled_in_vitro.time,
            scaled_in_vitro.value,
            scaled_in_vivo.time,
            scaled_in_vivo.value,
        )

    t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2 = data
    if has_uncertainty and any(
        not np.array_equal(returned, observed)
        for returned, observed in (
            (t1, scaled_in_vitro.time), (m1, scaled_in_vitro.value),
            (t2, scaled_in_vivo.time), (m2, scaled_in_vivo.value),
        )
    ):
        raise ValueError('Interpolation changed original source rows; supplied uncertainty cannot be transported.')
    metadata = PreprocessingMetadata(
        selected_interpolation=tuple(interpolation_names),
        interpolation=interpolation_choices[-1],
        selected_scalings=tuple(scaling_names),
        applied_scalings=applied_scalings,
        selected_normalizations=tuple(normalization_names),
        normalizations=normalization_records,
        warnings=preprocessing_warnings,
        sources={
            'in_vitro': _source_record(in_vitro, vitro_uncertainty, vitro_rows),
            'in_vivo': _source_record(in_vivo, vivo_uncertainty, vivo_rows),
        } if has_uncertainty else {},
    )

    return PreparedIVIVCData(
        in_vitro=TimeCourse(t1, m1, label=scaled_in_vitro.label,
                           observation_uncertainty=scaled_in_vitro.observation_uncertainty),
        in_vivo=TimeCourse(t2, m2, label=scaled_in_vivo.label,
                          observation_uncertainty=scaled_in_vivo.observation_uncertainty),
        response_grid=mm,
        time_grid=tt,
        in_vitro_time_at_response=ti1,
        in_vivo_time_at_response=ti2,
        in_vitro_response_on_time_grid=mi1,
        in_vivo_response_on_time_grid=mi2,
        preprocessing=metadata,
    )


def prepare_data(
    t1,
    m1,
    t2=None,
    m2=None,
    *,
    selected_interpolation: Iterable[str] | None = None,
    selected_scalings: Iterable[str] | None = None,
    selected_normalizations: Iterable[str] | None = None,
    fallback_invalid_log_scaling: bool = False,
    in_vitro_uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None = None,
    in_vivo_uncertainty: ObservationUncertainty | ResolvedObservationUncertainty | None = None,
) -> PreparedIVIVCData:
    """Convenience entry point matching the existing four-array workflow."""
    two_datasets = (t2 is not None) and (m2 is not None)
    if not two_datasets:
        if in_vivo_uncertainty is not None or (
            in_vitro_uncertainty is not None and (t2 is not None or m2 is not None)
        ):
            raise ValueError('In vivo uncertainty or partial paired input requires both in vivo time and response arrays.')
        in_vitro, metadata = prepare_time_course(
            t1,
            m1,
            selected_scalings=selected_scalings,
            selected_normalizations=selected_normalizations,
            fallback_invalid_log_scaling=fallback_invalid_log_scaling,
            label='in_vitro',
            observation_uncertainty=in_vitro_uncertainty,
        )
        return PreparedIVIVCData(
            in_vitro=in_vitro,
            in_vivo=None,
            preprocessing=metadata,
        )

    return prepare_ivivc_data(
        TimeCourse(t1, m1, label='in_vitro'),
        TimeCourse(t2, m2, label='in_vivo'),
        selected_interpolation=selected_interpolation,
        selected_scalings=selected_scalings,
        selected_normalizations=selected_normalizations,
        fallback_invalid_log_scaling=fallback_invalid_log_scaling,
        in_vitro_uncertainty=in_vitro_uncertainty,
        in_vivo_uncertainty=in_vivo_uncertainty,
    )
