"""Reusable structured preprocessing for IVIVC time-course data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from preprocessing import preprocessing_options

from .types import (
    NormalizationRecord,
    PreparedIVIVCData,
    PreprocessingMetadata,
    PreprocessingWarning,
    TimeCourse,
)


def _as_selection_list(selection: Iterable[str] | None) -> list[str]:
    # This deliberately mirrors the existing ``selection or []`` behavior.
    return list(selection or [])


def _clean_time_course(course: TimeCourse) -> TimeCourse:
    """Remove rows with missing responses while preserving true zero times."""
    mask = ~pd.isna(course.value)
    return TimeCourse(course.time[mask], course.value[mask], label=course.label)


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
    return {'applied': True}


def _apply_normalizations(
    in_vitro: TimeCourse,
    in_vivo: TimeCourse | None,
    selected_normalizations: list[str],
) -> tuple[TimeCourse, TimeCourse | None, tuple[NormalizationRecord, ...]]:
    records: list[NormalizationRecord] = []
    current_in_vitro = in_vitro
    current_in_vivo = in_vivo

    for normalization_name in selected_normalizations:
        in_vitro_parameters = _normalization_parameters(
            current_in_vitro.value,
            normalization_name,
        )
        normalized_in_vitro = preprocessing_options['normalization'][normalization_name](
            current_in_vitro.value
        )

        in_vivo_parameters = None
        normalized_in_vivo = None
        if current_in_vivo is not None:
            in_vivo_parameters = _normalization_parameters(
                current_in_vivo.value,
                normalization_name,
            )
            normalized_in_vivo = preprocessing_options['normalization'][normalization_name](
                current_in_vivo.value
            )

        current_in_vitro = TimeCourse(
            current_in_vitro.time,
            normalized_in_vitro,
            label=current_in_vitro.label,
        )
        if current_in_vivo is not None:
            current_in_vivo = TimeCourse(
                current_in_vivo.time,
                normalized_in_vivo,
                label=current_in_vivo.label,
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
                )
                if two_datasets:
                    current_in_vivo = TimeCourse(
                        scaled_t2,
                        current_in_vivo.value,
                        label=current_in_vivo.label,
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
) -> tuple[TimeCourse, PreprocessingMetadata]:
    """Clean and transform one time course without performing interpolation."""
    scaling_names = _as_selection_list(selected_scalings)
    normalization_names = _as_selection_list(selected_normalizations)

    cleaned = _clean_time_course(TimeCourse(time, value, label=label))
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
) -> PreparedIVIVCData:
    """Prepare paired in vitro and in vivo time courses for IVIVC analysis."""
    interpolation_names = _as_selection_list(selected_interpolation)
    scaling_names = _as_selection_list(selected_scalings)
    normalization_names = _as_selection_list(selected_normalizations)

    cleaned_in_vitro = _clean_time_course(in_vitro)
    cleaned_in_vivo = _clean_time_course(in_vivo)
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
    metadata = PreprocessingMetadata(
        selected_interpolation=tuple(interpolation_names),
        interpolation=interpolation_choices[-1],
        selected_scalings=tuple(scaling_names),
        applied_scalings=applied_scalings,
        selected_normalizations=tuple(normalization_names),
        normalizations=normalization_records,
        warnings=preprocessing_warnings,
    )

    return PreparedIVIVCData(
        in_vitro=TimeCourse(t1, m1, label=scaled_in_vitro.label),
        in_vivo=TimeCourse(t2, m2, label=scaled_in_vivo.label),
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
) -> PreparedIVIVCData:
    """Convenience entry point matching the existing four-array workflow."""
    two_datasets = (t2 is not None) and (m2 is not None)
    if not two_datasets:
        in_vitro, metadata = prepare_time_course(
            t1,
            m1,
            selected_scalings=selected_scalings,
            selected_normalizations=selected_normalizations,
            fallback_invalid_log_scaling=fallback_invalid_log_scaling,
            label='in_vitro',
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
    )
