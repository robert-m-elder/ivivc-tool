from __future__ import annotations

import numpy as np
import pytest

from ivivc_core import (
    PreparedIVIVCData,
    TimeCourse,
    get_approach_spec,
    get_model_spec,
    iter_model_specs,
    prepare_data,
    prepare_ivivc_data,
    prepare_time_course,
)


def test_time_course_validates_structure_and_copies_inputs():
    time = np.array([0.0, 1.0])
    value = np.array([1.0, 0.5])
    course = TimeCourse(time, value, label='in_vivo')

    time[0] = 99.0
    value[0] = 99.0
    np.testing.assert_array_equal(course.time, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(course.value, np.array([1.0, 0.5]))
    assert course.label == 'in_vivo'

    with pytest.raises(ValueError, match='same length'):
        TimeCourse(np.array([0.0]), np.array([1.0, 0.5]))


def test_structured_single_course_preserves_legacy_data_and_records_normalization():
    prepared = prepare_data(
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 5.0, np.nan]),
        selected_normalizations=['initial_value'],
    )

    assert isinstance(prepared, PreparedIVIVCData)
    assert prepared.in_vivo is None
    np.testing.assert_array_equal(prepared.in_vitro.time, np.array([0.0, 1.0]))
    np.testing.assert_allclose(prepared.in_vitro.value, np.array([1.0, 0.5]))
    legacy = prepared.to_legacy_data()
    assert isinstance(legacy, list)
    assert len(legacy) == 2

    record = prepared.preprocessing.normalizations[0]
    assert record.name == 'initial_value'
    assert record.in_vitro_parameters == {'initial_value': 10.0, 'applied': True}
    assert record.in_vivo_parameters is None


def test_structured_paired_preprocessing_exposes_alignment_and_metadata():
    in_vitro = TimeCourse(
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 5.0, 2.5]),
        label='in_vitro',
    )
    in_vivo = TimeCourse(
        np.array([0.0, 2.0, 4.0]),
        np.array([20.0, 10.0, 5.0]),
        label='in_vivo',
    )

    prepared = prepare_ivivc_data(
        in_vitro,
        in_vivo,
        selected_interpolation=['default'],
        selected_normalizations=['initial_value'],
        selected_scalings=['log_y'],
    )

    assert prepared.has_in_vivo
    assert prepared.preprocessing.interpolation == 'default'
    assert prepared.preprocessing.applied_scalings == ('log_y',)
    assert prepared.preprocessing.selected_normalizations == ('initial_value',)
    np.testing.assert_array_equal(prepared.time_grid, np.array([0.0, 1.0, 2.0, 4.0]))
    np.testing.assert_allclose(
        prepared.in_vitro.value,
        np.log10(np.array([1.0, 0.5, 0.25])),
    )
    np.testing.assert_allclose(
        prepared.in_vivo.value,
        np.log10(np.array([1.0, 0.5, 0.25])),
    )

    record = prepared.preprocessing.normalizations[0]
    assert record.in_vitro_parameters['initial_value'] == 10.0
    assert record.in_vivo_parameters['initial_value'] == 20.0
    assert isinstance(prepared.to_legacy_data(), tuple)
    assert len(prepared.to_legacy_data()) == 10


def test_invalid_log_fallback_is_structured_and_legacy_compatible():
    course, metadata = prepare_time_course(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.5]),
        selected_scalings=['log_x'],
        fallback_invalid_log_scaling=True,
    )

    np.testing.assert_array_equal(course.time, np.array([0.0, 1.0]))
    assert metadata.applied_scalings == ()
    assert metadata.warnings[0].option == 'Log10 time scaling'
    assert metadata.to_legacy_dict() == {
        'applied_scalings': [],
        'warnings': [
            {
                'option': 'Log10 time scaling',
                'message': (
                    'Log10 time scaling was selected but was not applied because at least one '
                    'analyzed time value is zero or negative. The analysis continued using '
                    'unscaled time values.'
                ),
            }
        ],
    }


def test_model_registry_adapter_exposes_stable_model_information():
    spec = get_model_spec('exponential')

    assert spec.key == 'exponential'
    assert spec.display_name == 'Exponential'
    assert spec.parameter_names == ('a', 'b')
    assert spec.approaches == ('approach1', 'approach2')
    assert callable(spec.model_function)
    assert callable(spec.fit_function)
    assert spec.latex_equation
    assert spec.fit_kwargs == {}
    assert spec.supports_tau is True

    approach2_names = tuple(spec.key for spec in iter_model_specs('approach2'))
    assert approach2_names == get_approach_spec('approach2').models
    assert 'exponential' in approach2_names
    assert get_model_spec('mass_loss_one_phase').supports_tau is False


def test_registry_adapter_reports_unknown_keys_cleanly():
    with pytest.raises(KeyError, match='Unknown model'):
        get_model_spec('not_a_model')
    with pytest.raises(KeyError, match='Unknown approach'):
        get_approach_spec('not_an_approach')
