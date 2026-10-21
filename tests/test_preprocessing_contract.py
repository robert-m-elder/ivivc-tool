from __future__ import annotations

import numpy as np
import pytest

from engine import preprocess_data


def test_single_dataset_preserves_zero_time_and_returns_two_arrays():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    response = np.array([1.0, 0.8, np.nan, 0.4])

    processed = preprocess_data(time, response, None, None)

    assert isinstance(processed, list)
    assert len(processed) == 2
    np.testing.assert_array_equal(processed[0], np.array([0.0, 1.0, 3.0]))
    np.testing.assert_array_equal(processed[1], np.array([1.0, 0.8, 0.4]))


def test_no_interpolation_aligns_only_exact_observations_and_averages_duplicates():
    t1 = np.array([0.0, 1.0, 1.0, 3.0])
    m1 = np.array([1.0, 0.8, 0.6, 0.2])
    t2 = np.array([0.0, 2.0, 3.0])
    m2 = np.array([1.0, 0.5, 0.2])

    result = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['none'],
    )
    _, _, _, _, mm, tt, ti1, ti2, mi1, mi2 = result

    np.testing.assert_array_equal(tt, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(mi1, np.array([1.0, 0.7, np.nan, 0.2]), equal_nan=True)
    np.testing.assert_allclose(mi2, np.array([1.0, np.nan, 0.5, 0.2]), equal_nan=True)
    assert mm[0] == 1.0
    assert mm[-1] == 0.2
    assert np.isfinite(ti1[mm == 0.2]).all()
    assert np.isfinite(ti2[mm == 0.2]).all()


def test_default_interpolation_uses_union_of_observed_times(paired_degradation_data):
    t1, m1, t2, m2 = paired_degradation_data

    result = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['default'],
    )
    _, _, _, _, _, tt, _, _, mi1, mi2 = result

    expected_time_union = np.array(sorted(set(t1).union(set(t2))))
    np.testing.assert_array_equal(tt, expected_time_union)
    # np.interp is bounded by each source dataset; union-grid points outside a
    # source range remain NaN rather than being extrapolated.
    np.testing.assert_allclose(mi1[:5], m1, atol=1e-12)
    assert np.isnan(mi1[-2:]).all()
    assert np.isfinite(mi2).all()


def test_alternative_interpolation_produces_denser_alignment(paired_degradation_data):
    t1, m1, t2, m2 = paired_degradation_data

    union_result = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['default'],
    )
    dense_result = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['alternative'],
    )

    assert len(dense_result[5]) > len(union_result[5])
    assert len(dense_result[4]) > len(union_result[4])


def test_normalization_is_applied_before_log_response_scaling():
    t1 = np.array([1.0, 2.0, 3.0])
    m1 = np.array([10.0, 5.0, 2.5])
    t2 = np.array([1.0, 2.0, 3.0])
    m2 = np.array([20.0, 10.0, 5.0])

    result, metadata = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['none'],
        selected_normalizations=['initial_value'],
        selected_scalings=['log_y'],
        return_metadata=True,
    )

    np.testing.assert_allclose(result[1], np.log10(np.array([1.0, 0.5, 0.25])))
    np.testing.assert_allclose(result[3], np.log10(np.array([1.0, 0.5, 0.25])))
    assert metadata == {'applied_scalings': ['log_y'], 'warnings': []}


def test_invalid_log_time_scaling_can_fall_back_without_partial_scaling():
    t1 = np.array([0.0, 1.0, 2.0])
    m1 = np.array([1.0, 0.8, 0.6])
    t2 = np.array([0.0, 2.0, 4.0])
    m2 = np.array([1.0, 0.8, 0.6])

    result, metadata = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=['none'],
        selected_scalings=['log_x'],
        fallback_invalid_log_scaling=True,
        return_metadata=True,
    )

    np.testing.assert_array_equal(result[0], t1)
    np.testing.assert_array_equal(result[2], t2)
    assert metadata['applied_scalings'] == []
    assert len(metadata['warnings']) == 1
    assert metadata['warnings'][0]['option'] == 'Log10 time scaling'
    assert 'zero or negative' in metadata['warnings'][0]['message']


def test_invalid_log_scaling_raises_when_fallback_is_disabled():
    with pytest.raises(ValueError, match='greater than zero'):
        preprocess_data(
            np.array([0.0, 1.0]),
            np.array([1.0, 0.5]),
            None,
            None,
            selected_scalings=['log_x'],
        )
