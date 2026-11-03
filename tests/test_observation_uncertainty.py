from __future__ import annotations

import json

import numpy as np
import pytest

from ivivc_core import (
    ObservationUncertainty,
    ResolvedObservationUncertainty,
    TimeCourse,
    resolve_observation_uncertainty,
)


def test_unspecified_uncertainty_stays_unspecified():
    assert resolve_observation_uncertainty(None, 3) is None
    assert resolve_observation_uncertainty(None, 3, row_mask=[True, False, True]) is None


def test_constant_and_pointwise_inputs_resolve_to_the_same_sigma():
    constant = resolve_observation_uncertainty(
        ObservationUncertainty(0.04, response_kind='mean'), 3,
    )
    pointwise = resolve_observation_uncertainty(
        ObservationUncertainty([0.04, 0.04, 0.04], response_kind='mean'), 3,
    )

    assert isinstance(constant, ResolvedObservationUncertainty)
    np.testing.assert_array_equal(constant.sigma, pointwise.sigma)
    assert constant.input_form == 'constant'
    assert pointwise.input_form == 'pointwise'
    assert constant.n_observations == 3
    assert constant.uncertainty_kind == 'sem'
    assert constant.replicate_counts is None


def test_sd_and_sem_have_the_same_numerical_representation_but_different_meanings():
    individual = resolve_observation_uncertainty(
        ObservationUncertainty([0.1, 0.2], response_kind='individual'), 2,
    )
    mean = resolve_observation_uncertainty(
        ObservationUncertainty([0.1, 0.2], response_kind='mean'), 2,
    )

    np.testing.assert_array_equal(individual.sigma, mean.sigma)
    assert individual.uncertainty_kind == 'sd'
    assert mean.uncertainty_kind == 'sem'
    assert individual.to_metadata()['response_kind'] == 'individual'
    assert mean.to_metadata()['response_kind'] == 'mean'


def test_optional_counts_are_metadata_and_do_not_divide_sem_again():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(
            [0.04, 0.08], response_kind='mean', replicate_counts=[4, 16],
        ),
        2,
    )

    np.testing.assert_array_equal(resolved.sigma, [0.04, 0.08])
    np.testing.assert_array_equal(resolved.replicate_counts, [4, 16])
    assert np.issubdtype(resolved.replicate_counts.dtype, np.integer)


def test_mask_applies_to_original_uncertainty_and_count_rows_together():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(
            [0.1, np.nan, 0.3, 0.4], response_kind='mean',
            replicate_counts=[3, np.nan, 7, 9],
        ),
        4,
        row_mask=[True, False, False, True],
    )

    np.testing.assert_array_equal(resolved.sigma, [0.1, 0.4])
    np.testing.assert_array_equal(resolved.replicate_counts, [3, 9])
    assert resolved.n_observations == 2


def test_constant_inputs_broadcast_after_row_selection():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(0.1, response_kind='mean', replicate_counts=5),
        4,
        row_mask=[False, True, False, True],
    )

    np.testing.assert_array_equal(resolved.sigma, [0.1, 0.1])
    np.testing.assert_array_equal(resolved.replicate_counts, [5, 5])


@pytest.mark.parametrize('field', ['standard_uncertainty', 'replicate_counts'])
def test_pointwise_lengths_must_match_original_rows_before_masking(field):
    kwargs = dict(standard_uncertainty=[0.1, 0.2, 0.3], response_kind='mean',
                  replicate_counts=[2, 3, 4])
    kwargs[field] = [0.1, 0.2] if field == 'standard_uncertainty' else [2, 3]

    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty(**kwargs), 3, row_mask=[True, True, False],
        )


@pytest.mark.parametrize('bad', [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_retained_uncertainty_must_be_finite_and_strictly_positive(bad):
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty([0.1, bad], response_kind='mean'), 2,
        )


@pytest.mark.parametrize('bad', [0.0, -0.1, np.nan, np.inf])
def test_invalid_constant_uncertainty_is_rejected_even_when_no_rows_are_retained(bad):
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty(bad, response_kind='mean'), 2,
            row_mask=[False, False],
        )


@pytest.mark.parametrize('bad', [0, -1, 2.5, np.nan, np.inf])
def test_retained_replicate_counts_must_be_finite_positive_integers(bad):
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty(0.1, response_kind='mean', replicate_counts=[3, bad]), 2,
        )


def test_missing_retained_sigma_is_not_silently_replaced_or_dropped():
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty([np.nan, 0.2], response_kind='mean'), 2,
            row_mask=[True, False],
        )


@pytest.mark.parametrize('mask', [[1, 0], [True], [[True, False]], True])
def test_masks_must_be_one_dimensional_boolean_arrays_matching_original_rows(mask):
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(None, 2, row_mask=mask)


@pytest.mark.parametrize('count', [-1, 2.0, True])
def test_original_observation_count_requires_a_nonnegative_integer(count):
    with pytest.raises((TypeError, ValueError)):
        resolve_observation_uncertainty(None, count)


@pytest.mark.parametrize('bad', [True, [True, False], 1 + 2j, [0.1 + 0.2j],
                                 [0.1, True], [[0.1, 0.2]], 'not numeric',
                                 np.array(['2020-01-01'], dtype='datetime64[D]'),
                                 np.array(['2020-01-01'], dtype='datetime64[ns]'),
                                 np.array([2], dtype='timedelta64[D]')])
def test_structurally_invalid_uncertainty_inputs_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        ObservationUncertainty(bad, response_kind='mean')


@pytest.mark.parametrize('kwargs', [
    {'response_kind': 'unknown'},
    {'response_kind': np.array('mean')},
    {'response_kind': np.array(['mean'])},
    {'response_kind': 'mean', 'source': ' '},
    {'response_kind': 'mean', 'response_units': ''},
    {'response_kind': 'individual', 'replicate_counts': 3},
    {'response_kind': 'mean', 'replicate_counts': True},
    {'response_kind': 'mean', 'replicate_counts': [2, True]},
])
def test_invalid_interpretations_and_provenance_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ObservationUncertainty(0.1, **kwargs)


def test_input_and_resolved_arrays_are_defensive_read_only_copies():
    supplied_sigma = np.array([0.1, 0.2])
    supplied_counts = np.array([3, 5])
    uncertainty = ObservationUncertainty(
        supplied_sigma, response_kind='mean', replicate_counts=supplied_counts,
    )
    supplied_sigma[:] = 99
    supplied_counts[:] = 99
    resolved = resolve_observation_uncertainty(uncertainty, 2)

    np.testing.assert_array_equal(resolved.sigma, [0.1, 0.2])
    np.testing.assert_array_equal(resolved.replicate_counts, [3, 5])
    with pytest.raises(ValueError):
        resolved.sigma[0] = 99
    with pytest.raises(ValueError):
        resolved.replicate_counts[0] = 99
    with pytest.raises(ValueError):
        uncertainty.standard_uncertainty[0] = 99


def test_integer_counts_are_preserved_without_float_rounding():
    counts = [2**53 + 1, np.iinfo(np.int64).max]
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(0.1, response_kind='mean', replicate_counts=counts), 2,
    )
    assert resolved.replicate_counts.tolist() == counts
    assert resolved.subset([True, False]).replicate_counts.tolist() == counts[:1]


def test_out_of_range_counts_are_rejected():
    with pytest.raises(ValueError):
        resolve_observation_uncertainty(
            ObservationUncertainty(0.1, response_kind='mean', replicate_counts=2**63), 1,
        )


def test_subset_preserves_alignment_and_provenance():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(
            [0.1, 0.2, 0.3], response_kind='mean', replicate_counts=[2, 4, 8],
            source='published SEM column', response_units='fraction remaining',
        ), 3,
    )
    subset = resolved.subset([False, True, True])

    np.testing.assert_array_equal(subset.sigma, [0.2, 0.3])
    np.testing.assert_array_equal(subset.replicate_counts, [4, 8])
    assert subset.source == resolved.source
    assert subset.response_units == resolved.response_units
    assert subset.input_form == resolved.input_form
    assert subset.response_kind == resolved.response_kind
    np.testing.assert_array_equal(resolved.sigma, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        resolved.subset([True])


def test_empty_selection_is_valid_and_retains_no_missing_values():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty([np.nan], response_kind='mean', replicate_counts=[np.nan]),
        1, row_mask=[False],
    )
    assert resolved.n_observations == 0
    assert resolved.sigma.shape == (0,)
    assert resolved.replicate_counts.shape == (0,)
    assert resolve_observation_uncertainty(None, 0) is None


def test_metadata_is_json_safe_and_records_absolute_uncertainty_assumptions():
    resolved = resolve_observation_uncertainty(
        ObservationUncertainty(
            0.05, response_kind='mean', replicate_counts=4,
            source='assay summary', response_units='mg',
        ), 2,
    )
    metadata = resolved.to_metadata()

    assert metadata == {
        'standard_uncertainty': [0.05, 0.05],
        'response_kind': 'mean',
        'uncertainty_kind': 'sem',
        'input_form': 'constant',
        'scale': 'absolute',
        'independent_observations': True,
        'source': 'assay summary',
        'response_units': 'mg',
        'replicate_counts': [4, 4],
    }
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    metadata['standard_uncertainty'][0] = 99
    assert resolved.sigma[0] == 0.05


def test_direct_resolved_objects_require_valid_pointwise_values():
    with pytest.raises(ValueError):
        ResolvedObservationUncertainty([0.1, np.nan], response_kind='mean')
    with pytest.raises(ValueError):
        ResolvedObservationUncertainty(0.1, response_kind='mean')
    with pytest.raises(ValueError):
        ResolvedObservationUncertainty([0.1, 0.2], response_kind='mean', replicate_counts=[3])


def test_resolved_input_form_must_be_a_string_for_json_metadata():
    with pytest.raises(ValueError):
        ResolvedObservationUncertainty([0.1], response_kind='mean', input_form=np.array('constant'))


def test_legacy_time_course_contract_does_not_acquire_implicit_uncertainty():
    course = TimeCourse([0.0, 1.0], [1.0, 0.5], label='in_vivo')

    np.testing.assert_array_equal(course.time, [0.0, 1.0])
    np.testing.assert_array_equal(course.value, [1.0, 0.5])
    assert course.label == 'in_vivo'
