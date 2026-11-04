"""Browser and CLI future inputs share validation and fixed-scale transport."""
import numpy as np
import pytest

from utilities.uncertainty_inputs import (
    future_uncertainty_from_options, validate_uncertainty_options,
)


def options(**changes):
    values = {'future_uncertainty_mode': 'constant', 'future_uncertainty_value': '6',
              'future_response_kind': 'mean', 'uncertainty_independence_confirmed': 'on'}
    values.update(changes)
    return values


def test_no_future_option_preserves_legacy_validation():
    assert future_uncertainty_from_options({}) == (None, None)
    assert not validate_uncertainty_options({}, ['approach1'], ['log_y'], has_prediction=False)


def test_future_constant_alone_requires_no_historical_error_model():
    assert validate_uncertainty_options(options(), ['approach2'], [], has_prediction=True)
    supplied, record = future_uncertainty_from_options(options())
    assert supplied.standard_uncertainty == 6.
    assert supplied.response_kind == 'mean'
    assert supplied.replicate_counts is None
    assert record['entered_standard_uncertainty'] == record['analyzed_standard_uncertainty'] == 6.


def test_future_constant_uses_only_fixed_in_vivo_normalizations_once():
    metadata = {'normalization_records': [
        {'dataset': 'in_vitro', 'operation': 'initial_value', 'reference': 100.},
        {'dataset': 'in_vivo', 'operation': 'initial_value', 'reference': -20.},
        {'dataset': 'in_vivo', 'operation': 'min_max', 'minimum': .1, 'maximum': .7},
        {'dataset': 'in_vivo', 'operation': 'one_minus', 'reference': 1.},
    ]}
    mean, record = future_uncertainty_from_options(options(), metadata)
    individual, _ = future_uncertainty_from_options(options(future_response_kind='individual'), metadata)
    np.testing.assert_allclose(mean.standard_uncertainty, .5)
    np.testing.assert_array_equal(mean.standard_uncertainty, individual.standard_uncertainty)
    assert record['entered_standard_uncertainty'] == 6.
    assert record['normalization_constants_fixed'] is True


def test_zero_initial_value_noop_keeps_future_constant_unchanged():
    supplied, _ = future_uncertainty_from_options(options(), {'normalization_records': [
        {'dataset': 'in_vivo', 'operation': 'initial_value', 'reference': 0., 'applied': False},
    ]})
    assert supplied.standard_uncertainty == 6.


@pytest.mark.parametrize('mode', ['prediction', 'calibration'])
def test_reused_errors_are_not_normalized_again(mode):
    spec, record = future_uncertainty_from_options(options(future_uncertainty_mode=mode), {
        'normalization_records': [{'dataset': 'in_vivo', 'operation': 'initial_value', 'reference': 100.}],
    })
    assert spec == mode and record == {'mode': mode}


@pytest.mark.parametrize('value', ['', 'nan', 'inf', 0, -1, True])
def test_future_constant_requires_a_finite_positive_error(value):
    with pytest.raises(ValueError):
        future_uncertainty_from_options(options(future_uncertainty_value=value))


@pytest.mark.parametrize('metadata', [
    {'applied_scalings': ['log_y']},
    {'normalization_records': [{'dataset': 'in_vivo', 'operation': 'initial_value', 'reference': 0.}]},
    {'normalization_records': [{'dataset': 'in_vivo', 'operation': 'min_max', 'minimum': 1., 'maximum': 1.}]},
    {'normalization_records': [{'dataset': 'in_vivo', 'operation': 'min_max', 'minimum': 2., 'maximum': 1.}]},
])
def test_invalid_future_response_scale_is_rejected(metadata):
    with pytest.raises(ValueError):
        future_uncertainty_from_options(options(), metadata)


@pytest.mark.parametrize('changes,approaches,scalings,has_prediction', [
    ({}, ['approach1'], [], True),
    ({}, ['approach2', 'approach3'], [], True),
    ({}, ['approach2'], ['log_y'], True),
    ({}, ['approach2'], [], False),
    ({'uncertainty_independence_confirmed': False}, ['approach2'], [], True),
    ({'future_uncertainty_mode': 'prediction'}, ['approach2'], [], True),
    ({'future_uncertainty_mode': 'calibration'}, ['approach2'], [], True),
    ({'future_uncertainty_mode': 'other'}, ['approach2'], [], True),
    ({'future_response_kind': 'other'}, ['approach2'], [], True),
])
def test_future_workflow_restrictions_are_enforced(changes, approaches, scalings, has_prediction):
    with pytest.raises(ValueError):
        validate_uncertainty_options(options(**changes), approaches, scalings, has_prediction=has_prediction)


@pytest.mark.parametrize('mode,source', [('prediction', 'prediction'), ('calibration', 'in_vivo')])
def test_source_reuse_requires_that_sources_supplied_error(mode, source):
    values = options(future_uncertainty_mode=mode)
    values[source + '_uncertainty_mode'] = 'constant'
    assert validate_uncertainty_options(values, ['approach2'], [], has_prediction=True)
