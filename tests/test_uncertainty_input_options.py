"""Browser and CLI use the same SD/SEM parsing and preprocessing record."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivivc_core import prepare_time_course
from utilities.uncertainty_inputs import (
    make_uncertainty_record, uncertainty_from_options, validate_uncertainty_options,
)


def options(**changes):
    result = {'in_vitro_uncertainty_mode': 'constant', 'in_vitro_uncertainty_value': '1',
              'in_vitro_response_kind': 'mean', 'uncertainty_independence_confirmed': 'on'}
    result.update(changes)
    return result


def frame():
    return pd.DataFrame({'t': [0., 1., np.nan], 'y': [10., 8., np.nan],
                         'SEM': [1., 2., np.nan], 'n': [4., 9., np.nan]})


def test_omitted_uncertainty_keeps_the_legacy_path():
    assert not validate_uncertainty_options({}, ['approach1'], ['log_y'], has_prediction=False)
    assert uncertainty_from_options(frame(), 'in_vitro', {}) == (None, None)


@pytest.mark.parametrize('mode', ['constant', 'column'])
def test_input_parser_preserves_mean_sem_count_metadata_and_original_rows(mode):
    supplied, raw = uncertainty_from_options(frame(), 'in_vitro', options(
        in_vitro_uncertainty_mode=mode, in_vitro_uncertainty_column='SEM',
        in_vitro_replicate_column='n',
    ))
    course, preprocessing = prepare_time_course(frame().t, frame().y,
        selected_normalizations=['initial_value'], observation_uncertainty=supplied)
    expected = [.1, .1] if mode == 'constant' else [.1, .2]
    np.testing.assert_allclose(course.observation_uncertainty.sigma, expected)
    np.testing.assert_array_equal(course.observation_uncertainty.replicate_counts, [4, 9])
    record = make_uncertainty_record(raw, course, preprocessing, source_name='profile')
    assert record['source_row_count'] == 3
    assert record['retained_rows'] == [0, 1]
    assert record['raw_time'][:2] == [0., 1.]
    assert record['raw_response'][:2] == [10., 8.]
    assert record['analyzed']['uncertainty_kind'] == 'sem'
    assert record['analyzed']['response_units'] == 'dimensionless'
    assert record['mode'] == mode
    assert record['replicate_column'] == 'n'


@pytest.mark.parametrize('value', ['', 'NaN', 'inf', '0', '-1', True])
def test_invalid_constant_uncertainty_is_not_accepted(value):
    with pytest.raises(ValueError):
        uncertainty_from_options(frame(), 'in_vitro', options(in_vitro_uncertainty_value=value))


def test_invalid_retained_error_column_is_an_error_not_a_row_filter():
    data = frame()
    data['SEM'] = data['SEM'].astype(object)
    data.loc[1, 'SEM'] = 'bad value'
    supplied, _ = uncertainty_from_options(data, 'in_vitro', options(
        in_vitro_uncertainty_mode='column', in_vitro_uncertainty_column='SEM'))
    with pytest.raises(ValueError):
        prepare_time_course(data.t, data.y, observation_uncertainty=supplied)


def test_missing_and_ambiguous_column_names_are_rejected():
    for data, column in ((frame(), 'missing'),
                         (pd.DataFrame([[.1, .2]], columns=['SEM', ' SEM ']), 'SEM')):
        with pytest.raises(ValueError):
            uncertainty_from_options(data, 'in_vitro', options(
                in_vitro_uncertainty_mode='column', in_vitro_uncertainty_column=column))


def test_optional_counts_cannot_be_misinterpreted_as_sem_divisors_for_individuals():
    with pytest.raises(ValueError):
        uncertainty_from_options(frame(), 'in_vitro', options(
            in_vitro_response_kind='individual', in_vitro_replicate_column='n'))


@pytest.mark.parametrize('approaches,scalings,extra', [
    (['approach1'], [], {}), (['approach2', 'approach3'], [], {}),
    (['approach2'], ['log_y'], {}),
    (['approach2'], [], {'uncertainty_independence_confirmed': False}),
    (['approach2'], [], {'prediction_uncertainty_mode': 'constant'}),
])
def test_workflow_restrictions_are_shared_by_browser_and_cli(approaches, scalings, extra):
    with pytest.raises(ValueError):
        validate_uncertainty_options(options(**extra), approaches, scalings, has_prediction=False)


def test_valid_uncertainty_requires_no_replicate_count_or_future_noise_assumption():
    assert validate_uncertainty_options(options(), ['approach2'], [], has_prediction=False)
    supplied, raw = uncertainty_from_options(frame(), 'in_vitro', options())
    assert supplied.replicate_counts is None
    assert raw['replicate_counts'] is None
