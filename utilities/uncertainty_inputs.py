"""Shared browser/CLI parsing of optional absolute SD or SEM inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivivc_core import ObservationUncertainty


SOURCES = ('in_vitro', 'in_vivo', 'prediction')
LABELS = {'in_vitro': 'In vitro calibration', 'in_vivo': 'In vivo calibration',
          'prediction': 'Prediction input'}


def _mode(options, prefix):
    mode = options.get(prefix + '_uncertainty_mode', 'none') or 'none'
    if mode not in ('none', 'constant', 'column'):
        raise ValueError(f'{LABELS[prefix]}: choose none, constant, or column uncertainty.')
    return mode


def validate_uncertainty_options(options, selected_approaches, selected_scalings, *, has_prediction):
    """Validate the supported workflow before fitting; omitted inputs stay legacy."""
    modes = {source: _mode(options, source) for source in SOURCES}
    future, _ = future_uncertainty_from_options(options)
    active = any(mode != 'none' for mode in modes.values()) or future is not None
    if not active:
        return False
    if set(selected_approaches) != {'approach2'}:
        raise ValueError('Supplied uncertainty is available with Approach 2 only. Select Approach 2 on its own.')
    if 'log_y' in selected_scalings:
        raise ValueError('Log10 response scaling cannot be used with supplied SD/SEM. Remove log10 response scaling or omit uncertainty.')
    if modes['prediction'] != 'none' and not has_prediction:
        raise ValueError('Upload a prediction dataset to supply prediction-input uncertainty.')
    if future is not None:
        if not has_prediction:
            raise ValueError('Future measurement variability requires a prediction dataset.')
        if future == 'prediction' and modes['prediction'] == 'none':
            raise ValueError('Reusing prediction uncertainty requires supplied prediction SD/SEM.')
        if future == 'calibration' and modes['in_vivo'] == 'none':
            raise ValueError('Calibration reuse requires supplied in vivo SD/SEM.')
    confirmed = options.get('uncertainty_independence_confirmed', False)
    if confirmed is not True and str(confirmed).lower() not in ('on', 'true', '1', 'yes'):
        raise ValueError('Confirm that original measurement errors are independent across observations and times, that prediction measurements are independent of calibration, and that future measurements are independent of entered data.')
    return True


def future_uncertainty_from_options(options, preprocessing_metadata=None):
    """Parse future variability and transport a constant from original in vivo units.

    Reused source errors are already on analyzed scales and are resolved by the
    backend at prediction time. A constant follows the fixed in vivo calibration
    normalization references once, independently of the prediction-input scale.
    """
    mode = options.get('future_uncertainty_mode', 'none') or 'none'
    if mode == 'none':
        return None, None
    if mode in ('prediction', 'calibration'):
        return mode, {'mode': mode}
    if mode != 'constant':
        raise ValueError('Future uncertainty: choose none, constant, prediction, or calibration.')
    kind = options.get('future_response_kind', 'individual')
    if kind not in ('individual', 'mean'):
        raise ValueError('Future uncertainty: choose individual measurements (SD) or reported means (SEM).')
    try:
        entered = options.get('future_uncertainty_value', '')
        if isinstance(entered, (bool, np.bool_)):
            raise ValueError('Boolean uncertainty is not a standard deviation.')
        entered = float(entered)
    except (TypeError, ValueError) as exc:
        raise ValueError('Future uncertainty: enter a positive absolute SD or SEM.') from exc
    if not np.isfinite(entered) or entered <= 0:
        raise ValueError('Future uncertainty: the constant SD or SEM must be finite and positive.')
    metadata = preprocessing_metadata or {}
    if 'log_y' in metadata.get('applied_scalings', ()):
        raise ValueError('Log10 response scaling cannot be used with future SD/SEM.')
    analyzed = entered
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        for record in metadata.get('normalization_records', ()):
            if record.get('dataset') != 'in_vivo' or not record.get('applied', True):
                continue
            operation = record['operation']
            if operation == 'one_minus':
                continue
            if operation == 'initial_value':
                divisor = record.get('reference')
            elif operation == 'min_max':
                minimum, maximum = record.get('minimum'), record.get('maximum')
                divisor = None if minimum is None or maximum is None else maximum - minimum
            else:
                raise ValueError('This normalization is unavailable with future SD/SEM.')
            if divisor is None or not np.isfinite(divisor) or divisor == 0 or (operation == 'min_max' and divisor < 0):
                raise ValueError('Future SD/SEM requires a finite, valid in vivo normalization reference.')
            analyzed /= abs(divisor)
    if not np.isfinite(analyzed) or analyzed <= 0:
        raise ValueError('Future SD/SEM is not finite and positive on the analyzed response scale.')
    supplied = ObservationUncertainty(analyzed, response_kind=kind, source='Future in vivo measurement: constant')
    return supplied, {
        'mode': mode, 'response_kind': kind,
        'uncertainty_kind': 'sd' if kind == 'individual' else 'sem',
        'entered_standard_uncertainty': entered,
        'analyzed_standard_uncertainty': analyzed,
        'normalization_constants_fixed': True,
    }


def _column(frame, name, label):
    names = [column for column in frame.columns if str(column).strip() == name.strip()]
    if not name or len(names) != 1:
        raise ValueError(f'{label}: enter one unambiguous existing column header.')
    # Preserve row alignment. Invalid retained values become validation errors
    # in core preprocessing; missing/padded response rows use the same mask.
    return pd.to_numeric(frame[names[0]], errors='coerce').to_numpy()


def uncertainty_from_options(frame, prefix, options):
    """Construct raw uncertainty; core preprocessing validates retained rows."""
    mode = _mode(options, prefix)
    if mode == 'none':
        return None, None
    label = LABELS[prefix]
    kind = options.get(prefix + '_response_kind', 'individual')
    if kind not in ('individual', 'mean'):
        raise ValueError(f'{label}: choose individual measurements (SD) or reported means (SEM).')
    column = str(options.get(prefix + '_uncertainty_column', '') or '').strip()
    if mode == 'constant':
        try:
            entered = options.get(prefix + '_uncertainty_value', '')
            if isinstance(entered, (bool, np.bool_)):
                raise ValueError('Boolean uncertainty is not a standard deviation.')
            values = float(entered)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label}: enter a positive absolute SD or SEM.') from exc
        if not np.isfinite(values) or values <= 0:
            raise ValueError(f'{label}: the constant SD or SEM must be finite and positive.')
    else:
        values = _column(frame, column, label + ' uncertainty')
    count_column = str(options.get(prefix + '_replicate_column', '') or '').strip()
    if count_column and kind != 'mean':
        raise ValueError(f'{label}: replicate counts are optional metadata for reported means only.')
    counts = _column(frame, count_column, label + ' replicate count') if count_column else None
    source = f'{label}: ' + (f'column {column}' if mode == 'column' else 'constant')
    supplied = ObservationUncertainty(values, response_kind=kind, replicate_counts=counts, source=source)
    raw_values = np.full(len(frame), values) if np.ndim(values) == 0 else np.asarray(values)
    record = {
        'mode': mode, 'response_kind': kind,
        'uncertainty_kind': 'sd' if kind == 'individual' else 'sem',
        'constant': float(values) if mode == 'constant' else None,
        'column': column if mode == 'column' else None,
        'replicate_column': count_column or None,
        'standard_uncertainty': raw_values.tolist(),
        'replicate_counts': None if counts is None else counts.tolist(),
        'source_row_count': len(frame),
    }
    return supplied, record


def make_uncertainty_record(raw_record, course, preprocessing, source_name='in_vitro'):
    """Retain original and analyzed errors for reporting without rescaling twice."""
    if raw_record is None:
        return None
    if getattr(course, 'observation_uncertainty', None) is None:
        raise ValueError('Supplied uncertainty is missing from the analyzed time course.')
    record = dict(raw_record)
    record['analyzed'] = course.observation_uncertainty.to_metadata()
    source = preprocessing.sources.get(source_name)
    if source is None and len(preprocessing.sources) == 1:
        source = next(iter(preprocessing.sources.values()))
    if source is None:
        raise ValueError('Supplied uncertainty is missing its preprocessing source record.')
    record['retained_rows'] = source.retained_rows.tolist()
    record['raw_time'] = source.time.tolist()
    record['raw_response'] = source.value.tolist()
    record['source_row_count'] = len(source.value)
    return record
