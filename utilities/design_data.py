"""Request-independent file reading and input interpretation for study design.

Files are supplied as file-like objects and are never saved by this module.
The returned numerical values remain on their supplied scales; fitting,
normalization, interpolation, and future measurement simulation happen later.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time
import io
from numbers import Integral, Real
from pathlib import PurePath
import re
from typing import Any

import numpy as np
import pandas as pd

from ivivc_core import (
    ObservationUncertainty, ResolvedObservationUncertainty, TimeCourse,
    resolve_observation_uncertainty,
)
from study_design_core import EmpiricalTimeScaleDistribution, ObservationModel


PREVIEW_ROWS = 10


class DataImportError(ValueError):
    """A correctable input issue, associated with a mapping/option field."""

    def __init__(self, message: str, field: str = 'file'):
        super().__init__(message)
        self.field = field


@dataclass
class LoadedTable:
    frame: pd.DataFrame | None
    sheets: list[str]
    sheet: str | None
    filename: str


@dataclass
class ImportedData:
    value: Any
    summary: dict
    rows: list[dict]
    columns: list[str]
    omitted_rows: list[int]


def _blank(value):
    if isinstance(value, str):
        return not value.strip()
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _headers(values):
    names = []
    for value in values:
        if _blank(value):
            raise DataImportError('Column headers must be nonempty and unique.')
        name = str(value).strip()
        if not name or name in names:
            raise DataImportError('Column headers must be nonempty and unique.')
        names.append(name)
    if not names:
        raise DataImportError('The file needs a header row and data rows.')
    return names


def read_upload(upload, sheet=None) -> LoadedTable:
    """Read a CSV/Excel upload, retaining headers before pandas can rename them.

    Multiple-sheet workbooks return their sheet names without choosing a
    worksheet. A one-sheet workbook can be read directly. CSV blank records
    are retained as empty rows so interpretation can report padded observations.
    """
    filename = str(getattr(upload, 'filename', '') or '')
    extension = PurePath(filename).suffix.lower()
    if extension not in ('.csv', '.xls', '.xlsx'):
        raise DataImportError('Choose a CSV, XLS, or XLSX file.')
    try:
        upload.seek(0)
        content = upload.read()
    except (AttributeError, OSError, ValueError) as exc:
        raise DataImportError('The uploaded file could not be read. Select it again.') from exc
    if not content:
        raise DataImportError('The uploaded file is empty.')
    if extension == '.csv':
        try:
            text = content.decode('utf-8-sig') if isinstance(content, bytes) else str(content).lstrip('\ufeff')
            reader = csv.reader(io.StringIO(text, newline=''), strict=True)
            header = _headers(next(reader))
            records = []
            for record_number, row in enumerate(reader, start=2):
                if not row:
                    row = [''] * len(header)
                if len(row) != len(header):
                    raise DataImportError(
                        f'CSV row {record_number} has {len(row)} fields; expected {len(header)}.'
                    )
                records.append(row)
        except DataImportError:
            raise
        except (UnicodeError, csv.Error, StopIteration) as exc:
            raise DataImportError('The CSV could not be read. Use UTF-8 text with a valid header and matching fields.') from exc
        if not records:
            raise DataImportError('The file has column headers but no data rows.')
        return LoadedTable(pd.DataFrame(records, columns=header), [], None, filename)

    try:
        workbook = pd.ExcelFile(io.BytesIO(content), engine='openpyxl' if extension == '.xlsx' else 'xlrd')
        with workbook:
            sheets = list(workbook.sheet_names)
            if not sheets:
                raise DataImportError('The workbook contains no readable worksheets.', 'sheet')
            chosen = str(sheet) if sheet is not None and str(sheet) else None
            if chosen is None:
                if len(sheets) != 1:
                    return LoadedTable(None, sheets, None, filename)
                chosen = sheets[0]
            if chosen not in sheets:
                raise DataImportError('Select an existing worksheet from this workbook.', 'sheet')
            raw = pd.read_excel(workbook, sheet_name=chosen, header=None, dtype=object)
    except DataImportError:
        raise
    except ImportError as exc:
        raise DataImportError('Excel support is unavailable. Install the app requirements or upload a CSV file.') from exc
    except Exception as exc:
        raise DataImportError('The workbook could not be read. Check its format and selected worksheet.') from exc
    if raw.empty:
        raise DataImportError('The selected worksheet is empty.', 'sheet')
    header = _headers(raw.iloc[0].tolist())
    frame = raw.iloc[1:].copy().reset_index(drop=True)
    frame.columns = header
    if frame.empty:
        raise DataImportError('The worksheet has column headers but no data rows.', 'sheet')
    return LoadedTable(frame, sheets, chosen, filename)


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataImportError('The selected table needs at least one data row.')
    headers = _headers(frame.columns)
    if headers != list(frame.columns):
        frame = frame.copy()
        frame.columns = headers
    return frame


def _mapping(frame, options, key, default=None, *, optional=False):
    supplied = options[key] if key in options else default
    name = str(supplied).strip() if supplied is not None else ''
    if optional and not name:
        return None
    if not name or name not in frame.columns:
        raise DataImportError('Select an existing column header.', key)
    return name


def _number(value, field, *, row=None, column=None, count=False):
    location = f'Row {row}, column {column!r}: ' if row is not None else ''
    invalid = isinstance(value, (bool, np.bool_, complex, np.complexfloating, date, datetime, time, np.datetime64))
    try:
        if invalid or _blank(value) or not isinstance(value, (str, Real)):
            raise ValueError
        if count and (isinstance(value, Integral) or (isinstance(value, str) and re.fullmatch(r'[+-]?\d+', value.strip()))):
            result = int(value)
            if not 1 <= result <= np.iinfo(np.int64).max:
                raise ValueError
            return result
        result = float(value)
        if not np.isfinite(result):
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        message = ('enter a positive integer count within the int64 range.' if count
                   else 'enter a finite real number; blanks, booleans, dates, and text are not valid here.')
        raise DataImportError(location + message, field) from exc


def _values(frame, column, field, indices=None, *, count=False):
    indices = list(range(len(frame))) if indices is None else list(indices)
    values = [_number(frame.iloc[index][column], field, row=index + 2, column=column, count=count)
              for index in indices]
    return np.asarray(values, dtype=object if count else float)


def _option_values(frame, options, prefix, indices, *, count=False):
    source = options.get(prefix + '_source', 'none' if count else 'constant')
    if source == 'none' and count:
        return None
    if source == 'constant':
        return _number(options.get(prefix + '_value'), prefix + '_value', count=count)
    if source == 'column':
        column = _mapping(frame, options, prefix + '_column')
        return _values(frame, column, prefix + '_column', indices, count=count)
    raise DataImportError('Choose a constant or a supplied column' + (' (or no count).' if count else '.'), prefix + '_source')


def _range(values):
    return [float(np.min(values)), float(np.max(values))]


def interpret_time_course(frame, options, label='Imported time course') -> ImportedData:
    """Resolve original-scale observations, including explicit mean SD to SEM.

    Only rows with both mapped time and response absent are omitted. Their
    source row numbers are reported. Partial or malformed rows are errors;
    repeated times and input order remain unchanged.
    """
    frame = _frame(frame)
    time_column = _mapping(frame, options, 'time_column', frame.columns[0])
    response_column = _mapping(frame, options, 'response_column', frame.columns[1] if len(frame.columns) > 1 else None)
    if time_column == response_column:
        raise DataImportError('Select distinct time and response columns.', 'response_column')
    retained, omitted = [], []
    for index in range(len(frame)):
        if _blank(frame.iloc[index][time_column]) and _blank(frame.iloc[index][response_column]):
            omitted.append(index + 2)
        else:
            retained.append(index)
    if not retained:
        raise DataImportError('No observations remain: all mapped time/response pairs are blank.', 'time_column')
    times = _values(frame, time_column, 'time_column', retained)
    values = _values(frame, response_column, 'response_column', retained)
    mode = options.get('uncertainty', 'none')
    if mode not in ('none', 'individual_sd', 'mean_sd', 'mean_sem'):
        raise DataImportError('Choose unspecified uncertainty, individual SD, mean SD with count, or mean SEM.', 'uncertainty')
    units = options.get('response_units')
    if units is not None and not isinstance(units, str):
        raise DataImportError('Response units must be text.', 'response_units')
    units = units.strip() or None if units is not None else None
    uncertainty, supplied_error, counts = None, None, None
    interpretation = 'Uncertainty is unspecified; existing fitting behavior is retained.'
    if mode != 'none':
        supplied_error = _option_values(frame, options, 'error', retained)
        counts = _option_values(frame, options, 'count', retained, count=True)
        mean = mode in ('mean_sd', 'mean_sem')
        if not mean and counts is not None:
            raise DataImportError('Replicate counts describe reported means; do not attach a count to an individual response.', 'count_source')
        if mode == 'mean_sd' and counts is None:
            raise DataImportError('Means with replicate SD require independent replicate counts.', 'count_source')
        source = f'{label}: ' + ('replicate SD converted to SEM' if mode == 'mean_sd' else 'supplied absolute SD/SEM')
        try:
            supplied = ObservationUncertainty(
                supplied_error, response_kind='mean' if mean else 'individual',
                replicate_counts=counts, source=source, response_units=units,
            )
            uncertainty = resolve_observation_uncertainty(supplied, len(retained))
        except (TypeError, ValueError, OverflowError) as exc:
            field = ('count_column' if options.get('count_source') == 'column' else 'count_value') if 'count' in str(exc) else (
                'error_column' if options.get('error_source') == 'column' else 'error_value')
            raise DataImportError(str(exc), field) from exc
        if mode == 'mean_sd':
            try:
                uncertainty = ResolvedObservationUncertainty(
                    uncertainty.sigma / np.sqrt(uncertainty.replicate_counts.astype(float)),
                    response_kind='mean', input_form=uncertainty.input_form,
                    replicate_counts=uncertainty.replicate_counts,
                    source=source, response_units=units,
                )
            except ValueError as exc:
                field = 'error_column' if options.get('error_source') == 'column' else 'error_value'
                raise DataImportError('The resulting SEM must be finite and positive.', field) from exc
            interpretation = 'Reported means: replicate SD is divided by the square root of the independent count once; the resulting SEM describes uncertainty in each reported mean.'
        elif mode == 'mean_sem':
            interpretation = 'Reported means: supplied SEM is retained directly as observation uncertainty. Optional replicate counts are metadata; there is no further division.'
        else:
            interpretation = 'Individual observations: supplied absolute SD is retained directly as uncertainty in each reported response.'
    course = TimeCourse(times, values, label=label, observation_uncertainty=uncertainty)
    columns = ['Source row', 'Time', 'Response']
    if uncertainty is not None:
        columns.append('Supplied SEM' if mode == 'mean_sem' else 'Supplied SD')
        if uncertainty.replicate_counts is not None:
            columns.append('Independent count')
        columns.append('Observation SEM' if uncertainty.response_kind == 'mean' else 'Observation SD')
    rows = []
    raw_errors = None if supplied_error is None else np.broadcast_to(np.asarray(supplied_error), (len(retained),))
    for position, original in enumerate(retained[:PREVIEW_ROWS]):
        row = {'Source row': original + 2, 'Time': float(times[position]), 'Response': float(values[position])}
        if uncertainty is not None:
            row['Supplied SEM' if mode == 'mean_sem' else 'Supplied SD'] = float(raw_errors[position])
            if uncertainty.replicate_counts is not None:
                row['Independent count'] = int(uncertainty.replicate_counts[position])
            row['Observation SEM' if uncertainty.response_kind == 'mean' else 'Observation SD'] = float(uncertainty.sigma[position])
        rows.append(row)
    summary = {
        'source_rows': len(frame), 'retained_rows': len(retained), 'omitted_rows': len(omitted),
        'interpretation': interpretation, 'time_range': _range(times),
        'response_range': _range(values), 'response_units': units,
        'uncertainty': mode, 'response_scale': 'supplied; no normalization applied',
        'independent_errors_assumed': True,
        'mapping': {'Time': time_column, 'Response': response_column},
    }
    if uncertainty is not None:
        summary['mapping']['Supplied SD/SEM'] = (options['error_column']
            if options.get('error_source') == 'column' else 'constant value')
        if uncertainty.replicate_counts is not None:
            summary['mapping']['Independent count'] = (options['count_column']
                if options.get('count_source') == 'column' else 'constant count')
    return ImportedData(course, summary, rows, columns, omitted)


def interpret_sd_profile(frame, options) -> ImportedData:
    """Validate a future individual-SD profile without estimating measurement SD."""
    frame = _frame(frame)
    time_column = _mapping(frame, options, 'time_column', 'time' if 'time' in frame else frame.columns[0])
    sd_column = _mapping(frame, options, 'sd_column', 'sd' if 'sd' in frame else (frame.columns[1] if len(frame.columns) > 1 else None))
    if time_column == sd_column:
        raise DataImportError('Select distinct time and individual-SD columns.', 'sd_column')
    times = _values(frame, time_column, 'time_column')
    sd = _values(frame, sd_column, 'sd_column')
    try:
        model = ObservationModel(kind='time_profile_sd', profile_times=times, profile_sd=sd,
                                 source='uploaded individual-SD profile')
    except (TypeError, ValueError) as exc:
        raise DataImportError(str(exc), 'time_column' if 'increasing' in str(exc) else 'sd_column') from exc
    columns = ['Time', 'Individual SD']
    rows = [{'Time': float(t), 'Individual SD': float(s)} for t, s in zip(times[:PREVIEW_ROWS], sd[:PREVIEW_ROWS])]
    summary = {
        'source_rows': len(frame), 'retained_rows': len(frame), 'omitted_rows': 0,
        'time_range': _range(times), 'individual_sd_range': _range(sd),
        'interpretation': 'Future individual-measurement SD: nearest-time lookup with endpoint extension. Replicate counts are applied separately during planning.',
        'profile_rule': 'nearest_time_with_endpoint_extension',
        'mapping': {'Time': time_column, 'Individual SD': sd_column},
    }
    return ImportedData(model, summary, rows, columns, [])


def interpret_time_scales(frame, options) -> ImportedData:
    """Retain empirical ratio records and normalize their supplied weights."""
    frame = _frame(frame)
    ratio_column = _mapping(frame, options, 'ratio_column', 'ratio' if 'ratio' in frame else frame.columns[0])
    weight_column = _mapping(frame, options, 'weight_column', 'weight' if 'weight' in frame else None, optional=True)
    source_column = _mapping(frame, options, 'source_column', 'source_id' if 'source_id' in frame else None, optional=True)
    ratios = _values(frame, ratio_column, 'ratio_column')
    weights = _values(frame, weight_column, 'weight_column') if weight_column else None
    identifiers = None
    if source_column:
        identifiers = []
        for index, value in enumerate(frame[source_column]):
            if _blank(value):
                raise DataImportError(f'Row {index + 2}: source identifiers must be nonempty.', 'source_column')
            identifiers.append(str(value).strip())
    try:
        distribution = EmpiricalTimeScaleDistribution(
            ratios, weights=weights, source_ids=identifiers,
            metadata={'input_source': 'upload', 'ratio_definition': 'tau_in_vivo / tau_in_vitro'},
        )
    except (TypeError, ValueError) as exc:
        field = 'weight_column' if 'weight' in str(exc) else 'ratio_column'
        raise DataImportError(str(exc), field) from exc
    columns = ['Ratio', 'Supplied weight', 'Normalized weight']
    if identifiers is not None:
        columns.append('Source ID')
    raw_weights = np.ones(len(ratios)) if weights is None else weights
    rows = []
    for index in range(min(len(frame), PREVIEW_ROWS)):
        row = {'Ratio': float(ratios[index]), 'Supplied weight': float(raw_weights[index]),
               'Normalized weight': float(distribution.weights[index])}
        if identifiers is not None:
            row['Source ID'] = identifiers[index]
        rows.append(row)
    summary = {
        'source_rows': len(frame), 'retained_rows': len(frame), 'omitted_rows': 0,
        'ratio_range': _range(ratios), 'weights': 'equal' if weights is None else 'supplied',
        'interpretation': 'Empirical in vivo/in vitro time-scale ratios; ' +
                          ('equal record weights.' if weights is None else 'supplied nonnegative weights normalized to sum to one.'),
        'ratio_definition': 'tau_in_vivo / tau_in_vitro',
        'mapping': {'Ratio': ratio_column, 'Weight': weight_column or 'equal weights',
                    'Source ID': source_column or 'not supplied'},
    }
    return ImportedData(distribution, summary, rows, columns, [])
