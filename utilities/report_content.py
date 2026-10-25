"""Run-specific report content shared by the browser preview and Word export.

This layer reads completed results. It does not fit, cross-validate, rank, or
sample uncertainty distributions. Tables are plain rows rather than browser DOM
snapshots, so pagination cannot change the numerical record.
"""

from html import unescape
from html.parser import HTMLParser
import math

import numpy as np

from metrics import metrics
from utilities.descriptions import CV_SCHEME_LABELS, PREPROCESSING_DESCRIPTIONS
from utilities.prediction_validity import tau_estimate_is_available, tau_uncertainty_is_available


REPORT_DETAIL_FIELDS = (
    ('analysis_title', 'Analysis title'),
    ('analysis_purpose', 'Analysis purpose and intended use'),
    ('material_device', 'Material or device'),
    ('endpoint', 'Measured response'),
    ('time_units', 'Time units'),
    ('response_units', 'Response units'),
    ('source_study_ids', 'Study and source identifiers'),
    ('selection_rationale', 'Model selection rationale'),
)

_CV_EXCLUDED = {'adjusted_r_squared', 'aic', 'aicc', 'bic', 'nrmse', 'mnrmse'}
_METHODS = {
    'direct_time_correlation': (
        'In vivo time is calculated from in vitro time using the fitted time relationship. '
        'The prediction response is carried into the plot but is not used to calculate the mapped time.',
        't_vivo,pred = f(t_vitro,pred)',
        'Horizontal intervals propagate fitted time-mapping parameters at fixed input times using the common parameter-only percentile calculation. The fitted relationship can be evaluated beyond its calibration range.'),
    'value_ratio': (
        'Both fitted calibration curves are evaluated at each prediction time. Their in vivo/in vitro response ratio scales the supplied prediction response.',
        'y_vivo,pred = y_vitro,pred × f_vivo(t) / f_vitro(t)',
        'Vertical intervals show approximate 95% parameter uncertainty conditional on fixed prediction times and responses. '
        'Independent multivariate normal draws from the two fitted parameter covariance matrices preserve covariance within each fit. '
        'The input response multiplies each sampled ratio; bounds are pointwise 2.5th and 97.5th percentiles, while the point estimate uses the fitted parameters. '
        'No additional observation noise, residual scatter, model-selection uncertainty, or model inadequacy is included; these are not prediction intervals for future measurements. '
        'A bounded interval is withheld when any requested draw is unusable at that time, sampling fails, or the sampled denominator interval includes zero. '
        'The zero-crossing check is a conservative display safeguard, not a formal ratio-confidence method. '
        'The ratio can become unstable near a zero fitted in vitro response. Prediction times may extend beyond either observed calibration time range.'),
    'tau_ratio': (
        'The prediction time is scaled by the ratio of the fitted in vivo and in vitro time constants. The prediction response is not used to calculate that time.',
        't_vivo,pred = t_vitro,pred × tau_vivo / tau_vitro',
        'Horizontal parameter-uncertainty intervals propagate independent draws of both fitted parameter vectors through checked tau integration and time scaling. Failed integrations make the interval unavailable; the central point uses fitted-parameter taus. '
        'A time constant summarizes the whole fitted curve; this scaling may miss differences in curve shape. Input and output times are not restricted to observed ranges.'),
    'observed_response_inverse': (
        'Each preprocessed observed prediction response is mapped to the time at which the fitted in vivo calibration curve reaches that response. '
        'The supplied prediction time and fitted in vitro curve are not used to calculate the mapped time.',
        't_vivo,pred = inverse(f_vivo)(y_vitro,pred)',
        'Horizontal parameter-uncertainty intervals invert sampled in vivo curves at the fixed supplied response. No response perturbations or residual noise are added. '
        'A common numerical mapper starts in the calibration interval and extends its search along the monotonic continuation when needed. '
        'Extrapolation is flagged, not clipped to a calibration endpoint. Nonunique or numerically flat responses, '
        'nonmonotonic continuations, nonfinite model domains, and unresolved roots remain unavailable. '
        'The search uses up to 16 doubling extensions of the calibration span; failure to find a root is not proof that no root exists. '
        'Unscaled time is searched no earlier than zero (or the observed minimum if negative); log10 time has no zero-coordinate floor. '
        'Monotonicity/uniqueness are numerical checks on the examined branch, not global proofs for arbitrary functions.'),
    'value_scaling': (
        'The calibration data supply an in vivo/in vitro response ratio at each supported prediction time. The interpolated ratio scales the supplied prediction response.',
        'y_vivo,pred = y_vitro,pred × response_ratio(t_vitro,pred)',
        'This deterministic mapping has no fitted-parameter covariance or formal propagated prediction interval. '
        'Inputs outside the calibration lookup range are unavailable; the resulting scaled in vivo response is not itself restricted to the observed in vivo response range.'),
    'time_scaling': (
        'The calibration data supply an in vivo/in vitro time ratio at each supported prediction response. The interpolated ratio scales the supplied prediction time.',
        't_vivo,pred = t_vitro,pred × time_ratio(y_vitro,pred)',
        'This deterministic mapping has no fitted-parameter covariance or formal propagated prediction interval. '
        'Inputs outside the calibration lookup range are unavailable; the resulting scaled in vivo time is not itself restricted to the observed in vivo time range.'),
}


def _fmt(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (str, bool)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return f'{number:.6g}' if math.isfinite(number) else 'N/A'


def _vector(values):
    try:
        return np.asarray([] if values is None else values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        # An uploaded prediction file can be readable but contain invalid cells.
        # Preserve those original cells in the appendix while summarizing only
        # numerical values in coverage calculations.
        result = []
        for value in np.asarray(values, dtype=object).reshape(-1):
            try:
                result.append(float(value))
            except (TypeError, ValueError, OverflowError):
                result.append(np.nan)
        return np.asarray(result, dtype=float)


def _at(value, index):
    if isinstance(value, (tuple, list)):
        return value[index] if index < len(value) else None
    return value if index == 0 else None


def _finite_pairs(time, response):
    x, y = _vector(time), _vector(response)
    n = min(len(x), len(y))
    finite = np.isfinite(x[:n]) & np.isfinite(y[:n])
    return x[:n][finite], y[:n][finite]


def _range(values):
    values = _vector(values)
    values = values[np.isfinite(values)]
    return f'{_fmt(np.min(values))} to {_fmt(np.max(values))}' if len(values) else 'N/A'


def _table(title, headers, rows, description=None):
    return {'type': 'table', 'title': title,
            'description': description or title + '. All rows are included.',
            'header_rows': 1, 'rows': [[str(x) for x in headers]] + [[_fmt(x) for x in row] for row in rows]}


def _paragraph(text, **metadata):
    return dict(type='paragraph', text=str(text), **metadata)


def _heading(text, level=1):
    return {'type': 'heading', 'text': text, 'level': level}


def _image(source_id, title, description):
    return {'type': 'image', 'source_id': source_id, 'title': title, 'description': description}


class _ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag in ('p', 'div', 'li', 'br'):
            self.fragments.append('\n')
        elif tag == 'span':
            self.fragments.append(' ')

    def handle_endtag(self, tag):
        if tag in ('p', 'div', 'li'):
            self.fragments.append('\n')

    def handle_data(self, data):
        self.fragments.append(data)


def _html_paragraphs(value):
    parser = _ReadableHTML()
    parser.feed(str(value or ''))
    return [' '.join(unescape(line).split()) for line in ''.join(parser.fragments).splitlines() if line.strip()]


def _preprocessing_label(category, options):
    return '; '.join(PREPROCESSING_DESCRIPTIONS.get(category, {}).get(key, {}).get('display_name', key)
                     for key in (options or [])) or 'None'


def _source_rows(label, source):
    if not source:
        return [[label, 'Not supplied']]
    filename = source.get('filename') or 'Not provided'
    worksheet = source.get('worksheet')
    rows = [[label + ' file', filename],
            [label + ' worksheet', worksheet if worksheet is not None else ('Not applicable (CSV)' if str(filename).lower().endswith('.csv') else 'Not provided')]]
    roles = (('In vitro time', 'In vitro response', 'In vivo time', 'In vivo response')
             if label == 'Calibration' else ('In vitro prediction time', 'In vitro prediction response'))
    columns = source.get('columns') or []
    rows.extend([label + f' column {index + 1}', str(column) + ' — ' + (roles[index] if index < len(roles) else 'Not used')]
                for index, column in enumerate(columns))
    if not columns:
        rows.append([label + ' columns', 'Not recorded'])
    if source.get('sha256'):
        rows.append([label + ' file SHA256', source['sha256']])
    return rows


def _observation_rows(raw):
    x = np.asarray([] if raw.get('time') is None else raw['time'], dtype=object).reshape(-1)
    y = np.asarray([] if raw.get('response') is None else raw['response'], dtype=object).reshape(-1)
    return [[i + 1, x[i] if i < len(x) else None, y[i] if i < len(y) else None]
            for i in range(max(len(x), len(y)))]


def _stats_rows(stats, selected_metrics, finite_scores=None, fit_cv_ratios=None, cv_comparison=None):
    if stats is None:
        return []
    stats = stats.iloc[0] if hasattr(stats, 'iloc') else stats
    rows = []
    for key in selected_metrics:
        full = stats.get(key)
        cv = 'Not applicable' if key in _CV_EXCLUDED else stats.get('CV ' + key)
        count = 'Not applicable' if key in _CV_EXCLUDED else (finite_scores or {}).get(key, 'Not recorded')
        ratio = 'Not meaningful' if key in _CV_EXCLUDED else (fit_cv_ratios or {}).get(key)
        comparison = 'Not meaningful' if key in _CV_EXCLUDED else (cv_comparison or {}).get(key, 'Not recorded')
        rows.append([metrics.get(key, {}).get('display_name', key), full, cv, count, ratio, comparison])
    return rows


def _model_label(key, registry):
    approach, _, model = key.partition(':')
    return approach, registry.get(model, {}).get('display_name', model) if model else 'Direct mapping'


def _parameter_table(result, index, metadata, label):
    params = _at(result.get('params'), index) or {}
    uncertain = _at(result.get('uparams'), index) or {}
    multiplier = metadata.get('ci_multiplier_95')
    if multiplier is None:
        # Backwards-compatible read of older completed results; no sampling.
        from scipy.stats import t
        df = metadata.get('residual_df', 0)
        multiplier = float(t.ppf(.975, df)) if df and df > 0 else 1.96
    rows = []
    for name, estimate in params.items():
        uncertainty = uncertain.get(name)
        sd = getattr(uncertainty, 's', np.nan)
        margin = float(multiplier) * float(sd)
        if not (np.isfinite(float(estimate)) and np.isfinite(margin) and margin >= 0):
            margin, lower, upper = np.nan, np.nan, np.nan
        else:
            lower, upper = float(estimate) - margin, float(estimate) + margin
        rows.append([name, estimate, margin, lower, upper])
    return _table(label + ' parameter estimates', ['Parameter', 'Estimate', '95% CI half-width', 'Lower 95%', 'Upper 95%'], rows)


def _fit_metadata(result, index):
    metadata = dict(_at(result.get('fit_metadata'), index) or {})
    x = _at(result.get('residual_x'), index)
    params = _at(result.get('params'), index) or {}
    metadata.setdefault('n_observations', len(_vector(x)))
    metadata.setdefault('n_parameters', len(params))
    metadata.setdefault('residual_df', metadata['n_observations'] - metadata['n_parameters'])
    return metadata


def _prediction_records(context, option):
    approach, model, method = option['approach_id'], option.get('model_name'), option.get('method_id')
    predictions = context.get('prediction_results') or {}
    if predictions.get('error'):
        return [], str(predictions['error'])
    if approach == 'approach3':
        key = 'method1_value_scaling' if method == 'value_scaling' else 'method2_time_scaling'
        selected = predictions.get(approach, {}).get(key, {})
    else:
        selected = predictions.get(approach, {}).get(model, {})
    if selected.get('error'):
        return [], str(selected['error'])
    records = selected.get('prediction_tables', {}).get(method)
    if records is not None:
        return records, None
    values = selected.get('predictions') or {}
    times, responses = _vector(values.get('in_vitro_time')), _vector(values.get('in_vitro_value'))
    output_key = 'predicted_in_vivo_value' if method in ('value_ratio', 'value_scaling') else 'predicted_in_vivo_time'
    output = _vector(values.get(output_key))
    records = []
    for i in range(max(len(times), len(responses), len(output))):
        point = output[i] if i < len(output) else np.nan
        records.append({'in_vitro_time': times[i] if i < len(times) else np.nan,
                        'in_vitro_value': responses[i] if i < len(responses) else np.nan,
                        output_key: point, 'lower_95': np.nan, 'upper_95': np.nan,
                        'status': 'Predicted' if np.isfinite(point) else 'Unavailable',
                        'interval_status': 'Not applicable' if approach == 'approach3' else 'Not recorded'})
    if not records:
        reason_key = {'tau_ratio': 'plot_tau_skip_reason', 'observed_response_inverse': 'plot_inverse_response_skip_reason'}.get(method)
        return [], selected.get(reason_key, 'Prediction output was not available for this selection.')
    return records, None


def _diagnostic_sections(result, context, option, index, label):
    sections = []
    diagnostics = _at(result.get('parameter_diagnostics_html'), index)
    sections.extend(_paragraph(text) for text in _html_paragraphs(diagnostics))
    config, approach = context.get('analysis_config') or {}, option['approach_id']
    suffix = '' if approach == 'approach1' else ('-in-vitro' if index == 0 else '-in-vivo')
    dom_id = option.get('model_dom_id') or option['model_key'].replace(':', '-')
    choices = (
        ('include_cv_fit_variability_plot', 'cv_fit_variability_plot', 'CV fit variability', 'plot-cv-fit-', 'The full-data fit and mean plus/minus one SD across successful CV-training fits. This describes sensitivity to omissions, not a confidence or prediction interval.'),
        ('include_raw_residual_plots', 'residual_plot', 'Residuals', 'plot-residual-', 'Observed minus fitted values on the analyzed response scale, shown against the fitting x-variable.'),
        ('include_residual_qq_plots', 'residual_qq_plot', 'Normal Q-Q plot', 'plot-residual-qq-', 'Ordered standardized residuals compared with a normal reference; intended for qualitative review.'),
    )
    statuses = []
    for flag, key, title, prefix, description in choices:
        if not config.get(flag, False):
            statuses.append([title, 'Not requested'])
            continue
        plot = _at(result.get(key), index) or {}
        if plot.get('available') and plot.get('plot'):
            statuses.append([title, 'Included'])
            sections.append(_image(prefix + dom_id + suffix, label + ' ' + title, description))
        else:
            message = plot.get('message') or 'Requested but unavailable'
            statuses.append([title, message])
    sections.insert(0, _table(label + ' diagnostic availability', ['Diagnostic', 'Status'], statuses))
    return sections


def _candidate_sections(context, option):
    registry, selected = context.get('models_registry') or {}, context.get('selected_models') or []
    results = context.get('results') or {}
    sections = [_heading('Candidate model context', 2)]
    outcomes = []
    for key in selected:
        approach, model = _model_label(key, registry)
        result = results.get(key)
        status = ('Not completed' if result is None else str(result['error']) if result.get('error') else
                  'Mapping completed' if approach == 'approach3' else 'Fit completed')
        outcomes.append([approach, model, status, 'Reported selection' if key == option['model_key'] else ''])
    sections.append(_table('Candidates evaluated in this run', ['Approach', 'Model', 'Outcome', 'Selection'], outcomes))
    panels = (context.get('evidence_tables') or {}).get(option['approach_id'], {})
    # Prefer joint evidence for Approach 2; never compare unlike approaches.
    panel = panels.get('combined' if option['approach_id'] == 'approach2' else 'fit') or {}
    criteria = panel.get('criteria') or []
    evidence = panel.get('rows') or []
    if evidence and criteria:
        rows = []
        for row in evidence:
            for criterion in criteria:
                rows.append([row.get('model'), criterion.upper(), row.get(criterion), row.get('delta_' + criterion),
                             row.get(criterion + '_weight'), row.get(criterion + '_evidence_ratio')])
        sections.append(_table('Relative evidence within the selected approach', ['Model', 'Criterion', 'Value', 'Delta', 'Weight', 'Evidence ratio'], rows))
        joint = 'This table combines the in vitro and in vivo evidence. ' if option['approach_id'] == 'approach2' else ''
        sections.append(_paragraph('These information criteria compare successfully fitted candidates for the same approach and data preparation. '
                                   + joint + 'Lower criterion values indicate stronger relative support within this candidate set; they are not quality grades.'))
    elif option['approach_id'] != 'approach3':
        # Selected full-fit metrics provide context when no IC comparison exists.
        rows = []
        for key in selected:
            approach, label = _model_label(key, registry)
            result = results.get(key) or {}
            if approach != option['approach_id'] or result.get('error') or 'stats' not in result:
                continue
            for index, dataset in enumerate(('In vitro', 'In vivo') if approach == 'approach2' else ('Correlation',)):
                stats = _at(result['stats'], index)
                if stats is None:
                    continue
                stats = stats.iloc[0] if hasattr(stats, 'iloc') else stats
                for metric in context.get('selected_metrics') or []:
                    rows.append([label, dataset, metrics.get(metric, {}).get('display_name', metric), stats.get(metric)])
        sections.append(_table('Candidate full-fit metrics within the selected approach', ['Model', 'Dataset', 'Metric', 'Full-fit value'], rows))
        sections.append(_paragraph('Relative information-criterion evidence was not available for this selection. Full-fit metrics are shown without an automatic model recommendation.'))
    return sections


def build_report_sections(context, option):
    """Build all report content from a completed analysis and selected option."""
    results, config = context.get('results') or {}, context.get('analysis_config') or {}
    result = results.get(option['model_key']) or {}
    if result.get('error'):
        raise ValueError('A report cannot describe a failed fit as a completed model.')
    approach, method = option['approach_id'], option.get('method_id')
    sections = [_heading('Analysis summary')]
    for key, label in REPORT_DETAIL_FIELDS:
        sections.append(_paragraph(label + ': Not provided', detail_key=key, detail_label=label))
    sections.append(_paragraph('Reported selection: ' + option.get('label', option['model_key']) + '. This identifies the exported configuration; it is not an app recommendation.'))
    info = context.get('app_info') or {}
    sections.append(_table('Analysis identification', ['Item', 'Value'], [
        ['Analysis completed', context.get('analysis_timestamp', 'Not recorded')],
        ['App version', info.get('version', 'Not recorded')],
        ['Source revision', info.get('revision', 'Not recorded')],
    ] + _source_rows('Calibration', context.get('fitting_source')) + _source_rows('Prediction', context.get('prediction_source'))))

    sections.append(_heading('Data preparation'))
    raw = context.get('raw_fitting_data') or {}
    summary = []
    for key, label in (('in_vitro', 'In vitro calibration'), ('in_vivo', 'In vivo calibration')):
        series = raw.get(key) or {}
        x, y = _finite_pairs(series.get('time'), series.get('response'))
        source_rows = max(len(_vector(series.get('time'))), len(_vector(series.get('response'))))
        summary.append([label, source_rows, len(x), len(np.unique(x)), _range(x), _range(y)])
    pred_raw = context.get('raw_prediction_data')
    if pred_raw is not None:
        x, y = _finite_pairs(pred_raw.get('time'), pred_raw.get('response'))
        source_rows = max(len(_vector(pred_raw.get('time'))), len(_vector(pred_raw.get('response'))))
        summary.append(['Prediction input', source_rows, len(x), len(np.unique(x)), _range(x), _range(y)])
    sections.append(_table('Source data coverage', ['Dataset', 'Source rows', 'Finite time-response pairs', 'Distinct times', 'Time range', 'Response range'], summary))
    sections.append(_paragraph('Counts describe supplied numerical observations, not biological replicate counts. Source rows include missing cells or padding for unequal profile lengths; those rows are not counted as finite time-response pairs. The source-data appendix preserves them. The in vitro and in vivo series need not have matching times or equal lengths.'))
    analyzed = context.get('processed_data')
    if analyzed is not None and len(analyzed) >= 4:
        coverage = []
        for index, label in ((0, 'In vitro calibration'), (2, 'In vivo calibration')):
            x, y = _finite_pairs(analyzed[index], analyzed[index + 1])
            coverage.append([label, len(x), _range(x), _range(y)])
        processed_prediction = context.get('processed_prediction_data')
        if processed_prediction is not None and len(processed_prediction) >= 2:
            x, y = _finite_pairs(processed_prediction[0], processed_prediction[1])
            coverage.append(['Prediction input', len(x), _range(x), _range(y)])
        sections.append(_table('Analyzed data coverage after preprocessing', ['Dataset', 'Finite time-response pairs', 'Analyzed time range', 'Analyzed response range'], coverage))
    preprocessing = context.get('preprocessing_metadata') or {}
    requested_scalings = preprocessing.get('requested_scalings', context.get('selected_scalings') or [])
    prep_rows = [
        ['Normalization', _preprocessing_label('normalization', context.get('selected_normalizations'))],
        ['Scaling requested', _preprocessing_label('scaling', requested_scalings)],
        ['Scaling applied', _preprocessing_label('scaling', context.get('selected_scalings'))],
        ['Interpolation selected', _preprocessing_label('interpolation', context.get('selected_interpolation'))],
    ]
    sections.append(_table('Applied preprocessing', ['Setting', 'Value'], prep_rows))
    for warning in preprocessing.get('warnings') or []:
        sections.append(_paragraph('Preprocessing warning: ' + str(warning.get('message', warning))))
    prediction_preprocessing = context.get('prediction_preprocessing_metadata') or {}
    references = [*(preprocessing.get('normalization_records') or []), *(prediction_preprocessing.get('normalization_records') or [])]
    if references:
        rows = [[r.get('dataset'), r.get('operation'), r.get('reference'), r.get('minimum'), r.get('maximum'),
                 'Applied' if r.get('applied') else 'Not applied', r.get('note', '')] for r in references]
        sections.append(_table('Normalization reference values in application order', ['Dataset', 'Operation', 'Initial reference', 'Minimum', 'Maximum', 'Status', 'Note'], rows))
    sections.append(_paragraph('Calculations use the preprocessed time and response scales. Normalization, when selected, precedes log10 scaling. Predictions are reported on those analyzed scales; normalization and logarithmic transformations are not automatically reversed.'))
    if approach == 'approach2':
        sections.append(_paragraph('The in vitro and in vivo models use preprocessed observed points independently. Interpolation does not add fitting observations for this approach.'))
    else:
        sections.append(_paragraph('Alignment uses the selected preprocessing to derive matching time or response positions. Interpolated values are deterministic estimates from the observations, not additional independent experiments.'))
        interpolation_sources = ([('plot-processing-interpolation-value', 'Response-aligned calibration data')] if approach == 'approach1' else
                                 [('plot-processing-interpolation-time', 'Time-aligned calibration data')] if method == 'value_scaling' else
                                 [('plot-processing-interpolation-value', 'Response-aligned calibration data')] if method == 'time_scaling' else
                                 [('plot-processing-interpolation-time', 'Time-aligned calibration data'), ('plot-processing-interpolation-value', 'Response-aligned calibration data')])
        for source_id, title in interpolation_sources:
            sections.append(_image(source_id, title, 'Observed and aligned calibration values on the analyzed scales.'))
    sections.insert(sections.index(_heading('Data preparation')) + 1, _image('plot-processing-raw-data', 'Source calibration data', 'The supplied in vitro and in vivo time courses before preprocessing.'))

    sections.append(_heading('Methods'))
    if approach == 'approach3':
        sections.append(_paragraph('Direct interpolation-based mapping was used. No parametric model, fitted-parameter confidence interval, cross-validation, or grid-search initialization applies.'))
    else:
        registry = context.get('models_registry') or {}
        model = registry.get(option.get('model_name'), {})
        latex = result.get('function') or model.get('latex_equation') or ''
        sections.append({'type': 'equation', 'latex': latex, 'text': latex})
        if model.get('description'):
            sections.append(_paragraph(model['description']))
        sections.append(_paragraph('Equation variables: x is analyzed in vitro time and y is the corresponding analyzed in vivo time.' if approach == 'approach1' else
                                   'Equation variables: x is analyzed time and y is analyzed response. The same model family is fitted separately to the in vitro and in vivo calibration series.'))
        if (model.get('fit_kwargs') or {}).get('bounds') is not None:
            bounds = model['fit_kwargs']['bounds']
            names = list((_at(result.get('params'), 0) or {}).keys())
            lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (len(names),))
            upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (len(names),))
            sections.append(_table('Parameter fitting constraints', ['Parameter', 'Lower bound', 'Upper bound'],
                                   [[name, lo if np.isfinite(lo) else 'Unbounded', hi if np.isfinite(hi) else 'Unbounded']
                                    for name, lo, hi in zip(names, lower, upper)]))
        scheme = config.get('cv_scheme', 'shuffle_split')
        settings = [['Cross-validation scheme', CV_SCHEME_LABELS.get(scheme, scheme)]]
        if scheme == 'shuffle_split':
            settings.extend([['CV splits requested', config.get('cv_n_splits')], ['CV test fraction', config.get('cv_test_size')], ['CV random seed', config.get('cv_random_state')]])
        elif scheme == 'leave_contiguous_block_out':
            settings.append(['Contiguous block size requested (observations)', config.get('cv_n_splits')])
        settings.extend([['Initialization random starts', config.get('grid_search_num_points')],
                         ['Initialization lower starting-value bound', config.get('grid_search_param_min')],
                         ['Initialization upper starting-value bound', config.get('grid_search_param_max')],
                         ['Initialization random seed', config.get('grid_search_random_state')]])
        sections.append(_table('Settings used', ['Setting', 'Value'], settings))
        sections.append(_paragraph('Nonlinear least squares fits were initialized by searching candidate starting values, including model-specific starts when available. The starting-value search bounds are not parameter constraints for the final fit.'))
        if scheme == 'none':
            sections.append(_paragraph('Cross-validation was disabled. No held-out prediction assessment was performed in this run.'))
        else:
            scope = ('the derived paired-time observations' if approach == 'approach1' else 'the preprocessed observed rows for each calibration series')
            text = 'Cross-validation refits the parametric model using subsets of ' + scope + '. Preprocessing and alignment are established before the splits, rather than recomputed within each training fold.'
            if scheme in ('leave_one_timepoint_out', 'leave_one_out'):
                text += ' In this version, the option labelled leave-one-timepoint-out omits one row at a time; replicates sharing a time are not held out together.'
            elif scheme == 'leave_contiguous_block_out':
                text += ' Blocks contain adjacent observations in fitting-x order, not necessarily all observations sharing the same time.'
            sections.append(_paragraph(text))

    if method in _METHODS:
        explanation, formula, uncertainty = _METHODS[method]
        sections.extend([_heading(option.get('method_display') or 'Selected prediction method', 2), _paragraph(explanation), _paragraph(formula)])

    sections.extend(_candidate_sections(context, option))
    sections.append(_heading('Fit and diagnostic results' if approach != 'approach3' else 'Mapping results'))
    fit_caption = ('Response-aligned analyzed in vitro and in vivo calibration times and their fitted relationship; shading shows the available approximate 95% parameter-uncertainty band.' if approach == 'approach1' else
                   'Independent in vitro and in vivo fits to preprocessed observed time-response points; shading shows available approximate 95% parameter-uncertainty bands.' if approach == 'approach2' else
                   'Deterministic ratios derived from the calibration observations and selected alignment; no statistical uncertainty band is calculated.')
    for source in option.get('fit_plot_sources') or []:
        sections.append(_image(source['source_id'], source['title'], fit_caption))
    for warning in (result.get('plot') or {}).get('uncertainty_warnings', []):
        sections.append(_paragraph('Fitting-plot warning: ' + str(warning.get('message', '')) + (' ' + str(warning['detail']) if warning.get('detail') else '')))
    if approach != 'approach3':
        sections.append(_paragraph('Parameter-table intervals remain approximate local 95% confidence intervals from fitted covariance. Fitting bands and prediction error bars are approximate pointwise 95% parameter-uncertainty intervals: 2.5th and 97.5th percentiles of covariance-aware parameter draws, conditional on the model, preprocessing, and fixed prediction inputs. Separate fits are sampled independently. No additional residual/observation noise, model-selection uncertainty, or model inadequacy is included. These are not prediction intervals for future observations or a simultaneous envelope. Calibration variability still informs fitted covariance.'))
        sections.append(_paragraph('Fitting and parametric prediction intervals use 1,000 parameter draws and uncertainty seed 12345, independently of the CV/initialization seeds. Central values use fitted parameters, not the sampled mean. Any unusable draw at a point makes its interval unavailable; unrelated requested times do not change an available pointwise interval. Export reuses the already calculated bounds without resampling or re-centering.'))
        for index, label in enumerate(('In vitro', 'In vivo') if approach == 'approach2' else ('Time correlation',)):
            metadata = _fit_metadata(result, index)
            sections.append(_heading(label, 2))
            sections.append(_table(label + ' fitting data and degrees of freedom', ['Quantity', 'Value'], [[key.replace('_', ' '), metadata[key]] for key in ('n_observations', 'n_parameters', 'residual_df')]))
            if metadata['residual_df'] <= 0:
                sections.append(_paragraph('Residual degrees of freedom are not positive. Where the app still reports a local interval, it uses its normal-multiplier fallback; that interval warrants particular caution.'))
            sections.append(_parameter_table(result, index, metadata, label))
            cv = _at(result.get('cv_diagnostics'), index) or {}
            if cv:
                cv_rows = [['Splits attempted', cv.get('total_splits')], ['Successful fits and finite test predictions', cv.get('successful_fits')], ['Failed fits or test predictions', cv.get('fitting_failure_count')]]
                for key, title in (('min_training_observations', 'Smallest training subset'), ('max_training_observations', 'Largest training subset'),
                                   ('min_test_observations', 'Smallest held-out subset'), ('max_test_observations', 'Largest held-out subset')):
                    if cv.get(key) is not None:
                        cv_rows.append([title, cv[key]])
                sections.append(_table(label + ' cross-validation outcomes', ['Outcome', 'Count'], cv_rows))
            stats_rows = _stats_rows(_at(result.get('stats'), index), context.get('selected_metrics') or [], cv.get('finite_scores'),
                                     _at(result.get('fit_cv_ratios'), index), _at(result.get('cv_comparison'), index))
            sections.append(_table(label + ' performance metrics', ['Metric', 'Full-data fit', 'Mean finite CV score', 'Finite CV splits', 'Final/CV ratio', 'CV comparison'], stats_rows))
            sections.extend(_diagnostic_sections(result, context, option, index, label))
        sections.append(_paragraph('CV means summarize finite split scores only; counts reveal partial failures or undefined metrics. Final/CV ratios and comparison labels reproduce the app\'s diagnostic summaries, not model-acceptability decisions. Adjusted R², information criteria, and the app\'s normalized-error metrics are not given CV comparisons. Metrics describe the analyzed scale.'))
        if method == 'tau_ratio':
            rows = []
            for index, label in enumerate(('In vitro', 'In vivo')):
                value = _at(result.get('tau'), index)
                metadata = _fit_metadata(result, index)
                diagnostics = _at(result.get('tau_diagnostics'), index) or {}
                estimate = value.n if tau_estimate_is_available(value) else np.nan
                multiplier = metadata.get('ci_multiplier_95', 1.96)
                half_width = (value.s * multiplier if tau_uncertainty_is_available(value)
                              and np.isfinite(multiplier) and multiplier > 0 else np.nan)
                status = ('Available' if np.isfinite(estimate) and np.isfinite(half_width)
                          else diagnostics.get('reason') or diagnostics.get('uncertainty_reason')
                          or ('Uncertainty unavailable' if np.isfinite(estimate) else 'Time constant unavailable'))
                rows.append([label, estimate, half_width, status])
            sections.append(_table('Time constants used for time scaling', ['Calibration series', 'Tau', '95% CI half-width', 'Status'], rows))
            sections.append(_paragraph('Central tau is the integral of the fitted curve from zero to infinity divided by its fitted initial response, evaluated at the fitted parameters. Integration warnings or invalid results make tau unavailable without discarding the fitted curve. The app estimates tau standard uncertainty using 50 parameter draws (seed 12345); if sampling fails or any sampled tau is invalid, uncertainty is unavailable rather than calculated after dropping those draws. Plotted tau-ratio intervals separately propagate 1,000 pairs of fitted parameter draws using percentile bounds, rather than multiplying the table SDs by a t multiplier. A valid central time scaling can still be shown without error bars. Tau may depend strongly on the extrapolated tail; these numerical checks do not prove convergence.'))
    else:
        sections.append(_paragraph('The plotted ratios and mappings are determined by the supplied data and interpolation choices. Formal propagated uncertainty is not available.'))

    if context.get('prediction_source') or context.get('raw_prediction_data') is not None:
        sections.append(_heading('Prediction results'))
        if not method:
            prediction_results = context.get('prediction_results') or {}
            error = prediction_results.get('error') or prediction_results.get(approach, {}).get(option.get('model_name'), {}).get('error')
            sections.append(_paragraph('A prediction dataset was supplied, but prediction processing failed: ' + str(error) if error else
                                       'A prediction dataset was supplied, but no prediction method is active for this report.'))
        else:
            if method in _METHODS:
                sections.append(_paragraph(_METHODS[method][2]))
            records, error = _prediction_records(context, option)
            if error:
                sections.append(_paragraph('Prediction status: ' + error))
            if records:
                output_key = 'predicted_in_vivo_value' if method in ('value_ratio', 'value_scaling') else 'predicted_in_vivo_time'
                rows = []
                for index, row in enumerate(records):
                    rows.append([index + 1, row.get('in_vitro_time', row.get('input_time')), row.get('in_vitro_value', row.get('input_response')),
                                 row.get(output_key), row.get('lower_95'), row.get('upper_95'), row.get('status', ''), row.get('interval_status', '')])
                sections.append(_table('Predictions and displayed uncertainty bounds', ['Row', 'Input time', 'Input response', 'Predicted in vivo response' if output_key.endswith('value') else 'Predicted in vivo time', 'Lower 95%', 'Upper 95%', 'Mapping status', 'Interval status'], rows))
                sections.append(_paragraph('Inputs and predictions in this table are preprocessed quantities. N/A denotes an unavailable value or interval, not a zero. Bounds match the displayed intervals; a numerical point can be available when its interval is unavailable.'))
                domain_rows = [
                    [i + 1, row.get('extrapolation_status'), row.get('extrapolation_reason'),
                     row.get('interval_extrapolation_status')]
                    for i, row in enumerate(records) if 'extrapolation_status' in row
                ]
                if domain_rows:
                    sections.append(_table('Calibration-domain cautions',
                        ['Row', 'Point domain status', 'Reason', 'Interval domain status'], domain_rows))
                    ranges = next((row['calibration_ranges'] for row in records if row.get('calibration_ranges')), '')
                    if ranges:
                        sections.append(_paragraph('Calibration ranges on analyzed scales: ' + ranges))
                    if any(row.get('extrapolated') or row.get('interval_extrapolated') for row in records):
                        sections.append(_paragraph('Extrapolation caution: one or more calculated points or interval bounds are outside relevant calibration ranges. Narrow parameter-uncertainty intervals do not establish reliability outside calibration.'))
                    sections.append(_paragraph('Range checks use the quantities used by the selected method, not unused prediction inputs. Within-range status does not establish scientific transferability or independent validation.'))
                if method == 'observed_response_inverse':
                    inverse_rows = [[i + 1, row.get('inverse_extrapolated_samples'),
                                     row.get('inverse_failed_draw_reasons', '')]
                                    for i, row in enumerate(records) if 'inverse_failed_draw_reasons' in row]
                    if inverse_rows:
                        sections.append(_table('Inverse search diagnostics',
                            ['Row', 'Draws with times outside calibration', 'Failed draws by reason'], inverse_rows))
                    first = records[0]
                    if 'inverse_max_expansions' in first:
                        minimum = first.get('inverse_minimum_time')
                        sections.append(_paragraph('Inverse search settings (analyzed time scale): minimum time = '
                            + ('no finite lower bound' if minimum is None else _fmt(minimum))
                            + '; maximum doubling extensions = ' + _fmt(first['inverse_max_expansions'])
                            + '; extrapolation enabled = ' + str(first.get('inverse_allow_extrapolation', True)) + '.'))
                if method == 'value_ratio':
                    extra = [[i + 1, row.get('interval_valid_samples'), row.get('interval_total_samples'),
                              row.get('interval_requested_samples'), row.get('denominator_lower_95'),
                              row.get('denominator_upper_95'), row.get('interval_random_seed')]
                             for i, row in enumerate(records) if 'interval_requested_samples' in row]
                    if extra:
                        sections.append(_table('Response-ratio uncertainty diagnostics',
                            ['Row', 'Usable draws', 'Generated draws', 'Requested draws',
                             'Denominator lower 95%', 'Denominator upper 95%', 'Random seed'], extra))
                        sections.append(_paragraph('Usable draws are counted separately at each prediction time. No residual noise is added. Intervals that exclude the point estimate are drawn as separate vertical segments at the calculated bounds, not re-centered.'))
                if method in ('direct_time_correlation', 'tau_ratio', 'observed_response_inverse'):
                    draw_rows = [[i + 1, row.get('interval_valid_samples'), row.get('interval_total_samples'),
                                  row.get('interval_requested_samples'), row.get('interval_random_seed')]
                                 for i, row in enumerate(records) if 'interval_requested_samples' in row]
                    if draw_rows:
                        sections.append(_table('Parameter-uncertainty diagnostics',
                            ['Row', 'Usable draws', 'Generated draws', 'Requested draws', 'Random seed'], draw_rows))
                if method == 'observed_response_inverse':
                    extra = [[i + 1, row.get('calculated_lower_95'), row.get('calculated_upper_95'), row.get('interval_valid_samples'), row.get('interval_total_samples', row.get('total_samples'))]
                             for i, row in enumerate(records) if row.get('calculated_lower_95') is not None or row.get('calculated_upper_95') is not None]
                    if extra:
                        sections.append(_table('Calculated inverse interval diagnostics', ['Row', 'Calculated lower 95%', 'Calculated upper 95%', 'Valid Monte Carlo draws', 'Requested draws'], extra))
                        sections.append(_paragraph('Finite calculated bounds are retained even when they do not contain the point estimate; such intervals are drawn as separate horizontal segments. Unavailable intervals and failed-draw counts are reported explicitly.'))
            selected_predictions = (context.get('prediction_results') or {}).get(approach, {}).get(option.get('model_name'), {})
            if method == 'direct_time_correlation' and selected_predictions.get('plot_note'):
                sections.append(_paragraph(selected_predictions['plot_note']))
            if method == 'value_ratio' and selected_predictions.get('plot_value_ratio_note'):
                sections.append(_paragraph(selected_predictions['plot_value_ratio_note']))
            if method == 'tau_ratio' and selected_predictions.get('plot_tau_note'):
                sections.append(_paragraph(selected_predictions['plot_tau_note']))
            if method == 'observed_response_inverse' and selected_predictions.get('plot_inverse_response_note'):
                sections.append(_paragraph(selected_predictions['plot_inverse_response_note']))
            for source in option.get('prediction_plot_sources') or []:
                caption = ('Predicted in vivo responses at the supplied analyzed times; vertical intervals show available approximate 95% parameter uncertainty conditional on fixed prediction inputs, without additional observation noise.' if method == 'value_ratio' else
                           'Predicted in vivo times from the supplied analyzed in vitro responses; horizontal intervals show available approximate 95% parameter uncertainty conditional on the fixed response.' if method == 'observed_response_inverse' else
                           'Predicted in vivo times from the supplied analyzed in vitro times; horizontal intervals show available approximate 95% parameter uncertainty conditional on fixed inputs.' if method in ('direct_time_correlation', 'tau_ratio') else
                           'Predictions from the selected deterministic mapping on the analyzed scales; no formal uncertainty bars are available.')
                sections.append(_image(source['source_id'], source['title'], caption))
    else:
        sections.append(_paragraph('No prediction dataset was supplied. This report describes calibration fitting or mapping only.'))
    assessment = ('The direct mapping uses the supplied calibration data.' if approach == 'approach3' else
                  'The fitting and internal cross-validation results assess the supplied calibration data.' if config.get('cv_scheme', 'shuffle_split') != 'none' else
                  'The fitting results assess the supplied calibration data.')
    sections.append(_paragraph(assessment + ' An in vitro prediction dataset alone does not provide independent validation against additional paired in vitro and in vivo measurements.'))

    sections.append(_heading('Numerical data appendix'))
    for key, label in (('in_vitro', 'In vitro calibration source observations'), ('in_vivo', 'In vivo calibration source observations')):
        sections.append(_table(label, ['Source row', 'Time', 'Response'], _observation_rows(raw.get(key) or {})))
    if pred_raw is not None:
        sections.append(_table('Prediction source observations', ['Source row', 'Time', 'Response'], _observation_rows(pred_raw)))
    if approach != 'approach3':
        for index, label in enumerate(('In vitro', 'In vivo') if approach == 'approach2' else ('Derived paired-time',)):
            x, y = _vector(_at(result.get('residual_x'), index)), _vector(_at(result.get('residual_y'), index))
            headers = ['Fitting row', 'Analyzed in vitro time', 'Analyzed in vivo time'] if approach == 'approach1' else ['Fitting row', 'Analyzed time', 'Analyzed response']
            sections.append(_table(label + ' data actually fitted', headers, [[i + 1, x[i], y[i]] for i in range(min(len(x), len(y)))]))
    else:
        data = context.get('processed_data')
        if data is not None and len(data) >= 10:
            _, _, _, _, values, times, vitro_times, vivo_times, vitro_values, vivo_values = data
            if method in (None, 'value_scaling'):
                rows = [[i + 1, x, y1, y2] for i, (x, y1, y2) in enumerate(zip(times, vitro_values, vivo_values))]
                sections.append(_table('Response-mapping lookup data', ['Row', 'Analyzed time', 'In vitro response', 'In vivo response'], rows))
            if method in (None, 'time_scaling'):
                rows = [[i + 1, x, y1, y2] for i, (x, y1, y2) in enumerate(zip(values, vitro_times, vivo_times))]
                sections.append(_table('Time-mapping lookup data', ['Row', 'Analyzed response', 'In vitro time', 'In vivo time'], rows))
    return sections
