"""Small reporting helpers for source precision and future observation error."""
from copy import deepcopy
from html import escape

import numpy as np
import pandas as pd


def future_variability_table(definition):
    """Record individual SD and mean SEM at candidate and fixed future times."""
    times = definition.constraints.all_available_times
    completed = definition.existing_observations
    if completed is not None:
        times = times[~np.any(np.isclose(times[:, None], completed.unique_times[None, :],
                                       rtol=0., atol=1e-12), axis=1)]
    expected = definition.reference.evaluate_central(times) if len(times) else np.array([])
    sd = definition.observation_model.standard_deviation(expected, times=times)
    count = int(definition.constraints.replicates_per_time)
    fixed = definition.constraints.fixed_times
    return pd.DataFrame({
        'time': times, 'individual_sd': sd, 'replicates_per_time': count,
        'future_mean_sem': sd / np.sqrt(count),
        'time_role': ['fixed_future' if np.any(np.isclose(t, fixed, rtol=0., atol=1e-12))
                      else 'candidate' for t in times],
    })


def uncertainty_metadata(definition):
    """Keep reference uncertainty, completed precision, and future SD distinct."""
    source = definition.reference.metadata
    completed = definition.existing_observations
    precision = None if completed is None else completed.observation_uncertainty
    return {
        'reference': {key: deepcopy(source.get(key)) for key in (
            'source_observation_uncertainty', 'source_covariance_scaling',
            'source_parameter_uncertainty_propagated', 'calibration_uncertainty',
            'prediction_input_uncertainty',
        )},
        'completed': None if completed is None else {
            'reported_rows': completed.n_observations,
            'time_points': completed.n_time_points,
            'response_kind': completed.response_kind,
            'observation_uncertainty': None if precision is None else precision.to_metadata(),
            'precision_source': ('observation_model_individual_sd' if precision is None else
                                 precision.source if precision.source == 'historical_nominal_sd_assumption'
                                 else 'supplied_' + precision.uncertainty_kind),
        },
        'future': definition.observation_model.to_metadata(),
        'planned_replicates_per_time': int(definition.constraints.replicates_per_time),
        'independent_measurements_within_and_across_times': True,
        'coordinates': 'analyzed_time_and_response',
        'future_mean_variance': 'individual_sd_squared / replicates_per_time',
        'future_measurement_variability_in_reference_envelope': False,
    }


def _precision_text(precision):
    values = np.asarray(precision['standard_uncertainty'], dtype=float)
    scale = (f'{values[0]:g}' if np.all(values == values[0]) else
             f'{values.min():g} to {values.max():g}')
    kind = 'SEM for reported means' if precision['response_kind'] == 'mean' else 'SD for individual values'
    counts = precision.get('replicate_counts')
    count_text = ''
    if counts is not None:
        counts = np.asarray(counts, dtype=int)
        count_range = (str(counts[0]) if np.all(counts == counts[0]) else
                       f'{counts.min()} to {counts.max()}')
        count_text = f' (reported n={count_range})'
    assumption = (' (nominal historical SD assumption)' if
                  precision.get('source') == 'historical_nominal_sd_assumption' else '')
    return f'{kind}: {scale}{count_text}{assumption}'


def uncertainty_summary_html(definition):
    """Describe the assumptions used for this run without requiring Plotly."""
    record = uncertainty_metadata(definition)
    reference = record['reference']
    calibration = reference['calibration_uncertainty'] or {}
    sources = [('Source profile', reference['source_observation_uncertainty'])]
    sources.extend((f'{name.replace("_", " ").capitalize()} calibration',
                    calibration.get(name, {}).get('observation_uncertainty'))
                   for name in ('in_vitro', 'in_vivo'))
    sources.append(('Target in vitro prediction', reference['prediction_input_uncertainty']))
    descriptions = [f'{escape(label)} — {escape(_precision_text(precision))}'
                    for label, precision in sources if precision is not None]
    source_text = ('; '.join(descriptions) + '.' if descriptions else
                   'No experimental SD/SEM was supplied for the reference; its existing uncertainty scope is retained.')
    if reference['source_parameter_uncertainty_propagated'] is False:
        source_text += (' The source curve is fixed: only the empirical time-scale distribution '
                        'contributes reference spread.')
    completed = record['completed']
    if completed is None:
        completed_text = 'No completed target-study observations.'
    else:
        precision = completed['observation_uncertainty']
        completed_text = (f'{completed["reported_rows"]} reported rows at '
                          f'{completed["time_points"]} times; ')
        completed_text += (_precision_text(precision) + '.' if precision is not None else
                           'individual SD is assumed from the observation model.')
    model = definition.observation_model
    profile_table = ''
    if model.kind == 'time_profile_sd':
        future_text = ('Individual SD uses the supplied time profile, choosing the nearest time '
                       '(earlier time for ties) and extending endpoint SDs beyond the profile range.')
        profile_table = pd.DataFrame({'Profile time': model.profile_times,
                                     'Supplied individual SD': model.profile_sd}).to_html(index=False, escape=True)
    elif model.kind == 'constant_cv':
        future_text = f'Individual SD uses constant CV {model.value:g} times the absolute expected response.'
    else:
        future_text = f'Constant individual SD: {model.value:g}.'
    if model.minimum_sd:
        future_text += f' Minimum individual SD: {model.minimum_sd:g}.'
    return (
        '<h2>Measurement uncertainty assumptions</h2>'
        f'<p><strong>Reference inputs:</strong> {source_text}</p>'
        f'<p><strong>Completed measurements:</strong> {escape(completed_text)}</p>'
        '<p>SEM is the standard uncertainty of a reported mean; its replicate count is not applied again. '
        'Completed observations remain fixed in simulated studies.</p>'
        f'<p><strong>Future measurements:</strong> {escape(future_text)} '
        f'{record["planned_replicates_per_time"]} independent replicates per new time; '
        'the future mean has SEM = individual SD / sqrt(replicate count).</p>'
        + profile_table +
        '<p>Times and errors use the analyzed time and response scales. Observation errors are assumed '
        'independent within and across times. The reference envelope describes plausible trajectories; '
        'future measurement error is added separately in Monte Carlo studies. The envelope is not an '
        'individual-measurement prediction band.</p>'
    )
