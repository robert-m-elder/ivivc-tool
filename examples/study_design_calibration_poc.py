#!/usr/bin/env python3
"""Demonstrate curve recovery plus IVIVC assessment, and calibration refinement.

This focused POC compares specified candidates using a normalized exponential
and Approach 2 value-ratio prediction. It complements the general study-design
POC; it does not change its scenarios or perform joint optimization.
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import sys
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivivc_core import ObservationUncertainty, TimeCourse, prepare_time_course  # noqa: E402
from study_design_core import (  # noqa: E402
    CalibrationDesign, IVIVCAssessmentRequest, ObservationModel,
    ReferenceTrajectoryEnsemble, compare_calibration_designs,
    create_simulation_plan, fit_reference_model_ensemble, simulate_design,
    summarize_calibration_simulation, summarize_simulation,
)


MODEL = 'exponential_normalized'
VITRO_RATE = .35
VIVO_RATES = np.array([.15, .16, .17])
REFERENCE_WEIGHTS = np.array([.25, .5, .25])
EVALUATION_TIMES = np.array([1., 2., 4., 6., 8.])


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}.')
    return int(value)


def _seed(seed, role):
    return np.random.SeedSequence([seed, role])


def _references():
    grid = np.linspace(0., 12., 241)
    sources = {}
    for name, rates, weights in (
        ('in_vitro', np.array([VITRO_RATE]), np.ones(1)),
        ('in_vivo', VIVO_RATES, REFERENCE_WEIGHTS),
    ):
        sources[name] = ReferenceTrajectoryEnsemble(
            grid, np.exp(-rates[:, None]*grid), np.ones(len(rates), dtype=bool),
            draw_weights=weights, interpolation='pchip', metadata={
                'source': 'specified_synthetic_generating_curves',
                'generating_model': MODEL, 'parameter_names': ['b'],
                'parameter_draws': (-rates[:, None]).tolist(),
                'future_measurement_noise_included': False,
            },
        )
    return sources


def _measured_means(times, rate, sd, count, seed, label):
    times = np.asarray(times, dtype=float)
    raw = (np.exp(-rate*times)[None, :]
           + np.random.default_rng(seed).normal(0., sd, size=(count, len(times))))
    course, _ = prepare_time_course(
        times, raw.mean(axis=0), label=label,
        observation_uncertainty=ObservationUncertainty(
            sd/np.sqrt(count), response_kind='mean', replicate_counts=count,
            source='known_synthetic_measurement_sd', response_units='fraction remaining',
        ),
    )
    return course


def _descriptor(vitro_times=(), vitro_count=0, vivo_times=(), vivo_count=0):
    return {
        'in_vitro_times': np.asarray(vitro_times, dtype=float),
        'in_vitro_replicates_per_time': int(vitro_count),
        'in_vitro_added_measurements': len(vitro_times)*int(vitro_count),
        'in_vivo_times': np.asarray(vivo_times, dtype=float),
        'in_vivo_replicates_per_time': int(vivo_count),
        'in_vivo_added_measurements': len(vivo_times)*int(vivo_count),
        'selection': 'specified_candidate',
    }


def _assumptions():
    return {
        'model': 'fraction remaining = exp(b*time), with initial response fixed at 1',
        'reference': 'one specified in vitro curve and three weighted in vivo curves; not an inferred posterior',
        'in_vitro_rate': VITRO_RATE, 'in_vivo_rates': VIVO_RATES.tolist(),
        'in_vivo_weights': REFERENCE_WEIGHTS.tolist(),
        'historical_observations': 'held fixed in every experiment and across candidates',
        'fixed_data_uncertainty': 'supplied historical SEM still contributes to fitted and propagated uncertainty',
        'measurement_errors': 'independent Gaussian errors within and across times and environments',
        'observation_processing': 'unclipped; no monotonicity enforcement or data-dependent normalization',
        'historical_precision': 'known individual SD divided by sqrt(actual count), reported as SEM once',
        'target': 'separate fixed, noiseless in vitro curve exp(-0.25*time)',
        'target_transfer': 'synthetic target in vivo truth is defined by the calibration value ratio',
        'uncertainty_intervals': 'approximate pointwise 95% propagated calibration uncertainty',
        'downstream_future_measurement_scatter': 'excluded from prediction intervals',
        'coverage': 'pooled pointwise coverage under the specified references, conditional on fixed historical observations',
        'missing_results': 'all requested experiments retained; missing intervals count against coverage when truth is known',
        'schedules': 'specified candidates, not optimized schedules',
        'ivivc_evaluation_times': EVALUATION_TIMES.tolist(),
        'resource_comparison': 'candidate counts differ; no automatic ranking or equal-cost assumption',
        'time_zero': 'measurements at time zero count toward resources but do not identify the rate in this fixed-initial-response model',
        'interpretation': 'illustrative assumed example, not external validation, a sufficiency threshold, or evidence of transferability',
    }


def _study(settings, refs, target, options):
    seed = settings['random_seed']
    observed = _measured_means([0., 1., 2., 4., 6.], VITRO_RATE, .03, 3,
                               _seed(seed, 101), 'fixed in vitro calibration')
    designs = {
        'early_times': _descriptor(vivo_times=[0., 1., 2.], vivo_count=3),
        'spread_times': _descriptor(vivo_times=[0., 4., 8.], vivo_count=3),
        'more_times': _descriptor(vivo_times=[0., 2., 4., 6., 8.], vivo_count=3),
        'more_replicates': _descriptor(vivo_times=[0., 4., 8.], vivo_count=6),
    }
    for name, label in (('early_times', 'Early times'), ('spread_times', 'Spread times'),
                        ('more_times', 'More times'), ('more_replicates', 'More replicates')):
        designs[name]['display_name'] = label
    master = np.unique(np.concatenate([design['in_vivo_times'] for design in designs.values()]))
    # Allocate the maximum count once; candidates reuse the first n replicates.
    shared = create_simulation_plan(
        refs['in_vivo'], ObservationModel('constant_sd', .05, 6), master,
        n_simulations=settings['n_simulations'], random_state=_seed(seed, 201),
    )
    reference_model = fit_reference_model_ensemble(refs['in_vivo'], model_name=MODEL)
    request = IVIVCAssessmentRequest(observed, options, in_vitro_reference=refs['in_vitro'])
    recovery, recovery_summaries, calibrations, summaries = {}, {}, {}, {}
    for name, design in designs.items():
        observation = ObservationModel('constant_sd', .05, design['in_vivo_replicates_per_time'])
        recovery[name] = simulate_design(
            refs['in_vivo'], design['in_vivo_times'], observation, model_name=MODEL,
            reference_model=reference_model, simulation_plan=shared,
            evaluation_times=np.linspace(0., 8., 161),
            fallback_grid_search=False, ivivc_assessment=request,
        )
        recovery_summaries[name] = summarize_simulation(recovery[name])
        calibrations[name] = recovery[name].ivivc_assessment
        summaries[name] = summarize_calibration_simulation(calibrations[name])
    assumptions = _assumptions()
    assumptions.update(historical_in_vitro_individual_sd=.03,
                       historical_in_vitro_count=3, proposed_in_vivo_individual_sd=.05,
                       curve_recovery_evaluation_range=[0., 8.],
                       backend_entry_point='simulate_design with optional IVIVC assessment',
                       same_simulated_measurements_for_both_assessments=True)
    return {
        'name': 'study', 'title': 'In vivo study: curve recovery and IVIVC assessment',
        'description': 'Compare specified in vivo schedules and counts while holding companion in vitro data fixed.',
        'assumptions': assumptions, 'designs': designs,
        'existing_data': {'in_vitro': observed}, 'references': refs, 'target': target,
        'calibration_results': calibrations, 'calibration_summaries': summaries,
        'recovery_results': recovery, 'recovery_summaries': recovery_summaries,
        'simulation_plans': {'in_vivo': shared},
    }


def _refinement(settings, refs, target, options):
    seed = settings['random_seed']
    observed = {
        'in_vitro': _measured_means([0., .5, 1.5], VITRO_RATE, .06, 3,
                                   _seed(seed, 301), 'existing in vitro calibration'),
        'in_vivo': _measured_means([0., 2., 4., 8.], .16, .05, 3,
                                  _seed(seed, 302), 'existing in vivo calibration'),
    }
    designs = {
        'baseline': _descriptor(),
        'repeat_vitro': _descriptor(vitro_times=[.5, 1.5], vitro_count=3),
        'new_vitro': _descriptor(vitro_times=[3., 5.], vitro_count=3),
        'new_vivo': _descriptor(vivo_times=[6., 10.], vivo_count=3),
        'both': _descriptor(vitro_times=[3., 5.], vitro_count=3,
                            vivo_times=[6., 10.], vivo_count=3),
    }
    for name, label in (('baseline', 'Existing data only'), ('repeat_vitro', 'Repeat in vitro times'),
                        ('new_vitro', 'New in vitro times'), ('new_vivo', 'New in vivo times'),
                        ('both', 'New times in both')):
        designs[name]['display_name'] = label
    candidates = {}
    for name, design in designs.items():
        candidates[name] = CalibrationDesign(
            in_vitro_times=design['in_vitro_times'],
            in_vitro_observation_model=(ObservationModel('constant_sd', .06, 3)
                                       if len(design['in_vitro_times']) else None),
            in_vivo_times=design['in_vivo_times'],
            in_vivo_observation_model=(ObservationModel('constant_sd', .05, 3)
                                      if len(design['in_vivo_times']) else None),
        )
    comparison = compare_calibration_designs(
        candidates, assessment_options=options,
        existing_in_vitro=observed['in_vitro'], existing_in_vivo=observed['in_vivo'],
        in_vitro_reference=refs['in_vitro'], in_vivo_reference=refs['in_vivo'],
        n_simulations=settings['n_simulations'], random_state=_seed(seed, 401),
    )
    assumptions = _assumptions()
    assumptions.update(historical_in_vitro_individual_sd=.06, historical_in_vivo_individual_sd=.05,
                       historical_count_per_time=3, added_count_per_time=3,
                       backend_entry_point='compare_calibration_designs',
                       cohorts='additions are independent new cohorts; times are exposure or implantation ages, not calendar dates',
                       baseline='existing-data-only IVIVC; no observations regenerated')
    return {
        'name': 'refinement', 'title': 'Refining existing IVIVC calibration data',
        'description': 'Compare independent additions at existing or new times in vitro, in vivo, or both.',
        'assumptions': assumptions, 'designs': designs, 'existing_data': observed,
        'references': refs, 'target': target,
        'calibration_results': comparison.results, 'calibration_summaries': comparison.summaries,
        'recovery_results': {}, 'recovery_summaries': {},
        'simulation_plans': comparison.simulation_plans,
    }


def run_calibration_poc(n_simulations=100, n_interval_samples=200,
                        random_seed=12345, scenario='all') -> dict:
    """Run the two focused examples without creating output files."""
    n_simulations = _integer(n_simulations, 'n_simulations', 1)
    n_interval_samples = _integer(n_interval_samples, 'n_interval_samples', 2)
    random_seed = _integer(random_seed, 'random_seed', 0)
    if scenario not in ('all', 'study', 'refinement'):
        raise ValueError('scenario must be all, study, or refinement.')
    settings = dict(n_simulations=n_simulations, n_interval_samples=n_interval_samples,
                    random_seed=random_seed, model_name=MODEL, prediction_method='value_ratio',
                    response_scale='fraction remaining', time_units='arbitrary time units')
    refs = _references()
    target = TimeCourse(EVALUATION_TIMES, np.exp(-.25*EVALUATION_TIMES),
                        label='separate fixed target in vitro data')
    options = dict(model_name=MODEL, method='value_ratio', evaluation_times=EVALUATION_TIMES,
                   prediction=target, n_samples=n_interval_samples, random_state=random_seed,
                   confidence_level=.95,
                   analysis_config=dict(cv_scheme='none', grid_search_num_points=12,
                                        grid_search_num_cores=1, grid_search_random_state=random_seed))
    settings.update(analysis_config=dict(options['analysis_config']),
                    selected_normalizations=[], confidence_level=.95,
                    evaluation_times=EVALUATION_TIMES.tolist(),
                    seed_roles=dict(study_historical_vitro=101, study_measurements=201,
                                    refinement_historical_vitro=301,
                                    refinement_historical_vivo=302, refinement_measurements=401))
    builders = {'study': _study, 'refinement': _refinement}
    names = tuple(builders) if scenario == 'all' else (scenario,)
    # The shared fitter emits this warning for empty CV scores when CV is off.
    # Do not suppress fitting, covariance, sampling, or integration warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        scenarios = {name: builders[name](settings, refs, target, options) for name in names}
    return {'settings': settings, 'scenarios': scenarios}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('all', 'study', 'refinement'), default='all')
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--interval-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output-dir', default='study_design_calibration_poc_output')
    parser.add_argument('--no-html', action='store_true')
    args = parser.parse_args(argv)
    try:
        run = run_calibration_poc(args.simulations, args.interval_samples, args.seed, args.scenario)
    except ValueError as error:
        parser.error(str(error))
    from examples.study_design_calibration_reporting import write_calibration_poc_outputs
    output = write_calibration_poc_outputs(run, args.output_dir, include_html=not args.no_html)
    print(json.dumps({
        'output_directory': str(output.resolve()), 'settings': run['settings'],
        'scenarios': {name: list(value['designs']) for name, value in run['scenarios'].items()},
        'index_html': None if args.no_html else str((output / 'index.html').resolve()),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
