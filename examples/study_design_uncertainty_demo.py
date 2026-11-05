#!/usr/bin/env python3
"""Compare time selection under constant and unequal measurement SD.

Both schedules are assessed under the same unequal-error assumption. The
illustrative reference trajectories are shared, as are the Monte Carlo draws.
Run directly to print JSON; this example creates no output files.
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_design_core import (  # noqa: E402
    DesignConstraints, ObservationModel, ReferenceTrajectoryEnsemble,
    compare_designs_by_simulation, create_simulation_plan,
    evaluate_local_i_optimal_design, fit_reference_model_ensemble,
    optimize_local_i_optimal_designs,
)


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{label} must be a positive integer.')
    return int(value)


def run_uncertainty_comparison(
    n_simulations: int = 1000, random_seed: int = 123, replicates_per_time: int = 3,
) -> dict:
    """Return a JSON-safe comparison of two planning assumptions.

    These are local heuristic schedules for a small assumed linear example,
    not evidence that unequal-error planning always improves curve recovery.
    """
    n_simulations = _positive_integer(n_simulations, 'n_simulations')
    replicates_per_time = _positive_integer(replicates_per_time, 'replicates_per_time')
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral) or random_seed < 0:
        raise ValueError('random_seed must be a nonnegative integer.')
    random_seed = int(random_seed)
    grid = np.linspace(0., 4., 81)
    parameter_draws = np.array([[1., -.15], [1.03, -.16], [.97, -.14]])
    reference = ReferenceTrajectoryEnsemble(
        grid, parameter_draws[:, 0, None] + parameter_draws[:, 1, None]*grid,
        np.ones(3, dtype=bool), central_curve=1. - .15*grid, interpolation='linear',
    )
    reference_model = fit_reference_model_ensemble(reference, model_name='linear')
    constraints = DesignConstraints(
        candidate_times=np.arange(.5, 4.1, .5), fixed_times=[0.],
        min_time=0., max_time=4., min_number_of_times=3, max_number_of_times=3,
        replicates_per_time=replicates_per_time,
    )
    constant = ObservationModel('constant_sd', .05, replicates_per_time=replicates_per_time)
    unequal = ObservationModel('time_profile_sd', replicates_per_time=replicates_per_time,
        profile_times=[0., 1., 2., 3., 4.], profile_sd=[.03, .03, .05, .2, .3])
    assumptions = {'constant_sd': constant, 'unequal_sd': unequal}
    designs = {
        label: optimize_local_i_optimal_designs(reference_model, model, constraints,
            refine=False).design_for_n(3).times
        for label, model in assumptions.items()
    }
    plan = create_simulation_plan(reference, unequal, constraints.all_available_times,
                                  n_simulations=n_simulations, random_state=random_seed)
    comparison = compare_designs_by_simulation(
        reference, designs, unequal, model_name='linear', constraints=constraints,
        reference_model=reference_model, simulation_plan=plan, fallback_grid_search=False,
    )
    rows = []
    for label, times in designs.items():
        summary = comparison.summaries[label]
        rows.append({
            'planning_assumption': label,
            'selected_times': times.tolist(),
            'assessment_information_criterion': evaluate_local_i_optimal_design(
                times, reference_model, unequal, constraints).criterion,
            'mean_irmse': summary.mean_irmse,
            'p90_irmse': summary.p90_irmse,
            'failure_rate': summary.failure_rate,
        })
    return {
        'model_name': 'linear', 'time_range': [0., 4.],
        'candidate_times': constraints.candidate_times.tolist(), 'fixed_times': [0.],
        'number_of_times': 3, 'replicates_per_time': replicates_per_time,
        'n_simulations': n_simulations, 'random_seed': random_seed,
        'reference': {
            'uncertainty': 'three illustrative equally weighted complete trajectories',
            'parameter_names': ['a', 'b'], 'parameter_draws': parameter_draws.tolist(),
            'draw_weights': reference.draw_weights.tolist(),
            'future_measurement_noise_included': False,
        },
        'planning_observation_models': {label: model.to_metadata() for label, model in assumptions.items()},
        'assessment_observation_model': unequal.to_metadata(),
        'common_random_numbers': True, 'refit_objective': 'weighted_least_squares',
        'designs': rows,
        'interpretation': (
            'Both schedules use the same reference ensemble and are assessed under '
            'the unequal-SD model with shared trajectory and measurement-error draws. '
            'Reference uncertainty and future measurement variability are separate. '
            'Results illustrate this assumed example, not a guarantee of improvement '
            'or globally optimal time selection.'
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--replicates-per-time', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = run_uncertainty_comparison(args.simulations, args.seed, args.replicates_per_time)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
