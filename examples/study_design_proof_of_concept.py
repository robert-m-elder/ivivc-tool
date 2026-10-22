#!/usr/bin/env python3
"""Run reproducible proof-of-concept degradation study-design examples.

The runner demonstrates three backend-only workflows:

1. fit measured in vivo data and optimize a prospective time course;
2. generate an uncertain in vivo trajectory through the IVIVC core and optimize
   sampling on that predicted trajectory;
3. fit an incomplete in vivo study, lock the completed times, and optimize the
   next time points.

Each scenario uses the same downstream workflow:

reference trajectory ensemble
    -> analysis-model parameter ensemble
    -> rank-aware I-optimal time selection
    -> Monte Carlo recovery assessment
    -> comparison with simple baseline schedules

No Flask views, templates, or existing IVIVC presentation code are used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
import warnings

# Permit direct execution with ``python examples/study_design_proof_of_concept.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivivc_core import (  # noqa: E402
    TimeCourse,
    fit_ivivc,
    fit_time_course,
    generate_trajectory_ensembles,
    get_model_spec,
    iter_model_specs,
    prepare_data,
)
from study_design_core import (  # noqa: E402
    DesignConstraints,
    DesignOptimizationResult,
    DesignSimulationComparison,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
    compare_designs_by_simulation,
    create_simulation_plan,
    equal_response_spaced_design,
    equally_spaced_design,
    fit_reference_model_ensemble,
    log_spaced_design,
    optimize_i_optimal_designs,
    reference_from_profile_fit,
    reference_from_trajectory_ensemble,
)


SCENARIO_NAMES = ('measured_in_vivo', 'ivivc_prediction', 'adaptive_refinement')


@contextmanager
def _quiet_expected_numerical_warnings():
    """Suppress expected fitting warnings while retaining structured failures."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning)
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            message='Mean of empty slice',
        )
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            message='.*encountered in.*',
        )
        yield


@dataclass(frozen=True)
class ProofOfConceptSettings:
    """Numerical and output settings shared by the built-in examples."""

    n_reference_draws: int = 200
    n_simulations: int = 250
    random_seed: int = 20261021
    refine: bool = True
    write_html: bool = True
    max_design_sizes: int | None = None
    analysis_model: str = 'mass_loss_one_phase'
    observation_sd: float = 0.05
    fit_initialization: str = 'data_driven'

    def __post_init__(self) -> None:
        if int(self.n_reference_draws) != self.n_reference_draws or self.n_reference_draws < 2:
            raise ValueError('n_reference_draws must be an integer of at least two.')
        if int(self.n_simulations) != self.n_simulations or self.n_simulations < 2:
            raise ValueError('n_simulations must be an integer of at least two.')
        if self.max_design_sizes is not None:
            if int(self.max_design_sizes) != self.max_design_sizes or self.max_design_sizes < 1:
                raise ValueError('max_design_sizes must be a positive integer when supplied.')
        spec = get_model_spec(self.analysis_model)
        if 'approach2' not in spec.approaches:
            raise ValueError(
                f'analysis_model {self.analysis_model!r} is not registered for Approach 2.'
            )
        if not np.isfinite(self.observation_sd) or self.observation_sd <= 0:
            raise ValueError('observation_sd must be finite and positive.')
        initialization = str(self.fit_initialization).strip().lower()
        if initialization not in {'data_driven', 'central_reference', 'oracle'}:
            raise ValueError(
                "fit_initialization must be 'data_driven', 'central_reference', or 'oracle'."
            )
        object.__setattr__(self, 'fit_initialization', initialization)


@dataclass
class ScenarioDefinition:
    """Inputs required by the common proof-of-concept workflow."""

    name: str
    title: str
    description: str
    reference: ReferenceTrajectoryEnsemble
    observation_model: ObservationModel
    constraints: DesignConstraints
    analysis_model: str
    observed_data: pd.DataFrame
    target_response: float | None = None
    target_direction: str = 'decreasing'
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in SCENARIO_NAMES:
            raise ValueError(f'Unknown proof-of-concept scenario name: {self.name!r}.')
        self.observed_data = self.observed_data.copy()
        self.metadata = dict(self.metadata)


@dataclass
class ScenarioResult:
    """Optimization and simulation results for one proof-of-concept scenario."""

    definition: ScenarioDefinition
    reference_model: ReferenceModelEnsemble
    optimization: DesignOptimizationResult
    comparisons_by_n: Mapping[int, DesignSimulationComparison]
    performance_table: pd.DataFrame
    selected_times_table: pd.DataFrame

    def __post_init__(self) -> None:
        self.comparisons_by_n = dict(self.comparisons_by_n)
        self.performance_table = self.performance_table.copy()
        self.selected_times_table = self.selected_times_table.copy()


@dataclass
class ProofOfConceptRun:
    """Complete set of requested proof-of-concept scenarios."""

    settings: ProofOfConceptSettings
    scenarios: Mapping[str, ScenarioResult]
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        self.scenarios = dict(self.scenarios)
        if not self.scenarios:
            raise ValueError('At least one proof-of-concept scenario is required.')
        if self.output_directory is not None:
            self.output_directory = Path(self.output_directory)



def _analysis_config(seed: int) -> dict[str, object]:
    """Use the current fitting machinery without unnecessary CV work."""
    return {
        'cv_scheme': 'none',
        'cv_n_splits': 2,
        'cv_test_size': 0.25,
        'cv_random_state': int(seed),
        'grid_search_num_points': 25,
        'grid_search_num_cores': 1,
        'grid_search_param_min': -5.0,
        'grid_search_param_max': 5.0,
        'grid_search_random_state': int(seed),
        'include_cv_fit_variability_plot': False,
    }




def _one_phase_curve(
    time: np.ndarray,
    *,
    t50: float,
    width: float,
    initial_response: float = 1.0,
    loss_magnitude: float = 1.0,
) -> np.ndarray:
    """Generate the built-in synthetic truth using the registered model."""
    spec = get_model_spec('mass_loss_one_phase')
    return np.asarray(
        spec.model_function(
            np.asarray(time, dtype=float),
            float(initial_response),
            float(loss_magnitude),
            float(t50),
            float(width),
        ),
        dtype=float,
    )



def _scenario_max_n(
    minimum: int,
    default_maximum: int,
    settings: ProofOfConceptSettings,
) -> int:
    if settings.max_design_sizes is None:
        return default_maximum
    return min(default_maximum, minimum + int(settings.max_design_sizes) - 1)



def _candidate_grid(
    lower: float,
    upper: float,
    *,
    step: float,
    include_zero: bool = True,
) -> np.ndarray:
    if not lower < upper:
        raise ValueError('Candidate-grid bounds must be increasing.')
    first = np.ceil(lower / step) * step
    last = np.floor(upper / step) * step
    grid = np.arange(first, last + 0.5 * step, step, dtype=float)
    if include_zero and lower <= 0.0 <= upper:
        grid = np.union1d(grid, [0.0])
    if len(grid) < 2:
        raise ValueError('The supported domain is too narrow for the candidate-time step.')
    return grid



def build_measured_in_vivo_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Fit replicate-level measured in vivo data and plan a new full study."""
    rng = np.random.default_rng(settings.random_seed + 101)
    distinct_times = np.array([0.0, 4.0, 10.0, 16.0, 22.0, 28.0, 36.0, 48.0])
    true_means = _one_phase_curve(distinct_times, t50=23.0, width=4.5)
    replicate_count = 3
    observed = true_means[:, None] + rng.normal(
        0.0,
        settings.observation_sd,
        size=(len(distinct_times), replicate_count),
    )
    observed = np.clip(observed, 0.002, None)
    raw_time = np.repeat(distinct_times, replicate_count)
    raw_response = observed.reshape(-1)

    with _quiet_expected_numerical_warnings():
        fit_batch = fit_time_course(
            raw_time,
            raw_response,
            model_names=settings.analysis_model,
            selected_metrics=['rmse'],
            analysis_config=_analysis_config(settings.random_seed + 102),
            approach_id='approach2',
        )
    if fit_batch.errors:
        raise RuntimeError(
            'Measured in vivo example fit failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in fit_batch.errors.items())
        )
    fit = fit_batch[settings.analysis_model]
    reference_grid = np.linspace(0.0, 48.0, 193)
    reference = reference_from_profile_fit(
        fit,
        reference_grid,
        n_samples=settings.n_reference_draws,
        random_state=settings.random_seed + 103,
    )

    minimum_n = 4
    maximum_n = _scenario_max_n(minimum_n, 7, settings)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 48.0 + 1e-12, 2.0),
        fixed_times=np.array([0.0]),
        min_time=0.0,
        max_time=48.0,
        minimum_spacing=2.0,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=4,
    )
    observed_data = pd.DataFrame({
        'time': raw_time,
        'response': raw_response,
        'replicate_id': np.tile(np.arange(1, replicate_count + 1), len(distinct_times)),
        'role': 'reference measured in vivo data',
    })
    return ScenarioDefinition(
        name='measured_in_vivo',
        title='Measured in vivo reference data',
        description=(
            'Replicate-level in vivo observations are fitted with the selected '
            'analysis model. Parameter uncertainty defines plausible reference '
            'trajectories, while future-study variability is specified separately.'
        ),
        reference=reference,
        observation_model=ObservationModel(
            kind='constant_sd',
            value=settings.observation_sd,
            replicates_per_time=4,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        target_response=0.5,
        metadata={
            'reference_source': 'measured_in_vivo_replicates',
            'reference_model': settings.analysis_model,
            'reference_data_time_points': distinct_times.tolist(),
            'reference_replicates_per_time': replicate_count,
            'reference_data_sd': settings.observation_sd,
            'future_observation_sd': settings.observation_sd,
        },
    )



def build_ivivc_prediction_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Generate an uncertain in vivo reference trajectory through Approach 1."""
    in_vitro_time = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0])
    in_vitro_response = _one_phase_curve(
        in_vitro_time,
        t50=18.0,
        width=3.8,
    )
    time_scale = 1.85
    mapping_offsets = np.array([0.0, 0.15, -0.20, 0.25, -0.15, 0.30, -0.10, 0.20])
    in_vivo_time = time_scale * in_vitro_time + mapping_offsets
    in_vivo_response = in_vitro_response.copy()

    prepared = prepare_data(
        in_vitro_time,
        in_vitro_response,
        in_vivo_time,
        in_vivo_response,
        selected_interpolation=['none'],
    )
    with _quiet_expected_numerical_warnings():
        analysis = fit_ivivc(
            prepared,
            selections='approach1:linear_zero_intercept',
            selected_metrics=['rmse'],
            analysis_config=_analysis_config(settings.random_seed + 201),
        )
    if analysis.errors:
        raise RuntimeError(
            'IVIVC example fit failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in analysis.errors.items())
        )

    prediction_time = np.linspace(0.0, 36.0, 37)
    prediction_response = _one_phase_curve(
        prediction_time,
        t50=18.0,
        width=3.8,
    )
    prediction = TimeCourse(
        prediction_time,
        prediction_response,
        label='in_vitro_prediction',
    )
    ensemble_batch = generate_trajectory_ensembles(
        analysis,
        prediction,
        methods={'approach1': 'time_mapping'},
        n_samples=settings.n_reference_draws,
        random_state=settings.random_seed + 202,
    )
    if ensemble_batch.errors:
        raise RuntimeError(
            'IVIVC trajectory generation failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in ensemble_batch.errors.items())
        )
    trajectory_ensemble = ensemble_batch['approach1:linear_zero_intercept:time_mapping']
    reference = reference_from_trajectory_ensemble(
        trajectory_ensemble,
        grid_size=193,
        interpolation='pchip',
    )

    lower, upper = reference.supported_time_range
    candidates = _candidate_grid(lower, upper, step=2.0, include_zero=True)
    fixed = np.array([0.0]) if np.any(np.isclose(candidates, 0.0)) else np.array([candidates[0]])
    minimum_n = 4
    maximum_n = _scenario_max_n(minimum_n, 7, settings)
    constraints = DesignConstraints(
        candidate_times=candidates,
        fixed_times=fixed,
        min_time=float(candidates[0]),
        max_time=float(candidates[-1]),
        minimum_spacing=2.0,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=4,
    )
    observed_data = pd.DataFrame({
        'in_vitro_time': in_vitro_time,
        'in_vitro_response': in_vitro_response,
        'in_vivo_time': in_vivo_time,
        'in_vivo_response': in_vivo_response,
        'role': 'IVIVC calibration data',
    })
    return ScenarioDefinition(
        name='ivivc_prediction',
        title='IVIVC-predicted in vivo reference trajectory',
        description=(
            'Paired calibration data are fitted with an Approach 1 time mapping. '
            'Joint mapping-parameter draws are propagated across the complete '
            'in vitro prediction profile to create plausible in vivo trajectories.'
        ),
        reference=reference,
        observation_model=ObservationModel(
            kind='constant_sd',
            value=settings.observation_sd,
            replicates_per_time=4,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        target_response=0.5,
        metadata={
            'reference_source': 'ivivc_approach1_prediction',
            'ivivc_selection': 'approach1:linear_zero_intercept',
            'prediction_method': 'time_mapping',
            'in_vitro_prediction_points': len(prediction_time),
            'future_observation_sd': settings.observation_sd,
        },
    )



def build_adaptive_refinement_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Fit an incomplete study and lock its completed sampling times."""
    rng = np.random.default_rng(settings.random_seed + 301)
    completed_times = np.array([0.0, 8.0, 16.0, 24.0, 28.0])
    true_means = _one_phase_curve(completed_times, t50=30.0, width=5.0)
    replicate_count = 3
    observed = true_means[:, None] + rng.normal(
        0.0,
        settings.observation_sd,
        size=(len(completed_times), replicate_count),
    )
    observed = np.clip(observed, 0.002, None)
    raw_time = np.repeat(completed_times, replicate_count)
    raw_response = observed.reshape(-1)

    with _quiet_expected_numerical_warnings():
        fit_batch = fit_time_course(
            raw_time,
            raw_response,
            model_names=settings.analysis_model,
            selected_metrics=['rmse'],
            analysis_config=_analysis_config(settings.random_seed + 302),
            approach_id='approach2',
        )
    if fit_batch.errors:
        raise RuntimeError(
            'Adaptive example fit failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in fit_batch.errors.items())
        )
    fit = fit_batch[settings.analysis_model]
    reference_grid = np.linspace(0.0, 52.0, 209)
    reference = reference_from_profile_fit(
        fit,
        reference_grid,
        n_samples=settings.n_reference_draws,
        random_state=settings.random_seed + 303,
    )

    minimum_n = len(completed_times) + 1
    maximum_n = _scenario_max_n(minimum_n, 7, settings)
    constraints = DesignConstraints(
        candidate_times=np.arange(0.0, 52.0 + 1e-12, 2.0),
        fixed_times=completed_times,
        min_time=0.0,
        max_time=52.0,
        minimum_spacing=2.0,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=3,
    )
    observed_data = pd.DataFrame({
        'time': raw_time,
        'response': raw_response,
        'replicate_id': np.tile(np.arange(1, replicate_count + 1), len(completed_times)),
        'role': 'completed target-study in vivo data',
    })
    return ScenarioDefinition(
        name='adaptive_refinement',
        title='Adaptive refinement after early in vivo observations',
        description=(
            'The completed target-study times are locked. The trajectory '
            'distribution is estimated from the observations available so far, '
            'and the optimizer selects additional future times.'
        ),
        reference=reference,
        observation_model=ObservationModel(
            kind='constant_sd',
            value=settings.observation_sd,
            replicates_per_time=3,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        target_response=0.5,
        metadata={
            'reference_source': 'partial_target_in_vivo_study',
            'reference_model': settings.analysis_model,
            'completed_times': completed_times.tolist(),
            'reference_replicates_per_time': replicate_count,
            'reference_data_sd': settings.observation_sd,
            'future_observation_sd': settings.observation_sd,
            'note': 'The fitted reference beyond the latest completed time is extrapolative.',
        },
    )


SCENARIO_BUILDERS: Mapping[
    str,
    Callable[[ProofOfConceptSettings], ScenarioDefinition],
] = {
    'measured_in_vivo': build_measured_in_vivo_scenario,
    'ivivc_prediction': build_ivivc_prediction_scenario,
    'adaptive_refinement': build_adaptive_refinement_scenario,
}



def _baseline_designs(
    definition: ScenarioDefinition,
    number_of_times: int,
) -> dict[str, np.ndarray]:
    designs = {
        'equal_time': equally_spaced_design(definition.constraints, number_of_times),
        'log_time': log_spaced_design(definition.constraints, number_of_times),
    }
    try:
        designs['equal_response'] = equal_response_spaced_design(
            definition.reference,
            definition.constraints,
            number_of_times,
        )
    except ValueError:
        # Equal-response spacing is intentionally unavailable for nonmonotonic
        # or effectively constant central reference curves.
        pass
    return designs



def _information_lookup(
    optimization: DesignOptimizationResult,
) -> dict[tuple[int, str], tuple[float, float, int, float]]:
    lookup: dict[tuple[int, str], tuple[float, float, int, float]] = {}
    for label, steps in (
        ('optimized_nested', optimization.nested_steps),
        ('optimized_refined', optimization.refined_steps),
    ):
        for step in steps:
            lookup[(step.number_of_times, label)] = (
                step.score.criterion,
                step.score.full_rank_fraction,
                step.score.min_rank,
                step.score.mean_log_pseudodeterminant,
            )
    return lookup



def _performance_tables(
    definition: ScenarioDefinition,
    optimization: DesignOptimizationResult,
    comparisons: Mapping[int, DesignSimulationComparison],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    information = _information_lookup(optimization)
    rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for number_of_times in sorted(comparisons):
        comparison = comparisons[number_of_times]
        for label, summary in comparison.summaries.items():
            info = information.get((number_of_times, label))
            rows.append({
                'scenario': definition.name,
                'design': label,
                'number_of_times': number_of_times,
                'replicates_per_time': definition.constraints.replicates_per_time,
                'total_experimental_units': definition.constraints.total_experimental_units(
                    number_of_times
                ),
                'times': '; '.join(f'{value:g}' for value in summary.design_times),
                'information_criterion': np.nan if info is None else info[0],
                'full_rank_fraction': np.nan if info is None else info[1],
                'minimum_information_rank': np.nan if info is None else info[2],
                'mean_log_pseudodeterminant': np.nan if info is None else info[3],
                'n_simulations': summary.n_simulations,
                'n_successful': summary.n_successful,
                'failure_rate': summary.failure_rate,
                'median_irmse': summary.median_irmse,
                'p90_irmse': summary.p90_irmse,
                'p95_irmse': summary.p95_irmse,
                'median_normalized_irmse': summary.median_normalized_irmse,
                'p90_normalized_irmse': summary.p90_normalized_irmse,
                'penalized_mean_irmse': summary.penalized_mean_irmse,
                'penalized_mean_normalized_irmse': (
                    summary.penalized_mean_normalized_irmse
                ),
                'median_curve_interval_width': summary.median_curve_interval_width,
                'median_absolute_target_time_error': (
                    summary.median_absolute_target_time_error
                ),
                'parameter_bound_hit_rate': summary.parameter_bound_hit_rate,
            })
            for order, time in enumerate(summary.design_times, start=1):
                time_rows.append({
                    'scenario': definition.name,
                    'design': label,
                    'number_of_times': number_of_times,
                    'order': order,
                    'time': time,
                    'fixed': bool(
                        np.any(
                            np.isclose(
                                definition.constraints.fixed_times,
                                time,
                                rtol=0.0,
                                atol=1e-12,
                            )
                        )
                    ),
                })
    performance = pd.DataFrame(rows)
    selected_times = pd.DataFrame(time_rows)
    performance = performance.sort_values(
        ['number_of_times', 'design'],
        kind='stable',
    ).reset_index(drop=True)
    selected_times = selected_times.sort_values(
        ['number_of_times', 'design', 'order'],
        kind='stable',
    ).reset_index(drop=True)

    performance['marginal_penalized_nirmse_improvement'] = np.nan
    for label, index in performance.groupby('design', sort=False).groups.items():
        ordered = performance.loc[index].sort_values('number_of_times')
        previous = ordered['penalized_mean_normalized_irmse'].shift(1)
        improvement = previous - ordered['penalized_mean_normalized_irmse']
        performance.loc[ordered.index, 'marginal_penalized_nirmse_improvement'] = improvement
    return performance, selected_times



def run_scenario(
    definition: ScenarioDefinition,
    settings: ProofOfConceptSettings,
    *,
    random_seed: int | None = None,
) -> ScenarioResult:
    """Optimize and simulate one scenario using the common backend workflow."""
    seed = settings.random_seed if random_seed is None else int(random_seed)
    reference_model = fit_reference_model_ensemble(
        definition.reference,
        model_name=definition.analysis_model,
    )
    optimization = optimize_i_optimal_designs(
        reference_model,
        definition.observation_model,
        definition.constraints,
        refine=settings.refine,
    )
    plan = create_simulation_plan(
        definition.reference,
        definition.observation_model,
        definition.constraints.all_available_times,
        n_simulations=settings.n_simulations,
        random_state=seed,
    )

    refined_by_n = {
        step.number_of_times: step
        for step in optimization.refined_steps
    }
    comparisons: dict[int, DesignSimulationComparison] = {}
    for nested_step in optimization.nested_steps:
        number_of_times = nested_step.number_of_times
        designs: dict[str, Sequence[float]] = {
            'optimized_nested': nested_step.times,
        }
        if number_of_times in refined_by_n:
            designs['optimized_refined'] = refined_by_n[number_of_times].times
        designs.update(_baseline_designs(definition, number_of_times))
        comparisons[number_of_times] = compare_designs_by_simulation(
            definition.reference,
            designs,
            definition.observation_model,
            model_name=definition.analysis_model,
            constraints=definition.constraints,
            reference_model=reference_model,
            simulation_plan=plan,
            target_response=definition.target_response,
            target_direction=definition.target_direction,
            fit_initialization=settings.fit_initialization,
        )

    performance, selected_times = _performance_tables(
        definition,
        optimization,
        comparisons,
    )
    return ScenarioResult(
        definition=definition,
        reference_model=reference_model,
        optimization=optimization,
        comparisons_by_n=comparisons,
        performance_table=performance,
        selected_times_table=selected_times,
    )



def run_proof_of_concept(
    scenario_names: Sequence[str] = SCENARIO_NAMES,
    *,
    settings: ProofOfConceptSettings | None = None,
) -> ProofOfConceptRun:
    """Run one or more built-in proof-of-concept scenarios."""
    settings = ProofOfConceptSettings() if settings is None else settings
    unique_names = tuple(dict.fromkeys(str(name) for name in scenario_names))
    unknown = [name for name in unique_names if name not in SCENARIO_BUILDERS]
    if unknown:
        raise ValueError('Unknown scenario(s): ' + ', '.join(unknown))

    seed_sequence = np.random.SeedSequence(settings.random_seed)
    child_seeds = seed_sequence.spawn(len(unique_names))
    results: dict[str, ScenarioResult] = {}
    for name, child_seed in zip(unique_names, child_seeds):
        with _quiet_expected_numerical_warnings():
            definition = SCENARIO_BUILDERS[name](settings)
            simulation_seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
            results[name] = run_scenario(
                definition,
                settings,
                random_seed=simulation_seed,
            )
    return ProofOfConceptRun(settings=settings, scenarios=results)



def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.ndim != 1 or values.shape[0] != len(weights):
        raise ValueError('Weighted-quantile inputs have incompatible shapes.')
    output = np.empty(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind='mergesort')
        ordered_values = values[order, column]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)
        cumulative /= cumulative[-1]
        output[column] = np.interp(quantile, cumulative, ordered_values)
    return output



def _reference_table(reference: ReferenceTrajectoryEnsemble) -> pd.DataFrame:
    valid = reference.valid_draws & (reference.draw_weights > 0)
    draws = reference.value_draws[valid]
    weights = reference.draw_weights[valid]
    weights = weights / np.sum(weights)
    return pd.DataFrame({
        'time': reference.time_grid,
        'central_response': reference.central_curve,
        'lower_95': _weighted_quantile(draws, weights, 0.025),
        'upper_95': _weighted_quantile(draws, weights, 0.975),
    })



def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)



def _scenario_metadata(result: ScenarioResult) -> dict[str, Any]:
    definition = result.definition
    return {
        'name': definition.name,
        'title': definition.title,
        'description': definition.description,
        'analysis_model': definition.analysis_model,
        'target_response': definition.target_response,
        'target_direction': definition.target_direction,
        'reference': {
            'draws': definition.reference.n_draws,
            'valid_draws': definition.reference.n_valid_draws,
            'supported_time_range': definition.reference.supported_time_range,
            'metadata': definition.reference.metadata,
        },
        'observation_model': {
            'kind': definition.observation_model.kind,
            'value': definition.observation_model.value,
            'minimum_sd': definition.observation_model.minimum_sd,
            'replicates_per_time': definition.observation_model.replicates_per_time,
            'source': definition.observation_model.source,
        },
        'constraints': {
            'candidate_times': definition.constraints.candidate_times,
            'fixed_times': definition.constraints.fixed_times,
            'min_time': definition.constraints.min_time,
            'max_time': definition.constraints.max_time,
            'minimum_spacing': definition.constraints.minimum_spacing,
            'min_number_of_times': definition.constraints.min_number_of_times,
            'max_number_of_times': definition.constraints.max_number_of_times,
            'replicates_per_time': definition.constraints.replicates_per_time,
        },
        'optimization': result.optimization.metadata,
        'scenario_metadata': definition.metadata,
    }



def _plotly_summary_html(result: ScenarioResult) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError as exception:  # pragma: no cover - project requirements include Plotly
        raise RuntimeError('Plotly is required to write proof-of-concept HTML output.') from exception

    reference_table = _reference_table(result.definition.reference)
    reference_figure = go.Figure()
    reference_figure.add_trace(go.Scatter(
        x=reference_table['time'],
        y=reference_table['upper_95'],
        mode='lines',
        line={'width': 0},
        showlegend=False,
        hoverinfo='skip',
    ))
    reference_figure.add_trace(go.Scatter(
        x=reference_table['time'],
        y=reference_table['lower_95'],
        mode='lines',
        fill='tonexty',
        name='95% reference envelope',
        line={'width': 0},
        hoverinfo='skip',
        fillcolor='rgba(136, 231, 136, 0.5)',
    ))
    reference_figure.add_trace(go.Scatter(
        x=reference_table['time'],
        y=reference_table['central_response'],
        mode='lines',
        name='Central reference trajectory',
        line=dict(color='rgb(27, 136, 37)'),
    ))

    observed = result.definition.observed_data
    if {'time', 'response'}.issubset(observed.columns):
        reference_figure.add_trace(go.Scatter(
            x=observed['time'],
            y=observed['response'],
            mode='markers',
            name='Observed in vivo data',
        ))
    elif {'in_vivo_time', 'in_vivo_response'}.issubset(observed.columns):
        reference_figure.add_trace(go.Scatter(
            x=observed['in_vivo_time'],
            y=observed['in_vivo_response'],
            mode='markers',
            name='IVIVC calibration in vivo data',
        ))

    maximum_n = int(result.performance_table['number_of_times'].max())
    available_labels = set(
        result.performance_table.loc[
            result.performance_table['number_of_times'] == maximum_n,
            'design',
        ]
    )
    msize = {'optimized_refined':18, 'optimized_nested':14, 'equal_time':10}
    for label in ('optimized_refined', 'optimized_nested', 'equal_time'):
        if label not in available_labels:
            continue
        times = result.selected_times_table.loc[
            (result.selected_times_table['number_of_times'] == maximum_n)
            & (result.selected_times_table['design'] == label),
            'time',
        ].to_numpy(dtype=float)
        response = result.definition.reference.evaluate_central(times)
        reference_figure.add_trace(go.Scatter(
            x=times,
            y=response,
            mode='markers',
            marker={'size': msize[label]},
            name=f'{label.replace("_", " ")} (N={maximum_n})',
        ))
    reference_figure.update_layout(
        title='Reference trajectory and representative schedules',
        xaxis_title='Time',
        yaxis_title='Response',
        template='plotly_white',
    )

    performance_figure = go.Figure()
    for label, group in result.performance_table.groupby('design', sort=False):
        group = group.sort_values('number_of_times')
        performance_figure.add_trace(go.Scatter(
            x=group['number_of_times'],
            y=group['penalized_mean_normalized_irmse'],
            mode='lines+markers',
            name=label.replace('_', ' '),
        ))
    performance_figure.update_layout(
        title='Expected curve-recovery error versus number of time points',
        xaxis_title='Number of distinct time points',
        yaxis_title='Penalized mean normalized IRMSE',
        template='plotly_white',
    )

    schedule_figure = go.Figure()
    schedule_data = result.selected_times_table[
        result.selected_times_table['design'].isin(
            ['optimized_nested', 'optimized_refined']
        )
    ]
    msize = {'optimized_nested':14, 'optimized_refined':10}
    for label, group in schedule_data.groupby('design', sort=False):
        schedule_figure.add_trace(go.Scatter(
            x=group['time'],
            y=group['number_of_times'],
            mode='markers',
            marker={'size': msize[label], 'symbol': 'circle'},
            name=label.replace('_', ' '),
            customdata=np.stack([group['fixed']], axis=-1),
            hovertemplate=(
                'time=%{x:g}<br>N=%{y}<br>fixed=%{customdata[0]}<extra></extra>'
            ),
        ))
    schedule_figure.update_layout(
        title='Selected optimized times across design sizes',
        xaxis_title='Time',
        yaxis_title='Number of distinct time points',
        template='plotly_white',
    )

    table_columns = [
        'design',
        'number_of_times',
        'times',
        'failure_rate',
        'median_normalized_irmse',
        'p90_normalized_irmse',
        'penalized_mean_normalized_irmse',
    ]
    table = result.performance_table[table_columns].copy()
    for column in table.select_dtypes(include=[np.number]).columns:
        table[column] = table[column].map(
            lambda value: '' if not np.isfinite(value) else f'{value:.5g}'
        )

    first_figure = to_html(
        reference_figure,
        full_html=False,
        include_plotlyjs='cdn',
    )
    second_figure = to_html(
        performance_figure,
        full_html=False,
        include_plotlyjs=False,
    )
    third_figure = to_html(
        schedule_figure,
        full_html=False,
        include_plotlyjs=False,
    )
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{result.definition.title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
    th {{ background: #f3f3f3; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>{result.definition.title}</h1>
  <p>{result.definition.description}</p>
  <p><strong>Analysis model:</strong> <code>{result.definition.analysis_model}</code>;
     <strong>reference draws:</strong> {result.definition.reference.n_valid_draws};
     <strong>future observation model:</strong> <code>{result.definition.observation_model.kind}={result.definition.observation_model.value:g}</code>;
     <strong>future replicates/time:</strong> {result.definition.constraints.replicates_per_time}.</p>
  {first_figure}
  {second_figure}
  {third_figure}
  <h2>Performance summary</h2>
  {table.to_html(index=False, escape=True)}
</body>
</html>
'''



def write_proof_of_concept_outputs(
    run: ProofOfConceptRun,
    output_directory: str | Path,
) -> ProofOfConceptRun:
    """Write transparent scenario outputs and a simple static index."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    combined_performance: list[pd.DataFrame] = []
    combined_times: list[pd.DataFrame] = []
    scenario_links: list[tuple[str, str]] = []

    for name, result in run.scenarios.items():
        scenario_directory = output_directory / name
        scenario_directory.mkdir(parents=True, exist_ok=True)
        result.definition.observed_data.to_csv(
            scenario_directory / 'observed_or_calibration_data.csv',
            index=False,
        )
        _reference_table(result.definition.reference).to_csv(
            scenario_directory / 'reference_trajectory.csv',
            index=False,
        )
        result.performance_table.to_csv(
            scenario_directory / 'design_performance.csv',
            index=False,
        )
        result.selected_times_table.to_csv(
            scenario_directory / 'selected_times.csv',
            index=False,
        )
        with (scenario_directory / 'metadata.json').open('w', encoding='utf-8') as handle:
            json.dump(
                _jsonable(_scenario_metadata(result)),
                handle,
                indent=2,
                sort_keys=True,
            )
        if run.settings.write_html:
            (scenario_directory / 'summary.html').write_text(
                _plotly_summary_html(result),
                encoding='utf-8',
            )
            scenario_links.append((result.definition.title, f'{name}/summary.html'))
        combined_performance.append(result.performance_table)
        combined_times.append(result.selected_times_table)

    pd.concat(combined_performance, ignore_index=True).to_csv(
        output_directory / 'combined_design_performance.csv',
        index=False,
    )
    pd.concat(combined_times, ignore_index=True).to_csv(
        output_directory / 'combined_selected_times.csv',
        index=False,
    )
    manifest = {
        'settings': {
            'n_reference_draws': run.settings.n_reference_draws,
            'n_simulations': run.settings.n_simulations,
            'random_seed': run.settings.random_seed,
            'refine': run.settings.refine,
            'write_html': run.settings.write_html,
            'max_design_sizes': run.settings.max_design_sizes,
            'analysis_model': run.settings.analysis_model,
            'observation_sd': run.settings.observation_sd,
            'fit_initialization': run.settings.fit_initialization,
        },
        'scenarios': list(run.scenarios),
        'files': {
            'combined_performance': 'combined_design_performance.csv',
            'combined_selected_times': 'combined_selected_times.csv',
        },
    }
    with (output_directory / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if run.settings.write_html:
        links = '\n'.join(
            f'<li><a href="{href}">{title}</a></li>'
            for title, href in scenario_links
        )
        index_html = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Study-design proof of concept</title></head>
<body style="font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto;">
<h1>Study-design proof of concept</h1>
<p>Each scenario uses the same information-based optimization and Monte Carlo evaluation engine.</p>
<ul>{links}</ul>
<p>Machine-readable combined results are available in <code>combined_design_performance.csv</code>
   and <code>combined_selected_times.csv</code>.</p>
</body>
</html>
'''
        (output_directory / 'index.html').write_text(index_html, encoding='utf-8')

    run.output_directory = output_directory
    return run



def _print_console_summary(run: ProofOfConceptRun) -> None:
    for name, result in run.scenarios.items():
        print(f'\n{name}: {result.definition.title}')
        preferred = (
            'optimized_refined'
            if 'optimized_refined' in set(result.performance_table['design'])
            else 'optimized_nested'
        )
        subset = result.performance_table[
            result.performance_table['design'] == preferred
        ].sort_values('number_of_times')
        for row in subset.itertuples(index=False):
            print(
                f'  N={row.number_of_times}: [{row.times}]  '
                f'penalized normalized IRMSE={row.penalized_mean_normalized_irmse:.4g}, '
                f'failure rate={row.failure_rate:.1%}'
            )



def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run backend-only proof-of-concept degradation study-design examples.',
    )
    parser.add_argument(
        '--scenario',
        choices=('all',) + SCENARIO_NAMES,
        default='all',
        help='Built-in scenario to run (default: all).',
    )
    parser.add_argument(
        '--output-dir',
        default='study_design_poc_output',
        help='Directory for CSV, JSON, and optional HTML output.',
    )
    parser.add_argument(
        '--reference-draws',
        type=int,
        default=200,
        help='Number of plausible reference trajectories (default: 200).',
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=250,
        help='Monte Carlo studies per design (default: 250).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=20261021,
        help='Random seed used throughout the run.',
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Skip coordinate-exchange refinement and retain nested greedy designs only.',
    )
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='Write machine-readable files only.',
    )
    parser.add_argument(
        '--analysis-model',
        choices=tuple(spec.key for spec in iter_model_specs('approach2')),
        default='mass_loss_one_phase',
        help=(
            'Registered Approach 2 model used to analyze simulated in vivo studies '
            '(default: mass_loss_one_phase).'
        ),
    )
    parser.add_argument(
        '--observation-sd',
        type=float,
        default=0.05,
        help=(
            'Constant absolute SD used for built-in measured data and all future-study '
            'simulations (default: 0.05).'
        ),
    )
    parser.add_argument(
        '--fit-initialization',
        choices=('data_driven', 'central_reference', 'oracle'),
        default='data_driven',
        help=(
            'Initialization used when refitting simulated studies. The default uses '
            'only the simulated data; oracle is diagnostic only.'
        ),
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use small draw/simulation counts and only two design sizes.',
    )
    return parser.parse_args(argv)



def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reference_draws = args.reference_draws
    simulations = args.simulations
    max_design_sizes = None
    if args.quick:
        reference_draws = min(reference_draws, 30)
        simulations = min(simulations, 40)
        max_design_sizes = 2
    settings = ProofOfConceptSettings(
        n_reference_draws=reference_draws,
        n_simulations=simulations,
        random_seed=args.seed,
        refine=not args.no_refine,
        write_html=not args.no_html,
        max_design_sizes=max_design_sizes,
        analysis_model=args.analysis_model,
        observation_sd=args.observation_sd,
        fit_initialization=args.fit_initialization,
    )
    scenario_names = SCENARIO_NAMES if args.scenario == 'all' else (args.scenario,)
    run = run_proof_of_concept(scenario_names, settings=settings)
    write_proof_of_concept_outputs(run, args.output_dir)
    _print_console_summary(run)
    print(f'\nOutputs written to: {Path(args.output_dir).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
