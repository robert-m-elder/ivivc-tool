#!/usr/bin/env python3
"""Run reproducible proof-of-concept degradation study-design examples.

The runner demonstrates five backend-only scenarios:

1. fit measured in vivo data and optimize a prospective time course;
2. generate an uncertain in vivo trajectory through the IVIVC core and optimize
   sampling on that predicted trajectory;
3. fit an incomplete in vivo study and optimize additional time points;
4. update an IVIVC-predicted in vivo trajectory using completed target-study
   observations and optimize additional time points;
5. use that IVIVC-informed workflow to fill a deliberately missed transition
   region between completed observations.

Each scenario uses the same downstream workflow:

reference trajectory ensemble
    -> central analysis-model fit
    -> rank-aware local I-optimal time selection
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
    prepare_data,
)
from examples.study_design_poc_models import (  # noqa: E402
    generate_noisy_monotonic_responses,
    get_synthetic_model_preset,
    supported_synthetic_models,
)
from study_design_core import (  # noqa: E402
    DesignConstraints,
    DesignOptimizationResult,
    ExistingObservations,
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
    optimize_local_i_optimal_designs,
    reference_from_profile_fit,
    reference_from_trajectory_ensemble,
    reweight_reference_ensemble,
    threshold_crossing_time,
)


SCENARIO_NAMES = (
    'measured_in_vivo',
    'ivivc_prediction',
    'adaptive_in_vivo_only',
    'adaptive_ivivc_prediction',
    'adaptive_gap_filling',
)
LEGACY_SCENARIO_NAMES = ('adaptive_refinement',)


DESIGN_LABELS: Mapping[str, str] = {
    'optimized_nested': 'I-optimal (add-one-at-a-time)',
    'optimized_refined': 'I-optimal (refined at fixed N)',
    'equal_time': 'Equal time spacing',
    'log_time': 'Log time spacing',
    'equal_response': 'Equal response spacing',
}

DESIGN_EXPLANATIONS: Mapping[str, str] = {
    'optimized_nested': (
        'Greedy local I-optimal sequence: each larger schedule retains the '
        'previously selected times and adds one new time.'
    ),
    'optimized_refined': (
        'Fixed-N local refinement of the add-one-at-a-time schedule using '
        'coordinate exchange; it uses the same local I-optimal criterion.'
    ),
}


# Shared proof-of-concept study settings. Keep numerical example constants here
# so exploratory changes propagate consistently through reference construction,
# design constraints, observation models, and adaptive workflows.
POC_FUTURE_REPLICATES_PER_TIME = 3
POC_MINIMUM_SPACING = 1.0
POC_PROSPECTIVE_MINIMUM_N = 3
POC_DEFAULT_MAXIMUM_N = 7
ADAPTIVE_CANDIDATE_POLICIES = ('future_only', 'any_unobserved')

# Common synthetic in vivo truth used by every proof-of-concept scenario.
POC_IN_VIVO_TIME_RANGE = (0.0, 52.0)
#POC_IN_VIVO_TRANSITION_TIME = 30.0
#POC_IN_VIVO_TRANSITION_WIDTH = 5.0
POC_IN_VIVO_TRANSITION_TIME = 25.0
POC_IN_VIVO_TRANSITION_WIDTH = 2.5

MEASURED_TIME_RANGE = POC_IN_VIVO_TIME_RANGE
MEASURED_REFERENCE_GRID_STEP = 0.5
MEASURED_CANDIDATE_STEP = 1.0
MEASURED_REFERENCE_REPLICATES_PER_TIME = 3

IVIVC_TRUE_TIME_SCALE = 1.85
IVIVC_IN_VITRO_TIME_RANGE = (
    POC_IN_VIVO_TIME_RANGE[0] / IVIVC_TRUE_TIME_SCALE,
    POC_IN_VIVO_TIME_RANGE[1] / IVIVC_TRUE_TIME_SCALE,
)
IVIVC_CALIBRATION_IN_VITRO_TIMES = (0.0, 6.0, 12.0, 16.0, 20.0, 24.0, 28.0)
IVIVC_CALIBRATION_IN_VIVO_TIMES = (0.0, 10.0, 20.0, 30.0, 40.0, 48.0, 52.0)
IVIVC_INTERPOLATION_METHOD = 'default'
IVIVC_PREDICTION_GRID_STEP = 0.5
IVIVC_CANDIDATE_STEP = 1.0

ADAPTIVE_IN_VIVO_TIME_RANGE = POC_IN_VIVO_TIME_RANGE
ADAPTIVE_REFERENCE_GRID_STEP = 0.5
ADAPTIVE_IN_VIVO_CANDIDATE_STEP = 1.0
ADAPTIVE_IVIVC_CANDIDATE_STEP = 1.0
#ADAPTIVE_COMPLETED_TIMES = (0, 8, 16, 24)
#ADAPTIVE_COMPLETED_TIMES = (0,8,16,24,28) # works with original ground truth (mass_loss_one_phase)
ADAPTIVE_COMPLETED_TIMES = (0.0, 10.0, 20.0) # works fine for mass_loss_one_phase_normalized
GAP_FILLING_COMPLETED_TIMES = (0.0, 10.0, 40.0)
#ADAPTIVE_COMPLETED_TIMES = (0.0, 10, 20, 22.0) 
#ADAPTIVE_COMPLETED_TIMES = (0.0, 10.0, 15, 20.0, 25) # XXX  does not work with narrower ground truth (mass_loss_one_phase)

# Reported curve-derived degradation endpoints. T50 means 50% degraded
# (50% of the initial response remaining); T90 means 90% degraded
# (10% of the initial response remaining). These do not affect optimization.
REPORTED_DEGRADATION_ENDPOINTS = (
    ('t50', 'T50 (50% degraded / 50% remaining)', 0.5),
    ('t90', 'T90 (90% degraded / 10% remaining)', 0.1),
)


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
    ivivc_calibration_response_sd: float = 0.05
    adaptive_candidate_policy: str = 'future_only'

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
        if self.analysis_model not in supported_synthetic_models():
            supported = ', '.join(supported_synthetic_models())
            raise ValueError(
                f'No proof-of-concept synthetic preset is defined for '
                f'{self.analysis_model!r}. Supported models: {supported}.'
            )
        if not np.isfinite(self.observation_sd) or self.observation_sd <= 0:
            raise ValueError('observation_sd must be finite and positive.')
        if (
            not np.isfinite(self.ivivc_calibration_response_sd)
            or self.ivivc_calibration_response_sd < 0
        ):
            raise ValueError(
                'ivivc_calibration_response_sd must be finite and nonnegative.'
            )
        initialization = str(self.fit_initialization).strip().lower()
        if initialization not in {'data_driven', 'central_reference', 'oracle'}:
            raise ValueError(
                "fit_initialization must be 'data_driven', 'central_reference', or 'oracle'."
            )
        object.__setattr__(self, 'fit_initialization', initialization)
        candidate_policy = str(self.adaptive_candidate_policy).strip().lower()
        if candidate_policy not in ADAPTIVE_CANDIDATE_POLICIES:
            allowed = ', '.join(ADAPTIVE_CANDIDATE_POLICIES)
            raise ValueError(
                f'adaptive_candidate_policy must be one of: {allowed}.'
            )
        object.__setattr__(self, 'adaptive_candidate_policy', candidate_policy)


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
    preprocessed_data: pd.DataFrame | None = None
    target_response: float | None = None
    target_direction: str = 'decreasing'
    ground_truth: TimeCourse | None = None
    existing_observations: ExistingObservations | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_model_initial_parameters: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.name not in SCENARIO_NAMES:
            raise ValueError(f'Unknown proof-of-concept scenario name: {self.name!r}.')
        self.observed_data = self.observed_data.copy()
        if self.preprocessed_data is not None:
            self.preprocessed_data = self.preprocessed_data.copy()
        self.metadata = dict(self.metadata)
        if self.ground_truth is not None:
            self.ground_truth = TimeCourse(
                self.ground_truth.time,
                self.ground_truth.value,
                label=self.ground_truth.label,
            )
            if len(self.ground_truth.time) < 2:
                raise ValueError('ground_truth must contain at least two points.')
            if not np.all(np.isfinite(self.ground_truth.time)) or not np.all(
                np.isfinite(self.ground_truth.value)
            ):
                raise ValueError('ground_truth must contain only finite values.')
            if np.any(np.diff(self.ground_truth.time) <= 0):
                raise ValueError('ground_truth times must be strictly increasing.')
        if self.existing_observations is not None:
            self.existing_observations = ExistingObservations(
                self.existing_observations.time,
                self.existing_observations.response,
                label=self.existing_observations.label,
            )
            for value in self.existing_observations.unique_times:
                if not np.any(
                    np.isclose(
                        self.constraints.fixed_times,
                        value,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        'Existing observation times must be included in fixed_times.'
                    )
        if self.reference_model_initial_parameters is not None:
            self.reference_model_initial_parameters = dict(
                self.reference_model_initial_parameters
            )


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





def _scenario_max_n(
    minimum: int,
    default_maximum: int,
    settings: ProofOfConceptSettings,
) -> int:
    if settings.max_design_sizes is None:
        return default_maximum
    return min(default_maximum, minimum + int(settings.max_design_sizes) - 1)



def _reference_grid(
    lower: float,
    upper: float,
    *,
    target_step: float,
) -> np.ndarray:
    """Return a dense grid spanning both requested endpoints."""
    lower = float(lower)
    upper = float(upper)
    target_step = float(target_step)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError('Reference-grid bounds must be finite and increasing.')
    if not np.isfinite(target_step) or target_step <= 0:
        raise ValueError('Reference-grid target_step must be finite and positive.')
    intervals = max(1, int(np.ceil((upper - lower) / target_step)))
    return np.linspace(lower, upper, intervals + 1)


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



def _adaptive_candidate_grid(
    lower: float,
    upper: float,
    *,
    completed_times: Sequence[float],
    step: float,
    policy: str,
) -> np.ndarray:
    """Return selectable elapsed times under one adaptive-candidate policy."""
    completed = np.unique(np.asarray(completed_times, dtype=float))
    if completed.ndim != 1 or len(completed) == 0 or np.any(~np.isfinite(completed)):
        raise ValueError('completed_times must contain at least one finite time.')
    if np.any(completed < lower) or np.any(completed > upper):
        raise ValueError('Completed times must lie within the candidate-grid bounds.')
    policy = str(policy).strip().lower()
    if policy not in ADAPTIVE_CANDIDATE_POLICIES:
        allowed = ', '.join(ADAPTIVE_CANDIDATE_POLICIES)
        raise ValueError(f'Unknown adaptive candidate policy; expected one of: {allowed}.')

    full = _candidate_grid(lower, upper, step=step, include_zero=False)
    observed = np.any(
        np.isclose(full[:, None], completed[None, :], rtol=0.0, atol=1e-12),
        axis=1,
    )
    candidates = full[~observed]
    if policy == 'future_only':
        candidates = candidates[candidates > float(completed[-1]) + 1e-12]
    if len(candidates) == 0:
        if policy == 'future_only':
            raise ValueError(
                'No candidate times remain after the latest completed elapsed time.'
            )
        raise ValueError('No unobserved candidate times remain in the supported range.')
    return candidates


def _adaptive_candidate_description(policy: str) -> str:
    if policy == 'future_only':
        return 'Only elapsed times later than the latest completed time are eligible.'
    return (
        'Any unobserved elapsed time in the supported range is eligible, including '
        'earlier gap-filling times.'
    )


def _common_in_vivo_preset(settings: ProofOfConceptSettings):
    """Return the common synthetic in vivo trajectory used across scenarios."""
    time_min, time_max = POC_IN_VIVO_TIME_RANGE
    span = time_max - time_min
    return get_synthetic_model_preset(
        settings.analysis_model,
        time_min=time_min,
        time_max=time_max,
        transition_fraction=(POC_IN_VIVO_TRANSITION_TIME - time_min) / span,
        width_fraction=POC_IN_VIVO_TRANSITION_WIDTH / span,
    )


def _ivivc_in_vitro_preset(settings: ProofOfConceptSettings):
    """Return the time-compressed in vitro counterpart of the common in vivo truth."""
    time_min, time_max = IVIVC_IN_VITRO_TIME_RANGE
    in_vivo_min, in_vivo_max = POC_IN_VIVO_TIME_RANGE
    in_vivo_span = in_vivo_max - in_vivo_min
    return get_synthetic_model_preset(
        settings.analysis_model,
        time_min=time_min,
        time_max=time_max,
        transition_fraction=(POC_IN_VIVO_TRANSITION_TIME - in_vivo_min) / in_vivo_span,
        width_fraction=POC_IN_VIVO_TRANSITION_WIDTH / in_vivo_span,
    )


def build_measured_in_vivo_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Fit replicate-level measured in vivo data and plan a new full study."""
    study_min_time, study_max_time = MEASURED_TIME_RANGE
    future_replicates = POC_FUTURE_REPLICATES_PER_TIME

    rng = np.random.default_rng(settings.random_seed + 101)
    distinct_times = np.array([0.0, 4.0, 10.0, 16.0, 22.0, 28.0, 36.0, 48.0])
    if np.any(distinct_times < study_min_time) or np.any(distinct_times > study_max_time):
        raise ValueError('Measured reference times must lie within MEASURED_TIME_RANGE.')

    preset = _common_in_vivo_preset(settings)
    true_means = preset.evaluate(distinct_times)
    replicate_count = MEASURED_REFERENCE_REPLICATES_PER_TIME
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
    reference_grid = _reference_grid(
        study_min_time,
        study_max_time,
        target_step=MEASURED_REFERENCE_GRID_STEP,
    )
    reference = reference_from_profile_fit(
        fit,
        reference_grid,
        n_samples=settings.n_reference_draws,
        random_state=settings.random_seed + 103,
    )
    ground_truth = TimeCourse(
        reference_grid,
        preset.evaluate(reference_grid),
        label='synthetic_ground_truth',
    )

    reference_lower, reference_upper = reference.supported_time_range
    candidates = _candidate_grid(
        reference_lower,
        reference_upper,
        step=MEASURED_CANDIDATE_STEP,
        include_zero=True,
    )
    minimum_n = POC_PROSPECTIVE_MINIMUM_N
    maximum_n = _scenario_max_n(minimum_n, POC_DEFAULT_MAXIMUM_N, settings)
    constraints = DesignConstraints(
        candidate_times=candidates,
        fixed_times=np.array([study_min_time]),
        min_time=float(reference_lower),
        max_time=float(reference_upper),
        minimum_spacing=POC_MINIMUM_SPACING,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=future_replicates,
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
            replicates_per_time=future_replicates,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        target_response=0.5,
        ground_truth=ground_truth,
        reference_model_initial_parameters=fit.parameters,
        metadata={
            'reference_source': 'measured_in_vivo_replicates',
            'reference_model': settings.analysis_model,
            'reference_data_time_points': distinct_times.tolist(),
            'reference_replicates_per_time': replicate_count,
            'reference_data_sd': settings.observation_sd,
            'future_observation_sd': settings.observation_sd,
            'future_replicates_per_time': future_replicates,
            'study_time_range': MEASURED_TIME_RANGE,
            'candidate_time_step': MEASURED_CANDIDATE_STEP,
            'minimum_spacing': POC_MINIMUM_SPACING,
            'generating_model': preset.model_name,
            'generating_parameters': preset.parameter_mapping,
        },
    )



def _ivivc_preprocessed_pairs(prepared) -> pd.DataFrame:
    """Return the response-aligned pairs actually supplied to Approach 1."""
    response = np.asarray(prepared.response_grid, dtype=float)
    in_vitro_time = np.asarray(prepared.in_vitro_time_at_response, dtype=float)
    in_vivo_time = np.asarray(prepared.in_vivo_time_at_response, dtype=float)
    used = np.isfinite(in_vitro_time) & np.isfinite(in_vivo_time)
    return pd.DataFrame({
        'response_level': response,
        'in_vitro_time_at_response': in_vitro_time,
        'in_vivo_time_at_response': in_vivo_time,
        'used_for_approach1_fit': used,
    })


def build_ivivc_prediction_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Build an Approach 1 prediction from noisy fixed-time calibration profiles."""
    in_vitro_min_time, in_vitro_max_time = IVIVC_IN_VITRO_TIME_RANGE
    future_replicates = POC_FUTURE_REPLICATES_PER_TIME

    in_vitro_time = np.asarray(IVIVC_CALIBRATION_IN_VITRO_TIMES, dtype=float)
    in_vivo_time = np.asarray(IVIVC_CALIBRATION_IN_VIVO_TIMES, dtype=float)
    if np.any(in_vitro_time < in_vitro_min_time) or np.any(in_vitro_time > in_vitro_max_time):
        raise ValueError('IVIVC in vitro calibration times must lie within IVIVC_IN_VITRO_TIME_RANGE.')
    if np.any(in_vivo_time < 0.0):
        raise ValueError('IVIVC in vivo calibration times must be nonnegative.')

    preset = _ivivc_in_vitro_preset(settings)
    in_vivo_preset = _common_in_vivo_preset(settings)
    time_scale = IVIVC_TRUE_TIME_SCALE
    in_vitro_response_true = preset.evaluate(in_vitro_time)
    in_vivo_response_true = preset.evaluate_time_scaled(in_vivo_time, time_scale)
    in_vitro_response, in_vitro_noise, in_vitro_noise_attempts = (
        generate_noisy_monotonic_responses(
            in_vitro_response_true,
            absolute_sd=settings.ivivc_calibration_response_sd,
            random_state=settings.random_seed + 200,
            minimum_response=0.0,
        )
    )
    in_vivo_response, in_vivo_noise, in_vivo_noise_attempts = (
        generate_noisy_monotonic_responses(
            in_vivo_response_true,
            absolute_sd=settings.ivivc_calibration_response_sd,
            random_state=settings.random_seed + 201,
            minimum_response=0.0,
        )
    )

    prepared = prepare_data(
        in_vitro_time,
        in_vitro_response,
        in_vivo_time,
        in_vivo_response,
        selected_interpolation=[IVIVC_INTERPOLATION_METHOD],
    )
    preprocessed_data = _ivivc_preprocessed_pairs(prepared)
    if not np.any(preprocessed_data['used_for_approach1_fit']):
        raise RuntimeError('IVIVC preprocessing produced no overlapping response-aligned pairs.')

    with _quiet_expected_numerical_warnings():
        analysis = fit_ivivc(
            prepared,
            selections='approach1:linear_zero_intercept',
            selected_metrics=['rmse'],
            analysis_config=_analysis_config(settings.random_seed + 202),
        )
    if analysis.errors:
        raise RuntimeError(
            'IVIVC example fit failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in analysis.errors.items())
        )
    mapping_fit = analysis['approach1:linear_zero_intercept'].time_mapping_fit
    if mapping_fit is None:  # pragma: no cover - Approach 1 invariant
        raise RuntimeError('Approach 1 did not return a time-mapping fit.')

    prediction_time = _reference_grid(
        in_vitro_min_time,
        in_vitro_max_time,
        target_step=IVIVC_PREDICTION_GRID_STEP,
    )
    prediction_response = preset.evaluate(prediction_time)
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
        random_state=settings.random_seed + 203,
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
    ground_truth = TimeCourse(
        reference.time_grid,
        in_vivo_preset.evaluate(reference.time_grid),
        label='synthetic_ground_truth',
    )

    lower, upper = reference.supported_time_range
    candidates = _candidate_grid(
        lower,
        upper,
        step=IVIVC_CANDIDATE_STEP,
        include_zero=True,
    )
    fixed = np.array([0.0]) if np.any(np.isclose(candidates, 0.0)) else np.array([candidates[0]])
    minimum_n = POC_PROSPECTIVE_MINIMUM_N
    maximum_n = _scenario_max_n(minimum_n, POC_DEFAULT_MAXIMUM_N, settings)
    constraints = DesignConstraints(
        candidate_times=candidates,
        fixed_times=fixed,
        min_time=float(candidates[0]),
        max_time=float(candidates[-1]),
        minimum_spacing=POC_MINIMUM_SPACING,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=future_replicates,
    )
    calibration_data = pd.DataFrame({
        'in_vitro_time': in_vitro_time,
        'in_vitro_response': in_vitro_response,
        'in_vitro_response_true': in_vitro_response_true,
        'in_vitro_response_noise': in_vitro_noise,
        'in_vivo_time': in_vivo_time,
        'in_vivo_response': in_vivo_response,
        'in_vivo_response_true': in_vivo_response_true,
        'in_vivo_response_noise': in_vivo_noise,
        'role': 'IVIVC calibration data',
    })
    prediction_data = pd.DataFrame({
        'in_vitro_time': prediction_time,
        'in_vitro_response': prediction_response,
        'in_vitro_response_true': prediction_response,
        'in_vitro_response_noise': np.nan,
        'in_vivo_time': np.nan,
        'in_vivo_response': np.nan,
        'in_vivo_response_true': np.nan,
        'in_vivo_response_noise': np.nan,
        'role': 'target in vitro prediction data',
    })
    observed_data = pd.concat(
        [calibration_data, prediction_data],
        ignore_index=True,
        sort=False,
    )
    return ScenarioDefinition(
        name='ivivc_prediction',
        title='IVIVC-predicted in vivo reference trajectory',
        description=(
            'Noisy in vitro and in vivo calibration response profiles are observed at '
            'fixed study times and passed through the same response-aligned interpolation '
            'pipeline used by the IVIVC app. An Approach 1 time mapping is then fitted, '
            'and joint mapping-parameter draws are propagated across the target in vitro '
            'prediction profile.'
        ),
        reference=reference,
        observation_model=ObservationModel(
            kind='constant_sd',
            value=settings.observation_sd,
            replicates_per_time=future_replicates,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        preprocessed_data=preprocessed_data,
        target_response=0.5,
        ground_truth=ground_truth,
        reference_model_initial_parameters=in_vivo_preset.parameter_mapping,
        metadata={
            'reference_source': 'ivivc_approach1_prediction',
            'ivivc_selection': 'approach1:linear_zero_intercept',
            'ivivc_preprocessing_interpolation': IVIVC_INTERPOLATION_METHOD,
            'prediction_method': 'time_mapping',
            'in_vitro_prediction_points': len(prediction_time),
            'future_observation_sd': settings.observation_sd,
            'future_replicates_per_time': future_replicates,
            'ivivc_time_scale': time_scale,
            'ivivc_calibration_response_sd': settings.ivivc_calibration_response_sd,
            'ivivc_in_vitro_noise_generation_attempts': in_vitro_noise_attempts,
            'ivivc_in_vivo_noise_generation_attempts': in_vivo_noise_attempts,
            'ivivc_in_vitro_calibration_times': in_vitro_time.tolist(),
            'ivivc_in_vivo_calibration_times': in_vivo_time.tolist(),
            'ivivc_preprocessed_pair_count': int(
                np.sum(preprocessed_data['used_for_approach1_fit'])
            ),
            'ivivc_fitted_time_mapping_parameters': mapping_fit.parameters,
            'ivivc_time_mapping_rmse': mapping_fit.fit_metrics.get('rmse'),
            'candidate_time_step': IVIVC_CANDIDATE_STEP,
            'minimum_spacing': POC_MINIMUM_SPACING,
            'generating_model': in_vivo_preset.model_name,
            'generating_parameters': in_vivo_preset.parameter_mapping,
            'ivivc_in_vitro_generating_parameters': preset.parameter_mapping,
        },
    )


def _generate_completed_target_observations(
    ground_truth: TimeCourse,
    completed_times: np.ndarray,
    *,
    replicate_count: int,
    observation_sd: float,
    random_state: int,
) -> tuple[ExistingObservations, pd.DataFrame]:
    """Generate replicate-level completed observations from synthetic truth."""
    completed_times = np.asarray(completed_times, dtype=float)
    true_means = np.interp(
        completed_times,
        ground_truth.time,
        ground_truth.value,
    )
    rng = np.random.default_rng(random_state)
    observed = true_means[:, None] + rng.normal(
        0.0,
        observation_sd,
        size=(len(completed_times), int(replicate_count)),
    )
    observed = np.clip(observed, 0.002, None)
    raw_time = np.repeat(completed_times, int(replicate_count))
    raw_response = observed.reshape(-1)
    existing = ExistingObservations(
        raw_time,
        raw_response,
        label='completed target-study in vivo data',
    )
    table = pd.DataFrame({
        'time': raw_time,
        'response': raw_response,
        'replicate_id': np.tile(
            np.arange(1, int(replicate_count) + 1),
            len(completed_times),
        ),
        'role': 'completed target-study in vivo data',
    })
    return existing, table


def build_adaptive_in_vivo_only_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Use incomplete target in vivo data alone to select future times."""
    study_min_time, study_max_time = ADAPTIVE_IN_VIVO_TIME_RANGE
    completed_times = np.asarray(ADAPTIVE_COMPLETED_TIMES, dtype=float)
    replicate_count = POC_FUTURE_REPLICATES_PER_TIME
    if np.any(completed_times < study_min_time) or np.any(completed_times > study_max_time):
        raise ValueError('Adaptive completed times must lie within ADAPTIVE_IN_VIVO_TIME_RANGE.')

    preset = _common_in_vivo_preset(settings)
    ground_truth_time = _reference_grid(
        study_min_time,
        study_max_time,
        target_step=ADAPTIVE_REFERENCE_GRID_STEP,
    )
    ground_truth = TimeCourse(
        ground_truth_time,
        preset.evaluate(ground_truth_time),
        label='synthetic_ground_truth',
    )
    existing, observed_data = _generate_completed_target_observations(
        ground_truth,
        completed_times,
        replicate_count=replicate_count,
        observation_sd=settings.observation_sd,
        random_state=settings.random_seed + 301,
    )

    with _quiet_expected_numerical_warnings():
        fit_batch = fit_time_course(
            existing.time,
            existing.response,
            model_names=settings.analysis_model,
            selected_metrics=['rmse'],
            analysis_config=_analysis_config(settings.random_seed + 302),
            approach_id='approach2',
        )
    if fit_batch.errors:
        raise RuntimeError(
            'Adaptive in vivo-only example fit failed: '
            + '; '.join(f'{key}: {error.message}' for key, error in fit_batch.errors.items())
        )
    fit = fit_batch[settings.analysis_model]
    reference = reference_from_profile_fit(
        fit,
        ground_truth.time,
        n_samples=settings.n_reference_draws,
        random_state=settings.random_seed + 303,
    )

    candidates = _adaptive_candidate_grid(
        reference.supported_time_range[0],
        reference.supported_time_range[1],
        completed_times=existing.unique_times,
        step=ADAPTIVE_IN_VIVO_CANDIDATE_STEP,
        policy=settings.adaptive_candidate_policy,
    )
    minimum_n = len(completed_times) + 1
    maximum_n = _scenario_max_n(minimum_n, POC_DEFAULT_MAXIMUM_N, settings)
    constraints = DesignConstraints(
        candidate_times=candidates,
        fixed_times=completed_times,
        min_time=float(reference.supported_time_range[0]),
        max_time=float(reference.supported_time_range[1]),
        minimum_spacing=POC_MINIMUM_SPACING,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=replicate_count,
    )
    return ScenarioDefinition(
        name='adaptive_in_vivo_only',
        title='Adaptive refinement from incomplete in vivo data only',
        description=(
            'Completed target-study observations are retained exactly. A provisional '
            'full trajectory is fitted from the incomplete in vivo data alone. '
            + _adaptive_candidate_description(settings.adaptive_candidate_policy)
        ),
        reference=reference,
        observation_model=ObservationModel(
            kind='constant_sd',
            value=settings.observation_sd,
            replicates_per_time=replicate_count,
            source='proof_of_concept_assumption',
        ),
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        target_response=0.5,
        ground_truth=ground_truth,
        existing_observations=existing,
        reference_model_initial_parameters=fit.parameters,
        metadata={
            'reference_source': 'partial_target_in_vivo_study',
            'adaptive_mode': 'in_vivo_only',
            'adaptive_candidate_policy': settings.adaptive_candidate_policy,
            'reference_model': settings.analysis_model,
            'completed_times': completed_times.tolist(),
            'latest_completed_time': existing.latest_time,
            'reference_replicates_per_time': replicate_count,
            'reference_data_sd': settings.observation_sd,
            'future_observation_sd': settings.observation_sd,
            'future_replicates_per_time': replicate_count,
            'study_time_range': ADAPTIVE_IN_VIVO_TIME_RANGE,
            'candidate_time_step': ADAPTIVE_IN_VIVO_CANDIDATE_STEP,
            'minimum_spacing': POC_MINIMUM_SPACING,
            'generating_model': preset.model_name,
            'generating_parameters': preset.parameter_mapping,
            'note': (
                'The fitted reference beyond the latest completed time is '
                'model-based extrapolation from the incomplete in vivo study.'
            ),
        },
    )



def build_adaptive_refinement_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Backward-compatible alias for the in vivo-only adaptive scenario."""
    return build_adaptive_in_vivo_only_scenario(settings)


def _build_adaptive_ivivc_scenario(
    settings: ProofOfConceptSettings,
    *,
    scenario_name: str,
    title: str,
    description: str,
    completed_times: Sequence[float],
    candidate_policy: str,
    candidate_policy_source: str,
    adaptive_mode: str,
    observation_seed_offset: int,
) -> ScenarioDefinition:
    """Build one IVIVC-informed adaptive scenario."""
    prior = build_ivivc_prediction_scenario(settings)
    if prior.ground_truth is None:  # pragma: no cover - synthetic scenario invariant
        raise RuntimeError('The synthetic IVIVC scenario did not provide ground truth.')

    completed_times = np.asarray(completed_times, dtype=float)
    replicate_count = POC_FUTURE_REPLICATES_PER_TIME
    if (
        np.any(completed_times < prior.reference.supported_time_range[0])
        or np.any(completed_times > prior.reference.supported_time_range[1])
    ):
        raise ValueError(
            'Adaptive completed times must lie within the IVIVC reference domain.'
        )
    existing, completed_table = _generate_completed_target_observations(
        prior.ground_truth,
        completed_times,
        replicate_count=replicate_count,
        observation_sd=settings.observation_sd,
        random_state=settings.random_seed + observation_seed_offset,
    )
    observation_model = ObservationModel(
        kind='constant_sd',
        value=settings.observation_sd,
        replicates_per_time=replicate_count,
        source='proof_of_concept_assumption',
    )
    update = reweight_reference_ensemble(
        prior.reference,
        existing,
        observation_model,
    )
    reference = update.reference

    candidates = _adaptive_candidate_grid(
        reference.supported_time_range[0],
        reference.supported_time_range[1],
        completed_times=existing.unique_times,
        step=ADAPTIVE_IVIVC_CANDIDATE_STEP,
        policy=candidate_policy,
    )
    minimum_n = len(completed_times) + 1
    maximum_n = _scenario_max_n(minimum_n, POC_DEFAULT_MAXIMUM_N, settings)
    constraints = DesignConstraints(
        candidate_times=candidates,
        fixed_times=completed_times,
        min_time=float(reference.supported_time_range[0]),
        max_time=float(reference.supported_time_range[1]),
        minimum_spacing=POC_MINIMUM_SPACING,
        min_number_of_times=minimum_n,
        max_number_of_times=maximum_n,
        replicates_per_time=replicate_count,
    )

    completed_ivivc_table = pd.DataFrame({
        'in_vitro_time': np.nan,
        'in_vitro_response': np.nan,
        'in_vivo_time': completed_table['time'],
        'in_vivo_response': completed_table['response'],
        'in_vitro_response_true': np.nan,
        'in_vitro_response_noise': np.nan,
        'in_vivo_response_true': np.nan,
        'in_vivo_response_noise': np.nan,
        'replicate_id': completed_table['replicate_id'],
        'role': completed_table['role'],
    })
    observed_data = pd.concat(
        [prior.observed_data, completed_ivivc_table],
        ignore_index=True,
        sort=False,
    )
    metadata = dict(prior.metadata)
    metadata.update({
        'reference_source': 'ivivc_prediction_updated_by_completed_target_in_vivo',
        'adaptive_mode': adaptive_mode,
        'adaptive_candidate_policy': candidate_policy,
        'adaptive_candidate_policy_source': candidate_policy_source,
        'completed_times': completed_times.tolist(),
        'latest_completed_time': existing.latest_time,
        'completed_replicates_per_time': replicate_count,
        'effective_sample_size': update.effective_sample_size,
        'effective_sample_size_fraction': update.metadata[
            'effective_sample_size_fraction'
        ],
        'prior_positive_weight_draws': update.metadata['positive_prior_draws'],
        'future_observation_sd': settings.observation_sd,
        'future_replicates_per_time': replicate_count,
        'candidate_time_step': ADAPTIVE_IVIVC_CANDIDATE_STEP,
        'minimum_spacing': POC_MINIMUM_SPACING,
        'note': (
            'The IVIVC-predicted target in vivo trajectory ensemble is reweighted '
            'by completed target-study in vivo observations. Completed observations '
            'remain fixed during future-study simulation.'
        ),
    })
    return ScenarioDefinition(
        name=scenario_name,
        title=title,
        description=(
            description + ' ' + _adaptive_candidate_description(candidate_policy)
        ),
        reference=reference,
        observation_model=observation_model,
        constraints=constraints,
        analysis_model=settings.analysis_model,
        observed_data=observed_data,
        preprocessed_data=prior.preprocessed_data,
        target_response=prior.target_response,
        ground_truth=prior.ground_truth,
        existing_observations=existing,
        reference_model_initial_parameters=prior.reference_model_initial_parameters,
        metadata=metadata,
    )


def build_adaptive_ivivc_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Update an IVIVC-predicted target trajectory with completed in vivo data."""
    return _build_adaptive_ivivc_scenario(
        settings,
        scenario_name='adaptive_ivivc_prediction',
        title='IVIVC-informed adaptive refinement',
        description=(
            'Paired calibration data and a target in vitro profile define a prior '
            'target in vivo trajectory ensemble. Completed target-study in vivo '
            'observations reweight that ensemble.'
        ),
        completed_times=ADAPTIVE_COMPLETED_TIMES,
        candidate_policy=settings.adaptive_candidate_policy,
        candidate_policy_source='proof_of_concept_settings',
        adaptive_mode='ivivc_informed',
        observation_seed_offset=401,
    )


def build_adaptive_gap_filling_scenario(
    settings: ProofOfConceptSettings,
) -> ScenarioDefinition:
    """Demonstrate selection inside a transition region missed by completed times."""
    scenario = _build_adaptive_ivivc_scenario(
        settings,
        scenario_name='adaptive_gap_filling',
        title='Adaptive gap filling after a missed transition',
        description=(
            'Completed observations at elapsed times 0, 10, and 40 bracket but do '
            'not sample the synthetic transition region. The IVIVC-informed '
            'reference is used to recommend an additional elapsed time inside '
            'that gap.'
        ),
        completed_times=GAP_FILLING_COMPLETED_TIMES,
        candidate_policy='any_unobserved',
        candidate_policy_source='fixed_gap_filling_demonstration',
        adaptive_mode='ivivc_informed_gap_filling_example',
        observation_seed_offset=501,
    )
    scenario.metadata.update({
        'example_purpose': 'demonstrate_earlier_elapsed_time_gap_filling',
        'missed_elapsed_time_interval': (10.0, 40.0),
    })
    return scenario



SCENARIO_BUILDERS: Mapping[
    str,
    Callable[[ProofOfConceptSettings], ScenarioDefinition],
] = {
    'measured_in_vivo': build_measured_in_vivo_scenario,
    'ivivc_prediction': build_ivivc_prediction_scenario,
    'adaptive_in_vivo_only': build_adaptive_in_vivo_only_scenario,
    'adaptive_ivivc_prediction': build_adaptive_ivivc_scenario,
    'adaptive_gap_filling': build_adaptive_gap_filling_scenario,
    # Backward-compatible runner alias retained for existing scripts/tests.
    'adaptive_refinement': build_adaptive_refinement_scenario,
}



def _baseline_designs(
    definition: ScenarioDefinition,
    number_of_times: int,
) -> dict[str, np.ndarray]:
    designs = {
        'equal_time': equally_spaced_design(definition.constraints, number_of_times),
        #'log_time': log_spaced_design(definition.constraints, number_of_times),
    }
#    try:
#        designs['equal_response'] = equal_response_spaced_design(
#            definition.reference,
#            definition.constraints,
#            number_of_times,
#        )
#    except ValueError:
#        # Equal-response spacing is intentionally unavailable for nonmonotonic
#        # or effectively constant central reference curves.
#        pass
    return designs




def _design_label(design_code: str) -> str:
    """Return a concise user-facing label for one machine-readable design code."""
    return DESIGN_LABELS.get(str(design_code), str(design_code).replace('_', ' '))


def _selection_rule_label(rule: str | None) -> str | None:
    if rule is None:
        return None
    if rule == 'i_optimality':
        return 'I-optimality'
    if rule == 'rank_pseudodeterminant_fallback':
        return 'rank / pseudo-determinant fallback'
    return str(rule).replace('_', ' ')


def _contains_time(values: Sequence[float], time: float) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(
        len(values)
        and np.any(np.isclose(values, float(time), rtol=0.0, atol=1e-12))
    )


def _split_schedule_times(
    definition: ScenarioDefinition,
    design_times: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Split a schedule into already observed and newly planned time points."""
    design_times = np.asarray(design_times, dtype=float)
    if definition.existing_observations is None:
        return np.array([], dtype=float), design_times.copy()

    existing = definition.existing_observations.unique_times
    completed_mask = np.array(
        [_contains_time(existing, value) for value in design_times],
        dtype=bool,
    )
    return design_times[completed_mask], design_times[~completed_mask]


def _adaptive_candidate_summary(
    definition: ScenarioDefinition,
) -> tuple[float, float]:
    """Return earliest overall and later-than-completed candidate times."""
    candidates = definition.constraints.available_candidates()
    if len(candidates) == 0:
        raise ValueError('No candidate time satisfies the current design constraints.')
    earliest = float(np.min(candidates))
    if definition.existing_observations is None:
        return earliest, np.nan
    later = candidates[
        candidates > definition.existing_observations.latest_time + 1e-12
    ]
    earliest_later = float(np.min(later)) if len(later) else np.nan
    return earliest, earliest_later


def _time_role(definition: ScenarioDefinition, time: float) -> str:
    """Describe how one schedule time is used in the prospective/adaptive workflow."""
    if definition.existing_observations is not None:
        if _contains_time(definition.existing_observations.unique_times, time):
            return 'completed observation'
        if time < definition.existing_observations.latest_time - 1e-12:
            return 'recommended earlier gap-filling time'
        return 'recommended future time'
    if _contains_time(definition.constraints.fixed_times, time):
        return 'required fixed time'
    return 'selected study time'


def _format_times(values: Sequence[float]) -> str:
    return '; '.join(f'{float(value):g}' for value in values)


def _fractional_remaining_times(
    times: Sequence[float],
    curves: np.ndarray,
    remaining_fraction: float,
) -> np.ndarray:
    """Return first times at a fraction of each curve's own initial response."""
    times = np.asarray(times, dtype=float)
    curves = np.asarray(curves, dtype=float)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.ndim != 2 or curves.shape[1] != len(times):
        raise ValueError('Endpoint curves must contain one response per time point.')
    remaining_fraction = float(remaining_fraction)
    if not 0 < remaining_fraction < 1:
        raise ValueError('remaining_fraction must lie strictly between zero and one.')

    output = np.full(curves.shape[0], np.nan, dtype=float)
    for index, curve in enumerate(curves):
        if not np.isfinite(curve[0]):
            continue
        threshold = remaining_fraction * float(curve[0])
        output[index] = threshold_crossing_time(
            times,
            curve,
            threshold,
            direction='decreasing',
        )[0]
    return output


def _simulation_endpoint_metrics(simulation_result) -> dict[str, float]:
    """Summarize Monte Carlo recovery of T50 and T90 without affecting design."""
    metrics: dict[str, float] = {}
    for code, _label, remaining_fraction in REPORTED_DEGRADATION_ENDPOINTS:
        true_times = _fractional_remaining_times(
            simulation_result.evaluation_times,
            simulation_result.true_curves,
            remaining_fraction,
        )
        fitted_times = _fractional_remaining_times(
            simulation_result.evaluation_times,
            simulation_result.fitted_curves,
            remaining_fraction,
        )
        true_reachable = np.isfinite(true_times)
        recovered = (
            true_reachable
            & simulation_result.fit_success
            & np.isfinite(fitted_times)
        )
        errors = fitted_times[recovered] - true_times[recovered]
        absolute_errors = np.abs(errors)

        metrics[f'{code}_true_reachable_rate'] = float(np.mean(true_reachable))
        metrics[f'{code}_recovery_rate'] = (
            float(np.count_nonzero(recovered) / np.count_nonzero(true_reachable))
            if np.any(true_reachable)
            else np.nan
        )
        metrics[f'median_absolute_{code}_error'] = (
            float(np.median(absolute_errors)) if len(absolute_errors) else np.nan
        )
    return metrics


def _information_lookup(
    optimization: DesignOptimizationResult,
) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for label, steps in (
        ('optimized_nested', optimization.nested_steps),
        ('optimized_refined', optimization.refined_steps),
    ):
        for step in steps:
            lookup[(step.number_of_times, label)] = {
                'criterion': step.score.criterion,
                'full_rank_fraction': step.score.full_rank_fraction,
                'rank': step.score.min_rank,
                'parameter_count': step.score.parameter_count,
                'log_pseudodeterminant': step.score.mean_log_pseudodeterminant,
                'selection_rule': step.metadata.get('selection_rule'),
            }
    return lookup



def _performance_tables(
    definition: ScenarioDefinition,
    optimization: DesignOptimizationResult,
    comparisons: Mapping[int, DesignSimulationComparison],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Monte Carlo-first performance tables with transparent schedule roles."""
    information = _information_lookup(optimization)
    rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for number_of_times in sorted(comparisons):
        comparison = comparisons[number_of_times]
        for design_code, summary in comparison.summaries.items():
            info = information.get((number_of_times, design_code))
            endpoint_metrics = _simulation_endpoint_metrics(
                comparison.results[design_code]
            )
            existing_times, new_times = _split_schedule_times(
                definition,
                summary.design_times,
            )
            complete_schedule = _format_times(summary.design_times)
            latest_completed_time = (
                np.nan
                if definition.existing_observations is None
                else definition.existing_observations.latest_time
            )
            earliest_candidate_time = float(np.min(definition.constraints.candidate_times))
            earliest_future_candidate_time = np.nan
            if definition.existing_observations is not None:
                (
                    earliest_candidate_time,
                    earliest_future_candidate_time,
                ) = _adaptive_candidate_summary(definition)
            rows.append({
                'scenario': definition.name,
                'design': design_code,
                'design_label': _design_label(design_code),
                'number_of_times': number_of_times,
                'number_of_existing_times': len(existing_times),
                'number_of_new_times': len(new_times),
                'existing_times': _format_times(existing_times),
                'new_times': _format_times(new_times),
                'latest_completed_time': latest_completed_time,
                'adaptive_candidate_policy': definition.metadata.get(
                    'adaptive_candidate_policy'
                ),
                'earliest_candidate_time': earliest_candidate_time,
                # Retained for compatibility; this is specifically the earliest
                # candidate later than the latest completed elapsed time.
                'earliest_future_candidate_time': earliest_future_candidate_time,
                'reference_effective_sample_size': definition.metadata.get(
                    'effective_sample_size',
                    np.nan,
                ),
                'reference_effective_sample_size_fraction': definition.metadata.get(
                    'effective_sample_size_fraction',
                    np.nan,
                ),
                'complete_schedule': complete_schedule,
                # Retain the original machine-readable column for backward compatibility.
                'times': complete_schedule,
                'replicates_per_time': definition.constraints.replicates_per_time,
                'total_experimental_units': definition.constraints.total_experimental_units(
                    number_of_times
                ),
                'new_experimental_units': (
                    len(new_times) * definition.constraints.replicates_per_time
                ),
                # Monte Carlo evaluation is the primary comparison.
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
                'median_absolute_t50_error': endpoint_metrics[
                    'median_absolute_t50_error'
                ],
                't50_recovery_rate': endpoint_metrics['t50_recovery_rate'],
                't50_true_reachable_rate': endpoint_metrics['t50_true_reachable_rate'],
                'median_absolute_t90_error': endpoint_metrics[
                    'median_absolute_t90_error'
                ],
                't90_recovery_rate': endpoint_metrics['t90_recovery_rate'],
                't90_true_reachable_rate': endpoint_metrics['t90_true_reachable_rate'],
                'parameter_bound_hit_rate': summary.parameter_bound_hit_rate,
                # Local information quantities are supporting schedule-generation diagnostics.
                'information_criterion': np.nan if info is None else info['criterion'],
                'full_rank_fraction': (
                    np.nan if info is None else info['full_rank_fraction']
                ),
                'minimum_information_rank': np.nan if info is None else info['rank'],
                'information_rank': np.nan if info is None else info['rank'],
                'information_parameter_count': (
                    np.nan if info is None else info['parameter_count']
                ),
                'information_selection_rule': (
                    None if info is None else info['selection_rule']
                ),
                'information_selection_rule_label': (
                    None if info is None else _selection_rule_label(info['selection_rule'])
                ),
                'mean_log_pseudodeterminant': (
                    np.nan if info is None else info['log_pseudodeterminant']
                ),
            })
            for order, time in enumerate(summary.design_times, start=1):
                role = _time_role(definition, float(time))
                is_existing = role == 'completed observation'
                time_rows.append({
                    'scenario': definition.name,
                    'design': design_code,
                    'design_label': _design_label(design_code),
                    'number_of_times': number_of_times,
                    'order': order,
                    'time': time,
                    'time_role': role,
                    'existing_observation': is_existing,
                    'new_study_time': not is_existing,
                    'fixed': bool(
                        _contains_time(definition.constraints.fixed_times, float(time))
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
    for design_code, index in performance.groupby('design', sort=False).groups.items():
        ordered = performance.loc[index].sort_values('number_of_times')
        previous = ordered['penalized_mean_normalized_irmse'].shift(1)
        improvement = previous - ordered['penalized_mean_normalized_irmse']
        performance.loc[
            ordered.index,
            'marginal_penalized_nirmse_improvement',
        ] = improvement
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
        initial_parameters=definition.reference_model_initial_parameters,
    )
    optimization = optimize_local_i_optimal_designs(
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
            existing_observations=definition.existing_observations,
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



def _reference_endpoint_table(definition: ScenarioDefinition) -> pd.DataFrame:
    """Summarize T50/T90 for ground truth, central reference, and reference draws."""
    reference = definition.reference
    valid = reference.valid_draws & (reference.draw_weights > 0)
    draws = reference.value_draws[valid]
    weights = reference.draw_weights[valid]
    weights = weights / np.sum(weights)

    rows: list[dict[str, Any]] = []
    for code, label, remaining_fraction in REPORTED_DEGRADATION_ENDPOINTS:
        central_time = _fractional_remaining_times(
            reference.time_grid,
            reference.central_curve,
            remaining_fraction,
        )[0]
        draw_times = _fractional_remaining_times(
            reference.time_grid,
            draws,
            remaining_fraction,
        )
        finite = np.isfinite(draw_times)
        reachable_weight = float(np.sum(weights[finite]))
        if np.any(finite):
            finite_times = draw_times[finite].reshape(-1, 1)
            finite_weights = weights[finite]
            median_time = _weighted_quantile(
                finite_times, finite_weights, 0.5
            )[0]
            lower_time = _weighted_quantile(
                finite_times, finite_weights, 0.025
            )[0]
            upper_time = _weighted_quantile(
                finite_times, finite_weights, 0.975
            )[0]
        else:
            median_time = lower_time = upper_time = np.nan

        ground_truth_time = np.nan
        if definition.ground_truth is not None:
            ground_truth_time = _fractional_remaining_times(
                definition.ground_truth.time,
                definition.ground_truth.value,
                remaining_fraction,
            )[0]

        rows.append({
            'endpoint': code.upper(),
            'definition': label,
            'degraded_fraction': 1.0 - remaining_fraction,
            'remaining_fraction': remaining_fraction,
            'ground_truth_time': ground_truth_time,
            'central_reference_time': central_time,
            'reference_median_time': median_time,
            'reference_lower_95_time': lower_time,
            'reference_upper_95_time': upper_time,
            'reference_reachable_weight': reachable_weight,
        })
    return pd.DataFrame(rows)


def _reference_table(
    reference: ReferenceTrajectoryEnsemble,
    ground_truth: TimeCourse | None = None,
) -> pd.DataFrame:
    valid = reference.valid_draws & (reference.draw_weights > 0)
    draws = reference.value_draws[valid]
    weights = reference.draw_weights[valid]
    weights = weights / np.sum(weights)
    data: dict[str, np.ndarray] = {
        'time': reference.time_grid,
        'central_response': reference.central_curve,
        'lower_95': _weighted_quantile(draws, weights, 0.025),
        'upper_95': _weighted_quantile(draws, weights, 0.975),
    }
    if ground_truth is not None:
        lower, upper = reference.supported_time_range
        if ground_truth.time[0] > lower or ground_truth.time[-1] < upper:
            raise ValueError(
                'The synthetic ground truth must span the full reference domain.'
            )
        data['ground_truth_response'] = np.interp(
            reference.time_grid,
            ground_truth.time,
            ground_truth.value,
        )
    return pd.DataFrame(data)



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
    adaptive_summary = None
    if definition.existing_observations is not None:
        earliest_candidate, earliest_future_candidate = _adaptive_candidate_summary(
            definition
        )
        adaptive_summary = {
            'completed_times': definition.existing_observations.unique_times,
            'latest_completed_time': definition.existing_observations.latest_time,
            'candidate_policy': definition.metadata.get(
                'adaptive_candidate_policy', 'future_only'
            ),
            'earliest_candidate_time': earliest_candidate,
            # Retained for compatibility with existing output consumers.
            'earliest_future_candidate_time': (
                earliest_future_candidate
                if np.isfinite(earliest_future_candidate)
                else None
            ),
        }
    return {
        'name': definition.name,
        'title': definition.title,
        'description': definition.description,
        'analysis_model': definition.analysis_model,
        'target_response': definition.target_response,
        'target_direction': definition.target_direction,
        'ground_truth': (
            None
            if definition.ground_truth is None
            else {
                'available': True,
                'label': definition.ground_truth.label,
                'time_range': (
                    float(definition.ground_truth.time[0]),
                    float(definition.ground_truth.time[-1]),
                ),
                'used_for_optimization': False,
            }
        ),
        'existing_observations': (
            None
            if definition.existing_observations is None
            else {
                'count': definition.existing_observations.n_observations,
                'time_point_count': definition.existing_observations.n_time_points,
                'times': definition.existing_observations.unique_times,
                'latest_time': definition.existing_observations.latest_time,
            }
        ),
        'adaptive_summary': adaptive_summary,
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
        'output_interpretation': {
            'primary_evaluation': 'monte_carlo_curve_recovery',
            'schedule_generator': 'local_i_optimality_at_central_reference',
            'reference_uncertainty_used_for_monte_carlo': True,
            'reference_uncertainty_used_for_local_schedule_generation': False,
            'reported_degradation_endpoints': {
                't50': '50% degraded / 50% of initial response remaining',
                't90': '90% degraded / 10% of initial response remaining',
            },
            'design_labels': DESIGN_LABELS,
            'optimized_design_explanations': DESIGN_EXPLANATIONS,
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

    definition = result.definition
    reference_table = _reference_table(
        definition.reference,
        definition.ground_truth,
    )
    endpoint_table = _reference_endpoint_table(definition)
    endpoint_display = endpoint_table[[
        'endpoint',
        'ground_truth_time',
        'central_reference_time',
        'reference_median_time',
        'reference_lower_95_time',
        'reference_upper_95_time',
        'reference_reachable_weight',
    ]].copy()
    endpoint_display = endpoint_display.rename(columns={
        'endpoint': 'Endpoint',
        'ground_truth_time': 'Ground truth',
        'central_reference_time': 'Central reference',
        'reference_median_time': 'Reference median',
        'reference_lower_95_time': 'Reference 2.5%',
        'reference_upper_95_time': 'Reference 97.5%',
        'reference_reachable_weight': 'Reference reaching endpoint',
    })
    for column in (
        'Ground truth',
        'Central reference',
        'Reference median',
        'Reference 2.5%',
        'Reference 97.5%',
    ):
        endpoint_display[column] = endpoint_display[column].map(
            lambda value: '' if not np.isfinite(value) else f'{value:.5g}'
        )
    endpoint_display['Reference reaching endpoint'] = endpoint_display[
        'Reference reaching endpoint'
    ].map(lambda value: '' if not np.isfinite(value) else f'{value:.1%}')

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
    if 'ground_truth_response' in reference_table:
        reference_figure.add_trace(go.Scatter(
            x=reference_table['time'],
            y=reference_table['ground_truth_response'],
            mode='lines',
            name='Synthetic ground truth — not used by optimizer',
            line={
                'color': 'rgba(40, 40, 40, 0.9)',
                'dash': 'dash',
                'width': 2,
            },
        ))
    reference_figure.add_trace(go.Scatter(
        x=reference_table['time'],
        y=reference_table['central_response'],
        mode='lines',
        name='Central reference trajectory',
        line=dict(color='rgb(27, 136, 37)'),
    ))

    observed = definition.observed_data
    if {'time', 'response'}.issubset(observed.columns):
        reference_figure.add_trace(go.Scatter(
            x=observed['time'],
            y=observed['response'],
            mode='markers',
            name=(
                'Completed target in vivo observations'
                if definition.existing_observations is not None
                else 'Observed in vivo data'
            ),
        ))
    elif {'in_vivo_time', 'in_vivo_response'}.issubset(observed.columns):
        if 'role' in observed.columns:
            calibration = observed['role'] == 'IVIVC calibration data'
            completed = observed['role'] == 'completed target-study in vivo data'
        else:
            calibration = np.ones(len(observed), dtype=bool)
            completed = np.zeros(len(observed), dtype=bool)
        if np.any(calibration):
            reference_figure.add_trace(go.Scatter(
                x=observed.loc[calibration, 'in_vivo_time'],
                y=observed.loc[calibration, 'in_vivo_response'],
                mode='markers',
                name='IVIVC calibration in vivo data',
            ))
        if np.any(completed):
            reference_figure.add_trace(go.Scatter(
                x=observed.loc[completed, 'in_vivo_time'],
                y=observed.loc[completed, 'in_vivo_response'],
                mode='markers',
                name='Completed target in vivo observations',
            ))

    maximum_n = int(result.performance_table['number_of_times'].max())
    available_codes = set(
        result.performance_table.loc[
            result.performance_table['number_of_times'] == maximum_n,
            'design',
        ]
    )
    marker_sizes = {'optimized_refined': 18, 'optimized_nested': 14, 'equal_time': 10}
    marker_colors = {'optimized_refined': 'rgb(136, 136, 231)', 'optimized_nested': 'rgb(231, 136, 136)', 'equal_time': 'rgb(136, 136, 136)'}
    for design_code in ('optimized_refined', 'optimized_nested', 'equal_time'):
        if design_code not in available_codes:
            continue
        times = result.selected_times_table.loc[
            (result.selected_times_table['number_of_times'] == maximum_n)
            & (result.selected_times_table['design'] == design_code),
            'time',
        ].to_numpy(dtype=float)
        response = definition.reference.evaluate_central(times)
        reference_figure.add_trace(go.Scatter(
            x=times,
            y=response,
            mode='markers',
            marker={'size': marker_sizes[design_code], 'color':marker_colors[design_code]},
            name=f'{_design_label(design_code)} (N={maximum_n})',
        ))
    reference_figure.update_layout(
        title='Reference trajectory and representative schedules',
        xaxis_title='Time',
        yaxis_title='Response',
        template='plotly_white',
    )

    performance_figure = go.Figure()
    for design_code, group in result.performance_table.groupby('design', sort=False):
        group = group.sort_values('number_of_times')
        performance_figure.add_trace(go.Scatter(
            x=group['number_of_times'],
            y=group['penalized_mean_normalized_irmse'],
            mode='lines+markers',
            name=_design_label(design_code),
        ))
    count_axis_title = (
        'Total distinct time points (completed + future)'
        if definition.existing_observations is not None
        else 'Number of distinct time points'
    )
    performance_figure.update_layout(
        title='Monte Carlo curve-recovery error versus number of time points',
        xaxis_title=count_axis_title,
        yaxis_title='Penalized mean normalized IRMSE',
        template='plotly_white',
    )

    schedule_figure = go.Figure()
    schedule_data = result.selected_times_table[
        result.selected_times_table['design'].isin(
            ['optimized_nested', 'optimized_refined']
        )
    ]
    marker_sizes = {'optimized_nested': 14, 'optimized_refined': 10}
    for design_code, group in schedule_data.groupby('design', sort=False):
        custom = np.column_stack([
            group['time_role'].astype(str).to_numpy(),
            group['fixed'].astype(str).to_numpy(),
        ])
        schedule_figure.add_trace(go.Scatter(
            x=group['time'],
            y=group['number_of_times'],
            mode='markers',
            marker={'size': marker_sizes[design_code], 'symbol': 'circle'},
            name=_design_label(design_code),
            customdata=custom,
            hovertemplate=(
                'time=%{x:g}<br>N=%{y}<br>role=%{customdata[0]}'
                '<br>fixed=%{customdata[1]}<extra></extra>'
            ),
        ))
    schedule_figure.update_layout(
        title='Optimized schedules across design sizes',
        xaxis_title='Time',
        yaxis_title=count_axis_title,
        template='plotly_white',
    )

    table_columns = ['design_label', 'number_of_times']
    if definition.existing_observations is not None:
        table_columns.extend(['number_of_new_times', 'new_times'])
    table_columns.extend([
        'complete_schedule',
        'failure_rate',
        'median_normalized_irmse',
        'p90_normalized_irmse',
        'penalized_mean_normalized_irmse',
        'median_absolute_t50_error',
        't50_recovery_rate',
        'median_absolute_t90_error',
        't90_recovery_rate',
    ])
    table = result.performance_table[table_columns].copy()
    table = table.rename(columns={
        'design_label': 'Design',
        'number_of_times': 'Total N',
        'number_of_new_times': 'New N',
        'new_times': 'New times',
        'complete_schedule': 'Complete schedule',
        'failure_rate': 'Fit failure rate',
        'median_normalized_irmse': 'Median normalized IRMSE',
        'p90_normalized_irmse': 'P90 normalized IRMSE',
        'penalized_mean_normalized_irmse': 'Penalized mean normalized IRMSE',
        'median_absolute_t50_error': 'Median absolute T50 error',
        't50_recovery_rate': 'T50 recovery rate',
        'median_absolute_t90_error': 'Median absolute T90 error',
        't90_recovery_rate': 'T90 recovery rate',
    })
    for column in ('T50 recovery rate', 'T90 recovery rate'):
        table[column] = table[column].map(
            lambda value: '' if not np.isfinite(value) else f'{value:.1%}'
        )
    for column in table.select_dtypes(include=[np.number]).columns:
        table[column] = table[column].map(
            lambda value: '' if not np.isfinite(value) else f'{value:.5g}'
        )

    diagnostics = result.performance_table[
        result.performance_table['design'].isin(['optimized_nested', 'optimized_refined'])
    ][[
        'design_label',
        'number_of_times',
        'information_selection_rule_label',
        'information_rank',
        'information_parameter_count',
        'information_criterion',
    ]].copy()
    diagnostics = diagnostics.rename(columns={
        'design_label': 'Design',
        'number_of_times': 'Total N',
        'information_selection_rule_label': 'Selection rule',
        'information_rank': 'Information rank',
        'information_parameter_count': 'Parameters',
        'information_criterion': 'Local I criterion',
    })
    diagnostics['Local I criterion'] = diagnostics['Local I criterion'].map(
        lambda value: (
            'not finite (rank deficient)'
            if not np.isfinite(value)
            else f'{float(value):.5g}'
        )
    )
    for column in ('Information rank', 'Parameters'):
        diagnostics[column] = diagnostics[column].map(
            lambda value: '' if not np.isfinite(value) else f'{int(value)}'
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

    scenario_metadata = definition.metadata
    generating_model = scenario_metadata.get('generating_model')
    if generating_model == definition.analysis_model:
        model_summary = (
            '<strong>Synthetic generating / analysis model:</strong> '
            f'<code>{definition.analysis_model}</code>'
        )
    else:
        model_summary = (
            f'<strong>Analysis model:</strong> <code>{definition.analysis_model}</code>'
        )
    reference_source = scenario_metadata.get('reference_source', 'unspecified')
    extra_summary: list[str] = []
    if 'ivivc_calibration_response_sd' in scenario_metadata:
        extra_summary.append(
            '<strong>IVIVC calibration response SD:</strong> '
            f'{float(scenario_metadata["ivivc_calibration_response_sd"]):g}'
        )
    if 'ivivc_preprocessing_interpolation' in scenario_metadata:
        extra_summary.append(
            '<strong>IVIVC preprocessing:</strong> response-aligned '
            f'<code>{scenario_metadata["ivivc_preprocessing_interpolation"]}</code> interpolation; '
            f'<strong>usable time pairs:</strong> {int(scenario_metadata["ivivc_preprocessed_pair_count"])}'
        )
    if definition.existing_observations is not None:
        completed = _format_times(definition.existing_observations.unique_times)
        earliest_candidate, earliest_future = _adaptive_candidate_summary(definition)
        candidate_policy = definition.metadata.get(
            'adaptive_candidate_policy', 'future_only'
        )
        timing_summary = (
            f'<strong>earliest eligible elapsed time:</strong> {earliest_candidate:g}'
        )
        if np.isfinite(earliest_future):
            timing_summary += (
                '; <strong>earliest eligible time after completion:</strong> '
                f'{earliest_future:g}'
            )
        extra_summary.append(
            '<strong>Completed times:</strong> '
            f'{completed}; <strong>candidate policy:</strong> '
            f'<code>{candidate_policy}</code>; {timing_summary}'
        )
    if 'effective_sample_size' in scenario_metadata:
        ess = float(scenario_metadata['effective_sample_size'])
        ess_fraction = float(scenario_metadata.get('effective_sample_size_fraction', np.nan))
        if np.isfinite(ess_fraction):
            extra_summary.append(
                '<strong>IVIVC trajectory effective sample size after update:</strong> '
                f'{ess:.2f} ({ess_fraction:.1%} of prior draws)'
            )
        else:
            extra_summary.append(
                '<strong>IVIVC trajectory effective sample size after update:</strong> '
                f'{ess:.2f}'
            )
    extra_summary_html = ''.join(f'<p>{item}.</p>' for item in extra_summary)

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{definition.title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
    th {{ background: #f3f3f3; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>{definition.title}</h1>
  <p>{definition.description}</p>
  <p>{model_summary};
     <strong>reference source:</strong> <code>{reference_source}</code>;
     <strong>reference draws:</strong> {definition.reference.n_valid_draws};
     <strong>future observation model:</strong> <code>{definition.observation_model.kind}={definition.observation_model.value:g}</code>;
     <strong>future replicates/time:</strong> {definition.constraints.replicates_per_time}.</p>
  {extra_summary_html}
  <p><strong>Schedule generation:</strong> local I-optimality is evaluated at the fitted central reference trajectory. Reference-trajectory uncertainty is used in the Monte Carlo evaluation rather than averaged into the schedule generator.</p>
  <p><strong>{DESIGN_LABELS['optimized_nested']}:</strong> {DESIGN_EXPLANATIONS['optimized_nested']} <strong>{DESIGN_LABELS['optimized_refined']}:</strong> {DESIGN_EXPLANATIONS['optimized_refined']} Both schedules are information-based; Monte Carlo simulation evaluates their finite-sample curve-recovery performance.</p>
  <p><strong>Synthetic ground truth:</strong> shown only for validation and never used to select the schedules.</p>
  {first_figure}
  <h2>Degradation endpoints</h2>
  <p>T50 is the time to 50% degraded (50% of the curve's initial response remaining); T90 is the time to 90% degraded (10% remaining). These endpoints are reported for interpretation and Monte Carlo evaluation, not used by the I-optimal schedule generator.</p>
  {endpoint_display.to_html(index=False, escape=True)}
  {second_figure}
  {third_figure}
  <h2>Monte Carlo performance summary</h2>
  <p>Curve-recovery error and fit failure are the primary schedule-comparison outputs.</p>
  {table.to_html(index=False, escape=True)}
  <h2>Optimization diagnostics (supporting)</h2>
  <p>The local information calculation generates candidate schedules. Rank and the local I criterion are supporting diagnostics rather than the final performance assessment.</p>
  {diagnostics.to_html(index=False, escape=True)}
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
        if result.definition.preprocessed_data is not None:
            result.definition.preprocessed_data.to_csv(
                scenario_directory / 'ivivc_preprocessed_pairs.csv',
                index=False,
            )
        _reference_table(
            result.definition.reference,
            result.definition.ground_truth,
        ).to_csv(
            scenario_directory / 'reference_trajectory.csv',
            index=False,
        )
        _reference_endpoint_table(result.definition).to_csv(
            scenario_directory / 'reference_endpoints.csv',
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
            'ivivc_calibration_response_sd': run.settings.ivivc_calibration_response_sd,
            'adaptive_candidate_policy': run.settings.adaptive_candidate_policy,
        },
        'scenarios': list(run.scenarios),
        'output_interpretation': {
            'primary_evaluation': 'monte_carlo_curve_recovery',
            'schedule_generator': 'local_i_optimality_at_central_reference',
            'design_labels': DESIGN_LABELS,
        },
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
<p>Each scenario uses local I-optimality to generate candidate schedules from the central reference trajectory. Monte Carlo simulation across plausible trajectories and measurement noise is the primary performance comparison.</p>
<p><strong>{DESIGN_LABELS['optimized_nested']}:</strong> adds one time at a time while retaining earlier selections. <strong>{DESIGN_LABELS['optimized_refined']}:</strong> locally adjusts that schedule for each fixed N using the same I-optimal criterion. Both are information-based; neither is Monte Carlo optimized.</p>
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
        print(f'  Reported schedule: {_design_label(preferred)}')
        subset = result.performance_table[
            result.performance_table['design'] == preferred
        ].sort_values('number_of_times')
        for row in subset.itertuples(index=False):
            if result.definition.existing_observations is not None:
                schedule_text = (
                    f'new=[{row.new_times}]  full=[{row.complete_schedule}]'
                )
                count_text = (
                    f'N={row.number_of_times} '
                    f'({row.number_of_existing_times} completed + '
                    f'{row.number_of_new_times} new)'
                )
            else:
                schedule_text = f'[{row.complete_schedule}]'
                count_text = f'N={row.number_of_times}'
            print(
                f'  {count_text}: {schedule_text}  '
                f'MC penalized normalized IRMSE='
                f'{row.penalized_mean_normalized_irmse:.4g}, '
                f'failure rate={row.failure_rate:.1%}'
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run backend-only proof-of-concept degradation study-design examples.',
    )
    parser.add_argument(
        '--scenario',
        choices=('all',) + SCENARIO_NAMES + LEGACY_SCENARIO_NAMES,
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
        help=(
            'Skip the fixed-N coordinate-exchange refinement and retain only the '
            'add-one-at-a-time I-optimal sequence.'
        ),
    )
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='Write machine-readable files only.',
    )
    parser.add_argument(
        '--analysis-model',
        choices=supported_synthetic_models(),
        default='mass_loss_one_phase',
        help=(
            'Registered Approach 2 model used to analyze simulated in vivo studies '
            '(default: mass_loss_one_phase).'
        ),
    )
    parser.add_argument(
        '--ivivc-calibration-response-sd',
        type=float,
        default=0.05,
        help=(
            'Absolute SD applied independently to the synthetic in vitro and in vivo '
            'calibration response measurements (default: 0.05).'
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
        '--adaptive-candidate-policy',
        choices=('future-only', 'any-unobserved'),
        default='future-only',
        help=(
            'Eligible elapsed times for adaptive recommendations: only times after '
            'the latest completed time, or any unobserved time in the supported '
            'range (default: future-only).'
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
        ivivc_calibration_response_sd=args.ivivc_calibration_response_sd,
        adaptive_candidate_policy=args.adaptive_candidate_policy.replace('-', '_'),
    )
    scenario_names = SCENARIO_NAMES if args.scenario == 'all' else (args.scenario,)
    run = run_proof_of_concept(scenario_names, settings=settings)
    write_proof_of_concept_outputs(run, args.output_dir)
    _print_console_summary(run)
    print(f'\nOutputs written to: {Path(args.output_dir).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
