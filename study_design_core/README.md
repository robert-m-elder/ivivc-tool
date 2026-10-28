# Study-design numerical foundation

`study_design_core` is a presentation-independent layer for the planned
degradation study-design assessment module. It does not change the existing
IVIVC browser or CLI behavior.

## Common reference representation

All sources of expected in vivo behavior are converted to a
`ReferenceTrajectoryEnsemble`:

- one strictly increasing in vivo time grid;
- one complete response trajectory per plausible draw;
- validity flags and normalized draw weights;
- a central reference curve;
- the shared supported time range.

`reference_from_trajectory_ensemble()` converts the joint time/response draws
produced by `ivivc_core`. It sorts each draw, removes nonfinite pairs, handles
duplicate times according to an explicit policy, restricts the result to the
intersection of the valid draw domains, and does not extrapolate.

`reference_from_profile_fit()` supports a fitted measured in vivo profile, and
`reference_from_time_course()` supports a deterministic external trajectory.

## Future-study variability

`ObservationModel` is separate from uncertainty in the reference trajectory.
The initial supported forms are constant absolute SD and constant coefficient
of variation. `observation_model_from_residuals()` can create a clearly labeled
residual-based surrogate from an existing `ProfileFit`; it can include both
measurement variability and model lack of fit.

## User constraints

`DesignConstraints` records candidate times, already fixed times, time range,
minimum spacing, allowable numbers of distinct times, and fixed replicates per
time. These are user-defined numerical constraints, not standards or regulatory
requirements.

## Repeated fitting

`fit_model_once()` fits one registered model without cross-validation, metrics,
plots, or HTML. It is intended for Monte Carlo study simulations. A caller can
supply reference parameters directly; model-specific starts are used when
available, and generic grid search can be requested or used once as a fallback.

The later optimization and simulation modules should consume these objects
rather than legacy IVIVC dictionaries.

## Information-based time-point optimization

Patch 8 adds a prediction-oriented information layer. A reference trajectory
ensemble is first represented by the selected analysis model:

```python
from study_design_core import fit_reference_model_ensemble

reference_model = fit_reference_model_ensemble(
    reference,
    model_name='stretched_exponential',
)
```

Each successful fit produces one complete parameter draw with the source draw's
weight. The backend retains the uncertainty-averaged I-optimality functions
introduced for early method development. They average integrated fitted
mean-curve variance over the parameter-draw ensemble.

```python
from study_design_core import (
    DesignConstraints,
    ObservationModel,
    optimize_i_optimal_designs,
)

constraints = DesignConstraints(
    candidate_times=np.arange(0.0, 53.0),
    fixed_times=np.array([0.0]),
    min_number_of_times=4,
    max_number_of_times=8,
    minimum_spacing=2.0,
    replicates_per_time=4,
)
observation = ObservationModel(kind='constant_cv', value=0.10, minimum_sd=0.01)
result = optimize_i_optimal_designs(
    reference_model,
    observation,
    constraints,
    refine=True,
)
```

The proof-of-concept runner now uses the simpler
`optimize_local_i_optimal_designs()` path. It evaluates the information
calculation only at the parameter vector fitted to the central reference
trajectory. Before that information matrix is full rank, candidate additions
are selected by rank and then log pseudo-determinant. Once full rank is reached,
selection minimizes local I-optimality.

The runner presents two versions of this same information-based design.
**I-optimal (add-one-at-a-time)** greedily adds one time while retaining all
previously selected times; this is why the internal implementation calls the
sequence *nested*. **I-optimal (refined at fixed N)** starts from that schedule
and uses coordinate exchange to move nonfixed times for each design size.
Coordinate exchange is not Monte Carlo optimization; both schedules use the
same local I-optimal criterion. Monte Carlo simulation is applied afterward to
evaluate finite-sample curve recovery and fit failure.

Reference-trajectory uncertainty is deliberately not averaged into the
analytical optimizer in this default workflow. Instead it is retained for the
Monte Carlo recovery assessment below. The information calculation is therefore
a fast schedule generator, while Monte Carlo recovery error and fit failure are
the primary measures used to compare schedules. The ensemble optimizer remains
available for method-development experiments.

## Simulation-based design evaluation

Patch 9 adds Monte Carlo validation of proposed schedules. A
`SimulationPlan` selects plausible reference trajectories and stores
standard-normal errors for every available candidate time. Reusing one plan
across schedules applies common random numbers, so differences between designs
are less obscured by Monte Carlo noise.

```python
from study_design_core import (
    compare_designs_by_simulation,
    create_simulation_plan,
    equally_spaced_design,
)

plan = create_simulation_plan(
    reference,
    observation,
    constraints.all_available_times,
    n_simulations=1000,
    random_state=123,
)
comparison = compare_designs_by_simulation(
    reference,
    {
        'optimized': optimized_times,
        'equal_time': equally_spaced_design(constraints, len(optimized_times)),
    },
    observation,
    model_name='stretched_exponential',
    constraints=constraints,
    simulation_plan=plan,
)
```

For every simulated study, the engine samples future observations, fits the
selected analysis model with `fit_model_once()`, and compares the recovered
curve with the complete generating trajectory. By default, each fit uses only
the simulated dataset and the model's registered data-driven initial guess. A
common central-reference start is available when prior information is intended
to guide fitting. The generating draw's true parameters are available only
through an explicitly requested `oracle` diagnostic mode and are not used by
default. The primary metric is integrated
root mean square error (IRMSE). Additional outputs include normalized IRMSE,
maximum absolute error, approximate mean-curve interval width, optional
threshold-time error, parameter-bound hits, and fit-failure rate.

Fit failures remain explicit. Successful-study metrics exclude failed fits,
while `penalized_mean_irmse` assigns a configurable response-scale penalty to
failures for design ranking. The default failure penalty is twice the response
range represented by the simulated reference curves.

Simple comparison schedules are available for equal calendar-time spacing,
shifted-log spacing, and equal-response spacing. The equal-response baseline
requires a monotonic central reference trajectory.

## Adaptive study support

`ExistingObservations` stores replicate-level measurements that have already
been collected. Adaptive simulations keep these values fixed and simulate only
new measurements at proposed elapsed times. `reweight_reference_ensemble()` can
update the weights of complete IVIVC-predicted trajectories using the
likelihood of those completed observations; trajectory shapes are retained and
the effective sample size of the reweighted ensemble is reported.

The backend therefore supports two distinct adaptive use cases:

- **in vivo-only adaptive refinement:** fit a provisional reference from an
  incomplete target in vivo study and use it to select additional observations;
- **IVIVC-informed adaptive refinement:** predict a target in vivo ensemble from
  calibration IVIVC data and a target in vitro profile, reweight that ensemble
  with completed target in vivo observations, and select additional observations.

In both cases, completed times are fixed. The proof-of-concept candidate policy
can either restrict recommendations to elapsed times strictly later than the
latest completed observation (`future_only`, the default), or permit any
unobserved elapsed time in the supported range (`any_unobserved`). The latter
supports a new experiment or retained specimen that fills an earlier sampling
gap; it does not imply retrospective measurement of an unavailable specimen.

## Proof-of-concept runner

`examples/study_design_proof_of_concept.py` composes the refactored IVIVC core,
local information optimizer, and simulation evaluator into five backend-only
examples:

- measured in vivo data used as the reference;
- an IVIVC-derived predicted in vivo trajectory;
- adaptive refinement from incomplete in vivo data alone;
- IVIVC-informed adaptive refinement using completed target in vivo data;
- IVIVC-informed gap filling after completed times 0, 10, and 40 miss the
  synthetic transition region.

Run a small demonstration from the repository root with:

```bash
python examples/study_design_proof_of_concept.py --quick
```

The default output directory, `study_design_poc_output/`, contains transparent
CSV and JSON files for each scenario and optional static Plotly HTML summaries.

All built-in scenarios are synthetic validation examples. Their HTML summaries
and `reference_trajectory.csv` files include the known generating curve as
**Synthetic ground truth — not used by optimizer**. This curve is supplied only
for evaluation and visualization; it is not passed to reference fitting, IVIVC
calibration, trajectory reweighting, optimization, or Monte Carlo refitting.
Real-data workflows will not have this trace.
The examples use constant absolute future-study error by default (`SD = 0.05`).
The selected proof-of-concept model is used both to generate the synthetic
trajectory and to analyze the simulated sparse studies:

```bash
python examples/study_design_proof_of_concept.py \
    --analysis-model mass_loss_one_phase \
    --observation-sd 0.05 \
    --ivivc-calibration-response-sd 0.05 \
    --fit-initialization data_driven
```

To permit adaptive recommendations at unobserved elapsed times earlier than the
latest completed observation, run with:

```bash
python examples/study_design_proof_of_concept.py \
    --scenario adaptive_in_vivo_only \
    --adaptive-candidate-policy any-unobserved
```

Completed times remain fixed. An earlier recommendation represents an
additional experiment or a suitable retained specimen at that elapsed time.
The built-in `adaptive_gap_filling` scenario demonstrates this behavior with
the `any_unobserved` policy fixed for that example:

```bash
python examples/study_design_proof_of_concept.py \
    --quick \
    --scenario adaptive_gap_filling
```

The built-in demonstration currently defines explicit example parameter presets
for `mass_loss_one_phase`, `mass_loss_one_phase_normalized`, `exponential`, and
`mass_loss_surface_erosion`. A model without a preset is rejected rather than
silently replaced by a one-phase curve.
The Approach 1 IVIVC scenario samples the in vitro and in vivo calibration
profiles at separate fixed time grids, adds configurable absolute response noise to
both profiles, and passes the noisy measurements through the same default
response-aligned interpolation path used by the IVIVC app. Synthetic noise is
redrawn when it would make a calibration response profile nonmonotonic, because
the response-to-time interpolation assumes a monotonic degradation trajectory.
Optional `parameter_bounds` declared by a model (or its existing fit
bounds) are used by the study-design backend to constrain repeated fits and to
reject and resample implausible parameter draws before trajectory ensembles are
constructed. This does not change the existing IVIVC fitting path. The runner encodes
no regulatory minimums; all candidate grids, fixed times, replicates, and
design-size ranges are ordinary `DesignConstraints` supplied by the example
definitions.
