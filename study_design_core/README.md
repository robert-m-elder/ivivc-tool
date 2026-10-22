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
weight. The optimizer then uses finite-difference model sensitivities and the
future-study observation model to calculate local Fisher-information matrices.
The default criterion is uncertainty-averaged I-optimality: integrated fitted
mean-curve variance over a user-specified evaluation grid.

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

The greedy sequence is nested. Before a design is full rank, selection is based
on robust rank and pseudo-determinant diagnostics; after full rank is attained,
selection minimizes the uncertainty-averaged I criterion. Optional coordinate
exchange refines each design size without preserving nesting. These information
scores are screening and optimization quantities; later Monte Carlo validation
will provide user-facing expected recovery performance.

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

## Proof-of-concept runner

`examples/study_design_proof_of_concept.py` composes the refactored IVIVC core,
information optimizer, and simulation evaluator into three backend-only
examples:

- measured in vivo data used as the reference;
- an IVIVC-derived predicted in vivo trajectory;
- adaptive refinement with completed time points held fixed.

Run a small demonstration from the repository root with:

```bash
python examples/study_design_proof_of_concept.py --quick
```

The default output directory, `study_design_poc_output/`, contains transparent
CSV and JSON files for each scenario and optional static Plotly HTML summaries.
The examples use constant absolute future-study error by default (`SD = 0.05`)
and the unnormalized registered one-phase mass-loss model. Both are configurable:

```bash
python examples/study_design_proof_of_concept.py \
    --analysis-model mass_loss_one_phase \
    --observation-sd 0.05 \
    --fit-initialization data_driven
```

Any model registered for Approach 2 in the existing `models` directory can be
selected. Optional `parameter_bounds` declared by a model (or its existing fit bounds)
are used by the study-design backend to constrain repeated fits and to reject
and resample implausible parameter draws before trajectory ensembles are
constructed. This does not change the existing IVIVC fitting path. The runner encodes
no regulatory minimums; all candidate grids, fixed times, replicates, and
design-size ranges are ordinary `DesignConstraints` supplied by the example
definitions.
