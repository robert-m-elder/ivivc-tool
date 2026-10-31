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

## Selecting a fitted IVIVC reference

`reference_from_ivivc()` connects one fitted IVIVC selection and prediction
method to the same reference representation. It accepts an existing
`IVIVCAnalysis` from `ivivc_core.fit_ivivc()` and a target in vitro
`TimeCourse`; it does not refit the calibration data or choose a method.

```python
from ivivc_core import TimeCourse, fit_ivivc
from study_design_core import reference_from_ivivc

analysis = fit_ivivc(
    prepared_calibration_data,
    selections='approach2:exponential',
    selected_metrics=['rmse'],
)
target_in_vitro = TimeCourse(target_time, target_response)
reference = reference_from_ivivc(
    analysis,
    target_in_vitro,
    selection='approach2:exponential',
    method='tau_ratio',
    n_samples=200,
    random_state=123,
    required_times=[0.0],
)
```

Approach 2 methods retain their existing IVIVC meanings:

| Method | Reference prediction |
| --- | --- |
| `value_ratio` | Multiply the target responses by the fitted IVV/IV response ratio at the target times. |
| `tau_ratio` | Multiply the target times by the fitted IVV/IV time-constant ratio, retaining the target responses. |
| `observed_response_inverse` | Map the target responses to times on the fitted in vivo curve. |

Approach 1's `time_mapping` and the existing Approach 3 methods can also be
selected through this adapter. Only the requested fit is sampled, so adding
other fits to an analysis does not alter its seeded reference draws. A failed
selection or unavailable nominal prediction raises `ValueError`; another
method is never substituted. In particular, tau-ratio prediction needs finite,
positive integral time constants. An unsupported or nonconvergent tau does not
prevent use of another explicitly selected prediction method.

The target data must use the analysis's response normalization and compatible
time units. The adapter rejects applied `log_x` or `log_y` preprocessing because
the IVIVC APIs return predictions on those transformed axes; this adapter does
not convert them back to study time/response coordinates. Normalized response
data remain supported.

The shared supported domain is the intersection of usable draw domains.
Partial inverse-response predictions may therefore supply a shorter reference;
the nominal prediction must contain at least two distinct finite times.
An explicit `time_grid` must lie entirely inside that domain. `required_times`
checks mandatory baseline, fixed, or completed times without changing the
grid; unsupported times cause an error rather than moving or inventing a
baseline. Existing method-specific model extrapolation can occur while
generating predictions, but conversion adds no extrapolation. Source point
statuses, support masks, calibration-range/extrapolation diagnostics, and
preprocessing details are retained in reference metadata under `ivivc_*` keys.
The central reference uses the nominal trajectory when it covers the output
grid; otherwise the existing converter uses the weighted draw mean and records
`central_curve_source='weighted_draw_mean'`.
Numerical support does not establish transferability to a different system.

The IVIVC calibration model and study analysis model are independent. For
example, an exponential IVIVC can supply the reference for a study analyzed
using `stretched_exponential`:

```python
from study_design_core import fit_reference_model_ensemble

reference_model = fit_reference_model_ensemble(
    reference, model_name='stretched_exponential',
)
```

Use the resulting reference and reference model with the existing local
I-optimal optimization, timing requirements, and Monte Carlo simulation APIs.
For adaptive planning, reweight the original reference with
`reweight_reference_ensemble(reference, completed, observation_model)`, then
fit the analysis model to `update.reference`. For successive updates, supply
the cumulative completed observations and the original prior, avoiding repeated
use of earlier observations in the likelihood. Keep those same completed
observations fixed in subsequent simulation and schedule constraints.
Calibration observations are reference inputs, not completed target-study data.

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

The five original proof-of-concept scenarios use the simpler
`optimize_local_i_optimal_designs()` path. It evaluates the information
calculation only at the parameter vector fitted to the central reference
trajectory. Before that information matrix is full rank, candidate additions
are selected by rank and then log pseudo-determinant. Once full rank is reached,
selection minimizes local I-optimality.

Those scenarios present two versions of this same information-based design.
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

## Crossing-time deadlines for one additional observation

The timing functions use complete reference trajectories to calculate an early
deadline for a selected amount of degradation. The two inputs have separate
meanings: `degraded_fraction=0.50` selects time to 50% degradation, while
`quantile=0.10` selects the weighted 10th percentile of those crossing times.
These are crossing-time quantiles, rather than percentiles of an information
criterion.

```python
from study_design_core import (
    reference_crossing_time_summary,
    latest_time_before_deadline,
    optimize_next_time_before_deadline,
)

timing = reference_crossing_time_summary(
    reference,
    degraded_fraction=0.50,
    quantile=0.10,
    baseline_time=0.0,
)
direct = latest_time_before_deadline(
    timing,
    constraints,
    latest_completed_time=existing.latest_time,
)
constrained = optimize_next_time_before_deadline(
    timing,
    reference_model,
    observation,
    constraints,
    latest_completed_time=existing.latest_time,
)
if direct.status == 'available':
    print(direct.new_time, direct.times)
```

The baseline time must be inside the supported reference domain. Each draw's
positive response at that time defines its own relative threshold; only
crossings at or after that baseline are considered. Crossings are linearly
interpolated between reference evaluations. The deadline is the smallest
crossing time whose cumulative normalized draw weight reaches the requested
quantile, without interpolation between different draws.

Valid trajectories that do not cross within the supported horizon retain their
weights and have crossing time `inf`. Original invalid and zero-weight draws
are excluded. A quantile beyond the horizon has status `beyond_horizon` and no
finite deadline. An unsuitable positive-weight trajectory, such as one with a
nonpositive baseline response, gives status `invalid_reference`; its weight is
not silently discarded.

Both selectors plan exactly one new time after `latest_completed_time`, which
must be at or after the baseline. All fixed times must represent completed
observations, and the original constraints must permit
`len(fixed_times) + 1` distinct times. The direct rule
chooses the latest feasible candidate at or before the deadline. The constrained
rule uses the existing central I-optimal optimizer over that same candidate
subset, retaining its rank fallback and the original evaluation horizon.
Completed times, spacing, observation variability, and replicates are preserved.

An elapsed deadline has status `deadline_elapsed`; a deadline without a legal
future candidate has status `no_feasible_candidate`. Unavailable policies return
no schedule, rather than moving the date past the deadline. With the reference,
grid, and central criterion held fixed, lowering the quantile can only keep or
move these one-point recommendations earlier, while they remain feasible.

The weights describe uncertainty in the supplied reference, rather than verified
population event rates. A quantile deadline is a planning rule; finite weighted
ties do not imply that exactly the requested fraction has crossed by that date.
The existing POC endpoint tables remain descriptive summaries conditional on
reaching the endpoint and are separate from these planning calculations.
The separate `adaptive_timing_comparison` POC scenario compares these timing
policies with unrestricted central I-optimality.

## Requirements for future schedule planning

`study_design_core.requirements` defines timing requirements independently of
the source of the reference and the selected analysis model. This first layer
resolves deadlines and checks requirement accounting. It does not select a
schedule, resolve multiple latest-before points, or establish that all
requirements can be satisfied together. The existing one-point policies and
POC examples retain their behavior.

- `CrossingTimeTarget` specifies a degraded fraction, crossing-time quantile,
  and reference baseline time. It uses `reference_crossing_time_summary()`
  with the original reference weights, including updated IVIVC weights.
- `SamplingWindow` requires a minimum number of distinct sampling times in
  `(lower_bound, upper_bound]`. Either bound can be a calendar time or a
  `CrossingTimeTarget`.
- `LatestBefore` records a request for a future point at or before a deadline,
  with an optional lower bound. This layer identifies eligible times; choosing
  the latest times jointly is a later scheduling step.
- `PlanningContext` identifies baseline and completed target-study times and
  the planning start. Baseline and completed times must be fixed in the
  supplied `DesignConstraints`. Other fixed times are future commitments.

Window counts exclude baseline. The default `count_scope='new'` also excludes
completed times but includes fixed future points. `count_scope='all'` includes
completed nonbaseline times when assessing coverage already achieved. Replicates
at the same time count once. Historical reference and calibration measurement
times never enter these counts automatically.

The planning start defaults to the latest completed time for adaptive studies,
or `constraints.min_time` for prospective studies. It can be specified
explicitly, but cannot precede completed observations. Newly planned times
must be strictly after the planning start. A prospective baseline can therefore
be fixed without being described as an observation already collected. This is
separate from the reference baseline used to define relative degradation.

For example, given a reference and constraints with completed times 0, 10, and
20 fixed, requirements can be resolved without choosing future points:

```python
from study_design_core import (
    CrossingTimeTarget,
    LatestBefore,
    PlanningContext,
    SamplingWindow,
    evaluate_timing_requirements,
    resolve_timing_requirements,
)

requirements = (
    SamplingWindow(
        lower_bound=20.0,
        upper_bound=CrossingTimeTarget(degraded_fraction=0.50, quantile=0.10),
        minimum_count=1,
        label='new point before early T50',
    ),
    LatestBefore(
        deadline=CrossingTimeTarget(degraded_fraction=0.90, quantile=0.50),
        label='point near median T90',
    ),
)
resolved = resolve_timing_requirements(
    requirements, reference, constraints,
    context=PlanningContext(
        baseline_times=[0.0],
        completed_times=[0.0, 10.0, 20.0],
    ),
)
print(resolved.status)
for requirement in resolved.requirements:
    print(requirement.label, requirement.status, requirement.eligible_times)

# Assess an independently supplied complete schedule; no optimization occurs.
assessment = evaluate_timing_requirements(proposed_times, resolved, constraints)
```

Repeated crossing targets are resolved once per request. Unavailable crossing
targets, reversed or empty windows, elapsed new-sampling deadlines, and windows
without enough candidates remain visible in the results. An elapsed window can
still be covered by completed observations under `count_scope='all'`.
Resolve requirements again whenever the reference or study constraints change.

Resolution status `ready` means the bounds and individual candidate checks are
usable. It does not prove that the requested budget, overlapping requirements,
and spacing permit a complete schedule. Candidate-count bounds are upper
bounds, rather than guarantees that those candidates can coexist. Actual
schedule evaluation validates ordinary design constraints and reports the
qualifying times and counts for each requirement. A latest-before evaluation
can establish `not_past_deadline`; it does not certify that the chosen point is
the latest feasible one or assign distinct points to multiple such requests.

## Optimizing schedules with window requirements

`optimize_window_constrained_designs()` selects complete schedules subject to
`SamplingWindow` requirements. It supports prospective planning and multiple
additional adaptive times, using the same central I-optimal criterion as the
existing optimizer. The original reference evaluation horizon, observation
model, integration weights, and rank fallback are retained. Windows constrain
sampling coverage; they do not truncate the prediction domain being scored.

Given a reference and its matching fitted `reference_model`, this example
retains three completed times and selects either two or three additional times:

```python
import numpy as np
from study_design_core import (
    DesignConstraints,
    PlanningContext,
    SamplingWindow,
    optimize_window_constrained_designs,
    resolve_timing_requirements,
)

constraints = DesignConstraints(
    candidate_times=np.arange(0.0, 53.0),
    fixed_times=np.array([0.0, 10.0, 20.0]),
    min_time=0.0,
    max_time=52.0,
    minimum_spacing=1.0,
    min_number_of_times=5,
    max_number_of_times=6,
    replicates_per_time=3,
)
resolved_windows = resolve_timing_requirements(
    [
        SamplingWindow(20.0, 30.0, label='early future coverage'),
        SamplingWindow(30.0, 45.0, label='later future coverage'),
    ],
    reference, constraints,
    context=PlanningContext(
        baseline_times=[0.0], completed_times=[0.0, 10.0, 20.0],
    ),
)
result = optimize_window_constrained_designs(
    reference_model, observation, constraints, resolved_windows,
    number_of_times=[5, 6],
    refine=True,
)
for design in result.designs:
    print(design.number_of_times, design.status, design.times)
    print(design.optimization_status, design.requirement_evaluation)
```

`number_of_times` is the total number of distinct target-study times, including
completed observations, baseline, and fixed future commitments. Pass an integer
for one final size, a sequence for selected sizes, or omit it for every allowed
size in `DesignConstraints`. Each size is solved independently; these results
are not a nested sequence. `result_for_n()` returns any outcome, while
`design_for_n()` requires a feasible schedule. The result retains the resolved
requirements, including individual unavailable-target diagnostics.

For prospective planning, supply the required planned baseline in `fixed_times`
and leave `completed_times` empty. Other fixed future times remain commitments
and count as new sampling. The same functions accept measured, IVIVC-predicted,
or adaptively updated references; construction of the reference remains
separate from scheduling.

`check_schedule_feasibility()` checks whether a complete schedule of a specified
size exists, without calculating information. Its discrete search enforces
fixed times, minimum spacing, future eligibility, and all window counts
together. Optional `required_times` test a provisional selection without
changing the original constraints. A feasible result includes a witness
schedule; an `infeasible` result establishes that no such schedule exists on
the supplied grid. Unresolved reference targets remain `unavailable`.
Ambiguous grid times within the existing `1e-12` matching tolerance are
rejected explicitly.

During optimization, greedy additions are accepted only when a feasible
completion remains possible. Coordinate exchange then improves the complete
schedule while preserving every window requirement. The information search
is heuristic and does not certify a global optimum. Feasible rank-deficient
schedules use the existing rank and log pseudo-determinant fallback.

`max_search_states` bounds each feasibility search. A search that stops before
finding or excluding a schedule reports `search_limit`, rather than
`infeasible`. If optimization already has a feasible schedule, it retains that
schedule and reports the incomplete improvement search separately through
`optimization_status`. An exchange-iteration limit is also reported separately
from schedule feasibility.

This window-only API continues to reject `LatestBefore` requests. Use
`optimize_timing_constrained_designs()` below for mixed requirements. The
existing one-point direct and deadline-constrained APIs remain available.

## Joint latest-before points and window coverage

`select_latest_before_times()` chooses distinct future points for any number
of `LatestBefore` requests. The selection preserves all window counts, fixed
times, spacing, and the exact final study size. A fixed future commitment can
fill a request, but baseline and completed observations cannot. A selected
point can also satisfy one or more coverage windows and is counted only once
toward the study size.

Later deadlines have priority, with request order breaking equal-deadline ties.
The selector chooses the latest time at the first priority that permits all
remaining requests and a complete schedule, then does the same at the next
priority. This is a deterministic ordering of preferences, rather than a
compromise between proximity to deadlines and information. For example, with
deadlines 24 and 25 and minimum spacing 4, the selected times can be 21 and 25.
Independently choosing 24 and 25 would violate spacing.

Use the general optimizer to mix these requests with informative sampling
within windows. Given the reference, fitted model, observation model, and
adaptive constraints from the preceding example:

```python
from study_design_core import LatestBefore, optimize_timing_constrained_designs

mixed = resolve_timing_requirements(
    [
        SamplingWindow(20.0, 30.0, label='early future coverage'),
        LatestBefore(24.0, label='earlier milestone'),
        LatestBefore(40.0, label='later milestone'),
    ],
    reference, constraints,
    context=PlanningContext(
        baseline_times=[0.0], completed_times=[0.0, 10.0, 20.0],
    ),
)
result = optimize_timing_constrained_designs(
    reference_model, observation, constraints, mixed,
    number_of_times=[5, 6],
)
for design in result.designs:
    print(design.number_of_times, design.status, design.times)
    for point in design.latest_before_selection.points:
        print(point.label, point.status, point.selected_time,
              point.deadline_slack, point.displacement)
```

Either bound can also be a `CrossingTimeTarget`. Targets are resolved once
using the original reference and weights. Locking selected points does not
recalculate targets or change their uncertainty quantiles. Prospective calls
use the same API with no completed observations. Reference construction and
adaptive updating remain separate from the choice of timing requirements.

The mixed optimizer first establishes the latest-before points, then locks
them while the central I-optimal search chooses the remaining times. It
retains the original information evaluation domain, noise model, and rank
fallback. Different final sizes are solved independently: a larger requested
study can force a deadline point earlier to leave room for other observations.
Window-only calls delegate to the existing window optimizer.

Each design's `latest_before_selection` retains the complete feasible witness
and every request's status, selected time, and priority. `priority_labels`
records the selection order. `latest_eligible_time` is the latest grid time
compatible with the original fixed times, before considering competing
requirements and the final size. `displacement` reports how far the selected
point is earlier than that time; `deadline_slack` uses the actual deadline.
An unchanged generic `evaluate_timing_requirements()` call still checks only
counts and deadline eligibility. The mixed optimizer's assessment additionally
records the distinct assignments established by its joint selector.

`max_anchor_search_states` bounds the total joint selection work for each
study size, including its window-feasibility checks. If it reaches the limit
before the latest choices are established, that size reports `search_limit`
with no certified selections or schedule. Individually unavailable and jointly
infeasible requests also remain explicit, without partial selections or
omitted requirements. `max_search_states` continues to limit each feasibility
check in the information search. If that later search stops, the already
verified complete schedule is retained with a separate optimization status.

Existing one-point policies and POC examples retain their behavior. The mixed
POC below composes these functions after reference construction in all four
input modes.

## Proof-of-concept runner

`examples/study_design_proof_of_concept.py` composes the refactored IVIVC core,
local information optimizer, and simulation evaluator into eight backend-only
examples:

- measured in vivo data used as the reference;
- an IVIVC-derived predicted in vivo trajectory, with selectable Approach 1 or
  Approach 2 prediction;
- adaptive refinement from incomplete in vivo data alone;
- IVIVC-informed adaptive refinement using completed target in vivo data;
- IVIVC-informed gap filling after completed times 0, 10, and 40 miss the
  synthetic transition region;
- a next-time comparison using incomplete in vivo data alone, with unrestricted
  central I-optimality, direct early-quantile timing, and deadline-constrained
  central I-optimality;
- one mixed-requirements example with selectable prospective or adaptive input
  mode, early window coverage, and multiple latest-before points;
- a separate `design_size_evolution` example that reuses the measured reference
  with N=1–5 and a required time-zero baseline.

Run a small demonstration from the repository root with:

```bash
python examples/study_design_proof_of_concept.py --quick
```

The default output directory, `study_design_poc_output/`, contains transparent
CSV and JSON files for each scenario and optional static Plotly HTML summaries.

### Approach 2 reference demonstration

The existing `ivivc_prediction` example is also the focused Approach 2
demonstration. Run the same example with the new selection controls:

```bash
python examples/study_design_proof_of_concept.py \
    --scenario ivivc_prediction \
    --ivivc-approach approach2 \
    --ivivc-model exponential_normalized \
    --ivivc-method tau_ratio \
    --analysis-model exponential \
    --ivivc-calibration-response-sd 0.01 \
    --reference-draws 100 --simulations 100
```

Change only `--ivivc-method` to `value_ratio` or
`observed_response_inverse` to exercise the other Approach 2 predictions.
The example uses a normalized exponential calibration model, which fixes its
initial response at one and keeps time zero supported by inverse prediction.
The separate `exponential` study analysis model estimates both amplitude and
rate. `--analysis-model` continues to control the synthetic generating curve
as well as the study analysis; `--ivivc-model` controls only IVIVC calibration.

The same controls apply to `adaptive_ivivc_prediction`,
`adaptive_gap_filling`, and the mixed example's `prospective-ivivc` and
`adaptive-ivivc` input modes. For a future mixed-requirement design after the
exponential example's completed times 0, 10, and 20:

```bash
python examples/study_design_proof_of_concept.py --quick \
    --scenario mixed_timing_requirements --planning-input-mode adaptive-ivivc \
    --ivivc-approach approach2 --ivivc-model exponential_normalized \
    --ivivc-method tau_ratio --analysis-model exponential \
    --ivivc-calibration-response-sd 0.01 \
    --timing-degraded-fraction 0.85 --timing-quantile 0.25 \
    --planning-latest-fraction 0.90 --planning-latest-fraction 0.95
```

Earlier exponential milestones can already precede the completed observations;
the mixed example reports those deadlines as elapsed rather than replacing
the requested fractions. Other curve shapes can require different milestones.

Without the new controls, the existing Approach 1 settings remain
`linear_zero_intercept` and `time_mapping`. Selecting Approach 2 alone defaults
to the study analysis model for calibration and `value_ratio` prediction.
Explicit model choices must be registered for the selected approach. One-phase
mass-loss models do not support the current tau calculation; unavailable tau
or inverse predictions raise an error rather than switching methods or moving
the required time-zero baseline.

Approach 2 fits the calibration profiles directly, without response-aligned
Approach 1 pairs. Its HTML summary and JSON metadata identify the calibration
model and parameters, prediction method, usable draw count, retained time
domain, and calibration-range extrapolation diagnostics. Missing numerical
metadata are written as JSON null. The study's whole-curve I-optimal criterion,
timing requirements, and Monte Carlo assessment retain their existing meanings.

Value-ratio prediction keeps input times as in vivo times, so its target
prediction profile spans the full 0–52 study horizon, including gap filling
after completed time 40. Time-mapping methods use the original in vitro
prediction range and retain the shared mapped domain. These domains need not
be identical across methods. Model evaluations outside calibration ranges are
reported, while reference conversion adds no extrapolation. Unsupported
completed times remain an explicit error.

Rerunning the same output folder with Approach 2 removes a previous
`ivivc_preprocessed_pairs.csv`; no such pair table was used for that fit.

The `design_size_evolution` example appears first in `index.html`, while the
existing examples retain their order of simulation seeds and calculations.
Run its full five-size illustration with:

```bash
python examples/study_design_proof_of_concept.py \
    --scenario design_size_evolution \
    --reference-draws 100 --simulations 100 \
    --analysis-model mass_loss_one_phase --observation-sd 0.05
```

Its four-panel figure shows the reference and N=4 schedules, optimized schedules
across N=1–5, the existing penalized mean IRMSE, and mean T50 estimation error
with the central 95% simulation range. Time zero is a required baseline; the
normal optimizer selects the remaining times. Nested schedules retain earlier
selections; fixed-N refinement can move them. The historical reference is not
added to the simulated new study as completed data.

Returned values at underdetermined sizes are included without rank filtering
or outlier clipping. The page explains that N<4 cannot identify the four-parameter
model (or N below the selected model's parameter count), regardless of the
number of replicates. The existing rank / pseudo-determinant fallback selects
points before full rank; no finite local I criterion is implied at these sizes.
Fit failures retain the existing IRMSE penalty. T50 errors compare each fitted
curve with its own simulated true curve; only successful fits with both T50
values available within the evaluation horizon contribute. The new
`t50_recovery.csv` reports their mean, 2.5th/97.5th percentiles, and counts,
including rows with no recovered endpoints. The range describes simulation
variability, not a confidence interval for the mean. Axes are not clipped to
hide extreme results.

`--quick` and `--max-design-sizes` can truncate this example; when N=4 is absent,
panel a uses the largest available size. The original `measured_in_vivo` example
and the prospective-measured mixed-requirements mode retain their fixed baseline
and previous design-size settings. No numerical backend changes are required.

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

The next-time comparison uses the same partial in vivo reference for all three
policies. It retains completed times 0, 10, and 20 and proposes exactly one new
time (total N=4). This example fixes the candidate policy to `future_only`;
earlier gap filling remains a separate example. Each timing policy appears
once, alongside an equal-time baseline, without duplicate nested/refined rows.
All feasible schedules share the same Monte Carlo trajectory draws and
observation errors, and completed measurements stay fixed.

Run the comparison with the normalized one-phase model:

```bash
python examples/study_design_proof_of_concept.py \
    --quick \
    --scenario adaptive_timing_comparison \
    --analysis-model mass_loss_one_phase_normalized \
    --timing-degraded-fraction 0.50 \
    --timing-quantile 0.10
```

`--timing-degraded-fraction` selects the degradation milestone; the separate
`--timing-quantile` selects its early crossing-time percentile. They also
control the mixed example described below, and do not affect the other
examples. Use
`--analysis-model stretched_exponential` to try another model family. The
ordinary unnormalized one-phase fit to partial data can produce an unsuitable
reference or fail to reach the milestone within the supported horizon; the
example reports that status without substituting another model.

The new scenario writes `timing_policies.csv` with the deadline, selected time,
deadline slack, weight crossed before selection, information diagnostics, and
status/reason for every policy. Unavailable policies remain visible in this
table and the HTML/JSON summaries, but are omitted from Monte Carlo performance
rows. The reference plot marks the deadline and eligible sampling region. The
five original examples retain their existing schedule generation and outputs.

### Mixed requirements across input modes

Run the single mixed-requirements example with incomplete in vivo data:

```bash
python examples/study_design_proof_of_concept.py --quick \
    --scenario mixed_timing_requirements \
    --analysis-model mass_loss_one_phase_normalized \
    --planning-input-mode adaptive-in-vivo
```

Change only `--planning-input-mode` to select the source:

| Mode | Reference and study being planned |
| --- | --- |
| `prospective-measured` | Historical full in vivo data; plan a new study |
| `prospective-ivivc` | IVIVC-predicted in vivo reference; plan a new study |
| `adaptive-in-vivo` | Incomplete target-study in vivo data; add future times |
| `adaptive-ivivc` | IVIVC reference updated using completed target-study data; add future times |

The example defaults to at least one new nonbaseline time before the 10th
percentile of T50, plus distinct latest-before points for the 10th percentiles
of T70 and T90. All targets use complete reference trajectories and their
original or updated weights. Set `--timing-degraded-fraction` to change the
coverage milestone and `--timing-quantile` to change the common crossing-time
quantile. Repeat `--planning-latest-fraction` to replace the default list of
latest-before milestones, for example `--planning-latest-fraction 0.8
--planning-latest-fraction 0.95`. `--planning-search-limit` controls the search
budget for this example.

With the two default latest-before requests, prospective runs plan total N=4
and N=5; adaptive runs retain times 0, 10, and 20 and add three or four future
times (total N=6 and N=7). The existing maximum-design-size setting can limit
the number of sizes. Each size is solved independently. A reserved point can
also satisfy window coverage; the remaining times are optimized around all
requirements. Baseline never counts toward milestone coverage. Historical and
calibration measurements are never treated as completed target-study data.

The example compares an unrestricted central I-optimal benchmark with the
mixed-requirements schedule, using the same Monte Carlo trajectories and
errors. The benchmark retains baseline, completed times, spacing, and future
eligibility; it does not apply the milestone requirements. This example fixes
future-only sampling independently of `--adaptive-candidate-policy`.

The example adds three output files:

- `schedule_status.csv`: every requested size and design, feasibility and
  optimization status, complete/new schedules, experimental units, and
  information diagnostics;
- `timing_requirements.csv`: target definitions, resolved bounds, counting
  scope, qualifying times, achieved counts, and reference crossing status;
- `latest_before_points.csv`: priority, selected time, latest eligible grid
  time, displacement, slack, and status for each requested point.

Metadata retains the planning specification, central parameters and evaluation
grid, and all results. The HTML page shows the reference envelope, coverage
windows, deadlines, selected points, status tables, and Monte Carlo recovery.
Selected times distinguish planned baseline, completed observations, fixed
future commitments, reserved points, and other new sampling. Unavailable or
limited searches remain visible and do not receive fabricated Monte Carlo
performance rows. A deadline that has elapsed or lies beyond the reference
horizon is reported rather than moved. With the exponential model and completed
times 0, 10, and 20, later targets may be needed; for example:

```bash
python examples/study_design_proof_of_concept.py --quick \
    --scenario mixed_timing_requirements --analysis-model exponential \
    --timing-degraded-fraction 0.9 --timing-quantile 0.5 \
    --planning-latest-fraction 0.93 --planning-latest-fraction 0.97
```

For custom requirements, attach a `SchedulePlanningSpec` from
`examples.study_design_mixed_workflow` to any POC
`ScenarioDefinition`. Its common run path accepts the existing reference,
observation model, constraints, and actual completed observations. The input
mode chooses reference construction; it does not select a different optimizer.
The six existing examples retain their individual output formats and calculations.

The built-in demonstration currently defines explicit example parameter presets
for `mass_loss_one_phase`, `mass_loss_one_phase_normalized`, `exponential`,
`stretched_exponential`, and `mass_loss_surface_erosion`. A model without a preset is rejected rather than
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
