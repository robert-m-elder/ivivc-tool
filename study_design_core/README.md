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

## Empirical time-scale reference

`EmpiricalTimeScaleDistribution` and `reference_from_time_scaling()` provide
an alternative reference when a target in vitro profile is available but the
target system has no paired IV/IVV calibration data. Supply positive ratios
from comparable systems, defined consistently as
`r = tau_in_vivo / tau_in_vitro`. Each complete reference curve is
`f_in_vivo(t) = f_in_vitro(t / r)`: a ratio of four makes the same degradation
take four times as long in vivo.

For an existing in vitro `ProfileFit` spanning days 0–40:

```python
from study_design_core import (
    EmpiricalTimeScaleDistribution,
    reference_from_time_scaling,
)

scales = EmpiricalTimeScaleDistribution(
    ratios=[1.5, 2.0, 3.0],
    weights=[1.0, 2.0, 1.0],
    source_ids=['paired-system-A', 'paired-system-B', 'paired-system-C'],
    metadata={
        'material_group': 'user-selected comparable systems',
        'time_unit': 'days',
        'tau_definition': 'exponential decay time constant',
    },
)
reference = reference_from_time_scaling(
    in_vitro_fit, scales,
    time_grid=np.linspace(0.0, 60.0, 241),
    required_times=[0.0],
)
```

These ratios are illustrative inputs, not supplied estimates for a material
class. A plain sequence can be passed directly to the constructor for equal
weights. The distribution preserves original records and source identifiers;
duplicate ratios contribute their combined mass to `scales.ecdf()`.
`scales.cdf(values)` returns cumulative probabilities and
`scales.quantile(probabilities)` uses the inverse weighted ECDF without
smoothing or interpolation. Zero-weight records remain traceable but do not
restrict the horizon or generate trajectories. Source identifiers are labels,
not automatic grouping rules: record weights determine the relative influence
of the historical systems.

The default fixes the supplied fitted parameters and propagates only the
empirical scale distribution. A usable fitted covariance is not required.
Set `n_parameter_draws=K` and `random_state=123` to propagate in vitro fitting
uncertainty using the existing covariance-aware joint parameter sampler. Each
of the K complete parameter vectors is paired with every positive-weight ratio,
giving `active_ratio_count * K` trajectories with weights `ratio_weight / K`.
This explicitly assumes independent source-fit and transferred-scale
uncertainty. Increasing K does not resample or narrow the ratio distribution.
Parameter and ratio indices are retained so later updates can be traced back
to their source records. Reference construction never adds observation noise.

The source may instead be a deterministic `ivivc_core.TimeCourse`, interpolated
using the chosen `linear` or `pchip` method. Duplicate source times use the
explicit `duplicate_time_policy` (default `mean`). Parameter uncertainty
requires a `ProfileFit`. The target model need not support the integral tau
calculation: historical ratios are already supplied, and no target tau is
computed or refitted by this constructor.

Every requested in vivo time must map inside the observed source domain for
every positive-weight ratio by default. A fitted source can explicitly permit
`allow_extrapolation=True`; external `TimeCourse` data remain bounded even
when that flag is supplied. No unsupported time is silently removed and no
failed positive-weight curve is dropped to renormalize the distribution.
Required baseline or completed times must fall within the requested reference
grid. The returned reference domain is that numerical grid; explicit model
extrapolation is separately recorded in metadata and does not imply observed
coverage.

The central trajectory is the weighted mean of complete scaled curves and
may differ in shape from an individual scaled trajectory. The returned object
uses the existing `ReferenceTrajectoryEnsemble` contract. Its metadata records
ratios, weights, identifiers, source model/parameters, propagated covariance,
parameter draws, source and required time ranges, and extrapolation masks.
The ordinary analysis-model fit, local I-optimal criterion, timing functions,
and Monte Carlo APIs can consume it directly.

Ratios must use compatible time-constant definitions and units on both sides;
source and study times must share a unit and elapsed-time origin. This is a
horizontal scaling assumption that preserves the source curve's response
scale, shape, and final plateau. The empirical envelope describes variation
implied by the selected systems, rather than uncertainty in their mean ratio.
The prospective/adaptive composition is described below. The POC runner's
`empirical_time_scaling` example exposes both modes and CSV ratio input.

### Prospective and adaptive planning from empirical scales

Prospective planning uses the reference from `reference_from_time_scaling()`
directly. Fit the selected study analysis model with
`fit_reference_model_ensemble(reference, model_name=...)`, then use the existing
local optimizer, or resolve timing requests and call
`optimize_timing_constrained_designs()`. The source in vitro model and study
analysis model can differ. Historical source observations and ratio records
construct the reference; they are not completed observations in the new study.

`time_scaling_weight_summary(reference)` reports this input as
`prospective-time-scaling`, including original and current record weights.
It can also summarize an updated reference. For adaptive planning, keep the
original prior and supply all completed target-study measurements:

```python
from study_design_core import (
    ExistingObservations,
    ObservationModel,
    reweight_time_scaled_reference,
    time_scaling_weight_summary,
)

prior = reference
completed = ExistingObservations(completed_times, completed_responses)
observation = ObservationModel(
    kind='constant_sd', value=0.05, replicates_per_time=3,
)
update = reweight_time_scaled_reference(prior, completed, observation)
reference_for_design = update.reference
weights = time_scaling_weight_summary(reference_for_design)
```

The update reuses `reweight_reference_ensemble()` and reports
`adaptive-time-scaling`. It changes weights, retaining the original curves and
scale values. The posterior central trajectory is the weighted mean of those
curves. Fit the study analysis model to `reference_for_design` and resolve
milestone deadlines from that same updated reference. The local information
criterion is unchanged; the updated weights also govern Monte Carlo truth
sampling and crossing-time quantiles.

For a successive update, call the helper again with `prior` and the cumulative
observations. Passing an already-updated reference is rejected, including one
updated with the generic helper, to prevent accidental repeated use of earlier
measurements. The original prior is not modified, so each cumulative dataset
is applied once. In simulations pass the same `ExistingObservations` object
using `existing_observations=completed`, and include its unique times in
`DesignConstraints.fixed_times`. Completed values are retained rather than
resimulated. Total design size includes those fixed times; it can request one
or multiple new sampling times.

Candidate grids retain the existing eligibility rules. Ordinary local
optimization can use a future-only grid or a grid of any unobserved elapsed
times, including earlier gaps. Timing-requirement planning uses the existing
`PlanningContext` convention of new points strictly after `planning_start`;
set completed and baseline roles explicitly. Both prospective and adaptive
references support sampling windows and multiple distinct latest-before
anchors through these common APIs, without introducing a source-specific
optimization criterion.

The weight summary preserves record order and zero-weight entries in
`ratios`, `source_ids`, `prior_weights`, and `current_weights`. Its
`unique_ratios`, `prior_unique_weights`, and `current_unique_weights` additionally
aggregate duplicate numerical ratios, including zero-mass entries for
traceability. This differs from the ECDF, whose support excludes zero mass.
Draw, record, and unique-scale effective sample sizes are named separately.
Parameter-draw expansion can increase draw ESS without adding independent
historical evidence; none of these ESS values is a historical-study count.

Adaptive results and their reference metadata contain
`time_scaling_weight_summary` and `observation_agreement`. The latter records
per-draw response RMSE and standardized residual RMS, the best complete-draw
standardized RMS, and prior/posterior weighted RMS values. Standardization uses
supplied completed-row SD/SEM when available. Otherwise it uses individual SD
from the observation model, without applying the future replicate count.
Weighted RMS is the square root of the weighted mean squared residual, not
the mean of per-draw RMS values. These continuous diagnostics show agreement
even when all supplied trajectories fit poorly and one merely receives the
largest posterior weight. No adequacy threshold, automatic distribution
widening, or alternative-method fallback is applied.

## Future-study variability

`ObservationModel` is separate from uncertainty in the reference trajectory.
Observation errors are assumed independent within and across times; covariance
between measurements is not represented.
Supported forms are constant absolute SD, constant coefficient of variation,
and an explicitly supplied time profile of individual SD. Historical pointwise
errors do not automatically define the future profile. For example:

```python
observation = ObservationModel(
    kind='time_profile_sd',
    profile_times=[0., 5., 10.],
    profile_sd=[0.02, 0.08, 0.03],
    replicates_per_time=3,
)
sd = observation.standard_deviation([1., 0.7, 0.2], times=[0., 6., 12.])
```

Times must be strictly increasing and use the study's analyzed time coordinate;
SD values use its analyzed response scale. Lookup selects the nearest profile
time, prefers the earlier time for ties, and extends the endpoint SD outside the
profile range. Profile arrays are copied and read-only. The SD floor still
applies. `variance_of_mean()` uses SD squared divided by planned replicate
count; `simulate()` generates individual replicates. Neither expects an SEM.
Profile calls require `times=`, including direct simulation and mean-variance
evaluation. Information, optimization, and measurement simulation pass their
actual observation times. Output metadata retains the complete profile.

`observation_model_from_residuals()` can create a clearly labeled
residual-based surrogate from an existing `ProfileFit`; it can include both
measurement variability and model lack of fit.

### Precision of completed measurements

Historical `TimeCourse` inputs already support the shared uncertainty types.
Use the same representation for completed observations:

```python
from ivivc_core import ObservationUncertainty, prepare_time_course
from study_design_core import existing_observations_from_time_course

course, _ = prepare_time_course(
    [1., 3.], [0.9, 0.6],
    observation_uncertainty=ObservationUncertainty(
        [0.04, 0.05], response_kind='mean', replicate_counts=[4, 9],
    ),
)
completed = existing_observations_from_time_course(course)
```

Supply individual SD for individual rows, or SEM for reported means. Counts
remain metadata and never divide SEM again. Normalize the time course before
conversion; the adapter preserves rows and existing uncertainty. Alternatively,
pass `observation_uncertainty=` directly to `ExistingObservations`. Missing
uncertainty preserves the existing individual-observation interpretation and
observation-model fallback. `n_observations` counts reported rows, not the
underlying units contributing to reported means. `mean_response_by_time()` is
only an arithmetic response summary, not precision-weighted pooling.

Adaptive likelihoods and empirical-scale agreement diagnostics use supplied
completed-row precision directly. Completed means do not acquire the future
study's replicate count. Simulated-study refits retain each completed row with
its SD/SEM and keep the measured response fixed across simulations. Missing
completed precision uses the observation model on the central reference curve.
These are backend APIs; POC input controls and completed-observation precision
in schedule information calculations follow separately.

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

Supply `observation_uncertainty=` using the shared `ObservationUncertainty` or
`ResolvedObservationUncertainty` types. Individual SD or mean SEM determines
the residual weights in candidate selection, grid search, and the final fit.
Parameter covariance uses that absolute scale (`absolute_sigma=True`) without
adding a residual-based variance estimate. Without supplied uncertainty, the
previous unweighted fit and residual-scaled covariance remain in use.

The lower-level `sigma=` alternative accepts absolute standard uncertainty for
each fitted value, allowing a refit to combine completed means and individual
future measurements. Supply either this numeric form or typed uncertainty,
not both. Missing time/response rows are removed with the same mask applied to
uncertainty; pointwise inputs must match the original rows. `ModelFitResult`
retains `observation_sigma` and, for typed inputs, `observation_uncertainty`.
Residuals and `residual_sd` remain response-scale descriptive errors.

Fits that project dense reference trajectories onto an analysis model remain
unweighted. Those grid values are not additional experimental observations.

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
default.

Refits use individual SD for every simulated replicate row, repeated in the
same order as the responses. Completed-row SEM is used directly, without
another division by its count or the planned count. The numerical weights and
covariance scaling are recorded in simulation metadata. Reference-trajectory
spread is not added to these measurement errors; complete trajectories are
already sampled separately.

Constant SD and supplied SD profiles give response-independent refitting
weights. For constant CV, the engine fixes refitting SDs at the central
reference curve for all simulated studies. Noise generation still uses each
draw's true response. These fixed planning weights approximate the
response-dependent variance model; they avoid weighting by noisy observations
or using the generating draw to set fitting weights.

Zero-noise scalar models and all-zero SD profiles retain unweighted refitting.
Mixing zero-SD and positive-SD rows is unsupported: specify a positive
`minimum_sd` rather than dropping exact rows or assigning them an arbitrary
large weight. A positive CV also needs a positive SD floor if the central
reference gives zero planning SD, even when all its values are zero.

The primary metric is integrated
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

`ExistingObservations` stores individual measurements or explicitly identified
reported means already collected. Supported adaptive simulations keep completed
values fixed and simulate only new measurements at proposed elapsed times.
`reweight_reference_ensemble()` can
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
POC below composes these functions after reference construction in all six
input modes.

## Proof-of-concept runner

`examples/study_design_proof_of_concept.py` composes the refactored IVIVC core,
local information optimizer, and simulation evaluator into nine backend-only
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
  with N=1–6 and a required time-zero baseline;
- one `empirical_time_scaling` example, selectable as prospective or adaptive,
  using target in vitro data and illustrative or user-supplied scale ratios.

Run a small demonstration from the repository root with:

```bash
python examples/study_design_proof_of_concept.py --quick
```

The default output directory, `study_design_poc_output/`, contains transparent
CSV and JSON files for each scenario and optional static Plotly HTML summaries.

### Empirical time-scale POC example

Run the focused example with its illustrative synthetic ratios:

```bash
python examples/study_design_proof_of_concept.py --quick \
    --scenario empirical_time_scaling \
    --analysis-model mass_loss_one_phase_normalized
```

The target in vitro observations are fitted over times 0–40, and the in vivo
study horizon is 0–52. The default ratios are 1.6, 1.8, 2.0, 2.2, and 2.4, with
equal record weights. These are illustrative values, not estimates for an
implant or polymer class. No paired IV/IVV calibration dataset is constructed.
Use `--time-scaling-mode adaptive` to condition the same original prior on
completed target-study observations at times 0, 10, and 20. Ordinary adaptive
eligibility follows `--adaptive-candidate-policy`.

The new scenario is appended to `--scenario all`, retaining the existing
scenarios' seed order. `design_size_evolution` remains first in `index.html`.

A user-supplied CSV can contain:

```csv
ratio,weight,source_id,group
1.6,1,system-A,polymer-X
2.0,2,system-B,polymer-X
2.4,1,system-C,polymer-Y
```

Pass its path with `--time-scaling-ratios-csv ratios.csv`. Only `ratio` is
required. Omitting the entire `weight` column gives equal weights; blank or
invalid cells in an existing weight column cause an error. Missing source IDs
are generated only when the entire `source_id` column is absent. Group labels
and other columns remain provenance; they do not automatically filter or
reweight records. Input order, duplicates, and zero-weight records are retained
in the output tables. Missing files and malformed rows never fall back to the
illustrative distribution.

`--time-scaling-model` chooses the in vitro fitting model independently of
`--analysis-model`; its default follows the analysis model. As in the other
examples, the analysis model also generates the synthetic truth.
`--ivivc-calibration-response-sd` controls in vitro source noise in this example,
while `--observation-sd` controls target-study measurements.
The default `--time-scaling-parameter-draws 0` fixes the fitted source curve;
a positive K crosses K joint parameter draws with every positive-weight ratio.
The resulting count is `active_ratio_count * K`, or one curve per active ratio
when K is zero. `--reference-draws` does not resample the empirical ratios or
control this count. The source curve does not need a computable integral tau.

The requested horizon must be covered for every active ratio. For fitted-source
extrapolation, explicitly add `--time-scaling-allow-extrapolation`; source-range
checks and extrapolation diagnostics retain their backend meanings.

The mixed-requirements example also accepts the two new input modes:

```bash
python examples/study_design_proof_of_concept.py --quick \
    --scenario mixed_timing_requirements \
    --planning-input-mode adaptive-time-scaling \
    --analysis-model mass_loss_one_phase_normalized \
    --timing-degraded-fraction 0.50 --timing-quantile 0.25
```

Use `prospective-time-scaling` for a new full study. In the mixed example,
`--planning-input-mode` determines the mode; `--time-scaling-mode` applies only
to the focused example. Ratio CSV, source-model, and uncertainty options apply
to both. The mixed example retains its existing future-only eligibility and
reports unavailable deadlines or schedules without changing the requests.

Empirical-mode pages add weighted prior/current ECDFs, prior/current reference
envelopes and central curves, record-weight tables, separate draw/record/unique
ESS, and adaptive observation-agreement diagnostics. The usual optimized
schedules, IRMSE, and endpoint summaries follow the existing calculation path.
In vitro source observations are not plotted as in vivo measurements.
Additional machine-readable files are `time_scale_input.csv`,
`time_scale_weights.csv`, `time_scale_ecdf.csv`, and, for adaptive runs,
`observation_agreement.csv`. JSON retains the full reference and agreement
metadata with unavailable values as null. Rerunning the same scenario folder
removes empirical-specific files that no longer apply.

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
Run its full study-size illustration with:

```bash
python examples/study_design_proof_of_concept.py \
    --scenario design_size_evolution \
    --reference-draws 100 --simulations 100 \
    --analysis-model mass_loss_one_phase --observation-sd 0.05
```

Its four-panel figure shows the reference and N=4 schedules, optimized schedules
across N=1–6, the existing penalized mean IRMSE, and mean T50 estimation error
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
| `prospective-time-scaling` | Target in vitro data and an empirical ratio distribution; plan a new study |
| `adaptive-time-scaling` | Empirical time-scale reference updated using completed target-study data; add future times |

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
