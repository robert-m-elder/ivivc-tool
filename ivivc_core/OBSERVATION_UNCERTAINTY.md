# Supplied observation uncertainty and profile fitting

`ObservationUncertainty` represents the absolute standard uncertainty of each
entered response, in that response's units. For individual measurements, supply
SD; for reported means, supply SEM. Constant values and pointwise arrays resolve
to the same `ResolvedObservationUncertainty.sigma` representation:

```python
from ivivc_core import ObservationUncertainty, resolve_observation_uncertainty

reported_sem = ObservationUncertainty(
    [0.03, 0.05, 0.04],
    response_kind="mean",
    source="published SEM column",
    response_units="fraction remaining",
)
resolved = resolve_observation_uncertainty(reported_sem, n_observations=3)
# resolved.sigma is [0.03, 0.05, 0.04].

constant_sd = ObservationUncertainty(0.05, response_kind="individual")
resolved_sd = resolve_observation_uncertainty(constant_sd, n_observations=3)
unspecified = resolve_observation_uncertainty(None, n_observations=3)  # None
```

Original experimental errors are assumed independent within and across time
points. Initial-value and min-max normalization use the fixed-reference working
model described below: their observed constants are held fixed, and uncertainty
in those constants is excluded. Fitted curves and other derived quantities can
still be dependent. Separate experiments must be independent for
independent-input propagation; reuse of calibration measurements is not a new
independent experiment.

Optional `replicate_counts` describe reported means and may be scalar or
pointwise. They are retained as metadata; they **never divide an SEM again** or
add another fitting weight. A mean accompanied by replicate SD must have that
SD converted to SEM before using this interface. Counts are not required.
Relative weights, percentage-error conversion, and raw-replicate aggregation
are not implemented by this representation. Future-observation variability is
selected separately through the backend option below.

Resolution requires finite, strictly positive uncertainties for every retained
response. Missing uncertainty is `None`, not zero or known unit uncertainty.
There is no implicit row removal or replacement of missing errors. To remove
missing time/response rows, pass the same boolean mask used for those arrays:

```python
reported_sem = ObservationUncertainty([0.03, float("nan"), 0.04], response_kind="mean")
keep = [True, False, True]
resolved = resolve_observation_uncertainty(reported_sem, 3, row_mask=keep)
# resolved.sigma is [0.03, 0.04], aligned to the two retained responses.
```

Pointwise uncertainty and count arrays must match the original row count before
masking. Excluded pointwise entries need not be valid; retained entries must
be. Scalar inputs must be valid even for an empty selection. Validated resolved
objects can be subset again with `.subset(keep)`. Arrays are defensive read-only
copies, and `.to_metadata()` returns a separate JSON-compatible record of the
resolved values, interpretation, provenance, units, and assumptions.

## Browser and CLI input boundary

The browser and CLI accept constant or named-column SD/SEM independently for in
vitro calibration, in vivo calibration, and prediction input. Their shared parser
in `utilities.uncertainty_inputs` constructs `ObservationUncertainty`; preprocessing
then cleans and rescales it with the response rows. Users enter errors on the
uploaded response scale, before transformations selected in the app.

These entry points require Approach 2 alone, confirmation of independent original
errors, and no `log_y` selection when any error is supplied. The CLI confirmation
flag is `--independent-errors`. Optional replicate-count columns describe reported
means and remain metadata. Reports retain both entered and analyzed errors with
source-row alignment. See the [repository README](../README.md) and
[guide source](../docs/user_guide.tex) for the controls and examples. The batch API
has its own per-selection error behavior, described below.

## Standalone weighted profile fitting

`fit_profile` accepts optional `observation_uncertainty` on the same analyzed
response scale as the supplied time/response arrays:

```python
from ivivc_core import ObservationUncertainty, fit_profile

fit = fit_profile(
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [1.0, 0.83, 0.66, 0.57, 0.45],
    model_name="exponential_normalized",
    selected_metrics=["rmse"],
    approach_id="approach2",
    analysis_config={"cv_scheme": "none", "grid_search_num_cores": 1},
    observation_uncertainty=ObservationUncertainty(
        [0.03, 0.04, 0.03, 0.06, 0.08], response_kind="mean",
    ),
)

sigma = fit.observation_uncertainty.sigma
method = fit.fit_diagnostics["covariance_scaling"]  # "absolute"
```

The supplied uncertainties control the starting-value search, each CV training
fit, and the final fit through the sum of squared standardized residuals,
`sum(((response - fitted_response) / sigma)**2)`. Final and CV fits use
`absolute_sigma=True`; the supplied error scale is not estimated again from
residual scatter. A perfect fit therefore need not have zero parameter
covariance. Multiplying every supplied SD/SEM by a common factor leaves the
objective's minimizer unchanged while scaling covariance by that factor squared.
The covariance remains a local approximation around the fitted parameters.

Time/response NaN rows are removed with the same explicit mask applied to
original-row-aligned uncertainties and optional counts. Supplied SEMs are not
divided by counts again. Uncertainty omitted as `None` preserves the existing
unweighted fitting and residual-scaled covariance behavior. Model optimizer
settings, including bounds, are preserved without modifying registry options.

`ProfileFit` retains the resolved uncertainty and `fit_diagnostics`, including
the objective, covariance convention, finite-covariance flag, retained observation
and parameter counts, and (for weighted fits) chi-squared and reduced chi-squared.
Reduced chi-squared is diagnostic only; it never rescales absolute covariance.
The retained observation count, not the sum of replicate counts, determines the
reported residual degrees of freedom.

RMSE, R-squared, and other descriptive metrics retain their unweighted definitions.
Supplied-error full fits calculate AIC/BIC from the normal likelihood on the
analyzed scale. They retain the application's conventional AICc correction
`2*k*(k+1)/(n-k-1)` when `n > k+1`, where `n` counts retained responses and `k`
counts fitted curve parameters. This convention does not assert exact small-sample
calibration for every nonlinear model or supplied-error setting.

Criteria are forwarded explicitly to source-wise evidence calculations, including
mixed supplied-error/residual-scaled sources. Combined Approach 2 evidence adds
the two source-specific criteria; combined AICc adds their separate corrections
and requires both to be defined. It is not reconstructed using pooled `n` and `k`.

Supplied-error CV information criteria remain unavailable. Its `observation_error`
summary pools standardized RMSE and normal observation negative log likelihood
over test-observation occurrences, including repeated uses of a row across splits.
These scores compare fitted predictions with the supplied observation errors;
they do not add training-parameter uncertainty. Pooled scores require CV to be
enabled and every requested split to provide a finite score. Ordinary CV means can
still summarize finite split scores and retain their finite-score counts.

Low-level `auto_grid_search` accepts optional scalar/pointwise `sigma`.
`cross_validation_curve_fit` accepts `sigma` and `absolute_sigma`, validates
uncertainty before splitting (including when CV is disabled), and subsets it to
each training fold. Preprocessing and starting vectors are established before the
CV splits; they are not rebuilt using each training subset alone. Low-level helper
metrics remain descriptive.

The standalone fitter accepts either supplied or already-resolved uncertainty.
Callers must supply time, response, and uncertainty on the same analyzed scale.
Uncertainty is rejected with explicit `approach_id="approach1"` or `"approach3"`;
the generic standalone fitter also accepts `approach_id=None`.

## Preprocessing transport

`prepare_time_course` accepts `observation_uncertainty` and returns a `TimeCourse`
with aligned, resolved `.observation_uncertainty`. It can feed the standalone
fitter directly:

```python
from ivivc_core import ObservationUncertainty, prepare_time_course, fit_profile

course, preprocessing = prepare_time_course(
    [0.0, 1.0, float("nan"), 3.0],
    [1.0, 0.8, 0.5, 0.4],
    observation_uncertainty=ObservationUncertainty(
        [0.03, 0.04, float("nan"), 0.06], response_kind="mean",
        source="reported SEM", response_units="fraction remaining",
    ),
)
fit = fit_profile(
    course.time, course.value,
    model_name="exponential_normalized", selected_metrics=["rmse"],
    approach_id="approach2",
    analysis_config={"cv_scheme": "none", "grid_search_num_cores": 1},
    observation_uncertainty=course.observation_uncertainty,
)
source = preprocessing.sources["profile"]
# source.time / source.value retain all four original rows.
# source.retained_rows is [0, 1, 3], linking analyzed rows to the original rows.
```

`prepare_data` and `prepare_ivivc_data` accept optional `in_vitro_uncertainty`
and `in_vivo_uncertainty` keywords. They can differ in format, SD/SEM meaning,
counts, or whether supplied at all. Paired source records are stored under
`preprocessing.sources["in_vitro"]` and `["in_vivo"]`. The uncertainty values
are never aligned between environments or assigned to interpolation grids.
Replicates at duplicate source times remain distinct source rows.

When any uncertainty is supplied, cleaning removes rows with missing time OR
response independently for each profile, using that same mask for uncertainty
and optional counts. Missing errors on retained rows are errors; they are not
filled or silently dropped. Original zero times are preserved. Without supplied
uncertainty the existing cleaning and numerical preprocessing behavior is
unchanged, and no additional source snapshots are created.

`PreprocessingSource` stores defensive read-only copies of original time/response
arrays, original uncertainty, a label, and the retained original-row indices.
Original missing rows and the input scale remain available for provenance.
These snapshots describe the inputs to this preprocessing call, not necessarily
an earlier experiment before caller-applied transformations.

`TimeCourse` accepts only already-resolved, row-aligned uncertainty. Raw uncertainty
should be passed to preprocessing so cleaning can precede numerical validation.
Paired preprocessing can consume attached resolved uncertainty, but rejects an
additional uncertainty keyword for that same profile instead of overwriting it.

### Supported transformations

| Control with supplied uncertainty | Treatment |
| --- | --- |
| No response transformation | Retain the original absolute SD/SEM |
| `one_minus` | Apply `1 - response`; absolute SD/SEM, counts, and response units remain unchanged |
| `initial_value` | Divide responses by the observed initial value and SD/SEM by its absolute value; treat that reference as fixed |
| `min_max` | Subtract the observed minimum and divide responses and SD/SEM by the observed range; treat both constants as fixed |
| `log_x` | Transform known times; response uncertainty is unchanged |
| `log_y` | Reject: requires an explicitly supported transformed observation-error model |
| Unknown/custom transformations | Reject until uncertainty handling is explicitly verified |

`one_minus` retains its existing interpretation for fractional responses.
Log-time fallback remains atomic across paired datasets: if an applicable time
is zero or negative and fallback is requested, both profiles keep their current
times and record the existing warning. Unsupported response transformations are
rejected BEFORE fallback; requesting fallback cannot bypass this restriction.

### Fixed normalization constants

Each profile's initial value, minimum, maximum, and range are calculated from its
current cleaned responses at the relevant normalization step. Those constants
are then treated as fixed. For `initial_value`, `sigma_new = sigma / abs(initial)`;
for `min_max`, `sigma_new = sigma / range`. A fixed subtracted minimum does not
add uncertainty. Chained normalizations apply these conversions sequentially;
`one_minus` remains last and is applied once.

This convention excludes uncertainty in estimating the reference values and
does not introduce covariance from a random denominator or random extrema.
Subsequent fitted-parameter, curve, and prediction uncertainty is conditional on
the chosen constants. Normalized baseline/extreme observations retain their
rescaled positive SD/SEM; their uncertainties are not forced to zero. Fixing a
model's amplitude or asymptotes remains a separate model choice.

Applicable `NormalizationRecord` parameter dictionaries include
`constants_treated_as_fixed: True`, together with the existing observed constants
and `applied` flag. With paired profiles, each uses its own constants, including
when only one profile has supplied uncertainty. Raw source snapshots, original
uncertainties, counts, and provenance remain unchanged. Rescaled uncertainties
retain SD/SEM interpretation, source, counts, and supplied input format; their
response units become `dimensionless` after an applied normalization.

The existing zero-initial-value convention is preserved: normalization is not
applied, and response values, uncertainty, and units stay unchanged. With supplied
uncertainty, an initial reference must be finite and min-max normalization
requires a finite, strictly positive range. Invalid or unrepresentable rescaled
uncertainties raise an error. Analyses without supplied uncertainty preserve
their existing behavior and normalization record shapes.

Already-normalized data must have SD/SEM on that same scale. Converting only the
response values is insufficient. Source snapshots describe the inputs received
by this preprocessing call, including any earlier caller-applied normalization.

The registered interpolation methods preserve observed source rows. The returned
source `TimeCourse` objects retain only their observed-row uncertainties. Derived
time/response grids have no supplied uncertainty model; increasing grid density
does not create new independent observations or interpolate error bars.

### Integration boundaries

`PreparedIVIVCData.has_observation_uncertainty` identifies supplied-error input.
Legacy positional-array and legacy-metadata adapters reject these objects rather
than drop their uncertainty. Existing no-uncertainty adapter shapes are unchanged.

`fit_time_course` accepts optional supplied or resolved `observation_uncertainty`
for a batch of source-profile fits. The same-scale, missing-row, and absolute-error
rules used by `fit_profile` apply to every candidate model. Successful models and
independent model errors retain the existing `ResultBatch` behavior.

`fit_ivivc` consumes each prepared source profile's uncertainty independently for
Approach 2. One profile may use supplied absolute errors while the other uses
the existing residual-scaled fit. Uncertainty retained in provenance but missing
from the processed source profile is an error, not permission to fit unweighted.

```python
from ivivc_core import (
    ObservationUncertainty, fit_ivivc, prepare_data,
    profile_curve_intervals, profile_parameter_intervals,
)

prepared = prepare_data(
    [0.0, 1.0, 2.0, 3.0], [1.0, 0.82, 0.67, 0.55],
    [0.0, 2.0, 4.0, 6.0], [1.0, 0.81, 0.68, 0.54],
    in_vitro_uncertainty=ObservationUncertainty(0.03, response_kind="mean"),
    in_vivo_uncertainty=ObservationUncertainty(0.05, response_kind="mean"),
)
analysis = fit_ivivc(
    prepared, selections=["approach2:exponential_normalized"],
    selected_metrics=["rmse"],
    analysis_config={"cv_scheme": "none", "grid_search_num_cores": 1},
)
fit = analysis["approach2:exponential_normalized"].in_vivo_fit
coefficients = profile_parameter_intervals(fit)
curve = profile_curve_intervals(fit, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
```

With supplied calibration errors, Approach 1 and 3 selections are returned as
explicit selection errors. Successful Approach 2 fits remain available in the
same batch; `fail_fast=True` raises on the first failure. Without supplied errors,
the existing approach-selection behavior and numerical results are unchanged.

## Fitted parameter and curve uncertainty

`profile_parameter_intervals` returns numeric estimates, standard errors, bounds,
availability flags, and the interval convention. It does not produce HTML or
change the fit's covariance:

| Source-profile covariance | Approximate parameter interval |
| --- | --- |
| Supplied absolute SD/SEM | Normal-Wald interval using the supplied covariance scale |
| Unspecified observation uncertainty | Existing residual-df Student-t convention |
| Unspecified uncertainty with unavailable/nonpositive residual df | Existing normal fallback, retaining exactly 1.96 for 95% intervals |

Supplied SDs/SEMs are treated as error estimates given to the analysis; uncertainty
in those error estimates is not modeled. The normal-Wald interval is a local
approximation, not a claim of exact finite-sample coverage. Optional replicate
counts neither change its multiplier nor add observations/degrees of freedom.
Fixed normalization constants remain fixed as described above.

The summary records `confidence_level`, `multiplier`, `distribution`,
`degrees_of_freedom`, `covariance_scaling`, `scale_source`, and `method`, alongside
`parameter_names`, `estimate`, `standard_error`, `lower`, `upper`, and `available`.
Invalid joint covariance makes interval uncertainty unavailable while retaining
central estimates. Singular valid covariance is allowed; negative variances or
invalid covariance are not clipped into apparent certainty.

`profile_curve_intervals` reuses the existing complete multivariate-normal
parameter draws and pointwise Monte Carlo bounds. These curve intervals retain
the normal-covariance approximation even when the coefficient table uses a
Student-t multiplier; they are distinct established summaries. Full covariance,
pointwise failure handling, fitted-parameter centers, confidence settings, and
reproducible seeds are preserved. Provenance identifies absolute versus residual
scaling, the covariance Monte Carlo method, and the absence of observation noise.

The app-facing fit adapter uses the same parameter multiplier for coefficient
and approximate symmetric tau tables. Its existing tau SD comes from sampled
fitted parameters; tau-ratio prediction intervals continue to use the existing
nonlinear parameter-draw percentiles. Weighted table metadata records the method
and scale source. The adapter forwards supplied-error information criteria
explicitly and preserves the existing source-wise combined-evidence convention;
it does not rebuild weighted criteria from unweighted RSS. The
[statistical appendix](../docs/appendices/interpreting_model_performance.tex)
describes the criterion conventions and interpretation limits.

## Calibration uncertainty in existing predictions

For fixed prediction inputs, Approach 2 value-ratio, time-constant-ratio, and
inverse-response calculations inherit the weighted calibration covariances.
Existing complete source parameter vectors are reused across each trajectory,
preserving dependence across times and within vectors. In vitro and in vivo
source vectors are sampled independently under the declared source independence
assumption. No additional calibration residual noise is added.

When either source has supplied errors, prediction auxiliary data and complete
ensemble metadata include each source's `calibration_uncertainty` and covariance
scale, `prediction_inputs_fixed=True` when no input errors are supplied, and
`future_observation_variability_included=False` unless future measurements are
explicitly requested. Applicable fixed normalization
records are retained under `fixed_normalization_constants`. The metadata is
provenance, not an extra uncertainty contribution. Outputs without supplied
errors retain their existing metadata shape.

## Prediction-input uncertainty (Approach 2)

`prepare_time_course` accepts the same SD/SEM representation for a new in vitro
prediction dataset. Its responses and errors must be on the analyzed scale used
by calibration. Preprocessing uses the established fixed-reference normalization
convention; logarithmic response scaling remains unsupported with supplied errors.

```python
from ivivc_core import (
    ObservationUncertainty, prepare_time_course,
    predict_in_vivo, generate_trajectory_ensembles,
)

prediction, input_preprocessing = prepare_time_course(
    [1.0, 2.0, 3.0], [0.8, 0.65, 0.5],
    observation_uncertainty=ObservationUncertainty(
        [0.03, 0.04, 0.04], response_kind="mean", replicate_counts=4,
    ),
)
methods = {"approach2": ("value_ratio", "observed_response_inverse")}
predictions = predict_in_vivo(analysis, prediction, methods=methods)
ensembles = generate_trajectory_ensembles(
    analysis, prediction, methods=methods, n_samples=1000, random_state=12345,
)
```

Each input response is sampled once per trajectory from a normal distribution
centered on its entered value with its supplied standard uncertainty. An
individual response uses SD; a reported mean uses SEM. Counts are metadata and
never divide SEM again or generate extra replicates. Input times, supplied error
estimates, and normalization constants are held fixed. No prediction-profile
refitting or additional calibration residual perturbation is performed.

| Method | Input-response contribution |
| --- | --- |
| Response-value ratio | Multiply the sampled response once by that trajectory's fitted in vivo/in vitro ratio. Response intervals combine calibration and input uncertainty. |
| Time-constant ratio | Carry the sampled response on the response axis. The time coordinate depends only on the shared calibration tau ratio, so response error does not widen time intervals. |
| Observed-response inverse | Map the sampled response through the sampled in vivo curve and retain that same response as the output value. Time and response coordinates remain dependent. |

Full parameter covariance and one shared source vector per trajectory are
retained. Separate input rows are independent, including distinct observations
at equal times. Prediction measurements must also be independent of calibration
measurements. The high-level APIs reject direct reuse of the same calibration
`TimeCourse` object as an uncertain prediction input. Copies cannot establish
independence: reusing calibration measurements, overlapping means, or shared
experimental errors requires a joint error model and is not supported.

Central predictions remain the nominal-input, fitted-parameter calculations.
The sampled mean does not replace them. Response draws are never clipped to
positive values or a normalized range, nor retried until an inverse is possible.
An unsupported nominal inverse remains unavailable even if some perturbed
responses can be mapped. Inverse time intervals require the configured minimum
fraction of requested draws to map successfully at each point (stock threshold:
95%); optional future-response intervals use the same rule. When some draws fail,
the conditional interval encloses the middle 95% of successful mappings, not all
requested outcomes. The policy, threshold, counts, and failure reasons are
retained. Other sampled intervals require every requested draw to be usable.
Denominator safeguards and inverse-domain checks remain active, with failed and
extrapolated inverse draws counted. Backend settings and overrides are described
in [the core README](README.md#common-plotted-parameter-uncertainty).

Primary response-ratio and inverse-time intervals record
`interval_kind="parameter_and_input_uncertainty"`. Tau time intervals retain
`interval_kind="parameter_uncertainty"`. Tau and inverse methods also provide
`value_lower`/`value_upper` from the exact marginal normal quantiles of the
entered response and SD/SEM, with
`value_interval_kind="prediction_input_uncertainty"` and
`value_interval_method="normal"`. These analytical marginal bounds need not
match finite-sample percentiles of the response draws exactly; they do not
describe a joint confidence region or simultaneous band. Tau time bounds propagate
calibration parameters only; inverse time bounds propagate in vivo parameters and
the sampled response together. Vertical input bounds can remain available when
parameter covariance prevents calculating horizontal uncertainty.

Input metadata records the resolved error representation under
`prediction_input_uncertainty`, the normal-distribution and independence
assumptions, fixed times, propagation status, and `prediction_inputs_fixed=False`.
This information is supplied even when calibration has no entered errors.
The point-only `predict_value_ratio` operation records
`prediction_input_uncertainty_propagated=False`; use its uncertainty wrapper for
intervals. Complete ensembles retain `prediction_response_draws`, paired output
draws, and separate complete-trajectory validity masks. Their input draws apply
only at supplied rows, never at the denser model plotting grid or unmeasured
candidate times.

Integer seeds leave the existing calibration streams unchanged and use a
separate input stream. A supplied Generator/RandomState is consumed sequentially;
omitted uncertainty consumes no extra random values and preserves legacy outputs.

Approach 1/3 prediction methods and their samplers still reject attached input
errors. Mixed high-level batches retain successful Approach 2 predictions and
record unsupported methods as errors, unless `fail_fast=True`.

By default no future in vivo measurement variability is added. Input uncertainty
describes the existing prediction measurements. Future study-design simulation
must use a separate measurement-error assumption for future times and must not
count these entered input errors again. Browser and CLI inputs use the shared
parsing boundary described above.
The existing `uncertainties` dependency is loaded only when the high-level
tau-prediction compatibility wrapper needs its nominal/SD value type.

## Optional future in vivo measurements (backend)

Approach 2 prediction functions and `predict_in_vivo` accept
`future_observation_uncertainty`. Omit it or pass `None` to preserve the existing
intervals. Supply an existing `ObservationUncertainty` or
`ResolvedObservationUncertainty` on the **analyzed response scale**, or select a
source for reuse:

```python
predictions = predict_in_vivo(
    analysis, prediction,
    future_observation_uncertainty=ObservationUncertainty(.03, response_kind="individual"),
)
# Alternatives: future_observation_uncertainty="prediction" or "calibration".
```

An individual future observation uses SD; a future mean uses SEM. Counts remain
metadata and never divide SEM again. Raw uncertainties must first be transported
through the selected preprocessing, as for other backend inputs.

`"prediction"` explicitly reuses the prediction inputs' precision and
individual/mean interpretation. `"calibration"` uses supplied in vivo calibration
SD/SEM at the nearest nominal predicted in vivo time. Lookup uses analyzed time
coordinates, prefers the earlier time for ties, and extends the endpoint error
outside calibration. The matched errors are held fixed during propagation and
their source and lookup assumptions are recorded. Missing source uncertainty is
an error; variability is not inferred from fit residuals.

Independent normal future measurement noise is added once to the propagated
response draws. The resulting `value_lower`/`value_upper` are combined future
response intervals; propagated-only bounds remain in auxiliary metadata.
For tau and inverse methods these future responses accompany the mapped
prediction; the fitted curve is not reevaluated at a fixed nominal time.
Horizontal time intervals and central predictions retain their existing meaning.
Inverse response intervals use the same successful mappings and conditional
policy as inverse time intervals. Separate axis bounds remain marginal intervals.

Future variability does not alter fitted-curve/parameter uncertainty or the raw
reference ensembles returned by `generate_trajectory_ensembles`. These intervals
remain conditional on the chosen model and supplied error assumptions; model
discrepancy, model-selection uncertainty, and variable normalization references
are excluded.

The browser and CLI expose `none`, `constant`, `prediction`, and `calibration`
future modes. Their constant values use the **original in vivo calibration
response units**; `utilities.uncertainty_inputs.future_uncertainty_from_options`
transports them once through the fixed in vivo normalization records before
calling this backend. Reused errors are already analyzed and are not rescaled
again. Future options require a prediction dataset, Approach 2 alone, no
`log_y`, and confirmation of independent errors. Reports and downloads retain
the entered/analyzed future errors, source/lookup assumptions, original
propagated response bounds, and calculated interval status.
