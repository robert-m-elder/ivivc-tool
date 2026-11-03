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
still be dependent. Separate experiments must be independent for later
independent-input propagation; reuse of calibration measurements is not a new
independent experiment.

Optional `replicate_counts` describe reported means and may be scalar or
pointwise. They are retained as metadata; they **never divide an SEM again** or
add another fitting weight. A mean accompanied by replicate SD must have that
SD converted to SEM before using this interface. Counts are not required.
Relative weights, percentage-error conversion, raw-replicate aggregation, and
future-observation variability are not implemented by this representation.

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

RMSE, R-squared, and other descriptive metrics retain their existing unweighted
definitions. AIC/AICc/BIC are returned as NaN for supplied-uncertainty fits and CV
scores until likelihood-aware evidence is implemented. The reason is retained in
`fit_diagnostics`; the legacy unweighted-RSS formulas must not be presented as
evidence for the weighted statistical model.

Low-level `auto_grid_search` accepts optional scalar/pointwise `sigma`.
`cross_validation_curve_fit` accepts `sigma` and `absolute_sigma`, validates
uncertainty before splitting (including when CV is disabled), and subsets it to
each training fold. Existing fold construction and full-data starting vectors are
unchanged; training-only initialization and rebuilding data-dependent preprocessing
inside CV are later validation work. Low-level helper metrics remain descriptive.

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

`fit_ivivc` rejects supplied source uncertainty until Approach 2 integration is
implemented; use the standalone source-profile fitter above in the meantime.
Prediction methods, complete prediction samplers, and high-level prediction APIs
also reject attached prediction-input uncertainty until supported propagation is
implemented. Fitted calibration-parameter uncertainty continues to be propagated
as before. Browser/CLI controls remain unchanged, as do existing Approach 1 and 3
calculations and prediction displays without supplied-error inputs.
