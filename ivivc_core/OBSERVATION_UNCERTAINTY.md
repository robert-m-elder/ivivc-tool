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
points. This does not imply that normalized responses, fitted curves, or other
derived quantities are independent. A later transformation must preserve any
dependence it introduces. Separate experiments must be independent for later
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

This patch does not attach uncertainty to `TimeCourse`, transform it, or expose
it through `fit_time_course`, `fit_ivivc`, browser, or CLI inputs. Later patches
will perform that integration. Callers of the standalone fitter must already
supply time, response, and uncertainty on the same analyzed scale; uncertainty
in normalization or transformation parameters is not handled here. Supplied
uncertainty is rejected with explicit `approach_id="approach1"` or `"approach3"`;
the generic standalone fitter also accepts `approach_id=None`. Existing Approach 1
and 3 application calculations and prediction displays remain unchanged.
