# Supplied observation uncertainty (foundation only)

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

This first patch only defines and validates the shared representation. It does
not attach uncertainty to `TimeCourse`, change fitting/prediction behavior,
estimate residual uncertainty, or add browser/CLI controls. Subsequent patches
will explicitly connect it to Approach 2 fitting and prediction; existing
Approaches 1 and 3 remain unchanged. Keeping the types in `observations.py`
also avoids modifying the design branch's parameter-sampling extensions.
