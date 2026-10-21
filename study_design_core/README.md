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
