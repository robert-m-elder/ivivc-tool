# IVIVC numerical core

`ivivc_core` contains the reusable numerical backend shared by the existing
IVIVC application and future study-design modules. It intentionally does not
create Plotly figures, HTML, Word reports, or Flask objects.

The current browser and CLI continue to use `engine.py`, which acts as a
compatibility and presentation facade over these functions.

## Data-flow overview

```text
raw arrays
    -> prepare_data / prepare_time_course
    -> PreparedIVIVCData or TimeCourse
    -> fit_time_course or fit_ivivc
    -> ProfileFit / IVIVCAnalysis
    -> predict_in_vivo
    -> PredictedTrajectory
    -> generate_trajectory_ensembles
    -> TrajectoryEnsemble
```

All arrays passed to fitting and prediction functions are assumed to be on the
same analyzed scale. The public API does not silently apply or reverse
normalization, scaling, or interpolation.

Preprocessing accepts `one_minus` to apply `1 - response` after the selected
normalizations and before logarithmic scaling or interpolation. It applies to
each supplied profile and records its fixed reference value of 1 in preprocessing
metadata. Already fractional inputs need no other normalization for this option.

Approach 2 also provides `exponential_normalized`, `double_exponential_normalized`,
and `stretched_exponential_normalized`. These fix amplitudes or their sum at 1;
they do not transform the input data. 

`logistic_logistic` and `logistic_logistic_normalized` are Approach 2 models
combining two decreasing logistic components, each normalized by its own value
at time zero. The standard parameters are `A, w, t1, t2, k1, k2`; the normalized
variant fixes `A=1`. Fits constrain `A >= 0`, `0 <= w <= 1`, and both rates to be
positive. Midpoints may precede zero. Both forms approach zero at long times
and use the existing generic tau and inverse calculations. Component labels are
interchangeable, and overlapping transitions may leave parameters poorly identified.

## In vivo-only fitting

```python
from ivivc_core import fit_time_course

fits = fit_time_course(
    time,
    response,
    model_names=["exponential", "stretched_exponential"],
    selected_metrics=["r_squared", "rmse", "aicc"],
    analysis_config={"cv_scheme": "leave_one_timepoint_out"},
    approach_id="approach2",
)

exponential_fit = fits.results["exponential"]
failed_models = fits.errors
```

`fit_time_course` is the intended entry point for measured in vivo data in the
study-design workflow. Candidate models are independent: unless `fail_fast=True`,
one failed model is recorded in `errors` without discarding successful fits.

## Paired IVIVC fitting

```python
from ivivc_core import fit_ivivc, prepare_data

prepared = prepare_data(
    in_vitro_time,
    in_vitro_response,
    in_vivo_time,
    in_vivo_response,
    selected_interpolation=["none"],
)

analysis = fit_ivivc(
    prepared,
    selections=[
        "approach1:linear",
        "approach2:exponential",
    ],
    selected_metrics=["r_squared", "rmse", "aicc"],
    analysis_config={"cv_scheme": "shuffle_split"},
)
```

Selection keys intentionally match the existing application. Approach 1 stores
a fitted time-to-time relationship. Approach 2 stores separate in vitro and in
vivo profile fits. Approach 3 has no parametric fit and is represented only when
explicitly selected.

## Central predictions and complete uncertainty draws

```python
from ivivc_core import (
    TimeCourse,
    generate_trajectory_ensembles,
    predict_in_vivo,
)

prediction = TimeCourse(prediction_time, prediction_response)

central = predict_in_vivo(
    analysis,
    prediction,
    methods={"approach2": ["value_ratio", "tau_ratio"]},
)

ensembles = generate_trajectory_ensembles(
    analysis,
    prediction,
    methods={"approach2": ["value_ratio", "tau_ratio"]},
    n_samples=1000,
    random_state=12345,
)
```

`PredictedTrajectory` contains one central in vivo trajectory.
`TrajectoryEnsemble` retains complete joint draws across all prediction times;
it does not treat pointwise uncertainty bounds as independent observations.
Approach 3 has no fitted covariance and therefore yields a deterministic
one-draw ensemble.

### Response-value ratio uncertainty

`predict_value_ratio` remains the point-only operation.
`predict_value_ratio_with_uncertainty` (also used by `predict_in_vivo`) returns
`value_lower`/`value_upper` and per-point diagnostics in `auxiliary`. It samples
the two parameter vectors independently from their fitted normal covariance
approximations and takes pointwise percentiles of **input response times ratio**.
Input times and responses are fixed; no residual or observation noise is added.
The center is the fitted-parameter prediction, not the Monte Carlo mean. These
are approximate parameter-uncertainty intervals, not future-observation
prediction intervals or simultaneous bands. Model choice, preprocessing, and
transferability of the fitted relationship are not included.

Use `random_state` on the low-level wrapper for repeatable draws. The app uses
1,000 draws, 95% intervals, and seed 12345. Covariance/sampling failures retain
usable central predictions but mark intervals unavailable. No denominator floor
or physical-response clipping is applied. A pointwise interval is withheld if
any requested draw is unusable there or the denominator's sampled equal-tailed
interval includes zero (a conservative display safeguard, not a formal ratio
confidence procedure). Statuses, draw counts, denominator bounds, and sampling
errors are retained, rather than silently conditioning on valid draws.

The ratio ensemble preserves partially finite rows for these pointwise checks.
Its `valid_draws` mask still identifies **complete** finite trajectories for
study-design callers; observation-noise simulation remains separate.

## Error handling

Low-level numerical functions such as `fit_profile` raise their original
exceptions. Batch-oriented public functions return structured `AnalysisError`
records by default. Set `fail_fast=True` while debugging or when any failure
should abort the operation.

## Compatibility boundary

Backend code for a new module should import from `ivivc_core`, not from
`engine.py`. Existing UI code can continue to import `preprocess_data`,
`process_data`, and `process_predictions` from `engine.py`; their legacy return
contracts are preserved by adapters in that facade.

### Time-constant availability

`ProfileFit.tau_estimate` is evaluated at the fitted parameters by one generic,
checked `quad` calculation (default tolerances, `limit=100`). Unsupported models,
integration warnings, invalid evaluations/results, or nonpositive tau produce
NaN, with a reason in `tau_diagnostics`; they do not invalidate the profile fit.
`tau_sd` is separate: sampling failure or any invalid sampled tau produces NaN
uncertainty while retaining a valid central estimate. The fitting API uses 50
draws and seed 12345 for the tau table. Plotted tau-ratio intervals separately
propagate 1,000 pairs of parameter draws; table-SD availability is not their
gate. Valid time-scaled points remain available if the new propagation fails.
Trajectory ensembles use the same checked evaluator and retain invalid-draw
masks; no model-specific formulas or custom convergence search are introduced.
The numerical safeguards are not a guarantee of convergence for arbitrary
functions. The table retains its local tau interval; plotted intervals follow
the shared percentile convention below.


## Common plotted parameter uncertainty

`ivivc_core.intervals.curve_parameter_intervals` and `parameter_interval_summary`
provide pointwise equal-tailed bounds and diagnostics. App fitting plots and
parametric predictions use 1,000 covariance-aware normal parameter draws, seed
12345, and no added residual/measurement noise. Separate fits are sampled
independently. The center remains the fitted-parameter calculation; percentile
bounds can exclude it and are never moved to fit error-bar geometry.

All requested draws must be usable at a point for its interval to be displayed.
Whole-trajectory validity masks remain separate for study-design callers.
`predict_in_vivo` now supplies this interpretation for every parametric method,
with an optional `random_state` to control reproducibility. Low-level
`predict_time_mapping` remains central-only unless `include_uncertainty=True`.
`predict_tau_ratio` requires both fitted parameter mappings/covariances for its
intervals; legacy tau values alone still supply central times, not intervals.
The legacy `generate_prediction_bands` accepts but ignores `residuals` and now
returns parameter-only bounds; new callers should use the diagnostic-aware
helper. The inverse helper likewise ignores its legacy residuals argument.

Inverse mapping uses one generic mapper (`ivivc_core.inverse`) for central and
sampled calculations, including trajectory-ensemble draws. It checks a finite,
numerically monotonic calibration grid, then searches the indicated continuation
with up to 16 doubling extensions (129 points per extension) and Brent roots.
There is no response clipping or model-specific inverse. Plateaus/rounded tails,
nonmonotonic continuations, nonfinite domains and unresolved roots are explicit
unavailable statuses. These checks establish numerical support on the examined
branch, not global mathematical uniqueness. The finite search budget is not a
physical endpoint. An invalid distant target cannot alter an earlier result.

Low-level inverse functions accept `allow_extrapolation`, `minimum_time` (default
0) and `max_expansions` (default 16). Use `minimum_time=None` for transformed time
coordinates such as log10 time. The public API and Flask adapter apply this choice
from **applied** scaling metadata. A negative calibration minimum on unlogged
coordinates remains allowed; the app does not invent a domain excluding data.
Failed sampled inverses are counted by reason and are not silently dropped from
parameter intervals. Extra raw-mass or raw-response sign constraints are not
imposed on normalized/log-transformed responses.

`annotate_prediction_domain` keeps calibration coverage distinct from numerical
validity. Prediction auxiliary metadata carries ranges, per-point reasons,
point/interval flags and caution messages. The high-level API and app use actual
paired-time fitting ranges for Approach 1, both evaluation-time ranges for
response ratios, input/output time ranges for tau rescaling, and only in vivo
response/time ranges for inverse mapping. Lookup-only Approach 3 remains bounded;
no endpoint-hold or linear extrapolation rule is silently introduced. Plots,
downloads, and reports reuse these annotations. Within-range status is not
validation of transfer between devices or of the underlying model.
Parameter-table and tau-table local intervals and descriptive CV variability
retain their distinct meanings. Normal covariance draws remain a local
approximation, conditional on the preprocessing and model selection; no
interpolation-error, model-discrepancy, or future-observation coverage is claimed.
