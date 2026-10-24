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
