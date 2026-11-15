# Study-design Flask frontend

The study-design interface has a separate Flask entry point and shares the
repository's Python environment with the IVIVC app. The IVIVC entry point,
templates, and numerical backend are unchanged.

## Run locally

From the repository root, with `requirements.txt` installed:

```bash
python design_app.py
```

Open `http://127.0.0.1:5001`. The separate port permits running the IVIVC app
on its usual port at the same time. Debug mode is off by default.

For a hosted installation, use the hosting service's normal WSGI configuration
with `design_app:app`, or import `app` from `design_app` as its application
object. The built-in development server is for local use. A future landing
page can link to the two applications without combining their entry points.

## Setup and request handling

- `GET /` serves the input and comparison workspace.
- `POST /review` accepts multipart form data and returns a JSON response.
  It validates the selected task, its active reference/environment choice,
  and the advanced numerical settings. Success includes server-rendered setup
  review HTML; errors include a message and an optional field identifier.
- **Review setup** checks these choices. File previews are requested separately;
  it does not fit curves, generate schedules, or run simulations. A setup review
  is not an analysis result.
- `POST /calculate` imports and validates the current files and settings, then
  completes prospective or adaptive study planning synchronously. It returns server-rendered
  comparison HTML or field-linked errors.
- **Calculate comparison** supports new studies and additions to completed
  observations, using measured reference data, an IVIVC prediction, or empirical
  time scaling. Timing windows and deadlines can constrain proposed future times.
  Calibration comparisons retain import/setup review but their numerical
  calculations are not yet connected to this frontend. Report generation remains
  unavailable.

JavaScript is required for the workspace. It leaves the input form mounted,
including selected browser files, while a normal request runs to completion.
Displayed results change only after a successful calculation. Setup reviews use
a separate region and never replace numerical results. Validation, network,
and server errors preserve earlier output. Edits during a request are
distinguished from the settings submitted in that request; successful results
retain their original input summary and are marked out of date when needed.

Only the current browser page holds the working form and displayed review.
Refreshing or closing it clears that state. There is no project/result store,
session storage, job queue, polling, progress reporting, or background worker.
The upload limit defaults to 25 MiB per request. Hosted request timeouts will
limit numerical calculations. The request helper also bounds candidate-grid
size, study sizes, reference draws, and simulated fits to keep this first
synchronous workflow manageable; an oversized request gives an input error.

The app factory `create_app(config=None)` supports ordinary Flask configuration
for deployment and testing. Server failures are logged; their details are not
inserted into browser error messages.

## Data import and interpretation

For each active file role:

1. Select a CSV, XLS, or XLSX file with a header row.
2. Choose **Read columns**. A workbook with multiple worksheets first offers
   a worksheet selector; select one and read its columns. A single worksheet
   is selected automatically and named in the preview.
3. Check the suggested column mappings. Time-course defaults use the first two
   columns; a paired IVIVC calibration file uses its first four columns, two per
   environment. Mappings can be changed to other named columns.
4. Select the uncertainty interpretation and its supplied constant or column.
   Enter actual historical precision/counts; no numerical values are assumed.
5. Choose **Preview interpreted data**. The preview records the file,
   worksheet, mappings, row counts, ranges, and uncertainty interpretation.

Data roles remain separate: prior reference data, IVIVC calibration/prediction
inputs, completed target-study observations, future individual-SD profiles,
and empirical time-scale distributions. Existing calibration files are also
available for the calibration-comparison task. Calculation imports only the
files required by the selected reference source and future measurement model.

### Reported observation uncertainty

| Supplied information | Interpreted observation uncertainty |
| --- | --- |
| Individual observations with SD | Supplied SD |
| Means with replicate SD and independent counts | SD divided by the square root of each count, once |
| Means with SEM | Supplied SEM; optional counts never divide it again |
| Unspecified uncertainty | No precision is invented; existing fitting behavior is retained |

SD/SEM must be positive and finite for retained observations. Independent counts
must be positive integers. The original measurement errors are assumed
independent. Counts should describe independent experimental units rather than
multiple dependent readings from the same unit. Supplying precision here does
not imply that every downstream IVIVC approach weights its calibration fit.

The reader rejects duplicate/empty column headers, malformed records, invalid
mapped values, and incomplete time/response pairs. Rows whose mapped time and
response are both blank are omitted and their source table-row numbers are
reported. This is handled independently for each side of paired files, allowing
unequal-length datasets padded with blank pairs. Original row order and repeated
times are preserved. No automatic aggregation, normalization, interpolation,
clipping, or uncertainty estimation occurs during import.

The preview shows at most ten retained rows; all retained rows are checked.
Response units can be recorded without converting values. Source row numbers
include the header as row 1; they are table-record numbers, rather than physical
line numbers in a CSV cell containing embedded newlines.

### Future variability profiles and empirical scaling

Future profiles map a time column and an **individual SD** column. Times must
increase strictly; finite nonnegative SDs follow the shared observation-model
rules. Planning applies replicate counts separately. The profile uses the
nearest time, extending the endpoint SDs beyond the supplied profile range.
It describes future measurement variability, not reference-curve uncertainty.

Empirical distributions map a positive in vivo/in vitro time-scale ratio and
optional weight/source-identifier columns. No weight column means equal
weights. A selected weight column must contain finite nonnegative values with
positive total weight; blank supplied weights are errors. Previews retain the
records and show supplied and normalized weights without fitting or resampling
the distribution. Time-scale definitions and time units must be compatible.

`POST /data/inspect` and `POST /data/preview` reread the supplied file in each
request. No uploaded-file cache, job, or server-side session store is created.
The browser keeps its File selections and mapping controls. Changing files,
worksheets, or interpretation marks earlier previews as out of date while
retaining their original captions. Failed or superseded requests do not replace
successful previews or apply errors to newer inputs.

The numerical import helpers live in `utilities/design_data.py`. They return
typed `TimeCourse`, `ObservationModel`, or `EmpiricalTimeScaleDistribution`
values for subsequent request handlers. Later calculations must reconstruct
and validate these values from the current uploaded file and form settings,
rather than trusting browser preview output.

## Measured-reference planning

1. Choose **Plan or extend an in vivo study**, **Measured in vivo data**, and
   supply prior reference observations. These observations inform expected
   behavior; they are not completed measurements in the proposed new study.
2. Check the data preview and uncertainty interpretation. Select the reference
   and study analysis model, optional response normalization, and optional
   one-minus response conversion. Both source values and supplied SD/SEM pass
   through the shared preprocessing once. Initial-value normalization uses the
   first retained row; place the intended baseline first. Normalization constants
   are fixed.
   Logarithmic axis scaling is unavailable for study planning.
3. Specify future individual-measurement variability as constant SD, constant
   CV with an absolute SD floor, or a time/SD profile. Use the **analyzed response
   scale after preprocessing**. Set the number of independent replicates at
   every proposed time. Historical uncertainty is never silently reused as
   future measurement variability.
4. Enter the first/last feasible times and step, a range of total distinct study
   times, and optional fixed times. The first/last range also defines the curve
   recovery assessment horizon. Every fixed time receives the planned replicate
   count. Extending beyond the historical observations relies on model
   extrapolation and is flagged in the output.
5. Optionally include evenly spaced comparisons at each size and named manual
   schedules, one per line (`Early: 0, 2, 4`). Manual schedules must satisfy the
   same allowed opportunities, fixed times, and study-size bounds. Alternative
   schedules form a separate comparison group in the inputs.
6. Choose **Calculate comparison**. Changing inputs does not trigger a
   calculation. Advanced settings control reference parameter draws, simulated
   studies per candidate, and the random seed.

The request helper in `utilities/design_planning.py` fits the measured reference,
samples joint fitted parameters into complete trajectories, and fits the study
analysis model to that ensemble. It uses the existing local I-optimal heuristic
to propose times and the existing simulation/refitting APIs to assess schedules.
The same model form is used for the measured source fit and study refits.
This is an assessment under the selected assumptions, not independent
evidence that a model or schedule is correct.

All schedules share trajectory selections and simulated measurement-error draws.
Reference-curve uncertainty and future measurement variability remain separate.
Replicate counts reduce the variability of simulated time-point means, and the
same observation model determines refit weighting. Supplied historical SEM is
not divided again. Source rows at repeated times are retained without pooling.

The comparison shows the reference curve and its parameter-uncertainty band,
expected curve recovery versus study size, explicit times and experimental-unit
counts, recovery errors, fitted-curve interval widths, and fit failures. Recovery
errors summarize successful fits; check failure rates alongside them. Small
differences may reflect Monte Carlo noise. No automatic winner, global optimality
claim, cost optimization, or universal data-adequacy cutoff is provided.

Calculation rebuilds typed inputs from the current upload and mapping on every
request; a previously generated preview is not required or trusted as input.
Neither reference construction nor simulation changes the original files.

## IVIVC-derived references

Select **IVIVC prediction**, supply the paired calibration file and target in vitro
file, and choose one approach, model where applicable, and prediction method.
The app fits that selection using the shared IVIVC pipeline. It does not select
a model or substitute a different method if the requested prediction fails.

The **study analysis model** controls schedule optimization and simulated-study
refitting. It is independent of the IVIVC model. The selected response
normalization and one-minus operation apply once to each calibration dataset
and the target dataset, using each dataset's own constants as in the IVIVC app.
Time and response logarithms remain unavailable for study planning.

- **Approach 1** uses the shared interpolation and fitted time mapping. Its
  reference uncertainty comes from that fitted relationship, without propagation
  of experimental response SD/SEM.
- **Approach 2** uses supplied calibration SD/SEM and target-input uncertainty
  through its existing response-ratio, time-constant-ratio, or inverse-response
  prediction. Target measurement errors are assumed independent of calibration
  measurement errors; overlapping experimental units are not detected.
- **Approach 3** supplies a deterministic reference, using response or time
  scaling. Its calibration and interpolation uncertainty are not quantified.

Supplied response SD/SEM is rejected for Approaches 1 and 3. Select Approach 2
to use that precision, or explicitly choose unspecified uncertainty for those
datasets when evaluating the other approaches. Nothing is silently discarded.
Future in vivo measurement variability remains a separate planning input for
every approach; it does not make an uncertain reference fully characterized.

The requested planning range, including fixed times, must fit inside the common
supported domain of the usable IVIVC trajectories. An error reports the
available range rather than moving the baseline or extending the reference.
Changing the prediction dataset or method can change this domain. Existing
method-specific extrapolation diagnostics remain relevant even within the
converted reference's numerical domain.

Results include basic source-data/fitting previews, the selected IVIVC settings,
input provenance, and the reference used for planning. These are planning aids.
Use the IVIVC app for detailed diagnostics and investigation of questionable
data or model choices.

## Empirical time scaling

Select **Empirical time scaling** and supply both an in vitro source file and
an empirical ratio file. Ratios are `tau_in_vivo / tau_in_vitro`, with optional
weights and source identifiers. Their definitions, time units, and elapsed-time
origin must be compatible with the source and planned study.

Choose the in vitro source model separately from the study analysis model.
The source is preprocessed and fitted using its supplied SD/SEM. The reference
then evaluates each complete curve as `f_in_vitro(in_vivo_time / ratio)`.
Source-parameter uncertainty and empirical ratios are assumed independent.
Every positive-weight ratio is crossed with the requested parameter draws;
the original ratio weights are preserved rather than replaced by random draws
of the empirical records. Duplicate and zero-weight records remain in provenance.

The **Reference parameter draws** setting is the number per positive-weight ratio,
so the total reference-curve count can exceed that setting. A request is limited
to 2000 complete curves and 5000 ratio records; reduce the parameter draw count
if needed. Results distinguish requested parameter draws from generated curves
and report the ratio distribution used.

Extrapolation beyond the observed in vitro source times requires explicitly
selecting **Allow source-model extrapolation**. The default rejects a horizon
that any positive-weight ratio cannot support. No empirical mass is dropped to
make the horizon fit. The results show any permitted extrapolation and keep
future in vivo measurement variability separate from source/ratio uncertainty.

All three sources enter the same schedule-generation and Monte Carlo assessment
path. Changing source or construction settings marks an earlier comparison as
out of date; requests continue to use the current files and mappings.

Source-parameter sampling and future-study simulation use separate reproducible
random streams derived from the entered seed. Schedules still share the same
simulation plan. This separation can change seeded Monte Carlo values relative
to F3, while preserving reproducibility within this version.

## Adaptive planning with completed observations

Select **This target study already has completed observations**, then import the
current cumulative target-study data. Repeated rows are retained. Each row's
supplied SD or SEM is used directly; reported counts do not divide SEM again.
Completed responses are held fixed in every simulated study and are never
generated again using the proposed future replicate count.

With a measured reference, choose one of two roles:

- **Refit completed data** uses the current study's completed observations alone
  to fit the reference. Shared preprocessing is applied once to their values and
  precision. These observations are not also used in a likelihood update.
- **Update a separate prior** fits the separately supplied prior reference data,
  then updates its complete trajectory weights using the completed target data.
  The two studies are not pooled into a common measured curve.

IVIVC and empirical-scaling references also update trajectory weights using the
completed target data. Each request reconstructs the original prior and applies
the cumulative observations once. It does not update an already updated prior.
Empirical scaling retains its ratio provenance and reports the changed weights.

For any prior-weight update, completed values and SD/SEM must **already use the
analyzed response scale of the reference**. This is the raw scale when no
response transformation is selected. A partial completed dataset is not
normalized independently by its own initial value or extrema. By contrast,
refitting the completed data as the source applies the selected preprocessing
to that source and uses those same transformed rows in study refits.

If completed precision is unspecified, explicitly choose to reuse the future
measurement model or supply SD/SEM. Reusing the future model treats those
historical rows as individual observations: their SD is evaluated at the
completed times and observed responses once, then held fixed for fitting,
weighting, and likelihood calculations. This is an additional historical-error
assumption; for constant CV it uses the observed response as a plug-in value.
Supplied completed SD/SEM always takes precedence over this fallback.

Completed times are automatically retained in every schedule. The study-size
inputs still count **total distinct times**, including completed and fixed
proposed times. The adaptive results additionally show new time counts, new
times, and new experimental-unit counts. Proposed replicate counts apply only
to new observations; unknown historical unit counts are not inferred from them.

Choose **Future times only** to exclude times at or before the latest completed
measurement. Alternatively, allow earlier unmeasured elapsed times for gap
filling when that is experimentally possible. Selecting **Latest completed
measurement** as the first opportunity anchors the candidate grid there; the
future-only rule still requires new times to be strictly later. Completed times
and the milestone baseline remain in the reference/evaluation domain rather
than being cut off by this candidate restriction.

An optional list of unavailable proposed times removes those opportunities
from the candidate grid. It cannot conflict with a fixed proposed measurement.
Completed observations remain in the analysis even if their time appears in
that exclusion list.

Weight concentration and agreement summaries help assess whether an updated
reference is still informative. Their effective draw counts are numerical
diagnostics, not numbers of independent animals or universal adequacy cutoffs.
An update cannot introduce behavior absent from the original reference curves.

## Timing windows and deadlines

Add up to eight timing requirements. For each bound, use a specified in vivo
time or a degradation milestone and reference quantile. For example, a degraded
fraction of `0.5` and quantile `0.25` uses the 25th percentile of reference T50.
The **milestone baseline time** defines the response from which degradation is
measured; specifying that time does not automatically add an experimental unit.

- **Informative point(s) in a window** requests a minimum number of distinct
  times in `(lower bound, upper bound]`, selected for information alongside the
  other schedule choices. Count either new observations only or all qualifying
  times, including completed nonbaseline observations.
- **Latest feasible point before a deadline** reserves a point as late as
  feasible before that deadline, jointly with the other latest-point requests.
  The remaining times are selected for information.

Fixed or completed observations at the designated baseline time do not count
toward window coverage. An already elapsed deadline
or a milestone not sufficiently reached within the reference horizon can make a
requirement unavailable. Increase the horizon or revise the requirement when
scientifically appropriate; the app does not discard nonreaching reference
weight or change the requested quantile to produce a deadline.

The result lists resolved bounds, generated schedules, and outcomes at every
requested study size. **Infeasible**, **unresolved**, and **search limit reached**
have different meanings. Reaching a search limit is not evidence that no
schedule exists. If no suggested schedule can be produced, the reference and
requirement diagnostics remain available without invented recovery metrics.
Evenly spaced and manual comparisons are assessed separately and labeled with
their requirement coverage; a comparison need not satisfy the requirements.

This frontend currently combines timing requirements with future-only adaptive
planning. Gap filling can be assessed without timing requirements; their
combination is rejected explicitly because the shared timing scheduler defines
new points as occurring after the latest completed time.

## Checks

```bash
python -m pytest tests/test_design_app.py
python -m pytest tests/test_design_data.py tests/test_design_data_routes.py
python -m pytest tests/test_design_planning.py tests/test_design_planning_results.py tests/test_design_planning_routes.py
python -m pytest tests/test_design_reference_*.py
python -m pytest tests/test_design_adaptive_*.py tests/test_design_timing.py
node tests/test_design_frontend.cjs
node tests/test_design_data_frontend.cjs
```

The frontend checks use Node's built-in test runner and a small DOM fixture;
they require no additional JavaScript packages. Existing numerical and IVIVC
tests should continue to run in the usual application environment.
