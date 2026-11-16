# Study-design Flask frontend

The study-design interface has a separate Flask entry point and shares the
repository's Python environment with the IVIVC app. The IVIVC entry point,
templates, and numerical backend are unchanged.

## Run locally

From the repository root, using the project's Python 3.12 environment with
`requirements.txt` installed:

```bash
python design_app.py
```

Open `http://127.0.0.1:5001`. The separate port permits running the IVIVC app
on its usual port at the same time. Debug mode is off by default.
The existing double-click launchers start the IVIVC app; use the command above
to start the design app. Both use the same installed Python dependencies.

For a hosted installation, use the hosting service's normal WSGI configuration
with `design_app:app`, or import `app` from `design_app` as its application
object. The built-in development server is for local use. A future landing
page can link to the two applications without combining their entry points.

The checkout must be importable by the WSGI worker. Templates and static files
are located relative to `design_app.py`, so the worker's current directory need
not be the checkout. Importing `design_app:app` creates the application without
starting the development server. `create_app(config)` is also available for a
host's application factory or a deployment-specific wrapper.

A deployment may serve the app at `/` or mount it at a prefix such as `/design`.
For a prefixed mount, the WSGI server or middleware must set `SCRIPT_NAME` to the
mount prefix and pass the remaining route in `PATH_INFO`. Flask then generates
prefixed links for scripts, styles, uploads, generation, and calculation. Merely
adding a prefix to the public URL without the corresponding WSGI mount is not
sufficient. No proxy-header middleware is enabled automatically.

Allow the web server's request body limit to accommodate the configured Flask
upload limit, including multipart overhead, and allow enough request time for
the selected numerical workload. All work finishes in the same request; a
server timeout does not create a result that can be retrieved later. Reducing
draws, simulated studies, candidate designs, or grid size can shorten a trial
calculation. No session affinity is needed between requests: required files
and settings are submitted each time.

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
- `POST /generate-candidates` imports the varied calibration environments and
  returns proposed curve-recovery schedules with a review preview. It uses the
  same generation path as calibration comparison, without calibration
  simulations or target assessment.
- **Calculate comparison** supports new studies and additions to completed
  observations, using measured reference data, an IVIVC prediction, or empirical
  time scaling. Timing windows and deadlines can constrain proposed future times.
  The separate **Compare calibration sampling** task compares candidate in vitro
  and in vivo calibration studies through the shared IVIVC workflow. Report
  generation remains unavailable.

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

### Interactive reference and study-size plots

The main in vivo reference plot uses Plotly for hover values, zoom and pan,
legend toggles, and SVG figure download. Its curves, observations, and weighted
pointwise uncertainty range are the same numerical results used by the static
plot. The plot remains on the analyzed response scale. Changing a plot view
does not rerun the analysis or change the submitted inputs. Double-clicking the
plot or using its reset control restores the axis range.

**Curve recovery versus study size** uses the same interactive renderer and
the same mean/90th-percentile IRMSE values as its static fallback and performance
table. The horizontal axis counts all distinct proposed times for prospective
planning and only new times for adaptive planning. Manual comparisons remain
separate points. Circle/diamond markers and hover labels distinguish means
from 90th percentiles, including when only one study size is assessed.

The design page loads the versioned Plotly 3.0.1 script once from
`https://cdn.plot.ly/plotly-3.0.1.min.js`, matching the version used by the IVIVC
results page. The browser needs access to that CDN for interactive plotting.
If the library cannot load or a plot cannot be rendered, the reference and
study-size plots retain their static fallbacks with a short status message.
Local serving can be added later without changing the numerical result format.
Source-fit and calibration plots retain their existing static presentation.

The small shared renderer is in `static/js/design-plots.js`. Figure payloads
contain ordinary numeric `data` and `layout` values from
`utilities.design_plotly.make_figure`, serialized separately from the HTML.
Nonfinite values are represented by gaps. The workspace initializes figures
after a successful result insertion, resizes them when Comparisons becomes
visible, and releases replaced figures. Switching workspace views retains the
existing plot state; a new calculation starts with a new figure.

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

## Calibration-sampling comparison

Choose **Compare calibration sampling** to assess how proposed measurements may
affect an IVIVC relationship or target prediction. The inputs have four roles:
fixed existing calibration observations, assumed generating trajectories,
future calibration measurement variability, and optional independent target
in vitro observations. They are not interchangeable.

1. Choose whether in vitro, in vivo, or both environments receive proposed
   measurements. For each varied environment, choose a prospective study or
   additions to existing calibration data. The other environment remains fixed
   at its uploaded calibration observations. Existing observations are held
   fixed in every simulation; additions never reduce historical SEM.
2. Specify expected behavior using a fitted source curve for each varied
   environment. Its source may be the existing calibration data or a separate
   uploaded time course. Select its generating model independently of the
   IVIVC model. A fixed environment may omit a generating reference when only
   precision and calculation success are needed; error or coverage calculations
   requiring that missing reference remain unavailable.
3. Specify individual future measurement variability in each varied environment
   using constant SD, constant CV with an SD floor, or a time/SD profile. Values
   use the analyzed response scale. Existing reported SD/SEM takes precedence.
   If old precision is missing and needed for weighted additions or schedule
   generation, supply it or explicitly assume the future error model for those
   historical rows. That assumption treats unqualified rows as individuals,
   evaluates SD at their observed responses/times once, and holds it fixed.
4. Choose Approach 1 or 2, the IVIVC model, and the prediction method. Define
   relationship evaluation times or response levels as appropriate, and
   optionally supply a separate target in vitro dataset. The same target inputs
   are used for every candidate and experiment.
5. Add named candidates, specifying proposed times and independent replicates
   per time in each varied environment. Alternatively, generate a candidate
   schedule from a first/last/step grid and a number of proposed times. Use
   **Generate and preview candidate times** to review the proposed times before calculating
   the full comparison, or calculate directly to generate and assess them
   in one request.

There are at most six candidates per request. Manually specified times may
include times already measured: the requested replicates are additional
independent observations, not replacements for the old measurements. In
additions mode, an empty manual schedule requests no additional measurements
in that environment. This permits an existing-data-only baseline when both
environments already have calibration observations. A prospective environment
must receive proposed measurements.

Synchronous requests are limited to 200 simulated experiments per candidate,
200 source parameter draws, 500 propagated interval draws, and 200 optional
target observations. Combined simulation/propagation limits can require smaller
settings when several candidates or many output coordinates are requested.

Generated schedules use the existing local information criterion for expected
**curve recovery**, followed by calibration assessment. Existing observations
contribute their actual precision and remain fixed. Generated additions use
previously unmeasured times; use manual schedules to request more replicates at
existing times. With both environments varied, their candidate schedules are
generated separately. This is not optimization specifically for IVIVC quality
or joint optimization of a complete paired study.

### Reviewing generated candidates

The separate generation action fits the selected generating sources and uses
the same curve-recovery optimizer as the full comparison. It does not simulate
new calibration datasets, fit an IVIVC, or assess target predictions. Only
inputs for the varied environments are needed; the fixed environment, target
uploads, relationship evaluation settings, and simulation/interval-draw
settings can be completed later. Source parameter draws and the random seed
still apply. All active candidates in the varied environments are interpreted
so generation uses the same source domains as the subsequent comparison.

The preview reports proposed times, new replicate counts, existing times held
fixed, source and measurement assumptions, and feasible generation grids.
Existing data contribute their supplied precision; the proposed replicate
count does not replace that precision. With both environments varied,
generation runs independently in each environment.

Use **Copy to editable schedule** to transfer a candidate's generated times to
its manual inputs. This is an explicit schedule snapshot: later source or
measurement changes do not silently optimize it again. Leave the candidate
in generated mode when it should be regenerated during calculation. After
changing inputs, the old preview remains visible and is marked out of date;
generate again before copying it. A failed generation preserves both the
previous preview and the last successful comparison. No preview is required
or trusted as numerical input by the calculation endpoint.

### Response scales and uncertainty

Initial-value or min-max normalization uses constants calculated once from each
environment's selected source, or its existing data when no generating curve is
requested. The same affine transformation is applied to its existing values,
source values, and SD/SEM. Optional one-minus reversal is also fixed. Those
constants remain unchanged across candidates and simulated experiments; each
newly simulated calibration dataset is not normalized by its own noisy extrema
or baseline. Optional target data use their own fixed preprocessing constants,
as in the IVIVC app. Logarithmic axis scaling is unavailable in this workflow.

The numerical adapter calls the shared calibration-simulation and IVIVC APIs.
Each experiment samples complete generating curves, simulates only the proposed
calibration measurements, assembles them with the fixed historical rows, and
repeats the selected IVIVC fit and prediction. New mean observations have
SD divided by the square root of their own replicate count; already supplied
historical SEM is never divided again. Independent raw observations and
reported means retain the existing backend's row/precision conventions.

Approach 2 uses observation uncertainty in its fitted curves and propagates
fitted and supported target-input uncertainty. Approach 1 repeats the ordinary
interpolation and time-to-time fit used by the IVIVC app. Measurement variability
affects the simulated datasets, but supplied response SD/SEM does not weight
that mapping fit or enter its regression-parameter intervals. Its variation
across simulated studies is reported separately from those fitted intervals.
Target response uncertainty is not supported for Approach 1.

Source-curve uncertainty is distinct from future calibration measurement noise.
Separate environment reference ensembles and measurement errors are treated as
independent. Candidates share trajectory selections and standard-normal errors
at common times, including shared replicate draws when counts differ. Future
calibration noise does not add scatter for individual downstream in vivo target
measurements to the target prediction intervals.

### Reading calibration results

Results retain explicit proposed times/counts, fixed-data and source provenance,
normalization constants, generating-model assumptions, and whether schedules
were specified or generated. The comparison reports fit success, point and
interval availability, interval widths, and error/coverage where generating
truth is available. Coordinates and units identify whether an output is an
in vivo response, an in vivo time, or a dimensionless ratio.

Widths and errors are conditional on the corresponding calculations being
available, with their counts retained. Coverage uses the backend's stated
denominator: every experiment with known truth, including unavailable intervals
as not covering; conditional coverage among available intervals is separate.
Missing truth produces unavailable error/coverage, not zero error. Calculation
failures remain in success and availability denominators, with stage details.
For example, a finite reference time grid does not define a time-constant
integral to infinity, and uncertain target observations do not by themselves
specify the latent target truth. Precision can still be reported when those
error or coverage quantities are unavailable.

Generating curves are assumptions, not externally observed truth for a future
study. A narrow interval or favorable simulated error does not establish an
overall best model, transferability, or sufficient calibration data. Compare
the tradeoffs and check scientific plausibility. Small differences can reflect
Monte Carlo variation; advanced controls set source parameter draws, propagated
interval draws, simulated experiments, and the seed.

## Schedule inspection data and endpoint ranges

The ordinary planning milestone table includes the reference median time,
central 95% range, and reference weight reaching T50 or T90 by the assessment
horizon. The bounds are the weighted 2.5th and 97.5th percentiles of crossing
times, using the same baseline and shared crossing-time API as timing
requirements. Non-reaching trajectories retain their weight beyond the
horizon. An unresolved percentile reads **Beyond the assessment horizon**;
it is neither discarded nor replaced by the last sample or assessment time.
An invalid baseline leaves the milestone unavailable. One usable reference
trajectory supplies a median when resolvable, but no sampled uncertainty range.

Each calculated planning result also contains compact numerical inspection
data. Select up to three schedules in the performance table to inspect their
timelines. The first suggested schedule is selected initially; this is a display
default rather than a ranking. All selected schedules share one time axis,
with a separate vertically stacked row for each schedule. Remaining selected
schedules keep their colors when another schedule is added or removed. Once
three are selected, deselect one before choosing another. Clearing all
selections hides the timelines until a schedule is chosen again.

The timeline uses open circles for completed observations, squares for fixed
proposed times, diamonds for explicitly selected latest-before anchors, and
filled circles for other proposed times. An anchor takes diamond precedence
when a point is also fixed; hover text and **Exact times and explanations**
retain all roles and overlapping timing contributions. The baseline remains a
vertical guide even when it is not sampled. Latest completion is a separate
guide where applicable, including gap filling with earlier new opportunities.
Shading shows resolved windows, with deadline lines for latest-before rules.
Exact requested/resolved bounds and statuses remain in the timing-requirement
table; unresolved bounds are not drawn at the study end. A time inside a
window is not thereby individually mandated.

Selection changes use the current result without further requests or
simulations, and never mark the analysis inputs changed. The comparison's
existing settings and out-of-date behavior apply to the inspection data as
well. Switching workspace views preserves the current selection; a successful
new calculation starts a fresh selection. A failed calculation or setup review
preserves it. The timeline supports hover, zoom/reset, and SVG download through
the shared Plotly renderer. If interactive rendering is unavailable, selection
still updates the exact-time tables, which open automatically. The other
performance results remain available. On narrow screens, the timeline scrolls
horizontally; its explanatory key wraps below the chart rather than clipping
inside it.

The same selection controls the overlaid recovery plots. **Recovered response**
shows the median recovered curve for each schedule alongside the central
reference. **Recovery error** shows summaries of recovered minus each simulated
study's own generating curve, with a zero-error guide. Errors are paired before
summarizing; subtracting the central reference from a median recovered curve
does not produce that error summary. Values use the analyzed response scale.

Each selected schedule shows its pointwise central 95% recovery range when
available. The plot legend identifies the central reference and recovered
curves. Clicking a schedule's legend entry hides or shows its median and range
together; numerical tables and availability counts still follow the schedule
selection in the performance table. Changing the view or selection restores
the current curves and their available ranges together. These ranges reflect
the sampled reference trajectories and future
measurement assumptions across successful simulated studies; they describe a
different quantity from individual-fit confidence intervals or a reference-only
range. They are not simultaneous bands. A singleton supplies a median without
a plotted range. Fit success/failure counts and pointwise usable counts remain
visible even with a curve hidden through the legend, and missing values stay unavailable.
Curve portions outside a schedule's sampled time range rely on the model's
extrapolation; the timeline shows that sampling coverage.

Expand **Illustrative simulated study** to see the same predetermined first
simulation for the selected schedules. It always shows response, independently
of the summary view. Its generating true curve appears once, alongside
successful recovered curves and the means of newly simulated replicates.
Original completed rows appear once with their unchanged precision and reported
counts. A failed fit retains its explanation and observations; another study is
not substituted. This example has no study-number selector or fitted confidence
band because pointwise fitted bounds are not retained by the simulation output.
The existing fitted mean-curve interval-width metric remains separate.

**Numerical recovery values** and **Numerical illustrative-study values** expose
the retained grid values, ranges, counts, and observations. They are populated
only when opened and include at most the selected schedules. If interactive
plotting is unavailable, the corresponding numerical table opens automatically.
Selection, response/error views, legend controls, and the collapsed example all
operate on retained results without calculation requests. The selected view
survives schedule selection changes and workspace navigation. A successful new
comparison starts fresh; failed requests retain the prior display.

The shared selection and color state is available through
`DesignScheduleInspection.getSelection(resultElement)` and the display-only
`design-schedule-selection-changed` event. The shared renderer's
`DesignPlots.update(wrapper, figure)` serializes plot updates and coalesces
rapid changes to the latest selection. Result replacement disposes the
inspection controls and plots before initializing the new comparison.

The result contains an inert `application/json` element marked
`data-planning-inspection`, serialized with Jinja's HTML-safe `tojson` filter.
Its versioned payload has these sections:

- `reference` retains the central curve and existing pointwise weighted
  reference range, on their original time/response coordinates and analyzed
  response scale, separately from simulated recovery spread.
- `schedule_data` retains numeric times, point roles, baseline guides,
  resolved windows/deadlines, and unavailable schedule-generation outcomes.
  A point may be completed, user-fixed, or a retained latest-before anchor,
  and may contribute to multiple windows. Window contributions do not imply
  that an individual time was mandated. A manual or evenly spaced comparison
  does not inherit certified latest-before attribution from a suggested
  schedule with the same times. Completed measurements and planned new
  replicates remain separate. The milestone baseline guide and explicitly
  fixed timing-baseline points have separate flags: only the latter are
  excluded by the timing backend from window counts. Latest completed time,
  first sampling opportunity, evaluation range, and timing-context start are
  retained separately, so gap filling does not hide earlier new opportunities.
- `recovery` contains pointwise 2.5th, 50th, and 97.5th percentiles for recovered
  responses and recovery errors, with total, successful, failed, and usable
  counts. Errors are recovered minus that simulation's own true response,
  calculated before summarization. Simulations already sample the reference
  weights; these summaries use equal simulation weights and NumPy's linear
  empirical quantile method. They describe variation across successful
  simulated studies, separately from fitted-model confidence intervals and
  reference uncertainty. Fewer than two usable values leave the range
  unavailable; a singleton median remains available. Missing coordinates are
  JSON `null`, preserving gaps rather than substituting zero.
- `recovery.shared_example` identifies the predetermined first simulated
  study, using the same shared trajectory selection for every schedule. A
  failed fit stays failed; another successful study is not substituted. Each
  schedule retains that study's true/recovered curves and observed time means.
  Those means summarize the simulated replicates, rather than displaying all
  individual replicate observations. Original completed rows, their SD/SEM,
  and reported counts remain available separately. Pointwise fitted confidence
  bounds are explicitly unavailable because the simulation result does not
  retain them; the existing mean fitted interval width remains separate.
- `endpoints` retains numeric crossing quantiles with explicit availability
  statuses, so unresolved horizon bounds do not require nonfinite JSON values.

Performance rows and both schedule inspection sections share stable
`schedule-N` identifiers within the calculated result. Only summary curves
and one example per schedule are included; the browser does not receive all
simulated studies. Presentation does not refit curves or rerun simulations.

## Checks

Install the test dependencies in the same environment used to run the app:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/test_design_*.py
python -m pytest tests/test_study_design*.py
node --test tests/test_design_frontend.cjs tests/test_design_data_frontend.cjs tests/test_design_plots.cjs tests/test_design_timeline_figure.cjs tests/test_design_schedule_frontend.cjs tests/test_design_recovery_figure.cjs tests/test_design_recovery_frontend.cjs
```

The frontend checks use Node's built-in test runner and a small DOM fixture;
they require no additional JavaScript packages. Existing numerical and IVIVC
tests should continue to run in the usual application environment.

The integration tests run small real numerical workflows through Flask's test
client: import and uncertainty preview, schedule generation, adoption of
explicit times, calibration comparison, completed-only adaptive planning, and
switching reference sources. They check that each request uses its own current
uploads and that reported SEM is not divided by a new replicate count.
The deployment tests exercise a real WSGI prefix mount, static files, all POST
upload limits, JSON errors, and import/startup from a different working
directory. They use the WSGI application directly; they do not start a network
server or verify a particular hosting provider. Run the repository's full test
suite with `python -m pytest` when checking integration with other app changes.

### Browser and hosting smoke checks

Before using a deployment, run these checks in an actual browser against that
installation. DOM-fixture tests do not replace this check of both browser
controllers working together.

1. Open the workspace and check that styles, the application scripts, and the
   Plotly CDN script load.
   If using a prefix, complete the remaining checks at that prefixed URL.
   After calculating a comparison, check hover values, zoom/reset, legend
   toggles, and figure download on the reference plot. Switch between Inputs
   and Comparisons and resize the browser; the plot should fit the visible
   panel. Recalculate to check that the replacement figure works. With CDN
   access blocked, the static reference plot and other results should remain
   available.
2. Import a small CSV and a workbook with multiple sheets. Read the columns,
   select the sheet and mappings, and preview mean/SD/count and mean/SEM data.
   Confirm the selected files remain available when changing workspace views.
3. Calculate a small measured-reference study comparison. Check the endpoint
   median and central 95% range, including a milestone that some or all
   reference trajectories do not reach by the horizon. Such bounds should
   state **Beyond the assessment horizon**. Select three schedules and verify
   stacked timeline rows, a shared time axis, point-role hover text, and exact
   times. Deselect and reselect schedules; the remaining colors should stay
   consistent. Clear all selections, then choose one again. Check that these
   actions leave the comparison/input status unchanged and make no new
   calculation request. With CDN access blocked, selection and exact-time
   tables should still work. Switch the recovery plot between response and
   error, use the legend to hide/show a schedule's curve and range together,
   and inspect numerical values. Check hover/zoom/download on the study-size
   plot as well. Expand
   the shared illustrative study, confirming that a failed example remains
   failed and that original completed observations appear once. With CDN
   access blocked, the recovery tables should open automatically and continue
   to follow the same selection. Exercise an IVIVC
   reference and an empirical time-scale reference with their own input files.
   Add completed observations and one timing requirement, then recalculate.
4. In calibration comparison, generate a candidate, copy it to editable times,
   and assess it alongside an explicit candidate. Check both Approach 1 and
   Approach 2 when they are to be used. Varying both environments should retain
   their separate source and measurement assumptions.
5. After a successful calculation, change an input, replace a file, and submit
   an invalid request. Earlier output should remain visible with its original
   provenance and an out-of-date label. Stale generated times must not be
   copied into the current inputs. Check that validation links reach the
   relevant control and that requests work again after correcting the inputs.
6. Open a second tab to confirm it starts with a separate draft. Refreshing the
   original tab should start a new workspace; uploads and results are not
   restored. If running the IVIVC app too, verify it remains available at its
   own entry point and port.

Reports remain a disabled placeholder. Long-running jobs and saved workspaces
are outside this synchronous interface.
