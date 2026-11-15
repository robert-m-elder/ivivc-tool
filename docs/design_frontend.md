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
  completes measured-reference planning synchronously. It returns server-rendered
  comparison HTML or field-linked errors.
- **Calculate comparison** currently supports new in vivo studies using measured
  reference data. IVIVC-derived references, empirical scaling, completed-study
  updates, and calibration comparisons retain import/setup review but are not
  yet connected to numerical calculation in this frontend. Report generation
  remains unavailable.

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
available for the calibration-comparison task. Numerical planning currently
uses only the measured reference and, when selected, its future SD profile.

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

## First complete planning workflow

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
The same model form is used for the source fit and study refits in this first
workflow. This is an assessment under the selected assumptions, not independent
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

## Checks

```bash
python -m pytest tests/test_design_app.py
python -m pytest tests/test_design_data.py tests/test_design_data_routes.py
python -m pytest tests/test_design_planning.py tests/test_design_planning_results.py tests/test_design_planning_routes.py
node tests/test_design_frontend.cjs
node tests/test_design_data_frontend.cjs
```

The frontend checks use Node's built-in test runner and a small DOM fixture;
they require no additional JavaScript packages. Existing numerical and IVIVC
tests should continue to run in the usual application environment.
