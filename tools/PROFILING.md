# Temporary performance investigation

This launcher adds timers only when explicitly run. It does not change fitting,
uncertainty draws/seeds, integration, inverse searches, or the normal launchers.
It requires the app's existing Python dependencies; no extra package or Node.js
is needed. The server runs locally, without Flask's reloader, one request at a
time so analyses do not compete with one another.

From the app's top-level folder, using its Python environment:

```bash
python tools/profile_app.py
```

Open <http://127.0.0.1:5000/> and reproduce the slow analysis with your usual
settings. Stop with Control-C. Use `--port 5001` if 5000 is occupied.

For each analysis submission, the console shows the slowest timed calls and
response size. The ignored `profiling-output/` folder receives a CSV of stage
timings and JSON request metadata. Word/Excel export requests are also measured.
Input values, fitted parameters, filenames, and uploaded file contents are not
written to these timing files. Only model identifiers and a user-supplied label
are recorded alongside timing/request/runtime information.

The CSV records the calling path, model, number of calls, total inclusive time,
exclusive time, longest call, and exceptions. Plot labels distinguish the three
Approach 2 prediction methods. Inclusive times include nested work: **do not add
them together**. Exclusive time subtracts other timed calls only; it is not a
complete line-by-line breakdown. Report assembly is grouped by its calling path
and counts all model/method report variants built during the submission.

The JSON's `application_seconds` covers upload parsing, analysis, plot/table and
report preparation, and template rendering until Flask returns the response.
`response_bytes` is the uncompressed response body size. Timing-file writes,
network transfer, JavaScript execution, and browser rendering are excluded from
the application time. `Server-Timing` also exposes time up to response headers
in the browser's Network panel.

## Suggested comparison

Use a representative slow fitting dataset and keep models, preprocessing, CV,
grid search, diagnostic options, and all other settings identical.

1. Run without a prediction dataset, then with prediction data.
2. Compare in-range prediction responses against extrapolated responses, with
   the same number of prediction rows.
3. Compare a few prediction rows against many, and repeated responses against
   distinct responses if relevant to the dataset.
4. If several models are selected, inspect their individual prediction timings;
   then rerun the slowest model alone to simplify the detailed profile.
5. Repeat each case after a warm-up run. Compare repeated runs, not a cold first
   run against an already warmed-up one. Record settings with your results.

Start with ordinary timers. Then rerun a slow case with detailed profiling:

```bash
python tools/profile_app.py --cprofile --label slow-case
```

This also saves `.prof` statistics and a readable `.txt` listing the top functions
by cumulative/self time and the callers of tau evaluation, inverse mapping,
curve-draw evaluation, and HTML serialization. The usual 1,000-draw tau-ratio
calculation makes 2,002 `evaluate_tau` calls per eligible model, plus the smaller
fit-time tau calculations. `quad` calls may be fewer if validity checks return
early. The inverse call counts expose repeated searches across draws. Function
statistics aggregate all models in a request; select one model to isolate them.

Detailed profiling adds overhead, especially to Python-heavy loops. Use ordinary
timer runs for approximate wall times and detailed runs to locate the work;
**do not compare their elapsed times as a speedup/regression estimate**. Grid
search timers include time waiting for worker processes, but cProfile does not
capture code executing inside those workers.

If server times are short but the page stays busy, use the browser's Network and
Performance panels to separate response transfer, scripts, and rendering. These
files alone cannot measure browser rendering or establish how much slower the
app became than an earlier revision.

This can remain on a temporary branch. To remove it, delete this script and
guide and remove `profiling-output/` from `.gitignore`; no production code was
edited. The profiling tests can also be removed.
