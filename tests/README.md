# Report regression checks

With the app requirements installed, run from the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests cover complete numerical tables, method availability, mathematical
equations, preprocessing and CV metadata, prediction/error-bar fidelity, and
request-to-report handling. Full request and plotting checks explicitly skip
when the corresponding app dependencies are unavailable.

For the browser payload functions, using Node.js:

```bash
node tests/test_report_export.cjs
node tests/test_accessibility_descriptions.cjs
```

The browser check verifies that all rows are retained, author text is preserved,
selection rationales stay with the correct report, and missing images remain
explicit. The accessibility check distinguishes parameter-uncertainty intervals
from descriptive CV-refit shading. Neither check requires installing JavaScript packages.

## Response-value ratio uncertainty

Run the full suite with `python -m pytest`. The targeted checks are:

```bash
python -m pytest tests/test_value_ratio_uncertainty.py tests/test_value_ratio_outputs.py tests/test_report_metadata.py tests/test_report_content.py
```

These cover parameter-only percentile propagation, non-unit ratio scaling,
pointwise validity, denominator safeguards, reproducibility, response-unit
invariance, and fidelity of plots/downloads/reports. They do not establish
frequentist coverage of the local parameter approximation on real datasets.
