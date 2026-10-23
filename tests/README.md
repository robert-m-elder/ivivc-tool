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
```

The browser check verifies that all rows are retained, author text is preserved,
selection rationales stay with the correct report, and missing images remain
explicit. It does not require installing JavaScript packages.
