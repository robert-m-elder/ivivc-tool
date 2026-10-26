# Regenerate the toy data

From the project directory, using the app's Python environment:

```bash
python tools/generate_toy_data.py
```

This overwrites `toy-data.xlsx`. To write elsewhere:

```bash
python tools/generate_toy_data.py --output /path/to/examples.xlsx
```

The script calls the registered model functions directly; it does not fit models
or run the web app. It uses NumPy and openpyxl from the app's existing requirements.
The workbook has smooth and noisy sheets for Exponential, Stretched Exponential,
Double Exponential, Logistic, Exponential + Logistic, Linear + Power Decay, and
Logistic + Logistic. Sheet names follow the registry's display names.

Each sheet retains the input format: in vitro time, in vitro response, in vivo
time, in vivo response (`time1, value1, time2, value2`). Shorter profiles have
blank trailing cells. Values are stored as numbers, without Excel formulas or
a recalculation requirement. The original five examples retain their smooth
parameters and time grids. The new Logistic + Logistic example uses denser grids
to show both transitions, with the in vivo profile slowed by a factor of 1.5.

Edit `EXAMPLES` in the script to change parameters, grids, or seeds, or to add an
example. Parameters use the backend function's keyword names. Each profile has
an independent fixed random stream, so changing example order does not change
its noise. Noisy sheets add Gaussian noise with SD 0.03, leave the time-zero
value unperturbed, then clip responses to [0, 1]. This illustrative noise is
separate from the app's uncertainty calculations. The regenerated noisy values
replace the old workbook's unrecorded noise realizations. Regeneration reproduces
cell values; XLSX metadata or archive timestamps can differ.
