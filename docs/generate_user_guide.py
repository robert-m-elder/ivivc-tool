#!/usr/bin/env python3
"""Generate README.md and LaTeX user-guide source from reusable app descriptions.

Run from the repository root:

    python utilities/generate_user_guide.py

The generator reads user-facing descriptions from utilities/descriptions.py,
which can load routine wording changes from utilities/descriptions.yaml, and
combines them with the live model, metric, and preprocessing registries when
available. Keep routine wording changes in utilities/descriptions.yaml where
possible, then rerun this script to refresh README.md and docs/user_guide.tex.
The PDF is generated separately from LaTeX source.
"""
from __future__ import annotations

from html import unescape
from pathlib import Path
from textwrap import dedent
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from metrics import metrics
except Exception:  # pragma: no cover - documentation fallback
    metrics = {}

try:
    from models import approaches, models
except Exception:  # pragma: no cover - documentation fallback
    approaches = {
        "approach1": {"display_name": "Approach 1", "models": []},
        "approach2": {"display_name": "Approach 2", "models": []},
        "approach3": {"display_name": "Approach 3", "models": []},
    }
    models = {}

try:
    from preprocessing import preprocessing_options
except Exception:  # pragma: no cover - documentation fallback
    preprocessing_options = {"normalization": {}, "scaling": {}, "interpolation": {}}

from utilities.descriptions import (  # noqa: E402
    ANALYSIS_CONFIG_LABELS,
    ANALYSIS_PARAMETER_DESCRIPTIONS,
    APPROACH_DESCRIPTIONS,
    CROSS_VALIDATION_DESCRIPTION_HTML,
    CV_SCHEME_DESCRIPTIONS,
    CV_SCHEME_LABELS,
    GRID_SEARCH_DESCRIPTION_HTML,
    METRIC_DESCRIPTIONS,
    PREPROCESSING_DESCRIPTIONS,
    RELATIVE_MODEL_EVIDENCE_TEXT,
    TOOL_CAPABILITY_SUMMARY,
    TOOL_PURPOSE_TEXT,
)

README_PATH = ROOT / "README.md"
GUIDE_TEX_PATH = ROOT / "docs" / "user_guide.tex"
APPENDICES_DIR = ROOT / "docs" / "appendices"
APP_DESCRIPTION = TOOL_PURPOSE_TEXT


def html_to_text(value: str) -> str:
    """Convert short HTML description fragments to plain text."""
    if not value:
        return ""
    text = value.strip()
    text = re.sub(r"</p>\s*<p>", "\n\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<li>", "- ", text)
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"</?(ul|ol)>", "\n", text)
    text = re.sub(r"<strong>(.*?)</strong>", r"\1", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def latex_escape(value: object) -> str:
    """Escape text for LaTeX while preserving a few common scientific symbols."""
    text = "" if value is None else str(value)

    replacements = [
        ("R²", r"\(R^2\)"),
        ("R^2", r"\(R^2\)"),
        ("r²", r"\(r^2\)"),
        ("r^2", r"\(r^2\)"),
        ("AICc", r"AIC\textsubscript{c}"),
        ("Δ", r"\(\Delta\)"),
        ("±", r"\(\pm\)"),
        ("≥", r"\(\ge\)"),
        ("≤", r"\(\le\)"),
        ("–", "-"),
        ("—", "-"),
        ("×", r"\(\times\)"),
    ]

    protected: list[str] = []
    for old, new in replacements:
        if old in text:
            placeholder = f"@@LATEX{len(protected)}@@"
            protected.append(new)
            text = text.replace(old, placeholder)

    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")

    for i, new in enumerate(protected):
        text = text.replace(f"@@LATEX{i}@@", new)

    return text


def latex_paragraphs(value: str) -> str:
    """Convert plain text with paragraph breaks to LaTeX paragraphs."""
    text = html_to_text(value) if "<" in value and ">" in value else value.strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(latex_escape(p) for p in paragraphs) + "\n\n"


def latex_itemize(items: list[str]) -> str:
    lines = [r"\begin{itemize}"]
    for item in items:
        lines.append(rf"\item {latex_escape(item)}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n\n"


def latex_code(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{", r"\{").replace("}", r"\}")
    text = text.replace("_", r"\_")
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    return rf"\texttt{{{text}}}"


def latex_section(level: int, title: str) -> str:
    commands = {1: "section", 2: "subsection", 3: "subsubsection"}
    command = commands.get(level, "paragraph")
    return rf"\{command}{{{latex_escape(title)}}}" + "\n\n"


def latex_table(
    headers: list[str],
    rows: list[list[object]],
    widths: list[str] | None = None,
    raw_columns: set[int] | None = None,
) -> str:
    """Create a compact longtable with paragraph columns.

    Columns listed in raw_columns are inserted without escaping. Use this only
    for LaTeX fragments generated by this script, such as model equations.
    """
    raw_columns = raw_columns or set()
    if widths is None:
        if len(headers) == 2:
            widths = ["0.24\\textwidth", "0.68\\textwidth"]
        elif len(headers) == 3:
            widths = ["0.20\\textwidth", "0.25\\textwidth", "0.47\\textwidth"]
        elif len(headers) == 4:
            widths = ["0.17\\textwidth", "0.17\\textwidth", "0.20\\textwidth", "0.38\\textwidth"]
        else:
            width = f"{0.92 / max(len(headers), 1):.2f}\\textwidth"
            widths = [width for _ in headers]

    spec = "@{}" + "".join(f">{{\\raggedright\\arraybackslash}}p{{{w}}}" for w in widths) + "@{}"
    lines = [rf"\begin{{longtable}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(rf"\textbf{{{latex_escape(h)}}}" for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(" & ".join(rf"\textbf{{{latex_escape(h)}}}" for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for row in rows:
        cells = [str(cell) if i in raw_columns else latex_escape(cell) for i, cell in enumerate(row)]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n\n"


def md_generated_notice() -> str:
    return "<!-- Generated by utilities/generate_user_guide.py. Edit routine wording in utilities/descriptions.yaml where possible. The user guide is generated as LaTeX source in docs/user_guide.tex; long-form appendices are sourced from docs/appendices/; references are maintained in docs/references.bib. -->\n\n"


def tex_generated_notice() -> str:
    return "% Generated by utilities/generate_user_guide.py. Edit routine wording in utilities/descriptions.yaml where possible. This is a content-only LaTeX file; include it from a top-level document. Long-form appendices are sourced from docs/appendices/; references are maintained in docs/references.bib.\n\n"


def preprocessing_label(category: str, key: str) -> str:
    item = PREPROCESSING_DESCRIPTIONS.get(category, {}).get(key, {})
    return item.get("display_name") or item.get("short_name") or key.replace("_", " ").title()


def preprocessing_description(category: str, key: str) -> str:
    return PREPROCESSING_DESCRIPTIONS.get(category, {}).get(key, {}).get("description", "")


def metric_description(metric_key: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric_key, {}).get("description", "")


def direction_label(direction: str) -> str:
    if direction == "higher":
        return "Higher is better"
    if direction == "lower":
        return "Lower is better"
    return direction or ""


def model_approach_labels(model_key: str) -> str:
    labels = []
    for approach_key, approach_info in approaches.items():
        if model_key in approach_info.get("models", []):
            labels.append(approach_info.get("display_name", approach_key))
    return ", ".join(labels)


def model_equation(model_key: str, model_info: dict) -> str:
    equation = model_info.get("latex_equation")
    if equation:
        return rf"\({equation}\)"
    return "Not specified"


def model_description(model_key: str, model_info: dict) -> str:
    description = model_info.get("description")
    if description:
        return description
    return "General empirical model used to describe monotonic or nonlinear relationships when supported by the data."


def analysis_parameter_description(key: str) -> str:
    return ANALYSIS_PARAMETER_DESCRIPTIONS.get(key, "")


def build_readme() -> str:
    capabilities_md = "\n".join(f"        - {item}" for item in TOOL_CAPABILITY_SUMMARY)
    return md_generated_notice() + dedent(
        f"""
        # IVIVC App

        {APP_DESCRIPTION}

        ## Quick start

        1. Install the Python dependencies listed in `requirements.txt`.
        2. Start the Flask app with `python app.py`.
        3. Open the local URL shown by Flask.
        4. Upload a fitting dataset with four columns: in vitro time, in vitro value, in vivo time, and in vivo value.
        5. Select approaches, models, preprocessing options, metrics, and analysis parameters.
        6. Optionally upload a two-column in vitro prediction dataset.
        7. Review model outputs, plots, downloadable spreadsheets, and report outputs.

        ## Main inputs

        - **Fitting dataset:** required four-column dataset used for model fitting.
        - **Prediction dataset:** optional two-column in vitro dataset used to generate predictions after fitting.
        - **Approaches/models:** mathematical methods and candidate functions to evaluate.
        - **Preprocessing:** optional normalization, scaling, and interpolation settings.
        - **Metrics:** goodness-of-fit and model-comparison metrics.
        - **Analysis parameters:** cross-validation and grid-search settings.

        ## What the app does

{capabilities_md}

        ## Main outputs

        - Preprocessing summaries and plots.
        - Fitted model parameters and performance metrics.
        - Cross-validation metrics for supported approaches.
        - Model-comparison tables.
        - Prediction plots and tables when a prediction dataset is supplied.
        - Downloadable spreadsheet outputs and report outputs where enabled.

        ## User guide

        See [`docs/user_guide.pdf`](docs/user_guide.pdf) for detailed descriptions of inputs, options, methods, and outputs. The main LaTeX content source is [`docs/user_guide.tex`](docs/user_guide.tex), with appendices in [`docs/appendices/`](docs/appendices/) and references in [`docs/references.bib`](docs/references.bib). These files are content-only so they can be copied into an Overleaf project and included from a top-level LaTeX document.

        ## Review packet

        To generate Word review artifacts for non-technical review, run:

        ```bash
        bash docs/generate_review_packet.sh
        ```

        This writes review-only `.docx` files to `docs/review/`. Reviewers can use Word comments or tracked changes, but accepted edits should be applied back to the source files.

        ## Updating documentation text

        User-facing descriptions are centralized in `utilities/descriptions.py`, with routine wording overrides in `utilities/descriptions.yaml`. The main guide content is generated as LaTeX source, long-form appendices are maintained as separate LaTeX files in `docs/appendices/`, and references are maintained in `docs/references.bib`.

        After editing source text, regenerate the main LaTeX guide with:

        ```bash
        python utilities/generate_user_guide.py
        ```

        Compile the PDF with:

        ```bash
        bash utilities/generate_user_guide_pdf.sh
        ```
        """
    ).lstrip()


def latex_preamble() -> str:
    return dedent(
        r"""
        \documentclass[10pt,letterpaper]{article}
        \usepackage[margin=0.75in]{geometry}
        \usepackage{amsmath}
        \usepackage{array}
        \usepackage{booktabs}
        \usepackage{enumitem}
        \usepackage{fancyhdr}
        \usepackage{longtable}
        \usepackage{tabularx}
        \usepackage{xcolor}
        \usepackage{xurl}
        \usepackage{hyperref}

        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0.55em}
        \setlist[itemize]{leftmargin=*}
        \setlist[enumerate]{leftmargin=*}
        \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue, breaklinks=true}

        \pagestyle{fancy}
        \fancyhf{}
        \lhead{IVIVC App User Guide}
        \rhead{\thepage}
        \fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}}

        \title{IVIVC App User Guide}
        \author{}
        \date{}
        """
    ).lstrip()


def build_user_guide_tex() -> str:
    parts: list[str] = [tex_generated_notice()]

    parts.append(latex_section(1, "Introduction"))
    parts.append(latex_paragraphs(APP_DESCRIPTION))
    parts.append(
        r"For general background on credibility assessment and reporting of computational modeling studies in medical device submissions, see FDA guidance on computational modeling credibility and reporting \cite{fda_modeling_guidance,fda_reporting_guidance}."
        + "\n\n"
    )

    parts.append(latex_section(1, "What the app does"))
    parts.append(latex_itemize(TOOL_CAPABILITY_SUMMARY))

    parts.append(latex_section(1, "Inputs"))
    parts.append(latex_paragraphs("The IVIVC app requires paired in vitro and in vivo degradation data, along with user-selected analysis settings that define how the data are processed, modeled, and summarized. At a high level, the inputs include:"))
    parts.append(latex_itemize(['a spreadsheet containing in vitro and in vivo time-course degradation data for fitting correlations;',
                                'optionally, a spreadsheet containing in vitro data from which to generate predictions using fitted correlations;',
                                'selection of one or more IVIVC modeling approaches;',
                                'selection of candidate models to evaluate, where applicable;',
                                'optionally, preprocessing options, such as normalization, interpolation, or scaling choices;',
                                'selected performance metrics to display in the results and reports; by default, the app selects R², RMSE, and AICc as a simple starting set that has been useful in internal testing with real datasets but may not be optimal for every analysis',
                                'optionally, cross-validation settings for applicable parametric models; and',
                                'grid-search settings for model initialization.']))
    parts.append(latex_paragraphs("The sections below provide additional detail on input file structure, required data columns, preprocessing options, model-selection inputs, and optional analysis settings."))

    parts.append(latex_section(1, "Outputs"))
    parts.append(latex_paragraphs("The IVIVC app generates interactive results and downloadable reports that summarize fitted model behavior, model-comparison metrics, diagnostic information, and user-selected analysis settings. Output availability depends on the selected approaches, whether fitting succeeds, and whether a prediction dataset was supplied."))
    parts.append(latex_table(
        ["Output", "Description"],
        [
            ["Preprocessing outputs", "Processed data tables and plots showing the data used for fitting and prediction."],
            ["Model fitting outputs", "Parameter estimates, parameter standard errors where available, fitted plots, residual (error) plots, and selected performance metrics."],
            ["Model comparison outputs", "Tables comparing selected approaches, models, metrics, and cross-validation summaries."],
            ["Prediction outputs", "Prediction plots and downloadable tables when a prediction dataset is supplied and the selected method can be applied."],
            ["Selected model & summary of outputs", "User-selected report summaries and optional Word report export for incorporation into external documentation."],
        ],
    ))

    parts.append(latex_section(1, "Input datasets"))
    parts.append(latex_table(
        ["Input", "Description"],
        [
            ["Fitting dataset", "Required four-column dataset used to establish the in vitro/in vivo relationship. Columns are interpreted as in vitro time, in vitro value, in vivo time, and in vivo value."],
            ["Prediction dataset", "Optional two-column in vitro dataset used after fitting to generate in vivo predictions. Columns are interpreted as in vitro time and in vitro value."],
        ],
    ))
    parts.append(latex_paragraphs("For Excel workbooks, select the sheet containing the relevant data. For CSV files, the first columns are read directly in the order described above."))

    parts.append(latex_section(1, "Analysis approaches and models"))
    approach_rows = []
    for approach_key, approach_info in approaches.items():
        desc = APPROACH_DESCRIPTIONS.get(approach_key, {})
        model_names = [models[m].get("display_name", m) for m in approach_info.get("models", []) if m in models]
        model_text = ", ".join(model_names) if model_names else "No parametric model selection"
        method_text = html_to_text(desc.get("method_html", ""))
        approach_rows.append([
            desc.get("title", approach_info.get("display_name", approach_key)),
            method_text,
            model_text,
        ])
    parts.append(latex_table(["Approach", "Description", "Available models"], approach_rows, ["0.18\\textwidth", "0.46\\textwidth", "0.28\\textwidth"]))

    parts.append(latex_section(2, "Candidate model functional forms"))
    model_rows = []
    for model_key, model_info in sorted(models.items(), key=lambda item: item[1].get("display_name", item[0])):
        model_rows.append([
            model_info.get("display_name", model_key),
            model_approach_labels(model_key),
            model_equation(model_key, model_info),
            model_description(model_key, model_info),
        ])
    if model_rows:
        parts.append(latex_table(
            ["Model", "Used in", "Functional form", "Notes"],
            model_rows,
            ["0.16\\textwidth", "0.14\\textwidth", "0.36\\textwidth", "0.26\\textwidth"],
            raw_columns={2},
        ))
        parts.append(r"\textbf{Mass-loss model source notes}" + "\n\n")
        parts.append(r"The surface-erosion mass-loss form is based on geometry-controlled surface erosion concepts such as the erodible slab, cylinder, and sphere model \cite{hopfenberg1976}. The one-phase mass-loss model uses a logistic/sigmoidal transition to represent a delayed loss phase \cite{tsoularis2002}. The two-phase mass-loss model combines an early burst-like loss term with a delayed sigmoidal loss phase. Burst and delayed/multiphase behavior in degradable polymer systems is discussed in the polymer erosion and PLGA release literature \cite{gopferich1997,fredenberg2011,allison2008}. The Gompertz mass-loss model uses a modified Gompertz-style sigmoidal form with lag and maximum-rate parameters \cite{zwietering1990}." + "\n\n")

    parts.append(latex_section(1, "Preprocessing options"))
    for category in ["normalization", "scaling", "interpolation"]:
        parts.append(latex_section(2, category.title()))
        keys = list(preprocessing_options.get(category, {}).keys())
        if not keys:
            keys = list(PREPROCESSING_DESCRIPTIONS.get(category, {}).keys())
        rows = [[preprocessing_label(category, key), key, preprocessing_description(category, key)] for key in keys]
        parts.append(latex_table(["Option", "Internal key", "Description"], rows, ["0.22\\textwidth", "0.20\\textwidth", "0.50\\textwidth"]))

    parts.append(latex_section(1, "Relative model evidence"))
    parts.append(latex_paragraphs(RELATIVE_MODEL_EVIDENCE_TEXT))
    parts.append(
        r"For additional background on information criteria, model-selection evidence, and evidence-weight interpretation, see references \cite{akaike1974,schwarz1978,burnham2002,kass1995,wagenmakers2004}."
        + "\n\n"
    )
    parts.append(latex_table(
        ["Evidence summary", "Interpretation"],
        [
            ["Delta AICc / Delta AIC / Delta BIC", "Difference from the lowest value in the displayed candidate set. Values near 0 indicate similar relative support; larger values indicate lower support relative to the leading candidate under that criterion."],
            ["AICc or BIC weight", "Normalized relative-support value across the candidate models in the displayed evidence table. Weights are relative, not absolute probabilities that a model is true."],
            ["Evidence ratio", "Ratio of the highest evidence weight to the model's evidence weight. Larger ratios indicate less relative support compared with the highest-weighted model in the set."],
            ["Interpretation badge", "Neutral rule-of-thumb label based on the displayed delta values. The label supports user review and is not an app recommendation."],
        ],
    ))

    parts.append(latex_section(1, "Cross-validation options"))
    parts.append(latex_paragraphs(CROSS_VALIDATION_DESCRIPTION_HTML))
    parts.append(latex_paragraphs("Leave-one-timepoint-out is the default cross-validation scheme. It is a practical starting point for small time-course datasets because each split uses all but one observation for fitting. Leave-contiguous-block-out and K-fold can be used as more demanding time-region holdout checks, while Shuffle split is retained as a familiar exploratory option."))
    parts.append(
        r"For background on cross-validation procedures and their use in model assessment and selection, see reference \cite{arlot2010}."
        + "\n\n"
    )
    parts.append(latex_table(
        ["Scheme", "Internal key", "Description"],
        [[label, key, CV_SCHEME_DESCRIPTIONS.get(key, "")] for key, label in CV_SCHEME_LABELS.items()],
        ["0.20\\textwidth", "0.20\\textwidth", "0.52\\textwidth"],
    ))
    cv_rows = []
    for key, label in ANALYSIS_CONFIG_LABELS.items():
        if key.startswith("cv_"):
            cv_rows.append([label, key, analysis_parameter_description(key)])
    parts.append(latex_table(["Parameter", "Internal key", "Description"], cv_rows, ["0.22\\textwidth", "0.22\\textwidth", "0.48\\textwidth"]))

    parts.append(latex_section(1, "Grid-search initialization"))
    parts.append(latex_paragraphs(GRID_SEARCH_DESCRIPTION_HTML))
    grid_rows = []
    for key, label in ANALYSIS_CONFIG_LABELS.items():
        if key.startswith("grid_search"):
            grid_rows.append([label, key, analysis_parameter_description(key)])
    parts.append(latex_table(["Parameter", "Internal key", "Description"], grid_rows, ["0.24\\textwidth", "0.24\\textwidth", "0.44\\textwidth"]))

    parts.append(latex_section(1, "Performance metrics"))
    parts.append(latex_paragraphs("By default, the app selects R², RMSE, and AICc. This compact default set has been simple and useful in internal testing with real datasets, but it is not necessarily the best set for every dataset or intended use. Users may select additional metrics when they provide useful context for a specific analysis."))
    metric_rows = []
    for metric_key, metric_info in metrics.items():
        metric_rows.append([
            metric_info.get("display_name", metric_key),
            metric_key,
            direction_label(metric_info.get("better_direction", "")),
            metric_description(metric_key),
        ])
    if not metric_rows:
        for metric_key, desc in METRIC_DESCRIPTIONS.items():
            metric_rows.append([metric_key.replace("_", " ").title(), metric_key, "", desc.get("description", "")])
    parts.append(latex_table(["Metric", "Internal key", "Preferred direction", "Description"], metric_rows, ["0.17\\textwidth", "0.18\\textwidth", "0.20\\textwidth", "0.37\\textwidth"]))
    parts.append(
        r"For additional background on common error metrics, scale-dependent accuracy measures, and adjusted \(R^2\), see references \cite{hodson2022,hyndman2006,miles2014}."
        + "\n\n"
    )
    parts.append(
        r"For AIC\textsubscript{c}, a commonly cited rule of thumb is to prefer AIC\textsubscript{c} over AIC when the ratio of fitted observations to estimated parameters is small, such as \(n/k < 40\), where \(n\) is the number of fitted observations and \(k\) is the number of estimated model parameters \cite{burnham2002}." + "\n\n"
    )

    parts.append(latex_section(1, "Uncertainty and error bars"))
    parts.append(latex_paragraphs(
        "The following descriptions summarize how fitted-parameter uncertainty, fitting-plot bands, and prediction error bars are calculated or interpreted for each approach."
    ))
    for approach_key, desc in APPROACH_DESCRIPTIONS.items():
        title = desc.get("title", approach_key)
        fit_text = html_to_text(desc.get("fit_uncertainty_html", ""))
        prediction_text = html_to_text(desc.get("prediction_error_html", ""))
        parts.append(latex_section(2, title))
        if fit_text:
            parts.append(r"\textbf{Fitting uncertainty}" + "\n\n")
            parts.append(latex_paragraphs(fit_text))
        if prediction_text:
            parts.append(r"\textbf{Prediction uncertainty}" + "\n\n")
            parts.append(latex_paragraphs(prediction_text))

    parts.append(latex_section(1, "Command-line interface"))
    parts.append(latex_paragraphs("The repository includes cli.py for batch-oriented fitting from the command line. Use python cli.py --help to see available arguments for input files, approaches, models, metrics, preprocessing options, cross-validation settings, and grid-search settings. The internal keys specified in this user guide are used as settings with cli.py."))

    parts.append(latex_section(1, "Extending the app"))
    parts.append(latex_paragraphs("The IVIVC app is designed so that common extensions can be added with minimal changes to the main application code. In many cases, new models, performance metrics, and preprocessing methods can be added by following the structure of the existing files in the relevant subdirectories."))

    return "".join(parts)


def main() -> None:
    GUIDE_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text(build_readme(), encoding="utf-8")
    GUIDE_TEX_PATH.write_text(build_user_guide_tex(), encoding="utf-8")
    print(f"Wrote {README_PATH.relative_to(ROOT)}")
    print(f"Wrote {GUIDE_TEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
