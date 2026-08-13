#!/usr/bin/env python3
"""Generate README.md and generated LaTeX includes for the user guide.

The main user-guide source, docs/user_guide.tex, is intentionally manual and
is not overwritten by this script. This generator writes only mechanically
derived material, such as tables and short text blocks sourced from
utilities/descriptions.yaml or from live model, metric, and preprocessing
registries.

Run from the repository root or from docs/:

    python docs/generate_user_guide.py

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
    CV_SCHEME_DESCRIPTIONS,
    CV_SCHEME_LABELS,
    GRID_SEARCH_DESCRIPTION_HTML,
    METRIC_DESCRIPTIONS,
    PREPROCESSING_DESCRIPTIONS,
    RELATIVE_MODEL_EVIDENCE_HTML,
    TOOL_CAPABILITY_SUMMARY,
    TOOL_PURPOSE_TEXT,
)

README_PATH = ROOT / "README.md"
INCLUDES_GENERATED_DIR = ROOT / "docs" / "includes" / "generated"
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
    text = re.sub(r"<em>(.*?)</em>", r"\1", text, flags=re.S)
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
    if not paragraphs:
        return ""
    return "\n\n".join(latex_escape(p) for p in paragraphs) + "\n\n"


def latex_itemize(items: list[str]) -> str:
    lines = [r"\begin{itemize}"]
    for item in items:
        lines.append(rf"\item {latex_escape(item)}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n\n"


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
    return "<!-- Generated by docs/generate_user_guide.py. Edit app-facing reusable wording in utilities/descriptions.yaml. Edit user-guide-only narrative in docs/user_guide.tex. Generated LaTeX inserts are written to docs/includes/generated/. -->\n\n"


def tex_generated_notice(source: str) -> str:
    return (
        "% Generated by docs/generate_user_guide.py. Do not edit directly.\n"
        f"% Source: {source}\n\n"
    )


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

        **For users without Python experience**, platform-specific local setup scripts are provided for Windows and macOS under `local_install/`. These installers (Setup IVIVC App) create a private Python environment for the IVIVC App and provide a separate double-click launcher (Start IVIVC App) for subsequent use. See the README in the corresponding `local_install/macos/` or `local_install/windows/` folder for instructions.

        Alternatively, **users comfortable with Python** can run the app directly using these steps: 
        1. Install the Python dependencies listed in `requirements.txt`. For example, using `conda create -n ivivc-app -c conda-forge python=3.12 --file requirements.txt`, then `conda activate ivivc-app` to set up a separate environment.
        2. Start the Flask app with `python app.py`.
        3. Open the local URL shown by Flask in a browser (`http://127.0.0.1:5000` by default).
        4. Upload a fitting dataset with four columns: in vitro time, in vitro value, in vivo time, and in vivo value.
        5. Select approaches, models, preprocessing options, metrics, and analysis parameters.
        6. Optionally upload a two-column in vitro prediction dataset.
        7. Review model outputs, plots, downloadable spreadsheets, and report outputs.

        `requirements.txt` contains the packages needed to run the app. Documentation/tutorial generation additionally requires the packages in `requirements-docs.txt`.

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

        See [`docs/user_guide.pdf`](docs/user_guide.pdf) for detailed descriptions of inputs, options, methods, and outputs. 

        ## Data handling
        The app does not save uploaded datasets or fitting results to persistent server storage. 
        Uploaded data are processed for the current analysis, and generated results and downloads are returned directly to the user.
        When running the app locally (either via `python app.py` or the `local_install` launchers), no data leaves the user's computer.
        """
    ).lstrip()


def build_approach_model_table() -> str:
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
    return latex_table(["Approach", "Description", "Available models"], approach_rows, ["0.18\\textwidth", "0.46\\textwidth", "0.28\\textwidth"])


def build_candidate_model_functional_forms_table() -> str:
    model_rows = []
    for model_key, model_info in sorted(models.items(), key=lambda item: item[1].get("display_name", item[0])):
        model_rows.append([
            model_info.get("display_name", model_key),
            model_approach_labels(model_key),
            model_equation(model_key, model_info),
            model_description(model_key, model_info),
        ])
    if not model_rows:
        return "% No candidate models were available when this file was generated.\n\n"
    return latex_table(
        ["Model", "Used in", "Functional form", "Notes"],
        model_rows,
        ["0.16\\textwidth", "0.14\\textwidth", "0.36\\textwidth", "0.26\\textwidth"],
        raw_columns={2},
    )


def build_preprocessing_options() -> str:
    parts: list[str] = []
    for category in ["normalization", "scaling", "interpolation"]:
        parts.append(latex_section(2, category.title()))
        keys = list(preprocessing_options.get(category, {}).keys())
        if not keys:
            keys = list(PREPROCESSING_DESCRIPTIONS.get(category, {}).keys())
        rows = [[preprocessing_label(category, key), key, preprocessing_description(category, key)] for key in keys]
        parts.append(latex_table(["Option", "Internal key", "Description"], rows, ["0.22\\textwidth", "0.20\\textwidth", "0.50\\textwidth"]))
    return "".join(parts)


def build_cv_scheme_reference_table() -> str:
    rows = [[label, key, CV_SCHEME_DESCRIPTIONS.get(key, "")] for key, label in CV_SCHEME_LABELS.items()]
    return latex_table(["Scheme", "Internal key", "Description"], rows, ["0.20\\textwidth", "0.28\\textwidth", "0.44\\textwidth"])


def build_cv_parameter_reference_table() -> str:
    cv_rows = []
    for key, label in ANALYSIS_CONFIG_LABELS.items():
        if key.startswith("cv_"):
            cv_rows.append([label, key, analysis_parameter_description(key)])
    return latex_table(["Parameter", "Internal key", "Description"], cv_rows, ["0.22\\textwidth", "0.22\\textwidth", "0.48\\textwidth"])


def build_grid_search_parameter_table() -> str:
    grid_rows = []
    for key, label in ANALYSIS_CONFIG_LABELS.items():
        if key.startswith("grid_search"):
            grid_rows.append([label, key, analysis_parameter_description(key)])
    return latex_table(["Parameter", "Internal key", "Description"], grid_rows, ["0.24\\textwidth", "0.24\\textwidth", "0.44\\textwidth"])


def build_performance_metrics_table() -> str:
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
    return latex_table(["Metric", "Internal key", "Preferred direction", "Description"], metric_rows, ["0.17\\textwidth", "0.18\\textwidth", "0.20\\textwidth", "0.37\\textwidth"])


def build_uncertainty_by_approach() -> str:
    parts: list[str] = []
    for approach_key, desc in APPROACH_DESCRIPTIONS.items():
        title = desc.get("title", approach_key)
        fit_text = html_to_text(desc.get("fit_uncertainty_html", ""))
        prediction_text = html_to_text(desc.get("prediction_error_html", ""))
        parts.append(latex_section(3, title))
        if fit_text:
            parts.append(r"\textbf{Fitting uncertainty}" + "\n\n")
            parts.append(latex_paragraphs(fit_text))
        if prediction_text:
            parts.append(r"\textbf{Prediction uncertainty}" + "\n\n")
            parts.append(latex_paragraphs(prediction_text))
    return "".join(parts)


def build_generated_includes() -> dict[str, tuple[str, str]]:
    return {
        "app_purpose.tex": (latex_paragraphs(APP_DESCRIPTION), "utilities/descriptions.yaml: tool.purpose_text"),
        "capability_summary.tex": (latex_itemize(TOOL_CAPABILITY_SUMMARY), "utilities/descriptions.yaml: tool.capability_summary"),
        "approach_model_table.tex": (build_approach_model_table(), "utilities/descriptions.yaml: approaches; models registry"),
        "candidate_model_functional_forms_table.tex": (build_candidate_model_functional_forms_table(), "models registry"),
        "preprocessing_options.tex": (build_preprocessing_options(), "utilities/descriptions.yaml: preprocessing; preprocessing registry"),
        "relative_model_evidence_description.tex": (latex_paragraphs(html_to_text(RELATIVE_MODEL_EVIDENCE_HTML)), "utilities/descriptions.yaml: relative_model_evidence_html"),
        "cv_scheme_reference_table.tex": (build_cv_scheme_reference_table(), "utilities/descriptions.yaml: cv_schemes"),
        "cv_parameter_reference_table.tex": (build_cv_parameter_reference_table(), "utilities/descriptions.yaml: analysis_parameters"),
        "grid_search_description.tex": (latex_paragraphs(GRID_SEARCH_DESCRIPTION_HTML), "utilities/descriptions.yaml: grid_search_description_html"),
        "grid_search_parameter_table.tex": (build_grid_search_parameter_table(), "utilities/descriptions.yaml: analysis_parameters"),
        "performance_metrics_table.tex": (build_performance_metrics_table(), "utilities/descriptions.yaml: metrics; metrics registry"),
        "uncertainty_by_approach.tex": (build_uncertainty_by_approach(), "utilities/descriptions.yaml: approaches"),
    }


def write_include(filename: str, content: str, source: str) -> Path:
    INCLUDES_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = INCLUDES_GENERATED_DIR / filename
    path.write_text(tex_generated_notice(source) + content.rstrip() + "\n", encoding="utf-8")
    return path


def write_generated_includes() -> list[Path]:
    written: list[Path] = []
    for filename, (content, source) in build_generated_includes().items():
        written.append(write_include(filename, content, source))
    return written


def main() -> None:
    README_PATH.write_text(build_readme(), encoding="utf-8")
    written = write_generated_includes()
    print(f"Wrote {README_PATH.relative_to(ROOT)}")
    for path in written:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
