"""Relative model-evidence calculations for fitted IVIVC models.

The functions in this module intentionally summarize relative evidence only.
They do not rank, select, recommend, or validate a model.
"""

from __future__ import annotations

from math import exp, isfinite, log

import numpy as np
import pandas as pd


_EPS = np.finfo(float).eps
CRITERION_LABELS = {
    "aic": "AIC",
    "aicc": "AICc",
    "bic": "BIC",
}
INTERPRETATION_PREFERENCE = ("aicc", "aic", "bic")


BADGE_CLASSES = {
    "Comparable support": "evidence-badge-comparable",
    "Somewhat lower support": "evidence-badge-lower",
    "Considerably lower support": "evidence-badge-lower",
    "Substantially lower support": "evidence-badge-minimal",
    "Minimal relative support": "evidence-badge-minimal",
    "Weak BIC difference": "evidence-badge-comparable",
    "Positive BIC evidence difference": "evidence-badge-lower",
    "Strong BIC evidence difference": "evidence-badge-minimal",
    "Very strong BIC evidence difference": "evidence-badge-minimal",
}


def selected_information_criteria(selected_metrics):
    """Return selected information criteria in the user's metric display order."""
    if selected_metrics is None:
        selected_metrics = CRITERION_LABELS.keys()

    criteria = []
    for metric in selected_metrics:
        metric_key = str(metric).lower()
        if metric_key in CRITERION_LABELS and metric_key not in criteria:
            criteria.append(metric_key)
    return criteria


def _finite(value) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if isfinite(value) else np.nan


def _format_number(value, digits=4):
    value = _as_float(value)
    if not isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _format_weight(value):
    value = _as_float(value)
    if not isfinite(value):
        return "N/A"
    if value < 0.0001 and value > 0:
        return "<0.0001"
    return f"{value:.4f}"


def _format_ratio(value):
    value = _as_float(value)
    if not isfinite(value):
        return "N/A"
    if value > 9999:
        return ">9999"
    return f"{value:.2f}"


def _badge(label):
    css_class = BADGE_CLASSES.get(label, "")
    class_attr = f"evidence-badge {css_class}".strip()
    return f'<span class="{class_attr}">{label}</span>'


def ic_from_n_rss_k(n, rss, k):
    """Calculate AIC, AICc, and BIC from sample size, RSS, and parameter count."""
    try:
        n = int(n)
        k = int(k)
    except (TypeError, ValueError):
        n = 0
        k = 0

    rss = _as_float(rss)
    if n <= 0 or k < 0 or not isfinite(rss):
        return {
            "n": n,
            "k": k,
            "rss": np.nan,
            "aic": np.nan,
            "aicc": np.nan,
            "bic": np.nan,
            "aicc_defined": False,
        }

    rss = max(rss, _EPS)
    aic = n * log(rss / n) + 2 * k
    bic = n * log(rss / n) + k * log(n) if n > 0 else np.nan
    if n > k + 1:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        aicc_defined = True
    else:
        aicc = np.nan
        aicc_defined = False

    return {
        "n": n,
        "k": k,
        "rss": rss,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "aicc_defined": aicc_defined,
    }


def ic_from_residuals(residuals, n_params):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    return ic_from_n_rss_k(len(residuals), float(np.sum(residuals ** 2)), n_params)


def add_relative_evidence(rows, criterion):
    """Add delta, weight, and evidence ratio fields for one criterion."""
    value_key = criterion
    delta_key = f"delta_{criterion}"
    weight_key = f"{criterion}_weight"
    ratio_key = f"{criterion}_evidence_ratio"

    finite_values = [_as_float(row.get(value_key)) for row in rows if _finite(row.get(value_key))]
    if len(finite_values) < 2:
        for row in rows:
            row[delta_key] = np.nan
            row[weight_key] = np.nan
            row[ratio_key] = np.nan
        return rows

    min_value = min(finite_values)
    raw_weights = []
    for row in rows:
        value = _as_float(row.get(value_key))
        if isfinite(value):
            delta = value - min_value
            raw_weight = exp(-0.5 * delta)
        else:
            delta = np.nan
            raw_weight = np.nan
        row[delta_key] = delta
        raw_weights.append(raw_weight)

    finite_weights = [w for w in raw_weights if _finite(w)]
    denominator = sum(finite_weights)
    normalized_weights = []
    for row, raw_weight in zip(rows, raw_weights):
        if denominator > 0 and _finite(raw_weight):
            weight = raw_weight / denominator
        else:
            weight = np.nan
        row[weight_key] = weight
        normalized_weights.append(weight)

    finite_normalized = [w for w in normalized_weights if _finite(w)]
    reference_weight = max(finite_normalized) if finite_normalized else np.nan
    for row in rows:
        weight = row.get(weight_key)
        if _finite(reference_weight) and _finite(weight) and weight > 0:
            row[ratio_key] = reference_weight / weight
        else:
            row[ratio_key] = np.nan

    return rows


def aic_support_label(delta):
    delta = _as_float(delta)
    if not isfinite(delta):
        return "N/A"
    if delta <= 2:
        return "Comparable support"
    if delta < 4:
        return "Somewhat lower support"
    if delta <= 7:
        return "Considerably lower support"
    if delta <= 10:
        return "Substantially lower support"
    return "Minimal relative support"


def bic_support_label(delta):
    delta = _as_float(delta)
    if not isfinite(delta):
        return "N/A"
    if delta <= 2:
        return "Weak BIC difference"
    if delta <= 6:
        return "Positive BIC evidence difference"
    if delta <= 10:
        return "Strong BIC evidence difference"
    return "Very strong BIC evidence difference"


def _primary_criterion(rows, criteria):
    for criterion in INTERPRETATION_PREFERENCE:
        if criterion in criteria and sum(_finite(row.get(criterion)) for row in rows) >= 2:
            return criterion
    return None


def _sort_rows(rows, criteria):
    primary = _primary_criterion(rows, criteria)
    if not primary:
        return sorted(rows, key=lambda row: row.get("model", ""))
    return sorted(rows, key=lambda row: (_as_float(row.get(primary)) if _finite(row.get(primary)) else float("inf"), row.get("model", "")))


def _interpretation(row, primary=None):
    if primary in ("aicc", "aic"):
        return aic_support_label(row.get(f"delta_{primary}"))
    if primary == "bic":
        return bic_support_label(row.get("delta_bic"))
    return "N/A"


def _model_key_parts(model_key):
    if ":" in model_key:
        return model_key.split(":", 1)
    return model_key, model_key


def _row_from_evidence_data(model_key, model_display, evidence_data, dataset_label, values=None):
    values = values or ic_from_n_rss_k(
        evidence_data.get("n"),
        evidence_data.get("rss"),
        evidence_data.get("n_params"),
    )
    return {
        "model_key": model_key,
        "model": model_display,
        "dataset": dataset_label,
        "n": values.get("n"),
        "k": values.get("k"),
        "rss": values.get("rss"),
        "aic": values.get("aic"),
        "aicc": values.get("aicc"),
        "bic": values.get("bic"),
        "aicc_defined": values.get("aicc_defined"),
    }


def _format_table_rows(rows, criteria, include_dataset=False):
    primary = _primary_criterion(rows, criteria)
    formatted_rows = []
    for row in rows:
        interpretation = row.get("interpretation_label") or _interpretation(row, primary)
        formatted = {
            "Model": row.get("model", ""),
            "n": row.get("n", ""),
            "k": row.get("k", ""),
        }
        for criterion in criteria:
            label = CRITERION_LABELS[criterion]
            formatted[label] = _format_number(row.get(criterion))
            formatted[f"Delta {label}"] = _format_number(row.get(f"delta_{criterion}"))
            formatted[f"{label} weight"] = _format_weight(row.get(f"{criterion}_weight"))
            formatted[f"{label} evidence ratio"] = _format_ratio(row.get(f"{criterion}_evidence_ratio"))
        formatted["Interpretation"] = _badge(interpretation) if interpretation != "N/A" else "N/A"
        if include_dataset:
            formatted = {"Evidence set": row.get("dataset", "")} | formatted
        formatted_rows.append(formatted)
    return formatted_rows


def _html_table(rows, criteria, include_dataset=False):
    if not rows:
        return ""
    df = pd.DataFrame(_format_table_rows(rows, criteria, include_dataset=include_dataset))
    return df.to_html(
        classes="table table-striped evidence-table",
        index=False,
        escape=False,
    )


def _panel_html(rows, criteria, extra_notes=None):
    extra_notes = extra_notes or []
    if not criteria:
        return '<p class="evidence-note">Relative model evidence is not shown because no information-criterion metric (AIC, AICc, or BIC) was selected.</p>'
    if not rows:
        return '<p class="evidence-note">Relative evidence is not available for this model set.</p>'

    sorted_rows = _sort_rows(rows, criteria)
    primary = _primary_criterion(sorted_rows, criteria)
    notes = []
    if primary is None:
        notes.append("Relative evidence requires at least two models with finite values for a selected information-criterion metric.")
    elif len(criteria) > 1:
        notes.append(f"Interpretation labels use {CRITERION_LABELS[primary]} differences because it is the selected information criterion prioritized for the badge summary.")

    for criterion in criteria:
        finite_count = sum(_finite(row.get(criterion)) for row in sorted_rows)
        if finite_count < 2:
            label = CRITERION_LABELS[criterion]
            notes.append(f"{label} delta values, weights, and evidence ratios require at least two finite {label} values.")

    if "aicc" in criteria and any(not row.get("aicc_defined") for row in sorted_rows):
        notes.append("AICc is undefined for one or more models because the number of observations is too small relative to the number of fitted parameters.")
    notes.extend(extra_notes)

    note_html = "".join(f'<p class="evidence-note">{note}</p>' for note in notes)
    return f'{_html_table(sorted_rows, criteria)}{note_html}'


def _unavailable_panel(message):
    return {"html": f'<p class="evidence-note">{message}</p>', "rows": []}


def _final_report_html(rows, criteria):
    if not rows or not criteria:
        return ""
    return _html_table(rows, criteria, include_dataset=True)


def _add_relative_fields(rows, criteria):
    for criterion in criteria:
        add_relative_evidence(rows, criterion)
    primary = _primary_criterion(rows, criteria)
    for row in rows:
        row["primary_criterion"] = primary
        row["interpretation_label"] = _interpretation(row, primary) if primary else "N/A"
    return rows


def _approach1_rows(results, models_registry, criteria):
    rows = []
    excluded_count = 0
    for model_key, model_results in results.items():
        approach_id, model_name = _model_key_parts(model_key)
        if approach_id != "approach1":
            continue
        if model_results.get("error"):
            excluded_count += 1
            continue
        evidence_data = model_results.get("evidence_data", {}).get("fit")
        if not evidence_data:
            excluded_count += 1
            continue
        display = models_registry.get(model_name, {}).get("display_name", model_name)
        rows.append(_row_from_evidence_data(model_key, display, evidence_data, "Approach 1 fit"))
    return _add_relative_fields(rows, criteria), excluded_count


def _approach2_rows(results, models_registry, criteria):
    combined_rows = []
    in_vitro_rows = []
    in_vivo_rows = []
    excluded_count = 0

    for model_key, model_results in results.items():
        approach_id, model_name = _model_key_parts(model_key)
        if approach_id != "approach2":
            continue
        if model_results.get("error"):
            excluded_count += 1
            continue

        evidence_data = model_results.get("evidence_data", {})
        in_vitro = evidence_data.get("in_vitro")
        in_vivo = evidence_data.get("in_vivo")
        if not in_vitro or not in_vivo:
            excluded_count += 1
            continue

        display = models_registry.get(model_name, {}).get("display_name", model_name)
        in_vitro_values = ic_from_n_rss_k(in_vitro.get("n"), in_vitro.get("rss"), in_vitro.get("n_params"))
        in_vivo_values = ic_from_n_rss_k(in_vivo.get("n"), in_vivo.get("rss"), in_vivo.get("n_params"))
        in_vitro_rows.append(_row_from_evidence_data(model_key, display, in_vitro, "In vitro fit", in_vitro_values))
        in_vivo_rows.append(_row_from_evidence_data(model_key, display, in_vivo, "In vivo fit", in_vivo_values))

        both_aicc_defined = in_vitro_values.get("aicc_defined") and in_vivo_values.get("aicc_defined")
        combined_values = {
            "n": int(in_vitro_values.get("n", 0)) + int(in_vivo_values.get("n", 0)),
            "k": int(in_vitro_values.get("k", 0)) + int(in_vivo_values.get("k", 0)),
            "rss": _as_float(in_vitro_values.get("rss")) + _as_float(in_vivo_values.get("rss")),
            "aic": _as_float(in_vitro_values.get("aic")) + _as_float(in_vivo_values.get("aic")),
            "aicc": (_as_float(in_vitro_values.get("aicc")) + _as_float(in_vivo_values.get("aicc"))) if both_aicc_defined else np.nan,
            "bic": _as_float(in_vitro_values.get("bic")) + _as_float(in_vivo_values.get("bic")),
            "aicc_defined": both_aicc_defined,
        }
        combined_rows.append(_row_from_evidence_data(model_key, display, {}, "Combined in vitro + in vivo", combined_values))

    return {
        "combined": _add_relative_fields(combined_rows, criteria),
        "in_vitro": _add_relative_fields(in_vitro_rows, criteria),
        "in_vivo": _add_relative_fields(in_vivo_rows, criteria),
    }, excluded_count


def _excluded_note(excluded_count):
    if excluded_count <= 0:
        return []
    plural = "s" if excluded_count != 1 else ""
    return [f"{excluded_count} model{plural} with fitting errors or incomplete evidence data were excluded from relative evidence calculations."]


def _panel_from_rows(rows, criteria, excluded_count=0):
    if not criteria:
        return _unavailable_panel("Relative model evidence is not shown because no information-criterion metric (AIC, AICc, or BIC) was selected.")
    if len(rows) < 2:
        return _unavailable_panel("Relative evidence requires at least two successfully fitted candidate models for this approach.")
    return {"html": _panel_html(rows, criteria, _excluded_note(excluded_count)), "rows": rows, "criteria": criteria}


def build_relative_evidence_tables(results, models_registry, selected_approaches, selected_metrics=None):
    """Build HTML evidence tables and selected-model lookup rows for templates."""
    criteria = selected_information_criteria(selected_metrics)
    evidence_tables = {}
    evidence_lookup = {}

    if "approach1" in selected_approaches:
        rows, excluded_count = _approach1_rows(results, models_registry, criteria)
        evidence_tables.setdefault("approach1", {})["fit"] = _panel_from_rows(rows, criteria, excluded_count)
        for row in rows:
            evidence_lookup.setdefault(row["model_key"], {}).setdefault("rows", []).append(row)

    if "approach2" in selected_approaches:
        row_sets, excluded_count = _approach2_rows(results, models_registry, criteria)
        evidence_tables.setdefault("approach2", {})
        for key, rows in row_sets.items():
            evidence_tables["approach2"][key] = _panel_from_rows(rows, criteria, excluded_count if key == "combined" else 0)
            for row in rows:
                evidence_lookup.setdefault(row["model_key"], {}).setdefault("rows", []).append(row)

    if "approach3" in selected_approaches:
        evidence_tables.setdefault("approach3", {})["fit"] = _unavailable_panel(
            "Relative AIC/AICc/BIC evidence is not shown for direct mapping because no parametric candidate model is fitted."
        )

    for model_key, payload in evidence_lookup.items():
        payload["report_html"] = _final_report_html(payload.get("rows", []), criteria)

    return evidence_tables, evidence_lookup
