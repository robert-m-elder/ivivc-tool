#!/usr/bin/env python3
"""Generate files for the worked model-comparison tutorial appendix.

The script is intended to live at::

    docs/tutorial/generate_model_comparison_tutorial.py

It reuses the IVIVC app's model registry, fitting machinery, metric functions,
shuffle-split cross-validation configuration, and relative-evidence calculations.
The generated files are written to ``docs/tutorial/generated`` by default.

Run from any working directory with::

    python docs/tutorial/generate_model_comparison_tutorial.py

The script does not compile the user guide or modify any maintained LaTeX file.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
TUTORIAL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TUTORIAL_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import through engine.py so the tutorial uses the same registered models,
# initial-search function, fitting functions, metrics, and CV constructor as the app.
import engine  # noqa: E402
from utilities.model_evidence import add_relative_evidence  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducible tutorial configuration
# -----------------------------------------------------------------------------

DEFAULT_RANDOM_SEED = 27
#DEFAULT_RANDOM_SEED = 215
DEFAULT_CV_SEED = 12345
DEFAULT_CV_SPLITS = 100
DEFAULT_CV_TEST_SIZE = 0.25
DEFAULT_GRID_SEARCH_POINTS = 100
DEFAULT_NOISE_SD = 3.0
DENSE_RANDOM_SEED_OFFSET = 100003

TIME_POINTS = np.array([0, 2, 4, 8, 12, 24, 36, 48, 60], dtype=float)
#TIME_POINTS = np.array([0, 1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 48, 60], dtype=float)
DENSE_TIME_POINTS = np.linspace(0.0, 60.0, 61)

TRUE_MODEL_ID = "exponential"
TRUE_PARAMETERS = np.array([100.0, -0.075], dtype=float)

MODEL_IDS = (
    "linear",
    "exponential",
    "stretched_exponential",
    "double_exponential",
)

TUTORIAL_ROLES = {
    "linear": "Clearly inadequate",
    "exponential": "Strong candidate",
    "stretched_exponential": "More flexible candidate",
    "double_exponential": "Most flexible candidate",
}

# App-derived colors, supplemented with neutral colors for the tutorial.
MODEL_COLORS = {
    "linear": "#7A7A7A",
    "exponential": engine.colors.get("in_vitro", "#2E5F8A"),
    "stretched_exponential": engine.colors.get("in_vivo", "#8B3A6B"),
    "double_exponential": "#2F2F2F",
}

MODEL_LINESTYLES = {
    "linear": "-.",
    "exponential": "-",
    "stretched_exponential": "--",
    "double_exponential": ":",
}

MODEL_MARKERS = {
    "linear": "D",
    "exponential": "o",
    "stretched_exponential": "s",
    "double_exponential": "^",
}

METRIC_IDS = (
    "r_squared",
    "adjusted_r_squared",
    "rmse",
    "mae",
    "aicc",
    "bic",
)


@dataclass
class FitResult:
    model_id: str
    model_name: str
    role: str
    model_function: Any
    parameter_names: list[str]
    estimates: np.ndarray
    covariance: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    metrics: dict[str, float]
    initial_values: np.ndarray
    cv_scores: dict[str, np.ndarray]


# -----------------------------------------------------------------------------
# Data generation and fitting
# -----------------------------------------------------------------------------


def generate_dataset(random_seed: int, noise_sd: float) -> pd.DataFrame:
    """Generate the fixed small-sample single-exponential dataset."""
    rng = np.random.default_rng(random_seed)
    true_function = engine.models[TRUE_MODEL_ID]["model_function"]
    true_response = true_function(TIME_POINTS, *TRUE_PARAMETERS)
    observed_response = true_response + rng.normal(0.0, noise_sd, size=TIME_POINTS.size)

    return pd.DataFrame(
        {
            "time": TIME_POINTS,
            "true_response": true_response,
            "observed_response": observed_response,
        }
    )


def generate_dense_dataset(random_seed: int, noise_sd: float) -> pd.DataFrame:
    """Generate an independent 61-point dataset from the same model."""
    rng = np.random.default_rng(random_seed + DENSE_RANDOM_SEED_OFFSET)
    true_function = engine.models[TRUE_MODEL_ID]["model_function"]
    true_response = true_function(DENSE_TIME_POINTS, *TRUE_PARAMETERS)
    observed_response = true_response + rng.normal(
        0.0,
        noise_sd,
        size=DENSE_TIME_POINTS.size,
    )

    return pd.DataFrame(
        {
            "time": DENSE_TIME_POINTS,
            "true_response": true_response,
            "observed_response": observed_response,
        }
    )


def tutorial_initial_points(model_id: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return deterministic candidate starts for the tutorial models.

    These are supplied to the app's generic grid-search initializer. They do not
    impose parameter bounds on the final nonlinear least-squares fit.
    """
    amplitude = max(float(np.nanmax(y)), float(y[0]), 1.0)

    if model_id == "linear":
        slope = (float(y[-1]) - float(y[0])) / max(float(x[-1] - x[0]), 1.0)
        return np.array(
            [
                [float(y[0]), slope],
                [amplitude, -amplitude / max(float(np.max(x)), 1.0)],
                [float(np.mean(y)), slope],
            ],
            dtype=float,
        )

    if model_id == "exponential":
        return np.array(
            [
                [amplitude, -0.04],
                [amplitude, -0.075],
                [amplitude, -0.12],
            ],
            dtype=float,
        )

    if model_id == "stretched_exponential":
        return np.array(
            [
                [amplitude, -0.08, 0.60],
                [amplitude, -0.16, 0.72],
                [amplitude, -0.25, 0.90],
                [amplitude, -0.05, 1.10],
            ],
            dtype=float,
        )

    if model_id == "double_exponential":
        return np.array(
            [
                [0.65 * amplitude, -0.09, 0.35 * amplitude, -0.06],
                [0.50 * amplitude, -0.08, 0.50 * amplitude, -0.07],
                [0.65 * amplitude, -0.25, 0.35 * amplitude, -0.035],
                [0.50 * amplitude, -0.35, 0.50 * amplitude, -0.055],
                [0.80 * amplitude, -0.15, 0.20 * amplitude, -0.015],
                [0.35 * amplitude, -0.45, 0.65 * amplitude, -0.070],
            ],
            dtype=float,
        )

    raise KeyError(f"No tutorial starting values are defined for {model_id!r}.")


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def fit_candidate_model(
    model_id: str,
    x: np.ndarray,
    y: np.ndarray,
    grid_search_points: int,
    cv_config: dict[str, Any],
) -> FitResult:
    """Fit one candidate model using the same machinery used by engine.py."""
    model_spec = engine.models[model_id]
    model_function = model_spec["model_function"]
    fit_kwargs = dict(model_spec.get("fit_kwargs", {}))
    candidate_starts = tutorial_initial_points(model_id, x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        initial_values = engine.auto_grid_search(
            model_function,
            x,
            y,
            param_min=-200.0,
            param_max=200.0,
            num_points=grid_search_points,
            num_cores=1,
            random_state=DEFAULT_RANDOM_SEED,
            initial_points=candidate_starts,
        )
        model_result = model_spec["fit_model"](x, y, p0=initial_values)

    parameter_names = list(model_result["params"].keys())
    estimates = np.asarray(list(model_result["params"].values()), dtype=float)
    covariance = np.asarray(model_result["pcov"], dtype=float)
    predictions = np.asarray(model_result["predict"](x), dtype=float)
    residuals = y - predictions
    n_parameters = len(estimates)

    metric_values = {
        metric_id: float(engine.calculate_metric(metric_id, y, predictions, n_parameters))
        for metric_id in METRIC_IDS
    }

    cv = engine.make_cross_validator(cv_config, approach_id="approach2")
    cv_scores = engine.cross_validation_curve_fit(
        x,
        y,
        model_function,
        cv,
        selected_metrics=["rmse", "mae"],
        p0=initial_values,
        kwargs=fit_kwargs,
    )

    return FitResult(
        model_id=model_id,
        model_name=str(model_spec["display_name"]),
        role=TUTORIAL_ROLES[model_id],
        model_function=model_function,
        parameter_names=parameter_names,
        estimates=estimates,
        covariance=covariance,
        predictions=predictions,
        residuals=residuals,
        metrics=metric_values,
        initial_values=np.asarray(initial_values, dtype=float),
        cv_scores=cv_scores,
    )


def fit_candidate_model_for_diagnostics(
    model_id: str,
    x: np.ndarray,
    y: np.ndarray,
    grid_search_points: int,
) -> FitResult:
    """Fit one model for the dense diagnostic figure without CV or metrics."""
    model_spec = engine.models[model_id]
    model_function = model_spec["model_function"]
    candidate_starts = tutorial_initial_points(model_id, x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        initial_values = engine.auto_grid_search(
            model_function,
            x,
            y,
            param_min=-200.0,
            param_max=200.0,
            num_points=grid_search_points,
            num_cores=1,
            random_state=DEFAULT_RANDOM_SEED,
            initial_points=candidate_starts,
        )
        model_result = model_spec["fit_model"](x, y, p0=initial_values)

    parameter_names = list(model_result["params"].keys())
    estimates = np.asarray(list(model_result["params"].values()), dtype=float)
    predictions = np.asarray(model_result["predict"](x), dtype=float)

    return FitResult(
        model_id=model_id,
        model_name=str(model_spec["display_name"]),
        role=TUTORIAL_ROLES[model_id],
        model_function=model_function,
        parameter_names=parameter_names,
        estimates=estimates,
        covariance=np.asarray(model_result["pcov"], dtype=float),
        predictions=predictions,
        residuals=y - predictions,
        metrics={},
        initial_values=np.asarray(initial_values, dtype=float),
        cv_scores={},
    )


# -----------------------------------------------------------------------------
# Summary calculations
# -----------------------------------------------------------------------------


def parameter_correlation_summary(
    covariance: np.ndarray,
    parameter_names: list[str],
) -> tuple[float, str]:
    """Return the largest absolute off-diagonal parameter correlation."""
    if covariance.ndim != 2 or covariance.shape[0] < 2:
        return np.nan, "N/A"
    diagonal = np.diag(covariance)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        return np.nan, "N/A"

    standard_errors = np.sqrt(diagonal)
    denominator = np.outer(standard_errors, standard_errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator

    best_value = np.nan
    best_pair = "N/A"
    for i in range(correlation.shape[0]):
        for j in range(i + 1, correlation.shape[1]):
            value = float(correlation[i, j])
            if not np.isfinite(value):
                continue
            if not np.isfinite(best_value) or abs(value) > abs(best_value):
                best_value = value
                best_pair = f"{parameter_names[i]}-{parameter_names[j]}"
    return best_value, best_pair


def format_number(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def build_output_tables(
    dataset: pd.DataFrame,
    fit_results: list[FitResult],
    noise_sd: float,
    random_seed: int,
    cv_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    n = len(dataset)

    evidence_rows: list[dict[str, Any]] = []
    summary_by_model: dict[str, dict[str, Any]] = {}
    parameter_rows: list[dict[str, Any]] = []

    for result in fit_results:
        n_parameters = len(result.estimates)
        residual_df = n - n_parameters
        cv_rmse = finite_mean(result.cv_scores["rmse"])
        cv_mae = finite_mean(result.cv_scores["mae"])
        fit_rmse = result.metrics["rmse"]
        cv_rmse_ratio = cv_rmse / fit_rmse if np.isfinite(cv_rmse) and fit_rmse > 0 else np.nan
        max_correlation, max_correlation_pair = parameter_correlation_summary(
            result.covariance,
            result.parameter_names,
        )

        row = {
            "model": result.model_name,
            "role": result.role,
            "nparams": n_parameters,
            "residualdf": residual_df,
            "rtwo": result.metrics["r_squared"],
            "adjrtwo": result.metrics["adjusted_r_squared"],
            "rmse": fit_rmse,
            "mae": result.metrics["mae"],
            "aicc": result.metrics["aicc"],
            "bic": result.metrics["bic"],
            "cvrmse": cv_rmse,
            "cvmae": cv_mae,
            "cvrmseratio": cv_rmse_ratio,
            "maxcorr": abs(max_correlation) if np.isfinite(max_correlation) else np.nan,
            "maxcorrpair": max_correlation_pair,
        }
        summary_by_model[result.model_id] = row
        evidence_rows.append(row)

        if (
            result.covariance.shape == (n_parameters, n_parameters)
            and np.all(np.isfinite(result.covariance))
            and np.all(np.diag(result.covariance) >= 0)
            and residual_df > 0
        ):
            standard_errors = np.sqrt(np.diag(result.covariance))
            t_critical = float(stats.t.ppf(0.975, residual_df))
        else:
            standard_errors = np.full(n_parameters, np.nan)
            t_critical = np.nan

        for parameter_name, estimate, standard_error in zip(
            result.parameter_names,
            result.estimates,
            standard_errors,
        ):
            half_width = t_critical * standard_error if np.isfinite(t_critical) else np.nan
            relative_standard_error = (
                100.0 * standard_error / abs(estimate)
                if np.isfinite(standard_error) and abs(estimate) > np.finfo(float).eps
                else np.nan
            )
            parameter_rows.append(
                {
                    "model": result.model_name,
                    "parameter": parameter_name,
                    "estimate": estimate,
                    "standarderror": standard_error,
                    "cihalfwidth": half_width,
                    "relativestandarderror": relative_standard_error,
                }
            )


    add_relative_evidence(evidence_rows, "aicc")
    for row in evidence_rows:
        model_id = next(
            result.model_id for result in fit_results if result.model_name == row["model"]
        )
        summary_by_model[model_id]["deltaaicc"] = row.get("delta_aicc", np.nan)
        summary_by_model[model_id]["aiccweight"] = row.get("aicc_weight", np.nan)

    summary_ordered = [summary_by_model[model_id] for model_id in MODEL_IDS]
    summary_columns = [
        "model",
        "role",
        "nparams",
        "residualdf",
        "rtwo",
        "adjrtwo",
        "rmse",
        "mae",
        "aicc",
        "deltaaicc",
        "aiccweight",
        "bic",
        "cvrmse",
        "cvmae",
        "cvrmseratio",
        "maxcorr",
        "maxcorrpair",
    ]
    summary_df = pd.DataFrame(summary_ordered)[summary_columns]
    summary_display = summary_df.copy()
    for column in (
        "rtwo",
        "adjrtwo",
        "rmse",
        "mae",
        "aicc",
        "deltaaicc",
        "bic",
        "cvrmse",
        "cvmae",
        "cvrmseratio",
        "maxcorr",
    ):
        summary_display[column] = summary_display[column].map(format_number)
    summary_display["aiccweight"] = summary_display["aiccweight"].map(
        lambda value: format_number(value, 4)
    )

    parameter_df = pd.DataFrame(parameter_rows)
    parameter_display = parameter_df.copy()
    for column in ("estimate", "standarderror", "cihalfwidth", "relativestandarderror"):
        parameter_display[column] = parameter_display[column].map(format_number)

    data_display = dataset.rename(
        columns={
            "true_response": "trueresponse",
            "observed_response": "observedresponse",
        }
    ).copy()
    for column in data_display.columns:
        data_display[column] = data_display[column].map(lambda value: format_number(value, 2))

    settings_rows = [
        {"settingname": "Generating model", "settingvalue": engine.models[TRUE_MODEL_ID]["display_name"]},
    ]
    true_parameter_names = list(engine.models[TRUE_MODEL_ID]["model_function"].__code__.co_varnames[1:1 + len(TRUE_PARAMETERS)])
    settings_rows.extend(
        {"settingname": f"True parameter {name}", "settingvalue": format_number(value, 3)}
        for name, value in zip(true_parameter_names, TRUE_PARAMETERS)
    )
    settings_rows.extend(
        [
            {"settingname": "Noise standard deviation", "settingvalue": format_number(noise_sd, 3)},
            {"settingname": "Dataset random seed", "settingvalue": str(random_seed)},
            {"settingname": "CV scheme", "settingvalue": "Shuffle split"},
            {"settingname": "CV splits", "settingvalue": str(cv_config["cv_n_splits"])},
            {"settingname": "CV test fraction", "settingvalue": format_number(cv_config["cv_test_size"], 2)},
            {"settingname": "CV random seed", "settingvalue": str(cv_config["cv_random_state"])},
        ]
    )
    settings_df = pd.DataFrame(settings_rows)

    return {
        "simulated_data.csv": data_display,
        "model_summary.csv": summary_display,
        "parameter_estimates.csv": parameter_display,
        "tutorial_settings.csv": settings_df,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def apply_clean_axis_style(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#767676")
    axis.tick_params(direction="out", width=0.8, colors="black")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fitted_curves(dataset: pd.DataFrame, fit_results: list[FitResult], output_path: Path) -> None:
    x = dataset["time"].to_numpy(dtype=float)
    y = dataset["observed_response"].to_numpy(dtype=float)
    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 600)

    fig, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.scatter(
        x,
        y,
        s=52,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        label="Simulated observations",
        zorder=5,
    )

    for result in fit_results:
        dense_y = result.model_function(dense_x, *result.estimates)
        axis.plot(
            dense_x,
            dense_y,
            color=MODEL_COLORS[result.model_id],
            linestyle=MODEL_LINESTYLES[result.model_id],
            linewidth=2.1,
            label=result.model_name,
        )

    axis.set_xlabel("Time")
    axis.set_ylabel("Response value")
    #axis.set_title("Candidate model fits to the same simulated dataset")
    axis.legend(frameon=False, loc="upper right")
    apply_clean_axis_style(axis)
    save_figure(fig, output_path)


def plot_residuals(dataset: pd.DataFrame, fit_results: list[FitResult], output_path: Path) -> None:
    x = dataset["time"].to_numpy(dtype=float)
    global_limit = max(float(np.nanmax(np.abs(result.residuals))) for result in fit_results)
    response_range = float(np.ptp(dataset["observed_response"].to_numpy(dtype=float)))
    global_limit = 1.12 * max(global_limit, 0.025 * response_range)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for index, (axis, result) in enumerate(zip(axes_flat, fit_results)):
        axis.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=1.1)
        axis.plot(
            x,
            result.residuals,
            color=MODEL_COLORS[result.model_id],
            linewidth=0.95,
            alpha=0.75,
        )
        axis.scatter(
            x,
            result.residuals,
            s=34,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=1.25,
            zorder=4,
        )
        axis.set_ylim(-global_limit, global_limit)
        axis.set_title(result.model_name, fontsize=10.2, fontweight="bold")
        if index // 2 == 1:
            axis.set_xlabel("Time")
        if index % 2 == 0:
            axis.set_ylabel("Residual")
        apply_clean_axis_style(axis)

    #fig.suptitle("Residuals versus time", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.0)
    save_figure(fig, output_path)


def plot_qq(fit_results: list[FitResult], output_path: Path) -> None:
    qq_data: list[tuple[np.ndarray, np.ndarray]] = []
    global_min = math.inf
    global_max = -math.inf

    for result in fit_results:
        residuals = np.asarray(result.residuals, dtype=float)
        residual_sd = float(np.std(residuals, ddof=1))
        standardized = (residuals - float(np.mean(residuals))) / residual_sd
        ordered = np.sort(standardized)
        probabilities = (np.arange(1, len(ordered) + 1, dtype=float) - 0.5) / len(ordered)
        theoretical = stats.norm.ppf(probabilities)
        qq_data.append((theoretical, ordered))
        global_min = min(global_min, float(np.min(theoretical)), float(np.min(ordered)))
        global_max = max(global_max, float(np.max(theoretical)), float(np.max(ordered)))

    padding = 0.10 * (global_max - global_min)
    lower = global_min - padding
    upper = global_max + padding

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for index, (axis, result, (theoretical, ordered)) in enumerate(
        zip(axes_flat, fit_results, qq_data)
    ):
        axis.plot([lower, upper], [lower, upper], color="#4D4D4D", linestyle="--", linewidth=1.0)
        axis.scatter(
            theoretical,
            ordered,
            s=34,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=1.2,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(result.model_name, fontsize=10.2, fontweight="bold")
        if index // 2 == 1:
            axis.set_xlabel("Theoretical quantile")
        if index % 2 == 0:
            axis.set_ylabel("Ordered standardized residual")
        apply_clean_axis_style(axis)

    #fig.suptitle("Normal Q-Q plots of fitted residuals", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.0)
    save_figure(fig, output_path)



def plot_dense_sample_diagnostics(
    dataset: pd.DataFrame,
    fit_results: list[FitResult],
    output_path: Path,
) -> None:
    """Plot fit, residual, and Q-Q diagnostics for the 61-point example."""
    x = dataset["time"].to_numpy(dtype=float)
    y = dataset["observed_response"].to_numpy(dtype=float)
    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 600)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45))
    fit_axis, residual_axis, qq_axis = axes

    fit_axis.scatter(
        x,
        y,
        s=18,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        label="Simulated observations",
        zorder=5,
    )
    for result in fit_results:
        fit_axis.plot(
            dense_x,
            result.model_function(dense_x, *result.estimates),
            color=MODEL_COLORS[result.model_id],
            linestyle=MODEL_LINESTYLES[result.model_id],
            linewidth=1.8,
            label=result.model_name,
        )
    fit_axis.set_xlabel("Time")
    fit_axis.set_ylabel("Response value")
    fit_axis.set_title("A. Data and fitted curves", fontsize=10.5, fontweight="bold")
    fit_axis.legend(frameon=False, fontsize=7.7, loc="upper right")
    apply_clean_axis_style(fit_axis)

    residual_axis.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=1.0)
    for result in fit_results:
        residual_axis.plot(
            x,
            result.residuals,
            color=MODEL_COLORS[result.model_id],
            linestyle=MODEL_LINESTYLES[result.model_id],
            linewidth=1.0,
            alpha=0.85,
        )
        residual_axis.scatter(
            x,
            result.residuals,
            s=19,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=0.9,
            label=result.model_name,
            zorder=4,
        )
    residual_axis.set_xlabel("Time")
    residual_axis.set_ylabel("Residual")
    residual_axis.set_title("B. Residuals versus time", fontsize=10.5, fontweight="bold")
    residual_axis.legend(frameon=False, fontsize=7.7, loc="best")
    apply_clean_axis_style(residual_axis)

    qq_series: list[tuple[FitResult, np.ndarray, np.ndarray]] = []
    lower = math.inf
    upper = -math.inf
    for result in fit_results:
        residuals = np.asarray(result.residuals, dtype=float)
        residual_sd = float(np.std(residuals, ddof=1))
        standardized = (residuals - float(np.mean(residuals))) / residual_sd
        ordered = np.sort(standardized)
        probabilities = (np.arange(1, len(ordered) + 1, dtype=float) - 0.5) / len(ordered)
        theoretical = stats.norm.ppf(probabilities)
        qq_series.append((result, theoretical, ordered))
        lower = min(lower, float(np.min(theoretical)), float(np.min(ordered)))
        upper = max(upper, float(np.max(theoretical)), float(np.max(ordered)))

    padding = 0.08 * (upper - lower)
    lower -= padding
    upper += padding
    qq_axis.plot([lower, upper], [lower, upper], color="#4D4D4D", linestyle="--", linewidth=1.0)
    for result, theoretical, ordered in qq_series:
        qq_axis.scatter(
            theoretical,
            ordered,
            s=20,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=0.9,
            label=result.model_name,
        )
    qq_axis.set_xlim(lower, upper)
    qq_axis.set_ylim(lower, upper)
    qq_axis.set_aspect("equal", adjustable="box")
    qq_axis.set_xlabel("Theoretical quantile")
    qq_axis.set_ylabel("Ordered standardized residual")
    qq_axis.set_title("C. Normal Q-Q plots", fontsize=10.5, fontweight="bold")
    qq_axis.legend(frameon=False, fontsize=7.7, loc="best")
    apply_clean_axis_style(qq_axis)

    fig.tight_layout(w_pad=1.4)
    save_figure(fig, output_path)

def generate_plots(
    dataset: pd.DataFrame,
    fit_results: list[FitResult],
    dense_dataset: pd.DataFrame,
    dense_fit_results: list[FitResult],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
        }
    )

    plot_fitted_curves(dataset, fit_results, output_dir / "fitted_curves.pdf")
    plot_residuals(dataset, fit_results, output_dir / "residuals_vs_time.pdf")
    plot_qq(fit_results, output_dir / "qq_plots.pdf")
    plot_dense_sample_diagnostics(
        dense_dataset,
        dense_fit_results,
        output_dir / "dense_sample_diagnostics.pdf",
    )


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TUTORIAL_DIR / "generated",
        help="Directory for generated PDF and CSV files.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Dataset random seed.")
    parser.add_argument("--noise-sd", type=float, default=DEFAULT_NOISE_SD, help="Gaussian noise SD.")
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED, help="Shuffle-split random seed.")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS, help="Number of shuffle splits.")
    parser.add_argument(
        "--cv-test-size",
        type=float,
        default=DEFAULT_CV_TEST_SIZE,
        help="Fraction of observations held out in each shuffle split.",
    )
    parser.add_argument(
        "--grid-search-points",
        type=int,
        default=DEFAULT_GRID_SEARCH_POINTS,
        help="Number of generic grid-search starts per model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.noise_sd < 0:
        raise ValueError("--noise-sd must be nonnegative.")
    if args.cv_splits < 1:
        raise ValueError("--cv-splits must be at least 1.")
    if not 0 < args.cv_test_size < 1:
        raise ValueError("--cv-test-size must be between 0 and 1.")
    if args.grid_search_points < 1:
        raise ValueError("--grid-search-points must be at least 1.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_name in ("cross_validation_predictions.pdf", "cv_predictions.csv"):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    dataset = generate_dataset(args.seed, args.noise_sd)
    x = dataset["time"].to_numpy(dtype=float)
    y = dataset["observed_response"].to_numpy(dtype=float)

    cv_config = {
        "cv_scheme": "shuffle_split",
        "cv_n_splits": args.cv_splits,
        "cv_test_size": args.cv_test_size,
        "cv_random_state": args.cv_seed,
    }

    fit_results: list[FitResult] = []
    for model_id in MODEL_IDS:
        fit_results.append(
            fit_candidate_model(
                model_id,
                x,
                y,
                grid_search_points=args.grid_search_points,
                cv_config=cv_config,
            )
        )

    dense_dataset = generate_dense_dataset(args.seed, args.noise_sd)
    dense_x = dense_dataset["time"].to_numpy(dtype=float)
    dense_y = dense_dataset["observed_response"].to_numpy(dtype=float)
    dense_fit_results = [
        fit_candidate_model_for_diagnostics(
            model_id,
            dense_x,
            dense_y,
            grid_search_points=args.grid_search_points,
        )
        for model_id in ("linear", "exponential")
    ]

    output_tables = build_output_tables(
        dataset,
        fit_results,
        noise_sd=args.noise_sd,
        random_seed=args.seed,
        cv_config=cv_config,
    )
    for filename, table in output_tables.items():
        table.to_csv(output_dir / filename, index=False)

    generate_plots(
        dataset,
        fit_results,
        dense_dataset,
        dense_fit_results,
        output_dir,
    )

    print(f"Generated tutorial files in: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
