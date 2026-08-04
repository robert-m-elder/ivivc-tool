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

DEFAULT_RANDOM_SEED = 1847
DEFAULT_CV_SEED = 12345
DEFAULT_CV_SPLITS = 100
DEFAULT_CV_TEST_SIZE = 0.25
DEFAULT_GRID_SEARCH_POINTS = 200
DEFAULT_NOISE_SD = 1.25

#TIME_POINTS = np.array(
#    [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 11.0, 16.0, 23.0, 32.0, 45.0, 60.0],
#    dtype=float,
#)

TIME_POINTS = np.array([0, 1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 48], dtype=float)

TRUE_MODEL_ID = "stretched_exponential"
TRUE_PARAMETERS = np.array([100.0, -0.31, 0.55], dtype=float)
#TRUE_PARAMETERS = np.array([100.0, -0.002, 2], dtype=float)
#TRUE_PARAMETERS = np.array([100.0, -0.16, 0.72], dtype=float)

MODEL_IDS = (
    "exponential",
    "stretched_exponential",
    "double_exponential",
)

TUTORIAL_ROLES = {
    "exponential": "Plausible but imperfect",
    "stretched_exponential": "Strong candidate",
    "double_exponential": "More flexible candidate",
}

# App-derived colors, with neutral gray added for the third model.
MODEL_COLORS = {
    "exponential": engine.colors.get("in_vitro", "#2E5F8A"),
    "stretched_exponential": engine.colors.get("in_vivo", "#8B3A6B"),
    "double_exponential": "#4D4D4D",
}

MODEL_LINESTYLES = {
    "exponential": "--",
    "stretched_exponential": "-",
    "double_exponential": ":",
}

MODEL_MARKERS = {
    "exponential": "s",
    "stretched_exponential": "o",
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
    cv_prediction_mean: np.ndarray
    cv_prediction_sd: np.ndarray
    cv_prediction_count: np.ndarray


# -----------------------------------------------------------------------------
# Data generation and fitting
# -----------------------------------------------------------------------------


def generate_dataset(random_seed: int, noise_sd: float) -> pd.DataFrame:
    """Generate one fixed stretched-exponential decay dataset."""
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


def tutorial_initial_points(model_id: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return deterministic candidate starts for the tutorial models.

    These are supplied to the app's generic grid-search initializer. They do not
    impose parameter bounds on the final nonlinear least-squares fit.
    """
    amplitude = max(float(np.nanmax(y)), float(y[0]), 1.0)

    if model_id == "exponential":
        return np.array(
            [
                [amplitude, -0.05],
                [amplitude, -0.10],
                [amplitude, -0.20],
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


def collect_shuffle_split_predictions(
    x: np.ndarray,
    y: np.ndarray,
    model_fit_function: Any,
    initial_values: np.ndarray,
    cv_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect held-out predictions using the app's shuffle-split definition."""
    predictions_by_observation: list[list[float]] = [[] for _ in range(len(x))]
    cv = engine.make_cross_validator(cv_config, approach_id="approach2")

    for train_index, test_index in cv.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model_result = model_fit_function(
                    x_train,
                    y_train,
                    p0=initial_values,
                )
            test_predictions = np.asarray(model_result["predict"](x_test), dtype=float)
        except Exception:
            continue

        if test_predictions.shape != test_index.shape or not np.all(np.isfinite(test_predictions)):
            continue
        for observation_index, predicted_value in zip(test_index, test_predictions):
            predictions_by_observation[int(observation_index)].append(float(predicted_value))

    means = np.full(len(x), np.nan, dtype=float)
    standard_deviations = np.full(len(x), np.nan, dtype=float)
    counts = np.zeros(len(x), dtype=int)

    for i, values in enumerate(predictions_by_observation):
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        means[i] = float(np.mean(arr))
        standard_deviations[i] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        counts[i] = int(arr.size)

    return means, standard_deviations, counts


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

    cv_prediction_mean, cv_prediction_sd, cv_prediction_count = collect_shuffle_split_predictions(
        x,
        y,
        model_spec["fit_model"],
        initial_values,
        cv_config,
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
        cv_prediction_mean=cv_prediction_mean,
        cv_prediction_sd=cv_prediction_sd,
        cv_prediction_count=cv_prediction_count,
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
    x = dataset["time"].to_numpy(dtype=float)
    y = dataset["observed_response"].to_numpy(dtype=float)
    n = len(dataset)

    evidence_rows: list[dict[str, Any]] = []
    summary_by_model: dict[str, dict[str, Any]] = {}
    parameter_rows: list[dict[str, Any]] = []
    cv_prediction_rows: list[dict[str, Any]] = []

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

        for time_value, observed_value, prediction_mean, prediction_sd, prediction_count in zip(
            x,
            y,
            result.cv_prediction_mean,
            result.cv_prediction_sd,
            result.cv_prediction_count,
        ):
            cv_prediction_rows.append(
                {
                    "model": result.model_name,
                    "time": time_value,
                    "observedresponse": observed_value,
                    "cvpredictionmean": prediction_mean,
                    "cvpredictionsd": prediction_sd,
                    "cvpredictioncount": prediction_count,
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

    settings_df = pd.DataFrame(
        [
            {"settingname": "Generating model", "settingvalue": engine.models[TRUE_MODEL_ID]["display_name"]},
            {"settingname": "True parameter a", "settingvalue": format_number(TRUE_PARAMETERS[0], 3)},
            {"settingname": "True parameter b", "settingvalue": format_number(TRUE_PARAMETERS[1], 3)},
            {"settingname": "True parameter c", "settingvalue": format_number(TRUE_PARAMETERS[2], 3)},
            {"settingname": "Noise standard deviation", "settingvalue": format_number(noise_sd, 3)},
            {"settingname": "Dataset random seed", "settingvalue": str(random_seed)},
            {"settingname": "CV scheme", "settingvalue": "Shuffle split"},
            {"settingname": "CV splits", "settingvalue": str(cv_config["cv_n_splits"])},
            {"settingname": "CV test fraction", "settingvalue": format_number(cv_config["cv_test_size"], 2)},
            {"settingname": "CV random seed", "settingvalue": str(cv_config["cv_random_state"])},
        ]
    )

    return {
        "simulated_data.csv": data_display,
        "model_summary.csv": summary_display,
        "parameter_estimates.csv": parameter_display,
        "cv_predictions.csv": pd.DataFrame(cv_prediction_rows),
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
    axis.set_title("Candidate model fits to the same simulated dataset")
    axis.legend(frameon=False, loc="upper right")
    apply_clean_axis_style(axis)
    save_figure(fig, output_path)


def plot_residuals(dataset: pd.DataFrame, fit_results: list[FitResult], output_path: Path) -> None:
    x = dataset["time"].to_numpy(dtype=float)
    global_limit = max(float(np.nanmax(np.abs(result.residuals))) for result in fit_results)
    response_range = float(np.ptp(dataset["observed_response"].to_numpy(dtype=float)))
    global_limit = 1.12 * max(global_limit, 0.025 * response_range)

    fig, axes = plt.subplots(len(fit_results), 1, figsize=(7.2, 7.7), sharex=True, sharey=True)
    for axis, result in zip(np.atleast_1d(axes), fit_results):
        axis.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=1.2)
        axis.plot(
            x,
            result.residuals,
            color=MODEL_COLORS[result.model_id],
            linewidth=1.0,
            alpha=0.75,
        )
        axis.scatter(
            x,
            result.residuals,
            s=42,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=1.4,
            zorder=4,
        )
        axis.set_ylim(-global_limit, global_limit)
        axis.set_ylabel("Residual")
        axis.set_title(result.model_name, loc="left", fontsize=10.5, fontweight="bold")
        apply_clean_axis_style(axis)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Residuals versus time", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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

    fig, axes = plt.subplots(1, len(fit_results), figsize=(9.0, 3.35), sharex=True, sharey=True)
    for axis, result, (theoretical, ordered) in zip(np.atleast_1d(axes), fit_results, qq_data):
        axis.plot([lower, upper], [lower, upper], color="#4D4D4D", linestyle="--", linewidth=1.1)
        axis.scatter(
            theoretical,
            ordered,
            s=40,
            marker=MODEL_MARKERS[result.model_id],
            facecolors="white",
            edgecolors=MODEL_COLORS[result.model_id],
            linewidths=1.3,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(result.model_name, fontsize=10.5, fontweight="bold")
        axis.set_xlabel("Theoretical quantile")
        apply_clean_axis_style(axis)

    axes[0].set_ylabel("Ordered standardized residual")
    fig.suptitle("Normal Q-Q plots of fitted residuals", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_cv_predictions(dataset: pd.DataFrame, fit_results: list[FitResult], output_path: Path) -> None:
    x = dataset["time"].to_numpy(dtype=float)
    y = dataset["observed_response"].to_numpy(dtype=float)

    fig, axes = plt.subplots(len(fit_results), 1, figsize=(7.2, 7.7), sharex=True, sharey=True)
    for axis, result in zip(np.atleast_1d(axes), fit_results):
        axis.scatter(
            x,
            y,
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=1.1,
            label="Observed",
            zorder=5,
        )
        finite = np.isfinite(result.cv_prediction_mean)
        axis.errorbar(
            x[finite],
            result.cv_prediction_mean[finite],
            yerr=result.cv_prediction_sd[finite],
            fmt=MODEL_MARKERS[result.model_id],
            color=MODEL_COLORS[result.model_id],
            markerfacecolor="white",
            markeredgewidth=1.3,
            markersize=6,
            capsize=2.5,
            linewidth=1.0,
            label="Mean held-out prediction +/- SD",
            zorder=4,
        )
        axis.set_ylabel("Response value")
        axis.set_title(result.model_name, loc="left", fontsize=10.5, fontweight="bold")
        apply_clean_axis_style(axis)

    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Shuffle-split cross-validation predictions", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_path)


def generate_plots(dataset: pd.DataFrame, fit_results: list[FitResult], output_dir: Path) -> None:
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
    plot_cv_predictions(dataset, fit_results, output_dir / "cross_validation_predictions.pdf")


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

    output_tables = build_output_tables(
        dataset,
        fit_results,
        noise_sd=args.noise_sd,
        random_seed=args.seed,
        cv_config=cv_config,
    )
    for filename, table in output_tables.items():
        table.to_csv(output_dir / filename, index=False)

    generate_plots(dataset, fit_results, output_dir)

    print(f"Generated tutorial files in: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
