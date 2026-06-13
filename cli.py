from pathlib import Path
import argparse
import glob
import warnings

import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import UndefinedMetricWarning

from engine import (
    num_cores_for_grid_search,
    num_points_for_grid_search,
    preprocess_data,
    process_data,
)
from metrics import metrics
from models import approaches, models
from preprocessing import preprocessing_options

def parse_sheet_spec(spec: str | None, sheet_names: list[str]) -> list[tuple[int, str]]:
    """
    Parse sheet selectors against workbook sheet_names.

    Supported examples:
      0        -> first sheet
      -1       -> last sheet
      0,2,4    -> sheets 0,2,4
      1:-1     -> Python-style slice, stop excluded
      2:       -> second-from-third sheet through end
      :3       -> first three sheets
      0,2:-1   -> mixed selectors

    User-facing sheet numbers are zero-based, matching Python and pandas.
    Negative numbers follow Python-style indexing from the end.
    Slice semantics match normal Python exactly: stop is excluded.
    """
    if spec is None:
        raise ValueError("--sheet is required for Excel input")

    n = len(sheet_names)
    selected: list[int] = []

    def normalize_index(token: str) -> int:
        idx = int(token)
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Sheet selector {token!r} is out of range for workbook with {n} sheets."
            )
        return idx

    def normalize_slice_bound(token: str | None):
        if token is None or token == "":
            return None
        return int(token)

    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue

        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid slice syntax: {part!r}")

            start = normalize_slice_bound(pieces[0])
            stop = normalize_slice_bound(pieces[1])
            step = normalize_slice_bound(pieces[2]) if len(pieces) == 3 else None

            rng = range(n)[slice(start, stop, step)]
            selected.extend(list(rng))
        else:
            selected.append(normalize_index(part))

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    return [(idx, sheet_names[idx]) for idx in ordered]


def load_fit_dataframes(path, sheet_spec=None):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return [(None, None, df)]

    xls = pd.ExcelFile(path)
    selected_sheets = parse_sheet_spec(sheet_spec, xls.sheet_names)

    dfs = []
    for sheet_idx, sheet_name in selected_sheets:
        df = pd.read_excel(path, sheet_name=sheet_name)
        dfs.append((sheet_idx, sheet_name, df))
    return dfs


def extract_fit_columns(df):
    # Same convention as the web app: first 4 columns are t1, m1, t2, m2
    t1, m1, t2, m2 = df.iloc[:, :4].to_numpy().T
    return t1, m1, t2, m2

def rows_from_result(file_path, sheet_idx, sheet_name, model_key, result):
    base = {
        "file": str(file_path),
        "sheet_index": "" if sheet_idx is None else sheet_idx,
        "sheet_name": "" if sheet_name is None else sheet_name,
        "model_key": model_key,
    }

    if "error" in result:
        return [base | {"error": result["error"]}]

    stats = result["stats"]
    rows = []

    if isinstance(stats, list):  # approach2
        for idx, stats_df in enumerate(stats, start=1):
            row = base | {"dataset": idx}
            for k, v in stats_df.iloc[0].items():
                row[k] = float(v)
            if "tau" in result:
                row["tau_nominal"] = float(result["tau"][idx - 1].n)
                row["tau_sd"] = float(result["tau"][idx - 1].s)
            rows.append(row)
    else:  # approach1
        row = dict(base)
        for k, v in stats.iloc[0].items():
            row[k] = float(v)
        rows.append(row)

    return rows


def build_analysis_config(args):
    return {
        "cv_scheme": args.cv_scheme,
        "cv_n_splits": args.cv_n_splits,
        "cv_test_size": args.cv_test_size,
        "cv_random_state": args.cv_random_state,
        "grid_search_num_points": args.grid_search_num_points,
        "grid_search_num_cores": args.grid_search_num_cores,
        "grid_search_param_min": args.grid_search_param_min,
        "grid_search_param_max": args.grid_search_param_max,
        "grid_search_random_state": args.grid_search_random_state,
    }


def run_one_df(file_path, sheet_idx, sheet_name, df, approach, model, metrics,
               interpolation, scalings, normalizations, analysis_config):
    if approach == "approach3":
        raise ValueError(
            "approach3 currently returns plots, not numeric fit stats. "
            "Use approach1 or approach2 for batch summaries."
        )

    if model not in approaches[approach]["models"]:
        raise ValueError(f"Model {model!r} is not available for {approach}.")

    t1, m1, t2, m2 = extract_fit_columns(df)
    data = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=[interpolation],
        selected_scalings=scalings,
        selected_normalizations=normalizations,
    )

    model_key = f"{approach}:{model}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        results = process_data(
            data,
            selected_models=[model_key],
            selected_approaches=[approach],
            selected_metrics=metrics,
            analysis_config=analysis_config,
        )

    return rows_from_result(file_path, sheet_idx, sheet_name, model_key, results[model_key])


def print_results(df: pd.DataFrame):
    if df.empty:
        print("No results.")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 80,
    ):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Files or glob patterns")
    parser.add_argument(
        "--sheet",
        help=(
            "Excel sheet selector using zero-based indexing. Supports single indices, "
            "negative indices, comma-separated lists, and Python-style slices. "
            "Examples: 0, -1, 0,2,4, 1:-1, 2:, :3"
        ),
    )
    parser.add_argument("--approach", required=True, choices=["approach1", "approach2"])
    parser.add_argument("--model", required=True, choices=sorted(models), help="Model name as used in your app")
    parser.add_argument("--metric", dest="metrics", action="append", choices=sorted(metrics), required=True)
    parser.add_argument(
        "--interpolation",
        choices=sorted(preprocessing_options["interpolation"]),
        default="default",
        help="Interpolation/alignment method. Default: default (shared-time interpolation).",
    )
    parser.add_argument("--scaling", dest="scalings", action="append", choices=sorted(preprocessing_options["scaling"]), default=[])
    parser.add_argument("--normalization", dest="normalizations", action="append", choices=sorted(preprocessing_options["normalization"]), default=[])

    parser.add_argument("--cv-scheme", choices=["shuffle_split", "kfold", "leave_one_out", "none"], default="shuffle_split")
    parser.add_argument("--cv-n-splits", type=int, default=20)
    parser.add_argument("--cv-test-size", type=float, default=0.25)
    parser.add_argument("--cv-random-state", type=int, default=12345)
    parser.add_argument("--grid-search-num-points", type=int, default=num_points_for_grid_search)
    parser.add_argument("--grid-search-num-cores", type=int, default=num_cores_for_grid_search)
    parser.add_argument("--grid-search-param-min", type=float, default=-1e6)
    parser.add_argument("--grid-search-param-max", type=float, default=1e6)
    parser.add_argument("--grid-search-random-state", type=int, default=12345)
    parser.add_argument(
        "--out",
        help="Optional output CSV path. If omitted, results are printed to stdout."
    )
    args = parser.parse_args()
    analysis_config = build_analysis_config(args)

    files = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])

    rows = []
    for file_path in files:
        path = Path(file_path)
        try:
            dataframes = load_fit_dataframes(path, sheet_spec=args.sheet)
            for sheet_idx, sheet_name, df in dataframes:
                rows.extend(
                    run_one_df(
                        path,
                        sheet_idx,
                        sheet_name,
                        df,
                        approach=args.approach,
                        model=args.model,
                        metrics=args.metrics,
                        interpolation=args.interpolation,
                        scalings=args.scalings,
                        normalizations=args.normalizations,
                        analysis_config=analysis_config,
                    )
                )
        except Exception as e:
            rows.append({
                "file": str(path),
                "sheet_index": "",
                "sheet_name": "",
                "model_key": f"{args.approach}:{args.model}",
                "error": str(e),
            })

    out_df = pd.DataFrame(rows)

    if args.out:
        out_df.to_csv(args.out, index=False)
    else:
        print_results(out_df)


if __name__ == "__main__":
    main()

# usage example
# python cli.py '/Users/robert.elder/OneDrive - FDA/IVIVC Tool/data/Literature_and_submission_data_v1.xlsx' --approach approach2 --model exponential --metric rmse --sheet 1: --out summary.csv
