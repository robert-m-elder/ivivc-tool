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

if 0:
    # usage example
    # python cli.py '/Users/robert.elder/OneDrive - FDA/IVIVC Tool/data/Literature_and_submission_data_v1.xlsx' --approach approach2 --model exponential --metric rmse --sheet 1: --out summary.csv

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt

    df = pd.read_excel('/Users/robert.elder/OneDrive - FDA/IVIVC Tool/data/Literature_and_submission_data_v1.xlsx')
    df['sheet_name'] = df['Name']
    df_fit = pd.read_csv('summary-stretched-alternative_interpolation.csv')
    #df_fit = pd.read_csv('summary-alternative_interpolation.csv')
    #df_fit = pd.read_csv('summary.csv')
    df_comb = pd.merge(df, df_fit.loc[df_fit['dataset']==1, ['sheet_name', 'tau_nominal', 'tau_sd']], how='left', on='sheet_name', suffixes=['_1','_2'])
    df_comb = pd.merge(df_comb, df_fit.loc[df_fit['dataset']==2, ['sheet_name', 'tau_nominal', 'tau_sd']], how='left', on='sheet_name', suffixes=['_1','_2'])
    df_comb['Material'] = df_comb['Material'].fillna('')
    df_comb['tau_ratio'] = df_comb['tau_nominal_1']/df_comb['tau_nominal_2']
    df_comb['tau_ratio_sd'] = np.abs(df_comb['tau_ratio']) * np.sqrt((df_comb['tau_sd_1'] / df_comb['tau_nominal_1'])**2 + (df_comb['tau_sd_2'] / df_comb['tau_nominal_2'])**2)
    v = df_comb['tau_ratio']
    v = v[~np.isnan(v)]

    mask_base = (
        (df_comb["Type"] == "IV/IV") &
        (df_comb["Quantity"].str.fullmatch("Mw|Mn|Mv|inherent viscosity", na=False))
    )

    mask_plla = df_comb["Material"].eq("PLLA")
    mask_not_plla = df_comb["Material"].str.contains("PLGA|DL", na=False)

    df_plla = df_comb.loc[mask_base & mask_plla] 
    df_not_plla = df_comb.loc[mask_base & mask_not_plla] 
    plla = df_plla["tau_ratio"].dropna()
    not_plla = df_not_plla["tau_ratio"].dropna()

    def compare(col):
        plla = df_comb.loc[mask_base & mask_plla, col].dropna()
        not_plla = df_comb.loc[mask_base & mask_not_plla, col].dropna()

        ks = sp.stats.ks_2samp(plla, not_plla, alternative="greater", method="auto")
        mw = sp.stats.mannwhitneyu(plla, not_plla, alternative="greater", method="auto")
        perm = sp.stats.permutation_test(
            (plla, not_plla),
            statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
            permutation_type="independent",
            alternative="greater",
            n_resamples=10000,
            random_state=0,
        )

        return {
            "variable": col,
            "n_PLLA": len(plla),
            "n_not_PLLA": len(not_plla),
            "PLLA_mean": plla.mean(),
            "not_PLLA_mean": not_plla.mean(),
            "PLLA_median": plla.median(),
            "not_PLLA_median": not_plla.median(),
            "KS_stat": ks.statistic,
            "KS_p_greater": ks.pvalue,
            "MWU_stat": mw.statistic,
            "MWU_p_greater": mw.pvalue,
            "Perm_diff_mean": plla.mean() - not_plla.mean(),
            "Perm_p_greater_mean": perm.pvalue,
        }

    def test_vs_1(x, group):
        x = x.dropna()

        sign_x = x[x != 1]
        sign_res = sp.stats.binomtest((sign_x > 1).sum(), len(sign_x), p=0.5, alternative="greater") if len(sign_x) else None

        log_x = x[x > 0]
        wilcox_res = sp.stats.wilcoxon(np.log(log_x), alternative="greater", zero_method="wilcox") if len(log_x) else None

        return {
            "group": group,
            "n": len(x),
            "mean": x.mean(),
            "median": x.median(),
            "sign_test_p_median_eq_1": np.nan if sign_res is None else sign_res.pvalue,
            "wilcoxon_logratio_p": np.nan if wilcox_res is None else wilcox_res.pvalue,
        }

    summary1 = pd.DataFrame([
        compare("tau_nominal_1"),
        compare("tau_nominal_2"),
        compare("tau_ratio"),
    ])
    print(summary1.to_string(index=False))

    summary2 = pd.DataFrame([
        test_vs_1(df_plla['tau_ratio'], "PLLA"),
        test_vs_1(df_not_plla['tau_ratio'], "not-PLLA"),
    ])
    print(summary2.to_string(index=False))

    plt.figure(figsize=(10,6)); plt.subplots_adjust(top=0.96, right=0.96, left=0.15, bottom=0.17)
    plt.ecdf(plla,c='b',lw=3,label='PLLA'); 
    plt.ecdf(not_plla,c='g',lw=3,label='P(DL)LA/PLGA'); 
    plt.xscale('log');
    plt.legend()
    plt.xlabel(r'$\tau_{vitro} / \tau_{vivo}$')
    plt.ylabel('cumulative probability')
    #plt.tight_layout()
    plt.savefig('/Users/robert.elder/OneDrive - FDA/IVIVC Tool/data/Analysis/tau-ratio-distribution.pdf', format='pdf'); plt.close()
    plt.show()

    plt.figure(figsize=(10,6)); plt.subplots_adjust(top=0.96, right=0.96, left=0.15, bottom=0.17)
    plt.ecdf(df_plla['tau_nominal_1'],c='b',lw=3,label='PLLA, in vitro');
    plt.ecdf(df_plla['tau_nominal_2'],c='lightblue',lw=3,label='PLLA, in vivo');
    plt.ecdf(df_not_plla['tau_nominal_1'],c='g',lw=3,label='P(DL)LA/PLGA, in vitro');
    plt.ecdf(df_not_plla['tau_nominal_2'],c='lightgreen',lw=3,label='P(DL)LA/PLGA, in vivo');
    plt.xscale('log');
    plt.legend()
    plt.xlabel(r'$\tau$ (weeks)')
    plt.ylabel('cumulative probability')
    #plt.tight_layout()
    plt.savefig('/Users/robert.elder/OneDrive - FDA/IVIVC Tool/data/Analysis/tau-distribution.pdf', format='pdf'); plt.close()
    plt.show()

