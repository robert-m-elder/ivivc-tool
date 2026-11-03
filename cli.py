from pathlib import Path
import argparse
import glob
import json
import math
import warnings

import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import UndefinedMetricWarning

from engine import (
    num_cores_for_grid_search,
    num_points_for_grid_search,
    preprocess_data,
    process_data,
    process_predictions,
)
from metrics import metrics
from models import approaches, models
from preprocessing import preprocessing_options
from utilities.uncertainty_inputs import uncertainty_from_options, validate_uncertainty_options

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
    if len(df.columns) < 4:
        raise ValueError('Fitting data require four columns: in vitro time/value, then in vivo time/value.')
    return tuple(pd.to_numeric(df.iloc[:, index], errors='raise').to_numpy() for index in range(4))


def _json_values(value):
    def clean(item):
        if hasattr(item, 'tolist'):
            item = item.tolist()
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), separators=(',', ':'), allow_nan=False)


def _uncertainty_columns(raw, analyzed, preprocessing_metadata=None):
    """Record supplied and analyzed precision without converting SEM again."""
    row = {'uncertainty_mode': raw.get('mode', 'none') if raw else 'none'}
    if analyzed:
        row.update({
            'response_kind': analyzed['response_kind'],
            'uncertainty_kind': analyzed['uncertainty_kind'],
            'standard_uncertainty': _json_values(analyzed['standard_uncertainty']),
            'replicate_counts': _json_values(analyzed.get('replicate_counts')),
            'uncertainty_column': raw.get('column') if raw else None,
            'replicate_column': raw.get('replicate_column') if raw else None,
            'entered_standard_uncertainty': _json_values(raw.get('standard_uncertainty')) if raw else None,
            'uncertainty_interpretation': 'SD for individual values; SEM for means; supplied absolute scale',
            'independent_errors_confirmed': True,
            'normalization_constants_fixed': True,
        })
        if preprocessing_metadata:
            row['normalization_records'] = _json_values(preprocessing_metadata.get('normalization_records', []))
    return row


def _prediction_rows(data, frame, results, approach, model, base, *, interpolation,
                     scalings, normalizations, options):
    supplied, record = uncertainty_from_options(frame, 'prediction', options)
    if len(frame.columns) < 2:
        raise ValueError('Prediction data require time and response in the first two columns.')
    time, response = (pd.to_numeric(frame.iloc[:, index], errors='raise').to_numpy() for index in range(2))
    preparation_options = ({} if supplied is None else
                           dict(return_prepared=True, in_vitro_uncertainty=supplied))
    prediction_data = preprocess_data(
        time, response, None, None, selected_interpolation=[interpolation],
        selected_scalings=scalings, selected_normalizations=normalizations,
        **preparation_options,
    )
    retained_rows = None
    if supplied is not None:
        retained_rows = next(iter(prediction_data.preprocessing.sources.values())).retained_rows
        prediction_data = prediction_data.in_vitro
    predictions = process_predictions(data, prediction_data, results, [approach],
                                      interpolated_data=data, selected_scalings=scalings)
    payload = predictions.get(approach, {}).get(model, {})
    if 'error' in payload:
        return [base | {'error': payload['error']}]
    if not payload.get('prediction_tables'):
        return [base | {'error': 'No prediction result was produced for this model.'}]
    rows = []
    for method, values in payload['prediction_tables'].items():
        metadata = payload.get('uncertainty_metadata', {}).get(method, {})
        for index, value in enumerate(values):
            row = base | {'method': method} | value
            for key in ('prediction_inputs_fixed', 'prediction_times_fixed',
                        'prediction_input_uncertainty_propagated',
                        'prediction_calibration_independence_assumed',
                        'future_observation_variability_included'):
                if key in metadata:
                    row[key] = metadata[key]
            for source, description in metadata.get('calibration_uncertainty', {}).items():
                row[source + '_covariance_scaling'] = description.get('covariance_scaling')
            if record:
                row.update(prediction_uncertainty_mode=record['mode'],
                           prediction_uncertainty_column=record.get('column'),
                           entered_input_standard_uncertainty=record['standard_uncertainty'][retained_rows[index]],
                           independent_errors_confirmed=True, normalization_constants_fixed=True)
            rows.append(row)
    return rows

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
               interpolation, scalings, normalizations, analysis_config, *,
               uncertainty_options=None, prediction_frame=None, prediction_file=None,
               prediction_rows=None):
    if approach == "approach3":
        raise ValueError(
            "approach3 currently returns plots, not numeric fit stats. "
            "Use approach1 or approach2 for batch summaries."
        )

    if model not in approaches[approach]["models"]:
        raise ValueError(f"Model {model!r} is not available for {approach}.")

    options = uncertainty_options or {}
    if uncertainty_options is not None:
        validate_uncertainty_options(options, [approach], scalings,
                                     has_prediction=prediction_frame is not None)
    vitro_uncertainty, vitro_record = uncertainty_from_options(df, 'in_vitro', options)
    vivo_uncertainty, vivo_record = uncertainty_from_options(df, 'in_vivo', options)
    supplied_calibration = vitro_uncertainty is not None or vivo_uncertainty is not None
    preparation_options = {}
    if supplied_calibration:
        preparation_options = dict(return_prepared=True, return_metadata=True,
                                   in_vitro_uncertainty=vitro_uncertainty,
                                   in_vivo_uncertainty=vivo_uncertainty)
    t1, m1, t2, m2 = extract_fit_columns(df)
    data = preprocess_data(
        t1,
        m1,
        t2,
        m2,
        selected_interpolation=[interpolation],
        selected_scalings=scalings,
        selected_normalizations=normalizations,
        **preparation_options,
    )
    preprocessing_metadata = None
    if supplied_calibration:
        data, preprocessing_metadata = data

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

    result = results[model_key]
    rows = rows_from_result(file_path, sheet_idx, sheet_name, model_key, result)
    if supplied_calibration and 'error' not in result:
        analyzed = result.get('observation_uncertainty', [None, None])
        for row in rows:
            index = row['dataset'] - 1
            row.update(_uncertainty_columns((vitro_record, vivo_record)[index], analyzed[index],
                                           preprocessing_metadata))
            row['covariance_scaling'] = result['fit_metadata'][index].get('covariance_scaling', 'residual_scaled')
            row['information_criteria_note'] = result['fit_metadata'][index].get('information_criteria_note')

    if prediction_frame is not None and prediction_rows is not None:
        base = dict(file=str(file_path), sheet_index='' if sheet_idx is None else sheet_idx,
                    sheet_name='' if sheet_name is None else sheet_name, model_key=model_key,
                    prediction_file=str(prediction_file or ''))
        if 'error' in result:
            prediction_rows.append(base | {'error': result['error']})
        else:
            try:
                prediction_rows.extend(_prediction_rows(
                    data, prediction_frame, results, approach, model, base,
                    interpolation=interpolation, scalings=scalings,
                    normalizations=normalizations, options=options,
                ))
            except Exception as exc:
                # Preserve completed fitting summaries when a prediction input
                # fails, while recording the failure in the prediction output.
                prediction_rows.append(base | {'error': str(exc)})
    return rows


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


def build_parser():
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

    parser.add_argument("--cv-scheme", choices=["shuffle_split", "leave_one_timepoint_out", "leave_one_out", "leave_contiguous_block_out", "none"], default="shuffle_split")
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
    for prefix, label in (('in_vitro', 'in vitro calibration'), ('in_vivo', 'in vivo calibration'),
                          ('prediction', 'prediction input')):
        flag = prefix.replace('_', '-')
        parser.add_argument(f'--{flag}-uncertainty-mode', choices=['none', 'constant', 'column'], default='none',
                            help=f'Absolute response uncertainty for {label}; default: none.')
        parser.add_argument(f'--{flag}-uncertainty-value', type=float,
                            help='Positive constant SD (individual values) or SEM (means).')
        parser.add_argument(f'--{flag}-uncertainty-column', help='Exact header of the SD/SEM column.')
        parser.add_argument(f'--{flag}-response-kind', choices=['individual', 'mean'], default='individual',
                            help='individual uses SD; mean uses SEM without dividing it again.')
        parser.add_argument(f'--{flag}-replicate-column',
                            help='Optional replicate-count column for means; recorded without rescaling SEM.')
    parser.add_argument('--independent-errors', action='store_true',
                        dest='uncertainty_independence_confirmed',
                        help='Confirm independent original observations and prediction measurements separate from calibration.')
    parser.add_argument('--prediction', help='Optional prediction CSV/Excel file; first two columns are time and response.')
    parser.add_argument('--prediction-sheet', help='Zero-based worksheet selector for prediction Excel input; select exactly one sheet.')
    parser.add_argument('--prediction-out', help='Optional prediction CSV path; otherwise predictions are printed separately.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.prediction_out and not args.prediction:
        parser.error('--prediction-out requires --prediction.')
    try:
        validate_uncertainty_options(vars(args), [args.approach], args.scalings,
                                     has_prediction=bool(args.prediction))
        prediction_frame = None
        if args.prediction:
            prediction_frames = load_fit_dataframes(args.prediction, sheet_spec=args.prediction_sheet)
            if len(prediction_frames) != 1:
                raise ValueError('Select exactly one prediction worksheet with --prediction-sheet.')
            prediction_frame = prediction_frames[0][2]
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    analysis_config = build_analysis_config(args)

    files = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])

    rows, prediction_rows = [], []
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
                        uncertainty_options=vars(args),
                        prediction_frame=prediction_frame,
                        prediction_file=args.prediction,
                        prediction_rows=prediction_rows,
                    )
                )
        except Exception as e:
            error_row = {
                "file": str(path),
                "sheet_index": "",
                "sheet_name": "",
                "model_key": f"{args.approach}:{args.model}",
                "error": str(e),
            }
            rows.append(error_row)
            if args.prediction:
                prediction_rows.append(error_row | {'prediction_file': str(args.prediction)})

    out_df = pd.DataFrame(rows)

    if args.out:
        out_df.to_csv(args.out, index=False)
    else:
        print_results(out_df)
    if args.prediction:
        prediction_df = pd.DataFrame(prediction_rows)
        if args.prediction_out:
            prediction_df.to_csv(args.prediction_out, index=False)
        else:
            print('\nPrediction results:')
            print_results(prediction_df)


if __name__ == "__main__":
    main()
