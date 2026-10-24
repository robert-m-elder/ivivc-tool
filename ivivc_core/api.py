"""High-level numerical API shared by the IVIVC and study-design modules.

The functions in this module compose the lower-level preprocessing, fitting,
prediction, and uncertainty primitives.  They do not create Plotly figures,
HTML, reports, or Flask-specific objects.  Inputs are assumed to have already
been preprocessed onto the scale on which the analysis should be performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import ufloat

from .fitting import fit_profile
from .predictions import (
    predict_direct_time_scaling,
    predict_direct_value_scaling,
    predict_inverse_response,
    predict_tau_ratio,
    predict_time_mapping,
    predict_value_ratio_with_uncertainty,
)
from .registry import get_approach_spec, get_model_spec
from .types import (
    AnalysisError,
    IVIVCAnalysis,
    IVIVCModelFit,
    PredictedTrajectory,
    PreparedIVIVCData,
    ProfileFit,
    ResultBatch,
    TimeCourse,
    TrajectoryEnsemble,
)
from .uncertainty import (
    deterministic_trajectory_ensemble,
    sample_inverse_response_trajectories,
    sample_tau_ratio_trajectories,
    sample_time_mapping_trajectories,
    sample_value_ratio_trajectories,
)


DEFAULT_PREDICTION_METHODS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    'approach1': ('time_mapping',),
    'approach2': (
        'value_ratio',
        'tau_ratio',
        'observed_response_inverse',
    ),
    'approach3': (
        'value_scaling',
        'time_scaling',
    ),
})


def _as_string_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _analysis_error(exception: BaseException) -> AnalysisError:
    return AnalysisError.from_exception(exception)


def _record_failure(
    errors: dict[str, AnalysisError],
    key: str,
    exception: BaseException,
    *,
    fail_fast: bool,
) -> None:
    if fail_fast:
        raise exception
    errors[key] = _analysis_error(exception)


def _selection_parts(selection: str) -> tuple[str, str | None]:
    if ':' in selection:
        approach_id, model_name = selection.split(':', 1)
        if not approach_id or not model_name:
            raise ValueError(f'Invalid IVIVC selection: {selection!r}')
        return approach_id, model_name
    return selection, None


def _validate_selection(approach_id: str, model_name: str | None) -> None:
    get_approach_spec(approach_id)
    if approach_id == 'approach3':
        if model_name is not None:
            raise ValueError('Approach 3 does not accept a parametric model.')
        return
    if model_name is None:
        raise ValueError(f'{approach_id} requires a model name.')
    model_spec = get_model_spec(model_name)
    if approach_id not in model_spec.approaches:
        raise ValueError(
            f'Model {model_name!r} is not registered for {approach_id!r}.'
        )


def _resolved_prediction_methods(
    methods: Mapping[str, Sequence[str] | str] | None,
) -> dict[str, tuple[str, ...]]:
    if methods is None:
        return dict(DEFAULT_PREDICTION_METHODS)

    resolved: dict[str, tuple[str, ...]] = {}
    for approach_id, selected_methods in methods.items():
        get_approach_spec(approach_id)
        method_names = _as_string_tuple(selected_methods)
        allowed = DEFAULT_PREDICTION_METHODS[approach_id]
        unknown = [name for name in method_names if name not in allowed]
        if unknown:
            raise ValueError(
                f'Unsupported prediction method(s) for {approach_id}: '
                + ', '.join(repr(name) for name in unknown)
            )
        resolved[approach_id] = method_names
    return resolved


def _prediction_key(model_fit: IVIVCModelFit, method: str) -> str:
    return f'{model_fit.key}:{method}'


def _split_random_state(random_state: Any, count: int) -> list[Any]:
    if count < 1:
        return []
    if isinstance(random_state, (int, np.integer)):
        seed_sequence = np.random.SeedSequence(int(random_state))
        return [np.random.default_rng(seed) for seed in seed_sequence.spawn(count)]
    return [random_state] * count


def fit_time_course(
    time: ArrayLike,
    response: ArrayLike,
    *,
    model_names: Sequence[str] | str,
    selected_metrics: Sequence[str],
    analysis_config: Mapping[str, object] | None = None,
    approach_id: str | None = None,
    calculate_tau: bool = False,
    config_defaults: Mapping[str, object] | None = None,
    fail_fast: bool = False,
) -> ResultBatch[ProfileFit]:
    """Fit one measured time course with one or more registered models.

    This is the primary entry point for an in vivo-only design-assessment
    workflow.  Models are fitted independently; by default one failed model is
    returned in ``errors`` without discarding the successful fits.
    """
    fits: dict[str, ProfileFit] = {}
    errors: dict[str, AnalysisError] = {}

    for model_name in _as_string_tuple(model_names):
        try:
            if approach_id is not None:
                _validate_selection(approach_id, model_name)
            fits[model_name] = fit_profile(
                time,
                response,
                model_name=model_name,
                selected_metrics=selected_metrics,
                analysis_config=analysis_config,
                approach_id=approach_id,
                calculate_tau=calculate_tau,
                config_defaults=config_defaults,
            )
        except Exception as exception:
            _record_failure(
                errors,
                model_name,
                exception,
                fail_fast=fail_fast,
            )

    return ResultBatch(results=fits, errors=errors)


def fit_ivivc(
    prepared_data: PreparedIVIVCData,
    *,
    selections: Sequence[str] | str,
    selected_metrics: Sequence[str],
    analysis_config: Mapping[str, object] | None = None,
    config_defaults: Mapping[str, object] | None = None,
    fail_fast: bool = False,
) -> IVIVCAnalysis:
    """Fit selected IVIVC approaches without creating presentation output.

    ``selections`` uses the existing stable selection keys, for example
    ``"approach1:linear"``, ``"approach2:exponential"``, and ``"approach3"``.
    Approach 1 fits the interpolated time-to-time relationship. Approach 2
    independently fits the observed in vitro and in vivo profiles. Approach 3
    records that the prepared interpolation mapping is available but performs
    no parametric fit.
    """
    if not prepared_data.has_in_vivo:
        raise ValueError('Paired in vitro and in vivo data are required for IVIVC fitting.')

    results: dict[str, IVIVCModelFit] = {}
    errors: dict[str, AnalysisError] = {}

    for selection in dict.fromkeys(_as_string_tuple(selections)):
        try:
            approach_id, model_name = _selection_parts(selection)
            _validate_selection(approach_id, model_name)

            if approach_id == 'approach1':
                assert model_name is not None
                if (
                    prepared_data.in_vitro_time_at_response is None
                    or prepared_data.in_vivo_time_at_response is None
                ):
                    raise ValueError(
                        'Approach 1 requires response-aligned interpolation arrays.'
                    )
                mapping_fit = fit_profile(
                    prepared_data.in_vitro_time_at_response,
                    prepared_data.in_vivo_time_at_response,
                    model_name=model_name,
                    selected_metrics=selected_metrics,
                    analysis_config=analysis_config,
                    approach_id=approach_id,
                    calculate_tau=False,
                    config_defaults=config_defaults,
                )
                result = IVIVCModelFit(
                    approach_id=approach_id,
                    model_name=model_name,
                    time_mapping_fit=mapping_fit,
                )
            elif approach_id == 'approach2':
                assert model_name is not None
                in_vitro_fit = fit_profile(
                    prepared_data.in_vitro.time,
                    prepared_data.in_vitro.value,
                    model_name=model_name,
                    selected_metrics=selected_metrics,
                    analysis_config=analysis_config,
                    approach_id=approach_id,
                    calculate_tau=True,
                    config_defaults=config_defaults,
                )
                in_vivo_fit = fit_profile(
                    prepared_data.in_vivo.time,
                    prepared_data.in_vivo.value,
                    model_name=model_name,
                    selected_metrics=selected_metrics,
                    analysis_config=analysis_config,
                    approach_id=approach_id,
                    calculate_tau=True,
                    config_defaults=config_defaults,
                )
                result = IVIVCModelFit(
                    approach_id=approach_id,
                    model_name=model_name,
                    in_vitro_fit=in_vitro_fit,
                    in_vivo_fit=in_vivo_fit,
                )
            else:
                result = IVIVCModelFit(approach_id='approach3')

            results[result.key] = result
        except Exception as exception:
            _record_failure(
                errors,
                selection,
                exception,
                fail_fast=fail_fast,
            )

    return IVIVCAnalysis(
        prepared_data=prepared_data,
        model_results=results,
        errors=errors,
        selected_metrics=tuple(selected_metrics),
        analysis_config={} if analysis_config is None else analysis_config,
    )


def _tau_uncertain_value(fit: ProfileFit):
    if fit.tau_estimate is None or fit.tau_sd is None:
        return ufloat(0.0, 1.0)
    return ufloat(fit.tau_estimate, fit.tau_sd)


def predict_in_vivo(
    analysis: IVIVCAnalysis,
    prediction: TimeCourse,
    *,
    methods: Mapping[str, Sequence[str] | str] | None = None,
    confidence_level: float = 0.95,
    n_samples: int = 1000,
    fail_fast: bool = False,
) -> ResultBatch[PredictedTrajectory]:
    """Generate central in vivo trajectories from a structured IVIVC analysis.

    The prediction time course must already be on the same preprocessed scale as
    ``analysis.prepared_data``.  Use ``methods`` to restrict the default method
    set; for example ``{'approach2': ('value_ratio', 'tau_ratio')}``.
    Response-ratio bounds propagate fitted-parameter uncertainty only, holding
    the prediction inputs fixed; uncertainty definitions for other methods are
    unchanged. Use ``predict_value_ratio_with_uncertainty`` for a seeded call.
    """
    selected_methods = _resolved_prediction_methods(methods)
    trajectories: dict[str, PredictedTrajectory] = {}
    errors: dict[str, AnalysisError] = {}
    prepared = analysis.prepared_data
    if prepared.in_vivo is None:
        raise ValueError('Paired in vitro and in vivo data are required for prediction.')

    for model_fit in analysis.model_results.values():
        for method in selected_methods.get(model_fit.approach_id, ()):
            key = _prediction_key(model_fit, method)
            try:
                if model_fit.approach_id == 'approach1':
                    assert model_fit.model_name is not None
                    assert model_fit.time_mapping_fit is not None
                    trajectory = predict_time_mapping(
                        prediction,
                        model_name=model_fit.model_name,
                        parameters=model_fit.time_mapping_fit.parameters,
                    )
                elif model_fit.approach_id == 'approach2':
                    assert model_fit.model_name is not None
                    assert model_fit.in_vitro_fit is not None
                    assert model_fit.in_vivo_fit is not None
                    in_vitro_fit = model_fit.in_vitro_fit
                    in_vivo_fit = model_fit.in_vivo_fit
                    if method == 'value_ratio':
                        trajectory = predict_value_ratio_with_uncertainty(
                            prediction,
                            model_name=model_fit.model_name,
                            in_vitro_parameters=in_vitro_fit.parameters,
                            in_vitro_covariance=in_vitro_fit.covariance,
                            in_vivo_parameters=in_vivo_fit.parameters,
                            in_vivo_covariance=in_vivo_fit.covariance,
                            confidence_level=confidence_level,
                            n_samples=n_samples,
                        )
                    elif method == 'tau_ratio':
                        trajectory = predict_tau_ratio(
                            prediction,
                            model_name=model_fit.model_name,
                            tau_values=(
                                _tau_uncertain_value(in_vitro_fit),
                                _tau_uncertain_value(in_vivo_fit),
                            ),
                            residual_dof=(
                                in_vitro_fit.residual_dof,
                                in_vivo_fit.residual_dof,
                            ),
                            confidence_level=confidence_level,
                        )
                    else:
                        trajectory = predict_inverse_response(
                            prediction,
                            model_name=model_fit.model_name,
                            parameters=in_vivo_fit.parameters,
                            covariance=in_vivo_fit.covariance,
                            residuals=in_vivo_fit.residuals,
                            calibration_time_range=(
                                float(np.min(prepared.in_vivo.time)),
                                float(np.max(prepared.in_vivo.time)),
                            ),
                            observed_responses=prepared.in_vivo.value,
                            confidence_level=confidence_level,
                            n_samples=n_samples,
                        )
                else:
                    if method == 'value_scaling':
                        trajectory = predict_direct_value_scaling(
                            prediction,
                            time_grid=prepared.time_grid,
                            in_vitro_response_on_time_grid=(
                                prepared.in_vitro_response_on_time_grid
                            ),
                            in_vivo_response_on_time_grid=(
                                prepared.in_vivo_response_on_time_grid
                            ),
                        )
                    else:
                        trajectory = predict_direct_time_scaling(
                            prediction,
                            response_grid=prepared.response_grid,
                            in_vitro_time_at_response=(
                                prepared.in_vitro_time_at_response
                            ),
                            in_vivo_time_at_response=(
                                prepared.in_vivo_time_at_response
                            ),
                        )
                trajectories[key] = trajectory
            except Exception as exception:
                _record_failure(
                    errors,
                    key,
                    exception,
                    fail_fast=fail_fast,
                )

    return ResultBatch(results=trajectories, errors=errors)


def generate_trajectory_ensembles(
    analysis: IVIVCAnalysis,
    prediction: TimeCourse,
    *,
    methods: Mapping[str, Sequence[str] | str] | None = None,
    n_samples: int = 1000,
    random_state: Any = None,
    fail_fast: bool = False,
) -> ResultBatch[TrajectoryEnsemble]:
    """Propagate fitted uncertainty into complete predicted trajectory draws.

    Parametric approaches retain joint parameter draws across the full
    prediction grid. Approach 3 has no fitted covariance and is therefore
    returned as a deterministic one-draw ensemble when requested.
    """
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError('n_samples must be at least one.')

    selected_methods = _resolved_prediction_methods(methods)
    tasks: list[tuple[IVIVCModelFit, str]] = []
    for model_fit in analysis.model_results.values():
        tasks.extend(
            (model_fit, method)
            for method in selected_methods.get(model_fit.approach_id, ())
        )
    random_states = _split_random_state(random_state, len(tasks))

    ensembles: dict[str, TrajectoryEnsemble] = {}
    errors: dict[str, AnalysisError] = {}
    prepared = analysis.prepared_data
    if prepared.in_vivo is None:
        raise ValueError('Paired in vitro and in vivo data are required for prediction.')

    for (model_fit, method), method_random_state in zip(tasks, random_states):
        key = _prediction_key(model_fit, method)
        try:
            if model_fit.approach_id == 'approach1':
                assert model_fit.model_name is not None
                assert model_fit.time_mapping_fit is not None
                fit = model_fit.time_mapping_fit
                ensemble = sample_time_mapping_trajectories(
                    prediction,
                    model_name=model_fit.model_name,
                    parameters=fit.parameters,
                    covariance=fit.covariance,
                    n_samples=n_samples,
                    random_state=method_random_state,
                )
            elif model_fit.approach_id == 'approach2':
                assert model_fit.model_name is not None
                assert model_fit.in_vitro_fit is not None
                assert model_fit.in_vivo_fit is not None
                in_vitro_fit = model_fit.in_vitro_fit
                in_vivo_fit = model_fit.in_vivo_fit
                if method == 'value_ratio':
                    ensemble = sample_value_ratio_trajectories(
                        prediction,
                        model_name=model_fit.model_name,
                        in_vitro_parameters=in_vitro_fit.parameters,
                        in_vitro_covariance=in_vitro_fit.covariance,
                        in_vivo_parameters=in_vivo_fit.parameters,
                        in_vivo_covariance=in_vivo_fit.covariance,
                        n_samples=n_samples,
                        random_state=method_random_state,
                    )
                elif method == 'tau_ratio':
                    ensemble = sample_tau_ratio_trajectories(
                        prediction,
                        model_name=model_fit.model_name,
                        in_vitro_parameters=in_vitro_fit.parameters,
                        in_vitro_covariance=in_vitro_fit.covariance,
                        in_vivo_parameters=in_vivo_fit.parameters,
                        in_vivo_covariance=in_vivo_fit.covariance,
                        n_samples=n_samples,
                        random_state=method_random_state,
                    )
                else:
                    ensemble = sample_inverse_response_trajectories(
                        prediction,
                        model_name=model_fit.model_name,
                        parameters=in_vivo_fit.parameters,
                        covariance=in_vivo_fit.covariance,
                        calibration_time_range=(
                            float(np.min(prepared.in_vivo.time)),
                            float(np.max(prepared.in_vivo.time)),
                        ),
                        observed_responses=prepared.in_vivo.value,
                        n_samples=n_samples,
                        random_state=method_random_state,
                    )
            else:
                if method == 'value_scaling':
                    central = predict_direct_value_scaling(
                        prediction,
                        time_grid=prepared.time_grid,
                        in_vitro_response_on_time_grid=(
                            prepared.in_vitro_response_on_time_grid
                        ),
                        in_vivo_response_on_time_grid=(
                            prepared.in_vivo_response_on_time_grid
                        ),
                    )
                else:
                    central = predict_direct_time_scaling(
                        prediction,
                        response_grid=prepared.response_grid,
                        in_vitro_time_at_response=(
                            prepared.in_vitro_time_at_response
                        ),
                        in_vivo_time_at_response=(
                            prepared.in_vivo_time_at_response
                        ),
                    )
                ensemble = deterministic_trajectory_ensemble(
                    central,
                    metadata={
                        'uncertainty_source': 'none',
                        'requested_samples': 1,
                    },
                )
            ensembles[key] = ensemble
        except Exception as exception:
            _record_failure(
                errors,
                key,
                exception,
                fail_fast=fail_fast,
            )

    return ResultBatch(results=ensembles, errors=errors)
