"""Read-only adapters around the existing dynamic model registries."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import MappingProxyType
from typing import Any, Callable, Mapping

from models import approaches as _approaches
from models import models as _models


@dataclass(frozen=True)
class ModelSpec:
    """Stable numerical description of a model registered by the IVIVC app."""

    key: str
    display_name: str
    model_function: Callable[..., Any]
    fit_function: Callable[..., Any]
    parameter_names: tuple[str, ...]
    fit_kwargs: Mapping[str, Any]
    approaches: tuple[str, ...]
    description: str | None = None
    latex_equation: str | None = None
    initial_guess: Callable[..., Any] | None = None
    supports_tau: bool = False
    parameter_bounds: tuple[Any, Any] | None = None
    parameter_validator: Callable[[Mapping[str, float]], bool] | None = None


@dataclass(frozen=True)
class ApproachSpec:
    """Stable description of an IVIVC approach and its registered models."""

    key: str
    display_name: str
    models: tuple[str, ...]


def get_model_spec(model_name: str) -> ModelSpec:
    """Return a structured view of one entry in ``models.models``.

    The adapter is constructed on demand rather than cached so existing tests
    and callers that temporarily modify registry entries continue to work.
    """
    try:
        entry = _models[model_name]
    except KeyError as exc:
        raise KeyError(f'Unknown model: {model_name!r}') from exc

    model_function = entry['model_function']
    parameter_names = tuple(inspect.signature(model_function).parameters.keys())[1:]
    approach_keys = tuple(
        approach_name
        for approach_name, approach_entry in _approaches.items()
        if model_name in approach_entry.get('models', ())
    )

    return ModelSpec(
        key=model_name,
        display_name=entry.get('display_name', model_name.replace('_', ' ').title()),
        model_function=model_function,
        fit_function=entry['fit_model'],
        parameter_names=parameter_names,
        fit_kwargs=MappingProxyType(dict(entry.get('fit_kwargs', {}))),
        approaches=approach_keys,
        description=entry.get('description'),
        latex_equation=entry.get('latex_equation'),
        initial_guess=entry.get('initial_guess'),
        supports_tau=bool(
            entry.get('supports_tau', getattr(model_function, 'supports_tau', True))
        ),
        parameter_bounds=entry.get('parameter_bounds'),
        parameter_validator=entry.get('parameter_validator'),
    )


def iter_model_specs(approach_id: str | None = None) -> tuple[ModelSpec, ...]:
    """Return model specifications in the existing display order."""
    if approach_id is None:
        names = sorted(_models, key=lambda name: _models[name].get('display_name', name))
    else:
        names = get_approach_spec(approach_id).models
    return tuple(get_model_spec(name) for name in names)


def get_approach_spec(approach_id: str) -> ApproachSpec:
    """Return a structured view of one entry in ``models.approaches``."""
    try:
        entry = _approaches[approach_id]
    except KeyError as exc:
        raise KeyError(f'Unknown approach: {approach_id!r}') from exc
    return ApproachSpec(
        key=approach_id,
        display_name=entry.get('display_name', approach_id),
        models=tuple(entry.get('models', ())),
    )


def iter_approach_specs() -> tuple[ApproachSpec, ...]:
    return tuple(get_approach_spec(name) for name in _approaches)
