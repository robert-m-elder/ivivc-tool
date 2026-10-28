"""Synthetic model presets used by the study-design proof of concept.

The proof-of-concept runner intentionally uses the same registered model to
both generate and analyze each synthetic degradation trajectory.  This module
contains a small, explicit set of example parameterizations; models without a
preset are not silently substituted with another curve form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ivivc_core import get_model_spec


@dataclass(frozen=True)
class SyntheticModelPreset:
    """One registered model and a deterministic example parameterization."""

    model_name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    time_range: tuple[float, float]

    @property
    def parameter_mapping(self) -> Mapping[str, float]:
        return dict(zip(self.parameter_names, self.parameters))

    def evaluate(self, time: np.ndarray) -> np.ndarray:
        spec = get_model_spec(self.model_name)
        values = spec.model_function(
            np.asarray(time, dtype=float),
            *self.parameters,
        )
        values = np.asarray(values, dtype=float)
        if values.shape != np.asarray(time).shape:
            raise ValueError(
                f'Synthetic model {self.model_name!r} returned an unexpected shape.'
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f'Synthetic model {self.model_name!r} produced nonfinite values.'
            )
        return values

    def evaluate_time_scaled(
        self,
        time: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Evaluate the exact curve produced by scaling time by ``scale``."""
        parameters = self.time_scaled_parameter_mapping(scale)
        spec = get_model_spec(self.model_name)
        ordered = tuple(parameters[name] for name in self.parameter_names)
        values = spec.model_function(np.asarray(time, dtype=float), *ordered)
        values = np.asarray(values, dtype=float)
        if values.shape != np.asarray(time).shape:
            raise ValueError(
                f'Synthetic model {self.model_name!r} returned an unexpected shape.'
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f'Synthetic model {self.model_name!r} produced nonfinite values.'
            )
        return values

    def time_scaled_parameter_mapping(self, scale: float) -> Mapping[str, float]:
        """Return parameters for ``y(t / scale)`` in the same model family."""
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive.')
        parameters = dict(self.parameter_mapping)
        if self.model_name == 'mass_loss_one_phase':
            parameters['t50'] *= scale
            parameters['s'] *= scale
        elif self.model_name == 'mass_loss_one_phase_normalized':
            parameters['t50'] *= scale
            parameters['s'] *= scale
        elif self.model_name == 'exponential':
            parameters['b'] /= scale
        elif self.model_name == 'stretched_exponential':
            parameters['b'] /= scale ** parameters['c']
        elif self.model_name == 'mass_loss_surface_erosion':
            parameters['k'] /= scale
        else:  # pragma: no cover - guarded by the supported preset list
            raise ValueError(
                f'Time scaling is not defined for {self.model_name!r}.'
            )
        return parameters


SUPPORTED_SYNTHETIC_MODELS = (
    'mass_loss_one_phase',
    'mass_loss_one_phase_normalized',
    'exponential',
    'mass_loss_surface_erosion',
    'stretched_exponential',
)


def supported_synthetic_models() -> tuple[str, ...]:
    """Return registered models with built-in proof-of-concept presets."""
    return SUPPORTED_SYNTHETIC_MODELS


def get_synthetic_model_preset(
    model_name: str,
    *,
    time_min: float,
    time_max: float,
    transition_fraction: float = 0.5,
    width_fraction: float = 0.09,
) -> SyntheticModelPreset:
    """Construct an example parameterization over a specified time domain.

    The presets are intentionally simple and are used only by the synthetic
    proof-of-concept examples.  They are not recommended parameter values for
    experimental studies.
    """
    model_name = str(model_name)
    if model_name not in SUPPORTED_SYNTHETIC_MODELS:
        supported = ', '.join(SUPPORTED_SYNTHETIC_MODELS)
        raise ValueError(
            f'No proof-of-concept synthetic preset is defined for {model_name!r}. '
            f'Supported models: {supported}.'
        )

    time_min = float(time_min)
    time_max = float(time_max)
    if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max <= time_min:
        raise ValueError('Synthetic time bounds must be finite and increasing.')
    if time_min < 0:
        raise ValueError('The built-in synthetic presets require nonnegative time.')
    if not 0 < transition_fraction < 1:
        raise ValueError('transition_fraction must lie strictly between zero and one.')
    if not np.isfinite(width_fraction) or width_fraction <= 0:
        raise ValueError('width_fraction must be finite and positive.')

    span = time_max - time_min
    terminal_fraction = 0.02

    if model_name == 'mass_loss_one_phase':
        parameters = (
            1.0,
            0.8,
            time_min + transition_fraction * span,
            width_fraction * span,
        )
    elif model_name == 'mass_loss_one_phase_normalized':
        parameters = (
            time_min + transition_fraction * span,
            width_fraction * span,
        )
    elif model_name == 'exponential':
        rate = np.log(terminal_fraction) / span
        amplitude = np.exp(-rate * time_min)
        parameters = (float(amplitude), float(rate))
    elif model_name == 'stretched_exponential':
        shape = 2.0  # c < 1 gives a stretched exponential
        rate = np.log(terminal_fraction) / (
            time_max**shape - time_min**shape
        )
        amplitude = np.exp(-rate * time_min**shape)
        parameters = (
            float(amplitude),
            float(rate),
            float(shape),
        )
    else:  # mass_loss_surface_erosion
        exponent = 1.5
        erosion_rate = 1.0 / time_max
        initial_fraction = max(1.0 - erosion_rate * time_min, np.finfo(float).eps)
        initial_level = 1.0 / initial_fraction**exponent
        final_level = 0.0
        parameters = (float(initial_level), float(erosion_rate), exponent, final_level)

    spec = get_model_spec(model_name)
    if len(parameters) != len(spec.parameter_names):
        raise ValueError(
            f'Preset for {model_name!r} does not match the registered parameter count.'
        )
    preset = SyntheticModelPreset(
        model_name=model_name,
        parameter_names=spec.parameter_names,
        parameters=tuple(float(value) for value in parameters),
        time_range=(time_min, time_max),
    )
    # Validate the preset at representative points before returning it.
    preset.evaluate(np.linspace(time_min, time_max, 101))
    return preset



def generate_noisy_monotonic_responses(
    true_response: np.ndarray,
    *,
    absolute_sd: float,
    random_state: int | np.random.Generator | None = None,
    minimum_response: float | None = None,
    max_attempts: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Add absolute response noise while preserving a decreasing profile.

    The IVIVC interpolation used by the app maps response levels back to time
    and therefore assumes a monotonic degradation profile.  Synthetic noise is
    redrawn when it reverses the observed ordering so the proof-of-concept data
    exercise the same interpolation path without creating an invalid response
    axis.  The returned offsets are the realized additive response errors.
    """
    true_response = np.asarray(true_response, dtype=float)
    if true_response.ndim != 1 or len(true_response) < 2:
        raise ValueError('true_response must be a one-dimensional array of length two or more.')
    if not np.all(np.isfinite(true_response)):
        raise ValueError('true_response must contain only finite values.')
    if np.any(np.diff(true_response) >= 0):
        raise ValueError('true_response must be strictly decreasing.')

    absolute_sd = float(absolute_sd)
    if not np.isfinite(absolute_sd) or absolute_sd < 0:
        raise ValueError('absolute_sd must be finite and nonnegative.')
    if minimum_response is not None:
        minimum_response = float(minimum_response)
        if not np.isfinite(minimum_response):
            raise ValueError('minimum_response must be finite when supplied.')
    if int(max_attempts) != max_attempts or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer.')

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    for attempt in range(1, int(max_attempts) + 1):
        offsets = (
            np.zeros_like(true_response)
            if absolute_sd == 0
            else rng.normal(0.0, absolute_sd, size=len(true_response))
        )
        observed = true_response + offsets
        if minimum_response is not None and np.any(observed < minimum_response):
            continue
        if np.all(np.isfinite(observed)) and np.all(np.diff(observed) < 0):
            return observed, offsets, attempt

    raise RuntimeError(
        'Unable to generate a strictly decreasing noisy response profile '
        f'after {max_attempts} attempts.'
    )
