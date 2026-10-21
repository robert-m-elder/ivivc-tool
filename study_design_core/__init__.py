"""Reusable numerical foundation for degradation study-design assessment."""

from .fitting import fit_model_once
from .observation import observation_model_from_residuals
from .reference import (
    evaluate_central_trajectory,
    evaluate_reference_trajectories,
    reference_from_profile_fit,
    reference_from_time_course,
    reference_from_trajectory_ensemble,
)
from .types import (
    DesignConstraints,
    ModelFitResult,
    ObservationModel,
    ReferenceTrajectoryEnsemble,
)

__all__ = [
    'DesignConstraints',
    'ModelFitResult',
    'ObservationModel',
    'ReferenceTrajectoryEnsemble',
    'evaluate_central_trajectory',
    'evaluate_reference_trajectories',
    'fit_model_once',
    'observation_model_from_residuals',
    'reference_from_profile_fit',
    'reference_from_time_course',
    'reference_from_trajectory_ensemble',
]
