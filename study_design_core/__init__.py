"""Reusable numerical foundation for degradation study-design assessment."""

from .fitting import fit_model_once
from .information import (
    ensemble_i_optimality,
    finite_difference_sensitivities,
    fit_reference_model_ensemble,
    information_matrix,
    local_i_optimality,
    score_information_matrices,
    trapezoidal_integration_weights,
)
from .observation import observation_model_from_residuals
from .optimize import (
    coordinate_exchange_i_optimal_design,
    evaluate_i_optimal_design,
    greedy_i_optimal_designs,
    optimize_i_optimal_designs,
)
from .reference import (
    evaluate_central_trajectory,
    evaluate_reference_trajectories,
    reference_from_profile_fit,
    reference_from_time_course,
    reference_from_trajectory_ensemble,
)
from .types import (
    CoordinateExchangeResult,
    DesignConstraints,
    DesignInformationScore,
    DesignOptimizationResult,
    DesignStep,
    ModelFitResult,
    ObservationModel,
    ReferenceModelEnsemble,
    ReferenceTrajectoryEnsemble,
)

__all__ = [
    'CoordinateExchangeResult',
    'DesignConstraints',
    'DesignInformationScore',
    'DesignOptimizationResult',
    'DesignStep',
    'ModelFitResult',
    'ObservationModel',
    'ReferenceModelEnsemble',
    'ReferenceTrajectoryEnsemble',
    'coordinate_exchange_i_optimal_design',
    'ensemble_i_optimality',
    'evaluate_central_trajectory',
    'evaluate_i_optimal_design',
    'evaluate_reference_trajectories',
    'finite_difference_sensitivities',
    'fit_model_once',
    'fit_reference_model_ensemble',
    'greedy_i_optimal_designs',
    'information_matrix',
    'local_i_optimality',
    'observation_model_from_residuals',
    'optimize_i_optimal_designs',
    'reference_from_profile_fit',
    'reference_from_time_course',
    'reference_from_trajectory_ensemble',
    'score_information_matrices',
    'trapezoidal_integration_weights',
]
