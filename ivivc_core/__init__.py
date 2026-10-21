"""Reusable numerical core for IVIVC analysis."""

from .fitting import fit_profile
from .preprocessing import prepare_data, prepare_ivivc_data, prepare_time_course
from .registry import (
    ApproachSpec,
    ModelSpec,
    get_approach_spec,
    get_model_spec,
    iter_approach_specs,
    iter_model_specs,
)
from .types import (
    NormalizationRecord,
    PreparedIVIVCData,
    PreprocessingMetadata,
    PreprocessingWarning,
    ProfileFit,
    TimeCourse,
)

__all__ = [
    'ApproachSpec',
    'ModelSpec',
    'NormalizationRecord',
    'PreparedIVIVCData',
    'PreprocessingMetadata',
    'PreprocessingWarning',
    'ProfileFit',
    'TimeCourse',
    'fit_profile',
    'get_approach_spec',
    'get_model_spec',
    'iter_approach_specs',
    'iter_model_specs',
    'prepare_data',
    'prepare_ivivc_data',
    'prepare_time_course',
]
