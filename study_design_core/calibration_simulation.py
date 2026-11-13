"""Compare explicit calibration sampling plans through the full IVIVC fit."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ivivc_core import AnalysisError, TimeCourse

from .calibration_assessment import IVIVCCalibrationAssessment, assess_ivivc_calibration
from .calibration_data import append_calibration_observations, _new_calibration_batch
from .calibration_performance import (
    CalibrationSimulationResult, _finish_calibration_simulation,
    _validate_assessment_options, _validate_output_truth, summarize_calibration_simulation,
)
from .types import ObservationModel, ReferenceTrajectoryEnsemble, SimulationPlan


_ENVIRONMENTS = ('in_vitro', 'in_vivo')


@dataclass(frozen=True)
class CalibrationDesign:
    """Additional times and independent replicate counts for either environment.

    An empty time vector and no observation model keep that environment's
    existing data fixed. Each nonempty vector has an ObservationModel, whose
    replicate count applies to every proposed time. Times already measured may
    be included to request additional independent replicates there.
    """

    in_vitro_times: object = ()
    in_vitro_observation_model: ObservationModel | None = None
    in_vivo_times: object = ()
    in_vivo_observation_model: ObservationModel | None = None

    def __post_init__(self):
        for side in _ENVIRONMENTS:
            times = np.asarray(getattr(self, side + '_times'), dtype=float)
            if times.ndim != 1 or not np.isfinite(times).all():
                raise ValueError(f'{side}_times must be a finite one-dimensional array.')
            if len(np.unique(times)) != len(times):
                raise ValueError('Use replicate counts rather than duplicate proposed times.')
            model = getattr(self, side + '_observation_model')
            if len(times):
                if not isinstance(model, ObservationModel):
                    raise TypeError(f'Proposed {side} times require an ObservationModel.')
            elif model is not None:
                raise ValueError(f'An empty {side} plan must omit its observation model.')
            times = np.sort(times).copy()
            times.setflags(write=False)
            object.__setattr__(self, side + '_times', times)


@dataclass
class CalibrationSimulationComparison:
    """Candidate results sharing trajectory selections and measurement errors."""

    simulation_plans: dict[str, SimulationPlan]
    results: dict[str, CalibrationSimulationResult]
    summaries: dict[str, dict]
    metadata: dict = field(default_factory=dict)


def _named_designs(designs):
    if not isinstance(designs, Mapping) or not designs:
        raise ValueError('Supply at least one named CalibrationDesign.')
    named = {}
    for label, design in designs.items():
        name = str(label)
        if name in named:
            raise ValueError('Calibration design labels must be distinct after conversion to strings.')
        if not isinstance(design, CalibrationDesign):
            raise TypeError('Each calibration candidate must be a CalibrationDesign.')
        named[name] = design
    return named


def _reference_pair(in_vitro_reference, in_vivo_reference):
    references = dict(zip(_ENVIRONMENTS, (in_vitro_reference, in_vivo_reference)))
    for name, reference in references.items():
        if reference is not None and not isinstance(reference, ReferenceTrajectoryEnsemble):
            raise TypeError(f'{name}_reference must be a ReferenceTrajectoryEnsemble or None.')
    return references


def _validate_paired_references(references):
    first, second = (references[name] for name in _ENVIRONMENTS)
    if first is None or second is None:
        raise ValueError('Paired reference draws require both calibration references.')
    if first.n_draws == 1 or second.n_draws == 1:
        return
    if (first.n_draws != second.n_draws
            or not np.array_equal(first.valid_draws, second.valid_draws)
            or not np.allclose(first.draw_weights, second.draw_weights, rtol=1e-12, atol=1e-15)):
        raise ValueError('Paired references require matching joint scenario rows, validity, and weights.')


def create_calibration_simulation_plans(
    designs: Mapping[str, CalibrationDesign],
    *,
    in_vitro_reference: ReferenceTrajectoryEnsemble | None = None,
    in_vivo_reference: ReferenceTrajectoryEnsemble | None = None,
    n_simulations: int = 1000,
    random_state=None,
    paired_reference_draws: bool = False,
) -> dict[str, SimulationPlan]:
    """Create shared source selections and standard-normal measurement draws.

    Separate reference ensembles are independent by default. Set
    ``paired_reference_draws=True`` only when their rows represent the same
    joint scenarios; deterministic one-row references broadcast. Errors remain
    independent between environments. A fixed environment with a reference
    uses one placeholder master time solely to retain its truth selections;
    that placeholder does not request or generate an extra measurement.
    """
    named = _named_designs(designs)
    references = _reference_pair(in_vitro_reference, in_vivo_reference)
    if (isinstance(n_simulations, (bool, np.bool_)) or not isinstance(n_simulations, Integral)
            or n_simulations < 1):
        raise ValueError('n_simulations must be a positive integer.')
    n_simulations = int(n_simulations)
    if paired_reference_draws:
        _validate_paired_references(references)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**63 - 1, size=3, dtype=np.int64)
    joint_indices = None
    if paired_reference_draws:
        joint = max(references.values(), key=lambda item: item.n_draws)
        joint_indices = np.random.default_rng(int(seeds[2])).choice(
            joint.n_draws, n_simulations, p=joint.draw_weights,
        )
    plans = {}
    for index, side in enumerate(_ENVIRONMENTS):
        reference = references[side]
        times = [getattr(design, side + '_times') for design in named.values()
                 if len(getattr(design, side + '_times'))]
        if reference is None:
            if times:
                raise ValueError(f'Proposed {side} measurements require an explicit generating reference.')
            continue
        master = np.unique(np.concatenate(times)) if times else np.array([reference.supported_time_range[0]])
        lower, upper = reference.supported_time_range
        if master[0] < lower or master[-1] > upper:
            raise ValueError(f'Proposed {side} times extend beyond the generating reference domain.')
        replicates = max((getattr(design, side + '_observation_model').replicates_per_time
                          for design in named.values() if len(getattr(design, side + '_times'))), default=1)
        child = np.random.default_rng(int(seeds[index]))
        selected = (np.zeros(n_simulations, dtype=int) if reference.n_draws == 1
                    else joint_indices.copy() if paired_reference_draws
                    else child.choice(reference.n_draws, n_simulations, p=reference.draw_weights))
        plans[side] = SimulationPlan(
            master, selected, child.standard_normal((n_simulations, int(replicates), len(master))),
            metadata={'common_random_numbers': True, 'error_scale': 'standard_normal',
                      'reference_draws': 'paired' if paired_reference_draws else 'independent'},
        )
    return plans


def _validate_plans(plans, references, designs, paired, n_simulations):
    if set(plans) != {name for name, ref in references.items() if ref is not None}:
        raise ValueError('Simulation plans must match the supplied calibration references.')
    counts = set()
    for side, plan in plans.items():
        if not isinstance(plan, SimulationPlan):
            raise TypeError('Calibration simulation plans must contain SimulationPlan objects.')
        counts.add(plan.n_simulations)
        reference = references[side]
        indices = plan.trajectory_draw_indices
        if (np.any(indices < 0) or np.any(indices >= reference.n_draws)
                or np.any(~reference.valid_draws[indices])
                or np.any(reference.draw_weights[indices] <= 0)):
            raise ValueError('Simulation plans select an invalid generating reference draw.')
        for design in designs.values():
            times = getattr(design, side + '_times')
            if len(times):
                plan.indices_for_times(times)
                if plan.n_replicates < getattr(design, side + '_observation_model').replicates_per_time:
                    raise ValueError('Simulation plan has too few replicate draws.')
    if len(counts) > 1:
        raise ValueError('Calibration simulation plans must have equal experiment counts.')
    if paired:
        _validate_paired_references(references)
        if all(references[name].n_draws > 1 for name in _ENVIRONMENTS):
            if not np.array_equal(plans['in_vitro'].trajectory_draw_indices,
                                  plans['in_vivo'].trajectory_draw_indices):
                raise ValueError('Paired plans must retain matching joint scenario selections.')
    return counts.pop() if counts else int(n_simulations)


def compare_calibration_designs(
    designs: Mapping[str, CalibrationDesign],
    *,
    assessment_options: Mapping[str, object],
    existing_in_vitro: TimeCourse | None = None,
    existing_in_vivo: TimeCourse | None = None,
    in_vitro_reference: ReferenceTrajectoryEnsemble | None = None,
    in_vivo_reference: ReferenceTrajectoryEnsemble | None = None,
    n_simulations: int = 1000,
    random_state=None,
    simulation_plans: Mapping[str, SimulationPlan] | None = None,
    paired_reference_draws: bool = False,
    truth: Mapping[str, object] | None = None,
) -> CalibrationSimulationComparison:
    """Generate additions, repeat preprocessing/fitting, and compare candidates.

    Existing rows stay fixed. New means use SD/sqrt(new count), while new
    individuals retain individual SD. A historical mean is never silently
    assigned a future SD or divided again. Future CV fitting errors are fixed
    at the central reference; measurement noise follows each generating draw.

    Approach 1 uses the same generated measurements with its ordinary shared
    time-mapping fit. When old and new rows have incomplete combined precision,
    numerical copies are assembled without inventing historical errors. Both
    original batches remain in each assessment's ``calibration_batches``
    metadata. Supplied precision drives simulation, not Approach 1 weighting.

    Only common response scaling is supported for comparisons: apply any
    prespecified normalization to data and references first. The shared
    assessment options permit no normalization or fixed one-minus reversal.
    References are explicit generating assumptions, not inferred from missing
    calibration data. Fits and evaluation failures stay in every result row.
    """
    named = _named_designs(designs)
    if isinstance(assessment_options, Mapping):
        target = assessment_options.get('prediction')
        if target is not None and (target is existing_in_vitro or target is existing_in_vivo):
            raise ValueError('Target prediction measurements must be independent of calibration measurements.')
    options = _validate_assessment_options(assessment_options)
    approach1 = options.get('approach_id', 'approach2') == 'approach1'
    if (isinstance(n_simulations, (bool, np.bool_)) or not isinstance(n_simulations, Integral)
            or n_simulations < 1):
        raise ValueError('n_simulations must be a positive integer.')
    references = _reference_pair(in_vitro_reference, in_vivo_reference)
    existing = dict(zip(_ENVIRONMENTS, (existing_in_vitro, existing_in_vivo)))
    for side in _ENVIRONMENTS:
        if existing[side] is not None:
            existing[side] = append_calibration_observations(existing[side], None)
        for design in named.values():
            times = getattr(design, side + '_times')
            if not len(times):
                if existing[side] is None:
                    raise ValueError(f'Each design requires existing or proposed {side} calibration data.')
                continue
            reference = references[side]
            if reference is None:
                raise ValueError(f'Proposed {side} measurements require an explicit generating reference.')
            lower, upper = reference.supported_time_range
            if times[0] < lower or times[-1] > upper:
                raise ValueError(f'Proposed {side} times extend beyond the generating reference domain.')
            model = getattr(design, side + '_observation_model')
            sigma = model.standard_deviation(reference.evaluate_central(times), times=times)
            if np.any(~np.isfinite(sigma)) or (np.any(sigma <= 0) and not np.all(sigma == 0)):
                raise ValueError('Calibration fitting requires positive errors or an entirely zero-noise model.')
            if np.all(sigma == 0) and model.kind == 'constant_cv' and model.value > 0:
                raise ValueError('Set a positive minimum_sd for zero central CV responses.')
            if existing[side] is not None and not approach1:
                known = existing[side].observation_uncertainty is not None
                if known != bool(np.any(sigma > 0)):
                    raise ValueError('Specify compatible historical uncertainty explicitly before appending weighted measurements.')
    plans = (create_calibration_simulation_plans(
        named, in_vitro_reference=in_vitro_reference, in_vivo_reference=in_vivo_reference,
        n_simulations=n_simulations, random_state=random_state,
        paired_reference_draws=paired_reference_draws,
    ) if simulation_plans is None else dict(simulation_plans))
    count = _validate_plans(plans, references, named, paired_reference_draws, n_simulations)
    if count < 1:
        raise ValueError('n_simulations must be a positive integer.')
    _validate_output_truth(options, truth, count)
    results, summaries = {}, {}
    for label, design in named.items():
        generated = {}
        fit_errors = {}
        for side in _ENVIRONMENTS:
            times = getattr(design, side + '_times')
            if not len(times):
                continue
            reference, plan = references[side], plans[side]
            model = getattr(design, side + '_observation_model')
            expected = reference.evaluate(times, draw_indices=plan.trajectory_draw_indices)
            noise_sd = model.standard_deviation(expected, times=times)
            columns = plan.indices_for_times(times)
            normals = plan.standard_normal_errors[:, :int(model.replicates_per_time), :][:, :, columns]
            with np.errstate(over='ignore', invalid='ignore'):
                generated[side] = expected[:, None, :] + noise_sd[:, None, :] * normals
            fit_errors[side] = model.standard_deviation(reference.evaluate_central(times), times=times)
        assessments, datasets = [], {side: [] for side in _ENVIRONMENTS}
        for experiment in range(count):
            assembled = {side: None for side in _ENVIRONMENTS}
            batches = {}
            try:
                for side in _ENVIRONMENTS:
                    additions = None
                    if side in generated:
                        old = existing[side]
                        kind = ('mean' if old is None else 'individual' if old.observation_uncertainty is None
                                else old.observation_uncertainty.response_kind)
                        additions = _new_calibration_batch(
                            getattr(design, side + '_times'), generated[side][experiment],
                            fit_errors[side], response_kind=kind, source='simulated_' + side + '_calibration',
                        )
                    old = existing[side]
                    if approach1:
                        batches[side] = {'existing': old, 'additions': additions}
                    if (approach1 and old is not None and additions is not None
                            and ((old.observation_uncertainty is None)
                                 != (additions.observation_uncertainty is None))):
                        # Precision is not used by this fit. Retain the full
                        # source batches below instead of fabricating missing SD.
                        old = TimeCourse(old.time, old.value, label=old.label)
                        additions = TimeCourse(additions.time, additions.value,
                                               label=additions.label)
                    assembled[side] = append_calibration_observations(old, additions)
            except (ValueError, TypeError, FloatingPointError) as exception:
                assessment = IVIVCCalibrationAssessment(errors={'generation': AnalysisError.from_exception(exception)})
            else:
                assessment = assess_ivivc_calibration(assembled['in_vitro'], assembled['in_vivo'], **options)
            if approach1:
                assessment.metadata['calibration_batches'] = deepcopy(batches)
            assessments.append(assessment)
            for side in _ENVIRONMENTS:
                datasets[side].append(assembled[side])
        metadata = {'design_label': label, 'common_random_numbers': True,
                    'reference_draws': 'paired' if paired_reference_draws else 'independent',
                    'calibration_measurement_noise_simulated': bool(generated),
                    'future_observation_variability_included': False,
                    'weighting': 'fixed_at_central_reference', 'n_simulations': count}
        if approach1:
            metadata.update(weighting='unweighted',
                            calibration_uncertainty_usage='simulation_only')
        for side in _ENVIRONMENTS:
            times = getattr(design, side + '_times')
            model = getattr(design, side + '_observation_model')
            metadata[side + '_additional_times'] = times.copy()
            metadata[side + '_additional_measurements'] = len(times) * (0 if model is None else int(model.replicates_per_time))
            metadata[side + '_observation_model'] = None if model is None else model.to_metadata()
            metadata[side + '_existing_rows'] = 0 if existing[side] is None else len(existing[side].time)
        results[label] = _finish_calibration_simulation(
            assessments, datasets['in_vitro'], datasets['in_vivo'], options=options,
            in_vitro_reference=in_vitro_reference, in_vivo_reference=in_vivo_reference,
            in_vitro_draw_indices=None if 'in_vitro' not in plans else plans['in_vitro'].trajectory_draw_indices,
            in_vivo_draw_indices=None if 'in_vivo' not in plans else plans['in_vivo'].trajectory_draw_indices,
            truth=truth, metadata=metadata,
        )
        summaries[label] = summarize_calibration_simulation(results[label])
    return CalibrationSimulationComparison(
        plans, results, summaries,
        metadata={'n_simulations': count, 'common_random_numbers': True,
                  'design_labels': tuple(named), 'reference_draws': 'paired' if paired_reference_draws else 'independent'},
    )
