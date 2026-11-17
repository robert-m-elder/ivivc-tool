/* Study-design requests complete synchronously; state lives in this browser page. */
(() => {
    'use strict';

    function initialize() {
        const root = document.getElementById('design-workspace');
        if (!root || root.dataset.ready === 'true') return;

        const form = document.getElementById('design-setup-form');
        const task = document.getElementById('task');
        const source = document.getElementById('reference_source');
        const inputs = document.getElementById('design-inputs');
        const results = document.getElementById('design-results');
        const content = document.getElementById('design-results-content');
        const resultStatus = document.getElementById('design-results-status');
        const summary = document.getElementById('design-errors');
        const draftStatus = document.getElementById('design-draft-status');
        const requestStatus = document.getElementById('design-request-status');
        const comparisonStatus = document.getElementById('design-comparison-status');
        const operationFeedback = document.getElementById('design-operation-feedback');
        const operationText = document.getElementById('design-operation-text');
        const navigation = document.getElementById('design-section-navigation');
        const actionBar = document.getElementById('design-calculation-bar');
        const sectionButtons = Array.from(root.querySelectorAll('[data-design-section]'));
        const calculate = document.getElementById('calculate-comparison');
        const completed = document.getElementById('has_completed');
        const measuredMode = document.getElementById('measured_adaptive_mode');
        const firstMode = document.getElementById('first_time_mode');
        const timingRows = Array.from(root.querySelectorAll('[data-timing-row]'));
        const observationKind = document.getElementById('observation_kind');
        const ivivcApproach = document.getElementById('ivivc_approach');
        const ivivcModel = document.getElementById('ivivc_model');
        const ivivcMethod = document.getElementById('ivivc_method');
        const ivivcChoices = new Map();
        let previousApproach = null;
        const calibrationEnvironment = document.getElementById('varied_environment');
        const calibrationApproach = document.getElementById('cal_approach');
        const calibrationModel = document.getElementById('cal_model');
        const calibrationMethod = document.getElementById('cal_method');
        const calibrationRelationship = document.getElementById('cal_relationship');
        const calibrationTarget = document.getElementById('cal_assess_target');
        const calibrationCandidates = Array.from(root.querySelectorAll('[data-calibration-candidate]'));
        const generateCandidates = document.getElementById('generate-calibration-candidates');
        const generationStatus = document.getElementById('calibration-generation-status');
        const generationContent = document.getElementById('calibration-generation-content');
        const calibrationChoices = new Map();
        let previousCalibrationApproach = null;
        const files = new WeakMap();
        let nextFile = 0;
        let pending = false;
        let calculatedSignature = null;
        let generating = false;
        let generatedSignature = null;
        let generatedSchedules = [];
        let generationNotice = '';
        let activeView = 'inputs';
        let activeSection = null;
        // Explicit edits and navigation should keep their focus even if the
        // user later returns the form values to the submitted signature.
        let draftRevision = 0;
        let navigationRevision = 0;
        const operations = new Map();

        function setOperation(key, active, message) {
            if (active) operations.set(key, message);
            else operations.delete(key);
            if (operationFeedback) operationFeedback.hidden = operations.size === 0;
            if (operationText) {
                const primary = operations.get('calculation') || operations.get('generation') || operations.values().next().value;
                operationText.textContent = primary ? primary + (operations.size > 1 ?
                    ' (' + (operations.size - 1) + ' other request' + (operations.size > 2 ? 's' : '') + ' running.)' : '') : '';
            }
            measureChrome();
        }

        root.addEventListener('design-request-activity', event => {
            const activity = event.detail || {};
            if (typeof activity.key !== 'string' || !activity.key.startsWith('import:')) return;
            if (activity.active === true) {
                requestStatus.textContent = '';
                requestStatus.dataset.state = 'busy';
            }
            setOperation(activity.key, activity.active === true,
                typeof activity.message === 'string' ? activity.message : 'Preparing data.');
        });

        function measureChrome() {
            if (!root.style || typeof root.style.setProperty !== 'function') return;
            for (const [element, property] of [[navigation, '--design-navigation-height'],
                [actionBar, '--design-action-bar-height']]) {
                if (!element || typeof element.getBoundingClientRect !== 'function') continue;
                const height = element.getBoundingClientRect().height;
                if (height > 0) root.style.setProperty(property, Math.ceil(height) + 'px');
            }
        }

        function focusTarget(target) {
            if (!target) return;
            measureChrome();
            if (target.tagName === 'DETAILS') target.open = true;
            for (let ancestor = target.parentElement; ancestor && ancestor !== root; ancestor = ancestor.parentElement) {
                if (ancestor.tagName === 'DETAILS') ancestor.open = true;
            }
            target.focus({preventScroll: true});
            if (typeof target.scrollIntoView === 'function') target.scrollIntoView({block: 'start', behavior: 'auto'});
        }

        function syncNavigation() {
            sectionButtons.forEach(button => {
                button.hidden = activeView !== 'inputs' ||
                    !['all', task.value].includes(button.dataset.sectionTask || 'all');
                button.setAttribute('aria-current', !button.hidden && button.dataset.designSection === activeSection ? 'location' : 'false');
            });
            root.querySelectorAll('[data-design-section-links]').forEach(row => { row.hidden = activeView !== 'inputs'; });
            measureChrome();
        }

        function noteInputChange() {
            draftRevision++;
            updateDraftStatus();
        }

        function planningAvailable() {
            return task.value === 'study' && ['measured', 'ivivc', 'scaling'].includes(source.value);
        }

        function calibrationAvailable() {
            return task.value === 'calibration' && calibrationEnvironment &&
                ['in_vitro', 'in_vivo', 'both'].includes(calibrationEnvironment.value);
        }

        function calculationAvailable() {
            return planningAvailable() || calibrationAvailable();
        }

        function syncButtons() {
            if (calculate) {
                calculate.disabled = pending || generating || !calculationAvailable();
                calculate.textContent = calculatedSignature === null ? 'Calculate comparison' : 'Recalculate comparison';
            }
            if (generateCandidates) {
                const requested = calibrationCandidates.some(row =>
                    row.dataset.calibrationActive === 'true' && ['in_vitro', 'in_vivo'].some(side => {
                        const kind = document.getElementById('cal_candidate_' + row.dataset.calibrationCandidate + '_' + side + '_kind');
                        return calibrationVaries(side) && kind && !kind.disabled && kind.value === 'generated';
                    }));
                generateCandidates.disabled = pending || generating || !calibrationAvailable() || !requested;
            }
            if (generationContent) generationContent.querySelectorAll('[data-copy-generated-candidate]').forEach(button => {
                button.disabled = pending || generating || generatedSignature === null || signature() !== generatedSignature;
            });
        }

        function updateGenerationStatus() {
            if (!generationStatus) return;
            if (generating) {
                generationStatus.textContent = 'Generating curve-recovery schedules. This request runs to completion without calibration simulations.';
            } else if (generatedSignature !== null) {
                generationStatus.textContent = signature() === generatedSignature ?
                    'Generated schedules match these inputs. Review the times, then copy a candidate to edit it or calculate the comparison.' :
                    'Inputs changed. The preview retains its original settings; generate again before copying a candidate.';
                if (generationNotice) generationStatus.textContent += ' ' + generationNotice;
            } else {
                generationStatus.textContent = generationNotice || 'Choose “Generate a curve-recovery schedule” for at least one environment to preview its times before comparison.';
            }
            syncButtons();
        }

        function setActive(container, active) {
            container.hidden = !active;
            container.querySelectorAll('input, select, textarea').forEach(field => {
                field.disabled = !active;
            });
        }

        function filterOptions(select, allowed, preferred) {
            const options = Array.from(select.querySelectorAll('option'));
            options.forEach(option => {
                option.hidden = !allowed(option);
                option.disabled = option.hidden;
            });
            const valid = options.filter(option => !option.disabled);
            const choice = [preferred, select.value].find(value => valid.some(option => option.value === value));
            select.value = choice || (valid.length ? valid[0].value : '');
        }

        function syncIVIVC(available) {
            if (!ivivcApproach || !ivivcModel || !ivivcMethod) return;
            const approach = ivivcApproach.value;
            if (previousApproach && previousApproach !== approach) {
                ivivcChoices.set(previousApproach, {model: ivivcModel.value, method: ivivcMethod.value});
            }
            const remembered = previousApproach !== approach && ivivcChoices.get(approach);
            filterOptions(ivivcModel, option => (option.dataset.ivivcApproaches || '').split(/\s+/).includes(approach), remembered && remembered.model);
            filterOptions(ivivcMethod, option => option.dataset.ivivcMethodApproach === approach, remembered && remembered.method);
            const enabled = available && source.value === 'ivivc';
            root.querySelectorAll('[data-ivivc-model-fields]').forEach(group => {
                setActive(group, enabled && approach !== 'approach3');
            });
            root.querySelectorAll('[data-ivivc-interpolation-fields]').forEach(group => {
                setActive(group, enabled && approach !== 'approach2');
            });
            previousApproach = approach;
        }

        function syncAdaptive(available) {
            const adaptive = available && Boolean(completed && completed.checked);
            const refit = adaptive && source.value === 'measured' &&
                (!measuredMode || measuredMode.value === 'refit_completed');
            root.querySelectorAll('[data-adaptive-controls]').forEach(group => {
                setActive(group, adaptive);
            });
            root.querySelectorAll('[data-measured-adaptive-options]').forEach(group => {
                setActive(group, adaptive && source.value === 'measured');
            });
            root.querySelectorAll('[data-measured-prior]').forEach(group => {
                setActive(group, available && source.value === 'measured' && !refit);
            });
            root.querySelectorAll('[data-completed-dataset]').forEach(group => {
                setActive(group, adaptive);
            });
            root.querySelectorAll('[data-first-specified]').forEach(group => {
                setActive(group, available && !(adaptive && firstMode && firstMode.value === 'latest_completed'));
            });
            root.querySelectorAll('[data-completed-scale-help]').forEach(note => {
                note.textContent = refit ?
                    'Enter completed responses and SD/SEM on their original scale. The selected preprocessing is applied once, and the same analyzed observations remain fixed during simulated-study refitting.' :
                    'Enter completed responses and SD/SEM already on the analyzed reference response scale. Do not normalize a partial completed dataset using its own initial value or extrema. These observations update the separate prior once.';
            });
        }

        function syncTiming(available) {
            timingRows.forEach(row => {
                const enabled = available && row.dataset.timingActive === 'true';
                setActive(row, enabled);
                const prefix = 'timing_' + row.dataset.timingRow + '_';
                row.querySelectorAll('[data-timing-mode]').forEach(group => {
                    const mode = document.getElementById(prefix + group.dataset.timingBound + '_mode');
                    setActive(group, enabled && mode && mode.value === group.dataset.timingMode);
                });
                const kind = document.getElementById(prefix + 'kind');
                row.querySelectorAll('[data-timing-window-fields]').forEach(group => {
                    setActive(group, enabled && kind && kind.value === 'window');
                });
                row.querySelectorAll('[data-remove-timing]').forEach(button => {
                    button.disabled = !enabled;
                });
            });
            root.querySelectorAll('[data-add-timing]').forEach(button => {
                button.disabled = !available || !timingRows.some(row => row.dataset.timingActive !== 'true');
            });
        }

        function calibrationVaries(side) {
            return calibrationEnvironment &&
                (calibrationEnvironment.value === 'both' || calibrationEnvironment.value === side);
        }

        function syncCalibration(available) {
            for (const side of ['in_vitro', 'in_vivo']) {
                const varied = calibrationVaries(side);
                const mode = document.getElementById('cal_' + side + '_mode');
                const sourceMode = document.getElementById('cal_' + side + '_source');
                const observation = document.getElementById('cal_' + side + '_observation_kind');
                const existing = !varied || Boolean(mode && mode.value === 'additions');
                if (sourceMode) {
                    filterOptions(sourceMode, option => option.value === 'file' ||
                        (option.value === 'existing' && existing) ||
                        (option.value === 'none' && !varied));
                }
                for (const [attribute, shown] of [
                    ['data-calibration-mode', varied],
                    ['data-calibration-existing', existing],
                    ['data-calibration-source-file', sourceMode && sourceMode.value === 'file'],
                    ['data-calibration-source-controls', sourceMode && sourceMode.value !== 'none'],
                    ['data-calibration-future', varied],
                    ['data-calibration-historical', varied && existing],
                ]) {
                    root.querySelectorAll('[' + attribute + '="' + side + '"]').forEach(group => {
                        setActive(group, available && shown);
                    });
                }
                root.querySelectorAll('[data-calibration-observation-side="' + side + '"]').forEach(group => {
                    const kinds = (group.dataset.calibrationObservationKinds || '').split(/\s+/);
                    setActive(group, available && varied && observation && kinds.includes(observation.value));
                });
            }
            if (calibrationApproach && calibrationModel && calibrationMethod) {
                const approach = calibrationApproach.value;
                if (previousCalibrationApproach && previousCalibrationApproach !== approach) {
                    calibrationChoices.set(previousCalibrationApproach,
                        {model: calibrationModel.value, method: calibrationMethod.value});
                }
                const remembered = previousCalibrationApproach !== approach && calibrationChoices.get(approach);
                filterOptions(calibrationModel, option =>
                    (option.dataset.calibrationApproaches || '').split(/\s+/).includes(approach), remembered && remembered.model);
                filterOptions(calibrationMethod, option =>
                    option.dataset.calibrationMethodApproach === approach, remembered && remembered.method);
                root.querySelectorAll('[data-calibration-interpolation]').forEach(group => {
                    setActive(group, available && approach === 'approach1');
                });
                const relationship = available && calibrationRelationship && calibrationRelationship.checked;
                root.querySelectorAll('[data-calibration-relationship]').forEach(group => {
                    setActive(group, relationship);
                });
                root.querySelectorAll('[data-calibration-evaluation]').forEach(group => {
                    const kind = group.dataset.calibrationEvaluation;
                    setActive(group, relationship && (kind === 'time' ?
                        approach === 'approach1' || calibrationMethod.value === 'value_ratio' :
                        kind === 'response' && calibrationMethod.value === 'observed_response_inverse'));
                });
                previousCalibrationApproach = approach;
            }
            root.querySelectorAll('[data-calibration-target]').forEach(group => {
                setActive(group, available && calibrationTarget && calibrationTarget.checked);
            });
            const activeCount = calibrationCandidates.filter(row => row.dataset.calibrationActive === 'true').length;
            calibrationCandidates.forEach(row => {
                const enabled = available && row.dataset.calibrationActive === 'true';
                setActive(row, enabled);
                for (const side of ['in_vitro', 'in_vivo']) {
                    const sideEnabled = enabled && calibrationVaries(side);
                    row.querySelectorAll('[data-calibration-candidate-side="' + side + '"]').forEach(group => {
                        setActive(group, sideEnabled);
                    });
                    const kind = document.getElementById('cal_candidate_' + row.dataset.calibrationCandidate + '_' + side + '_kind');
                    row.querySelectorAll('[data-calibration-schedule-side="' + side + '"]').forEach(group => {
                        setActive(group, sideEnabled && kind && kind.value === group.dataset.calibrationScheduleKind);
                    });
                }
                row.querySelectorAll('[data-remove-calibration-candidate]').forEach(button => {
                    button.disabled = !enabled || activeCount === 1;
                });
            });
            root.querySelectorAll('[data-add-calibration-candidate]').forEach(button => {
                button.disabled = !available || activeCount >= calibrationCandidates.length;
            });
        }

        function syncInputs() {
            root.querySelectorAll('[data-design-task]').forEach(group => {
                setActive(group, group.dataset.designTask === task.value);
            });
            root.querySelectorAll('[data-reference-source]').forEach(group => {
                setActive(group, task.value === 'study' && group.dataset.referenceSource === source.value);
            });
            const available = planningAvailable();
            root.querySelectorAll('[data-planning-controls]').forEach(group => {
                group.querySelectorAll('input, select, textarea').forEach(field => {
                    field.disabled = !available;
                });
            });
            root.querySelectorAll('[data-reference-options]').forEach(group => {
                setActive(group, available && group.dataset.referenceOptions === source.value);
            });
            syncIVIVC(available);
            syncAdaptive(available);
            syncCalibration(calibrationAvailable());
            root.querySelectorAll('[data-observation-kind]').forEach(group => {
                const kinds = group.dataset.observationKind.split(/\s+/);
                setActive(group, available && observationKind && kinds.includes(observationKind.value));
            });
            root.querySelectorAll('[data-planning-availability]').forEach(note => {
                note.textContent = calibrationAvailable() ?
                    'Ready to compare calibration sampling candidates through the selected IVIVC workflow.' : available ?
                    (completed && completed.checked ?
                        'Ready to assess additional sampling using completed observations and the selected reference source.' :
                        'Ready to calculate a prospective study using the selected reference source.') :
                    'Select a supported study-planning or calibration-comparison task.';
            });
            syncTiming(available);
            syncButtons();
            syncNavigation();
            root.dispatchEvent(new CustomEvent('design-visibility-updated'));
        }

        function signature(data = new FormData(form)) {
            return JSON.stringify(Array.from(data.entries(), ([key, value]) => [key,
                typeof value === 'string' ? value : (() => {
                    if (!value.name) return null;
                    if (!files.has(value)) files.set(value, ++nextFile);
                    return [value.name, value.size, value.lastModified, files.get(value)];
                })(),
            ]));
        }

        function showView(view) {
            activeView = view;
            if (view !== 'inputs') activeSection = null;
            inputs.hidden = view !== 'inputs';
            results.hidden = view !== 'results';
            root.querySelectorAll('[data-design-view]').forEach(button => {
                button.setAttribute('aria-pressed', String(button.dataset.designView === view));
            });
            if (view === 'results') plotAction('resize', results);
            syncNavigation();
        }

        function plotAction(action, container) {
            // Plot loading/rendering is independent of a successful calculation.
            // The renderer retains the static figure if the CDN is unavailable.
            try {
                if (typeof window !== 'undefined' && window.DesignPlots &&
                        typeof window.DesignPlots[action] === 'function') {
                    const rendering = window.DesignPlots[action](container);
                    if (rendering && typeof rendering.catch === 'function') rendering.catch(() => {});
                }
            } catch (error) {
                // Keep numerical results and their static plots available.
            }
        }

        function inspectionAction(action, container) {
            if (typeof window === 'undefined') return;
            // Comparison displays subscribe before the schedule controller
            // emits its initial selection, retaining the same result colors.
            for (const api of [window.DesignPerformanceInspection,
                    window.DesignRecoveryInspection, window.DesignScheduleInspection,
                    window.DesignCalibrationRecovery]) {
                try {
                    if (api && typeof api[action] === 'function') api[action](container);
                } catch (error) { /* Display controls cannot invalidate a successful calculation. */ }
            }
        }

        function updateDraftStatus() {
            updateSamplingSummaries();
            updateCalculationSummaries();
            updateGenerationStatus();
            if (calculatedSignature !== null) {
                const current = signature() === calculatedSignature;
                if (comparisonStatus) {
                    comparisonStatus.dataset.state = current ? 'current' : 'stale';
                    comparisonStatus.textContent = current ? 'Comparison is current.' :
                        'Comparison is out of date. Recalculate for these inputs.';
                }
                draftStatus.textContent = current ?
                    'The displayed comparison matches these settings.' :
                    'Inputs changed. The displayed comparison retains its original settings; calculate again to assess these choices.';
                if (resultStatus) resultStatus.textContent = current ?
                    'Comparison for the current inputs.' :
                    'Inputs changed. This comparison still uses its original inputs and settings.';
                measureChrome();
                return;
            }
            if (comparisonStatus) {
                comparisonStatus.dataset.state = 'none';
                comparisonStatus.textContent = 'No comparison yet.';
            }
            if (resultStatus) resultStatus.textContent = 'No numerical comparison has been calculated.';
            draftStatus.textContent = 'Draft settings. Choose Calculate comparison to assess these inputs.';
            measureChrome();
        }

        function updateSamplingSummaries() {
            const value = id => {
                const field = document.getElementById(id);
                return field ? String(field.value).trim() : '';
            };
            const short = text => text.length > 45 ? text.slice(0, 42) + '…' : text;
            const even = document.getElementById('include_even');
            const manual = value('manual_schedules').split(/\r?\n/).filter(line => line.trim()).length;
            const count = timingRows.filter(row => row.dataset.timingActive === 'true').length;
            const summaries = {
                alternatives: 'Evenly spaced: ' + (even && even.checked ? 'on' : 'off') + ' · Manual schedules: ' + manual,
                timing: count + ' requirement' + (count === 1 ? '' : 's'),
                advanced: 'Fixed times: ' + short(value('fixed_times') || 'none') +
                    ' · Milestone baseline: ' + short(value('baseline_time') || 'not specified') +
                    ' · Excluded times: ' + short(value('excluded_times') || 'none') +
                    (completed && completed.checked ? ' · New times: ' +
                        (value('future_policy') === 'gap_fill' ? 'gap filling allowed' : 'future only') : ''),
            };
            root.querySelectorAll('[data-sampling-summary]').forEach(summary => {
                summary.textContent = summaries[summary.dataset.samplingSummary] || '';
            });
        }

        function updateCalculationSummaries() {
            const field = id => document.getElementById(id);
            const value = id => String(field(id)?.value || '').trim();
            const short = text => text.length > 45 ? text.slice(0, 42) + '…' : text;
            const endpoints = [];
            for (const [id, label] of [['recovery_t50', 'T50'], ['recovery_t90', 'T90']]) {
                if (field(id)?.checked) endpoints.push(label);
            }
            if (value('recovery_tx_custom')) endpoints.push('Custom Tx: ' + short(value('recovery_tx_custom')));
            if (value('recovery_absolute_thresholds')) endpoints.push('Absolute ' +
                (value('recovery_absolute_direction') === 'increasing' ? '≥ ' : '≤ ') +
                short(value('recovery_absolute_thresholds')));
            const advanced = ['Parameter draws: ' + (value('reference_draws') || 'not specified')];
            if (task.value === 'calibration') advanced.push('IVIVC uncertainty draws: ' +
                (value('cal_interval_draws') || 'not specified'));
            advanced.push('Simulated studies: ' + (value('n_simulations') || 'not specified'),
                'Seed: ' + (value('random_seed') || 'not specified'));
            const summaries = {endpoints: endpoints.join(' · ') || 'None selected',
                advanced: advanced.join(' · ')};
            root.querySelectorAll('[data-calculation-summary]').forEach(summary => {
                summary.textContent = summaries[summary.dataset.calculationSummary] || '';
            });
        }

        function clearErrors() {
            summary.replaceChildren();
            summary.hidden = true;
            root.querySelectorAll('[data-error-for]').forEach(container => {
                container.replaceChildren();
                container.hidden = true;
                const field = document.getElementById(container.dataset.errorFor);
                if (!field) return;
                field.setAttribute('aria-invalid', 'false');
                const describedBy = (field.getAttribute('aria-describedby') || '').split(/\s+/)
                    .filter(id => id && id !== container.id).join(' ');
                field.setAttribute('aria-describedby', describedBy);
            });
        }

        function showErrors(errors, context = {}) {
            clearErrors();
            const heading = document.createElement('h2');
            heading.textContent = context.heading || 'The request could not be completed';
            const list = document.createElement('ul');
            const fieldMessages = new Map();
            errors.forEach(error => {
                const item = document.createElement('li');
                const field = error.field_id && document.getElementById(error.field_id);
                if (field && !field.disabled) {
                    const link = document.createElement('a');
                    link.href = '#' + field.id;
                    link.textContent = error.message;
                    link.addEventListener('click', event => {
                        event.preventDefault();
                        showView('inputs');
                        let inactive = field.disabled || field.hidden || document.getElementById(field.id) !== field;
                        for (let ancestor = field.parentElement; ancestor && ancestor !== root; ancestor = ancestor.parentElement) {
                            if (ancestor.hidden) inactive = true;
                        }
                        if (inactive) {
                            requestStatus.textContent = 'That setting is no longer active. Check the current inputs again.';
                            requestStatus.dataset.state = 'notice';
                            focusTarget(summary);
                            return;
                        }
                        focusTarget(field);
                    });
                    item.append(link);
                    field.setAttribute('aria-invalid', 'true');
                    if (!fieldMessages.has(error.field_id)) fieldMessages.set(error.field_id, []);
                    fieldMessages.get(error.field_id).push(error.message);
                } else {
                    item.textContent = error.message;
                }
                list.append(item);
            });
            root.querySelectorAll('[data-error-for]').forEach(container => {
                const messages = fieldMessages.get(container.dataset.errorFor);
                if (!messages) return;
                container.textContent = messages.join(' ');
                container.hidden = false;
                const field = document.getElementById(container.dataset.errorFor);
                const ids = new Set((field.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
                ids.add(container.id);
                field.setAttribute('aria-describedby', Array.from(ids).join(' '));
            });
            summary.append(heading, list);
            summary.hidden = false;
            showView('inputs');
            focusTarget(summary);
            requestStatus.textContent = context.status || 'Request failed. Earlier displayed results remain available.';
            requestStatus.dataset.state = 'error';
        }

        root.addEventListener('design-input-errors', event => {
            const context = event.detail || {};
            if (!Array.isArray(context.errors)) return;
            if (context.errors.length) showErrors(context.errors, context);
            else clearErrors();
        });
        root.addEventListener('design-inputs-updated', noteInputChange);

        root.querySelectorAll('[data-design-view]').forEach(button => {
            button.addEventListener('click', () => {
                navigationRevision++;
                activeSection = null;
                showView(button.dataset.designView);
                focusTarget(document.getElementById(button.dataset.designView === 'inputs' ?
                    'design-inputs-heading' : 'design-results-heading') ||
                    (button.dataset.designView === 'inputs' ? inputs : results));
            });
        });
        sectionButtons.forEach(button => button.addEventListener('click', () => {
            if (button.hidden || button.disabled) return;
            const target = document.getElementById(button.dataset.designSection);
            if (!target) return;
            navigationRevision++;
            showView('inputs');
            activeSection = button.dataset.designSection;
            syncNavigation();
            focusTarget(target);
        }));
        root.addEventListener('click', event => {
            const add = event.target.closest('[data-add-timing]');
            const remove = event.target.closest('[data-remove-timing]');
            if ((!add && !remove) || !planningAvailable()) return;
            event.preventDefault();
            const row = add ? timingRows.find(item => item.dataset.timingActive !== 'true') :
                timingRows.find(item => item.dataset.timingRow === remove.dataset.removeTiming);
            if (!row) return;
            row.dataset.timingActive = add ? 'true' : 'false';
            syncInputs();
            noteInputChange();
            if (add) focusTarget(document.getElementById('timing_' + row.dataset.timingRow + '_kind'));
        });
        root.addEventListener('click', event => {
            const add = event.target.closest('[data-add-calibration-candidate]');
            const remove = event.target.closest('[data-remove-calibration-candidate]');
            if ((!add && !remove) || !calibrationAvailable() || (remove && remove.disabled)) return;
            event.preventDefault();
            const row = add ? calibrationCandidates.find(item => item.dataset.calibrationActive !== 'true') :
                calibrationCandidates.find(item => item.dataset.calibrationCandidate === remove.dataset.removeCalibrationCandidate);
            if (!row) return;
            row.dataset.calibrationActive = add ? 'true' : 'false';
            syncInputs();
            noteInputChange();
            if (add) focusTarget(document.getElementById('cal_candidate_' + row.dataset.calibrationCandidate + '_name'));
        });
        root.addEventListener('click', event => {
            const button = event.target.closest('[data-copy-generated-candidate]');
            if (!button || button.disabled || pending || generating || !calibrationAvailable() ||
                    generatedSignature === null || signature() !== generatedSignature) return;
            event.preventDefault();
            const index = button.dataset.copyGeneratedCandidate;
            const schedules = generatedSchedules.filter(row => String(row.candidate_index) === index);
            if (!schedules.length) return;
            for (const schedule of schedules) {
                const prefix = 'cal_candidate_' + index + '_' + schedule.side + '_';
                document.getElementById(prefix + 'times').value = schedule.times.join(', ');
                document.getElementById(prefix + 'kind').value = 'manual';
            }
            syncInputs();
            generationNotice = 'Copied times are now explicitly specified and can be edited. Calculate to assess them.';
            noteInputChange();
            focusTarget(document.getElementById('cal_candidate_' + index + '_' + schedules[0].side + '_times'));
        });
        if (generateCandidates) generateCandidates.addEventListener('click', async event => {
            event.preventDefault();
            if (generateCandidates.disabled || pending || generating || !calibrationAvailable()) return;
            const data = new FormData(form);
            const submittedSignature = signature(data);
            const submittedRevision = draftRevision;
            const submittedNavigation = navigationRevision;
            generating = true;
            generationNotice = '';
            clearErrors();
            form.setAttribute('aria-busy', 'true');
            requestStatus.textContent = '';
            requestStatus.dataset.state = 'busy';
            setOperation('generation', true, 'Generating calibration candidate schedules.');
            updateGenerationStatus();
            try {
                const response = await fetch(generateCandidates.dataset.generationUrl, {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                });
                const payload = await response.json();
                if (signature() !== submittedSignature) {
                    generationNotice = 'Inputs changed during generation; the submitted preview was not applied. Generate again for the current choices.';
                    requestStatus.textContent = generationNotice;
                    requestStatus.dataset.state = 'notice';
                    return;
                }
                if (!response.ok || payload.ok !== true) {
                    generationNotice = 'Generation failed. Earlier generated schedules and comparison results remain available.';
                    showErrors(Array.isArray(payload.errors) && payload.errors.length ? payload.errors :
                        [{message: 'Schedule generation could not be completed. Check the inputs and try again.'}], {
                            heading: 'Candidate schedules could not be generated', status: generationNotice,
                        });
                    return;
                }
                if (typeof payload.html !== 'string' || !Array.isArray(payload.schedules) ||
                        payload.schedules.some(row => !['in_vitro', 'in_vivo'].includes(row.side) ||
                            !Array.isArray(row.times) || !row.times.length ||
                            row.times.some(time => typeof time !== 'number' || !Number.isFinite(time)) ||
                            !calibrationCandidates.some(candidate => candidate.dataset.calibrationCandidate === String(row.candidate_index) &&
                                candidate.dataset.calibrationActive === 'true'))) throw new Error('Invalid generation response.');
                // Server-rendered escaped markup stays separate from saved comparison results.
                generationContent.innerHTML = payload.html;
                generatedSchedules = payload.schedules;
                generatedSignature = submittedSignature;
                requestStatus.textContent = 'Candidate schedules generated. Review the times before calculating a comparison.';
                requestStatus.dataset.state = 'success';
                if (draftRevision === submittedRevision && navigationRevision === submittedNavigation) {
                    showView('inputs');
                    focusTarget(generationContent);
                }
            } catch (error) {
                generationNotice = signature() !== submittedSignature ?
                    'Inputs changed during generation; generate again for the current choices.' :
                    'Generation could not be completed. Check the connection and try again. Earlier previews and comparisons remain available.';
                if (signature() === submittedSignature) showErrors([{message: generationNotice}], {
                    heading: 'Candidate schedules could not be generated', status: generationNotice,
                });
                else {
                    requestStatus.textContent = generationNotice;
                    requestStatus.dataset.state = 'notice';
                }
            } finally {
                generating = false;
                setOperation('generation', false);
                form.setAttribute('aria-busy', 'false');
                updateGenerationStatus();
            }
        });
        task.addEventListener('change', () => {
            syncInputs();
            clearErrors();
            showView('inputs');
            // The task selector is associated with the form from above the
            // navigation bar, so its edits do not bubble through the form.
            noteInputChange();
        });
        source.addEventListener('change', syncInputs);
        form.addEventListener('input', noteInputChange);
        form.addEventListener('change', () => { syncInputs(); noteInputChange(); });

        form.addEventListener('submit', async event => {
            event.preventDefault();
            if (pending || generating) return;
            // Native Enter submission may omit a submitter. The form has one
            // calculation action, shared by keyboard and button submission.
            if (event.submitter && event.submitter !== calculate) return;
            const comparisonName = task.value === 'calibration' ? 'calibration' : 'study';
            const context = {
                heading: 'The ' + comparisonName + ' comparison could not be completed',
                status: 'Calculation failed. Earlier displayed results remain available.',
            };
            if (!calculationAvailable()) {
                showErrors([{message: 'Choose a supported study-planning or calibration-comparison task.'}], context);
                return;
            }
            const data = new FormData(form);
            const submittedSignature = signature(data);
            const submittedRevision = draftRevision;
            const submittedNavigation = navigationRevision;
            pending = true;
            syncButtons();
            form.setAttribute('aria-busy', 'true');
            clearErrors();
            requestStatus.textContent = '';
            requestStatus.dataset.state = 'busy';
            setOperation('calculation', true, 'Calculating the ' + comparisonName + ' comparison. Keep this page open.');
            try {
                // Read the explicit attribute so named controls cannot shadow
                // the URL through HTMLFormElement's named-property lookup.
                const response = await fetch(form.getAttribute('action'), {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                });
                const payload = await response.json();
                if (!response.ok || payload.ok !== true) {
                    showErrors(signature() !== submittedSignature ? [{
                        message: 'Submitted settings could not be calculated. Inputs changed during the request; check the current choices again.',
                    }] : Array.isArray(payload.errors) && payload.errors.length ? payload.errors :
                        [{message: 'The request could not be completed. Try again.'}], context);
                    return;
                }
                if (typeof payload.html !== 'string') throw new Error('Missing response HTML.');
                // Only this app's server-rendered, escaped template is inserted here.
                inspectionAction('dispose', content);
                plotAction('purge', content);
                content.innerHTML = payload.html;
                inspectionAction('initialize', content);
                plotAction('initialize', content);
                calculatedSignature = submittedSignature;
                updateDraftStatus();
                const unchanged = signature() === submittedSignature;
                requestStatus.textContent = unchanged ?
                    comparisonName[0].toUpperCase() + comparisonName.slice(1) + ' comparison calculated.' :
                    'Submitted ' + comparisonName + ' comparison calculated. Inputs changed during the request; calculate again to assess the current choices.';
                requestStatus.dataset.state = 'success';
                if (unchanged && draftRevision === submittedRevision && navigationRevision === submittedNavigation) {
                    showView('results');
                    focusTarget(document.getElementById('design-results-heading') || content);
                }
            } catch (error) {
                showErrors([{message: 'The request could not be completed. Check the connection and try again.'}], context);
            } finally {
                pending = false;
                setOperation('calculation', false);
                syncButtons();
                form.setAttribute('aria-busy', 'false');
            }
        });

        syncInputs();
        showView('inputs');
        updateDraftStatus();
        syncButtons();
        measureChrome();
        if (typeof window !== 'undefined') {
            if (typeof window.ResizeObserver === 'function') {
                const observer = new window.ResizeObserver(measureChrome);
                if (navigation) observer.observe(navigation);
                if (actionBar) observer.observe(actionBar);
            }
            if (typeof window.addEventListener === 'function') window.addEventListener('resize', measureChrome);
        }
        root.dataset.ready = 'true';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
})();
