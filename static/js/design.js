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
        const reviewContent = document.getElementById('design-review-content') || content;
        const resultStatus = document.getElementById('design-results-status');
        const summary = document.getElementById('design-errors');
        const draftStatus = document.getElementById('design-draft-status');
        const requestStatus = document.getElementById('design-request-status');
        const submit = document.getElementById('review-setup');
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
        const calibrationChoices = new Map();
        let previousCalibrationApproach = null;
        const files = new WeakMap();
        let nextFile = 0;
        let pending = false;
        let reviewedSignature = null;
        let calculatedSignature = null;

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
            submit.disabled = pending;
            if (calculate) calculate.disabled = pending || !calculationAvailable();
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
            inputs.hidden = view !== 'inputs';
            results.hidden = view !== 'results';
            root.querySelectorAll('[data-design-view]').forEach(button => {
                button.setAttribute('aria-pressed', String(button.dataset.designView === view));
            });
        }

        function updateDraftStatus() {
            if (calculatedSignature !== null) {
                const current = signature() === calculatedSignature;
                draftStatus.textContent = current ?
                    'The displayed comparison matches these settings.' :
                    'Inputs changed. The displayed comparison retains its original settings; calculate again to assess these choices.';
                if (resultStatus) resultStatus.textContent = current ?
                    'Comparison for the current inputs.' :
                    'Inputs changed. This comparison still uses its original inputs and settings.';
                return;
            }
            if (resultStatus) resultStatus.textContent = 'No numerical comparison has been calculated.';
            draftStatus.textContent = reviewedSignature === null ?
                'Draft settings. Review setup to check these choices.' :
                signature() === reviewedSignature ?
                    'The displayed setup review matches these settings.' :
                    'Inputs changed. The displayed setup review retains its original settings; review again to check these choices.';
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
            heading.textContent = context.heading || 'The setup review could not be completed';
            const list = document.createElement('ul');
            const fieldMessages = new Map();
            errors.forEach(error => {
                const item = document.createElement('li');
                const field = error.field_id && document.getElementById(error.field_id);
                if (field) {
                    const link = document.createElement('a');
                    link.href = '#' + field.id;
                    link.textContent = error.message;
                    link.addEventListener('click', event => {
                        event.preventDefault();
                        const details = field.closest('details');
                        if (details) details.open = true;
                        field.focus();
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
            summary.focus();
            requestStatus.textContent = context.status || 'Setup review failed. Earlier displayed results remain available.';
        }

        root.addEventListener('design-input-errors', event => {
            const context = event.detail || {};
            if (!Array.isArray(context.errors)) return;
            if (context.errors.length) showErrors(context.errors, context);
            else clearErrors();
        });
        root.addEventListener('design-inputs-updated', updateDraftStatus);

        root.querySelectorAll('[data-design-view]').forEach(button => {
            button.addEventListener('click', () => showView(button.dataset.designView));
        });
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
            updateDraftStatus();
            if (add) document.getElementById('timing_' + row.dataset.timingRow + '_kind').focus();
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
            updateDraftStatus();
            if (add) document.getElementById('cal_candidate_' + row.dataset.calibrationCandidate + '_name').focus();
        });
        task.addEventListener('change', () => {
            syncInputs();
            clearErrors();
            showView('inputs');
            updateDraftStatus();
        });
        source.addEventListener('change', syncInputs);
        form.addEventListener('input', updateDraftStatus);
        form.addEventListener('change', () => { syncInputs(); updateDraftStatus(); });

        form.addEventListener('submit', async event => {
            event.preventDefault();
            if (pending) return;
            const calculating = Boolean(calculate && event.submitter === calculate);
            const comparisonName = task.value === 'calibration' ? 'calibration' : 'study';
            const context = calculating ? {
                heading: 'The planning comparison could not be completed',
                status: 'Calculation failed. Earlier displayed results remain available.',
            } : {};
            if (calculating && !calculationAvailable()) {
                showErrors([{message: 'Choose a supported study-planning or calibration-comparison task.'}], context);
                return;
            }
            const data = new FormData(form);
            const submittedSignature = signature(data);
            pending = true;
            syncButtons();
            form.setAttribute('aria-busy', 'true');
            clearErrors();
            requestStatus.textContent = calculating ? 'Calculating the ' + comparisonName + ' comparison. This request runs to completion.' : 'Reviewing setup.';
            try {
                const response = await fetch(calculating ? calculate.getAttribute('formaction') : form.action, {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                });
                const payload = await response.json();
                if (!response.ok || payload.ok !== true) {
                    showErrors(signature() !== submittedSignature ? [{
                        message: 'Submitted settings could not be ' + (calculating ? 'calculated' : 'reviewed') + '. Inputs changed during the request; check the current choices again.',
                    }] : Array.isArray(payload.errors) && payload.errors.length ? payload.errors :
                        [{message: 'The request could not be completed. Try again.'}], context);
                    return;
                }
                if (typeof payload.html !== 'string') throw new Error('Missing response HTML.');
                // Only this app's server-rendered, escaped template is inserted here.
                const destination = calculating ? content : reviewContent;
                destination.innerHTML = payload.html;
                if (calculating) calculatedSignature = submittedSignature;
                else reviewedSignature = submittedSignature;
                updateDraftStatus();
                const unchanged = signature() === submittedSignature;
                requestStatus.textContent = calculating ? (unchanged ?
                    comparisonName[0].toUpperCase() + comparisonName.slice(1) + ' comparison calculated.' :
                    'Submitted ' + comparisonName + ' comparison calculated. Inputs changed during the request; calculate again to assess the current choices.') : unchanged ?
                    'Setup reviewed. This request did not run a numerical calculation.' :
                    'Submitted settings reviewed. Inputs changed during the request; review again to check the current choices.';
                if (unchanged) {
                    showView('results');
                    destination.focus();
                }
            } catch (error) {
                showErrors([{message: 'The request could not be completed. Check the connection and try again.'}], context);
            } finally {
                pending = false;
                syncButtons();
                form.setAttribute('aria-busy', 'false');
            }
        });

        syncInputs();
        showView('inputs');
        updateDraftStatus();
        syncButtons();
        root.dataset.ready = 'true';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
})();
