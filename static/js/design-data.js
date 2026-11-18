/* Stateless upload inspection and interpreted data previews. */
(() => {
    'use strict';

    function initialize() {
        const root = document.getElementById('design-workspace');
        if (!root || root.dataset.dataReady === 'true') return;
        const states = new Map();
        const completed = document.getElementById('has_completed');
        const completedGroup = root.querySelector('[data-completed-dataset]');

        function emit(name, detail = {}) {
            root.dispatchEvent(new CustomEvent(name, {bubbles: true, detail}));
        }

        function fields(container) {
            return Array.from(container.querySelectorAll('input, select, textarea'));
        }

        function active(container) {
            for (let node = container; node && node !== root; node = node.parentElement) {
                // Switching workspace views does not invalidate imported data.
                if (node.hidden && node.id !== 'design-inputs') return false;
            }
            return true;
        }

        function showGroup(container, shown, enabled) {
            if (!container) return;
            container.hidden = !shown;
            fields(container).forEach(field => { field.disabled = !shown || !enabled; });
        }

        function syncCourse(course, enabled) {
            const prefix = course.dataset.courseOptions;
            const field = key => document.getElementById(prefix + key);
            const mode = field('uncertainty').value;
            const supplied = mode !== 'none';
            const mean = mode === 'mean_sd' || mode === 'mean_sem';
            showGroup(course.querySelector('[data-error-fields]'), supplied, enabled);
            course.querySelectorAll('[data-error-source]').forEach(group => {
                showGroup(group, supplied && group.dataset.errorSource === field('error_source').value, enabled);
            });
            if (mode === 'mean_sd' && field('count_source').value === 'none') {
                field('count_source').value = 'constant';
            }
            showGroup(course.querySelector('[data-count-fields]'), mean, enabled);
            course.querySelectorAll('[data-count-source]').forEach(group => {
                showGroup(group, mean && group.dataset.countSource === field('count_source').value, enabled);
            });
        }

        function openOptions(state) {
            if (state.panel) state.panel.open = true;
            const mapping = state.config.querySelector('[data-import-mapping]');
            if (mapping) mapping.open = true;
        }

        function updateSummaries(state) {
            const file = state.file.files && state.file.files[0];
            const sheet = document.getElementById(state.id + '__sheet');
            if (state.datasetSummary) {
                const status = state.pending ? (state.pendingAction === 'inspect' ? 'Reading columns…' : 'Preparing preview…') :
                    state.hasPreview && state.previewCurrent ? 'Preview ready' :
                    state.hasPreview ? 'Preview out of date' :
                    state.columnsCurrent ? 'Columns ready; preview not run' :
                    sheet && !sheet.value ? 'Choose a worksheet' : 'Columns not loaded';
                state.datasetSummary.textContent = file ? file.name + (sheet && sheet.value ? ' · ' + sheet.value : '') + ' · ' + status : 'Select a file';
            }
            const summary = state.config.querySelector('[data-import-mapping-summary]');
            if (!summary) return;
            const modes = {none: 'uncertainty not supplied', individual_sd: 'individual observations with SD',
                mean_sd: 'means with SD and independent count', mean_sem: 'means with SEM'};
            const descriptions = Array.from(state.config.querySelectorAll('[data-course-options]'), course => {
                const prefix = course.dataset.courseOptions;
                const value = key => document.getElementById(prefix + key)?.value || 'Not selected';
                const label = course.querySelector('legend')?.textContent;
                return (label ? label + ': ' : '') + value('time_column') + ' → ' + value('response_column') +
                    '; ' + (modes[value('uncertainty')] || value('uncertainty'));
            });
            if (!descriptions.length) {
                for (const [key, label] of [['ratio_column', 'Ratio'], ['weight_column', 'Weights'],
                    ['source_column', 'Source'], ['time_column', 'Time'], ['sd_column', 'SD']]) {
                    const field = document.getElementById(state.id + '__' + key);
                    if (field) descriptions.push(label + ': ' + (field.value || 'Not supplied'));
                }
            }
            summary.textContent = descriptions.join(' · ');
        }

        function cancelRequest(state) {
            if (!state.pending) return;
            state.requestToken++;
            if (state.controller) state.controller.abort();
            state.controller = null;
            state.pending = false;
            emit('design-request-activity', {key: 'import:' + state.id, active: false});
        }

        function sync() {
            if (completed && completedGroup) {
                showGroup(completedGroup, completed.checked && active(completedGroup.parentElement), true);
            }
            states.forEach(state => {
                const enabled = active(state.container);
                if (state.lastActive !== undefined && state.lastActive !== enabled) {
                    state.revision++;
                    if (!enabled && state.pending) {
                        cancelRequest(state);
                        state.status.textContent = 'Import interrupted. Reload columns or preview again when this data role is active.';
                    }
                }
                state.lastActive = enabled;
                state.file.disabled = !enabled;
                fields(state.config).forEach(field => { field.disabled = !enabled; });
                const columnFields = state.config.querySelector('[data-column-fields]');
                showGroup(columnFields, state.columnsCurrent, enabled);
                state.config.querySelectorAll('[data-course-options]').forEach(course => {
                    syncCourse(course, enabled && state.columnsCurrent);
                });
                state.container.querySelectorAll('[data-import-action]').forEach(button => {
                    button.disabled = !enabled || state.pending || !state.file.files?.length ||
                        (button.dataset.importAction === 'preview' && !state.columnsCurrent);
                });
                state.preview.hidden = !state.hasPreview || !state.previewVisible;
                if (state.toggle) {
                    state.toggle.hidden = !state.hasPreview;
                    state.toggle.disabled = !enabled;
                    state.toggle.textContent = state.previewVisible ? 'Hide preview' : 'Show preview';
                    state.toggle.setAttribute('aria-expanded', String(state.previewVisible));
                }
                updateSummaries(state);
            });
        }

        function markStale(state, text) {
            state.previewCurrent = false;
            state.stale.hidden = !state.hasPreview;
            state.stale.textContent = state.hasPreview ? text : '';
        }

        function error(state, errors) {
            openOptions(state);
            emit('design-input-errors', {
                errors,
                heading: 'Check the data import settings',
                status: 'Data import failed. Earlier displayed results remain available.',
            });
            state.status.textContent = 'Data import could not be completed. Earlier previews are unchanged.';
        }

        async function request(state, action, replace = false) {
            if ((state.pending && !replace) || !active(state.container)) return;
            const file = state.file.files && state.file.files[0];
            if (!file) {
                error(state, [{field_id: state.id, message: 'Select a file to inspect.'}]);
                return;
            }
            if (action === 'preview' && !state.columnsCurrent) {
                error(state, [{field_id: state.id, message: 'Wait for the selected file or worksheet columns to load before previewing data.'}]);
                return;
            }
            const data = new FormData();
            data.append('dataset_id', state.id);
            data.append('file', file);
            const sheet = document.getElementById(state.id + '__sheet');
            data.append('sheet', sheet ? sheet.value : '');
            if (action === 'preview') {
                fields(state.config).forEach(field => {
                    if (field.disabled || !field.name) return;
                    if (['checkbox', 'radio'].includes(field.getAttribute('type')) && !field.checked) return;
                    data.append(field.name, field.value);
                });
            }
            const revision = state.revision;
            const token = ++state.requestToken;
            if (state.controller) state.controller.abort();
            const controller = typeof AbortController === 'function' ? new AbortController() : null;
            state.controller = controller;
            const current = () => token === state.requestToken && state.revision === revision &&
                state.file.files && state.file.files[0] === file && active(state.container);
            state.pending = true;
            state.pendingAction = action;
            state.status.textContent = action === 'inspect' ? 'Reading file columns.' : 'Preparing data preview.';
            emit('design-request-activity', {
                key: 'import:' + state.id, active: true, message: state.status.textContent,
            });
            emit('design-input-errors', {errors: []});
            sync();
            try {
                const response = await fetch(action === 'inspect' ? state.container.dataset.inspectUrl :
                    state.container.dataset.previewUrl, {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                    ...(controller ? {signal: controller.signal} : {}),
                });
                const payload = await response.json();
                if (!current()) {
                    if (token === state.requestToken) state.status.textContent = 'Inputs changed during the request. Reload columns or preview the current choices again.';
                    return;
                }
                if (!response.ok || !payload || payload.ok !== true) {
                    error(state, payload && Array.isArray(payload.errors) && payload.errors.length ? payload.errors :
                        [{field_id: state.id, message: 'The file request could not be completed. Try again.'}]);
                    return;
                }
                if (typeof payload.html !== 'string') throw new Error('Missing data response.');
                // Both fragments are rendered and escaped by this app's server.
                if (action === 'inspect') {
                    const sheetFocused = document.activeElement?.id === state.id + '__sheet';
                    state.config.innerHTML = payload.html;
                    state.columnsCurrent = !payload.requires_sheet &&
                        Boolean(state.config.querySelector('[data-column-fields]'));
                    state.status.textContent = payload.requires_sheet ?
                        'Select a worksheet to load its columns.' : 'Check the mapped columns and uncertainty interpretation, then preview data.';
                    if (sheetFocused) document.getElementById(state.id + '__sheet')?.focus({preventScroll: true});
                    markStale(state, 'Column choices were reloaded. This earlier preview retains its original file and settings. Preview again to update it.');
                } else {
                    state.preview.innerHTML = payload.html;
                    state.hasPreview = true;
                    state.previewVisible = true;
                    state.previewCurrent = true;
                    state.stale.hidden = true;
                    state.stale.textContent = '';
                    state.status.textContent = 'Data preview ready. No fitting or simulation has run.';
                    const mapping = state.config.querySelector('[data-import-mapping]');
                    if (mapping) mapping.open = false;
                }
                sync();
                emit('design-inputs-updated');
            } catch (exception) {
                if (current()) {
                    error(state, [{field_id: state.id, message: 'The file request could not be completed. Check the connection and try again.'}]);
                } else if (token === state.requestToken) {
                    state.status.textContent = 'Inputs changed during the request. Reload columns or preview the current choices again.';
                }
            } finally {
                if (token === state.requestToken) {
                    state.pending = false;
                    state.controller = null;
                    sync();
                    emit('design-request-activity', {key: 'import:' + state.id, active: false});
                }
            }
        }

        root.querySelectorAll('[data-design-dataset]').forEach(container => {
            const id = container.dataset.designDataset;
            const state = {
                id, container, file: document.getElementById(id),
                config: container.querySelector('[data-import-config]'),
                preview: container.querySelector('[data-import-preview]'),
                toggle: container.querySelector('[data-import-preview-toggle]'),
                status: container.querySelector('[data-import-status]'),
                stale: container.querySelector('[data-import-stale]'),
                panel: container.querySelector('[data-import-panel]'),
                datasetSummary: container.querySelector('[data-import-dataset-summary]'),
                revision: 0, pending: false, columnsCurrent: false, hasPreview: false,
                previewVisible: false, previewCurrent: false, requestToken: 0, controller: null,
            };
            states.set(container, state);
            function edited(event) {
                const target = event.target;
                let inspect = false;
                if (target === state.file) {
                    state.revision++;
                    state.config.replaceChildren();
                    state.columnsCurrent = false;
                    markStale(state, 'The file selection changed. This earlier preview still describes its original file. Preview again after the new columns load.');
                    state.status.textContent = state.file.files?.length ? 'Loading columns for the selected file.' : 'Select a file to load its columns.';
                    inspect = event.type === 'change' && Boolean(state.file.files?.length);
                    openOptions(state);
                    if (!inspect) cancelRequest(state);
                } else if (state.config.contains(target)) {
                    state.revision++;
                    if (target.id === id + '__sheet') {
                        state.columnsCurrent = false;
                        state.status.textContent = target.value ? 'Loading columns for the selected worksheet.' : 'Select a worksheet to load its columns.';
                        inspect = event.type === 'change' && Boolean(target.value);
                        openOptions(state);
                        if (!target.value) cancelRequest(state);
                    }
                    markStale(state, 'File or interpretation settings changed. This earlier preview retains its original settings. Preview again to update it.');
                } else return;
                sync();
                emit('design-inputs-updated');
                if (inspect) void request(state, 'inspect', true);
            }
            container.addEventListener('change', edited);
            container.addEventListener('input', event => { if (event.target !== state.file) edited(event); });
        });
        root.addEventListener('click', event => {
            const toggle = event.target.closest('[data-import-preview-toggle]');
            if (toggle) {
                event.preventDefault();
                const state = states.get(toggle.closest('[data-design-dataset]'));
                if (!state || !state.hasPreview || toggle.disabled || !active(state.container)) return;
                state.previewVisible = !state.previewVisible;
                sync();
                return;
            }
            const button = event.target.closest('[data-import-action]');
            if (!button) return;
            const state = states.get(button.closest('[data-design-dataset]'));
            if (!state) return;
            event.preventDefault();
            return request(state, button.dataset.importAction);
        });
        root.addEventListener('design-visibility-updated', sync);
        if (completed) completed.addEventListener('change', () => { sync(); emit('design-inputs-updated'); });
        sync();
        root.dataset.dataReady = 'true';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
})();
