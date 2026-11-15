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

        function sync() {
            if (completed && completedGroup) {
                showGroup(completedGroup, completed.checked && active(completedGroup.parentElement), true);
            }
            states.forEach(state => {
                const enabled = active(state.container);
                if (state.lastActive !== undefined && state.lastActive !== enabled) state.revision++;
                state.lastActive = enabled;
                state.file.disabled = !enabled;
                fields(state.config).forEach(field => { field.disabled = !enabled; });
                const columnFields = state.config.querySelector('[data-column-fields]');
                showGroup(columnFields, state.columnsCurrent, enabled);
                state.config.querySelectorAll('[data-course-options]').forEach(course => {
                    syncCourse(course, enabled && state.columnsCurrent);
                });
                state.container.querySelectorAll('[data-import-action]').forEach(button => {
                    button.disabled = !enabled || state.pending ||
                        (button.dataset.importAction === 'preview' && !state.columnsCurrent);
                });
            });
        }

        function markStale(state, text) {
            state.stale.hidden = !state.hasPreview;
            state.stale.textContent = state.hasPreview ? text : '';
        }

        function error(state, errors) {
            emit('design-input-errors', {
                errors,
                heading: 'Check the data import settings',
                status: 'Data import failed. Earlier displayed results remain available.',
            });
            state.status.textContent = 'Data import could not be completed. Earlier previews are unchanged.';
        }

        async function request(state, action) {
            if (state.pending || !active(state.container)) return;
            const file = state.file.files && state.file.files[0];
            if (!file) {
                error(state, [{field_id: state.id, message: 'Select a file to inspect.'}]);
                return;
            }
            if (action === 'preview' && !state.columnsCurrent) {
                error(state, [{field_id: state.id, message: 'Read the selected worksheet columns before previewing data.'}]);
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
            const current = () => state.revision === revision &&
                state.file.files && state.file.files[0] === file && active(state.container);
            state.pending = true;
            state.status.textContent = action === 'inspect' ? 'Reading file columns.' : 'Preparing data preview.';
            emit('design-input-errors', {errors: []});
            sync();
            try {
                const response = await fetch(action === 'inspect' ? state.container.dataset.inspectUrl :
                    state.container.dataset.previewUrl, {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                });
                const payload = await response.json();
                if (!current()) {
                    state.status.textContent = 'Inputs changed during the request. Inspect or preview the current choices again.';
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
                    state.config.innerHTML = payload.html;
                    state.columnsCurrent = !payload.requires_sheet &&
                        Boolean(state.config.querySelector('[data-column-fields]'));
                    state.status.textContent = payload.requires_sheet ?
                        'Select a worksheet, then read its columns.' : 'Check the mapped columns and uncertainty interpretation, then preview data.';
                    markStale(state, 'Column choices were reloaded. This earlier preview retains its original file and settings. Preview again to update it.');
                } else {
                    state.preview.innerHTML = payload.html;
                    state.hasPreview = true;
                    state.stale.hidden = true;
                    state.stale.textContent = '';
                    state.status.textContent = 'Data preview ready. No fitting or simulation has run.';
                }
                sync();
                emit('design-inputs-updated');
            } catch (exception) {
                if (current()) {
                    error(state, [{field_id: state.id, message: 'The file request could not be completed. Check the connection and try again.'}]);
                } else {
                    state.status.textContent = 'Inputs changed during the request. Inspect or preview the current choices again.';
                }
            } finally {
                state.pending = false;
                sync();
            }
        }

        root.querySelectorAll('[data-design-dataset]').forEach(container => {
            const id = container.dataset.designDataset;
            const state = {
                id, container, file: document.getElementById(id),
                config: container.querySelector('[data-import-config]'),
                preview: container.querySelector('[data-import-preview]'),
                status: container.querySelector('[data-import-status]'),
                stale: container.querySelector('[data-import-stale]'),
                revision: 0, pending: false, columnsCurrent: false, hasPreview: false,
            };
            states.set(container, state);
            function edited(event) {
                const target = event.target;
                if (target === state.file) {
                    state.revision++;
                    state.config.replaceChildren();
                    state.columnsCurrent = false;
                    markStale(state, 'The file selection changed. This earlier preview still describes its original file. Read columns and preview again.');
                    state.status.textContent = 'Read columns for the selected file.';
                } else if (state.config.contains(target)) {
                    state.revision++;
                    if (target.id === id + '__sheet') {
                        state.columnsCurrent = false;
                        state.status.textContent = 'Worksheet changed. Read its columns before previewing data.';
                    }
                    markStale(state, 'File or interpretation settings changed. This earlier preview retains its original settings. Preview again to update it.');
                } else return;
                sync();
                emit('design-inputs-updated');
            }
            container.addEventListener('change', edited);
            container.addEventListener('input', event => { if (event.target !== state.file) edited(event); });
        });
        root.addEventListener('click', event => {
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
