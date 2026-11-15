/* Study-design setup. Each POST completes in its request; no saved application state. */
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
        const summary = document.getElementById('design-errors');
        const draftStatus = document.getElementById('design-draft-status');
        const requestStatus = document.getElementById('design-request-status');
        const submit = document.getElementById('review-setup');
        let pending = false;
        let reviewedSignature = null;

        function setActive(container, active) {
            container.hidden = !active;
            container.querySelectorAll('input, select, textarea').forEach(field => {
                field.disabled = !active;
            });
        }

        function syncInputs() {
            root.querySelectorAll('[data-design-task]').forEach(group => {
                setActive(group, group.dataset.designTask === task.value);
            });
            root.querySelectorAll('[data-reference-source]').forEach(group => {
                setActive(group, task.value === 'study' && group.dataset.referenceSource === source.value);
            });
            root.dispatchEvent(new CustomEvent('design-visibility-updated'));
        }

        function signature(data = new FormData(form)) {
            return JSON.stringify(Array.from(data.entries(), ([key, value]) => [key,
                typeof value === 'string' ? value :
                    (value.name ? [value.name, value.size, value.lastModified] : null),
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
            const data = new FormData(form);
            const submittedSignature = signature(data);
            pending = true;
            submit.disabled = true;
            form.setAttribute('aria-busy', 'true');
            clearErrors();
            requestStatus.textContent = 'Reviewing setup.';
            try {
                const response = await fetch(form.action, {
                    method: 'POST', body: data, headers: {'Accept': 'application/json'},
                });
                const payload = await response.json();
                if (!response.ok || payload.ok !== true) {
                    showErrors(signature() !== submittedSignature ? [{
                        message: 'Submitted settings could not be reviewed. Inputs changed during the request; review the current choices again.',
                    }] : Array.isArray(payload.errors) && payload.errors.length ? payload.errors :
                        [{message: 'The request could not be completed. Try again.'}]);
                    return;
                }
                if (typeof payload.html !== 'string') throw new Error('Missing setup review.');
                // Only this app's server-rendered, escaped template is inserted here.
                content.innerHTML = payload.html;
                reviewedSignature = submittedSignature;
                updateDraftStatus();
                const unchanged = signature() === submittedSignature;
                requestStatus.textContent = unchanged ?
                    'Setup reviewed. No numerical calculation has run.' :
                    'Submitted settings reviewed. Inputs changed during the request; review again to check the current choices.';
                if (unchanged) {
                    showView('results');
                    content.focus();
                }
            } catch (error) {
                showErrors([{message: 'The request could not be completed. Check the connection and try again.'}]);
            } finally {
                pending = false;
                submit.disabled = false;
                form.setAttribute('aria-busy', 'false');
            }
        });

        syncInputs();
        showView('inputs');
        updateDraftStatus();
        submit.disabled = false;
        root.dataset.ready = 'true';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
})();
