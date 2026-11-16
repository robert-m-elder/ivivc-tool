/* Recovery displays use retained results and V3's shared schedule selection. */
(() => {
    'use strict';
    if (window.DesignRecoveryInspection) return;

    const states = new WeakMap();
    const palette = ['#24535c', '#b34f00', '#7550a2'];
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const array = value => Array.isArray(value) ? value : [];
    const text = value => finite(value) ? String(value) : '—';

    function roots(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-planning-result]'));
        if (container.matches && container.matches('[data-planning-result]')) found.unshift(container);
        return found;
    }

    function purge(wrapper) {
        try {
            if (window.DesignPlots && typeof window.DesignPlots.purge === 'function') window.DesignPlots.purge(wrapper);
        } catch (error) { /* Numerical results survive unavailable plotting. */ }
    }

    function chosen(state) {
        return state.schedules.filter(row => state.selection.selectedIds.includes(row.id));
    }

    function summary(state, schedule) {
        return schedule[state.view.value === 'error' ? 'recovery_error' : 'recovered_response'] || {};
    }

    function usable(schedule, values, index) {
        return schedule.n_successful > 0 && finite(values.available_counts?.[index]) ?
            values.available_counts[index] : 0;
    }

    function bandAvailable(schedule, values) {
        return array(schedule.evaluation_times).some((time, index) => finite(time) &&
            usable(schedule, values, index) >= 2 && finite(values.lower?.[index]) &&
            finite(values.upper?.[index]) && values.lower[index] <= values.upper[index]);
    }

    function row(body, values) {
        const document = body.ownerDocument || globalThis.document;
        const tr = document.createElement('tr');
        values.forEach(value => {
            const td = document.createElement('td');
            td.textContent = String(value);
            tr.append(td);
        });
        body.append(tr);
    }

    function summaryValues(state) {
        if (!state.values.open || !state.selection.selectedIds.length) return;
        state.valuesBody.replaceChildren();
        if (state.caption) state.caption.textContent = state.view.value === 'error' ?
            'Recovery error: recovered minus each simulated study’s own true response' :
            'Recovered response across successful simulated studies';
        chosen(state).forEach(schedule => {
            const values = summary(state, schedule);
            array(schedule.evaluation_times).forEach((time, index) => {
                const count = usable(schedule, values, index);
                const validBand = count >= 2 && finite(values.lower?.[index]) &&
                    finite(values.upper?.[index]) && values.lower[index] <= values.upper[index];
                row(state.valuesBody, [schedule.name, text(time),
                    count > 0 ? text(values.median?.[index]) : '—',
                    validBand ? text(values.lower[index]) : '—',
                    validBand ? text(values.upper[index]) : '—', text(count)]);
            });
        });
    }

    function exampleValues(state) {
        if (!state.example.open || !state.exampleValues.open || !state.selection.selectedIds.length ||
            !state.inspection.recovery.shared_example?.available) return;
        state.exampleCurves.replaceChildren();
        state.examplePoints.replaceChildren();
        chosen(state).forEach(schedule => {
            const example = schedule.example;
            if (!example) return;
            array(schedule.evaluation_times).forEach((time, index) => {
                row(state.exampleCurves, [schedule.name, text(time), text(example.true_response?.[index]),
                    example.fit_success ? text(example.recovered_response?.[index]) : '—']);
            });
            array(example.observed_time_means).filter(point => point.role === 'new').forEach(point => {
                row(state.examplePoints, [schedule.name, text(point.time), text(point.response),
                    'New simulated per-time mean', finite(point.planned_replicate_count) ?
                        text(point.planned_replicate_count) + ' planned independent replicates' : 'Count unavailable']);
            });
        });
        // Original completed rows occur once, rather than once per selected schedule.
        array(state.inspection.recovery.completed_observations).forEach(point => {
            const precision = finite(point.observation_sigma) ?
                String(point.uncertainty_kind || 'SD/SEM') + ' ' + text(point.observation_sigma) :
                'Precision unavailable';
            const count = finite(point.reported_replicate_count) ?
                '; reported independent count ' + text(point.reported_replicate_count) : '';
            row(state.examplePoints, ['Shared completed observations', text(point.time), text(point.response),
                point.response_kind === 'mean' ? 'Completed mean (original row)' : 'Completed individual (original row)',
                precision + count]);
        });
    }

    function unavailable(wrapper, message) {
        purge(wrapper);
        const target = wrapper.querySelector('[data-design-plot-target]');
        const fallback = wrapper.querySelector('[data-design-plot-fallback]');
        const status = wrapper.querySelector('[data-design-plot-status]');
        if (target) target.hidden = true;
        if (fallback) fallback.hidden = false;
        if (status) status.textContent = message;
    }

    function draw(state, type, figure, version) {
        const example = type === 'example';
        const wrapper = example ? state.examplePlot : state.plot;
        const current = () => !state.disposed && version === state[example ? 'exampleVersion' : 'version'];
        const showValues = () => {
            if (!current()) return;
            if (example) {
                if (!state.example.open) return;
                state.exampleValues.open = true;
                exampleValues(state);
            } else {
                state.values.open = true;
                summaryValues(state);
            }
        };
        if (!figure) {
            wrapper.hidden = true;
            purge(wrapper);
            showValues();
            return;
        }
        wrapper.hidden = false;
        try {
            if (!window.DesignPlots || typeof window.DesignPlots.update !== 'function') throw new Error('Unavailable renderer.');
            Promise.resolve(window.DesignPlots.update(wrapper, figure)).then(() => {
                if (!current()) return;
                const target = wrapper.querySelector('[data-design-plot-target]');
                if (!target || target.hidden) showValues();
            }).catch(() => {
                if (!current()) return;
                unavailable(wrapper, 'Interactive recovery plot unavailable. Numerical values remain available below.');
                showValues();
            });
        } catch (error) {
            unavailable(wrapper, 'Interactive recovery plot unavailable. Numerical values remain available below.');
            showValues();
        }
    }

    function renderSummary(state) {
        const version = ++state.version;
        const schedules = chosen(state);
        const count = schedules.length;
        state.empty.hidden = count > 0;
        state.view.disabled = !count;
        state.values.hidden = !count;
        if (state.controls) state.controls.hidden = !count;
        state.bandControls.forEach(control => {
            const schedule = state.byId.get(control.dataset.recoveryBand);
            const available = schedule && state.selection.selectedIds.includes(schedule.id) &&
                bandAvailable(schedule, summary(state, schedule));
            control.disabled = !available;
            control.checked = Boolean(available && state.bands.get(schedule.id));
        });
        state.root.querySelectorAll('[data-recovery-schedule]').forEach(element => {
            element.hidden = !state.selection.selectedIds.includes(element.dataset.recoverySchedule);
        });
        state.root.querySelectorAll('[data-recovery-swatch]').forEach(element => {
            const id = element.dataset.recoverySwatch;
            element.hidden = !state.selection.selectedIds.includes(id);
            if (!element.hidden) element.style.setProperty('--schedule-color', state.selection.colors[id]);
        });
        state.root.querySelectorAll('[data-recovery-availability]').forEach(element => {
            const schedule = state.byId.get(element.dataset.recoveryAvailability);
            if (!schedule) return;
            const values = summary(state, schedule);
            const counts = array(schedule.evaluation_times).map((_, index) => usable(schedule, values, index));
            const minimum = counts.length ? Math.min(...counts) : 0;
            const maximum = counts.length ? Math.max(...counts) : 0;
            element.textContent = text(schedule.n_successful) + ' / ' + text(schedule.n_simulations) +
                ' successful fits; ' + text(schedule.n_failed) + ' failed. Pointwise usable studies: ' +
                (minimum === maximum ? text(minimum) : text(minimum) + '–' + text(maximum)) + '.' +
                (!bandAvailable(schedule, values) ? ' No central 95% range is available in this view.' : '');
        });
        if (!count) {
            state.plot.hidden = true;
            state.status.textContent = 'Choose up to three schedules in the performance table to inspect curve recovery.';
            purge(state.plot);
            return;
        }
        const available = schedules.some(schedule => array(summary(state, schedule).median).some((value, index) =>
            usable(schedule, summary(state, schedule), index) > 0 && finite(value)));
        state.status.textContent = available ? (state.view.value === 'error' ?
            'Recovery error uses each simulated study’s own true curve. Fit and usable-study counts are shown for every selected schedule.' :
            'Recovered responses are summarized across successful simulated studies. Fit and usable-study counts are shown for every selected schedule.') :
            'No recovery values are available for the selected schedules in this view. Review their fit and usable-study counts.';
        summaryValues(state);
        let figure = null;
        try {
            figure = window.DesignRecoveryFigure?.summary(state.inspection, state.selection,
                {view: state.view.value, bands: Object.fromEntries(state.bands)});
        } catch (error) { /* The numeric tables remain available if figure construction fails. */ }
        draw(state, 'summary', figure, version);
    }

    function renderExample(state) {
        const version = ++state.exampleVersion;
        const count = state.selection.selectedIds.length;
        const shared = state.inspection.recovery.shared_example;
        state.example.hidden = !count;
        state.root.querySelectorAll('[data-recovery-example-schedule]').forEach(element => {
            element.hidden = !state.selection.selectedIds.includes(element.dataset.recoveryExampleSchedule);
        });
        state.root.querySelectorAll('[data-recovery-example-outcome]').forEach(element => {
            const example = state.byId.get(element.dataset.recoveryExampleOutcome)?.example;
            element.textContent = !shared?.available || !example ? 'No shared simulated example is available.' :
                example.fit_success ? 'The predetermined simulated study fitted successfully.' :
                'Fit failed in the predetermined simulated study: ' + String(example.fit_message || 'Reason unavailable.') +
                ' No successful study has been substituted.';
        });
        state.exampleStatus.textContent = !shared?.available ?
            String(shared?.reason || 'No shared simulated example is available.') :
            'Study ' + text(shared.study_number) + ' is predetermined for every selected schedule, including failed fits. This example always shows response values.';
        state.exampleValues.hidden = !count || !shared?.available;
        if (!count || !state.example.open || !shared?.available) {
            state.examplePlot.hidden = true;
            purge(state.examplePlot);
            return;
        }
        exampleValues(state);
        let figure = null;
        try { figure = window.DesignRecoveryFigure?.example(state.inspection, state.selection); }
        catch (error) { /* Preserve the failed example and its numeric values. */ }
        draw(state, 'example', figure, version);
    }

    function select(state, value) {
        const wanted = new Set(array(value?.selectedIds));
        const selectedIds = state.schedules.filter(schedule => wanted.has(schedule.id)).slice(0, 3).map(schedule => schedule.id);
        const colors = Object.fromEntries(selectedIds.map((id, index) => [id,
            /^#[0-9a-f]{6}$/i.test(value?.colors?.[id] || '') ? value.colors[id] : palette[index]]));
        state.selection = {selectedIds, colors};
        renderSummary(state);
        renderExample(state);
    }

    function initialize(root) {
        if (states.has(root) || !root.querySelector('[data-recovery-panel]')) return;
        const status = root.querySelector('[data-recovery-status]');
        try {
            const encoded = root.querySelector('[data-planning-inspection]');
            const inspection = JSON.parse(encoded ? encoded.textContent : 'null');
            const schedules = inspection?.recovery?.schedules;
            if (inspection?.schema_version !== 1 || inspection.recovery.schema_version !== 1 || !Array.isArray(schedules) ||
                schedules.some(schedule => !schedule || typeof schedule.id !== 'string' || !Array.isArray(schedule.evaluation_times)) ||
                new Set(schedules.map(schedule => schedule.id)).size !== schedules.length) throw new Error('Unavailable recovery data.');
            const state = {
                root, inspection, schedules, byId: new Map(schedules.map(schedule => [schedule.id, schedule])),
                bands: new Map(schedules.map(schedule => [schedule.id, true])),
                bandControls: Array.from(root.querySelectorAll('[data-recovery-band]')),
                selection: {selectedIds: [], colors: {}}, version: 0, exampleVersion: 0, disposed: false,
                status, controls: root.querySelector('[data-recovery-controls]'),
                view: root.querySelector('[data-recovery-view]'), empty: root.querySelector('[data-recovery-empty]'),
                plot: root.querySelector('[data-recovery-plot]'), values: root.querySelector('[data-recovery-values]'),
                valuesBody: root.querySelector('[data-recovery-values-body]'), caption: root.querySelector('[data-recovery-values-caption]'),
                example: root.querySelector('[data-recovery-example]'), examplePlot: root.querySelector('[data-recovery-example-plot]'),
                exampleStatus: root.querySelector('[data-recovery-example-status]'),
                exampleValues: root.querySelector('[data-recovery-example-values]'),
                exampleCurves: root.querySelector('[data-recovery-example-curves-body]'),
                examplePoints: root.querySelector('[data-recovery-example-points-body]'),
            };
            if ([state.status, state.view, state.empty, state.plot, state.values, state.valuesBody,
                state.example, state.examplePlot, state.exampleStatus, state.exampleValues,
                state.exampleCurves, state.examplePoints].some(element => !element)) throw new Error('Incomplete recovery display.');
            state.view.value = 'response';
            state.selected = event => select(state, event.detail);
            state.changed = event => {
                if (event.target === state.view) {
                    if (!['response', 'error'].includes(state.view.value)) state.view.value = 'response';
                    renderSummary(state);
                } else if (state.bandControls.includes(event.target)) {
                    const control = event.target;
                    if (!control.disabled) state.bands.set(control.dataset.recoveryBand, Boolean(control.checked));
                    renderSummary(state);
                }
            };
            state.exampleToggled = () => renderExample(state);
            state.valuesToggled = () => summaryValues(state);
            state.exampleValuesToggled = () => exampleValues(state);
            states.set(root, state);
            root.addEventListener('design-schedule-selection-changed', state.selected);
            root.addEventListener('change', state.changed);
            state.example.addEventListener('toggle', state.exampleToggled);
            state.values.addEventListener('toggle', state.valuesToggled);
            state.exampleValues.addEventListener('toggle', state.exampleValuesToggled);
            select(state, window.DesignScheduleInspection?.getSelection(root));
        } catch (error) {
            if (status) status.textContent = 'Recovery inspection unavailable. The performance table and calculation settings remain available.';
        }
    }

    window.DesignRecoveryInspection = Object.freeze({
        initialize(container) { roots(container).forEach(initialize); },
        dispose(container) {
            roots(container).forEach(root => {
                const state = states.get(root);
                if (!state) return;
                state.disposed = true;
                state.version++;
                state.exampleVersion++;
                root.removeEventListener('design-schedule-selection-changed', state.selected);
                root.removeEventListener('change', state.changed);
                state.example.removeEventListener('toggle', state.exampleToggled);
                state.values.removeEventListener('toggle', state.valuesToggled);
                state.exampleValues.removeEventListener('toggle', state.exampleValuesToggled);
                purge(state.plot);
                purge(state.examplePlot);
                states.delete(root);
            });
        },
    });
})();
