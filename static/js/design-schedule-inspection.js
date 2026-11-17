/* Display-only schedule selection from the current calculated comparison. */
(() => {
    'use strict';
    if (window.DesignScheduleInspection) return;

    const states = new WeakMap();
    const palette = Object.freeze(['#24535c', '#b34f00', '#7550a2']);
    const maximum = 3;

    function results(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-planning-result]'));
        if (container.matches && container.matches('[data-planning-result]')) found.unshift(container);
        return found;
    }

    function plotAction(action, ...args) {
        try {
            const api = window.DesignPlots;
            if (api && typeof api[action] === 'function') return Promise.resolve(api[action](...args));
        } catch (error) { /* Keep the selected schedules' numerical details available. */ }
        return Promise.resolve();
    }

    function selection(state) {
        const selectedIds = state.schedules.filter(row => state.selected.has(row.id)).map(row => row.id);
        return {selectedIds, colors: Object.fromEntries(selectedIds.map(id => [id, state.colors.get(id)]))};
    }

    function render(state) {
        const version = ++state.version;
        const chosen = selection(state);
        const count = chosen.selectedIds.length;
        state.controls.forEach(control => {
            const id = control.dataset.scheduleSelect;
            control.checked = state.selected.has(id);
            control.disabled = !state.byId.has(id) || (!control.checked && count === maximum);
        });
        state.root.querySelectorAll('[data-schedule-row]').forEach(row => {
            row.dataset.inspectionSelected = String(state.selected.has(row.dataset.scheduleRow));
        });
        state.root.querySelectorAll('[data-schedule-swatch]').forEach(swatch => {
            const id = swatch.dataset.scheduleSwatch;
            const selected = state.selected.has(id);
            swatch.hidden = !selected;
            if (selected) swatch.style.setProperty('--schedule-color', state.colors.get(id));
        });
        state.root.querySelectorAll('[data-schedule-detail]').forEach(detail => {
            detail.hidden = !state.selected.has(detail.dataset.scheduleDetail);
        });
        state.empty.hidden = count > 0;
        state.details.hidden = !count;
        state.help.hidden = !count;
        state.timeline.hidden = !count;
        state.status.textContent = count ? count + ' of ' + maximum + ' schedules selected.' +
            (count === maximum ? ' Deselect a schedule to choose another.' : '') :
            'No schedules selected. Choose up to three in the performance table.';
        if (!count) {
            plotAction('purge', state.timeline);
        } else {
            try {
                const builder = window.DesignTimelineFigure;
                const figure = builder && builder.build(state.inspection, chosen.selectedIds, chosen.colors);
                if (!figure || !window.DesignPlots || typeof window.DesignPlots.update !== 'function') {
                    throw new Error('Timeline renderer unavailable.');
                }
                plotAction('update', state.timeline, figure).then(() => {
                    if (state.disposed || version !== state.version) return;
                    const target = state.timeline.querySelector('[data-design-plot-target]');
                    if (!target || target.hidden) state.details.open = true;
                }).catch(() => {
                    if (!state.disposed && version === state.version) state.details.open = true;
                });
            } catch (error) {
                state.details.open = true;
                const status = state.timeline.querySelector('[data-design-plot-status]');
                if (status) status.textContent = 'Interactive timeline unavailable. Exact times and explanations remain available below.';
            }
        }
        // V4 can use this same selection/colors without changing analysis inputs.
        state.root.dispatchEvent(new CustomEvent('design-schedule-selection-changed', {
            detail: chosen, bubbles: true,
        }));
    }

    function initializeResult(root) {
        if (states.has(root)) return;
        const controls = Array.from(root.querySelectorAll('[data-schedule-select]'));
        if (!controls.length) return;
        const status = root.querySelector('[data-schedule-selection-status]');
        try {
            const encoded = root.querySelector('[data-planning-inspection]');
            const inspection = JSON.parse(encoded ? encoded.textContent : 'null');
            const schedules = inspection && inspection.schedule_data && inspection.schedule_data.schedules;
            if (inspection.schema_version !== 1 || !Array.isArray(schedules) || !schedules.length ||
                schedules.some(row => !row || typeof row.id !== 'string' || !Array.isArray(row.points)) ||
                new Set(schedules.map(row => row.id)).size !== schedules.length) {
                throw new Error('Unavailable inspection data.');
            }
            const state = {
                root, inspection, schedules, controls, status,
                byId: new Map(schedules.map(row => [row.id, row])),
                selected: new Set(), colors: new Map(), version: 0, disposed: false,
                timeline: root.querySelector('[data-schedule-timeline]'),
                details: root.querySelector('[data-schedule-details]'),
                empty: root.querySelector('[data-schedule-selection-empty]'),
                help: root.querySelector('[data-schedule-timeline-help]'),
            };
            if (!state.status || !state.timeline || !state.details || !state.empty || !state.help) {
                throw new Error('Incomplete inspection display.');
            }
            state.changed = event => {
                const control = event.target;
                if (!state.controls.includes(control)) return;
                const id = control.dataset.scheduleSelect;
                if (control.checked && state.byId.has(id) && state.selected.size < maximum) {
                    if (!state.selected.has(id)) {
                        const used = new Set(selection(state).selectedIds.map(key => state.colors.get(key)));
                        const previous = state.colors.get(id);
                        state.colors.set(id, previous && !used.has(previous) ? previous :
                            palette.find(color => !used.has(color)));
                        state.selected.add(id);
                    }
                } else if (!control.checked) {
                    state.selected.delete(id);
                }
                render(state);
            };
            const initial = schedules.find(row => row.kind === 'suggested') || schedules[0];
            state.selected.add(initial.id);
            state.colors.set(initial.id, palette[0]);
            states.set(root, state);
            root.addEventListener('change', state.changed);
            render(state);
        } catch (error) {
            if (status) status.textContent = 'Schedule inspection unavailable. Exact times and explanations for all schedules remain available below.';
        }
    }

    window.DesignScheduleInspection = Object.freeze({
        initialize(container) { results(container).forEach(initializeResult); },
        dispose(container) {
            results(container).forEach(root => {
                const state = states.get(root);
                if (!state) return;
                state.disposed = true;
                state.version++;
                root.removeEventListener('change', state.changed);
                plotAction('purge', state.timeline);
                states.delete(root);
            });
        },
        getSelection(root) {
            const state = states.get(root);
            return state ? selection(state) : {selectedIds: [], colors: {}};
        },
    });
})();
