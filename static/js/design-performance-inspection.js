/* Display-only recovery measures from the current simulated comparison. */
(() => {
    'use strict';
    if (window.DesignPerformanceInspection) return;

    const states = new WeakMap();
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const array = value => Array.isArray(value) ? value : [];
    const number = value => finite(value) ? String(Number(value.toPrecision(6))) : '—';
    const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
    const endpointTime = (value, status) => status === 'beyond_horizon' ?
        'Beyond the assessment horizon' : number(value);

    function roots(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-planning-result]'));
        if (container.matches?.('[data-planning-result]')) found.unshift(container);
        return found;
    }

    function plotAction(action, ...args) {
        try {
            if (typeof window.DesignPlots?.[action] === 'function') return Promise.resolve(window.DesignPlots[action](...args));
        } catch (error) { /* The numerical table remains available. */ }
        return Promise.resolve();
    }

    function values(schedule, measure, view) {
        return schedule.measures?.[measure.id]?.[view] || {};
    }

    function availability(schedule, value, measure, view) {
        if (measure.kind === 'curve') return number(schedule.curve_interval_width) +
            ' (' + number(schedule.interval_count) + ' fits)';
        const total = number(schedule.n_simulations);
        const reached = number(value.n_reached) + ' / ' + total + ' reached';
        if (view === 'error') return number(value.n_pairs) + ' / ' + total + ' finite pairs; ' + reached;
        return number(value.n_valid) + ' / ' + total + ' valid; ' + reached +
            '; ' + number(value.n_failed) + ' failed; ' + number(value.n_invalid) + ' invalid';
    }

    function updateTable(state, measure, view, labels) {
        const caption = state.table.querySelector('[data-performance-caption]');
        if (caption) caption.textContent = 'Schedule performance: ' + measure.label +
            (view === 'time' ? '; recovered endpoint times across valid outcomes' :
                measure.kind === 'endpoint' ? '; absolute timing errors among finite true–recovered pairs' :
                    '; curve recovery errors among successful fits');
        state.primaryHeading.textContent = labels.primary_label;
        state.secondaryHeading.textContent = labels.secondary_label;
        state.availabilityHeading.textContent = measure.kind === 'curve' ? 'Mean fitted curve interval width' :
            view === 'error' ? 'Finite pairs and endpoint reach' : 'Endpoint availability';
        state.table.querySelectorAll('[data-schedule-row]').forEach(row => {
            const schedule = state.byId.get(row.dataset.scheduleRow);
            if (!schedule) return;
            const value = values(schedule, measure, view);
            const primary = row.querySelector('[data-performance-primary]');
            const secondary = row.querySelector('[data-performance-secondary]');
            const available = row.querySelector('[data-performance-availability]');
            if (primary) primary.textContent = view === 'time' ? endpointTime(value.primary, value.primary_status) : number(value.primary);
            if (secondary) secondary.textContent = view === 'time' ?
                value.n_valid >= 2 ? endpointTime(value.lower, value.lower_status) + ' to ' + endpointTime(value.upper, value.upper_status) :
                    'Unavailable (fewer than two valid outcomes)' : number(value.secondary);
            if (available) available.textContent = availability(schedule, value, measure, view);
        });
    }

    function hoverDetail(schedule, value, measure, view) {
        let detail = escape(schedule.name) + '<br>' + escape(availability(schedule, value, measure, view));
        if (view === 'time') detail += '<br>Central 95% range: ' + escape(value.n_valid >= 2 ?
            endpointTime(value.lower, value.lower_status) + ' to ' + endpointTime(value.upper, value.upper_status) :
            'Unavailable (fewer than two valid outcomes)');
        return detail;
    }

    function figure(state, measure, view, labels) {
        const data = [], groups = new Map();
        state.schedules.forEach(schedule => {
            if (!groups.has(schedule.kind)) groups.set(schedule.kind, []);
            groups.get(schedule.kind).push(schedule);
        });
        const statistics = view === 'time' ? ['primary'] : ['primary', 'secondary'];
        function trace(rows, statistic, name, color, selected, uid) {
            const y = rows.map(schedule => {
                const value = values(schedule, measure, view);
                return finite(value[statistic]) && finite(schedule.x) &&
                    (view !== 'time' || value.primary_status === 'resolved') ? value[statistic] : null;
            });
            if (!y.some(finite)) return null;
            const secondary = statistic === 'secondary';
            const result = {
                type: 'scatter', uid, name: escape(name), legendgroup: uid.split(':')[0],
                x: rows.map(schedule => finite(schedule.x) ? schedule.x : null), y,
                mode: selected || String(rows[0]?.kind).toLowerCase() === 'manual' ? 'markers' : 'lines+markers',
                connectgaps: false,
                line: {color, width: 1.5, dash: secondary ? 'dash' : 'solid'},
                marker: {color, size: selected ? 11 : 7, symbol: secondary ? 'diamond' : 'circle',
                    ...(selected ? {line: {color: '#ffffff', width: 1.5}} : {})},
                showlegend: !selected || !secondary,
                customdata: rows.map(schedule => hoverDetail(schedule, values(schedule, measure, view), measure, view)),
                hovertemplate: '%{customdata}<br>' + escape(state.performance.x_label) + ': %{x:.6g}<br>' +
                    escape(secondary ? labels.secondary_label : labels.primary_label) + ': %{y:.6g}<extra>%{fullData.name}</extra>',
                meta: {selected, statistic, schedule_ids: rows.map(schedule => schedule.id)},
            };
            if (view === 'time') {
                const ranges = rows.map((schedule, index) => {
                    const value = values(schedule, measure, view);
                    return value.n_valid >= 2 && value.lower_status === 'resolved' && value.upper_status === 'resolved' &&
                        finite(y[index]) && finite(value.lower) && finite(value.upper) &&
                        value.lower <= y[index] && y[index] <= value.upper ?
                        [value.upper - y[index], y[index] - value.lower] : [null, null];
                });
                if (ranges.some(range => finite(range[0]))) result.error_y = {
                    type: 'data', symmetric: false, array: ranges.map(range => range[0]),
                    arrayminus: ranges.map(range => range[1]), visible: true, color, thickness: selected ? 2 : 1,
                };
            }
            return result;
        }
        let groupIndex = 0;
        groups.forEach((rows, kind) => {
            rows.sort((a, b) => a.x - b.x);
            const color = ['#78858f', '#929da6', '#acb5bc'][groupIndex++ % 3];
            statistics.forEach(statistic => {
                const label = statistic === 'primary' ? labels.primary_label : labels.secondary_label;
                const result = trace(rows, statistic, kind + ': ' + label, color, false, 'group-' + groupIndex + ':' + statistic);
                if (result) data.push(result);
            });
        });
        state.selection.selectedIds.forEach(id => {
            const schedule = state.byId.get(id), color = state.selection.colors[id];
            if (!schedule || typeof color !== 'string' || !/^#[0-9a-f]{6}$/i.test(color)) return;
            statistics.forEach(statistic => {
                const result = trace([schedule], statistic, schedule.name, color, true, 'selected-' + id + ':' + statistic);
                if (result) data.push(result);
            });
        });
        return {data, layout: {
            title: {text: escape(labels.title), x: 0},
            xaxis: {title: {text: escape(state.performance.x_label)}, tickmode: 'linear', tick0: 0, dtick: 1,
                showgrid: false, zeroline: false, automargin: true},
            yaxis: {title: {text: escape(labels.y_label)}, gridcolor: '#e5ebef', zeroline: false, automargin: true},
            font: {family: 'Arial, sans-serif', color: '#20313d'}, paper_bgcolor: 'white', plot_bgcolor: 'white',
            margin: {l: 75, r: 25, t: 65, b: 175}, height: 500,
            legend: {orientation: 'h', x: 0, y: -.25, yanchor: 'top', font: {size: 11}, groupclick: 'togglegroup'},
            hovermode: 'closest', uirevision: 'performance-' + measure.id + '-' + view,
        }};
    }

    function numericalFallback(state, message) {
        const target = state.plot.querySelector('[data-design-plot-target]');
        const fallback = state.plot.querySelector('[data-design-plot-fallback]');
        const status = state.plot.querySelector('[data-design-plot-status]');
        if (target) target.hidden = true;
        if (fallback) fallback.hidden = false;
        if (status) status.textContent = message;
    }

    function render(state) {
        const version = ++state.version;
        const measure = state.measures.get(state.measure.value) || state.performance.measures[0];
        state.measure.value = measure.id;
        const endpoint = measure.kind === 'endpoint';
        state.viewField.hidden = !endpoint;
        state.view.disabled = !endpoint;
        if (!['error', 'time'].includes(state.view.value)) state.view.value = 'error';
        const view = endpoint ? state.view.value : 'error';
        const labels = measure.views[view];
        if (!labels) return;
        updateTable(state, measure, view, labels);
        const missing = view === 'time' ? state.schedules.filter(schedule =>
            !finite(values(schedule, measure, view).primary)).map(schedule => schedule.name) : [];
        state.help.textContent = labels.help + ' Gray traces show all schedules by type; colored markers identify the selected schedules.' +
            (missing.length ? ' No finite median time is plotted for ' + missing.join('; ') +
                '. Read their beyond-horizon or unavailable outcomes in the table.' : '');
        const target = state.plot.querySelector('[data-design-plot-target]');
        if (target) target.setAttribute('aria-label', labels.title + '. ' + labels.y_label + ' versus ' + state.performance.x_label + '.');
        // The server-rendered SVG describes IRMSE. Do not show it under a
        // different measure if plotting is unavailable or a render is pending.
        const fallback = state.plot.querySelector('[data-design-plot-fallback]');
        if (fallback) fallback.textContent = 'Numerical results for this measure are available in the schedule performance table below.';
        const specification = figure(state, measure, view, labels);
        if (!specification.data.length) {
            plotAction('purge', state.plot);
            numericalFallback(state, 'No finite values to plot for this measure. Beyond-horizon and unavailable outcomes remain in the table.');
            return;
        }
        if (typeof window.DesignPlots?.update !== 'function') {
            numericalFallback(state, 'Interactive plot unavailable. Numerical results remain available below.');
            return;
        }
        plotAction('update', state.plot, specification).then(() => {
            if (state.disposed || version !== state.version) return;
            if (target?.hidden) numericalFallback(state, 'Interactive plot unavailable. Numerical results remain available below.');
        }).catch(() => {
            if (!state.disposed && version === state.version) numericalFallback(state,
                'Interactive plot unavailable. Numerical results remain available below.');
        });
    }

    function select(state, selection) {
        state.selection = {selectedIds: array(selection?.selectedIds).filter(id => state.byId.has(id)).slice(0, 3),
            colors: {...selection?.colors}};
        render(state);
    }

    function initialize(root) {
        if (states.has(root) || !root.querySelector('[data-performance-inspection]')) return;
        try {
            const encoded = root.querySelector('[data-planning-inspection]');
            const inspection = JSON.parse(encoded?.textContent || 'null');
            const performance = inspection?.performance;
            if (inspection?.schema_version !== 1 || !array(performance?.measures).length ||
                    !array(performance?.schedules).length) return;
            const state = {root, performance, schedules: performance.schedules,
                byId: new Map(performance.schedules.map(row => [row.id, row])),
                measures: new Map(performance.measures.map(row => [row.id, row])), version: 0, disposed: false,
                selection: {selectedIds: [], colors: {}},
                measure: root.querySelector('[data-performance-measure]'),
                view: root.querySelector('[data-performance-view]'), viewField: root.querySelector('[data-performance-view-field]'),
                plot: root.querySelector('[data-performance-plot]'), help: root.querySelector('[data-performance-help]'),
                table: root.querySelector('[data-performance-table]'),
                primaryHeading: root.querySelector('[data-performance-primary-heading]'),
                secondaryHeading: root.querySelector('[data-performance-secondary-heading]'),
                availabilityHeading: root.querySelector('[data-performance-availability-heading]')};
            if ([state.measure, state.view, state.viewField, state.plot, state.help, state.table,
                state.primaryHeading, state.secondaryHeading, state.availabilityHeading].some(element => !element)) return;
            state.measure.disabled = false;
            state.measure.value = state.measures.has('irmse') ? 'irmse' : performance.measures[0].id;
            state.view.value = 'error';
            state.changed = event => {
                if (event.target === state.measure || event.target === state.view) render(state);
            };
            state.selected = event => select(state, event.detail);
            states.set(root, state);
            root.addEventListener('change', state.changed);
            root.addEventListener('design-schedule-selection-changed', state.selected);
            select(state, window.DesignScheduleInspection?.getSelection(root));
        } catch (error) { /* Preserve the server-rendered IRMSE table on invalid data. */ }
    }

    window.DesignPerformanceInspection = Object.freeze({
        initialize(container) { roots(container).forEach(initialize); },
        dispose(container) {
            roots(container).forEach(root => {
                const state = states.get(root);
                if (!state) return;
                state.disposed = true;
                state.version++;
                root.removeEventListener('change', state.changed);
                root.removeEventListener('design-schedule-selection-changed', state.selected);
                plotAction('purge', state.plot);
                states.delete(root);
            });
        },
    });
})();
