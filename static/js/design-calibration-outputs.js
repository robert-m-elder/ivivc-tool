/* Method-specific IVIVC comparisons from retained simulated evaluations. */
(() => {
    'use strict';
    if (window.DesignCalibrationOutputs) return;

    const states = new WeakMap();
    const palette = ['#24535c', '#b34f00', '#7550a2'];
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const array = value => Array.isArray(value) ? value : [];
    const number = value => finite(value) ? String(Number(value.toPrecision(7))) : '—';
    const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));

    function legendName(value) {
        const lines = []; let remaining = String(value ?? '');
        while (remaining.length > 24) {
            let split = remaining.lastIndexOf(' ', 24);
            if (split < 8) split = 24;
            lines.push(remaining.slice(0, split)); remaining = remaining.slice(split).trimStart();
        }
        return [...lines, remaining].map(escape).join('<br>');
    }

    function roots(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-calibration-result]'));
        if (container.matches?.('[data-calibration-result]')) found.unshift(container);
        return found;
    }

    const selected = (state, panel) => panel.output.candidates.filter(candidate => state.selection.selectedIds.includes(candidate.id));
    const values = (panel, candidate, example) => {
        const view = candidate.views?.[panel.viewKey] || {};
        return example ? view.example || {} : view;
    };
    const counts = (value, example) => array(example ? value.point_counts : value.counts);
    const color = (state, candidate, index) => /^#[0-9a-f]{6}$/i.test(state.selection.colors[candidate.id] || '') ?
        state.selection.colors[candidate.id] : palette[index];

    function monotonic(values) {
        const points = values.filter(finite);
        if (points.length < 2) return true;
        const increasing = points[1] > points[0];
        return points.slice(1).every((value, index) => increasing ? value > points[index] : value < points[index]);
    }

    function series(panel, candidate, example, meta) {
        const value = values(panel, candidate, example), available = counts(value, example);
        const scalar = Boolean(meta.scalar || panel.output.scalar && panel.viewKey === 'native');
        const x = array(value.x).map((coordinate, index) =>
            finite(coordinate) && finite(value.y?.[index]) && available[index] > 0 ?
                scalar ? escape(candidate.name) : coordinate : null);
        const y = x.map((coordinate, index) => coordinate !== null ? value.y[index] : null);
        return {value, available, scalar, x, y,
            connected: !scalar && meta.connect_points !== false && monotonic(
                (meta.range_axis === 'x' ? y : x).filter(finite))};
    }

    function candidateTraces(state, panel, candidate, index, example, meta) {
        const {value, available, scalar, x, y, connected} = series(panel, candidate, example, meta);
        if (!y.some(finite)) return [];
        const shade = color(state, candidate, index), group = 'output-' + candidate.id;
        const prefix = panel.key + '-' + panel.viewKey + '-' + (example ? 'example-' : 'summary-') + candidate.id;
        const data = [], axis = meta.range_axis === 'x' ? 'x' : 'y';
        const main = {type: 'scatter', mode: connected && y.filter(finite).length > 1 ? 'lines+markers' : 'markers',
            x, y, uid: prefix + '-center', legendgroup: group, name: legendName(candidate.name), showlegend: true,
            line: {color: shade, width: 2}, marker: {color: shade, size: 7}, connectgaps: false,
            customdata: available.map(count => [count, candidate.n_simulations]),
            hovertemplate: escape(meta.x_label) + ': %{x}<br>' + escape(meta.y_label) + ': %{y:.7g}' +
                (example ? '' : '<br>Available estimates: %{customdata[0]}/%{customdata[1]}') + '<extra>%{fullData.name}</extra>',
            meta: {kind: 'center', candidate_id: candidate.id, range_axis: axis}};
        const valid = x.map((coordinate, position) => coordinate !== null && finite(y[position]) &&
            finite(value.lower?.[position]) && finite(value.upper?.[position]) && value.lower[position] <= value.upper[position] &&
            (example ? value.interval_counts?.[position] > 0 : available[position] >= 2));
        const segments = []; let segment = [];
        valid.forEach((usable, position) => {
            if (usable) segment.push(position);
            else if (segment.length) { segments.push(segment); segment = []; }
        });
        if (segment.length) segments.push(segment);
        const rgb = [1, 3, 5].map(start => parseInt(shade.slice(start, start + 2), 16)).join(',');
        const barIndices = [];
        segments.forEach((positions, segmentIndex) => {
            if (!connected || positions.length < 2) { barIndices.push(...positions); return; }
            const common = {type: 'scatter', mode: 'lines', name: legendName(candidate.name), legendgroup: group,
                showlegend: false, line: {width: 0}, connectgaps: false,
                customdata: positions.map(position => [value.lower[position], value.upper[position]])};
            for (const bound of ['lower', 'upper']) {
                const varying = positions.map(position => value[bound][position]);
                data.push({...common, uid: prefix + '-range-' + segmentIndex + '-' + bound,
                    x: axis === 'x' ? varying : positions.map(position => x[position]),
                    y: axis === 'y' ? varying : positions.map(position => y[position]),
                    ...(bound === 'upper' ? {fill: axis === 'x' ? 'tonextx' : 'tonexty', fillcolor: 'rgba(' + rgb + ',0.13)',
                        hovertemplate: (example ? 'Fitted interval: ' : 'Across-study central 95% range: ') +
                            '%{customdata[0]:.7g} to %{customdata[1]:.7g}<extra>%{fullData.name}</extra>'} : {hoverinfo: 'skip'}),
                    meta: {kind: 'range_' + bound, candidate_id: candidate.id, range_axis: axis}});
            }
        });
        if (barIndices.length) {
            const plus = x.map(() => null), minus = x.map(() => null), outside = [];
            barIndices.forEach(position => {
                const center = axis === 'x' ? x[position] : y[position];
                if (value.lower[position] <= center && center <= value.upper[position]) {
                    plus[position] = value.upper[position] - center; minus[position] = center - value.lower[position];
                } else outside.push(position);
            });
            if (plus.some(finite)) main['error_' + axis] = {type: 'data', symmetric: false, array: plus,
                arrayminus: minus, visible: true, color: shade, thickness: 1.5};
            // Some fitted inverse intervals do not contain the point estimate.
            // Draw their actual limits as separate segments, never negative bars.
            if (outside.length) {
                const intervalX = [], intervalY = [];
                outside.forEach(position => {
                    intervalX.push(axis === 'x' ? value.lower[position] : x[position], axis === 'x' ? value.upper[position] : x[position], null);
                    intervalY.push(axis === 'y' ? value.lower[position] : y[position], axis === 'y' ? value.upper[position] : y[position], null);
                });
                data.push({type: 'scatter', mode: 'lines', x: intervalX, y: intervalY, uid: prefix + '-fitted-limits',
                    legendgroup: group, name: legendName(candidate.name), showlegend: false, line: {color: shade, width: 2},
                    connectgaps: false, hovertemplate: 'Original fitted interval limit<extra>%{fullData.name}</extra>',
                    meta: {kind: 'interval_limits', candidate_id: candidate.id}});
            }
        }
        data.push(main);
        return data;
    }

    function truthTraces(state, panel, example, meta) {
        const groups = new Map();
        selected(state, panel).forEach(candidate => {
            const truth = values(panel, candidate, example).truth;
            if (!truth) return;
            const scalar = Boolean(meta.scalar || panel.output.scalar && panel.viewKey === 'native');
            const x = array(truth.x).map((coordinate, index) => finite(coordinate) && finite(truth.y?.[index]) && truth.counts?.[index] > 0 ?
                scalar ? escape(candidate.name) : coordinate : null);
            const y = x.map((coordinate, index) => coordinate !== null ? truth.y[index] : null);
            if (!y.some(finite)) return;
            const key = JSON.stringify(scalar ? y : [x, y]);
            if (!groups.has(key)) groups.set(key, {x, y, names: [candidate.name], scalar});
            else if (scalar) { const item = groups.get(key); item.x.push(...x); item.y.push(...y); item.names.push(candidate.name); }
        });
        return Array.from(groups.values(), (item, index) => ({type: 'scatter',
            mode: item.y.filter(finite).length > 1 && meta.connect_points !== false ? 'lines+markers' : 'markers',
            x: item.x, y: item.y, uid: panel.key + '-' + panel.viewKey + '-truth-' + index,
            name: legendName((example ? 'Assumed truth — study 1' : 'Median assumed truth') +
                (groups.size > 1 ? ' — ' + item.names[0] : '')), showlegend: true,
            legendgroup: 'truth-' + index, line: {color: '#62717c', width: 2, dash: 'dash'},
            marker: {color: '#62717c', size: 6}, connectgaps: false,
            hovertemplate: escape(meta.x_label) + ': %{x}<br>' + escape(meta.y_label) + ': %{y:.7g}<extra>%{fullData.name}</extra>',
            meta: {kind: 'truth'}}));
    }

    function figure(state, panel, example) {
        const meta = panel.output.views[panel.viewKey];
        const data = truthTraces(state, panel, example, meta);
        selected(state, panel).forEach((candidate, index) => data.push(...candidateTraces(state, panel, candidate, index, example, meta)));
        const key = ['calibration-output', panel.key, panel.viewKey, example ? 'example' : 'summary',
            ...state.selection.selectedIds.map(id => id + '-' + (state.selection.colors[id] || '').replace('#', ''))].join('-');
        const lines = data.filter(trace => trace.showlegend).reduce((sum, trace) => sum + trace.name.split('<br>').length, 0);
        const bottom = 70 + 20 * lines;
        const specification = {data, render_key: key, layout: {
            title: {text: escape((example ? 'Shared simulated study 1 — ' : '') + meta.title), x: 0},
            xaxis: {title: {text: escape(meta.x_label)}, showgrid: false, zeroline: false, automargin: true,
                ...(meta.scalar ? {type: 'category'} : {})},
            yaxis: {title: {text: escape(meta.y_label)}, gridcolor: '#e5ebef', zeroline: false, automargin: true},
            font: {family: 'Arial, sans-serif', color: '#20313d'}, paper_bgcolor: 'white', plot_bgcolor: 'white',
            margin: {l: 75, r: 25, t: 65, b: bottom}, height: 320 + bottom,
            legend: {orientation: 'h', x: 0, y: -45 / 255, yanchor: 'top', groupclick: 'togglegroup', font: {size: 11}},
            hovermode: 'closest', uirevision: key}};
        const recovered = data.some(trace => trace.meta?.kind === 'center');
        if (!recovered) specification.empty_message = state.selection.selectedIds.length ?
            'No selected estimates are available; assumed truth is shown only where supplied. Read the availability counts.' : 'Choose a design in the candidate table.';
        if (!data.length) specification.layout.annotations = [{xref: 'paper', yref: 'paper', x: .5, y: .5,
            text: escape(specification.empty_message), showarrow: false}];
        return specification;
    }

    function addRow(body, values) {
        const document = body.ownerDocument || globalThis.document;
        const row = document.createElement('tr');
        values.forEach(value => { const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell); });
        body.append(row);
    }

    function numerical(state, panel, example) {
        if (example ? !panel.example.open : !panel.values.open) return;
        const body = example ? panel.exampleBody : panel.valuesBody;
        body.replaceChildren();
        const meta = panel.output.views[panel.viewKey], axis = meta.range_axis === 'x' ? 'x' : 'y';
        if (!example && panel.caption) panel.caption.textContent = meta.title + ' — across simulated studies';
        selected(state, panel).forEach(candidate => {
            const item = series(panel, candidate, example, meta), value = item.value;
            array(value.x).forEach((_, index) => {
                const x = item.scalar ? candidate.name : number(value.x[index]), y = number(item.y[index]);
                if (example) addRow(body, [candidate.name, x, y,
                    number(value.lower?.[index]), number(value.upper?.[index]),
                    number(value.truth?.[axis]?.[index]), item.available[index] > 0 ?
                        value.interval_counts?.[index] > 0 ? 'Point and fitted interval available' : 'Point available; fitted interval unavailable' : 'Point unavailable']);
                else addRow(body, [candidate.name, x, y,
                    item.available[index] >= 2 ? number(value.lower?.[index]) : '—',
                    item.available[index] >= 2 ? number(value.upper?.[index]) : '—',
                    (item.available[index] || 0) + ' / ' + candidate.n_simulations]);
            });
        });
    }

    function fallback(state, panel, example) {
        const wrapper = example ? panel.examplePlot : panel.plot;
        const target = wrapper.querySelector('[data-design-plot-target]');
        const alternate = wrapper.querySelector('[data-design-plot-fallback]');
        if (target) target.hidden = true;
        if (alternate) alternate.hidden = false;
        const status = wrapper.querySelector('[data-design-plot-status]');
        if (status) status.textContent = 'Interactive plot unavailable. Numerical values remain available below.';
        if (!example) panel.values.open = true;
        numerical(state, panel, example);
    }

    function draw(state, panel, example) {
        if (example && !panel.example.open) return;
        const wrapper = example ? panel.examplePlot : panel.plot;
        const version = example ? ++panel.exampleVersion : ++panel.version;
        const current = () => !state.disposed && version === (example ? panel.exampleVersion : panel.version);
        wrapper.hidden = false;
        const spec = figure(state, panel, example), target = wrapper.querySelector('[data-design-plot-target]');
        if (target) target.setAttribute('aria-label', panel.output.views[panel.viewKey].title);
        try {
            if (typeof window.DesignPlots?.update !== 'function') throw new Error('Renderer unavailable.');
            Promise.resolve(window.DesignPlots.update(wrapper, spec)).then(result => {
                if (current() && (result?.ok === false || (result == null && target?.hidden))) fallback(state, panel, example);
            }).catch(() => { if (current()) fallback(state, panel, example); });
        } catch (error) { if (current()) fallback(state, panel, example); }
    }

    function renderPanel(state, panel) {
        const meta = panel.output.views[panel.viewKey], rows = selected(state, panel);
        const special = meta.connect_points === false ? ' Repeated or nonmonotonic coordinates are shown as points with pointwise ranges, without a connecting band.' : '';
        const unavailable = panel.viewKey !== 'response' && panel.output.views.response?.available === false ?
            ' ' + (panel.output.views.response.help || 'A response-versus-time view is unavailable for these retained coordinates.') : '';
        if (panel.help) panel.help.textContent = (meta.help || panel.output.message || '') + special + unavailable;
        for (const [selector, label] of [['[data-calibration-output-x-heading]', meta.x_label], ['[data-calibration-output-y-heading]', meta.y_label],
            ['[data-calibration-output-example-x-heading]', meta.x_label], ['[data-calibration-output-example-y-heading]', meta.y_label]]) {
            const heading = panel.element.querySelector(selector); if (heading) heading.textContent = label;
        }
        panel.status.textContent = rows.length ? rows.map(candidate => {
            const available = counts(values(panel, candidate, false), false).filter(finite);
            const minimum = available.length ? Math.min(...available) : 0, maximum = available.length ? Math.max(...available) : 0;
            return candidate.name + ': ' + (minimum === maximum ? minimum : minimum + '–' + maximum) + ' / ' + candidate.n_simulations + ' available estimates at each point.';
        }).join(' ') : 'Choose up to three designs in the candidate table.';
        panel.exampleStatus.textContent = 'Study 1 is used for every selected design, including failed calculations. ' +
            rows.map(candidate => candidate.name + ': ' + (values(panel, candidate, true).message || 'Read point and interval availability below.')).join(' ');
        numerical(state, panel, false); numerical(state, panel, true);
        draw(state, panel, false); draw(state, panel, true);
    }

    function select(state, selection) {
        state.selection = {selectedIds: array(selection?.selectedIds).slice(0, 3), colors: {...selection?.colors}};
        state.panels.forEach(panel => renderPanel(state, panel));
    }

    function initialize(root) {
        if (states.has(root)) return;
        try {
            const inspection = JSON.parse(root.querySelector('[data-calibration-inspection]')?.textContent || 'null');
            if (inspection?.schema_version !== 1 || !inspection.outputs) return;
            const state = {root, panels: [], disposed: false, selection: {selectedIds: [], colors: {}}};
            root.querySelectorAll('[data-calibration-output]').forEach(element => {
                const key = element.dataset.calibrationOutput, output = inspection.outputs[key];
                if (!output || !array(output.candidates).length) return;
                const get = attribute => element.querySelector('[' + attribute + ']');
                const panel = {element, key, output, version: 0, exampleVersion: 0,
                    view: get('data-calibration-output-view'), status: get('data-calibration-output-status'), help: get('data-calibration-output-help'),
                    plot: get('data-calibration-output-plot'), values: get('data-calibration-output-values'), valuesBody: get('data-calibration-output-values-body'),
                    caption: get('data-calibration-output-values-caption'), example: get('data-calibration-output-example'),
                    examplePlot: get('data-calibration-output-example-plot'), exampleStatus: get('data-calibration-output-example-status'),
                    exampleBody: get('data-calibration-output-example-values-body')};
                if ([panel.status, panel.plot, panel.values, panel.valuesBody, panel.example, panel.examplePlot, panel.exampleStatus, panel.exampleBody].some(value => !value)) return;
                const valid = Object.keys(output.views).filter(view => output.views[view].available !== false);
                if (!valid.length) return;
                panel.viewKey = key === 'prediction' && valid.includes('response') ? 'response' : valid.includes('native') ? 'native' : valid[0];
                if (panel.view) {
                    panel.view.querySelectorAll('option').forEach(option => { option.disabled = !valid.includes(option.value); });
                    panel.view.value = panel.viewKey;
                    panel.view.disabled = valid.length < 2;
                }
                panel.changed = event => {
                    if (!panel.view || event.target !== panel.view) return;
                    panel.viewKey = valid.includes(panel.view.value) ? panel.view.value : valid[0];
                    panel.view.value = panel.viewKey;
                    renderPanel(state, panel);
                };
                panel.toggled = () => { numerical(state, panel, true); draw(state, panel, true); };
                panel.valuesToggled = () => numerical(state, panel, false);
                element.addEventListener('change', panel.changed);
                panel.example.addEventListener('toggle', panel.toggled);
                panel.values.addEventListener('toggle', panel.valuesToggled);
                state.panels.push(panel);
            });
            if (!state.panels.length) return;
            state.selected = event => select(state, event.detail);
            states.set(root, state);
            root.addEventListener('design-calibration-selection-changed', state.selected);
            select(state, window.DesignCalibrationRecovery?.getSelection?.(root));
        } catch (error) { /* Existing numerical performance summaries remain available. */ }
    }

    window.DesignCalibrationOutputs = Object.freeze({
        initialize(container) { roots(container).forEach(initialize); },
        dispose(container) {
            roots(container).forEach(root => {
                const state = states.get(root); if (!state) return;
                state.disposed = true;
                root.removeEventListener('design-calibration-selection-changed', state.selected);
                state.panels.forEach(panel => {
                    panel.version++; panel.exampleVersion++;
                    panel.element.removeEventListener('change', panel.changed);
                    panel.example.removeEventListener('toggle', panel.toggled);
                    panel.values.removeEventListener('toggle', panel.valuesToggled);
                    for (const wrapper of [panel.plot, panel.examplePlot]) {
                        try { window.DesignPlots?.purge(wrapper); } catch (error) { /* Removed results need no further cleanup. */ }
                    }
                });
                states.delete(root);
            });
        },
    });
})();
