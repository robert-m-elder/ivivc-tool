/* Calibration-curve comparisons use retained fits and a shared simulated study. */
(() => {
    'use strict';
    if (window.DesignCalibrationRecovery) return;

    const states = new WeakMap();
    const palette = ['#24535c', '#b34f00', '#7550a2'];
    const neutral = '#62717c';
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const array = value => Array.isArray(value) ? value : [];
    const number = value => finite(value) ? String(Number(value.toPrecision(7))) : '—';
    const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));

    function legendName(value) {
        const lines = [];
        let remaining = String(value ?? '');
        while (remaining.length > 24) {
            let split = remaining.lastIndexOf(' ', 24);
            if (split < 8) split = 24;
            lines.push(remaining.slice(0, split)); remaining = remaining.slice(split).trimStart();
        }
        lines.push(remaining);
        return lines.map(escape).join('<br>');
    }

    function roots(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-calibration-result]'));
        if (container.matches?.('[data-calibration-result]')) found.unshift(container);
        return found;
    }

    function selection(state) {
        const selectedIds = state.candidates.filter(row => state.selected.has(row.id)).map(row => row.id);
        return {selectedIds, colors: Object.fromEntries(selectedIds.map(id => [id, state.colors.get(id)]))};
    }

    function chosen(state) { return state.candidates.filter(row => state.selected.has(row.id)); }
    function side(state, candidate) { return candidate.sides?.[state.environment.value] || {}; }
    function environment(state) { return state.inspection.environments[state.environment.value]; }
    function sharedFixed(env) { return env.fixed && env.fixed_curve_shared && array(env.fixed_curve).some(finite); }

    function line(times, values, name, color, uid, group = uid, counts = null) {
        const y = array(times).map((time, index) => finite(time) && finite(values?.[index]) &&
            (!counts || counts[index] > 0) ? values[index] : null);
        if (!y.some(finite)) return null;
        return {type: 'scatter', mode: y.filter(finite).length > 1 ? 'lines' : 'markers',
            x: array(times).map(time => finite(time) ? time : null), y,
            uid, legendgroup: group, name: legendName(name), showlegend: true,
            line: {color, width: 2}, marker: {color, size: 7}, connectgaps: false,
            customdata: array(times).map((_, index) => counts ? counts[index] : null),
            hovertemplate: 'Time: %{x:.7g}<br>Response: %{y:.7g}' +
                (counts ? '<br>Usable simulated studies: %{customdata}' : '') + '<extra>%{fullData.name}</extra>'};
    }

    function ranges(times, values, candidate, color) {
        const segments = [], data = [];
        let segment = [];
        array(times).forEach((time, index) => {
            const lower = values.lower?.[index], upper = values.upper?.[index];
            if (finite(time) && finite(lower) && finite(upper) && lower <= upper && values.available_counts?.[index] >= 2) {
                segment.push({time, lower, upper, count: values.available_counts[index]});
            } else if (segment.length) { segments.push(segment); segment = []; }
        });
        if (segment.length) segments.push(segment);
        const rgb = [1, 3, 5].map(start => parseInt(color.slice(start, start + 2), 16)).join(',');
        segments.forEach((points, index) => {
            const common = {type: 'scatter', mode: 'lines', x: points.map(point => point.time),
                legendgroup: 'calibration-' + candidate.id, name: escape(candidate.name),
                line: {width: 0}, connectgaps: false, showlegend: false};
            data.push({...common, uid: candidate.id + '-range-' + index + '-lower',
                y: points.map(point => point.lower), hoverinfo: 'skip'});
            data.push({...common, uid: candidate.id + '-range-' + index + '-upper',
                y: points.map(point => point.upper), fill: 'tonexty', fillcolor: 'rgba(' + rgb + ',0.13)',
                customdata: points.map(point => [point.lower, point.upper, point.count]),
                hovertemplate: 'Time: %{x:.7g}<br>Across-study central 95% range: %{customdata[0]:.7g} to %{customdata[1]:.7g}' +
                    '<br>Usable simulated studies: %{customdata[2]}<extra>%{fullData.name}</extra>'});
        });
        return data;
    }

    function points(values, name, color, uid, group, existing = false) {
        const usable = array(values).filter(point => finite(point.time) && finite(point.response));
        return usable.length ? {type: 'scatter', mode: 'markers', name: legendName(name), uid, legendgroup: group,
            x: usable.map(point => point.time), y: usable.map(point => point.response),
            marker: {color, size: existing ? 9 : 7, symbol: existing ? 'circle-open' : 'circle'},
            showlegend: existing, hovertemplate: 'Time: %{x:.7g}<br>Observed response: %{y:.7g}<extra>%{fullData.name}</extra>'} : null;
    }

    function specification(state, example) {
        const env = environment(state), rows = chosen(state), data = [], medians = [];
        const envKey = state.environment.value;
        const key = ['calibration', example ? 'example' : 'summary', envKey,
            ...rows.map(row => row.id + '-' + state.colors.get(row.id).slice(1))].join('-');
        if (rows.length) {
            if (example) {
                const truth = rows.map(row => side(state, row).example?.truth).find(values => array(values).some(finite));
                const trace = line(env.time, truth, 'Assumed true curve — study 1', neutral, 'calibration-example-truth');
                if (trace) { trace.line.dash = 'dash'; data.push(trace); }
            } else {
                const trace = line(env.time, env.reference?.central, 'Central assumed reference', neutral, 'calibration-reference');
                if (trace) { trace.line.dash = 'dash'; data.push(trace); }
            }
            const fixed = sharedFixed(env) && rows.some(row => example ? side(state, row).example?.fit_available : side(state, row).n_curves > 0);
            if (fixed) {
                const trace = line(env.time, env.fixed_curve, 'Shared fit to fixed calibration data', '#20313d', 'calibration-fixed-fit');
                if (trace) data.push(trace);
            }
            rows.forEach(candidate => {
                const values = side(state, candidate), color = state.colors.get(candidate.id);
                if (example) {
                    const sample = values.example || {}, group = 'calibration-' + candidate.id;
                    if (!fixed && sample.fit_available) {
                        const trace = line(env.time, sample.curve, candidate.name, color, candidate.id + '-example-fit', group);
                        if (trace) data.push(trace);
                    }
                    const observed = points(sample.new_points, candidate.name + ' — new observations', color,
                        candidate.id + '-example-new', group);
                    if (observed) { observed.showlegend = !sample.fit_available; data.push(observed); }
                } else if (!fixed) {
                    const trace = line(env.time, values.median, candidate.name, color, candidate.id + '-median',
                        'calibration-' + candidate.id, values.available_counts);
                    if (trace) {
                        data.push(...ranges(env.time, values, candidate, color));
                        medians.push(trace);
                    }
                }
            });
            if (example) {
                const original = points(env.existing_points, 'Original calibration observations', '#20313d',
                    'calibration-existing', 'calibration-existing', true);
                if (original) data.push(original);
            }
        }
        data.push(...medians);
        const lines = data.filter(trace => trace.showlegend).reduce((count, trace) => count + trace.name.split('<br>').length, 0);
        const bottom = 70 + 18 * lines;
        const result = {data, render_key: key, layout: {
            title: {text: escape((example ? 'Illustrative simulated study 1 — ' : 'Recovered calibration responses — ') + env.label), x: 0},
            xaxis: {title: {text: escape(env.label + ' time')}, showgrid: false, zeroline: false, automargin: true},
            yaxis: {title: {text: 'Analyzed response'}, gridcolor: '#e5ebef', zeroline: false, automargin: true},
            font: {family: 'Arial, sans-serif', color: '#20313d'}, paper_bgcolor: 'white', plot_bgcolor: 'white',
            margin: {l: 70, r: 25, t: 65, b: bottom}, height: 320 + bottom,
            legend: {orientation: 'h', x: 0, y: -45 / 255, yanchor: 'top', groupclick: 'togglegroup', font: {size: 11}},
            hovermode: 'closest', uirevision: key,
        }};
        if (!data.length) {
            result.empty_message = rows.length ? 'No recovered curves are available. Review the fit and usable-study counts.' :
                'Choose up to three designs in the candidate table.';
            result.layout.annotations = [{xref: 'paper', yref: 'paper', x: .5, y: .5,
                text: escape(result.empty_message), showarrow: false}];
        }
        return result;
    }

    function row(body, values) {
        const document = body.ownerDocument || globalThis.document;
        const tr = document.createElement('tr');
        values.forEach(value => { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); });
        body.append(tr);
    }

    function summaryValues(state) {
        if (!state.values.open) return;
        const env = environment(state);
        state.valuesBody.replaceChildren();
        if (state.caption) state.caption.textContent = env.label + ' recovered calibration responses for selected designs';
        if (sharedFixed(env) && chosen(state).some(candidate => side(state, candidate).n_curves > 0)) {
            array(env.time).forEach((time, index) => row(state.valuesBody,
                ['Shared fixed calibration fit', number(time), number(env.fixed_curve[index]), '—', '—', 'Fixed data']));
            return;
        }
        chosen(state).forEach(candidate => {
            const values = side(state, candidate);
            array(env.time).forEach((time, index) => {
                const usable = values.available_counts?.[index] || 0;
                row(state.valuesBody, [candidate.name, number(time), usable ? number(values.median?.[index]) : '—',
                    usable >= 2 ? number(values.lower?.[index]) : '—', usable >= 2 ? number(values.upper?.[index]) : '—',
                    usable + ' / ' + candidate.n_simulations]);
            });
        });
    }

    function exampleValues(state) {
        if (!state.example.open || !state.inspection.example?.available) return;
        const env = environment(state), rows = chosen(state);
        state.exampleBody.replaceChildren();
        const truth = rows.map(candidate => side(state, candidate).example?.truth).find(values => array(values).some(finite));
        array(env.time).forEach((time, index) => {
            if (finite(truth?.[index])) row(state.exampleBody, ['Shared generating curve', 'True', number(time), number(truth[index])]);
        });
        const fixed = sharedFixed(env) && rows.some(candidate => side(state, candidate).example?.fit_available);
        if (fixed) array(env.time).forEach((time, index) => row(state.exampleBody,
            ['Shared fixed calibration fit', 'Recovered', number(time), number(env.fixed_curve[index])]));
        rows.forEach(candidate => {
            const sample = side(state, candidate).example || {};
            if (!fixed && sample.fit_available) array(env.time).forEach((time, index) => row(state.exampleBody,
                [candidate.name, 'Recovered', number(time), number(sample.curve?.[index])]));
            array(sample.new_points).forEach(point => row(state.exampleBody,
                [candidate.name, 'New observation', number(point.time), number(point.response)]));
        });
        if (rows.length) array(env.existing_points).forEach(point => row(state.exampleBody,
            ['Shared calibration observations', 'Existing observation', number(point.time), number(point.response)]));
    }

    function fallback(state, wrapper, example) {
        const target = wrapper.querySelector('[data-design-plot-target]');
        const alternate = wrapper.querySelector('[data-design-plot-fallback]');
        const status = wrapper.querySelector('[data-design-plot-status]');
        if (target) target.hidden = true;
        if (alternate) alternate.hidden = false;
        if (status) status.textContent = 'Interactive plot unavailable. Numerical values remain available below.';
        if (example) exampleValues(state);
        else { state.values.open = true; summaryValues(state); }
    }

    function draw(state, example) {
        const wrapper = example ? state.examplePlot : state.plot;
        if (!wrapper || (example && (!state.example.open || !state.inspection.example?.available))) return;
        const version = example ? ++state.exampleVersion : ++state.version;
        const current = () => !state.disposed && version === (example ? state.exampleVersion : state.version);
        wrapper.hidden = false;
        const figure = specification(state, example);
        const target = wrapper.querySelector('[data-design-plot-target]');
        if (target) target.setAttribute('aria-label', figure.layout.title.text + '. Response versus time.');
        try {
            if (typeof window.DesignPlots?.update !== 'function') throw new Error('Renderer unavailable.');
            Promise.resolve(window.DesignPlots.update(wrapper, figure)).then(result => {
                if (current() && (result?.ok === false || (result == null && target?.hidden))) fallback(state, wrapper, example);
            }).catch(() => { if (current()) fallback(state, wrapper, example); });
        } catch (error) { if (current()) fallback(state, wrapper, example); }
    }

    function render(state) {
        const rows = chosen(state), picked = selection(state);
        state.controls.forEach(control => {
            const id = control.dataset.calibrationSelect;
            control.checked = state.selected.has(id);
            control.disabled = !state.byId.has(id) || (!control.checked && rows.length >= 3);
        });
        state.root.querySelectorAll('[data-calibration-swatch]').forEach(swatch => {
            const id = swatch.dataset.calibrationSwatch;
            swatch.hidden = !state.selected.has(id);
            if (!swatch.hidden) swatch.style.setProperty('--schedule-color', state.colors.get(id));
        });
        state.selectionStatus.textContent = rows.length + ' of 3 designs selected.' +
            (rows.length === 3 ? ' Deselect a design to choose another.' : '');
        if (!state.inspection.supported) {
            state.root.dispatchEvent(new CustomEvent('design-calibration-selection-changed', {detail: picked, bubbles: true}));
            return;
        }
        const env = environment(state);
        state.root.querySelectorAll('[data-calibration-availability]').forEach(element => {
            const candidate = state.byId.get(element.dataset.calibrationAvailability);
            if (!candidate) return;
            element.hidden = !state.selected.has(candidate.id);
            const values = side(state, candidate), counts = array(values.available_counts).filter(finite);
            const usable = counts.length ? Math.min(...counts) + '–' + Math.max(...counts) : '0';
            element.textContent = candidate.name + ': ' + candidate.n_fit_successful + ' / ' + candidate.n_simulations +
                ' successful IVIVC fits; ' + (candidate.n_simulations - candidate.n_fit_successful) + ' failed. ' +
                env.label + ' usable studies at each time: ' + usable + '.';
        });
        if (state.empty) state.empty.hidden = rows.length > 0;
        state.status.textContent = !rows.some(candidate => side(state, candidate).n_curves > 0) ?
            'No usable reconstructed curves are available for the selected designs in this environment. Any dashed assumed reference is a generating assumption, not a recovered fit.' :
            env.fixed ? 'The unchanged calibration data use a shared fitted curve where their reconstructed fits are identical. Availability counts still describe every selected design.' :
                'Lines show median reconstructed responses; shading shows pointwise central 95% ranges across usable simulated studies.';
        const exampleMessages = rows.map(candidate => candidate.name + ': ' +
            (side(state, candidate).example?.message || (side(state, candidate).example?.fit_available ? 'Fit available.' : 'Fit unavailable; no replacement study is substituted.')));
        state.exampleStatus.textContent = (state.inspection.example?.message || 'Study 1 is predetermined for every design.') +
            (exampleMessages.length ? ' ' + exampleMessages.join(' ') : ' Choose a design to inspect the example.');
        state.example.hidden = !state.inspection.example?.available;
        summaryValues(state); exampleValues(state); draw(state, false); draw(state, true);
        state.root.dispatchEvent(new CustomEvent('design-calibration-selection-changed', {detail: picked, bubbles: true}));
    }

    function initialize(root) {
        if (states.has(root) || !root.querySelector('[data-calibration-recovery]')) return;
        try {
            const inspection = JSON.parse(root.querySelector('[data-calibration-inspection]')?.textContent || 'null');
            if (inspection?.schema_version !== 1 || !array(inspection.candidates).length) return;
            const candidates = inspection.candidates;
            const state = {root, inspection, candidates, byId: new Map(candidates.map(row => [row.id, row])),
                selected: new Set([candidates[0].id]), colors: new Map([[candidates[0].id, palette[0]]]),
                controls: Array.from(root.querySelectorAll('[data-calibration-select]')), version: 0, exampleVersion: 0, disposed: false,
                environment: root.querySelector('[data-calibration-environment]'), selectionStatus: root.querySelector('[data-calibration-selection-status]'),
                status: root.querySelector('[data-calibration-recovery-status]'), empty: root.querySelector('[data-calibration-recovery-empty]'),
                plot: root.querySelector('[data-calibration-recovery-plot]'), values: root.querySelector('[data-calibration-values]'),
                valuesBody: root.querySelector('[data-calibration-values-body]'), caption: root.querySelector('[data-calibration-values-caption]'),
                example: root.querySelector('[data-calibration-example]'), examplePlot: root.querySelector('[data-calibration-example-plot]'),
                exampleStatus: root.querySelector('[data-calibration-example-status]'), exampleBody: root.querySelector('[data-calibration-example-values-body]')};
            if (!state.selectionStatus || (inspection.supported && [state.environment, state.status, state.plot,
                state.values, state.valuesBody, state.example, state.examplePlot, state.exampleStatus,
                state.exampleBody].some(element => !element))) return;
            if (inspection.supported) {
                state.environment.value = inspection.environments[inspection.default_environment] ? inspection.default_environment : 'in_vivo';
                state.environment.disabled = false;
            }
            state.changed = event => {
                if (event.target === state.environment) {
                    if (!inspection.environments[state.environment.value]) state.environment.value = inspection.default_environment;
                    render(state); return;
                }
                if (!state.controls.includes(event.target)) return;
                const id = event.target.dataset.calibrationSelect;
                if (event.target.checked && state.byId.has(id) && state.selected.size < 3) {
                    const used = new Set(selection(state).selectedIds.map(key => state.colors.get(key)));
                    const previous = state.colors.get(id);
                    state.colors.set(id, previous && !used.has(previous) ? previous : palette.find(color => !used.has(color)));
                    state.selected.add(id);
                } else if (!event.target.checked) state.selected.delete(id);
                render(state);
            };
            state.toggled = () => { exampleValues(state); draw(state, true); };
            state.valuesToggled = () => summaryValues(state);
            states.set(root, state);
            root.addEventListener('change', state.changed);
            if (inspection.supported) {
                state.example.addEventListener('toggle', state.toggled);
                state.values.addEventListener('toggle', state.valuesToggled);
            }
            render(state);
        } catch (error) { /* Existing numerical calibration results stay available. */ }
    }

    window.DesignCalibrationRecovery = Object.freeze({
        initialize(container) { roots(container).forEach(initialize); },
        dispose(container) {
            roots(container).forEach(root => {
                const state = states.get(root); if (!state) return;
                state.disposed = true; state.version++; state.exampleVersion++;
                root.removeEventListener('change', state.changed);
                state.example?.removeEventListener('toggle', state.toggled);
                state.values?.removeEventListener('toggle', state.valuesToggled);
                for (const wrapper of [state.plot, state.examplePlot]) {
                    try { window.DesignPlots?.purge(wrapper); } catch (error) { /* Detached output is already disposed. */ }
                }
                states.delete(root);
            });
        },
        getSelection(root) { const state = states.get(root); return state ? selection(state) : {selectedIds: [], colors: {}}; },
    });
})();
