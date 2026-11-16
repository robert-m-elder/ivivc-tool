'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = () => fs.readFileSync(path.join(__dirname, '../static/js/design-recovery-inspection.js'), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };

function inspectionData() {
    const schedules = Array.from({length: 4}, (_, index) => ({
        id: 'schedule-' + index, name: index === 0 ? '<img src=x> first schedule' : 'Schedule ' + index,
        evaluation_times: [0, 1, 2], n_simulations: 4, n_successful: 3, n_failed: 1,
        n_complete_recovered_curves: 2,
        recovered_response: {lower: [.8, .5, .2], median: [1, .7, .4], upper: [1.2, .9, .6], available_counts: [3, 3, 2]},
        recovery_error: {lower: [-.2, -.3, -.4], median: [-.1, -.2, -.3], upper: [0, -.1, -.2], available_counts: [3, 3, 2]},
        example: {simulation_index: 0, study_number: 1, trajectory_draw_index: 8,
            fit_success: index !== 1, fit_message: index === 1 ? 'Example fit failed without replacement.' : null,
            true_response: [1, .7, .4], recovered_response: index === 1 ? [null, null, null] : [.99, .69, .39],
            recovery_error: index === 1 ? [null, null, null] : [-.01, -.01, -.01],
            observed_time_means: [
                {time: 0, response: .995, expected_response: 1, role: 'completed', planned_replicate_count: null},
                {time: 1.25, response: .65 + index * .01, expected_response: .625, role: 'new', planned_replicate_count: 3},
            ]},
    }));
    schedules[2].recovery_error = {lower: [null, null, null], median: [-.1, -.2, null],
        upper: [null, null, null], available_counts: [1, 1, 0]};
    return {schema_version: 1, reference: {time: [0, 1, 2], central_response: [1, .7, .4]},
        schedule_data: {schedules: schedules.map((row, index) => ({id: row.id, name: row.name,
            kind: index === 1 ? 'suggested' : 'manual', points: []}))},
        recovery: {schema_version: 1, schedules, shared_example: {available: true,
            simulation_index: 0, study_number: 1, trajectory_draw_index: 8,
            reason: 'Study 1 is predetermined for every schedule, including failed fits.'},
            completed_observations: [
                {time: 0, response: .98, role: 'completed', response_kind: 'mean', observation_sigma: .03,
                    uncertainty_kind: 'SEM', reported_replicate_count: 9},
                {time: 0, response: 1.01, role: 'completed', response_kind: 'mean', observation_sigma: .04,
                    uncertainty_kind: 'SEM', reported_replicate_count: 4},
            ], fit_reported_intervals: {available: false, reason: 'Pointwise fit bounds are not retained.'}}};
}

function fixture({data = inspectionData(), encoded, initialSelection = {selectedIds: [], colors: {}},
    missingRenderer = false, update, summaryResult, exampleResult, integrated = false} = {}) {
    let holder;
    const attributeName = key => 'data-' + key.replace(/[A-Z]/g, letter => '-' + letter.toLowerCase());
    class Element {
        constructor(tag = 'div', attributes = {}) {
            this.tagName = tag.toUpperCase(); this.attributes = {...attributes}; this.children = [];
            this.parentElement = null; this.listeners = {}; this._text = '';
            this.hidden = false; this.disabled = false; this.checked = false; this.open = false; this.value = '';
            this.dataset = new Proxy({}, {get: (_, key) => this.attributes[attributeName(key)],
                set: (_, key, value) => { this.attributes[attributeName(key)] = String(value); return true; }});
            this.style = {values: {}, setProperty(key, value) { this.values[key] = value; }};
        }
        get ownerDocument() { return document; }
        get isConnected() { for (let node = this; node; node = node.parentElement) if (node === holder) return true; return false; }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
        set textContent(value) { this._text = String(value); this.children = []; }
        set innerHTML(_value) { throw new Error('Numerical tables must use safe text nodes.'); }
        append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } return children.at(-1); }
        appendChild(child) { return this.append(child); }
        replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
        getAttribute(key) { return this.attributes[key] ?? null; }
        setAttribute(key, value) { this.attributes[key] = String(value); }
        matches(selector) {
            const match = /^\[([^=\]]+)(?:="([^"]*)")?\]$/.exec(selector);
            return match ? this.attributes[match[1]] !== undefined &&
                (match[2] === undefined || this.attributes[match[1]] === match[2]) : this.tagName.toLowerCase() === selector;
        }
        querySelectorAll(selector) {
            const found = [];
            const visit = node => node.children.forEach(child => {
                if (selector.split(',').some(part => child.matches(part.trim()))) found.push(child);
                visit(child);
            });
            visit(this); return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        addEventListener(type, callback) { (this.listeners[type] ||= new Set()).add(callback); }
        removeEventListener(type, callback) { this.listeners[type]?.delete(callback); }
        dispatchEvent(event) {
            if (!event.target) event.target = this;
            for (const callback of this.listeners[event.type] || []) callback(event);
            if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
            return true;
        }
    }
    const document = {createElement: tag => new Element(tag), createDocumentFragment: () => new Element('fragment')};
    holder = new Element();
    const root = holder.append(new Element('div', {'data-planning-result': ''}));
    const add = (name, parent = root, tag = 'div', value = '') => parent.append(new Element(tag, {[name]: value}));
    const payload = add('data-planning-inspection', root, 'script');
    payload.textContent = encoded ?? JSON.stringify(data);
    const panel = add('data-recovery-panel');
    const view = add('data-recovery-view', panel, 'select'); view.value = 'response'; view.disabled = true;
    const empty = add('data-recovery-empty', panel, 'p');
    const status = add('data-recovery-status', panel, 'p');
    const rows = [], bands = [], swatches = [], availability = [], exampleRows = [], outcomes = [];
    for (const schedule of data.recovery.schedules) {
        const row = add('data-recovery-schedule', panel, 'div', schedule.id); rows.push(row);
        const band = add('data-recovery-band', row, 'input', schedule.id);
        band.checked = true; band.disabled = true; bands.push(band);
        swatches.push(add('data-recovery-swatch', row, 'span', schedule.id));
        availability.push(add('data-recovery-availability', row, 'span', schedule.id));
    }
    const plot = add('data-recovery-plot', panel); plot.attributes['data-design-plot'] = '';
    const target = add('data-design-plot-target', plot); target.hidden = true;
    add('data-design-plot-fallback', plot); add('data-design-plot-status', plot, 'p');
    const values = add('data-recovery-values', panel, 'details');
    const caption = add('data-recovery-values-caption', values, 'caption');
    const valuesBody = add('data-recovery-values-body', values, 'tbody');
    const example = add('data-recovery-example', panel, 'details');
    const exampleStatus = add('data-recovery-example-status', example, 'p');
    for (const schedule of data.recovery.schedules) {
        const row = add('data-recovery-example-schedule', example, 'p', schedule.id); exampleRows.push(row);
        outcomes.push(add('data-recovery-example-outcome', row, 'span', schedule.id));
    }
    const examplePlot = add('data-recovery-example-plot', example); examplePlot.attributes['data-design-plot'] = '';
    const exampleTarget = add('data-design-plot-target', examplePlot); exampleTarget.hidden = true;
    add('data-design-plot-fallback', examplePlot); add('data-design-plot-status', examplePlot, 'p');
    const exampleValues = add('data-recovery-example-values', example, 'details');
    const curvesBody = add('data-recovery-example-curves-body', exampleValues, 'tbody');
    const pointsBody = add('data-recovery-example-points-body', exampleValues, 'tbody');
    const scheduleControls = [];
    if (integrated) {
        for (const schedule of data.schedule_data.schedules) {
            const row = add('data-schedule-row', root, 'div', schedule.id);
            const control = add('data-schedule-select', row, 'input', schedule.id);
            control.disabled = true; scheduleControls.push(control);
            add('data-schedule-swatch', row, 'span', schedule.id);
            add('data-schedule-detail', root, 'div', schedule.id);
        }
        add('data-schedule-selection-status', root, 'p');
        const timeline = add('data-schedule-timeline'); timeline.attributes['data-design-plot'] = '';
        add('data-design-plot-target', timeline); add('data-design-plot-status', timeline);
        add('data-schedule-details', root, 'details');
        add('data-schedule-selection-empty'); add('data-schedule-timeline-help');
    }
    const calls = {summary: [], example: [], updates: [], purges: []};
    const builders = {
        summary(inspection, selection, options) {
            calls.summary.push(plain({inspection, selection, options}));
            return summaryResult === undefined ? {data: [], layout: {}, kind: 'summary'} : summaryResult;
        },
        example(inspection, selection) {
            calls.example.push(plain({inspection, selection}));
            return exampleResult === undefined ? {data: [], layout: {}, kind: 'example'} : exampleResult;
        },
    };
    const plots = {
        update(wrapper, spec) {
            calls.updates.push({wrapper, spec});
            const plotTarget = wrapper.querySelector('[data-design-plot-target]');
            plotTarget.hidden = false;
            return update ? update(wrapper, spec, plotTarget) : Promise.resolve();
        },
        purge(wrapper) { calls.purges.push(wrapper); },
    };
    const window = {DesignRecoveryFigure: builders, document,
        ...(integrated ? {DesignTimelineFigure: {build: () => ({data: [], layout: {}, kind: 'timeline'})}} :
            {DesignScheduleInspection: {getSelection: () => initialSelection}}),
        ...(missingRenderer ? {} : {DesignPlots: plots})};
    class CustomEvent {
        constructor(type, options) { this.type = type; Object.assign(this, options); }
    }
    const context = {window, document, CustomEvent};
    if (integrated) vm.runInNewContext(fs.readFileSync(path.join(__dirname,
        '../static/js/design-schedule-inspection.js'), 'utf8'), context);
    vm.runInNewContext(source(), context);
    const api = window.DesignRecoveryInspection;
    const selected = (ids, colors = {}) => root.dispatchEvent({type: 'design-schedule-selection-changed',
        detail: {selectedIds: ids, colors}, bubbles: true});
    const change = element => element.dispatchEvent({type: 'change', bubbles: true});
    const toggle = (element, open = true) => { element.open = open; element.dispatchEvent({type: 'toggle'}); };
    return {api, root, holder, payload, panel, view, empty, status, rows, bands, swatches, availability,
        plot, target, values, caption, valuesBody, example, exampleStatus, exampleRows, outcomes,
        examplePlot, exampleTarget, exampleValues, curvesBody, pointsBody, calls, selected, change, toggle, window, data,
        scheduleControls};
}

test('recovery uses V3 selection and colors without choosing another schedule', () => {
    const f = fixture(); f.api.initialize(f.holder);
    assert.equal(f.calls.summary.length, 0);
    f.selected(['schedule-0', 'schedule-1'], {'schedule-0': '#123456', 'schedule-1': '#abcdef'});
    const call = f.calls.summary.at(-1);
    assert.deepEqual(call.selection.selectedIds, ['schedule-0', 'schedule-1']);
    assert.equal(call.options.view, 'response');
    assert.equal(call.options.bands['schedule-0'], true);
    assert.equal(f.rows[0].hidden, false); assert.equal(f.rows[2].hidden, true);
    assert.equal(f.swatches[0].style.values['--schedule-color'], '#123456');
    assert.match(f.availability[0].textContent, /3\s*(?:\/|of)\s*4/);
    assert.equal(f.calls.example.length, 0, 'The collapsed example is not rendered.');
});

test('existing selection is adopted and duplicate or unknown IDs never exceed three schedules', () => {
    const f = fixture({initialSelection: {selectedIds: ['schedule-2'], colors: {'schedule-2': '#123456'}}});
    f.api.initialize(f.root);
    assert.deepEqual(f.calls.summary.at(-1).selection.selectedIds, ['schedule-2']);
    f.selected(['schedule-0', 'schedule-0', 'unknown', 'schedule-1', 'schedule-2', 'schedule-3']);
    const ids = f.calls.summary.at(-1).selection.selectedIds;
    assert.equal(ids.length, 3); assert.equal(new Set(ids).size, 3); assert.equal(ids.includes('unknown'), false);
});

test('response/error display and each band preference survive schedule changes', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0', 'schedule-1']);
    f.bands[0].checked = false; f.change(f.bands[0]);
    assert.equal(f.calls.summary.at(-1).options.bands['schedule-0'], false);
    f.view.value = 'error'; f.change(f.view);
    assert.equal(f.calls.summary.at(-1).options.view, 'error');
    f.selected(['schedule-1']); f.selected(['schedule-0', 'schedule-1']);
    assert.equal(f.view.value, 'error'); assert.equal(f.bands[0].checked, false);
    assert.equal(f.calls.summary.at(-1).options.bands['schedule-0'], false);
    assert.equal(f.bands[1].checked, true);
});

test('a sparse pointwise range disables its toggle only for the affected view', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-2']);
    assert.equal(f.bands[2].disabled, false);
    f.view.value = 'error'; f.change(f.view);
    assert.equal(f.bands[2].disabled, true);
    f.view.value = 'response'; f.change(f.view);
    assert.equal(f.bands[2].disabled, false); assert.equal(f.bands[2].checked, true);
});

test('zero successful fits suppress incidental retained summaries while preserving failure counts', () => {
    const data = inspectionData();
    data.recovery.schedules[0].n_successful = 0;
    data.recovery.schedules[0].n_failed = 4;
    // Keep incidental finite summary coordinates/counts to verify the success guard.
    const f = fixture({data}); f.api.initialize(f.root); f.selected(['schedule-0']);
    assert.equal(f.bands[0].disabled, true);
    assert.match(f.availability[0].textContent, /0\s*(?:\/|of)\s*4/);
    assert.match(f.availability[0].textContent, /4 failed/);
    f.toggle(f.values);
    assert.equal(f.valuesBody.children.length, 3);
    for (const row of f.valuesBody.children) {
        assert.deepEqual(row.children.slice(2, 5).map(cell => cell.textContent), ['—', '—', '—']);
        assert.equal(row.children[5].textContent, '0');
    }
});

test('the optional example is lazy, shares study one, and reports a failed fit without replacement', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0', 'schedule-1']);
    assert.equal(f.calls.example.length, 0);
    f.toggle(f.example);
    assert.equal(f.calls.example.length, 1);
    assert.equal(f.calls.example[0].inspection.recovery.shared_example.simulation_index, 0);
    assert.match(f.outcomes[1].textContent, /failed/i);
    assert.match(f.outcomes[1].textContent, /without replacement/);
    f.view.value = 'error'; f.change(f.view);
    const call = f.calls.example.at(-1);
    assert.equal(call.inspection.recovery.schedules[1].example.simulation_index, 0);
    assert.equal(call.inspection.recovery.schedules[1].example.fit_success, false);
});

test('unavailable shared examples retain their reason without constructing a figure', () => {
    const data = inspectionData(); data.recovery.shared_example.available = false;
    data.recovery.shared_example.reason = 'Shared trajectory selections do not match.';
    const f = fixture({data}); f.api.initialize(f.root); f.selected(['schedule-0']); f.toggle(f.example);
    assert.equal(f.calls.example.length, 0);
    assert.match(f.exampleStatus.textContent, /Shared trajectory selections do not match/);
    assert.equal(f.examplePlot.hidden, true);
});

test('numerical recovery values are lazy, view-specific, and independent of hidden bands', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0', 'schedule-2']);
    assert.equal(f.valuesBody.children.length, 0);
    f.toggle(f.values);
    assert.equal(f.valuesBody.children.length, 6);
    assert.match(f.valuesBody.textContent, /<img src=x> first schedule/);
    assert.match(f.valuesBody.textContent, /0\.8/);
    f.bands[0].checked = false; f.change(f.bands[0]);
    assert.match(f.valuesBody.textContent, /0\.8/, 'Band display does not delete numerical bounds.');
    f.view.value = 'error'; f.change(f.view);
    assert.match(f.valuesBody.textContent, /-0\.2/);
    assert.match(f.valuesBody.textContent, /—/);
    assert.match(f.caption.textContent, /error/i);
});

test('example tables retain new means and each original completed row once with its SEM/count', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0', 'schedule-1']); f.toggle(f.example);
    assert.equal(f.curvesBody.children.length, 0);
    f.toggle(f.exampleValues);
    assert.equal(f.curvesBody.children.length, 6);
    assert.equal(f.pointsBody.children.length, 4, 'Two new means plus two shared original completed rows.');
    assert.match(f.pointsBody.textContent, /0\.98/); assert.match(f.pointsBody.textContent, /1\.01/);
    assert.match(f.pointsBody.textContent, /SEM/); assert.match(f.pointsBody.textContent, /0\.03/);
    assert.match(f.pointsBody.textContent, /9/);
    assert.doesNotMatch(f.pointsBody.textContent, /0\.995/, 'Collapsed completed means must not duplicate original rows.');
    assert.match(f.curvesBody.textContent, /—/, 'The failed example remains unavailable.');
});

test('missing or failed interactive rendering opens and fills the numerical fallback', async () => {
    const missing = fixture({missingRenderer: true}); missing.api.initialize(missing.root); missing.selected(['schedule-0']);
    assert.equal(missing.values.open, true); assert.equal(missing.valuesBody.children.length, 3);
    missing.toggle(missing.example);
    assert.equal(missing.exampleValues.open, true); assert.equal(missing.curvesBody.children.length, 3);
    const failed = fixture({update: (_wrapper, _spec, target) => { target.hidden = true; return Promise.resolve(); }});
    failed.api.initialize(failed.root); failed.selected(['schedule-0']); await flush();
    assert.equal(failed.values.open, true); assert.equal(failed.valuesBody.children.length, 3);
});

test('clearing and restoring selection preserves view/bands while hiding empty plots and details', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0']);
    f.view.value = 'error'; f.change(f.view); f.bands[0].checked = false; f.change(f.bands[0]);
    f.selected([]);
    assert.equal(f.plot.hidden, true); assert.equal(f.values.hidden, true); assert.equal(f.empty.hidden, false);
    assert.equal(f.example.hidden, true);
    f.selected(['schedule-0']);
    assert.equal(f.values.hidden, false); assert.equal(f.example.hidden, false);
    assert.equal(f.view.value, 'error'); assert.equal(f.bands[0].checked, false);
});

test('disposal and later resolutions cannot reopen old fallbacks or respond to selection', async () => {
    let finish;
    const f = fixture({update: (_wrapper, _spec, target) => { target.hidden = true; return new Promise(resolve => { finish = resolve; }); }});
    f.api.initialize(f.root); f.selected(['schedule-0']);
    f.api.dispose(f.holder);
    const calls = f.calls.summary.length;
    finish(); await flush();
    assert.equal(f.values.open, false);
    f.selected(['schedule-1']); f.view.value = 'error'; f.change(f.view);
    assert.equal(f.calls.summary.length, calls);
    assert.equal(f.calls.purges.includes(f.plot), true); assert.equal(f.calls.purges.includes(f.examplePlot), true);
});

test('malformed data does not execute code and leaves controls unavailable', () => {
    for (const encoded of ['window.injected=true', 'null', '{}', '{"schema_version":2}']) {
        const f = fixture({encoded});
        assert.doesNotThrow(() => f.api.initialize(f.root));
        assert.match(f.status.textContent, /unavailable/i);
        assert.equal(f.view.disabled, true); assert.equal(f.bands.every(band => band.disabled), true);
        assert.equal(f.window.injected, undefined); assert.equal(f.calls.summary.length, 0);
    }
});

test('initialization is idempotent and unrelated result changes do not redraw recovery', () => {
    const f = fixture(); f.api.initialize(f.root); f.selected(['schedule-0']);
    const calls = f.calls.summary.length; f.api.initialize(f.holder);
    f.root.dispatchEvent({type: 'change', target: {dataset: {}}, bubbles: true});
    assert.equal(f.calls.summary.length, calls);
});

test('actual V3 selection initializes recovery colors and remains unchanged by recovery display controls', () => {
    const f = fixture({integrated: true});
    f.api.initialize(f.root);
    assert.equal(f.calls.summary.length, 0);
    f.window.DesignScheduleInspection.initialize(f.root);
    let selected = plain(f.window.DesignScheduleInspection.getSelection(f.root));
    assert.deepEqual(selected.selectedIds, ['schedule-1']);
    assert.deepEqual(f.calls.summary.at(-1).selection, selected);
    assert.equal(f.swatches[1].style.values['--schedule-color'], selected.colors['schedule-1']);
    f.scheduleControls[0].checked = true; f.change(f.scheduleControls[0]);
    selected = plain(f.window.DesignScheduleInspection.getSelection(f.root));
    assert.deepEqual(selected.selectedIds, ['schedule-0', 'schedule-1']);
    assert.deepEqual(f.calls.summary.at(-1).selection, selected);
    f.view.value = 'error'; f.change(f.view);
    f.bands[0].checked = false; f.change(f.bands[0]);
    assert.deepEqual(plain(f.window.DesignScheduleInspection.getSelection(f.root)), selected);
    assert.deepEqual(f.calls.summary.at(-1).selection, selected);
    assert.equal(f.calls.summary.at(-1).options.view, 'error');
    assert.equal(f.calls.summary.at(-1).options.bands['schedule-0'], false);
});
