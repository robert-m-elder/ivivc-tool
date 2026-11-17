'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/js/design-performance-inspection.js'), 'utf8');
const rendererSource = fs.readFileSync(path.join(__dirname, '../static/js/design-plots.js'), 'utf8');
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const settle = () => new Promise(resolve => setImmediate(resolve));

function data() {
    const labels = (title, primary, secondary) => ({title, primary_label: primary,
        secondary_label: secondary, y_label: title, help: 'Read counts with each summary.'});
    return {schema_version: 1, performance: {x_label: 'Sampling times', measures: [
        {id: 'irmse', label: 'IRMSE', kind: 'curve', views: {error: labels('Curve error', 'Mean IRMSE', 'P90 IRMSE')}},
        {id: 'endpoint-0', label: 'T50', kind: 'endpoint', views: {
            error: labels('T50 timing error', 'Median absolute error', 'P90 absolute error'),
            time: labels('Recovered T50', 'Median time', 'Central 95% range')}}],
    schedules: [0, 1, 2].map(index => ({id: 'schedule-' + index,
        name: index === 0 ? '<img src=x> schedule' : 'Schedule ' + index,
        kind: index === 2 ? 'Manual' : 'Suggested', x: 3 + index,
        n_simulations: 4, n_successful: 3, curve_interval_width: .5, interval_count: 3,
        measures: {irmse: {error: {primary: 4 - index, secondary: 9 - index, n_pairs: 3}},
            'endpoint-0': {
                error: {primary: 2 - index / 2, secondary: 6 - index, n_pairs: 2,
                    n_reached: 2, n_valid: 3, n_invalid: 0, n_failed: 1},
                time: {primary: 5 + index, lower: 3, upper: index === 1 ? null : 9,
                    primary_status: 'resolved', lower_status: 'resolved',
                    upper_status: index === 1 ? 'beyond_horizon' : 'resolved',
                    n_valid: 3, n_reached: 2, n_invalid: 0, n_failed: 1}}}}))}};
}

function fixture({payload = data(), missingRenderer = false, failEndpoint = false, sharedRenderer = false} = {}) {
    class Element {
        constructor(attributes = {}) {
            this.attributes = {...attributes}; this.children = []; this.listeners = {};
            this.parentElement = null; this.value = ''; this.hidden = false; this.disabled = true; this._text = '';
            this.dataset = new Proxy({}, {get: (_, key) => this.attributes['data-' + key.replace(/[A-Z]/g, c => '-' + c.toLowerCase())]});
        }
        append(child) { child.parentElement = this; this.children.push(child); return child; }
        matches(selector) { return this.attributes[selector.slice(1, -1)] !== undefined; }
        querySelectorAll(selector) {
            const found = [];
            const visit = element => element.children.forEach(child => {
                if (child.matches(selector)) found.push(child);
                visit(child);
            });
            visit(this); return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        setAttribute(key, value) { this.attributes[key] = String(value); }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
        set innerHTML(_) { throw new Error('Use safe text updates.'); }
        addEventListener(type, listener) { (this.listeners[type] ||= new Set()).add(listener); }
        removeEventListener(type, listener) { this.listeners[type]?.delete(listener); }
        dispatchEvent(event) {
            event.target ||= this;
            for (const listener of this.listeners[event.type] || []) listener(event);
            if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
        }
    }
    const holder = new Element();
    const root = holder.append(new Element({'data-planning-result': ''}));
    const add = (name, parent = root, value = '') => parent.append(new Element({[name]: value}));
    add('data-planning-inspection').textContent = JSON.stringify(payload);
    const panel = add('data-performance-inspection');
    const measure = add('data-performance-measure', panel), viewField = add('data-performance-view-field', panel);
    const view = add('data-performance-view', viewField), help = add('data-performance-help', panel);
    const plot = add('data-performance-plot', panel), target = add('data-design-plot-target', plot);
    plot.setAttribute('data-design-plot', '');
    target.setAttribute('id', 'planning-performance-plot-interactive');
    target.hidden = true;
    const fallback = add('data-design-plot-fallback', plot); fallback.textContent = 'Original IRMSE SVG';
    const status = add('data-design-plot-status', plot), table = add('data-performance-table', panel);
    const caption = add('data-performance-caption', table);
    const headings = ['primary', 'secondary', 'availability'].map(name => add('data-performance-' + name + '-heading', table));
    const cells = payload.performance.schedules.map(schedule => {
        const row = add('data-schedule-row', table, schedule.id);
        return ['primary', 'secondary', 'availability'].map(name => add('data-performance-' + name, row));
    });
    const calls = {updates: [], purges: [], renders: [], warnings: []};
    const window = {};
    if (sharedRenderer) {
        const draw = (method, target, data, layout) => {
            calls.renders.push({method, data, layout});
            if (failEndpoint && /T50/.test(layout.title.text)) {
                return Promise.reject(new Error('Simulated Plotly draw failure'));
            }
            return Promise.resolve();
        };
        window.Plotly = {
            react: (...args) => draw('react', ...args),
            newPlot: (...args) => draw('newPlot', ...args),
            purge: target => calls.purges.push(target),
            Plots: {resize: () => Promise.resolve()},
        };
        vm.runInNewContext(rendererSource, {window, console: {warn: (...args) => calls.warnings.push(args)}});
    } else if (!missingRenderer) window.DesignPlots = {
        update(wrapper, spec) {
            calls.updates.push({wrapper, spec});
            if (failEndpoint && /T50/.test(spec.layout.title.text)) {
                target.hidden = true;
                return Promise.resolve({ok: false});
            }
            target.hidden = false;
            fallback.hidden = true;
            status.textContent = spec.empty_message || 'Interactive plot ready.';
            return Promise.resolve({ok: true, empty: !spec.data.length});
        },
        purge(wrapper) { calls.purges.push(wrapper); },
    };
    vm.runInNewContext(source, {window});
    const api = window.DesignPerformanceInspection;
    api.initialize(holder);
    return {root, holder, panel, measure, view, viewField, help, plot, target, fallback, status, table,
        caption, headings, cells, calls, api,
        change(control, value) { control.value = value; root.dispatchEvent({type: 'change', target: control}); }};
}

test('shared measure selectors update the numerical table and plot without changing retained results', () => {
    const payload = data(), before = JSON.stringify(payload), f = fixture({payload});
    assert.equal(f.cells[0][0].textContent, '4');
    assert.equal(f.viewField.hidden, true);
    assert.equal(f.calls.updates.at(-1).spec.layout.height, 460);
    f.change(f.measure, 'endpoint-0');
    assert.equal(f.cells[0][0].textContent, '2');
    assert.equal(f.headings[0].textContent, 'Median absolute error');
    assert.match(f.cells[0][2].textContent, /2 \/ 4 finite pairs; 2 \/ 4 reached/);
    assert.equal(f.viewField.hidden, false);
    assert.match(f.calls.updates.at(-1).spec.layout.title.text, /T50 timing error/);
    assert.equal(JSON.stringify(payload), before);
    assert.doesNotMatch(f.fallback.textContent, /Original IRMSE/);
    f.change(f.view, 'time');
    assert.equal(f.cells[0][0].textContent, '5');
    assert.match(f.caption.textContent, /recovered endpoint times/);
    assert.match(f.target.attributes['aria-label'], /Recovered T50/);
    f.change(f.measure, 'irmse');
    assert.equal(f.cells[0][0].textContent, '4');
    assert.equal(f.viewField.hidden, true);
});

test('endpoint ranges preserve beyond-horizon statuses instead of drawing a capped bound', () => {
    const f = fixture(); f.change(f.measure, 'endpoint-0'); f.change(f.view, 'time');
    assert.match(f.cells[1][1].textContent, /Beyond the assessment horizon/);
    const spec = f.calls.updates.at(-1).spec;
    const suggested = spec.data.find(trace => trace.meta.schedule_ids.length === 2);
    assert.deepEqual(Array.from(suggested.error_y.array), [4, null]);
    assert.match(suggested.customdata[1], /Beyond the assessment horizon/);
    assert.match(suggested.customdata[0], /&lt;img/);
    assert.equal(JSON.stringify(spec).includes('Infinity'), false);
});

test('schedule-type colors remain stable and selection does not duplicate or redraw points', () => {
    const payload = data(); payload.performance.schedules[1].kind = 'Evenly spaced';
    const f = fixture({payload}); f.change(f.measure, 'endpoint-0');
    const count = f.calls.updates.length;
    f.root.dispatchEvent({type: 'design-schedule-selection-changed', detail: {
        selectedIds: ['schedule-1'], colors: {'schedule-1': '#b34f00'}}});
    const spec = f.calls.updates.at(-1).spec;
    assert.equal(f.calls.updates.length, count);
    assert.equal(spec.data.length, 6);
    assert.deepEqual(Array.from(new Set(spec.data.map(trace => trace.line.color))),
        ['#24535c', '#a75328', '#6f4a92']);
    assert.ok(spec.data.every(trace => !trace.meta.selected));
    assert.ok(spec.data.some(trace => trace.meta.schedule_ids.includes('schedule-0')));
    assert.ok(spec.data.some(trace => trace.mode === 'markers' && trace.meta.schedule_ids.includes('schedule-2')));
    f.root.dispatchEvent({type: 'design-schedule-selection-changed', detail: {selectedIds: [], colors: {}}});
    assert.equal(f.calls.updates.length, count);
});

test('all noncrossing or invalid outcomes remain explicit when no finite median can be plotted', () => {
    const payload = data();
    payload.performance.schedules.forEach(schedule => {
        const value = schedule.measures['endpoint-0'].time;
        Object.assign(value, {primary: null, primary_status: 'beyond_horizon', upper: null, upper_status: 'beyond_horizon'});
    });
    const f = fixture({payload}); f.change(f.measure, 'endpoint-0'); f.change(f.view, 'time');
    assert.equal(f.target.hidden, false);
    assert.match(f.cells[0][0].textContent, /Beyond the assessment horizon/);
    assert.match(f.status.textContent, /No finite values to plot/);
    assert.match(f.help.textContent, /No finite median time is plotted/);
    assert.equal(f.calls.purges.length, 0);
    assert.equal(f.calls.updates.at(-1).spec.data.length, 0);
    assert.match(f.calls.updates.at(-1).spec.layout.annotations[0].text, /No finite values/);
    f.change(f.measure, 'irmse');
    assert.equal(f.target.hidden, false);
    assert.match(f.status.textContent, /Interactive plot ready/);
    assert.ok(f.calls.updates.at(-1).spec.data.length > 0);
});

test('numerical controls survive plotting failure and returning to IRMSE recovers', async () => {
    const f = fixture({missingRenderer: true}); f.change(f.measure, 'endpoint-0'); f.change(f.view, 'time');
    assert.equal(f.cells[0][0].textContent, '5');
    assert.match(f.status.textContent, /Interactive plot unavailable/);
    assert.doesNotMatch(f.fallback.textContent, /Original IRMSE/);
    const payload = data(); payload.performance.measures = payload.performance.measures.slice(0, 1);
    const only = fixture({payload});
    assert.equal(only.viewField.hidden, true);
    assert.equal(only.cells[0][0].textContent, '4');
    const failed = fixture({failEndpoint: true, sharedRenderer: true});
    await settle();
    failed.change(failed.measure, 'endpoint-0');
    await settle();
    assert.equal(failed.cells[0][0].textContent, '2');
    assert.match(failed.status.textContent, /Interactive plot unavailable/);
    assert.match(failed.calls.warnings[0][2].message, /Simulated Plotly draw failure/);
    assert.equal(failed.calls.warnings[0][1].view, 'performance-endpoint-0-error');
    failed.change(failed.measure, 'irmse');
    await settle();
    assert.equal(failed.target.hidden, false);
    assert.match(failed.status.textContent, /Interactive plot ready/);
    assert.ok(failed.calls.renders.every(call => call.method === 'newPlot'));
});

test('performance controller and shared renderer switch through finite, empty, and rapid views', async () => {
    const payload = data();
    payload.performance.schedules.forEach(schedule => Object.assign(schedule.measures['endpoint-0'].time,
        {primary: null, primary_status: 'beyond_horizon', upper: null, upper_status: 'beyond_horizon'}));
    const f = fixture({payload, sharedRenderer: true});
    await settle();
    assert.match(f.calls.renders.at(-1).layout.title.text, /Curve error/);
    f.change(f.measure, 'endpoint-0');
    await settle();
    assert.match(f.calls.renders.at(-1).layout.title.text, /T50 timing error/);
    f.change(f.view, 'time');
    await settle();
    assert.equal(f.target.hidden, false);
    assert.equal(f.calls.renders.at(-1).data.length, 0);
    assert.match(f.status.textContent, /No finite values/);
    assert.doesNotMatch(f.status.textContent, /unavailable\./);
    f.change(f.measure, 'irmse');
    f.change(f.measure, 'endpoint-0');
    f.change(f.measure, 'irmse');
    await settle();
    assert.equal(f.target.hidden, false);
    assert.equal(f.cells[0][0].textContent, '4');
    assert.match(f.status.textContent, /Interactive plot ready/);
    assert.match(f.calls.renders.at(-1).layout.title.text, /Curve error/);
    assert.ok(f.calls.renders.every(call => call.method === 'newPlot'));
    assert.equal(f.calls.warnings.length, 0);
});

test('initialization and disposal are idempotent and detached results ignore later selection', async () => {
    const f = fixture();
    const count = f.calls.updates.length;
    f.api.initialize(f.holder);
    assert.equal(f.calls.updates.length, count);
    f.api.dispose(f.holder); f.api.dispose(f.holder);
    f.root.dispatchEvent({type: 'design-schedule-selection-changed', detail: {selectedIds: [], colors: {}}});
    f.change(f.measure, 'endpoint-0');
    await flush();
    assert.equal(f.calls.updates.length, count);
    assert.equal(f.calls.purges.length, 1);
});
