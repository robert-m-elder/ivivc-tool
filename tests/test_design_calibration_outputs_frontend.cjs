'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/js/design-calibration-outputs.js'), 'utf8');
const rendererSource = fs.readFileSync(path.join(__dirname, '../static/js/design-plots.js'), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));
const settle = () => new Promise(resolve => setImmediate(resolve));

function payload() {
    const value = () => ({x: [0, 1, 2], y: [1, .7, .4], lower: [.8, .5, .2], upper: [1.2, .9, .6],
        counts: [3, 3, 3], truth: {x: [0, 1, 2], y: [1, .6, .2], counts: [4, 4, 4]},
        example: {x: [0, 1, 2], y: [1, .65, .3], lower: [.9, .55, .2], upper: [1.1, .75, .4],
            point_counts: [1, 1, 1], interval_counts: [1, 1, 1],
            truth: {x: [0, 1, 2], y: [1, .6, .2], counts: [1, 1, 1]}, message: 'Study 1 is available.'}});
    const meta = (title, axis = 'y') => ({title, label: title, x_label: 'Input time', y_label: 'Output response',
        range_axis: axis, available: true, connect_points: true, help: 'Across-study ranges; read counts.'});
    const candidates = () => [0, 1].map(index => ({id: 'calibration-design-' + index,
        name: index ? 'Second' : '<b>First</b>', n_simulations: 4, n_fit_successful: 3,
        views: {native: value(), response: value()}}));
    return {schema_version: 1, outputs: {
        relationship: {title: 'IVIVC relationship', quantity: 'value_ratio', scalar: false,
            views: {native: meta('Response ratio')}, candidates: candidates()},
        prediction: {title: 'Target predictions', quantity: 'in_vivo_response', scalar: false,
            views: {native: meta('Prediction output'), response: meta('Response versus in vivo time')}, candidates: candidates()},
    }};
}

function fixture({data = payload(), missingRenderer = false, sharedRenderer = false} = {}) {
    let document;
    class Element {
        constructor(tag = 'div', attrs = {}) {
            this.tagName = tag; this.attributes = {...attrs}; this.children = []; this.parentElement = null;
            this.listeners = {}; this.hidden = false; this.disabled = true; this.open = false; this.value = attrs.value || ''; this._text = '';
            this.dataset = new Proxy({}, {get: (_, key) => this.attributes['data-' + key.replace(/[A-Z]/g, c => '-' + c.toLowerCase())]});
            this.ownerDocument = document;
        }
        append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); return children.at(-1); }
        replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
        set textContent(value) { this._text = String(value); this.children = []; }
        set innerHTML(_) { throw new Error('Use textContent for output labels.'); }
        setAttribute(key, value) { this.attributes[key] = String(value); }
        matches(selector) { return selector[0] === '[' ? this.attributes[selector.slice(1, -1)] !== undefined : this.tagName === selector; }
        querySelectorAll(selector) {
            const found = []; const visit = node => node.children.forEach(child => {
                if (child.matches(selector)) found.push(child); visit(child);
            }); visit(this); return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        addEventListener(type, listener) { (this.listeners[type] ||= new Set()).add(listener); }
        removeEventListener(type, listener) { this.listeners[type]?.delete(listener); }
        dispatchEvent(event) {
            event.target ||= this; for (const listener of this.listeners[event.type] || []) listener(event);
            if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
        }
    }
    document = {createElement: tag => new Element(tag)};
    const holder = new Element(), root = holder.append(new Element('div', {'data-calibration-result': ''}));
    const add = (key, parent = root, tag = 'div', value = '') => parent.append(new Element(tag, {[key]: value}));
    add('data-calibration-inspection', root, 'script').textContent = JSON.stringify(data);
    const panels = {};
    for (const [key, output] of Object.entries(data.outputs)) {
        const element = add('data-calibration-output', root, 'section', key);
        const view = key === 'prediction' ? add('data-calibration-output-view', element, 'select') : null;
        if (view) for (const name of Object.keys(output.views)) view.append(new Element('option', {value: name}));
        const status = add('data-calibration-output-status', element, 'p');
        const help = add('data-calibration-output-help', element, 'p');
        const wrapper = name => {
            const plot = add(name, element); plot.setAttribute('data-design-plot', '');
            const target = add('data-design-plot-target', plot); target.hidden = true;
            add('data-design-plot-fallback', plot); add('data-design-plot-status', plot, 'p'); return plot;
        };
        const plot = wrapper('data-calibration-output-plot');
        const values = add('data-calibration-output-values', element, 'details');
        const body = add('data-calibration-output-values-body', values, 'tbody');
        add('data-calibration-output-values-caption', values, 'caption');
        const headings = ['x', 'y'].map(axis => add('data-calibration-output-' + axis + '-heading', values, 'th'));
        const example = add('data-calibration-output-example', element, 'details');
        const exampleStatus = add('data-calibration-output-example-status', example, 'p');
        const examplePlot = wrapper('data-calibration-output-example-plot');
        const exampleBody = add('data-calibration-output-example-values-body', example, 'tbody');
        ['x', 'y'].forEach(axis => add('data-calibration-output-example-' + axis + '-heading', example, 'th'));
        panels[key] = {element, view, status, help, plot, values, body, headings, example, examplePlot, exampleStatus, exampleBody};
    }
    const calls = {updates: [], purges: [], renders: [], warnings: [], failNext: false};
    const window = {DesignCalibrationRecovery: {getSelection: () => ({selectedIds: ['calibration-design-0'],
        colors: {'calibration-design-0': '#24535c'}})}};
    if (sharedRenderer) {
        const draw = (method, target, data, layout) => {
            calls.renders.push({method, target, data, layout});
            if (calls.failNext) { calls.failNext = false; return Promise.reject(new Error('Simulated Plotly failure')); }
            return Promise.resolve();
        };
        window.Plotly = {react: (...args) => draw('react', ...args), newPlot: (...args) => draw('newPlot', ...args),
            purge: target => calls.purges.push(target), Plots: {resize: () => Promise.resolve()}};
        vm.runInNewContext(rendererSource, {window, console: {warn: (...args) => calls.warnings.push(args)}});
        const renderer = window.DesignPlots;
        window.DesignPlots = {...renderer, update(wrapper, spec) { calls.updates.push({wrapper, spec}); return renderer.update(wrapper, spec); }};
    } else if (!missingRenderer) window.DesignPlots = {
        update(wrapper, spec) { calls.updates.push({wrapper, spec}); wrapper.querySelector('[data-design-plot-target]').hidden = false; return Promise.resolve({ok: true}); },
        purge: wrapper => calls.purges.push(wrapper),
    };
    vm.runInNewContext(source, {window, document});
    const api = window.DesignCalibrationOutputs; api.initialize(holder);
    return {root, holder, panels, calls, api,
        latest(key, example = false) { return calls.updates.filter(call => call.wrapper === panels[key][example ? 'examplePlot' : 'plot']).at(-1).spec; },
        change(key, value) { const panel = panels[key]; panel.view.value = value; panel.element.dispatchEvent({type: 'change', target: panel.view}); },
        toggle(key, example = false) { const control = panels[key][example ? 'example' : 'values']; control.open = true; control.dispatchEvent({type: 'toggle'}); },
        select(ids, colors) { root.dispatchEvent({type: 'design-calibration-selection-changed', detail: {selectedIds: ids, colors}}); }};
}

test('relationship and target overlays preserve source data and group each curve with its empirical range', () => {
    const data = payload(), before = JSON.stringify(data), f = fixture({data});
    assert.equal(f.panels.prediction.view.value, 'response');
    assert.equal(f.panels.relationship.view, null);
    assert.ok(f.latest('relationship').data.some(trace => trace.meta.kind === 'center'));
    const spec = f.latest('prediction'), group = spec.data.filter(trace => trace.legendgroup === 'output-calibration-design-0');
    assert.equal(group.length, 3);
    assert.equal(group.filter(trace => trace.showlegend).length, 1);
    assert.equal(group.find(trace => trace.showlegend).name, '&lt;b&gt;First&lt;/b&gt;');
    assert.equal(spec.layout.legend.groupclick, 'togglegroup');
    assert.ok(spec.data.every(trace => /^[A-Za-z0-9-]+$/.test(trace.uid)));
    assert.equal(JSON.stringify(data), before);
});

test('time-method response ranges are horizontal and nonmonotonic target levels stay unjoined', () => {
    const data = payload(), output = data.outputs.prediction;
    Object.assign(output.views.response, {range_axis: 'x', x_label: 'Predicted in vivo time', y_label: 'Target response'});
    for (const candidate of output.candidates) Object.assign(candidate.views.response,
        {x: [4, 6, 8], y: [1, .5, .2], lower: [3, 5, 7], upper: [5, 7, 9]});
    const f = fixture({data}), spec = f.latest('prediction');
    assert.ok(spec.data.some(trace => trace.fill === 'tonextx'));
    assert.equal(spec.data.some(trace => trace.fill === 'tonexty'), false);
    assert.equal(f.panels.prediction.headings[0].textContent, 'Predicted in vivo time');
    output.views.response.connect_points = false;
    for (const candidate of output.candidates) candidate.views.response.y = [1, .2, .5];
    const nonmonotonic = fixture({data}), center = nonmonotonic.latest('prediction').data.find(trace => trace.meta.kind === 'center');
    assert.equal(center.mode, 'markers');
    assert.ok(center.error_x);
    assert.deepEqual(Array.from(center.y), [1, .2, .5]);
    assert.equal(nonmonotonic.latest('prediction').data.some(trace => trace.fill), false);
    assert.match(nonmonotonic.panels.prediction.help.textContent, /nonmonotonic/);
});

test('constant-ratio relationships use one categorical design point and interval per candidate', () => {
    const data = payload(), output = data.outputs.relationship;
    output.scalar = output.views.native.scalar = true; output.views.native.x_label = 'Design';
    for (const candidate of output.candidates) Object.assign(candidate.views.native,
        {x: [1], y: [2], lower: [1.5], upper: [2.5], counts: [3], truth: {x: [1], y: [2], counts: [4]}});
    const f = fixture({data});
    f.select(['calibration-design-0', 'calibration-design-1'], {'calibration-design-0': '#24535c', 'calibration-design-1': '#b34f00'});
    const spec = f.latest('relationship'), centers = spec.data.filter(trace => trace.meta.kind === 'center');
    assert.equal(spec.layout.xaxis.type, 'category');
    assert.equal(centers.length, 2);
    assert.ok(centers.every(trace => trace.mode === 'markers' && trace.error_y));
    assert.equal(spec.data.filter(trace => trace.meta.kind === 'truth').length, 1);
});

test('study 1 failures stay failed and original fitted intervals may exclude their estimates', () => {
    const data = payload(), failed = data.outputs.relationship.candidates[0].views.native.example;
    Object.assign(failed, {y: [9, 9, 9], point_counts: [0, 0, 0], interval_counts: [0, 0, 0], message: 'Study 1 failed; no replacement.'});
    const f = fixture({data}); f.toggle('relationship', true);
    assert.equal(f.latest('relationship', true).data.some(trace => trace.meta.kind === 'center'), false);
    assert.match(f.panels.relationship.exampleStatus.textContent, /Study 1 failed/);
    const response = data.outputs.prediction;
    response.views.response.connect_points = false;
    Object.assign(response.candidates[0].views.response.example,
        {x: [1], y: [1], lower: [10], upper: [12], point_counts: [1], interval_counts: [1]});
    const outside = fixture({data}); outside.toggle('prediction', true);
    const spec = outside.latest('prediction', true);
    const limits = spec.data.find(trace => trace.meta.kind === 'interval_limits');
    assert.deepEqual(Array.from(limits.y), [10, 12, null]);
    assert.equal(spec.data.find(trace => trace.meta.kind === 'center').error_y, undefined);
    assert.match(outside.panels.prediction.exampleBody.textContent, /10.*12/);
});

test('shared selection updates colors across panels and an empty selection stays a usable empty view', () => {
    const f = fixture(); f.change('prediction', 'native');
    f.select(['calibration-design-1'], {'calibration-design-1': '#7550a2'});
    for (const key of ['relationship', 'prediction']) {
        const centers = f.latest(key).data.filter(trace => trace.meta.kind === 'center');
        assert.equal(centers.length, 1); assert.equal(centers[0].line.color, '#7550a2');
    }
    assert.equal(f.panels.prediction.view.value, 'native');
    f.select([], {});
    assert.equal(f.latest('prediction').data.length, 0);
    assert.match(f.latest('prediction').empty_message, /Choose a design/);
});

test('numerical fallbacks, unavailable response view, and controller cleanup retain usable results', async () => {
    const data = payload(); data.outputs.prediction.views.response.available = false;
    data.outputs.prediction.views.response.help = 'Cannot verify a shared fixed target coordinate.';
    const f = fixture({data, missingRenderer: true}); await settle();
    assert.equal(f.panels.prediction.view.value, 'native');
    assert.equal(f.panels.prediction.values.open, true);
    assert.equal(f.panels.prediction.body.children.length, 3);
    assert.match(f.panels.prediction.help.textContent, /Cannot verify a shared fixed target coordinate/);
    const normal = fixture(), count = normal.calls.updates.length;
    normal.api.initialize(normal.holder); assert.equal(normal.calls.updates.length, count);
    normal.api.dispose(normal.holder); normal.api.dispose(normal.holder);
    normal.change('prediction', 'native'); normal.select([], {});
    assert.equal(normal.calls.updates.length, count);
    assert.equal(normal.calls.purges.length, 4);
});

test('output controller and shared renderer recover after empty and failed views with rapid switches', async () => {
    const data = payload();
    for (const candidate of data.outputs.prediction.candidates) Object.assign(candidate.views.response,
        {y: [null, null, null], counts: [0, 0, 0], truth: null});
    const f = fixture({data, sharedRenderer: true}); await settle();
    assert.match(f.panels.prediction.plot.querySelector('[data-design-plot-status]').textContent, /No selected estimates/);
    f.calls.failNext = true; f.change('prediction', 'native'); await settle();
    assert.equal(f.panels.prediction.values.open, true);
    assert.equal(f.calls.warnings.length, 1);
    f.change('prediction', 'response'); f.change('prediction', 'native'); await settle();
    assert.equal(f.panels.prediction.plot.querySelector('[data-design-plot-target]').hidden, false);
    assert.match(f.panels.prediction.plot.querySelector('[data-design-plot-status]').textContent, /Interactive plot ready/);
    assert.ok(f.calls.renders.every(call => call.method === 'newPlot'));
});
