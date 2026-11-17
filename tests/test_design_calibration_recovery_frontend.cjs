'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/js/design-calibration-recovery.js'), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));
const settle = () => new Promise(resolve => setImmediate(resolve));

function payload() {
    const values = () => ({median: [1, .7, .4], lower: [.8, .5, .2], upper: [1.2, .9, .6],
        available_counts: [3, 3, 3], n_curves: 3,
        example: {fit_available: true, curve: [1, .65, .3], truth: [1, .6, .2],
            new_points: [{time: 0, response: 1.1}, {time: 2, response: .3}], message: 'Study 1 fit available.'}});
    return {schema_version: 1, supported: true, approach_id: 'approach2', default_environment: 'in_vivo',
        example: {study_number: 1, available: true, message: 'Study 1 is predetermined.'},
        environments: {
            in_vivo: {label: 'In vivo', time: [0, 1, 2], fixed: false, reference: {central: [1, .6, .2]}, existing_points: []},
            in_vitro: {label: 'In vitro', time: [0, 1, 2], fixed: true, fixed_curve_shared: true,
                fixed_curve: [1, .8, .6], reference: {central: [1, .8, .6]}, existing_points: [{time: 0, response: 1}]},
        }, candidates: Array.from({length: 4}, (_, index) => ({id: 'calibration-design-' + index,
            name: index === 0 ? '<b>First</b>' : 'Design ' + index, n_simulations: 4, n_fit_successful: 3,
            sides: {in_vitro: values(), in_vivo: values()}}))};
}

function fixture({data = payload(), missingRenderer = false} = {}) {
    let document;
    class Element {
        constructor(tag = 'div', attrs = {}) {
            this.tagName = tag; this.attributes = {...attrs}; this.children = []; this.parentElement = null;
            this.listeners = {}; this.hidden = false; this.disabled = true; this.checked = false; this.open = false;
            this.value = ''; this._text = '';
            this.dataset = new Proxy({}, {get: (_, key) => this.attributes['data-' + key.replace(/[A-Z]/g, c => '-' + c.toLowerCase())]});
            this.style = {values: {}, setProperty(key, value) { this.values[key] = value; }};
            this.ownerDocument = document;
        }
        append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); return children.at(-1); }
        replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
        set textContent(value) { this._text = String(value); this.children = []; }
        set innerHTML(_) { throw new Error('Use safe text updates.'); }
        setAttribute(key, value) { this.attributes[key] = String(value); }
        matches(selector) { return selector[0] === '[' ? this.attributes[selector.slice(1, -1)] !== undefined : this.tagName === selector; }
        querySelectorAll(selector) {
            const found = [];
            const visit = node => node.children.forEach(child => { if (child.matches(selector)) found.push(child); visit(child); });
            visit(this); return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        addEventListener(type, listener) { (this.listeners[type] ||= new Set()).add(listener); }
        removeEventListener(type, listener) { this.listeners[type]?.delete(listener); }
        dispatchEvent(event) {
            event.target ||= this;
            for (const listener of this.listeners[event.type] || []) listener(event);
            if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
        }
    }
    document = {createElement: tag => new Element(tag)};
    const holder = new Element(), root = holder.append(new Element('div', {'data-calibration-result': ''}));
    const add = (key, parent = root, tag = 'div', value = '') => parent.append(new Element(tag, {[key]: value}));
    add('data-calibration-inspection', root, 'script').textContent = JSON.stringify(data);
    const panel = add('data-calibration-recovery');
    const controls = [], swatches = [], available = [];
    data.candidates.forEach(candidate => {
        const control = add('data-calibration-select', root, 'input', candidate.id); controls.push(control);
        swatches.push(add('data-calibration-swatch', root, 'span', candidate.id));
        available.push(add('data-calibration-availability', panel, 'p', candidate.id));
    });
    const environment = add('data-calibration-environment', panel, 'select');
    const selectionStatus = add('data-calibration-selection-status', panel, 'p');
    const status = add('data-calibration-recovery-status', panel, 'p');
    add('data-calibration-recovery-empty', panel, 'p');
    const wrapper = name => {
        const plot = add(name, panel); plot.setAttribute('data-design-plot', '');
        const target = add('data-design-plot-target', plot); target.hidden = true;
        add('data-design-plot-fallback', plot); add('data-design-plot-status', plot, 'p');
        return plot;
    };
    const plot = wrapper('data-calibration-recovery-plot');
    const values = add('data-calibration-values', panel, 'details');
    add('data-calibration-values-caption', values, 'caption');
    const valuesBody = add('data-calibration-values-body', values, 'tbody');
    const example = add('data-calibration-example', panel, 'details');
    const exampleStatus = add('data-calibration-example-status', example, 'p');
    const examplePlot = wrapper('data-calibration-example-plot');
    const exampleBody = add('data-calibration-example-values-body', example, 'tbody');
    const calls = {updates: [], purges: [], selections: []};
    root.addEventListener('design-calibration-selection-changed', event => calls.selections.push(plain(event.detail)));
    const window = missingRenderer ? {} : {DesignPlots: {
        update(wrapper, spec) { calls.updates.push({wrapper, spec}); wrapper.querySelector('[data-design-plot-target]').hidden = false; return Promise.resolve({ok: true}); },
        purge(wrapper) { calls.purges.push(wrapper); },
    }};
    vm.runInNewContext(source, {window, document, CustomEvent: class {constructor(type, init) {this.type = type; Object.assign(this, init);}}});
    const api = window.DesignCalibrationRecovery;
    api.initialize(holder);
    return {holder, root, environment, controls, swatches, available, plot, values, valuesBody, example, exampleStatus, examplePlot,
        exampleBody, status, selectionStatus, calls, api,
        change(control, value) { if (typeof value === 'boolean') control.checked = value; else control.value = value; root.dispatchEvent({type: 'change', target: control}); },
        toggle(control) { control.open = true; control.dispatchEvent({type: 'toggle'}); }};
}

test('median and range traces form one safe legend group per selected calibration design', () => {
    const data = payload(), before = JSON.stringify(data), f = fixture({data});
    const spec = f.calls.updates[0].spec;
    assert.equal(f.environment.value, 'in_vivo');
    const group = spec.data.filter(trace => trace.legendgroup === 'calibration-calibration-design-0');
    assert.equal(group.length, 3);
    assert.equal(group.filter(trace => trace.showlegend).length, 1);
    assert.equal(group.find(trace => trace.showlegend).name, '&lt;b&gt;First&lt;/b&gt;');
    assert.equal(spec.layout.legend.groupclick, 'togglegroup');
    assert.ok(spec.data.every(trace => /^[a-zA-Z0-9-]+$/.test(trace.uid)));
    assert.match(spec.render_key, /summary-in_vivo/);
    assert.equal(JSON.stringify(data), before);
});

test('selection is capped at three and colors remain stable across environment changes and reselection', () => {
    const f = fixture();
    f.change(f.controls[1], true); f.change(f.controls[2], true);
    assert.equal(f.controls[3].disabled, true);
    const original = plain(f.api.getSelection(f.root));
    f.change(f.environment, 'in_vitro');
    assert.deepEqual(plain(f.api.getSelection(f.root)), original);
    f.change(f.controls[1], false); f.change(f.controls[1], true);
    assert.deepEqual(plain(f.api.getSelection(f.root)), original);
    assert.match(f.selectionStatus.textContent, /3 of 3/);
    assert.equal(f.swatches[1].style.values['--schedule-color'], original.colors['calibration-design-1']);
});

test('fixed calibration curve and existing example observations appear once', () => {
    const f = fixture(); f.change(f.controls[1], true); f.change(f.environment, 'in_vitro');
    let spec = f.calls.updates.at(-1).spec;
    assert.equal(spec.data.filter(trace => trace.uid === 'calibration-fixed-fit').length, 1);
    assert.equal(spec.data.some(trace => trace.fill), false);
    f.toggle(f.values);
    assert.equal(f.valuesBody.children.length, 3);
    assert.match(f.valuesBody.textContent, /Shared fixed calibration fit/);
    f.toggle(f.example); spec = f.calls.updates.at(-1).spec;
    assert.equal(spec.data.filter(trace => trace.uid === 'calibration-existing').length, 1);
    assert.equal(spec.data.filter(trace => trace.uid === 'calibration-fixed-fit').length, 1);
    const data = payload();
    Object.assign(data.candidates[0].sides.in_vitro,
        {n_curves: 0, median: [null, null, null], lower: [null, null, null], upper: [null, null, null], available_counts: [0, 0, 0]});
    const failed = fixture({data}); failed.change(failed.environment, 'in_vitro'); failed.toggle(failed.values);
    assert.equal(failed.calls.updates.at(-1).spec.data.some(trace => trace.uid === 'calibration-fixed-fit'), false);
    assert.match(failed.status.textContent, /No usable reconstructed curves/);
    assert.equal(failed.valuesBody.textContent.includes('Shared fixed calibration fit'), false);
});

test('failed study 1 remains failed while observations and matched truth remain visible', () => {
    const data = payload();
    data.candidates[0].sides.in_vivo.example = {fit_available: false, curve: [9, 9, 9], truth: [1, .6, .2],
        new_points: [{time: 1, response: .7}], message: 'Fit failed in study 1; no replacement.'};
    const f = fixture({data}); f.toggle(f.example);
    const spec = f.calls.updates.at(-1).spec;
    assert.equal(spec.data.some(trace => trace.uid === 'calibration-design-0-example-fit'), false);
    assert.ok(spec.data.some(trace => trace.uid === 'calibration-design-0-example-new'));
    assert.ok(spec.data.some(trace => trace.uid === 'calibration-example-truth'));
    assert.match(f.exampleStatus.textContent, /Fit failed in study 1/);
    assert.equal(f.exampleBody.textContent.includes('Recovered'), false);
});

test('numerical fallback and empty selection remain available without plotting', async () => {
    const f = fixture({missingRenderer: true});
    await settle();
    assert.equal(f.values.open, true);
    assert.equal(f.valuesBody.children.length, 3);
    f.change(f.controls[0], false);
    assert.equal(f.valuesBody.children.length, 0);
    assert.equal(f.api.getSelection(f.root).selectedIds.length, 0);
    const ordinary = fixture(); ordinary.change(ordinary.controls[0], false);
    assert.equal(ordinary.calls.updates.at(-1).spec.data.length, 0);
    assert.match(ordinary.calls.updates.at(-1).spec.empty_message, /Choose up to three/);
    assert.equal(ordinary.calls.purges.length, 0);
});

test('unsupported Approach 1 stays explicit and controller disposal is idempotent', () => {
    const data = payload(); data.supported = false; data.approach_id = 'approach1';
    const unsupported = fixture({data});
    assert.equal(unsupported.environment.disabled, true);
    assert.equal(unsupported.calls.updates.length, 0);
    assert.equal(unsupported.controls[0].disabled, false);
    assert.deepEqual(plain(unsupported.api.getSelection(unsupported.root).selectedIds), ['calibration-design-0']);
    const f = fixture(), count = f.calls.updates.length;
    f.api.initialize(f.holder); assert.equal(f.calls.updates.length, count);
    f.api.dispose(f.holder); f.api.dispose(f.holder);
    f.change(f.environment, 'in_vitro');
    assert.equal(f.calls.updates.length, count);
    assert.equal(f.calls.purges.length, 2);
});
