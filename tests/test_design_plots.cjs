'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/design-plots.js'), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));

function fixture({missing = false, react, resize} = {}) {
    class Element {
        constructor(attributes = {}) {
            this.attributes = attributes;
            this.children = [];
            this.parentElement = null;
            this.hidden = false;
            this.textContent = '';
        }
        get id() { return this.attributes.id || ''; }
        get isConnected() {
            for (let node = this; node; node = node.parentElement) if (node === root) return true;
            return false;
        }
        append(child) { child.parentElement = this; this.children.push(child); return child; }
        matches(selector) { return this.attributes[selector.slice(1, -1)] !== undefined; }
        querySelectorAll(selector) {
            const found = [];
            const visit = node => node.children.forEach(child => {
                if (child.matches(selector)) found.push(child);
                visit(child);
            });
            visit(this);
            return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        getClientRects() {
            for (let node = this; node; node = node.parentElement) if (node.hidden) return [];
            return this.isConnected ? [{}] : [];
        }
    }
    const root = new Element();
    const wrapper = root.append(new Element({'data-design-plot': ''}));
    const target = wrapper.append(new Element({'data-design-plot-target': '', id: 'reference-plot'}));
    target.hidden = true;
    const spec = wrapper.append(new Element({'data-design-plot-spec': ''}));
    spec.textContent = JSON.stringify({
        data: [{x: [0, 1, 2], y: [1, null, .5], name: 'Reference'}],
        layout: {title: {text: 'Reference'}, uirevision: 'reference'},
    });
    const fallback = wrapper.append(new Element({'data-design-plot-fallback': ''}));
    const status = wrapper.append(new Element({'data-design-plot-status': ''}));
    const calls = {react: [], purge: [], resize: []};
    const Plotly = {
        react(...args) { calls.react.push(args); return react ? react(...args) : Promise.resolve(); },
        purge(element) { calls.purge.push(element); },
        Plots: {resize(element) { calls.resize.push(element); return resize ? resize(element) : Promise.resolve(); }},
    };
    const window = missing ? {} : {Plotly};
    vm.runInNewContext(source, {window});
    return {root, wrapper, target, spec, fallback, status, calls, window, api: window.DesignPlots};
}

test('initialization renders finite numerical data once and uses shared interactive settings', async () => {
    const f = fixture();
    await Promise.all([f.api.initialize(f.root), f.api.initialize(f.wrapper)]);
    assert.equal(f.calls.react.length, 1);
    const [target, data, layout, config] = f.calls.react[0];
    assert.equal(target, f.target);
    assert.deepEqual(plain(data[0].y), [1, null, .5]);
    assert.equal(layout.uirevision, 'reference');
    assert.equal(config.responsive, true);
    assert.equal(config.scrollZoom, false);
    assert.equal(config.displaylogo, false);
    assert.equal(config.toImageButtonOptions.format, 'svg');
    assert.equal(f.target.hidden, false);
    assert.equal(f.fallback.hidden, true);
    assert.match(f.status.textContent, /Interactive plot ready/);
});

test('a failed CDN load preserves static output and does not reject the result request', async () => {
    const f = fixture({missing: true});
    await f.api.initialize(f.root);
    assert.equal(f.calls.react.length, 0);
    assert.equal(f.target.hidden, true);
    assert.equal(f.fallback.hidden, false);
    assert.match(f.status.textContent, /static plot and numerical results remain available/);
});

test('malformed or incomplete figure data falls back without executing payload text', async () => {
    for (const encoded of ['window.injected = true', '{}', '{"data": [], "layout": []}']) {
        const f = fixture();
        f.spec.textContent = encoded;
        await f.api.initialize(f.wrapper);
        assert.equal(f.calls.react.length, 0);
        assert.equal(f.window.injected, undefined);
        assert.equal(f.fallback.hidden, false);
        assert.equal(f.target.hidden, true);
    }
});

test('a rejected render is purged and leaves the numerical/static result available', async () => {
    const f = fixture({react: () => Promise.reject(new Error('Plotly render failed'))});
    await f.api.initialize(f.root);
    assert.equal(f.calls.purge.length, 1);
    assert.equal(f.calls.purge[0], f.target);
    assert.equal(f.fallback.hidden, false);
    assert.equal(f.target.hidden, true);
    assert.match(f.status.textContent, /Interactive plot unavailable/);
});

test('plots rendered in a hidden workspace resize when its result view becomes visible', async () => {
    const f = fixture();
    f.root.hidden = true;
    await f.api.initialize(f.root);
    assert.equal(f.calls.react.length, 1);
    assert.equal(f.calls.resize.length, 0);
    f.root.hidden = false;
    f.api.resize(f.root);
    assert.deepEqual(f.calls.resize, [f.target]);
    assert.equal(f.calls.react.length, 1, 'Resizing must retain the plot and its zoom/legend state.');
});

test('a view opened before a pending render resolves is resized after completion', async () => {
    let resolve;
    const f = fixture({react: () => new Promise(done => { resolve = done; })});
    f.root.hidden = true;
    const ready = f.api.initialize(f.root);
    await Promise.resolve();
    f.root.hidden = false;
    f.api.resize(f.root);
    assert.equal(f.calls.resize.length, 0);
    resolve();
    await ready;
    assert.deepEqual(f.calls.resize, [f.target]);
});

test('removing pending results purges resources and does not apply a late successful plot', async () => {
    let resolve;
    const f = fixture({react: () => new Promise(done => { resolve = done; })});
    const ready = f.api.initialize(f.root);
    await Promise.resolve();
    f.api.purge(f.root);
    const pendingStatus = f.status.textContent;
    f.root.children = [];
    f.wrapper.parentElement = null;
    resolve();
    await ready;
    assert.equal(f.status.textContent, pendingStatus);
    assert.equal(f.fallback.hidden, false);
    assert.equal(f.calls.resize.length, 0);
    assert.equal(f.calls.purge.length, 2, 'Clean up both the partial and late completed render.');
});

test('purging before rendering starts prevents the obsolete result from drawing', async () => {
    const f = fixture();
    const ready = f.api.initialize(f.root);
    f.api.purge(f.root);
    await ready;
    assert.equal(f.calls.react.length, 0);
});

test('purging a completed plot supports a fresh render without retaining old UI state', async () => {
    const f = fixture();
    await f.api.initialize(f.root);
    f.api.purge(f.root);
    assert.deepEqual(f.calls.purge, [f.target]);
    f.spec.textContent = JSON.stringify({data: [], layout: {uirevision: 'new-result'}});
    await f.api.initialize(f.root);
    assert.equal(f.calls.react.length, 2);
    assert.equal(f.calls.react[1][2].uirevision, 'new-result');
});

test('missing optional containers and rejected resizes leave the successful chart intact', async () => {
    const f = fixture({resize: () => Promise.reject(new Error('View resized again'))});
    await f.api.initialize(null);
    await f.api.initialize(f.root);
    f.api.resize(f.wrapper);
    await Promise.resolve();
    assert.equal(f.target.hidden, false);
    assert.equal(f.fallback.hidden, true);
    assert.match(f.status.textContent, /Interactive plot ready/);
    f.api.purge(null);
    f.api.resize(null);
});
