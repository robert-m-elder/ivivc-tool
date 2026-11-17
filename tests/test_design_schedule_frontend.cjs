'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const builder = require('../static/js/design-timeline-figure.js');
const source = fs.readFileSync(path.join(__dirname, '../static/js/design-schedule-inspection.js'), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));

function fixture({encoded, update, missingRenderer = false} = {}) {
    class Element {
        constructor(attributes = {}) {
            this.attributes = {...attributes};
            this.children = [];
            this.parentElement = null;
            this.listeners = {};
            this.dataset = Object.fromEntries(Object.entries(attributes).filter(([key]) =>
                key.startsWith('data-')).map(([key, value]) =>
                [key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()), value]));
            this.style = {values: {}, setProperty(key, value) { this.values[key] = value; }};
            this.hidden = false;
            this.disabled = false;
            this.checked = false;
            this.open = false;
            this.textContent = '';
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
        addEventListener(type, callback) { (this.listeners[type] ||= new Set()).add(callback); }
        removeEventListener(type, callback) { this.listeners[type]?.delete(callback); }
        dispatchEvent(event) {
            if (!event.target) event.target = this;
            for (const callback of this.listeners[event.type] || []) callback(event);
            if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
            return true;
        }
    }
    const holder = new Element();
    const root = holder.append(new Element({'data-planning-result': ''}));
    const payload = root.append(new Element({'data-planning-inspection': ''}));
    const schedules = ['manual', 'suggested', 'evenly_spaced', 'suggested'].map((kind, index) => ({
        id: 'schedule-' + index, name: kind + ' ' + index, kind,
        times: [0, index + 1, 10], points: [0, index + 1, 10].map(time => ({
            time, origin: kind, roles: ['new'], completed: false, user_fixed: false,
            latest_before: [], contributes_to: [], deadline_eligibility: [], new_replicates: 2,
            explanations: ['Proposed new measurement.'],
        })),
    }));
    payload.textContent = encoded ?? JSON.stringify({schema_version: 1,
        reference: {time: [0, 10]}, schedule_data: {schedules, baseline_time: 0,
            requirements: [], evaluation_time_range: [0, 10]}});
    const rows = [], controls = [], swatches = [], detailRows = [];
    for (const schedule of schedules) {
        const row = root.append(new Element({'data-schedule-row': schedule.id}));
        rows.push(row);
        const control = row.append(new Element({'data-schedule-select': schedule.id}));
        control.disabled = true;
        controls.push(control);
        swatches.push(row.append(new Element({'data-schedule-swatch': schedule.id})));
        detailRows.push(root.append(new Element({'data-schedule-detail': schedule.id})));
    }
    const status = root.append(new Element({'data-schedule-selection-status': ''}));
    const timeline = root.append(new Element({'data-schedule-timeline': '', 'data-design-plot': ''}));
    const target = timeline.append(new Element({'data-design-plot-target': ''}));
    target.hidden = true;
    const plotStatus = timeline.append(new Element({'data-design-plot-status': ''}));
    const details = root.append(new Element({'data-schedule-details': ''}));
    const empty = root.append(new Element({'data-schedule-selection-empty': ''}));
    const help = root.append(new Element({'data-schedule-timeline-help': ''}));
    const calls = {update: [], purge: [], events: []};
    const plots = {
        update(wrapper, spec) {
            calls.update.push(spec);
            target.hidden = false;
            return update ? update(wrapper, spec, target) : Promise.resolve();
        },
        purge(wrapper) { calls.purge.push(wrapper); },
    };
    root.addEventListener('design-schedule-selection-changed', event => calls.events.push(event.detail));
    class CustomEvent {
        constructor(type, options) { this.type = type; Object.assign(this, options); }
    }
    const window = {DesignTimelineFigure: builder, ...(missingRenderer ? {} : {DesignPlots: plots})};
    vm.runInNewContext(source, {window, CustomEvent});
    const api = window.DesignScheduleInspection;
    const select = (index, checked) => {
        controls[index].checked = checked;
        controls[index].dispatchEvent({type: 'change', bubbles: true});
    };
    return {api, window, holder, root, payload, rows, controls, swatches, detailRows, status,
        timeline, target, plotStatus, details, empty, help, calls, select};
}

test('initial selection uses the first suggested schedule and aligns table and timeline colors', () => {
    const f = fixture();
    f.api.initialize(f.holder);
    const chosen = plain(f.api.getSelection(f.root));
    assert.deepEqual(chosen.selectedIds, ['schedule-1']);
    assert.equal(f.controls[1].checked, true);
    assert.equal(f.rows[1].dataset.inspectionSelected, 'true');
    assert.equal(f.swatches[1].hidden, false);
    assert.equal(f.swatches[1].style.values['--schedule-color'], chosen.colors['schedule-1']);
    assert.equal(f.detailRows[1].hidden, false);
    assert.equal(f.detailRows[0].hidden, true);
    assert.equal(f.timeline.hidden, false);
    assert.equal(f.empty.hidden, true);
    assert.match(f.status.textContent, /1 of 3/);
    assert.equal(f.calls.update.length, 1);
});

test('three-schedule cap disables unselected controls while selected ones remain usable', () => {
    const f = fixture();
    f.api.initialize(f.root);
    f.select(0, true);
    f.select(2, true);
    assert.equal(f.controls[3].disabled, true);
    assert.equal(f.controls[1].disabled, false);
    assert.match(f.status.textContent, /Deselect a schedule/);
    f.select(3, true); // Guard programmatic or queued events too.
    assert.equal(f.controls[3].checked, false);
    assert.deepEqual(plain(f.api.getSelection(f.root).selectedIds), ['schedule-0', 'schedule-1', 'schedule-2']);
    f.select(0, false);
    assert.equal(f.controls[3].disabled, false);
    f.select(3, true);
    assert.equal(new Set(Object.values(f.api.getSelection(f.root).colors)).size, 3);
});

test('a comparison without suggestions initially selects its first available schedule', () => {
    const f = fixture();
    const data = JSON.parse(f.payload.textContent);
    data.schedule_data.schedules.forEach(schedule => { schedule.kind = 'manual'; });
    f.payload.textContent = JSON.stringify(data);
    f.api.initialize(f.root);
    assert.deepEqual(plain(f.api.getSelection(f.root).selectedIds), ['schedule-0']);
});

test('remaining selections keep their colors and all selected details use fixed result order', () => {
    const f = fixture();
    f.api.initialize(f.root);
    f.select(3, true);
    f.select(0, true);
    const before = plain(f.api.getSelection(f.root));
    f.select(1, false);
    const after = plain(f.api.getSelection(f.root));
    assert.deepEqual(after.selectedIds, ['schedule-0', 'schedule-3']);
    assert.equal(after.colors['schedule-0'], before.colors['schedule-0']);
    assert.equal(after.colors['schedule-3'], before.colors['schedule-3']);
    assert.equal(f.detailRows[1].hidden, true);
    assert.equal(f.detailRows[3].hidden, false);
    assert.deepEqual(plain(f.calls.events.at(-1)), after);
});

test('clearing selection hides and purges only the timeline and supports reselecting', () => {
    const f = fixture();
    f.api.initialize(f.root);
    f.select(1, false);
    assert.equal(f.timeline.hidden, true);
    assert.equal(f.details.hidden, true);
    assert.equal(f.empty.hidden, false);
    assert.equal(f.calls.purge[0], f.timeline);
    assert.equal(f.controls.every(control => !control.disabled), true);
    f.select(2, true);
    assert.equal(f.timeline.hidden, false);
    assert.equal(f.calls.update.length, 2);
});

test('reinitialization and unrelated changes retain the display selection without recalculation', () => {
    const f = fixture();
    f.api.initialize(f.root);
    f.select(2, true);
    const before = plain(f.api.getSelection(f.root));
    const calls = f.calls.update.length;
    f.api.initialize(f.holder);
    f.root.hidden = true;
    f.root.hidden = false;
    f.root.dispatchEvent({type: 'change', target: {dataset: {}}, bubbles: true});
    assert.deepEqual(plain(f.api.getSelection(f.root)), before);
    assert.equal(f.calls.update.length, calls);
});

test('missing plotting capabilities keep selection and exact time details usable', async () => {
    const f = fixture({missingRenderer: true});
    f.api.initialize(f.root);
    assert.equal(f.details.open, true);
    assert.equal(f.detailRows[1].hidden, false);
    assert.match(f.plotStatus.textContent, /Exact times and explanations remain available/);
    f.select(2, true);
    assert.equal(f.detailRows[2].hidden, false);
    const failed = fixture({update: (_wrapper, _figure, target) => { target.hidden = true; }});
    failed.api.initialize(failed.root);
    await Promise.resolve();
    assert.equal(failed.details.open, true);
});

test('unavailable or malformed inspection data disable controls without executing input text', () => {
    for (const encoded of ['window.injected = true', 'null', '{}', '{"schema_version":2}']) {
        const f = fixture({encoded});
        assert.doesNotThrow(() => f.api.initialize(f.root));
        assert.match(f.status.textContent, /Schedule inspection unavailable/);
        assert.equal(f.controls.every(control => control.disabled), true);
        assert.equal(f.details.hidden, false);
        assert.equal(f.detailRows.every(detail => !detail.hidden), true);
        assert.equal(f.window.injected, undefined);
        assert.equal(f.calls.update.length, 0);
    }
});

test('late rendering after clearing or disposal cannot reopen obsolete detail sections', async () => {
    let finish;
    const f = fixture({update: (_wrapper, _figure, target) => {
        target.hidden = true;
        return new Promise(resolve => { finish = resolve; });
    }});
    f.api.initialize(f.root);
    f.select(1, false);
    finish();
    await Promise.resolve();
    assert.equal(f.details.open, false);
    f.select(1, true);
    f.api.dispose(f.holder);
    finish();
    await Promise.resolve();
    assert.equal(f.details.open, false);
    const before = f.calls.update.length;
    f.select(0, true);
    assert.equal(f.calls.update.length, before);
    assert.deepEqual(plain(f.api.getSelection(f.root)), {selectedIds: [], colors: {}});
});

test('selection snapshots cannot mutate controller state and new comparisons start independently', () => {
    const f = fixture();
    f.api.initialize(f.root);
    const snapshot = f.api.getSelection(f.root);
    snapshot.selectedIds.push('schedule-3');
    snapshot.colors['schedule-1'] = '#000000';
    assert.deepEqual(plain(f.api.getSelection(f.root).selectedIds), ['schedule-1']);
    assert.notEqual(f.api.getSelection(f.root).colors['schedule-1'], '#000000');
    const second = fixture();
    second.api.initialize(second.root);
    assert.deepEqual(plain(second.api.getSelection(second.root).selectedIds), ['schedule-1']);
});
