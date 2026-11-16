'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const figure = require('../static/js/design-timeline-figure.js');

function point(time, options = {}) {
    return {time, completed: false, baseline: false, timing_baseline: false, user_fixed: false,
        latest_before: [], contributes_to: [], deadline_eligibility: [], completed_rows: 0, new_replicates: 3, ...options};
}

function fixture() {
    return {schema_version: 1, reference: {time: [0, 10]}, schedule_data: {
        baseline_time: .5, latest_completed_time: 2, first_opportunity_time: 0,
        evaluation_time_range: [0, 10], future_policy: 'gap_fill', requirements: [
            {kind: 'window', label: 'Early window', lower: 0, upper: 4, minimum_count: 2, count_scope: 'all', status: 'available'},
            {kind: 'latest_before', label: 'Latest by six', lower: 2, upper: 6, status: 'available'},
            {kind: 'window', label: 'Beyond horizon', lower: 1, upper: null, status: 'beyond_horizon'},
        ], schedules: [
            {id: 'schedule-0', name: 'Suggested · 4 times', points: [point(0, {completed: true, completed_rows: 2}), point(1), point(4, {user_fixed: true, latest_before: ['Latest by six'], contributes_to: ['Early window', 'Other window']}), point(9)]},
            {id: 'schedule-1', name: 'Evenly spaced · 3 times', points: [point(0, {completed: true}), point(5, {user_fixed: true}), point(10)]},
            {id: 'schedule-2', name: 'Manual · comparison', points: [point(0, {completed: true}), point(2), point(8)]},
            {id: 'schedule-3', name: 'Manual · fourth', points: [point(0), point(2), point(7)]},
        ],
    }};
}

const schedules = spec => spec.data.filter(trace => trace.meta.kind === 'schedule');

test('selected schedules stack in original order on one shared numeric time axis', () => {
    const value = figure.build(fixture(), ['schedule-2', 'schedule-0']);
    const lines = schedules(value);
    assert.equal(lines.length, 2);
    assert.deepEqual(lines.map(line => line.meta.schedule_id), ['schedule-0', 'schedule-2']);
    assert.deepEqual(lines[0].y, [0, 0, 0, 0]);
    assert.deepEqual(lines[1].y, [1, 1, 1]);
    assert.equal(value.layout.xaxis.title.text, 'In vivo time');
    assert.equal(value.layout.xaxis2, undefined);
    assert.deepEqual(value.layout.yaxis.tickvals, [0, 1]);
    assert.deepEqual(value.layout.yaxis.range, [1.5, -.5]);
    assert.equal(value.layout.uirevision, 'planning-schedule-timeline');
});

test('schedule colors and marker symbols distinguish actual retained roles', () => {
    const value = figure.build(fixture(), ['schedule-0', 'schedule-1'], {'schedule-0': '#123456', 'schedule-1': '#abcdef'});
    const [first, second] = schedules(value);
    assert.equal(first.marker.color, '#123456');
    assert.equal(second.marker.color, '#abcdef');
    assert.deepEqual(first.marker.symbol, ['circle-open', 'circle', 'diamond', 'circle']);
    assert.deepEqual(second.marker.symbol, ['circle-open', 'square', 'circle']);
    assert.equal(value.layout.showlegend, false, 'The wrapping point-type key is outside the plot.');
    assert.equal(value.data.some(trace => trace.meta.kind === 'legend'), false);
});

test('overlapping fixed, latest-before and window roles all remain in hover text', () => {
    const point = schedules(figure.build(fixture(), ['schedule-0']))[0].text[2];
    assert.match(point, /Fixed by the user/);
    assert.match(point, /Selected by latest-before rule: Latest by six/);
    assert.match(point, /Contributes to window: Early window/);
    assert.match(point, /Contributes to window: Other window/);
    assert.match(point, /Planned independent replicates: 3/);
});

test('unsampled baseline is a guide and does not add a sampling point', () => {
    const value = figure.build(fixture(), ['schedule-0']);
    const baseline = value.layout.shapes.find(shape => shape.name === 'milestone-baseline');
    assert.equal(baseline.x0, .5);
    assert.equal(baseline.x1, .5);
    assert.equal(baseline.yref, 'paper');
    assert.equal(baseline.line.dash, 'dash');
    assert.equal(schedules(value)[0].x.includes(.5), false);
    assert.match(value.data.find(trace => trace.meta.guide === 'milestone-baseline').hovertemplate, /no observation is added/);
});

test('resolved windows and deadlines retain exact bounds while unresolved bounds are not drawn', () => {
    const value = figure.build(fixture(), ['schedule-0']);
    const window = value.layout.shapes.find(shape => shape.name === 'window-0');
    const deadline = value.layout.shapes.find(shape => shape.name === 'deadline-1');
    assert.deepEqual([window.x0, window.x1], [0, 4]);
    assert.deepEqual([deadline.x0, deadline.x1], [6, 6]);
    assert.equal(value.layout.shapes.some(shape => shape.name === 'window-2'), false);
    assert.match(value.data.find(trace => trace.meta.kind === 'window').hovertemplate, /Required window: \(0, 4\]/);
});

test('gap filling keeps earlier new times and latest completion on the shared axis', () => {
    const value = figure.build(fixture(), ['schedule-0']);
    assert.equal(schedules(value)[0].x.includes(1), true);
    assert.ok(value.layout.xaxis.range[0] <= 0);
    assert.ok(value.layout.xaxis.range[1] >= 10);
    const latest = value.data.find(trace => trace.meta.guide === 'latest-completed');
    assert.match(latest.hovertemplate, /Gap filling may include new earlier times/);
    assert.deepEqual(latest.x, [2, 2]);
});

test('coincident baseline and latest completion use one combined guide', () => {
    const data = fixture();
    data.schedule_data.baseline_time = 2;
    const value = figure.build(data, ['schedule-0']);
    assert.equal(value.layout.shapes.filter(shape => ['milestone-baseline', 'latest-completed'].includes(shape.name)).length, 1);
    assert.match(value.data.find(trace => trace.meta.guide === 'milestone-baseline').hovertemplate, /Also the latest completed time/);
});

test('hostile labels are escaped in ticks, names, and hover content', () => {
    const data = fixture();
    data.schedule_data.schedules[0].name = '<b>Source & schedule</b>';
    data.schedule_data.schedules[0].points[2].latest_before = ['<img src=x>'];
    data.schedule_data.requirements[0].label = '<a href="x">window</a>';
    const value = figure.build(data, ['schedule-0']);
    const first = schedules(value)[0];
    assert.equal(first.name, '&lt;b&gt;Source &amp; schedule&lt;/b&gt;');
    assert.match(first.text[2], /&lt;img src=x&gt;/);
    assert.match(value.data.find(trace => trace.meta.kind === 'window').hovertemplate, /&lt;a href=&quot;x&quot;&gt;/);
    assert.equal(JSON.stringify(value).includes('<img'), false);
    assert.equal(value.layout.yaxis.ticktext[0].includes('<b>'), false);
});

test('no selection is empty, unknown IDs are ignored, duplicate IDs collapse and selection is capped at three', () => {
    const data = fixture();
    assert.equal(figure.build(data, []), null);
    assert.equal(figure.build(data, ['unknown']), null);
    assert.equal(figure.build(null, []), null);
    assert.equal(figure.build(data, ['schedule-0', 'schedule-0']).layout.yaxis.tickvals.length, 1);
    assert.equal(schedules(figure.build(data, ['schedule-0', 'schedule-1', 'schedule-2', 'schedule-3'])).length, 3);
});

test('building a timeline does not mutate inspection data or selection', () => {
    const data = fixture(), ids = ['schedule-1', 'schedule-0'];
    const before = JSON.stringify(data), selected = JSON.stringify(ids);
    const first = figure.build(data, ids);
    assert.equal(JSON.stringify(data), before);
    assert.equal(JSON.stringify(ids), selected);
    assert.deepEqual(first, figure.build(data, ids));
    assert.ok(first.layout.height < figure.build(data, ['schedule-0', 'schedule-1', 'schedule-2']).layout.height);
    assert.equal(JSON.stringify(first).includes('Infinity'), false);
});
