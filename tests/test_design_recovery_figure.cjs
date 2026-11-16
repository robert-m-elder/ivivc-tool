'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const figure = require('../static/js/design-recovery-figure.js');

function fixture() {
    const times = [0, 1, 2, 3, 4, 5];
    const schedules = [0, 1, 2, 3].map(index => ({
        id: 'schedule-' + index, name: 'Schedule ' + index,
        evaluation_times: [...times], n_simulations: 4, n_successful: 3, n_failed: 1,
        recovered_response: {lower: [.8, .7, null, .3, .2, .1], median: [1, .8, null, .5, .3, .2],
            upper: [1.2, 1, null, .7, .4, .3], available_counts: [3, 3, 0, 3, 3, 3]},
        recovery_error: {lower: [-.2, -.1, null, -.1, -.2, -.1], median: [42, .2, null, .1, 0, 0],
            upper: [.2, .3, null, .3, .2, .1], available_counts: [3, 3, 0, 3, 3, 3]},
        example: {simulation_index: 0, study_number: 1, trajectory_draw_index: 7, fit_success: true,
            fit_message: null, true_response: [1, .8, .6, .4, .2, .1],
            recovered_response: [1.1, .9, .7, .5, .3, .2],
            observed_time_means: [
                {time: 0, response: 123, role: 'completed', planned_replicate_count: null},
                {time: 2 + index / 10, response: .65, role: 'new', planned_replicate_count: index + 2},
                {time: 4, response: .25, role: 'new', planned_replicate_count: index + 2},
            ]},
    }));
    return {schema_version: 1,
        reference: {time: times, central_response: [1, .8, .6, .4, .2, .1], lower: [0, 0], upper: [2, 2]},
        recovery: {schema_version: 1, schedules,
            shared_example: {available: true, simulation_index: 0, study_number: 1, trajectory_draw_index: 7},
            completed_observations: [
                {time: 0, response: 1, observation_sigma: .03, uncertainty_kind: 'SEM', reported_replicate_count: 4},
                {time: 0, response: .98, observation_sigma: .04, uncertainty_kind: 'SEM', reported_replicate_count: 9},
            ], fit_reported_intervals: {available: false}},
    };
}

const selection = {selectedIds: ['schedule-2', 'schedule-0'], colors: {'schedule-0': '#123456', 'schedule-2': '#abcdef'}};
const kind = (spec, name) => spec.data.filter(trace => trace.meta.kind === name);

test('summary overlays selected schedules in result order with shared selection colors', () => {
    const value = figure.summary(fixture(), selection);
    const medians = kind(value, 'median');
    assert.deepEqual(medians.map(trace => trace.meta.schedule_id), ['schedule-0', 'schedule-2']);
    assert.deepEqual(medians.map(trace => trace.line.color), ['#123456', '#abcdef']);
    assert.equal(value.layout.showlegend, true);
    assert.equal(value.layout.uirevision, 'planning-recovery-response');
    assert.equal(kind(value, 'reference').length, 1);
    assert.equal(kind(value, 'reference')[0].line.dash, 'dash');
    assert.equal(medians[0].name, 'Schedule 0');
    assert.match(medians[0].hovertemplate, /Successful fits: 3\/4/);
    assert.deepEqual(medians[0].customdata, [[3], [3], [0], [3], [3], [3]]);
});

test('response range segments never fill across unavailable time points', () => {
    const value = figure.summary(fixture(), {selectedIds: ['schedule-0']});
    assert.deepEqual(kind(value, 'range_upper').map(trace => trace.x), [[0, 1], [3, 4, 5]]);
    assert.deepEqual(kind(value, 'range_lower').map(trace => trace.x), [[0, 1], [3, 4, 5]]);
    assert.equal(kind(value, 'range_upper').every(trace => trace.fill === 'tonexty'), true);
    assert.equal(kind(value, 'median')[0].connectgaps, false);
    assert.deepEqual(kind(value, 'median')[0].x, [0, 1, null, 3, 4, 5]);
    assert.match(kind(value, 'range_upper')[0].name, /across-study central 95% range/);
    assert.match(value.layout.meta.range_interpretation, /not fitted confidence intervals/);
});

test('available ranges are shown and share one legend group with their corresponding median', () => {
    const value = figure.summary(fixture(), selection, {view: 'response'});
    assert.equal(kind(value, 'median').length, 2);
    assert.equal(kind(value, 'range_upper').length, 4);
    for (const median of kind(value, 'median')) {
        assert.equal(median.showlegend, true);
        const ranges = value.data.filter(trace => trace.meta.schedule_id === median.meta.schedule_id &&
            ['range_lower', 'range_upper'].includes(trace.meta.kind));
        assert.equal(ranges.every(trace => trace.legendgroup === median.legendgroup && !trace.showlegend), true);
    }
    assert.equal(value.layout.legend.groupclick, 'togglegroup');
    assert.equal(value.data.filter(trace => trace.showlegend).length, 3, 'Reference plus one entry per recovered curve.');
});

test('long legend names wrap safely without losing the full schedule name or obscuring counts', () => {
    const data = fixture();
    data.recovery.schedules[0].name = '<script>' + 'Long alternative schedule name '.repeat(3) + '</script>';
    const value = figure.summary(data, selection);
    const median = kind(value, 'median')[0];
    assert.match(median.name, /<br>/);
    assert.equal(median.name.includes('<script>'), false);
    assert.equal(median.name.replaceAll('<br>', '').replaceAll(' ', ''),
        data.recovery.schedules[0].name.replaceAll(' ', '').replaceAll('<', '&lt;').replaceAll('>', '&gt;'));
    assert.match(median.hovertemplate, /Successful fits: 3\/4/);
    assert.equal(value.layout.height - value.layout.margin.t - value.layout.margin.b, 255);
    assert.ok(value.layout.margin.b > 100);
});

test('legend visibility resets when the view or range group membership changes', () => {
    const data = fixture();
    const response = figure.summary(data, selection);
    assert.equal(figure.summary(data, selection).layout.legend.uirevision, response.layout.legend.uirevision);
    assert.notEqual(figure.summary(data, selection, {view: 'error'}).layout.legend.uirevision,
        response.layout.legend.uirevision);
    assert.notEqual(figure.summary(data, {selectedIds: ['schedule-0']}).layout.legend.uirevision,
        response.layout.legend.uirevision);
    data.recovery.schedules[0].recovered_response.available_counts.fill(1);
    assert.notEqual(figure.summary(data, selection).layout.legend.uirevision, response.layout.legend.uirevision);
});

test('recovery-error curves pass through the already paired summaries and use a separate view scale', () => {
    const value = figure.summary(fixture(), {selectedIds: ['schedule-0']}, {view: 'error'});
    assert.equal(kind(value, 'median')[0].y[0], 42, 'Never subtract central/marginal reference curves to recreate the error.');
    assert.equal(kind(value, 'reference').length, 0);
    assert.equal(value.layout.shapes[0].y0, 0);
    assert.equal(value.layout.shapes[0].y1, 0);
    assert.equal(value.layout.uirevision, 'planning-recovery-error');
    assert.match(value.layout.yaxis.title.text, /Recovery error/);
});

test('singleton estimates keep their medians without manufacturing a range', () => {
    const data = fixture(), row = data.recovery.schedules[0];
    row.n_successful = 1;
    row.recovered_response.available_counts = [1, 1, 0, 1, 1, 1];
    const value = figure.summary(data, {selectedIds: ['schedule-0']});
    assert.equal(kind(value, 'range_upper').length, 0);
    assert.equal(kind(value, 'median').length, 1);
});

test('reversed range bounds split the range and unavailable median values preserve gaps', () => {
    const data = fixture(), row = data.recovery.schedules[0];
    row.recovered_response.lower[4] = .8;
    row.recovered_response.available_counts[1] = 0;
    const value = figure.summary(data, {selectedIds: ['schedule-0']});
    assert.deepEqual(kind(value, 'range_upper').map(trace => trace.x), [[0], [3], [5]]);
    assert.equal(kind(value, 'median')[0].y[1], null);
});

test('failed schedules cannot contribute incidental curve values or zero-width bands', () => {
    const data = fixture();
    data.recovery.schedules[0].n_successful = 0;
    const value = figure.summary(data, {selectedIds: ['schedule-0']});
    assert.equal(kind(value, 'median').length, 0);
    assert.equal(kind(value, 'range_upper').length, 0);
    assert.equal(kind(value, 'reference').length, 1, 'The separately labeled central reference remains reviewable.');
    assert.equal(figure.summary(data, {selectedIds: ['schedule-0']}, {view: 'error'}), null);
});

test('empty selection, unavailable payloads and invalid views stay unavailable', () => {
    assert.equal(figure.summary(fixture(), {selectedIds: []}), null);
    assert.equal(figure.summary(fixture(), {selectedIds: ['unknown']}), null);
    assert.equal(figure.summary(null, selection), null);
    assert.equal(figure.summary(fixture(), selection, {view: 'unknown'}), null);
    const data = fixture();
    data.schema_version = 2;
    assert.equal(figure.summary(data, selection), null);
});

test('shared example retains the predetermined failed study rather than finding a successful replacement', () => {
    const data = fixture();
    data.recovery.schedules[0].example.fit_success = false;
    data.recovery.schedules[0].example.fit_message = 'Fixed example failed';
    const value = figure.example(data, selection);
    assert.equal(value.layout.meta.simulation_index, 0);
    assert.equal(value.layout.meta.trajectory_draw_index, 7);
    assert.equal(kind(value, 'truth').length, 1);
    assert.deepEqual(kind(value, 'truth')[0].y, data.recovery.schedules[0].example.true_response);
    assert.deepEqual(kind(value, 'recovered').map(trace => trace.meta.schedule_id), ['schedule-2']);
    assert.equal(kind(value, 'observations').length, 2, 'Even the failed study retains its measured points.');
});

test('example observations are the retained new time means and completed rows are shown only once', () => {
    const data = fixture(), value = figure.example(data, selection);
    const completed = kind(value, 'completed');
    assert.equal(completed.length, 1);
    assert.deepEqual(completed[0].x, [0, 0]);
    assert.deepEqual(completed[0].y, [1, .98]);
    assert.match(completed[0].text[0], /SEM: 0.03/);
    assert.match(completed[0].text[1], /independent count: 9/);
    assert.equal(completed[0].marker.symbol, 'circle-open');
    assert.deepEqual(kind(value, 'observations')[0].x, [2, 4]);
    assert.deepEqual(kind(value, 'observations')[0].customdata, [[2], [2]]);
    assert.equal(value.data.some(trace => trace.error_y || trace.fill), false,
        'Historical precision is described faithfully; no pointwise fitted intervals are reconstructed.');
});

test('a shared example remains meaningful when all selected recovery fits failed', () => {
    const data = fixture();
    data.recovery.schedules.forEach(row => { row.example.fit_success = false; });
    const value = figure.example(data, selection);
    assert.equal(kind(value, 'recovered').length, 0);
    assert.equal(kind(value, 'truth').length, 1);
    assert.equal(kind(value, 'observations').length, 2);
});

test('unavailable or mismatched shared examples do not substitute another simulation', () => {
    const data = fixture();
    data.recovery.shared_example.available = false;
    assert.equal(figure.example(data, selection), null);
    data.recovery.shared_example.available = true;
    data.recovery.schedules[2].example.simulation_index = 1;
    assert.equal(figure.example(data, selection), null);
    assert.equal(figure.example(data, {selectedIds: []}), null);
});

test('hostile labels are escaped in Plotly titles, names and completed-point hover text', () => {
    const data = fixture();
    data.recovery.schedules[0].name = '<script>alert(1)</script>';
    data.recovery.completed_observations[0].uncertainty_kind = '<b>SEM</b>';
    const main = figure.summary(data, selection), one = figure.example(data, selection);
    assert.equal(JSON.stringify(main).includes('<script>'), false);
    assert.equal(JSON.stringify(one).includes('<script>'), false);
    assert.match(kind(main, 'median')[0].name, /&lt;script&gt;/);
    assert.match(kind(one, 'completed')[0].text[0], /&lt;b&gt;SEM&lt;\/b&gt;/);
});

test('nonfinite numerical coordinates remain JSON-safe gaps without modifying retained data', () => {
    const data = fixture();
    data.recovery.schedules[0].recovered_response.median[0] = Infinity;
    data.recovery.schedules[0].evaluation_times[1] = NaN;
    const before = JSON.stringify(data);
    const value = figure.summary(data, selection);
    assert.deepEqual(kind(value, 'median')[0].y.slice(0, 3), [null, null, null]);
    figure.example(data, selection);
    assert.equal(JSON.stringify(data), before);
    const hasNonfinite = value => value && typeof value === 'object' ? Object.values(value).some(hasNonfinite) :
        typeof value === 'number' && !Number.isFinite(value);
    assert.equal(hasNonfinite(value), false);
});

test('more than three selections are capped and duplicate selections do not duplicate curves', () => {
    const value = figure.summary(fixture(), {selectedIds: ['schedule-3', 'schedule-3', 'schedule-2', 'schedule-1', 'schedule-0']});
    assert.deepEqual(kind(value, 'median').map(trace => trace.meta.schedule_id), ['schedule-0', 'schedule-1', 'schedule-2']);
});
