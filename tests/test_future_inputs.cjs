/* Run the real uncertainty-input controls without browser dependencies. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/uncertainty-inputs.js'), 'utf8');
function fixture() {
    const elements = {}, events = {}, groups = [];
    function control(id, value = '') {
        const item = {id, value, disabled: false, required: false, message: '',
            setCustomValidity(message) { this.message = message; }};
        elements[id] = item;
        return item;
    }
    function row(field, controls) {
        return {hidden: false, getAttribute: () => field, querySelectorAll: () => controls};
    }
    for (const prefix of ['in_vitro', 'in_vivo', 'prediction']) {
        control(prefix + '_uncertainty_mode', 'none');
        const rows = [row('kind', [control(prefix + '_response_kind', 'individual')]),
            row('constant', [control(prefix + '_uncertainty_value')]),
            row('column', [control(prefix + '_uncertainty_column')]),
            row('counts', [control(prefix + '_replicate_column')])];
        groups.push({getAttribute: () => prefix, querySelectorAll: () => rows});
    }
    const future = control('future_uncertainty_mode', 'none');
    const options = {prediction: {}, calibration: {}};
    future.querySelector = selector => options[selector.includes('prediction') ? 'prediction' : 'calibration'];
    const futureRows = [row('kind', [control('future_response_kind', 'individual')]),
        row('constant', [control('future_uncertainty_value')])];
    elements['future-uncertainty-group'] = {querySelectorAll: () => futureRows};
    control('uncertainty_independence_confirmed');
    elements['uncertainty-independence-row'] = {hidden: false};
    elements['uncertainty-options'] = {open: false};
    elements.prediction_file = {files: [{}]};
    const state = {approaches: [{value: 'approach2'}], logResponse: false};
    const form = {
        querySelectorAll: selector => selector === '[data-uncertainty-prefix]' ? groups : state.approaches,
        querySelector: () => state.logResponse ? {} : null,
        addEventListener(name, handler) { events[name] = handler; },
    };
    elements['upload-form'] = form;
    const context = {document: {readyState: 'complete', getElementById: id => elements[id]},
        window: {addEventListener() {}}};
    vm.runInNewContext(source, context);
    return {elements, options, state, futureRows, update: events.change, submit: events.submit};
}
let cases = 0;
function check(name, run) {
    run(fixture());
    cases += 1;
}

check('default state preserves optional input behavior', h => {
    assert.equal(h.elements.uncertainty_independence_confirmed.disabled, true);
    assert.equal(h.elements.uncertainty_independence_confirmed.required, false);
    assert.equal(h.elements.future_uncertainty_value.disabled, true);
    assert.ok(h.futureRows.every(item => item.hidden));
    assert.equal(h.options.prediction.disabled, true);
    assert.equal(h.options.calibration.disabled, true);
});
check('future constant alone requires confirmation and both constant controls', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.elements.future_uncertainty_value.value = '.03';
    h.update();
    assert.ok(h.futureRows.every(item => !item.hidden));
    assert.equal(h.elements.future_uncertainty_value.required, true);
    assert.equal(h.elements.future_response_kind.disabled, false);
    assert.equal(h.elements.uncertainty_independence_confirmed.required, true);
    assert.equal(h.elements.uncertainty_independence_confirmed.disabled, false);
    assert.equal(h.elements.future_uncertainty_mode.message, '');
});
for (const value of ['0', '-.03', 'Infinity']) {
    check('invalid future SD/SEM ' + value, h => {
        h.elements.future_uncertainty_mode.value = 'constant';
        h.elements.future_uncertainty_value.value = value;
        h.update();
        assert.match(h.elements.future_uncertainty_value.message, /greater than zero/);
    });
}
check('future measurements require a prediction dataset', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.elements.prediction_file.files = [];
    h.update();
    assert.match(h.elements.future_uncertainty_mode.message, /Upload a prediction dataset/);
});
check('future-only requests require Approach 2 alone', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.state.approaches = [{value: 'approach1'}, {value: 'approach2'}];
    h.update();
    assert.match(h.elements.future_uncertainty_mode.message, /Approach 2 alone/);
});
check('future-only requests reject log-response scaling', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.state.logResponse = true;
    h.update();
    assert.match(h.elements.future_uncertainty_mode.message, /Log10 response/);
});
for (const [mode, prefix] of [['prediction', 'prediction'], ['calibration', 'in_vivo']]) {
    check(mode + ' reuse requires its source uncertainty', h => {
        h.elements.future_uncertainty_mode.value = mode;
        h.update();
        assert.equal(h.options[mode].disabled, true);
        assert.match(h.elements.future_uncertainty_mode.message, /before reusing/);
        h.elements[prefix + '_uncertainty_mode'].value = 'constant';
        h.elements[prefix + '_uncertainty_value'].value = '.02';
        h.update();
        assert.equal(h.options[mode].disabled, false);
        assert.equal(h.elements.future_uncertainty_mode.message, '');
        assert.equal(h.elements.future_response_kind.disabled, true);
        assert.equal(h.elements.future_uncertainty_value.required, false);
    });
}
check('turning future variability off clears stale validity', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.elements.future_uncertainty_value.value = '-1';
    h.update();
    h.elements.future_uncertainty_mode.value = 'none';
    h.update();
    assert.equal(h.elements.future_uncertainty_value.message, '');
    assert.equal(h.elements.future_uncertainty_value.disabled, true);
    assert.equal(h.elements.uncertainty_independence_confirmed.required, false);
});
check('existing column and mean controls remain active without future variability', h => {
    h.elements.prediction_uncertainty_mode.value = 'column';
    h.elements.prediction_response_kind.value = 'mean';
    h.update();
    assert.equal(h.elements.prediction_uncertainty_column.required, true);
    assert.equal(h.elements.prediction_replicate_column.disabled, false);
    assert.equal(h.elements.prediction_replicate_column.required, false);
    assert.equal(h.elements.uncertainty_independence_confirmed.required, true);
    assert.equal(h.elements.future_uncertainty_mode.message, '');
});
check('submit refresh opens uncertainty details when future inputs are active', h => {
    h.elements.future_uncertainty_mode.value = 'constant';
    h.submit();
    assert.equal(h.elements['uncertainty-options'].open, true);
});
console.log(`PASS ${cases} future-input control cases`);
