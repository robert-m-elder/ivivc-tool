/* Exercise the app's plot descriptions without a browser or extra packages. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/accessibility.js'), 'utf8');
const start = source.indexOf('    function axisTitle(');
const end = source.indexOf('    function enhancePlotAccessibility(', start);
assert(start >= 0 && end > start);
const context = {stripHtml: value => String(value || '').replace(/<[^>]+>/g, '')};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
function describe(id, data, title = 'Results') {
    return context.plotTypeDescription({id, previousElementSibling: null}, {data, layout: {}}, title);
}
let cases = 0;
function check(name, callback) {
    callback();
    cases += 1;
}
const band = {name: '95% Parameter-Uncertainty Band', fill: 'tonexty'};
check('legacy fitted-curve bands', () => {
    const fit = describe('plot-fit-approach1-linear', [band], 'Time fit');
    assert.match(fit, /pointwise 95% parameter-uncertainty intervals/);
    assert.match(fit, /do not include the additional scatter of future observations/);
    assert.doesNotMatch(fit, /95% prediction bands/);
});
for (const data of [[{name: 'Fitted curve'}], [{name: 'Other area', fill: 'tonexty'}]]) {
    check('no invented fitted-curve band', () => {
        const fit = describe('plot-fit-example', data);
        assert.doesNotMatch(fit, /Shaded areas show|95% parameter-uncertainty/);
    });
}
for (const [kind, meaning] of [['sd', 'SD of individual measurements'], ['sem', 'SEM of reported means']]) {
    check('calibration-only ' + kind + ' differs from fitted-curve bands', () => {
        const fit = describe('plot-fit-approach2-example', [band, {
            name: 'In Vitro Data', meta: {observation_uncertainty_kind: kind},
            error_y: {array: [0.1], visible: true},
        }]);
        assert.ok(fit.includes('plus or minus one supplied ' + meaning));
        assert.match(fit, /separate from 95% fitted-curve intervals/);
        assert.match(fit, /Shaded areas show approximate pointwise 95% parameter-uncertainty intervals/);
        assert.doesNotMatch(fit, /Prediction inputs are treated as fixed/);
    });
}
check('observation bars alone are not parameter intervals', () => {
    const fit = describe('plot-fit-approach2-example', [{
        name: 'In Vivo Data', meta: {observation_uncertainty_kind: 'sem'},
        error_y: {array: [0.1], visible: true},
    }]);
    assert.match(fit, /one supplied SEM/);
    assert.doesNotMatch(fit, /Shaded areas show|95% parameter-uncertainty intervals/);
});
for (const axis of ['error_x', 'error_y']) {
    check('legacy prediction ' + axis, () => {
        const prediction = describe('plot-prediction-approach1', [{name: 'In Vivo (Predicted)', [axis]: {array: [1]}}]);
        assert.match(prediction, /95% parameter-uncertainty intervals/);
        assert.match(prediction, /Prediction inputs are treated as fixed/);
        assert.match(prediction, /exclude future in vivo measurement variability/);
    });
}
function predicted(output, rows, errors = {}) {
    return {name: 'In Vivo (Predicted)', x: [2], y: [0.5],
            meta: {prediction_records: rows.map(row => ({in_vitro_time: 1, in_vitro_value: 0.5,
                ['predicted_in_vivo_' + output]: output === 'time' ? 2 : 0.5, ...row})),
                prediction_output_kind: output}, ...errors};
}
const parameter = {lower_95: 1.5, upper_95: 2.5, interval_kind: 'parameter_uncertainty'};
const input = {input_standard_uncertainty: 0.1, input_uncertainty_kind: 'sem',
               response_lower_95: 0.3, response_upper_95: 0.7,
               value_interval_kind: 'prediction_input_uncertainty'};
check('calibration uncertainty without prediction-input errors', () => {
    const prediction = describe('plot-prediction-approach2-tau', [predicted('time', [parameter], {error_x: {array: [0.5]}})]);
    assert.match(prediction, /Horizontal bars.*95% parameter-uncertainty intervals/);
    assert.match(prediction, /Prediction inputs are treated as fixed/);
    assert.doesNotMatch(prediction, /combining fitted-parameter and supplied/);
});
check('prediction-input response-ratio intervals', () => {
    const prediction = describe('plot-prediction-approach2-value', [predicted('value', [{
        ...parameter, ...input, interval_kind: 'parameter_and_input_uncertainty',
    }], {error_y: {array: [0.2]}})]);
    assert.match(prediction, /Vertical bars.*combining fitted-parameter and supplied prediction-response uncertainty/);
    assert.doesNotMatch(prediction, /Prediction inputs are treated as fixed|Horizontal bars|joint 95% region/);
});
check('time-constant scaling with calibration and prediction-input uncertainty', () => {
    const prediction = describe('plot-prediction-approach2-tau', [predicted('time', [{...parameter, ...input}], {
        error_x: {array: [0.5]}, error_y: {array: [0.2]},
    })]);
    assert.match(prediction, /Horizontal bars.*95% parameter-uncertainty intervals/);
    assert.match(prediction, /does not change these mapped-time intervals/);
    assert.match(prediction, /Vertical bars.*supplied prediction-response uncertainty/);
    assert.match(prediction, /describe each axis separately; they do not form a joint 95% region/);
    assert.doesNotMatch(prediction, /Prediction inputs are treated as fixed|combining fitted-parameter/);
});
check('inverse mapping combines input and fitted-parameter uncertainty', () => {
    const prediction = describe('plot-prediction-approach2-inverse', [predicted('time', [{
        ...parameter, ...input, interval_kind: 'parameter_and_input_uncertainty',
    }], {error_x: {array: [0.5]}, error_y: {array: [0.2]}})], 'Time Prediction with Example Fit');
    assert.match(prediction, /Horizontal bars.*combining fitted-parameter and supplied prediction-response uncertainty/);
    assert.match(prediction, /Vertical bars.*supplied prediction-response uncertainty/);
    assert.match(prediction, /do not form a joint 95% region/);
    assert.doesNotMatch(prediction, /Prediction inputs are treated as fixed|Markers show observations and lines show the fitted model/);
});
check('separate propagated interval segments without error bars', () => {
    const prediction = describe('plot-prediction-approach2-inverse', [
        predicted('time', [{...parameter, ...input, interval_kind: 'parameter_and_input_uncertainty'}], {error_x: {visible: false}, error_y: {array: [0.2]}}),
        {name: '95% propagated-uncertainty interval', mode: 'lines', meta: {exclude_from_data_table: true}},
    ]);
    assert.match(prediction, /separate interval segments.*combining fitted-parameter/);
    assert.doesNotMatch(prediction, /Prediction inputs are treated as fixed/);
});
check('separate legacy parameter interval segments', () => {
    const prediction = describe('plot-prediction-approach1', [
        {name: 'In Vivo (Predicted)'}, {name: '95% parameter-uncertainty interval', mode: 'lines'},
    ]);
    assert.match(prediction, /Error bars or separate interval segments/);
});
for (const status of ['unavailable (failed draw)', 'not displayed (disabled)']) {
    check('no invented interval for ' + status, () => {
        const prediction = describe('plot-prediction-approach2-inverse', [predicted('time', [{
            ...parameter, ...input, lower_95: null, upper_95: null,
            response_lower_95: null, response_upper_95: null, interval_status: status,
        }], {error_x: {array: [null]}, error_y: {array: [NaN]}})]);
        assert.doesNotMatch(prediction, /bars or separate interval segments show|bounds are marginal|inputs are treated as fixed/);
    });
}
check('no response interval is described at an unavailable mapped time', () => {
    const trace = predicted('time', [{...parameter, ...input, lower_95: null, upper_95: null,
        predicted_in_vivo_time: null}], {error_y: {array: [null], visible: false}});
    trace.x = [null];
    const prediction = describe('plot-prediction-approach2-inverse', [trace]);
    assert.doesNotMatch(prediction, /bars or separate interval segments show/);
});
check('response intervals remain distinct when mapped-time bounds are unavailable', () => {
    const trace = predicted('time', [{...parameter, ...input, lower_95: null, upper_95: null}], {
        error_x: {array: [null], visible: false}, error_y: {array: [0.2]},
    });
    const prediction = describe('plot-prediction-approach2-tau', [trace]);
    assert.match(prediction, /Vertical bars.*supplied prediction-response uncertainty/);
    assert.doesNotMatch(prediction, /Horizontal bars|joint 95% region|95% parameter-uncertainty intervals/);
});
check('legacy missing prediction bounds', () => {
    const prediction = describe('plot-prediction-approach2', [{name: 'In Vivo (Predicted)'}]);
    assert.doesNotMatch(prediction, /interval segments show/);
});
const cvTraces = [{name: 'CV mean +/- 1 SD', fill: 'tonexty'}, {name: 'Mean CV fit'}];
for (const id of ['plot-cv-fit-example', 'report-placeholder']) {
    check('CV fit variability for ' + id, () => {
        const cv = describe(id, cvTraces, 'CV fit variability');
        assert.match(cv, /mean plus or minus one standard deviation/);
        assert.match(cv, /not a 95% uncertainty interval/);
        assert.doesNotMatch(cv, /95% parameter-uncertainty intervals|95% prediction bands/);
    });
}
check('direct mapping without parametric intervals', () => {
    const direct = describe('plot-fit-approach3', [{name: 'Response ratio'}]);
    assert.match(direct, /no parametric fitted curve or model-based uncertainty band/);
});
check('residuals use analyzed fitted-output units', () => {
    const residual = describe('plot-residual-example', [{name: 'Residuals'}]);
    assert.match(residual, /units of the fitted output on the analyzed scale/);
    assert.doesNotMatch(residual, /original response units|supplied SD|supplied SEM/);
});
console.log('PASS ' + cases + ' plot-description cases: supplied SD/SEM, parameter/input propagation, both axes, missing bounds, CV and residuals');
