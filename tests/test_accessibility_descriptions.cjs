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
const band = {name: '95% Parameter-Uncertainty Band', fill: 'tonexty'};
const fit = describe('plot-fit-approach1-linear', [band], 'Time fit');
assert.match(fit, /pointwise 95% parameter-uncertainty intervals/);
assert.match(fit, /do not include the additional scatter of future observations/);
assert.doesNotMatch(fit, /95% prediction bands/);
const unshaded = describe('plot-fit-approach1-linear', [{name: 'Fitted curve'}]);
assert.doesNotMatch(unshaded, /Shaded areas show/);
const otherFill = describe('plot-fit-example', [{name: 'Other area', fill: 'tonexty'}]);
assert.doesNotMatch(otherFill, /95% parameter-uncertainty/);

for (const axis of ['error_x', 'error_y']) {
    const prediction = describe('plot-prediction-approach2', [{name: 'In Vivo (Predicted)', [axis]: {array: [1]}}]);
    assert.match(prediction, /95% parameter-uncertainty intervals/);
    assert.match(prediction, /Prediction inputs are treated as fixed/);
}
const separate = describe('plot-prediction-approach2', [
    {name: 'In Vivo (Predicted)'}, {name: '95% parameter-uncertainty interval', mode: 'lines'},
]);
assert.match(separate, /Error bars or separate interval segments/);
const unavailable = describe('plot-prediction-approach2', [{name: 'In Vivo (Predicted)'}]);
assert.doesNotMatch(unavailable, /interval segments show/);

const cvTraces = [{name: 'CV mean +/- 1 SD', fill: 'tonexty'}, {name: 'Mean CV fit'}];
for (const id of ['plot-cv-fit-example', 'report-placeholder']) {
    const cv = describe(id, cvTraces, 'CV fit variability');
    assert.match(cv, /mean plus or minus one standard deviation/);
    assert.match(cv, /not a 95% uncertainty interval/);
    assert.doesNotMatch(cv, /95% parameter-uncertainty intervals|95% prediction bands/);
}
const direct = describe('plot-fit-approach3', [{name: 'Response ratio'}]);
assert.match(direct, /no parametric fitted curve or model-based uncertainty band/);
console.log('PASS 10 plot-description cases: fitting, horizontal/vertical bars, separate intervals, unavailable bounds, CV, direct mapping');
