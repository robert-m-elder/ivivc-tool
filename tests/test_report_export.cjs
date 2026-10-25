/* Regression checks for the actual structured Word-payload functions. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/results.js'), 'utf8');
const start = source.indexOf('var reportSectionsById = null;');
const end = source.indexOf('function getPlotForWordExport(', start);
assert(start >= 0 && end > start);
const tableRows = [['Source row', 'Response'], ...Array.from({length: 37}, (_, i) => [String(i), String(i / 10)])];
const base = [{type: 'paragraph', detail_key: 'endpoint', detail_label: 'Measured response', text: 'Measured response: Not provided'},
              {type: 'paragraph', detail_key: 'selection_rationale', detail_label: 'Selection rationale', text: 'Selection rationale: Not provided'},
              {type: 'table', title: 'All observations', rows: tableRows},
              {type: 'image', source_id: 'fit-plot', title: 'Fitted curve', description: 'Approximate pointwise 95% parameter-uncertainty band.'}];
const fields = {endpoint: {value: '<mass remaining>'}, selection_rationale: {value: ''}};
function report(id, withImage) {
    const attributes = {'data-report-id': id};
    const placeholder = {getAttribute(key) { return {'data-plot-source': 'fit-plot',
        'data-report-image-id': withImage ? 'one-image' : null}[key] || null; }};
    const paragraphs = ['endpoint', 'selection_rationale'].map(key => ({textContent: '',
        getAttribute(attr) { return attr === 'data-report-detail' ? key : (key === 'endpoint' ? 'Measured response' : 'Selection rationale'); }}));
    return {paragraphs,
        getAttribute(key) { return attributes[key]; },
        querySelectorAll(selector) { return selector === '.report-plot-placeholder' ? [placeholder] : paragraphs; }};
}
const reports = [report('final-1', true), report('final-2', false)];
const context = {reportImageDataById: {'one-image': 'data:image/png;base64,example'},
    document: {
        getElementById() { return {textContent: JSON.stringify({'final-1': base, 'final-2': base})}; },
        querySelector(selector) { const key = /="([^"]+)"/.exec(selector)[1]; return fields[key]; },
        querySelectorAll() { return reports; }
    }};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
context.reportSelectionRationales['final-1'] = 'Simple model with suitable residuals';
context.reportSelectionRationales['final-2'] = 'Alternative mapping';
context.updateReportDetails('final-1');
assert.equal(fields.selection_rationale.value, 'Simple model with suitable residuals');
assert.equal(reports[0].paragraphs[0].textContent, 'Measured response: <mass remaining>');
const payload = context.collectReportSections(reports[0]);
assert.equal(payload[2].rows.length, 38); // all records, independent of displayed page
assert.equal(payload[0].text, 'Measured response: <mass remaining>');
assert.equal(payload[1].text, 'Selection rationale: Simple model with suitable residuals');
assert.equal(payload[3].data_url, 'data:image/png;base64,example');
assert.equal(payload[3].caption, base[3].description);
context.updateReportDetails('final-2');
assert.equal(fields.selection_rationale.value, 'Alternative mapping');
assert.equal(context.collectReportSections(reports[1])[3].data_url, '');
assert.equal(base[1].text, 'Selection rationale: Not provided'); // original record was not mutated
fields.endpoint.value = '';
assert.equal(context.collectReportSections(reports[0])[0].text, 'Measured response: Not provided');
console.log('PASS complete structured tables, author details, per-selection rationale, image status, immutable calculation record');
