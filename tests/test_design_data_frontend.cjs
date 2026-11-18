"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/design-data.js"), "utf8");

// A small browser fixture for the import controller. Server-rendered mapping
// responses install matching controls; no data-file parser is implemented here.
function fixture() {
    const document = {readyState: "complete"};
    const dataAttribute = key => "data-" + key.replace(/[A-Z]/g, letter => "-" + letter.toLowerCase());
    let installMapping;

    class Element {
        constructor(tag, attributes = {}) {
            this.tagName = tag.toUpperCase();
            this.attributes = {...attributes};
            this.children = [];
            this.parentElement = null;
            this.listeners = {};
            this.files = [];
            this.value = attributes.value || "";
            this._html = "";
            this._text = "";
            this.dataset = new Proxy({}, {
                get: (_, key) => this.attributes[dataAttribute(key)],
                set: (_, key, value) => { this.attributes[dataAttribute(key)] = String(value); return true; },
            });
        }
        get id() { return this.attributes.id || ""; }
        get name() { return this.attributes.name || ""; }
        get type() { return this.attributes.type || ""; }
        get disabled() { return this.attributes.disabled !== undefined; }
        set disabled(value) { if (value) this.attributes.disabled = ""; else delete this.attributes.disabled; }
        get hidden() { return this.attributes.hidden !== undefined; }
        set hidden(value) { if (value) this.attributes.hidden = ""; else delete this.attributes.hidden; }
        get open() { return this.attributes.open !== undefined; }
        set open(value) { if (value) this.attributes.open = ""; else delete this.attributes.open; }
        get checked() { return this.attributes.checked !== undefined; }
        set checked(value) { if (value) this.attributes.checked = ""; else delete this.attributes.checked; }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
        set textContent(value) { this._text = String(value); this._html = ""; this.children = []; }
        get innerHTML() { return this._html; }
        set innerHTML(value) {
            this._html = String(value); this._text = ""; this.children = [];
            if (this.attributes["data-import-config"] !== undefined) installMapping(this, value);
            else if (value) this.append(new Element("section"));
        }
        get childElementCount() { return this.children.length; }
        append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); }
        replaceChildren(...children) { this._html = ""; this._text = ""; this.children = []; this.append(...children); }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        getAttribute(name) { return this.attributes[name] ?? null; }
        hasAttribute(name) { return this.attributes[name] !== undefined; }
        addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
        matches(selector) {
            if (selector.startsWith("#")) return this.id === selector.slice(1);
            const match = /^([\w-]+)?((?:\[[^\]]+\])*)$/.exec(selector);
            if (!match || (match[1] && this.tagName.toLowerCase() !== match[1])) return false;
            return [...match[2].matchAll(/\[([^=\]]+)(?:=["']?([^\]"']*)["']?)?\]/g)].every(([, name, value]) =>
                this.hasAttribute(name) && (value === undefined || this.attributes[name] === value));
        }
        querySelectorAll(selector) {
            const selectors = selector.split(",").map(part => part.trim());
            const found = [];
            const visit = node => node.children.forEach(child => {
                if (selectors.some(part => child.matches(part))) found.push(child);
                visit(child);
            });
            visit(this); return found;
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        closest(selector) {
            for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node;
            return null;
        }
        contains(other) {
            for (let node = other; node; node = node.parentElement) if (node === this) return true;
            return false;
        }
        dispatchEvent(event) {
            if (!event.target) event.target = this;
            const ancestors = [];
            for (let node = this; node; node = node.parentElement) ancestors.push(node);
            for (const node of ancestors) for (const listener of node.listeners[event.type] || []) listener(event);
            return true;
        }
    }

    const element = (tag, attributes = {}, parent = null) => {
        const node = new Element(tag, attributes);
        if (parent) parent.append(node);
        return node;
    };
    const root = element("main", {id: "design-workspace"});
    const form = element("form", {id: "design-setup-form"}, root);
    const task = element("select", {id: "task", name: "task", value: "study"}, form);
    const studyGroup = element("div", {"data-design-task": "study"}, form);
    const hasCompleted = element("input", {id: "has_completed", name: "has_completed", type: "checkbox"}, studyGroup);
    const completedGroup = element("div", {"data-completed-dataset": "", hidden: ""}, studyGroup);

    function makeDataset(id, parent) {
        const container = element("div", {"data-design-dataset": id, "data-dataset-kind": "course",
            "data-inspect-url": "/data/inspect", "data-preview-url": "/data/preview"}, parent);
        const panel = element("details", {"data-import-panel": "", open: ""}, container);
        const heading = element("summary", {}, panel);
        const datasetSummary = element("span", {"data-import-dataset-summary": ""}, heading);
        const file = element("input", {id, name: id.replaceAll("-", "_"), type: "file"}, panel);
        const config = element("div", {"data-import-config": ""}, panel);
        const inspect = element("button", {type: "button", "data-import-action": "inspect"}, panel);
        const previewButton = element("button", {type: "button", "data-import-action": "preview", disabled: ""}, panel);
        const toggle = element("button", {type: "button", "data-import-preview-toggle": "",
            "aria-controls": id + "-preview", "aria-expanded": "false", hidden: ""}, panel);
        const preview = element("div", {id: id + "-preview", "data-import-preview": "", hidden: ""}, panel);
        const status = element("p", {"data-import-status": ""}, panel);
        const stale = element("p", {"data-import-stale": "", hidden: ""}, panel);
        return {id, container, panel, datasetSummary, file, inspect, config, previewButton, preview, toggle, status, stale,
            get mapping() { return config.querySelector('[data-import-mapping]'); }};
    }
    const dataset = makeDataset("reference-file", studyGroup);
    const completed = makeDataset("completed-target-file", completedGroup);
    const prefix = dataset.id + "__profile__";

    installMapping = (container, token) => {
        if (!token) return;
        const id = container.closest("[data-design-dataset]").dataset.designDataset;
        const key = id + "__profile__";
        if (token.includes("SHEETS") || token.includes("EXCEL_MAPPING")) {
            const worksheet = element("div", {"data-import-worksheet": ""}, container);
            element("select", {id: id + "__sheet", name: id + "__sheet", "data-import-sheet": "", value: token.includes("MAPPING") ? "Data" : ""}, worksheet);
        }
        if (!token.includes("MAPPING")) return;
        const mapping = element("details", {"data-import-mapping": "", open: ""}, container);
        const title = element("summary", {}, mapping);
        element("span", {"data-import-mapping-summary": ""}, title);
        const columns = element("div", {"data-column-fields": ""}, mapping);
        const options = element("fieldset", {"data-course-options": key}, columns);
        const field = (name, tag, value, parent = options) => element(tag, {id: key + name, name: key + name, value}, parent);
        field("time_column", "select", "time");
        field("response_column", "select", "response");
        field("uncertainty", "select", "none");
        const errorFields = element("div", {"data-error-fields": ""}, options);
        field("error_source", "select", "constant", errorFields);
        const errorConstant = element("div", {"data-error-source": "constant"}, errorFields);
        field("error_value", "input", ".05", errorConstant);
        const errorColumn = element("div", {"data-error-source": "column"}, errorFields);
        field("error_column", "select", "sd", errorColumn);
        const countFields = element("div", {"data-count-fields": ""}, options);
        field("count_source", "select", "none", countFields);
        const countConstant = element("div", {"data-count-source": "constant"}, countFields);
        field("count_value", "input", "3", countConstant);
        const countColumn = element("div", {"data-count-source": "column"}, countFields);
        field("count_column", "select", "n", countColumn);
        field("response_units", "input", "fraction remaining");
    };

    document.getElementById = id => root.id === id ? root : root.querySelectorAll("[id]").find(node => node.id === id) || null;
    document.createElement = tag => element(tag);
    const requests = [], errors = [], updates = [], activity = [];
    root.addEventListener("design-input-errors", event => { if (event.detail.errors.length) errors.push(event.detail); });
    root.addEventListener("design-inputs-updated", event => updates.push(event));
    root.addEventListener("design-request-activity", event => activity.push(event.detail));
    class FormDataFixture {
        constructor() { this.values = []; }
        append(name, value) { this.values.push([name, value]); }
        set(name, value) { this.values = this.values.filter(([key]) => key !== name); this.append(name, value); }
        entries() { return this.values[Symbol.iterator](); }
        get(name) { return this.values.find(([key]) => key === name)?.[1] ?? null; }
    }
    class CustomEventFixture {
        constructor(type, options = {}) { this.type = type; Object.assign(this, options); }
    }
    vm.runInNewContext(source, {
        document, FormData: FormDataFixture, CustomEvent: CustomEventFixture,
        fetch(url, options) { return new Promise((resolve, reject) => requests.push({url, options, resolve, reject})); },
    }, {filename: "design-data.js"});

    async function emit(node, type) {
        const event = {type, target: node, preventDefault() { this.defaultPrevented = true; }};
        const ancestors = []; for (let current = node; current; current = current.parentElement) ancestors.push(current);
        const calls = []; for (const current of ancestors) for (const listener of current.listeners[type] || []) calls.push(listener(event));
        await Promise.all(calls);
    }
    const flush = () => new Promise(resolve => setImmediate(resolve));
    async function respond(html, extra = {}, index = requests.length - 1) {
        requests[index].resolve({ok: true, json: async () => ({ok: true, html, dataset_id: dataset.id, filename: dataset.file.files[0]?.name, ...extra})});
        await flush();
    }
    function file(name = "measurements.csv") { return {name, size: 42, lastModified: 123}; }
    function control(name) { return document.getElementById(prefix + name); }
    async function readColumns(token = "CSV_MAPPING") {
        dataset.file.files = [file(token === "EXCEL_MAPPING" ? "measurements.xlsx" : "measurements.csv")];
        await emit(dataset.file, "change");
        await respond(token);
    }
    return {root, form, task, studyGroup, hasCompleted, completedGroup, dataset, completed, requests, errors, updates, activity,
        emit, flush, respond, file, control, readColumns, document};
}

test("a missing file reports its field without sending a request", async () => {
    const page = fixture();
    await page.emit(page.dataset.inspect, "click");
    assert.equal(page.requests.length, 0);
    assert.equal(page.activity.length, 0);
    assert.equal(page.errors.length, 1);
    assert.equal(page.errors[0].errors[0].field_id, page.dataset.id);
});

test("selecting a file automatically reads columns while preserving its input and leaves preview explicit", async () => {
    const page = fixture(), input = page.dataset.file, selected = page.file();
    input.files = [selected];
    await page.emit(input, "change");
    assert.equal(page.requests.length, 1);
    assert.equal(page.requests[0].url, "/data/inspect");
    assert.equal(page.requests[0].options.body.get("file"), selected);
    assert.equal(page.requests[0].options.body.get("dataset_id"), page.dataset.id);
    await page.respond("CSV_MAPPING");
    assert.equal(page.document.getElementById(page.dataset.id), input);
    assert.equal(input.files[0], selected);
    assert(page.control("time_column"));
    assert(page.updates.length > 0);
    assert.equal(page.dataset.preview.innerHTML, '');
    assert.equal(page.requests.length, 1);
    assert.match(page.dataset.datasetSummary.textContent, /measurements\.csv/);
});

test("Excel automatically loads the explicitly selected worksheet and blank choices do not start requests", async () => {
    const page = fixture();
    page.dataset.file.files = [page.file("measurements.xlsx")];
    await page.emit(page.dataset.file, "change");
    await page.respond("SHEETS", {requires_sheet: true, sheets: ["Data", "Other"]});
    const sheet = page.document.getElementById(page.dataset.id + "__sheet");
    assert(sheet);
    assert.equal(page.control("time_column"), null);
    sheet.value = "Data"; await page.emit(sheet, "change");
    assert.equal(page.requests.at(-1).options.body.get("sheet"), "Data");
    await page.respond("EXCEL_MAPPING", {requires_sheet: false, selected_sheet: "Data"});
    const columns = page.dataset.config.querySelector("[data-column-fields]");
    const original = page.control("time_column");
    const changedSheet = page.document.getElementById(page.dataset.id + "__sheet");
    changedSheet.value = "Other"; await page.emit(changedSheet, "change");
    assert.equal(columns.hidden, true);
    assert.equal(original.disabled, true);
    assert.equal(page.requests.length, 3);
    assert.equal(page.requests.at(-1).options.body.get('sheet'), 'Other');
    changedSheet.value = ''; await page.emit(changedSheet, 'change');
    assert.equal(page.requests.length, 3);
    await page.respond('EXCEL_MAPPING', {selected_sheet: 'Other'});
    assert.equal(columns.hidden, true, 'The response for an obsolete sheet cannot reactivate its mapping.');
});

test('a later file selection owns the mapping and activity even when the earlier file finishes last', async () => {
    const page = fixture();
    page.dataset.file.files = [page.file('first.xlsx')];
    await page.emit(page.dataset.file, 'change');
    const earlier = page.requests[0];
    const selected = page.file('latest.csv');
    page.dataset.file.files = [selected];
    await page.emit(page.dataset.file, 'change');
    assert.equal(page.requests.length, 2);
    await page.respond('CSV_MAPPING');
    const current = page.control('time_column');
    earlier.resolve({ok: true, json: async () => ({ok: true, html: 'SHEETS', requires_sheet: true})});
    await page.flush();
    assert.equal(page.control('time_column'), current);
    assert.equal(page.document.getElementById(page.dataset.id + '__sheet'), null);
    assert.equal(page.dataset.file.files[0], selected);
    assert.equal(page.dataset.inspect.disabled, false);
    assert.deepEqual(page.activity.map(item => item.active), [true, true, false]);
    assert.equal(page.errors.length, 0);
});

test("uncertainty controls distinguish means with SD from means with SEM", async () => {
    const page = fixture(); await page.readColumns();
    const mode = page.control("uncertainty");
    assert.equal(page.control("error_value").disabled, true);
    mode.value = "mean_sd"; await page.emit(mode, "change");
    assert.equal(page.control("count_source").value, "constant");
    assert.equal(page.control("count_value").disabled, false);
    assert.equal(page.control("error_value").disabled, false);
    mode.value = "mean_sem"; await page.emit(mode, "change");
    const counts = page.control("count_source"); counts.value = "none"; await page.emit(counts, "change");
    assert.equal(counts.value, "none");
    assert.equal(page.control("count_value").disabled, true);
    mode.value = "individual_sd"; await page.emit(mode, "change");
    assert.equal(counts.disabled, true);
    assert.equal(page.control("error_value").disabled, false);
    assert.equal(page.requests.length, 1);
});

test("preview submits mapped controls and preserves the earlier preview on failure", async () => {
    const page = fixture(); await page.readColumns();
    void page.emit(page.dataset.previewButton, "click");
    assert.equal(page.requests.at(-1).url, "/data/preview");
    assert.equal(page.requests.at(-1).options.body.get(page.dataset.id + "__profile__time_column"), "time");
    await page.respond("<table>First interpreted data</table>");
    const previous = page.dataset.preview.innerHTML;
    page.control("response_column").value = "other"; await page.emit(page.control("response_column"), "change");
    assert.equal(page.dataset.stale.hidden, false);
    assert.equal(page.requests.length, 2);
    void page.emit(page.dataset.previewButton, "click");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [{field_id: page.dataset.id + "__profile__response_column", message: "Choose a response column."}]})});
    await page.flush();
    assert.equal(page.dataset.preview.innerHTML, previous);
    assert.equal(page.errors.at(-1).errors[0].message, "Choose a response column.");
});

test("file replacement clears old mappings while retaining the labeled earlier preview", async () => {
    const page = fixture(); await page.readColumns();
    void page.emit(page.dataset.previewButton, "click");
    await page.respond("<table>Earlier file preview</table>");
    page.dataset.file.files = [page.file("replacement.csv")];
    await page.emit(page.dataset.file, "change");
    assert.equal(page.dataset.config.children.length, 0);
    assert.equal(page.dataset.preview.innerHTML, "<table>Earlier file preview</table>");
    assert.equal(page.dataset.stale.hidden, false);
    assert.equal(page.requests.length, 3);
    assert.equal(page.requests.at(-1).url, '/data/inspect');
    await page.respond('CSV_MAPPING');
});

test("network and malformed-response failures keep the last preview and release import buttons", async () => {
    for (const failure of ["network", "invalid-json"]) {
        const page = fixture(); await page.readColumns();
        const button = page.dataset.previewButton;
        void page.emit(button, "click"); await page.respond("<table>Retained interpreted data</table>");
        void page.emit(button, "click");
        if (failure === "network") page.requests.at(-1).reject(new Error("offline"));
        else page.requests.at(-1).resolve({ok: false, json: async () => { throw new Error("not JSON"); }});
        await page.flush();
        assert.equal(page.dataset.preview.innerHTML, "<table>Retained interpreted data</table>");
        assert.equal(page.dataset.inspect.disabled, false);
        assert.equal(button.disabled, false);
        assert.match(page.errors.at(-1).errors[0].message, /connection/);
    }
});

test("a changed file or mapping discards in-flight responses and stale field errors", async () => {
    const page = fixture(); await page.readColumns();
    const button = page.dataset.previewButton;
    void page.emit(button, "click");
    void page.emit(button, "click");
    assert.equal(page.requests.length, 2);
    page.control("response_column").value = "changed"; await page.emit(page.control("response_column"), "change");
    await page.respond("<table>Obsolete response</table>");
    assert.equal(page.dataset.preview.innerHTML, "");
    void page.emit(button, "click");
    const previous = page.requests.at(-1);
    page.dataset.file.files = [page.file("new.csv")]; await page.emit(page.dataset.file, "change");
    previous.resolve({ok: false, json: async () => ({ok: false, errors: [{field_id: page.dataset.id, message: "Old file invalid."}]})});
    await page.flush();
    assert.equal(page.errors.length, 0);
    assert.equal(page.dataset.preview.innerHTML, "");
    assert.equal(page.dataset.inspect.disabled, true, 'Obsolete completion must not release the latest column request.');
    await page.respond('CSV_MAPPING');
    assert.equal(page.dataset.inspect.disabled, false);
});

test("visibility changes cannot enable inactive dataset or completed-observation controls", async () => {
    const page = fixture(); await page.readColumns();
    assert.equal(page.completed.file.disabled, true);
    page.hasCompleted.checked = true; await page.emit(page.hasCompleted, "change");
    assert.equal(page.completedGroup.hidden, false);
    assert.equal(page.completed.file.disabled, false);
    page.studyGroup.hidden = true;
    page.root.dispatchEvent({type: "design-visibility-updated"});
    assert.equal(page.dataset.file.disabled, true);
    assert.equal(page.control("time_column").disabled, true);
    page.studyGroup.hidden = false;
    page.root.dispatchEvent({type: "design-visibility-updated"});
    assert.equal(page.dataset.file.disabled, false);
    assert.equal(page.control("time_column").disabled, false);
    assert.equal(page.control("error_value").disabled, true);
});

test('file inspection and preview announce request activity only while their requests are pending', async () => {
    const page = fixture();
    page.dataset.file.files = [page.file()];
    void page.emit(page.dataset.inspect, 'click');
    assert.equal(page.activity.length, 1);
    assert.equal(page.activity[0].key, 'import:' + page.dataset.id);
    assert.equal(page.activity[0].active, true);
    assert.match(page.activity[0].message, /reading/i);
    await page.respond('CSV_MAPPING');
    assert.deepEqual(page.activity.map(item => item.active), [true, false]);
    assert.equal(page.activity[1].key, page.activity[0].key);
    page.activity.length = 0;
    void page.emit(page.dataset.previewButton, 'click');
    assert.equal(page.activity.length, 1);
    assert.match(page.activity[0].message, /preview/i);
    await page.respond('<table>Interpreted data</table>');
    assert.deepEqual(page.activity.map(item => item.active), [true, false]);
    assert.equal(page.activity[1].key, 'import:' + page.dataset.id);
});

test('failed and superseded data requests always release their persistent activity indicator', async () => {
    for (const outcome of ['network', 'malformed', 'validation', 'superseded']) {
        const page = fixture();
        await page.readColumns();
        page.activity.length = 0;
        void page.emit(page.dataset.previewButton, 'click');
        assert.equal(page.activity[0].active, true);
        const request = page.requests.at(-1);
        if (outcome === 'network') request.reject(new Error('offline'));
        if (outcome === 'malformed') request.resolve({ok: true, json: async () => { throw new Error('not JSON'); }});
        if (outcome === 'validation') request.resolve({ok: false, json: async () => ({ok: false, errors: [{message: 'Check the mapping.'}]})});
        if (outcome === 'superseded') {
            page.dataset.file.files = [page.file('replacement.csv')];
            await page.emit(page.dataset.file, 'change');
            request.resolve({ok: true, json: async () => ({ok: true, html: '<p>Obsolete preview</p>'})});
        }
        await page.flush();
        if (outcome === 'superseded') {
            assert.deepEqual(page.activity.map(item => item.active), [true, true], outcome);
            await page.respond('CSV_MAPPING');
            assert.deepEqual(page.activity.map(item => item.active), [true, true, false], outcome);
        } else assert.deepEqual(page.activity.map(item => item.active), [true, false], outcome);
        assert.equal(page.activity.at(-1).key, page.activity[0].key, outcome);
        assert.equal(page.dataset.preview.innerHTML, '', outcome);
    }
});

test('concurrent file roles use independent activity keys and each ends only its own request', async () => {
    const page = fixture();
    page.hasCompleted.checked = true;
    await page.emit(page.hasCompleted, 'change');
    page.dataset.file.files = [page.file('reference.csv')];
    page.completed.file.files = [page.file('completed.csv')];
    void page.emit(page.dataset.inspect, 'click');
    void page.emit(page.completed.inspect, 'click');
    assert.equal(page.requests.length, 2);
    assert.deepEqual(page.activity.map(item => item.key), [
        'import:' + page.dataset.id, 'import:' + page.completed.id,
    ]);
    assert(page.activity.every(item => item.active));
    await page.respond('CSV_MAPPING', {}, 0);
    assert.equal(page.activity.length, 3);
    assert.equal(page.activity[2].key, 'import:' + page.dataset.id);
    assert.equal(page.activity[2].active, false);
    assert.equal(page.completed.inspect.disabled, true);
    await page.respond('CSV_MAPPING', {dataset_id: page.completed.id}, 1);
    assert.equal(page.activity.length, 4);
    assert.equal(page.activity[3].key, 'import:' + page.completed.id);
    assert.equal(page.activity[3].active, false);
    assert.equal(page.completed.inspect.disabled, false);
});

test('preview visibility toggles without requests, input updates, or loss of selected files and mappings', async () => {
    const page = fixture();
    assert.equal(page.dataset.toggle.hidden, true);
    assert.equal(page.dataset.preview.hidden, true);
    await page.readColumns();
    assert.equal(page.dataset.toggle.hidden, true);
    const previewButton = page.dataset.previewButton;
    void page.emit(previewButton, 'click');
    await page.respond('<figure>Original file, mapped rows and caption</figure>');
    assert.equal(page.dataset.preview.hidden, false);
    assert.equal(page.dataset.toggle.hidden, false);
    assert.equal(page.dataset.toggle.textContent, 'Hide preview');
    assert.equal(page.dataset.toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(page.dataset.toggle.getAttribute('aria-controls'), page.dataset.preview.id);
    const file = page.dataset.file.files[0];
    const mapping = page.control('response_column');
    const requests = page.requests.length, updates = page.updates.length, activity = page.activity.length;
    await page.emit(page.dataset.toggle, 'click');
    assert.equal(page.dataset.preview.hidden, true);
    assert.equal(page.dataset.toggle.textContent, 'Show preview');
    assert.equal(page.dataset.toggle.getAttribute('aria-expanded'), 'false');
    await page.emit(page.dataset.toggle, 'click');
    assert.equal(page.dataset.preview.hidden, false);
    assert.equal(page.dataset.preview.innerHTML, '<figure>Original file, mapped rows and caption</figure>');
    assert.equal(page.dataset.file.files[0], file);
    assert.equal(page.control('response_column'), mapping);
    assert.equal(mapping.value, 'response');
    assert.equal(mapping.disabled, false);
    assert.equal(page.requests.length, requests);
    assert.equal(page.updates.length, updates, 'Presentation toggles must not mark a comparison as edited.');
    assert.equal(page.activity.length, activity);
});

test('mapping and dataset panels collapse without disabling inputs and validation reopens the relevant controls', async () => {
    const page = fixture();
    await page.readColumns();
    const mapping = page.dataset.mapping;
    assert.equal(mapping.open, true);
    void page.emit(page.dataset.previewButton, 'click');
    await page.respond('<table>Verified interpreted data</table>');
    assert.equal(mapping.open, false);
    assert.equal(page.dataset.panel.open, true);
    assert.equal(page.dataset.preview.hidden, false);
    assert.match(page.dataset.datasetSummary.textContent, /measurements\.csv/);
    assert.match(page.dataset.datasetSummary.textContent, /preview/i);
    const requests = page.requests.length, updates = page.updates.length;
    page.dataset.panel.open = false;
    await page.emit(page.dataset.panel, 'toggle');
    assert.equal(page.control('time_column').disabled, false);
    assert.equal(page.requests.length, requests);
    assert.equal(page.updates.length, updates);
    page.dataset.panel.open = true;
    void page.emit(page.dataset.previewButton, 'click');
    page.dataset.panel.open = false;
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [{
        field_id: page.control('response_column').id, message: 'Review the response mapping.',
    }]})});
    await page.flush();
    assert.equal(page.dataset.panel.open, true);
    assert.equal(mapping.open, true);
    assert.equal(page.dataset.preview.innerHTML, '<table>Verified interpreted data</table>');
});

test('collapsed stale previews retain their original content and can be shown after mapping or file changes', async () => {
    const page = fixture();
    await page.readColumns('EXCEL_MAPPING');
    void page.emit(page.dataset.previewButton, 'click');
    await page.respond('<figure>Original worksheet Data and original column mapping</figure>');
    await page.emit(page.dataset.toggle, 'click');
    page.control('response_column').value = 'changed-response';
    await page.emit(page.control('response_column'), 'change');
    assert.equal(page.dataset.preview.hidden, true);
    assert.equal(page.dataset.stale.hidden, false, 'The stale notice must remain visible outside the collapsed preview.');
    const sheet = page.document.getElementById(page.dataset.id + '__sheet');
    sheet.value = 'Other';
    await page.emit(sheet, 'change');
    const before = page.requests.length;
    await page.emit(page.dataset.toggle, 'click');
    assert.equal(page.requests.length, before);
    assert.equal(page.dataset.preview.hidden, false);
    assert.match(page.dataset.preview.innerHTML, /Original worksheet Data/);
    await page.emit(page.dataset.toggle, 'click');
    page.dataset.file.files = [page.file('replacement.csv')];
    await page.emit(page.dataset.file, 'change');
    assert.equal(page.dataset.preview.hidden, true);
    assert.equal(page.dataset.toggle.hidden, false);
    assert.equal(page.dataset.stale.hidden, false);
    void page.emit(page.dataset.inspect, 'click');
    await page.respond('CSV_MAPPING');
    assert.equal(page.dataset.preview.hidden, true, 'Rereading columns must not reopen old data.');
    assert.match(page.dataset.preview.innerHTML, /Original worksheet Data/);
    await page.emit(page.dataset.toggle, 'click');
    assert.equal(page.dataset.preview.hidden, false);
    assert.equal(page.dataset.stale.hidden, false);
});

test('failed and superseded preview refreshes preserve collapsed earlier data while the toggle remains usable', async () => {
    for (const outcome of ['network', 'malformed', 'validation', 'mapping', 'file', 'inactive']) {
        const page = fixture();
        await page.readColumns();
        const previewButton = page.dataset.previewButton;
        void page.emit(previewButton, 'click');
        await page.respond('<figure>Retained original preview</figure>');
        await page.emit(page.dataset.toggle, 'click');
        void page.emit(previewButton, 'click');
        assert.equal(page.dataset.toggle.disabled, false, outcome);
        const before = page.requests.length;
        await page.emit(page.dataset.toggle, 'click');
        assert.equal(page.dataset.preview.hidden, false);
        await page.emit(page.dataset.toggle, 'click');
        assert.equal(page.requests.length, before, 'Toggling during a refresh must not start another request.');
        const request = page.requests.at(-1);
        if (outcome === 'network') request.reject(new Error('offline'));
        if (outcome === 'malformed') request.resolve({ok: true, json: async () => { throw new Error('not JSON'); }});
        if (outcome === 'validation') request.resolve({ok: false, json: async () => ({ok: false, errors: [{message: 'Check the mapping.'}]})});
        if (outcome === 'mapping') {
            page.control('response_column').value = 'other';
            await page.emit(page.control('response_column'), 'change');
        }
        if (outcome === 'file') {
            page.dataset.file.files = [page.file('replacement.csv')];
            await page.emit(page.dataset.file, 'change');
        }
        if (outcome === 'inactive') {
            page.studyGroup.hidden = true;
            page.root.dispatchEvent({type: 'design-visibility-updated'});
            assert.equal(page.dataset.toggle.disabled, true);
        }
        if (['mapping', 'file', 'inactive'].includes(outcome)) request.resolve({ok: true, json: async () => ({ok: true, html: '<p>Obsolete replacement</p>'})});
        await page.flush();
        assert.equal(page.dataset.preview.innerHTML, '<figure>Retained original preview</figure>', outcome);
        assert.equal(page.dataset.preview.hidden, true, outcome);
        assert.equal(page.dataset.toggle.textContent, 'Show preview', outcome);
        assert.equal(page.dataset.toggle.getAttribute('aria-expanded'), 'false', outcome);
    }
});

test('a successful explicit preview refresh opens the newly interpreted data', async () => {
    const page = fixture();
    await page.readColumns();
    const previewButton = page.dataset.previewButton;
    void page.emit(previewButton, 'click');
    await page.respond('<p>Earlier interpreted data</p>');
    await page.emit(page.dataset.toggle, 'click');
    page.control('response_column').value = 'other';
    await page.emit(page.control('response_column'), 'change');
    void page.emit(previewButton, 'click');
    await page.respond('<p>Updated interpreted data and caption</p>');
    assert.equal(page.dataset.preview.innerHTML, '<p>Updated interpreted data and caption</p>');
    assert.equal(page.dataset.preview.hidden, false);
    assert.equal(page.dataset.toggle.textContent, 'Hide preview');
    assert.equal(page.dataset.toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(page.dataset.stale.hidden, true);
});

test('preview visibility belongs to each dataset and inactive roles cannot change it', async () => {
    const page = fixture();
    await page.readColumns();
    void page.emit(page.dataset.previewButton, 'click');
    await page.respond('<p>Reference preview</p>');
    page.hasCompleted.checked = true;
    await page.emit(page.hasCompleted, 'change');
    page.completed.file.files = [page.file('completed.csv')];
    void page.emit(page.completed.inspect, 'click');
    await page.respond('CSV_MAPPING', {dataset_id: page.completed.id});
    void page.emit(page.completed.previewButton, 'click');
    await page.respond('<p>Completed observations preview</p>', {dataset_id: page.completed.id});
    await page.emit(page.dataset.toggle, 'click');
    assert.equal(page.dataset.preview.hidden, true);
    assert.equal(page.completed.preview.hidden, false);
    page.hasCompleted.checked = false;
    await page.emit(page.hasCompleted, 'change');
    assert.equal(page.completed.toggle.disabled, true);
    const before = page.requests.length;
    await page.emit(page.completed.toggle, 'click');
    assert.equal(page.completed.preview.hidden, false, 'An inactive role must retain, rather than toggle, its preview state.');
    page.hasCompleted.checked = true;
    await page.emit(page.hasCompleted, 'change');
    assert.equal(page.completed.toggle.disabled, false);
    assert.equal(page.completed.preview.hidden, false);
    assert.equal(page.completed.toggle.textContent, 'Hide preview');
    assert.equal(page.requests.length, before);
});
