"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/design.js"), "utf8");

// Minimal DOM for the behavior used by this frontend. No browser or extra
// JavaScript dependency is needed to exercise its request/state handling.
function fixture() {
    const document = {readyState: "complete", activeElement: null};
    const dataAttribute = key => "data-" + key.replace(/[A-Z]/g, letter => "-" + letter.toLowerCase());

    class Element {
        constructor(tag, attributes = {}) {
            this.tagName = tag.toUpperCase();
            this.attributes = {...attributes};
            this.children = [];
            this.parentElement = null;
            this.listeners = {};
            this.files = [];
            this.value = attributes.value || "";
            this.open = false;
            this._text = "";
            this._html = "";
            this.dataset = new Proxy({}, {
                get: (_, key) => this.attributes[dataAttribute(key)],
                set: (_, key, value) => { this.attributes[dataAttribute(key)] = String(value); return true; },
            });
        }
        get id() { return this.attributes.id || ""; }
        get name() { return this.attributes.name || ""; }
        get disabled() { return this.attributes.disabled !== undefined; }
        set disabled(value) { if (value) this.attributes.disabled = ""; else delete this.attributes.disabled; }
        get hidden() { return this.attributes.hidden !== undefined; }
        set hidden(value) { if (value) this.attributes.hidden = ""; else delete this.attributes.hidden; }
        get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
        set textContent(value) { this._text = String(value); this._html = ""; this.children = []; }
        get innerHTML() { return this._html; }
        set innerHTML(value) { this._html = String(value); this._text = ""; this.children = []; }
        append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); }
        replaceChildren(...children) { this._text = ""; this._html = ""; this.children = []; this.append(...children); }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        getAttribute(name) { return this.attributes[name] ?? null; }
        addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
        dispatchEvent(event) {
            if (!event.target) event.target = this;
            const ancestors = [];
            for (let node = this; node; node = event.bubbles ? node.parentElement : null) ancestors.push(node);
            for (const node of ancestors) for (const listener of node.listeners[event.type] || []) listener(event);
            return true;
        }
        focus() { document.activeElement = this; }
        matches(selector) {
            const attribute = /^\[([^=\]]+)(?:="([^"]*)")?\]$/.exec(selector);
            if (attribute) return this.attributes[attribute[1]] !== undefined &&
                (attribute[2] === undefined || this.attributes[attribute[1]] === attribute[2]);
            return this.tagName.toLowerCase() === selector;
        }
        querySelectorAll(selector) {
            const selectors = selector.split(",").map(part => part.trim());
            const found = [];
            const visit = node => node.children.forEach(child => {
                if (selectors.some(part => child.matches(part))) found.push(child);
                visit(child);
            });
            visit(this);
            return found;
        }
        closest(selector) {
            for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node;
            return null;
        }
    }

    const element = (tag, attributes, parent) => {
        const node = new Element(tag, attributes);
        if (parent) parent.append(node);
        return node;
    };
    const root = element("main", {id: "design-workspace"});
    const form = element("form", {id: "design-setup-form"}, root);
    form.action = "/review";
    const task = element("select", {id: "task", name: "task", value: "study", "aria-describedby": "task-help"}, form);
    const inputsButton = element("button", {"data-design-view": "inputs"}, form);
    const resultsButton = element("button", {"data-design-view": "results"}, form);
    const inputs = element("section", {id: "design-inputs"}, form);
    const summary = element("section", {id: "design-errors", hidden: ""}, inputs);
    const draftStatus = element("p", {id: "design-draft-status"}, inputs);
    const studyGroup = element("div", {"data-design-task": "study"}, inputs);
    const sourceField = element("select", {id: "reference_source", name: "reference_source", value: "measured"}, studyGroup);
    const sourceGroups = {};
    const fileInputs = {};
    for (const mode of ["measured", "ivivc", "scaling"]) {
        sourceGroups[mode] = element("div", {"data-reference-source": mode}, studyGroup);
        fileInputs[mode] = element("input", {id: mode + "-file", name: mode + "_file", type: "file"}, sourceGroups[mode]);
    }
    const calibrationGroup = element("div", {"data-design-task": "calibration"}, inputs);
    const environment = element("select", {id: "varied_environment", name: "varied_environment", value: "in_vivo"}, calibrationGroup);
    const calibrationFile = element("input", {id: "calibration-file", name: "calibration_file", type: "file"}, calibrationGroup);
    const details = element("details", {}, inputs);
    const simulations = element("input", {id: "n_simulations", name: "n_simulations", type: "number", value: "100", "aria-describedby": "n-simulations-help"}, details);
    const seed = element("input", {id: "random_seed", name: "random_seed", type: "number", value: "12345", "aria-describedby": "random-seed-help"}, details);
    const submit = element("button", {id: "review-setup", disabled: ""}, inputs);
    const errors = {};
    for (const field of [task, sourceField, environment, simulations, seed]) {
        errors[field.id] = element("p", {id: field.id + "-error", "data-error-for": field.id, hidden: ""}, inputs);
    }
    const requestStatus = element("p", {id: "design-request-status"}, root);
    const results = element("section", {id: "design-results", hidden: ""}, root);
    const content = element("div", {id: "design-results-content"}, results);
    content.innerHTML = "<p>No review yet.</p>";

    document.getElementById = id => root.id === id ? root :
        root.querySelectorAll("[id]").find(node => node.id === id) || null;
    document.createElement = tag => element(tag, {});
    const requests = [];

    class SubmittedFormData {
        constructor(submittedForm) {
            this.items = [];
            for (const field of submittedForm.querySelectorAll("input, select, textarea")) {
                if (field.disabled || !field.name) continue;
                if (field.getAttribute("type") === "file") {
                    this.items.push([field.name, field.files[0] || {name: "", size: 0, lastModified: 0}]);
                } else this.items.push([field.name, field.value]);
            }
        }
        entries() { return this.items[Symbol.iterator](); }
    }

    vm.runInNewContext(source, {
        document,
        FormData: SubmittedFormData,
        CustomEvent: class {
            constructor(type, options = {}) { this.type = type; Object.assign(this, options); }
        },
        fetch(url, options) {
            return new Promise((resolve, reject) => requests.push({url, options, resolve, reject}));
        },
    }, {filename: "design.js"});

    async function emit(node, type) {
        const event = {type, target: node, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }};
        const ancestors = [];
        for (let parent = node; parent; parent = parent.parentElement) ancestors.push(parent);
        const pending = [];
        for (const parent of ancestors) for (const listener of parent.listeners[type] || []) pending.push(listener(event));
        await Promise.all(pending);
        return event;
    }

    async function successfulReview(html = "<section>Reviewed original settings</section>") {
        const response = emit(form, "submit");
        requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html})});
        await response;
    }

    return {root, form, task, sourceField, sourceGroups, fileInputs, calibrationGroup, environment,
        calibrationFile, details, simulations, seed, inputs, results, content, summary, draftStatus,
        requestStatus, submit, errors, requests, document, inputsButton, resultsButton, emit, successfulReview};
}

test("task and reference changes enable only active inputs without fetching", async () => {
    const page = fixture();
    assert.equal(page.root.dataset.ready, "true");
    assert.equal(page.submit.disabled, false);
    assert.equal(page.inputs.hidden, false);
    assert.equal(page.results.hidden, true);
    assert.equal(page.fileInputs.measured.disabled, false);
    assert.equal(page.fileInputs.ivivc.disabled, true);
    assert.equal(page.fileInputs.scaling.disabled, true);
    assert.equal(page.environment.disabled, true);

    page.sourceField.value = "scaling";
    await page.emit(page.sourceField, "change");
    assert.equal(page.sourceGroups.scaling.hidden, false);
    assert.equal(page.fileInputs.scaling.disabled, false);
    assert.equal(page.fileInputs.measured.disabled, true);

    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(page.environment.disabled, false);
    assert.equal(page.calibrationFile.disabled, false);
    assert.equal(page.sourceField.disabled, true);
    assert(Object.values(page.fileInputs).every(field => field.disabled));

    page.task.value = "study";
    await page.emit(page.task, "change");
    assert.equal(page.fileInputs.scaling.disabled, false);
    assert.equal(page.calibrationFile.disabled, true);
    page.seed.value = "999";
    await page.emit(page.seed, "input");
    assert.equal(page.requests.length, 0);
});

test("successful review replaces only results and preserves selected File objects", async () => {
    const page = fixture();
    const file = {name: "measurements.csv", size: 42, lastModified: 123};
    const fileInput = page.fileInputs.measured;
    fileInput.files = [file];
    const response = page.emit(page.form, "submit");
    assert.equal(page.requests.length, 1);
    assert.equal(page.requests[0].url, "/review");
    assert.equal(page.requests[0].options.method, "POST");
    assert.equal(page.requests[0].options.headers.Accept, "application/json");
    assert.equal(Array.from(page.requests[0].options.body.entries()).find(([name]) => name === "measured_file")[1], file);

    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<section>Measured setup review</section>"})});
    await response;
    assert.equal(page.content.innerHTML, "<section>Measured setup review</section>");
    assert.equal(page.document.getElementById("design-setup-form"), page.form);
    assert.equal(page.document.getElementById("measured-file"), fileInput);
    assert.equal(fileInput.files[0], file);
    assert.equal(page.results.hidden, false);
    assert.equal(page.inputs.hidden, true);
    assert.equal(page.document.activeElement, page.content);
    assert.match(page.draftStatus.textContent, /matches these settings/);

    await page.emit(page.inputsButton, "click");
    page.simulations.value = "120";
    await page.emit(page.simulations, "input");
    assert.match(page.draftStatus.textContent, /Inputs changed/);
    assert.equal(page.content.innerHTML, "<section>Measured setup review</section>");
    assert.equal(page.requests.length, 1);
    page.simulations.value = "100";
    await page.emit(page.simulations, "input");
    assert.match(page.draftStatus.textContent, /matches these settings/);
});

test("server validation links errors, opens advanced fields, and retains earlier results", async () => {
    const page = fixture();
    await page.successfulReview();
    const previous = page.content.innerHTML;
    await page.emit(page.inputsButton, "click");
    page.seed.value = "invalid";
    const failed = page.emit(page.form, "submit");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "random_seed", message: "Enter a whole-number seed."},
        {field_id: "random_seed", message: "The seed must be nonnegative."},
    ]})});
    await failed;
    assert.equal(page.content.innerHTML, previous);
    assert.equal(page.summary.hidden, false);
    assert.equal(page.document.activeElement, page.summary);
    assert.equal(page.seed.getAttribute("aria-invalid"), "true");
    assert.equal(page.errors.random_seed.hidden, false);
    assert.match(page.errors.random_seed.textContent, /whole-number.*nonnegative/);
    assert.match(page.seed.getAttribute("aria-describedby"), /random-seed-help/);
    assert.equal(page.seed.getAttribute("aria-describedby").split(/\s+/).filter(id => id === "random_seed-error").length, 1);
    assert.equal(page.inputs.hidden, false);
    const link = page.summary.querySelectorAll("a")[0];
    const clicked = await page.emit(link, "click");
    assert.equal(clicked.defaultPrevented, true);
    assert.equal(page.details.open, true);
    assert.equal(page.document.activeElement, page.seed);

    page.seed.value = "12345";
    await page.successfulReview("<section>Corrected review</section>");
    assert.equal(page.summary.hidden, true);
    assert.equal(page.seed.getAttribute("aria-invalid"), "false");
    assert.equal(page.errors.random_seed.hidden, true);
    assert.equal(page.seed.getAttribute("aria-describedby"), "random-seed-help");
});

test("network and malformed-response failures preserve existing content and release the form", async () => {
    for (const failure of ["network", "invalid-json", "missing-html"]) {
        const page = fixture();
        await page.successfulReview("<section>Retained review</section>");
        const request = page.emit(page.form, "submit");
        if (failure === "network") page.requests.at(-1).reject(new Error("offline"));
        if (failure === "invalid-json") page.requests.at(-1).resolve({ok: false, json: async () => { throw new Error("not JSON"); }});
        if (failure === "missing-html") page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true})});
        await request;
        assert.equal(page.content.innerHTML, "<section>Retained review</section>");
        assert.equal(page.submit.disabled, false);
        assert.equal(page.form.getAttribute("aria-busy"), "false");
        assert.equal(page.summary.hidden, false);
        assert.match(page.summary.textContent, /connection/);
        await page.emit(page.resultsButton, "click");
        assert.equal(page.results.hidden, false);
    }
});

test("only one review request runs at a time and busy state is restored", async () => {
    const page = fixture();
    const first = page.emit(page.form, "submit");
    assert.equal(page.submit.disabled, true);
    assert.equal(page.form.getAttribute("aria-busy"), "true");
    await page.emit(page.form, "submit");
    assert.equal(page.requests.length, 1);
    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<p>First review</p>"})});
    await first;
    assert.equal(page.submit.disabled, false);
    assert.equal(page.form.getAttribute("aria-busy"), "false");
    await page.successfulReview("<p>Second review</p>");
    assert.equal(page.requests.length, 2);
    assert.equal(page.content.innerHTML, "<p>Second review</p>");
});

test("editing while a request runs retains captured settings and leaves current inputs visible", async () => {
    const page = fixture();
    const pending = page.emit(page.form, "submit");
    assert.equal(Array.from(page.requests[0].options.body.entries()).find(([name]) => name === "random_seed")[1], "12345");
    page.seed.value = "999";
    await page.emit(page.seed, "input");
    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<section>Submitted random seed: 12345</section>"})});
    await pending;
    assert.equal(page.seed.value, "999");
    assert.equal(page.content.innerHTML, "<section>Submitted random seed: 12345</section>");
    assert.equal(page.inputs.hidden, false);
    assert.equal(page.results.hidden, true);
    assert.match(page.requestStatus.textContent, /Inputs changed during the request/);
    assert.match(page.draftStatus.textContent, /retains its original settings/);
    await page.emit(page.resultsButton, "click");
    assert.equal(page.content.innerHTML, "<section>Submitted random seed: 12345</section>");
});

test("failed submitted settings do not mark changed or inactive fields as invalid", async () => {
    const page = fixture();
    await page.successfulReview("<section>Retained review</section>");
    const pending = page.emit(page.form, "submit");
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "reference_source", message: "Invalid submitted reference source."},
    ]})});
    await pending;
    assert.equal(page.content.innerHTML, "<section>Retained review</section>");
    assert.equal(page.task.value, "calibration");
    assert.equal(page.sourceField.disabled, true);
    assert.notEqual(page.sourceField.getAttribute("aria-invalid"), "true");
    assert.equal(page.summary.querySelectorAll("a").length, 0);
    assert.match(page.summary.textContent, /Inputs changed during the request/);
    assert.equal(page.inputs.hidden, false);
});
