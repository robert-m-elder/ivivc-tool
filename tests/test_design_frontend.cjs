"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/design.js"), "utf8");

// Minimal DOM for the behavior used by this frontend. No browser or extra
// JavaScript dependency is needed to exercise its request/state handling.
function fixture({planning = false} = {}) {
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
    let calculate, completed, planningGroup, observationKind, observationValue, minimumSd, profileFile;
    let availability, ivivcGroup, scalingGroup, ivivcApproach, ivivcModel, ivivcMethod;
    let interpolation, scalingModel, extrapolation, ivivcTarget, ratioFile;
    if (planning) {
        completed = element("input", {id: "has_completed", name: "has_completed", type: "checkbox", value: "on"}, studyGroup);
        completed.checked = false;
        planningGroup = element("div", {"data-planning-controls": ""}, studyGroup);
        ivivcGroup = element("div", {"data-reference-options": "ivivc"}, planningGroup);
        ivivcApproach = element("select", {id: "ivivc_approach", name: "ivivc_approach", value: "approach2"}, ivivcGroup);
        const modelGroup = element("div", {"data-ivivc-model-fields": ""}, ivivcGroup);
        ivivcModel = element("select", {id: "ivivc_model", name: "ivivc_model", value: "exponential"}, modelGroup);
        element("option", {value: "linear", "data-ivivc-approaches": "approach1"}, ivivcModel);
        element("option", {value: "exponential", "data-ivivc-approaches": "approach1 approach2"}, ivivcModel);
        element("option", {value: "normalized", "data-ivivc-approaches": "approach2"}, ivivcModel);
        ivivcMethod = element("select", {id: "ivivc_method", name: "ivivc_method", value: "value_ratio"}, ivivcGroup);
        for (const [approach, methods] of [["approach1", ["time_mapping"]], ["approach2", ["value_ratio", "tau_ratio", "observed_response_inverse"]], ["approach3", ["value_scaling", "time_scaling"]]]) {
            for (const method of methods) element("option", {value: method, "data-ivivc-method-approach": approach}, ivivcMethod);
        }
        const interpolationGroup = element("div", {"data-ivivc-interpolation-fields": ""}, ivivcGroup);
        interpolation = element("select", {id: "ivivc_interpolation", name: "ivivc_interpolation", value: "default"}, interpolationGroup);
        scalingGroup = element("div", {"data-reference-options": "scaling"}, planningGroup);
        scalingModel = element("select", {id: "scaling_model", name: "scaling_model", value: "exponential"}, scalingGroup);
        extrapolation = element("input", {id: "scaling_allow_extrapolation", name: "scaling_allow_extrapolation", type: "checkbox", value: "on"}, scalingGroup);
        extrapolation.checked = false;
        ivivcTarget = element("input", {name: "prediction_file", type: "file"}, sourceGroups.ivivc);
        ratioFile = element("input", {name: "time_scale_file", type: "file"}, sourceGroups.scaling);
        observationKind = element("select", {id: "observation_kind", name: "observation_kind", value: "constant_sd"}, planningGroup);
        const constant = element("div", {"data-observation-kind": "constant_sd constant_cv"}, planningGroup);
        observationValue = element("input", {id: "observation_value", name: "observation_value", value: ".05"}, constant);
        const cv = element("div", {"data-observation-kind": "constant_cv"}, planningGroup);
        minimumSd = element("input", {id: "minimum_sd", name: "minimum_sd", value: ".001"}, cv);
        const profile = element("div", {"data-observation-kind": "time_profile_sd"}, planningGroup);
        profileFile = element("input", {id: "profile-file", name: "profile_file", type: "file"}, profile);
        calculate = element("button", {id: "calculate-comparison", formaction: "/calculate", disabled: ""}, inputs);
        availability = element("p", {"data-planning-availability": ""}, inputs);
    }
    const errors = {};
    for (const field of [task, sourceField, environment, simulations, seed]) {
        errors[field.id] = element("p", {id: field.id + "-error", "data-error-for": field.id, hidden: ""}, inputs);
    }
    const requestStatus = element("p", {id: "design-request-status"}, root);
    const results = element("section", {id: "design-results", hidden: ""}, root);
    const content = element("div", {id: "design-results-content"}, results);
    content.innerHTML = "<p>No review yet.</p>";
    const reviewContent = planning ? element("div", {id: "design-review-content"}, results) : content;
    const resultStatus = planning ? element("p", {id: "design-results-status"}, results) : null;

    document.getElementById = id => root.id === id ? root :
        root.querySelectorAll("[id]").find(node => node.id === id) || null;
    document.createElement = tag => element(tag, {});
    const requests = [];

    class SubmittedFormData {
        constructor(submittedForm) {
            this.items = [];
            for (const field of submittedForm.querySelectorAll("input, select, textarea")) {
                if (field.disabled || !field.name) continue;
                if (["checkbox", "radio"].includes(field.getAttribute("type")) && !field.checked) continue;
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

    async function emit(node, type, extra = {}) {
        const event = {type, target: node, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra};
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

    async function successfulCalculation(html = "<section>Calculated original settings</section>") {
        const response = emit(form, "submit", {submitter: calculate});
        requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html})});
        await response;
    }

    return {root, form, task, sourceField, sourceGroups, fileInputs, calibrationGroup, environment,
        calibrationFile, details, simulations, seed, inputs, results, content, summary, draftStatus,
        requestStatus, submit, errors, requests, document, inputsButton, resultsButton, emit, successfulReview,
        calculate, completed, planningGroup, observationKind, observationValue, minimumSd, profileFile,
        availability, reviewContent, resultStatus, successfulCalculation,
        ivivcGroup, scalingGroup, ivivcApproach, ivivcModel, ivivcMethod,
        interpolation, scalingModel, extrapolation, ivivcTarget, ratioFile};
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

test("calculation posts the mounted form to its own endpoint and preserves selected files", async () => {
    const page = fixture({planning: true});
    const file = {name: "reference.csv", size: 42, lastModified: 123};
    page.fileInputs.measured.files = [file];
    assert.equal(page.calculate.disabled, false);
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    assert.equal(page.requests[0].url, "/calculate");
    assert.equal(page.requests[0].options.method, "POST");
    assert.equal(Array.from(page.requests[0].options.body.entries()).find(([name]) => name === "measured_file")[1], file);
    assert.equal(page.submit.disabled, true);
    assert.equal(page.calculate.disabled, true);
    await page.emit(page.form, "submit");
    assert.equal(page.requests.length, 1);
    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<section>Curve recovery</section>"})});
    await pending;
    assert.equal(page.content.innerHTML, "<section>Curve recovery</section>");
    assert.equal(page.fileInputs.measured.files[0], file);
    assert.equal(page.results.hidden, false);
    assert.equal(page.document.activeElement, page.content);
    assert.match(page.resultStatus.textContent, /current inputs/);
    assert.equal(page.calculate.disabled, false);
});

test("setup review never replaces an existing numerical comparison or its freshness", async () => {
    const page = fixture({planning: true});
    await page.successfulCalculation("<section>Simulation seed 12345</section>");
    page.seed.value = "999";
    await page.emit(page.seed, "change");
    await page.successfulReview("<section>Reviewed seed 999</section>");
    assert.equal(page.content.innerHTML, "<section>Simulation seed 12345</section>");
    assert.equal(page.reviewContent.innerHTML, "<section>Reviewed seed 999</section>");
    assert.match(page.resultStatus.textContent, /original inputs/);
    assert.match(page.draftStatus.textContent, /Inputs changed/);
});

test("calculation availability follows source and completed-study choices without running requests", async () => {
    const page = fixture({planning: true});
    for (const mode of ["ivivc", "scaling"]) {
        page.sourceField.value = mode;
        await page.emit(page.sourceField, "change");
        assert.equal(page.calculate.disabled, false);
        assert.equal(page.observationKind.disabled, false);
        assert.equal(page.submit.disabled, false);
        assert.match(page.availability.textContent, /selected reference source/);
    }
    page.sourceField.value = "measured";
    await page.emit(page.sourceField, "change");
    assert.equal(page.calculate.disabled, false);
    page.completed.checked = true;
    await page.emit(page.completed, "change");
    assert.equal(page.calculate.disabled, true);
    assert.equal(page.observationValue.disabled, true);
    await page.emit(page.form, "submit", {submitter: page.calculate});
    assert.equal(page.requests.length, 0);
    assert.match(page.summary.textContent, /without completed/);
    page.completed.checked = false;
    await page.emit(page.completed, "change");
    assert.equal(page.calculate.disabled, false);
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(page.calculate.disabled, true);
});

test("only the selected future-error controls are submitted after repeated visibility updates", async () => {
    const page = fixture({planning: true});
    assert.equal(page.observationValue.disabled, false);
    assert.equal(page.minimumSd.disabled, true);
    assert.equal(page.profileFile.disabled, true);
    page.observationKind.value = "constant_cv";
    await page.emit(page.observationKind, "change");
    assert.equal(page.observationValue.disabled, false);
    assert.equal(page.minimumSd.disabled, false);
    page.observationKind.value = "time_profile_sd";
    await page.emit(page.observationKind, "change");
    assert.equal(page.observationValue.disabled, true);
    assert.equal(page.minimumSd.disabled, true);
    assert.equal(page.profileFile.disabled, false);
    page.seed.value = "456";
    await page.emit(page.seed, "change");
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    const names = Array.from(page.requests[0].options.body.entries()).map(([name]) => name);
    assert(names.includes("profile_file"));
    assert(!names.includes("observation_value"));
    assert(!names.includes("minimum_sd"));
    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<p>Profile calculation</p>"})});
    await pending;
});

test("a completed calculation retains submitted settings when the current draft changes", async () => {
    const page = fixture({planning: true});
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    page.seed.value = "999";
    await page.emit(page.seed, "input");
    page.requests[0].resolve({ok: true, json: async () => ({ok: true, html: "<section>Seed 12345</section>"})});
    await pending;
    assert.equal(page.seed.value, "999");
    assert.equal(page.content.innerHTML, "<section>Seed 12345</section>");
    assert.equal(page.inputs.hidden, false);
    assert.match(page.resultStatus.textContent, /original inputs/);
    assert.match(page.requestStatus.textContent, /Inputs changed during the request/);
});

test("a same-name replacement File makes calculation output stale and prevents obsolete field errors", async () => {
    const page = fixture({planning: true});
    page.fileInputs.measured.files = [{name: "reference.csv", size: 42, lastModified: 123}];
    await page.successfulCalculation("<section>Original file results</section>");
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    page.fileInputs.measured.files = [{name: "reference.csv", size: 42, lastModified: 123}];
    await page.emit(page.fileInputs.measured, "change");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "random_seed", message: "An obsolete field error"},
    ]})});
    await pending;
    assert.equal(page.content.innerHTML, "<section>Original file results</section>");
    assert.match(page.resultStatus.textContent, /original inputs/);
    assert.notEqual(page.seed.getAttribute("aria-invalid"), "true");
    assert.equal(page.summary.querySelectorAll("a").length, 0);
    assert.match(page.summary.textContent, /Inputs changed during the request/);
    assert.equal(page.calculate.disabled, false);
});

test("failed calculations retain both earlier numerical results and setup review", async () => {
    for (const failure of ["network", "invalid-json", "missing-html", "validation"]) {
        const page = fixture({planning: true});
        await page.successfulCalculation("<section>Earlier comparison</section>");
        await page.successfulReview("<section>Earlier review</section>");
        const pending = page.emit(page.form, "submit", {submitter: page.calculate});
        const request = page.requests.at(-1);
        if (failure === "network") request.reject(new Error("offline"));
        if (failure === "invalid-json") request.resolve({ok: false, json: async () => { throw new Error("not JSON"); }});
        if (failure === "missing-html") request.resolve({ok: true, json: async () => ({ok: true})});
        if (failure === "validation") request.resolve({ok: false, json: async () => ({ok: false, errors: [{field_id: "random_seed", message: "Invalid seed"}]})});
        await pending;
        assert.equal(page.content.innerHTML, "<section>Earlier comparison</section>");
        assert.equal(page.reviewContent.innerHTML, "<section>Earlier review</section>");
        assert.equal(page.form.getAttribute("aria-busy"), "false");
        assert.equal(page.calculate.disabled, false);
        assert.equal(page.submit.disabled, false);
        assert.match(page.summary.textContent, /planning comparison/);
    }
});

test("other reference sources keep shared planning controls active and submit only their own inputs", async () => {
    const page = fixture({planning: true});
    const target = {name: "target.csv", size: 20, lastModified: 1};
    const ratios = {name: "ratios.csv", size: 30, lastModified: 2};
    page.ivivcTarget.files = [target];
    page.ratioFile.files = [ratios];
    page.sourceField.value = "ivivc";
    await page.emit(page.sourceField, "change");
    assert.equal(page.ivivcGroup.hidden, false);
    assert.equal(page.scalingGroup.hidden, true);
    assert.equal(page.ivivcApproach.disabled, false);
    assert.equal(page.scalingModel.disabled, true);
    assert.equal(page.observationKind.disabled, false);
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    let values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("prediction_file"), target);
    assert(!values.has("measured_file"));
    assert(!values.has("time_scale_file"));
    assert(!values.has("scaling_model"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>IVIVC comparison</p>"})});
    await pending;
    page.sourceField.value = "scaling";
    await page.emit(page.sourceField, "change");
    assert.equal(page.ivivcApproach.disabled, true);
    assert.equal(page.scalingModel.disabled, false);
    assert.equal(page.extrapolation.checked, false);
    pending = page.emit(page.form, "submit", {submitter: page.calculate});
    values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("time_scale_file"), ratios);
    assert(!values.has("prediction_file"));
    assert(!values.has("ivivc_approach"));
    assert(!values.has("scaling_allow_extrapolation"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Scaled comparison</p>"})});
    await pending;
    assert.equal(page.ivivcTarget.files[0], target);
    assert.equal(page.ratioFile.files[0], ratios);
});

test("IVIVC approach changes filter compatible choices and retain each approach's selections", async () => {
    const page = fixture({planning: true});
    page.sourceField.value = "ivivc";
    await page.emit(page.sourceField, "change");
    assert.equal(page.interpolation.disabled, true);
    page.ivivcModel.value = "normalized";
    page.ivivcMethod.value = "tau_ratio";
    await page.emit(page.ivivcMethod, "change");
    page.ivivcApproach.value = "approach1";
    await page.emit(page.ivivcApproach, "change");
    assert.equal(page.ivivcModel.value, "linear");
    assert.equal(page.ivivcMethod.value, "time_mapping");
    assert.equal(page.interpolation.disabled, false);
    assert(page.ivivcMethod.querySelectorAll("option").filter(option => !option.disabled)
        .every(option => option.dataset.ivivcMethodApproach === "approach1"));
    page.ivivcModel.value = "exponential";
    page.ivivcApproach.value = "approach3";
    await page.emit(page.ivivcApproach, "change");
    assert.equal(page.ivivcModel.disabled, true);
    assert.equal(page.ivivcMethod.value, "value_scaling");
    assert.equal(page.interpolation.disabled, false);
    page.ivivcMethod.value = "time_scaling";
    page.ivivcApproach.value = "approach2";
    await page.emit(page.ivivcApproach, "change");
    assert.equal(page.ivivcModel.value, "normalized");
    assert.equal(page.ivivcMethod.value, "tau_ratio");
    assert.equal(page.ivivcModel.disabled, false);
    assert.equal(page.interpolation.disabled, true);
    page.ivivcApproach.value = "approach3";
    await page.emit(page.ivivcApproach, "change");
    assert.equal(page.ivivcMethod.value, "time_scaling");
    assert.equal(page.requests.length, 0);
});

test("Approach 3 excludes a model and Approach 2 excludes interpolation from calculation", async () => {
    const page = fixture({planning: true});
    page.sourceField.value = "ivivc";
    page.ivivcApproach.value = "approach3";
    await page.emit(page.sourceField, "change");
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    let values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("ivivc_method"), "value_scaling");
    assert.equal(values.get("ivivc_interpolation"), "default");
    assert(!values.has("ivivc_model"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Deterministic reference</p>"})});
    await pending;
    page.ivivcApproach.value = "approach2";
    await page.emit(page.ivivcApproach, "change");
    pending = page.emit(page.form, "submit", {submitter: page.calculate});
    values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("ivivc_model"), "exponential");
    assert.equal(values.get("ivivc_method"), "value_ratio");
    assert(!values.has("ivivc_interpolation"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Weighted reference</p>"})});
    await pending;
});

test("changing reference source during calculation keeps its successful output labeled with original inputs", async () => {
    const page = fixture({planning: true});
    page.sourceField.value = "ivivc";
    await page.emit(page.sourceField, "change");
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    page.sourceField.value = "scaling";
    await page.emit(page.sourceField, "change");
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Submitted IVIVC reference</p>"})});
    await pending;
    assert.equal(page.content.innerHTML, "<p>Submitted IVIVC reference</p>");
    assert.equal(page.sourceField.value, "scaling");
    assert.equal(page.scalingModel.disabled, false);
    assert.equal(page.inputs.hidden, false);
    assert.match(page.resultStatus.textContent, /original inputs/);
    assert.equal(page.calculate.disabled, false);
    assert.equal(page.requests.length, 1);
});
