"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/design.js"), "utf8");

// Minimal DOM for the behavior used by this frontend. No browser or extra
// JavaScript dependency is needed to exercise its request/state handling.
function fixture({planning = false, calibration = false, plots} = {}) {
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
        const uploadGroup = planning && mode === "measured" ? element("div", {"data-measured-prior": ""}, sourceGroups[mode]) : sourceGroups[mode];
        fileInputs[mode] = element("input", {id: mode + "-file", name: mode + "_file", type: "file"}, uploadGroup);
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
    let measuredMode, completedError, completedFile, completedScaleHelp, firstMode, firstTime, futurePolicy, addTiming;
    const timingRows = [];
    if (planning) {
        completed = element("input", {id: "has_completed", name: "has_completed", type: "checkbox", value: "on"}, studyGroup);
        completed.checked = false;
        const completedGroup = element("div", {"data-completed-dataset": "", hidden: ""}, studyGroup);
        completedFile = element("input", {id: "completed-target-file", name: "completed_target_file", type: "file"}, completedGroup);
        const measuredModeGroup = element("div", {"data-measured-adaptive-options": "", hidden: ""}, studyGroup);
        measuredMode = element("select", {id: "measured_adaptive_mode", name: "measured_adaptive_mode", value: "refit_completed"}, measuredModeGroup);
        const adaptiveGroup = element("div", {"data-adaptive-controls": "", hidden: ""}, studyGroup);
        completedError = element("select", {id: "completed_error_mode", name: "completed_error_mode", value: "supplied"}, adaptiveGroup);
        completedScaleHelp = element("p", {"data-completed-scale-help": ""}, adaptiveGroup);
        planningGroup = element("div", {"data-planning-controls": ""}, studyGroup);
        const samplingAdaptive = element("div", {"data-adaptive-controls": "", hidden: ""}, planningGroup);
        firstMode = element("select", {id: "first_time_mode", name: "first_time_mode", value: "specified"}, samplingAdaptive);
        futurePolicy = element("select", {id: "future_policy", name: "future_policy", value: "future_only"}, samplingAdaptive);
        const specified = element("div", {"data-first-specified": ""}, planningGroup);
        firstTime = element("input", {id: "first_time", name: "first_time", value: "0"}, specified);
        addTiming = element("button", {type: "button", "data-add-timing": ""}, planningGroup);
        for (let index = 0; index < 8; index++) {
            const prefix = "timing_" + index + "_";
            const row = element("fieldset", {"data-timing-row": String(index), "data-timing-active": "false", hidden: ""}, planningGroup);
            const kind = element("select", {id: prefix + "kind", name: prefix + "kind", value: "window"}, row);
            const label = element("input", {id: prefix + "label", name: prefix + "label", value: "Requirement " + (index + 1)}, row);
            const bounds = {};
            for (const side of ["lower", "upper"]) {
                const mode = element("select", {id: prefix + side + "_mode", name: prefix + side + "_mode", value: "numeric"}, row);
                const numeric = element("div", {"data-timing-bound": side, "data-timing-mode": "numeric"}, row);
                const time = element("input", {id: prefix + side + "_time", name: prefix + side + "_time", value: side === "lower" ? "0" : "12"}, numeric);
                const milestone = element("div", {"data-timing-bound": side, "data-timing-mode": "milestone"}, row);
                const fraction = element("input", {id: prefix + side + "_fraction", name: prefix + side + "_fraction", value: side === "lower" ? ".5" : ".9"}, milestone);
                const quantile = element("input", {id: prefix + side + "_quantile", name: prefix + side + "_quantile", value: ".25"}, milestone);
                element("p", {id: quantile.id + "-error", "data-error-for": quantile.id, hidden: ""}, milestone);
                bounds[side] = {mode, time, fraction, quantile};
            }
            const windowFields = element("div", {"data-timing-window-fields": ""}, row);
            const count = element("input", {id: prefix + "minimum_count", name: prefix + "minimum_count", value: "1"}, windowFields);
            const scope = element("select", {id: prefix + "count_scope", name: prefix + "count_scope", value: "new"}, windowFields);
            const remove = element("button", {type: "button", "data-remove-timing": String(index)}, row);
            timingRows.push({row, kind, label, bounds, count, scope, remove});
        }
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
    let calibrationControls = null;
    if (calibration) {
        const cal = calibrationControls = {sides: {}, candidates: []};
        function field(id, value, parent, tag = "input", type) {
            return element(tag, {id, name: id, value, ...(type ? {type} : {})}, parent);
        }
        for (const side of ["in_vitro", "in_vivo"]) {
            const container = element("section", {"data-calibration-side": side}, calibrationGroup);
            const modeGroup = element("div", {"data-calibration-mode": side}, container);
            const mode = field("cal_" + side + "_mode", "prospective", modeGroup, "select");
            const existingGroup = element("div", {"data-calibration-existing": side}, container);
            const existing = field("existing_" + side + "_file", "", existingGroup, "input", "file");
            const source = field("cal_" + side + "_source", "existing", container, "select");
            for (const value of ["existing", "file", "none"]) element("option", {value}, source);
            const sourceGroup = element("div", {"data-calibration-source-file": side}, container);
            const sourceFile = field("cal_" + side + "_source_file", "", sourceGroup, "input", "file");
            const sourceControls = element("div", {"data-calibration-source-controls": side}, container);
            const model = field("cal_" + side + "_model", "exponential", sourceControls, "select");
            const first = field("cal_" + side + "_first_time", "0", sourceControls);
            const future = element("div", {"data-calibration-future": side}, container);
            const observation = field("cal_" + side + "_observation_kind", "constant_sd", future, "select");
            const constant = element("div", {"data-calibration-observation-side": side,
                "data-calibration-observation-kinds": "constant_sd constant_cv"}, future);
            const value = field("cal_" + side + "_observation_value", ".05", constant);
            const cv = element("div", {"data-calibration-observation-side": side,
                "data-calibration-observation-kinds": "constant_cv"}, future);
            const minimum = field("cal_" + side + "_minimum_sd", ".001", cv);
            const profileGroup = element("div", {"data-calibration-observation-side": side,
                "data-calibration-observation-kinds": "time_profile_sd"}, future);
            const profile = field("calibration_" + side + "_sd_profile_file", "", profileGroup, "input", "file");
            const historyGroup = element("div", {"data-calibration-historical": side}, container);
            const history = field("cal_" + side + "_historical_error", "supplied", historyGroup, "select");
            cal.sides[side] = {container, mode, existing, source, sourceFile, model, first,
                observation, value, minimum, profile, history};
        }
        cal.approach = field("cal_approach", "approach2", calibrationGroup, "select");
        cal.model = field("cal_model", "exponential", calibrationGroup, "select");
        for (const [value, approaches] of [["linear", "approach1"], ["exponential", "approach1 approach2"], ["normalized", "approach2"]]) {
            element("option", {value, "data-calibration-approaches": approaches}, cal.model);
        }
        cal.method = field("cal_method", "value_ratio", calibrationGroup, "select");
        for (const [approach, methods] of [["approach1", ["time_mapping"]], ["approach2", ["value_ratio", "tau_ratio", "observed_response_inverse"]]]) {
            for (const method of methods) element("option", {value: method, "data-calibration-method-approach": approach}, cal.method);
        }
        const interpolationGroup = element("div", {"data-calibration-interpolation": ""}, calibrationGroup);
        cal.interpolation = field("cal_interpolation", "default", interpolationGroup, "select");
        cal.relationship = field("cal_relationship", "on", calibrationGroup, "input", "checkbox");
        cal.relationship.checked = true;
        const relationshipGroup = element("div", {"data-calibration-relationship": ""}, calibrationGroup);
        const evaluationGroup = element("div", {"data-calibration-evaluation": "time"}, relationshipGroup);
        cal.times = field("cal_evaluation_times", "0,2,4", evaluationGroup);
        const responseGroup = element("div", {"data-calibration-evaluation": "response"}, relationshipGroup);
        cal.levels = field("cal_response_levels", ".9,.5,.1", responseGroup);
        cal.target = field("cal_assess_target", "on", calibrationGroup, "input", "checkbox");
        cal.target.checked = false;
        const targetGroup = element("div", {"data-calibration-target": ""}, calibrationGroup);
        cal.targetFile = field("cal_target_file", "", targetGroup, "input", "file");
        cal.intervalDraws = field("cal_interval_draws", "100", calibrationGroup);
        cal.referenceDraws = field("reference_draws", "100", details);
        cal.add = element("button", {type: "button", "data-add-calibration-candidate": ""}, calibrationGroup);
        cal.generate = element("button", {id: "generate-calibration-candidates", type: "button",
            "data-generation-url": "/generate-candidates"}, calibrationGroup);
        cal.generationStatus = element("p", {id: "calibration-generation-status"}, calibrationGroup);
        cal.generationContent = element("div", {id: "calibration-generation-content"}, calibrationGroup);
        cal.copyButton = index => element("button", {type: "button",
            "data-copy-generated-candidate": String(index)}, cal.generationContent);
        for (let index = 0; index < 6; index++) {
            const prefix = "cal_candidate_" + index + "_";
            const row = element("fieldset", {"data-calibration-candidate": String(index),
                "data-calibration-active": String(index === 0)}, calibrationGroup);
            const name = field(prefix + "name", "Design " + (index + 1), row);
            element("p", {id: name.id + "-error", "data-error-for": name.id, hidden: ""}, row);
            const sides = {};
            for (const side of ["in_vitro", "in_vivo"]) {
                const container = element("div", {"data-calibration-candidate-side": side}, row);
                const kind = field(prefix + side + "_kind", "manual", container, "select");
                const manual = element("div", {"data-calibration-schedule-side": side,
                    "data-calibration-schedule-kind": "manual"}, container);
                const times = field(prefix + side + "_times", "0,4,12", manual);
                const generated = element("div", {"data-calibration-schedule-side": side,
                    "data-calibration-schedule-kind": "generated"}, container);
                const count = field(prefix + side + "_number_of_times", "3", generated);
                const replicates = field(prefix + side + "_replicates", "3", container);
                sides[side] = {kind, times, count, replicates};
            }
            const remove = element("button", {type: "button", "data-remove-calibration-candidate": String(index)}, row);
            cal.candidates.push({row, name, sides, remove});
        }
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
        window: {DesignPlots: plots},
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
        interpolation, scalingModel, extrapolation, ivivcTarget, ratioFile,
        measuredMode, completedError, completedFile, completedScaleHelp, firstMode, firstTime, futurePolicy, addTiming, timingRows,
        calibration: calibrationControls};
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

test("successful calculations dispose the previous plot and initialize the inserted result", async () => {
    const calls = [];
    const plots = {
        purge(node) { calls.push(['purge', node, node.innerHTML]); },
        initialize(node) { calls.push(['initialize', node, node.innerHTML]); },
        resize(node) { calls.push(['resize', node, node.hidden]); },
    };
    const page = fixture({planning: true, plots});
    const file = {name: 'reference.csv', size: 12, lastModified: 1};
    page.fileInputs.measured.files = [file];
    const previous = page.content.innerHTML;
    await page.successfulCalculation('<section>Interactive reference</section>');
    assert.deepEqual(calls.map(call => call[0]), ['purge', 'initialize', 'resize']);
    assert.equal(calls[0][1], page.content);
    assert.equal(calls[0][2], previous);
    assert.equal(calls[1][2], '<section>Interactive reference</section>');
    assert.equal(calls[2][1], page.results);
    assert.equal(calls[2][2], false, 'Resize must happen after the result view becomes visible.');
    assert.equal(page.fileInputs.measured.files[0], file);
    assert.equal(page.requests.length, 1);
});

test("failed calculations retain the previous rendered plot without disposing it", async () => {
    const calls = [];
    const plots = Object.fromEntries(['purge', 'initialize', 'resize'].map(action =>
        [action, () => calls.push(action)]));
    const page = fixture({planning: true, plots});
    await page.successfulCalculation('<section>Keep this figure</section>');
    calls.length = 0;
    const pending = page.emit(page.form, 'submit', {submitter: page.calculate});
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false,
        errors: [{field_id: 'random_seed', message: 'Check the seed.'}]})});
    await pending;
    assert.deepEqual(calls, []);
    assert.equal(page.content.innerHTML, '<section>Keep this figure</section>');
});

test("review and workspace navigation preserve plots and resize without a new calculation", async () => {
    const calls = [];
    const plots = Object.fromEntries(['purge', 'initialize', 'resize'].map(action =>
        [action, () => calls.push(action)]));
    const page = fixture({planning: true, plots});
    await page.successfulCalculation('<section>Numerical plot</section>');
    calls.length = 0;
    await page.successfulReview('<section>Setup review</section>');
    assert.deepEqual(calls, ['resize']);
    assert.equal(page.content.innerHTML, '<section>Numerical plot</section>');
    const status = page.resultStatus.textContent;
    const requests = page.requests.length;
    calls.length = 0;
    await page.emit(page.inputsButton, 'click');
    await page.emit(page.resultsButton, 'click');
    assert.deepEqual(calls, ['resize']);
    assert.equal(page.requests.length, requests);
    assert.equal(page.resultStatus.textContent, status);
});

test("plot initialization errors cannot turn successful calculations into failed requests", async () => {
    for (const asynchronous of [false, true]) {
        const page = fixture({planning: true, plots: {
            purge() { throw new Error('Old plot cleanup failed.'); },
            initialize() {
                if (asynchronous) return Promise.reject(new Error('Plot unavailable.'));
                throw new Error('Plot unavailable.');
            },
            resize() { throw new Error('Plot unavailable.'); },
        }});
        await page.successfulCalculation('<section>Numerical result and static figure</section>');
        assert.equal(page.content.innerHTML, '<section>Numerical result and static figure</section>');
        assert.equal(page.results.hidden, false);
        assert.equal(page.summary.hidden, true);
        assert.equal(page.calculate.disabled, false);
        assert.match(page.resultStatus.textContent, /current inputs/);
        assert.match(page.requestStatus.textContent, /comparison calculated/);
    }
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
    assert.equal(page.calculate.disabled, false);
    assert.equal(page.observationValue.disabled, false);
    assert.equal(page.completedFile.disabled, false);
    assert.equal(page.fileInputs.measured.disabled, true);
    assert.equal(page.requests.length, 0);
    assert.match(page.availability.textContent, /additional sampling/);
    page.completed.checked = false;
    await page.emit(page.completed, "change");
    assert.equal(page.calculate.disabled, false);
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(page.calculate.disabled, false);
    assert.equal(page.observationValue.disabled, true);
    assert.equal(page.requests.length, 0);
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

test("adaptive measured modes retain files and submit the correct reference role", async () => {
    const page = fixture({planning: true});
    const prior = {name: "prior.csv", size: 123, lastModified: 1};
    const completed = {name: "completed.csv", size: 90, lastModified: 2};
    page.fileInputs.measured.files = [prior];
    page.completedFile.files = [completed];
    page.completed.checked = true;
    await page.emit(page.completed, "change");
    assert.equal(page.measuredMode.disabled, false);
    assert.equal(page.completedError.disabled, false);
    assert.match(page.completedScaleHelp.textContent, /original scale/);
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    let values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("measured_adaptive_mode"), "refit_completed");
    assert.equal(values.get("completed_target_file"), completed);
    assert(!values.has("measured_file"));
    assert.equal(values.get("completed_error_mode"), "supplied");
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Refit completed observations</p>"})});
    await pending;
    page.measuredMode.value = "reweight_prior";
    page.completedError.value = "future_model";
    await page.emit(page.measuredMode, "change");
    assert.match(page.completedScaleHelp.textContent, /already on the analyzed/);
    assert.equal(page.fileInputs.measured.files[0], prior);
    assert.equal(page.completedFile.files[0], completed);
    pending = page.emit(page.form, "submit", {submitter: page.calculate});
    values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("measured_file"), prior);
    assert.equal(values.get("completed_target_file"), completed);
    assert.equal(values.get("completed_error_mode"), "future_model");
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Updated prior</p>"})});
    await pending;
    page.completed.checked = false;
    await page.emit(page.completed, "change");
    assert.equal(page.fileInputs.measured.disabled, false);
    assert.equal(page.completedFile.disabled, true);
    assert.equal(page.measuredMode.disabled, true);
    assert.equal(page.completedError.disabled, true);
});

test("latest-completed grid starts and gap-filling policy are explicit without changing stored times", async () => {
    const page = fixture({planning: true});
    page.completed.checked = true;
    page.firstTime.value = "3";
    page.firstMode.value = "latest_completed";
    page.futurePolicy.value = "gap_fill";
    await page.emit(page.completed, "change");
    assert.equal(page.firstTime.disabled, true);
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    const values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("first_time_mode"), "latest_completed");
    assert.equal(values.get("future_policy"), "gap_fill");
    assert(!values.has("first_time"));
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [{message: "Check sampling choices."}]})});
    await pending;
    page.firstMode.value = "specified";
    await page.emit(page.firstMode, "change");
    assert.equal(page.firstTime.disabled, false);
    assert.equal(page.firstTime.value, "3");
    page.sourceField.value = "ivivc";
    await page.emit(page.sourceField, "change");
    assert.equal(page.measuredMode.disabled, true);
    assert.equal(page.completedError.disabled, false);
    assert.match(page.completedScaleHelp.textContent, /update the separate prior once/);
});

test("timing editor submits active numeric or milestone controls and removes rows without losing their settings", async () => {
    const page = fixture({planning: true});
    assert(page.timingRows.every(item => item.row.hidden && item.kind.disabled));
    await page.emit(page.addTiming, "click");
    const first = page.timingRows[0], second = page.timingRows[1];
    assert.equal(first.row.hidden, false);
    assert.equal(page.document.activeElement, first.kind);
    first.bounds.upper.mode.value = "milestone";
    first.bounds.upper.fraction.value = ".5";
    first.bounds.upper.quantile.value = ".25";
    await page.emit(first.bounds.upper.mode, "change");
    await page.emit(page.addTiming, "click");
    second.kind.value = "latest_before";
    second.bounds.upper.time.value = "20";
    await page.emit(second.kind, "change");
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    const values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("timing_0_kind"), "window");
    assert.equal(values.get("timing_0_upper_fraction"), ".5");
    assert.equal(values.get("timing_0_upper_quantile"), ".25");
    assert(!values.has("timing_0_upper_time"));
    assert.equal(values.get("timing_1_kind"), "latest_before");
    assert.equal(values.get("timing_1_upper_time"), "20");
    assert(!values.has("timing_1_minimum_count"));
    assert(!values.has("timing_1_count_scope"));
    assert(!values.has("timing_2_kind"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Timing comparison</p>"})});
    await pending;
    await page.emit(first.remove, "click");
    assert.equal(first.kind.disabled, true);
    assert.equal(second.kind.disabled, false);
    assert.match(page.resultStatus.textContent, /Inputs changed/);
    await page.emit(page.addTiming, "click");
    assert.equal(first.bounds.upper.mode.value, "milestone");
    assert.equal(first.bounds.upper.quantile.value, ".25");
    assert.equal(first.bounds.upper.time.disabled, true);
});

test("timing slots remain bounded and inactive requirements stay disabled across task changes", async () => {
    const page = fixture({planning: true});
    for (let index = 0; index < 8; index++) await page.emit(page.addTiming, "click");
    assert.equal(page.addTiming.disabled, true);
    assert.equal(page.timingRows.filter(item => !item.row.hidden).length, 8);
    await page.emit(page.timingRows[3].remove, "click");
    assert.equal(page.addTiming.disabled, false);
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert(page.timingRows.every(item => item.kind.disabled));
    assert.equal(page.addTiming.disabled, true);
    page.task.value = "study";
    await page.emit(page.task, "change");
    assert.equal(page.timingRows[0].kind.disabled, false);
    assert.equal(page.timingRows[3].kind.disabled, true);
    assert.equal(page.requests.length, 0);
});

test("timing validation targets current controls and ignores obsolete errors after removal", async () => {
    const page = fixture({planning: true});
    await page.successfulCalculation();
    const previous = page.content.innerHTML;
    await page.emit(page.addTiming, "click");
    const row = page.timingRows[0];
    row.bounds.upper.mode.value = "milestone";
    row.bounds.upper.quantile.value = "2";
    await page.emit(row.bounds.upper.mode, "change");
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "timing_0_upper_quantile", message: "Use a quantile between zero and one."},
    ]})});
    await pending;
    assert.equal(row.bounds.upper.quantile.getAttribute("aria-invalid"), "true");
    assert.equal(page.content.innerHTML, previous);
    pending = page.emit(page.form, "submit", {submitter: page.calculate});
    await page.emit(row.remove, "click");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "timing_0_upper_quantile", message: "Obsolete error."},
    ]})});
    await pending;
    assert.notEqual(row.bounds.upper.quantile.getAttribute("aria-invalid"), "true");
    assert(!page.summary.textContent.includes("Obsolete error."));
    assert.equal(page.content.innerHTML, previous);
});

test("calibration comparison enables only the varied environments and their active source roles", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration, vitro = cal.sides.in_vitro, vivo = cal.sides.in_vivo;
    assert.equal(cal.approach.disabled, true);
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(page.calculate.disabled, false);
    assert.equal(page.observationValue.disabled, true);
    assert.equal(page.sourceField.disabled, true);
    assert.equal(cal.referenceDraws.disabled, false);
    assert.equal(vitro.mode.disabled, true);
    assert.equal(vitro.existing.disabled, false);
    assert.equal(vitro.source.value, "existing");
    assert.equal(vitro.sourceFile.disabled, true);
    assert.equal(vitro.value.disabled, true);
    assert.equal(vivo.mode.disabled, false);
    assert.equal(vivo.existing.disabled, true);
    assert.equal(vivo.source.value, "file");
    assert.equal(vivo.sourceFile.disabled, false);
    assert.equal(vivo.value.disabled, false);
    assert.equal(vivo.history.disabled, true);
    assert.equal(cal.candidates[0].sides.in_vitro.replicates.disabled, true);
    assert.equal(cal.candidates[0].sides.in_vivo.replicates.disabled, false);
    vivo.mode.value = "additions";
    vivo.source.value = "existing";
    await page.emit(vivo.mode, "change");
    assert.equal(vivo.existing.disabled, false);
    assert.equal(vivo.history.disabled, false);
    assert.equal(vivo.sourceFile.disabled, true);
    vivo.source.value = "none";
    await page.emit(vivo.source, "change");
    assert.notEqual(vivo.source.value, "none");
    vitro.source.value = "none";
    await page.emit(vitro.source, "change");
    assert.equal(vitro.model.disabled, true);
    assert.equal(vitro.first.disabled, true);
    assert.equal(vitro.existing.disabled, false);
    assert.equal(page.requests.length, 0);
});

test("calibration measurement models exclude inactive fields and preserve selected files", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration, vitro = cal.sides.in_vitro, vivo = cal.sides.in_vivo;
    const historical = {name: "historical.csv", size: 30, lastModified: 1};
    const source = {name: "expected.csv", size: 40, lastModified: 2};
    const profile = {name: "precision.csv", size: 15, lastModified: 3};
    vivo.existing.files = [historical];
    vivo.sourceFile.files = [source];
    vivo.profile.files = [profile];
    page.task.value = "calibration";
    page.environment.value = "both";
    await page.emit(page.task, "change");
    vivo.mode.value = "additions";
    vivo.observation.value = "time_profile_sd";
    vitro.observation.value = "constant_cv";
    await page.emit(vivo.observation, "change");
    assert.equal(vivo.profile.disabled, false);
    assert.equal(vivo.value.disabled, true);
    assert.equal(vivo.minimum.disabled, true);
    assert.equal(vitro.minimum.disabled, false);
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    let values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get(vivo.existing.name), historical);
    assert.equal(values.get(vivo.sourceFile.name), source);
    assert.equal(values.get(vivo.profile.name), profile);
    assert(!values.has(vivo.value.name));
    assert(!values.has(vivo.minimum.name));
    assert.equal(values.get(vitro.minimum.name), ".001");
    assert(!values.has("observation_kind"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Calibration uncertainty</p>"})});
    await pending;
    page.environment.value = "in_vitro";
    await page.emit(page.environment, "change");
    assert.equal(vivo.profile.disabled, true);
    assert.equal(vivo.value.disabled, true);
    assert.equal(vivo.existing.disabled, false);
    assert.equal(vivo.profile.files[0], profile);
    assert.equal(vivo.sourceFile.files[0], source);
    assert.equal(vivo.existing.files[0], historical);
});

test("calibration approach and target choices submit compatible assessment controls", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration;
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(cal.times.disabled, false);
    assert.equal(cal.levels.disabled, true);
    assert.equal(cal.interpolation.disabled, true);
    assert.equal(cal.targetFile.disabled, true);
    cal.model.value = "normalized";
    cal.method.value = "observed_response_inverse";
    cal.target.checked = true;
    await page.emit(cal.method, "change");
    assert.equal(cal.times.disabled, true);
    assert.equal(cal.levels.disabled, false);
    assert.equal(cal.targetFile.disabled, false);
    cal.approach.value = "approach1";
    await page.emit(cal.approach, "change");
    assert.equal(cal.method.value, "time_mapping");
    assert.equal(cal.model.value, "linear");
    assert.equal(cal.interpolation.disabled, false);
    assert.equal(cal.times.disabled, false);
    assert.equal(cal.levels.disabled, true);
    cal.approach.value = "approach2";
    await page.emit(cal.approach, "change");
    assert.equal(cal.model.value, "normalized");
    assert.equal(cal.method.value, "observed_response_inverse");
    cal.method.value = "tau_ratio";
    await page.emit(cal.method, "change");
    assert.equal(cal.times.disabled, true);
    assert.equal(cal.levels.disabled, true);
    cal.relationship.checked = false;
    await page.emit(cal.relationship, "change");
    assert.equal(cal.targetFile.disabled, false);
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    const names = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(names.get("cal_method"), "tau_ratio");
    assert(!names.has("cal_relationship"));
    assert(!names.has("cal_evaluation_times"));
    assert(!names.has("cal_response_levels"));
    assert(!names.has("cal_interpolation"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Target prediction comparison</p>"})});
    await pending;
});

test("calibration candidates retain manual or generated choices while inactive rows stay excluded", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration;
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    assert.equal(cal.candidates[0].remove.disabled, true);
    assert.equal(cal.candidates[1].name.disabled, true);
    await page.emit(cal.add, "click");
    const row = cal.candidates[1], side = row.sides.in_vivo;
    assert.equal(page.document.activeElement, row.name);
    row.name.value = "Generated later times";
    side.kind.value = "generated";
    side.count.value = "7";
    side.times.value = "2,5,8";
    await page.emit(side.kind, "change");
    assert.equal(side.times.disabled, true);
    assert.equal(side.count.disabled, false);
    assert.equal(row.sides.in_vitro.kind.disabled, true);
    await page.emit(row.remove, "click");
    assert.equal(row.name.disabled, true);
    assert.equal(side.count.disabled, true);
    await page.emit(cal.add, "click");
    assert.equal(row.name.value, "Generated later times");
    assert.equal(side.kind.value, "generated");
    assert.equal(side.count.value, "7");
    for (let index = 2; index < 6; index++) await page.emit(cal.add, "click");
    assert.equal(cal.add.disabled, true);
    assert.equal(cal.candidates.filter(item => !item.row.hidden).length, 6);
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    const values = new Map(page.requests.at(-1).options.body.entries());
    assert.equal(values.get("cal_candidate_1_in_vivo_number_of_times"), "7");
    assert(!values.has("cal_candidate_1_in_vivo_times"));
    assert(!values.has("cal_candidate_1_in_vitro_times"));
    assert(!values.has("cal_candidate_6_name"));
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Six candidates</p>"})});
    await pending;
    await page.emit(row.remove, "click");
    assert.match(page.resultStatus.textContent, /Inputs changed/);
});

test("calibration errors retain earlier results and do not mark removed candidates invalid", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration;
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    await page.emit(cal.add, "click");
    await page.successfulCalculation("<p>Existing calibration result</p>");
    let pending = page.emit(page.form, "submit", {submitter: page.calculate});
    assert.equal(page.calculate.disabled, true);
    await page.emit(cal.candidates[1].remove, "click");
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: "cal_candidate_1_name", message: "Obsolete candidate name error"},
    ]})});
    await pending;
    assert.equal(page.content.innerHTML, "<p>Existing calibration result</p>");
    assert.notEqual(cal.candidates[1].name.getAttribute("aria-invalid"), "true");
    assert(!page.summary.textContent.includes("Obsolete candidate name error"));
    pending = page.emit(page.form, "submit", {submitter: page.calculate});
    page.requests.at(-1).reject(new Error("Request failed"));
    await pending;
    assert.equal(page.content.innerHTML, "<p>Existing calibration result</p>");
    assert.equal(page.calculate.disabled, false);
});

test("a calibration request preserves submitted results when the user returns to study planning", async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration;
    const expected = {name: "expected.csv", size: 20, lastModified: 1};
    cal.sides.in_vivo.sourceFile.files = [expected];
    page.task.value = "calibration";
    await page.emit(page.task, "change");
    const pending = page.emit(page.form, "submit", {submitter: page.calculate});
    assert.equal(page.requests.at(-1).url, "/calculate");
    assert.equal(new Map(page.requests.at(-1).options.body.entries()).get("task"), "calibration");
    page.task.value = "study";
    await page.emit(page.task, "change");
    page.requests.at(-1).resolve({ok: true, json: async () => ({ok: true, html: "<p>Submitted calibration candidates</p>"})});
    await pending;
    assert.equal(page.content.innerHTML, "<p>Submitted calibration candidates</p>");
    assert.equal(page.inputs.hidden, false);
    assert.equal(page.observationValue.disabled, false);
    assert.equal(cal.candidates[0].name.disabled, true);
    assert.equal(cal.sides.in_vivo.sourceFile.files[0], expected);
    assert.match(page.resultStatus.textContent, /original inputs/);
    assert.match(page.requestStatus.textContent, /Submitted calibration comparison/);
});

async function generationPage({both = false} = {}) {
    const page = fixture({planning: true, calibration: true});
    page.task.value = 'calibration';
    page.environment.value = both ? 'both' : 'in_vivo';
    await page.emit(page.task, 'change');
    const sides = both ? ['in_vitro', 'in_vivo'] : ['in_vivo'];
    for (const side of sides) page.calibration.candidates[0].sides[side].kind.value = 'generated';
    await page.emit(page.calibration.candidates[0].sides.in_vivo.kind, 'change');
    return page;
}

function generationPayload(schedules = [{candidate_index: 0, side: 'in_vivo', times: [0, 6, 12]}]) {
    return {ok: true, html: '<section>Generated proposed times and source assumptions</section>', schedules};
}

async function successfulGeneration(page, payload = generationPayload()) {
    const pending = page.emit(page.calibration.generate, 'click');
    page.requests.at(-1).resolve({ok: true, json: async () => payload});
    await pending;
    return page.calibration.copyButton(0);
}

test('generation is enabled only for active generated environments', async () => {
    const page = fixture({planning: true, calibration: true});
    const cal = page.calibration;
    assert.equal(cal.generate.disabled, true);
    page.task.value = 'calibration';
    await page.emit(page.task, 'change');
    assert.equal(cal.generate.disabled, true);
    cal.candidates[0].sides.in_vitro.kind.value = 'generated';
    await page.emit(cal.candidates[0].sides.in_vitro.kind, 'change');
    assert.equal(cal.generate.disabled, true);
    cal.candidates[0].sides.in_vivo.kind.value = 'generated';
    await page.emit(cal.candidates[0].sides.in_vivo.kind, 'change');
    assert.equal(cal.generate.disabled, false);
    assert.equal(page.requests.length, 0);
});

test('generation previews without modifying candidates, files, or the saved comparison', async () => {
    const page = await generationPage();
    const cal = page.calibration;
    const file = {name: 'expected.csv', size: 40, lastModified: 2};
    cal.sides.in_vivo.sourceFile.files = [file];
    await page.successfulCalculation('<p>Original comparison remains</p>');
    await page.emit(page.inputsButton, 'click');
    const pending = page.emit(cal.generate, 'click');
    assert.equal(page.requests.at(-1).url, '/generate-candidates');
    assert.equal(new Map(page.requests.at(-1).options.body.entries()).get('cal_in_vivo_source_file'), file);
    assert.equal(page.calculate.disabled, true);
    assert.equal(page.submit.disabled, true);
    assert.equal(cal.generate.disabled, true);
    assert.match(cal.generationStatus.textContent, /without calibration simulations/);
    const count = page.requests.length;
    await page.emit(page.form, 'submit', {submitter: page.calculate});
    assert.equal(page.requests.length, count);
    page.requests.at(-1).resolve({ok: true, json: async () => generationPayload()});
    await pending;
    assert.equal(page.content.innerHTML, '<p>Original comparison remains</p>');
    assert.match(cal.generationContent.innerHTML, /Generated proposed times/);
    assert.match(cal.generationStatus.textContent, /match these inputs/);
    assert.equal(cal.candidates[0].sides.in_vivo.kind.value, 'generated');
    assert.equal(cal.candidates[0].sides.in_vivo.times.value, '0,4,12');
    assert.equal(cal.sides.in_vivo.sourceFile.files[0], file);
    assert.equal(page.calculate.disabled, false);
    assert.equal(page.document.activeElement, cal.generationContent);
});

test('copying a candidate makes only its generated sides editable and marks the comparison stale', async () => {
    const page = await generationPage({both: true});
    const cal = page.calibration;
    await page.emit(cal.add, 'click');
    const second = cal.candidates[1];
    second.sides.in_vivo.times.value = '2, 8';
    await page.emit(second.sides.in_vivo.times, 'input');
    await page.successfulCalculation('<p>Comparison before copying</p>');
    const copy = await successfulGeneration(page, generationPayload([
        {candidate_index: 0, side: 'in_vitro', times: [0, 3, 12]},
        {candidate_index: 0, side: 'in_vivo', times: [1, 5, 10]},
    ]));
    await page.emit(copy, 'click');
    assert.equal(cal.candidates[0].sides.in_vitro.times.value, '0, 3, 12');
    assert.equal(cal.candidates[0].sides.in_vivo.times.value, '1, 5, 10');
    assert.equal(cal.candidates[0].sides.in_vitro.kind.value, 'manual');
    assert.equal(cal.candidates[0].sides.in_vivo.kind.value, 'manual');
    assert.equal(cal.candidates[0].sides.in_vivo.times.disabled, false);
    assert.equal(second.sides.in_vivo.times.value, '2, 8');
    assert.equal(cal.candidates[0].sides.in_vivo.replicates.value, '3');
    assert.match(cal.generationStatus.textContent, /explicitly specified/);
    assert.match(page.resultStatus.textContent, /Inputs changed/);
    assert.equal(page.content.innerHTML, '<p>Comparison before copying</p>');
    assert.equal(copy.disabled, true);
});

test('generation preview becomes stale when a same-name same-size file is replaced', async () => {
    const page = await generationPage();
    const cal = page.calibration;
    const file = {name: 'expected.csv', size: 40, lastModified: 2};
    cal.sides.in_vivo.sourceFile.files = [file];
    const copy = await successfulGeneration(page);
    cal.sides.in_vivo.sourceFile.files = [{...file}];
    await page.emit(cal.sides.in_vivo.sourceFile, 'change');
    assert.equal(copy.disabled, true);
    assert.match(cal.generationStatus.textContent, /Inputs changed/);
    await page.emit(copy, 'click');
    assert.equal(cal.candidates[0].sides.in_vivo.kind.value, 'generated');
    cal.sides.in_vivo.sourceFile.files = [file];
    await page.emit(cal.sides.in_vivo.sourceFile, 'change');
    assert.equal(copy.disabled, false);
});

test('failed generation retains both earlier preview and comparison and restores actions', async () => {
    const page = await generationPage();
    const cal = page.calibration;
    await page.successfulCalculation('<p>Saved comparison</p>');
    await successfulGeneration(page);
    const preview = cal.generationContent.innerHTML;
    let pending = page.emit(cal.generate, 'click');
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: 'cal_candidate_0_in_vivo_number_of_times', message: 'Too many new times.'},
    ]})});
    await pending;
    assert.equal(cal.generationContent.innerHTML, preview);
    assert.equal(page.content.innerHTML, '<p>Saved comparison</p>');
    assert.match(page.summary.textContent, /Too many new times/);
    assert.equal(page.calculate.disabled, false);
    assert.equal(cal.generate.disabled, false);
    pending = page.emit(cal.generate, 'click');
    page.requests.at(-1).reject(new Error('Connection lost'));
    await pending;
    assert.equal(cal.generationContent.innerHTML, preview);
    assert.equal(page.content.innerHTML, '<p>Saved comparison</p>');
    assert.match(cal.generationStatus.textContent, /Check the connection/);
});

test('superseded generated times and field errors cannot replace the current draft or preview', async () => {
    const page = await generationPage();
    const cal = page.calibration;
    await successfulGeneration(page);
    const previous = cal.generationContent.innerHTML;
    let pending = page.emit(cal.generate, 'click');
    cal.candidates[0].sides.in_vivo.count.value = '4';
    await page.emit(cal.candidates[0].sides.in_vivo.count, 'input');
    page.requests.at(-1).resolve({ok: true, json: async () => ({...generationPayload(), html: '<p>Obsolete four times</p>'})});
    await pending;
    assert.equal(cal.generationContent.innerHTML, previous);
    assert.match(cal.generationStatus.textContent, /not applied/);
    assert.equal(cal.candidates[0].sides.in_vivo.kind.value, 'generated');
    pending = page.emit(cal.generate, 'click');
    await page.emit(cal.add, 'click');
    page.requests.at(-1).resolve({ok: false, json: async () => ({ok: false, errors: [
        {field_id: 'cal_candidate_0_name', message: 'Obsolete server error'},
    ]})});
    await pending;
    assert.equal(cal.generationContent.innerHTML, previous);
    assert(!page.summary.textContent.includes('Obsolete server error'));
    assert.notEqual(cal.candidates[0].name.getAttribute('aria-invalid'), 'true');
});

test('changing task during generation cannot apply or copy an obsolete calibration preview', async () => {
    const page = await generationPage();
    const cal = page.calibration;
    const pending = page.emit(cal.generate, 'click');
    page.task.value = 'study';
    await page.emit(page.task, 'change');
    page.requests.at(-1).resolve({ok: true, json: async () => generationPayload()});
    await pending;
    assert.equal(cal.generationContent.innerHTML, '');
    assert.equal(cal.generate.disabled, true);
    assert.equal(page.inputs.hidden, false);
    assert.equal(page.calculate.disabled, false);
});
