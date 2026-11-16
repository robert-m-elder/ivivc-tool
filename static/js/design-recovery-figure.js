/* Plotly specifications from retained recovery summaries and one shared study. */
(() => {
    'use strict';

    const palette = ['#24535c', '#b34f00', '#7550a2'];
    const neutral = '#62717c';
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
    const array = value => Array.isArray(value) ? value : [];
    const count = value => Number.isInteger(value) && value >= 0 ? value : 0;

    function selected(inspection, selection) {
        if (!inspection || inspection.schema_version !== 1 || !inspection.recovery ||
                !selection || !Array.isArray(selection.selectedIds)) return [];
        const wanted = new Set(selection.selectedIds);
        return array(inspection.recovery.schedules).filter(row => row && wanted.has(row.id)).slice(0, 3);
    }

    function colorFor(selection, row, index) {
        const requested = selection.colors && selection.colors[row.id];
        return typeof requested === 'string' && /^#[0-9a-f]{6}$/i.test(requested) ? requested : palette[index];
    }

    function pairs(times, values, usable) {
        const x = [], y = [];
        array(times).forEach((time, index) => {
            const valid = finite(time) && finite(array(values)[index]) && (!usable || usable(index));
            x.push(valid ? time : null);
            y.push(valid ? values[index] : null);
        });
        return {x, y};
    }

    function line(times, values, {name, color = neutral, dash = 'solid', uid, meta, customdata, hover, usable}) {
        const points = pairs(times, values, usable);
        const available = points.x.filter(finite).length;
        if (!available) return null;
        return {
            type: 'scatter', mode: available > 1 ? 'lines' : 'markers',
            ...points, name: escape(name), uid, legendgroup: uid,
            line: {color, width: 2, dash}, marker: {color, size: 6},
            connectgaps: false, showlegend: false, meta,
            ...(customdata ? {customdata} : {}),
            hovertemplate: hover || 'In vivo time: %{x:.8g}<br>Response: %{y:.8g}<extra>%{fullData.name}</extra>',
        };
    }

    function layout(title, yLabel, identifier, meta = {}) {
        return {
            title: {text: escape(title), x: 0},
            xaxis: {title: {text: 'In vivo time'}, showgrid: false, zeroline: false, automargin: true},
            yaxis: {title: {text: escape(yLabel)}, gridcolor: '#e5ebef', zeroline: false, automargin: true},
            font: {family: 'Arial, sans-serif', color: '#20313d'},
            paper_bgcolor: 'white', plot_bgcolor: 'white',
            margin: {l: 70, r: 25, t: 60, b: 65},
            height: 380, showlegend: false,
            hovermode: 'closest', uirevision: identifier, meta,
        };
    }

    function legendName(value) {
        let remaining = String(value ?? '');
        const lines = [];
        while (remaining.length > 24) {
            let boundary = remaining.lastIndexOf(' ', 24);
            if (boundary < 8) boundary = 24;
            lines.push(remaining.slice(0, boundary));
            remaining = remaining.slice(boundary).trimStart();
        }
        lines.push(remaining);
        return lines.map(escape).join('<br>');
    }

    function summaryLegend(figureLayout, data) {
        const lineCount = data.filter(trace => trace.showlegend).reduce((total, trace) =>
            total + trace.name.split('<br>').length, 0);
        // Reserve enough room even when all entries wrap to separate rows on
        // a narrow screen. Keep the original 255px plotting area above it.
        const bottom = 65 + 18 * lineCount;
        figureLayout.margin.b = bottom;
        figureLayout.height = 315 + bottom;
        figureLayout.showlegend = true;
        figureLayout.legend = {orientation: 'h', x: 0, y: -45 / 255, yanchor: 'top',
            font: {size: 11}, groupclick: 'togglegroup',
            // Plotly remembers visibility per trace, not per group. Reset it
            // when a different view/selection changes the range segments, so
            // a hidden curve cannot leave a newly added segment visible.
            uirevision: figureLayout.uirevision + ':' + data.map(trace => trace.uid).join('|')};
    }

    function ranges(times, summary, row, color) {
        const segments = [];
        let segment = [];
        array(times).forEach((time, index) => {
            const lower = array(summary.lower)[index], upper = array(summary.upper)[index];
            const usable = count(array(summary.available_counts)[index]);
            if (finite(time) && finite(lower) && finite(upper) && lower <= upper && usable >= 2) {
                segment.push({time, lower, upper, usable});
            } else if (segment.length) {
                segments.push(segment);
                segment = [];
            }
        });
        if (segment.length) segments.push(segment);
        const red = parseInt(color.slice(1, 3), 16), green = parseInt(color.slice(3, 5), 16), blue = parseInt(color.slice(5, 7), 16);
        const data = [];
        segments.forEach((points, index) => {
            const common = {
                type: 'scatter', mode: 'lines', x: points.map(point => point.time),
                name: escape(row.name) + ' — across-study central 95% range',
                legendgroup: 'recovery-' + row.id, line: {width: 0},
                connectgaps: false, showlegend: false,
            };
            data.push({...common, y: points.map(point => point.lower),
                uid: row.id + '-range-' + index + '-lower', hoverinfo: 'skip',
                meta: {kind: 'range_lower', schedule_id: row.id}});
            data.push({...common, y: points.map(point => point.upper),
                uid: row.id + '-range-' + index + '-upper', fill: 'tonexty',
                fillcolor: 'rgba(' + red + ',' + green + ',' + blue + ',0.13)',
                customdata: points.map(point => [point.lower, point.upper, point.usable]),
                hovertemplate: 'In vivo time: %{x:.8g}<br>Across-study 2.5th–97.5th percentiles: ' +
                    '%{customdata[0]:.8g} to %{customdata[1]:.8g}<br>Usable successful studies: %{customdata[2]}' +
                    '<br>Successful fits: ' + count(row.n_successful) + '/' + count(row.n_simulations) + '<extra>%{fullData.name}</extra>',
                meta: {kind: 'range_upper', schedule_id: row.id}});
        });
        return data;
    }

    function summary(inspection, selection, options = {}) {
        const rows = selected(inspection, selection);
        if (!rows.length) return null;
        const view = options.view ?? 'response';
        if (!['response', 'error'].includes(view)) return null;
        const data = [], medians = [];
        rows.forEach((row, index) => {
            if (count(row.n_successful) === 0) return;
            const values = row[view === 'error' ? 'recovery_error' : 'recovered_response'];
            if (!values) return;
            const color = colorFor(selection, row, index);
            data.push(...ranges(row.evaluation_times, values, row, color));
            const trace = line(row.evaluation_times, values.median, {
                name: row.name,
                color, uid: row.id + '-median',
                usable: index => count(array(values.available_counts)[index]) > 0,
                customdata: array(row.evaluation_times).map((_, index) => [count(array(values.available_counts)[index])]),
                hover: 'In vivo time: %{x:.8g}<br>Median ' + (view === 'error' ? 'recovery error' : 'recovered response') +
                    ': %{y:.8g}<br>Usable successful studies: %{customdata[0]}' +
                    '<br>Successful fits: ' + count(row.n_successful) + '/' + count(row.n_simulations) +
                    '<extra>' + escape(row.name) + '</extra>',
                meta: {kind: 'median', schedule_id: row.id, view},
            });
            if (trace) {
                trace.name = legendName(row.name);
                trace.showlegend = true;
                trace.legendgroup = 'recovery-' + row.id;
                medians.push(trace);
            }
        });
        const figureLayout = layout(view === 'error' ? 'Recovery error across simulated studies' : 'Recovered response across simulated studies',
            view === 'error' ? 'Recovery error (analyzed response)' : 'Analyzed response',
            'planning-recovery-' + view,
            {kind: 'recovery_summary', view,
             range_interpretation: 'Pointwise central 95% ranges across successful simulated studies; not fitted confidence intervals.'});
        if (view === 'response') {
            const reference = inspection.reference || {};
            const central = line(reference.time, reference.central_response, {
                name: 'Central assumed reference', dash: 'dash', uid: 'central-reference',
                meta: {kind: 'reference'},
            });
            if (central) {
                central.name = legendName('Central assumed reference');
                central.showlegend = true;
                data.push(central);
            }
        } else {
            figureLayout.shapes = [{type: 'line', xref: 'paper', x0: 0, x1: 1,
                yref: 'y', y0: 0, y1: 0, layer: 'below', line: {color: neutral, width: 1.5, dash: 'dash'}}];
        }
        data.push(...medians);
        summaryLegend(figureLayout, data);
        return data.length ? {data, layout: figureLayout} : null;
    }

    function example(inspection, selection) {
        const rows = selected(inspection, selection);
        const recovery = inspection && inspection.recovery;
        const shared = recovery && recovery.shared_example;
        if (!rows.length || !shared || shared.available !== true ||
            rows.some(row => !row.example || row.example.simulation_index !== shared.simulation_index ||
                row.example.trajectory_draw_index !== shared.trajectory_draw_index)) return null;
        const data = [];
        const truth = line(rows[0].evaluation_times, rows[0].example.true_response, {
            name: 'Simulated true curve — shared study ' + shared.study_number,
            dash: 'dash', uid: 'shared-study-truth', meta: {kind: 'truth'},
        });
        if (truth) data.push(truth);
        rows.forEach((row, index) => {
            const sample = row.example, color = colorFor(selection, row, index);
            if (sample.fit_success === true) {
                const recovered = line(row.evaluation_times, sample.recovered_response, {
                    name: row.name + ' — recovered curve', color, uid: row.id + '-example-recovered',
                    meta: {kind: 'recovered', schedule_id: row.id},
                });
                if (recovered) data.push(recovered);
            }
            const observations = array(sample.observed_time_means).filter(point =>
                point.role === 'new' && finite(point.time) && finite(point.response));
            if (observations.length) data.push({
                type: 'scatter', mode: 'markers', name: escape(row.name) + ' — new observed time means',
                uid: row.id + '-example-observations', legendgroup: row.id,
                x: observations.map(point => point.time), y: observations.map(point => point.response),
                marker: {color, size: 8, symbol: 'circle', line: {color: 'white', width: 1}},
                customdata: observations.map(point => [Number.isInteger(point.planned_replicate_count) ? point.planned_replicate_count : null]),
                hovertemplate: 'In vivo time: %{x:.8g}<br>Observed replicate mean: %{y:.8g}<br>Planned independent replicates: %{customdata[0]}<extra>%{fullData.name}</extra>',
                showlegend: false, meta: {kind: 'observations', schedule_id: row.id},
            });
        });
        const completed = array(recovery.completed_observations).filter(point => finite(point.time) && finite(point.response));
        if (completed.length) data.push({
            type: 'scatter', mode: 'markers', name: 'Original completed observations',
            uid: 'original-completed-observations', x: completed.map(point => point.time), y: completed.map(point => point.response),
            marker: {color: '#20313d', size: 9, symbol: 'circle-open', line: {color: '#20313d', width: 1.6}},
            text: completed.map(point => {
                const precision = finite(point.observation_sigma) && point.uncertainty_kind ?
                    escape(point.uncertainty_kind) + ': ' + point.observation_sigma : 'Observation uncertainty not supplied';
                const replicates = Number.isInteger(point.reported_replicate_count) ? point.reported_replicate_count : 'not supplied';
                return precision + '<br>Reported independent count: ' + replicates;
            }),
            hovertemplate: 'In vivo time: %{x:.8g}<br>Completed response: %{y:.8g}<br>%{text}<extra>%{fullData.name}</extra>',
            showlegend: false, meta: {kind: 'completed'},
        });
        return data.length ? {data, layout: layout('Illustrative simulated study ' + shared.study_number,
            'Analyzed response', 'planning-recovery-example',
            {kind: 'recovery_example', simulation_index: shared.simulation_index,
             trajectory_draw_index: shared.trajectory_draw_index})} : null;
    }

    const api = Object.freeze({summary, example});
    if (typeof window !== 'undefined') window.DesignRecoveryFigure = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
