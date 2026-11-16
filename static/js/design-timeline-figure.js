/* Numerical Plotly specifications for stacked schedule timelines. */
(() => {
    'use strict';

    const palette = ['#24535c', '#a75328', '#6f4a92'];
    const pointTypes = [
        {key: 'completed', symbol: 'circle-open', label: 'Completed time'},
        {key: 'fixed', symbol: 'square', label: 'User-fixed proposed time'},
        {key: 'latest', symbol: 'diamond', label: 'Latest-before selection'},
        {key: 'other', symbol: 'circle', label: 'Other proposed time'},
    ];
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));

    function kind(point) {
        if (point.completed) return 'completed';
        if (Array.isArray(point.latest_before) && point.latest_before.length) return 'latest';
        return point.user_fixed ? 'fixed' : 'other';
    }

    function pointText(point) {
        const descriptions = [point.completed ? 'Completed measurement time.' : 'Proposed new measurement time.'];
        if (point.baseline) descriptions.push('At the milestone response baseline.');
        if (point.timing_baseline) descriptions.push('Fixed timing baseline; excluded from timing requirement counts.');
        if (point.user_fixed) descriptions.push('Fixed by the user for a proposed measurement.');
        (point.latest_before || []).forEach(label => descriptions.push('Selected by latest-before rule: ' + label + '.'));
        (point.contributes_to || []).forEach(label => descriptions.push('Contributes to window: ' + label + '.'));
        (point.deadline_eligibility || []).filter(label => !(point.latest_before || []).includes(label))
            .forEach(label => descriptions.push('Eligible before deadline: ' + label + '; eligibility alone does not certify latestness.'));
        if (point.completed && Number.isInteger(point.completed_rows)) {
            descriptions.push('Retained completed rows at this time: ' + point.completed_rows + '.');
        } else if (!point.completed && Number.isInteger(point.new_replicates)) {
            descriptions.push('Planned independent replicates: ' + point.new_replicates + '.');
        }
        return descriptions.map(escape).join('<br>');
    }

    function wrappedLabel(value) {
        const text = String(value ?? '');
        const shortened = text.length > 56 ? text.slice(0, 53) + '…' : text;
        if (shortened.length <= 28) return escape(shortened);
        let at = shortened.lastIndexOf(' ', 28);
        if (at < 12) at = 28;
        return escape(shortened.slice(0, at)) + '<br>' + escape(shortened.slice(at).trimStart());
    }

    function bounds(values) {
        const available = values.filter(finite);
        if (!available.length) return [0, 1];
        const lower = Math.min(...available), upper = Math.max(...available);
        const span = upper - lower;
        const padding = span > 0 && Number.isFinite(span) ? span * .035 : Math.max(Math.abs(lower) * .035, .5);
        const paddedLower = lower - padding, paddedUpper = upper + padding;
        return [Number.isFinite(paddedLower) ? paddedLower : lower,
                Number.isFinite(paddedUpper) ? paddedUpper : upper];
    }

    function build(inspection, selectedIds, colors = {}) {
        if (!inspection || inspection.schema_version !== 1 || !inspection.schedule_data || !Array.isArray(selectedIds)) return null;
        const schedules = inspection.schedule_data.schedules;
        if (!Array.isArray(schedules)) return null;
        const wanted = new Set(selectedIds);
        const selected = schedules.filter(schedule => wanted.has(schedule.id)).slice(0, 3);
        if (!selected.length) return null;

        const guides = inspection.schedule_data;
        const requirements = Array.isArray(guides.requirements) ? guides.requirements : [];
        const coordinates = [...(inspection.reference?.time || []), ...(guides.evaluation_time_range || [])];
        const data = [], shapes = [];
        const top = -.4, bottom = selected.length - .6;

        selected.forEach(schedule => {
            (schedule.points || []).forEach(point => { if (finite(point.time)) coordinates.push(point.time); });
        });
        [guides.baseline_time, guides.latest_completed_time, guides.first_opportunity_time].forEach(value => {
            if (finite(value)) coordinates.push(value);
        });

        requirements.forEach((requirement, index) => {
            const lower = requirement.lower, upper = requirement.upper;
            if (requirement.kind === 'window' && finite(lower) && finite(upper) && lower < upper) {
                coordinates.push(lower, upper);
                const description = escape(requirement.label) + '<br>Required window: (' + lower + ', ' + upper + ']' +
                    '<br>Minimum distinct times: ' + escape(requirement.minimum_count) +
                    '<br>' + (requirement.count_scope === 'all' ? 'Completed and new qualifying times.' : 'New qualifying times.');
                shapes.push({type: 'rect', name: 'window-' + index, xref: 'x', yref: 'paper',
                    x0: lower, x1: upper, y0: 0, y1: 1, layer: 'below',
                    line: {width: 0}, fillcolor: 'rgba(74,111,139,0.07)'});
                data.push({type: 'scatter', mode: 'lines', name: escape(requirement.label),
                    x: [lower, upper, upper, lower, lower], y: [top, top, bottom, bottom, top],
                    fill: 'toself', fillcolor: 'rgba(0,0,0,0)', line: {width: 0},
                    hoveron: 'fills', hovertemplate: description + '<extra></extra>',
                    showlegend: false, meta: {kind: 'window', requirement_index: index}});
            } else if (requirement.kind === 'latest_before' && finite(upper)) {
                coordinates.push(upper);
                vertical(upper, 'deadline-' + index, '#8a6e4a', 'dot',
                    escape(requirement.label) + '<br>Deadline: ' + upper +
                    '<br>A time before this line is not automatically a certified latest-before selection.');
            }
        });

        function vertical(time, name, color, dash, description) {
            shapes.push({type: 'line', name, xref: 'x', yref: 'paper',
                x0: time, x1: time, y0: 0, y1: 1, layer: 'below', line: {color, width: 1.5, dash}});
            data.push({type: 'scatter', mode: 'lines', name, x: [time, time], y: [top, bottom],
                line: {width: 7, color: 'rgba(0,0,0,0)'}, hovertemplate: description + '<extra></extra>',
                showlegend: false, meta: {kind: 'guide', guide: name}});
        }

        if (finite(guides.baseline_time)) {
            const alsoLatest = guides.baseline_time === guides.latest_completed_time;
            vertical(guides.baseline_time, 'milestone-baseline', '#62717c', 'dash',
                'Milestone response baseline: ' + guides.baseline_time +
                '<br>A guide only; no observation is added.' + (alsoLatest ? '<br>Also the latest completed time.' : ''));
        }
        if (finite(guides.latest_completed_time) && guides.latest_completed_time !== guides.baseline_time) {
            vertical(guides.latest_completed_time, 'latest-completed', '#62717c', 'dashdot',
                'Latest completed time: ' + guides.latest_completed_time +
                '<br>' + (guides.future_policy === 'gap_fill' ? 'Gap filling may include new earlier times.' : 'New future-only times occur after this guide.'));
        }

        const range = bounds(coordinates);
        selected.forEach((schedule, row) => {
            const points = (schedule.points || []).filter(point => finite(point.time));
            const requested = colors[schedule.id];
            const color = typeof requested === 'string' && /^#[0-9a-f]{6}$/i.test(requested) ? requested : palette[schedules.indexOf(schedule) % palette.length];
            const symbols = points.map(point => {
                const type = kind(point);
                return pointTypes.find(value => value.key === type).symbol;
            });
            shapes.push({type: 'line', name: 'schedule-track-' + schedule.id,
                xref: 'x', yref: 'y', x0: range[0], x1: range[1], y0: row, y1: row,
                layer: 'below', line: {color: '#d8e1e7', width: 1}});
            data.push({type: 'scatter', mode: 'markers', name: escape(schedule.name),
                x: points.map(point => point.time), y: points.map(() => row),
                text: points.map(pointText),
                marker: {color, size: 12, symbol: symbols, line: {color, width: 1.6}},
                hovertemplate: 'In vivo time: %{x:.8g}<br>%{text}<extra>' + escape(schedule.name) + '</extra>',
                showlegend: false, meta: {kind: 'schedule', schedule_id: schedule.id}});
        });
        // The wrapping HTML key remains available beside the numerical table.
        // Omitting an in-plot key also keeps narrow timelines from clipping it.
        const height = 225 + selected.length * 62;
        return {data, layout: {
            title: {text: 'Selected sampling schedules', x: 0},
            shapes,
            height,
            xaxis: {title: {text: 'In vivo time'}, range, showgrid: true, gridcolor: '#edf1f4', zeroline: false},
            yaxis: {tickmode: 'array', tickvals: selected.map((_, row) => row),
                ticktext: selected.map(schedule => wrappedLabel(schedule.name)),
                range: [selected.length - .5, -.5], showgrid: false, zeroline: false,
                fixedrange: true, automargin: true, tickfont: {size: 11}},
            font: {family: 'Arial, sans-serif', color: '#20313d'},
            paper_bgcolor: 'white', plot_bgcolor: 'white',
            margin: {l: 185, r: 25, t: 55, b: 75},
            showlegend: false,
            hovermode: 'closest', uirevision: 'planning-schedule-timeline',
        }};
    }

    const api = Object.freeze({build});
    if (typeof window !== 'undefined') window.DesignTimelineFigure = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
