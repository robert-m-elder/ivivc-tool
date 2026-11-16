/* Shared rendering for numerical Plotly specifications in returned results. */
(() => {
    'use strict';
    if (window.DesignPlots) return;

    const states = new WeakMap();

    function figures(container) {
        if (!container) return [];
        const found = Array.from(container.querySelectorAll('[data-design-plot]'));
        if (container.matches && container.matches('[data-design-plot]')) found.unshift(container);
        return found;
    }

    function visible(target) {
        for (let node = target; node; node = node.parentElement) if (node.hidden) return false;
        return !target.getClientRects || target.getClientRects().length > 0;
    }

    function clear(state) {
        if (!state.started) return;
        try { state.plotly.purge(state.target); } catch (error) { /* Results may already be removed. */ }
    }

    function fallback(state) {
        if (state.target) state.target.hidden = true;
        if (state.fallback) state.fallback.hidden = false;
        if (state.status) state.status.textContent = 'Interactive plot unavailable. The accompanying numerical results remain available.';
    }

    function resizeFigure(state) {
        if (state.disposed || !state.rendered || !visible(state.target)) return;
        try {
            // A view may become hidden again before Plotly finishes resizing.
            Promise.resolve(state.plotly.Plots.resize(state.target)).catch(() => {});
        } catch (error) { /* A failed resize must not change a successful calculation. */ }
    }

    function stateFor(wrapper) {
        const previous = states.get(wrapper);
        if (previous && !previous.disposed) return previous;
        const state = {
            target: wrapper.querySelector('[data-design-plot-target]'),
            fallback: wrapper.querySelector('[data-design-plot-fallback]'),
            status: wrapper.querySelector('[data-design-plot-status]'),
            plotly: window.Plotly, started: false, rendered: false, disposed: false,
            draining: false, pending: null, ready: Promise.resolve(),
            // A wrapper can be intentionally reused after purge. Let a prior
            // in-flight render finish its cleanup before touching its target.
            predecessor: previous && previous.draining ? previous.ready : null,
        };
        states.set(wrapper, state);
        return state;
    }

    function enqueue(state, spec) {
        state.pending = {spec};
        if (state.draining) return state.ready;
        state.draining = true;
        state.ready = Promise.resolve().then(async () => {
            try {
                if (state.predecessor) await state.predecessor;
                state.predecessor = null;
                while (!state.disposed && state.pending) {
                    const next = state.pending;
                    state.pending = null;
                    try {
                        const figure = next.spec;
                        const plotly = window.Plotly;
                        if (!state.target || !plotly || typeof plotly.react !== 'function' ||
                                !figure || !Array.isArray(figure.data) || !figure.layout ||
                                typeof figure.layout !== 'object' || Array.isArray(figure.layout)) throw new Error('Unavailable plot specification.');
                        state.plotly = plotly;
                        state.rendered = false;
                        if (state.fallback) state.fallback.hidden = false;
                        if (state.status) state.status.textContent = 'Preparing interactive plot.';
                        // Plotly needs a layout box even when the workspace
                        // containing this result will be shown just afterward.
                        state.target.hidden = false;
                        state.started = true;
                        await state.plotly.react(state.target, figure.data, {
                            ...figure.layout,
                            uirevision: figure.layout.uirevision ?? (state.target.id || 'design-plot'),
                        }, {
                            responsive: true, displaylogo: false, scrollZoom: false,
                            toImageButtonOptions: {format: 'svg', filename: state.target.id || 'study-design-plot'},
                        });
                        if (state.disposed || state.target.isConnected === false) {
                            clear(state);
                            return;
                        }
                        // A more recent selection may arrive while react is
                        // pending. Render only its latest spec next, without
                        // marking the superseded frame as ready.
                        if (state.pending) continue;
                        state.rendered = true;
                        if (state.fallback) state.fallback.hidden = true;
                        if (state.status) state.status.textContent = 'Interactive plot ready. Hover for values; use the plot toolbar to zoom, reset, or download an SVG.';
                        resizeFigure(state);
                    } catch (error) {
                        clear(state);
                        state.started = false;
                        state.rendered = false;
                        if (!state.disposed && !state.pending) fallback(state);
                    }
                }
            } finally {
                state.draining = false;
            }
        });
        return state.ready;
    }

    function initializeFigure(wrapper) {
        const previous = states.get(wrapper);
        if (previous && !previous.disposed) return previous.ready;
        const state = stateFor(wrapper);
        let spec = null;
        try {
            const encoded = wrapper.querySelector('[data-design-plot-spec]');
            spec = JSON.parse(encoded ? encoded.textContent : 'null');
        } catch (error) { /* Invalid specs use the ordinary numerical fallback. */ }
        return enqueue(state, spec);
    }

    window.DesignPlots = Object.freeze({
        initialize(container) {
            return Promise.all(figures(container).map(initializeFigure));
        },
        update(wrapper, spec) {
            if (!wrapper || typeof wrapper.querySelector !== 'function') return Promise.resolve();
            return enqueue(stateFor(wrapper), spec);
        },
        purge(container) {
            figures(container).forEach(wrapper => {
                const state = states.get(wrapper);
                if (!state || state.disposed) return;
                state.disposed = true;
                state.pending = null;
                clear(state);
            });
        },
        resize(container) {
            figures(container).forEach(wrapper => {
                const state = states.get(wrapper);
                if (state) resizeFigure(state);
            });
        },
    });
})();
