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
        if (state.status) state.status.textContent = 'Interactive plot unavailable. The static plot and numerical results remain available.';
    }

    function resizeFigure(state) {
        if (state.disposed || !state.rendered || !visible(state.target)) return;
        try {
            // A view may become hidden again before Plotly finishes resizing.
            Promise.resolve(state.plotly.Plots.resize(state.target)).catch(() => {});
        } catch (error) { /* A failed resize must not change a successful calculation. */ }
    }

    function initializeFigure(wrapper) {
        const previous = states.get(wrapper);
        if (previous) return previous.ready;
        const state = {
            target: wrapper.querySelector('[data-design-plot-target]'),
            fallback: wrapper.querySelector('[data-design-plot-fallback]'),
            status: wrapper.querySelector('[data-design-plot-status]'),
            plotly: window.Plotly, started: false, rendered: false, disposed: false,
        };
        states.set(wrapper, state);
        state.ready = Promise.resolve().then(async () => {
            try {
                const encoded = wrapper.querySelector('[data-design-plot-spec]');
                const spec = JSON.parse(encoded ? encoded.textContent : 'null');
                if (!state.target || !state.plotly || typeof state.plotly.react !== 'function' ||
                        !spec || !Array.isArray(spec.data) || !spec.layout ||
                        typeof spec.layout !== 'object' || Array.isArray(spec.layout)) throw new Error('Unavailable plot specification.');
                if (state.disposed) return;
                if (state.fallback) state.fallback.hidden = false;
                if (state.status) state.status.textContent = 'Preparing interactive plot.';
                // Plotly needs an element with a layout box, including when a
                // containing workspace view will be made visible just afterward.
                state.target.hidden = false;
                state.started = true;
                await state.plotly.react(state.target, spec.data, {
                    ...spec.layout,
                    uirevision: spec.layout.uirevision ?? (state.target.id || 'design-plot'),
                }, {
                    responsive: true, displaylogo: false, scrollZoom: false,
                    toImageButtonOptions: {format: 'svg', filename: state.target.id || 'study-design-plot'},
                });
                if (state.disposed || state.target.isConnected === false) {
                    clear(state);
                    return;
                }
                state.rendered = true;
                if (state.fallback) state.fallback.hidden = true;
                if (state.status) state.status.textContent = 'Interactive plot ready. Hover for values; use the plot toolbar to zoom, reset, or download an SVG.';
                resizeFigure(state);
            } catch (error) {
                clear(state);
                if (!state.disposed) fallback(state);
            }
        });
        return state.ready;
    }

    window.DesignPlots = Object.freeze({
        initialize(container) {
            return Promise.all(figures(container).map(initializeFigure));
        },
        purge(container) {
            figures(container).forEach(wrapper => {
                const state = states.get(wrapper);
                if (!state) return;
                state.disposed = true;
                clear(state);
                states.delete(wrapper);
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
