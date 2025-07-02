$(function() {
    $("#vertical-tabs").tabs().addClass("ui-tabs-vertical ui-helper-clearfix");
    $("#vertical-tabs li").removeClass("ui-corner-top").addClass("ui-corner-left");
});

$(document).ready( function() {
    $('.dataframe').DataTable({
        paging: false,
        searching: false,
        info: false,
        ordering: true,
        responsive: true,
        layout: {
            bottomStart: {
                buttons: ['copy', 'csv', 'excel']
            }
        }
    });
});

// Store original sizes of plots
var plotSizes = {};

function initializePlotSizes() {
    var plotContainers = document.querySelectorAll('.plot-container');
    plotContainers.forEach(function(plotContainer) {
        var plot = plotContainer.querySelector('.plotly-graph-div');
        if (plot && plot.layout) {
            plotSizes[plot.id] = {
                width: plot.layout.width,
                height: plot.layout.height
            };
        }
    });
}

function resizePlots(forceResize = false) {
    var plotContainers = document.querySelectorAll('.plot-container');
    plotContainers.forEach(function(plotContainer) {
        var plot = plotContainer.querySelector('.plotly-graph-div');
        if (plot && plot.layout) {
            var containerWidth = plotContainer.offsetWidth;
            var containerHeight = plotContainer.offsetHeight;
            
            if (forceResize || 
                (containerWidth > 0 && containerHeight > 0 && 
                 (containerWidth !== plot.layout.width || containerHeight !== plot.layout.height))) {
                
                Plotly.relayout(plot, {
                    width: containerWidth,
                    height: containerHeight
                });
                
                plotSizes[plot.id] = {
                    width: containerWidth,
                    height: containerHeight
                };
            } else if (containerWidth === 0 || containerHeight === 0) {
                Plotly.relayout(plot, plotSizes[plot.id]);
            }
        }
    });
}

function ensureResizePlots(forceResize = false) {
    resizePlots(forceResize);
    // Add a slight delay to catch any plots that might render late
    setTimeout(function() { resizePlots(forceResize); }, 10);
    // Add another check after a longer delay
    setTimeout(function() { resizePlots(forceResize); }, 50);
}

function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this, args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}

// Initialize plot sizes when the page loads
document.addEventListener('DOMContentLoaded', function() {
    initializePlotSizes();
    ensureResizePlots(true);
});

// Use debounced version for window resize, and force resize
window.addEventListener('resize', debounce(function() { ensureResizePlots(true); }, 250));

// Resize plots when tab is changed (if using tabs)
$(document).ready(function() {
    if (typeof $.ui !== 'undefined' && typeof $.ui.tabs !== 'undefined') {
        $(".tabs").tabs({
            activate: function(event, ui) {
                // Use setTimeout to allow the new tab to render
                setTimeout(function() { ensureResizePlots(false); }, 0);
            }
        });
    }
});

// MutationObserver to watch for changes in the DOM
var observer = new MutationObserver(debounce(function() { ensureResizePlots(false); }, 250));

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });

/*
function resizePlots() {
    var plotContainers = document.querySelectorAll('.plot-container');
    plotContainers.forEach(function(plotContainer) {
        var plot = plotContainer.querySelector('.plotly-graph-div');
        if (plot && plot.layout) {
            var containerWidth = plotContainer.offsetWidth;
            var containerHeight = plotContainer.offsetHeight;
            Plotly.relayout(plot, {
                width: containerWidth,
                height: containerHeight
            });
        }
    });
}

function ensureResizePlots() {
    resizePlots();
    // Add a slight delay to catch any plots that might render late
    setTimeout(resizePlots, 10);
    // Add another check after a longer delay
    setTimeout(resizePlots, 500);
}

// Debounce function to limit how often resizePlots is called
function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this, args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}

// Use debounced version for window resize
window.addEventListener('resize', debounce(resizePlots, 250));

// Initial resize
document.addEventListener('DOMContentLoaded', ensureResizePlots);

// Additional check after all resources have loaded
window.addEventListener('load', ensureResizePlots);

// Resize plots when tab is changed (if using tabs)
$(document).ready(function() {
    if (typeof $.ui !== 'undefined' && typeof $.ui.tabs !== 'undefined') {
        $(".tabs").tabs({
            activate: function(event, ui) {
                ensureResizePlots();
            }
        });
    }
});

// MutationObserver to watch for changes in the DOM
var observer = new MutationObserver(debounce(ensureResizePlots, 250));

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });

// Additional check for plots that might be initially hidden
function checkHiddenPlots() {
    var plotContainers = document.querySelectorAll('.plot-container');
    plotContainers.forEach(function(plotContainer) {
        if (plotContainer.offsetParent !== null && !plotContainer.dataset.resized) {
            resizePlots();
            plotContainer.dataset.resized = 'true';
        }
    });
}

// Run checkHiddenPlots periodically
setInterval(checkHiddenPlots, 1000);
*/

