
function announceResultsStatus(message) {
    if (window.IVIVCAccessibility) {
        window.IVIVCAccessibility.setLiveMessage('results-status', message);
    }
}

function showResultsAlert(message) {
    var alertBox = $('#results-alert');
    if (!message) {
        alertBox.prop('hidden', true).empty();
        return;
    }
    alertBox.text(message).prop('hidden', false);
    if (alertBox.length) {
        alertBox[0].focus();
    }
}

function refreshAccessibility(root) {
    if (!window.IVIVCAccessibility) {
        return;
    }
    window.IVIVCAccessibility.enhanceTableSemantics(root || document);
    window.IVIVCAccessibility.initializeDataTableControls(root || document);
    window.IVIVCAccessibility.enhancePlotAccessibility(root || document);
}

function getSummaryTables(root) {
    var scope = root ? $(root) : $(document);
    var tables = scope.is('table.dataframe')
        ? scope.add(scope.find('table.dataframe'))
        : scope.find('table.dataframe');

    return tables.filter(function() {
        // Plot-source tables are hidden implementation details used only by
        // the copy and download controls beneath each plot.
        return $(this).closest('.data-table-container').length === 0;
    });
}


function tableContextLabel(tableElement) {
    var table = $(tableElement);
    var explicit = table.attr('data-table-context');
    if (explicit) {
        return explicit;
    }

    var caption = table.find('caption').first().text().replace(/\s+/g, ' ').trim();
    if (caption) {
        return caption;
    }

    var heading = table.prevAll('h2, h3, h4, h5, h6').first().text().replace(/\s+/g, ' ').trim();
    return heading || 'data table';
}

function initializeSummaryTableDownloads(tableElement) {
    if (!$.fn.DataTable.isDataTable(tableElement)) {
        return;
    }

    var table = $(tableElement);
    if (table.attr('data-summary-downloads-initialized') === 'true') {
        return;
    }

    var api = table.DataTable();
    var context = tableContextLabel(tableElement);
    var container = $(api.table().container());
    var filename = plotDataFilename(context, table.attr('id') || 'table-data');
    var exportOptions = {
        columns: ':not(.no-export)',
        modifier: {
            page: 'all',
            search: 'applied',
            order: 'applied'
        }
    };
    var buttons = new DataTable.Buttons(api, {
        buttons: [
            {
                extend: 'copyHtml5',
                text: 'Copy data',
                title: null,
                exportOptions: exportOptions,
                attr: {
                    'aria-label': 'Copy data from ' + context,
                    'title': 'Copy data from ' + context
                }
            },
            {
                extend: 'csvHtml5',
                text: 'Download CSV',
                title: null,
                filename: filename,
                exportOptions: exportOptions,
                attr: {
                    'aria-label': 'Download ' + context + ' as CSV',
                    'title': 'Download ' + context + ' as CSV'
                }
            },
            {
                extend: 'excelHtml5',
                text: 'Download Excel',
                title: null,
                filename: filename,
                exportOptions: exportOptions,
                attr: {
                    'aria-label': 'Download ' + context + ' as Excel',
                    'title': 'Download ' + context + ' as Excel'
                }
            }
        ]
    });

    var toolbar = $('<div>', {
        'class': 'summary-table-downloads',
        'role': 'group',
        'aria-label': 'Table data downloads for ' + context
    });
    $('<span>', {
        'class': 'summary-table-download-label',
        'aria-hidden': 'true',
        'text': 'Table data:'
    }).appendTo(toolbar);
    var buttonHost = $('<div>', {
        'class': 'summary-table-button-container'
    }).appendTo(toolbar);
    $(buttons.container()).appendTo(buttonHost);
    toolbar.appendTo(container);

    toolbar
        .off('click.ivivcSummaryDataStatus')
        .on('click.ivivcSummaryDataStatus', 'button', function() {
            var button = $(this);
            if (button.hasClass('buttons-copy')) {
                setTimeout(function() {
                    announceResultsStatus('Table data copied from ' + context + '.');
                }, 100);
            } else if (button.hasClass('buttons-csv')) {
                announceResultsStatus('CSV download started for ' + context + '.');
            } else if (button.hasClass('buttons-excel')) {
                announceResultsStatus('Excel download started for ' + context + '.');
            }
        });

    table.attr('data-summary-downloads-initialized', 'true');
}

function plotDataFilename(context, target) {
    var base = String(context || target || 'plot-data')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return base || 'plot-data';
}

function initializePlotDataDownloads(root) {
    var scope = root ? $(root) : $(document);
    var controls = scope.is('.plot-data-downloads')
        ? scope.add(scope.find('.plot-data-downloads'))
        : scope.find('.plot-data-downloads');

    controls.each(function() {
        var group = $(this);
        if (group.attr('data-downloads-initialized') === 'true') {
            return;
        }

        var target = group.attr('data-table-target');
        var context = group.attr('data-plot-title') || 'plot';
        var sourceContainer = target ? $('#' + target + '-container') : $();
        var table = sourceContainer.find('table').first();
        var buttonHost = group.find('.plot-data-button-container').first();

        if (!target || !sourceContainer.length || !table.length || !buttonHost.length) {
            buttonHost.text('Plot data are unavailable.');
            group.attr('data-downloads-initialized', 'true');
            return;
        }

        table.attr('data-table-context', context + ' data');

        if (!$.fn.DataTable.isDataTable(table[0])) {
            table.DataTable({
                paging: false,
                searching: false,
                info: false,
                ordering: false,
                order: [],
                responsive: false,
                autoWidth: false,
                dom: null,
                buttons: [],
                layout: {
                    topStart: null,
                    topEnd: null,
                    bottomStart: null,
                    bottomEnd: null
                }
            });
        }

        var api = table.DataTable();
        var filename = plotDataFilename(context, target);
        var exportOptions = {
            columns: ':not(.no-export)',
            modifier: {
                page: 'all',
                search: 'none',
                order: 'original'
            }
        };
        var buttons = new DataTable.Buttons(api, {
            buttons: [
                {
                    extend: 'copyHtml5',
                    text: 'Copy data',
                    title: null,
                    exportOptions: exportOptions,
                    attr: {
                        'aria-label': 'Copy data for ' + context,
                        'title': 'Copy data for ' + context
                    }
                },
                {
                    extend: 'csvHtml5',
                    text: 'Download CSV',
                    title: null,
                    filename: filename,
                    exportOptions: exportOptions,
                    attr: {
                        'aria-label': 'Download CSV data for ' + context,
                        'title': 'Download CSV data for ' + context
                    }
                },
                {
                    extend: 'excelHtml5',
                    text: 'Download Excel',
                    title: null,
                    filename: filename,
                    exportOptions: exportOptions,
                    attr: {
                        'aria-label': 'Download Excel data for ' + context,
                        'title': 'Download Excel data for ' + context
                    }
                }
            ]
        });

        var buttonContainer = $(buttons.container());
        buttonContainer.appendTo(buttonHost);
        buttonContainer
            .off('click.ivivcPlotDataStatus')
            .on('click.ivivcPlotDataStatus', 'button', function() {
                var button = $(this);
                if (button.hasClass('buttons-copy')) {
                    setTimeout(function() {
                        announceResultsStatus('Plot data copied for ' + context + '.');
                    }, 100);
                } else if (button.hasClass('buttons-csv')) {
                    announceResultsStatus('CSV download started for ' + context + '.');
                } else if (button.hasClass('buttons-excel')) {
                    announceResultsStatus('Excel download started for ' + context + '.');
                }
            });

        group.attr('data-downloads-initialized', 'true');
    });

    refreshAccessibility(root || document);
}

function announceSummaryTableSort(tableElement) {
    var table = $(tableElement);
    var api = table.DataTable();
    var order = api.order();

    if (!order || order.length === 0) {
        announceResultsStatus('Table returned to its original order.');
        return;
    }

    var columnIndex = order[0][0];
    var direction = order[0][1] === 'asc' ? 'ascending' : 'descending';
    var heading = $(api.column(columnIndex).header()).text().replace(/\s+/g, ' ').trim();
    announceResultsStatus('Table sorted by ' + (heading || 'selected column') + ', ' + direction + '.');
}

function initializeSummaryDataTables(root) {
    getSummaryTables(root).each(function() {
        var tableElement = this;
        var table = $(tableElement);

        // DataTables calculates column widths from the rendered table. Defer
        // initialization until the table's tab or report panel is visible.
        if (!table.is(':visible')) {
            return;
        }

        if (!$.fn.DataTable.isDataTable(tableElement)) {
            table.DataTable();
        }

        table.off('order.dt.ivivcSummarySort')
            .on('order.dt.ivivcSummarySort', function() {
                announceSummaryTableSort(tableElement);
            });
        table.DataTable().columns.adjust();
        initializeSummaryTableDownloads(tableElement);
    });

    refreshAccessibility(root || document);
}

$(function() {
    $("#vertical-tabs").tabs({
        activate: function(event, ui) {
            setTimeout(function() {
                initializeSummaryDataTables(ui.newPanel);
                handleVisiblePlotResize();
            }, 0);
        }
    }).addClass("ui-tabs-vertical ui-helper-clearfix");
    $("#vertical-tabs li").removeClass("ui-corner-top").addClass("ui-corner-left");
});

// DataTables default configuration
$.extend(true, $.fn.dataTable.defaults, {
    paging: true,
    pageLength: 10,
    searching: false,
    info: false,
    ordering: true,
    order: [],
    responsive: true,
    dom: 'rtp',
    buttons: [],
    layout: {
        topStart: null,
        topEnd: null,
        bottomStart: 'paging',
        bottomEnd: null
    },
    language: {
        aria: {
            orderable: ': Activate to sort this column',
            orderableReverse: ': Activate to reverse the sort',
            orderableRemove: ': Activate to remove sorting'
        }
    }
});

$(document).ready(function() {
    refreshAccessibility(document);

    // Initialize summary and comparison tables in the initially visible panel.
    initializeSummaryDataTables(document);

    // Plot-source tables remain hidden; only their copy and download controls
    // are exposed beneath the corresponding plots.
    initializePlotDataDownloads(document);

    if (window.IVIVCAccessibility) {
        window.IVIVCAccessibility.initializeHelpModal({
            'data-processing-help': 'data-processing-help-content',
            'model-comparison-help': 'model-comparison-help-content',
            'final-model-report-help': 'final-model-report-help-content',
            'approach1-results-help': 'approach1-results-help-content',
            'approach2-results-help': 'approach2-results-help-content',
            'approach3-results-help': 'approach3-results-help-content',
            'prediction-approach1-help': 'prediction-approach1-help-content',
            'prediction-approach2-help': 'prediction-approach2-help-content',
            'prediction-approach3-help': 'prediction-approach3-help-content'
        });
    }
});


/*function sortColumns(dt, rowIndex) {
    var table = dt.table();
    var rowData = table.row(rowIndex).data();
    var indices = Object.keys(rowData).slice(1);  // Exclude the first column (arrows)
    var $arrow = $(table.row(rowIndex).node()).find('.row-sorter');
    var isDescending = $arrow.text() === '↓';
    
    // Sort indices based on the values in the selected row
    indices.sort(function(a, b) {
        return isDescending ? rowData[a] - rowData[b] : rowData[b] - rowData[a];
    });
    
    // Reorder columns based on the sorted indices
    var newOrder = [0].concat(indices.map(i => parseInt(i)));  // Keep arrow column first
    table.colReorder.order(newOrder);
    
    // Update sorting arrows
    $('.row-sorter').text('↕️');  // Reset all arrows
    $arrow.text(isDescending ? '↑' : '↓');  // Set arrow for sorted row
}*/

// Fit plots to containers
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
    //setTimeout(function() { resizePlots(forceResize); }, 10);
    // Add another check after a longer delay
    //setTimeout(function() { resizePlots(forceResize); }, 50);
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
    refreshAccessibility(document);
});

// Use debounced version for window resize, and force resize
window.addEventListener('resize', debounce(function() { handleVisiblePlotResize(); }, 250));

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
var observer = new MutationObserver(debounce(function() {
    ensureResizePlots(false);
    refreshAccessibility(document);
}, 250));

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });

function sanitizePlotId(value) {
    return String(value || 'plot').replace(/[^A-Za-z0-9_-]/g, '-');
}

function clonePlotlyPayload(payload) {
    return JSON.parse(JSON.stringify(payload || {}));
}

function getReportPlotHeight(sourcePlot) {
    var height = 450;
    if (sourcePlot && sourcePlot._fullLayout && sourcePlot._fullLayout.height) {
        height = sourcePlot._fullLayout.height;
    } else if (sourcePlot && sourcePlot.layout && sourcePlot.layout.height) {
        height = sourcePlot.layout.height;
    }
    return Math.max(360, Math.min(height, 650));
}

function resizeReportPlots(scope) {
    var $scope = scope ? $(scope) : $(document);
    $scope.find('.report-plotly-div').each(function() {
        var plotDiv = this;
        var $placeholder = $(plotDiv).closest('.report-plot-placeholder');
        var width = $placeholder.width();
        var height = parseInt(plotDiv.getAttribute('data-report-height'), 10) || getReportPlotHeight(plotDiv);

        if (width > 0 && height > 0 && plotDiv.layout) {
            Plotly.relayout(plotDiv, {
                width: width,
                height: height,
                autosize: true
            });
            Plotly.Plots.resize(plotDiv);
        }
    });
}

function populateReportPlots(scope) {
    var $scope = scope ? $(scope) : $(document);
    var plotPromises = [];

    $scope.find('.report-plot-placeholder').each(function(index) {
        var placeholder = this;
        var $placeholder = $(placeholder);
        var sourceId = $placeholder.data('plot-source');
        var source = sourceId ? document.getElementById(sourceId) : null;

        if (!source) {
            $placeholder.html('<p class="plot-unavailable">Referenced plot is not available.</p>');
            return;
        }

        var sourcePlot = source.querySelector('.plotly-graph-div');
        if (!sourcePlot || !sourcePlot.data || !sourcePlot.layout) {
            $placeholder.html('<p class="plot-unavailable">Referenced plot has not rendered yet.</p>');
            return;
        }

        var existingPlot = placeholder.querySelector('.report-plotly-div');
        if (existingPlot && existingPlot.getAttribute('data-plot-source') === sourceId) {
            resizeReportPlots(placeholder);
            return;
        }

        var reportPlot = document.createElement('div');
        var reportHeight = getReportPlotHeight(sourcePlot);
        reportPlot.id = 'report-plot-' + sanitizePlotId(sourceId) + '-' + index + '-' + Date.now();
        reportPlot.className = 'report-plotly-div';
        reportPlot.setAttribute('data-plot-source', sourceId);
        reportPlot.setAttribute('data-report-height', reportHeight);
        reportPlot.style.width = '100%';
        reportPlot.style.height = reportHeight + 'px';

        $placeholder.empty().append(reportPlot);

        var data = clonePlotlyPayload(sourcePlot.data);
        var layout = clonePlotlyPayload(sourcePlot.layout);
        delete layout.width;
        layout.height = reportHeight;
        layout.autosize = true;
        layout.margin = layout.margin || {l: 30, r: 30, t: 30, b: 30};

        var config = {
            responsive: true,
            displaylogo: false
        };

        var plotPromise = Plotly.newPlot(reportPlot, data, layout, config).then(function() {
            resizeReportPlots(placeholder);
        });
        plotPromises.push(plotPromise);
    });

    return Promise.all(plotPromises).then(function() {
        resizeReportPlots($scope);
    });
}

function setActiveFinalReport(selectedId) {
    $('.final-report-content').hide().removeClass('active-report').attr('aria-hidden', 'true');
    var report = document.getElementById(selectedId + '-report');
    if (!report) {
        return Promise.resolve();
    }

    $(report).show().addClass('active-report').attr('aria-hidden', 'false');
    initializeSummaryDataTables(report);

    return populateReportPlots(report).then(function() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            return window.MathJax.typesetPromise([report]);
        }
    }).then(function() {
        initializeSummaryDataTables(report);
        resizeReportPlots(report);
    });
}

function handleVisiblePlotResize() {
    setTimeout(function() {
        var activeReport = document.querySelector('.final-report-content.active-report');
        if (activeReport) {
            populateReportPlots(activeReport);
        }
        ensureResizePlots(true);
        resizeReportPlots(activeReport || document);
    }, 0);

    setTimeout(function() {
        var activeReport = document.querySelector('.final-report-content.active-report');
        ensureResizePlots(true);
        resizeReportPlots(activeReport || document);
    }, 150);
}

var reportImageDataById = {};

function getPreviousHeadingText(element) {
    var sibling = element.previousElementSibling;
    while (sibling) {
        if (/^H[1-6]$/.test(sibling.tagName)) {
            return cleanReportText(sibling.textContent);
        }
        sibling = sibling.previousElementSibling;
    }
    return '';
}

function cleanReportText(value) {
    return String(value || '').replace(/\s+/g, ' ').trim();
}

function extractTableRows(table) {
    var rows = [];
    Array.from(table.querySelectorAll('tr')).forEach(function(row) {
        var cells = Array.from(row.querySelectorAll('th, td')).map(function(cell) {
            return cleanReportText(cell.textContent);
        });
        if (cells.some(function(cell) { return cell.length > 0; })) {
            rows.push(cells);
        }
    });
    return rows;
}

function appendReportElementToPayload(element, sections) {
    if (!element || element.nodeType !== 1) {
        return;
    }

    if (element.classList.contains('report-plot-placeholder')) {
        var imageId = element.getAttribute('data-report-image-id');
        if (imageId) {
            sections.push({
                type: 'image',
                title: '',
                data_url: reportImageDataById[imageId] || ''
            });
        } else if (element.getAttribute('data-report-export-attempted') === 'true') {
            var plotTitle = getPreviousHeadingText(element) || 'Report plot';
            var exportError = element.getAttribute('data-report-export-error') || 'Image export was unavailable.';
            sections.push({
                type: 'paragraph',
                text: plotTitle + ': plot image could not be exported to Word. ' + exportError
            });
        }
        return;
    }

    var tagName = element.tagName.toLowerCase();
    if (/^h[1-6]$/.test(tagName)) {
        var headingText = cleanReportText(element.textContent);
        if (headingText) {
            var htmlLevel = parseInt(tagName.substring(1), 10);
            sections.push({
                type: 'heading',
                level: Math.max(1, Math.min(htmlLevel - 2, 3)),
                text: headingText
            });
        }
        return;
    }

    if (tagName === 'p') {
        if (element.classList.contains('report-equation')) {
            var equationLatex = element.getAttribute('data-report-equation') || '';
            var fallbackText = cleanReportText(element.textContent);
            if (equationLatex || fallbackText) {
                sections.push({
                    type: 'equation',
                    latex: equationLatex,
                    text: fallbackText
                });
            }
            return;
        }

        var paragraphText = cleanReportText(element.textContent);
        if (paragraphText) {
            sections.push({type: 'paragraph', text: paragraphText});
        }
        return;
    }

    if (tagName === 'ul' || tagName === 'ol') {
        Array.from(element.children).forEach(function(child) {
            if (child.tagName && child.tagName.toLowerCase() === 'li') {
                var itemText = cleanReportText(child.textContent);
                if (itemText) {
                    sections.push({type: 'bullet', text: itemText});
                }
            }
        });
        return;
    }

    if (tagName === 'table') {
        var rows = extractTableRows(element);
        if (rows.length > 0) {
            sections.push({type: 'table', rows: rows});
        }
        return;
    }

    Array.from(element.children).forEach(function(child) {
        appendReportElementToPayload(child, sections);
    });
}

function collectReportSections(report) {
    var sections = [];
    Array.from(report.children).forEach(function(child) {
        appendReportElementToPayload(child, sections);
    });
    return sections;
}

function getPlotForWordExport(placeholder) {
    // Prefer the report plot if it has already been rendered.
    var reportPlot = placeholder.querySelector('.report-plotly-div');
    if (reportPlot && reportPlot.data && reportPlot.layout) {
        return reportPlot;
    }

    // Fall back to the original source plot. This avoids depending on report-tab
    // sizing/state and is more reliable when the user has clicked through tabs in
    // different orders.
    var sourceId = placeholder.getAttribute('data-plot-source');
    var source = sourceId ? document.getElementById(sourceId) : null;
    var sourcePlot = source ? source.querySelector('.plotly-graph-div') : null;
    if (sourcePlot && sourcePlot.data && sourcePlot.layout) {
        return sourcePlot;
    }

    return null;
}

function clonePlotForFixedSizeExport(sourcePlot, width, height) {
    if (!sourcePlot || typeof Plotly === 'undefined' || !Plotly.newPlot || !Plotly.toImage) {
        return Promise.reject(new Error('Plotly image export is not available.'));
    }

    var data = clonePlotlyPayload(sourcePlot.data || sourcePlot._fullData || []);
    var layout = clonePlotlyPayload(sourcePlot.layout || {});
    layout.width = width;
    layout.height = height;
    layout.autosize = false;
    layout.margin = layout.margin || {l: 60, r: 30, t: 50, b: 60};
    layout.paper_bgcolor = layout.paper_bgcolor || 'white';
    layout.plot_bgcolor = layout.plot_bgcolor || 'white';

    var tempPlot = document.createElement('div');
    tempPlot.style.position = 'fixed';
    tempPlot.style.left = '-10000px';
    tempPlot.style.top = '0';
    tempPlot.style.width = width + 'px';
    tempPlot.style.height = height + 'px';
    tempPlot.style.background = 'white';
    document.body.appendChild(tempPlot);

    var cleanup = function() {
        try {
            if (typeof Plotly !== 'undefined' && Plotly.purge) {
                Plotly.purge(tempPlot);
            }
        } catch (e) {
            // Cleanup should never block report generation.
        }
        if (tempPlot.parentNode) {
            tempPlot.parentNode.removeChild(tempPlot);
        }
    };

    return Plotly.newPlot(tempPlot, data, layout, {
        staticPlot: true,
        displayModeBar: false,
        responsive: false,
        displaylogo: false
    }).then(function() {
        return Plotly.toImage(tempPlot, {
            format: 'png',
            width: width,
            height: height,
            scale: 2
        });
    }).then(function(dataUrl) {
        cleanup();
        return dataUrl;
    }).catch(function(error) {
        cleanup();
        throw error;
    });
}

function exportReportPlotImages(report) {
    reportImageDataById = {};
    var placeholders = Array.from(report.querySelectorAll('.report-plot-placeholder'));
    var imagePromises = placeholders.map(function(placeholder, index) {
        placeholder.setAttribute('data-report-export-attempted', 'true');
        placeholder.removeAttribute('data-report-image-id');
        placeholder.removeAttribute('data-report-export-error');

        var plotDiv = getPlotForWordExport(placeholder);
        if (!plotDiv) {
            placeholder.setAttribute('data-report-export-error', 'Referenced plot could not be found or has not rendered.');
            return Promise.resolve(false);
        }

        var height = parseInt(plotDiv.getAttribute('data-report-height'), 10) || getReportPlotHeight(plotDiv) || 450;
        height = Math.max(360, Math.min(height, 650));
        var imageId = 'report-image-' + index;

        return clonePlotForFixedSizeExport(plotDiv, 1000, height).then(function(dataUrl) {
            if (!dataUrl || dataUrl.indexOf('data:image/') !== 0) {
                throw new Error('Plotly returned an invalid image payload.');
            }
            placeholder.setAttribute('data-report-image-id', imageId);
            reportImageDataById[imageId] = dataUrl;
            return true;
        }).catch(function(error) {
            var message = error && error.message ? error.message : 'Unknown plot export error.';
            console.warn('Unable to export report plot image:', message, placeholder);
            placeholder.removeAttribute('data-report-image-id');
            placeholder.setAttribute('data-report-export-error', message);
            delete reportImageDataById[imageId];
            return false;
        });
    });

    return Promise.all(imagePromises).then(function(results) {
        return {
            total: placeholders.length,
            exported: results.filter(Boolean).length
        };
    });
}

function downloadBlob(blob, filename) {
    var url = window.URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = filename || 'IVIVC_Report.docx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}

function filenameFromContentDisposition(header) {
    if (!header) {
        return null;
    }
    var match = header.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
    if (!match) {
        return null;
    }
    return decodeURIComponent(match[1] || match[2]);
}

function downloadSelectedWordReport(selectedId, button) {
    var report = document.getElementById(selectedId + '-report');
    if (!report) {
        return;
    }

    var originalText = button.find('span').text();
    showResultsAlert('');
    button.prop('disabled', true).attr('aria-busy', 'true');
    button.find('span').text('Preparing Word Report...');
    announceResultsStatus('Preparing the selected Word report.');

    setActiveFinalReport(selectedId)
        .then(function() {
            return exportReportPlotImages(report);
        })
        .then(function(imageExportSummary) {
            if (imageExportSummary && imageExportSummary.total > 0 && imageExportSummary.exported === 0) {
                console.warn('No report plot images were exported for the Word report.');
                showResultsAlert('No report plot images could be exported. The Word report will still download and will include notes where images were unavailable.');
            } else if (imageExportSummary && imageExportSummary.exported < imageExportSummary.total) {
                console.warn('Some report plot images were not exported for the Word report.', imageExportSummary);
            }

            var payload = {
                report_label: report.getAttribute('data-report-label') || 'IVIVC Report',
                sections: collectReportSections(report),
                image_export_summary: imageExportSummary || {total: 0, exported: 0}
            };

            return fetch('/download_word_report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
        })
        .then(function(response) {
            if (!response.ok) {
                return response.json().catch(function() {
                    return {error: 'Unable to generate Word report.'};
                }).then(function(errorPayload) {
                    throw new Error(errorPayload.error || 'Unable to generate Word report.');
                });
            }
            var filename = filenameFromContentDisposition(response.headers.get('Content-Disposition')) || 'IVIVC_Report.docx';
            return response.blob().then(function(blob) {
                downloadBlob(blob, filename);
                announceResultsStatus('The selected Word report was downloaded.');
            });
        })
        .catch(function(error) {
            showResultsAlert(error.message || 'Unable to generate Word report.');
        })
        .finally(function() {
            button.prop('disabled', false).removeAttr('aria-busy');
            button.find('span').text(originalText);
        });
}

// Final model/report selector
$(document).ready(function() {
    var initialSelection = $('input[name="final_model_option"]:checked').val();
    if (initialSelection) {
        setActiveFinalReport(initialSelection);
    }

    $('input[name="final_model_option"]').on('change', function() {
        var selectedLabel = $(this).closest('label').text().replace(/\s+/g, ' ').trim();
        setActiveFinalReport($(this).val()).then(function() {
            announceResultsStatus('Report summary updated to ' + selectedLabel + '.');
            refreshAccessibility(document);
        });
    });

    $('#print-final-report').on('click', function() {
        var selectedId = $('input[name="final_model_option"]:checked').val();
        if (!selectedId) {
            return;
        }

        setActiveFinalReport(selectedId).then(function() {
            $('body').addClass('printing-final-report');
            resizeReportPlots(document.querySelector('.final-report-content.active-report'));

            setTimeout(function() {
                window.print();
                setTimeout(function() {
                    $('body').removeClass('printing-final-report');
                    handleVisiblePlotResize();
                }, 250);
            }, 300);
        });
    });

    $('#download-word-report').on('click', function() {
        var selectedId = $('input[name="final_model_option"]:checked').val();
        if (!selectedId) {
            return;
        }
        downloadSelectedWordReport(selectedId, $(this));
    });
});
