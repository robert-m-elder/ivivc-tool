$(function() {
    $("#vertical-tabs").tabs({
        activate: function(event, ui) {
            handleVisiblePlotResize();
        }
    }).addClass("ui-tabs-vertical ui-helper-clearfix");
    $("#vertical-tabs li").removeClass("ui-corner-top").addClass("ui-corner-left");
});

// DataTables
/*$.extend(true, $.fn.dataTable.defaults, {
    paging: false,
    searching: false,
    info: false,
    ordering: true,
    responsive: true,
    dom: 'frtip<"custom-button-container"B>',
    buttons: [
        {
            extend: 'excel',
            text: 'Download Data'
        }
    ],
    layout: {
        bottomStart: {
            buttons: ['excel']
        }
    },
});

$(document).ready(function() {
    $('.dataframe').DataTable();
});*/

// DataTables default configuration
$.extend(true, $.fn.dataTable.defaults, {
    paging: true,
    pageLength: 10,
    searching: false,
    info: false,
    ordering: true,
    order: [],
    responsive: true,
    dom: 'frtip<"custom-button-container"B>',
    buttons: [
        {
            extend: 'excel',
            text: 'Download Data'
        }
    ],
    layout: {
        bottomStart: {
            buttons: ['excel']
        }
    },
});

$(document).ready(function() {
    // Initialize existing visible DataTables
    $('.dataframe').DataTable();

    // Handle show/hide data table buttons for plot data tables
    $('.show-data-btn').on('click', function() {
        var targetTable = $(this).data('target');
        var container = $('#' + targetTable + '-container');
        var button = $(this);

        if (container.is(':visible')) {
            // Hide table
            container.hide();
            button.find('span').text('Show Data Table');

            // Destroy DataTable if it exists
            var table = container.find('table.display');
            if ($.fn.DataTable.isDataTable(table)) {
                table.DataTable().destroy();
            }
        } else {
            // Show table and initialize DataTable
            container.show();
            button.find('span').text('Hide Data Table');

            // Find the table within the container and initialize DataTable
            var table = container.find('table.display');
            if (table.length > 0) {
                // Initialize with the same configuration as existing tables
                // but with additional export buttons for plot data
                table.DataTable({
                    paging: false,
                    searching: false,
                    info: false,
                    ordering: true,
                    order: [],
                    responsive: true,
                    dom: 'frtip<"custom-button-container"B>',
                    buttons: [
                        {
                            extend: 'copy',
                            text: 'Copy'
                        },
                        {
                            extend: 'csv',
                            text: 'CSV'
                        },
                        {
                            extend: 'excel',
                            text: 'Excel'
                        },
                        {
                            extend: 'pdf',
                            text: 'PDF'
                        },
                        {
                            extend: 'print',
                            text: 'Print'
                        }
                    ],
                    layout: {
                        bottomStart: {
                            buttons: ['copy', 'csv', 'excel', 'pdf', 'print']
                        }
                    },
                    columnDefs: [
                        {
                            targets: '_all',
                            className: 'dt-center'
                        }
                    ]
                });
            }
        }
    });
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
var observer = new MutationObserver(debounce(function() { ensureResizePlots(false); }, 250));

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });

// Modal functionality for results page
$(document).ready(function() {
    // Modal content mapping
    const modalContent = {
        'data-processing-help': 'data-processing-help-content',
        'model-comparison-help': 'model-comparison-help-content',
        'final-model-report-help': 'final-model-report-help-content',
        'approach1-results-help': 'approach1-results-help-content',
        'approach2-results-help': 'approach2-results-help-content',
        'approach3-results-help': 'approach3-results-help-content',
        'prediction-approach1-help': 'prediction-approach1-help-content',
        'prediction-approach2-help': 'prediction-approach2-help-content',
        'prediction-approach3-help': 'prediction-approach3-help-content'
    };

    // Open modal when help button is clicked
    $('.help-btn').on('click', function(e) {
        e.preventDefault();
        e.stopPropagation();

        const modalId = $(this).data('modal');
        const contentId = modalContent[modalId];

        if (contentId) {
            const content = $('#' + contentId).html();
            if (content) {
                $('#modal-text').html(content);
                $('#help-modal').fadeIn(300);
                $('body').addClass('modal-open');
            } else {
                // Fallback content
                $('#modal-text').html('<h4>Help</h4><p>Help information for this section is coming soon.</p>');
                $('#help-modal').fadeIn(300);
                $('body').addClass('modal-open');
            }
        }
    });

    // Close modal when X is clicked
    $('.close').on('click', function() {
        closeModal();
    });

    // Close modal when close button is clicked
    $('.modal-close-btn').on('click', function() {
        closeModal();
    });

    // Close modal when clicking outside of it
    $('#help-modal').on('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });

    // Close modal with Escape key
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' && $('#help-modal').is(':visible')) {
            closeModal();
        }
    });

    function closeModal() {
        $('#help-modal').fadeOut(300);
        $('body').removeClass('modal-open');
    }

    // Prevent modal from closing when clicking inside modal content
    $('.modal-content').on('click', function(e) {
        e.stopPropagation();
    });
});


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
    $('.final-report-content').hide().removeClass('active-report');
    var report = document.getElementById(selectedId + '-report');
    if (!report) {
        return Promise.resolve();
    }

    $(report).show().addClass('active-report');

    return populateReportPlots(report).then(function() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            return window.MathJax.typesetPromise([report]);
        }
    }).then(function() {
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
    button.prop('disabled', true);
    button.find('span').text('Preparing Word Report...');

    setActiveFinalReport(selectedId)
        .then(function() {
            return exportReportPlotImages(report);
        })
        .then(function(imageExportSummary) {
            if (imageExportSummary && imageExportSummary.total > 0 && imageExportSummary.exported === 0) {
                console.warn('No report plot images were exported for the Word report.');
                alert('No report plot images could be exported. The Word report will still download and will include notes where images were unavailable.');
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
            });
        })
        .catch(function(error) {
            alert(error.message || 'Unable to generate Word report.');
        })
        .finally(function() {
            button.prop('disabled', false);
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
        setActiveFinalReport($(this).val());
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
