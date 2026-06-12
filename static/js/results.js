$(function() {
    $("#vertical-tabs").tabs().addClass("ui-tabs-vertical ui-helper-clearfix");
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


// Final model/report selector
$(document).ready(function() {
    $('input[name="final_model_option"]').on('change', function() {
        var selectedId = $(this).val();
        $('.final-report-content').hide();
        $('#' + selectedId + '-report').show();

        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([document.getElementById(selectedId + '-report')]);
        }
    });

    $('#print-final-report').on('click', function() {
        var reportHtml = $('.final-report-content:visible').html();
        if (!reportHtml) {
            return;
        }
        var printWindow = window.open('', '_blank');
        printWindow.document.write('<html><head><title>IVIVC Final Model Report</title></head><body>');
        printWindow.document.write(reportHtml);
        printWindow.document.write('</body></html>');
        printWindow.document.close();
        printWindow.focus();
        printWindow.print();
    });
});
