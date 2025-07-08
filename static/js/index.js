/*$(document).ready(function() {
    console.log('Document ready');
});

    function populateSheetSelection(sheets) {
        var sheetSelect = $('#sheet');
        sheetSelect.empty();
        sheets.forEach(function(sheet) {
            sheetSelect.append($('<option></option>').attr('value', sheet).text(sheet));
        });
        $('#sheet-selection').show();

        var selectedSheet = sessionStorage.getItem('selectedSheet');
        if (selectedSheet) {
            sheetSelect.val(selectedSheet);
        }
    }

    function clearSheetSelection() {
        $('#sheet').empty();
        $('#sheet-selection').hide();
        sessionStorage.removeItem('sheetNames');
        sessionStorage.removeItem('selectedSheet');
    }

    function handleFileSelection() {
        var file = $('#file')[0].files[0];
        if (file) {
            var fileName = file.name;
            var fileExtension = fileName.split('.').pop().toLowerCase();

            if (fileExtension === 'csv') {
                clearSheetSelection();
            } else {
                var formData = new FormData();
                formData.append('file', file);

                $.ajax({
                    url: '/get_sheets',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        console.log('Sheets received:', response);
                        sessionStorage.setItem('sheetNames', JSON.stringify(response.sheets));
                        populateSheetSelection(response.sheets);
                    },
                    error: function(xhr, status, error) {
                        console.error('Error getting sheets:', status, error);
                        alert('Error getting sheet names. Please try again.');
                        clearSheetSelection();
                    }
                });
            }
        } else {
            clearSheetSelection();
        }
    }

    // Check if file is selected on page load
    if ($('#file')[0].files.length === 0) {
        clearSheetSelection();
    } else {
        handleFileSelection();
    }

    $('#file').on('change', handleFileSelection);

    $('#upload-form').on('submit', function() {
        if ($('#sheet-selection').is(':visible')) {
            sessionStorage.setItem('selectedSheet', $('#sheet').val());
        }
    });

function populateSheetSelection(sheets, sheetSelectId, sheetSelectionId, storageKey) {
    var sheetSelect = $(sheetSelectId);
    sheetSelect.empty();
    sheets.forEach(function(sheet) {
        sheetSelect.append($('<option></option>').attr('value', sheet).text(sheet));
    });
    $(sheetSelectionId).show();

    var selectedSheet = sessionStorage.getItem(storageKey);
    if (selectedSheet) {
        sheetSelect.val(selectedSheet);
    }
}

function clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    $(sheetSelectId).empty();
    $(sheetSelectionId).hide();
    sessionStorage.removeItem(sheetStorageKey);
    sessionStorage.removeItem(selectedSheetStorageKey);
}

function handleFileSelection(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    var file = $(fileInputId)[0].files[0];
    if (file) {
        var fileName = file.name;
        var fileExtension = fileName.split('.').pop().toLowerCase();
        if (fileExtension === 'csv') {
            clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
        } else {
            var formData = new FormData();
            formData.append('file', file);
            $.ajax({
                url: '/get_sheets',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    console.log('Sheets received for ' + fileInputId + ':', response);
                    sessionStorage.setItem(sheetStorageKey, JSON.stringify(response.sheets));
                    populateSheetSelection(response.sheets, sheetSelectId, sheetSelectionId, selectedSheetStorageKey);
                },
                error: function(xhr, status, error) {
                    console.error('Error getting sheets for ' + fileInputId + ':', status, error);
                    alert('Error getting sheet names. Please try again.');
                    clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
                }
            });
        }
    } else {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
    }
}

// Main file handlers
function handleMainFileSelection() {
    handleFileSelection('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet');
}

// Prediction file handlers
function handlePredictionFileSelection() {
    handleFileSelection('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet');
}

// Check if files are selected on page load
if ($('#file')[0].files.length === 0) {
    clearSheetSelection('#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet');
} else {
    handleMainFileSelection();
}

if ($('#prediction_file')[0].files.length === 0) {
    clearSheetSelection('#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet');
} else {
    handlePredictionFileSelection();
}

// Event listeners
$('#file').on('change', handleMainFileSelection);
$('#prediction_file').on('change', handlePredictionFileSelection);

$('#upload-form').on('submit', function() {
    // Save selected sheets to session storage
    if ($('#sheet-selection').is(':visible')) {
        sessionStorage.setItem('selectedSheet', $('#sheet').val());
    }
    if ($('#sheet-selection2').is(':visible')) {
        sessionStorage.setItem('selectedPredictionSheet', $('#sheet2').val());
    }
});

$('input[name="approaches"]').change(function() {
    var approach = $(this).val();
    var isChecked = $(this).prop('checked');
    $('input[name="models"]').each(function() {
        if ($(this).val().startsWith(approach + ':')) {
            $(this).prop('disabled', !isChecked);
            if (!isChecked) {
                $(this).prop('checked', false);
            }
        }
    });
});

$('#upload-form').on('submit', function(e) {
    var checkedModels = $('input[name="models"]:checked');
    if (checkedModels.length === 0) {
        e.preventDefault();
        alert('Please select at least one model for analysis.');
    }
});
*/

function populateSheetSelection(sheets, sheetSelectId, sheetSelectionId, storageKey) {
    var sheetSelect = $(sheetSelectId);
    sheetSelect.empty();
    sheets.forEach(function(sheet) {
        sheetSelect.append($('<option></option>').attr('value', sheet).text(sheet));
    });
    $(sheetSelectionId).show();

    var selectedSheet = sessionStorage.getItem(storageKey);
    if (selectedSheet) {
        sheetSelect.val(selectedSheet);
    }
}

function clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    $(sheetSelectId).empty();
    $(sheetSelectionId).hide();
    sessionStorage.removeItem(sheetStorageKey);
    sessionStorage.removeItem(selectedSheetStorageKey);
}

function handleFileSelection(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    var file = $(fileInputId)[0].files[0];
    if (file) {
        var fileName = file.name;
        var fileExtension = fileName.split('.').pop().toLowerCase();
        if (fileExtension === 'csv') {
            clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
        } else {
            var formData = new FormData();
            formData.append('file', file);
            $.ajax({
                url: '/get_sheets',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    console.log('Sheets received for ' + fileInputId + ':', response);
                    sessionStorage.setItem(sheetStorageKey, JSON.stringify(response.sheets));
                    populateSheetSelection(response.sheets, sheetSelectId, sheetSelectionId, selectedSheetStorageKey);
                },
                error: function(xhr, status, error) {
                    console.error('Error getting sheets for ' + fileInputId + ':', status, error);
                    alert('Error getting sheet names. Please try again.');
                    clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
                }
            });
        }
    } else {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
    }
}

// NEW: Clear file input function
function clearFileInput(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    $(fileInputId)[0].value = '';
    clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
    $(fileInputId).trigger('change');
}

// Main file handlers
function handleMainFileSelection() {
    handleFileSelection('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet');
}

// NEW: Clear main file function
function clearMainFile() {
    clearFileInput('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet');
}

// Prediction file handlers
function handlePredictionFileSelection() {
    handleFileSelection('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet');
}

// NEW: Clear prediction file function
function clearPredictionFile() {
    clearFileInput('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet');
}

// Check if files are selected on page load
if ($('#file')[0].files.length === 0) {
    clearSheetSelection('#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet');
} else {
    handleMainFileSelection();
}

if ($('#prediction_file')[0].files.length === 0) {
    clearSheetSelection('#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet');
} else {
    handlePredictionFileSelection();
}

// Event listeners
$('#file').on('change', handleMainFileSelection);
$('#prediction_file').on('change', handlePredictionFileSelection);

// NEW: Clear button event listeners
$('.clear-file-btn').on('click', function() {
    var target = $(this).data('target');
    if (target === 'file') {
        clearMainFile();
    } else if (target === 'prediction_file') {
        clearPredictionFile();
    }
});

$('#upload-form').on('submit', function() {
    // Save selected sheets to session storage
    if ($('#sheet-selection').is(':visible')) {
        sessionStorage.setItem('selectedSheet', $('#sheet').val());
    }
    if ($('#sheet-selection2').is(':visible')) {
        sessionStorage.setItem('selectedPredictionSheet', $('#sheet2').val());
    }
});

$('input[name="approaches"]').change(function() {
    var approach = $(this).val();
    var isChecked = $(this).prop('checked');
    $('input[name="models"]').each(function() {
        if ($(this).val().startsWith(approach + ':')) {
            $(this).prop('disabled', !isChecked);
            if (!isChecked) {
                $(this).prop('checked', false);
            }
        }
    });
});

$('#upload-form').on('submit', function(e) {
    var checkedModels = $('input[name="models"]:checked');
    if (checkedModels.length === 0) {
        e.preventDefault();
        alert('Please select at least one model for analysis.');
    }
});

