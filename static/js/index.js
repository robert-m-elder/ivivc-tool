'use strict';

function announceFormStatus(message) {
    if (window.IVIVCAccessibility) {
        window.IVIVCAccessibility.setLiveMessage('form-status', message);
    }
}

function populateSheetSelection(sheets, sheetSelectId, sheetSelectionId, storageKey, announce) {
    var sheetSelect = $(sheetSelectId);
    sheetSelect.empty();
    sheetSelect.append($('<option></option>').attr('value', '').text('Select a sheet...'));
    sheets.forEach(function(sheet) {
        sheetSelect.append($('<option></option>').attr('value', sheet).text(sheet));
    });

    var selection = $(sheetSelectionId);
    selection.prop('hidden', false).attr('aria-busy', 'false');

    var selectedSheet = sessionStorage.getItem(storageKey);
    if (selectedSheet) {
        sheetSelect.val(selectedSheet);
    }
    if (announce !== false) {
        announceFormStatus(sheets.length + ' worksheet' + (sheets.length === 1 ? '' : 's') + ' available. Select the worksheet to analyze.');
    }
}

function getFileExtension(file) {
    return file && file.name.indexOf('.') !== -1 ? file.name.split('.').pop().toLowerCase() : '';
}

function getFileFingerprint(file) {
    if (!file) {
        return '';
    }
    return [file.name, file.size, file.lastModified || 0].join('|');
}

function getStoredSheetRecord(storageKey) {
    var storedValue = sessionStorage.getItem(storageKey);
    if (!storedValue) {
        return null;
    }

    try {
        var parsed = JSON.parse(storedValue);
        if (Array.isArray(parsed)) {
            return {sheets: parsed, fingerprint: ''};
        }
        if (parsed && Array.isArray(parsed.sheets)) {
            return parsed;
        }
    } catch (error) {
        console.warn('Could not restore worksheet names from session storage:', error);
    }
    return null;
}

function hideSheetSelection(sheetSelectId, sheetSelectionId) {
    $(sheetSelectId).empty().append($('<option></option>').attr('value', '').text('Select a sheet...'));
    $(sheetSelectionId).prop('hidden', true).attr('aria-busy', 'false');
}

function clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey) {
    hideSheetSelection(sheetSelectId, sheetSelectionId);
    sessionStorage.removeItem(sheetStorageKey);
    sessionStorage.removeItem(selectedSheetStorageKey);
}

function showClientErrors(errors) {
    var summary = $('#form-error-summary');
    summary.empty();

    if (!errors.length) {
        summary.prop('hidden', true);
        return;
    }

    var heading = $('<h2></h2>').text('There is a problem with the analysis settings');
    var list = $('<ul></ul>');
    var describedTargets = {};

    errors.forEach(function(error) {
        var item = $('<li></li>');
        if (error.targetId && document.getElementById(error.targetId)) {
            item.append($('<a></a>').attr('href', '#' + error.targetId).text(error.message));
            describedTargets[error.targetId] = true;
            $('#' + error.targetId).attr('aria-invalid', 'true').attr('aria-describedby', 'form-error-summary');
        } else {
            item.text(error.message);
        }
        list.append(item);
    });

    summary.append(heading, list).prop('hidden', false);
    summary[0].focus();
}

function clearClientErrors() {
    $('#form-error-summary').prop('hidden', true).empty();
    $('.field-error').remove();
    $('#upload-form [aria-invalid="true"]').removeAttr('aria-invalid').each(function() {
        $(this).removeAttr('aria-describedby');
    });
}

function handleFileSelection(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey, datasetLabel, resetStoredState) {
    var file = $(fileInputId)[0].files[0];
    if (!file) {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
        return;
    }

    var fileExtension = getFileExtension(file);
    if (fileExtension === 'csv') {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
        announceFormStatus(datasetLabel + ' CSV file selected. Worksheet selection is not needed.');
        return;
    }

    if (resetStoredState) {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
    }

    var selection = $(sheetSelectionId);
    selection.prop('hidden', false).attr('aria-busy', 'true');
    announceFormStatus('Loading worksheet names for the ' + datasetLabel.toLowerCase() + '.');

    var formData = new FormData();
    formData.append('file', file);
    $.ajax({
        url: '/get_sheets',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            sessionStorage.setItem(sheetStorageKey, JSON.stringify({
                sheets: response.sheets,
                fingerprint: getFileFingerprint(file)
            }));
            populateSheetSelection(response.sheets, sheetSelectId, sheetSelectionId, selectedSheetStorageKey, true);
        },
        error: function(xhr, status, error) {
            console.error('Error getting sheets for ' + fileInputId + ':', status, error);
            clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
            showClientErrors([{
                message: 'Worksheet names could not be read from the selected ' + datasetLabel.toLowerCase() + '. Select a valid Excel file and try again.',
                targetId: fileInputId.replace('#', '')
            }]);
        }
    });
}

function clearFileInput(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey, datasetLabel) {
    $(fileInputId)[0].value = '';
    clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
    announceFormStatus(datasetLabel + ' selection cleared.');
}

function handleMainFileSelection() {
    handleFileSelection('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet', 'Fitting dataset', true);
}

function clearMainFile() {
    clearFileInput('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet', 'Fitting dataset');
}

function handlePredictionFileSelection() {
    handleFileSelection('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet', 'Prediction dataset', true);
}

function clearPredictionFile() {
    clearFileInput('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet', 'Prediction dataset');
}


function restoreFileSelection(fileInputId, sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey, datasetLabel) {
    var input = $(fileInputId)[0];
    var file = input && input.files ? input.files[0] : null;

    if (!file) {
        // Browsers may restore file inputs after DOM ready when navigating back.
        // Hide the selector for now but retain stored worksheet metadata for pageshow.
        hideSheetSelection(sheetSelectId, sheetSelectionId);
        return;
    }

    if (getFileExtension(file) === 'csv') {
        clearSheetSelection(sheetSelectId, sheetSelectionId, sheetStorageKey, selectedSheetStorageKey);
        return;
    }

    var storedRecord = getStoredSheetRecord(sheetStorageKey);
    var fingerprint = getFileFingerprint(file);
    var recordMatchesFile = storedRecord && (!storedRecord.fingerprint || storedRecord.fingerprint === fingerprint);

    if (recordMatchesFile && storedRecord.sheets.length) {
        populateSheetSelection(storedRecord.sheets, sheetSelectId, sheetSelectionId, selectedSheetStorageKey, false);
        return;
    }

    sessionStorage.removeItem(sheetStorageKey);
    sessionStorage.removeItem(selectedSheetStorageKey);

    if ($(sheetSelectionId).attr('aria-busy') !== 'true') {
        handleFileSelection(
            fileInputId,
            sheetSelectId,
            sheetSelectionId,
            sheetStorageKey,
            selectedSheetStorageKey,
            datasetLabel,
            false
        );
    }
}

function restoreFileSelections() {
    restoreFileSelection('#file', '#sheet', '#sheet-selection', 'sheetNames', 'selectedSheet', 'Fitting dataset');
    restoreFileSelection('#prediction_file', '#sheet2', '#sheet-selection2', 'predictionSheetNames', 'selectedPredictionSheet', 'Prediction dataset');
    $('#upload-form').removeAttr('aria-busy');
}

function updateApproachModelState(approachInput) {
    var approach = approachInput.val();
    var isChecked = approachInput.prop('checked');
    $('#' + approach + '-models').attr('aria-disabled', isChecked ? 'false' : 'true');

    $('input[name="models"]').each(function() {
        if ($(this).val().startsWith(approach + ':')) {
            $(this).prop('disabled', !isChecked);
            $(this).closest('label').toggleClass('checkbox-disabled', !isChecked);
            if (!isChecked) {
                $(this).prop('checked', false);
            }
        }
    });
}

function updateCrossValidationSettingState(resetValues) {
    const scheme = $('#cv_scheme').val();
    const controls = {
        nSplits: $('#cv_n_splits'),
        nSplitsLabel: $('#cv_n_splits_label'),
        testSize: $('#cv_test_size'),
        randomState: $('#cv_random_state')
    };

    function setFieldState(field, enabled) {
        field.prop('disabled', !enabled);
        field.closest('.advanced-field').toggleClass('advanced-field-disabled', !enabled);
    }

    if (resetValues) {
        if (scheme === 'shuffle_split') {
            controls.nSplits.val('20');
            controls.testSize.val('0.25');
        } else if (scheme === 'leave_contiguous_block_out') {
            controls.nSplits.val('3');
            controls.testSize.val('0.25');
        }
    }

    const usesSplitsField = scheme === 'shuffle_split' || scheme === 'leave_contiguous_block_out';
    const usesTestSize = scheme === 'shuffle_split';
    const usesRandomState = scheme === 'shuffle_split';
    const splitLabels = {
        'shuffle_split': 'Splits',
        'leave_contiguous_block_out': 'Block size'
    };

    controls.nSplitsLabel.text(splitLabels[scheme] || 'Splits/block size');
    setFieldState(controls.nSplits, usesSplitsField);
    setFieldState(controls.testSize, usesTestSize);
    setFieldState(controls.randomState, usesRandomState);
}

function collectFormErrors(form) {
    var errors = [];
    var seenTargets = {};

    Array.from(form.elements).forEach(function(control) {
        if (!control.disabled && typeof control.checkValidity === 'function' && !control.checkValidity()) {
            var targetId = control.id || '';
            if (!seenTargets[targetId]) {
                errors.push({message: control.validationMessage, targetId: targetId});
                seenTargets[targetId] = true;
            }
        }
    });

    var selectedApproaches = $('input[name="approaches"]:checked').map(function() { return this.value; }).get();
    if (selectedApproaches.length === 0) {
        errors.push({message: 'Select at least one analysis approach.', targetId: 'approaches-group'});
    }

    var selectedParametricApproaches = selectedApproaches.filter(function(value) { return value !== 'approach3'; });
    if (selectedParametricApproaches.length > 0 && $('input[name="models"]:checked:not(:disabled)').length === 0) {
        errors.push({message: 'Select at least one model for the selected parametric approach.', targetId: 'approaches-group'});
    }

    if ($('input[name="metrics"]:checked').length === 0) {
        errors.push({message: 'Select at least one performance metric.', targetId: 'metrics-group'});
    }

    var fittingFile = $('#file')[0].files[0];
    if (fittingFile && ['xls', 'xlsx'].indexOf(getFileExtension(fittingFile)) !== -1 && !$('#sheet').val()) {
        errors.push({
            message: $('#sheet-selection').attr('aria-busy') === 'true'
                ? 'Wait for the fitting-dataset worksheet names to finish loading.'
                : 'Select a worksheet for the fitting dataset.',
            targetId: $('#sheet-selection').prop('hidden') ? 'file' : 'sheet'
        });
    }

    var predictionFile = $('#prediction_file')[0].files[0];
    if (predictionFile && ['xls', 'xlsx'].indexOf(getFileExtension(predictionFile)) !== -1 && !$('#sheet2').val()) {
        errors.push({
            message: $('#sheet-selection2').attr('aria-busy') === 'true'
                ? 'Wait for the prediction-dataset worksheet names to finish loading.'
                : 'Select a worksheet for the prediction dataset.',
            targetId: $('#sheet-selection2').prop('hidden') ? 'prediction_file' : 'sheet2'
        });
    }

    return errors;
}

$(document).ready(function() {
    restoreFileSelections();
    updateCrossValidationSettingState(false);
    $('#cv_scheme').on('change', function() {
        updateCrossValidationSettingState(true);
        announceFormStatus('Cross-validation controls updated for ' + $('#cv_scheme option:selected').text() + '.');
    });

    $('#file').on('change', handleMainFileSelection);
    $('#prediction_file').on('change', handlePredictionFileSelection);
    $('#sheet').on('change', function() {
        sessionStorage.setItem('selectedSheet', this.value);
    });
    $('#sheet2').on('change', function() {
        sessionStorage.setItem('selectedPredictionSheet', this.value);
    });

    $('.clear-file-btn').on('click', function() {
        var target = $(this).data('target');
        if (target === 'file') {
            clearMainFile();
        } else if (target === 'prediction_file') {
            clearPredictionFile();
        }
    });

    $('input[name="approaches"]').on('change', function() {
        updateApproachModelState($(this));
    }).each(function() {
        updateApproachModelState($(this));
    });

    $('#upload-form').on('submit', function(event) {
        clearClientErrors();
        var errors = collectFormErrors(this);
        if (errors.length) {
            event.preventDefault();
            showClientErrors(errors);
            return;
        }

        if (!$('#sheet-selection').prop('hidden')) {
            sessionStorage.setItem('selectedSheet', $('#sheet').val());
        }
        if (!$('#sheet-selection2').prop('hidden')) {
            sessionStorage.setItem('selectedPredictionSheet', $('#sheet2').val());
        }
        $('#upload-form').attr('aria-busy', 'true');
        announceFormStatus('Analysis submitted. Results are loading.');
    });

    $(document).on('click', '#form-error-summary a', function() {
        var target = document.querySelector(this.getAttribute('href'));
        if (target) {
            window.setTimeout(function() { target.focus(); }, 0);
        }
    });

    if (!$('#form-error-summary').prop('hidden') && $('#form-error-summary').children().length) {
        $('#form-error-summary')[0].focus();
    }

    if (window.IVIVCAccessibility) {
        window.IVIVCAccessibility.initializeHelpModal({
            'data-help': 'data-help-content',
            'fitting-data-help': 'fitting-data-help-content',
            'prediction-data-help': 'prediction-data-help-content',
            'fitting-sheet-help': 'fitting-sheet-help-content',
            'prediction-sheet-help': 'prediction-sheet-help-content',
            'approaches-help': 'approaches-help-content',
            'approach1-help': 'approach1-help-content',
            'approach2-help': 'approach2-help-content',
            'approach3-help': 'approach3-help-content',
            'preprocessing-help': 'preprocessing-help-content',
            'normalization-help': 'normalization-help-content',
            'scaling-help': 'scaling-help-content',
            'interpolation-help': 'interpolation-help-content',
            'metrics-help': 'metrics-help-content',
            'advanced-analysis-help': 'advanced-analysis-help-content',
            'residual-diagnostics-help': 'residual-diagnostics-help-content',
            'cross-validation-help': 'cross-validation-help-content',
            'grid-search-help': 'grid-search-help-content',
            'r2-help': 'r2-help-content'
        });
    }
});

$(window).on('pageshow', function() {
    restoreFileSelections();
});
