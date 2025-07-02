$(document).ready(function() {
    console.log('Document ready');

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
});

