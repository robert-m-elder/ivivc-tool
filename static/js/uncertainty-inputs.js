'use strict';

(function () {
    function updateUncertaintyInputs() {
        var form = document.getElementById('upload-form');
        if (!form) return;
        var groups = form.querySelectorAll('[data-uncertainty-prefix]');
        var active = [];
        groups.forEach(function (group) {
            var prefix = group.getAttribute('data-uncertainty-prefix');
            var mode = document.getElementById(prefix + '_uncertainty_mode');
            var kind = document.getElementById(prefix + '_response_kind');
            var supplied = mode.value !== 'none';
            mode.setCustomValidity('');
            if (supplied) active.push(mode);
            group.querySelectorAll('[data-uncertainty-field]').forEach(function (row) {
                var field = row.getAttribute('data-uncertainty-field');
                var show = supplied && (field === 'kind' || field === mode.value ||
                    (field === 'counts' && kind.value === 'mean'));
                row.hidden = !show;
                row.querySelectorAll('input, select').forEach(function (control) {
                    control.disabled = !show;
                    control.required = show && (field === 'constant' || field === 'column');
                    control.setCustomValidity('');
                });
            });
            var value = document.getElementById(prefix + '_uncertainty_value');
            if (mode.value === 'constant' && value.value !== '' &&
                    (!Number.isFinite(Number(value.value)) || Number(value.value) <= 0)) {
                value.setCustomValidity('Enter a finite SD or SEM greater than zero.');
            }
        });
        var future = document.getElementById('future_uncertainty_mode');
        if (future) {
            future.setCustomValidity('');
            ['prediction', 'calibration'].forEach(function (source) {
                var prefix = source === 'calibration' ? 'in_vivo' : 'prediction';
                future.querySelector('option[value="' + source + '"]').disabled =
                    document.getElementById(prefix + '_uncertainty_mode').value === 'none';
            });
            var constant = future.value === 'constant';
            document.getElementById('future-uncertainty-group').querySelectorAll('[data-future-field]').forEach(function (row) {
                row.hidden = !constant;
                row.querySelectorAll('input, select').forEach(function (control) {
                    control.disabled = !constant;
                    control.required = constant;
                    control.setCustomValidity('');
                });
            });
            var futureValue = document.getElementById('future_uncertainty_value');
            if (constant && futureValue.value !== '' &&
                    (!Number.isFinite(Number(futureValue.value)) || Number(futureValue.value) <= 0)) {
                futureValue.setCustomValidity('Enter a finite future SD or SEM greater than zero.');
            }
            if (future.value !== 'none') active.push(future);
        }
        var confirmed = document.getElementById('uncertainty_independence_confirmed');
        confirmed.disabled = !active.length;
        confirmed.required = Boolean(active.length);
        document.getElementById('uncertainty-independence-row').hidden = !active.length;
        if (!active.length) return;

        var selected = Array.from(form.querySelectorAll('input[name="approaches"]:checked'));
        var incompatible = selected.some(function (control) { return control.value !== 'approach2'; });
        var logResponse = form.querySelector('input[name="scalings"][value="log_y"]:checked');
        active.forEach(function (mode) {
            if (incompatible) {
                mode.setCustomValidity('Supplied uncertainty requires Approach 2 alone.');
            } else if (logResponse) {
                mode.setCustomValidity('Log10 response scaling is unavailable with supplied SD or SEM.');
            } else if ((mode.id === 'prediction_uncertainty_mode' || mode.id === 'future_uncertainty_mode') &&
                    !document.getElementById('prediction_file').files.length) {
                mode.setCustomValidity('Upload a prediction dataset or set this uncertainty source to Not supplied.');
            } else if (mode === future && mode.value === 'prediction' &&
                    document.getElementById('prediction_uncertainty_mode').value === 'none') {
                mode.setCustomValidity('Supply prediction-input SD/SEM before reusing it for future measurements.');
            } else if (mode === future && mode.value === 'calibration' &&
                    document.getElementById('in_vivo_uncertainty_mode').value === 'none') {
                mode.setCustomValidity('Supply in vivo calibration SD/SEM before reusing it for future measurements.');
            }
        });
    }

    function initialize() {
        var form = document.getElementById('upload-form');
        if (!form) return;
        updateUncertaintyInputs();
        form.addEventListener('change', updateUncertaintyInputs);
        form.addEventListener('input', updateUncertaintyInputs);
        // Refresh validity before index.js collects the existing form errors.
        form.addEventListener('submit', function () {
            updateUncertaintyInputs();
            if (!document.getElementById('uncertainty_independence_confirmed').disabled) {
                document.getElementById('uncertainty-options').open = true;
            }
        }, true);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
    window.addEventListener('pageshow', updateUncertaintyInputs);
}());
