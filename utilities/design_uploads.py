"""File roles and form mappings for the stateless study-design frontend.

Keep this metadata independent of the numerical/import modules so starting
the Flask app does not import the IVIVC application or numerical workflow.
"""


def _course(identifier, file_field, label, series_label=None):
    return {
        'id': identifier, 'file_field': file_field, 'label': label, 'kind': 'course',
        'series': [{'key': 'profile', 'label': series_label or label,
                    'time_default': 0, 'response_default': 1}],
    }


DATASETS = {
    'reference-file': _course('reference-file', 'reference_file', 'Prior in vivo reference data'),
    'ivivc-calibration-file': {
        'id': 'ivivc-calibration-file', 'file_field': 'ivivc_calibration_file',
        'label': 'Paired in vitro/in vivo calibration data', 'kind': 'pair',
        'series': [
            {'key': 'in_vitro', 'label': 'In vitro calibration', 'time_default': 0, 'response_default': 1},
            {'key': 'in_vivo', 'label': 'In vivo calibration', 'time_default': 2, 'response_default': 3},
        ],
    },
    'prediction-file': _course('prediction-file', 'prediction_file', 'Target in vitro prediction data'),
    'scaling-source-file': _course('scaling-source-file', 'scaling_source_file', 'Target in vitro time-course data'),
    'time-scale-file': {
        'id': 'time-scale-file', 'file_field': 'time_scale_file',
        'label': 'Empirical time-scale distribution', 'kind': 'ratios', 'series': [],
    },
    'existing-in-vitro-file': _course('existing-in-vitro-file', 'existing_in_vitro_file',
                                     'Existing in vitro calibration data'),
    'existing-in-vivo-file': _course('existing-in-vivo-file', 'existing_in_vivo_file',
                                    'Existing in vivo calibration data'),
    'cal-in-vitro-source-file': _course('cal-in-vitro-source-file', 'cal_in_vitro_source_file',
                                       'Expected in vitro calibration trajectory source'),
    'cal-in-vivo-source-file': _course('cal-in-vivo-source-file', 'cal_in_vivo_source_file',
                                      'Expected in vivo calibration trajectory source'),
    'cal-target-file': _course('cal-target-file', 'cal_target_file',
                               'Independent target in vitro prediction data'),
    'completed-target-file': _course('completed-target-file', 'completed_target_file',
                                    'Completed target-study observations'),
}
for _identifier, _field, _label in (
    ('future-sd-profile-file', 'future_sd_profile_file', 'Future in vivo individual-SD profile'),
    ('calibration-in-vitro-sd-profile-file', 'calibration_in_vitro_sd_profile_file',
     'Future calibration in vitro individual-SD profile'),
    ('calibration-in-vivo-sd-profile-file', 'calibration_in_vivo_sd_profile_file',
     'Future calibration in vivo individual-SD profile'),
):
    DATASETS[_identifier] = {
        'id': _identifier, 'file_field': _field, 'label': _label,
        'kind': 'sd_profile', 'series': [],
    }
del _identifier, _field, _label


COURSE_FIELDS = (
    'time_column', 'response_column', 'uncertainty', 'error_source',
    'error_value', 'error_column', 'count_source', 'count_value',
    'count_column', 'response_units',
)


def options_for(form, prefix, fields):
    """Keep absent options absent; an explicitly empty mapping is not a default."""
    return {field: form[prefix + field] for field in fields if prefix + field in form}
