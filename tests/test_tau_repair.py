"""Minimal integral-tau safeguards and separate point/uncertainty availability."""
from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning

from utilities import time_constants
from utilities.evaluation import calculate_tau, calculate_tau_with_uncertainty
from utilities.prediction_validity import describe_tau_prediction_skip_reason
from ivivc_core import TimeCourse, fit_profile, predict_tau_ratio
from ivivc_core.uncertainty import _tau_from_parameter_vector, sample_tau_ratio_trajectories
from models import models
from models.exponential import model_function as exponential
from models.double_exponential import model_function as double_exponential


@pytest.mark.parametrize('amplitude,rate', [(1., -.1), (3., -.2), (.005, -.05)])
def test_checked_tau_agrees_with_exponential_integral(amplitude, rate):
    # Closed-form values are test references, not production special cases.
    tau, reason = time_constants.evaluate_tau(exponential, [amplitude, rate])
    assert reason is None
    assert tau == pytest.approx(-1. / rate)
    assert calculate_tau(exponential, [amplitude, rate]) == tau
    assert _tau_from_parameter_vector(exponential, [amplitude, rate]) == tau


def test_generic_evaluator_handles_a_nonexponential_curve():
    def decay(time, scale):
        return (1. + time / scale) ** -2
    tau, reason = time_constants.evaluate_tau(decay, [7.])
    assert reason is None
    assert tau == pytest.approx(7.)


def test_no_model_specific_tau_formulas_are_introduced(monkeypatch):
    calls = []
    def quad(function, lower, upper, **kwargs):
        calls.append((lower, upper, kwargs))
        assert function(0.) == 2.
        return 20., 1e-9
    monkeypatch.setattr(time_constants, 'quad', quad)
    assert calculate_tau(exponential, [2., -.1]) == 10.
    assert calls == [(0., np.inf, {'limit': 100})]


def test_integration_warning_is_not_silenced_by_an_outer_ignore_filter(monkeypatch):
    def warned_quad(*args, **kwargs):
        warnings.warn('unreliable integral', IntegrationWarning)
        return 10., 1e-9
    monkeypatch.setattr(time_constants, 'quad', warned_quad)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tau, reason = time_constants.evaluate_tau(exponential, [1., -.1])
    assert np.isnan(tau)
    assert 'did not converge reliably' in reason


def test_divergent_positive_plateau_does_not_become_a_finite_tau():
    tau, reason = time_constants.evaluate_tau(double_exponential, [1., -.1, .2, 0.])
    assert np.isnan(tau)
    assert reason is not None
    assert np.isnan(_tau_from_parameter_vector(double_exponential, [1., -.1, .2, 0.]))


@pytest.mark.parametrize('area,error', [(np.nan, 0.), (np.inf, 1.), (1., np.inf),
                                       (1., np.nan), (1., -1.), (0., 0.), (-1., .01)])
def test_invalid_quadrature_results_are_unavailable(monkeypatch, area, error):
    monkeypatch.setattr(time_constants, 'quad', lambda *a, **kw: (area, error))
    tau, reason = time_constants.evaluate_tau(exponential, [1., -.1])
    assert np.isnan(tau)
    assert reason


@pytest.mark.parametrize('initial', [0., np.inf, np.nan])
def test_invalid_initial_response_does_not_reach_quad(monkeypatch, initial):
    def forbidden(*args, **kwargs):
        raise AssertionError('quad should not be called')
    monkeypatch.setattr(time_constants, 'quad', forbidden)
    tau, reason = time_constants.evaluate_tau(lambda time: initial, [])
    assert np.isnan(tau)
    assert 'initial response' in reason


def test_nonfinite_integrand_is_unavailable():
    tau, reason = time_constants.evaluate_tau(lambda time: 1. if time == 0. else np.inf, [])
    assert np.isnan(tau)
    assert 'nonfinite evaluation' in reason


def test_opted_out_model_is_not_evaluated():
    def forbidden(time):
        raise AssertionError('unsupported model should not be evaluated')
    forbidden.supports_tau = False
    tau, reason = time_constants.evaluate_tau(forbidden, [])
    assert np.isnan(tau)
    assert 'not supported' in reason


def test_central_tau_does_not_depend_on_draw_count_or_random_seed():
    centers = []
    for seed, count in [(1, 12), (2, 50), (3, 80)]:
        tau, sd = calculate_tau_with_uncertainty(
            exponential, [1., -.1], np.diag([1e-4, 1e-5]), count, seed,
        )
        centers.append(tau)
        assert sd > 0
    assert centers == [calculate_tau(exponential, [1., -.1])] * 3
    assert centers[0] == pytest.approx(10.)


def test_exact_parameters_give_zero_tau_uncertainty_and_legacy_pair_return():
    result = calculate_tau_with_uncertainty(exponential, [1., -.1], np.zeros((2, 2)), random_state=5)
    assert len(result) == 2
    assert result[0] == pytest.approx(10.)
    assert result[1] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('covariance', [np.full((2, 2), np.inf), np.eye(3), np.diag([1., -1.])])
def test_bad_covariance_retains_central_tau(covariance):
    (tau, sd), diagnostics = calculate_tau_with_uncertainty(
        exponential, [1., -.1], covariance, return_diagnostics=True,
    )
    assert tau == pytest.approx(10.)
    assert np.isnan(sd)
    assert diagnostics['reason'] is None
    assert 'Parameter sampling failed' in diagnostics['uncertainty_reason']


def test_invalid_sampled_tau_is_not_silently_dropped(monkeypatch):
    import ivivc_core.uncertainty as uncertainty
    samples = np.array([[1., -.1], [1., .1], [1., -.2]])
    monkeypatch.setattr(uncertainty, 'sample_parameter_sets', lambda *a, **kw: SimpleNamespace(values=samples))
    (tau, sd), diagnostics = calculate_tau_with_uncertainty(
        exponential, [1., -.1], np.eye(2), num_samples=3, return_diagnostics=True,
    )
    assert tau == pytest.approx(10.)
    assert np.isnan(sd)
    assert diagnostics['valid_samples'] == 2
    assert diagnostics['requested_samples'] == 3
    assert 'Only 2 of 3' in diagnostics['uncertainty_reason']


def test_unavailable_central_tau_is_not_replaced_by_a_placeholder():
    (tau, sd), diagnostics = calculate_tau_with_uncertainty(
        double_exponential, [1., -.1, .2, 0.], np.zeros((4, 4)), return_diagnostics=True,
    )
    assert np.isnan(tau) and np.isnan(sd)
    assert diagnostics['reason']
    assert diagnostics['valid_samples'] == 0


def test_single_draw_does_not_imply_zero_tau_uncertainty():
    (tau, sd), diagnostics = calculate_tau_with_uncertainty(
        exponential, [1., -.1], np.eye(2), num_samples=1, return_diagnostics=True,
    )
    assert tau == pytest.approx(10.)
    assert np.isnan(sd)
    assert 'At least two' in diagnostics['uncertainty_reason']


def test_optional_tau_uncertainty_failure_does_not_discard_profile_fit(monkeypatch, fast_analysis_config):
    import ivivc_core.fitting as fitting
    monkeypatch.setattr(fitting, 'auto_grid_search', lambda *a, **kw: np.array([1., -.1]))
    monkeypatch.setitem(models['exponential'], 'fit_model', lambda *a, **kw: {
        'params': {'a': 1., 'b': -.1}, 'pcov': np.full((2, 2), np.inf),
        'predict': lambda time: np.exp(-.1 * np.asarray(time)),
    })
    result = fit_profile([0., 1.], np.exp(-.1 * np.arange(2.)), model_name='exponential',
                         selected_metrics=['rmse'], analysis_config=fast_analysis_config,
                         approach_id='approach2', calculate_tau=True)
    assert result.fit_metrics['rmse'] == 0.
    assert result.tau_estimate == pytest.approx(10.)
    assert np.isnan(result.tau_sd)
    assert result.tau_diagnostics['reason'] is None
    assert result.tau_diagnostics['uncertainty_reason']


def test_fit_profile_tau_sampling_is_reproducible(monkeypatch, fast_analysis_config):
    import ivivc_core.fitting as fitting
    monkeypatch.setattr(fitting, 'auto_grid_search', lambda *a, **kw: np.array([1., -.1]))
    time = np.arange(8., dtype=float)
    response = np.exp(-.1 * time) + np.array([.005, -.006, .004, -.002, .002, -.003, .001, -.001])
    first = fit_profile(time, response, model_name='exponential', selected_metrics=['rmse'],
                        analysis_config=fast_analysis_config, calculate_tau=True)
    np.random.seed(761)
    second = fit_profile(time, response, model_name='exponential', selected_metrics=['rmse'],
                         analysis_config=fast_analysis_config, calculate_tau=True)
    assert first.tau_estimate == second.tau_estimate
    assert first.tau_sd == second.tau_sd
    assert first.tau_diagnostics['random_state'] == 12345
    assert first.tau_diagnostics['valid_samples'] == 50


def test_tau_ensemble_central_uses_the_same_checked_integral():
    prediction = TimeCourse(np.array([0., 2., 4.]), np.array([1., .8, .6]))
    result = sample_tau_ratio_trajectories(
        prediction, model_name='exponential',
        in_vitro_parameters={'a': 1., 'b': -.2}, in_vitro_covariance=np.zeros((2, 2)),
        in_vivo_parameters={'a': 1., 'b': -.1}, in_vivo_covariance=np.zeros((2, 2)),
        n_samples=4, random_state=123,
    )
    assert result.central.available
    np.testing.assert_allclose(result.central.in_vivo_time, 2. * prediction.time)
    assert result.valid_draws.all()


def test_tau_ensemble_rejects_divergent_nominal_and_sampled_integrals():
    prediction = TimeCourse(np.array([0., 1.]), np.array([1., .8]))
    # Use the registry's declared names to avoid imposing a model-specific API.
    from ivivc_core import get_model_spec
    names = get_model_spec('double_exponential').parameter_names
    parameters = dict(zip(names, [1., -.1, .2, 0.]))
    result = sample_tau_ratio_trajectories(
        prediction, model_name='double_exponential',
        in_vitro_parameters=parameters, in_vitro_covariance=np.zeros((4, 4)),
        in_vivo_parameters=parameters, in_vivo_covariance=np.zeros((4, 4)),
        n_samples=3, random_state=1,
    )
    assert not result.central.available
    assert not result.valid_draws.any()
    assert np.isnan(result.metadata['in_vitro_tau_draws']).all()


def test_tau_points_remain_available_without_uncertainty():
    prediction = TimeCourse(np.array([0., 1., 2.]), np.array([1., .8, .6]))
    taus = [SimpleNamespace(n=5., s=np.nan), SimpleNamespace(n=10., s=.1)]
    assert describe_tau_prediction_skip_reason(taus) is None
    result = predict_tau_ratio(prediction, model_name='exponential', tau_values=taus,
                               tau_diagnostics=({'uncertainty_reason': 'Sampling failed.'}, {}))
    assert result.available
    np.testing.assert_allclose(result.in_vivo_time, 2. * prediction.time)
    assert result.time_error_upper is None
    assert result.time_error_lower is None
    assert not result.auxiliary['interval_available']
    assert 'parameter vectors and covariance' in result.note


def test_failed_tau_prediction_explains_integration_reason():
    result = predict_tau_ratio(
        TimeCourse(np.array([1.]), np.array([.8])), model_name='exponential',
        tau_values=[SimpleNamespace(n=np.nan, s=np.nan), SimpleNamespace(n=10., s=.1)],
        tau_diagnostics=({'reason': 'Numerical integration did not converge reliably.'}, {}),
    )
    assert not result.available
    assert 'did not converge' in result.skip_reason


def test_ui_separates_tau_and_interval_availability():
    from engine import _tau_interval_row
    invalid = _tau_interval_row(SimpleNamespace(n=0., s=1.), 2.)
    assert invalid['estimate'] == 'N/A'
    assert invalid['half_width'] == 'N/A'
    missing_sd = _tau_interval_row(SimpleNamespace(n=5., s=np.nan), 2.,
                                   diagnostics={'uncertainty_reason': 'Covariance unavailable.'})
    assert missing_sd['is_available']
    assert not missing_sd['interval_available']
    assert float(missing_sd['estimate']) == 5.
    assert missing_sd['half_width'] == 'N/A'
    assert missing_sd['note'] == 'Covariance unavailable.'
    valid = _tau_interval_row(SimpleNamespace(n=5., s=.2), 2.)
    assert valid['interval_available']
    assert float(valid['half_width']) == pytest.approx(.4)


def test_report_hides_legacy_sentinel_and_retains_tau_with_missing_sd():
    from test_report_content import fixture, table
    from utilities.report_content import build_report_sections
    context, option = fixture(method='tau_ratio')
    result = context['results']['approach2:example']
    result['tau'] = [SimpleNamespace(n=0., s=1.), SimpleNamespace(n=5., s=np.nan)]
    result['tau_diagnostics'] = [{'reason': 'Integration unavailable.'},
                                 {'uncertainty_reason': 'Covariance unavailable.'}]
    rows = table(build_report_sections(context, option), 'Time constants used for time scaling')['rows']
    assert rows[1] == ['In vitro', 'N/A', 'N/A', 'Integration unavailable.']
    assert rows[2] == ['In vivo', '5', 'N/A', 'Covariance unavailable.']


def test_process_predictions_retains_tau_points_and_explanatory_note(monkeypatch):
    import engine
    time = np.array([0., 2., 4., 6.])
    data = engine.preprocess_data(time, np.exp(-.2 * time), time, np.exp(-.1 * time),
                                  selected_interpolation=['default'])
    plotted = {}
    def plot(*args, **kwargs):
        plotted[kwargs['prediction_mode']] = (args[3], args[8] if len(args) > 8 else None)
        return {'plot': '<plot>', 'table_html': '<table>'}
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', plot)
    fitted = {'params': [{'a': 1., 'b': -.2}, {'a': 1., 'b': -.1}],
              'pcov': [np.full((2, 2), np.nan), np.zeros((2, 2))],
              'residuals': [np.zeros(4), np.zeros(4)], 'residual_dof': [2, 2],
              'tau': [SimpleNamespace(n=5., s=np.nan), SimpleNamespace(n=10., s=.1)],
              'tau_diagnostics': [{'uncertainty_reason': 'Covariance unavailable.'}, {}]}
    output = engine.process_predictions(data, [time[:2], np.exp(-.2 * time[:2])],
                                         {'approach2:exponential': fitted}, ['approach2'])['approach2']['exponential']
    assert 'error' not in output
    np.testing.assert_allclose(plotted['tau_ratio'][0], 2. * time[:2])
    assert plotted['tau_ratio'][1] is None
    assert 'Parameter sampling failed' in output['plot_tau_note']
    for row in output['prediction_tables']['tau_ratio']:
        assert row['lower_95'] is None and row['upper_95'] is None
        assert 'unavailable' in row['interval_status']


def test_tau_api_adapter_with_real_uncertainties_preserves_missing_sd():
    from ivivc_core.api import _tau_uncertain_value
    from engine import _profile_fit_to_legacy_components
    from ivivc_core import ProfileFit
    fit = ProfileFit(
        model_name='exponential', time=np.array([0., 1., 2.]), response=np.exp(-.1 * np.arange(3.)),
        parameter_names=('a', 'b'), parameters={'a': 1., 'b': -.1}, covariance=np.eye(2),
        initial_parameters=np.array([1., -.1]), fitted_values=np.exp(-.1 * np.arange(3.)),
        residuals=np.zeros(3), residual_dof=1, fit_metrics={'rmse': 0.},
        cv_scores={'rmse': np.array([np.nan])}, cv_means={'rmse': np.nan},
        tau_estimate=10., tau_sd=np.nan, tau_diagnostics={'uncertainty_reason': 'Example failure.'},
    )
    tau = _tau_uncertain_value(fit)
    assert tau.n == 10. and np.isnan(tau.s)
    components = _profile_fit_to_legacy_components(fit, include_tau=True)
    assert components['tau'].n == 10.
    assert components['tau_interval']['is_available']
    assert not components['tau_interval']['interval_available']
    assert components['tau_diagnostics']['uncertainty_reason'] == 'Example failure.'
    fit.tau_estimate = np.nan
    components = _profile_fit_to_legacy_components(fit, include_tau=True)
    assert np.isnan(components['tau'].n)
    assert components['tau_interval']['estimate'] == 'N/A'


def test_unexpected_tau_prediction_failure_does_not_hide_other_predictions(monkeypatch):
    import engine
    time = np.array([0., 2., 4., 6.])
    data = engine.preprocess_data(time, np.exp(-.2 * time), time, np.exp(-.1 * time),
                                  selected_interpolation=['default'])
    def failed_tau(*args, **kwargs):
        raise ValueError('Example optional tau failure')
    monkeypatch.setattr(engine, 'predict_tau_ratio', failed_tau)
    monkeypatch.setattr(engine, 'create_prediction_plot_a2', lambda *a, **kw: {'plot': '<plot>', 'table_html': '<table>'})
    fitted = {'params': [{'a': 1., 'b': -.2}, {'a': 1., 'b': -.1}],
              'pcov': [np.zeros((2, 2)), np.zeros((2, 2))],
              'residuals': [np.zeros(4), np.zeros(4)], 'residual_dof': [2, 2]}
    output = engine.process_predictions(data, [time[:2], np.exp(-.2 * time[:2])],
                                         {'approach2:exponential': fitted}, ['approach2'])['approach2']['exponential']
    assert 'error' not in output
    assert output['plot_tau'] is None
    assert 'Example optional tau failure' in output['plot_tau_skip_reason']
    assert output['plot'] is not None
    assert output['plot_inverse_response'] is not None
    assert len(output['prediction_tables']['tau_ratio']) == 2


def test_tau_templates_handle_estimate_and_uncertainty_separately():
    from pathlib import Path
    from jinja2 import Environment
    from engine import _tau_interval_row
    source = (Path(__file__).resolve().parents[1] / 'templates/results.html').read_text()
    # Parse the full template, then render both actual tau display fragments.
    environment = Environment()
    environment.parse(source)
    for index in (0, 1):
        start = source.index('{% set tau_row = model_results.tau_intervals[' + str(index) + '] %}')
        end = source.index('<h4>Goodness of Fit</h4>', start)
        fragment = environment.from_string(source[start:end])
        row = _tau_interval_row(SimpleNamespace(n=5., s=np.nan), 2.)
        rendered = fragment.render(model_results={'tau_intervals': [row, row]})
        assert 'uncertainty unavailable' in rendered
        assert 'Tau: N/A' not in rendered
        row = _tau_interval_row(SimpleNamespace(n=np.nan, s=np.nan), 2.)
        rendered = fragment.render(model_results={'tau_intervals': [row, row]})
        assert 'Tau: N/A' in rendered


def test_tau_prediction_note_is_included_in_report():
    from test_report_content import fixture, text
    from utilities.report_content import build_report_sections
    context, option = fixture(method='tau_ratio')
    context['prediction_source'] = {'filename': 'prediction.csv'}
    context['prediction_results'] = {'approach2': {'example': {
        'plot_tau_note': 'Time scaling is shown without an interval: example covariance failure.',
    }}}
    assert 'example covariance failure' in text(build_report_sections(context, option))
