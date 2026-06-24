# Appendix: How to Interpret Model Performance

No single performance metric determines whether an IVIVC model is meaningful. Model selection should consider goodness of fit, prediction error, model complexity, cross-validation behavior, residual patterns, parameter uncertainty, and whether the fitted relationship is scientifically plausible for the polymer degradation mechanism and the intended context of use.

The app reports metrics and comparison tables to support user review. It does not select, recommend, validate, or establish regulatory acceptability for any model. Users should document the rationale for the final model or method selected for a specific analysis.

## 1. Confirm that models are comparable before interpreting metrics

Model-performance metrics are most useful when the compared models were evaluated under the same analysis conditions. In general, compare models only within the same uploaded dataset, preprocessing configuration, response scale, selected IVIVC approach, and candidate-model set.

Do not directly compare AIC, AICc, or BIC values across different datasets, transformed responses, preprocessing settings, or substantially different fitting contexts. These information criteria are relative comparison tools, not absolute quality scores. Approach 3 should also be interpreted differently from Approaches 1 and 2 because it uses direct interpolation-based mappings rather than fitting a parametric model.

## 2. Metric ranges, direction, and interpretation

| Metric | Range / direction | Interpretation guidance |
|---|---|---|
| R^2 | Best possible value is 1. Values near 0 indicate little improvement over predicting the mean response. R^2 can be negative when the model performs worse than the mean-response baseline. Higher is generally better. | R^2 summarizes the fraction of response variance explained by the fitted model. High R^2 does not prove that the model is mechanistic, causal, predictive outside the fitted data range, or appropriate for the intended use. |
| Adjusted R^2 | Higher is generally better. It can be lower than R^2 and may be negative. | Adjusted R^2 penalizes additional fitted parameters and may be more informative than R^2 when comparing models with different complexity. |
| MSE | Lower is generally better. Units are the squared response units. | MSE summarizes average squared error. It is useful computationally but can be less intuitive than RMSE because the units are squared. |
| RMSE | Lower is generally better. Units match the response units. | RMSE summarizes typical prediction error on the response scale. Interpret RMSE relative to assay variability, measurement uncertainty, clinically or biologically meaningful differences, and the intended use. |
| NRMSE | Lower is generally better. The value is normalized and is intended to be more comparable across response scales. | NRMSE expresses prediction error relative to a normalization quantity, such as the response mean or response standard deviation. Interpretation depends on the normalization method. |
| AIC | Lower is generally better, but only relative to other comparable models. AIC has no fixed lower or upper bound, and negative values are allowed. | AIC balances fitted error and model complexity. Absolute AIC values should not be interpreted as standalone quality grades. |
| AICc | Lower is generally better, but only relative to other comparable models. AICc is undefined when there are too few observations relative to the number of fitted parameters. | AICc is AIC with a small-sample correction. It is often more appropriate than AIC when the number of observations is limited relative to the number of parameters. |
| BIC | Lower is generally better, but only relative to other comparable models. BIC has no fixed lower or upper bound, and negative values are allowed. | BIC also balances fitted error and model complexity, but it usually penalizes additional parameters more strongly than AIC. |
| Cross-validation metrics | Direction follows the underlying metric: higher is generally better for R^2-type metrics, and lower is generally better for error metrics. | Cross-validation metrics estimate held-out performance under the selected validation scheme. They can help identify overfitting, instability, or sensitivity to individual observations or data splits. |
| Fit/CV ratio | Values near 1 indicate similar fitted-data and cross-validation performance. | For models with consistent fitted and cross-validation performance, the fit/CV ratio is expected to be near 1. Substantial deviations from 1 may indicate reduced generalizability, overfitting, model instability, or sensitivity to the validation procedure, although no universal acceptance threshold is established. |

## 3. Interpreting goodness of fit

Goodness-of-fit metrics describe how well the model fits the data used to estimate the model parameters. These metrics include R^2, adjusted R^2, MSE, RMSE, and NRMSE.

High fitted-data performance can be encouraging, but it can also be optimistic, especially for small datasets, flexible models, or models with many parameters. A model with strong fitted-data performance may still be unsuitable if it has unstable parameters, poor cross-validation performance, systematic residual patterns, or a fitted shape that is not scientifically plausible for the degradation process.

When interpreting fitted-data metrics, consider whether the uncertainty is small enough to be meaningful for the specific degradation endpoint and context of use. There is no universal performance cutoff for IVIVC adequacy.

## 4. Interpreting prediction error

Prediction-error metrics should be interpreted in practical context. RMSE is often easier to interpret than MSE because it is reported in the same units as the response. NRMSE can be useful for comparing error across response scales, but the denominator matters. For example, NRMSE normalized by the response mean and NRMSE normalized by the response standard deviation answer different questions.

A prediction error may be numerically small but practically important if the relevant assay, device attribute, or degradation endpoint has narrow acceptable variation. Conversely, a larger numerical error may be acceptable for exploratory model development if the intended use does not require high predictive precision.

## 5. Interpreting relative model evidence

Relative Model Evidence tables summarize information-criterion comparisons among candidate parametric models fitted under comparable conditions. These tables may include AIC, AICc, BIC, delta values, evidence weights, and evidence ratios.

| Evidence quantity | Interpretation |
|---|---|
| AIC, AICc, or BIC | Information criterion value. Lower values indicate stronger relative support within the displayed candidate set. The absolute value is not interpreted directly. |
| Delta AIC, Delta AICc, or Delta BIC | Difference between a model's criterion value and the lowest value in the displayed candidate set. Values near 0 indicate similar relative support. Larger values indicate lower relative support under that criterion. |
| Evidence weight | Normalized relative-support value across the displayed candidate models. Weights are comparative summaries, not absolute probabilities that a model is true. |
| Evidence ratio | Ratio of the highest evidence weight to the model's evidence weight. Larger values indicate greater separation from the highest-weighted model under that criterion. |

Common rules of thumb for AIC-family comparisons are: delta values from 0 to 2 suggest similar or substantial empirical support; values from 4 to 7 suggest considerably less support; and values greater than 10 suggest essentially no support relative to the leading candidate in that model set. Common rules of thumb for BIC-family comparisons are: delta values from 0 to 2 suggest weak differences; 2 to 6 suggest positive evidence against the higher-BIC model; 6 to 10 suggest strong evidence; and values greater than 10 suggest very strong evidence. These are interpretive conventions, not regulatory acceptance criteria.

Relative evidence does not prove that a model is correct, mechanistic, predictive outside the observed range, or suitable for a regulatory submission. If multiple models have similar relative support, users should also consider simplicity, parameter plausibility, residual behavior, cross-validation behavior, and scientific interpretability.

## 6. Interpreting cross-validation performance

Cross-validation estimates how a fitted model performs on held-out data. It is useful for assessing whether fitted-data performance appears to generalize beyond the observations used for parameter estimation.

Cross-validation results depend on the selected validation method, number of folds or splits, random seed, test-set size, sample size, and data distribution. For small IVIVC datasets, cross-validation should usually be treated as an exploratory diagnostic rather than a definitive validation result. Leave-one-out cross-validation can be useful when datasets are small, but it may be sensitive to individual observations. K-fold and shuffle-split methods can provide repeated or fold-based estimates, but they require enough observations to create meaningful training and holdout sets.

Fit/CV ratios summarize the relationship between fitted-data and held-out performance for the same metric. Ratios near 1 indicate that fitted and cross-validation scores are similar. Ratios far from 1 may suggest reduced generalizability, overfitting, model instability, or sensitivity to the validation scheme. The direction of concern depends on the metric and context, so the ratio should be interpreted together with the raw fit and CV values.

AIC, AICc, and BIC should not be interpreted as ordinary cross-validation scores because the final model and the cross-validation models are fitted to different subsets of the data and may involve different effective sample sizes. Use information criteria for relative evidence among comparable fitted models, and use cross-validation metrics to evaluate held-out predictive behavior.

<!--
## 7. Inspecting residuals and diagnostic plots

Summary metrics can hide important model behavior. Residual plots should be reviewed for systematic patterns, bias, curvature, clustering, changing variance, or influential observations.

| Residual pattern | Possible implication |
|---|---|
| Random scatter around zero | More consistent with an adequate model form, although not proof of validity. |
| Curvature or time-dependent trend | The model form may not capture the observed relationship. |
| Increasing or decreasing spread | Error variability may change across time, response, or degradation range. |
| Mostly positive or mostly negative residuals | Potential systematic bias. |
| One point dominates the apparent fit | The model may be sensitive to individual observations or limited data density. |

Outliers should be investigated using scientific and data-quality rationale. Removing observations solely to improve metrics can make the model-selection rationale difficult to justify.
--!>

## 7. Interpreting parameter uncertainty

Parameter estimates should be reviewed for scientific plausibility and numerical stability. Large standard errors, wide confidence intervals, failed covariance estimation, or highly correlated parameters may indicate that the data do not strongly identify the model parameters.

Parameter uncertainty is especially important when a fitted model will be interpreted mechanistically, used to support extrapolation, or used to compare degradation behavior across materials, processes, or test conditions. A model with strong fitted metrics but poorly estimated or implausible parameters may be difficult to justify.

## 8. Considering model complexity

More complex models can often fit observed data more closely, but they may overfit small datasets or produce unstable parameter estimates. Adjusted R^2, AIC, AICc, BIC, cross-validation metrics, and residual diagnostics can help evaluate whether additional complexity is justified.

When two models have similar fitted performance and similar relative evidence, the simpler and more interpretable model may be easier to justify, provided it is scientifically plausible and performs adequately for the intended use. Model complexity should be supported by data density, degradation-mechanism expectations, and the intended context of use.

## 9. Scientific plausibility and intended use

Statistical performance should be interpreted together with knowledge of the polymer, degradation mechanism, test conditions, and proposed use of the IVIVC. A model may be statistically competitive but scientifically difficult to justify if its fitted shape, parameter values, or extrapolated behavior conflict with known degradation behavior.

The intended use should determine the amount and type of supporting evidence needed. Exploratory model development may tolerate more uncertainty than a model intended to support a specific regulatory argument, design change, or prediction of in vivo performance.

## 10. Suggested model-selection workflow

1. Confirm that the dataset, preprocessing choices, and selected response variables are appropriate.
2. Review fitted curves and uncertainty bands for each candidate model.
3. Compare fitted-data metrics, such as R^2 and RMSE.
4. Review prediction-error metrics in the context of assay variability, endpoint importance, and intended use.
5. Review cross-validation performance and fit/CV ratios, where cross-validation is applicable.
6. Review Relative Model Evidence using AIC, AICc, and/or BIC, where parametric model comparisons are applicable.
7. Review fitted-parameter estimates, standard errors, and scientific plausibility.
8. Consider model complexity and whether additional parameters are justified by the data and degradation mechanism.
9. Select the model or method based on a documented balance of statistical performance, scientific plausibility, and intended use.

## 11. Example interpretation statements

The following examples illustrate wording that may be adapted for reports. Users should tailor the language to the specific analysis and supporting evidence.

- Model A had the lowest RMSE and highest adjusted R^2 among the evaluated models, indicating strong performance under the selected preprocessing conditions.
- Model B had similar AICc support to Model A, with Delta AICc less than 2, suggesting that the two models were not clearly distinguished by AICc within this candidate set.
- Although Model C had strong fitted-data performance, its cross-validation error was substantially higher than its fitted error, suggesting possible sensitivity to the validation procedure or reduced generalizability.
- The selected model was chosen based on a balance of fitted performance, cross-validation behavior, relative evidence, parameter plausibility, and scientific interpretability.

## 12. Limitations

Performance metrics are descriptive and comparative tools, not acceptance criteria. Small datasets can produce unstable estimates, and similar metric values do not necessarily indicate equivalent scientific validity. Users should document preprocessing assumptions, fitting settings, model selection rationale, limitations, and any planned use of the model outside the fitted data range.

## References and suggested reading

1. U.S. Food and Drug Administration. *Assessing the Credibility of Computational Modeling and Simulation in Medical Device Submissions: Guidance for Industry and Food and Drug Administration Staff*. November 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/assessing-credibility-computational-modeling-and-simulation-medical-device-submissions
2. U.S. Food and Drug Administration. *Reporting of Computational Modeling Studies in Medical Device Submissions: Guidance for Industry and Food and Drug Administration Staff*. September 2016. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/reporting-computational-modeling-studies-medical-device-submissions
3. Burnham, K. P., and Anderson, D. R. *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach*. 2nd ed. Springer, 2002.
4. Heinze, G., Wallisch, C., and Dunkler, D. Variable selection - A review and recommendations for the practicing statistician. *Biometrical Journal*. 2018;60(3):431-449. https://doi.org/10.1002/bimj.201700067
5. Kass, R. E., and Raftery, A. E. Bayes Factors. *Journal of the American Statistical Association*. 1995;90(430):773-795. https://doi.org/10.1080/01621459.1995.10476572
6. Hodson, T. O. Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not. *Geoscientific Model Development*. 2022;15:5481-5487. https://doi.org/10.5194/gmd-15-5481-2022
7. scikit-learn developers. `sklearn.metrics.r2_score` documentation. https://sklearn.org/stable/modules/generated/sklearn.metrics.r2_score.html
