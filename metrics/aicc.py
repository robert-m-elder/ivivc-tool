import numpy as np


display_name = "AICc"
sort_order = 6
better_direction = 'lower'


def metric_function(y_true, y_pred, n_params=None):
    """
    Corrected Akaike Information Criterion (AICc).

    AICc = AIC + (2k(k + 1)) / (n - k - 1)

    where n is sample size, RSS is residual sum of squares, and k is the
    number of fitted model parameters. AICc is undefined when n <= k + 1.
    """
    if n_params is None:
        n_params = 2  # Default assumption: slope + intercept

    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)

    if rss <= 0:
        rss = np.finfo(float).eps

    aic = n * np.log(rss / n) + 2 * n_params
    denominator = n - n_params - 1
    if denominator <= 0:
        return np.nan

    return aic + (2 * n_params * (n_params + 1)) / denominator
