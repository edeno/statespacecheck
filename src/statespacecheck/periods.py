"""Period-level aggregation utilities for time-series metrics.

This module provides functions to aggregate time-series goodness-of-fit metrics
(e.g., KL divergence, HPD overlap, predictive checks) over specified time periods
using indicator functions.
"""

import warnings

import numpy as np
from numpy.typing import NDArray


def aggregate_over_period(
    metric_values: NDArray[np.floating],
    time_mask: NDArray[np.bool_],
    *,
    reduction: str = "mean",
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Aggregate metric values over specified time period.

    Aggregates time-series metrics (e.g., KL divergence, HPD overlap, or
    predictive checks) over specified time periods using an indicator
    function approach from the paper.

    Parameters
    ----------
    metric_values : np.ndarray
        Time-series metric array. Must be 1-dimensional.
        Shape (n_time,).
    time_mask : np.ndarray
        Boolean array indicating which time points to include.
        True values indicate time points to aggregate.
        Must have same length as metric_values.
        Shape (n_time,).
    reduction : {'mean', 'sum'}, optional
        Aggregation method. Default is 'mean'.
        - 'mean': Compute mean over selected time points (optionally weighted)
        - 'sum': Compute sum over selected time points
    weights : np.ndarray, optional
        Optional weights for weighted mean (e.g., occupancy/time weighting).
        Must be non-negative and have same length as metric_values.
        Only used when reduction='mean'. Ignored for 'sum' with a warning.
        Shape (n_time,).

    Returns
    -------
    aggregated_value : float
        Aggregated metric value (scalar float).
        Returns NaN if no time points are selected (all-false mask).

    Raises
    ------
    ValueError
        If metric_values is not 1-dimensional, if shapes don't match,
        if reduction is invalid, or if weights are negative.

    Warns
    -----
    UserWarning
        If weights are provided when reduction='sum' (weights are ignored).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import aggregate_over_period
    >>> # Aggregate KL divergence over non-local events
    >>> kl_values = np.array([0.5, 1.0, 0.3, 0.8, 0.6])
    >>> is_non_local = np.array([True, False, True, True, False])
    >>> result = aggregate_over_period(kl_values, is_non_local, reduction="mean")
    >>> result  # Mean of [0.5, 0.3, 0.8]
    0.5333333333333333

    >>> # Aggregate log-likelihoods using sum
    >>> log_likes = np.array([-1.0, -2.0, -1.5, -3.0])
    >>> period_mask = np.array([True, True, True, True])
    >>> total = aggregate_over_period(log_likes, period_mask, reduction="sum")
    >>> total  # Sum of all values
    -7.5

    >>> # Weighted mean with occupancy weights
    >>> metrics = np.array([1.0, 2.0, 3.0])
    >>> mask = np.array([True, True, True])
    >>> occupancy = np.array([10.0, 5.0, 10.0])  # Time spent in each state
    >>> weighted = aggregate_over_period(metrics, mask, weights=occupancy)
    >>> weighted  # (1*10 + 2*5 + 3*10) / (10 + 5 + 10)
    2.0

    See Also
    --------
    kl_divergence : Compute KL divergence between distributions
    hpd_overlap : Compute spatial overlap between HPD regions
    predictive_density : Compute predictive density
    log_predictive_density : Compute log predictive density

    Notes
    -----
    This function implements the period-level aggregation approach from the paper,
    using indicator functions (time_mask) to select time points for aggregation.

    Use cases:
    - Period-level KL divergence: weighted mean over non-local events
    - Period-level log-likelihood: sum for predictive checks
    - Consistent with paper's weighted average equations

    When no time points are selected (all-false mask), returns NaN to indicate
    an undefined aggregation.
    """
    # Validate metric_values is 1D
    metric_arr = np.asarray(metric_values, dtype=float)
    if metric_arr.ndim != 1:
        raise ValueError(
            f"metric_values must be 1-dimensional, "
            f"got {metric_arr.ndim}D array with shape {metric_arr.shape}"
        )

    # Validate time_mask
    mask_arr = np.asarray(time_mask, dtype=bool)
    if mask_arr.shape != metric_arr.shape:
        raise ValueError(
            f"time_mask must have same length as metric_values, "
            f"got {mask_arr.shape} vs {metric_arr.shape}"
        )

    # Validate reduction parameter
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

    # Validate weights if provided
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != metric_arr.shape:
            raise ValueError(
                f"weights must have same length as metric_values, "
                f"got {weights_arr.shape} vs {metric_arr.shape}"
            )
        if not np.isfinite(weights_arr).all():
            raise ValueError("weights must be finite (no NaN or inf values)")
        if np.any(weights_arr < 0):
            raise ValueError("weights must be non-negative")

        # Warn if weights provided with sum reduction
        if reduction == "sum":
            warnings.warn(
                "weights are ignored when reduction='sum'",
                UserWarning,
                stacklevel=2,
            )

    # Select values based on time_mask
    selected_values = metric_arr[mask_arr]

    # Handle empty period (no time points selected)
    if len(selected_values) == 0:
        return np.nan

    # Perform aggregation
    if reduction == "sum":
        return float(np.sum(selected_values))
    else:  # reduction == "mean"
        if weights is None:
            return float(np.mean(selected_values))
        else:
            # Weighted mean
            selected_weights = weights_arr[mask_arr]
            weight_sum = np.sum(selected_weights)
            if weight_sum == 0:
                # All weights are zero -> return NaN
                return np.nan
            return float(np.sum(selected_values * selected_weights) / weight_sum)
