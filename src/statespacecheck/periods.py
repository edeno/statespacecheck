"""Period-level aggregation and detection utilities for time-series metrics.

This module provides functions to:
1. Aggregate time-series goodness-of-fit metrics over specified time periods
2. Detect problematic periods based on threshold exceedances
3. Combine multiple diagnostic methods via majority voting
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation


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
    metric_values : np.ndarray, shape (n_time,)
        Time-series metric array. Must be 1-dimensional.
    time_mask : np.ndarray, shape (n_time,)
        Boolean array indicating which time points to include.
        True values indicate time points to aggregate.
        Must have same length as metric_values.
    reduction : {'mean', 'sum'}, optional
        Aggregation method. Default is 'mean'.
        - 'mean': Compute mean over selected time points (optionally weighted)
        - 'sum': Compute sum over selected time points
    weights : np.ndarray, shape (n_time,), optional
        Optional weights for weighted mean (e.g., occupancy/time weighting).
        Must be non-negative and have same length as metric_values.
        Only used when reduction='mean'. Ignored for 'sum' with a warning.

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


# ---------- Helper functions for period detection ----------


def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs for True-runs in a 1D boolean mask.

    Parameters
    ----------
    mask : np.ndarray, shape (n_time,)
        Boolean mask array.

    Returns
    -------
    runs : list[tuple[int, int]]
        List of (start, stop) index pairs for contiguous True regions.

    Raises
    ------
    ValueError
        If mask is not 1-dimensional.
    """
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 1:
        raise ValueError("mask must be 1D")
    # Pad with False on both ends so diff catches edges
    padded = np.concatenate(([False], mask_arr, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    # Even indices are starts, odd are stops
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def _enforce_min_len(mask: NDArray[np.bool_], min_len: int) -> NDArray[np.bool_]:
    """Remove True-runs shorter than min_len.

    Parameters
    ----------
    mask : np.ndarray, shape (n_time,)
        Boolean mask array.
    min_len : int
        Minimum length for runs to be kept.

    Returns
    -------
    filtered_mask : np.ndarray, shape (n_time,)
        Boolean mask with short runs removed.
    """
    runs = _contiguous_runs(mask)
    if not runs:
        return np.zeros_like(mask, dtype=bool)

    # Vectorized filtering: convert to arrays for length comparison
    starts = np.array([start for start, _ in runs], dtype=int)
    stops = np.array([stop for _, stop in runs], dtype=int)
    lengths = stops - starts
    keep = lengths >= max(1, int(min_len))

    # Build output by setting kept runs to True
    out = np.zeros_like(mask, dtype=bool)
    for start, stop in zip(starts[keep], stops[keep], strict=True):
        out[start:stop] = True
    return out


def _robust_zscore(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Median/MAD-based z-score; returns NaN where values is NaN/Inf.

    Uses scipy's median_abs_deviation with Gaussian scaling factor.

    Parameters
    ----------
    values : np.ndarray, shape (n_time,)
        Input array.

    Returns
    -------
    zscores : np.ndarray, shape (n_time,)
        Robust z-scores.
    """
    values_arr = np.asarray(values, dtype=float)
    zscores = np.full_like(values_arr, np.nan)
    finite = np.isfinite(values_arr)
    if not np.any(finite):
        return zscores
    finite_vals = values_arr[finite]
    median = np.median(finite_vals)
    # Use scipy's median_abs_deviation with scale='normal' for 1.4826 factor
    mad = median_abs_deviation(finite_vals, scale="normal", nan_policy="propagate")
    if mad == 0.0:
        # Fall back to IQR-based scale if MAD is zero (all equal or extremely tied)
        q75, q25 = np.percentile(finite_vals, [75, 25])
        scale = (q75 - q25) / 1.349 if (q75 - q25) > 0 else 1.0
    else:
        scale = mad
    zscores[finite] = (finite_vals - median) / scale
    return zscores


# ---------- Public API for period detection ----------


def flag_low_overlap(
    overlap: NDArray[np.floating],
    threshold: float = 0.4,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Flag times where HPD overlap is below threshold.

    This is the boolean array version of find_low_overlap_intervals().
    Use this when combining multiple diagnostics with combine_flags().
    Use find_low_overlap_intervals() when you need interval boundaries.

    Parameters
    ----------
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    threshold : float, optional
        Threshold below which overlap is considered problematic. Default is 0.4.
        A value of 0.4 identifies periods where less than 40% of HPD regions
        overlap, indicating substantial spatial disagreement.
    min_len : int, optional
        Minimum length for flagged runs. Default is 5.
        Filters out transient single-timepoint artifacts. Adjust based on
        temporal resolution and expected duration of model failures.

    Returns
    -------
    flags : np.ndarray, shape (n_time,)
        Boolean array indicating flagged time points.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import flag_low_overlap, combine_flags
    >>> overlap = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8])
    >>> flags = flag_low_overlap(overlap, threshold=0.4, min_len=5)
    >>> flags
    array([False, False,  True,  True,  True,  True,  True, False])
    >>> # Combine with other diagnostics
    >>> kl_flags = flag_extreme_kl(kl_values, z_thresh=3.0, min_len=5)
    >>> combined = combine_flags(flags, kl_flags, min_votes=2, min_len=5)

    See Also
    --------
    find_low_overlap_intervals : Returns interval boundaries instead of boolean mask
    combine_flags : Combine multiple diagnostic flag arrays
    """
    overlap_arr = np.asarray(overlap, dtype=float)
    flags = (overlap_arr < threshold) & np.isfinite(overlap_arr)
    return _enforce_min_len(flags, min_len)


def find_low_overlap_intervals(
    overlap: NDArray[np.floating],
    threshold: float = 0.4,
    min_len: int = 5,
) -> list[tuple[int, int]]:
    """Identify contiguous intervals where HPD overlap < threshold and length >= min_len.

    Returns interval boundaries rather than boolean flags. Use flag_low_overlap()
    if you need a boolean array compatible with combine_flags().

    Parameters
    ----------
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    threshold : float, optional
        Threshold below which overlap is considered problematic. Default is 0.4.
    min_len : int, optional
        Minimum length for intervals to be reported. Default is 5.

    Returns
    -------
    intervals : list[tuple[int, int]]
        List of (start, stop) index pairs for problematic periods.
        Uses Python slice notation: interval includes start but excludes stop,
        so to extract values use array[start:stop] not array[start:stop+1].

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import find_low_overlap_intervals
    >>> overlap = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8])
    >>> intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=5)
    >>> intervals
    [(2, 7)]
    >>> # Extract first problematic interval
    >>> if intervals:
    >>>     start, stop = intervals[0]
    >>>     problem_overlap = overlap[start:stop]  # Correct: excludes stop
    >>>     print(f"Problem period: timepoints {start}-{stop-1}")

    See Also
    --------
    flag_low_overlap : Returns boolean mask instead of interval boundaries
    """
    overlap_arr = np.asarray(overlap, dtype=float)
    bad = (overlap_arr < threshold) & np.isfinite(overlap_arr)
    bad = _enforce_min_len(bad, min_len)
    return _contiguous_runs(bad)


def flag_extreme_kl(
    kl: NDArray[np.floating],
    z_thresh: float = 3.0,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Flag times where KL divergence is extreme via robust z-score.

    Parameters
    ----------
    kl : np.ndarray, shape (n_time,)
        KL divergence values.
    z_thresh : float, optional
        Z-score threshold above which values are flagged. Default is 3.0.
        A value of 3.0 corresponds to p < 0.003 for normal distributions,
        providing a conservative threshold to avoid false positives.
        Lower values (e.g., 2.0) are more sensitive but may flag more noise.
    min_len : int, optional
        Minimum length for flagged runs. Default is 5.
        Filters out transient single-timepoint artifacts. Adjust based on
        temporal resolution and expected duration of model failures.

    Returns
    -------
    flags : np.ndarray, shape (n_time,)
        Boolean array indicating flagged time points.

    Notes
    -----
    Inf/NaN KL values are ignored (not flagged) by default.

    The min_len parameter filters short runs to reduce false positives from
    single-timepoint artifacts or noise. This is a practical filter, not a
    statistical requirement. Appropriate values depend on:
    - Temporal resolution of your data (higher sampling → larger min_len)
    - Expected duration of real model failures (persistent vs transient)
    - Tolerance for false alarms (strict → larger min_len)

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import flag_extreme_kl
    >>> kl = np.ones(20)
    >>> kl[5:10] = 100.0  # Extreme spike
    >>> flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
    >>> np.sum(flags[5:10])
    5

    See Also
    --------
    flag_low_overlap : Flag periods with low HPD overlap
    flag_extreme_pvalues : Flag extreme predictive p-values
    combine_flags : Combine multiple diagnostic methods
    """
    kl_arr = np.asarray(kl, dtype=float)
    zscores = _robust_zscore(kl_arr)
    flags = np.isfinite(zscores) & (zscores > z_thresh)
    return _enforce_min_len(flags, min_len)


def flag_extreme_pvalues(
    pvalues: NDArray[np.floating],
    alpha: float = 0.05,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Two-sided extremeness test for predictive p-values.

    Flags when pvalues < alpha/2 or pvalues > 1 - alpha/2.

    Parameters
    ----------
    pvalues : np.ndarray, shape (n_time,)
        Predictive p-values.
    alpha : float, optional
        Significance level for two-sided test. Default is 0.05.
        This flags the extreme 5% of p-values (2.5% in each tail),
        identifying observations that are unusually extreme under the model.
    min_len : int, optional
        Minimum length for flagged runs. Default is 5.

    Returns
    -------
    flags : np.ndarray, shape (n_time,)
        Boolean array indicating flagged time points.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import flag_extreme_pvalues
    >>> pvalues = np.ones(20) * 0.5
    >>> pvalues[5:10] = 0.01  # Very low p-values
    >>> flags = flag_extreme_pvalues(pvalues, alpha=0.05, min_len=5)
    >>> np.sum(flags[5:10])
    5
    >>> # Combine with KL-based flags
    >>> kl_flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
    >>> combined = combine_flags(flags, kl_flags, min_votes=2, min_len=5)

    See Also
    --------
    flag_extreme_kl : Flag extreme KL divergence times
    flag_low_overlap : Flag low HPD overlap periods
    combine_flags : Combine multiple diagnostic methods
    """
    pvalues_arr = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(pvalues_arr)
    too_low = pvalues_arr < (alpha / 2.0)
    too_high = pvalues_arr > (1.0 - alpha / 2.0)
    flags = finite & (too_low | too_high)
    return _enforce_min_len(flags, min_len)


def combine_flags(
    *flags: NDArray[np.bool_],
    min_votes: int = 2,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Majority-vote combination of multiple boolean flag arrays.

    Parameters
    ----------
    *flags : bool arrays, each shape (n_time,)
        Variable number of boolean flag arrays to combine.
        Each should be 1D; all must have equal length.
    min_votes : int, optional
        Number of agreeing methods required to flag a time point. Default is 2.
        For example, with 3 input flags and min_votes=2, a time point is
        flagged only if at least 2 of the 3 methods flag it.
    min_len : int, optional
        Minimum length for flagged runs in final output. Default is 5.

    Returns
    -------
    combined : np.ndarray, shape (n_time,)
        Final boolean mask with short runs removed.

    Raises
    ------
    ValueError
        If no flag arrays provided or if arrays have mismatched lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import (
    >>>     flag_extreme_kl, flag_extreme_pvalues, flag_low_overlap, combine_flags
    >>> )
    >>> # Combine two diagnostic methods (require both to agree)
    >>> kl_flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
    >>> overlap_flags = flag_low_overlap(overlap, tau=0.4, min_len=5)
    >>> strict = combine_flags(kl_flags, overlap_flags, min_votes=2, min_len=5)
    >>>
    >>> # Combine three methods (require any 2 to agree)
    >>> pval_flags = flag_extreme_pvalues(pvals, alpha=0.05, min_len=5)
    >>> moderate = combine_flags(
    >>>     kl_flags, overlap_flags, pval_flags, min_votes=2, min_len=5
    >>> )
    >>>
    >>> # Require all three methods to agree (strict consensus)
    >>> consensus = combine_flags(
    >>>     kl_flags, overlap_flags, pval_flags, min_votes=3, min_len=5
    >>> )

    See Also
    --------
    flag_extreme_kl : Flag extreme KL divergence times
    flag_extreme_pvalues : Flag extreme p-values
    flag_low_overlap : Flag low HPD overlap periods
    """
    if len(flags) == 0:
        raise ValueError(
            "Error: No flag arrays provided.\n\n"
            "What went wrong: combine_flags() requires at least one flag array.\n"
            "How to fix: Pass one or more boolean flag arrays as arguments, e.g.:\n"
            "    combined = combine_flags(kl_flags, overlap_flags, min_votes=2)"
        )
    flag_arrays = [np.asarray(flag_arr, dtype=bool) for flag_arr in flags]
    n_time = flag_arrays[0].shape[0]
    if any(flag_arr.shape != (n_time,) for flag_arr in flag_arrays):
        shapes = [flag_arr.shape for flag_arr in flag_arrays]
        raise ValueError(
            f"Error: All flag arrays must be 1D with matching length.\n\n"
            f"What went wrong: Flag arrays have mismatched shapes: {shapes}\n"
            f"Why: combine_flags() performs element-wise majority voting across time,\n"
            f"     requiring all flags to represent the same time points.\n\n"
            f"How to fix:\n"
            f"  1. Check that all input arrays have shape (n_time,)\n"
            f"  2. Verify arrays come from same dataset with same time axis\n"
            f"  3. Ensure no accidental transposition or subsetting"
        )
    votes = np.sum(np.stack(flag_arrays, axis=0), axis=0)
    combined = votes >= int(min_votes)
    return _enforce_min_len(combined, min_len)
