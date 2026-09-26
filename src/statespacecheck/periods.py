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


def aggregate_over_period(
    metric_values: NDArray[np.floating],
    time_mask: NDArray[np.bool_],
    *,
    reduction: str = "mean",
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Aggregate metric values over specified time period.

    Aggregates time-series metrics (e.g., KL divergence, HPD overlap, or
    predictive checks) over the time points selected by an indicator
    (boolean mask).

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
        Returns NaN if no time points are selected (all-false mask), or if
        reduction='mean' and every selected weight is zero.

    Raises
    ------
    ValueError
        If metric_values is not 1-dimensional, time_mask is not boolean, the
        shapes don't match, reduction is invalid, or weights are negative.

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
    The indicator ``time_mask`` selects the time points to aggregate, so one
    time series can be summarized over several periods of interest.

    Use cases:
    - Period-level KL divergence: weighted mean over non-local events
    - Period-level log-likelihood: sum for predictive checks

    When no time points are selected (all-false mask), returns NaN to indicate
    an undefined aggregation.
    """
    # Validate metric_values is 1D
    metric_arr = np.asarray(metric_values, dtype=float)
    if metric_arr.ndim != 1:
        msg = (
            f"metric_values must be 1-dimensional, "
            f"got {metric_arr.ndim}D array with shape {metric_arr.shape}"
        )
        raise ValueError(msg)

    # Validate time_mask: casting an index array to bool would select everything
    mask_arr = np.asarray(time_mask)
    if mask_arr.dtype != np.bool_:
        msg = (
            "time_mask must be a boolean array (True where a time point is included); "
            f"got dtype {mask_arr.dtype}. To select time points by index, build a "
            "mask: mask = np.zeros(n_time, dtype=bool); mask[indices] = True"
        )
        raise ValueError(msg)
    if mask_arr.shape != metric_arr.shape:
        msg = (
            f"time_mask must have same length as metric_values, "
            f"got {mask_arr.shape} vs {metric_arr.shape}"
        )
        raise ValueError(msg)

    # Validate reduction parameter
    if reduction not in ("mean", "sum"):
        msg = f"reduction must be 'mean' or 'sum', got '{reduction}'"
        raise ValueError(msg)

    # Validate weights if provided
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != metric_arr.shape:
            msg = (
                f"weights must have same length as metric_values, "
                f"got {weights_arr.shape} vs {metric_arr.shape}"
            )
            raise ValueError(msg)
        if not np.isfinite(weights_arr).all():
            msg = "weights must be finite (no NaN or inf values)"
            raise ValueError(msg)
        if np.any(weights_arr < 0):
            msg = "weights must be non-negative"
            raise ValueError(msg)

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
    # reduction == "mean"
    if weights is None:
        return float(np.mean(selected_values))
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
        msg = "mask must be 1D"
        raise ValueError(msg)
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


# scipy.special.ndtri(0.75): the MAD of a standard normal distribution
_NORMAL_MAD_SCALE = 0.6744897501960817


def _robust_zscore(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Median/MAD-based z-score; returns NaN where values is NaN/Inf.

    The MAD is scaled by 1 / Phi^-1(3/4), so it estimates the standard
    deviation of normally distributed values.

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
    # Same as scipy.stats.median_abs_deviation(finite_vals, scale="normal")
    mad = np.median(np.abs(finite_vals - median)) / _NORMAL_MAD_SCALE
    if mad == 0.0:
        # Fall back to IQR-based scale if MAD is zero (all equal or extremely tied)
        q75, q25 = np.percentile(finite_vals, [75, 25])
        scale = (q75 - q25) / 1.349 if (q75 - q25) > 0 else 1.0
    else:
        scale = mad
    zscores[finite] = (finite_vals - median) / scale
    return zscores


# ---------- Public API for period detection ----------


def _as_series(values: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    """Return ``values`` as a 1-D float array, or raise naming the argument."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        msg = f"{name} must be 1-D, shape (n_time,); got shape {series.shape}"
        raise ValueError(msg)
    return series


def flag_low_overlap(
    overlap: NDArray[np.floating],
    *,
    threshold: float = 0.4,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Flag times where HPD overlap is at or below a threshold.

    This is the boolean array version of find_low_overlap_intervals().
    Use this when combining multiple diagnostics with combine_flags().
    Use find_low_overlap_intervals() when you need interval boundaries.

    Parameters
    ----------
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    threshold : float, optional
        Overlap at or below this value is flagged. Default is 0.4, a
        convenience value with no statistical basis; the paper sets the
        threshold from a baseline period with
        :func:`~statespacecheck.baseline_threshold` (its 1st percentile).
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
    >>> from statespacecheck.periods import flag_low_overlap
    >>> overlap = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8])
    >>> flags = flag_low_overlap(overlap, threshold=0.4, min_len=5)
    >>> flags
    array([False, False,  True,  True,  True,  True,  True, False])

    See Also
    --------
    find_low_overlap_intervals : Returns interval boundaries instead of boolean mask
    combine_flags : Combine multiple diagnostic flag arrays
    """
    overlap_arr = _as_series(overlap, "overlap")
    flags = (overlap_arr <= threshold) & np.isfinite(overlap_arr)
    return _enforce_min_len(flags, min_len)


def find_low_overlap_intervals(
    overlap: NDArray[np.floating],
    *,
    threshold: float = 0.4,
    min_len: int = 5,
) -> list[tuple[int, int]]:
    """Find runs of at least ``min_len`` time points with HPD overlap at or below a threshold.

    Returns interval boundaries rather than boolean flags. Use flag_low_overlap()
    if you need a boolean array compatible with combine_flags().

    Parameters
    ----------
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    threshold : float, optional
        Overlap at or below this value is flagged. Default is 0.4 (see
        :func:`flag_low_overlap`).
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
    >>> # Extract the first problematic interval; stop is exclusive
    >>> start, stop = intervals[0]
    >>> print(f"Problem period: timepoints {start}-{stop - 1}")
    Problem period: timepoints 2-6

    See Also
    --------
    flag_low_overlap : Returns boolean mask instead of interval boundaries
    """
    overlap_arr = _as_series(overlap, "overlap")
    bad = (overlap_arr <= threshold) & np.isfinite(overlap_arr)
    bad = _enforce_min_len(bad, min_len)
    return _contiguous_runs(bad)


def flag_extreme_kl(
    kl: NDArray[np.floating],
    *,
    z_thresh: float = 3.0,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Flag times where KL divergence is extreme relative to the rest of the recording.

    A time point is flagged when its robust z-score (median and MAD of the
    finite values) exceeds ``z_thresh``, or when its KL divergence is
    infinite, which happens when the two distributions have disjoint support.

    The z-score is computed from the same values it tests, so the rule finds
    time points that stand out from the recording; a model that fits equally
    badly everywhere is not flagged. The paper instead flags values at or
    above a threshold set on a baseline period
    (:func:`~statespacecheck.baseline_threshold`, its 99th percentile) and
    uses KL divergence only as a reference, because it also flags consistent
    observations when the prediction is broad; see
    :func:`~statespacecheck.flag_events`.

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
    NaN values are never flagged; ``+inf`` values are always flagged.

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
    >>> np.flatnonzero(flags).tolist()
    [5, 6, 7, 8, 9]

    See Also
    --------
    flag_events : The paper's per-event flagging rule
    flag_low_overlap : Flag periods with low HPD overlap
    flag_extreme_pvalues : Flag extreme predictive p-values
    combine_flags : Combine multiple diagnostic methods
    """
    kl_arr = _as_series(kl, "kl")
    zscores = _robust_zscore(kl_arr)
    flags = (np.isfinite(zscores) & (zscores > z_thresh)) | np.isposinf(kl_arr)
    return _enforce_min_len(flags, min_len)


def flag_extreme_pvalues(
    pvalues: NDArray[np.floating],
    *,
    alpha: float = 0.05,
    min_len: int = 5,
) -> NDArray[np.bool_]:
    """Flag time points whose predictive p-value is at or below a cutoff.

    A small predictive p-value means the observation was unexpected under the
    model's prediction; a p-value near 1 means it was typical, which is good
    fit. The test is therefore one-sided: a time point is flagged when
    ``p <= alpha``.

    Parameters
    ----------
    pvalues : np.ndarray, shape (n_time,)
        Predictive p-values.
    alpha : float, optional
        Cutoff: p-values at or below it are flagged. Default is 0.05.
    min_len : int, optional
        Minimum length of a run of consecutive flagged time points; shorter
        runs are dropped. Default is 5. Use 1 to flag individual values (for
        example per-event p-values, which are not a time series).

    Returns
    -------
    flags : np.ndarray, shape (n_time,)
        Boolean array indicating flagged time points. NaN p-values are never
        flagged.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck.periods import flag_extreme_pvalues
    >>> pvalues = np.ones(20) * 0.5
    >>> pvalues[5:10] = 0.01  # Very low p-values
    >>> flags = flag_extreme_pvalues(pvalues, alpha=0.05, min_len=5)
    >>> np.flatnonzero(flags).tolist()
    [5, 6, 7, 8, 9]

    See Also
    --------
    flag_events : The paper's per-event flagging rule
    flag_extreme_kl : Flag extreme KL divergence times
    flag_low_overlap : Flag low HPD overlap periods
    combine_flags : Combine multiple diagnostic methods
    """
    pvalues_arr = _as_series(pvalues, "pvalues")
    flags = np.isfinite(pvalues_arr) & (pvalues_arr <= alpha)
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
    >>> from statespacecheck.periods import combine_flags
    >>> kl_flags = np.array([False, True, True, True, True, True, False, False])
    >>> overlap_flags = np.array([False, False, True, True, True, True, True, False])
    >>> pval_flags = np.array([False, False, False, True, True, True, True, True])
    >>> # Require both of two methods to agree
    >>> combine_flags(kl_flags, overlap_flags, min_votes=2, min_len=3)
    array([False, False,  True,  True,  True,  True, False, False])
    >>> # Require any two of three methods to agree
    >>> combine_flags(kl_flags, overlap_flags, pval_flags, min_votes=2, min_len=3)
    array([False, False,  True,  True,  True,  True,  True, False])
    >>> # Require all three methods to agree (strict consensus)
    >>> combine_flags(kl_flags, overlap_flags, pval_flags, min_votes=3, min_len=3)
    array([False, False, False,  True,  True,  True, False, False])

    See Also
    --------
    flag_extreme_kl : Flag extreme KL divergence times
    flag_extreme_pvalues : Flag extreme p-values
    flag_low_overlap : Flag low HPD overlap periods
    """
    if len(flags) == 0:
        msg = (
            "Error: No flag arrays provided.\n\n"
            "What went wrong: combine_flags() requires at least one flag array.\n"
            "How to fix: Pass one or more boolean flag arrays as arguments, e.g.:\n"
            "    combined = combine_flags(kl_flags, overlap_flags, min_votes=2)"
        )
        raise ValueError(msg)
    flag_arrays = [np.asarray(flag_arr, dtype=bool) for flag_arr in flags]
    n_time = flag_arrays[0].shape[0]
    if any(flag_arr.shape != (n_time,) for flag_arr in flag_arrays):
        shapes = [flag_arr.shape for flag_arr in flag_arrays]
        msg = (
            f"Error: All flag arrays must be 1D with matching length.\n\n"
            f"What went wrong: Flag arrays have mismatched shapes: {shapes}\n"
            f"Why: combine_flags() performs element-wise majority voting across time,\n"
            f"     requiring all flags to represent the same time points.\n\n"
            f"How to fix:\n"
            f"  1. Check that all input arrays have shape (n_time,)\n"
            f"  2. Verify arrays come from same dataset with same time axis\n"
            f"  3. Ensure no accidental transposition or subsetting"
        )
        raise ValueError(msg)
    votes = np.sum(np.stack(flag_arrays, axis=0), axis=0)
    combined = votes >= int(min_votes)
    return _enforce_min_len(combined, min_len)
