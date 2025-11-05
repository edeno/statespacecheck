"""Visualization utilities for diagnostic plots.

This module provides functions to create diagnostic plots for state space model
goodness-of-fit assessment, including HPD overlap, KL divergence, and predictive
p-values with optional flagged period highlighting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Import helper functions from periods module
from .periods import _contiguous_runs, _robust_zscore


def plot_diagnostics(
    time: NDArray[np.floating],
    overlap: NDArray[np.floating],
    kl: NDArray[np.floating],
    pvals: NDArray[np.floating],
    flags: NDArray[np.bool_] | None = None,
    tau: float = 0.4,
    z_thresh: float = 3.0,
    alpha: float = 0.05,
) -> Figure:
    """Plot HPD overlap, KL (with robust z-band), and predictive p-values.

    Creates a three-panel diagnostic plot showing:
    1. HPD overlap with threshold line
    2. KL divergence with robust z-score on secondary axis
    3. Predictive p-values with two-sided significance bounds

    Optionally shades flagged periods across all panels.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time values for x-axis.
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    kl : np.ndarray, shape (n_time,)
        KL divergence values.
    pvals : np.ndarray, shape (n_time,)
        Predictive p-values.
    flags : np.ndarray, shape (n_time,), optional
        Boolean array indicating problematic time points to highlight with shading.
        True values mark periods where model-data agreement is poor (e.g., from
        combine_flags(), flag_extreme_kl(), or flag_low_overlap()). These regions
        will be shaded across all diagnostic panels to facilitate visual identification
        of problematic periods. Default is None (no shading).
    tau : float, optional
        Threshold for HPD overlap visualization (dashed line). Default is 0.4.
    z_thresh : float, optional
        Z-score threshold for KL visualization (dashed line). Default is 3.0.
    alpha : float, optional
        Significance level for p-value bounds (dashed lines). Default is 0.05.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the diagnostic plots.

    Raises
    ------
    ValueError
        If flags array has different shape than metrics.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statespacecheck.viz import plot_diagnostics
    >>> time = np.arange(100)
    >>> overlap = np.random.uniform(0.3, 0.9, 100)
    >>> kl = np.random.uniform(0.1, 2.0, 100)
    >>> pvals = np.random.uniform(0.1, 0.9, 100)
    >>> fig = plot_diagnostics(time, overlap, kl, pvals)
    >>> plt.show()
    """
    time_arr = np.asarray(time)
    overlap_arr = np.asarray(overlap, dtype=float)
    kl_arr = np.asarray(kl, dtype=float)
    pvals_arr = np.asarray(pvals, dtype=float)

    if flags is not None:
        flags_arr = np.asarray(flags, dtype=bool)
        if flags_arr.shape != overlap_arr.shape:
            raise ValueError("flags must have same shape as metrics")
    else:
        flags_arr = None

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    # 1) Overlap
    ax = axes[0]
    ax.plot(time_arr, overlap_arr, linewidth=1)
    ax.axhline(tau, linestyle="--", linewidth=1)
    ax.set_ylabel("HPD overlap")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2) KL (show raw KL and robust z as twin y if desired)
    ax = axes[1]
    ax.plot(time_arr, kl_arr, linewidth=1)
    ax.set_ylabel("KL divergence")
    ax.grid(True, alpha=0.3)

    # Optionally draw a z-threshold band using right y-axis (robust z)
    z = _robust_zscore(kl_arr)
    ax2 = ax.twinx()
    ax2.plot(time_arr, z, linewidth=0.8, alpha=0.6)
    ax2.axhline(z_thresh, linestyle="--", linewidth=1)
    ax2.set_ylabel("robust z(KL)")

    # 3) Predictive p-values
    ax = axes[2]
    ax.plot(time_arr, pvals_arr, linewidth=1)
    ax.axhline(alpha / 2.0, linestyle="--", linewidth=1)
    ax.axhline(1.0 - alpha / 2.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Predictive p")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Shade flagged intervals across all panels
    if flags_arr is not None and np.any(flags_arr):
        for a, b in _contiguous_runs(flags_arr):
            # Handle edge case where b might be at the end
            time_end = time_arr[b - 1] if b - 1 < len(time_arr) else time_arr[-1]
            for axi in axes:
                axi.axvspan(time_arr[a], time_end, alpha=0.15)

    fig.tight_layout()
    return fig
