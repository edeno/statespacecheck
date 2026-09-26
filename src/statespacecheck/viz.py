"""Visualization utilities for diagnostic plots.

This module plots time series of HPD overlap, KL divergence, and predictive
p-values, with optional shading of flagged periods. It is an extension beyond
the paper, whose figures plot one marker per event; matplotlib is imported
only when a plot is made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .periods import _as_flags, _contiguous_runs, _robust_zscore

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_diagnostics(
    time: ArrayLike,
    overlap: ArrayLike,
    kl: ArrayLike,
    pvals: ArrayLike,
    flags: ArrayLike | None = None,
    *,
    overlap_threshold: float = 0.4,
    kl_z_threshold: float = 3.0,
    pvalue_threshold: float = 0.05,
) -> Figure:
    """Plot HPD overlap, KL divergence (with its robust z-score), and p-values.

    Creates a three-panel figure:

    1. HPD overlap, with a line at ``overlap_threshold``;
    2. KL divergence, with its robust z-score on a secondary axis and a line
       at ``kl_z_threshold`` (the rule of
       :func:`~statespacecheck.periods.flag_extreme_kl`);
    3. predictive p-values, with a line at ``pvalue_threshold``. Small
       p-values indicate misfit; p-values near 1 indicate a typical
       observation.

    Optionally shades flagged periods across all panels.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for the x-axis.
    overlap : array_like, shape (n_time,)
        HPD overlap values.
    kl : array_like, shape (n_time,)
        KL divergence values.
    pvals : array_like, shape (n_time,)
        Predictive p-values.
    flags : array_like of bool, shape (n_time,), optional
        Time points to shade, for example from
        :func:`~statespacecheck.periods.combine_flags`. Default is None (no
        shading).
    overlap_threshold : float, optional
        Where to draw the HPD overlap threshold. Default is 0.4.
    kl_z_threshold : float, optional
        Where to draw the robust z-score threshold for KL divergence.
        Default is 3.0.
    pvalue_threshold : float, optional
        Where to draw the p-value cutoff. Default is 0.05.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the diagnostic plots.

    Raises
    ------
    ValueError
        If a metric or ``flags`` has a different length from ``time``, or if
        ``flags`` is not boolean.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statespacecheck.viz import plot_diagnostics
    >>> rng = np.random.default_rng(0)
    >>> time = np.arange(100)
    >>> overlap = rng.uniform(0.3, 0.9, 100)
    >>> kl = rng.uniform(0.1, 2.0, 100)
    >>> pvals = rng.uniform(0.1, 0.9, 100)
    >>> fig = plot_diagnostics(time, overlap, kl, pvals)
    >>> plt.close(fig)
    """
    import matplotlib.pyplot as plt

    time_arr = np.asarray(time)
    metrics = {
        "overlap": np.asarray(overlap, dtype=float),
        "kl": np.asarray(kl, dtype=float),
        "pvals": np.asarray(pvals, dtype=float),
    }
    for name, values in metrics.items():
        if values.shape != time_arr.shape:
            msg = (
                f"{name} must have the same length as time ({time_arr.shape}); "
                f"got shape {values.shape}"
            )
            raise ValueError(msg)
    overlap_arr, kl_arr, pvals_arr = metrics.values()

    flags_arr = None
    if flags is not None:
        flags_arr = _as_flags(flags, "flags")
        if flags_arr.shape != time_arr.shape:
            msg = (
                f"flags must have the same length as time ({time_arr.shape}); "
                f"got shape {flags_arr.shape}"
            )
            raise ValueError(msg)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax = axes[0]
    ax.plot(time_arr, overlap_arr, linewidth=1)
    ax.axhline(overlap_threshold, linestyle="--", linewidth=1)
    ax.set_ylabel("HPD overlap")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time_arr, kl_arr, linewidth=1)
    ax.set_ylabel("KL divergence")
    ax.grid(True, alpha=0.3)
    z_ax = ax.twinx()
    z_ax.plot(time_arr, _robust_zscore(kl_arr), linewidth=0.8, alpha=0.6)
    z_ax.axhline(kl_z_threshold, linestyle="--", linewidth=1)
    z_ax.set_ylabel("robust z(KL)")

    ax = axes[2]
    ax.plot(time_arr, pvals_arr, linewidth=1)
    ax.axhline(pvalue_threshold, linestyle="--", linewidth=1)
    ax.set_ylabel("Predictive p")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    if flags_arr is not None and flags_arr.any():
        # Shade each run over its samples' full width, half the median step either
        # side, so a run of one sample is visible too. Numbers are padded in their
        # own units (by 0.5 for a single sample), datetime64 in the axis's float
        # date numbers, because halving a timedelta64 truncates to its unit (half
        # of 1 s would be 0 s); a single datetime and other time types get no
        # padding.
        span_time = time_arr
        if np.issubdtype(time_arr.dtype, np.datetime64):
            span_time = np.asarray(axes[0].convert_xunits(time_arr), dtype=float)
        if time_arr.size > 1 and np.issubdtype(span_time.dtype, np.number):
            half_step = np.median(np.diff(span_time)) / 2
        elif np.issubdtype(time_arr.dtype, np.number):
            half_step = 0.5
        else:
            half_step = span_time[0] - span_time[0]
        for start, stop in _contiguous_runs(flags_arr):
            for axi in axes:
                axi.axvspan(
                    span_time[start] - half_step, span_time[stop - 1] + half_step, alpha=0.15
                )

    fig.tight_layout()
    return fig
