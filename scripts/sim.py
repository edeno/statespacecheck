"""Demonstration simulation for state space model diagnostics.

This script simulates a Bayesian decoder with periods of good and poor model fit,
then computes diagnostic metrics using the statespacecheck package.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, poisson

import statespacecheck as ssc

# -----------------------------
# Utilities (DRY helpers)
# -----------------------------


def normalize(
    p: NDArray[np.floating], axis: int | None = None, eps: float = 1e-12
) -> NDArray[np.floating]:
    """Return p / sum(p) with numerical safety."""
    s = np.sum(p, axis=axis, keepdims=True)
    s = np.maximum(s, eps)
    return p / s


def reflect_into_interval(x: NDArray[np.floating], lo: float, hi: float) -> NDArray[np.floating]:
    """Reflect a walk into [lo, hi] using the 'triangle wave' trick like the MATLAB triple-abs."""
    length = hi - lo
    y = np.mod(x - lo, 2 * length)
    y = np.where(y <= length, y, 2 * length - y)
    return y + lo


def gaussian_transition_matrix(xs: NDArray[np.floating], sig: float) -> NDArray[np.floating]:
    """OSM[i, j] = p(x_t = xs[i] | x_{t-1} = xs[j]) from Gaussian RW with std sig."""
    diff = xs[:, None] - xs[None, :]
    matrix = norm.pdf(diff, loc=0.0, scale=sig)
    return normalize(matrix, axis=0)  # columns sum to 1


def safe_log(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    """Return log(x) with numerical safety to avoid log(0)."""
    return np.log(np.maximum(x, eps))


def placefield_rates(
    xs: NDArray[np.floating], centers: NDArray[np.floating], width: float, scale: float
) -> NDArray[np.floating]:
    """lambda_mat[bin, cell] = scaled Gaussian place field evaluated on xs for each center."""
    return norm.pdf(xs[:, None], loc=centers[None, :], scale=width) * scale


def spike_prob_rank(
    prior: NDArray[np.floating],
    lambda_ratio: NDArray[np.floating],
    normalize_rank: bool = False,
) -> NDArray[np.floating]:
    """Compute rank of expected per-cell contribution under state uncertainty.

    Matches MATLAB: sum(lambda_expect(lambda_expect <= lambda_expect(j)))

    prior: (n_bins,)
    lambda_ratio: (n_bins, n_cells), rows sum to 1.
    returns: (n_cells,), each in [0,1] if normalize_rank=True, else raw count in [1..n_cells].
    """
    contrib = prior @ lambda_ratio  # (n_cells,)
    # rank by value; handle ties by "less or equal" like MATLAB
    ranks = (contrib[:, None] >= contrib[None, :]).sum(axis=1)
    return ranks / len(contrib) if normalize_rank else ranks


# -----------------------------
# Data containers
# -----------------------------


@dataclass
class DecodeParams:
    """Parameters for decoding simulation."""

    # Timeline with recovery periods between misfits:
    # 0-6k: Clean baseline
    # 6k-10k: Remapping misfit (4k)
    # 10k-14k: Clean recovery (4k)
    # 14k-16k: Flat firing misfit (2k)
    # 16k-20k: Clean recovery (4k)
    # 20k-24k: Fast movement misfit (4k)
    T_remap_start: int = 6_000
    T_remap_end: int = 10_000
    T_recovery1_end: int = 14_000
    T_flat_end: int = 16_000
    T_recovery2_end: int = 20_000
    T_fast_end: int = 24_000
    sigx_pred: float = 0.5  # decoder's dynamics std (kept fixed, even in part 3)
    sigx_true_fast: float = (
        10.0  # true dynamics std in part 3 (20x faster!) - optimized for strong misfit
    )
    xs_min: int = 0
    xs_max: int = 100
    xs_step: int = 1
    pf_width: float = 10.0
    pf_centers: NDArray[np.floating] | None = None  # set in __post_init__
    rate_scale: float = 0.10  # High spike rate for clear tracking visualization
    base_seed: int = 1
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (1, 6),  # Position 10 → 60
        (2, 7),  # Position 20 → 70
        (3, 8),  # Position 30 → 80
        (4, 9),  # Position 40 → 90
        (5, 10),  # Position 50 → 100
        (6, 0),  # Position 60 → 0
        (7, 1),  # Position 70 → 10
    )  # Remap 7 out of 11 cells with large spatial shifts for strong global mismatch

    @property
    def remap_window(self) -> tuple[int, int]:
        """Remapping window for backward compatibility."""
        return (self.T_remap_start, self.T_remap_end)

    def __post_init__(self) -> None:
        """Initialize pf_centers if not provided."""
        if self.pf_centers is None:
            self.pf_centers = np.arange(self.xs_min, self.xs_max + 1, 10, dtype=float)


# -----------------------------
# Simulation
# -----------------------------


def simulate_walk(
    n_time: int,
    sig: float,
    x0: float,
    xs_min: float,
    xs_max: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Simulate random walk with reflecting boundary conditions."""
    steps = rng.normal(loc=0.0, scale=sig, size=n_time)
    x = x0 + np.cumsum(steps)
    return reflect_into_interval(x, xs_min, xs_max)


def simulate_spikes_position_tuned(
    x: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Poisson spikes for each time and cell: spikes[t, j]."""
    lam = norm.pdf(x[:, None], loc=pf_centers[None, :], scale=pf_width) * rate_scale
    return rng.poisson(lam)


def simulate_spikes_flat_rate(
    n_time: int, n_cells: int, rate: float, rng: np.random.Generator
) -> NDArray[np.int_]:
    """Simulate spikes with flat (non-position-tuned) firing rate."""
    return rng.poisson(rate, size=(n_time, n_cells))


# -----------------------------
# Decoder step (vectorized across cells/bins)
# -----------------------------


def likelihood_grid_for_counts(
    xs: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    counts: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute likelihood grid for spike counts.

    L_grid[bin, cell] ∝ P(counts[cell] | position=xs[bin])
    Not normalized across bins; we normalize later per-cell.
    """
    lam = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    # Poisson PMF per bin, per cell for this time's counts
    # counts is (n_cells,), lam is (n_bins, n_cells)
    likelihood_grid = poisson.pmf(counts[None, :], lam)
    # Avoid degenerate zeros; normalize per cell (over bins) to a proper density on xs
    likelihood_grid = normalize(likelihood_grid, axis=0)
    return likelihood_grid


def apply_remap_for_likelihoods(
    likelihood: NDArray[np.floating],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    active: bool,
) -> NDArray[np.floating]:
    """Optionally replace one or more columns by others (remapping cell identities)."""
    if not active:
        return likelihood
    likelihood = likelihood.copy()

    # Handle both single tuple and tuple of tuples
    if (
        isinstance(remap_from_to, tuple)
        and len(remap_from_to) == 2
        and isinstance(remap_from_to[0], int)
    ):
        # Single remapping: (src, dst)
        src, dst = remap_from_to
        likelihood[:, src] = likelihood[:, dst]
    else:
        # Multiple remappings: ((src1, dst1), (src2, dst2), ...)
        for src, dst in remap_from_to:
            likelihood[:, src] = likelihood[:, dst]

    return likelihood


def decode_and_diagnostics(
    spikes: NDArray[np.int_],
    xs: NDArray[np.floating],
    osm: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    remap_window: tuple[int, int],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray]:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    Returns dict with: post, HPDO, KL, spikeProb
    """
    n_time, n_cells = spikes.shape
    n_bins = xs.size

    post = np.zeros((n_time, n_bins), dtype=float)
    hpdo = np.full((n_time, n_cells), np.nan, dtype=float)
    kl = np.full((n_time, n_cells), np.nan, dtype=float)
    spike_prob = np.full((n_time, n_cells), np.nan, dtype=float)

    # t=0 (MATLAB used a flat prior at t=1)
    post[0] = normalize(np.ones(n_bins))

    lam_grid_all = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    lambda_ratio = normalize(lam_grid_all, axis=1)  # per-bin cell-fractions, rows sum to 1

    start_r, end_r = remap_window
    for t in range(1, n_time):
        # Predict (prior)
        prior = normalize(post[t - 1] @ osm)  # (n_bins,)

        # Likelihood grid for this time's counts (vectorized over bins & cells)
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, spikes[t])
        # Optional remap (imitating MATLAB's j==10 uses field of j==1 in a window)
        active_remap = start_r <= t <= end_r
        likelihood = apply_remap_for_likelihoods(likelihood, remap_from_to, active_remap)

        # Compute diagnostics using statespacecheck functions
        # Process each cell separately since statespacecheck expects matching spatial dims
        prior_t = prior[np.newaxis, :]  # (1, n_bins)

        # Compute per-cell diagnostics
        for cell_idx in range(n_cells):
            # Get likelihood for this cell: (n_bins,) -> (1, n_bins)
            like_cell = likelihood[:, cell_idx][np.newaxis, :]

            # HPD overlap between prior and this cell's likelihood
            hpdo_t = ssc.hpd_overlap(prior_t, like_cell, coverage=0.95)
            hpdo[t, cell_idx] = hpdo_t[0]

            # KL divergence between prior and this cell's likelihood
            kl_t = ssc.kl_divergence(prior_t, like_cell)
            kl[t, cell_idx] = kl_t[0]

        # Posterior update with product over cells (independence)
        combined = np.prod(likelihood, axis=1)  # (n_bins,)
        post[t] = normalize(prior * combined)

        # spike_prob rank statistic (raw count, matching MATLAB)
        spike_prob[t] = spike_prob_rank(prior, lambda_ratio, normalize_rank=False)

    # Mask individual cells with zero spikes (match MATLAB: HPDO(spikes == 0) = nan)
    hpdo[spikes == 0] = np.nan
    kl[spikes == 0] = np.nan
    spike_prob[spikes == 0] = np.nan

    return {"post": post, "HPDO": hpdo, "KL": kl, "spikeProb": spike_prob}


# -----------------------------
# Thresholds & transforms
# -----------------------------


@dataclass
class Thresholds:
    """Threshold values for diagnostic metrics."""

    HPDO: float
    KL: float
    spike_prob: float


def compute_thresholds(metrics: dict[str, NDArray], baseline_end: int = 60_000) -> Thresholds:
    """Compute threshold values from baseline period."""
    hpdo_thresh = np.nanquantile(metrics["HPDO"][:baseline_end], 0.01)
    kl_thresh = np.nanquantile(metrics["KL"][:baseline_end], 0.99)
    # MATLAB uses 0.05 as fixed threshold (raw count, not normalized)
    spike_prob_thresh = 0.05
    return Thresholds(HPDO=hpdo_thresh, KL=kl_thresh, spike_prob=spike_prob_thresh)


@dataclass
class Transformed:
    """Transformed diagnostic metrics and thresholds."""

    HPDO: NDArray[np.floating]
    KL: NDArray[np.floating]
    spike_prob: NDArray[np.floating]
    HPDO_th: float
    KL_th: float
    spike_prob_th: float


def transform_metrics(
    metrics: dict[str, NDArray], th: Thresholds, eps1: float = 1e-2, eps2: float = 1e-10
) -> Transformed:
    """Apply transformations to metrics for better visualization."""
    hpdo_transformed = -safe_log(metrics["HPDO"] + eps1)
    kl_transformed = np.sqrt(metrics["KL"])
    spike_prob_transformed = -safe_log(metrics["spikeProb"] + eps2)

    return Transformed(
        HPDO=hpdo_transformed,
        KL=kl_transformed,
        spike_prob=spike_prob_transformed,
        HPDO_th=-np.log(th.HPDO + eps1),
        KL_th=np.sqrt(th.KL),
        spike_prob_th=-np.log(th.spike_prob + eps2),
    )


# -----------------------------
# Plotting
# -----------------------------


def plot_original(
    xs: NDArray,
    x_true: NDArray,
    metrics: dict[str, NDArray],
    th: Thresholds,
    title: str = "Original Metrics",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, ...] | None = None,
) -> plt.Figure:
    """Plot original diagnostic metrics with thresholds.

    Parameters
    ----------
    remap_window : tuple[int, int] | None
        Time window where cell remapping occurs (start, end)
    phase_boundaries : tuple[int, ...] | None
        Boundaries between phases: (T_remap_start, T_remap_end, T_recovery1_end,
        T_flat_end, T_recovery2_end, T_fast_end)
    """
    n_time = metrics["post"].shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(7, 6), constrained_layout=True, sharex=True, dpi=150)

    im = axes[0].imshow(
        metrics["post"].T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.quantile(metrics["post"], 0.975),
        cmap="viridis",
    )
    axes[0].plot(np.arange(n_time), x_true, "k", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Position (bin)", fontsize=9, labelpad=8)
    axes[0].tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
    cbar.set_label("Probability", fontsize=8, labelpad=8)
    cbar.ax.tick_params(labelsize=7)

    for ax in axes:
        # Highlight phase boundaries with different colors for misfit vs recovery
        if phase_boundaries is not None and len(phase_boundaries) == 6:
            (
                t_remap_start,
                t_remap_end,
                t_recovery1_end,
                t_flat_end,
                t_recovery2_end,
                t_fast_end,
            ) = phase_boundaries

            # Misfit periods (colored)
            ax.axvspan(t_remap_start, t_remap_end, alpha=0.2, color="orange", label="Remapping")
            ax.axvspan(
                t_recovery1_end,
                t_flat_end,
                alpha=0.2,
                color="gray",
                label="Flat Firing",
            )
            ax.axvspan(
                t_recovery2_end,
                t_fast_end,
                alpha=0.2,
                color="red",
                label="Fast Movement",
            )

            # Recovery periods (very light green background)
            ax.axvspan(t_remap_end, t_recovery1_end, alpha=0.08, color="green")
            ax.axvspan(t_flat_end, t_recovery2_end, alpha=0.08, color="green")

    axes[1].plot(
        metrics["HPDO"],
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[1].axhline(th.HPDO, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("HPD Overlap", fontsize=9, labelpad=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    axes[2].plot(metrics["KL"], ".", markersize=1.5, alpha=0.6, color="#56B4E9", rasterized=True)
    axes[2].axhline(th.KL, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("KL Divergence", fontsize=9, labelpad=8)
    axes[2].tick_params(labelsize=7)

    axes[3].plot(
        metrics["spikeProb"],
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[3].axhline(th.spike_prob, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("Spike Probability", fontsize=9, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=9, labelpad=8)
    axes[3].tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, y=0.995)
    return fig


def plot_transformed(
    xs: NDArray,
    x_true: NDArray,
    post: NDArray,
    tr: Transformed,
    title: str = "Transformed Metrics (-log, sqrt)",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot transformed diagnostic metrics with thresholds.

    Parameters
    ----------
    remap_window : tuple[int, int] | None
        Time window where cell remapping occurs (start, end)
    phase_boundaries : tuple[int, int] | None
        Boundaries between phases: (T1, T2) where T3 is end of data
    """
    n_time = post.shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(7, 6), constrained_layout=True, sharex=True, dpi=150)

    im = axes[0].imshow(post.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].plot(np.arange(n_time), x_true, "k", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Position (bin)", fontsize=9, labelpad=8)
    axes[0].tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
    cbar.set_label("Probability", fontsize=8, labelpad=8)
    cbar.ax.tick_params(labelsize=7)

    for ax in axes:
        # Highlight remap window (cell 10->1)
        if remap_window is not None:
            ax.axvspan(
                remap_window[0],
                remap_window[1],
                alpha=0.15,
                color="orange",
                label="Remap",
            )

        # Highlight phase boundaries
        if phase_boundaries is not None:
            t1, t2 = phase_boundaries
            ax.axvspan(t1, t2, alpha=0.15, color="gray", label="Flat rate")
            ax.axvspan(t2, n_time, alpha=0.15, color="red", label="Fast movement")

    axes[1].plot(tr.HPDO, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[1].axhline(tr.HPDO_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("-log(HPD Overlap)", fontsize=9, labelpad=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    axes[2].plot(tr.KL, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[2].axhline(tr.KL_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("sqrt(KL Divergence)", fontsize=9, labelpad=8)
    axes[2].tick_params(labelsize=7)

    axes[3].plot(tr.spike_prob, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[3].axhline(tr.spike_prob_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("-log(Spike Prob)", fontsize=9, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=9, labelpad=8)
    axes[3].tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, y=0.995)
    return fig


# -----------------------------
# Main orchestration
# -----------------------------


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with three simulation phases."""
    rng = np.random.default_rng(params.base_seed)

    # Ensure pf_centers is initialized
    assert params.pf_centers is not None, "pf_centers must be initialized"

    # Grid & transition (decoder keeps sigx_pred fixed across all parts)
    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    osm = gaussian_transition_matrix(xs, params.sigx_pred)

    # Generate all phases with recovery periods
    phases = []
    phase_labels = []

    # Phase 1: Clean baseline (0 - T_remap_start)
    x_last = 0.0
    n_time = params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Baseline")
    x_last = x_true_phase[-1]

    # Phase 2: Remapping misfit (T_remap_start - T_remap_end)
    n_time = params.T_remap_end - params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Remapping Misfit")
    x_last = x_true_phase[-1]

    # Phase 3: Recovery 1 (T_remap_end - T_recovery1_end)
    n_time = params.T_recovery1_end - params.T_remap_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 4: Flat firing misfit (T_recovery1_end - T_flat_end)
    n_time = params.T_flat_end - params.T_recovery1_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_flat_rate(n_time, len(params.pf_centers), rate=7e-3, rng=rng)
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Flat Firing Misfit")
    x_last = x_true_phase[-1]

    # Phase 5: Recovery 2 (T_flat_end - T_recovery2_end)
    n_time = params.T_recovery2_end - params.T_flat_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 6: Fast movement misfit (T_recovery2_end - T_fast_end)
    n_time = params.T_fast_end - params.T_recovery2_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_fast, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Fast Movement Misfit")

    # Concatenate all phases
    x_true = np.concatenate([x for x, _ in phases], axis=0)
    spikes = np.vstack([s for _, s in phases])

    # Decode (vectorized within time)
    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        osm=osm,
        pf_centers=params.pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
    )

    # Thresholds from clean baseline window (first 6k timesteps, before remapping starts)
    th = compute_thresholds(metrics, baseline_end=params.T_remap_start)

    # Plots (original) with highlighted regions
    # Mark all phase boundaries for visualization
    phase_boundaries = (
        params.T_remap_start,
        params.T_remap_end,
        params.T_recovery1_end,
        params.T_flat_end,
        params.T_recovery2_end,
        params.T_fast_end,
    )
    plot_original(
        xs,
        x_true,
        metrics,
        th,
        title="State Space Model Diagnostics with Recovery Periods",
        remap_window=params.remap_window,
        phase_boundaries=phase_boundaries,
    )

    # # Transforms & plots with highlighted regions
    # tr = transform_metrics(metrics, th)
    # plot_transformed(
    #     xs,
    #     x_true,
    #     metrics["post"],
    #     tr,
    #     title="Transformed Metrics (-log, sqrt)",
    #     remap_window=params.remap_window,
    #     phase_boundaries=(params.T1, params.T2),
    # )

    # Save figure instead of showing (for non-interactive execution)
    plt.savefig("sim_diagnostics.png", dpi=300, bbox_inches="tight")
    print("\nFigure saved to sim_diagnostics.png")


if __name__ == "__main__":
    # Default params mirror the MATLAB script. To run quickly while prototyping,
    # reduce T1/T2/T3 here.
    params = DecodeParams()
    # e.g., for a fast smoke test:
    # params = DecodeParams(T1=3_000, T2=4_000, T3=5_000)
    run_demo(params)
