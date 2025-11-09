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
    """Compute one-step transition matrix for Gaussian random walk.

    Returns matrix[i, j] = p(x_t = xs[i] | x_{t-1} = xs[j]) with std sig.
    """
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
) -> NDArray[np.floating]:
    """Compute cumulative probability mass of cells with low expected contribution.

    Matches MATLAB: sum(lambda_expect(lambda_expect <= lambda_expect(j)))
    where lambda_expect are probabilities summing to 1.

    prior: (n_bins,)
    lambda_ratio: (n_bins, n_cells), rows sum to 1.
    returns: (n_cells,), each in [0,1] representing cumulative probability mass.
    """
    contrib = prior @ lambda_ratio  # (n_cells,) - probabilities summing to ~1
    # Sum probability mass of cells with contribution <= each cell's contribution
    mask = contrib[:, None] <= contrib[None, :]  # (n_cells, n_cells)
    spike_probs = (contrib[None, :] * mask).sum(axis=1)  # Sum probabilities, not counts
    return spike_probs


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
    # 24k-28k: Clean recovery (4k)
    # 28k-32k: Slow movement misfit (4k)
    T_remap_start: int = 6_000
    T_remap_end: int = 10_000
    T_recovery1_end: int = 14_000
    T_flat_end: int = 16_000
    T_recovery2_end: int = 20_000
    T_fast_end: int = 24_000
    T_recovery3_end: int = 28_000
    T_slow_end: int = 32_000
    sigx_pred: float = 0.5  # decoder's dynamics std (baseline)
    sigx_pred_fast_phase: float = 0.1  # narrow decoder for fast phase (5x too narrow!)
    sigx_pred_slow_phase: float = 20.0  # inflated decoder for slow phase (40x too broad!)
    sigx_true_fast: float = 10.0  # true dynamics std in fast phase (100x faster than decoder!)
    sigx_true_slow: float = 0.0  # true dynamics std in slow phase (completely stationary!)
    xs_min: int = 0
    xs_max: int = 100
    xs_step: int = 1
    pf_width: float = 5.0  # Narrow place fields for sharp spatial selectivity
    pf_centers: NDArray[np.floating] | None = None  # set in __post_init__
    rate_scale: float = 0.15  # Higher spike rate to reduce uncertainty
    base_seed: int = 1
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 5),  # Position 0 → 50 (shift +50cm)
        (1, 6),  # Position 10 → 60
        (2, 7),  # Position 20 → 70
        (3, 8),  # Position 30 → 80
        (4, 9),  # Position 40 → 90
        (5, 10),  # Position 50 → 100
        (6, 0),  # Position 60 → 0
        (7, 1),  # Position 70 → 10
        (8, 2),  # Position 80 → 20
        (9, 3),  # Position 90 → 30
        (10, 4),  # Position 100 → 40
    )  # Remap ALL 11 cells with +50cm circular shift

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
    transition_matrix: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    remap_window: tuple[int, int],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    rng: np.random.Generator | None = None,
    transition_matrix_narrow: NDArray[np.floating] | None = None,
    narrow_window: tuple[int, int] | None = None,
    transition_matrix_inflated: NDArray[np.floating] | None = None,
    inflate_window: tuple[int, int] | None = None,
) -> dict[str, NDArray]:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    Returns dict with: post, HPDO, KL, spikeProb
    """
    n_time, n_cells = spikes.shape
    n_bins = xs.size

    post = np.zeros((n_time, n_bins), dtype=float)
    hpdo = np.full(n_time, np.nan, dtype=float)  # Single value per timestep
    kl = np.full(n_time, np.nan, dtype=float)  # Single value per timestep
    spike_prob = np.full((n_time, n_cells), np.nan, dtype=float)  # Keep per-cell for this metric

    # t=0 (MATLAB used a flat prior at t=1)
    post[0] = normalize(np.ones(n_bins))

    lam_grid_all = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    lambda_ratio = normalize(lam_grid_all, axis=1)  # per-bin cell-fractions, rows sum to 1

    start_r, end_r = remap_window
    start_narrow, end_narrow = narrow_window if narrow_window else (n_time + 1, n_time + 1)
    start_inflate, end_inflate = inflate_window if inflate_window else (n_time + 1, n_time + 1)

    for t in range(1, n_time):
        # Select transition matrix based on which window we're in
        if transition_matrix_narrow is not None and start_narrow <= t <= end_narrow:
            current_transition = transition_matrix_narrow
        elif transition_matrix_inflated is not None and start_inflate <= t <= end_inflate:
            current_transition = transition_matrix_inflated
        else:
            current_transition = transition_matrix

        # Predict (prior from state dynamics)
        prior = normalize(post[t - 1] @ current_transition)  # (n_bins,)

        # Likelihood grid for this time's counts (vectorized over bins & cells)
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, spikes[t])
        # Optional remap (imitating MATLAB's j==10 uses field of j==1 in a window)
        active_remap = start_r <= t <= end_r
        likelihood = apply_remap_for_likelihoods(likelihood, remap_from_to, active_remap)

        # Compute combined likelihood from all cells (product over cells)
        combined_likelihood = np.prod(likelihood, axis=1)  # (n_bins,)

        # Compute diagnostics using statespacecheck functions
        # Compare one-step prediction (prior) with combined likelihood (observation model)
        prior_t = prior[np.newaxis, :]  # (1, n_bins)
        combined_likelihood_t = combined_likelihood[np.newaxis, :]  # (1, n_bins)

        # HPD overlap between prior and combined likelihood
        hpdo_t = ssc.hpd_overlap(prior_t, combined_likelihood_t, coverage=0.95)
        hpdo[t] = hpdo_t[0]

        # KL divergence between prior and combined likelihood
        kl_t = ssc.kl_divergence(prior_t, combined_likelihood_t)
        kl[t] = kl_t[0]

        # Posterior update
        post[t] = normalize(prior * combined_likelihood)

        # spike_prob: cumulative probability mass for cells with low expected contribution
        spike_prob[t] = spike_prob_rank(prior, lambda_ratio)

    # Mask spike_prob for cells with zero spikes (match MATLAB: spikeProb(spikes == 0) = nan)
    # Note: HPDO and KL are now per-timestep (not per-cell) since they compare
    # the combined likelihood with the prior, so we don't mask them
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
        T_flat_end, T_recovery2_end, T_fast_end, T_recovery3_end, T_slow_end)
    """
    n_time = metrics["post"].shape[0]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(8, 6),
        constrained_layout={
            "h_pad": 0.02,
            "w_pad": 0.02,
            "hspace": 0,
            "wspace": 0,
            "rect": [0, 0, 1, 0.97],
        },
        sharex=True,
        dpi=150,
    )

    im = axes[0].imshow(
        metrics["post"].T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.quantile(metrics["post"], 0.975),
        cmap="bone_r",
    )
    # Plot true position in magenta for visibility against bone_r colormap
    axes[0].plot(
        np.arange(n_time), x_true, color="magenta", linewidth=1.5, alpha=0.85, label="True position"
    )
    axes[0].set_ylabel("Position (bin)", fontsize=9, labelpad=8)
    axes[0].tick_params(labelsize=7)

    # Create colorbar with better formatting
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label("Probability (×10⁻¹²)", fontsize=8, labelpad=8)
    cbar.ax.tick_params(labelsize=7, length=3, width=0.5)
    # Scale tick labels by 1e12 to avoid offset text
    import matplotlib.ticker as ticker

    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x * 1e12:.1f}"))

    # Add phase boundaries to all axes, but only add labels to first axis for legend
    for i, ax in enumerate(axes):
        # Highlight phase boundaries with different colors for misfit vs recovery
        if phase_boundaries is not None and len(phase_boundaries) == 8:
            (
                t_remap_start,
                t_remap_end,
                t_recovery1_end,
                t_flat_end,
                t_recovery2_end,
                t_fast_end,
                t_recovery3_end,
                t_slow_end,
            ) = phase_boundaries

            # Only add labels for the first axis (for legend)
            add_labels = i == 0

            # Misfit periods (colored)
            ax.axvspan(
                t_remap_start,
                t_remap_end,
                alpha=0.2,
                color="orange",
                label="Remapping" if add_labels else "",
            )
            ax.axvspan(
                t_recovery1_end,
                t_flat_end,
                alpha=0.2,
                color="gray",
                label="Flat firing" if add_labels else "",
            )
            ax.axvspan(
                t_recovery2_end,
                t_fast_end,
                alpha=0.2,
                color="red",
                label="Fast movement" if add_labels else "",
            )
            ax.axvspan(
                t_recovery3_end,
                t_slow_end,
                alpha=0.2,
                color="blue",
                label="Stationary" if add_labels else "",
            )

    axes[1].plot(
        metrics["HPDO"],
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[1].axhline(th.HPDO, color="#E69F00", linewidth=1.5, zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("HPD Overlap", fontsize=9, labelpad=8)
    axes[1].tick_params(labelsize=7)

    axes[2].plot(metrics["KL"], ".", markersize=1.5, alpha=0.6, color="#56B4E9", rasterized=True)
    axes[2].axhline(th.KL, color="#E69F00", linewidth=1.5, zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("KL Divergence", fontsize=9, labelpad=8)
    axes[2].tick_params(labelsize=7)

    # Transform spike probability to -log scale
    eps2 = 1e-12
    spike_prob_transformed = -safe_log(metrics["spikeProb"] + eps2)
    spike_prob_thresh_transformed = -np.log(th.spike_prob + eps2)

    axes[3].plot(
        spike_prob_transformed,
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[3].axhline(spike_prob_thresh_transformed, color="#E69F00", linewidth=1.5, zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("-log(Spike Prob)", fontsize=9, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=9, labelpad=8)
    axes[3].tick_params(labelsize=7)

    # Add comprehensive legend outside the plot area at the bottom
    # Get handles and labels from axes[0] where they were defined
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        fontsize=8,
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=5,
    )

    fig.suptitle(title, fontsize=10, y=0.99)
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

    fig.suptitle(title, fontsize=10, y=0.998)
    return fig


def plot_misfit_examples(
    xs: NDArray,
    x_true: NDArray,
    spikes: NDArray,
    metrics: dict[str, NDArray],
    params: DecodeParams,
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
) -> None:
    """Plot examples of high misfit moments for each scenario.

    Finds the worst time point in each misfit phase and shows the distributions.
    Shows 4 columns for the 4 misfit types.
    """
    # Define phase windows for misfit phases only (no baseline)
    remap_window = slice(params.T_remap_start, params.T_remap_end)
    flat_window = slice(params.T_recovery1_end, params.T_flat_end)
    fast_window = slice(params.T_recovery2_end, params.T_fast_end)
    slow_window = slice(params.T_recovery3_end, params.T_slow_end)

    phases = [
        ("Remapping", remap_window),
        ("Flat Firing", flat_window),
        ("Fast Movement", fast_window),
        ("Slow Movement", slow_window),
    ]

    # Publication quality: 450 DPI, single row with 4 columns
    fig = plt.figure(figsize=(10.0, 2.5), dpi=450, constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # Wong colorblind-friendly palette
    wong = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    for phase_idx, (phase_name, phase_slice) in enumerate(phases):
        # Find worst fit (lowest HPDO) but only consider time points with spikes
        phase_hpdo = metrics["HPDO"][phase_slice]
        phase_spikes = spikes[phase_slice]

        # Mask times without spikes (likelihood will be flat/uninformative)
        has_spikes = phase_spikes.sum(axis=1) > 0
        valid_hpdo = phase_hpdo.copy()
        valid_hpdo[~has_spikes] = np.nan  # Exclude times without spikes

        example_idx_in_phase = np.nanargmin(valid_hpdo)  # Worst fit with spikes
        example_time = phase_slice.start + example_idx_in_phase

        # Recompute prior and likelihood at this time point
        # Get posterior from previous timestep
        if example_time > 0:
            prev_post = metrics["post"][example_time - 1]
        else:
            prev_post = np.ones_like(xs) / len(xs)

        # Select appropriate transition matrix
        if params.T_recovery2_end <= example_time <= params.T_fast_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
        elif params.T_recovery3_end <= example_time <= params.T_slow_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)
        else:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)

        # Compute prior
        prior = normalize(prev_post @ transition_matrix)

        # Compute combined likelihood
        likelihood = likelihood_grid_for_counts(
            xs, pf_centers, pf_width, rate_scale, spikes[example_time]
        )

        # Apply remapping if in remap window
        if params.T_remap_start <= example_time <= params.T_remap_end:
            likelihood = apply_remap_for_likelihoods(likelihood, params.remap_from_to, active=True)

        combined_likelihood = normalize(np.prod(likelihood, axis=1))

        # Plot prior and likelihood with twin axes - use Wong colorblind-friendly palette
        ax1 = axes[phase_idx]
        ax2 = ax1.twinx()

        # Plot prior on left axis (blue from Wong palette) with transparency
        line1 = ax1.plot(xs, prior, color=wong[5], linewidth=1.5, alpha=0.7, label="Prior")
        ax1.set_ylabel("Prior", fontsize=7, color=wong[5], labelpad=3)
        ax1.tick_params(axis="y", labelcolor=wong[5], labelsize=6)
        ax1.set_ylim(0, None)
        # Use scientific notation for small/large values
        ax1.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2), useMathText=True)
        ax1.yaxis.get_offset_text().set_fontsize(6)
        ax1.yaxis.get_offset_text().set_color(wong[5])

        # Plot likelihood on right axis (orange from Wong palette) - solid line
        line2 = ax2.plot(
            xs, combined_likelihood, color=wong[1], linewidth=1.5, alpha=0.9, label="Likelihood"
        )
        ax2.set_ylabel("Likelihood", fontsize=7, color=wong[1], labelpad=3)
        ax2.tick_params(axis="y", labelcolor=wong[1], labelsize=6)
        ax2.set_ylim(0, None)
        # Use scientific notation for small/large values
        ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2), useMathText=True)
        ax2.yaxis.get_offset_text().set_fontsize(6)
        ax2.yaxis.get_offset_text().set_color(wong[1])

        # Add true position line (purple from Wong palette)
        ax1.axvline(x_true[example_time], color=wong[7], linestyle="--", linewidth=1.0, alpha=0.7)

        # Get diagnostic values
        hpdo_val = metrics["HPDO"][example_time]
        kl_val = metrics["KL"][example_time]
        spike_prob_vals = metrics["spikeProb"][example_time]

        # Calculate -log(min spike prob) with only significant digits
        if not np.all(np.isnan(spike_prob_vals)):
            spike_prob_min = np.nanmin(spike_prob_vals)
            log_spike_prob = -np.log(spike_prob_min + 1e-12)
        else:
            log_spike_prob = np.nan

        # Add phase name and metrics as title (always use engineering format for -log)
        title_text = (
            f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  -log: {log_spike_prob:.2e}"
        )
        if np.isnan(log_spike_prob):
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  -log: N/A"
        ax1.set_title(title_text, fontsize=7, pad=5, fontweight="bold")

        ax1.tick_params(axis="x", labelsize=6)
        ax1.set_xlabel("Position", fontsize=7, labelpad=3)

        # Add legend to first panel only
        if phase_idx == 0:
            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, fontsize=5, loc="lower right", frameon=False)

    # Save to scripts directory (publication quality: both PDF and PNG)
    import os

    save_path_base = os.path.join(os.path.dirname(__file__), "misfit_examples")

    # Save PDF (vector) for publication
    plt.savefig(f"{save_path_base}.pdf", dpi=450, bbox_inches="tight")
    # Save PNG (raster) for quick viewing
    plt.savefig(f"{save_path_base}.png", dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Misfit examples figure saved to {save_path_base}.{{pdf,png}}")


# -----------------------------
# Main orchestration
# -----------------------------


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with three simulation phases."""
    rng = np.random.default_rng(params.base_seed)

    # Ensure pf_centers is initialized
    assert params.pf_centers is not None, "pf_centers must be initialized"

    # Grid & transition matrices
    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    transition_matrix_narrow = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
    transition_matrix_inflated = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)

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
    # Transition model misfit: decoder uses narrow transition matrix (sigx=0.1),
    # animal moves fast (sigx=10.0)
    # Prior will be far too narrow/concentrated compared to actual movement (100x mismatch!)
    n_time = params.T_fast_end - params.T_recovery2_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_fast, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Fast Movement Misfit")
    x_last = x_true_phase[-1]

    # Phase 7: Recovery 3 (T_fast_end - T_recovery3_end)
    n_time = params.T_recovery3_end - params.T_fast_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 8: Slow movement misfit (T_recovery3_end - T_slow_end)
    # Transition model misfit: decoder uses inflated transition matrix (sigx=20.0),
    # animal stationary (sigx=0.0)
    # Prior will be far too broad/diffuse compared to actual lack of movement
    n_time = params.T_slow_end - params.T_recovery3_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_slow, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Slow Movement Misfit")

    # Concatenate all phases
    x_true = np.concatenate([x for x, _ in phases], axis=0)
    spikes = np.vstack([s for _, s in phases])

    # Decode (vectorized within time)
    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        transition_matrix=transition_matrix,
        pf_centers=params.pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
        transition_matrix_narrow=transition_matrix_narrow,
        narrow_window=(params.T_recovery2_end, params.T_fast_end),
        transition_matrix_inflated=transition_matrix_inflated,
        inflate_window=(params.T_recovery3_end, params.T_slow_end),
    )

    # Thresholds from clean baseline window (first 6k timesteps, before remapping starts)
    th = compute_thresholds(metrics, baseline_end=params.T_remap_start)

    # Plot misfit examples showing distributions
    plot_misfit_examples(
        xs, x_true, spikes, metrics, params, params.pf_centers, params.pf_width, params.rate_scale
    )

    # Plots (original) with highlighted regions
    # Mark all phase boundaries for visualization
    phase_boundaries = (
        params.T_remap_start,
        params.T_remap_end,
        params.T_recovery1_end,
        params.T_flat_end,
        params.T_recovery2_end,
        params.T_fast_end,
        params.T_recovery3_end,
        params.T_slow_end,
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
