"""Utility functions for generating example data in notebooks.

This module provides functions to generate synthetic state space model data
for pedagogical examples demonstrating the statespacecheck package.
"""

import numpy as np
from scipy.stats import norm


def generate_1d_gaussian_distribution(
    position_bins: np.ndarray,
    mean: float | np.ndarray,
    std: float | np.ndarray,
) -> np.ndarray:
    """Generate 1D Gaussian distribution over position bins.

    Parameters
    ----------
    position_bins : np.ndarray, shape (n_bins,)
        Position bin centers.
    mean : float or np.ndarray, shape (n_time,)
        Mean of Gaussian distribution(s).
    std : float or np.ndarray, shape (n_time,)
        Standard deviation of Gaussian distribution(s).

    Returns
    -------
    distribution : np.ndarray, shape (n_time, n_bins) or (n_bins,)
        Gaussian probability distribution over position bins.
        If mean and std are scalars, returns (n_bins,) array.
        If mean or std are arrays, returns (n_time, n_bins) array.
    """
    mean_arr = np.atleast_1d(mean)
    std_arr = np.atleast_1d(std)

    # Broadcast std to match mean if needed
    if len(std_arr) == 1 and len(mean_arr) > 1:
        std_arr = np.full_like(mean_arr, std_arr[0])
    elif len(mean_arr) == 1 and len(std_arr) > 1:
        mean_arr = np.full_like(std_arr, mean_arr[0])
    elif mean_arr.shape != std_arr.shape:
        raise ValueError(
            f"mean and std must have same shape, got {mean_arr.shape} vs {std_arr.shape}"
        )

    n_time = len(mean_arr)

    # Generate distributions
    distributions = np.array(
        [norm.pdf(position_bins, loc=m, scale=s) for m, s in zip(mean_arr, std_arr, strict=True)]
    )

    # If inputs were both scalars, return 1D array
    if n_time == 1 and np.isscalar(mean) and np.isscalar(std):
        return distributions[0]

    return distributions


def generate_spatial_navigation_data(
    n_time: int = 100,
    track_length: float = 100.0,
    n_bins: int = 50,
    velocity: float = 10.0,
    state_uncertainty: float = 2.0,
    likelihood_uncertainty: float = 3.0,
    drift: float = 0.0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic spatial navigation data for examples.

    Simulates an animal moving along a linear track with position estimates
    from a state space model (predictive distribution) and neural observations
    (likelihood).

    Parameters
    ----------
    n_time : int, optional
        Number of time steps. Default is 100.
    track_length : float, optional
        Length of the track in cm. Default is 100.0.
    n_bins : int, optional
        Number of spatial bins. Default is 50.
    velocity : float, optional
        Average velocity in cm/s. Default is 10.0.
    state_uncertainty : float, optional
        Standard deviation of state distribution in cm. Default is 2.0.
    likelihood_uncertainty : float, optional
        Standard deviation of likelihood distribution in cm. Default is 3.0.
    drift : float, optional
        Systematic bias between state and likelihood means in cm.
        Positive values mean likelihood is shifted forward. Default is 0.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'position_bins': np.ndarray, shape (n_bins,) - Position bin centers
        - 'true_position': np.ndarray, shape (n_time,) - True position trajectory
        - 'state_dist': np.ndarray, shape (n_time, n_bins) - State distributions
        - 'likelihood': np.ndarray, shape (n_time, n_bins) - Likelihood distributions
        - 'time': np.ndarray, shape (n_time,) - Time points
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Position bins
    position_bins = np.linspace(0, track_length, n_bins)

    # Generate true position trajectory (back and forth movement)
    time = np.arange(n_time)
    true_position = (track_length / 2) + (track_length / 3) * np.sin(2 * np.pi * time / n_time * 2)

    # Add small random walk
    true_position += rng.normal(0, 0.5, size=n_time).cumsum()
    true_position = np.clip(true_position, 0, track_length)

    # Generate state distribution (centered on true position)
    state_dist = generate_1d_gaussian_distribution(
        position_bins,
        mean=true_position,
        std=state_uncertainty,
    )

    # Generate likelihood (centered on true position + drift)
    likelihood = generate_1d_gaussian_distribution(
        position_bins,
        mean=true_position + drift,
        std=likelihood_uncertainty,
    )

    return {
        "position_bins": position_bins,
        "true_position": true_position,
        "state_dist": state_dist,
        "likelihood": likelihood,
        "time": time,
    }


def generate_misspecified_model_data(
    n_time: int = 100,
    track_length: float = 100.0,
    n_bins: int = 50,
    misfit_start: int = 40,
    misfit_end: int = 60,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate data where model is misspecified during a specific period.

    During the misfit period, the state distribution and likelihood disagree
    substantially, simulating a period where the model fails to capture the data.

    Parameters
    ----------
    n_time : int, optional
        Number of time steps. Default is 100.
    track_length : float, optional
        Length of the track in cm. Default is 100.0.
    n_bins : int, optional
        Number of spatial bins. Default is 50.
    misfit_start : int, optional
        Time index where misfit period starts. Default is 40.
    misfit_end : int, optional
        Time index where misfit period ends. Default is 60.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'position_bins': np.ndarray, shape (n_bins,) - Position bin centers
        - 'true_position': np.ndarray, shape (n_time,) - True position trajectory
        - 'state_dist': np.ndarray, shape (n_time, n_bins) - State distributions
        - 'likelihood': np.ndarray, shape (n_time, n_bins) - Likelihood distributions
        - 'time': np.ndarray, shape (n_time,) - Time points
        - 'misfit_period': tuple[int, int] - Start and end of misfit period
    """
    # Note: seed parameter is kept for API compatibility but not currently used
    # All data generation is deterministic based on input parameters
    _ = seed  # Mark as intentionally unused

    # Position bins
    position_bins = np.linspace(0, track_length, n_bins)
    time = np.arange(n_time)

    # Generate true position trajectory
    true_position = (track_length / 2) + (track_length / 3) * np.sin(2 * np.pi * time / n_time * 2)

    # Initialize distributions
    state_means = true_position.copy()
    likelihood_means = true_position.copy()
    state_stds = np.ones(n_time) * 2.0
    likelihood_stds = np.ones(n_time) * 3.0

    # Create misfit period: likelihood disagrees with state
    likelihood_means[misfit_start:misfit_end] += 20.0  # Large spatial offset
    likelihood_stds[misfit_start:misfit_end] = 5.0  # Increased uncertainty

    # Clip to track bounds
    likelihood_means = np.clip(likelihood_means, 0, track_length)

    # Generate distributions
    state_dist = generate_1d_gaussian_distribution(position_bins, state_means, state_stds)
    likelihood = generate_1d_gaussian_distribution(position_bins, likelihood_means, likelihood_stds)

    return {
        "position_bins": position_bins,
        "true_position": true_position,
        "state_dist": state_dist,
        "likelihood": likelihood,
        "time": time,
        "misfit_period": (misfit_start, misfit_end),
    }


def generate_multimodal_distribution(
    position_bins: np.ndarray,
    means: list[float],
    stds: list[float],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Generate multimodal distribution as mixture of Gaussians.

    Parameters
    ----------
    position_bins : np.ndarray, shape (n_bins,)
        Position bin centers.
    means : list of float
        Mean of each Gaussian component.
    stds : list of float
        Standard deviation of each Gaussian component.
    weights : list of float, optional
        Weight of each component (must sum to 1).
        If None, components are equally weighted.

    Returns
    -------
    distribution : np.ndarray, shape (n_bins,)
        Multimodal probability distribution.
    """
    n_components = len(means)

    if len(stds) != n_components:
        raise ValueError(f"means and stds must have same length, got {n_components} vs {len(stds)}")

    if weights is None:
        weights = [1.0 / n_components] * n_components
    elif len(weights) != n_components:
        raise ValueError(
            f"weights must have same length as means, got {len(weights)} vs {n_components}"
        )

    # Normalize weights
    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()

    # Generate mixture
    distribution = np.zeros_like(position_bins)
    for mean, std, weight in zip(means, stds, weights_arr, strict=True):
        distribution += weight * norm.pdf(position_bins, loc=mean, scale=std)

    return distribution


def configure_notebook_plotting() -> None:
    """Configure matplotlib settings for notebook figures.

    Sets up consistent styling for all example notebooks following
    best practices for scientific presentations.
    """
    import matplotlib.pyplot as plt

    # Use larger fonts for readability
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Use clear, professional styling
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Better default figure size
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            # Better color cycle
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                ],
            ),
        }
    )
