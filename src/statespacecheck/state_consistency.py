"""State consistency tests for state space model goodness of fit.

This module provides functions to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models. These tests help identify issues with prior specification
and model assumptions.
"""

import numpy as np
from scipy.stats import entropy

from .highest_density import highest_density_region


def kl_divergence(state_dist: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Compute Kullback-Leibler divergence between state distribution and likelihood.

    Measures the information divergence between the state distribution and likelihood
    distributions at each time point. Large divergences may indicate issues
    with the prior specification or model assumptions.

    Parameters
    ----------
    state_dist : np.ndarray
        State probability distributions over position at each time point.
        Can be either one-step predictive distribution or smoother output.
        Must be properly normalized probability distributions.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray
        Normalized likelihood distributions at each time point (equivalent to
        posterior with uniform prior). Must have same shape as state_dist and
        be properly normalized.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.

    Returns
    -------
    kl_divergence : np.ndarray
        Kullback-Leibler divergence D_KL(state_dist || likelihood) at each
        time point. Values are non-negative, with 0 indicating identical
        distributions.
        Shape (n_time,).

    Raises
    ------
    ValueError
        If state_dist and likelihood have different shapes, or if distributions
        contain negative values.

    Notes
    -----
    The KL divergence is computed using scipy.stats.entropy with the formula:
    D_KL(P || Q) = sum(P * log(P / Q))
    where P is the state distribution and Q is the likelihood.

    Time slices where distributions sum to zero or contain invalid values
    return inf for the divergence.

    """
    state = np.asarray(state_dist)
    like = np.asarray(likelihood)

    if state.shape != like.shape:
        raise ValueError(
            f"state_dist and likelihood must have same shape, got {state.shape} vs {like.shape}"
        )

    n_time = state.shape[0]
    # Flatten all spatial dimensions
    state_flat = state.reshape(n_time, -1)
    like_flat = like.reshape(n_time, -1)

    # Check for negative values
    if np.any(state_flat < 0) or np.any(like_flat < 0):
        raise ValueError("Distributions must be non-negative.")

    # Compute sums for normalization check
    state_sum = state_flat.sum(axis=1)
    like_sum = like_flat.sum(axis=1)

    # Initialize output
    kl_div = np.full(n_time, np.inf, dtype=float)

    # Find valid time slices (both distributions sum to positive values)
    valid = (state_sum > 0) & (like_sum > 0)

    if np.any(valid):
        # Normalize and compute KL divergence for valid slices
        state_norm = state_flat[valid] / state_sum[valid, np.newaxis]
        like_norm = like_flat[valid] / like_sum[valid, np.newaxis]
        kl_div[valid] = entropy(state_norm, like_norm, axis=1)

    return kl_div


def hpd_overlap(
    state_dist: np.ndarray, likelihood: np.ndarray, coverage: float = 0.95
) -> np.ndarray:
    """Compute overlap between HPD regions of state distribution and likelihood.

    Measures the spatial overlap between the highest posterior density regions
    of the state distribution and likelihood distributions. High overlap suggests
    consistency between the likelihood and prior contributions to the state estimate.

    Parameters
    ----------
    state_dist : np.ndarray
        State probability distributions over position at each time point.
        Can be either one-step predictive distribution or smoother output.
        Must be properly normalized probability distributions.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray
        Normalized likelihood distributions at each time point (equivalent to
        posterior with uniform prior). Must have same shape as state_dist and
        be properly normalized.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    coverage : float, optional
        Coverage probability for the HPD regions. Must be between 0 and 1.
        Default is 0.95 for 95% HPD regions.

    Returns
    -------
    hpd_overlap : np.ndarray
        Proportion of overlap between the HPD regions of state_dist and
        likelihood at each time point. Values range from 0 (no overlap)
        to 1 (complete overlap).
        Shape (n_time,).

    Raises
    ------
    ValueError
        If state_dist and likelihood have different shapes, or if coverage
        is not in (0, 1).

    Notes
    -----
    The overlap is computed as:
        overlap = intersection(HPD_state, HPD_like) / min(size(HPD_state), size(HPD_like))

    This normalization ensures that:
    - overlap = 1.0 when one region completely contains the other
    - overlap = 0.0 when regions don't overlap at all
    - Values are comparable even when HPD regions have different sizes

    When both HPD regions are empty (both sizes are 0), overlap is defined as 0.

    """
    state = np.asarray(state_dist)
    like = np.asarray(likelihood)

    if state.shape != like.shape:
        raise ValueError(
            f"state_dist and likelihood must have same shape, got {state.shape} vs {like.shape}"
        )

    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")

    # Get HPD regions
    mask_state = highest_density_region(state, coverage=coverage)
    mask_like = highest_density_region(like, coverage=coverage)

    # Sum over all spatial dimensions (everything except time)
    spatial_axes = tuple(range(1, state.ndim))
    size_state = mask_state.sum(axis=spatial_axes)
    size_like = mask_like.sum(axis=spatial_axes)
    intersection = (mask_state & mask_like).sum(axis=spatial_axes)

    # Compute denominator (minimum of the two sizes)
    denom = np.minimum(size_state, size_like)

    # Avoid division by zero: if both sizes are 0, define overlap as 0
    denom = np.where(denom == 0, 1, denom)
    overlap = intersection / denom

    # Set overlap to 0 where both regions were empty
    overlap = np.where((size_state == 0) & (size_like == 0), 0.0, overlap)

    return overlap
