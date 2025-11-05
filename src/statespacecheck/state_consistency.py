"""State consistency tests for state space model goodness of fit.

This module provides functions to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models. These tests help identify issues with prior specification
and model assumptions.
"""

import numpy as np
from scipy.stats import entropy

from ._validation import get_spatial_axes, validate_coverage, validate_paired_distributions
from .highest_density import highest_density_region


def _validate_and_normalize_distributions(
    state_dist: np.ndarray, likelihood: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize distributions, handling NaN values correctly.

    Parameters
    ----------
    state_dist : np.ndarray
        State distributions. Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray
        Likelihood distributions. Must have same shape as state_dist.

    Returns
    -------
    state_normalized : np.ndarray
        Normalized state distributions (NaNs preserved, normalization over valid bins).
    likelihood_normalized : np.ndarray
        Normalized likelihood distributions (NaNs preserved, normalization over valid bins).

    Raises
    ------
    ValueError
        If shapes don't match or distributions contain negative values.

    Notes
    -----
    - NaN/inf values in input are treated as invalid bins:
      * First converted to 0 by validation utilities for computation
      * Excluded from normalization sum (using nansum)
      * After normalization, output has 0.0 probability for invalid bins
    - Each time slice is normalized to sum to 1.0 over valid bins
    - Zero-sum rows (all NaN or all zero) remain as zeros and will cause downstream
      functions to return inf (KL divergence) or empty regions (HPD overlap)
    """
    # Use validation utilities for consistent validation
    # This converts NaN/inf to 0 but keeps zeros that represent actual zero probability
    state, state_flat, like, like_flat = validate_paired_distributions(
        state_dist, likelihood, name1="state_dist", name2="likelihood", min_ndim=2
    )

    # Normalize each time slice
    # After validation, NaN/inf already converted to 0, so use regular sum
    state_sum = state_flat.sum(axis=1)  # (n_time,)
    like_sum = like_flat.sum(axis=1)  # (n_time,)

    # Normalize: divide by sum where sum > 0, otherwise keep as zeros
    # Avoid division by zero by replacing zeros with 1 (will multiply by 0 anyway)
    valid_state = state_sum > 0
    valid_like = like_sum > 0

    # Safe division: replace sum=0 with 1 to avoid div by zero warnings
    # Then multiply by mask to zero out invalid rows
    state_sum_safe = np.where(valid_state, state_sum, 1.0)
    like_sum_safe = np.where(valid_like, like_sum, 1.0)

    state_norm_flat = (state_flat / state_sum_safe[:, np.newaxis]) * valid_state[:, np.newaxis]
    like_norm_flat = (like_flat / like_sum_safe[:, np.newaxis]) * valid_like[:, np.newaxis]

    # Reshape back to original shape
    state_norm = state_norm_flat.reshape(state.shape)
    like_norm = like_norm_flat.reshape(like.shape)

    # Note: No need to call nan_to_num here since validation already converted NaN to 0
    # The output arrays have 0.0 for invalid bins and normalized probabilities for valid bins

    return state_norm, like_norm


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
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray
        Likelihood distributions at each time point. This is the
        likelihood p(y_t | x_t) across spatial positions.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Must have same shape as state_dist.
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

    Distributions are automatically normalized over valid (non-NaN) bins.
    NaN values mark invalid spatial bins (e.g., inaccessible locations)
    and are excluded from both normalization and KL computation.

    Time slices where distributions have no valid mass return inf for the divergence.

    """
    # Validate and normalize distributions (handles NaN correctly)
    state_norm, like_norm = _validate_and_normalize_distributions(state_dist, likelihood)

    n_time = state_norm.shape[0]
    # Flatten all spatial dimensions
    state_flat = state_norm.reshape(n_time, -1)
    like_flat = like_norm.reshape(n_time, -1)

    # Check for empty rows (sum == 0)
    # After normalization, arrays have no NaNs: valid rows sum to 1.0, empty rows sum to 0.0
    state_sum = state_flat.sum(axis=1)
    like_sum = like_flat.sum(axis=1)

    # Initialize output
    kl_div = np.full(n_time, np.inf, dtype=float)

    # Find valid time slices (both distributions have positive mass over valid bins)
    valid = (state_sum > 0) & (like_sum > 0)

    if np.any(valid):
        # Compute entropy directly (NaN already converted to 0 by validation)
        kl_div[valid] = entropy(state_flat[valid], like_flat[valid], axis=1)

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
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray
        Likelihood distributions at each time point. This is the
        likelihood p(y_t | x_t) across spatial positions.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Must have same shape as state_dist.
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
        If state_dist and likelihood have different shapes, if coverage
        is not in (0, 1), or if distributions contain negative values.

    Notes
    -----
    The overlap is computed as:
        overlap = intersection(HPD_state, HPD_like) / min(size(HPD_state), size(HPD_like))

    This normalization ensures that:
    - overlap = 1.0 when one region completely contains the other
    - overlap = 0.0 when regions don't overlap at all
    - Values are comparable even when HPD regions have different sizes

    When both HPD regions are empty (both sizes are 0), overlap is defined as 0.

    Distributions are automatically normalized over valid (non-NaN) bins.
    NaN values mark invalid spatial bins (e.g., inaccessible locations)
    and are excluded from both normalization and HPD region computation.

    """
    validate_coverage(coverage)

    # Validate and normalize distributions (handles NaN correctly)
    state_norm, like_norm = _validate_and_normalize_distributions(state_dist, likelihood)

    # Get HPD regions (highest_density_region handles NaN by treating as zero mass)
    mask_state = highest_density_region(state_norm, coverage=coverage)
    mask_like = highest_density_region(like_norm, coverage=coverage)

    # Sum over all spatial dimensions (everything except time)
    spatial_axes = get_spatial_axes(state_norm)
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
