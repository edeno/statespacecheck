"""State consistency tests for state space model goodness of fit.

This module provides functions to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models. These tests help identify issues with prior specification
and model assumptions.
"""

import numpy as np
from scipy.stats import entropy

from ._validation import (
    DistributionArray,
    flatten_time_spatial,
    get_spatial_axes,
    validate_coverage,
    validate_paired_distributions,
)
from .highest_density import DEFAULT_COVERAGE, highest_density_region


def _validate_and_normalize_distributions(
    state_dist: DistributionArray, likelihood: DistributionArray
) -> tuple[DistributionArray, DistributionArray]:
    """Validate and normalize distributions, handling NaN values correctly.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, ...)
        State distributions where ... represents arbitrary spatial dimensions.
    likelihood : np.ndarray, shape (n_time, ...)
        Likelihood distributions. Must have same shape as state_dist.

    Returns
    -------
    state_normalized : np.ndarray, shape (n_time, ...)
        Normalized state distributions. NaN/inf values in input are converted to 0.0.
        Each time slice normalized to sum to 1.0 over valid (non-zero) bins.
    likelihood_normalized : np.ndarray, shape (n_time, ...)
        Normalized likelihood distributions. NaN/inf values in input are converted to 0.0.
        Each time slice normalized to sum to 1.0 over valid (non-zero) bins.

    Raises
    ------
    ValueError
        If shapes don't match or distributions contain negative values.

    Notes
    -----
    - Non-finite inputs (NaN/inf) are treated as invalid bins:
      * Converted to 0.0 by validation for computation
      * Excluded from normalization sums
      * Output has 0.0 for invalid bins (no NaNs present in output)
    - Each time slice normalized to sum to 1.0 over valid bins
    - Zero-sum rows remain all zeros; downstream returns inf (KL) or empty HPD
    """
    # Use validation utilities for consistent validation
    # This converts NaN/inf to 0 but keeps zeros that represent actual zero probability
    state, like = validate_paired_distributions(
        state_dist, likelihood, name1="state_dist", name2="likelihood", min_ndim=2
    )

    # Flatten for vectorized operations
    state_flat = flatten_time_spatial(state)
    like_flat = flatten_time_spatial(like)

    # Normalize each time slice
    # After validation, NaN/inf already converted to 0, so use regular sum
    # Shape: (n_time,)
    state_sum = state_flat.sum(axis=1)
    like_sum = like_flat.sum(axis=1)

    # Normalize, setting inf/nan results to 0
    # Division by zero is expected and handled, so suppress warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        state_norm_flat = state_flat / state_sum[:, np.newaxis]
        like_norm_flat = like_flat / like_sum[:, np.newaxis]

    # Replace non-finite values (from zero-sum rows) with 0
    state_norm_flat = np.nan_to_num(state_norm_flat, nan=0.0, posinf=0.0, neginf=0.0)
    like_norm_flat = np.nan_to_num(like_norm_flat, nan=0.0, posinf=0.0, neginf=0.0)

    # Reshape back to original shape
    state_norm = state_norm_flat.reshape(state.shape)
    like_norm = like_norm_flat.reshape(like.shape)

    return state_norm, like_norm


def kl_divergence(
    state_dist: DistributionArray, likelihood: DistributionArray
) -> DistributionArray:
    """Compute Kullback-Leibler divergence between state distribution and likelihood.

    Measures the information divergence between the state distribution and likelihood
    distributions at each time point. Large divergences may indicate issues
    with the prior specification or model assumptions.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, ...)
        State probability distributions over position at each time point where
        ... represents arbitrary spatial dimensions.
        Can be either one-step predictive distribution or smoother output.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
    likelihood : np.ndarray, shape (n_time, ...)
        Likelihood distributions at each time point. This is the
        likelihood p(y_t | x_t) across spatial positions.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Must have same shape as state_dist.

    Returns
    -------
    kl_divergence : np.ndarray, shape (n_time,)
        Kullback-Leibler divergence D_KL(state_dist || likelihood) at each
        time point. Values are non-negative, with 0 indicating identical
        distributions.

    Raises
    ------
    ValueError
        If state_dist and likelihood have different shapes, or if distributions
        contain negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import kl_divergence
    >>> # Identical distributions have zero divergence
    >>> state = np.array([[0.3, 0.4, 0.3]])
    >>> like = np.array([[0.3, 0.4, 0.3]])
    >>> div = kl_divergence(state, like)
    >>> div.shape
    (1,)
    >>> bool(np.isclose(div[0], 0.0))
    True

    See Also
    --------
    hpd_overlap : Compute spatial overlap between HPD regions
    highest_density_region : Compute highest density region mask

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

    # Initialize output with inf for invalid time slices
    kl_div: DistributionArray = np.full(n_time, np.inf, dtype=float)

    # Find valid time slices (both distributions have positive mass over valid bins)
    valid = (state_sum > 0) & (like_sum > 0)

    # Compute entropy for valid time slices
    # NaN already converted to 0 by validation
    if np.any(valid):
        kl_div[valid] = entropy(state_flat[valid], like_flat[valid], axis=1)

    # Clip to non-negative values to handle floating point precision errors
    # scipy.stats.entropy can return tiny negative values (~1e-113) with subnormal numbers
    # KL divergence is mathematically always non-negative, so clip spurious negatives to 0
    kl_div = np.maximum(kl_div, 0.0)

    return kl_div


def hpd_overlap(
    state_dist: DistributionArray,
    likelihood: DistributionArray,
    *,
    coverage: float = DEFAULT_COVERAGE,
) -> DistributionArray:
    """Compute overlap between HPD regions of state distribution and likelihood.

    Measures the spatial overlap between the highest posterior density regions
    of the state distribution and likelihood distributions. High overlap suggests
    consistency between the likelihood and prior contributions to the state estimate.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, ...)
        State probability distributions over position at each time point where
        ... represents arbitrary spatial dimensions.
        Can be either one-step predictive distribution or smoother output.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
    likelihood : np.ndarray, shape (n_time, ...)
        Likelihood distributions at each time point. This is the
        likelihood p(y_t | x_t) across spatial positions.
        Non-negative values (NaN allowed to mark invalid bins).
        Automatically normalized over valid (non-NaN) bins.
        Must have same shape as state_dist.
    coverage : float, optional
        Coverage probability for the HPD regions. Must be between 0 and 1.
        Default is 0.95 for 95% HPD regions.

    Returns
    -------
    hpd_overlap : np.ndarray, shape (n_time,)
        Proportion of overlap between the HPD regions of state_dist and
        likelihood at each time point. Values range from 0 (no overlap)
        to 1 (complete overlap).

    Raises
    ------
    ValueError
        If state_dist and likelihood have different shapes, if coverage
        is not in (0, 1), or if distributions contain negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import hpd_overlap
    >>> # Identical distributions have perfect overlap
    >>> state = np.array([[0.3, 0.4, 0.3]])
    >>> like = np.array([[0.3, 0.4, 0.3]])
    >>> overlap = hpd_overlap(state, like, coverage=0.9)
    >>> overlap.shape
    (1,)
    >>> bool(overlap[0] >= 0.0 and overlap[0] <= 1.0)
    True

    See Also
    --------
    kl_divergence : Measure information divergence between distributions
    highest_density_region : Compute highest density region mask

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

    # Validate but don't normalize - HPD works on relative magnitudes (unnormalized weights)
    # This saves 2 full array normalizations for large datasets
    state, like = validate_paired_distributions(
        state_dist, likelihood, name1="state_dist", name2="likelihood", min_ndim=2
    )

    # Get HPD regions (highest_density_region works on unnormalized weights)
    mask_state = highest_density_region(state, coverage=coverage)
    mask_like = highest_density_region(like, coverage=coverage)

    # Sum over all spatial dimensions (everything except time)
    spatial_axes = get_spatial_axes(state)
    size_state = mask_state.sum(axis=spatial_axes)
    size_like = mask_like.sum(axis=spatial_axes)
    intersection = (mask_state & mask_like).sum(axis=spatial_axes)

    # Compute denominator (minimum of the two sizes)
    denom = np.minimum(size_state, size_like)

    # Handle division by zero: when denom is 0, overlap is 0
    # This matches the normalization pattern used elsewhere in the codebase
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap: DistributionArray = intersection / denom
    overlap = np.nan_to_num(overlap, nan=0.0, posinf=0.0, neginf=0.0)

    return overlap
