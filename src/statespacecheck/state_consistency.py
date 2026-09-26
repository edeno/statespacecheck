"""Compare a state distribution with a likelihood: HPD overlap and KL divergence.

Each row (a time bin, or an event) pairs a state distribution, such as the
one-step predictive distribution, with a likelihood over the same states.
:func:`hpd_overlap` asks whether the two are consistent (their high-probability
regions overlap); :func:`kl_divergence` measures how different they are.

The paper applies both to each spike, with the spike's single-event likelihood;
:func:`~statespacecheck.event_diagnostics` does this. Applying them to whole time
bins with a whole-bin likelihood, which also includes the Poisson exposure term
and silent units, is an extension beyond the paper.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy

from ._validation import (
    DistributionArray,
    flatten_time_spatial,
    get_spatial_axes,
    rescale_subnormal_rows,
    row_chunks,
    validate_coverage,
    validate_paired_distributions,
)
from .highest_density import DEFAULT_COVERAGE, highest_density_region


def _exclude_bins_invalid_in_either(
    state_dist: DistributionArray, likelihood: DistributionArray
) -> tuple[DistributionArray, DistributionArray]:
    """Mark a bin NaN in both arrays when it is non-finite in either.

    Both distributions are then normalized over, and compared on, the same
    set of valid bins. Arrays of different shapes are returned unchanged for
    the shape check to report.
    """
    state = np.asarray(state_dist, dtype=float)
    like = np.asarray(likelihood, dtype=float)
    if state.shape == like.shape:
        invalid = ~(np.isfinite(state) & np.isfinite(like))
        if invalid.any():
            state = np.where(invalid, np.nan, state)
            like = np.where(invalid, np.nan, like)
    return state, like


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
    # A bin invalid in either input is excluded from both; validation then
    # converts NaN to 0 but keeps zeros that represent actual zero probability.
    state, like = validate_paired_distributions(
        *_exclude_bins_invalid_in_either(state_dist, likelihood),
        name1="state_dist",
        name2="likelihood",
        min_ndim=2,
    )

    # Flatten for vectorized operations
    state_flat = flatten_time_spatial(state)
    like_flat = flatten_time_spatial(like)

    # Normalize each time slice
    # After validation, NaN/inf already converted to 0, so use regular sum
    # Shape: (n_time,)
    state_sum = state_flat.sum(axis=1)
    like_sum = like_flat.sum(axis=1)
    state_flat, state_sum = rescale_subnormal_rows(state_flat, state_sum)
    like_flat, like_sum = rescale_subnormal_rows(like_flat, like_sum)

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


def kl_divergence(state_dist: ArrayLike, likelihood: ArrayLike) -> DistributionArray:
    """Compute Kullback-Leibler divergence between state distribution and likelihood.

    Measures how different the likelihood is from the state distribution at each
    time point, D(state_dist || likelihood). The divergence is large when the two
    put their mass in different places, but also when the state distribution is
    broad relative to a consistent likelihood, so the paper uses it as a reference
    alongside :func:`hpd_overlap` and the predictive p-value.

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
    >>> # Identical distributions have zero divergence; the likelihood in the
    >>> # second time bin puts its mass where the state distribution does not
    >>> state = np.array([[0.3, 0.4, 0.3], [0.3, 0.4, 0.3]])
    >>> like = np.array([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
    >>> kl_divergence(state, like).round(3)
    array([0.   , 0.353])

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
    NaN values mark invalid spatial bins (e.g., inaccessible locations); a bin
    that is NaN (or infinite) in either input is excluded from both, for
    normalization and for the divergence.

    Time slices where distributions have no valid mass return inf for the divergence.

    """
    state, like = _as_paired_arrays(state_dist, likelihood)
    divergence: DistributionArray = np.empty(state.shape[0])
    for rows in row_chunks(state.shape):
        divergence[rows] = _kl_divergence_rows(state[rows], like[rows])
    return divergence


def hpd_overlap(
    state_dist: ArrayLike,
    likelihood: ArrayLike,
    *,
    coverage: float = DEFAULT_COVERAGE,
) -> DistributionArray:
    """Compute overlap between HPD regions of state distribution and likelihood.

    Measures the overlap between the highest probability-density (HPD) regions of
    the state distribution and the likelihood, as a fraction of the smaller region
    (the Szymkiewicz-Simpson overlap coefficient). It is 1 when one region lies
    inside the other, so a broad prediction and a precise, consistent likelihood
    score 1, and 0 when the regions are disjoint.

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
    >>> # 80% HPD regions: bins {0, 1} for the state distribution and {1, 2}
    >>> # for the likelihood share one bin out of the smaller region's two
    >>> state = np.array([[0.4, 0.4, 0.2, 0.0, 0.0]])
    >>> like = np.array([[0.0, 0.4, 0.4, 0.2, 0.0]])
    >>> hpd_overlap(state, like, coverage=0.8)
    array([0.5])

    See Also
    --------
    kl_divergence : Measure information divergence between distributions
    highest_density_region : Compute highest density region mask

    Notes
    -----
    The overlap is computed as:
        overlap = intersection(HPD_state, HPD_like) / min(size(HPD_state), size(HPD_like))

    where a region's size is its number of bins. This equals the paper's
    region volume when all bins have the same volume; on a nonuniform grid,
    resample to a uniform one first.

    This normalization ensures that:
    - overlap = 1.0 when one region completely contains the other
    - overlap = 0.0 when regions don't overlap at all
    - Values are comparable even when HPD regions have different sizes

    When either HPD region is empty (a row with no probability mass), the
    denominator is 0 and overlap is defined as 0, so such rows read as
    disagreement. Check for all-zero rows separately if they can occur.

    Distributions are automatically normalized over valid (non-NaN) bins.
    NaN values mark invalid spatial bins (e.g., inaccessible locations); a bin
    that is NaN (or infinite) in either input is excluded from both HPD regions.

    """
    validate_coverage(coverage)
    state, like = _as_paired_arrays(state_dist, likelihood)
    overlap: DistributionArray = np.empty(state.shape[0])
    for rows in row_chunks(state.shape):
        overlap[rows] = _hpd_overlap_rows(state[rows], like[rows], coverage)
    return overlap


def _as_paired_arrays(
    state_dist: ArrayLike, likelihood: ArrayLike
) -> tuple[DistributionArray, DistributionArray]:
    """Return both inputs as float arrays, raising if their shapes cannot pair."""
    state = np.asarray(state_dist, dtype=float)
    like = np.asarray(likelihood, dtype=float)
    if state.ndim < 2 or state.shape != like.shape:
        # Raise the usual error, which reports both full shapes
        validate_paired_distributions(
            state, like, name1="state_dist", name2="likelihood", min_ndim=2
        )
    return state, like


def _kl_divergence_rows(
    state_dist: DistributionArray, likelihood: DistributionArray
) -> DistributionArray:
    """Compute :func:`kl_divergence` for one chunk of time points."""
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
    return np.maximum(kl_div, 0.0)


def _hpd_overlap_rows(
    state_dist: DistributionArray, likelihood: DistributionArray, coverage: float
) -> DistributionArray:
    """Compute :func:`hpd_overlap` for one chunk of time points."""
    # Validate but don't normalize - HPD works on relative magnitudes (unnormalized
    # weights). A bin invalid in either input is excluded from both.
    state, like = validate_paired_distributions(
        *_exclude_bins_invalid_in_either(state_dist, likelihood),
        name1="state_dist",
        name2="likelihood",
        min_ndim=2,
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

    # An empty region on either side makes denom 0; overlap is then defined as 0
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap: DistributionArray = intersection / denom
    return np.nan_to_num(overlap, nan=0.0, posinf=0.0, neginf=0.0)
