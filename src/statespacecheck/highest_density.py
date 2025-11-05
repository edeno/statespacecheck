"""Functions for computing highest density regions."""

import numpy as np
from numpy.typing import NDArray

from ._validation import (
    DistributionArray,
    flatten_time_spatial,
    validate_coverage,
    validate_distribution,
)

# Default coverage probability for highest density regions
DEFAULT_COVERAGE = 0.95


def highest_density_region(
    distribution: DistributionArray, *, coverage: float = DEFAULT_COVERAGE
) -> NDArray[np.bool_]:
    """Compute boolean mask indicating highest density region membership.

    Vectorized HPD mask for arrays shaped (n_time, *spatial). For each time t,
    includes all bins with value >= threshold_t, where threshold_t is chosen so
    cumulative mass >= coverage * total_t.

    Parameters
    ----------
    distribution : np.ndarray, shape (n_time, ...)
        Probability distributions over position at each time point where
        ... represents arbitrary spatial dimensions.
    coverage : float, optional
        Desired coverage probability for the highest density region. Must be between 0 and 1.
        Default is 0.95 for 95% coverage.

    Returns
    -------
    isin_hd : np.ndarray, shape (n_time, ...)
        Boolean mask indicating which positions are in the highest density region at each
        time point, matching input shape.

    Raises
    ------
    ValueError
        If coverage is not in the range (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import highest_density_region
    >>> # Simple 1D example with peaked distribution
    >>> distribution = np.array([[0.1, 0.6, 0.3], [0.2, 0.5, 0.3]])
    >>> region = highest_density_region(distribution, coverage=0.9)
    >>> region.shape
    (2, 3)
    >>> region.dtype
    dtype('bool')

    See Also
    --------
    hpd_overlap : Compute overlap between HPD regions of two distributions
    kl_divergence : Measure information divergence between distributions

    Notes
    -----
    - NaNs are ignored (treated as 0 mass).
    - If total mass at time t <= 0 or not finite, returns all-False for that t.
    - Works in unnormalized space to avoid numerical issues.
    - Fully vectorized with no Python loops for efficiency.
    - Uses `>=` threshold: all bins with value equal to cutoff are included.
    - Due to ties, actual coverage may slightly exceed requested coverage.
    - This ensures consistent behavior across equivalent distributions.

    References
    ----------
    .. [1] https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    """
    validate_coverage(coverage)

    # Use centralized validation: handles NaN/inf → 0, checks non-negativity, validates dimensions
    clean = validate_distribution(
        distribution,
        name="distribution",
        min_ndim=2,  # Require at least (n_time, n_spatial)
        allow_nan=True,
    )

    # Flatten to (n_time, n_spatial) for vectorized operations
    flat = flatten_time_spatial(clean)

    n_time = clean.shape[0]
    n_spatial = flat.shape[1]

    # Compute total mass and target mass for each time point
    # Shape: (n_time,)
    totals = flat.sum(axis=1)
    target = coverage * totals

    # Identify rows with no mass -> empty HPD (all False)
    empty = ~np.isfinite(totals) | (totals <= 0)

    # Sort each row descending (vectorized)
    # Shape: (n_time, n_spatial)
    flat_sorted = np.sort(flat, axis=1)[:, ::-1]

    # Row-wise cumulative sums
    # Shape: (n_time, n_spatial)
    csum = np.cumsum(flat_sorted, axis=1)

    # Find the first index where cumulative >= target (per row)
    # Shape: (n_time, n_spatial) boolean
    ge = csum >= target[:, None]

    # Check if each row has at least one True value
    # Shape: (n_time,)
    has_true = ge.any(axis=1)

    # argmax gives first True index; if none True, returns 0 (we fix below)
    # Shape: (n_time,)
    idx = ge.argmax(axis=1)

    # If a row never reaches target but has positive mass (rare numeric case),
    # choose the last index. If it's truly empty, handle later.
    idx = np.where(has_true, idx, n_spatial - 1)

    # Per-row cutoff (unnormalized)
    # Shape: (n_time,)
    cutoff = np.take_along_axis(flat_sorted, idx[:, None], axis=1).squeeze(1)

    # Empty rows -> set cutoff to +inf so mask is all False
    cutoff = np.where(empty, np.inf, cutoff)

    # Broadcast cutoff back to spatial shape and build mask
    # Use the **clean** array for the comparison to keep behavior consistent
    # Broadcasting: reshape cutoff from (n_time,) to (n_time, 1, 1, ...) to match spatial dims
    # Using tuple unpacking for clarity
    broadcast_shape = (n_time,) + (1,) * (clean.ndim - 1)
    return clean >= cutoff.reshape(broadcast_shape)
