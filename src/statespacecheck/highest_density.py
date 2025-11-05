"""Functions for computing highest density regions."""

import numpy as np

from ._validation import validate_coverage, validate_distribution


def highest_density_region(distribution: np.ndarray, *, coverage: float = 0.95) -> np.ndarray:
    """Compute boolean mask indicating highest density region membership.

    Vectorized HPD mask for arrays shaped (n_time, *spatial). For each time t,
    includes all bins with value >= threshold_t, where threshold_t is chosen so
    cumulative mass >= coverage * total_t.

    Parameters
    ----------
    distribution : np.ndarray
        Probability distributions over position at each time point.
        Shape (n_time, ...) where ... represents arbitrary spatial dimensions.
    coverage : float, optional
        Desired coverage probability for the highest density region. Must be between 0 and 1.
        Default is 0.95 for 95% coverage.

    Returns
    -------
    isin_hd : np.ndarray
        Boolean mask indicating which positions are in the highest density region at each
        time point. Shape (n_time, ...) matching input shape.

    Raises
    ------
    ValueError
        If coverage is not in the range (0, 1).

    Notes
    -----
    - NaNs are ignored (treated as 0 mass).
    - If total mass at time t <= 0 or not finite, returns all-False for that t.
    - Works in unnormalized space to avoid numerical issues.
    - Fully vectorized with no Python loops for efficiency.

    For reference see: https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    """
    validate_coverage(coverage)

    # Use centralized validation: handles NaN/inf → 0, checks non-negativity, validates dimensions
    clean, flat = validate_distribution(
        distribution,
        name="distribution",
        min_ndim=2,  # Require at least (n_time, n_spatial)
        allow_nan=True,
    )

    n_time = clean.shape[0]
    n_spatial = flat.shape[1]

    totals = flat.sum(axis=1)  # (n_time,)
    target = coverage * totals  # (n_time,)

    # Rows with no mass -> empty HPD (all False)
    empty = ~np.isfinite(totals) | (totals <= 0)

    # Sort each row descending (vectorized)
    flat_sorted = np.sort(flat, axis=1)[:, ::-1]  # (n_time, n_spatial)

    # Row-wise cumulative sums
    csum = np.cumsum(flat_sorted, axis=1)  # (n_time, n_spatial)

    # Find the first index where cumulative >= target (per row)
    ge = csum >= target[:, None]  # (n_time, n_spatial) boolean
    has_true = ge.any(axis=1)  # (n_time,)

    # argmax gives first True index; if none True, returns 0 (we fix below)
    idx = ge.argmax(axis=1)  # (n_time,)

    # If a row never reaches target but has positive mass (rare numeric case),
    # choose the last index. If it's truly empty, handle later.
    idx = np.where(has_true, idx, n_spatial - 1)

    # Per-row cutoff (unnormalized)
    cutoff = np.take_along_axis(flat_sorted, idx[:, None], axis=1).squeeze(1)  # (n_time,)

    # Empty rows -> set cutoff to +inf so mask is all False
    cutoff = np.where(empty, np.inf, cutoff)

    # Broadcast cutoff back to spatial shape and build mask
    # Use the **clean** array for the comparison to keep behavior consistent
    reshape = (n_time,) + (1,) * (clean.ndim - 1)
    return clean >= cutoff.reshape(reshape)
