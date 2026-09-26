"""Validation utilities for distributions and parameters."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Arrays of float64 values the package returns (inputs may be any array-like)
DistributionArray = NDArray[np.float64]

# The time-bin functions process this many array elements at a time, which
# bounds their temporary arrays (about 32 MB of float64 each) however long the
# recording. Every operation is row-wise, so results do not depend on it.
_CHUNK_ELEMENTS = 2**22


def row_chunks(shape: tuple[int, ...]) -> Iterator[slice]:
    """Yield slices over the first (time) axis covering about ``_CHUNK_ELEMENTS`` each."""
    n_rows = shape[0]
    row_size = max(1, int(np.prod(shape[1:])))
    step = max(1, _CHUNK_ELEMENTS // row_size)
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))


def check_threshold_not_nan(value: float, name: str) -> None:
    """Raise if a flag threshold is NaN, which would silently flag nothing.

    Parameters
    ----------
    value : float
        The threshold.
    name : str
        The argument's name, for the error message.

    Raises
    ------
    ValueError
        If ``value`` is NaN.
    """
    if np.isnan(value):
        msg = (
            f"{name} is NaN, which would flag nothing; a quantile of values that "
            "include +inf is NaN, so compute baseline thresholds with baseline_threshold"
        )
        raise ValueError(msg)


def rescale_subnormal_rows(
    flat: DistributionArray, row_sums: DistributionArray
) -> tuple[DistributionArray, DistributionArray]:
    """Scale rows whose total is subnormal by 2**1000 before they are normalized.

    Dividing by a subnormal total raises a spurious "overflow encountered in
    divide" RuntimeWarning on NumPy 1.26. Scaling by a power of two is exact, so
    the normalized row is unchanged; other rows are not touched.

    Parameters
    ----------
    flat : np.ndarray, shape (n_rows, n_bins)
        Nonnegative values.
    row_sums : np.ndarray, shape (n_rows,)
        ``flat.sum(axis=1)``.

    Returns
    -------
    flat, row_sums : np.ndarray
        The inputs, with subnormal rows (and their sums) rescaled.
    """
    subnormal = (row_sums > 0.0) & (row_sums < np.finfo(np.float64).tiny)
    if not subnormal.any():
        return flat, row_sums
    flat = flat.copy()
    flat[subnormal] *= 2.0**1000
    row_sums = row_sums.copy()
    row_sums[subnormal] = flat[subnormal].sum(axis=1)
    return flat, row_sums


def validate_coverage(coverage: float) -> None:
    """Validate that coverage is in the valid range (0, 1).

    Parameters
    ----------
    coverage : float
        Coverage value to validate

    Raises
    ------
    ValueError
        If coverage is not in (0, 1)
    """
    if not (0.0 < coverage < 1.0):
        msg = (
            f"coverage must be in (0, 1), got {coverage}. "
            f"Coverage represents the probability mass of the highest density region "
            f"and must be a value between 0 and 1 (exclusive). "
            f"For example, use 0.95 for a 95% credible region."
        )
        raise ValueError(msg)


def validate_distribution(
    distribution: ArrayLike,
    name: str = "distribution",
    min_ndim: int = 1,
    allow_nan: bool = True,
) -> DistributionArray:
    """Validate and clean distribution array.

    Parameters
    ----------
    distribution : np.ndarray
        Distribution to validate
    name : str
        Name for error messages
    min_ndim : int
        Minimum number of dimensions required
    allow_nan : bool
        Whether to allow NaN values (converted to 0 if True)

    Returns
    -------
    clean : np.ndarray
        Original shape array with NaN/inf converted to 0 if allow_nan=True

    Raises
    ------
    ValueError
        If validation fails
    """
    arr = np.asarray(distribution, dtype=float)

    if arr.ndim < min_ndim:
        if min_ndim == 1:
            expected_shape = "(n_time,)"
        elif min_ndim == 2:
            expected_shape = "(n_time, n_position)"
        else:
            expected_shape = f"{min_ndim}D"
        msg = (
            f"{name} must be at least {min_ndim}D with shape {expected_shape}, "
            f"got shape {arr.shape}. "
            f"State space diagnostics require time-series data where the first "
            f"dimension is time. "
            f"For 1D spatial data use shape (n_time, n_position_bins), "
            f"for 2D spatial data use shape (n_time, n_x_bins, n_y_bins). "
            f"Did you forget to add the time dimension?"
        )
        raise ValueError(msg)

    # Handle non-finite values
    clean: DistributionArray
    if allow_nan:
        # Use standard NumPy idiom: convert NaN/inf to 0
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        clean = arr.copy()
        if not np.all(np.isfinite(clean)):
            msg = (
                f"{name} contains non-finite values (NaN or inf). "
                f"Probability distributions must have finite values. "
                f"If you have invalid spatial bins (e.g., inaccessible locations), "
                f"consider setting them to 0 instead of NaN, or ensure "
                f"allow_nan=True in the validation."
            )
            raise ValueError(msg)

    # Check for negative values
    finite_mask = np.isfinite(arr)
    if np.any(clean[finite_mask] < 0):
        msg = (
            f"{name} must be non-negative (probability or weight). "
            f"Found negative values in the distribution. "
            f"Probability distributions and weights must be >= 0. "
            f"Check your data for errors or ensure proper normalization."
        )
        raise ValueError(msg)

    return clean


def flatten_time_spatial(arr: DistributionArray) -> DistributionArray:
    """Flatten array to (n_time, n_spatial) shape.

    Parameters
    ----------
    arr : np.ndarray, shape (n_time, ...)
        Array where ... represents arbitrary spatial dimensions.

    Returns
    -------
    flat : np.ndarray, shape (n_time, n_spatial)
        Flattened array.
    """
    n_time = arr.shape[0]
    # Use numpy's automatic dimension calculation with -1
    return arr.reshape(n_time, -1)


def validate_paired_distributions(
    dist1: ArrayLike,
    dist2: ArrayLike,
    name1: str = "state_dist",
    name2: str = "likelihood",
    min_ndim: int = 2,
) -> tuple[DistributionArray, DistributionArray]:
    """Validate two distributions have matching shapes.

    Parameters
    ----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    name1 : str
        Name for first distribution (error messages)
    name2 : str
        Name for second distribution (error messages)
    min_ndim : int
        Minimum number of dimensions required

    Returns
    -------
    clean1 : np.ndarray
        First distribution, cleaned
    clean2 : np.ndarray
        Second distribution, cleaned

    Raises
    ------
    ValueError
        If shapes don't match or validation fails
    """
    clean1 = validate_distribution(dist1, name1, min_ndim=min_ndim)
    clean2 = validate_distribution(dist2, name2, min_ndim=min_ndim)

    if clean1.shape != clean2.shape:
        msg = (
            f"{name1} and {name2} must have same shape, got {clean1.shape} vs {clean2.shape}. "
            f"Both distributions must cover the same time points and spatial bins. "
            f"Common causes: different spatial discretization, mismatched time periods, "
            f"or one distribution missing time/spatial dimensions. "
            f"Ensure both arrays use consistent binning and time indexing."
        )
        raise ValueError(msg)

    return clean1, clean2


def get_spatial_axes(arr: NDArray[Any]) -> tuple[int, ...]:
    """Get tuple of spatial dimension axes (all except time axis 0).

    Parameters
    ----------
    arr : np.ndarray, shape (n_time, ...)
        Array where ... are spatial dimensions.

    Returns
    -------
    spatial_axes : tuple[int, ...]
        Tuple of axis indices for spatial dimensions.
    """
    return tuple(range(1, arr.ndim))
