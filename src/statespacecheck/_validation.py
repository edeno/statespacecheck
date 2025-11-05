"""Validation utilities for distributions and parameters."""

import numpy as np
from numpy.typing import NDArray


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
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")


def validate_distribution(
    distribution: NDArray[np.floating],
    name: str = "distribution",
    min_ndim: int = 1,
    allow_nan: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Validate and prepare distribution array.

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
    flat : np.ndarray
        Flattened to (n_time, n_spatial)

    Raises
    ------
    ValueError
        If validation fails
    """
    arr = np.asarray(distribution, dtype=float)

    if arr.ndim < min_ndim:
        raise ValueError(
            f"{name} must be at least {min_ndim}D with shape (n_time, ...), got shape {arr.shape}"
        )

    n_time = arr.shape[0]
    n_spatial = int(np.prod(arr.shape[1:], dtype=np.int64))

    if n_spatial <= 0 and len(arr.shape) > 1:
        raise ValueError(
            f"Spatial dimensions too large or invalid: {arr.shape[1:]}. "
            f"Product of spatial dimensions must be positive and fit in int64."
        )

    # Handle non-finite values
    if allow_nan:
        # Use standard NumPy idiom: convert NaN/inf to 0
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        clean = arr.copy()
        if not np.all(np.isfinite(clean)):
            raise ValueError(f"{name} contains non-finite values (NaN or inf).")

    # Check for negative values
    finite_mask = np.isfinite(arr)
    if np.any(clean[finite_mask] < 0):
        raise ValueError(f"{name} must be non-negative (probability or weight).")

    flat = clean.reshape(n_time, n_spatial)

    return clean, flat


def validate_paired_distributions(
    dist1: NDArray[np.floating],
    dist2: NDArray[np.floating],
    name1: str = "state_dist",
    name2: str = "likelihood",
    min_ndim: int = 2,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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
    flat1 : np.ndarray
        First distribution, flattened to (n_time, n_spatial)
    clean2 : np.ndarray
        Second distribution, cleaned
    flat2 : np.ndarray
        Second distribution, flattened to (n_time, n_spatial)

    Raises
    ------
    ValueError
        If shapes don't match or validation fails
    """
    clean1, flat1 = validate_distribution(dist1, name1, min_ndim=min_ndim)
    clean2, flat2 = validate_distribution(dist2, name2, min_ndim=min_ndim)

    if clean1.shape != clean2.shape:
        raise ValueError(
            f"{name1} and {name2} must have same shape, got {clean1.shape} vs {clean2.shape}"
        )

    return clean1, flat1, clean2, flat2


def get_spatial_axes(arr: NDArray[np.floating]) -> tuple[int, ...]:
    """Get tuple of spatial dimension axes (all except time axis 0).

    Parameters
    ----------
    arr : np.ndarray
        Array with shape (n_time, ...) where ... are spatial dimensions

    Returns
    -------
    spatial_axes : tuple[int, ...]
        Tuple of axis indices for spatial dimensions
    """
    return tuple(range(1, arr.ndim))
