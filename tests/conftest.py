"""Pytest fixtures and test utilities."""

import matplotlib

# Use non-interactive backend for testing
# This prevents TclError on Windows CI and allows tests to run headlessly
matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Shared random number generator with fixed seed for reproducible tests."""
    return np.random.default_rng(seed=42)


# Test data generation helpers


def make_random_distribution_1d(rng: np.random.Generator, n_time: int, n_bins: int) -> np.ndarray:
    """Create random 1D distributions using Dirichlet.

    Returns array of shape (n_time, n_bins) where each time slice
    is a valid probability distribution (sums to 1).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n_time : int
        Number of time steps
    n_bins : int
        Number of spatial bins

    Returns
    -------
    distribution : np.ndarray
        Shape (n_time, n_bins)
    """
    return rng.dirichlet(np.ones(n_bins), size=n_time)


def make_random_distribution_2d(
    rng: np.random.Generator, n_time: int, n_x: int, n_y: int
) -> np.ndarray:
    """Create random 2D spatial distributions using Dirichlet.

    Returns array of shape (n_time, n_x, n_y) where each time slice
    is a valid probability distribution (sums to 1).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n_time : int
        Number of time steps
    n_x : int
        Number of bins in x dimension
    n_y : int
        Number of bins in y dimension

    Returns
    -------
    distribution : np.ndarray
        Shape (n_time, n_x, n_y)
    """
    n_bins = n_x * n_y
    return rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)


def make_gaussian_1d(n_time: int, n_bins: int, mean: float, std: float) -> np.ndarray:
    """Create 1D Gaussian-like distributions.

    Parameters
    ----------
    n_time : int
        Number of time steps
    n_bins : int
        Number of spatial bins
    mean : float
        Center of Gaussian
    std : float
        Standard deviation of Gaussian

    Returns
    -------
    distribution : np.ndarray
        Shape (n_time, n_bins), each row is same Gaussian
    """
    x = np.arange(n_bins)
    dist = np.exp(-((x - mean) ** 2) / (2 * std**2))
    dist = dist / dist.sum()
    return np.tile(dist, (n_time, 1))


def make_gaussian_2d(
    n_time: int, n_x: int, n_y: int, mean_x: float, mean_y: float, std: float
) -> np.ndarray:
    """Create 2D Gaussian-like distributions.

    Parameters
    ----------
    n_time : int
        Number of time steps
    n_x : int
        Number of bins in x dimension
    n_y : int
        Number of bins in y dimension
    mean_x : float
        Center in x dimension
    mean_y : float
        Center in y dimension
    std : float
        Standard deviation (isotropic)

    Returns
    -------
    distribution : np.ndarray
        Shape (n_time, n_x, n_y), each slice is same 2D Gaussian
    """
    x = np.arange(n_x)
    y = np.arange(n_y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dist = np.exp(-(((xx - mean_x) ** 2 + (yy - mean_y) ** 2) / (2 * std**2)))
    dist = dist / dist.sum()
    return np.tile(dist, (n_time, 1, 1))


def make_bimodal_gaussian_1d(
    n_time: int,
    n_bins: int,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    weight1: float = 0.5,
) -> np.ndarray:
    """Create bimodal Gaussian mixture distributions.

    Parameters
    ----------
    n_time : int
        Number of time steps
    n_bins : int
        Number of spatial bins
    mean1 : float
        Center of first Gaussian
    std1 : float
        Standard deviation of first Gaussian
    mean2 : float
        Center of second Gaussian
    std2 : float
        Standard deviation of second Gaussian
    weight1 : float
        Weight of first Gaussian (0 to 1)

    Returns
    -------
    distribution : np.ndarray
        Shape (n_time, n_bins), each row is same bimodal distribution
    """
    x = np.arange(n_bins)
    g1 = np.exp(-((x - mean1) ** 2) / (2 * std1**2))
    g2 = np.exp(-((x - mean2) ** 2) / (2 * std2**2))
    dist = weight1 * g1 + (1 - weight1) * g2
    dist = dist / dist.sum()
    return np.tile(dist, (n_time, 1))


# Test assertion helpers


def assert_shape_matches(
    actual: np.ndarray, expected: np.ndarray, message: str = "Shape mismatch"
) -> None:
    """Assert that actual array has same shape as expected."""
    assert actual.shape == expected.shape, (
        f"{message}: expected shape {expected.shape}, got {actual.shape}"
    )


def assert_time_vector(
    actual: np.ndarray, n_time: int, message: str = "Expected time vector"
) -> None:
    """Assert that array is 1D with length n_time."""
    assert actual.shape == (n_time,), f"{message}: expected shape ({n_time},), got {actual.shape}"


def sum_over_spatial(arr: np.ndarray) -> np.ndarray:
    """Sum over all spatial dimensions, keeping time dimension.

    Parameters
    ----------
    arr : np.ndarray
        Array with shape (n_time, ...) where ... are spatial dimensions

    Returns
    -------
    summed : np.ndarray
        Array with shape (n_time,) summed over all spatial dimensions
    """
    spatial_axes = tuple(range(1, arr.ndim))
    return arr.sum(axis=spatial_axes)
