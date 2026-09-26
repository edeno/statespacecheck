"""Test data generators shared by the test modules."""

import numpy as np
from scipy.stats import norm

from statespacecheck import MarkModel


def make_random_distribution_1d(
    rng: np.random.Generator, n_time: int, n_bins: int
) -> np.ndarray:
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


def unit_sampler(rates: np.ndarray):
    """A mark sampler that draws, for each state bin, a unit in proportion to its rate.

    ``rates`` has shape ``(n_bins, n_units)``; the sampler takes flat state-bin
    indices ``(n,)`` and a generator and returns unit indices ``(n,)``.
    """
    cumulative = np.cumsum(rates / rates.sum(axis=1, keepdims=True), axis=1)

    def sample(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(len(bins))
        # Inverse CDF: the first unit whose cumulative probability exceeds u
        return np.minimum((cumulative[bins] <= u[:, None]).sum(axis=1), rates.shape[1] - 1)

    return sample


def integer_mark_model(rates: np.ndarray) -> MarkModel:
    """The MarkModel of integer marks (units) with rates ``(n_bins, n_units)``."""

    def log_intensity(marks: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):  # zero rates are impossible marks
            return np.log(rates[:, np.asarray(marks)].T)

    return MarkModel(log_intensity, unit_sampler(rates), rates.sum(axis=1))


def gaussian_mark_model(
    place_fields: np.ndarray, waveform_means: np.ndarray, sigma: float
) -> MarkModel:
    """Units with place fields ``(n_bins, n_units)`` and Gaussian 1-D waveform amplitudes.

    ``lambda(x, y) = sum_u r_u(x) N(y; mu_u, sigma)``, so the ground intensity is
    ``sum_u r_u(x)``. Marks have shape ``(n, 1)``.
    """
    sample_unit = unit_sampler(place_fields)

    def log_mark_intensity(marks: np.ndarray) -> np.ndarray:
        # log sum_u r_u(x) N(y; mu_u, sigma), shape (n, n_bins), with the largest
        # log N(y; mu_u, sigma) factored out so the sum over units is a stable
        # matrix product (every r_u > 0, so the sum is positive)
        log_amplitude = norm.logpdf(np.asarray(marks)[:, :1], waveform_means, sigma)
        largest = log_amplitude.max(axis=1, keepdims=True)
        return largest + np.log(np.exp(log_amplitude - largest) @ place_fields.T)

    def sample_marks(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(waveform_means[sample_unit(bins, rng)], sigma)[:, None]

    return MarkModel(log_mark_intensity, sample_marks, place_fields.sum(axis=1))
