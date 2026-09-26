"""Pytest fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm


@pytest.fixture
def rng():
    """Shared random number generator with fixed seed for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def discrete_mark_model():
    """Sorted-spike model written as a marked point process with integer marks.

    Returns ``(rates, mark_intensity, sample_marks)``: place fields of 12 units
    on 50 positions, shape ``(n_bins, n_marks)``; the joint intensity of integer
    marks ``(n,)`` at every position, shape ``(n, n_bins)``; and a sampler of
    each event's unit given its position bin.
    """
    x = np.linspace(0.0, 1.0, 50)
    centers = np.random.default_rng(0).random(12)
    rates = 0.2 + 10.0 * np.exp(-0.5 * ((x[:, None] - centers) / 0.1) ** 2)
    cumulative = np.cumsum(rates / rates.sum(axis=1, keepdims=True), axis=1)

    def mark_intensity(marks):
        return rates[:, np.asarray(marks)].T

    def sample_marks(bins, rng):
        u = rng.random(len(bins))
        # Inverse CDF: the first unit whose cumulative probability exceeds u
        return np.minimum((cumulative[bins] <= u[:, None]).sum(axis=1), rates.shape[1] - 1)

    return rates, mark_intensity, sample_marks


@pytest.fixture(scope="session")
def clusterless_1d_model():
    """Clusterless model: 6 units on a 1-D track, each with a Gaussian waveform amplitude.

    ``lambda(x, y) = sum_u r_u(x) N(y; mu_u, sigma)``, so the ground intensity is
    ``sum_u r_u(x)``. Marks have shape ``(n, 1)``.
    """
    position = np.linspace(0.0, 1.0, 60)
    centers = np.linspace(0.1, 0.9, 6)
    place_fields = 0.5 + 20.0 * np.exp(-0.5 * ((position[:, None] - centers) / 0.12) ** 2)
    waveform_means = np.linspace(1.0, 4.0, 6)
    sigma = 0.3
    ground_intensity = place_fields.sum(axis=1)
    cumulative = np.cumsum(place_fields / ground_intensity[:, None], axis=1)

    def mark_intensity(marks):
        amplitude_density = norm.pdf(np.asarray(marks)[:, :1], waveform_means, sigma)
        return amplitude_density @ place_fields.T

    def sample_marks(bins, rng):
        u = rng.random(len(bins))
        unit = np.minimum((cumulative[bins] <= u[:, None]).sum(axis=1), len(centers) - 1)
        return rng.normal(waveform_means[unit], sigma)[:, None]

    return SimpleNamespace(
        position=position,
        place_fields=place_fields,
        waveform_means=waveform_means,
        sigma=sigma,
        ground_intensity=ground_intensity,
        mark_intensity=mark_intensity,
        sample_marks=sample_marks,
    )
