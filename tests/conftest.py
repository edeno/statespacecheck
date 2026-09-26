"""Pytest fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers import integer_mark_model, unit_sampler
from scipy.stats import norm

from statespacecheck import MarkModel


@pytest.fixture
def rng():
    """Shared random number generator with fixed seed for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def discrete_mark_model():
    """Sorted-spike model written as a marked point process with integer marks.

    Returns ``(rates, model)``: place fields of 12 units on 50 positions, shape
    ``(n_bins, n_marks)``, and the :class:`MarkModel` of integer marks ``(n,)``:
    their log joint intensity at every position, shape ``(n, n_bins)``, a
    sampler of each event's unit given its position bin, and the total rate.
    """
    x = np.linspace(0.0, 1.0, 50)
    centers = np.random.default_rng(0).random(12)
    rates = 0.2 + 10.0 * np.exp(-0.5 * ((x[:, None] - centers) / 0.1) ** 2)
    return rates, integer_mark_model(rates)


@pytest.fixture(scope="session")
def clusterless_1d_model():
    """Clusterless model: 6 units on a 1-D track, each with a Gaussian waveform amplitude.

    ``lambda(x, y) = sum_u r_u(x) N(y; mu_u, sigma)``, so the ground intensity is
    ``sum_u r_u(x)``. Marks have shape ``(n, 1)``. ``model`` is its
    :class:`MarkModel`.
    """
    position = np.linspace(0.0, 1.0, 60)
    centers = np.linspace(0.1, 0.9, 6)
    place_fields = 0.5 + 20.0 * np.exp(-0.5 * ((position[:, None] - centers) / 0.12) ** 2)
    waveform_means = np.linspace(1.0, 4.0, 6)
    sigma = 0.3
    ground_intensity = place_fields.sum(axis=1)
    sample_unit = unit_sampler(place_fields)

    def log_mark_intensity(marks):
        # log sum_u r_u(x) N(y; mu_u, sigma), shape (n, n_bins), with the largest
        # log N(y; mu_u, sigma) factored out so the sum over units is a stable
        # matrix product (every r_u > 0, so the sum is positive)
        log_amplitude = norm.logpdf(np.asarray(marks)[:, :1], waveform_means, sigma)
        largest = log_amplitude.max(axis=1, keepdims=True)
        return largest + np.log(np.exp(log_amplitude - largest) @ place_fields.T)

    def sample_marks(bins, rng):
        return rng.normal(waveform_means[sample_unit(bins, rng)], sigma)[:, None]

    return SimpleNamespace(
        position=position,
        place_fields=place_fields,
        waveform_means=waveform_means,
        sigma=sigma,
        model=MarkModel(log_mark_intensity, sample_marks, ground_intensity),
    )
