"""Pytest fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers import gaussian_mark_model, integer_mark_model, markov_trajectory, spike_events
from scipy.stats import norm


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
    return SimpleNamespace(
        position=position,
        place_fields=place_fields,
        waveform_means=waveform_means,
        sigma=sigma,
        model=gaussian_mark_model(place_fields, waveform_means, sigma),
    )


@pytest.fixture(scope="session")
def clusterless_session(clusterless_1d_model):
    """A simulated recording of ``clusterless_1d_model``, decoded by an exact grid filter.

    The position bin follows a Markov chain with a Gaussian random-walk transition;
    each unit fires Poisson counts with mean ``r_u(x) dt`` and each spike's mark is
    drawn from its unit's waveform Gaussian. The filter uses the same transition
    and the clusterless Poisson likelihood
    ``exp(-Lambda(x) dt) prod_j lambda(x, y_j) dt``, once with the true model and
    once with every waveform mean shifted by +0.8 (``misspecified_model``).

    Attributes: ``predictive`` and ``misspecified_predictive`` ``(n_time, n_bins)``,
    ``event_time_ind`` ``(n_events,)``, ``event_marks`` ``(n_events, 1)``, the unit
    that fired each event ``event_unit`` ``(n_events,)``, ``model`` and
    ``misspecified_model``.
    """
    clusterless = clusterless_1d_model
    rng = np.random.default_rng(20260925)
    n_time, dt = 2000, 0.01
    position, place_fields = clusterless.position, clusterless.place_fields
    n_bins = position.size

    # transition[i, j] = P(bin i at t | bin j at t - 1)
    transition = norm.pdf(position[:, None], position[None, :], 0.03)
    transition /= transition.sum(axis=0, keepdims=True)
    state_bin = markov_trajectory(rng, transition, n_time)

    counts = rng.poisson(place_fields[state_bin] * dt)  # (n_time, n_units)
    event_time_ind, event_unit = spike_events(counts)
    event_marks = rng.normal(clusterless.waveform_means[event_unit], clusterless.sigma)[
        :, None
    ]

    def decode(model):
        """One-step predictive distributions of a log-space grid filter."""
        log_likelihood = np.tile(-model.ground_intensity * dt, (n_time, 1))
        np.add.at(log_likelihood, event_time_ind, model.log_intensity(event_marks))
        predictive = np.empty((n_time, n_bins))
        posterior = np.full(n_bins, 1.0 / n_bins)
        for t in range(n_time):
            predictive[t] = transition @ posterior
            log_posterior = np.log(predictive[t]) + log_likelihood[t]
            posterior = np.exp(log_posterior - log_posterior.max())
            posterior /= posterior.sum()
        return predictive

    misspecified_model = gaussian_mark_model(
        place_fields, clusterless.waveform_means + 0.8, clusterless.sigma
    )
    return SimpleNamespace(
        predictive=decode(clusterless.model),
        misspecified_predictive=decode(misspecified_model),
        event_time_ind=event_time_ind,
        event_marks=event_marks,
        event_unit=event_unit,
        model=clusterless.model,
        misspecified_model=misspecified_model,
    )
