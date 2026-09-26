"""Pytest fixtures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Shared random number generator with fixed seed for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures after every test, pass or fail.

    With warnings as errors, figures left open by failing tests would trip
    matplotlib's too-many-open-figures warning in unrelated tests.
    """
    yield
    plt.close("all")
