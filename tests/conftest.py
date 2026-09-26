"""Pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Shared random number generator with fixed seed for reproducible tests."""
    return np.random.default_rng(seed=42)
