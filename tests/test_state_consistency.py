"""Tests for posterior consistency functions."""

import numpy as np
import pytest

from statespacecheck.state_consistency import (
    hpd_overlap,
    kl_divergence,
)

# Use modern NumPy random API with fixed seed for reproducibility
rng = np.random.default_rng(seed=42)


class TestKLDivergence:
    """Tests for kl_divergence function."""

    def test_1d_spatial_identical_distributions(self) -> None:
        """Test KL divergence is zero for identical distributions."""
        n_time, n_bins = 10, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)

        kl_div = kl_divergence(state_dist, state_dist)

        assert kl_div.shape == (n_time,)
        assert np.allclose(kl_div, 0.0, atol=1e-10)

    def test_2d_spatial_identical_distributions(self) -> None:
        """Test KL divergence is zero for identical 2D distributions."""
        n_time, n_x, n_y = 10, 5, 5
        n_bins = n_x * n_y
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)

        kl_div = kl_divergence(state_dist, state_dist)

        # CRITICAL: Must return (n_time,) shape, not (n_time, n_x)
        assert kl_div.shape == (n_time,), f"Expected shape (n_time,)={n_time}, got {kl_div.shape}"
        assert np.allclose(kl_div, 0.0, atol=1e-10)

    def test_1d_spatial_different_distributions(self) -> None:
        """Test KL divergence is positive for different distributions."""
        n_time, n_bins = 5, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)
        likelihood = rng.dirichlet(np.ones(n_bins), size=n_time)

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        # KL divergence should be positive for different distributions
        assert np.all(kl_div > 0)

    def test_2d_spatial_different_distributions(self) -> None:
        """Test KL divergence is positive for different 2D distributions."""
        n_time, n_x, n_y = 5, 5, 5
        n_bins = n_x * n_y
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)
        likelihood = rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        assert np.all(kl_div > 0)

    def test_shape_mismatch_raises_error(self) -> None:
        """Test that shape mismatch raises ValueError."""
        n_time = 5
        state_dist = rng.dirichlet(np.ones(20), size=n_time)
        likelihood = rng.dirichlet(np.ones(10), size=n_time)

        with pytest.raises(ValueError, match="must have same shape"):
            kl_divergence(state_dist, likelihood)

    def test_negative_values_raise_error(self) -> None:
        """Test that negative values raise ValueError."""
        n_time, n_bins = 5, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)
        likelihood = state_dist.copy()
        likelihood[0, 0] = -0.1  # Add negative value

        with pytest.raises(ValueError, match="non-negative"):
            kl_divergence(state_dist, likelihood)

    def test_handles_zero_sum_rows(self) -> None:
        """Test handling of rows with zero sum."""
        n_time, n_bins = 5, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)
        likelihood = state_dist.copy()
        # Set one row to zero
        state_dist[2, :] = 0.0

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        # Row with zero sum should return inf
        assert np.isinf(kl_div[2])
        # Other rows should be valid
        assert np.all(np.isfinite(kl_div[[0, 1, 3, 4]]))


class TestHPDOverlap:
    """Tests for hpd_overlap function."""

    def test_1d_spatial_identical_distributions(self) -> None:
        """Test overlap is 1.0 for identical distributions."""
        n_time, n_bins = 10, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)

        overlap = hpd_overlap(state_dist, state_dist, coverage=0.95)

        assert overlap.shape == (n_time,)
        assert np.allclose(overlap, 1.0)

    def test_2d_spatial_identical_distributions(self) -> None:
        """Test overlap is 1.0 for identical 2D distributions."""
        n_time, n_x, n_y = 10, 5, 5
        n_bins = n_x * n_y
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)

        overlap = hpd_overlap(state_dist, state_dist, coverage=0.95)

        # CRITICAL: Must return (n_time,) shape
        assert overlap.shape == (n_time,), f"Expected shape (n_time,)={n_time}, got {overlap.shape}"
        assert np.allclose(overlap, 1.0)

    def test_1d_spatial_completely_different_distributions(self) -> None:
        """Test overlap for distributions with non-overlapping peaks."""
        n_time, n_bins = 5, 20
        state_dist = np.zeros((n_time, n_bins))
        likelihood = np.zeros((n_time, n_bins))
        # Put mass at different positions
        state_dist[:, 5] = 1.0
        likelihood[:, 15] = 1.0

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # No overlap expected
        assert np.allclose(overlap, 0.0)

    def test_2d_spatial_completely_different_distributions(self) -> None:
        """Test overlap for 2D distributions with non-overlapping peaks."""
        n_time, n_x, n_y = 5, 10, 10
        state_dist = np.zeros((n_time, n_x, n_y))
        likelihood = np.zeros((n_time, n_x, n_y))
        # Put mass at different positions
        state_dist[:, 2, 2] = 1.0
        likelihood[:, 7, 7] = 1.0

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        assert np.allclose(overlap, 0.0)

    def test_1d_spatial_partial_overlap(self) -> None:
        """Test overlap for partially overlapping Gaussian distributions."""
        n_time, n_bins = 5, 50
        # Create two Gaussian distributions with partial overlap
        x = np.arange(n_bins)
        state_dist = np.exp(-((x - 20) ** 2) / (2 * 5**2))
        state_dist = state_dist / state_dist.sum()
        likelihood = np.exp(-((x - 30) ** 2) / (2 * 5**2))
        likelihood = likelihood / likelihood.sum()
        state_dist = np.tile(state_dist, (n_time, 1))
        likelihood = np.tile(likelihood, (n_time, 1))

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # Should have some overlap but not complete
        assert np.all((overlap > 0.0) & (overlap < 1.0))

    def test_2d_spatial_partial_overlap(self) -> None:
        """Test overlap for partially overlapping 2D Gaussian distributions."""
        n_time, n_x, n_y = 5, 20, 20
        x = np.arange(n_x)
        y = np.arange(n_y)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        # Two Gaussians at different locations
        state_dist = np.exp(-(((xx - 8) ** 2 + (yy - 8) ** 2) / (2 * 3**2)))
        state_dist = state_dist / state_dist.sum()
        likelihood = np.exp(-(((xx - 12) ** 2 + (yy - 12) ** 2) / (2 * 3**2)))
        likelihood = likelihood / likelihood.sum()
        state_dist = np.tile(state_dist, (n_time, 1, 1))
        likelihood = np.tile(likelihood, (n_time, 1, 1))

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # Should have some overlap but not complete
        assert np.all((overlap > 0.0) & (overlap < 1.0))

    def test_invalid_coverage_raises_error(self) -> None:
        """Test that invalid coverage values raise ValueError."""
        n_time, n_bins = 5, 20
        state_dist = rng.dirichlet(np.ones(n_bins), size=n_time)

        with pytest.raises(ValueError, match="coverage must be in"):
            hpd_overlap(state_dist, state_dist, coverage=0.0)

        with pytest.raises(ValueError, match="coverage must be in"):
            hpd_overlap(state_dist, state_dist, coverage=1.0)

        with pytest.raises(ValueError, match="coverage must be in"):
            hpd_overlap(state_dist, state_dist, coverage=1.5)

    def test_shape_mismatch_raises_error(self) -> None:
        """Test that shape mismatch raises ValueError."""
        n_time = 5
        state_dist = rng.dirichlet(np.ones(20), size=n_time)
        likelihood = rng.dirichlet(np.ones(10), size=n_time)

        with pytest.raises(ValueError, match="must have same shape"):
            hpd_overlap(state_dist, likelihood)

    def test_handles_empty_hpd_regions(self) -> None:
        """Test handling when both HPD regions are empty."""
        n_time, n_bins = 5, 20
        state_dist = np.zeros((n_time, n_bins))
        likelihood = np.zeros((n_time, n_bins))

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # When both regions are empty, overlap should be 0
        assert np.allclose(overlap, 0.0)

    def test_exact_overlap_calculation(self) -> None:
        """Test exact overlap with simple binary distributions."""
        # Single time point for clarity
        n_time = 1

        # state_dist has mass at positions 2 and 3
        # likelihood has mass only at position 2
        # Expected: intersection = 1, min(2, 1) = 1, overlap = 1/1 = 1.0
        state_dist = np.array([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]])
        likelihood = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # With 95% coverage, both regions include their non-zero positions
        # state_dist HPD: positions 2,3 (size=2)
        # likelihood HPD: position 2 (size=1)
        # intersection: position 2 (size=1)
        # overlap = 1 / min(2, 1) = 1 / 1 = 1.0
        assert np.allclose(overlap, 1.0)

        # Test case 2: Partial overlap
        # state_dist has mass at positions 2 and 3
        # likelihood has mass at positions 3 and 4
        # Expected: intersection = 1, min(2, 2) = 2, overlap = 1/2 = 0.5
        state_dist = np.array([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]])
        likelihood = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.0]])

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        # state_dist HPD: positions 2,3 (size=2)
        # likelihood HPD: positions 3,4 (size=2)
        # intersection: position 3 (size=1)
        # overlap = 1 / min(2, 2) = 1 / 2 = 0.5
        assert np.allclose(overlap, 0.5)

    def test_exact_overlap_calculation_2d(self) -> None:
        """Test exact overlap with simple 2D binary distributions."""
        # Single time point for clarity
        n_time = 1

        # Test case 1: Complete overlap in 2D
        # state_dist has mass at positions (0,0) and (0,1)
        # likelihood has mass only at position (0,0)
        state_dist = np.array([[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        likelihood = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # state_dist HPD: positions (0,0), (0,1) (size=2)
        # likelihood HPD: position (0,0) (size=1)
        # intersection: position (0,0) (size=1)
        # overlap = 1 / min(2, 1) = 1.0
        assert np.allclose(overlap, 1.0)

        # Test case 2: Partial overlap in 2D - the key test case
        # state_dist has mass at (0,0), (0,1), (1,0), (1,1) - a 2x2 square
        # likelihood has mass at (0,1), (1,0), (1,1), (2,1) - overlapping 2x2 square shifted
        state_dist = np.array([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
        likelihood = np.array([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        # state_dist HPD: positions (0,0), (0,1), (1,0), (1,1) (size=4)
        # likelihood HPD: positions (0,1), (1,0), (1,1), (2,1) (size=4)
        # intersection: positions (0,1), (1,0), (1,1) (size=3)
        # overlap = 3 / min(4, 4) = 3 / 4 = 0.75
        assert np.allclose(overlap, 0.75)

        # Test case 3: Exactly half overlap in 2D
        # state_dist has mass at (0,0), (0,1)
        # likelihood has mass at (0,1), (1,0)
        state_dist = np.array([[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        likelihood = np.array([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        # state_dist HPD: positions (0,0), (0,1) (size=2)
        # likelihood HPD: positions (0,1), (1,0) (size=2)
        # intersection: position (0,1) (size=1)
        # overlap = 1 / min(2, 2) = 1 / 2 = 0.5
        assert np.allclose(overlap, 0.5)
