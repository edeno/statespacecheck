"""Tests for posterior consistency functions."""

import numpy as np
import pytest
from conftest import (
    make_gaussian_1d,
    make_gaussian_2d,
    make_random_distribution_1d,
    make_random_distribution_2d,
)

from statespacecheck.state_consistency import (
    hpd_overlap,
    kl_divergence,
)


class TestKLDivergence:
    """Tests for kl_divergence function."""

    def test_1d_spatial_identical_distributions(self, rng) -> None:
        """Test KL divergence is zero for identical distributions."""
        n_time, n_bins = 10, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)

        kl_div = kl_divergence(state_dist, state_dist)

        assert kl_div.shape == (n_time,)
        assert np.allclose(kl_div, 0.0, atol=1e-10)

    def test_2d_spatial_identical_distributions(self, rng) -> None:
        """Test KL divergence is zero for identical 2D distributions."""
        n_time, n_x, n_y = 10, 5, 5
        state_dist = make_random_distribution_2d(rng, n_time, n_x, n_y)

        kl_div = kl_divergence(state_dist, state_dist)

        # CRITICAL: Must return (n_time,) shape, not (n_time, n_x)
        assert kl_div.shape == (n_time,), f"Expected shape (n_time,)={n_time}, got {kl_div.shape}"
        assert np.allclose(kl_div, 0.0, atol=1e-10)

    def test_1d_spatial_different_distributions(self, rng) -> None:
        """Test KL divergence is positive for different distributions."""
        n_time, n_bins = 5, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)
        likelihood = make_random_distribution_1d(rng, n_time, n_bins)

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        # KL divergence should be positive for different distributions
        assert np.all(kl_div > 0)

    def test_2d_spatial_different_distributions(self, rng) -> None:
        """Test KL divergence is positive for different 2D distributions."""
        n_time, n_x, n_y = 5, 5, 5
        state_dist = make_random_distribution_2d(rng, n_time, n_x, n_y)
        likelihood = make_random_distribution_2d(rng, n_time, n_x, n_y)

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        assert np.all(kl_div > 0)

    def test_shape_mismatch_raises_error(self, rng) -> None:
        """Test that shape mismatch raises ValueError."""
        n_time = 5
        state_dist = make_random_distribution_1d(rng, n_time, 20)
        likelihood = make_random_distribution_1d(rng, n_time, 10)

        with pytest.raises(ValueError, match="must have same shape"):
            kl_divergence(state_dist, likelihood)

    def test_negative_values_raise_error(self, rng) -> None:
        """Test that negative values raise ValueError."""
        n_time, n_bins = 5, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)
        likelihood = state_dist.copy()
        likelihood[0, 0] = -0.1  # Add negative value

        with pytest.raises(ValueError, match="non-negative"):
            kl_divergence(state_dist, likelihood)

    def test_handles_zero_sum_rows(self, rng) -> None:
        """Test handling of rows with zero sum."""
        n_time, n_bins = 5, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)
        likelihood = state_dist.copy()
        # Set one row to zero
        state_dist[2, :] = 0.0

        kl_div = kl_divergence(state_dist, likelihood)

        assert kl_div.shape == (n_time,)
        # Row with zero sum should return inf
        assert np.isinf(kl_div[2])
        # Other rows should be valid
        assert np.all(np.isfinite(kl_div[[0, 1, 3, 4]]))

    def test_1d_input_raises_error(self) -> None:
        """Test that 1D input raises ValueError."""
        state_dist = np.array([0.2, 0.3, 0.5])  # 1D array
        likelihood = np.array([0.3, 0.4, 0.3])

        with pytest.raises(ValueError, match="must be at least 2D"):
            kl_divergence(state_dist, likelihood)

    def test_negative_state_dist_raises_error(self, rng) -> None:
        """Test that negative values in state_dist raise ValueError."""
        n_time, n_bins = 5, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)
        likelihood = state_dist.copy()
        state_dist[0, 0] = -0.1  # Add negative value to state_dist

        with pytest.raises(ValueError, match="state_dist must be non-negative"):
            kl_divergence(state_dist, likelihood)


class TestHPDOverlap:
    """Tests for hpd_overlap function."""

    def test_1d_spatial_identical_distributions(self, rng) -> None:
        """Test overlap is 1.0 for identical distributions."""
        n_time, n_bins = 10, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)

        overlap = hpd_overlap(state_dist, state_dist, coverage=0.95)

        assert overlap.shape == (n_time,)
        assert np.allclose(overlap, 1.0)

    def test_2d_spatial_identical_distributions(self, rng) -> None:
        """Test overlap is 1.0 for identical 2D distributions."""
        n_time, n_x, n_y = 10, 5, 5
        state_dist = make_random_distribution_2d(rng, n_time, n_x, n_y)

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
        state_dist = make_gaussian_1d(n_time, n_bins, mean=20, std=5)
        likelihood = make_gaussian_1d(n_time, n_bins, mean=30, std=5)

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # Should have some overlap but not complete
        assert np.all((overlap > 0.0) & (overlap < 1.0))

    def test_2d_spatial_partial_overlap(self) -> None:
        """Test overlap for partially overlapping 2D Gaussian distributions."""
        n_time, n_x, n_y = 5, 20, 20
        # Two Gaussians at different locations
        state_dist = make_gaussian_2d(n_time, n_x, n_y, mean_x=8, mean_y=8, std=3)
        likelihood = make_gaussian_2d(n_time, n_x, n_y, mean_x=12, mean_y=12, std=3)

        overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

        assert overlap.shape == (n_time,)
        # Should have some overlap but not complete
        assert np.all((overlap > 0.0) & (overlap < 1.0))

    @pytest.mark.parametrize("invalid_coverage", [0.0, 1.0, 1.5])
    def test_invalid_coverage_raises_error(self, rng, invalid_coverage) -> None:
        """Test that invalid coverage values raise ValueError."""
        n_time, n_bins = 5, 20
        state_dist = make_random_distribution_1d(rng, n_time, n_bins)

        with pytest.raises(ValueError, match="coverage must be in"):
            hpd_overlap(state_dist, state_dist, coverage=invalid_coverage)

    def test_shape_mismatch_raises_error(self, rng) -> None:
        """Test that shape mismatch raises ValueError."""
        n_time = 5
        state_dist = make_random_distribution_1d(rng, n_time, 20)
        likelihood = make_random_distribution_1d(rng, n_time, 10)

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
