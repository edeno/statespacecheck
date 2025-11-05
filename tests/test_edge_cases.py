"""Edge case tests for highest_density and state_consistency functions."""

import numpy as np
import pytest

from statespacecheck import highest_density_region, kl_divergence


class TestNonFiniteValues:
    """Test handling of non-finite values (NaN, inf, -inf)."""

    def test_highest_density_with_positive_infinity(self) -> None:
        """Test that +inf values are treated as zero mass."""
        distribution = np.array([[0.2, 0.3, 0.5, np.inf], [0.4, 0.3, 0.3, np.inf]])

        # Should not raise error
        region = highest_density_region(distribution, coverage=0.95)

        # +inf positions should NOT be in the region (treated as 0)
        assert not region[0, 3]
        assert not region[1, 3]
        assert region.shape == distribution.shape

    def test_highest_density_with_negative_infinity(self) -> None:
        """Test that -inf values are treated as zero mass."""
        distribution = np.array([[0.2, 0.3, 0.5, -np.inf], [0.4, 0.3, 0.3, -np.inf]])

        # Should not raise error
        region = highest_density_region(distribution, coverage=0.95)

        # -inf positions should NOT be in the region (treated as 0)
        assert not region[0, 3]
        assert not region[1, 3]
        assert region.shape == distribution.shape

    def test_highest_density_with_mixed_non_finite(self) -> None:
        """Test mixed NaN, +inf, -inf values."""
        distribution = np.array([[0.2, 0.3, np.nan, np.inf, -np.inf, 0.5]])

        region = highest_density_region(distribution, coverage=0.95)

        # All non-finite positions should NOT be in the region
        assert not region[0, 2]  # NaN
        assert not region[0, 3]  # +inf
        assert not region[0, 4]  # -inf
        # Highest density position should be in region
        assert region[0, 5]  # 0.5 is highest

    def test_highest_density_all_nan_row(self) -> None:
        """Test row with all NaN values returns all False."""
        distribution = np.array([[0.2, 0.3, 0.5], [np.nan, np.nan, np.nan], [0.4, 0.3, 0.3]])

        region = highest_density_region(distribution, coverage=0.95)

        # Second row (all NaN) should be all False
        assert not np.any(region[1, :])
        # Other rows should have valid regions
        assert np.any(region[0, :])
        assert np.any(region[2, :])

    def test_highest_density_all_inf_row(self) -> None:
        """Test row with all +inf values returns all False."""
        distribution = np.array([[0.2, 0.3, 0.5], [np.inf, np.inf, np.inf], [0.4, 0.3, 0.3]])

        region = highest_density_region(distribution, coverage=0.95)

        # Second row (all +inf -> cleaned to 0) should be all False
        assert not np.any(region[1, :])


class TestNegativeValues:
    """Test handling of negative values."""

    def test_highest_density_rejects_negative_values(self) -> None:
        """Test that negative values raise ValueError."""
        distribution = np.array([[0.2, 0.3, -0.1, 0.5], [0.4, 0.3, 0.3, 0.0]])

        with pytest.raises(ValueError, match="must be non-negative"):
            highest_density_region(distribution, coverage=0.95)

    def test_highest_density_rejects_all_negative(self) -> None:
        """Test that all-negative row raises ValueError."""
        distribution = np.array([[0.2, 0.3, 0.5], [-0.1, -0.2, -0.3], [0.4, 0.3, 0.3]])

        with pytest.raises(ValueError, match="must be non-negative"):
            highest_density_region(distribution, coverage=0.95)


class TestTiedCutoffs:
    """Test handling of tied values at cutoff threshold."""

    def test_highest_density_with_many_tied_values(self) -> None:
        """Test that tied values at cutoff are all included (coverage >= target)."""
        # Create distribution with many tied values
        distribution = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

        region = highest_density_region(distribution, coverage=0.5)

        # With 50% coverage target (0.5 * 1.0 = 0.5), we need at least 5 bins
        # But because all have equal value (0.1), using >= cutoff includes all
        # This is expected HPD behavior with ties
        region_count = np.sum(region)
        region_mass = np.sum(distribution[region])

        # All bins are tied, so either all are in or we get exactly the ones that
        # reach the threshold. With the current implementation using >=, all equal
        # values at the cutoff are included.
        assert region_mass >= 0.5  # At least the target coverage
        assert region_count >= 5  # At least 5 bins

    def test_highest_density_ties_at_cutoff_threshold(self) -> None:
        """Test behavior when multiple bins have exactly the cutoff value."""
        # 60% coverage needs cumsum >= 0.6
        # Sorted: [0.3, 0.3, 0.2, 0.2] -> cumsum [0.3, 0.6, 0.8, 1.0]
        # Cutoff at index 1 (where cumsum first >= 0.6) -> cutoff = 0.3
        # Using >= 0.3 includes both 0.3 values
        distribution = np.array([[0.3, 0.2, 0.3, 0.2]])

        region = highest_density_region(distribution, coverage=0.6)

        # Both 0.3 values should be included (indices 0 and 2)
        assert region[0, 0]  # 0.3
        assert region[0, 2]  # 0.3
        # Check that we achieved at least target coverage
        assert np.sum(distribution[region]) >= 0.6


class TestKLDivergenceEdgeCases:
    """Test KL divergence edge cases."""

    def test_kl_divergence_with_zero_in_likelihood_positive_in_state(self) -> None:
        """Test KL divergence when Q=0 where P>0 (should be inf)."""
        # P has mass where Q has zero -> KL should be inf
        state_dist = np.array([[0.5, 0.5, 0.0], [0.3, 0.4, 0.3]])
        likelihood = np.array([[0.5, 0.0, 0.5], [0.3, 0.4, 0.3]])

        kl_div = kl_divergence(state_dist, likelihood)

        # First time: state has 0.5 at index 1, likelihood has 0
        assert np.isinf(kl_div[0])
        # Second time: both distributions are identical
        assert kl_div[1] < 0.01  # Very small (near zero)

    def test_kl_divergence_auto_normalization(self) -> None:
        """Test that distributions are automatically normalized."""
        # Create unnormalized distributions
        state_dist = np.array([[0.3, 0.3, 0.4]])  # Sums to 1.0 already
        likelihood = np.array([[0.6, 0.6, 0.8]])  # Sums to 2.0

        # Should auto-normalize and not raise error
        kl_div = kl_divergence(state_dist, likelihood)

        # Should compute successfully (identical after normalization)
        assert np.isfinite(kl_div[0])

    def test_kl_divergence_auto_normalization_with_nan(self) -> None:
        """Test that auto-normalization handles NaN bins correctly."""
        # 2D spatial array with NaN marking invalid bins
        # Valid bins: [0.2, 0.3, 0.2, 0.2] sum to 0.9 (need normalization)
        state_dist = np.array([[[0.2, 0.3], [np.nan, 0.2], [0.2, np.nan]]])
        # Valid bins: [0.25, 0.25, 0.25, 0.25] sum to 1.0 already
        likelihood = np.array([[[0.25, 0.25], [np.nan, 0.25], [0.25, np.nan]]])

        # Should auto-normalize over valid bins and preserve NaN
        kl_div = kl_divergence(state_dist, likelihood)

        # Should compute successfully
        assert np.isfinite(kl_div[0])

    def test_normalization_converts_nan_to_zero(self) -> None:
        """Test that auto-normalization converts NaN to 0 and normalizes valid bins."""
        # Import the private function for testing
        from statespacecheck.state_consistency import _validate_and_normalize_distributions

        # Unnormalized with NaN
        state_dist = np.array([[0.2, 0.4, np.nan, 0.8]])  # valid bins sum to 1.4
        likelihood = np.array([[0.3, np.nan, 0.3, 0.6]])  # valid bins sum to 1.2

        state_norm, like_norm = _validate_and_normalize_distributions(state_dist, likelihood)

        # Check valid bins sum to 1.0
        assert np.isclose(state_norm.sum(), 1.0)
        assert np.isclose(like_norm.sum(), 1.0)

        # Check NaN positions converted to 0
        assert state_norm[0, 2] == 0.0
        assert like_norm[0, 1] == 0.0

        # Check valid bins normalized correctly
        assert np.isclose(state_norm[0, 0], 0.2 / 1.4)  # 0.142857...
        assert np.isclose(like_norm[0, 0], 0.3 / 1.2)  # 0.25


class TestSmallSupports:
    """Test behavior with very small supports (1-3 bins)."""

    def test_highest_density_single_bin(self) -> None:
        """Test with single spatial bin."""
        distribution = np.array([[1.0], [1.0], [0.0]])

        region = highest_density_region(distribution, coverage=0.95)

        assert region.shape == (3, 1)
        # Rows with mass should include the single bin
        assert region[0, 0]
        assert region[1, 0]
        # Row with no mass should be all False
        assert not region[2, 0]

    def test_highest_density_two_bins(self) -> None:
        """Test with two spatial bins."""
        distribution = np.array([[0.7, 0.3], [0.5, 0.5]])

        region = highest_density_region(distribution, coverage=0.95)

        # First row: need 0.95 * 1.0 = 0.95, so need both bins (0.7 + 0.3 = 1.0)
        assert region[0, 0]
        assert region[0, 1]

        # Second row: equal bins, need >= 0.95, so need both
        assert region[1, 0]
        assert region[1, 1]

    def test_highest_density_three_bins(self) -> None:
        """Test with three spatial bins."""
        distribution = np.array([[0.6, 0.3, 0.1]])

        region = highest_density_region(distribution, coverage=0.95)

        # Need cumsum >= 0.95
        # Sorted: [0.6, 0.3, 0.1] -> cumsum [0.6, 0.9, 1.0]
        # Need index 2 to reach >= 0.95, cutoff = 0.1
        # All bins >= 0.1 are included
        assert np.all(region[0, :])


class TestLargeSpatialGrids:
    """Test behavior with large 2D spatial grids."""

    def test_highest_density_large_2d_grid(self) -> None:
        """Test with large 2D spatial grid (100x100)."""
        n_time = 5
        n_x, n_y = 100, 100

        # Create distributions with peaks in different locations
        distributions = []
        for t in range(n_time):
            dist = np.zeros((n_x, n_y))
            # Place Gaussian peak at different locations
            peak_x, peak_y = 20 + t * 15, 30 + t * 10
            x, y = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
            dist = np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / (2 * 10**2))
            dist = dist / dist.sum()
            distributions.append(dist)

        distribution = np.array(distributions)

        # Should handle large array without issues
        region = highest_density_region(distribution, coverage=0.95)

        assert region.shape == (n_time, n_x, n_y)
        # Each time should have a region near the peak
        for t in range(n_time):
            assert np.any(region[t, :, :])
            # Region should be a reasonable size (not too many bins)
            region_size = np.sum(region[t, :, :])
            # For a Gaussian, 95% HPD should be relatively compact
            assert region_size < n_x * n_y * 0.5  # Less than 50% of total bins
