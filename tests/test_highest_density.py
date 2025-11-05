"""Tests for highest density functions."""

import numpy as np

from statespacecheck.highest_density import (
    highest_density_region,
)

# Use modern NumPy random API with fixed seed for reproducibility
rng = np.random.default_rng(seed=42)


class TestHighestDensityRegion:
    """Tests for highest_density_region function."""

    def test_1d_spatial_output_shape(self) -> None:
        """Test that output shape matches input for 1D spatial."""
        n_time, n_bins = 10, 20
        posterior = rng.dirichlet(np.ones(n_bins), size=n_time)

        region = highest_density_region(posterior, coverage=0.95)

        assert region.shape == posterior.shape
        assert region.dtype == bool

    def test_2d_spatial_output_shape(self) -> None:
        """Test that output shape matches input for 2D spatial."""
        n_time, n_x, n_y = 10, 5, 5
        n_bins = n_x * n_y
        # Create and reshape to 2D spatial
        posterior = rng.dirichlet(np.ones(n_bins), size=n_time).reshape(n_time, n_x, n_y)

        region = highest_density_region(posterior, coverage=0.95)

        # CRITICAL: Output shape must match input shape exactly
        assert region.shape == posterior.shape, (
            f"Expected shape {posterior.shape}, got {region.shape}. "
            f"HPD region must preserve spatial structure."
        )
        assert region.dtype == bool

    def test_1d_spatial_peaked_coverage(self) -> None:
        """Test that peaked 1D distribution has small HPD region."""
        n_time, n_bins = 5, 20
        posterior = np.zeros((n_time, n_bins))
        posterior[:, 10] = 1.0  # All mass at one position

        region = highest_density_region(posterior, coverage=0.95)

        # For peaked distribution, HPD should be very small (just 1 bin)
        region_size = region.sum(axis=1)
        assert np.all(region_size == 1)
        # Should select the peaked position
        assert np.all(region[:, 10])

    def test_2d_spatial_peaked_coverage(self) -> None:
        """Test that peaked 2D distribution has small HPD region."""
        n_time, n_x, n_y = 5, 10, 10
        posterior = np.zeros((n_time, n_x, n_y))
        posterior[:, 5, 5] = 1.0  # All mass at one position

        region = highest_density_region(posterior, coverage=0.95)

        # CRITICAL: Sum over BOTH spatial dimensions
        region_size = region.sum(axis=(1, 2))

        assert region_size.shape == (n_time,), (
            f"Expected region_size shape (n_time,)={n_time}, got {region_size.shape}"
        )
        # For peaked distribution, HPD should be very small (just 1 bin)
        assert np.all(region_size == 1)
        # Should select the peaked position
        assert np.all(region[:, 5, 5])

    def test_1d_spatial_gaussian_coverage(self) -> None:
        """Test coverage for Gaussian-like 1D distribution."""
        n_time, n_bins = 5, 50
        # Create Gaussian-like distribution
        x = np.arange(n_bins)
        posterior = np.exp(-((x - 25) ** 2) / (2 * 5**2))
        posterior = posterior / posterior.sum()
        posterior = np.tile(posterior, (n_time, 1))
        coverage = 0.95

        region = highest_density_region(posterior, coverage=coverage)

        # Check actual coverage is at least the requested coverage
        actual_coverage = (region * posterior).sum(axis=1)
        assert np.all(actual_coverage >= coverage)

        # For Gaussian, 95% HPD should be relatively compact
        region_size = region.sum(axis=1)
        assert np.all(region_size < n_bins * 0.8)  # Less than 80% of bins

    def test_2d_spatial_gaussian_coverage(self) -> None:
        """Test coverage for Gaussian-like 2D distribution."""
        n_time, n_x, n_y = 5, 20, 20
        # Create 2D Gaussian
        x = np.arange(n_x)
        y = np.arange(n_y)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        posterior = np.exp(-(((xx - 10) ** 2 + (yy - 10) ** 2) / (2 * 3**2)))
        posterior = posterior / posterior.sum()
        posterior = np.tile(posterior, (n_time, 1, 1))
        coverage = 0.95

        region = highest_density_region(posterior, coverage=coverage)

        # Check actual coverage by summing over both spatial dimensions
        actual_coverage = (region * posterior).sum(axis=(1, 2))
        assert actual_coverage.shape == (n_time,)
        # Check coverage is at least the requested coverage
        assert np.all(actual_coverage >= coverage)

    def test_2d_spatial_multimodal_distribution(self) -> None:
        """Test 2D spatial distribution with two peaks."""
        n_time, n_x, n_y = 5, 10, 10
        posterior = np.zeros((n_time, n_x, n_y))
        # Two peaks with equal mass
        posterior[:, 3, 3] = 0.5
        posterior[:, 7, 7] = 0.5

        region = highest_density_region(posterior, coverage=0.95)

        # CRITICAL: Both peaks should be in HPD region
        assert np.all(region[:, 3, 3])
        assert np.all(region[:, 7, 7])

        # Region size should be small (just the peaks)
        region_size = region.sum(axis=(1, 2))
        assert np.all(region_size <= 4)  # At most 2 bins per peak

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        n_time, n_bins = 5, 20
        posterior = rng.dirichlet(np.ones(n_bins), size=n_time)
        # Add some NaN values
        posterior[:, 0] = np.nan

        # Should not raise an error
        region = highest_density_region(posterior, coverage=0.95)

        assert region.shape == posterior.shape
        # NaN columns should be excluded from region
        assert not np.any(region[:, 0])
