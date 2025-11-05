"""Tests for highest density functions."""

import numpy as np
import pytest
from conftest import (
    make_bimodal_gaussian_1d,
    make_gaussian_1d,
    make_gaussian_2d,
    make_random_distribution_1d,
    make_random_distribution_2d,
    sum_over_spatial,
)

from statespacecheck.highest_density import (
    highest_density_region,
)


class TestHighestDensityRegion:
    """Tests for highest_density_region function."""

    def test_1d_spatial_output_shape(self, rng) -> None:
        """Test that output shape matches input for 1D spatial."""
        n_time, n_bins = 10, 20
        posterior = make_random_distribution_1d(rng, n_time, n_bins)

        region = highest_density_region(posterior, coverage=0.95)

        assert region.shape == posterior.shape
        assert region.dtype == bool

    def test_2d_spatial_output_shape(self, rng) -> None:
        """Test that output shape matches input for 2D spatial."""
        n_time, n_x, n_y = 10, 5, 5
        posterior = make_random_distribution_2d(rng, n_time, n_x, n_y)

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
        region_size = sum_over_spatial(region)

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
        posterior = make_gaussian_1d(n_time, n_bins, mean=25, std=5)
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
        posterior = make_gaussian_2d(n_time, n_x, n_y, mean_x=10, mean_y=10, std=3)
        coverage = 0.95

        region = highest_density_region(posterior, coverage=coverage)

        # Check actual coverage by summing over both spatial dimensions
        actual_coverage = sum_over_spatial(region * posterior)
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
        region_size = sum_over_spatial(region)
        assert np.all(region_size <= 4)  # At most 2 bins per peak

    def test_handles_nan_values(self, rng) -> None:
        """Test that NaN values are handled correctly."""
        n_time, n_bins = 5, 20
        posterior = make_random_distribution_1d(rng, n_time, n_bins)
        # Add some NaN values
        posterior[:, 0] = np.nan

        # Should not raise an error
        region = highest_density_region(posterior, coverage=0.95)

        assert region.shape == posterior.shape
        # NaN columns should be excluded from region
        assert not np.any(region[:, 0])

    @pytest.mark.parametrize("invalid_coverage", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_coverage_raises_error(self, rng, invalid_coverage) -> None:
        """Test that invalid coverage values raise ValueError."""
        n_time, n_bins = 5, 20
        distribution = make_random_distribution_1d(rng, n_time, n_bins)

        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            highest_density_region(distribution, coverage=invalid_coverage)

    def test_exact_hd_region_1d(self) -> None:
        """Test exact HD region with simple 1D binary distributions."""
        # Test case 1: Two positions with equal mass
        # With 95% coverage, both positions should be in HD region
        distribution = np.array([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]])

        region = highest_density_region(distribution, coverage=0.95)

        # Expected: positions 2 and 3 should be in HD region
        expected = np.array([[False, False, True, True, False, False]])
        assert np.array_equal(region, expected)

        # Test case 2: Three positions with different mass
        # All three positions needed to reach 95% coverage
        distribution = np.array([[0.0, 0.5, 0.3, 0.2, 0.0, 0.0]])

        region = highest_density_region(distribution, coverage=0.95)

        # Expected: positions 1, 2, 3 (total 1.0 >= 0.95 * 1.0)
        # Cutoff is at 0.2, so all positions >= 0.2 are included
        expected = np.array([[False, True, True, True, False, False]])
        assert np.array_equal(region, expected)

        # Test case 3: Single peaked position
        distribution = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

        region = highest_density_region(distribution, coverage=0.95)

        # Expected: only position 2
        expected = np.array([[False, False, True, False, False, False]])
        assert np.array_equal(region, expected)

    def test_exact_hd_region_2d(self) -> None:
        """Test exact HD region with simple 2D binary distributions."""
        # Test case 1: Four corners with equal mass
        # With 95% coverage, all four should be in HD region
        distribution = np.array([[[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]]])

        region = highest_density_region(distribution, coverage=0.95)

        expected = np.array([[[True, False, True], [False, False, False], [True, False, True]]])
        assert np.array_equal(region, expected)

        # Test case 2: Two positions with different mass
        # Both positions needed to reach 95% coverage
        distribution = np.array([[[0.7, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]]])

        region = highest_density_region(distribution, coverage=0.95)

        # Expected: both positions (0,0) and (1,1) (total 1.0 >= 0.95 * 1.0)
        # Cutoff is at 0.3, so all positions >= 0.3 are included
        expected = np.array([[[True, False, False], [False, True, False], [False, False, False]]])
        assert np.array_equal(region, expected)

        # Test case 3: Center peak
        distribution = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])

        region = highest_density_region(distribution, coverage=0.95)

        # Expected: only center position (1,1)
        expected = np.array([[[False, False, False], [False, True, False], [False, False, False]]])
        assert np.array_equal(region, expected)

    def test_fine_discretization_unimodal_gaussian(self) -> None:
        """Test HD region for finely discretized unimodal Gaussian distribution.

        For a Gaussian with fine discretization, we expect:
        1. The HD region to be contiguous (single interval)
        2. The region to be symmetric around the mean
        3. Coverage to match the requested coverage
        """
        n_bins = 200
        mean = 100
        std = 20

        distribution = make_gaussian_1d(n_time=1, n_bins=n_bins, mean=mean, std=std)

        coverage = 0.95
        region = highest_density_region(distribution, coverage=coverage)

        # Test 1: Check actual coverage
        actual_coverage = (region * distribution).sum()
        assert actual_coverage >= coverage
        # With fine discretization, should be very close to requested coverage
        assert actual_coverage < coverage + 0.01

        # Test 2: Check region is contiguous (single interval)
        # Find the indices where region is True
        region_indices = np.where(region[0])[0]
        # Check that indices form a contiguous range
        expected_contiguous = np.arange(region_indices[0], region_indices[-1] + 1)
        assert np.array_equal(region_indices, expected_contiguous), (
            "HD region should be contiguous for unimodal Gaussian"
        )

        # Test 3: Check symmetry around mean
        # Region should be approximately symmetric around the mean
        left_edge = region_indices[0]
        right_edge = region_indices[-1]
        region_center = (left_edge + right_edge) / 2
        assert abs(region_center - mean) < 2, (
            f"HD region center ({region_center}) should be near mean ({mean})"
        )

        # Test 4: Check region size is reasonable for 95% coverage
        # For 95% coverage of Gaussian: ~1.96 standard deviations on each side
        # Total width should be approximately 2 * 1.96 * std = 3.92 * std
        region_width = right_edge - left_edge + 1
        expected_width = 2 * 1.96 * std
        # Allow 10% tolerance due to discretization
        assert abs(region_width - expected_width) < 0.1 * expected_width, (
            f"HD region width ({region_width}) should be near {expected_width:.1f}"
        )

    def test_fine_discretization_multimodal_gaussian(self) -> None:
        """Test HD region for finely discretized multimodal Gaussian distribution.

        For a mixture of two Gaussians with fine discretization:
        1. Both peaks should be included in the HD region
        2. Coverage should match the requested coverage
        3. Region may be non-contiguous if the valley between peaks is deep enough
        """
        n_bins = 200
        mean1, std1 = 60, 10
        mean2, std2 = 140, 10

        distribution = make_bimodal_gaussian_1d(
            n_time=1, n_bins=n_bins, mean1=mean1, std1=std1, mean2=mean2, std2=std2, weight1=0.5
        )

        coverage = 0.95
        region = highest_density_region(distribution, coverage=coverage)

        # Test 1: Check actual coverage
        actual_coverage = (region * distribution).sum()
        assert actual_coverage >= coverage
        # With fine discretization, should be very close to requested coverage
        assert actual_coverage < coverage + 0.01

        # Test 2: Both peaks should be included
        # Check that regions around both means are in the HD region
        assert region[0, mean1], f"Peak at position {mean1} should be in HD region"
        assert region[0, mean2], f"Peak at position {mean2} should be in HD region"

        # Test 3: Region should consist of two separate intervals (non-contiguous)
        # Find transitions (False->True and True->False)
        region_diff = np.diff(region[0].astype(int))
        n_rising_edges = (region_diff == 1).sum()  # False to True transitions

        # For well-separated bimodal distribution, should have 2 separate regions
        assert n_rising_edges >= 2, (
            f"Expected at least 2 separate regions for bimodal distribution, got {n_rising_edges}"
        )

        # Test 4: Check that each peak region is approximately symmetric
        # Find the region around each peak
        region_indices = np.where(region[0])[0]

        # Split indices into two groups (around each peak)
        midpoint = (mean1 + mean2) // 2
        region1_indices = region_indices[region_indices < midpoint]
        region2_indices = region_indices[region_indices > midpoint]

        # Check each region is non-empty
        assert len(region1_indices) > 0, "First peak should have HD region"
        assert len(region2_indices) > 0, "Second peak should have HD region"

        # Check approximate symmetry around each mean
        if len(region1_indices) > 0:
            center1 = (region1_indices[0] + region1_indices[-1]) / 2
            assert abs(center1 - mean1) < 3, (
                f"First HD region center ({center1}) should be near mean1 ({mean1})"
            )

        if len(region2_indices) > 0:
            center2 = (region2_indices[0] + region2_indices[-1]) / 2
            assert abs(center2 - mean2) < 3, (
                f"Second HD region center ({center2}) should be near mean2 ({mean2})"
            )
