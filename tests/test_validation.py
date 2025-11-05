"""Tests for validation utilities."""

import numpy as np
import pytest

from statespacecheck._validation import (
    flatten_time_spatial,
    get_spatial_axes,
    validate_coverage,
    validate_distribution,
    validate_paired_distributions,
)


class TestValidateCoverage:
    """Tests for validate_coverage function."""

    def test_valid_coverage_values(self) -> None:
        """Test that valid coverage values don't raise errors."""
        # These should not raise
        validate_coverage(0.5)
        validate_coverage(0.95)
        validate_coverage(0.99)
        validate_coverage(0.01)

    def test_coverage_at_boundaries_raises_error(self) -> None:
        """Test that coverage at boundaries raises ValueError."""
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            validate_coverage(0.0)

        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            validate_coverage(1.0)

    def test_coverage_outside_bounds_raises_error(self) -> None:
        """Test that coverage outside bounds raises ValueError."""
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            validate_coverage(-0.1)

        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            validate_coverage(1.5)


class TestValidateDistribution:
    """Tests for validate_distribution function."""

    def test_valid_1d_distribution(self) -> None:
        """Test validation of valid 1D distribution."""
        dist = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        clean = validate_distribution(dist, min_ndim=1)

        assert clean.shape == dist.shape
        assert np.allclose(clean, dist)

        # Test flattening separately
        flat = flatten_time_spatial(clean)
        assert flat.shape == (2, 3)

    def test_valid_2d_distribution(self) -> None:
        """Test validation of valid 2D distribution."""
        dist = np.array([[[0.2, 0.3], [0.2, 0.3]], [[0.4, 0.3], [0.2, 0.1]]])
        clean = validate_distribution(dist, min_ndim=2)

        assert clean.shape == dist.shape

        # Test flattening separately
        flat = flatten_time_spatial(clean)
        assert flat.shape == (2, 4)

    def test_distribution_with_nan_allowed(self) -> None:
        """Test that NaN values are converted to 0 when allowed."""
        dist = np.array([[0.2, np.nan, 0.5], [0.4, 0.3, 0.3]])
        clean = validate_distribution(dist, allow_nan=True)

        assert clean[0, 1] == 0.0

        # Test flattening separately
        flat = flatten_time_spatial(clean)
        assert flat[0, 1] == 0.0

    def test_distribution_with_inf_allowed(self) -> None:
        """Test that inf values are converted to 0 when allowed."""
        dist = np.array([[0.2, np.inf, 0.5], [0.4, 0.3, -np.inf]])
        clean = validate_distribution(dist, allow_nan=True)

        assert clean[0, 1] == 0.0
        assert clean[1, 2] == 0.0

    def test_distribution_with_nan_not_allowed_raises_error(self) -> None:
        """Test that NaN values raise error when not allowed."""
        dist = np.array([[0.2, np.nan, 0.5], [0.4, 0.3, 0.3]])

        with pytest.raises(ValueError, match="contains non-finite values"):
            validate_distribution(dist, allow_nan=False)

    def test_negative_values_raise_error(self) -> None:
        """Test that negative values raise ValueError."""
        dist = np.array([[0.2, -0.1, 0.5], [0.4, 0.3, 0.3]])

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_distribution(dist)

    def test_insufficient_dimensions_raises_error(self) -> None:
        """Test that insufficient dimensions raise ValueError."""
        dist = np.array([0.2, 0.3, 0.5])  # 1D array

        with pytest.raises(ValueError, match="must be at least 2D"):
            validate_distribution(dist, min_ndim=2)

    def test_insufficient_dimensions_min_ndim_1(self) -> None:
        """Test error message for min_ndim=1 with scalar input."""
        dist = np.array(0.5)  # 0D array (scalar)

        with pytest.raises(ValueError, match=r"must be at least 1D.*\(n_time,\)"):
            validate_distribution(dist, min_ndim=1)

    def test_insufficient_dimensions_min_ndim_3(self) -> None:
        """Test error message for min_ndim=3 with 2D input."""
        dist = np.array([[0.2, 0.3, 0.5]])  # 2D array

        with pytest.raises(ValueError, match="must be at least 3D"):
            validate_distribution(dist, min_ndim=3)

    def test_spatial_dimension_flattening(self) -> None:
        """Test that spatial dimensions are correctly flattened."""
        # Test basic flattening
        dist = np.array([[0.2, 0.3, 0.5]])
        clean = validate_distribution(dist, min_ndim=1)
        flat = flatten_time_spatial(clean)
        assert flat.shape == (1, 3)

        # Test with larger dimensions
        large_dist = np.ones((2, 100, 100))
        clean_large = validate_distribution(large_dist, min_ndim=2)
        flat_large = flatten_time_spatial(clean_large)
        assert flat_large.shape == (2, 10000)

    def test_zero_spatial_dimension_handled(self) -> None:
        """Test that zero spatial dimension is handled correctly by numpy."""
        # Create array with one spatial dimension being 0
        # Shape (2, 0) means 2 time points, 0 spatial bins
        dist = np.array([[], []])  # Shape (2, 0)

        # This should succeed - numpy handles empty arrays fine
        clean = validate_distribution(dist, min_ndim=2)
        flat = flatten_time_spatial(clean)
        assert flat.shape == (2, 0)


class TestValidatePairedDistributions:
    """Tests for validate_paired_distributions function."""

    def test_valid_paired_distributions(self) -> None:
        """Test validation of valid paired distributions."""
        dist1 = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        dist2 = np.array([[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])

        clean1, clean2 = validate_paired_distributions(dist1, dist2)

        assert clean1.shape == dist1.shape
        assert clean2.shape == dist2.shape

        # Test flattening separately
        flat1 = flatten_time_spatial(clean1)
        flat2 = flatten_time_spatial(clean2)
        assert flat1.shape == (2, 3)
        assert flat2.shape == (2, 3)

    def test_shape_mismatch_raises_error(self) -> None:
        """Test that shape mismatch raises ValueError."""
        dist1 = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        dist2 = np.array([[0.3, 0.4], [0.5, 0.2]])  # Different shape

        with pytest.raises(ValueError, match="must have same shape"):
            validate_paired_distributions(dist1, dist2)

    def test_paired_distributions_with_custom_names(self) -> None:
        """Test that custom names appear in error messages."""
        dist1 = np.array([[0.2, 0.3, 0.5]])
        dist2 = np.array([[0.3, 0.4]])  # Different shape

        with pytest.raises(ValueError, match="foo.*bar.*must have same shape"):
            validate_paired_distributions(dist1, dist2, name1="foo", name2="bar")


class TestGetSpatialAxes:
    """Tests for get_spatial_axes function."""

    def test_2d_array(self) -> None:
        """Test spatial axes for 2D array (n_time, n_bins)."""
        arr = np.zeros((10, 20))
        axes = get_spatial_axes(arr)
        assert axes == (1,)

    def test_3d_array(self) -> None:
        """Test spatial axes for 3D array (n_time, n_x, n_y)."""
        arr = np.zeros((10, 5, 5))
        axes = get_spatial_axes(arr)
        assert axes == (1, 2)

    def test_4d_array(self) -> None:
        """Test spatial axes for 4D array."""
        arr = np.zeros((10, 5, 5, 3))
        axes = get_spatial_axes(arr)
        assert axes == (1, 2, 3)

    def test_1d_array(self) -> None:
        """Test spatial axes for 1D array (edge case)."""
        arr = np.zeros(10)
        axes = get_spatial_axes(arr)
        assert axes == ()
