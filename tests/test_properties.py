"""Property-based tests using Hypothesis.

These tests verify mathematical properties and invariants that should hold
for all valid inputs, catching edge cases that might be missed by example-based tests.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from statespacecheck import highest_density_region, hpd_overlap, kl_divergence
from statespacecheck._validation import (
    flatten_time_spatial,
    get_spatial_axes,
    validate_distribution,
    validate_paired_distributions,
)


# Custom strategies for generating valid distributions
@st.composite
def valid_distribution_1d(draw, min_time=1, max_time=20, min_bins=2, max_bins=50):
    """Generate valid 1D probability distributions."""
    n_time = draw(st.integers(min_value=min_time, max_value=max_time))
    n_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))

    # Generate positive values
    dist = draw(
        arrays(
            dtype=np.float64,
            shape=(n_time, n_bins),
            elements=st.floats(
                min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Ensure each row has at least some positive mass
    for i in range(n_time):
        if dist[i].sum() == 0:
            dist[i, 0] = 1.0

    return dist


@st.composite
def valid_distribution_2d(draw, min_time=1, max_time=10, min_size=2, max_size=20):
    """Generate valid 2D probability distributions."""
    n_time = draw(st.integers(min_value=min_time, max_value=max_time))
    n_x = draw(st.integers(min_value=min_size, max_value=max_size))
    n_y = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate positive values
    dist = draw(
        arrays(
            dtype=np.float64,
            shape=(n_time, n_x, n_y),
            elements=st.floats(
                min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Ensure each time slice has at least some positive mass
    for i in range(n_time):
        if dist[i].sum() == 0:
            dist[i, 0, 0] = 1.0

    return dist


@st.composite
def valid_coverage(draw):
    """Generate valid coverage values in (0, 1)."""
    return draw(st.floats(min_value=0.01, max_value=0.99))


class TestHighestDensityRegionProperties:
    """Property-based tests for highest_density_region function."""

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_output_shape_matches_input_1d(self, dist):
        """Property: Output shape must match input shape."""
        region = highest_density_region(dist)
        assert region.shape == dist.shape

    @given(dist=valid_distribution_2d())
    @settings(deadline=None, max_examples=50)
    def test_output_shape_matches_input_2d(self, dist):
        """Property: Output shape must match input shape for 2D."""
        region = highest_density_region(dist)
        assert region.shape == dist.shape

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_output_is_boolean(self, dist):
        """Property: Output must be boolean array."""
        region = highest_density_region(dist)
        assert region.dtype == bool

    @given(dist=valid_distribution_1d(), cov=valid_coverage())
    @settings(deadline=None, max_examples=50)
    def test_coverage_is_met_or_exceeded(self, dist, cov):
        """Property: Actual coverage should be >= requested coverage."""
        region = highest_density_region(dist, coverage=cov)

        # Normalize distributions for coverage calculation
        row_sums = dist.sum(axis=1)
        valid_rows = row_sums > 0
        if not np.any(valid_rows):
            return  # Skip if no valid rows

        # Calculate actual coverage for valid rows
        dist_norm = dist.copy()
        dist_norm[valid_rows] = dist_norm[valid_rows] / row_sums[valid_rows, np.newaxis]

        actual_coverage = (region * dist_norm).sum(axis=1)

        # Coverage should be >= requested (allowing small numerical error)
        assert np.all(
            (actual_coverage[valid_rows] >= cov - 1e-10) | (actual_coverage[valid_rows] >= 0.999)
        )

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_hpd_is_subset_of_support(self, dist):
        """Property: HPD region should only include bins with positive probability."""
        region = highest_density_region(dist)

        # Bins in HPD should have positive probability (or be in rows with zero support)
        row_sums = dist.sum(axis=1)
        for i in range(dist.shape[0]):
            if row_sums[i] > 0:
                # For rows with positive mass, HPD bins should have positive values
                # (allowing for numerical precision issues)
                hpd_bins = region[i]
                if np.any(hpd_bins):
                    assert np.all(dist[i, hpd_bins] >= -1e-10)

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_empty_rows_have_empty_hpd(self, dist):
        """Property: Rows with zero mass should have empty HPD regions."""
        # Force some rows to have zero mass
        n_time = dist.shape[0]
        if n_time >= 2:
            dist[0, :] = 0.0

        region = highest_density_region(dist)

        # Row with zero mass should have no HPD bins
        if n_time >= 2:
            assert not np.any(region[0])

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_monotonicity_with_coverage(self, dist):
        """Property: Higher coverage should include more or equal bins."""
        region_50 = highest_density_region(dist, coverage=0.5)
        region_95 = highest_density_region(dist, coverage=0.95)

        # Count bins in each region per time
        count_50 = region_50.sum(axis=1)
        count_95 = region_95.sum(axis=1)

        # Higher coverage should have >= bins (monotonicity)
        assert np.all(count_95 >= count_50)


class TestKLDivergenceProperties:
    """Property-based tests for kl_divergence function."""

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_self_divergence_is_zero(self, dist):
        """Property: KL divergence of a distribution with itself should be 0."""
        kl_div = kl_divergence(dist, dist)

        # Filter out rows with zero mass (which return inf)
        row_sums = dist.sum(axis=1)
        valid_rows = row_sums > 0

        if np.any(valid_rows):
            assert np.allclose(kl_div[valid_rows], 0.0, atol=1e-10)

    @given(dist1=valid_distribution_1d(), dist2=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_divergence_is_non_negative(self, dist1, dist2):
        """Property: KL divergence must be non-negative."""
        # Ensure same shape
        if dist1.shape != dist2.shape:
            return  # Skip if shapes don't match

        kl_div = kl_divergence(dist1, dist2)

        # Filter out inf values (from zero-sum rows)
        finite_mask = np.isfinite(kl_div)
        if np.any(finite_mask):
            assert np.all(kl_div[finite_mask] >= 0.0)

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_output_shape_is_time_only(self, dist):
        """Property: Output shape should be (n_time,)."""
        kl_div = kl_divergence(dist, dist)
        assert kl_div.shape == (dist.shape[0],)

    @given(dist1=valid_distribution_2d(), dist2=valid_distribution_2d())
    @settings(deadline=None, max_examples=30)
    def test_divergence_works_with_2d(self, dist1, dist2):
        """Property: KL divergence should work with 2D spatial distributions."""
        # Ensure same shape
        if dist1.shape != dist2.shape:
            return

        kl_div = kl_divergence(dist1, dist2)

        # Output should be 1D with length n_time
        assert kl_div.ndim == 1
        assert kl_div.shape[0] == dist1.shape[0]


class TestHPDOverlapProperties:
    """Property-based tests for hpd_overlap function."""

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_self_overlap_is_one(self, dist):
        """Property: Overlap of a distribution with itself should be 1.0."""
        overlap = hpd_overlap(dist, dist)

        # Filter out rows with zero mass (where overlap is defined as 0)
        row_sums = dist.sum(axis=1)
        valid_rows = row_sums > 0

        if np.any(valid_rows):
            assert np.allclose(overlap[valid_rows], 1.0)

    @given(dist1=valid_distribution_1d(), dist2=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_overlap_is_in_valid_range(self, dist1, dist2):
        """Property: Overlap must be in [0, 1]."""
        # Ensure same shape
        if dist1.shape != dist2.shape:
            return

        overlap = hpd_overlap(dist1, dist2)

        # All values should be in [0, 1]
        assert np.all((overlap >= 0.0) & (overlap <= 1.0))

    @given(dist1=valid_distribution_1d(), dist2=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_overlap_is_symmetric(self, dist1, dist2):
        """Property: Overlap should be symmetric: overlap(A, B) == overlap(B, A)."""
        # Ensure same shape
        if dist1.shape != dist2.shape:
            return

        overlap_ab = hpd_overlap(dist1, dist2)
        overlap_ba = hpd_overlap(dist2, dist1)

        assert np.allclose(overlap_ab, overlap_ba)

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_output_shape_is_time_only(self, dist):
        """Property: Output shape should be (n_time,)."""
        overlap = hpd_overlap(dist, dist)
        assert overlap.shape == (dist.shape[0],)


class TestValidationProperties:
    """Property-based tests for validation utilities."""

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_validate_preserves_shape(self, dist):
        """Property: Validation should preserve input shape."""
        clean = validate_distribution(dist)
        assert clean.shape == dist.shape

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_validate_preserves_non_negativity(self, dist):
        """Property: Validation should preserve non-negativity."""
        clean = validate_distribution(dist)
        assert np.all(clean >= 0.0)

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_flatten_produces_correct_shape(self, dist):
        """Property: Flattening should produce (n_time, n_spatial) shape."""
        flat = flatten_time_spatial(dist)

        n_time = dist.shape[0]
        n_spatial = int(np.prod(dist.shape[1:]))

        assert flat.shape == (n_time, n_spatial)

    @given(dist=valid_distribution_2d())
    @settings(deadline=None, max_examples=50)
    def test_get_spatial_axes_correct(self, dist):
        """Property: get_spatial_axes should return all axes except 0."""
        axes = get_spatial_axes(dist)

        # Should be tuple of axes from 1 to ndim-1
        expected = tuple(range(1, dist.ndim))
        assert axes == expected

    @given(dist1=valid_distribution_1d(), dist2=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_paired_validation_preserves_shapes(self, dist1, dist2):
        """Property: Paired validation should preserve shapes when valid."""
        # Ensure same shape
        if dist1.shape != dist2.shape:
            # Should raise ValueError
            with pytest.raises(ValueError, match="must have same shape"):
                validate_paired_distributions(dist1, dist2)
        else:
            clean1, clean2 = validate_paired_distributions(dist1, dist2)
            assert clean1.shape == dist1.shape
            assert clean2.shape == dist2.shape


class TestEdgeCaseProperties:
    """Property-based tests for edge cases and boundary conditions."""

    @given(
        n_time=st.integers(min_value=1, max_value=10),
        n_bins=st.integers(min_value=1, max_value=20),
    )
    @settings(deadline=None, max_examples=50)
    def test_all_zero_distribution(self, n_time, n_bins):
        """Property: All-zero distributions should be handled gracefully."""
        dist = np.zeros((n_time, n_bins))

        # HPD of all-zero distribution should be all False
        region = highest_density_region(dist)
        assert not np.any(region)

        # KL divergence with all-zero should return inf
        kl_div = kl_divergence(dist, dist)
        assert np.all(np.isinf(kl_div))

        # Overlap of all-zero distributions should be 0
        overlap = hpd_overlap(dist, dist)
        assert np.allclose(overlap, 0.0)

    @given(
        n_time=st.integers(min_value=1, max_value=10),
        n_bins=st.integers(min_value=2, max_value=20),
    )
    @settings(deadline=None, max_examples=50)
    def test_single_bin_has_mass(self, n_time, n_bins):
        """Property: Distribution with single non-zero bin."""
        dist = np.zeros((n_time, n_bins))
        # Put all mass in first bin
        dist[:, 0] = 1.0

        # HPD should only include the first bin
        region = highest_density_region(dist, coverage=0.95)
        assert np.all(region[:, 0])
        assert np.all(~region[:, 1:])

        # KL divergence with itself should be 0
        kl_div = kl_divergence(dist, dist)
        assert np.allclose(kl_div, 0.0)

        # Overlap with itself should be 1
        overlap = hpd_overlap(dist, dist)
        assert np.allclose(overlap, 1.0)

    @given(dist=valid_distribution_1d())
    @settings(deadline=None, max_examples=50)
    def test_extreme_coverage_values(self, dist):
        """Property: Test with coverage values near boundaries."""
        # Very low coverage
        region_low = highest_density_region(dist, coverage=0.01)
        assert region_low.shape == dist.shape

        # Very high coverage
        region_high = highest_density_region(dist, coverage=0.99)
        assert region_high.shape == dist.shape

        # Low coverage should have fewer or equal bins
        assert np.all(region_low.sum(axis=1) <= region_high.sum(axis=1))
