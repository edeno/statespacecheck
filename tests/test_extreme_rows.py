"""Rows with extreme total mass (subnormal or overflowing) and rows with no bins."""

import numpy as np
import pytest

from statespacecheck import (
    highest_density_region,
    hpd_overlap,
    kl_divergence,
    log_predictive_density,
    predictive_density,
)


class TestKLDivergenceSubnormalNumbers:
    """Test KL divergence handles subnormal numbers correctly."""

    def test_kl_divergence_with_subnormal_values_is_non_negative(self):
        """Test that KL divergence with subnormal values doesn't return negative.

        This test uses the exact failing case found by Hypothesis that triggers
        floating point precision errors in scipy.stats.entropy, which can return
        tiny negative values (~10^-113) instead of 0 for nearly identical distributions.

        KL divergence is mathematically always non-negative. The implementation
        must clip spurious negative floating point artifacts to ensure this property.
        """
        # Exact failing case from Hypothesis property test
        dist1 = np.array([[4.39835706e-113, 2.00000000e000]])
        dist2 = np.array([[4.39835706e-113, 1.00000000e000]])

        kl_div = kl_divergence(dist1, dist2)

        # KL divergence must be non-negative (mathematical property)
        # No tolerance - must be exactly >= 0 since we clip in implementation
        assert kl_div[0] >= 0.0, (
            f"KL divergence must be non-negative, got {kl_div[0]:.20e}. "
            "This indicates floating point precision issues with subnormal numbers."
        )

    def test_kl_divergence_clips_tiny_negative_to_zero(self):
        """Test that tiny negative values from floating point errors are clipped to 0."""
        # Another case that might trigger similar issues
        dist1 = np.array([[1e-200, 1.0]])
        dist2 = np.array([[2e-200, 1.0]])

        kl_div = kl_divergence(dist1, dist2)

        # Should be non-negative
        assert kl_div[0] >= 0.0

        # Should be very close to 0 (distributions are nearly identical except for tiny values)
        assert kl_div[0] < 1e-10


# Each takes rows of nonnegative values; results must not depend on their overall scale
ROW_COMPUTATIONS = {
    "kl_divergence": lambda s: kl_divergence(s, s[:, ::-1]),
    "hpd_overlap": lambda s: hpd_overlap(s, s[:, ::-1]),
    "highest_density_region": lambda s: highest_density_region(s).astype(float),
    "predictive_density": lambda s: predictive_density(s, np.arange(1.0, 10.0)[None]),
    "log_predictive_density": lambda s: log_predictive_density(s, np.arange(1.0, 10.0)[None]),
}


@pytest.mark.parametrize("name", ROW_COMPUTATIONS)
def test_row_with_subnormal_total_mass(name) -> None:
    """A row whose total mass is subnormal gives the same result as the row scaled
    up; dividing by the subnormal total warned about an overflow on NumPy 1.26."""
    tiny = np.zeros((1, 9))
    tiny[0, [0, 3, 5]] = [1.0e-309, 1.2e-309, 0.3e-309]  # 1 / sum exceeds the largest float
    compute = ROW_COMPUTATIONS[name]
    np.testing.assert_allclose(compute(tiny), compute(tiny * 2.0**1000), rtol=1e-12)


@pytest.mark.parametrize("name", ROW_COMPUTATIONS)
def test_row_whose_total_mass_overflows(name) -> None:
    """A row of finite values whose sum overflows gives the same result as the row
    scaled down, instead of infinite KL divergence or an empty region."""
    huge = np.zeros((1, 9))
    huge[0, [0, 3, 5]] = [1.0e308, 1.2e308, 0.3e308]
    compute = ROW_COMPUTATIONS[name]
    np.testing.assert_allclose(compute(huge), compute(huge * 2.0**-1000), rtol=1e-12)


@pytest.mark.parametrize(
    "compute",
    [
        lambda s: kl_divergence(s, s),
        lambda s: hpd_overlap(s, s),
        lambda s: highest_density_region(s),
        lambda s: predictive_density(s, s),
        lambda s: log_predictive_density(s, s),
    ],
    ids=["kl", "hpd", "hdr", "predictive", "log_predictive"],
)
def test_empty_spatial_axis_raises(compute) -> None:
    with pytest.raises(ValueError, match="no bins"):
        compute(np.ones((3, 0)))
