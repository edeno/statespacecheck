"""Test KL divergence with subnormal numbers that trigger floating point errors."""

import numpy as np

from statespacecheck.state_consistency import kl_divergence


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
