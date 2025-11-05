"""Consistency lock tests for predictive check functions.

These tests verify invariants and mathematical properties that should hold
across the predictive check implementations to guard against regressions.
"""

import numpy as np
import pytest

from statespacecheck.predictive_checks import (
    log_predictive_density,
    predictive_density,
)


class TestPredictiveConsistency:
    """Test consistency relationships between predictive functions."""

    def test_log_of_predictive_equals_log_predictive(self):
        """Test np.log(predictive_density(...)) ≈ log_predictive_density(...)."""
        # Use stable case where both methods are accurate
        state = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 1.0]])
        likelihood = np.array([[0.5, 1.0, 0.8], [1.2, 0.9, 1.1]])

        # Compute via both methods
        pred = predictive_density(state, likelihood)
        log_pred_from_density = np.log(pred)

        log_pred_direct = log_predictive_density(state, likelihood=likelihood)

        # Should be very close (within numerical precision)
        np.testing.assert_allclose(log_pred_from_density, log_pred_direct, rtol=1e-10)

    def test_scaling_likelihood_scales_predictive(self):
        """Test that scaling likelihood by c scales predictive by c."""
        state = np.array([[1.0, 2.0, 3.0]])
        likelihood = np.array([[0.5, 1.0, 0.8]])
        scale_factor = 2.5

        # Compute base predictive
        pred_base = predictive_density(state, likelihood)

        # Compute with scaled likelihood
        pred_scaled = predictive_density(state, likelihood * scale_factor)

        # Should be scaled by the same factor
        np.testing.assert_allclose(pred_scaled, pred_base * scale_factor, rtol=1e-10)

    def test_scaling_likelihood_scales_log_predictive_via_likelihood(self):
        """Test log predictive with likelihood: log(c * like) = log(c) + log(like)."""
        state = np.array([[1.0, 2.0, 3.0]])
        likelihood = np.array([[0.5, 1.0, 0.8]])
        scale_factor = 2.5

        # Compute base log predictive
        log_pred_base = log_predictive_density(state, likelihood=likelihood)

        # Compute with scaled likelihood
        log_pred_scaled = log_predictive_density(state, likelihood=likelihood * scale_factor)

        # Should differ by log(scale_factor)
        np.testing.assert_allclose(
            log_pred_scaled, log_pred_base + np.log(scale_factor), rtol=1e-10
        )

    def test_scaling_log_likelihood_scales_log_predictive(self):
        """Test log predictive with log_likelihood: adding log(c) scales by c."""
        state = np.array([[1.0, 2.0, 3.0]])
        likelihood = np.array([[0.5, 1.0, 0.8]])
        scale_factor = 2.5

        log_likelihood = np.log(likelihood)
        log_scale = np.log(scale_factor)

        # Compute base log predictive
        log_pred_base = log_predictive_density(state, log_likelihood=log_likelihood)

        # Compute with scaled log_likelihood (adding log(c))
        log_pred_scaled = log_predictive_density(state, log_likelihood=log_likelihood + log_scale)

        # Should differ by log(scale_factor)
        np.testing.assert_allclose(log_pred_scaled, log_pred_base + log_scale, rtol=1e-10)

    def test_zero_sum_state_returns_nan_with_warning_predictive(self):
        """Test that zero-sum state rows return NaN with warning (predictive_density)."""
        state = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])  # First row zero
        likelihood = np.array([[0.5, 1.0, 0.8], [1.2, 0.9, 1.1]])

        with pytest.warns(UserWarning, match="state_dist has zero-sum rows"):
            pred = predictive_density(state, likelihood)

        # First time point should be NaN, second should be valid
        assert np.isnan(pred[0])
        assert np.isfinite(pred[1])
        assert pred[1] > 0  # Should be positive

    def test_zero_sum_state_returns_nan_with_warning_log_predictive(self):
        """Test that zero-sum state rows return NaN with warning (log_predictive_density)."""
        state = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])  # Second row zero
        likelihood = np.array([[0.5, 1.0, 0.8], [1.2, 0.9, 1.1]])

        with pytest.warns(UserWarning, match="state_dist has zero-sum rows"):
            log_pred = log_predictive_density(state, likelihood=likelihood)

        # Second time point should be NaN, first should be valid
        assert np.isfinite(log_pred[0])
        assert np.isnan(log_pred[1])

    def test_posinf_in_log_likelihood_raises_error(self):
        """Test that +inf in log_likelihood raises ValueError."""
        state = np.array([[1.0, 2.0, 3.0]])
        log_likelihood = np.array([[0.5, np.inf, 0.8]])  # Contains +inf

        with pytest.raises(ValueError, match="log_likelihood contains \\+inf"):
            log_predictive_density(state, log_likelihood=log_likelihood)

    def test_neginf_in_log_likelihood_is_allowed(self):
        """Test that -inf in log_likelihood is allowed (represents zero probability)."""
        state = np.array([[1.0, 2.0, 3.0]])
        log_likelihood = np.array([[0.5, -np.inf, 0.8]])  # Contains -inf (OK)

        # Should work without error
        log_pred = log_predictive_density(state, log_likelihood=log_likelihood)

        # Result should be finite (the -inf term contributes 0 to sum)
        assert np.isfinite(log_pred[0])

    def test_nan_in_log_likelihood_treated_as_neginf(self):
        """Test that NaN in log_likelihood is treated as -inf (zero probability)."""
        state = np.array([[1.0, 2.0, 3.0]])

        # Create two versions: one with NaN, one with -inf
        log_likelihood_nan = np.array([[0.5, np.nan, 0.8]])
        log_likelihood_neginf = np.array([[0.5, -np.inf, 0.8]])

        log_pred_nan = log_predictive_density(state, log_likelihood=log_likelihood_nan)
        log_pred_neginf = log_predictive_density(state, log_likelihood=log_likelihood_neginf)

        # Should produce same result
        np.testing.assert_allclose(log_pred_nan, log_pred_neginf, rtol=1e-10)

    def test_both_methods_handle_zero_sum_consistently(self):
        """Test that both methods handle zero-sum state consistently."""
        state = np.array([[0.0, 0.0, 0.0]])
        likelihood = np.array([[0.5, 1.0, 0.8]])

        # Both should warn
        with pytest.warns(UserWarning, match="state_dist has zero-sum rows"):
            pred = predictive_density(state, likelihood)

        with pytest.warns(UserWarning, match="state_dist has zero-sum rows"):
            log_pred = log_predictive_density(state, likelihood=likelihood)

        # Both should return NaN
        assert np.isnan(pred[0])
        assert np.isnan(log_pred[0])

    def test_multimodal_distribution_consistency(self):
        """Test consistency with multimodal distributions."""
        # Create bimodal state distribution
        state = np.array([[5.0, 1.0, 1.0, 1.0, 5.0]])  # Two peaks
        likelihood = np.array([[0.1, 0.2, 0.3, 0.2, 0.1]])

        pred = predictive_density(state, likelihood)
        log_pred_direct = log_predictive_density(state, likelihood=likelihood)
        log_pred_from_density = np.log(pred)

        np.testing.assert_allclose(log_pred_from_density, log_pred_direct, rtol=1e-10)

    def test_single_nonzero_bin_consistency(self):
        """Test consistency when state has single nonzero bin."""
        # State concentrated in one bin
        state = np.array([[0.0, 10.0, 0.0]])
        likelihood = np.array([[0.5, 1.5, 0.8]])

        pred = predictive_density(state, likelihood)
        log_pred_direct = log_predictive_density(state, likelihood=likelihood)

        # Result should be the likelihood value at the nonzero bin
        # State normalizes to [0, 1, 0], so predictive = 1 * 1.5 = 1.5
        np.testing.assert_allclose(pred[0], 1.5, rtol=1e-10)
        np.testing.assert_allclose(log_pred_direct[0], np.log(1.5), rtol=1e-10)

    def test_uniform_state_averages_likelihood(self):
        """Test that uniform state produces average of likelihood."""
        # Uniform state
        state = np.array([[1.0, 1.0, 1.0]])  # Will normalize to [1/3, 1/3, 1/3]
        likelihood = np.array([[3.0, 6.0, 9.0]])

        pred = predictive_density(state, likelihood)

        # Should be mean of likelihood: (3 + 6 + 9) / 3 = 6.0
        np.testing.assert_allclose(pred[0], 6.0, rtol=1e-10)

    def test_2d_spatial_consistency(self):
        """Test consistency with 2D spatial distributions."""
        # 2D spatial distribution (n_time=2, n_x=2, n_y=2)
        state = np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 2.0], [2.0, 2.0]]])
        likelihood = np.array([[[0.5, 1.0], [1.5, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])

        pred = predictive_density(state, likelihood)
        log_pred_direct = log_predictive_density(state, likelihood=likelihood)
        log_pred_from_density = np.log(pred)

        np.testing.assert_allclose(log_pred_from_density, log_pred_direct, rtol=1e-10)


class TestWeightsValidation:
    """Test finite weights validation in aggregate_over_period."""

    def test_nan_weights_raises_error(self):
        """Test that NaN weights raise ValueError."""
        from statespacecheck.periods import aggregate_over_period

        metrics = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, True, True])
        weights = np.array([1.0, np.nan, 1.0])

        with pytest.raises(ValueError, match="weights must be finite"):
            aggregate_over_period(metrics, mask, weights=weights)

    def test_inf_weights_raises_error(self):
        """Test that inf weights raise ValueError."""
        from statespacecheck.periods import aggregate_over_period

        metrics = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, True, True])
        weights = np.array([1.0, np.inf, 1.0])

        with pytest.raises(ValueError, match="weights must be finite"):
            aggregate_over_period(metrics, mask, weights=weights)

    def test_neginf_weights_raises_error(self):
        """Test that -inf weights raise ValueError."""
        from statespacecheck.periods import aggregate_over_period

        metrics = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, True, True])
        weights = np.array([1.0, -np.inf, 1.0])

        with pytest.raises(ValueError, match="weights must be finite"):
            aggregate_over_period(metrics, mask, weights=weights)
