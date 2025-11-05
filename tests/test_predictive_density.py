"""Tests for predictive density functions.

CRITICAL TEST: test_likelihood_not_normalized verifies that we DO NOT
normalize the likelihood distribution, only the state distribution.
This is mathematically required because p(y|x) is a likelihood function,
not a distribution over x.
"""

import numpy as np
import pytest

from statespacecheck.predictive_checks import (
    log_predictive_density,
    predictive_density,
)


class TestPredictiveDensity:
    """Test suite for predictive_density function."""

    def test_likelihood_not_normalized(self):
        """CRITICAL: Verify we normalize state_dist ONLY, NOT likelihood.

        This is the most important test. The formula is:
            f_predictive(y) = ∑_x p(x) * p(y|x)

        Where:
        - p(x) is state distribution (MUST normalize to 1)
        - p(y|x) is likelihood function (DO NOT normalize over x)

        Normalizing p(y|x) over x changes its value and masks model misfit.
        """
        n_time = 10
        n_position = 20

        # Create unnormalized state distribution
        rng = np.random.default_rng(42)
        state_dist = rng.random((n_time, n_position))

        # Create likelihood that is intentionally not normalized
        # Use values that would change significantly if normalized
        likelihood = rng.random((n_time, n_position)) * 10.0

        # Compute correct predictive density (should normalize state only)
        pred_correct = predictive_density(state_dist, likelihood)

        # Manually compute WRONG version (normalizing both)
        state_normalized = state_dist / state_dist.sum(axis=1, keepdims=True)
        likelihood_wrong = likelihood / likelihood.sum(axis=1, keepdims=True)
        pred_wrong = (state_normalized * likelihood_wrong).sum(axis=1)

        # Results should be DIFFERENT (proving we don't normalize likelihood)
        # If they're the same, we have a bug!
        assert not np.allclose(pred_correct, pred_wrong), (
            "ERROR: likelihood was normalized (should only normalize state_dist)"
        )

        # Verify correct computation: normalize state only, then integrate
        pred_expected = (state_normalized * likelihood).sum(axis=1)
        np.testing.assert_allclose(
            pred_correct,
            pred_expected,
            rtol=1e-10,
            err_msg="predictive_density should normalize state_dist ONLY",
        )

    def test_basic_1d(self):
        """Test basic functionality with 1D spatial distribution."""
        # Simple uniform distributions
        state = np.array([[1.0, 1.0, 1.0]])
        likelihood = np.array([[2.0, 3.0, 4.0]])

        pred = predictive_density(state, likelihood)

        # Expected: normalize state to [1/3, 1/3, 1/3], then sum with likelihood
        # (1/3)*2 + (1/3)*3 + (1/3)*4 = (2+3+4)/3 = 3.0
        assert pred.shape == (1,)
        np.testing.assert_allclose(pred[0], 3.0, rtol=1e-10)

    def test_basic_2d(self):
        """Test basic functionality with 2D spatial distribution."""
        # 2x2 spatial grid
        state = np.array([[[1.0, 1.0], [1.0, 1.0]]])  # shape (1, 2, 2)
        likelihood = np.array([[[2.0, 3.0], [4.0, 5.0]]])  # shape (1, 2, 2)

        pred = predictive_density(state, likelihood)

        # Expected: normalize state to [1/4, 1/4, 1/4, 1/4], then sum
        # (1/4)*2 + (1/4)*3 + (1/4)*4 + (1/4)*5 = (2+3+4+5)/4 = 3.5
        assert pred.shape == (1,)
        np.testing.assert_allclose(pred[0], 3.5, rtol=1e-10)

    def test_multiple_time_points(self):
        """Test with multiple time points processed independently."""
        n_time = 5
        n_position = 10

        rng = np.random.default_rng(42)
        state = rng.random((n_time, n_position))
        likelihood = rng.random((n_time, n_position)) * 10.0

        pred = predictive_density(state, likelihood)

        assert pred.shape == (n_time,)
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0)

        # Verify each time point computed independently
        for t in range(n_time):
            state_norm = state[t] / state[t].sum()
            expected = (state_norm * likelihood[t]).sum()
            np.testing.assert_allclose(pred[t], expected, rtol=1e-10)

    def test_nan_handling(self):
        """Test that NaN values are converted to 0 and excluded."""
        state = np.array([[1.0, np.nan, 1.0]])
        likelihood = np.array([[2.0, 3.0, 4.0]])

        pred = predictive_density(state, likelihood)

        # NaN in state should become 0, so only bins 0 and 2 contribute
        # Normalize [1, 0, 1] to [0.5, 0, 0.5]
        # Result: 0.5*2 + 0*3 + 0.5*4 = 3.0
        assert pred.shape == (1,)
        np.testing.assert_allclose(pred[0], 3.0, rtol=1e-10)

    def test_zero_state_distribution(self):
        """Test with all-zero state distribution returns NaN with warning."""
        state = np.array([[0.0, 0.0, 0.0]])
        likelihood = np.array([[2.0, 3.0, 4.0]])

        with pytest.warns(UserWarning, match="state_dist has zero-sum rows"):
            pred = predictive_density(state, likelihood)

        # Zero-sum state should result in NaN (no valid state mass)
        assert pred.shape == (1,)
        assert np.isnan(pred[0])

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        state = np.array([[1.0, 2.0, 3.0]])
        likelihood = np.array([[1.0, 2.0]])  # Different shape

        with pytest.raises(ValueError, match="must have same shape"):
            predictive_density(state, likelihood)

    def test_negative_values_error(self):
        """Test that negative values raise ValueError."""
        state = np.array([[1.0, -1.0, 1.0]])  # Negative state
        likelihood = np.array([[2.0, 3.0, 4.0]])

        with pytest.raises(ValueError, match="must be non-negative"):
            predictive_density(state, likelihood)

    def test_accepts_unnormalized_state(self):
        """Test that unnormalized state distribution is acceptable."""
        # State not normalized (sums to 10, not 1)
        state = np.array([[2.0, 3.0, 5.0]])
        likelihood = np.array([[1.0, 1.0, 1.0]])

        pred = predictive_density(state, likelihood)

        # Should normalize state to [0.2, 0.3, 0.5]
        # Result: 0.2*1 + 0.3*1 + 0.5*1 = 1.0
        assert pred.shape == (1,)
        np.testing.assert_allclose(pred[0], 1.0, rtol=1e-10)


class TestLogPredictiveDensity:
    """Test suite for log_predictive_density function."""

    def test_requires_exactly_one_likelihood_argument(self):
        """Test that exactly one of likelihood or log_likelihood must be provided."""
        state = np.array([[1.0, 2.0, 3.0]])

        # Neither provided - should error
        with pytest.raises(
            ValueError,
            match="Exactly one of 'likelihood' or 'log_likelihood' must be provided",
        ):
            log_predictive_density(state)

        # Both provided - should error
        likelihood = np.array([[1.0, 2.0, 3.0]])
        log_likelihood = np.log(likelihood)
        with pytest.raises(
            ValueError,
            match="Exactly one of 'likelihood' or 'log_likelihood' must be provided",
        ):
            log_predictive_density(state, likelihood=likelihood, log_likelihood=log_likelihood)

    def test_matches_log_of_predictive_density_stable_case(self):
        """Test that log_predictive_density matches log(predictive_density) for stable inputs."""
        n_time = 5
        n_position = 10

        rng = np.random.default_rng(42)
        state = rng.random((n_time, n_position))
        likelihood = rng.random((n_time, n_position)) + 0.1  # Avoid zeros

        # Compute using both methods
        log_pred = log_predictive_density(state, likelihood=likelihood)
        pred = predictive_density(state, likelihood)

        # Should match (for stable case)
        np.testing.assert_allclose(log_pred, np.log(pred), rtol=1e-10)

    def test_log_likelihood_input(self):
        """Test that log_likelihood input produces correct results."""
        state = np.array([[1.0, 1.0, 1.0]])
        likelihood = np.array([[2.0, 3.0, 4.0]])
        log_likelihood = np.log(likelihood)

        # Compute using both likelihood and log_likelihood
        log_pred_from_like = log_predictive_density(state, likelihood=likelihood)
        log_pred_from_log = log_predictive_density(state, log_likelihood=log_likelihood)

        # Should produce identical results
        np.testing.assert_allclose(log_pred_from_like, log_pred_from_log, rtol=1e-10)

    def test_numerical_stability_sparse_distribution(self):
        """Test numerical stability with very sparse/peaked distributions."""
        n_time = 10
        n_position = 1000

        state = np.zeros((n_time, n_position))
        state[:, 0] = 1.0  # Concentrated at position 0

        # Very small likelihood values that would underflow in linear space
        likelihood = np.full((n_time, n_position), 1e-200)
        likelihood[:, 0] = 1e-100  # Slightly larger at position 0

        # Using likelihood input (will convert internally)
        log_pred = log_predictive_density(state, likelihood=likelihood)

        # Should not be -inf (numerical underflow)
        assert np.all(np.isfinite(log_pred))
        # Should be very negative but computable
        assert np.all(log_pred < -200)

    def test_uses_logsumexp_not_log_of_sum(self):
        """Test that we use logsumexp, not log(sum(exp(...)))."""
        state = np.array([[1.0, 1.0, 1.0]])
        log_likelihood = np.array([[-500.0, -500.0, -500.0]])

        # If we use log(sum(exp(...))), this will underflow to -inf
        # But logsumexp handles it correctly
        log_pred = log_predictive_density(state, log_likelihood=log_likelihood)

        # Should not be -inf
        assert np.isfinite(log_pred[0])
        # Expected: log(1/3 * exp(-500) + 1/3 * exp(-500) + 1/3 * exp(-500))
        # = log(exp(-500)) = -500
        np.testing.assert_allclose(log_pred[0], -500.0, rtol=1e-10)

    def test_basic_computation(self):
        """Test basic computation with simple values."""
        # Uniform state, simple likelihood
        state = np.array([[1.0, 1.0, 1.0]])
        likelihood = np.array([[2.0, 3.0, 4.0]])

        log_pred = log_predictive_density(state, likelihood=likelihood)

        # Expected: log((1/3)*2 + (1/3)*3 + (1/3)*4) = log(3.0)
        assert log_pred.shape == (1,)
        np.testing.assert_allclose(log_pred[0], np.log(3.0), rtol=1e-10)

    def test_2d_spatial(self):
        """Test with 2D spatial distribution."""
        state = np.array([[[1.0, 1.0], [1.0, 1.0]]])  # shape (1, 2, 2)
        likelihood = np.array([[[2.0, 3.0], [4.0, 5.0]]])  # shape (1, 2, 2)

        log_pred = log_predictive_density(state, likelihood=likelihood)

        # Expected: log((1/4)*(2+3+4+5)) = log(3.5)
        assert log_pred.shape == (1,)
        np.testing.assert_allclose(log_pred[0], np.log(3.5), rtol=1e-10)

    def test_log_likelihood_wrong_dimensions_error(self):
        """Test that log_likelihood with wrong dimensions raises ValueError."""
        state = np.array([[1.0, 2.0, 3.0]])
        log_likelihood = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D

        with pytest.raises(ValueError, match="must be at least 2D"):
            log_predictive_density(state, log_likelihood=log_likelihood)

    def test_log_likelihood_shape_mismatch_error(self):
        """Test that log_likelihood shape mismatch raises ValueError."""
        state = np.array([[1.0, 2.0, 3.0]])
        log_likelihood = np.array([[1.0, 2.0]])  # Different shape

        with pytest.raises(ValueError, match="must have same shape"):
            log_predictive_density(state, log_likelihood=log_likelihood)
