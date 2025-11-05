"""Tests for predictive_pvalue function."""

import numpy as np
import pytest
from scipy.stats import kstest

from statespacecheck.predictive_checks import predictive_pvalue


class TestPredictivePValue:
    """Test suite for predictive_pvalue function."""

    def test_pvalue_in_valid_range(self):
        """Test that p-values are always in [0, 1]."""
        n_time = 10
        observed = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

        # Sampler that returns random values around observed
        def sampler(n_samples):
            rng = np.random.default_rng(42)
            return rng.normal(loc=observed, scale=0.5, size=(n_samples, n_time))

        p_values = predictive_pvalue(observed, sampler, n_samples=100)

        # All p-values must be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_pvalue_uniform_under_null(self):
        """Test that p-values are uniform under null hypothesis (correct model).

        When observed data comes from the same distribution as simulated samples
        (null hypothesis is true), p-values should be uniformly distributed.
        We test this using the Kolmogorov-Smirnov test.
        """
        n_time = 50
        n_samples = 200
        n_replications = 100  # Repeat to get distribution of p-values

        # Generate multiple p-values under null hypothesis
        p_values_collected = []

        for rep in range(n_replications):
            rng = np.random.default_rng(rep)

            # Generate "observed" from same distribution as samples
            true_mean = rng.normal(0, 1, size=n_time)
            observed = true_mean + rng.normal(0, 0.1, size=n_time)

            # Sampler generates from same distribution
            # Use default arguments to capture loop variables
            def sampler(n_samp, _mean=true_mean, _rng=rng):
                return _mean + _rng.normal(0, 0.1, size=(n_samp, n_time))

            p_vals = predictive_pvalue(observed, sampler, n_samples=n_samples)
            p_values_collected.extend(p_vals)

        # Test that collected p-values are uniformly distributed
        # Using Kolmogorov-Smirnov test against uniform(0, 1)
        ks_stat, ks_pvalue = kstest(p_values_collected, "uniform")

        # We should NOT reject null hypothesis (p-values are uniform)
        # Using alpha=0.01 for robustness
        assert ks_pvalue > 0.01, (
            f"P-values should be uniform under null hypothesis. "
            f"KS test p-value: {ks_pvalue:.4f} (should be > 0.01)"
        )

    def test_sampler_reproducibility(self):
        """Test that sampler with fixed seed produces reproducible results."""
        n_time = 10
        rng = np.random.default_rng(42)
        observed = rng.standard_normal(n_time)

        # Sampler with fixed internal seed
        def sampler(n_samples):
            rng = np.random.default_rng(123)
            return rng.normal(size=(n_samples, n_time))

        # Compute twice - should be identical due to sampler's internal seed
        p_values_1 = predictive_pvalue(observed, sampler, n_samples=100)
        p_values_2 = predictive_pvalue(observed, sampler, n_samples=100)

        np.testing.assert_array_equal(p_values_1, p_values_2)

    def test_different_sampler_seeds_give_different_results(self):
        """Test that samplers with different seeds give different results."""
        n_time = 10
        rng = np.random.default_rng(42)
        observed = rng.standard_normal(n_time)

        # Sampler with seed 42
        def sampler1(n_samples):
            rng = np.random.default_rng(42)
            return rng.normal(size=(n_samples, n_time))

        # Sampler with seed 99
        def sampler2(n_samples):
            rng = np.random.default_rng(99)
            return rng.normal(size=(n_samples, n_time))

        p_values_1 = predictive_pvalue(observed, sampler1, n_samples=100)
        p_values_2 = predictive_pvalue(observed, sampler2, n_samples=100)

        # Should be different (with very high probability)
        assert not np.allclose(p_values_1, p_values_2)

    def test_output_shape_matches_input(self):
        """Test that output shape is (n_time,)."""
        n_time = 15
        rng = np.random.default_rng(42)
        observed = rng.standard_normal(n_time)

        def sampler(n_samples):
            return rng.standard_normal((n_samples, n_time))

        p_values = predictive_pvalue(observed, sampler, n_samples=50)

        assert p_values.shape == (n_time,)
        assert p_values.ndim == 1

    def test_callable_interface_works(self):
        """Test that callable interface for sampler works correctly."""
        n_time = 10
        observed = np.zeros(n_time)  # Observed = 0

        # Sampler that returns values > 0 (so p-value should be ~1.0)
        def sampler(n_samples):
            return np.ones((n_samples, n_time)) * 2.0  # All samples > observed

        p_values = predictive_pvalue(observed, sampler, n_samples=100)

        # Since all samples > observed, p-values should be close to 1.0
        assert np.all(p_values > 0.9)

    def test_extreme_pvalue_zero(self):
        """Test that p-value can be 0 when observed is extreme."""
        n_time = 5
        observed = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # Very high

        # Sampler returns small values
        def sampler(n_samples):
            return np.zeros((n_samples, n_time))

        p_values = predictive_pvalue(observed, sampler, n_samples=100)

        # Since no samples >= observed, p-value should be 0
        np.testing.assert_array_equal(p_values, 0.0)

    def test_extreme_pvalue_one(self):
        """Test that p-value can be 1 when observed is extreme."""
        n_time = 5
        observed = np.array([-10.0, -10.0, -10.0, -10.0, -10.0])  # Very low

        # Sampler returns large values
        def sampler(n_samples):
            return np.ones((n_samples, n_time)) * 10.0

        p_values = predictive_pvalue(observed, sampler, n_samples=100)

        # Since all samples >= observed, p-value should be 1.0
        np.testing.assert_array_equal(p_values, 1.0)

    def test_observed_wrong_dimensions_error(self):
        """Test that non-1D observed raises ValueError."""
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D

        def sampler(n_samples):
            rng = np.random.default_rng(42)
            return rng.standard_normal((n_samples, 4))

        with pytest.raises(ValueError, match="observed_log_pred must be 1-dimensional"):
            predictive_pvalue(observed, sampler)

    def test_n_samples_validation_error(self):
        """Test that n_samples <= 0 raises ValueError."""
        observed = np.array([1.0, 2.0, 3.0])

        def sampler(n_samples):
            rng = np.random.default_rng(42)
            return rng.standard_normal((n_samples, 3))

        with pytest.raises(ValueError, match="n_samples must be positive"):
            predictive_pvalue(observed, sampler, n_samples=0)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            predictive_pvalue(observed, sampler, n_samples=-10)

    def test_sampler_not_callable_error(self):
        """Test that non-callable sampler raises TypeError."""
        observed = np.array([1.0, 2.0, 3.0])
        not_callable = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(TypeError, match="sample_log_pred must be callable"):
            predictive_pvalue(observed, not_callable)

    def test_sampler_wrong_output_shape_error(self):
        """Test that sampler returning wrong shape raises ValueError."""
        observed = np.array([1.0, 2.0, 3.0])

        # Sampler returns wrong shape
        def bad_sampler(n_samples):
            rng = np.random.default_rng(42)
            return rng.standard_normal((n_samples, 5))  # Should be 3, not 5

        with pytest.raises(
            ValueError,
            match="sample_log_pred output must have shape \\(n_samples, n_time\\)",
        ):
            predictive_pvalue(observed, bad_sampler, n_samples=10)

    def test_moderate_pvalue(self):
        """Test realistic case where p-value is moderate."""
        n_time = 10
        observed = np.zeros(n_time)

        # Sampler returns standard normal (mean 0, std 1)
        def sampler(n_samples):
            rng = np.random.default_rng(42)
            return rng.normal(0, 1, size=(n_samples, n_time))

        p_values = predictive_pvalue(observed, sampler, n_samples=1000)

        # With observed = 0 and samples ~ N(0,1), p-value should be ~0.5
        # Allow range [0.4, 0.6] with large n_samples
        assert np.all(p_values > 0.4)
        assert np.all(p_values < 0.6)
