"""Tests for aggregate_over_period function (generic period aggregation utility)."""

import numpy as np
import pytest

from statespacecheck.periods import aggregate_over_period


class TestAggregateOverPeriod:
    """Test suite for aggregate_over_period function."""

    def test_mean_reduction_basic(self):
        """Test mean reduction with simple values."""
        metric_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time_mask = np.array([True, True, True, False, False])

        result = aggregate_over_period(metric_values, time_mask, reduction="mean")

        # Mean of [1, 2, 3] = 2.0
        assert isinstance(result, float)
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_sum_reduction_basic(self):
        """Test sum reduction with simple values."""
        metric_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time_mask = np.array([True, True, True, False, False])

        result = aggregate_over_period(metric_values, time_mask, reduction="sum")

        # Sum of [1, 2, 3] = 6.0
        assert isinstance(result, float)
        np.testing.assert_allclose(result, 6.0, rtol=1e-10)

    def test_weighted_mean(self):
        """Test weighted mean with custom weights."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])
        weights = np.array([1.0, 2.0, 1.0])  # Weight middle value more

        result = aggregate_over_period(metric_values, time_mask, reduction="mean", weights=weights)

        # Weighted mean: (1*1 + 2*2 + 3*1) / (1 + 2 + 1) = 8/4 = 2.0
        assert isinstance(result, float)
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_default_reduction_is_mean(self):
        """Test that default reduction is 'mean'."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])

        # Without specifying reduction
        result_default = aggregate_over_period(metric_values, time_mask)

        # Explicitly specifying mean
        result_mean = aggregate_over_period(metric_values, time_mask, reduction="mean")

        np.testing.assert_allclose(result_default, result_mean, rtol=1e-10)

    def test_all_false_mask_returns_nan(self):
        """Test that all-false mask returns NaN."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([False, False, False])

        result = aggregate_over_period(metric_values, time_mask)

        assert np.isnan(result)

    def test_partial_mask(self):
        """Test with partial mask selecting subset of time points."""
        metric_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        time_mask = np.array([False, True, False, True, False])

        result = aggregate_over_period(metric_values, time_mask, reduction="mean")

        # Mean of [20, 40] = 30.0
        np.testing.assert_allclose(result, 30.0, rtol=1e-10)

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            aggregate_over_period(metric_values, time_mask)

    def test_invalid_reduction_error(self):
        """Test that invalid reduction raises ValueError."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])

        with pytest.raises(ValueError, match="reduction must be 'mean' or 'sum'"):
            aggregate_over_period(metric_values, time_mask, reduction="invalid")

    def test_negative_weights_error(self):
        """Test that negative weights raise ValueError."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])
        weights = np.array([1.0, -1.0, 1.0])  # Negative weight

        with pytest.raises(ValueError, match="weights must be non-negative"):
            aggregate_over_period(metric_values, time_mask, weights=weights)

    def test_weights_shape_mismatch_error(self):
        """Test that weights shape mismatch raises ValueError."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])
        weights = np.array([1.0, 2.0])  # Different length

        with pytest.raises(ValueError, match="weights must have same length"):
            aggregate_over_period(metric_values, time_mask, weights=weights)

    def test_multidimensional_metric_error(self):
        """Test that multidimensional metric_values raise ValueError."""
        metric_values = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D
        time_mask = np.array([True, True])

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            aggregate_over_period(metric_values, time_mask)

    def test_warns_weights_with_sum(self):
        """Test that using weights with sum reduction issues warning."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])
        weights = np.array([1.0, 2.0, 1.0])

        with pytest.warns(UserWarning, match="weights are ignored when reduction='sum'"):
            aggregate_over_period(metric_values, time_mask, reduction="sum", weights=weights)

    def test_single_time_point(self):
        """Test with single time point selected."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([False, True, False])

        result = aggregate_over_period(metric_values, time_mask)

        # Mean of [2.0] = 2.0
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_handles_inf_in_metrics(self):
        """Test handling of inf values in metric_values."""
        metric_values = np.array([1.0, np.inf, 3.0])
        time_mask = np.array([True, True, True])

        result = aggregate_over_period(metric_values, time_mask)

        # Mean of [1, inf, 3] = inf
        assert np.isinf(result)

    def test_handles_nan_in_metrics(self):
        """Test handling of NaN values in metric_values."""
        metric_values = np.array([1.0, np.nan, 3.0])
        time_mask = np.array([True, True, True])

        result = aggregate_over_period(metric_values, time_mask)

        # Mean with NaN = NaN
        assert np.isnan(result)

    def test_weighted_mean_with_all_zero_weights(self):
        """Test that all-zero weights return NaN."""
        metric_values = np.array([1.0, 2.0, 3.0])
        time_mask = np.array([True, True, True])
        weights = np.array([0.0, 0.0, 0.0])  # All zeros

        result = aggregate_over_period(metric_values, time_mask, reduction="mean", weights=weights)

        # All-zero weights should return NaN (undefined weighted mean)
        assert np.isnan(result)
