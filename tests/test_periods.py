"""Tests for period detection and aggregation functions."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck.periods import (
    _contiguous_runs,
    _enforce_min_len,
    _robust_zscore,
    aggregate_over_period,
    combine_flags,
    find_low_overlap_intervals,
    flag_extreme_kl,
    flag_extreme_pvalues,
    flag_low_overlap,
)


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


class TestContiguousRuns:
    """Test _contiguous_runs helper function."""

    def test_single_run(self) -> None:
        """Test detection of a single contiguous run."""
        mask = np.array([False, True, True, True, False])
        runs = _contiguous_runs(mask)
        assert runs == [(1, 4)]

    def test_multiple_runs(self) -> None:
        """Test detection of multiple runs."""
        mask = np.array([True, True, False, True, False, False, True])
        runs = _contiguous_runs(mask)
        assert runs == [(0, 2), (3, 4), (6, 7)]

    def test_all_false(self) -> None:
        """Test with no True values."""
        mask = np.array([False, False, False])
        runs = _contiguous_runs(mask)
        assert runs == []

    def test_all_true(self) -> None:
        """Test with all True values."""
        mask = np.array([True, True, True])
        runs = _contiguous_runs(mask)
        assert runs == [(0, 3)]

    def test_empty_array(self) -> None:
        """Test with empty array."""
        mask = np.array([], dtype=bool)
        runs = _contiguous_runs(mask)
        assert runs == []

    def test_requires_1d(self) -> None:
        """Test that 2D arrays raise ValueError."""
        mask = np.array([[True, False], [False, True]])
        with pytest.raises(ValueError, match="mask must be 1D"):
            _contiguous_runs(mask)


class TestEnforceMinLen:
    """Test _enforce_min_len helper function."""

    def test_remove_short_runs(self) -> None:
        """Test that runs shorter than min_len are removed."""
        mask = np.array([True, False, True, True, True, False, True])
        result = _enforce_min_len(mask, min_len=2)
        expected = np.array([False, False, True, True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_keep_long_runs(self) -> None:
        """Test that runs >= min_len are kept."""
        mask = np.array([True, True, True, False, True, True])
        result = _enforce_min_len(mask, min_len=2)
        expected = np.array([True, True, True, False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_min_len_one(self) -> None:
        """Test with min_len=1 keeps all runs."""
        mask = np.array([True, False, True, False, True])
        result = _enforce_min_len(mask, min_len=1)
        np.testing.assert_array_equal(result, mask)

    def test_empty_array(self) -> None:
        """Test with empty array."""
        mask = np.array([], dtype=bool)
        result = _enforce_min_len(mask, min_len=5)
        np.testing.assert_array_equal(result, mask)


class TestRobustZscore:
    """Test _robust_zscore helper function."""

    def test_standard_normal(self) -> None:
        """Test with standard normal-like data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        z = _robust_zscore(x)
        # For normal data, median-based z should be close to standard z
        assert np.abs(np.median(z)) < 0.1
        assert 0.8 < np.std(z) < 1.2

    def test_with_nans(self) -> None:
        """Test that NaN values are preserved."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        z = _robust_zscore(x)
        assert np.isnan(z[2])
        assert np.all(np.isfinite(z[[0, 1, 3, 4]]))

    def test_with_infs(self) -> None:
        """Test that Inf values result in NaN z-scores."""
        x = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        z = _robust_zscore(x)
        assert np.isnan(z[2])

    def test_constant_array(self) -> None:
        """Test with constant array (zero MAD)."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        z = _robust_zscore(x)
        # Should handle gracefully with fallback to IQR
        assert np.all(np.isfinite(z))

    def test_all_nan(self) -> None:
        """Test with all NaN values."""
        x = np.array([np.nan, np.nan, np.nan])
        z = _robust_zscore(x)
        assert np.all(np.isnan(z))

    def test_single_finite_value(self) -> None:
        """Test with only one finite value."""
        x = np.array([np.nan, 3.0, np.nan])
        z = _robust_zscore(x)
        # With only one value, z-score should be 0
        assert z[1] == 0.0
        assert np.isnan(z[0]) and np.isnan(z[2])


class TestFlagLowOverlap:
    """Test flag_low_overlap function."""

    def test_basic_flagging(self) -> None:
        """Test basic low overlap flagging."""
        overlap = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8])
        flags = flag_low_overlap(overlap, threshold=0.4, min_len=5)
        expected = np.array([False, False, True, True, True, True, True, False])
        np.testing.assert_array_equal(flags, expected)

    def test_no_flags(self) -> None:
        """Test when all overlap is above threshold."""
        overlap = np.array([0.8, 0.9, 0.7, 0.6, 0.85])
        flags = flag_low_overlap(overlap, threshold=0.4, min_len=5)
        np.testing.assert_array_equal(flags, np.zeros(5, dtype=bool))

    def test_short_runs_filtered(self) -> None:
        """Test that short runs are filtered out."""
        overlap = np.array([0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
        flags = flag_low_overlap(overlap, threshold=0.4, min_len=5)
        # First run (length 2) should be filtered, second (length 5) kept
        expected = np.array([False, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(flags, expected)

    def test_with_nan_values(self) -> None:
        """Test that NaN values are not flagged."""
        overlap = np.array([0.2, 0.2, np.nan, 0.2, 0.2, 0.2])
        flags = flag_low_overlap(overlap, threshold=0.4, min_len=3)
        # NaN should not be flagged, breaks run
        assert not flags[2]

    def test_different_threshold(self) -> None:
        """Test with different threshold values."""
        overlap = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        flags_low = flag_low_overlap(overlap, threshold=0.4, min_len=3)
        flags_high = flag_low_overlap(overlap, threshold=0.6, min_len=3)
        np.testing.assert_array_equal(flags_low, np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(flags_high, np.ones(5, dtype=bool))


class TestFindLowOverlapIntervals:
    """Test find_low_overlap_intervals function."""

    def test_single_interval(self) -> None:
        """Test detection of a single low overlap interval."""
        overlap = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8])
        intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=5)
        assert intervals == [(2, 7)]

    def test_multiple_intervals(self) -> None:
        """Test detection of multiple intervals."""
        overlap = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
        intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=5)
        assert intervals == [(0, 5), (6, 11)]

    def test_no_intervals(self) -> None:
        """Test when all overlap is above threshold."""
        overlap = np.array([0.8, 0.9, 0.7, 0.6, 0.85])
        intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=5)
        assert intervals == []

    def test_short_runs_filtered(self) -> None:
        """Test that short runs are filtered out."""
        overlap = np.array([0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
        intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=5)
        # First run (length 2) should be filtered, second (length 5) kept
        assert intervals == [(3, 8)]

    def test_with_nan_values(self) -> None:
        """Test that NaN values are not flagged."""
        overlap = np.array([0.2, 0.2, np.nan, 0.2, 0.2, 0.2])
        intervals = find_low_overlap_intervals(overlap, threshold=0.4, min_len=3)
        # NaN breaks the run into two parts: [0:2] length 2 (too short), [3:6] length 3 (kept)
        assert intervals == [(3, 6)]

    def test_different_threshold(self) -> None:
        """Test with different threshold values."""
        overlap = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        intervals_low = find_low_overlap_intervals(overlap, threshold=0.4, min_len=3)
        intervals_high = find_low_overlap_intervals(overlap, threshold=0.6, min_len=3)
        assert intervals_low == []
        assert intervals_high == [(0, 5)]


class TestFlagExtremeKL:
    """Test flag_extreme_kl function."""

    def test_normal_values_not_flagged(self) -> None:
        """Test that normal KL values are not flagged."""
        rng = np.random.default_rng(42)
        kl = np.abs(rng.standard_normal(100)) * 0.5 + 1.0
        flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
        # Most values should not be flagged
        assert np.sum(flags) < len(flags) * 0.1

    def test_extreme_values_flagged(self) -> None:
        """Test that extreme KL values are flagged."""
        kl = np.ones(20)
        kl[5:10] = 100.0  # Extreme spike
        flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
        assert np.sum(flags[5:10]) == 5

    def test_short_spikes_filtered(self) -> None:
        """Test that short extreme spikes are filtered out."""
        kl = np.ones(20)
        kl[5:7] = 100.0  # Short spike (length 2)
        flags = flag_extreme_kl(kl, z_thresh=3.0, min_len=5)
        assert np.sum(flags) == 0

    def test_with_nan_values(self) -> None:
        """Test that NaN values are not flagged."""
        kl = np.array([1.0, 2.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0])
        flags = flag_extreme_kl(kl, z_thresh=2.0, min_len=5)
        # NaN should not be flagged
        assert not flags[2]

    def test_with_inf_values(self) -> None:
        """Test that Inf values are not flagged."""
        kl = np.array([1.0, 2.0, np.inf, 100.0, 100.0, 100.0, 100.0, 100.0])
        flags = flag_extreme_kl(kl, z_thresh=2.0, min_len=5)
        # Inf should not be flagged
        assert not flags[2]

    def test_different_threshold(self) -> None:
        """Test with different z_thresh values."""
        kl = np.ones(20)
        kl[5:15] = 10.0
        flags_strict = flag_extreme_kl(kl, z_thresh=5.0, min_len=5)
        flags_lenient = flag_extreme_kl(kl, z_thresh=1.0, min_len=5)
        assert np.sum(flags_strict) <= np.sum(flags_lenient)


class TestFlagExtremePvalues:
    """Test flag_extreme_pvalues function."""

    def test_normal_pvalues_not_flagged(self) -> None:
        """Test that p-values in middle range are not flagged."""
        p = np.linspace(0.1, 0.9, 20)
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert np.sum(flags) == 0

    def test_low_pvalues_flagged(self) -> None:
        """Test that very low p-values are flagged."""
        p = np.ones(20) * 0.5
        p[5:10] = 0.01  # Very low p-values
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert np.sum(flags[5:10]) == 5

    def test_high_pvalues_flagged(self) -> None:
        """Test that very high p-values are flagged."""
        p = np.ones(20) * 0.5
        p[5:10] = 0.99  # Very high p-values
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert np.sum(flags[5:10]) == 5

    def test_both_extremes_flagged(self) -> None:
        """Test that both low and high extremes are flagged."""
        p = np.ones(30) * 0.5
        p[5:10] = 0.01  # Low
        p[20:25] = 0.99  # High
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert np.sum(flags[5:10]) == 5
        assert np.sum(flags[20:25]) == 5

    def test_short_runs_filtered(self) -> None:
        """Test that short extreme runs are filtered out."""
        p = np.ones(20) * 0.5
        p[5:7] = 0.01  # Short run (length 2)
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert np.sum(flags) == 0

    def test_with_nan_values(self) -> None:
        """Test that NaN values are not flagged."""
        p = np.array([0.5, 0.5, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01])
        flags = flag_extreme_pvalues(p, alpha=0.05, min_len=5)
        assert not flags[2]

    def test_different_alpha(self) -> None:
        """Test with different alpha values."""
        p = np.ones(20) * 0.5
        p[5:10] = 0.03
        flags_strict = flag_extreme_pvalues(p, alpha=0.01, min_len=5)
        flags_lenient = flag_extreme_pvalues(p, alpha=0.1, min_len=5)
        assert np.sum(flags_strict) == 0
        assert np.sum(flags_lenient) == 5


class TestCombineFlags:
    """Test combine_flags function."""

    def test_unanimous_agreement(self) -> None:
        """Test when all methods agree."""
        flag1 = np.array([True, True, True, True, True, False, False])
        flag2 = np.array([True, True, True, True, True, False, False])
        flag3 = np.array([True, True, True, True, True, False, False])
        combined = combine_flags(flag1, flag2, flag3, min_votes=2, min_len=3)
        np.testing.assert_array_equal(combined, flag1)

    def test_majority_vote(self) -> None:
        """Test majority voting with 2/3 agreement."""
        flag1 = np.array([True, True, True, True, True, False, False])
        flag2 = np.array([True, True, True, True, True, False, False])
        flag3 = np.array([False, False, False, False, False, True, True])
        combined = combine_flags(flag1, flag2, flag3, min_votes=2, min_len=3)
        expected = np.array([True, True, True, True, True, False, False])
        np.testing.assert_array_equal(combined, expected)

    def test_partial_overlap(self) -> None:
        """Test with partial overlap between methods."""
        flag1 = np.array([True, True, True, False, False, False, False, False])
        flag2 = np.array([False, True, True, True, False, False, False, False])
        flag3 = np.array([False, False, True, True, True, False, False, False])
        combined = combine_flags(flag1, flag2, flag3, min_votes=2, min_len=2)
        expected = np.array([False, True, True, True, False, False, False, False])
        np.testing.assert_array_equal(combined, expected)

    def test_no_agreement(self) -> None:
        """Test when no time points have min_votes agreement."""
        flag1 = np.array([True, False, False, False, False])
        flag2 = np.array([False, True, False, False, False])
        flag3 = np.array([False, False, True, False, False])
        combined = combine_flags(flag1, flag2, flag3, min_votes=2, min_len=1)
        np.testing.assert_array_equal(combined, np.zeros(5, dtype=bool))

    def test_single_flag_array(self) -> None:
        """Test with only one flag array."""
        flag = np.array([True, True, True, False, False])
        combined = combine_flags(flag, min_votes=1, min_len=2)
        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(combined, expected)

    def test_short_runs_filtered(self) -> None:
        """Test that short runs are filtered in combined output."""
        flag1 = np.array([True, True, False, True, True, True, True, True])
        flag2 = np.array([True, True, False, True, True, True, True, True])
        combined = combine_flags(flag1, flag2, min_votes=2, min_len=5)
        # First run (length 2) should be filtered
        expected = np.array([False, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(combined, expected)

    def test_no_flags_provided(self) -> None:
        """Test that error is raised when no flags provided."""
        with pytest.raises(ValueError, match="at least one flag array"):
            combine_flags(min_votes=2, min_len=5)

    def test_mismatched_lengths(self) -> None:
        """Test that error is raised for mismatched array lengths."""
        flag1 = np.array([True, True, False])
        flag2 = np.array([True, True, False, False])
        with pytest.raises(ValueError, match="matching length"):
            combine_flags(flag1, flag2, min_votes=2, min_len=1)

    def test_different_min_votes(self) -> None:
        """Test with different min_votes thresholds."""
        flag1 = np.array([True, True, True, True, True])
        flag2 = np.array([True, True, True, True, True])
        flag3 = np.array([False, False, False, False, False])
        # With min_votes=2, should get True
        combined2 = combine_flags(flag1, flag2, flag3, min_votes=2, min_len=3)
        # With min_votes=3, should get False (only 2/3 agree)
        combined3 = combine_flags(flag1, flag2, flag3, min_votes=3, min_len=3)
        np.testing.assert_array_equal(combined2, flag1)
        np.testing.assert_array_equal(combined3, np.zeros(5, dtype=bool))
