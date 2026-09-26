"""Tests for the per-event (marked point-process) diagnostics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statespacecheck import (
    EventDiagnostics,
    EventFlags,
    baseline_threshold,
    event_diagnostics,
    event_likelihood,
    event_weighted_predictive,
    flag_events,
    hpd_overlap,
    kl_divergence,
    mark_predictive_pvalue,
    predictive_mark_probabilities,
)


@pytest.fixture
def random_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random predictive distributions, intensity table, and events."""
    rng = np.random.default_rng(1)
    n_time, n_bins, n_marks, n_events = 30, 12, 5, 40
    predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
    intensities = rng.random((n_bins, n_marks))
    time_ind = rng.integers(0, n_time, size=n_events)
    marks = rng.integers(0, n_marks, size=n_events)
    return predictive, intensities, time_ind, marks


class TestEventLikelihood:
    def test_matches_normalized_intensity_and_rows_sum_to_one(self):
        intensities = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.2]])
        out = event_likelihood(intensities)
        assert_allclose(out, intensities / intensities.sum(axis=1, keepdims=True))
        assert_allclose(out.sum(axis=1), 1.0)

    def test_scale_invariant(self):
        """A common bin width (rates vs expected counts) cancels."""
        intensities = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.2]])
        assert_allclose(event_likelihood(17.0 * intensities), event_likelihood(intensities))

    def test_tiny_intensities_keep_their_shape(self):
        out = event_likelihood(np.array([[1e-20, 2e-20, 4e-20]]))
        assert_allclose(out[0], np.array([1.0, 2.0, 4.0]) / 7.0, rtol=1e-6)

    def test_normalizes_over_all_spatial_axes(self):
        intensities = np.arange(1.0, 13.0).reshape(2, 2, 3)  # (n_events, n_x, n_y)
        out = event_likelihood(intensities)
        assert out.shape == intensities.shape
        assert_allclose(out.sum(axis=(1, 2)), 1.0)
        assert_allclose(out[1], intensities[1] / intensities[1].sum())

    def test_zero_row_raises(self):
        with pytest.raises(ValueError, match="zero everywhere"):
            event_likelihood(np.array([[2.0, 0.5, 1.0], [0.0, 0.0, 0.0]]))

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_invalid_values_raise(self, bad):
        with pytest.raises(ValueError, match="finite nonnegative"):
            event_likelihood(np.array([[1.0, bad]]))

    def test_returns_float64_for_other_float_inputs(self):
        likelihood = event_likelihood(np.array([[1.0, 2.0, 1.0]], dtype=np.float32))
        assert likelihood.dtype == np.float64

    def test_requires_event_axis(self):
        with pytest.raises(ValueError, match="n_events"):
            event_likelihood(np.array([1.0, 2.0]))


class TestPredictiveMarkProbabilities:
    def test_integrates_raw_intensities_before_normalizing(self):
        """The total event rate differs across states, so averaging each state's
        mark fractions would give [0.7, 0.3]; the event-weighted answer is
        [5/6, 1/6]."""
        state = np.array([[0.5, 0.5]])
        intensities = np.array([[9.0, 1.0], [1.0, 1.0]])
        assert_allclose(predictive_mark_probabilities(state, intensities), [[5 / 6, 1 / 6]])

    def test_scale_invariant(self):
        state = np.array([[0.5, 0.3, 0.2]])
        intensities = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
        assert_allclose(
            predictive_mark_probabilities(state, 17.0 * intensities),
            predictive_mark_probabilities(state, intensities),
        )

    def test_two_dimensional_state_matches_flattened(self):
        rng = np.random.default_rng(0)
        state = rng.dirichlet(np.ones(6), size=4).reshape(4, 2, 3)
        intensities = rng.random((2, 3, 5))
        assert_allclose(
            predictive_mark_probabilities(state, intensities),
            predictive_mark_probabilities(state.reshape(4, 6), intensities.reshape(6, 5)),
        )

    def test_zero_total_intensity_row_raises(self):
        state = np.array([[1.0, 0.0], [0.0, 1.0]])
        intensities = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="zero total"):
            predictive_mark_probabilities(state, intensities)

    def test_overflow_in_total_raises(self):
        with pytest.raises(ValueError, match="total event intensity is non-finite"):
            predictive_mark_probabilities(np.array([[0.5, 0.5]]), np.full((2, 2), 1e308))

    def test_overflow_in_product_raises(self):
        with pytest.raises(ValueError, match="expected mark intensities are non-finite"):
            predictive_mark_probabilities(np.array([[1.0, 1.0]]), np.full((2, 1), 1e308))

    @pytest.mark.parametrize("bad", [-0.5, np.nan, np.inf])
    def test_invalid_state_values_raise(self, bad):
        with pytest.raises(
            ValueError, match="state_dist must contain only finite nonnegative"
        ):
            predictive_mark_probabilities(np.array([[1.5, bad]]), np.ones((2, 2)))

    @pytest.mark.parametrize("bad", [-0.5, np.nan, np.inf])
    def test_invalid_intensity_values_raise(self, bad):
        intensities = np.array([[1.0, bad], [1.0, 1.0]])
        with pytest.raises(
            ValueError, match="mark_intensities must contain only finite nonnegative"
        ):
            predictive_mark_probabilities(np.full((1, 2), 0.5), intensities)

    def test_no_marks_raises(self):
        with pytest.raises(ValueError, match="at least one mark"):
            predictive_mark_probabilities(np.full((1, 2), 0.5), np.ones((2, 0)))

    def test_state_without_spatial_axis_raises(self):
        with pytest.raises(
            ValueError, match=r"state_dist must have shape \(n_events, \.\.\.\)"
        ):
            predictive_mark_probabilities(np.array([0.5, 0.5]), np.ones((2, 2)))

    def test_spatial_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="mark_intensities must have shape"):
            predictive_mark_probabilities(np.array([[0.5, 0.5]]), np.ones((3, 2)))


class TestEventWeightedPredictive:
    def test_constant_ground_intensity_normalizes_state(self):
        rng = np.random.default_rng(0)
        state = rng.random((5, 7))
        weighted = event_weighted_predictive(state, np.full(7, 3.5))
        assert_allclose(weighted, state / state.sum(axis=1, keepdims=True), rtol=1e-12)

    def test_matches_weighting_by_ground_intensity_on_2d_grid(self):
        rng = np.random.default_rng(1)
        state = rng.random((6, 4, 3))
        ground = rng.random((4, 3)) * 5
        weighted = event_weighted_predictive(state, ground)
        expected = state * ground
        expected /= expected.sum(axis=(1, 2), keepdims=True)
        assert weighted.shape == state.shape
        assert_allclose(weighted, expected, rtol=1e-12)

    def test_sorted_marks_ground_intensity_gives_predictive_mark_probabilities(self):
        """P_event summed against each mark's share of the intensity is q."""
        rng = np.random.default_rng(2)
        state = rng.dirichlet(np.ones(9), size=4)
        rates = rng.gamma(2.0, size=(9, 5))
        ground = rates.sum(axis=-1)
        weighted = event_weighted_predictive(state, ground)
        assert_allclose(
            weighted @ (rates / ground[:, None]),
            predictive_mark_probabilities(state, rates),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        ("state", "ground", "match"),
        [
            (np.ones((2, 3)), np.ones(4), "ground_intensity must have shape"),
            (np.ones((2, 3)), -np.ones(3), "ground_intensity must contain only finite"),
            (np.ones((2, 3)), np.array([1.0, np.inf, 1.0]), "ground_intensity must contain"),
            (-np.ones((2, 3)), np.ones(3), "state_dist must contain only finite"),
            (np.ones(3), np.ones(3), "state_dist must have shape"),
        ],
    )
    def test_invalid_input_raises(self, state, ground, match):
        with pytest.raises(ValueError, match=match):
            event_weighted_predictive(state, ground)

    def test_zero_total_rows_are_listed(self):
        state = np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        with pytest.raises(ValueError, match=r"row indices: \[0, 2\]"):
            event_weighted_predictive(state, np.array([0.0, 1.0]))


class TestMarkPredictivePvalue:
    def test_event_weighted_value(self):
        """Regression test for the normalization order: the less likely mark's
        p-value is 1/6, not 0.3."""
        state = np.array([[0.5, 0.5], [0.5, 0.5]])
        intensities = np.array([[9.0, 1.0], [1.0, 1.0]])
        assert_allclose(
            mark_predictive_pvalue(state, intensities, np.array([0, 1])), [1, 1 / 6]
        )

    def test_zero_rate_state_carries_no_event_mass(self):
        state = np.array([[0.2, 0.5, 0.3]])
        intensities = np.array([[0.5, 0.5], [0.0, 0.0], [0.5, 0.5]])
        assert_allclose(mark_predictive_pvalue(state, intensities, np.array([0])), 1.0)

    def test_near_ties_receive_equal_pvalues(self):
        """Marks whose predictive probabilities differ by less than the
        floating-point tolerance tie instead of splitting."""
        n_bins = 4
        delta = 1e-15
        probabilities = np.array([0.05, 0.30, 0.30 + delta, 0.35 - delta])
        assert delta < np.finfo(float).eps * n_bins * 16 * probabilities.max()
        intensities = np.zeros((n_bins, 4))
        intensities[0] = probabilities
        state = np.zeros((2, n_bins))
        state[:, 0] = 1.0
        pvalue = mark_predictive_pvalue(state, intensities, np.array([1, 2]))
        assert pvalue[0] == pvalue[1]
        assert 0.0 < pvalue[0] < 1.0

    def test_pvalue_does_not_depend_on_other_events(self):
        """The tie tolerance is set by each event's own predictive probabilities,
        so batching or subsetting events cannot change a p-value."""
        state = np.eye(2)
        intensities = np.ones((2, 10))
        intensities[0, 1] = 1 + 3e-14  # nearly ties mark 0 at bin 0
        intensities[1, 2] = 1000.0  # at bin 1, mark 2 is far more probable
        marks = np.array([0, 2])
        together = mark_predictive_pvalue(state, intensities, marks)
        alone = [
            mark_predictive_pvalue(state[[i]], intensities, marks[[i]])[0] for i in (0, 1)
        ]
        assert_array_equal(together, alone)

    def test_values_in_unit_interval(self, random_model):
        predictive, intensities, time_ind, marks = random_model
        pvalue = mark_predictive_pvalue(predictive[time_ind], intensities, marks)
        assert np.all((pvalue >= 0.0) & (pvalue <= 1.0))

    def test_not_anticonservative_under_the_model(self):
        """Marks drawn from the predictive mark distribution give
        P(p <= alpha) <= alpha (a discrete p-value is conservative)."""
        rng = np.random.default_rng(3)
        n_bins, n_marks, n_events = 20, 8, 20_000
        state = np.tile(rng.dirichlet(np.ones(n_bins)), (n_events, 1))
        intensities = rng.gamma(0.5, size=(n_bins, n_marks))
        q = predictive_mark_probabilities(state[:1], intensities)[0]
        marks = rng.choice(n_marks, size=n_events, p=q)
        pvalue = mark_predictive_pvalue(state, intensities, marks)
        for alpha in (0.05, 0.1, 0.25, 0.5):
            # Allow three binomial standard errors of Monte Carlo slack.
            slack = 3 * np.sqrt(alpha * (1 - alpha) / n_events)
            assert np.mean(pvalue <= alpha) <= alpha + slack

    @pytest.mark.parametrize("marks", [np.array([0, 5]), np.array([-1, 0])])
    def test_out_of_range_marks_raise(self, marks):
        with pytest.raises(ValueError, match=r"observed_marks must lie in \[0, 3\)"):
            mark_predictive_pvalue(np.full((2, 2), 0.5), np.ones((2, 3)), marks)

    def test_non_integer_marks_raise(self):
        with pytest.raises(ValueError, match="observed_marks must be a 1-D integer array"):
            mark_predictive_pvalue(np.full((2, 2), 0.5), np.ones((2, 3)), np.array([0.0, 1.0]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="one entry per event"):
            mark_predictive_pvalue(np.full((2, 2), 0.5), np.ones((2, 3)), np.array([0]))


class TestEventDiagnostics:
    def test_matches_component_functions(self, random_model):
        predictive, intensities, time_ind, marks = random_model
        result = event_diagnostics(
            predictive, intensities, time_ind, marks, coverage=0.9, return_likelihood=True
        )
        likelihood = event_likelihood(intensities[:, marks].T)
        assert_array_equal(result.likelihood, likelihood)
        assert_array_equal(
            result.hpd_overlap, hpd_overlap(predictive[time_ind], likelihood, coverage=0.9)
        )
        assert_array_equal(
            result.kl_divergence, kl_divergence(predictive[time_ind], likelihood)
        )
        assert_array_equal(
            result.predictive_pvalue,
            mark_predictive_pvalue(predictive[time_ind], intensities, marks),
        )

    def test_batch_size_does_not_change_results(self, random_model):
        predictive, intensities, time_ind, marks = random_model
        whole = event_diagnostics(predictive, intensities, time_ind, marks)
        batched = event_diagnostics(predictive, intensities, time_ind, marks, batch_size=7)
        assert_array_equal(batched.hpd_overlap, whole.hpd_overlap)
        assert_array_equal(batched.kl_divergence, whole.kl_divergence)
        # The matrix product's reduction order depends on the number of rows, so
        # the p-value may differ in the last bit between batch sizes.
        assert_allclose(batched.predictive_pvalue, whole.predictive_pvalue, rtol=0, atol=1e-12)

    def test_likelihood_omitted_by_default(self, random_model):
        predictive, intensities, time_ind, marks = random_model
        assert event_diagnostics(predictive, intensities, time_ind, marks).likelihood is None

    def test_two_dimensional_state_space(self):
        rng = np.random.default_rng(2)
        predictive = rng.dirichlet(np.ones(12), size=5)
        intensities = rng.random((12, 3))
        time_ind, marks = np.array([0, 2, 4]), np.array([2, 0, 1])
        flat = event_diagnostics(
            predictive, intensities, time_ind, marks, return_likelihood=True
        )
        grid = event_diagnostics(
            predictive.reshape(5, 3, 4),
            intensities.reshape(3, 4, 3),
            time_ind,
            marks,
            return_likelihood=True,
        )
        assert grid.likelihood is not None
        assert grid.likelihood.shape == (3, 3, 4)
        assert_allclose(grid.likelihood.reshape(3, 12), flat.likelihood)
        for name in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            assert_allclose(getattr(grid, name), getattr(flat, name))

    def test_no_events(self):
        result = event_diagnostics(
            np.full((3, 2), 0.5),
            np.ones((2, 2)),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
        assert result.hpd_overlap.shape == (0,)

    def test_all_zero_intensity_for_observed_mark_raises(self):
        with pytest.raises(ValueError, match="zero everywhere"):
            event_diagnostics(
                np.full((5, 3), 1 / 3), np.zeros((3, 2)), np.array([0]), np.array([0])
            )

    @pytest.mark.parametrize("batch_size", [0, -3])
    def test_invalid_batch_size_raises(self, batch_size):
        # Without the check, a non-positive step would skip the batch loop and
        # return uninitialized arrays.
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            event_diagnostics(
                np.full((2, 2), 0.5),
                np.ones((2, 2)),
                np.array([0]),
                np.array([0]),
                batch_size=batch_size,
            )

    def test_predictive_without_spatial_axis_raises(self):
        with pytest.raises(ValueError, match=r"predictive must have shape \(n_time, \.\.\.\)"):
            event_diagnostics(
                np.array([0.5, 0.5]), np.ones((2, 2)), np.array([0]), np.array([0])
            )

    def test_out_of_range_time_index_raises(self):
        with pytest.raises(ValueError, match="event_time_ind"):
            event_diagnostics(
                np.full((2, 2), 0.5), np.ones((2, 2)), np.array([2]), np.array([0])
            )

    def test_mismatched_event_arrays_raise(self):
        with pytest.raises(ValueError, match="same length"):
            event_diagnostics(
                np.full((2, 2), 0.5), np.ones((2, 2)), np.array([0, 1]), np.array([0])
            )


class TestBaselineThreshold:
    def test_matches_nanquantile(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(0.5, 1.0, (50, 5))
        assert baseline_threshold(values, 0.01) == np.nanquantile(values.ravel(), 0.01)

    def test_ignores_nan(self):
        values = np.array([np.nan, 1.0, 2.0, 3.0])
        assert baseline_threshold(values, 0.5) == 2.0

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            baseline_threshold(np.full(4, np.nan), 0.5)

    def test_positive_infinity_can_be_the_threshold(self):
        """KL is +inf for disjoint supports; a high quantile can land on it."""
        assert baseline_threshold(np.array([1.0, 2.0, np.inf]), 0.99) == np.inf
        assert baseline_threshold(np.array([1.0, np.inf, np.inf]), 0.5) == np.inf

    def test_positive_infinity_above_the_quantile_is_ignored(self):
        """Below the infinite values, the threshold equals the usual quantile."""
        values = np.r_[np.arange(100.0), np.inf]
        assert baseline_threshold(values, 0.5) == np.quantile(np.arange(101.0), 0.5)
        assert baseline_threshold(values, 0.0) == 0.0
        # Between two finite values: interpolated, as np.quantile would without the inf
        assert baseline_threshold(values, 0.505) == 50.5

    @pytest.mark.parametrize(
        ("values", "quantile", "expected"),
        [
            (np.r_[np.arange(100.0), np.inf], 0.99, 99.0),  # exact position, inf above
            (np.array([1.0, np.inf]), 0.0, 1.0),
            (np.array([1.0, 2.0, np.inf]), 0.5, 2.0),
            (np.array([1.0, np.inf]), 1.0, np.inf),
        ],
    )
    def test_exact_positions_next_to_infinity(self, values, quantile, expected):
        """At an exact order-statistic position, +inf just above does not make it NaN."""
        assert baseline_threshold(values, quantile) == expected

    def test_negative_infinity_raises(self):
        with pytest.raises(ValueError, match="-inf"):
            baseline_threshold(np.array([1.0, -np.inf]), 0.5)

    def test_all_infinite_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            baseline_threshold(np.array([np.inf, np.inf]), 0.5)

    @pytest.mark.parametrize("quantile", [-0.1, 1.1])
    def test_invalid_quantile_raises(self, quantile):
        with pytest.raises(ValueError, match="quantile"):
            baseline_threshold(np.arange(3.0), quantile)


@pytest.fixture
def small_diagnostics() -> EventDiagnostics:
    return EventDiagnostics(
        hpd_overlap=np.array([0.0, 0.05, 0.5, np.nan]),
        kl_divergence=np.array([3.0, np.inf, 0.1, 2.0]),
        predictive_pvalue=np.array([0.05, 0.9, 0.01, 1.0]),
        likelihood=None,
    )


class TestFlagEvents:
    def test_paper_rule_is_inclusive(self, small_diagnostics):
        """HPD at or below, KL at or above, p at or below their thresholds."""
        flags = flag_events(
            small_diagnostics,
            hpd_overlap_threshold=0.05,
            kl_divergence_threshold=2.0,
            pvalue_threshold=0.05,
        )
        assert isinstance(flags, EventFlags)
        assert_array_equal(flags.hpd_overlap, [True, True, False, False])
        assert_array_equal(flags.kl_divergence, [True, True, False, True])
        assert_array_equal(flags.predictive_pvalue, [True, False, True, False])

    def test_unset_thresholds_skip_the_metric(self, small_diagnostics):
        """Only the p-value has a default cutoff (0.05); the others need a threshold."""
        flags = flag_events(small_diagnostics)
        assert flags.hpd_overlap is None
        assert flags.kl_divergence is None
        assert_array_equal(flags.predictive_pvalue, [True, False, True, False])

    def test_pvalue_can_be_skipped(self, small_diagnostics):
        flags = flag_events(small_diagnostics, pvalue_threshold=None)
        assert flags.predictive_pvalue is None

    def test_infinite_kl_threshold_flags_only_infinite(self, small_diagnostics):
        flags = flag_events(small_diagnostics, kl_divergence_threshold=np.inf)
        assert_array_equal(flags.kl_divergence, [False, True, False, False])

    def test_with_baseline_thresholds(self, random_model):
        """The paper's workflow: thresholds from a baseline, then flag every event."""
        diagnostics = event_diagnostics(*random_model)
        baseline = slice(0, 20)
        flags = flag_events(
            diagnostics,
            hpd_overlap_threshold=baseline_threshold(diagnostics.hpd_overlap[baseline], 0.01),
            kl_divergence_threshold=baseline_threshold(
                diagnostics.kl_divergence[baseline], 0.99
            ),
        )
        # The baseline's lowest overlap and highest divergence lie beyond its own
        # 1st and 99th percentiles, so they are flagged
        assert flags.hpd_overlap is not None
        assert flags.kl_divergence is not None
        assert flags.hpd_overlap[np.argmin(diagnostics.hpd_overlap[baseline])]
        assert flags.kl_divergence[np.argmax(diagnostics.kl_divergence[baseline])]

    @pytest.mark.parametrize(
        "threshold",
        ["hpd_overlap_threshold", "kl_divergence_threshold", "pvalue_threshold"],
    )
    def test_nan_threshold_raises(self, small_diagnostics, threshold):
        """A NaN threshold (for example np.quantile of values including +inf) would
        silently flag nothing."""
        with pytest.raises(ValueError, match=f"{threshold} is NaN"):
            flag_events(small_diagnostics, **{threshold: np.nan})


class TestEventDiagnosticsErrors:
    """Errors name the user's argument and point to absolute indices."""

    @pytest.fixture
    def model(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        predictive = rng.dirichlet(np.ones(6), size=10)  # (n_time=10, n_bins=6)
        fields = rng.gamma(2.0, size=(6, 4))  # (n_bins=6, n_marks=4)
        return predictive, fields

    def test_nan_predictive_names_the_argument_and_bin(self, model):
        predictive, fields = model
        predictive[7, 2] = np.nan
        with pytest.raises(ValueError, match=r"predictive .*time bins \[7\]"):
            event_diagnostics(predictive, fields, np.array([0, 7]), np.array([0, 1]))

    def test_zero_mass_time_bin_reported_by_absolute_indices(self, model):
        predictive, fields = model
        predictive[5] = 0.0
        time_ind = np.arange(10)
        with pytest.raises(ValueError, match=r"time bins \[5\].*events \[5\]"):
            event_diagnostics(predictive, fields, time_ind, np.zeros(10, int), batch_size=4)

    def test_mark_with_no_intensity_reported_by_absolute_indices(self, model):
        predictive, fields = model
        fields[:, 3] = 0.0
        marks = np.array([0, 1, 2, 0, 1, 2, 3])
        with pytest.raises(ValueError, match=r"zero everywhere.*marks \[3\].*events \[6\]"):
            event_diagnostics(predictive, fields, np.arange(7), marks, batch_size=2)

    def test_bins_and_marks_without_events_are_not_checked(self, model):
        """Decoder output often has invalid time bins and silent units where no event
        falls; only the time bins and marks that events use must be valid."""
        predictive, fields = model
        time_ind, marks = np.array([0, 2, 4]), np.array([0, 1, 2])
        clean = event_diagnostics(predictive, fields[:, :3], time_ind, marks)
        predictive[1] = np.nan
        predictive[3] = 0.0
        fields[:, 3] = 0.0  # a unit with zero rate everywhere, which never fires
        result = event_diagnostics(predictive, fields, time_ind, marks)
        for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            # The extra all-zero column can change the matrix product in the last bit
            assert_allclose(getattr(result, field), getattr(clean, field), rtol=1e-12)

    def test_transposed_mark_intensities_suggests_transpose(self, model):
        predictive, fields = model
        with pytest.raises(ValueError, match=r"mark_intensities\.T"):
            event_diagnostics(predictive, fields.T, np.array([0]), np.array([0]))

    def test_float_time_indices_suggest_binning(self, model):
        predictive, fields = model
        with pytest.raises(ValueError, match="time-bin indices"):
            event_diagnostics(predictive, fields, np.array([0.3, 1.7]), np.array([0, 1]))

    def test_empty_lists_are_accepted(self, model):
        predictive, fields = model
        result = event_diagnostics(predictive, fields, [], [])
        assert result.hpd_overlap.shape == (0,)


class TestNoEvents:
    """With no events, the building blocks return empty results, as event_diagnostics does."""

    def test_event_likelihood(self):
        assert event_likelihood(np.empty((0, 5))).shape == (0, 5)

    def test_predictive_mark_probabilities(self):
        assert predictive_mark_probabilities(np.empty((0, 5)), np.ones((5, 2))).shape == (0, 2)

    def test_mark_predictive_pvalue(self):
        pvalue = mark_predictive_pvalue(np.empty((0, 5)), np.ones((5, 2)), np.array([], int))
        assert pvalue.shape == (0,)

    def test_event_diagnostics_with_no_time_bins(self):
        diagnostics = event_diagnostics(np.empty((0, 5)), np.ones((5, 2)), [], [])
        assert diagnostics.hpd_overlap.shape == (0,)
