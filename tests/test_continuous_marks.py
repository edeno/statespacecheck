"""Tests for the Monte Carlo predictive check of continuous (or intractable) marks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp
from scipy.stats import kstest, norm

from statespacecheck import (
    MarkPredictiveCheck,
    mark_predictive_pvalue,
    monte_carlo_mark_pvalue,
    predictive_mark_probabilities,
)
from statespacecheck.continuous_marks import _sample_state_bins


def _binomial_se(p: np.ndarray, n: int) -> np.ndarray:
    return np.sqrt(p * (1 - p) / n)


@pytest.fixture(scope="module")
def discrete_events(discrete_mark_model):
    """40 events with random predictive distributions and observed units."""
    rates, _, _ = discrete_mark_model
    rng = np.random.default_rng(1)
    state = rng.dirichlet(np.full(rates.shape[0], 0.3), size=40)
    marks = rng.integers(0, rates.shape[1], size=40)
    return state, marks


def _discrete_check(discrete_mark_model, state, marks, **kwargs):
    rates, log_mark_intensity, sample_marks = discrete_mark_model
    return monte_carlo_mark_pvalue(
        state,
        log_mark_intensity,
        marks,
        ground_intensity=rates.sum(axis=-1),
        sample_marks=sample_marks,
        **kwargs,
    )


class TestSampleStateBins:
    def test_frequencies_match_probabilities(self):
        rng = np.random.default_rng(0)
        probabilities = rng.dirichlet(np.ones(6), size=3)
        n_samples = 200_000
        bins = _sample_state_bins(probabilities, n_samples, rng)
        assert bins.shape == (3, n_samples)
        for row, target in zip(bins, probabilities, strict=True):
            frequency = np.bincount(row, minlength=6) / n_samples
            assert np.all(np.abs(frequency - target) <= 5 * _binomial_se(target, n_samples))

    def test_zero_probability_bins_are_never_drawn(self):
        probabilities = np.array(
            [[0.0, 0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.7]]
        )
        bins = _sample_state_bins(probabilities, 50_000, np.random.default_rng(1))
        for row, target in zip(bins, probabilities, strict=True):
            assert np.all(target[row] > 0.0)


class TestAgainstExactDiscreteMarks:
    """For integer marks the Monte Carlo check estimates mark_predictive_pvalue."""

    @pytest.mark.slow
    def test_pvalue_matches_exact_within_monte_carlo_error(
        self, discrete_mark_model, discrete_events
    ):
        rates, _, _ = discrete_mark_model
        state, marks = discrete_events
        n_samples = 20_000
        check = _discrete_check(
            discrete_mark_model, state, marks, n_samples=n_samples, rng=2, batch_size=4
        )
        exact = mark_predictive_pvalue(state, rates, marks)
        allowed = 4 * _binomial_se(exact, n_samples) + 1 / n_samples
        assert np.all(np.abs(check.pvalue - exact) <= allowed)

    def test_observed_log_density_is_log_predictive_mark_probability(
        self, discrete_mark_model, discrete_events
    ):
        rates, _, _ = discrete_mark_model
        state, marks = discrete_events
        check = _discrete_check(discrete_mark_model, state, marks, n_samples=10, rng=0)
        q = predictive_mark_probabilities(state, rates)
        assert_allclose(
            check.observed_log_density, np.log(q[np.arange(40), marks]), rtol=1e-12
        )


@pytest.mark.slow
def test_calibrated_under_the_true_model(clusterless_1d_model):
    """Marks drawn from the model's own predictive give uniform p-values."""
    model = clusterless_1d_model
    rng = np.random.default_rng(3)
    n_events = 400
    centers = rng.uniform(0.1, 0.9, n_events)
    state = norm.pdf(model.position, centers[:, None], 0.08)
    state /= state.sum(axis=1, keepdims=True)
    # Each event's state is drawn from the predictive weighted by the total rate
    # (computed here independently of the package), then its mark at that state
    weights = state * model.ground_intensity
    weights /= weights.sum(axis=1, keepdims=True)
    state_bins = np.array([rng.choice(len(model.position), p=row) for row in weights])
    observed = model.sample_marks(state_bins, rng)

    check = monte_carlo_mark_pvalue(
        state,
        model.log_mark_intensity,
        observed,
        ground_intensity=model.ground_intensity,
        sample_marks=model.sample_marks,
        n_samples=2000,
        rng=4,
    )
    assert kstest(check.pvalue, "uniform").pvalue > 0.01


def test_figure2_scenario_matches_quadrature():
    """The paper's Figure 2 example: prediction N(35, 8), mark density N(y; x, 12),
    observed mark 60, constant ground intensity."""
    position = np.linspace(0.0, 100.0, 200)
    predictive = norm.pdf(position, 35.0, 8.0)
    predictive /= predictive.sum()
    sigma = 12.0

    def log_mark_intensity(marks):
        return norm.logpdf(np.asarray(marks)[:, :1], position, sigma)

    def sample_marks(bins, rng):
        return rng.normal(position[bins], sigma)[:, None]

    n_samples = 20_000
    check = monte_carlo_mark_pvalue(
        predictive[None],
        log_mark_intensity,
        np.array([[60.0]]),
        ground_intensity=np.ones_like(position),
        sample_marks=sample_marks,
        n_samples=n_samples,
        rng=5,
    )

    y = np.linspace(-100.0, 200.0, 20_001)
    f_pred = norm.pdf(y[:, None], position, sigma) @ predictive
    f_observed = norm.pdf(60.0, position, sigma) @ predictive
    reference = np.sum(f_pred * (f_pred <= f_observed)) * (y[1] - y[0])
    assert abs(check.pvalue[0] - reference) <= 4 * _binomial_se(reference, n_samples)


class TestReproducibility:
    def test_same_seed_and_batch_size_give_identical_results(
        self, discrete_mark_model, discrete_events
    ):
        state, marks = discrete_events
        first = _discrete_check(discrete_mark_model, state, marks, n_samples=200, rng=7)
        second = _discrete_check(discrete_mark_model, state, marks, n_samples=200, rng=7)
        assert_array_equal(first.pvalue, second.pvalue)
        other = _discrete_check(discrete_mark_model, state, marks, n_samples=200, rng=8)
        assert not np.array_equal(first.pvalue, other.pvalue)

    def test_batch_size_changes_only_the_draws(self, discrete_mark_model, discrete_events):
        state, marks = discrete_events
        n_samples = 4000
        one = _discrete_check(
            discrete_mark_model, state, marks, n_samples=n_samples, rng=9, batch_size=1
        )
        many = _discrete_check(
            discrete_mark_model, state, marks, n_samples=n_samples, rng=9, batch_size=32
        )
        assert_array_equal(one.observed_log_density, many.observed_log_density)
        p = (one.pvalue + many.pvalue) / 2
        # Two independent estimates: their difference has variance 2 p (1 - p) / n
        allowed = 5 * np.sqrt(2) * _binomial_se(p, n_samples) + 2 / n_samples
        assert np.all(np.abs(one.pvalue - many.pvalue) <= allowed)


def test_return_samples(discrete_mark_model, discrete_events):
    state, marks = discrete_events
    check = _discrete_check(
        discrete_mark_model, state, marks, n_samples=300, rng=10, return_samples=True
    )
    assert isinstance(check, MarkPredictiveCheck)
    assert check.simulated_log_density is not None
    assert check.simulated_log_density.shape == (40, 300)
    # Replicates as probable as the observed mark are exact ties here; any tolerance
    # between their rounding and the gaps between distinct marks' densities works
    recomputed = np.mean(
        check.simulated_log_density <= check.observed_log_density[:, None] + 1e-9, axis=1
    )
    assert_array_equal(check.pvalue, recomputed)
    without = _discrete_check(discrete_mark_model, state, marks, n_samples=300, rng=10)
    assert without.simulated_log_density is None
    assert_array_equal(without.pvalue, check.pvalue)


def test_impossible_observed_mark_gives_zero_pvalue():
    """A mark with zero intensity wherever the prediction has mass is maximally unexpected."""
    state = np.array([[0.5, 0.5, 0.0]])
    rates = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 5.0]])  # mark 1 only at bin 2

    def sample_marks(bins, _rng):
        return np.zeros(len(bins), dtype=int)  # at bins 0 and 1 only mark 0 occurs

    check = monte_carlo_mark_pvalue(
        state,
        _log_intensity_of(rates),
        np.array([1]),
        ground_intensity=rates.sum(axis=-1),
        sample_marks=sample_marks,
        n_samples=50,
        rng=0,
    )
    assert check.observed_log_density[0] == -np.inf
    assert check.pvalue[0] == 0.0


def _log_intensity_of(rates):
    """Log joint intensity of integer marks for a (n_bins, n_marks) rate table."""

    def log_mark_intensity(marks):
        with np.errstate(divide="ignore"):  # zero rates are impossible marks
            return np.log(rates[:, np.asarray(marks)].T)

    return log_mark_intensity


def _two_mark_check(state, rates, marks, n_samples=10_000):
    cumulative = np.cumsum(rates / rates.sum(axis=1, keepdims=True), axis=1)

    def sample_marks(bins, rng):
        return np.minimum((cumulative[bins] <= rng.random(len(bins))[:, None]).sum(axis=1), 1)

    return monte_carlo_mark_pvalue(
        state,
        _log_intensity_of(rates),
        np.asarray(marks),
        ground_intensity=rates.sum(axis=1),
        sample_marks=sample_marks,
        n_samples=n_samples,
        rng=0,
    )


class TestScale:
    """The p-value depends on the model, not on the scale of its numbers."""

    @pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
    def test_equal_density_marks_tie_at_any_intensity_scale(self, scale):
        """Both marks have predictive probability 0.5, so both p-values are 1."""
        rates = np.array([[0.2, 1.0], [1.8, 1.0]]) * scale
        check = _two_mark_check(np.full((2, 2), 0.5), rates, [0, 1])
        assert_array_equal(check.pvalue, [1.0, 1.0])

    def test_large_opposite_log_terms_still_tie(self):
        """log P and log lambda of about -230 and +230 cancel; both marks have
        probability 0.5, so both p-values are 1."""
        check = _two_mark_check(np.array([[1e-100, 1.0]] * 2), np.diag([1.1e100, 1.1]), [0, 1])
        assert_array_equal(check.pvalue, [1.0, 1.0])

    @pytest.mark.parametrize("far_probability", [0.0, 1e-300])
    def test_states_that_do_not_contribute_do_not_loosen_ties(self, far_probability):
        """A state far from the observed mark (log intensity about -5e15) with no or
        negligible predictive mass must not change the p-value (0 without it)."""
        means = np.array([0.0, 1e8])
        check = monte_carlo_mark_pvalue(
            np.array([[1.0, far_probability]]),
            lambda m: norm.logpdf(np.asarray(m)[:, :1], means, 1.0),
            np.array([[5.0]]),
            ground_intensity=np.ones(2),
            sample_marks=lambda bins, rng: rng.normal(means[bins], 1.0)[:, None],
            n_samples=1000,
            rng=0,
        )
        assert check.pvalue[0] == 0.0

    def test_state_and_intensity_far_apart_in_scale(self):
        """Expected intensities 1e300 * 1e-300 = 1 and 1e-100 * 3e100 = 3: the
        observed mark 0 has predictive probability 0.25, so p = 0.25."""
        check = _two_mark_check(np.array([[1e300, 1e-100]]), np.diag([1e-300, 3e100]), [0])
        assert abs(check.pvalue[0] - 0.25) <= 4 * np.sqrt(0.25 * 0.75 / 10_000)
        assert_allclose(check.observed_log_density, np.log(0.25), rtol=1e-12)

    def test_state_whose_sum_overflows(self):
        """Rows of finite values whose sum overflows still define the prediction."""
        state = np.full((2, 2), 1e308)
        rates = np.diag([1e-308, 1e-308])
        check = _two_mark_check(state, rates, [0, 1])
        assert_array_equal(check.pvalue, [1.0, 1.0])
        assert_allclose(check.observed_log_density, np.log(0.5), rtol=1e-12)

    @pytest.mark.parametrize("state_scale", [1e-300, 1e300])
    def test_state_scale_does_not_change_the_pvalue(self, state_scale):
        state = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        rates = np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 4.0]])
        reference = _two_mark_check(state, rates, [1, 0], n_samples=2000)
        scaled = _two_mark_check(state * state_scale, rates, [1, 0], n_samples=2000)
        assert_allclose(
            scaled.observed_log_density, reference.observed_log_density, rtol=1e-12
        )
        assert_allclose(scaled.pvalue, reference.pvalue, atol=2 / 2000)


def test_many_feature_marks_whose_densities_underflow():
    """With 32 waveform features, a typical mark's intensity is about exp(-140),
    below the smallest float32; in log space the check still works."""
    rng = np.random.default_rng(0)
    position = np.linspace(0.0, 1.0, 40)
    waveforms = rng.normal(0.0, 60.0, (4, 32))
    fields = 0.5 + 20.0 * np.exp(
        -0.5 * ((position[:, None] - [0.2, 0.4, 0.6, 0.8]) / 0.1) ** 2
    )
    cumulative = np.cumsum(fields / fields.sum(axis=1, keepdims=True), axis=1)
    bandwidth = 24.0

    def log_mark_intensity(marks):
        log_waveform = norm.logpdf(np.asarray(marks)[:, None, :], waveforms, bandwidth).sum(-1)
        return logsumexp(log_waveform[:, None, :] + np.log(fields), axis=-1)

    def sample_marks(bins, rng):
        unit = np.minimum((cumulative[bins] <= rng.random(len(bins))[:, None]).sum(axis=1), 3)
        return rng.normal(waveforms[unit], bandwidth)

    state = norm.pdf(position, 0.5, 0.1)[None].repeat(2, axis=0)
    typical = sample_marks(np.array([20]), np.random.default_rng(1))
    assert log_mark_intensity(typical).max() < np.log(np.finfo(np.float32).tiny)
    check = monte_carlo_mark_pvalue(
        state,
        log_mark_intensity,
        np.vstack([typical, waveforms[:1] + 400.0]),  # a typical mark, then an extreme one
        ground_intensity=fields.sum(axis=1),
        sample_marks=sample_marks,
        n_samples=500,
        rng=2,
    )
    assert check.pvalue[0] > 0.05
    assert check.pvalue[1] == 0.0


def test_sampler_inconsistent_with_intensity_raises():
    """Replicates the intensity calls impossible everywhere (or an intensity that
    underflowed) would otherwise tie with an impossible observed mark: p = 1."""
    rates = np.array([[1.0, 0.0], [2.0, 0.0]])  # mark 1 never occurs
    with pytest.raises(ValueError, match="marks drawn by sample_marks have zero intensity"):
        monte_carlo_mark_pvalue(
            np.full((1, 2), 0.5),
            _log_intensity_of(rates),
            np.array([1]),
            ground_intensity=rates.sum(axis=1),
            sample_marks=lambda bins, _rng: np.ones(len(bins), dtype=int),
            n_samples=20,
            rng=0,
        )


class TestValidation:
    @pytest.mark.parametrize(
        ("log_mark_intensity", "match"),
        [
            (lambda m: np.zeros((len(m), 4)), "log_mark_intensity must return shape"),
            (lambda m: np.full((len(m), 3), np.nan), "finite values, or -inf"),
            (lambda m: np.full((len(m), 3), np.inf), "finite values, or -inf"),
        ],
    )
    def test_bad_log_intensity_output_raises(self, log_mark_intensity, match):
        with pytest.raises(ValueError, match=match):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3),
                log_mark_intensity,
                np.zeros(2, dtype=int),
                ground_intensity=np.ones(3),
                sample_marks=lambda bins, _rng: np.zeros(len(bins), dtype=int),
            )

    def test_sampler_with_wrong_length_raises(self):
        with pytest.raises(ValueError, match="sample_marks must return"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3),
                lambda m: np.zeros((len(m), 3)),
                np.zeros(2, dtype=int),
                ground_intensity=np.ones(3),
                sample_marks=lambda bins, _rng: np.zeros(len(bins) + 1, dtype=int),
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_samples": 0}, "n_samples must be a positive integer"),
            ({"batch_size": 0}, "batch_size must be a positive integer"),
            ({"observed_marks": np.zeros(3, dtype=int)}, "one entry per event"),
            ({"ground_intensity": np.ones(4)}, "ground_intensity must have shape"),
        ],
    )
    def test_bad_arguments_raise(self, kwargs, match):
        arguments = {
            "observed_marks": np.zeros(2, dtype=int),
            "ground_intensity": np.ones(3),
            "sample_marks": lambda bins, _rng: np.zeros(len(bins), dtype=int),
            **kwargs,
        }
        observed = arguments.pop("observed_marks")
        with pytest.raises(ValueError, match=match):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), lambda m: np.zeros((len(m), 3)), observed, **arguments
            )


def test_no_events_calls_nothing():
    def never(*_):
        msg = "called with no events"
        raise AssertionError(msg)

    check = monte_carlo_mark_pvalue(
        np.empty((0, 3)),
        never,
        np.empty(0, dtype=int),
        ground_intensity=np.ones(3),
        sample_marks=never,
        n_samples=5,
        return_samples=True,
    )
    assert check.pvalue.shape == (0,)
    assert check.observed_log_density.shape == (0,)
    assert check.simulated_log_density is not None
    assert check.simulated_log_density.shape == (0, 5)
