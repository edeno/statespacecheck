"""Tests for the Monte Carlo predictive check of continuous (or intractable) marks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import kstest, norm

from statespacecheck import (
    MarkPredictiveCheck,
    event_weighted_predictive,
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
    rates, mark_intensity, sample_marks = discrete_mark_model
    return monte_carlo_mark_pvalue(
        state,
        mark_intensity,
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
    # Draw each event's state from the event-weighted predictive, then its mark
    state_bins = _sample_state_bins(
        event_weighted_predictive(state, model.ground_intensity), 1, rng
    )[:, 0]
    observed = model.sample_marks(state_bins, rng)

    check = monte_carlo_mark_pvalue(
        state,
        model.mark_intensity,
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

    def mark_intensity(marks):
        return norm.pdf(np.asarray(marks)[:, :1], position, sigma)

    def sample_marks(bins, rng):
        return rng.normal(position[bins], sigma)[:, None]

    n_samples = 20_000
    check = monte_carlo_mark_pvalue(
        predictive[None],
        mark_intensity,
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
    rates, _, _ = discrete_mark_model
    state, marks = discrete_events
    check = _discrete_check(
        discrete_mark_model, state, marks, n_samples=300, rng=10, return_samples=True
    )
    assert isinstance(check, MarkPredictiveCheck)
    assert check.simulated_log_density is not None
    assert check.simulated_log_density.shape == (40, 300)
    tolerance = 16 * np.finfo(float).eps * rates.shape[0]
    recomputed = np.mean(
        check.simulated_log_density <= check.observed_log_density[:, None] + tolerance, axis=1
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
        lambda m: rates[:, np.asarray(m)].T,
        np.array([1]),
        ground_intensity=rates.sum(axis=-1),
        sample_marks=sample_marks,
        n_samples=50,
        rng=0,
    )
    assert check.observed_log_density[0] == -np.inf
    assert check.pvalue[0] == 0.0


class TestValidation:
    @pytest.mark.parametrize(
        ("mark_intensity", "match"),
        [
            (lambda m: np.ones((len(m), 4)), "mark_intensity must return shape"),
            (lambda m: -np.ones((len(m), 3)), "finite nonnegative"),
            (lambda m: np.full((len(m), 3), np.nan), "finite nonnegative"),
        ],
    )
    def test_bad_mark_intensity_output_raises(self, mark_intensity, match):
        with pytest.raises(ValueError, match=match):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3),
                mark_intensity,
                np.zeros(2, dtype=int),
                ground_intensity=np.ones(3),
                sample_marks=lambda bins, _rng: np.zeros(len(bins), dtype=int),
            )

    def test_sampler_with_wrong_length_raises(self):
        with pytest.raises(ValueError, match="sample_marks must return"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3),
                lambda m: np.ones((len(m), 3)),
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
                np.full((2, 3), 1 / 3), lambda m: np.ones((len(m), 3)), observed, **arguments
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
