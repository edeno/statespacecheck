"""Tests for the Monte Carlo predictive check of continuous (or intractable) marks."""

import numpy as np
import pytest
from helpers import integer_mark_model, unit_sampler
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp
from scipy.stats import kstest, norm

from statespacecheck import (
    EventDiagnostics,
    MarkModel,
    MarkPredictiveCheck,
    clusterless_event_diagnostics,
    event_diagnostics,
    event_likelihood,
    hpd_overlap,
    kl_divergence,
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
    rates, _ = discrete_mark_model
    rng = np.random.default_rng(1)
    state = rng.dirichlet(np.full(rates.shape[0], 0.3), size=40)
    marks = rng.integers(0, rates.shape[1], size=40)
    return state, marks


def _discrete_check(discrete_mark_model, state, marks, **kwargs):
    _, model = discrete_mark_model
    return monte_carlo_mark_pvalue(state, model, marks, **kwargs)


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

    def test_never_draws_a_trailing_zero_probability_bin(self):
        """A uniform draw just below 1 can round past a row's last bin; it must fall back
        to the row's last bin with probability, not to a bin with none."""

        class AlmostOne:
            def random(self, shape):
                return np.full(shape, 1.0 - 2.0**-53)

        probabilities = np.array([[0.5, 0.5, 0.0]] * 3)
        bins = _sample_state_bins(probabilities, 2, AlmostOne())
        assert_array_equal(bins, 1)


class TestAgainstExactDiscreteMarks:
    """For integer marks the Monte Carlo check estimates mark_predictive_pvalue."""

    @pytest.mark.slow
    def test_pvalue_matches_exact_within_monte_carlo_error(
        self, discrete_mark_model, discrete_events
    ):
        rates, _ = discrete_mark_model
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
        rates, _ = discrete_mark_model
        state, marks = discrete_events
        check = _discrete_check(discrete_mark_model, state, marks, n_samples=10, rng=0)
        q = predictive_mark_probabilities(state, rates)
        assert_allclose(
            check.observed_log_density, np.log(q[np.arange(40), marks]), rtol=1e-12
        )


def test_two_dimensional_grid_matches_exact():
    """States on a (4, 3) grid, flattened in C order for the sampler: the observed log
    density is exact and the p-values agree with mark_predictive_pvalue."""
    rng = np.random.default_rng(6)
    rates = rng.gamma(2.0, size=(4, 3, 5))  # (n_x, n_y, n_marks)
    model = integer_mark_model(rates)  # log intensity (n, 4, 3)
    state = rng.dirichlet(np.full(12, 0.5), size=8).reshape(8, 4, 3)
    marks = rng.integers(0, 5, 8)
    n_samples = 20_000
    check = monte_carlo_mark_pvalue(state, model, marks, n_samples=n_samples, rng=7)
    q = predictive_mark_probabilities(state, rates)
    assert_allclose(check.observed_log_density, np.log(q[np.arange(8), marks]), rtol=1e-12)
    exact = mark_predictive_pvalue(state, rates, marks)
    assert np.all(np.abs(check.pvalue - exact) <= 4 * _binomial_se(exact, n_samples) + 1e-4)


@pytest.mark.slow
def test_calibrated_under_the_true_model(clusterless_1d_model):
    """Marks drawn from the model's own predictive give uniform p-values."""
    clusterless = clusterless_1d_model
    model = clusterless.model
    rng = np.random.default_rng(3)
    n_events = 400
    centers = rng.uniform(0.1, 0.9, n_events)
    state = norm.pdf(clusterless.position, centers[:, None], 0.08)
    state /= state.sum(axis=1, keepdims=True)
    # Each event's state is drawn from the predictive weighted by the total rate
    # (computed here independently of the package), then its mark at that state
    weights = state * model.ground_intensity
    weights /= weights.sum(axis=1, keepdims=True)
    state_bins = np.array([rng.choice(len(clusterless.position), p=row) for row in weights])
    observed = model.sample(state_bins, rng)

    check = monte_carlo_mark_pvalue(state, model, observed, n_samples=500, rng=4)
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
        MarkModel(log_mark_intensity, sample_marks, np.ones_like(position)),
        np.array([[60.0]]),
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

    model = integer_mark_model(rates)._replace(sample=sample_marks)
    check = monte_carlo_mark_pvalue(state, model, np.array([1]), n_samples=50, rng=0)
    assert check.observed_log_density[0] == -np.inf
    assert check.pvalue[0] == 0.0


def _two_mark_check(state, rates, marks, n_samples=10_000):
    return monte_carlo_mark_pvalue(
        state, integer_mark_model(rates), np.asarray(marks), n_samples=n_samples, rng=0
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
        model = MarkModel(
            lambda m: norm.logpdf(np.asarray(m)[:, :1], means, 1.0),
            lambda bins, rng: rng.normal(means[bins], 1.0)[:, None],
            np.ones(2),
        )
        check = monte_carlo_mark_pvalue(
            np.array([[1.0, far_probability]]), model, np.array([[5.0]]), n_samples=1000, rng=0
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
    sample_unit = unit_sampler(fields)
    bandwidth = 24.0

    def log_mark_intensity(marks):
        log_waveform = norm.logpdf(np.asarray(marks)[:, None, :], waveforms, bandwidth).sum(-1)
        return logsumexp(log_waveform[:, None, :] + np.log(fields), axis=-1)

    def sample_marks(bins, rng):
        return rng.normal(waveforms[sample_unit(bins, rng)], bandwidth)

    state = norm.pdf(position, 0.5, 0.1)[None].repeat(2, axis=0)
    typical = sample_marks(np.array([20]), np.random.default_rng(1))
    assert log_mark_intensity(typical).max() < np.log(np.finfo(np.float32).tiny)
    check = monte_carlo_mark_pvalue(
        state,
        MarkModel(log_mark_intensity, sample_marks, fields.sum(axis=1)),
        np.vstack([typical, waveforms[:1] + 400.0]),  # a typical mark, then an extreme one
        n_samples=500,
        rng=2,
    )
    assert check.pvalue[0] > 0.05
    assert check.pvalue[1] == 0.0


def test_sampler_inconsistent_with_intensity_raises():
    """Replicates the intensity calls impossible everywhere (or an intensity that
    underflowed) would otherwise tie with an impossible observed mark: p = 1."""
    rates = np.array([[1.0, 0.0], [2.0, 0.0]])  # mark 1 never occurs
    model = MarkModel(
        integer_mark_model(rates).log_intensity,
        lambda bins, _rng: np.ones(len(bins), dtype=int),
        rates.sum(axis=1),
    )
    with pytest.raises(ValueError, match=r"marks drawn by model\.sample have zero intensity"):
        monte_carlo_mark_pvalue(
            np.full((1, 2), 0.5), model, np.array([1]), n_samples=20, rng=0
        )


def _never(*_):
    msg = "called with no events"
    raise AssertionError(msg)


def _mark_one_log_intensity(value):
    """Log intensity 0 at every one of 3 bins, except ``value`` for mark 1."""
    return lambda m: np.where(np.asarray(m)[:, None] == 1, value, np.zeros((len(m), 3)))


def _uniform_model(log_intensity=None, sample=None, ground_intensity=None):
    """A 3-bin model with one mark (0) at log intensity 0 everywhere; parts can be replaced."""
    return MarkModel(
        log_intensity or (lambda m: np.zeros((len(m), 3))),
        sample or (lambda bins, _rng: np.zeros(len(bins), dtype=int)),
        np.ones(3) if ground_intensity is None else ground_intensity,
    )


class TestValidation:
    @pytest.mark.parametrize(
        ("log_intensity", "match"),
        [
            (lambda m: np.zeros((len(m), 4)), r"model\.log_intensity must return shape"),
            (lambda m: np.full((len(m), 3), np.nan), "finite values, or -inf"),
            (lambda m: np.full((len(m), 3), np.inf), "finite values, or -inf"),
        ],
    )
    def test_bad_log_intensity_output_raises(self, log_intensity, match):
        with pytest.raises(ValueError, match=match):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), _uniform_model(log_intensity), np.zeros(2, dtype=int)
            )

    def test_transposed_log_intensity_raises(self):
        """rates[:, marks] without .T has the same size but the axes swapped."""
        rates = np.array([[1.0, 2.0], [1.0, 1.0], [2.0, 1.0]])
        model = _uniform_model(lambda m: np.log(rates[:, np.asarray(m)]))
        with pytest.raises(ValueError, match=r"model\.log_intensity must return shape"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), model, np.zeros(2, dtype=int))

    def test_read_only_log_intensity_is_not_modified(self):
        """The state term is added in place to a copy, not to the caller's array."""
        table = np.zeros((1000, 3))
        table.flags.writeable = False
        model = _uniform_model(lambda m: table[: len(m)])
        check = monte_carlo_mark_pvalue(
            np.full((2, 3), 1 / 3), model, np.zeros(2, dtype=int), n_samples=10, rng=0
        )
        assert_array_equal(check.pvalue, 1.0)
        assert_array_equal(table, 0.0)

    def test_sampler_with_wrong_length_raises(self):
        model = _uniform_model(sample=lambda bins, _rng: np.zeros(len(bins) + 1, dtype=int))
        with pytest.raises(ValueError, match=r"model\.sample must return"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), model, np.zeros(2, dtype=int))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_samples": 0}, "n_samples must be a positive integer"),
            ({"batch_size": 0}, "batch_size must be a positive integer"),
        ],
    )
    def test_bad_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), _uniform_model(), np.zeros(2, dtype=int), **kwargs
            )

    def test_observed_marks_of_another_length_raise(self):
        with pytest.raises(ValueError, match="one entry per event"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), _uniform_model(), np.zeros(3, int))

    def test_ground_intensity_of_another_shape_raises(self):
        model = _uniform_model(ground_intensity=np.ones(4))
        with pytest.raises(ValueError, match="ground_intensity must have shape"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), model, np.zeros(2, dtype=int))

    def test_model_that_is_not_a_mark_model_raises(self):
        """A function where the model goes, as the separate-argument form would pass."""
        with pytest.raises(TypeError, match="model must be a MarkModel"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), lambda m: np.zeros((len(m), 3)), np.zeros(2, int)
            )


def test_no_events_calls_nothing():
    check = monte_carlo_mark_pvalue(
        np.empty((0, 3)),
        MarkModel(_never, _never, np.ones(3)),
        np.empty(0, dtype=int),
        n_samples=5,
        return_samples=True,
    )
    assert check.pvalue.shape == (0,)
    assert check.observed_log_density.shape == (0,)
    assert check.simulated_log_density is not None
    assert check.simulated_log_density.shape == (0, 5)


class TestSilentFailures:
    """Inputs that used to give wrong p-values without an error."""

    def test_masked_log_intensity_raises(self):
        """np.ma.log stores 0 (intensity 1) under masked zero-intensity entries."""
        rates = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        model = MarkModel(
            lambda m: np.ma.log(rates[:, np.asarray(m)].T),
            lambda bins, _rng: np.zeros(len(bins), dtype=int),
            rates.sum(axis=1),
        )
        with pytest.raises(ValueError, match="masked array"):
            monte_carlo_mark_pvalue(np.array([[0.05, 0.05, 0.9]]), model, np.array([0]))

    def test_intensity_where_ground_intensity_is_zero_raises(self):
        """Lambda(x) = 0 means no events at x, so lambda(x, y) must be 0 there too."""
        rates = np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 4.0]])
        ground = rates.sum(axis=1)
        ground[2] = 0.0
        model = MarkModel(
            integer_mark_model(rates).log_intensity,
            lambda bins, _rng: np.zeros(len(bins), int),
            ground,
        )
        with pytest.raises(ValueError, match="ground_intensity is zero"):
            monte_carlo_mark_pvalue(np.array([[0.1, 0.1, 0.8]]), model, np.array([1]))

    def test_replicated_marks_of_another_shape_raise(self):
        """Observed marks with one feature, a sampler returning two."""
        means = np.array([[0.0, 0.0], [3.0, 3.0]])
        model = MarkModel(
            lambda m: norm.logpdf(np.asarray(m)[:, None, :], means, 0.5).sum(-1),
            lambda bins, rng: rng.normal(means[bins], 0.5),
            np.ones(2),
        )
        with pytest.raises(ValueError, match=r"model\.sample must return marks of shape"):
            monte_carlo_mark_pvalue(
                np.full((2, 2), 0.5), model, np.array([[0.0], [3.0]]), n_samples=10, rng=0
            )

    def test_nan_observed_mark_raises(self):
        model = _uniform_model(lambda m: np.zeros((len(m), 3)))
        with pytest.raises(ValueError, match=r"observed_marks .* events \[1\]"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), model, np.array([[0.5], [np.nan]]))

    def test_float32_log_intensity_raises(self):
        """float32 rounding (about 1e-7) splits marks of equal density: p = 0.49, not 1."""
        rates = np.array([[0.2, 1.0], [1.8, 1.0]])
        model = MarkModel(
            lambda m: np.log(rates[:, np.asarray(m)].T).astype(np.float32),
            lambda bins, _rng: np.zeros(len(bins), dtype=int),
            rates.sum(axis=1),
        )
        with pytest.raises(ValueError, match="float64"):
            monte_carlo_mark_pvalue(np.full((2, 2), 0.5), model, np.array([0, 1]))

    def test_errors_name_the_event_not_its_position_in_the_batch(self):
        state = np.full((20, 3), 1 / 3)
        state[13] = [0.0, 0.0, 1.0]  # all its mass where there are no events
        model = _uniform_model(
            lambda m: np.tile([0.0, 0.0, -np.inf], (len(m), 1)),
            ground_intensity=np.array([1.0, 1.0, 0.0]),
        )
        with pytest.raises(ValueError, match=r"row indices: \[13\]"):
            monte_carlo_mark_pvalue(state, model, np.zeros(20, dtype=int), batch_size=8)


@pytest.fixture(scope="module")
def discrete_session(discrete_mark_model):
    """Predictive distributions over 30 time bins and 25 events, several per bin."""
    rates, _ = discrete_mark_model
    rng = np.random.default_rng(8)
    predictive = rng.dirichlet(np.full(rates.shape[0], 0.3), size=30)
    time_ind = np.sort(rng.integers(0, 30, size=25))
    marks = rng.integers(0, rates.shape[1], size=25)
    return predictive, time_ind, marks


def _assert_same_local_diagnostics(result, expected):
    assert_array_equal(result.hpd_overlap, expected.hpd_overlap)
    assert_array_equal(result.kl_divergence, expected.kl_divergence)
    assert_array_equal(result.likelihood, expected.likelihood)


class TestClusterlessMatchesDiscrete:
    """Integer marks written as a MarkModel reproduce event_diagnostics."""

    @pytest.mark.parametrize("batch_size", [1, 4, 25])
    def test_local_diagnostics_are_bit_identical(
        self, discrete_mark_model, discrete_session, batch_size
    ):
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        result = clusterless_event_diagnostics(
            predictive,
            model,
            time_ind,
            marks,
            n_samples=10,
            rng=0,
            return_likelihood=True,
            batch_size=batch_size,
        )
        assert isinstance(result, EventDiagnostics)
        expected = event_diagnostics(
            predictive, rates, time_ind, marks, return_likelihood=True
        )
        _assert_same_local_diagnostics(result, expected)

    def test_bit_identical_whatever_the_callables_memory_layout(
        self, discrete_mark_model, discrete_session
    ):
        """Sums reduce in an order that depends on memory layout; a Fortran-ordered
        log intensity must not change the likelihood."""
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        fortran = model._replace(
            log_intensity=lambda m: np.asfortranarray(model.log_intensity(m))
        )
        result = clusterless_event_diagnostics(
            predictive, fortran, time_ind, marks, n_samples=10, rng=0, return_likelihood=True
        )
        expected = event_diagnostics(
            predictive, rates, time_ind, marks, return_likelihood=True
        )
        _assert_same_local_diagnostics(result, expected)

    def test_two_dimensional_grid(self):
        rng = np.random.default_rng(9)
        rates = rng.gamma(2.0, size=(4, 3, 5))  # (n_x, n_y, n_marks)
        model = integer_mark_model(rates)  # log intensity (n, 4, 3)
        predictive = rng.dirichlet(np.full(12, 0.5), size=6).reshape(6, 4, 3)
        time_ind, marks = rng.integers(0, 6, 10), rng.integers(0, 5, 10)
        result = clusterless_event_diagnostics(
            predictive, model, time_ind, marks, n_samples=10, rng=0, return_likelihood=True
        )
        expected = event_diagnostics(
            predictive, rates, time_ind, marks, return_likelihood=True
        )
        assert result.likelihood is not None
        assert result.likelihood.shape == (10, 4, 3)
        _assert_same_local_diagnostics(result, expected)

    @pytest.mark.slow
    def test_pvalue_matches_exact_within_monte_carlo_error(
        self, discrete_mark_model, discrete_session
    ):
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        n_samples = 20_000
        result = clusterless_event_diagnostics(
            predictive, model, time_ind, marks, n_samples=n_samples, rng=3
        )
        exact = mark_predictive_pvalue(predictive[time_ind], rates, marks)
        allowed = 4 * _binomial_se(exact, n_samples) + 1 / n_samples
        assert np.all(np.abs(result.predictive_pvalue - exact) <= allowed)


class TestClusterlessEventDiagnostics:
    @pytest.mark.parametrize("batch_size", [3, 8])
    def test_pvalue_is_monte_carlo_mark_pvalue(self, clusterless_1d_model, batch_size):
        """The same seed and batch size give the same draws, so identical p-values."""
        model = clusterless_1d_model.model
        rng = np.random.default_rng(10)
        predictive = rng.dirichlet(np.ones(60), size=12)
        time_ind = rng.integers(0, 12, size=20)
        marks = rng.uniform(0.5, 4.5, size=(20, 1))
        result = clusterless_event_diagnostics(
            predictive, model, time_ind, marks, n_samples=200, rng=5, batch_size=batch_size
        )
        check = monte_carlo_mark_pvalue(
            predictive[time_ind], model, marks, n_samples=200, rng=5, batch_size=batch_size
        )
        assert_array_equal(result.predictive_pvalue, check.pvalue)

    def test_likelihood_is_normalized_intensity(self, clusterless_1d_model):
        clusterless = clusterless_1d_model
        marks = np.array([[1.0], [2.5], [4.0]])
        result = clusterless_event_diagnostics(
            np.full((1, 60), 1 / 60),
            clusterless.model,
            np.zeros(3, dtype=int),
            marks,
            n_samples=10,
            rng=0,
            return_likelihood=True,
        )
        intensity = norm.pdf(marks, clusterless.waveform_means, clusterless.sigma)
        expected = event_likelihood(intensity @ clusterless.place_fields.T)
        assert_allclose(result.likelihood, expected, rtol=1e-12)

    def test_repeated_time_bins(self, clusterless_1d_model):
        """Events in one bin are each compared with that bin's predictive distribution."""
        model = clusterless_1d_model.model
        predictive = np.random.default_rng(11).dirichlet(np.ones(60), size=3)
        marks = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = clusterless_event_diagnostics(
            predictive,
            model,
            np.array([1, 1, 1, 2]),
            marks,
            n_samples=10,
            rng=0,
            return_likelihood=True,
        )
        assert result.likelihood is not None
        in_bin = predictive[[1, 1, 1, 2]]
        assert_array_equal(result.hpd_overlap, hpd_overlap(in_bin, result.likelihood))
        assert_array_equal(result.kl_divergence, kl_divergence(in_bin, result.likelihood))
        # Different marks in the same bin get different diagnostics
        assert np.unique(result.kl_divergence[:3]).size == 3

    def test_likelihood_omitted_by_default(self, clusterless_1d_model):
        result = clusterless_event_diagnostics(
            np.full((2, 60), 1 / 60),
            clusterless_1d_model.model,
            np.array([0, 1]),
            np.array([[1.0], [2.0]]),
            n_samples=10,
            rng=0,
        )
        assert result.likelihood is None

    def test_no_events_calls_nothing(self):
        result = clusterless_event_diagnostics(
            np.full((4, 3), 1 / 3),
            MarkModel(_never, _never, np.ones(3)),
            [],
            np.empty((0, 2)),
            return_likelihood=True,
        )
        assert result.hpd_overlap.shape == (0,)
        assert result.kl_divergence.shape == (0,)
        assert result.predictive_pvalue.shape == (0,)
        assert result.likelihood is not None
        assert result.likelihood.shape == (0, 3)


class TestClusterlessValidation:
    @pytest.fixture
    def inputs(self):
        predictive = np.random.default_rng(12).dirichlet(np.ones(3), size=6)
        return predictive, np.array([0, 2, 5]), np.zeros(3, dtype=int)

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"event_time_ind": np.array([0, 2, 6])}, r"event_time_ind must lie in \[0, 6\)"),
            ({"event_time_ind": np.array([0.0, 2.5, 5.0])}, "time-bin indices"),
            (
                {"event_marks": np.zeros(2, dtype=int)},
                "event_marks must have one entry per event",
            ),
            ({"event_marks": np.array([0.0, np.nan, 0.0])}, r"event_marks .*\[1\]"),
            ({"coverage": 1.5}, "coverage"),
            ({"batch_size": 0}, "batch_size must be a positive integer"),
            ({"n_samples": 0}, "n_samples must be a positive integer"),
            (
                {"predictive": np.full(3, 1 / 3)},
                r"predictive must have shape \(n_time, \.\.\.\)",
            ),
            (
                {"predictive": np.ma.masked_array(np.full((6, 3), 1 / 3))},
                "predictive is a masked array",
            ),
            (
                {"event_time_ind": np.ma.masked_array([0, 2, 5])},
                "event_time_ind is a masked array",
            ),
            (
                {"event_marks": np.ma.masked_array(np.zeros(3, dtype=int))},
                "event_marks is a masked array",
            ),
        ],
    )
    def test_bad_arguments_raise(self, inputs, change, match):
        predictive, time_ind, marks = inputs
        arguments = {
            "predictive": predictive,
            "model": _uniform_model(),
            "event_time_ind": time_ind,
            "event_marks": marks,
        } | change
        with pytest.raises(ValueError, match=match):
            clusterless_event_diagnostics(**arguments)

    def test_model_that_is_not_a_mark_model_raises(self, inputs):
        predictive, time_ind, marks = inputs
        model = _uniform_model()
        with pytest.raises(TypeError, match="MarkModel"):
            clusterless_event_diagnostics(predictive, tuple(model), time_ind, marks)

    def test_zero_likelihood_names_the_event(self):
        """An observed mark with zero intensity everywhere has no likelihood."""
        model = _uniform_model(_mark_one_log_intensity(-np.inf))
        marks = np.array([0, 0, 0, 0, 0, 1, 0])
        with pytest.raises(
            ValueError, match=r"events \[5\] have zero intensity at every state"
        ):
            clusterless_event_diagnostics(
                np.full((7, 3), 1 / 3), model, np.arange(7), marks, n_samples=5, batch_size=2
            )

    def test_bad_predictive_bin_names_the_bin_and_event(self, inputs):
        predictive, time_ind, marks = inputs
        predictive[5, 1] = np.nan
        with pytest.raises(ValueError, match=r"time bins \[5\] .*events \[2\]"):
            clusterless_event_diagnostics(predictive, _uniform_model(), time_ind, marks)

    def test_predictive_without_mass_where_events_occur_names_the_bin(self, inputs):
        predictive, time_ind, marks = inputs
        predictive[2] = [0.0, 0.0, 1.0]
        model = _uniform_model(ground_intensity=np.array([1.0, 1.0, 0.0]))
        with pytest.raises(ValueError, match=r"time bins \[2\].*events \[1\]"):
            clusterless_event_diagnostics(predictive, model, time_ind, marks)

    def test_time_bins_without_events_are_not_checked(self, inputs):
        """Decoder output often has invalid time bins where no event falls."""
        predictive, time_ind, marks = inputs
        clean = clusterless_event_diagnostics(
            predictive, _uniform_model(), time_ind, marks, n_samples=10, rng=0
        )
        predictive[1] = np.nan
        predictive[3] = 0.0
        result = clusterless_event_diagnostics(
            predictive, _uniform_model(), time_ind, marks, n_samples=10, rng=0
        )
        for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            assert_array_equal(getattr(result, field), getattr(clean, field))

    def test_intensity_where_ground_intensity_is_zero_raises_without_predictive_mass(self):
        """The likelihood is normalized over every state, so a finite intensity where
        the model has no events would take likelihood mass even where the prediction
        has none."""
        model = _uniform_model(ground_intensity=np.array([1.0, 1.0, 0.0]))
        predictive = np.full((7, 3), 0.5)
        predictive[:, 2] = 0.0
        with pytest.raises(ValueError, match=r"ground_intensity is zero \(events \[0, 1\]\)"):
            clusterless_event_diagnostics(
                predictive, model, np.arange(7), np.zeros(7, dtype=int), batch_size=2
            )

    def test_zero_ground_error_names_the_global_event(self):
        # Mark 1 has intensity at bin 2, where the ground intensity is zero; the
        # sampler draws only mark 0, so only event 5's observed mark is inconsistent
        model = integer_mark_model(np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]))._replace(
            sample=lambda bins, _rng: np.zeros(len(bins), dtype=int),
            ground_intensity=np.array([1.0, 1.0, 0.0]),
        )
        with pytest.raises(ValueError, match=r"zero \(events \[5\]\)"):
            clusterless_event_diagnostics(
                np.full((7, 3), 1 / 3),
                model,
                np.arange(7),
                np.array([0, 0, 0, 0, 0, 1, 0]),
                n_samples=5,
                batch_size=2,
                rng=0,
            )

    def test_invalid_coverage_raises_before_any_event(self):
        with pytest.raises(ValueError, match="coverage"):
            clusterless_event_diagnostics(
                np.full((3, 3), 1 / 3), _uniform_model(), [], np.empty(0), coverage=1.5
            )

    def test_coverage_is_used(self, discrete_mark_model, discrete_session):
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        result = clusterless_event_diagnostics(
            predictive, model, time_ind, marks, coverage=0.5, n_samples=10, rng=0
        )
        expected = event_diagnostics(predictive, rates, time_ind, marks, coverage=0.5)
        default = event_diagnostics(predictive, rates, time_ind, marks)
        assert not np.array_equal(expected.hpd_overlap, default.hpd_overlap)
        assert_array_equal(result.hpd_overlap, expected.hpd_overlap)


class TestClusterlessScale:
    """Log intensities and predictive distributions far from unit scale."""

    @pytest.mark.parametrize("offset", [-1e6, -800.0, 800.0, 1e6])
    def test_log_intensity_offset_does_not_change_the_local_diagnostics(
        self, discrete_mark_model, discrete_session, offset
    ):
        """A constant factor exp(offset) in the intensity, which underflows or
        overflows in linear space, cancels in the likelihood."""
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        shifted = MarkModel(
            lambda m: model.log_intensity(m) + offset,
            model.sample,
            rates.sum(axis=1),  # the ground intensity is not used by the likelihood
        )
        kwargs = {"n_samples": 10, "rng": 0, "return_likelihood": True}
        result = clusterless_event_diagnostics(predictive, shifted, time_ind, marks, **kwargs)
        reference = clusterless_event_diagnostics(predictive, model, time_ind, marks, **kwargs)
        # The model's own log + offset rounds to within eps * |offset|
        rtol = 16 * np.finfo(float).eps * abs(offset)
        assert_allclose(result.likelihood, reference.likelihood, rtol=rtol)
        assert_allclose(result.kl_divergence, reference.kl_divergence, rtol=rtol, atol=rtol)
        assert_array_equal(result.hpd_overlap, reference.hpd_overlap)

    @pytest.mark.parametrize("scale", [1e-310, 1e-200, 1e200])
    def test_predictive_scale_matches_event_diagnostics(
        self, discrete_mark_model, discrete_session, scale
    ):
        """Subnormal and huge predictive rows (not normalized) give event_diagnostics'
        local diagnostics exactly, and finite p-values."""
        rates, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        result = clusterless_event_diagnostics(
            predictive * scale, model, time_ind, marks, n_samples=50, rng=0
        )
        expected = event_diagnostics(predictive * scale, rates, time_ind, marks)
        assert_array_equal(result.hpd_overlap, expected.hpd_overlap)
        assert_array_equal(result.kl_divergence, expected.kl_divergence)
        assert np.all((result.predictive_pvalue >= 0) & (result.predictive_pvalue <= 1))

    def test_underflowing_intensity_products_match_event_diagnostics(self):
        """Products below the smallest subnormal underflow to 0; both paths accept
        the events and give the same local diagnostics."""
        u = np.finfo(float).smallest_subnormal
        rates = np.array([[u, 2 * u], [0.0, 0.0]])
        predictive = np.array([[0.125, 0.875]])
        time_ind, marks = np.array([0, 0]), np.array([0, 1])
        result = clusterless_event_diagnostics(
            predictive, integer_mark_model(rates), time_ind, marks, n_samples=50, rng=0
        )
        expected = event_diagnostics(predictive, rates, time_ind, marks)
        assert_array_equal(result.hpd_overlap, expected.hpd_overlap)
        assert_array_equal(result.kl_divergence, expected.kl_divergence)

    def test_mark_possible_only_where_the_prediction_has_no_mass(self):
        """The observed mark's likelihood lies where the prediction is zero: no overlap,
        infinite KL divergence, and p = 0, without warnings."""
        model = integer_mark_model(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        result = clusterless_event_diagnostics(
            np.array([[0.5, 0.5, 0.0]]), model, np.array([0]), np.array([1]), rng=0
        )
        assert_array_equal(result.hpd_overlap, [0.0])
        assert_array_equal(result.kl_divergence, [np.inf])
        assert_array_equal(result.predictive_pvalue, [0.0])


@pytest.fixture(scope="module")
def session_diagnostics(clusterless_session):
    """Diagnostics of the simulated session under the true and the misspecified model."""
    session = clusterless_session
    return {
        name: clusterless_event_diagnostics(
            predictive,
            model,
            session.event_time_ind,
            session.event_marks,
            n_samples=500,
            rng=0,
        )
        for name, predictive, model in [
            ("true", session.predictive, session.model),
            ("misspecified", session.misspecified_predictive, session.misspecified_model),
        ]
    }


# Thresholds set from observed values (fixture seed, true model: KS p = 0.59, 4.7% of
# p <= 0.05, 6.7% for unit 0; misspecified: KS p = 1.0e-6, 8.4%, 47.2%). Over seven
# seeds (20260925 and 0 to 5), the misspecified fraction of p <= 0.05 ranged from 8.4% to 20.9%, always
# above the true model's (4.3% to 6.4%), so the two are compared with each other.
@pytest.mark.slow
class TestSimulatedSession:
    def test_calibrated_under_the_true_model(self, session_diagnostics):
        pvalue = session_diagnostics["true"].predictive_pvalue
        assert kstest(pvalue, "uniform").pvalue > 0.01
        assert np.mean(pvalue <= 0.05) <= 0.08

    def test_detects_misspecified_marks(self, session_diagnostics):
        """Shifting every waveform mean by +0.8, more than their 0.6 spacing, reads most
        spikes as the neighboring unit's: the decoder follows a consistently shifted
        position, so most spikes still agree with the prediction. The p-values are
        no longer uniform, and more of them are small."""
        pvalue = session_diagnostics["misspecified"].predictive_pvalue
        true_pvalue = session_diagnostics["true"].predictive_pvalue
        assert kstest(pvalue, "uniform").pvalue < 0.01
        assert np.mean(pvalue <= 0.05) > np.mean(true_pvalue <= 0.05)

    def test_flags_the_unit_the_misspecified_model_cannot_explain(
        self, clusterless_session, session_diagnostics
    ):
        """Unit 0's marks (mean 1.0) lie below every shifted mean (1.8 and up), so no
        relabeling explains them: most of its spikes are flagged."""
        unit_0 = clusterless_session.event_unit == 0
        flagged = {
            name: np.mean(result.predictive_pvalue[unit_0] <= 0.05)
            for name, result in session_diagnostics.items()
        }
        assert flagged["misspecified"] > 0.4
        assert flagged["true"] < 0.25

    def test_local_diagnostics_are_valid(self, session_diagnostics):
        for result in session_diagnostics.values():
            assert np.all((result.hpd_overlap >= 0.0) & (result.hpd_overlap <= 1.0))
            assert np.all(result.kl_divergence >= 0.0)  # also False for NaN


class TestModelChecks:
    """Model output and inputs that would otherwise give wrong numbers without an error."""

    @pytest.mark.parametrize(
        "log_intensity",
        [
            lambda m: np.ones((len(m), 3), dtype=bool),  # a support mask, not a log
            lambda m: np.zeros((len(m), 3), dtype=np.int64),  # cannot hold -inf
            lambda m: np.zeros((len(m), 3), dtype=complex),  # np.emath.log of a negative
        ],
    )
    def test_non_real_float_log_intensity_raises(self, log_intensity):
        with pytest.raises(ValueError, match="real floating-point"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), _uniform_model(log_intensity), np.zeros(2, dtype=int)
            )

    def test_nan_log_intensity_names_the_observed_event(self):
        model = _uniform_model(_mark_one_log_intensity(np.nan))
        marks = np.array([0, 0, 0, 0, 0, 1, 0])
        with pytest.raises(
            ValueError, match=r"NaN or \+inf for the observed marks of events \[5\]"
        ):
            monte_carlo_mark_pvalue(
                np.full((7, 3), 1 / 3), model, marks, n_samples=5, batch_size=2
            )

    @pytest.mark.parametrize("n_samples", [1, 5])
    def test_nan_log_intensity_names_the_replicated_event(self, n_samples):
        model = _uniform_model(
            _mark_one_log_intensity(np.nan),
            sample=lambda bins, _rng: np.ones(len(bins), dtype=int),
        )
        with pytest.raises(
            ValueError, match=r"marks drawn by model\.sample of events \[0, 1\]"
        ):
            monte_carlo_mark_pvalue(
                np.full((4, 3), 1 / 3),
                model,
                np.zeros(4, dtype=int),
                n_samples=n_samples,
                batch_size=2,
            )

    def test_non_finite_replicated_marks_raise(self):
        """NaN replicates used to be ranked like any other mark."""
        calls = []

        def sample(bins, _rng):
            calls.append(len(bins))
            marks = np.zeros((len(bins), 1))
            if len(calls) == 2:  # the second batch: events 2 and 3
                marks[::2] = np.nan
            return marks

        model = _uniform_model(sample=sample)
        with pytest.raises(ValueError, match=r"non-finite marks for events \[2, 3\]"):
            monte_carlo_mark_pvalue(
                np.full((4, 3), 1 / 3), model, np.zeros((4, 1)), n_samples=4, batch_size=2
            )

    def test_sampler_that_draws_marks_impossible_at_their_state_raises(self):
        """Mark 0 occurs only at bin 0 and mark 1 only at bins 1 and 2, but the sampler
        draws the other one; every replicate is possible somewhere with predictive mass,
        so only the check at the drawn state catches it."""
        rates = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        model = integer_mark_model(rates)._replace(
            sample=lambda bins, _rng: (bins == 0).astype(int)
        )
        with pytest.raises(ValueError, match="at the state bin they were drawn at"):
            monte_carlo_mark_pvalue(
                np.full((1, 3), 1 / 3), model, np.array([0]), n_samples=50, rng=0
            )

    def test_drawn_state_error_counts_and_names_the_events(self):
        """The sampler draws correctly for the first two batches, then the other mark."""
        rates = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        calls = []

        def sample(bins, _rng):
            calls.append(len(bins))
            wrong = len(calls) > 2
            return ((bins == 0) == wrong).astype(int)

        model = integer_mark_model(rates)._replace(sample=sample)
        with pytest.raises(ValueError, match=r"of 10 marks .*\(events \[4, 5\]\)"):
            monte_carlo_mark_pvalue(
                np.full((6, 3), 1 / 3),
                model,
                np.zeros(6, dtype=int),
                n_samples=5,
                batch_size=2,
            )

    def test_infinite_log_intensity_names_the_observed_event(self):
        model = _uniform_model(_mark_one_log_intensity(np.inf))
        with pytest.raises(ValueError, match=r"observed marks of events \[3\]"):
            monte_carlo_mark_pvalue(
                np.full((5, 3), 1 / 3), model, np.array([0, 0, 0, 1, 0]), batch_size=2
            )

    def test_one_non_finite_feature_makes_a_mark_non_finite(self):
        """A dead channel's NaN, in a feature the model might ignore."""
        model = _uniform_model(sample=lambda bins, _rng: np.zeros((len(bins), 2)))
        marks = np.array([[0.5, 0.0], [0.5, np.nan], [0.5, 0.0]])
        with pytest.raises(ValueError, match=r"observed_marks must be finite; events \[1\]"):
            monte_carlo_mark_pvalue(np.full((3, 3), 1 / 3), model, marks)

        def sample(bins, _rng):
            replicated = np.zeros((len(bins), 2))
            replicated[0, 1] = np.nan
            return replicated

        with pytest.raises(ValueError, match=r"non-finite marks for events \[0\]"):
            monte_carlo_mark_pvalue(
                np.full((3, 3), 1 / 3),
                _uniform_model(sample=sample),
                np.zeros((3, 2)),
            )

    def test_callers_marks_stay_writable(self):
        marks = np.zeros(2, dtype=int)
        monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), _uniform_model(), marks, n_samples=5)
        assert marks.flags.writeable

    def test_masked_sampler_output_raises(self):
        model = _uniform_model(
            sample=lambda bins, _rng: np.ma.masked_array(np.zeros(len(bins), dtype=int))
        )
        with pytest.raises(ValueError, match=r"model\.sample's output is a masked array"):
            monte_carlo_mark_pvalue(np.full((1, 3), 1 / 3), model, np.array([0]), n_samples=5)

    def test_masked_observed_marks_raise(self):
        marks = np.ma.masked_array([0, 0], mask=[False, True])
        with pytest.raises(ValueError, match="observed_marks is a masked array"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), _uniform_model(), marks)

    def test_model_cannot_change_the_callers_marks(self):
        def log_intensity(marks):
            marks -= 1.0  # a callable that modifies its input in place
            return np.zeros((len(marks), 3))

        marks = np.array([[5.0], [7.0]])
        with pytest.raises(ValueError, match="read-only"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), _uniform_model(log_intensity), marks
            )
        assert_array_equal(marks, [[5.0], [7.0]])


class TestLargeLogIntensities:
    @pytest.mark.parametrize("mark", [1e8, -1e8, 3e5])
    def test_likelihood_of_a_state_independent_mark_is_uniform(self, mark):
        """log N(1e8; 0, 1) is about -5e15, where subtracting the log normalizer from
        each value rounds by about 1 and left the rows summing to 0.74."""
        model = MarkModel(
            lambda m: np.tile(norm.logpdf(np.asarray(m)[:, :1], 0.0, 1.0), (1, 2)),
            lambda bins, rng: rng.normal(0.0, 1.0, (len(bins), 1)),
            np.ones(2),
        )
        result = clusterless_event_diagnostics(
            np.full((1, 2), 0.5),
            model,
            [0],
            np.array([[mark]]),
            n_samples=10,
            rng=0,
            return_likelihood=True,
        )
        assert_allclose(result.likelihood, [[0.5, 0.5]], rtol=1e-15)

    @pytest.mark.parametrize("offset", [-1e12, -1e6, -2000.0, 2000.0, 1e6, 1e12])
    def test_likelihood_rows_sum_to_one_at_any_offset(self, offset):
        """Log intensities offset + (0, log 2, log 4) give a likelihood of 1:2:4. The
        offset's own rounding (ulp 1.2e-4 at 1e12) limits the shape, not the sum."""
        model = _uniform_model(
            lambda m: np.tile(offset + np.log([1.0, 2.0, 4.0]), (len(m), 1))
        )
        result = clusterless_event_diagnostics(
            np.full((1, 3), 1 / 3),
            model,
            [0],
            [0],
            n_samples=10,
            rng=0,
            return_likelihood=True,
        )
        assert_allclose(result.likelihood.sum(), 1.0, rtol=1e-15)
        rtol = 4 * np.finfo(float).eps * abs(offset)
        assert_allclose(result.likelihood, [[1 / 7, 2 / 7, 4 / 7]], rtol=rtol)


def test_float32_predictive_is_not_copied_whole():
    """A decoder's float32 predictive (non_local_detector's) is converted batch by
    batch; converting it all would allocate twice its size in float64."""
    import tracemalloc

    predictive = np.full((4096, 256), 1 / 256, dtype=np.float32)
    model = MarkModel(
        lambda m: np.zeros((len(m), 256)),
        lambda bins, _rng: np.zeros((len(bins), 1)),
        np.ones(256),
    )
    tracemalloc.start()
    try:
        clusterless_event_diagnostics(
            predictive, model, [5], np.zeros((1, 1)), n_samples=10, rng=0, batch_size=1
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < predictive.nbytes  # a float64 copy alone is twice predictive.nbytes


class TestInputTypes:
    def test_sampler_cannot_reorder_the_state_bins(self):
        """Sorting the bins in place would move replicates between events."""

        def sample(bins, _rng):
            bins.sort()
            return np.zeros(len(bins), dtype=int)

        with pytest.raises(ValueError, match="read-only"):
            monte_carlo_mark_pvalue(
                np.full((2, 3), 1 / 3), _uniform_model(sample=sample), np.zeros(2, dtype=int)
            )

    def test_object_log_intensity_raises(self):
        """float32 or integer values wrapped in an object array would pass unchecked."""
        model = _uniform_model(
            lambda m: np.zeros((len(m), 3), dtype=np.float32).astype(object)
        )
        with pytest.raises(ValueError, match="real floating-point"):
            monte_carlo_mark_pvalue(np.full((2, 3), 1 / 3), model, np.zeros(2, dtype=int))

    def test_object_predictive_is_converted(self, discrete_mark_model, discrete_session):
        _, model = discrete_mark_model
        predictive, time_ind, marks = discrete_session
        kwargs = {"n_samples": 10, "rng": 0}
        result = clusterless_event_diagnostics(
            predictive.astype(object), model, time_ind, marks, **kwargs
        )
        expected = clusterless_event_diagnostics(predictive, model, time_ind, marks, **kwargs)
        for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            assert_array_equal(getattr(result, field), getattr(expected, field))

    @pytest.mark.skipif(
        np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
        reason="longdouble is float64 on this platform",
    )
    def test_longdouble_predictive_beyond_float64_raises(self):
        """1e400 is finite in extended precision but overflows float64."""
        predictive = np.ones((2, 3), dtype=np.longdouble)
        predictive[0, 0] = np.longdouble("1e400")
        with pytest.raises(ValueError, match=r"predictive must contain only finite"):
            clusterless_event_diagnostics(
                predictive, _uniform_model(), [0, 1], np.zeros(2, dtype=int)
            )


def test_complex_marks_are_passed_to_the_model():
    """Marks are not converted to floats; a complex mark gives the p-value of its real
    and imaginary parts as two features, with the same draws."""
    centers = np.array([0.0 + 0.0j, 1.0 + 1.0j, 2.0 - 1.0j])

    def sample_complex(bins, rng):
        real, imaginary = rng.normal(size=len(bins)), rng.normal(size=len(bins))
        return (centers[bins] + (real + 1j * imaginary) / np.sqrt(2))[:, None]

    def sample_pairs(bins, rng):
        real, imaginary = rng.normal(size=len(bins)), rng.normal(size=len(bins))
        return np.column_stack(
            [
                centers[bins].real + real / np.sqrt(2),
                centers[bins].imag + imaginary / np.sqrt(2),
            ]
        )

    complex_model = MarkModel(
        lambda m: -(np.abs(np.asarray(m)[:, :1] - centers) ** 2), sample_complex, np.ones(3)
    )
    pair_model = MarkModel(
        lambda m: (
            -((np.asarray(m)[:, :1] - centers.real) ** 2)
            - (np.asarray(m)[:, 1:] - centers.imag) ** 2
        ),
        sample_pairs,
        np.ones(3),
    )
    state = np.full((2, 3), 1 / 3)
    observed = np.array([0.5 + 0.2j, 3.0 + 2.0j])
    kwargs = {"n_samples": 500, "rng": 0}
    complex_check = monte_carlo_mark_pvalue(state, complex_model, observed[:, None], **kwargs)
    pair_check = monte_carlo_mark_pvalue(
        state, pair_model, np.column_stack([observed.real, observed.imag]), **kwargs
    )
    assert_allclose(complex_check.pvalue, pair_check.pvalue, atol=1 / 500)
    assert complex_check.pvalue[1] < complex_check.pvalue[0]
