"""What the per-spike diagnostics can and cannot detect.

Deterministic cases pin the consequences of ties in baseline thresholds and of a
prediction that is uniform over units. A seeded simulation of place-cell recordings,
decoded by an exact grid filter, then checks each diagnostic under a correctly
specified model and three misspecifications. Rates are compared across independent
recordings, with tolerances from their recording-to-recording standard error; spikes
within a recording share predictions and are not independent.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers import markov_trajectory, spike_events
from numpy.testing import assert_array_equal
from scipy.stats import norm

from statespacecheck import (
    EventDiagnostics,
    baseline_threshold,
    event_diagnostics,
    flag_events,
    mark_predictive_pvalue,
)


def _diagnostics(hpd_overlap, kl_divergence):
    values = np.asarray(hpd_overlap, dtype=float)
    return EventDiagnostics(
        hpd_overlap=values,
        kl_divergence=np.asarray(kl_divergence, dtype=float),
        predictive_pvalue=np.ones_like(values),
        likelihood=None,
    )


class TestTiedBaselines:
    """The comparisons are inclusive, as in the paper, so values tied at a threshold
    are all flagged, and the flagged fraction can exceed the requested quantile."""

    def test_all_hpd_overlaps_one(self):
        """Nested regions give an overlap of exactly 1; if every baseline spike's is,
        the 1st percentile is 1 and every spike at 1 is flagged."""
        baseline = _diagnostics(np.ones(200), np.full(200, 0.5))
        threshold = baseline_threshold(baseline.hpd_overlap, 0.01)
        flags = flag_events(baseline, hpd_overlap_threshold=threshold, pvalue_threshold=None)
        assert threshold == 1.0
        assert flags.hpd_overlap.mean() == 1.0

    def test_all_kl_divergences_zero(self):
        """A prediction equal to each spike's likelihood gives KL divergence 0; the
        99th percentile is 0 and every spike at 0 is flagged."""
        baseline = _diagnostics(np.ones(200), np.zeros(200))
        threshold = baseline_threshold(baseline.kl_divergence, 0.99)
        flags = flag_events(baseline, kl_divergence_threshold=threshold, pvalue_threshold=None)
        assert threshold == 0.0
        assert flags.kl_divergence.mean() == 1.0

    def test_hpd_overlaps_tied_at_zero(self):
        """5% of baseline spikes without overlap: the 1st percentile is 0, and the 5%
        at 0 are flagged, not 1%."""
        overlaps = np.linspace(0.1, 1.0, 200)
        overlaps[:10] = 0.0
        baseline = _diagnostics(overlaps, np.full(200, 0.5))
        threshold = baseline_threshold(baseline.hpd_overlap, 0.01)
        flags = flag_events(baseline, hpd_overlap_threshold=threshold, pvalue_threshold=None)
        assert threshold == 0.0
        assert flags.hpd_overlap.mean() == 0.05


def test_uniform_prediction_over_units_gives_every_unit_p_one():
    """Units with identical place fields are equally probable under any prediction, so
    every spike's p-value is 1 however skewed the observed mixture of units is: the
    p-value tests each spike, not the frequencies of units across spikes."""
    place_fields = np.ones((10, 4))  # every unit fires at the same rate everywhere
    units = np.array([0] * 95 + [1, 2, 3, 3, 3])
    diagnostics = event_diagnostics(
        np.full((100, 10), 0.1), place_fields, np.arange(100), units
    )
    assert_array_equal(diagnostics.predictive_pvalue, np.ones(100))


def test_rare_unit_is_flagged_under_a_correct_model():
    """The least probable unit, at 0.02, always has p = 0.02, so all its spikes are
    flagged at 0.05 although the model is correct; overall, 2% of spikes are."""
    place_fields = np.array([[0.49, 0.49, 0.02]])  # one state, three units
    rng = np.random.default_rng(5)
    units = rng.choice(3, size=20_000, p=place_fields[0])
    pvalue = mark_predictive_pvalue(np.ones((20_000, 1)), place_fields, units)
    assert_array_equal(pvalue[units == 2], 0.02)
    assert_array_equal(pvalue[units != 2], 1.0)
    assert np.mean(pvalue <= 0.05) == np.mean(units == 2)


# A 1-D track of 50 bins with 20 place cells, decoded in 20 ms bins
N_BINS, N_UNITS, N_TIME, DT = 50, 20, 2000, 0.02
POSITION = np.linspace(0.0, 1.0, N_BINS)
N_RECORDINGS = 12


def _transition(width):
    """Gaussian random-walk transition; ``[i, j]`` is the probability of i after j."""
    transition = norm.pdf(POSITION[:, None], POSITION[None, :], width)
    return transition / transition.sum(axis=0, keepdims=True)


def _place_fields(centers):
    """Firing rate (Hz) of each unit at each position, (n_bins, n_units)."""
    return 0.5 + 25.0 * np.exp(-0.5 * ((POSITION[:, None] - centers) / 0.06) ** 2)


def _decode(counts, place_fields, transition):
    """One-step predictive distributions (n_time, n_bins) of a Poisson grid filter."""
    expected = place_fields * DT
    log_likelihood = counts @ np.log(expected).T - expected.sum(axis=1)
    predictive = np.empty((N_TIME, N_BINS))
    posterior = np.full(N_BINS, 1.0 / N_BINS)
    for t in range(N_TIME):
        predictive[t] = transition @ posterior
        log_posterior = np.log(predictive[t]) + log_likelihood[t]
        posterior = np.exp(log_posterior - log_posterior.max())
        posterior /= posterior.sum()
    return predictive


@pytest.fixture(scope="module")
def scenarios():
    """Per-recording fractions of flagged spikes and median KL, for each scenario.

    Every recording follows the decoder's own transition, so the correctly specified
    filter is exact. Thresholds come from separate correctly specified recordings,
    as from a baseline period: HPD overlap at its 1st percentile, KL divergence at
    its 99th, the p-value at 0.05. The misspecified decoders keep the true
    transition unless stated:

    - changed place fields: every third unit fires at a new location, which the
      decoder does not know;
    - broad prediction: the decoder's transition is 17 times too wide;
    - overall rate: the decoder's rates are 4 or 0.25 times the true ones.
    """
    transition = _transition(0.03)
    centers = np.linspace(0.02, 0.98, N_UNITS)
    fields = _place_fields(centers)
    changed_centers = centers.copy()
    changed_centers[::3] = np.random.default_rng(99).uniform(0.0, 1.0, centers[::3].size)
    changed_fields = _place_fields(changed_centers)

    def run(seed, data_fields, decoder_fields, decoder_transition):
        rng = np.random.default_rng(seed)
        counts = rng.poisson(data_fields[markov_trajectory(rng, transition, N_TIME)] * DT)
        time_ind, unit = spike_events(counts)
        predictive = _decode(counts, decoder_fields, decoder_transition)
        return event_diagnostics(predictive, decoder_fields, time_ind, unit)

    baseline = [run(1000 + r, fields, fields, transition) for r in range(N_RECORDINGS)]
    hpd_threshold = baseline_threshold(np.concatenate([b.hpd_overlap for b in baseline]), 0.01)
    kl_threshold = baseline_threshold(
        np.concatenate([b.kl_divergence for b in baseline]), 0.99
    )

    settings = {
        "correct": (fields, fields, transition),
        "changed fields": (changed_fields, fields, transition),
        "broad prediction": (fields, fields, _transition(0.5)),
        "rate x4": (fields, 4.0 * fields, transition),
        "rate x0.25": (fields, 0.25 * fields, transition),
    }
    results = {}
    for name, (data_fields, decoder_fields, decoder_transition) in settings.items():
        runs = [
            run(r, data_fields, decoder_fields, decoder_transition)
            for r in range(N_RECORDINGS)
        ]
        results[name] = SimpleNamespace(
            pvalue=np.array([np.mean(d.predictive_pvalue <= 0.05) for d in runs]),
            hpd=np.array([np.mean(d.hpd_overlap <= hpd_threshold) for d in runs]),
            kl=np.array([np.mean(d.kl_divergence >= kl_threshold) for d in runs]),
            mean_hpd=np.array([np.mean(d.hpd_overlap) for d in runs]),
            median_kl=np.array([np.median(d.kl_divergence) for d in runs]),
        )
    return results


def _standard_error(values):
    return np.std(values, ddof=1) / np.sqrt(values.size)


def _exceeds(larger, smaller, n_se=3.0):
    """Whether the mean of ``larger`` exceeds that of ``smaller`` by more than n_se
    standard errors of the difference, over independent recordings."""
    difference = larger.mean() - smaller.mean()
    return difference > n_se * np.hypot(_standard_error(larger), _standard_error(smaller))


# Observed with these seeds (mean over 12 recordings, about 3,300 spikes each):
#   scenario          p <= 0.05  HPD flagged  KL flagged  median KL
#   correct             0.044       0.029       0.011       0.52
#   changed fields      0.181       0.062       0.026       1.09
#   broad prediction    0.011       0.000       0.000       1.06
#   rate x4             0.036       0.032       0.034       0.69
#   rate x0.25          0.045       0.029       0.008       0.51
# The HPD threshold is 0 (over 1% of baseline spikes have no overlap), so 2.9% of
# correctly specified spikes are flagged, not 1%: see TestTiedBaselines.
@pytest.mark.slow
class TestSimulatedMisspecification:
    def test_correct_model_pvalues_are_calibrated(self, scenarios):
        """Exact discrete p-values are conservative: at most 5% at or below 0.05."""
        pvalue = scenarios["correct"].pvalue
        assert pvalue.mean() <= 0.05 + 3 * _standard_error(pvalue)

    def test_changed_place_fields_are_detected(self, scenarios):
        correct, changed = scenarios["correct"], scenarios["changed fields"]
        assert _exceeds(changed.pvalue, correct.pvalue)
        assert _exceeds(changed.hpd, correct.hpd)
        assert _exceeds(changed.kl, correct.kl)

    def test_broad_prediction_passes_hpd_overlap_and_pvalue(self, scenarios):
        """A blind spot: a prediction too broad to be informative contains each
        spike's likelihood, so HPD overlap rises and fewer p-values are small. Only
        the typical KL divergence moves, and not past the baseline's 99th percentile."""
        correct, broad = scenarios["correct"], scenarios["broad prediction"]
        assert _exceeds(broad.mean_hpd, correct.mean_hpd)
        assert not _exceeds(broad.pvalue, correct.pvalue, n_se=0.0)
        assert not _exceeds(broad.hpd, correct.hpd, n_se=0.0)
        assert _exceeds(broad.median_kl, correct.median_kl)

    @pytest.mark.parametrize("scenario", ["rate x4", "rate x0.25"])
    def test_wrong_overall_rate_passes_hpd_overlap_and_pvalue(self, scenarios, scenario):
        """A blind spot: the single-event likelihood and the predictive mark
        probabilities are normalized, so a common rate factor cancels; it reaches the
        diagnostics only through the prediction. The recordings are the same as for
        the correct model, so the differences are paired, and HPD overlap and the
        p-value move by less than 1 percentage point."""
        correct, wrong = scenarios["correct"], scenarios[scenario]
        assert np.abs(wrong.pvalue - correct.pvalue).mean() < 0.01
        assert np.abs(wrong.hpd - correct.hpd).mean() < 0.01
