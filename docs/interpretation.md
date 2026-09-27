# Interpreting the diagnostics

A state space model estimates a latent state from two sources of information: the
**one-step predictive distribution** $P_k(x) = p(x_k \mid y_{1:k-1})$, which carries
everything learned from past observations through the model's dynamics, and the
**likelihood** of the current observation. For a spike, the likelihood is the
**single-event likelihood** $Q(x)$: the firing rate of the unit that fired, normalized
over positions (`event_likelihood`).

## Consistency, not similarity

The diagnostics ask whether the two sources are **consistent**: whether they put their
probability in overlapping regions of the state space. They do not ask whether the two
distributions are similar. A broad prediction and a narrow spike likelihood inside it
are consistent; so are a precise prediction and a spike that fires over a wide range.
Consistency is also different from accuracy: the decoded position may be far from the
animal's physical position (during replay, for example) and still be consistent.

## The three diagnostics

| Diagnostic | Function | Range | Poor fit when | What it measures |
| --- | --- | --- | --- | --- |
| HPD overlap | `hpd_overlap` | 0 to 1 | Low | How much of the smaller of the two 95% highest-density regions lies inside the other. It is 1 whenever one region is nested inside the other. |
| Predictive p-value | `mark_predictive_pvalue` | 0 to 1 | Low | The probability, under the prediction, that the next spike comes from a unit at most as probable as the one that fired. |
| KL divergence | `kl_divergence` | 0 to ∞ | High | How different the prediction $P$ is from the likelihood $Q$, $D(P \| Q)$. |

`event_diagnostics` computes all three for every spike.

**HPD overlap and the predictive p-value are the primary diagnostics.** KL divergence
measures difference rather than consistency: it grows whenever the prediction is broad
relative to the likelihood, which happens for consistent spikes too (for example, a
precise spike from a sparsely firing cell). Use it as a reference.

A p-value near 1 means the observed unit was among the most probable ones. Only small
p-values indicate misfit. With spike-sorted data the p-value is exact and takes only a
few distinct values, so it is conservative: under a correct model, a spike drawn from
its predictive mark distribution has probability at most 5% of `p <= 0.05`.

## What passing a diagnostic means

A high HPD overlap or a large predictive p-value means the spike **passes that
diagnostic**: it is consistent with the prediction in that one respect. It does not
show that the model is correct.

- **Broad predictions pass easily.** A prediction spread over most of the track has a
  95% region that contains almost any spike's likelihood region, so its HPD overlap is
  near one. Its predictive probabilities of units approach each unit's overall share of
  spikes, so few p-values are small. Such a prediction says little about position. In the
  simulation below, a decoder whose transition was far too wide raised the mean HPD
  overlap from 0.96 to 0.99 and flagged fewer spikes by p-value than the correct one;
  only the typical KL divergence rose.
- **Rare units get small p-values under a correct model.** The p-value of a spike is
  the predictive probability of all units at most as probable as the one that fired,
  so the least probable unit, predicted with probability 0.02, has p = 0.02 on every
  spike, and all its spikes are flagged at 0.05, although a spike drawn from the
  prediction still has probability at most 5% of being flagged. Flags concentrated in a few units therefore do not, alone,
  show that their place fields are wrong. Compare each unit's flagged fraction with the
  fraction the model itself would flag for that unit, which can be far above 5% (in
  this example, 100%): estimate it by simulating spikes from the model (drawing each
  spike's unit from its predictive probabilities in the same time bins) and flagging
  them. Or evaluate a revised model on the same spikes.

Each spike is checked on its own. A prediction that makes every unit equally probable
gives every spike p = 1, however unbalanced the observed mixture of units:

```python
import numpy as np
import statespacecheck as ssc

place_fields = np.ones((10, 4))  # four units, each at the same rate everywhere
predictive = np.full((100, 10), 0.1)  # (n_time, n_bins)
units = np.array([0] * 95 + [1, 2, 3, 3, 3])  # 95 of 100 spikes from unit 0
diagnostics = ssc.event_diagnostics(predictive, place_fields, np.arange(100), units)
print(np.unique(diagnostics.predictive_pvalue), np.bincount(units))
```

```text
[1.] [95  1  1  3]
```

The prediction gives each unit a probability of 0.25, and 95 of 100 spikes from one unit
is far from that, but no single spike is unexpected. Testing the proportions of units
over many spikes needs a separate test of the counts (for example a chi-squared test
against the summed predictive probabilities).

## Calibration

Under a correct model, a spike drawn from its predictive mark distribution has
probability at most `alpha` of `p <= alpha`. Over a recording, the expected number of
flagged spikes is therefore at most `alpha` times the expected number of spikes:
`E[flagged] <= alpha E[spikes]`. That does not bound the fraction observed in a given
recording, which varies around its expectation and can exceed `alpha` by chance (more
than for independent spikes, because spikes in the same time bin share a prediction and
successive predictions depend on the same spikes). Nor does it bound an average of
per-recording fractions: a recording with few spikes weighs as much as one with many.
If most recordings have a few spikes from a rarely predicted unit and the rest have
many from a common one, most recordings are entirely flagged while few spikes overall
are. Pool the counts instead: total flagged spikes over total spikes.

The guarantee holds only relative to the reference the p-value is computed from:

- **The one-step predictive distribution**, which does not use the spike being tested.
  A filtered or smoothed posterior already includes the spike, so the spike looks more
  expected than it is and too few spikes are flagged.
- **A model fit on other data.** Place fields estimated from the spikes being evaluated
  fit those spikes better than they would fit new ones, which also flags too few.
  Where possible, fit on one part of a session and evaluate on another.
- **Monte Carlo p-values** (clusterless marks) add sampling error (see below).

In the simulation below, with an exact filter and the true place fields, 4.4% of all
spikes over 12 recordings had `p <= 0.05`.

## Choosing thresholds

A diagnostic's typical values depend on the data and the model, so there is no
universal cutoff for HPD overlap or KL divergence. The paper uses two approaches:

- **From a baseline period** in which the model is believed to fit: flag HPD overlap
  at or below the baseline's 1st percentile and KL divergence at or above its 99th
  percentile. `baseline_threshold` computes these. The paper's simulation used its
  opening, well-specified period; in real data, a period such as running, when
  decoding is reliable, could serve.
- **Fixed cutoffs**: the predictive p-value at or below 0.05 throughout; for the
  paper's real data, which had no clean baseline period, also HPD overlap at or below
  0.05 (and no cutoff for KL divergence).

`flag_events` applies either kind, flagging each diagnostic separately and treating
values equal to the threshold as flagged. Report each threshold with its rule and the
fraction of baseline spikes it actually flags:

```python
import numpy as np
import statespacecheck as ssc

rng = np.random.default_rng(1)
predictive = rng.dirichlet(np.ones(20), size=300)  # (n_time, n_bins)
place_fields = rng.gamma(2.0, size=(20, 6))  # (n_bins, n_units)
time_ind, units = rng.integers(0, 300, 800), rng.integers(0, 6, 800)
diagnostics = ssc.event_diagnostics(predictive, place_fields, time_ind, units)

baseline = time_ind < 100
thresholds = {
    "hpd_overlap": (
        "at or below",
        ssc.baseline_threshold(diagnostics.hpd_overlap[baseline], 0.01),
    ),
    "kl_divergence": (
        "at or above",
        ssc.baseline_threshold(diagnostics.kl_divergence[baseline], 0.99),
    ),
    "predictive_pvalue": ("at or below", 0.05),
}
flags = ssc.flag_events(
    diagnostics,
    hpd_overlap_threshold=thresholds["hpd_overlap"][1],
    kl_divergence_threshold=thresholds["kl_divergence"][1],
    pvalue_threshold=thresholds["predictive_pvalue"][1],
)
for name, (rule, value) in thresholds.items():
    flagged = getattr(flags, name)
    print(
        f"{name:18s} flagged {rule} {value:.3f}: {flagged[baseline].mean():.1%} of "
        f"baseline spikes, {flagged[~baseline].mean():.1%} of the rest"
    )
```

```text
hpd_overlap        flagged at or below 0.714: 1.5% of baseline spikes, 2.0% of the rest
kl_divergence      flagged at or above 1.000: 1.1% of baseline spikes, 2.6% of the rest
predictive_pvalue  flagged at or below 0.050: 0.0% of baseline spikes, 0.0% of the rest
```

### Ties

The comparisons are inclusive, as in the paper, so every value tied at a threshold is
flagged, and the fraction of baseline spikes flagged can exceed the requested quantile.
Ties are common:

- HPD overlap is exactly 1 whenever one region is nested inside the other (94% of
  spikes in the simulation below) and exactly 0 when the regions are disjoint. If more
  than 1% of baseline spikes have no overlap, the 1st percentile is 0 and all of them
  are flagged: 2.9% of spikes in the simulation, not 1%. If every baseline overlap is
  1, the threshold is 1 and every spike at 1 is flagged.
- KL divergence is exactly 0 when the prediction equals a spike's likelihood. If every
  baseline value is 0, the 99th percentile is 0 and every spike is flagged.

Report the flagged baseline fraction with the threshold, as above, so that such cases
are visible.

### A baseline fraction is not a false-alarm rate

The fraction of baseline spikes flagged describes the baseline, from which the threshold
was also estimated. It does not guarantee the rate of false alarms elsewhere: other
periods differ in behavior, firing and the breadth of the prediction; spikes in a
recording are not independent; and nothing but the model's correctness makes the
baseline representative. Treat it as a reference point, not an error rate.

## From flags to a modeling decision

The diagnostics are local, so flags can be traced to their cause:

- **When**: flags concentrated in a period (a behavior, a replay event) point to what
  the model does not capture then.
- **Which units**: flags concentrated in some units point to the observation model,
  for example place fields that have changed.
- **Which model**: evaluating two models on the same spikes and counting the spikes
  flagged under one but not the other shows which observations a revision explains.
  In the paper, adding a fragmented state to a continuous decoder removed the HPD
  overlap flag from most of the spikes flagged under the continuous model, across the
  session.

The [per-spike tutorial](tutorials/05_per_event_diagnostics.ipynb) shows each step.

## Monte Carlo p-values

For marks that cannot be enumerated, `monte_carlo_mark_pvalue` (and
`clusterless_event_diagnostics`) return `r / B`: the fraction of `B` replicated marks
whose predictive density is at most the observed mark's. This estimates the predictive
tail probability, with standard error `sqrt(p (1 - p) / B)`, and can be exactly 0 when no
replicate is as unexpected as the observed mark. `predictive_pvalue` is computed the
same way.

A rank test at a finite `B` uses `(r + 1) / (B + 1)` instead (Phipson and Smyth, 2010,
[*Permutation p-values should never be zero*](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)).
It is valid in finite samples when the observed mark and the replicates are
exchangeable under the model (the observed mark drawn from the same predictive
distribution as the replicates): rejecting at `p <= alpha` has probability at most
`alpha`. It can be conservative: the smallest attainable value is `1 / (B + 1)`, so
with `B = 10` no p-value reaches 0.05, and the level equals `alpha` only when
`alpha (B + 1)` is an integer and the densities have no ties. The package reports
`r / B`; to get the rank-test value, compute `(p * B + 1) / (B + 1)` from the returned
`p`.

## What the diagnostics do not check

The per-spike diagnostics compare each spike with its prediction. They leave gaps:

- **Counts.** The single-event likelihood leaves out terms shared by all spikes in a
  time bin: the Poisson exposure term, silent units, and the total spike count. A wrong
  overall firing rate (every place field scaled by the same factor) cancels from both
  the likelihood and the predictive probabilities of units and reaches the diagnostics
  only through the prediction.
- **Timing and dependence.** All spikes in a time bin are compared with the same
  prediction, and each spike on its own: when in the bin a spike occurs, bursts,
  refractoriness and other dependence between spikes are not checked.
- **Proportions of units** across spikes, as the example above shows.
- **Decoding accuracy.** A consistent prediction can be far from the animal's position,
  and a prediction too broad to be useful passes HPD overlap and the p-value.

Misfit in these needs a complementary check, such as a count-based test or
time-rescaling.

### What a simulation detected

A seeded simulation of 20 place cells on a 1-D track (`tests/test_statistical_validation.py`)
decoded 12 independent recordings per scenario with a grid filter, with thresholds
from separate correctly specified recordings (HPD overlap at its 1st percentile, KL
divergence at its 99th, the p-value at 0.05). Fractions of all spikes flagged, pooled
over the recordings (about 38,700 spikes per scenario):

| Scenario | p-value | HPD overlap | KL divergence |
| --- | --- | --- | --- |
| Correct model | 4.4% | 2.9% | 1.1% |
| Changed place fields (a third of the units moved) | **18.2%** | **6.2%** | **2.6%** |
| Prediction far too broad (transition 17 times too wide) | 1.1% | 0.0% | 0.0% |
| Overall rate 4 times too high | 3.6% | 3.2% | **3.4%** |
| Overall rate 4 times too low | 4.5% | 2.9% | 0.8% |

Changed place fields were detected by all three diagnostics. The broad prediction and
the wrong overall rates were blind spots of HPD overlap and the p-value: they moved by
less than a percentage point, or in the direction of better fit. KL divergence rose
only when the rate was too high, and its median rose for the broad prediction without
crossing the baseline's 99th percentile. One simulation shows what can happen, not
how often; other models and data will differ.

## Time bins instead of spikes

`kl_divergence` and `hpd_overlap` also accept a whole-bin likelihood, and
`statespacecheck.periods` flags runs of time bins. These are extensions beyond the
paper. A whole-bin likelihood includes the exposure term and the silent units, and
run-based flags (`min_len`) assume a regular time series, so they do not apply to
per-spike values.
