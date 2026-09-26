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

A p-value near 1 is good fit: the observed unit was among the most probable ones. Only
small p-values indicate misfit. With spike-sorted data the p-value is exact and takes
only a few distinct values, so it is conservative (flagging at `p <= 0.05` flags at most
5% of well-fit spikes).

## Choosing thresholds

A diagnostic's typical values depend on the data and the model, so there is no
universal cutoff for HPD overlap or KL divergence. The paper uses two approaches:

- **From a baseline period** in which the model is believed to fit (in a simulation, a
  well-specified period; in data, for example, a period of running when decoding is
  reliable): flag HPD overlap at or below the baseline's 1st percentile and KL
  divergence at or above its 99th percentile. `baseline_threshold` computes these.
- **Fixed cutoffs**: the predictive p-value at or below 0.05; in the paper's real data,
  also HPD overlap at or below 0.05.

`flag_events` applies either kind, flagging each diagnostic separately and treating
values equal to the threshold as flagged:

```python
import numpy as np
import statespacecheck as ssc

rng = np.random.default_rng(1)
predictive = rng.dirichlet(np.ones(20), size=300)  # (n_time, n_bins)
place_fields = rng.gamma(2.0, size=(20, 6))  # (n_bins, n_units)
time_ind, units = rng.integers(0, 300, 800), rng.integers(0, 6, 800)
diagnostics = ssc.event_diagnostics(predictive, place_fields, time_ind, units)

baseline = time_ind < 100
flags = ssc.flag_events(
    diagnostics,
    hpd_overlap_threshold=ssc.baseline_threshold(diagnostics.hpd_overlap[baseline], 0.01),
    kl_divergence_threshold=ssc.baseline_threshold(diagnostics.kl_divergence[baseline], 0.99),
    pvalue_threshold=0.05,
)
print(f"{flags.hpd_overlap.mean():.1%} of spikes flagged by HPD overlap")
```

## From flags to a modeling decision

The diagnostics are local, so flags can be traced to their cause:

- **When**: flags concentrated in a period (a behavior, a replay event) point to what
  the model does not capture then.
- **Which units**: flags concentrated in some units point to the observation model,
  for example place fields that have changed.
- **Which model**: evaluating two models on the same spikes and counting the spikes
  flagged under one but not the other shows which observations a revision explains.
  In the paper, adding a fragmented state to a continuous decoder removed most HPD
  overlap flags during a candidate replay event.

The [per-spike tutorial](tutorials/05_per_event_diagnostics.ipynb) shows each step.

## What the diagnostics do not check

The single-event likelihood leaves out terms shared by all spikes in a time bin: the
Poisson exposure term, silent units, and the total spike count. Misfit in those (for
example, a wrong overall firing rate) needs a complementary count-based or
time-rescaling check.

## Time bins instead of spikes

`kl_divergence` and `hpd_overlap` also accept a whole-bin likelihood, and
`statespacecheck.periods` flags runs of time bins. These are extensions beyond the
paper. A whole-bin likelihood includes the exposure term and the silent units, and
run-based flags (`min_len`) assume a regular time series, so they do not apply to
per-spike values.
