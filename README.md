# statespacecheck

[![PyPI version](https://img.shields.io/pypi/v/statespacecheck.svg)](https://pypi.org/project/statespacecheck/)
[![Python versions](https://img.shields.io/pypi/pyversions/statespacecheck.svg)](https://pypi.org/project/statespacecheck/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/edeno/statespacecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/edeno/statespacecheck/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edeno/statespacecheck/branch/main/graph/badge.svg)](https://codecov.io/gh/edeno/statespacecheck)

**Local goodness-of-fit diagnostics for state space models: find the observations, down
to individual spikes, where a decoder disagrees with the data.**

<!-- --8<-- [start:intro] -->
A state space model decodes a latent state (for example an animal's position) by
combining a prediction from the past with the evidence in each new observation.
`statespacecheck` asks, for every observation, whether the two agree: does the
observation fall where the model's one-step prediction expected it? Global scores
such as the likelihood of a whole session cannot say *when* a model fails; these
diagnostics can, so a misfit can be traced to a period, a behavior, or a part of the
model.

The package implements the methods of the paper *Local goodness-of-fit measures for
neural decoding* (Zeng, Comrie, Frank, Eden and Denovellis). Its analysis code and an
interactive website are at
[statespacecheck-paper](https://github.com/edeno/statespacecheck-paper) and
<https://edeno.github.io/statespacecheck-paper/>.
<!-- --8<-- [end:intro] -->

## Installation

```bash
pip install statespacecheck
```

## Quick start

<!-- --8<-- [start:quickstart] -->
Given a decoder's one-step predictive distribution, each unit's firing rate at each
position, and which unit fired in which time bin, compute three diagnostics for every
spike and flag the poorly fit ones. Here a simulated decoder predicts the animal's
position correctly in the first half of the recording and the mirror-image position
in the second half:

```python
import numpy as np
import statespacecheck as ssc

rng = np.random.default_rng(0)
position = np.linspace(0, 100, 51)  # position bins (cm)
n_time, n_units, dt = 2000, 20, 0.02  # time bins, units, bin width (s)

# Place fields: firing rate (Hz) of each unit at each position, (n_bins, n_units)
centers = np.linspace(0, 100, n_units)
place_fields = 0.1 + 20 * np.exp(-0.5 * ((position[:, None] - centers) / 8) ** 2)

# The animal runs back and forth; spikes follow the place fields
true_position = 50 + 45 * np.sin(np.arange(n_time) * dt * 0.6)
true_bin = np.abs(position[:, None] - true_position).argmin(axis=0)
spike_counts = rng.poisson(place_fields[true_bin] * dt)  # (n_time, n_units)
time_bin, unit = np.nonzero(spike_counts)
n_spikes = spike_counts[time_bin, unit]
time_ind, unit = np.repeat(time_bin, n_spikes), np.repeat(unit, n_spikes)  # one per spike

# A decoder's one-step predictive distribution, (n_time, n_bins): it tracks the
# animal in the first half and predicts the mirror-image position in the second
predicted = np.where(np.arange(n_time) < n_time // 2, true_position, 100 - true_position)
predictive = np.exp(-0.5 * ((position - predicted[:, None]) / 5) ** 2)
predictive /= predictive.sum(axis=1, keepdims=True)

# Diagnostics for every spike; thresholds from a baseline period; flags
diagnostics = ssc.event_diagnostics(predictive, place_fields, time_ind, unit)
baseline = time_ind < n_time // 4
flags = ssc.flag_events(
    diagnostics,
    hpd_overlap_threshold=ssc.baseline_threshold(diagnostics.hpd_overlap[baseline], 0.01),
)
misfit = time_ind >= n_time // 2
for name, flagged in [
    ("HPD overlap", flags.hpd_overlap),
    ("p-value", flags.predictive_pvalue),
]:
    print(
        f"{name:12s} flagged: {flagged[~misfit].mean():.0%} of spikes before, "
        f"{flagged[misfit].mean():.0%} during the misfit"
    )
```

```text
HPD overlap  flagged: 2% of spikes before, 81% during the misfit
p-value      flagged: 2% of spikes before, 85% during the misfit
```
<!-- --8<-- [end:quickstart] -->

The [per-event tutorial](https://edeno.github.io/statespacecheck/tutorials/05_per_event_diagnostics/)
walks through this workflow with a real filter, and the
[decoder guide](https://edeno.github.io/statespacecheck/decoders/) shows how to get
these inputs from your own decoder.

<!-- --8<-- [start:reading] -->
## What your decoder provides

| Argument | Shape | What it is |
| --- | --- | --- |
| `predictive` | `(n_time, n_bins)` or `(n_time, n_x, n_y)` | The one-step predictive distribution p(x_t \| y_1:t-1) in each time bin |
| `mark_intensities` | `(n_bins, n_units)` or `(n_x, n_y, n_units)` | Each unit's firing rate at each position (its place field) |
| `event_time_ind` | `(n_spikes,)` | The time bin of each spike (an integer index) |
| `event_marks` | `(n_spikes,)` | The unit of each spike (an integer index) |

Common pitfalls:

- **Place fields stored one unit per row**, `(n_units, n_bins)`: pass `place_fields.T`.
- **Positions outside the track** marked NaN: the per-event functions need finite
  values, so keep only the valid bins in both arrays.
- **Switching models** (for example continuous and fragmented dynamics): sum the
  predictive distribution over the discrete states first, so every model is compared
  on the same position grid.
- **Spike times**: `event_time_ind` holds bin indices; convert times with
  `np.digitize(spike_times, time_bin_edges) - 1`.

## Reading the results

| Diagnostic | Measures | Poor fit when | The paper's rule |
| --- | --- | --- | --- |
| HPD overlap | Overlap of the 95% highest-density regions of the prediction and the spike's likelihood | Low | At or below the 1st percentile of a baseline period (`baseline_threshold`), or a fixed cutoff |
| Predictive p-value | How unexpected the unit that fired is, given the prediction | Low | At or below 0.05 |
| KL divergence | How different the two distributions are | High | A reference only: it is also large when a broad prediction is consistent with a precise spike |

The diagnostics measure *consistency*, not similarity: a spike is consistent with the
prediction when it falls where the prediction put probability, even if the prediction is
much broader. See [Interpreting the diagnostics](https://edeno.github.io/statespacecheck/interpretation/).
<!-- --8<-- [end:reading] -->

## The paper's quantities in the package

| Paper | Function |
| --- | --- |
| Normalized single-event likelihood | `event_likelihood` |
| HPD overlap (Szymkiewicz–Simpson overlap of 95% HPD regions) | `hpd_overlap`, `highest_density_region` |
| KL divergence D(predictive ‖ likelihood) | `kl_divergence` |
| Predictive distribution over units | `predictive_mark_probabilities` |
| Rank-based predictive p-value (exact sum over units) | `mark_predictive_pvalue` |
| All three diagnostics for every spike | `event_diagnostics` |
| Thresholds from a baseline period; flagging | `baseline_threshold`, `flag_events` |

The package also has tools the paper does not use: time-bin versions of the diagnostics
for a whole-bin likelihood, run-based flagging of time series (`statespacecheck.periods`),
a generic Monte Carlo predictive check (`predictive_pvalue`), and `plot_diagnostics`. The
[API reference](https://edeno.github.io/statespacecheck/reference/) marks which is which.

## Documentation

<https://edeno.github.io/statespacecheck>: tutorials, interpreting the diagnostics,
using the package with your decoder, and the API reference.

## Citation

<!-- --8<-- [start:citation] -->
If you use this package in your research, please cite it; `CITATION.cff` in
the repository records the version and release date:

```bibtex
@software{statespacecheck,
  title={statespacecheck: Goodness-of-fit diagnostics for state space models},
  author={Denovellis, Eric and Zeng, Sirui and Eden, Uri T.},
  url={https://github.com/edeno/statespacecheck}
}
```

A DOI will be added once releases are archived on Zenodo.

Please also cite the companion paper for the methods: *Local goodness-of-fit measures
for neural decoding* (Zeng, Comrie, Frank, Eden and Denovellis); its analysis code is
at <https://github.com/edeno/statespacecheck-paper>.
<!-- --8<-- [end:citation] -->

## Contributing and license

Contributions are welcome; see
[CONTRIBUTING.md](https://github.com/edeno/statespacecheck/blob/main/CONTRIBUTING.md).
MIT License.
