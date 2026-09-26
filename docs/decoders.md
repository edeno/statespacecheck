# Using statespacecheck with your decoder

The per-spike diagnostics need four arrays, all of which a grid-based decoder already
computes:

| Argument of `event_diagnostics` | Shape | What it is |
| --- | --- | --- |
| `predictive` | `(n_time, n_bins)` or `(n_time, n_x, n_y)` | The one-step predictive distribution p(x_k \| y_1:k-1) in each time bin |
| `mark_intensities` | `(n_bins, n_units)` or `(n_x, n_y, n_units)` | Each unit's firing rate (or expected count per bin) at each position |
| `event_time_ind` | `(n_spikes,)` | The time bin of each spike, as an integer index |
| `event_marks` | `(n_spikes,)` | The unit of each spike, as an integer index |

The predictive distribution and the firing rates must use the same position bins. Rates
and expected counts give the same results, because a common bin width cancels.

## Any grid decoder

If your decoder runs a filter over position bins, record its prediction before each
update. For spikes stored as a count matrix, list one event per spike:

```python
import numpy as np
import statespacecheck as ssc

rng = np.random.default_rng(0)
n_time, n_bins, n_units = 500, 40, 12
predictive = rng.dirichlet(np.ones(n_bins), size=n_time)  # from your filter
place_fields = rng.gamma(2.0, size=(n_bins, n_units))  # from your encoding model
spike_counts = rng.poisson(0.05, size=(n_time, n_units))  # (n_time, n_units)

time_bin, unit = np.nonzero(spike_counts)
n_spikes = spike_counts[time_bin, unit]
event_time_ind, event_marks = np.repeat(time_bin, n_spikes), np.repeat(unit, n_spikes)

diagnostics = ssc.event_diagnostics(predictive, place_fields, event_time_ind, event_marks)
print(diagnostics.hpd_overlap.shape == event_time_ind.shape)
```

```text
True
```

For spike times instead of counts, convert each spike's time to its time-bin index,
for example `np.digitize(spike_times, time_bin_edges) - 1`.

## non_local_detector

For a fitted sorted-spikes model from
[non_local_detector](https://github.com/LorenFrankLab/non_local_detector), the
predictive distribution is `results.predictive_posterior` from `model.predict(...)`, and
the place fields are stored one unit per row, over all position bins. Keep only the
bins on the track, and transpose the place fields:

<!-- not-executed -->
```python
import numpy as np
import statespacecheck as ssc

results = model.predict(spike_times=spike_times, time=time)
predictive = results.predictive_posterior.dropna("state_bins")  # on-track bins only

# A model with several discrete states (e.g. continuous and fragmented) indexes
# state_bins by (state, position): sum over the states.
if "state" in predictive.indexes["state_bins"].names:
    predictive = predictive.unstack("state_bins").sum("state", skipna=False)
predictive = np.asarray(predictive)  # (n_time, n_bins)

place_fields = model.encoding_model_[("", 0)]["place_fields"]  # (n_units, all bins)
n_positions = place_fields.shape[1]
# The track mask is repeated for each discrete state; the place fields are shared
on_track = np.asarray(model.is_track_interior_state_bins_, dtype=bool)
on_track = on_track.reshape(-1, n_positions)[0]
place_fields = place_fields[:, on_track].T  # (n_bins, n_units)

# The decoder's convention for assigning spikes to time bins
event_time_ind = np.concatenate([np.digitize(t, time[1:-1]) for t in spike_times])
event_marks = np.concatenate([np.full(len(t), u) for u, t in enumerate(spike_times)])

diagnostics = ssc.event_diagnostics(predictive, place_fields, event_time_ind, event_marks)
```

Check that `predictive.shape[1] == place_fields.shape[0]`. The companion paper's code
handles more cases (several environments, state-specific track masks); see
[`figure04_place_fields.py`](https://github.com/edeno/statespacecheck-paper/blob/main/src/statespacecheck_paper/figure04_place_fields.py)
in the paper repository.

## A Gaussian (Kalman filter) prediction

The diagnostics compare distributions on a grid. For a decoder whose prediction is
Gaussian, evaluate the predictive density and each unit's rate on a grid of positions
covering the range of interest:

```python
import numpy as np
import statespacecheck as ssc

grid = np.linspace(0, 100, 101)  # positions (cm)
predicted_mean = np.array([20.0, 22.0, 60.0])  # from the Kalman filter, per time bin
predicted_sd = np.array([4.0, 4.0, 5.0])
predictive = np.exp(-0.5 * ((grid - predicted_mean[:, None]) / predicted_sd[:, None]) ** 2)
predictive /= predictive.sum(axis=1, keepdims=True)  # (n_time, n_bins)

centers = np.array([20.0, 60.0])  # each unit's preferred position
place_fields = 0.02 + 15 * np.exp(-0.5 * ((grid[:, None] - centers) / 7) ** 2)

diagnostics = ssc.event_diagnostics(predictive, place_fields, [0, 1, 2], [0, 0, 0])
print(diagnostics.hpd_overlap.round(2))
print(diagnostics.predictive_pvalue.round(3))
```

```text
[1. 1. 0.]
[1.    1.    0.002]
```

The third spike, from a unit with a field at 20 cm while the prediction is at 60 cm,
has no overlap and a small p-value. (A higher background rate spreads each spike's
likelihood over the whole track and raises its HPD overlap everywhere; thresholds from
a baseline period account for this.)

## Common problems

- **"mark_intensities must have shape (..., n_marks)"**: the table is probably stored
  one unit per row; pass its transpose. The error message suggests this when the
  shapes fit.
- **"predictive must contain only finite nonnegative values"**: positions outside the
  track are marked NaN. Keep only the valid bins, in both arrays.
- **A unit with zero rate at every position**: such a unit cannot have spikes, so its
  spikes have no likelihood; the error names the unit and its spikes.
- **Which distribution to compare**: the paper uses the one-step predictive
  distribution. A filter or smoother distribution already includes the current
  spike's information, so comparing it with that spike is not an independent check
  and will flag less.
- **Nonuniform bins**: the HPD overlap counts bins, which matches the paper's region
  volume only when bins have equal size. Resample to a uniform grid if they do not.
