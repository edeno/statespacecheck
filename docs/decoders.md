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

The examples in this section need `non_local_detector` from its `main` branch (commit
`84259d4` of 2026-09-22 or later): the released version 0.6.9 cannot return the
predictive distribution (`predict` has no `return_outputs`), and its clusterless KDE model
stores no encoding weights. Install it with
`pip install git+https://github.com/LorenFrankLab/non_local_detector`.

For a fitted sorted-spikes model from
[non_local_detector](https://github.com/LorenFrankLab/non_local_detector), the
predictive distribution is `results.predictive_posterior` from
`model.predict(..., return_outputs="predictive_posterior")`, and
the place fields are stored one unit per row, over all position bins. Keep only the
bins on the track, and transpose the place fields. Assign spikes to time bins as the
decoder does, with the `time` passed to `predict`. (A change in development makes
`predict` take the decoding bin edges `time_edges` instead, with `results.time` then
holding the bins' centers; with it, use the same code with `time_edges` in place of
`time`.)

<!-- not-executed -->
```python
import numpy as np
import statespacecheck as ssc

# predict returns only the smoother posterior unless asked for the predictive one
results = model.predict(
    spike_times=spike_times, time=time, return_outputs="predictive_posterior"
)
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

# The decoder's convention: it uses only the spikes in [time[0], time[-1]] and assigns
# them to time bins with np.digitize (other spikes would be put in the first or last bin)
decoded_spikes = [np.asarray(t) for t in spike_times]
decoded_spikes = [t[(t >= time[0]) & (t <= time[-1])] for t in decoded_spikes]
event_time_ind = np.concatenate([np.digitize(t, time[1:-1]) for t in decoded_spikes])
event_marks = np.concatenate([np.full(len(t), u) for u, t in enumerate(decoded_spikes)])

diagnostics = ssc.event_diagnostics(predictive, place_fields, event_time_ind, event_marks)
```

Check that `predictive.shape[1] == place_fields.shape[0]`. The companion paper's code
handles more cases (several environments, state-specific track masks); see
[`figure04_place_fields.py`](https://github.com/edeno/statespacecheck-paper/blob/main/src/statespacecheck_paper/figure04_place_fields.py)
in the paper repository.

## Clusterless decoders (non_local_detector KDE)

This needs the same `non_local_detector` version as the section above. A clusterless
spike's mark is its waveform features, so the predictive p-value comes from
`monte_carlo_mark_pvalue`, which needs the model as a `MarkModel` of three pieces: the log
joint mark intensity, a sampler of marks, and the ground intensity (the total spike rate
at each position). For `non_local_detector`'s clusterless KDE model, all three follow from the
fitted encoding model:

- The electrode is part of the mark, `[electrode, feature_1, ..., feature_d]`: the
  intensity of a spike depends on which electrode recorded it.
- The intensity at electrode `e` is `rate_e * sum_j w_j K(x, p_j) K(y, f_j) / sum_j w_j /
  occupancy(x)` over its encoding spikes `j` (positions `p_j`, features `f_j`, weights
  `w_j`). Integrating over `y` gives the electrode's ground intensity, so a mark can be
  sampled exactly: an electrode in proportion to its ground intensity at the position,
  then an encoding spike in proportion to `w_j K(x, p_j)`, then features around it.
- It is computed in log space, from the fitted parameters. `non_local_detector`'s own
  log-intensity helpers floor small values (at `log(1e-15)`), which turns distinct
  tail densities into ties, and exponentiating in float32 underflows with many
  features.

<!-- not-executed -->
```python
import numpy as np
from scipy.stats import norm

from statespacecheck import MarkModel


def clusterless_kde_model(encoding_model, chunk_size=256):
    """The MarkModel (log mark intensity, mark sampler, ground intensity) of a fitted
    non_local_detector clusterless KDE encoding model, over its interior bins.

    A mark is ``[electrode, feature_1, ..., feature_d]``: the electrode is part of the
    mark. An electrode without (weighted) training spikes never fires under the model,
    so its marks have zero intensity. The other electrodes must have the same number of
    features: densities over different numbers of features are in different units, so
    ranking them against each other would depend on the features' units. Marks are
    evaluated ``chunk_size`` at a time, which bounds memory.
    """
    environment = encoding_model["environment"]
    bins = np.asarray(environment.place_bin_centers_)[environment.is_track_interior_.ravel()]
    occupancy = np.asarray(encoding_model["occupancy"])
    position_std = np.asarray(encoding_model["position_std"])
    electrodes = []  # None for an electrode that never fires
    for features, positions, weights, rate in zip(
        encoding_model["encoding_spike_waveform_features"],
        encoding_model["encoding_positions"],
        encoding_model["encoding_weights"],
        encoding_model["mean_rates"],
        strict=True,
    ):
        features, positions, weights = map(np.asarray, (features, positions, weights))
        if float(rate) == 0.0 or weights.sum() == 0.0:
            electrodes.append(None)
            continue
        # rate * w_j K(x, p_j) / sum_j w_j / occupancy(x): encoding spike j's share of
        # the intensity at each bin, (n_encoding, n_bins); zero where occupancy is zero
        scale = np.divide(
            float(rate) / weights.sum(),
            occupancy,
            out=np.zeros_like(occupancy),
            where=occupancy > 0,
        )
        position_kernel = np.exp(
            norm.logpdf(bins[None], positions[:, None], position_std).sum(-1)
        )
        electrodes.append(
            {"features": features, "kernel": weights[:, None] * position_kernel * scale}
        )
    feature_counts = {e["features"].shape[1] for e in electrodes if e is not None}
    if len(feature_counts) != 1:
        msg = (
            "The electrodes with spikes must all have the same number of waveform "
            f"features; got {sorted(feature_counts)}. Check groups of electrodes with the "
            "same number of features separately."
        )
        raise ValueError(msg)
    (n_features,) = feature_counts
    waveform_std = np.broadcast_to(encoding_model["waveform_std"], n_features)
    # Each electrode's ground intensity (the waveform kernel integrates to 1), (n_bins, n_electrodes)
    electrode_ground = np.stack(
        [np.zeros(len(bins)) if e is None else e["kernel"].sum(axis=0) for e in electrodes],
        axis=1,
    )
    ground = electrode_ground.sum(axis=1)

    def draw(weights, state_bins, rng):
        """An index for each state bin, in proportion to that bin's row of ``weights``."""
        uniform = rng.random(len(state_bins))
        choice = np.empty(len(state_bins), dtype=int)
        for state_bin in np.unique(state_bins):
            rows = state_bins == state_bin
            # Scaled to the largest weight, then normalized to [0, 1]: very small (even
            # subnormal) weights keep their proportions
            cdf = np.cumsum(weights[state_bin] / weights[state_bin].max())
            cdf /= cdf[-1]
            # side="right" skips zero weights; rounding can pass the last positive one
            index = np.searchsorted(cdf, uniform[rows], side="right")
            choice[rows] = np.minimum(index, np.flatnonzero(weights[state_bin])[-1])
        return choice

    def log_mark_intensity(marks):
        marks = np.asarray(marks, dtype=float)
        if marks.ndim != 2 or marks.shape[1] != 1 + n_features:
            msg = f"marks must have shape (n, {1 + n_features}): the electrode, then the features"
            raise ValueError(msg)
        electrode_ids = marks[:, 0]
        if not np.all(np.isin(electrode_ids, np.arange(len(electrodes)))):
            msg = f"electrodes must be indices 0..{len(electrodes) - 1}, in the encoding model's order"
            raise ValueError(msg)
        out = np.full((len(marks), len(bins)), -np.inf)
        for index, electrode in enumerate(electrodes):
            rows = np.flatnonzero(electrode_ids == index)
            if electrode is None:
                continue
            for start in range(0, len(rows), chunk_size):
                chunk = rows[start : start + chunk_size]
                # log K_wf(y, f_j), (n, n_encoding); factor out each mark's largest term
                log_waveform = norm.logpdf(
                    marks[chunk, None, 1:], electrode["features"], waveform_std
                ).sum(-1)
                largest = log_waveform.max(axis=1, keepdims=True)
                with np.errstate(divide="ignore"):
                    out[chunk] = largest + np.log(
                        np.exp(log_waveform - largest) @ electrode["kernel"]
                    )
        return out

    def sample_marks(state_bins, rng):
        marks = np.empty((len(state_bins), 1 + n_features))
        marks[:, 0] = draw(electrode_ground, state_bins, rng)
        for index, electrode in enumerate(electrodes):
            rows = np.flatnonzero(marks[:, 0] == index)
            if electrode is None or not len(rows):
                continue
            spike = draw(electrode["kernel"].T, state_bins[rows], rng)
            marks[rows, 1:] = rng.normal(electrode["features"][spike], waveform_std)
        return marks

    return MarkModel(log_mark_intensity, sample_marks, ground)
```

Applied to a fitted model and its predictive distribution:

<!-- not-executed -->
```python
mark_model = clusterless_kde_model(model.encoding_model_[("", 0)])
# predictive: (n_time, n_bins) on the interior bins, as in the example above.
# The spikes the decoder used, those in [time[0], time[-1]], electrode by electrode in
# the encoding model's order; the time bins and marks are built in the same order
in_bounds = [(t >= time[0]) & (t <= time[-1]) for t in spike_times]
decoded_times = [t[keep] for t, keep in zip(spike_times, in_bounds)]
decoded_features = [f[keep] for f, keep in zip(spike_waveform_features, in_bounds)]
event_time_ind = np.concatenate([np.digitize(t, time[1:-1]) for t in decoded_times])
# Marks: the electrode index, then the waveform features. An electrode without spikes
# adds no rows (and its features may not have the others' width), so it is skipped;
# the other electrodes keep their indices
blocks = [
    np.column_stack([np.full(len(f), e), f]) for e, f in enumerate(decoded_features) if len(f)
]
observed_marks = np.concatenate(blocks) if blocks else np.empty((0, 1))  # no spikes

check = ssc.monte_carlo_mark_pvalue(
    predictive[event_time_ind], mark_model, observed_marks, rng=0
)
```

On a fitted two-electrode model, these p-values agreed with numerical integration over the
marks to within Monte Carlo error, and the log intensity agreed with
`non_local_detector`'s to float32 precision. Two cases need more than this adapter:

- **The clusterless GMM backend** fits its ground intensity and its joint position and
  waveform model separately, so the saved ground intensity need not be the integral of
  the joint model's intensity, and a mismatch biases the p-values. Derive the ground
  intensity from the joint model's position marginal before using it here.
- **Detectors with more than one observation model**: local states (whose intensity
  depends on the animal's actual position), several encoding groups or environments, or
  a no-spike state. Summing the predictive distribution over such states and applying one
  mark model is not correct; each state's predictive mass needs its own model, with the
  predictive rows, valid bins and intensities kept aligned. Decoders whose states share
  one encoding model, such as continuous and fragmented dynamics, fit this interface.

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
  distribution, which does not include the current spike. A filter or smoother
  distribution already includes that spike's information, so the comparison is no
  longer with an independent prediction; the paper discusses a smoother, which also
  uses future observations, as an extension that could add statistical power.
- **Nonuniform bins**: the HPD overlap counts bins, which matches the paper's region
  volume only when bins have equal size. Resample to a uniform grid if they do not.
