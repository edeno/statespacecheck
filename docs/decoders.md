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
predictive distribution is `results.predictive_posterior` from
`model.predict(..., return_outputs="predictive_posterior")`, and
the place fields are stored one unit per row, over all position bins. Keep only the
bins on the track, and transpose the place fields:

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

A clusterless spike's mark is its waveform features, so the predictive p-value comes from
`monte_carlo_mark_pvalue`, which needs the model as three pieces: the log joint mark
intensity, a sampler of marks, and the ground intensity (the total spike rate at each
position). For `non_local_detector`'s clusterless KDE model, all three follow from the
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


def clusterless_kde_model(encoding_model):
    """The log mark intensity, mark sampler and ground intensity of a fitted
    non_local_detector clusterless KDE encoding model, over its interior bins.

    A mark is ``[electrode, feature_1, ..., feature_d]``: the electrode is part of the
    mark. Assumes every electrode has the same number of features.
    """
    environment = encoding_model["environment"]
    bins = np.asarray(environment.place_bin_centers_)[environment.is_track_interior_.ravel()]
    log_occupancy = np.log(np.asarray(encoding_model["occupancy"]))
    position_std = np.asarray(encoding_model["position_std"])
    electrodes = []
    for features, positions, weights, rate in zip(
        encoding_model["encoding_spike_waveform_features"],
        encoding_model["encoding_positions"],
        encoding_model["encoding_weights"],
        encoding_model["mean_rates"],
        strict=True,
    ):
        features, positions, weights = map(np.asarray, (features, positions, weights))
        waveform_std = np.broadcast_to(
            np.asarray(encoding_model["waveform_std"]), features.shape[1]
        )
        # Weighted position kernel of each encoding spike at each bin, (n_encoding, n_bins)
        kernel = weights[:, None] * np.exp(
            norm.logpdf(bins[None, :, :], positions[:, None, :], position_std).sum(-1)
        )
        position_density = kernel.sum(axis=0)  # sum_j w_j K(x, p_j)
        electrodes.append(
            {
                "features": features,
                "waveform_std": waveform_std,
                "kernel": kernel,
                "log_scale": np.log(float(rate)) - np.log(weights.sum()) - log_occupancy,
                "ground": float(rate)
                * position_density
                / weights.sum()
                / np.exp(log_occupancy),
                "spike_cdf": np.cumsum(kernel / position_density, axis=0).T,  # (n_bins, n_enc)
            }
        )
    ground = sum(electrode["ground"] for electrode in electrodes)
    electrode_cdf = np.cumsum(
        np.stack([e["ground"] for e in electrodes], axis=1) / ground[:, None], axis=1
    )

    def log_mark_intensity(marks):
        marks = np.asarray(marks)
        out = np.full((len(marks), len(bins)), -np.inf)
        for index, electrode in enumerate(electrodes):
            rows = marks[:, 0] == index
            if not rows.any():
                continue
            # log K_wf(y, f_j), (n, n_encoding); factor out each mark's largest term
            log_waveform = norm.logpdf(
                marks[rows, None, 1:], electrode["features"], electrode["waveform_std"]
            ).sum(-1)
            largest = log_waveform.max(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                out[rows] = (
                    largest
                    + np.log(np.exp(log_waveform - largest) @ electrode["kernel"])
                    + electrode["log_scale"]
                )
        return out

    def sample_marks(state_bins, rng):
        marks = np.empty((len(state_bins), 1 + electrodes[0]["features"].shape[1]))
        which = (electrode_cdf[state_bins] <= rng.random(len(state_bins))[:, None]).sum(1)
        marks[:, 0] = np.minimum(which, len(electrodes) - 1)
        for index, electrode in enumerate(electrodes):
            rows = np.flatnonzero(marks[:, 0] == index)
            cdf = electrode["spike_cdf"][state_bins[rows]]
            spike = np.minimum(
                (cdf <= rng.random(len(rows))[:, None]).sum(1), cdf.shape[1] - 1
            )
            marks[rows, 1:] = rng.normal(
                electrode["features"][spike], electrode["waveform_std"]
            )
        return marks

    return log_mark_intensity, sample_marks, ground
```

Applied to a fitted model and its predictive distribution:

<!-- not-executed -->
```python
log_mark_intensity, sample_marks, ground_intensity = clusterless_kde_model(
    model.encoding_model_[("", 0)]
)
# predictive: (n_time, n_bins) on the interior bins, as in the example above.
# Marks of the spikes in [time[0], time[-1]], with the electrode as the first column:
observed_marks = np.concatenate(
    [np.column_stack([np.full(len(f), e), f]) for e, f in enumerate(decoded_features)]
)
check = ssc.monte_carlo_mark_pvalue(
    predictive[event_time_ind],
    log_mark_intensity,
    observed_marks,
    ground_intensity=ground_intensity,
    sample_marks=sample_marks,
    rng=0,
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
