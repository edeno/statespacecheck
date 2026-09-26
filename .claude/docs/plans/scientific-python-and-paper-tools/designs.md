# Designs: continuous-mark diagnostics

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md)

- [Event-weighted predictive](#event-weighted-predictive)
- [Vectorized state sampling](#vectorized-state-sampling)
- [Monte Carlo p-value](#monte-carlo-p-value)
- [Memory and batching](#memory-and-batching)
- [Clusterless diagnostics](#clusterless-diagnostics)
- [Test models](#test-models)

These algorithms were prototyped and checked against the package on 2026-09-25, using the discrete test model in [Test models](#test-models) and `mark_predictive_pvalue` as the reference:

- **Parity:** with 20,000 samples, the Monte Carlo p-value matched the exact one for all 40 events, with max |z| = 2.0.
- **Sampler:** frequencies from 200,000 draws matched the target probabilities; the largest cell error was 0.0023. Zero-probability bins were never drawn.

## Event-weighted predictive

Put this in `events.py` after `predictive_mark_probabilities` (`events.py:244`). Reuse `_validate_state_distribution` (`events.py:85-95`).

```python
def event_weighted_predictive(
    state_dist: DistributionArray, ground_intensity: NDArray[np.floating]
) -> DistributionArray:
    state = _validate_state_distribution(state_dist, "state_dist")
    spatial_shape = np.shape(state_dist)[1:]
    ground = np.asarray(ground_intensity)
    if ground.shape != spatial_shape:
        msg = (
            f"ground_intensity must have shape {spatial_shape} to match the state "
            f"distribution's spatial axes; got {ground.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(ground)) or np.any(ground < 0.0):
        msg = "ground_intensity must contain only finite nonnegative values"
        raise ValueError(msg)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = state * ground.ravel()
        total = weighted.sum(axis=1, keepdims=True)
    undefined = ~np.isfinite(total[:, 0]) | (total[:, 0] == 0.0)
    if undefined.any():
        bad = np.flatnonzero(undefined)
        msg = (
            "Event-weighted predictive distribution is undefined for rows with zero or "
            f"non-finite total event intensity; row indices: {bad[:10].tolist()}"
        )
        raise ValueError(msg)
    result: DistributionArray = (weighted / total).reshape(np.shape(state_dist))
    return result
```

Phase 1 turns on the EM rules, so the error messages must be assigned to `msg` before each `raise`.

## Vectorized state sampling

This is a private helper in `continuous_marks.py`. It draws `n_samples` flat bin indices per row with no Python loop. Each row's CDF is shifted by its row index, which makes all rows one increasing sequence that a single `searchsorted` can search.

```python
def _sample_state_bins(
    probabilities: NDArray[np.floating], n_samples: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw flat state-bin indices, shape (n_rows, n_samples), from each row of
    ``probabilities`` (n_rows, n_bins); rows must sum to 1."""
    n_rows, n_bins = probabilities.shape
    cdf = np.cumsum(probabilities, axis=1)
    cdf /= cdf[:, -1:]
    offsets = np.arange(n_rows)[:, np.newaxis]
    u = rng.random((n_rows, n_samples))
    flat = np.searchsorted((cdf + offsets).ravel(), (u + offsets).ravel(), side="right")
    ind = flat.reshape(n_rows, n_samples) - offsets * n_bins
    # Rounding in cdf + offset can land one past a row's last bin.
    np.clip(ind, 0, n_bins - 1, out=ind)
    return ind.astype(np.intp, copy=False)
```

- `side="right"` never selects a zero-probability bin, because a zero-probability bin adds no width to the CDF.
- Adding the offset costs about `eps * batch_size` of precision. That is negligible because rows are batched ([Memory and batching](#memory-and-batching)).

## Monte Carlo p-value

This processes one batch of events. `state` has shape `(nb, n_bins)` and is the flattened `state_dist[start:stop]`; `marks` is `observed_marks[start:stop]`.

```python
log_state = _safe_log(state / state.sum(axis=1, keepdims=True))  # (nb, n_bins)
log_norm = logsumexp(log_state + _safe_log(ground_flat), axis=1)  # log Σ Λ P
observed_intensity = _evaluate_intensity(mark_intensity, marks, nb, n_bins)
observed_log = logsumexp(log_state + _safe_log(observed_intensity), axis=1) - log_norm

event_weighted = event_weighted_predictive(state, ground_flat).reshape(nb, n_bins)
state_bins = _sample_state_bins(event_weighted, n_samples, rng)  # (nb, S)
replicated_marks = sample_marks(state_bins.ravel(), rng)
_check_leading_axis(replicated_marks, nb * n_samples, "sample_marks")
replicated_intensity = _evaluate_intensity(
    mark_intensity, replicated_marks, nb * n_samples, n_bins
).reshape(nb, n_samples, n_bins)
simulated_log = (
    logsumexp(_safe_log(replicated_intensity) + log_state[:, np.newaxis, :], axis=2)
    - log_norm[:, np.newaxis]
)
tol = 16 * np.finfo(float).eps * n_bins
pvalue[start:stop] = np.mean(simulated_log <= observed_log[:, np.newaxis] + tol, axis=1)
```

Helpers, all private in `continuous_marks.py`:

- `_safe_log(a)` is `np.log(a)` under `np.errstate(divide="ignore")`, so a zero becomes `-inf` without a warning. That matters because pytest will run with `filterwarnings=error`.
- `_evaluate_intensity(fn, marks, n, n_bins)`:
  - calls `fn(marks)`
  - checks the result has shape `(n, *spatial_shape)` and is finite and nonnegative
  - returns it reshaped to `(n, n_bins)`
  - raises `ValueError` otherwise
- `_check_leading_axis(arr, n, name)`: `ValueError` if `np.shape(arr)[0] != n`.

Edge cases, each with a test:

| Case | Behaviour | Why |
| --- | --- | --- |
| `λ(x, y_obs) = 0` wherever `P(x) > 0` | `observed_log = -inf`; `pvalue` counts replicates with `simulated_log = -inf` (usually 0) | The observed mark is impossible under the prediction, so a p-value near 0 is correct. |
| `Σ Λ P = 0` | `ValueError`, raised by `event_weighted_predictive` | Same contract as `predictive_mark_probabilities`. |
| `n_events = 0` | Empty outputs; neither callable is called | |

## Memory and batching

- The temporaries over every replicated mark at every state are `batch × n_samples × n_bins × 8 B` each, and peak memory is about six of them, because `logsumexp` copies its input. Measured in phase 4a at 1,000 samples × 512 bins: 101, 201, 403 and 805 MB at batch sizes 4, 8, 16 and 32, at the same speed (6.6–7.5 ms per event).
- So `DEFAULT_MONTE_CARLO_BATCH_SIZE = 8` (about 200 MB), not the 32 first planned. The comment next to the constant gives the arithmetic.
- The docstring gives the peak-memory formula and says how to lower `batch_size` for large grids.

## Clusterless diagnostics

`clusterless_event_diagnostics` mirrors the loop in `event_diagnostics` (`events.py:381-427`) and reuses its validation:

- `validate_coverage`
- `_validate_marks` for `event_time_ind`, against `n_time`
- `batch_size >= 1`
- `event_marks` must have one entry per event

Per batch:

```python
predictive_batch = predictive_flat[time_ind[start:stop]]
observed_intensity = _evaluate_intensity(mark_intensity, event_marks[start:stop], nb, n_bins)
likelihood_batch = event_likelihood(observed_intensity)  # raises on all-zero rows
event_hpd[start:stop] = hpd_overlap(predictive_batch, likelihood_batch, coverage=coverage)
event_kl[start:stop] = kl_divergence(predictive_batch, likelihood_batch)
event_pvalue[start:stop] = _monte_carlo_batch(...)  # shared with monte_carlo_mark_pvalue
```

- Factor the per-batch Monte Carlo body out as `_monte_carlo_batch`, which both public functions call, so there is one implementation.
- Consume `rng` in the same order in both functions: batch by batch, the state draws, then `sample_marks`. With the same seed and batch size, `clusterless_event_diagnostics(...).predictive_pvalue` then equals `monte_carlo_mark_pvalue(predictive[time_ind], ...).pvalue` exactly. Test this.

## Test models

Define these as fixtures in `tests/conftest.py` so phases 4a and 4b share them.

### Discrete model

- Gaussian place fields: `x = linspace(0, 1, 50)`, `centers = rng.random(12)` with seed 0, `rates = 0.2 + 10 * exp(-0.5 ((x[:, None] - centers) / 0.1)^2)`, giving shape `(n_bins=50, n_marks=12)`.
- `mark_intensity = lambda m: rates[:, m].T`.
- `sample_marks(bins, rng)` is an inverse-CDF categorical over `rates[bins] / rates[bins].sum(1)`, built with `(cum[bins] <= u[:, None]).sum(1)`.

### Clusterless 1-D model

- `n_units = 6` units on a 1-D track, each with a place field `r_u(x)` and a Gaussian waveform amplitude `N(y; μ_u, σ=0.3)`, with the `μ_u` spread over [1, 4].
- Joint intensity: `λ(x, y) = Σ_u r_u(x) N(y; μ_u, σ)`.
- `Λ(x) = Σ_u r_u(x)`. The Gaussian integrates to 1; there is no truncation.
- Sampler: pick a unit `u` with probability `r_u(x) / Λ(x)`, then draw `y ~ N(μ_u, σ)`.
- Marks have shape `(n, 1)`.

**Simulate data:**

- a random-walk trajectory, 2,000 bins
- Poisson spikes from `r_u(x_t) dt`, with `dt = 0.002`
- marks drawn from each spiking unit's Gaussian
- the predictive distribution from an exact grid filter:
  - Gaussian random-walk transition
  - clusterless Poisson likelihood `exp(-Λ(x) dt) Π_j λ(x, y_j) dt`

**Misspecified variant:** decode with every `μ_u` shifted by +0.8, while the data come from the true model.
