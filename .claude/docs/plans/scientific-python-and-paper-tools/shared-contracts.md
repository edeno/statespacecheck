# Shared contracts: new public API

[← back to PLAN.md](PLAN.md)

Phases 4a and 4b implement these contracts and phase 5 consumes them. **Do not weaken them.** Changing a signature after 4a merges means updating every later phase file.

- [Model and notation](#model-and-notation)
- [`event_weighted_predictive`](#event_weighted_predictive)
- [Callable protocols: `MarkIntensity`, `MarkSampler`](#callable-protocols)
- [`MarkPredictiveCheck` and `monte_carlo_mark_pvalue`](#monte_carlo_mark_pvalue)
- [`clusterless_event_diagnostics`](#clusterless_event_diagnostics)
- [Placement and exports](#placement-and-exports)

## Model and notation

This follows `statespacecheck-paper/manuscript/main.tex:131-217`.

A marked point process has joint mark intensity `λ(x, y)` over state `x` (a grid of `n_bins` bins, possibly multi-dimensional) and mark `y`.

- **Ground intensity:** `Λ(x) = ∫ λ(x, y) dy`, or `Σ_c λ_c(x)` for discrete marks.
- **Mark distribution of an event at state `x`:** `g(y | x) = λ(x, y) / Λ(x)`.

For an event in a bin with predictive distribution `P(x)`:

| Quantity | Definition | Manuscript |
| --- | --- | --- |
| Event-weighted predictive | `P_event(x) = Λ(x) P(x) / Σ_x Λ(x) P(x)` | `main.tex:174-178` |
| Predictive mark density | `f_pred(y) = Σ_x λ(x, y) P(x) / Σ_x Λ(x) P(x) = Σ_x g(y\|x) P_event(x)` | `main.tex:195-201` |
| Single-event likelihood | `Q(x) ∝ λ(x, y_obs)`, i.e. the existing `event_likelihood` | `main.tex:147-152` |
| Predictive p-value | `p = Pr_{Ỹ ~ f_pred}[ f_pred(Ỹ) ≤ f_pred(y_obs) ]` | `main.tex:203-217` |

For discrete marks, `f_pred(c)` equals the existing `predictive_mark_probabilities`, and `p` equals `mark_predictive_pvalue`. That identity is the main parity test.

## `event_weighted_predictive`

```python
def event_weighted_predictive(
    state_dist: DistributionArray,
    ground_intensity: NDArray[np.floating],
) -> DistributionArray:
```

- `state_dist`: shape `(n_events, ...)`, nonnegative and finite. Rows need not be normalized.
- `ground_intensity`: shape `(...)`, equal to `state_dist.shape[1:]`; nonnegative and finite.
- **Returns** shape `(n_events, ...)`: rows proportional to `state_dist * ground_intensity`, each summing to 1.
- **Raises** `ValueError` for:
  - a shape mismatch, or no spatial axis
  - negative or non-finite input
  - a row whose weighted total is zero or non-finite. The message lists up to 10 row indices, like `predictive_mark_probabilities` (`events.py:236-242`).
- For sorted marks, pass `ground_intensity = mark_intensities.sum(axis=-1)`.

## Callable protocols

These are defined as public type aliases in `statespacecheck/continuous_marks.py`:

```python
from collections.abc import Callable
from typing import Any

MarkIntensity = Callable[[NDArray[Any]], NDArray[np.floating]]
"""marks (n, *mark_shape) -> lambda(x, y) at every state bin, shape (n, *spatial_shape)."""

MarkSampler = Callable[[NDArray[np.intp], np.random.Generator], NDArray[Any]]
"""(flat state-bin indices (n,), rng) -> marks (n, *mark_shape), each drawn from g(. | x)."""
```

**Semantics:**

- State-bin indices are flat indices into `np.prod(spatial_shape)`, the same C-order flattening as `state_dist.reshape(n, -1)`.
- `MarkIntensity` must return finite, nonnegative values of shape `(n, *spatial_shape)`. The package validates this and raises `ValueError` otherwise.
- `MarkSampler` must return an array whose first axis has length `n`. It must draw only from the `rng` it is given; that is what makes seeded results reproducible.
- **Caller's responsibility:** `ground_intensity` must equal `∫ λ(x, y) dy` for the same model as `MarkIntensity` and `MarkSampler`. The package cannot check this; each docstring must say so.

## `monte_carlo_mark_pvalue`

```python
class MarkPredictiveCheck(NamedTuple):
    pvalue: NDArray[np.floating]                        # (n_events,)
    observed_log_density: NDArray[np.floating]          # (n_events,)  log f_pred(y_obs)
    simulated_log_density: NDArray[np.floating] | None  # (n_events, n_samples), or None

def monte_carlo_mark_pvalue(
    state_dist: DistributionArray,          # (n_events, ...)
    mark_intensity: MarkIntensity,
    observed_marks: NDArray[Any],           # (n_events, *mark_shape)
    *,
    ground_intensity: NDArray[np.floating], # (...)
    sample_marks: MarkSampler,
    n_samples: int = 1000,
    rng: np.random.Generator | int | None = None,
    return_samples: bool = False,
    batch_size: int = DEFAULT_MONTE_CARLO_BATCH_SIZE,  # 8
) -> MarkPredictiveCheck:
```

**Positional order** follows `mark_predictive_pvalue(state_dist, mark_intensities, observed_marks)` (`events.py:247-251`).

**Returns:**

- `pvalue = mean_s 1{ log f_pred(ỹ_s) ≤ log f_pred(y_obs) + tol }`, with `tol = 16 * eps * (n_bins + M)`, where `M` sums the magnitudes of the finite log sums compared (the two numerators and twice the normalizer). A fixed `16 * eps * n_bins` split exact ties when the intensities were far from 1 (for example, all scaled by 1e-30), because rounding in the log sums grows with their magnitude. This is the log-space counterpart of `mark_predictive_pvalue`'s tie tolerance.
- `observed_log_density` is always returned.
- `simulated_log_density` is returned only if `return_samples=True`. It can be large; Figure 2 of the paper uses it for its histogram.

**Validation:**

- `n_samples >= 1` and `batch_size >= 1`, else `ValueError`.
- `observed_marks.shape[0] == n_events`.
- The output of each callable is validated as described under [Callable protocols](#callable-protocols).

**Reproducibility:**

- `rng` goes through `np.random.default_rng(rng)`.
- Results are bit-reproducible for a fixed integer seed **and** a fixed `batch_size`; the docstring says so.

## `clusterless_event_diagnostics`

```python
def clusterless_event_diagnostics(
    predictive: DistributionArray,          # (n_time, ...)
    mark_intensity: MarkIntensity,
    event_time_ind: NDArray[np.integer],    # (n_events,)
    event_marks: NDArray[Any],              # (n_events, *mark_shape)
    *,
    ground_intensity: NDArray[np.floating], # (...)
    sample_marks: MarkSampler,
    coverage: float = DEFAULT_COVERAGE,
    n_samples: int = 1000,
    rng: np.random.Generator | int | None = None,
    return_likelihood: bool = False,
    batch_size: int = DEFAULT_MONTE_CARLO_BATCH_SIZE,
) -> EventDiagnostics:
```

**Positional order** follows `event_diagnostics(predictive, mark_intensities, event_time_ind, event_marks)` (`events.py:319-328`).

**Returns** the existing `EventDiagnostics` (`events.py:43-65`), unchanged, with these fields:

- `hpd_overlap` and `kl_divergence`: between `predictive[event_time_ind]` and `event_likelihood(mark_intensity(event_marks))`.
- `predictive_pvalue`: `monte_carlo_mark_pvalue(...).pvalue`.
- `likelihood`: returned only if requested.

**Invariant (do not weaken):** for discrete marks encoded as integer arrays, it must reproduce `event_diagnostics` as follows:

- `hpd_overlap`, `kl_divergence` and `likelihood` are **bit-identical**.
- `predictive_pvalue` agrees within Monte Carlo error.

The discrete-mark encoding:

- `mark_intensity = lambda m: rates[:, m].T`
- `ground_intensity = rates.sum(-1)`
- `sample_marks` draws a categorical from `rates[x] / rates[x].sum()`

## Placement and exports

- `event_weighted_predictive` goes in `src/statespacecheck/events.py`, next to `predictive_mark_probabilities`.
- Everything else goes in a new module, `src/statespacecheck/continuous_marks.py`: `MarkIntensity`, `MarkSampler`, `MarkPredictiveCheck`, `monte_carlo_mark_pvalue`, `clusterless_event_diagnostics`, `DEFAULT_MONTE_CARLO_BATCH_SIZE`.
- Add each new public name to `src/statespacecheck/__init__.py` imports and to `__all__`, kept sorted (RUF022). `DEFAULT_MONTE_CARLO_BATCH_SIZE` stays module-level only, like `DEFAULT_EVENT_BATCH_SIZE` (`events.py:40`).
