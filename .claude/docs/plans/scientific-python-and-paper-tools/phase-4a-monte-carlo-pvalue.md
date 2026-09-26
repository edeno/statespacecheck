# Phase 4a: Event-weighted predictive distribution and Monte Carlo mark p-value

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md) · [designs](designs.md)

**Branch:** `mc-mark-pvalue`, from `main` after phase 3 merges. One PR. No release.

**Inputs to read first:**

- [src/statespacecheck/events.py](../../../../src/statespacecheck/events.py). This phase copies its conventions:
  - the module docstring and array conventions (1-25)
  - the batch-size constant with its memory comment (37-40)
  - validation helpers `_validate_state_distribution` (85-95) and `_validate_marks` (98-105)
  - `predictive_mark_probabilities` (176-244): the error-message style for undefined rows
  - `mark_predictive_pvalue` (247-316): the tie tolerance and output clipping
- [src/statespacecheck/__init__.py:1-67](../../../../src/statespacecheck/__init__.py): exports and `__all__`, sorted since phase 1.
- [tests/conftest.py](../../../../tests/conftest.py) and [tests/test_events.py](../../../../tests/test_events.py): fixture and test style.
- `statespacecheck-paper/manuscript/main.tex:174-217`: the definitions being implemented.
- `statespacecheck-paper/src/statespacecheck_paper/figure02_panels.py:170-211`: the hand-rolled Monte Carlo that phase 5 will replace. A regression test here reproduces its setup.

**Contracts referenced (do not weaken):**

- [Model and notation](shared-contracts.md#model-and-notation)
- [`event_weighted_predictive`](shared-contracts.md#event_weighted_predictive)
- [Callable protocols](shared-contracts.md#callable-protocols)
- [`monte_carlo_mark_pvalue`](shared-contracts.md#monte_carlo_mark_pvalue)
- [Placement and exports](shared-contracts.md#placement-and-exports)

**Designs referenced:**

- [designs.md#event-weighted-predictive](designs.md#event-weighted-predictive)
- [#vectorized-state-sampling](designs.md#vectorized-state-sampling)
- [#monte-carlo-p-value](designs.md#monte-carlo-p-value)
- [#memory-and-batching](designs.md#memory-and-batching)
- [#test-models](designs.md#test-models)

## Tasks

1. **`event_weighted_predictive`** in `events.py`, placed after `predictive_mark_probabilities`, as in [designs.md](designs.md#event-weighted-predictive).
   - Full NumPy docstring:
     - the formula `P_event(x) ∝ Λ(x) P(x)`
     - one sentence on why: a randomly chosen event is more likely to come from states with higher total intensity
     - shapes in the `Shape (n_events, ...)` style
     - Raises, and a doctest example
   - Add a bullet to the module docstring's list (`events.py:10-19`).
   - Do **not** refactor `predictive_mark_probabilities` to call it. Its output must stay bit-identical for the paper, and the two differ in floating-point operation order.

2. **New module `src/statespacecheck/continuous_marks.py`**, containing:
   - the module docstring: the marked point-process model in words; when to use it (continuous or intractable marks, e.g. clusterless waveform features) versus `events.py` (finite marks)
   - the `LogMarkIntensity` and `MarkSampler` aliases (the intensity is passed as its log; see [shared-contracts.md#callable-protocols](shared-contracts.md#callable-protocols))
   - `MarkPredictiveCheck`
   - `DEFAULT_MONTE_CARLO_BATCH_SIZE` (8; see [designs.md#memory-and-batching](designs.md#memory-and-batching)), with its memory comment
   - the private helpers `_safe_log`, `_evaluate_log_intensity`, `_check_leading_axis`, `_sample_state_bins` and `_monte_carlo_batch`
   - `monte_carlo_mark_pvalue`

   All code as in [designs.md](designs.md#monte-carlo-p-value). `_monte_carlo_batch` takes the flattened state batch, the flattened ground intensity, the observed marks for the batch, the two callables, `n_samples` and `rng`. It returns `(pvalue, observed_log, simulated_log)` for the batch; phase 4b reuses it.

   The `monte_carlo_mark_pvalue` docstring must include:
   - the p-value definition and the tie tolerance
   - the caller's responsibility that `ground_intensity` matches the other two callables
   - reproducibility: same integer seed and same `batch_size` give identical results
   - the memory note
   - a doctest on a 3-bin, 2-mark discrete model with `rng=0` and `n_samples=2000`, printing `pvalue.round(1)`. Print a coarse value so the doctest doesn't pin RNG bits.
   - See Also: `mark_predictive_pvalue` (the exact finite-mark version) and `predictive_pvalue` (the generic Monte Carlo helper)

3. **Exports.** Add `event_weighted_predictive`, `monte_carlo_mark_pvalue`, `MarkPredictiveCheck`, `LogMarkIntensity` and `MarkSampler` to `__init__.py` and `__all__`, keeping it sorted. `docs/gen_ref_pages.py` picks up the new module automatically; check that the API reference page renders.

4. **Tests.**
   - New `tests/test_continuous_marks.py`, plus additions to `tests/test_events.py` for `event_weighted_predictive`.
   - Put the discrete test model from [designs.md#test-models](designs.md#test-models) in `tests/conftest.py` as a fixture returning `(rates, log_mark_intensity, sample_marks)`. Phase 4b reuses it.
   - Cases are in the validation slice below.

5. **User-facing docs.**
   - README: a "Continuous marks" subsection after "Per-Spike Diagnostics" (`README.md:156`). Show a 1-D Gaussian-mark example of `monte_carlo_mark_pvalue` in ~15 lines, modelled on the paper's Figure 2 setup:
     - `log λ(x, y) = log N(y; x, σ)` (`norm.logpdf`)
     - `Λ(x) = 1`
     - `sample_marks = lambda b, rng: rng.normal(bins[b], σ)[:, None]`
   - CHANGELOG `[Unreleased]` → `### Added`: `event_weighted_predictive()`, `monte_carlo_mark_pvalue()`, `MarkPredictiveCheck`, and the `LogMarkIntensity`/`MarkSampler` types.
   - CLAUDE.md "Core Modules": add `continuous_marks.py`. Also update the stale module list, which omits `events.py`, `periods.py`, `predictive_checks.py` and `viz.py`.

## Deliberately not in this phase

- `clusterless_event_diagnostics` and the clusterless tutorial (phase 4b).
- Any change to `event_diagnostics`, `predictive_mark_probabilities` or `mark_predictive_pvalue`. Their error-message and validation hardening is [phase 4b task 1](phase-4b-clusterless-diagnostics.md#tasks).
- Changing the generic `predictive_pvalue` (`predictive_checks.py:300`) to use the new sampler. It stays a user-supplied-sampler helper.
- Releasing (phase 4b tags v0.3.0).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_event_weighted_predictive_constant_ground_is_identity` | `Λ ≡ c` returns the normalized `state_dist` (`allclose`, rtol 1e-12) |
| `test_event_weighted_predictive_matches_manual` | 2-D spatial input `(n, 4, 3)`: equals `state*Λ / sum`, and the shape is preserved |
| `test_event_weighted_predictive_rejects_bad_input` | `ValueError` for: shape mismatch, negative, non-finite, and zero-total rows (message lists the row index) |
| `test_sample_state_bins_frequencies` | 200 000 draws per row: empirical frequencies within 5 binomial SE of the target |
| `test_sample_state_bins_skips_zero_probability_bins` | Zero-probability bins are never drawn, even in the first or last position |
| `test_monte_carlo_matches_exact_for_discrete_marks` (**slow**) | Discrete model, 40 events, `n_samples=20_000`, `rng=2`: `abs(p_mc - p_exact) <= 4*sqrt(p(1-p)/n) + 1/n` for every event (prototype max z-score = 2.0) |
| `test_monte_carlo_observed_log_density_matches_exact` | For discrete marks, `observed_log_density == log(predictive_mark_probabilities[obs])` to rtol 1e-12 |
| `test_monte_carlo_calibrated_under_true_model` (**slow**) | Clusterless 1-D model, observed marks drawn from the model's own predictive: KS test of p-values against U(0,1) has p > 0.01 (fixed seed) |
| `test_monte_carlo_reproducible` | Same seed and batch size give bit-identical results; different seeds give different results |
| `test_monte_carlo_return_samples` | `simulated_log_density` shape `(n_events, n_samples)` when requested, `None` otherwise; `pvalue == mean(sim <= obs + tol)` recomputed from the returned samples |
| `test_monte_carlo_batch_size_invariance_of_distribution` | `batch_size=1` vs `32`: p-values agree within MC error (not bitwise) |
| `test_monte_carlo_impossible_observed_mark` | `λ(x, y_obs)=0` wherever `P>0`: `observed_log_density == -inf`, `pvalue == 0`, no warning raised |
| `test_monte_carlo_validates_callables` | Wrong-shaped `log_mark_intensity` output, NaN or `+inf` log intensity, and `sample_marks` returning the wrong length each raise `ValueError` |
| `test_monte_carlo_validates_arguments` | `n_samples=0`, `batch_size=0` and a mismatched `observed_marks` length raise `ValueError` |
| `test_monte_carlo_empty_events` | `n_events=0` gives empty outputs; the callables are not called (a sentinel callable raises if called) |
| `test_monte_carlo_figure2_scenario` | Replicates `figure02_panels.py:130-211`: 200 bins on [0, 100], predictive N(35, 8), mark density N(y; x, 12), `y_obs=60`, `Λ=1`. Compute the reference p by quadrature: `f_pred(y) = Σ_x P(x) N(y; x, 12)` on a fine y-grid (e.g. 20 001 points over [-100, 200]), then `p = ∫ f_pred 1{f_pred ≤ f_pred(60)} dy`. With `n_samples=20_000`, the Monte Carlo p is within 4 binomial SE of the quadrature value |
| `uv run pytest --doctest-modules src/statespacecheck/continuous_marks.py src/statespacecheck/events.py` | Docstring examples pass |
| `uv run mypy`, `ruff check`, `ruff format --check` | Clean under the strict phase-1 settings |

Register a `slow` marker in `[tool.pytest.ini_options] markers` (required by `--strict-markers`). CI runs slow tests; locally they can be skipped with `-m "not slow"`. Measure the slow tests' runtime and put it in the PR description. The target is under 20 s total.

## Fixtures

- `discrete_mark_model`, in `tests/conftest.py`: rates, `log_mark_intensity`, `sample_marks` ([designs.md#discrete-model](designs.md#discrete-model)).
- `clusterless_1d_model`, in `tests/conftest.py`: place fields, waveform means, `log_mark_intensity`, `sample_marks` and `ground_intensity` ([designs.md#clusterless-1-d-model](designs.md#clusterless-1-d-model)). Phase 4b extends it with the simulated trajectory and filter.
- Real data: none in this phase. Phase 5 uses the paper's Figure 2 setup as a real-use smoke test.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also run the `scientific-code-change-audit` skill on the diff. This code produces a statistic the paper reports.
