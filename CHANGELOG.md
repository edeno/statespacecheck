# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `event_weighted_predictive()`: the state distribution of the next event, the predictive distribution weighted by the total event intensity (the paper's event-weighted predictive distribution).
- `flag_events()` and `EventFlags`: the paper's per-event flagging rule. HPD overlap at or below its threshold, KL divergence at or above its threshold, and the predictive p-value at or below its cutoff (default 0.05), each flagged separately.
- A tutorial of the paper's per-spike workflow (`examples/05_per_event_diagnostics`), and documentation pages on interpreting the diagnostics and on getting the inputs from a decoder (a grid filter, `non_local_detector`, a Kalman filter). The API reference opens with an overview grouped by task that marks the extensions beyond the paper.

### Changed

- **Breaking:** `combine_flags()` and `plot_diagnostics()` require boolean flags; other values were cast to bool, so metric values passed by mistake flagged every nonzero or NaN point. `combine_flags()` requires `1 <= min_votes <= len(flags)` (0 flagged everything, too many flagged nothing), and every flag function requires `min_len >= 1` (smaller values were treated as 1).
- The README and documentation lead with the paper's per-spike workflow: `pip install statespacecheck`, a runnable quick start that flags a simulated misfit, what a decoder must provide, and how to read the results. Fixed cutoff tables (KL < 0.1, HPD overlap > 0.7) that contradicted the paper are removed, the terminology follows the paper (one-step predictive distribution, highest probability-density region), and the extensions beyond the paper are labeled as such. The README and docs examples run in the test suite.
- **Breaking:** `flag_extreme_pvalues()` is one-sided and flags `p <= alpha`, as in the paper. It used to flag `p < alpha/2 or p > 1 - alpha/2`, which marked the best-fitting observations (p near 1) as misfit. `alpha` and `min_len` are keyword-only.
- **Breaking:** `plot_diagnostics()` draws a single p-value line at the cutoff, and its threshold arguments are renamed and keyword-only: `tau` → `overlap_threshold`, `z_thresh` → `kl_z_threshold`, `alpha` → `pvalue_threshold`. It checks that the metrics match `time` in length, and shades single flagged samples visibly. Infinite KL divergence (disjoint supports), which a line cannot show, is marked with triangles.
- **Breaking:** `flag_low_overlap()` and `find_low_overlap_intervals()` flag overlap *at or below* the threshold (was strictly below), as in the paper; with a threshold of 0 from `baseline_threshold()`, zero overlap is now flagged. `threshold` and `min_len` are keyword-only.
- **Breaking:** `flag_extreme_kl()` always flags infinite KL divergence (disjoint supports), which it used to ignore. `z_thresh` and `min_len` are keyword-only. When more than 3/4 of the finite values are tied it warns that the robust z-score has fallen back to a scale of 1 (the rule becomes KL above the median by more than `z_thresh`), which it used to do silently.
- `baseline_threshold()` accepts `+inf` values (KL divergence of disjoint supports); when the quantile falls among them, the threshold is `+inf`. Finite results are unchanged. It raises on `-inf`.
- `kl_divergence()` and `hpd_overlap()` exclude a bin from both distributions when it is NaN (or infinite) in either. A NaN only in the likelihood used to give infinite KL divergence, contrary to the docstring. Results where NaN bins match in both inputs are unchanged.
- `event_diagnostics()` checks its inputs before computing and reports problems with the argument names and **absolute** event, time-bin and mark indices (they were relative to the current batch). A `mark_intensities` table stored one unit per row gets a hint to transpose it, float time indices get a hint to bin the event times, and empty event lists are accepted. Results are unchanged.
- **Breaking:** `aggregate_over_period()` requires a boolean `time_mask`; an index array used to be cast to bool and silently select every time point. The flag functions report a non-1-D input by its argument name.
- **Breaking:** `predictive_density()` and `log_predictive_density()` call the unnormalized p(y|x) `observation_likelihood` (was `likelihood`, which elsewhere in the package means a likelihood normalized over states); `log_predictive_density()`'s `log_likelihood` is renamed `log_observation_likelihood` and is keyword-only.
- `kl_divergence()`, `hpd_overlap()`, `highest_density_region()`, `predictive_density()` and `log_predictive_density()` process time in chunks, cutting peak memory about 8–12× (for 100,000 time bins × 500 positions, from 1.3–4.5 GB to 0.16–0.37 GB) at the same speed. Results are unchanged.
- Type hints: inputs accept any array-like (lists, integer arrays), outputs are typed `NDArray[np.float64]`, and `aggregate_over_period()`'s `reduction` is `Literal["mean", "sum"]`, so strict type checkers accept the documented usage and catch misspelled options. The per-event functions convert inputs to float64, as the time-bin functions already did.
- `import statespacecheck` no longer imports `matplotlib.pyplot`; `plot_diagnostics()` imports it when called.
- Development tooling follows the [Scientific Python development guide](https://learn.scientific-python.org/development/): strict pytest (warnings are errors, docstring examples run as doctests), strict mypy, a ruff rule set based on the guide's at line length 95, a `dev` dependency group with a committed `uv.lock`, and codespell.
- The package ships `py.typed`, so type checkers use its annotations (`__version__` is in `__all__` so strict checkers accept it), and a `CITATION.cff`.
- The license is declared as the SPDX expression `MIT` (PEP 639); Python 3.14 is supported and tested.
- The minimum SciPy is 1.11.1: 1.11.0 was yanked from PyPI. CI now tests at the declared minimum versions (NumPy 1.26.0, SciPy 1.11.1, matplotlib 3.8.0).
- The tutorial notebooks are executed in CI, which also checks that each tutorial's `.py` and `.ipynb` agree.
- `predictive_pvalue()` no longer checks that `sample_log_pred` is callable before calling it; a non-callable still raises `TypeError`, now with Python's own message.
- `predictive_pvalue()` raises `ValueError` when the sampler returns NaN, which previously pulled p-values toward 0 and read as misfit. An observed log predictive density of `-inf` now gives a p-value of 0 and `+inf` gives 1, instead of NaN; only a NaN observation gives NaN.

### Fixed

- `event_likelihood()`, `predictive_mark_probabilities()` and `mark_predictive_pvalue()` return empty results for zero events, and `event_diagnostics()` for zero time bins, instead of raising `IndexError` or a reshape error.
- `predictive_density()` and `log_predictive_density()` exclude a bin whose observation likelihood is NaN from the state as well, as `kl_divergence()` and `hpd_overlap()` do; they treated it as zero likelihood while the state kept its mass there, lowering the predictive density. A `+inf` observation likelihood raises `ValueError` (it was read as zero in linear space; the log path already raised).
- `flag_events()`, `flag_low_overlap()`, `find_low_overlap_intervals()`, `flag_extreme_kl()` and `flag_extreme_pvalues()` raise `ValueError` for a NaN threshold, which silently flagged nothing. `np.quantile` of KL divergences that include `+inf` is NaN; `baseline_threshold()` handles them.
- `mark_predictive_pvalue()` and `event_diagnostics()` scale the tie tolerance by each event's own largest predictive mark probability. It used the largest in the call, so a near-tied event's p-value could depend on `batch_size` or on which other events were passed with it.
- `kl_divergence()`, `predictive_density()` and `log_predictive_density()` no longer warn "overflow encountered in divide" with NumPy 1.26 when a row's total mass is subnormal; results are unchanged. A row of finite values whose total overflows (for example, values near 1e308) is rescaled instead of giving infinite KL divergence, zero HPD overlap with itself, or an empty highest-density region.
- The time-bin functions raise `ValueError` for distributions with no spatial bins (for example, shape `(n_time, 0)`); `kl_divergence()` returned infinity and `hpd_overlap()` raised a NumPy `argmax` error.
- The predictive-checks tutorial described the p-value with `>=` and treated p near 1 as misfit; it now uses `<=`, one-sided flags, and the paper's terminology. Links between tutorials work on the documentation site.
- The predictive-checks tutorial showed p-values computed before the 0.1.1 fix to `predictive_pvalue()`, so misfit periods appeared as p ≈ 1 instead of p ≈ 0; its outputs are regenerated.
- `predictive_mark_probabilities()`, `mark_predictive_pvalue()` and `event_diagnostics()` no longer emit a spurious "divide by zero encountered in matmul" `RuntimeWarning` on macOS with NumPy < 2.3; results are unchanged.
- Docstring examples in `periods.py`, `predictive_pvalue()` and `plot_diagnostics()` now run and show their actual output; they run as doctests.

## [0.2.0] - 2026-09-25

### Added

- **Per-event diagnostics for marked point-process data** (`statespacecheck.events`):
  - `event_likelihood()`: normalized single-event likelihood from mark intensities
  - `predictive_mark_probabilities()`: event-weighted predictive distribution over marks
  - `mark_predictive_pvalue()`: exact predictive p-value over a finite set of marks
  - `event_diagnostics()`: HPD overlap, KL divergence, and predictive p-value for every event, batched for long recordings
  - `baseline_threshold()`: flagging threshold from a quantile of baseline per-event values
  - `EventDiagnostics`: named tuple returned by `event_diagnostics()`

## [0.1.1] - 2025-11-19

### Fixed

- **`predictive_pvalue()`**: Corrected p-value computation to use `<=` comparison instead of `>=`, ensuring accurate calculation as the proportion of simulated values less than or equal to observed values. This fixes the interpretation for extreme value testing.

## [0.1.0] - 2025-11-11

### Added

#### Core Diagnostics
- `kl_divergence()`: Measure information divergence between state distributions and likelihood
- `hpd_overlap()`: Compute spatial overlap between highest posterior density regions
- `highest_density_region()`: Compute boolean masks for highest density regions

#### Predictive Checks
- `predictive_density()`: Compute predictive density from state distribution and likelihood
- `log_predictive_density()`: Numerically stable log-space computation of predictive density
- `predictive_pvalue()`: Monte Carlo p-values for goodness-of-fit testing

#### Period Detection
- `flag_low_overlap()`: Identify time periods with poor posterior-likelihood agreement
- `flag_extreme_kl()`: Detect anomalous KL divergence using robust z-scores
- `flag_extreme_pvalues()`: Flag extreme predictive p-values
- `combine_flags()`: Aggregate multiple diagnostic flags with majority voting
- `find_low_overlap_intervals()`: Extract contiguous intervals of poor model fit
- `aggregate_over_period()`: Summarize metrics over time periods with flexible aggregation

#### Visualization
- `plot_diagnostics()`: Three-panel diagnostic visualization with time-resolved metrics
- Support for flagging problematic time periods in plots

#### Documentation
- Comprehensive README with neuroscience examples
- Four tutorial notebooks covering core concepts and workflows
- API reference documentation with mkdocs-material
- CONTRIBUTING.md with development guidelines

#### Infrastructure
- Full test suite with 230 tests and 100% code coverage
- Property-based testing with Hypothesis
- Type hints with strict mypy compliance
- Code quality enforcement with ruff (formatting and linting)
- Pre-commit hooks for automated checks
- CI/CD with GitHub Actions
- Automated documentation deployment

### Features

- **Flexible Dimensionality**: Supports 1D `(n_time, n_position)` and 2D `(n_time, n_x, n_y)` spatial arrays
- **Robust Edge Cases**: Proper handling of NaN values, zero sums, and empty distributions
- **Automatic Normalization**: All functions normalize inputs automatically
- **Time-Resolved Analysis**: All metrics return time series for local model evaluation
- **Vectorized Operations**: Efficient NumPy-based implementation with no Python loops
- **Scientific Python Standards**: Follows SPEC 0 for version support and best practices

[0.1.1]: https://github.com/edeno/statespacecheck/releases/tag/v0.1.1
[0.1.0]: https://github.com/edeno/statespacecheck/releases/tag/v0.1.0
