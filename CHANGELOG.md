# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Development tooling follows the [Scientific Python development guide](https://learn.scientific-python.org/development/): strict pytest (warnings are errors, docstring examples run as doctests), strict mypy, a ruff rule set based on the guide's at line length 95, a `dev` dependency group with a committed `uv.lock`, and codespell.
- The package ships `py.typed`, so type checkers use its annotations (`__version__` is in `__all__` so strict checkers accept it), and a `CITATION.cff`.
- The license is declared as the SPDX expression `MIT` (PEP 639); Python 3.14 is supported and tested.
- The minimum SciPy is 1.11.1: 1.11.0 was yanked from PyPI. CI now tests at the declared minimum versions (NumPy 1.26.0, SciPy 1.11.1, matplotlib 3.8.0).
- `predictive_pvalue()` no longer checks that `sample_log_pred` is callable before calling it; a non-callable still raises `TypeError`, now with Python's own message.
- `predictive_pvalue()` raises `ValueError` when the sampler returns NaN, which previously pulled p-values toward 0 and read as misfit. An observed log predictive density of `-inf` now gives a p-value of 0 and `+inf` gives 1, instead of NaN; only a NaN observation gives NaN.

### Fixed

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
