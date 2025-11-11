# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/edeno/statespacecheck/releases/tag/v0.1.0
