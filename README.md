# statespacecheck

[![PyPI version](https://img.shields.io/pypi/v/statespacecheck.svg)](https://pypi.org/project/statespacecheck/)
[![Python versions](https://img.shields.io/pypi/pyversions/statespacecheck.svg)](https://pypi.org/project/statespacecheck/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/edeno/statespacecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/edeno/statespacecheck/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/edeno/statespacecheck/branch/main/graph/badge.svg)](https://codecov.io/gh/edeno/statespacecheck)

**Goodness-of-fit diagnostics for state space models in neuroscience**

`statespacecheck` provides tools to assess how well Bayesian state space models fit neural data by examining the consistency between posterior distributions and their component likelihood distributions. These diagnostics help identify issues with prior specification and model assumptions, enabling iterative model refinement.

## Overview

State space models are powerful tools for relating neural activity to latent dynamic brain states (e.g., memory, attention, spatial navigation). The core assumption is that complex, high-dimensional neural activity can be related to low-dimensional latent states through:

1. **State transition model**: How latent states evolve over time
2. **Observation model**: How neural activity relates to the current latent state

The posterior distribution combines information from both models, weighing current data (normalized likelihood) against accumulated history (prediction distribution). When these distributions agree, the model's prior expectations and data-driven evidence are consistent. When they diverge, the mismatch reveals where and when the model fails to capture the structure of the data.

## Features

- **KL Divergence**: Measure information divergence between posterior and likelihood distributions at each time point
- **HPD Overlap**: Compute spatial overlap between highest posterior density regions
- **Per-spike diagnostics**: For spike trains and other marked point-process data, compute HPD overlap, KL divergence, and an exact predictive p-value for every event
- **Vectorized Operations**: Efficient NumPy-based implementation with no Python loops
- **Flexible Dimensionality**: Supports both 1D `(n_time, n_position_bins)` and 2D `(n_time, n_x_bins, n_y_bins)` spatial arrays
- **Robust Edge Case Handling**: Proper treatment of NaN values, zero sums, and empty distributions

## Terminology

This package uses specific terminology to match standard state space model conventions:

### State Distributions (`state_dist` parameter)

- **One-step-ahead predictive distribution**: p(x_t | y_{1:t-1}) - The distribution over current state given all past observations
- **Smoothed distribution**: p(x_t | y_{1:T}) - The distribution over state at time t given all observations (past and future)
- **Filtered distribution**: p(x_t | y_{1:t}) - The posterior distribution at time t (filtered estimate)

For goodness-of-fit diagnostics, you typically use the **one-step predictive** or **smoothed** distribution as `state_dist`. These represent your model's predictions before (predictive) or after (smoother) incorporating all available data.

### Likelihood (`likelihood` parameter)

- **Normalized likelihood**: p(y_t | x_t) / Σ_x p(y_t | x_t) - The likelihood normalized across spatial positions
- This is mathematically equivalent to the posterior p(x_t | y_t) with a uniform prior
- Represents what your data alone says about the state, without temporal smoothing

### Important Note: Discrete Distributions

All functions expect **discrete probability distributions** represented as histograms over spatial bins. For continuous distributions (e.g., Gaussian), discretize them first:

- Distributions are **automatically normalized** over valid (non-NaN) bins
- Each bin represents the probability mass in that spatial region
- **NaN values** can be used to mark invalid/inaccessible spatial bins (e.g., walls in a maze)
- Finer binning provides better approximation but increases computation

### Interpretation

- **Consistency**: When state distribution and likelihood agree (low KL divergence, high overlap), your model's predictions align with the data
- **Inconsistency**: When they diverge, it indicates:
  - Prior/transition model may be too rigid or misspecified
  - Observation model may not capture the true relationship between states and observations
  - Model capacity may be insufficient

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

### Basic Example

```python
import numpy as np
from statespacecheck import (
    kl_divergence,
    hpd_overlap,
    highest_density_region,
)

# Example: 1D spatial arrays (time x position)
n_time, n_bins = 100, 50
state_dist = np.random.dirichlet(np.ones(n_bins), size=n_time)  # predictive or smoother
likelihood = np.random.dirichlet(np.ones(n_bins), size=n_time)

# Compute KL divergence at each time point
kl_div = kl_divergence(state_dist, likelihood)
# Returns: (n_time,) array of divergence values

# Compute HPD region overlap
overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)
# Returns: (n_time,) array of overlap proportions (0 = no overlap, 1 = complete)

# Get highest density region mask
hd_mask = highest_density_region(state_dist, coverage=0.95)
# Returns: (n_time, n_bins) boolean mask
```

### Neuroscience Example

```python
import numpy as np
from scipy.stats import norm
from statespacecheck import kl_divergence, hpd_overlap

# Assume you have state space model output for spatial navigation task
# with position bins representing locations in a linear track

# Position bins (e.g., 50 cm track discretized into 100 bins)
position_bins = np.linspace(0, 50, 100)  # cm
n_time = 1000  # Number of time steps

# Example: One-step-ahead predictive distribution from Kalman filter
# predicted_position: (n_time,) array of predicted positions in cm
# predicted_std: (n_time,) array of prediction uncertainty
predicted_position = 25 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_time))
predicted_std = np.ones(n_time) * 2.0

# Convert to spatial probability distribution over position bins
# Note: Distributions are automatically normalized, no need to normalize manually
state_dist = np.array(
    [
        norm.pdf(position_bins, loc=pred_pos, scale=pred_std)
        for pred_pos, pred_std in zip(predicted_position, predicted_std)
    ]
)

# Example: Likelihood from place cell firing (observation model)
# spike_counts: (n_cells, n_time) array of spike counts
# place_fields: (n_cells, n_bins) array of firing rate maps
# For this example, we'll simulate the likelihood
# Note: Automatically normalized, no manual normalization needed
likelihood = np.array(
    [
        norm.pdf(position_bins, loc=pred_pos + np.random.randn(), scale=3.0)
        for pred_pos in predicted_position
    ]
)

# Assess goodness-of-fit
divergence = kl_divergence(state_dist, likelihood)
overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

# Interpret results
print(f"Mean KL divergence: {np.mean(divergence):.3f}")
print(f"Mean HPD overlap: {np.mean(overlap):.3f}")

# Identify time points with poor fit
high_divergence = divergence > 1.0
low_overlap = overlap < 0.3
print(f"Time points with high divergence: {np.sum(high_divergence)}/{n_time}")
print(f"Time points with low overlap: {np.sum(low_overlap)}/{n_time}")
```

### Per-Spike Diagnostics

For spike-sorted data decoded with a point-process observation model, each
spike's likelihood is its unit's place field (intensity) normalized over
position. `event_diagnostics` compares that single-spike likelihood with the
one-step predictive distribution of the spike's time bin, and evaluates the
predictive check exactly over the finite set of units:

```python
import numpy as np
from statespacecheck import baseline_threshold, event_diagnostics

# predictive:   (n_time, n_bins) one-step predictive distribution from your decoder
# place_fields: (n_bins, n_units) expected spike count of each unit in each position bin
# spike_time_ind, spike_unit: (n_spikes,) time bin and unit of each spike
result = event_diagnostics(predictive, place_fields, spike_time_ind, spike_unit)
result.hpd_overlap  # (n_spikes,) low values indicate poor local fit
result.kl_divergence  # (n_spikes,) high values indicate poor local fit
result.predictive_pvalue  # (n_spikes,) low values indicate poor local fit

# Flag spikes against thresholds from a period where the model is trusted
baseline = spike_time_ind < n_baseline_bins
hpd_threshold = baseline_threshold(result.hpd_overlap[baseline], 0.01)
flagged = (result.hpd_overlap <= hpd_threshold) | (result.predictive_pvalue <= 0.05)
```

The building blocks are also available individually: `event_likelihood`,
`predictive_mark_probabilities`, and `mark_predictive_pvalue`.

## API Reference

### `kl_divergence(state_dist, likelihood)`

Compute Kullback-Leibler divergence between state distribution and likelihood.

**Parameters:**

- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Non-negative values, automatically normalized. NaN marks invalid bins. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Likelihood distributions. Non-negative values, automatically normalized. NaN marks invalid bins. Must have same shape as state_dist

**Returns:**

- `kl_divergence` (np.ndarray): KL divergence at each time point. Shape `(n_time,)`

**Interpretation:**

- **Low divergence (< 0.1)**: State distribution and likelihood agree well, indicating consistency between prior and data
- **Moderate divergence (0.1 - 1.0)**: Some disagreement, worth investigating
- **High divergence (> 1.0)**: Substantial mismatch, suggests issues with prior specification or observation model

### `hpd_overlap(state_dist, likelihood, coverage=0.95)`

Compute overlap between highest posterior density regions.

**Parameters:**

- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Non-negative values, automatically normalized. NaN marks invalid bins. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Likelihood distributions. Non-negative values, automatically normalized. NaN marks invalid bins. Must have same shape as state_dist
- `coverage` (float): Coverage probability for HPD regions (default: 0.95)

**Returns:**

- `overlap` (np.ndarray): Overlap proportion at each time point. Shape `(n_time,)`. Values range from 0 (no overlap) to 1 (complete overlap)

**Interpretation:**

- **High overlap (> 0.7)**: State distribution and likelihood concentrate probability mass in similar regions
- **Moderate overlap (0.3 - 0.7)**: Partial agreement, may indicate transition periods or model uncertainty
- **Low overlap (< 0.3)**: Distributions are spatially inconsistent, suggests model issues

### `highest_density_region(distribution, coverage=0.95)`

Compute boolean mask indicating highest density region membership.

**Parameters:**

- `distribution` (np.ndarray): Probability distributions. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `coverage` (float): Desired coverage probability (default: 0.95)

**Returns:**

- `isin_hd` (np.ndarray): Boolean mask. Same shape as input

**Notes:**

- Highest density regions can be multimodal (non-contiguous)
- Regions are defined by selecting positions with highest density until cumulative mass reaches coverage
- NaN values are treated as zero mass

## Development

### Setup

```bash
# Create the environment: the package in editable mode plus the dev tools,
# at the versions pinned in uv.lock
uv sync
```

`uv run <command>` runs a command in that environment; there is no nox or tox
file. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

### Running Tests

```bash
# Run the tests and the docstring examples, with coverage
uv run pytest

# Run a specific test file
uv run pytest tests/test_state_consistency.py -v
```

### Code Quality

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .

# Type checking (strict)
uv run mypy
```

### Standards

- **Python**: 3.10+ (following [SPEC 0](https://scientific-python.org/specs/spec-0000/))
- **Dependencies**: numpy>=1.26.0, scipy>=1.11.1, matplotlib>=3.8.0
- **Docstrings**: NumPy format with parameter types and return values
- **Type hints**: Full mypy strict mode compliance
- **Style**: ruff for formatting and linting (95 char line length)
- **No `# type: ignore`**: Fix type issues by refactoring, not suppressing

## Scientific Context

This package implements goodness-of-fit diagnostics for state space models used in neuroscience. The methods are based on the principle that a well-specified model should have consistent posterior and likelihood distributions. Large divergences or low overlap indicate:

1. **Prior issues**: State transition model too rigid or misspecified
2. **Observation model issues**: Tuning curves or noise assumptions incorrect
3. **Model capacity**: Latent state dimensionality insufficient

These diagnostics complement but are distinct from:

- **Cross-validation**: Measures predictive generalization to new data
- **Permutation tests**: Assess whether model captures structure vs. random patterns

## Citation

<!-- --8<-- [start:citation] -->
If you use this package in your research, please cite it. `CITATION.cff` holds
the metadata (GitHub's *Cite this repository* button reads it):

```bibtex
@software{statespacecheck,
  title={statespacecheck: Goodness-of-fit diagnostics for state space models},
  author={Denovellis, Eric and Zeng, Sirui and Eden, Uri T.},
  year={2026},
  version={0.2.0},
  url={https://github.com/edeno/statespacecheck}
}
```

A DOI will be added once releases are archived on Zenodo.

The methods are described in the companion paper, *Local goodness-of-fit
measures for neural decoding* (Zeng, Comrie, Frank, Eden and Denovellis), whose
analysis code is at <https://github.com/edeno/statespacecheck-paper>.
<!-- --8<-- [end:citation] -->

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code meets quality standards
5. Submit a pull request

## References

- Auger-Méthé, M., et al. (2021). A guide to state-space modeling of ecological time series. *Ecological Monographs*, 91(4), e01470.
- Newman, K. B., & Thomas, L. (2014). Goodness of fit for state-space models. In *Statistical Inference from Stochastic Processes* (pp. 153-191).
- Gelman, A., et al. (2020). *Bayesian Data Analysis* (3rd ed.). CRC Press.
