# statespacecheck

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
- **Vectorized Operations**: Efficient NumPy-based implementation with no Python loops
- **Flexible Dimensionality**: Supports both 1D `(n_time, n_position_bins)` and 2D `(n_time, n_x_bins, n_y_bins)` spatial arrays
- **Robust Edge Case Handling**: Proper treatment of NaN values, zero sums, and empty distributions

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

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

## API Reference

### `kl_divergence(state_dist, likelihood)`

Compute Kullback-Leibler divergence between state distribution and likelihood.

**Parameters:**
- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Normalized likelihood distributions (equivalent to posterior with uniform prior). Must have same shape as state_dist

**Returns:**
- `kl_divergence` (np.ndarray): KL divergence at each time point. Shape `(n_time,)`

**Interpretation:**
- **Low divergence (< 0.1)**: State distribution and likelihood agree well, indicating consistency between prior and data
- **Moderate divergence (0.1 - 1.0)**: Some disagreement, worth investigating
- **High divergence (> 1.0)**: Substantial mismatch, suggests issues with prior specification or observation model

### `hpd_overlap(state_dist, likelihood, coverage=0.95)`

Compute overlap between highest posterior density regions.

**Parameters:**
- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Normalized likelihood distributions (equivalent to posterior with uniform prior). Must have same shape as state_dist
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
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v

# Run specific test file
pytest tests/test_posterior_consistency.py -v

# Run with coverage report
pytest tests/ --cov=src/statespacecheck --cov-report=html
```

### Code Quality

```bash
# Check code style
ruff check .

# Format code
ruff format .

# Type checking
mypy src/
```

### Standards

- **Python**: 3.10+ (following [SPEC 0](https://scientific-python.org/specs/spec-0000/))
- **Dependencies**: numpy>=1.26.0, scipy>=1.11.0, matplotlib>=3.8.0
- **Docstrings**: NumPy format with parameter types and return values
- **Type hints**: Full mypy strict mode compliance
- **Style**: ruff for formatting and linting (100 char line length)
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

If you use this package in your research, please cite:

```bibtex
@software{statespacecheck2025,
  title={statespacecheck: Goodness-of-fit diagnostics for state space models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/statespacecheck}
}
```

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
