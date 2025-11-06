# statespacecheck

**Goodness-of-fit diagnostics for state space models in neuroscience**

`statespacecheck` provides tools to assess how well Bayesian state space models fit neural data by examining the consistency between posterior distributions and their component likelihood distributions. These diagnostics help identify issues with prior specification and model assumptions, enabling iterative model refinement.

## Overview

State space models are powerful tools for relating neural activity to latent dynamic brain states (e.g., memory, attention, spatial navigation). The core assumption is that complex, high-dimensional neural activity can be related to low-dimensional latent states through:

1. **State transition model**: How latent states evolve over time
2. **Observation model**: How neural activity relates to the current latent state

The posterior distribution combines information from both models, weighing current data (normalized likelihood) against accumulated history (prediction distribution). When these distributions agree, the model's prior expectations and data-driven evidence are consistent. When they diverge, the mismatch reveals where and when the model fails to capture the structure of the data.

## Scientific Context

This package implements goodness-of-fit diagnostics for state space models used in neuroscience. The methods are based on the principle that a well-specified model should have consistent posterior and likelihood distributions. Large divergences or low overlap indicate:

1. **Prior issues**: State transition model too rigid or misspecified
2. **Observation model issues**: Tuning curves or noise assumptions incorrect
3. **Model capacity**: Latent state dimensionality insufficient

These diagnostics complement but are distinct from:

- **Cross-validation**: Measures predictive generalization to new data
- **Permutation tests**: Assess whether model captures structure vs. random patterns

## Features

- **KL Divergence**: Measure information divergence between posterior and likelihood distributions at each time point
- **HPD Overlap**: Compute spatial overlap between Highest Posterior Density (HPD) regions
- **Vectorized Operations**: Efficient NumPy-based implementation with no Python loops
- **Flexible Dimensionality**: Supports both 1D `(n_time, n_position_bins)` and 2D `(n_time, n_x_bins, n_y_bins)` spatial arrays
- **Robust Edge Case Handling**: Proper treatment of NaN values, zero sums, and empty distributions

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install statespacecheck

# Using pip
pip install statespacecheck
```

### Verify Installation

```python
# Verify installation
import statespacecheck
print(f"✓ statespacecheck v{statespacecheck.__version__} installed successfully")

# Quick functionality test
import numpy as np
from statespacecheck import kl_divergence

# Test with identical distributions (should give near-zero divergence)
dist = np.array([[0.5, 0.5]])
divergence = kl_divergence(dist, dist)
print(f"✓ KL divergence test: {divergence[0]:.6f} (expected: ~0.0)")
```

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
state_dist = np.random.dirichlet(np.ones(n_bins), size=n_time)  # (1)!
likelihood = np.random.dirichlet(np.ones(n_bins), size=n_time)  # (2)!

# Compute KL divergence at each time point
kl_div = kl_divergence(state_dist, likelihood)  # (3)!
# Returns: (n_time,) array of divergence values

# Compute HPD region overlap
overlap = hpd_overlap(
    state_dist,
    likelihood,
    coverage=0.95  # (4)!
)
# Returns: (n_time,) array of overlap proportions (0 = no overlap, 1 = complete)

# Get highest density region mask
hd_mask = highest_density_region(state_dist, coverage=0.95)
# Returns: (n_time, n_bins) boolean mask
```

1. Predictive distribution - your model's prediction before seeing current data
2. Normalized likelihood - what the data alone says about the state
3. Computes Kullback-Leibler divergence for each time point
4. 95% credible region coverage probability

## What's Next?

**New to state space model diagnostics?**
Start with [Tutorial 1: Introduction](tutorials/01_introduction.ipynb) for a gentle introduction with visualizations and real examples.

**Want to understand HPD regions?**
Jump to [Tutorial 2: Highest Density Regions](tutorials/02_highest_density_regions.ipynb) for deep dive into spatial diagnostics.

**Ready for time-series analysis?**
Explore [Tutorial 3: Time-Resolved Diagnostics](tutorials/03_time_resolved_diagnostics.ipynb) to identify when and where your model fails.

**Need complete API details?**
Browse the [API Reference](reference/statespacecheck/highest_density.md) for detailed function documentation with parameter specifications.

## Documentation Structure

- **[Tutorials](tutorials/index.md)**: Step-by-step guides with interactive Jupyter notebooks
- **[API Reference](reference/statespacecheck/highest_density.md)**: Complete API documentation with detailed parameter descriptions
- **[Contributing](contributing.md)**: Guide for contributing to the project

## Citation

If you use this package in your research, please cite:

```bibtex
@software{statespacecheck2025,
  title={statespacecheck: Goodness-of-fit diagnostics for state space models},
  author={Denovellis, Eric and Zeng, Sirui and Eden, Uri T.},
  year={2025},
  url={https://github.com/edeno/statespacecheck}
}
```

## License

MIT License - see [LICENSE](https://github.com/edeno/statespacecheck/blob/main/LICENSE) file for details.

## References

- Auger-Méthé, M., et al. (2021). A guide to state-space modeling of ecological time series. *Ecological Monographs*, 91(4), e01470.
- Newman, K. B., & Thomas, L. (2014). Goodness of fit for state-space models. In *Statistical Inference from Stochastic Processes* (pp. 153-191).
- Gelman, A., et al. (2020). *Bayesian Data Analysis* (3rd ed.). CRC Press.
