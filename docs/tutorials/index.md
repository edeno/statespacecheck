# Tutorials

Welcome to the `statespacecheck` tutorials! These interactive Jupyter notebooks guide you through the core concepts and practical applications of goodness-of-fit diagnostics for state space models.

## Tutorial Overview

### [1. Introduction](01_introduction.ipynb)

Get started with the fundamentals of model checking for state space models. This tutorial covers:

- What are state space models and why do we need goodness-of-fit diagnostics?
- Understanding posterior-likelihood consistency
- Basic usage of KL divergence and HPD overlap metrics
- Interpreting diagnostic results

**Duration**: 20-25 minutes
**Prerequisites**: Basic understanding of probability distributions and state space models

---

### [2. Highest Density Regions](02_highest_density_regions.ipynb)

Deep dive into highest posterior density (HPD) regions and their role in model diagnostics. Topics include:

- Computing HPD regions for univariate and multivariate distributions
- Handling multimodal distributions
- Visualizing HPD regions in 1D and 2D
- Understanding coverage probabilities

**Duration**: 25-30 minutes
**Prerequisites**:
- Tutorial 1 (understanding of posterior-likelihood consistency)
- Familiarity with probability density functions

---

### [3. Time-Resolved Diagnostics](03_time_resolved_diagnostics.ipynb)

Learn how to identify *when* and *where* your model fails using time-resolved metrics. This tutorial demonstrates:

- Computing diagnostics across time series
- Detecting periods of model-data mismatch
- Visualizing temporal patterns in model fit
- Identifying systematic biases vs. random errors

**Duration**: 25-30 minutes
**Prerequisites**:
- Tutorial 1 (KL divergence and HPD overlap basics)
- Tutorial 2 (HPD region computation)
- Understanding of time-series data

---

### [4. Predictive Checks](04_predictive_checks.ipynb)

Advanced techniques for comprehensive model evaluation. Covers:

- Posterior predictive checks for state space models
- Comparing observed vs. predicted data patterns
- Detecting model misspecification
- Iterative model refinement workflows

**Duration**: 30-35 minutes
**Prerequisites**:
- Tutorials 1-3 (core diagnostic methods)
- Experience with Bayesian inference
- Familiarity with posterior predictive distributions

---

## Running the Tutorials

All tutorials are provided as Jupyter notebooks with pre-computed outputs for quick reference. To run them interactively:

### Which option should I choose?

- **Local Installation**: Best for exploring code, modifying examples, and integrating with your own data. Requires Python setup but gives full control.
- **Google Colab**: Quick start in the cloud with no installation needed. Requires a Google account. Good for trying out examples.
- **Binder**: Fully reproducible environment in the browser. No account needed, but slower to launch (~2-3 minutes). Best for workshops or teaching.

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/edeno/statespacecheck.git
cd statespacecheck

# Install with notebook dependencies
uv pip install -e ".[dev,docs]"

# Launch Jupyter
jupyter lab examples/
```

### Option 2: Google Colab

Open any tutorial notebook on GitHub and click the "Open in Colab" badge at the top (if available), or manually upload the notebook to [Google Colab](https://colab.research.google.com/).

### Option 3: Binder

Launch an interactive environment with all dependencies pre-installed:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edeno/statespacecheck/main?labpath=examples)

---

## Learning Path

We recommend following the tutorials in order, as each builds on concepts from the previous ones. However, if you're already familiar with state space models and want to jump to specific topics:

- **Quick start**: Tutorial 1
- **Understanding HPD metrics**: Tutorial 2
- **Time series analysis**: Tutorial 3
- **Advanced diagnostics**: Tutorial 4

## Feedback and Questions

Found an issue or have a question? Please [open an issue](https://github.com/edeno/statespacecheck/issues) on GitHub.

Happy learning!
