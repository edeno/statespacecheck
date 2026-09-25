# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`statespacecheck` is a Python package for goodness-of-fit diagnostics in state space models, particularly for neuroscience applications. The package provides methods to assess model-data agreement in Bayesian state space models by analyzing the relationship between posterior distributions, likelihood distributions, and prediction distributions.

### Scientific Context

State space models are used to relate neural activity to latent dynamic brain states. They consist of:

- **State transition model**: How latent states evolve over time
- **Observation model**: How observed neural activity relates to current latent state

The package addresses a critical gap: evaluating goodness-of-fit for latent variable models where ground truth cannot be observed directly. Traditional global metrics often fail to detect local misfits or diagnose whether errors arise from the state or observation model.

## Architecture

The codebase implements three complementary model checking methods designed to:

1. Identify time periods of poor model-data agreement (local evaluation)
2. Distinguish errors from state vs observation components
3. Provide intuitive visualizations

### Core Modules

**[src/statespacecheck/state_consistency.py](src/statespacecheck/state_consistency.py)**

- Primary diagnostic functions for assessing posterior-likelihood consistency
- `kl_divergence()`: Measures information divergence using KL divergence to detect issues with prior specification
- `hpd_overlap()`: Computes spatial overlap between highest posterior density (HPD) regions to assess consistency between likelihood and prior contributions

**[src/statespacecheck/highest_density.py](src/statespacecheck/highest_density.py)**

- Highest density region computation utilities
- `highest_density_region()`: Returns boolean mask indicating highest density region membership. Computes threshold values inline for specified coverage, handling multimodal distributions correctly

### Data Structures

Functions expect probability distributions as numpy arrays with shapes:

- `(n_time, n_position_bins)` for 1D spatial distributions
- `(n_time, n_x_bins, n_y_bins)` for 2D spatial distributions

All distributions must be properly normalized. The time dimension is always first, and methods operate along time to provide time-resolved diagnostics.

## Development Commands

`uv run <command>` against the locked environment is the task runner. There is
no nox or tox file, on purpose: each check is one `uv run` command, and CI runs
the same commands. This is a deliberate departure from the Scientific Python
development guide, shared with ripple_detection and spectral_connectivity.

**Environment setup:**

```bash
uv sync  # editable install plus the dev dependency group, pinned by uv.lock
uv sync --extra docs  # also the documentation tools
```

**Code quality:**

```bash
uv run ruff format .      # format
uv run ruff check .       # lint
uv run ruff check --fix . # fix auto-fixable lint issues
uv run mypy               # strict type check (files set in pyproject.toml)
uv lock --check           # uv.lock matches pyproject.toml
uvx pre-commit run --all-files
```

**Testing:**

```bash
uv run pytest             # tests and docstring examples, with coverage
uv run pytest --no-cov    # without the coverage report
uv run pytest tests/test_filename.py::test_function_name
```

Every warning is a test error (`filterwarnings = ["error"]`); a test that
expects a warning uses `pytest.warns`.

## Key Design Principles

- **Time-resolved diagnostics**: All metrics return time series (shape `(n_time,)`) to identify when/where models fail
- **Normalized metrics**: HPD overlap is normalized by minimum region size to handle varying region sizes
- **Robust to edge cases**: Functions handle NaN values and avoid division by zero
- **Type hints**: All functions use type annotations for clarity
- **Multimodal support**: HPD methods work with multimodal distributions by selecting highest density regions rather than contiguous intervals

## Code Quality Standards

This project follows scientific Python best practices:

- **Package management**: Use `uv` for all dependency management and virtual environment operations
- **Environment**: Always activate and work within `.venv` virtual environment
- **Formatting**: Code is formatted with `ruff format` (95 character line length)
- **Linting**: Code is linted with `ruff check` using the Scientific Python guide's rule set (the same as ripple_detection and spectral_connectivity) plus pydocstyle
- **Type checking**: Code is type-checked with `mypy` in strict mode
  - **IMPORTANT**: Never use `# type: ignore` comments. If mypy complains, fix the underlying issue by refactoring code, improving type annotations, or adjusting mypy configuration
- **Docstrings**: All public functions must have numpy-style docstrings with shape specifications in the format `Shape (n_time, n_position)` on a separate line after the parameter description
- **Testing**: Tests use pytest with coverage reporting
- **Version support**: Follows scientific Python SPEC 0 (supports Python 3.10+, recent numpy/scipy/matplotlib versions)
