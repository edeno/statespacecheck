# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`statespacecheck` implements the local goodness-of-fit diagnostics of the paper *Local goodness-of-fit measures for neural decoding* (Zeng, Comrie, Frank, Eden and Denovellis; analysis code in the sibling repository `statespacecheck-paper`). For each observation, down to individual spikes, it compares a state space model's **one-step predictive distribution** with the observation's **likelihood** over the same state grid, to find when and where a decoder disagrees with the data.

Use the paper's terminology: one-step predictive distribution, single-event likelihood, HPD overlap (highest probability-density region), KL divergence D(predictive || likelihood), rank-based predictive p-value. The diagnostics measure *consistency* (overlapping high-probability regions), not similarity. HPD overlap and the predictive p-value are the primary diagnostics; KL divergence is a reference.

## Architecture

### Modules

- **`events.py`**: the paper's method. `event_likelihood`, `predictive_mark_probabilities`, `mark_predictive_pvalue` (exact p-value over units), `event_diagnostics` (all three diagnostics per spike, batched), `baseline_threshold`, `flag_events` (the paper's flag rule: HPD <= t, KL >= t, p <= 0.05).
- **`state_consistency.py`**: `hpd_overlap` and `kl_divergence` row by row; used per spike by `event_diagnostics`, or per time bin (an extension).
- **`highest_density.py`**: `highest_density_region`, the regions HPD overlap compares.
- **`predictive_checks.py`** (extension): whole-bin predictive densities and a Monte Carlo `predictive_pvalue` with a user sampler.
- **`periods.py`** (extension): run-based flagging of time series (`min_len`), robust-z KL flags, majority vote, `aggregate_over_period`.
- **`viz.py`** (extension): `plot_diagnostics`; imports matplotlib.pyplot only when called.
- **`_validation.py`**: input validation, the `DistributionArray` output type, and `row_chunks`, which bounds memory in the time-bin functions.

Per-event outputs must stay bit-identical across refactors: the paper's reported numbers depend on them.

### Data Structures

Functions expect probability distributions as numpy arrays with shapes:

- `(n_time, n_position_bins)` for 1D spatial distributions
- `(n_time, n_x_bins, n_y_bins)` for 2D spatial distributions

All distributions must be properly normalized. The time dimension is always first, and methods operate along time to provide time-resolved diagnostics.

## Development Commands

`uv run <command>` against the locked environment is the task runner. There is
no nox or tox file, on purpose: each check is one `uv run` command, and CI runs
the same checks. This is a deliberate departure from the Scientific Python
development guide, the same choice ripple_detection made.

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
uvx codespell             # spell check (configured in pyproject.toml)
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

- **Local diagnostics**: every metric returns one value per event (`(n_events,)`) or per time bin (`(n_time,)`), to identify when and where models fail
- **Normalized metrics**: HPD overlap is normalized by minimum region size to handle varying region sizes
- **Robust to edge cases**: Functions handle NaN values and avoid division by zero
- **Type hints**: All functions use type annotations for clarity
- **Multimodal support**: HPD methods work with multimodal distributions by selecting highest density regions rather than contiguous intervals

## Code Quality Standards

This project follows scientific Python best practices:

- **Package management**: Use `uv` for all dependency management and virtual environment operations
- **Environment**: `uv sync` creates `.venv`; run tools with `uv run` (no activation needed)
- **Formatting**: Code is formatted with `ruff format` (wraps code at 95 characters; long strings and comments are not checked)
- **Linting**: Code is linted with `ruff check` using ripple_detection's rule set (a subset of the Scientific Python guide's; it leaves out PL, TRY and others) plus pydocstyle
- **Type checking**: Code is type-checked with `mypy` in strict mode
  - **IMPORTANT**: Never use `# type: ignore` comments. If mypy complains, fix the underlying issue by refactoring code, improving type annotations, or adjusting mypy configuration
- **Docstrings**: All public functions must have numpy-style docstrings with shape specifications in the format `Shape (n_time, n_position)` on a separate line after the parameter description
- **Testing**: Tests use pytest with coverage reporting
- **Version support**: Python 3.10+, NumPy 1.26+, SciPy 1.11.1+, matplotlib 3.8+ (checked by the CI floors job). This is a wider window than SPEC 0 recommends, kept so the paper repository (Python 3.11) can use current releases.
