# Contributing to statespacecheck

Thank you for your interest in contributing to statespacecheck! This document provides guidelines for development, testing, and releasing.

## Development Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/edeno/statespacecheck.git
   cd statespacecheck
   ```

2. **Create the environment**
   ```bash
   # Using uv (recommended): installs the package in editable mode plus the
   # `dev` dependency group, at the versions pinned in uv.lock
   uv sync

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   uv run python -c "import statespacecheck; print(statespacecheck.__version__)"
   ```

`uv run <command>` runs a command in the locked environment; it is the task
runner. There is no nox or tox file, on purpose: every check below is a single
`uv run` command, and CI runs the same checks.

## Development Workflow

### Code Quality Standards

This project follows strict code quality standards:

- **Formatting**: [ruff format](https://docs.astral.sh/ruff/formatter/) (wraps code at 95 characters)
- **Linting**: [ruff check](https://docs.astral.sh/ruff/) (comprehensive rules including NumPy-specific)
- **Type checking**: [mypy](https://mypy-lang.org/) in strict mode (no `# type: ignore` allowed)
- **Testing**: [pytest](https://pytest.org/) with coverage reporting; new code should be fully covered
- **Docstrings**: [NumPy style](https://numpydoc.readthedocs.io/)

### Running Checks Locally

Before committing, run all quality checks:

```bash
# Format code
uv run ruff format .

# Check formatting (CI runs this)
uv run ruff format --check .

# Lint code
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .

# Type check (strict; the files to check are set in pyproject.toml)
uv run mypy

# Run the tests and the docstring examples, with coverage
uv run pytest

# Run tests without coverage report
uv run pytest --no-cov

# Run specific test file
uv run pytest tests/test_highest_density.py -v

# Run specific test
uv run pytest tests/test_highest_density.py::TestHighestDensityRegion::test_exact_hd_region_1d -xvs

# Check that uv.lock matches pyproject.toml
uv lock --check

# Spell check (configured in pyproject.toml)
uvx codespell
```

Warnings are errors in the test suite (`filterwarnings = ["error"]`): a test
that expects a warning says so with `pytest.warns`.

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before every commit.

**Setup (one-time):**
```bash
# Install the git hooks (uvx runs pre-commit without installing it;
# with pip, `pip install pre-commit` and drop the `uvx`)
uvx pre-commit install
```

**Usage:**
```bash
# Hooks run automatically on `git commit`

# Run manually on all files
uvx pre-commit run --all-files

# Run manually on staged files only
uvx pre-commit run

# Update hook versions
uvx pre-commit autoupdate
```

**What it checks:**
- Code formatting with ruff
- Linting with ruff (auto-fixes when possible)
- Spelling with codespell
- Type checking with mypy, in the uv environment
- File hygiene: large files, merge conflicts, TOML/YAML syntax, trailing
  whitespace, line endings

The hooks do not run the tests; run `uv run pytest` yourself, and CI runs them
on every pull request. [pre-commit.ci](https://pre-commit.ci) runs the hooks on
pull requests and updates their versions monthly.

## Continuous Integration

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`, on
`v*` tags, and on demand (Actions → CI → Run workflow). Its jobs:

1. **Code Quality** (`quality`): `ruff format --check`, `ruff check` and
   `mypy` from the locked environment (`uv sync --frozen`), and
   [zizmor](https://docs.zizmor.sh) on the workflows.
2. **Lockfile** (`lockfile`): `uv lock --check`.
3. **Tests** (`test`): Python 3.10–3.14 on Linux, macOS and Windows, with
   warnings as errors; coverage goes to Codecov from Python 3.12 on Linux.
4. **Dependency floors** (`test-minimum-pins`): the tests on Python 3.10 with
   the lowest NumPy, SciPy and matplotlib that `pyproject.toml` allows.
5. **Build** (`build`) and **install tests** (`test-package`): builds the wheel
   and sdist, then installs each and runs a smoke test.
6. **Publish** and **GitHub release**: on `v*` tags only; see below.

`.github/workflows/docs.yml` runs on every pull request and push to `main`: it
executes the tutorial notebooks, checks that their committed text outputs match
the fresh run and that the run does not warn, checks each tutorial's jupytext
pair, builds the site with `mkdocs build --strict`, and deploys it to GitHub
Pages from `main`.

Every action is pinned to a commit SHA; Dependabot proposes updates monthly.

## Release Process

Releases publish from CI when a `vX.Y.Z` tag is pushed. The one-time setup
(PyPI trusted publishing, the protected `pypi` environment) and the step-by-step
release are in [.github/RELEASE_SETUP.md](https://github.com/edeno/statespacecheck/blob/main/.github/RELEASE_SETUP.md).

The version comes from the git tag through `hatch-vcs`: a development install
reports something like `0.2.1.dev3+g1a2b3c4`, and a tagged commit reports
`X.Y.Z`. **Do not** edit version numbers in the code.

## Testing

### Test Structure

```
conftest.py                          # Root: matplotlib backend, closes figures after each test
tests/
├── conftest.py                      # Shared fixtures
├── helpers.py                       # Shared test data generators
├── test_events.py                   # Per-spike diagnostics and flagging (the paper's method)
├── test_state_consistency.py        # KL divergence, HPD overlap
├── test_highest_density.py          # HPD regions
├── test_predictive_density.py       # Predictive densities
├── test_predictive_pvalue.py        # Monte Carlo predictive p-values
├── test_predictive_consistency.py   # Predictive checks on simulated models
├── test_periods.py                  # Time-series flagging and aggregation
├── test_viz.py                      # plot_diagnostics
├── test_chunking.py                 # Results do not depend on the chunk size
├── test_docs_examples.py            # README and docs code blocks run as shown
├── test_validation.py               # Input validation
├── test_edge_cases.py               # Edge cases
├── test_kl_subnormal.py             # KL divergence with subnormal numbers
├── test_properties.py               # Property-based tests (Hypothesis)
└── test_version.py                  # __version__
```

### Running Tests

```bash
# All tests and docstring examples, with coverage
uv run pytest

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Show print statements
uv run pytest -s

# Run specific test class
uv run pytest tests/test_highest_density.py::TestHighestDensityRegion -v

# Run specific test method
uv run pytest tests/test_highest_density.py::TestHighestDensityRegion::test_exact_hd_region_1d -xvs

# Run tests matching pattern
uv run pytest -k "test_hpd" -v

# HTML coverage report
uv run pytest --cov-report=html
# Then open htmlcov/index.html
```

### Writing Tests

Follow these guidelines:

1. **Use descriptive test names**: `test_kl_divergence_with_identical_distributions`
2. **Use pytest fixtures**: Defined in `conftest.py`
3. **Test edge cases**: NaN, inf, zeros, empty arrays
4. **Use property-based testing**: With Hypothesis for robustness
5. **Aim for 100% coverage**: Every line should be tested
6. **Document test intent**: Add docstrings to complex tests

Example test:

```python
def test_highest_density_region_with_peaked_distribution() -> None:
    """Test HPD region correctly identifies peak in simple 1D distribution."""
    # Arrange
    distribution = np.array([[0.1, 0.6, 0.3]])

    # Act
    region = highest_density_region(distribution, coverage=0.95)

    # Assert
    expected = np.array([[True, True, True]])
    np.testing.assert_array_equal(region, expected)
    assert region.shape == distribution.shape
```

## Documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) from
`docs/`, the docstrings, the tutorial notebooks, and sections of `README.md`
and `CONTRIBUTING.md` (included with `--8<--`):

```bash
uv run --extra docs mkdocs serve          # live preview at http://127.0.0.1:8000
uv run --extra docs mkdocs build --strict # what the docs workflow runs
```

Tutorials are edited in `examples/`: each is a jupytext pair, a `.py` script
and a `.ipynb` notebook with its outputs. The site shows the notebook's
committed outputs, so after editing a tutorial, sync the pair and re-run the
notebook:

```bash
uv run --extra docs jupytext --sync examples/NN_name.py
uv run --extra docs jupyter nbconvert --to notebook --execute --inplace examples/NN_name.ipynb
```

`docs/tutorials/` holds the tutorials' index page and symlinks to the
notebooks; a new tutorial needs a symlink there, a line in that index, and a
`nav` entry in `mkdocs.yml`. The docs workflow (`.github/workflows/docs.yml`)
executes every notebook on every pull request, and fails if a committed text
output differs from the fresh run, if the run warns, or if a pair's cells differ.
Figures are not compared.

## Code Style Guidelines

### General Principles

1. **Readability**: Code is read more often than written
2. **Simplicity**: Prefer simple solutions over clever ones
3. **Explicitness**: Explicit is better than implicit
4. **Documentation**: All public APIs must be documented
5. **Type safety**: Use type hints everywhere

### Python Style

- **Line length**: 95 characters (`ruff format` wraps code; long strings and comments are not checked)
- **Imports**: Sorted and grouped (ruff handles this)
- **Quotes**: Double quotes for strings (ruff enforces this)
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`

### NumPy Style

- **Array operations**: Prefer vectorized operations over loops
- **Broadcasting**: Use NumPy broadcasting for clarity
- **Type hints**: Use `np.ndarray` or `NDArray[np.floating]`
- **Docstrings**: Include shape information in parameter descriptions

Example:

```python
def compute_something(
    data: DistributionArray,
    threshold: float = 0.5,
) -> DistributionArray:
    """Compute something useful from data.

    Parameters
    ----------
    data : np.ndarray, shape (n_time, n_spatial)
        Input data array.
    threshold : float, optional
        Threshold value, by default 0.5.

    Returns
    -------
    result : np.ndarray, shape (n_time,)
        Computed result.
    """
    # Implementation
    pass
```

### Type Hints

- **Use everywhere**: All function signatures must have type hints
- **Import from typing**: Use `from collections.abc import Callable`
- **Union types**: Use `X | None` (Python 3.10+ syntax)
- **Generic types**: Use `DistributionArray` type alias
- **No `# type: ignore`**: Fix the issue instead

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'statespacecheck'`

**Solution**: Install in editable mode:
```bash
pip install -e .
```

### Version Shows Development String

**Problem**: `__version__` is `X.Y.Z.devN+g...` instead of `X.Y.Z`

**Solution**: This is expected in development: the version comes from the most
recent git tag plus the commits since. Only a tagged commit reports `X.Y.Z`.

### Tests Failing Locally But Pass in CI

**Problem**: Tests pass on your machine but fail in CI

**Possible causes**:
1. **Missing file**: Not committed to git
2. **Platform differences**: Windows vs Linux
3. **Python version**: Test with multiple versions
4. **Dependencies**: Check `pyproject.toml` is up to date

**Debug**:
```bash
# Run with same Python version as CI
python3.12 -m pytest

# Check which files are committed
git status

# Check differences from main
git diff main
```

### Mypy Errors

**Problem**: `mypy` complains about types

**Solution**: Never use `# type: ignore`! Instead:
1. **Add proper type hints** to function signatures
2. **Use type aliases** like `DistributionArray`
3. **Import types correctly**: `from collections.abc import Callable`
4. **Use explicit types**: `result: DistributionArray = np.array(...)`

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/edeno/statespacecheck/issues) (bug
  reports and feature requests have templates)
- **Security problems**: report privately; see
  [SECURITY.md](https://github.com/edeno/statespacecheck/blob/main/SECURITY.md)
- **Email**: eric.denovellis@ucsf.edu

This project follows the
[Contributor Covenant Code of Conduct](https://github.com/edeno/statespacecheck/blob/main/CODE_OF_CONDUCT.md).

## License

By contributing to statespacecheck, you agree that your contributions will be licensed under the MIT License.
