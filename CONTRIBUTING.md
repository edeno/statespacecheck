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

2. **Create a virtual environment and install dependencies**
   ```bash
   # Using uv (recommended - much faster)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   python -c "import statespacecheck; print(statespacecheck.__version__)"
   pytest --version
   ruff --version
   mypy --version
   ```

## Development Workflow

### Code Quality Standards

This project follows strict code quality standards:

- **Formatting**: [ruff format](https://docs.astral.sh/ruff/formatter/) (100 char line length)
- **Linting**: [ruff check](https://docs.astral.sh/ruff/) (comprehensive rules including NumPy-specific)
- **Type checking**: [mypy](https://mypy-lang.org/) in strict mode (no `# type: ignore` allowed)
- **Testing**: [pytest](https://pytest.org/) with 100% coverage requirement
- **Docstrings**: [NumPy style](https://numpydoc.readthedocs.io/)

### Running Checks Locally

Before committing, run all quality checks:

```bash
# Format code
ruff format .

# Check formatting (CI runs this)
ruff format --check .

# Lint code
ruff check .

# Fix auto-fixable linting issues
ruff check --fix .

# Type check
mypy src/

# Run tests with coverage
pytest

# Run tests without coverage report
pytest --no-cov

# Run specific test file
pytest tests/test_highest_density.py -v

# Run specific test
pytest tests/test_highest_density.py::TestHighestDensityRegion::test_exact_hd_region_1d -xvs
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before every commit.

**Setup (one-time):**
```bash
# Install dependencies (includes pre-commit)
uv pip install -e ".[dev]"

# Install the git hooks
uv run pre-commit install
```

**Usage:**
```bash
# Hooks run automatically on `git commit`

# Run manually on all files
uv run pre-commit run --all-files

# Run manually on staged files only
uv run pre-commit run

# Update hook versions
uv run pre-commit autoupdate
```

**What it checks:**
- Code formatting with ruff
- Linting with ruff (auto-fixes when possible)
- Type checking with mypy
- All tests with pytest

This matches exactly what CI runs, so commits that pass hooks will pass CI.

## Continuous Integration

### GitHub Actions Workflow

The CI/CD pipeline runs automatically on:
- **Pull requests** to `main` branch
- **Pushes** to `main` branch
- **Git tags** matching `v*` pattern

### CI Jobs

1. **Code Quality** (`quality`)
   - Runs on Python 3.12
   - Checks: `ruff format --check`, `ruff check`, `mypy src/`
   - Fast feedback (~1-2 minutes)

2. **Tests** (`test`)
   - Matrix: Python 3.10, 3.11, 3.12, 3.13
   - Runs: pytest with coverage
   - Uploads coverage to Codecov (Python 3.12 only)

3. **Build** (`build`)
   - Requires: `quality` and `test` jobs to pass
   - Builds: wheel and sdist
   - Validates: `twine check dist/*`
   - Uploads: distribution artifacts

4. **Install Tests** (`test-install`)
   - Matrix: wheel/sdist × Python 3.10/3.13
   - Tests actual installation from built packages
   - Verifies: imports, version, public API

5. **Publish to TestPyPI** (`publish-testpypi`)
   - Trigger: git tags matching `v*`
   - Tests publishing to TestPyPI first (safety check)

6. **Publish to PyPI** (`publish-pypi`)
   - Trigger: after TestPyPI succeeds
   - Publishes to production PyPI

7. **Create GitHub Release** (`create-release`)
   - Trigger: after PyPI publish succeeds
   - Creates GitHub release with notes
   - Attaches distribution files

### Viewing CI Results

- **In Pull Requests**: Check the "Checks" tab
- **In Commits**: Look for ✓ or ✗ next to commit hash
- **In Actions**: Go to the "Actions" tab in GitHub

## Release Process

### Version Management

This project uses **VCS-based versioning** with `hatch-vcs`:
- Version is derived from **git tags**
- Development versions: `0.1.dev19+gf37b8a4e4.d20251105`
- Release versions: `0.1.0` (from tag `v0.1.0`)

**Do not** edit version numbers manually in code!

### Creating a Release

1. **Ensure main branch is ready**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Verify tests pass locally**
   ```bash
   pytest
   ruff format --check .
   ruff check .
   mypy src/
   ```

3. **Create and push a version tag**
   ```bash
   # For version 0.1.0
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. **Monitor the CI/CD pipeline**
   - Go to GitHub Actions tab
   - Watch the workflow progress through:
     - ✓ Code quality and tests
     - ✓ Build distributions
     - ✓ Test installations
     - ✓ Publish to TestPyPI
     - ⏸️  **Requires approval** → Publish to PyPI
     - ✓ Create GitHub release

5. **Approve PyPI deployment** (if required)
   - Go to Actions tab → Click on the workflow run
   - Click "Review deployments" button
   - Approve the `pypi` environment

6. **Verify the release**
   - Check [PyPI](https://pypi.org/project/statespacecheck/)
   - Check [GitHub Releases](https://github.com/edeno/statespacecheck/releases)
   - Test installation:
     ```bash
     pip install statespacecheck==0.1.0
     python -c "import statespacecheck; print(statespacecheck.__version__)"
     ```

### Release Checklist

- [ ] All tests pass on main branch
- [ ] CHANGELOG updated (if you maintain one)
- [ ] Documentation is up to date
- [ ] Version tag follows semantic versioning
- [ ] Tag pushed to GitHub
- [ ] CI pipeline completes successfully
- [ ] Package available on PyPI
- [ ] GitHub release created

## PyPI Trusted Publishing Setup

### First-time Setup

The CI/CD pipeline uses **Trusted Publishing** (no API tokens needed!). Set it up once:

#### 1. PyPI Configuration

1. Go to [pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
2. Scroll to "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `statespacecheck`
   - **Owner**: `edeno` (your GitHub username)
   - **Repository name**: `statespacecheck`
   - **Workflow name**: `ci.yml`
   - **Environment name**: `pypi`
4. Click "Add"

#### 2. TestPyPI Configuration (Optional but Recommended)

1. Go to [test.pypi.org/manage/account/publishing/](https://test.pypi.org/manage/account/publishing/)
2. Repeat the same steps with environment name: `testpypi`

#### 3. GitHub Environments (Optional)

Add approval gates for extra safety:

1. Go to your repo → Settings → Environments
2. Create `pypi` environment:
   - Click "New environment"
   - Name: `pypi`
   - Add required reviewers (yourself or team members)
   - Protection rules: Require approval before deployment
3. Create `testpypi` environment (optional, can be auto-approved)

### How Trusted Publishing Works

1. GitHub Actions generates a short-lived OIDC token
2. PyPI verifies the token matches the configured repository/workflow
3. No long-lived API tokens needed!
4. More secure than using PyPI API tokens

## Testing

### Test Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── test_highest_density.py          # HPD region tests
├── test_state_consistency.py        # KL divergence, HPD overlap tests
├── test_predictive_density.py       # Predictive checks tests
├── test_validation.py               # Input validation tests
├── test_edge_cases.py              # Edge case handling
└── test_properties.py              # Property-based tests (Hypothesis)
```

### Running Tests

```bash
# All tests with coverage
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run specific test class
pytest tests/test_highest_density.py::TestHighestDensityRegion -v

# Run specific test method
pytest tests/test_highest_density.py::TestHighestDensityRegion::test_exact_hd_region_1d -xvs

# Run tests matching pattern
pytest -k "test_hpd" -v

# Run with coverage report
pytest --cov=statespacecheck --cov-report=html
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

## Code Style Guidelines

### General Principles

1. **Readability**: Code is read more often than written
2. **Simplicity**: Prefer simple solutions over clever ones
3. **Explicitness**: Explicit is better than implicit
4. **Documentation**: All public APIs must be documented
5. **Type safety**: Use type hints everywhere

### Python Style

- **Line length**: 100 characters (ruff enforces this)
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

**Problem**: `__version__` is `0.1.dev19+g...` instead of `0.1.0`

**Solution**: This is expected in development! Version comes from git tags. To test release version:
```bash
git tag v0.1.0
pip install -e .
```

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

- **Issues**: [GitHub Issues](https://github.com/edeno/statespacecheck/issues)
- **Discussions**: [GitHub Discussions](https://github.com/edeno/statespacecheck/discussions)
- **Email**: eric.denovellis@ucsf.edu

## License

By contributing to statespacecheck, you agree that your contributions will be licensed under the MIT License.
