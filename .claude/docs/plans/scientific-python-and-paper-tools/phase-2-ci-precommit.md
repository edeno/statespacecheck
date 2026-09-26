# Phase 2: pre-commit and a hardened CI/release workflow

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Branch:** `sp-ci`, from `main` after phase 1 merges. One PR.

**Inputs to read first:**

- [.pre-commit-config.yaml](../../../../.pre-commit-config.yaml) (42 lines): replaced wholesale.
- [.github/workflows/ci.yml](../../../../.github/workflows/ci.yml) (283 lines): rewritten. The job layout today:

  | Job | Lines |
  | --- | --- |
  | quality | 22-50 |
  | test | 52-88 |
  | build | 90-132 |
  | test-install | 134-185, including the public-API check at 169 |
  | publish-testpypi | 187-207 |
  | publish-pypi | 209-227 |
  | create-release | 229-283 |

- `../ripple_detection/.pre-commit-config.yaml` at `cf17183`: the hook set to copy.
- `../ripple_detection/.github/workflows/release.yml` at `cf17183`: the workflow to copy.
  - triggers: 1-23
  - quality + zizmor: 26-48
  - lockfile: 50-59
  - test: 61-92
  - floors: 94-111
  - build: 113-135
  - test-package: 137-178
  - publish: 180-197
  - create-release: 199-230
- `../spectral_connectivity/.github/workflows/release.yml` at `c2dbe60`, for two parts:
  - the three-OS test matrix (`test`, from line 71)
  - the publish job with build-provenance attestations (254-283)
- `../spectral_connectivity/.github/dependabot.yml` at `c2dbe60`: copy it verbatim.
- `../ripple_detection/.github/RELEASE_SETUP.md`: the model for this repo's release doc.

**Contracts referenced:** none.

**Designs referenced:** none.

## Tasks

1. **pre-commit.** Replace `.pre-commit-config.yaml` with ripple_detection's structure. Header comment: "Run `uvx pre-commit install` once …".

   ```yaml
   ci:
     autoupdate_schedule: monthly
     autofix_commit_msg: "pre-commit: apply automatic fixes"
     autoupdate_commit_msg: "pre-commit: update hook versions"
   ```

   Hooks:
   - **pre-commit-hooks v6.0.0:**
     - `check-added-large-files` with `--maxkb=1024`. Exclude the tutorial notebooks; the current config excludes `docs/examples/.*\.ipynb`, which doesn't exist.
     - `check-case-conflict`, `check-merge-conflict`, `check-toml`
     - `check-yaml` with `exclude: '^mkdocs\.yml$'`; mkdocs uses `!!python/name` tags.
     - `end-of-file-fixer` and `trailing-whitespace`, both excluding `uv.lock` and `*.ipynb`
     - `mixed-line-ending`
   - **ruff-pre-commit**, at the rev matching the ruff version in `uv.lock`:
     - `ruff-check` with `["--fix", "--show-fixes"]`
     - `ruff-format`
     - both with `files: '^(src|tests|examples)/'`
   - **codespell v2.4.3**: `additional_dependencies: [tomli]`; exclude `uv.lock` and `*.ipynb`.
   - **local mypy**: `entry: uv run --no-sync mypy`, `language: system`, `pass_filenames: false`, `files: '^(src/.*\.py|pyproject\.toml)$'`.

   Remove these hooks:
   - `mirrors-mypy`. It installs its own numpy and missed the scipy stubs.
   - the local `pytest` hook. It is slow at commit time, and CI runs the tests.
   - `debug-statements`. Ruff's `T20` covers it.

   Run `uvx pre-commit run --all-files` and commit any fixes separately. Enable pre-commit.ci on the GitHub repo (manual; note it in the PR description).

2. **Rewrite `.github/workflows/ci.yml`** following ripple_detection's `release.yml`. **Keep the filename `ci.yml` and the `pypi` environment name**, because PyPI's trusted publisher is bound to both (see [overview Risks](overview.md#risks-and-mitigations)). Workflow `name: CI`.

   **Top-level settings:**
   - `on:` push to `main` and tags `v*`, pull_request to `main`, and `workflow_dispatch` (GH103).
   - `permissions: contents: read`, `concurrency` with cancel-in-progress, `defaults.run.shell: bash`.
   - Every action pinned to the same commit SHAs as ripple_detection `cf17183`, with a `# vX.Y.Z` comment. Every checkout sets `persist-credentials: false`.

   **Jobs:**
   - **quality:**
     - setup-uv, then `uv sync --frozen` (spectral_connectivity's variant: it installs exactly the locked tools)
     - `uv run ruff format --check .`
     - `uv run ruff check . --output-format=github`
     - `uv run mypy`
     - `zizmorcore/zizmor-action` with `persona: regular`
   - **lockfile:** `uv lock --check`.
   - **test:**
     - matrix `os: [ubuntu-latest, macos-latest, windows-latest]` × `python-version: ["3.10","3.11","3.12","3.13","3.14"]`, with `fail-fast: false` and `allow-prereleases: true`
     - `pip install -e .[dev]`, then `pytest --cov-report=xml`
     - codecov upload on ubuntu/3.12 only, with `continue-on-error: true` and `token: ${{ secrets.CODECOV_TOKEN }}`
   - **test-minimum-pins**, Python 3.10, one resolver call:

     ```bash
     pip install numpy==1.26.0 scipy==1.11.1 matplotlib==3.8.0 -e . pytest pytest-cov hypothesis
     pip check
     python -c "import numpy, scipy, matplotlib; assert (numpy.__version__, scipy.__version__, matplotlib.__version__) == ('1.26.0', '1.11.1', '3.8.0')"
     pytest --no-cov -p no:warnings
     ```

     - Use scipy 1.11.1 because 1.11.0 is yanked (spectral_connectivity notes this; confirm with `pip index versions scipy`). Raise `pyproject.toml`'s floor to `scipy>=1.11.1` to match, and add a CHANGELOG "Changed" line.
     - Disable warnings in this job only, with spectral_connectivity's explanatory comment. Old floors emit their own deprecations: at these floors, importing matplotlib 3.8.0 with pyparsing 3.3 raises `PyparsingDeprecationWarning`, which `filterwarnings = ["error"]` would turn into a collection error. Checked on 2026-09-25: with numpy 1.26.0, scipy 1.11.1, matplotlib 3.8.0 and pyparsing 3.3.3 on Python 3.10, the suite fails under warnings-as-errors and passes (303) with `-p no:warnings`.
   - **build:**
     - needs quality, lockfile, test and test-minimum-pins
     - `pip install build twine`, `python -m build`, `twine check dist/*`
     - upload the artifact named `dist`
   - **test-package:**
     - matrix `[wheel, sdist]` on Python 3.12
     - install from `dist/`, then `pip check`
     - run a smoke test that replaces today's "Check public API" step (`ci.yml:169-185`):

     ```python
     import numpy as np
     import statespacecheck as ssc

     assert ssc.__version__ != "0.0.0"
     predictive = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
     fields = np.array([[5.0, 0.1], [1.0, 1.0], [0.1, 5.0]])
     result = ssc.event_diagnostics(predictive, fields, np.array([0, 1]), np.array([0, 0]))
     np.testing.assert_allclose(result.predictive_pvalue, [1.0, 0.172], atol=1e-3)
     ```

     These are the values from the `event_diagnostics` docstring (`events.py:375-379`). Delete the hard-coded `expected = {...}` export set. `__all__` is covered by the unit tests, and the hard-coded set would need editing on every API addition.
   - **publish:**
     - tag pushes only; `environment: pypi`
     - permissions `id-token: write`, `attestations: write`, `contents: read`
     - steps: `actions/attest-build-provenance` on `dist/*`, then `pypa/gh-action-pypi-publish` with `attestations: true`
     - Copy spectral_connectivity's SPEC 8 comment about protecting the `pypi` environment.
   - **create-release:** ripple_detection's `sed`-from-CHANGELOG plus `gh release create`, with `TAG` passed through `env:`, not interpolated (zizmor template-injection).

   **Delete:**
   - the `publish-testpypi` job. Remove the `testpypi` environment from the repo settings afterwards (manual, in the PR description).
   - the `softprops/action-gh-release` step.

3. **Dependabot.** Add `.github/dependabot.yml`, copied from spectral_connectivity: github-actions, monthly, 7-day cooldown, one group.

4. **Release doc.** Add `.github/RELEASE_SETUP.md`, adapted from ripple_detection's. It covers:
   - the trusted publisher: project `statespacecheck`, owner `edeno`, repository `statespacecheck`, workflow `ci.yml`, environment `pypi`
   - the protected `pypi` environment
   - the release checklist: update the CHANGELOG section, update `CITATION.cff` `version` and `date-released`, push the `vX.Y.Z` tag

   Link it from `CONTRIBUTING.md`'s release section (`CONTRIBUTING.md:175-230`). Replace the hard-coded `0.1.0` examples at lines 180, 202-204 and 227 with `X.Y.Z`, and add a step for `CITATION.cff`.

5. **README badges.** Add or refresh the CI workflow badge (`ci.yml`), PyPI, Python versions and codecov, in ripple_detection's style.

## Deliberately not in this phase

- `docs.yml` hardening and notebook execution (phase 3). zizmor will flag `docs.yml`. If the quality job's zizmor step scans every workflow and fails on `docs.yml`, apply the minimal pin and permissions fixes to `docs.yml` here and leave the rest to phase 3.
- A nox/tox file (by design).
- Any package source change.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uvx pre-commit run --all-files` | All hooks pass on the tree |
| `uvx zizmor .github/workflows/` | No findings at the `regular` persona |
| `uvx --from "sp-repo-review[cli]" repo-review .` | Fails exactly PY007, PC140, PC170, PC180, the same set as the references |
| PR CI run | quality, lockfile, 15-cell test matrix, test-minimum-pins, build and test-package green |
| Manual `workflow_dispatch` on the branch | Runs everything up to test-package; publish and create-release are skipped (no tag) |
| `actionlint` (optional, `uvx actionlint-py`) | Workflow YAML valid |

## Fixtures

None.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Workflow filename and environment name are unchanged (`ci.yml`, `pypi`); every action is SHA-pinned.
