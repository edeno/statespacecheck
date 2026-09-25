# Phase 1: Packaging and tool configuration per the Scientific Python guide

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Branch:** `sp-packaging`, from `main`. One PR. Commit per task; the order follows ripple_detection's migration, `git -C ../ripple_detection log --oneline 7fd5d12^..d322796`.

**Inputs to read first:**

- [pyproject.toml](../../../../pyproject.toml): the current config.
  - `[project]`: 1-37 (`license` at 7, authors 8-12, keywords 13-20, the `Typing :: Typed` classifier at 35)
  - urls: 38-44
  - extras: 45-64
  - build-system: 65-67
  - hatch: 69-80
  - ruff: 81-127
  - pytest: 128-139
  - coverage: 140-153
  - mypy: 154-171
- `../ripple_detection/pyproject.toml` at commit `cf17183`, the template:
  - build: 1-3
  - `[project]`: 5-27
  - dev extra and group: 29-52
  - sdist: 65-69
  - pytest: 77-93
  - coverage: 95-110
  - mypy: 112-122
  - ruff: 124-180
  - codespell: 182-184
- `../spectral_connectivity/pyproject.toml` at `c2dbe60`: how it uses `scipy-stubs` instead of a global `ignore_missing_imports`.
- [src/statespacecheck/predictive_checks.py:408-411](../../../../src/statespacecheck/predictive_checks.py): the one strict-mypy error (`unreachable`).
- [tests/test_predictive_pvalue.py:189-195](../../../../tests/test_predictive_pvalue.py): the test that pins that dead branch's message.
- [.github/workflows/ci.yml:22-50](../../../../.github/workflows/ci.yml): the quality job. It runs on Python 3.10 and must move to 3.12 in this phase; see the task on strict mypy.

**Contracts referenced:** none.

**Designs referenced:** none.

## Tasks

Each bullet is one commit. After every commit, run `uv run pytest`, `uv run mypy` and `uv run ruff check`.

1. **Doctests pass.** Seven docstring examples are broken today; this is a prerequisite for `--doctest-modules`. Found by `MPLBACKEND=Agg pytest --doctest-modules src`.

   | Location | Problem | Fix |
   | --- | --- | --- |
   | `periods.py:519` | Multi-line import lacks `...` continuation lines | Add the continuation lines |
   | `periods.py:361` | Multi-line `if` block lacks `...` continuation lines | Add the continuation lines |
   | `periods.py:420` and `flag_low_overlap`, `flag_extreme_pvalues` | Expect `5`, get `np.int64(5)` (NumPy 2 repr) | Wrap in `int(...)` |
   | `predictive_checks.py`, `predictive_pvalue` example | Fails | Fix after inspecting with `pytest --doctest-modules src/statespacecheck/predictive_checks.py` |
   | `viz.py:74-81` | `plt.show()` blocks interactively and warns under Agg | Replace with `plt.close(fig)` |

   Doctests in `src/` also need the Agg backend. `tests/conftest.py:3,7` sets it, but only for `tests/`. Move those two lines to a new root `conftest.py` so both test paths get them.

2. **uv dependency group and lockfile.**
   - Add `[dependency-groups] dev = [...]`, identical to the `dev` extra, with the comment "Keep identical" (ripple_detection lines 29-52).
   - Dev list:
     - existing: `pytest>=8.0`, `pytest-cov>=6.0.0`, `hypothesis>=6.0.0`, `mypy>=1.13.0`
     - pin ruff: `ruff>=0.16,<0.17`
     - add `scipy-stubs`
     - drop `pre-commit`; the docs now say `uvx pre-commit`.
   - Leave the `docs` extra as is.
   - Run `uv lock` and commit `uv.lock`.

3. **`py.typed`.** Add an empty `src/statespacecheck/py.typed`. The `Typing :: Typed` classifier already claims it (`pyproject.toml:35`), but the file is missing.

4. **`CITATION.cff`** at the repo root, cff-version 1.2.0. Model it on ripple_detection's.
   - Authors from `pyproject.toml:8-12`, as `family-names`/`given-names`.
   - `version: 0.2.0`, `date-released: "2026-09-25"`, license `MIT`.
   - `repository-code` and `url` set to the GitHub URL.
   - Keywords from `pyproject.toml:13-20`.
   - Add a `preferred-citation` stub only when the paper has a DOI; phase 5 decides.

5. **Strict pytest.** Replace `[tool.pytest.ini_options]` with:

   ```toml
   [tool.pytest.ini_options]
   minversion = "8"
   # src/ for the docstring examples
   testpaths = ["tests", "src"]
   addopts = [
       "-ra",
       "--doctest-modules",
       "--strict-config",
       "--strict-markers",
       "--cov=src/statespacecheck",
       "--cov-report=term-missing",
   ]
   xfail_strict = true
   log_level = "INFO"
   # every warning is an error: a test that expects one says so with pytest.warns
   filterwarnings = ["error"]
   ```

   Drop `-v` and `--cov-report=html`. `htmlcov/` stays gitignored.

   Replace `[tool.coverage.run]` with:

   ```toml
   branch = true
   # by path: with the src layout the module name resolves ambiguously
   source = ["src/statespacecheck"]
   omit = ["*/tests/*", "*/_version.py"]
   ```

   Keep `[tool.coverage.report]` as is. It already matches ripple_detection.

6. **The references' ruff rule set at line length 95.** Two commits.
   - **(a) Config and fixes.** `line-length = 95`; delete `target-version` (fixes RF002). `select` is ripple_detection's 26 codes plus `"D"`:

     ```
     "E","W","F","I","B","C4","UP","NPY","PD","RUF","ARG","EM","EXE","FURB","G","ICN",
     "ISC","PERF","PGH","PIE","PT","PTH","RET","SIM","T20","YTT","D"
     ```

     `ignore = ["E501", "ISC001", "D100", "D104"]`. This drops `N` (pep8-naming), as the references do.

     Per-file ignores:
     - keep the existing examples/notebooks entries
     - `"tests/**/*.py" = ["D", "ARG"]`
     - `"examples/*.py"` adds `"T201"`

     Keep `[tool.ruff.lint.pydocstyle] convention = "numpy"` and `[tool.ruff.format]` as they are. Add `exclude = ["src/statespacecheck/_version.py"]`.

     Then fix the 51 findings measured on 2026-09-25:
     - 39 EM101/EM102: move each message into `msg = ...; raise X(msg)`, the style of `predictive_checks.py:395-399`
     - RET504/RET505, PT018, RUF022 (sort `__all__`), ICN001, RUF043, RUF059, SIM118
     - Error messages must not change; existing `pytest.raises(match=...)` tests check this.
   - **(b) Reformat.** `ruff format` alone, in its own commit. Then add `.git-blame-ignore-revs` containing that commit's SHA.

7. **Strict mypy.** Replace `[tool.mypy]` and its override with:

   ```toml
   [tool.mypy]
   files = ["src/statespacecheck"]
   mypy_path = "src"
   # Target 3.12 so mypy can parse NumPy's stubs, which use PEP 695 `type`
   # statements. Runtime 3.10 support is verified by the test matrix.
   python_version = "3.12"
   strict = true
   warn_unreachable = true
   enable_error_code = ["ignore-without-code", "redundant-expr", "truthy-bool"]
   ```

   The override for `scipy.*` goes; `scipy-stubs` (task 2) provides the types. If mypy then reports errors in scipy calls, fix the annotations; never use `# type: ignore` (CLAUDE.md).

   **Fix the unreachable branch.** Delete `predictive_checks.py:408-411`, the `if not callable(...)` check. The parameter is typed `Callable`, and calling a non-callable still raises `TypeError`, so the docstring's `Raises TypeError` stays true. In `tests/test_predictive_pvalue.py:194`, change `match="sample_log_pred must be callable"` to `match="not callable"`, which is the interpreter's message.

   **Switch the quality job to Python 3.12.** In `.github/workflows/ci.yml:35`, change `python-version: "3.10"` to `"3.12"` and update the comment at lines 30-31. Under 3.10, mypy cannot parse the NumPy stubs once `python_version = "3.12"`. Phase 2 replaces this job, but the phase-1 PR must pass CI.

8. **Metadata.**
   - `[build-system] requires = ["hatchling>=1.27", "hatch-vcs"]`, with the comment `# 1.27: PEP 639 license expressions`.
   - `license = "MIT"` and `license-files = ["LICENSE"]` replace `license = { file = "LICENSE" }`.
   - Delete the `License :: OSI Approved :: MIT License` classifier (PEP 639 forbids it alongside an SPDX license).
   - Add the `Programming Language :: Python :: 3.14` classifier.
   - Replace `[tool.hatch.build.targets.sdist] exclude = [".git_archival.txt"]`, which refers to a file that doesn't exist, with:

     ```toml
     include = ["/src", "/tests", "/examples", "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md"]
     ```

9. **codespell config.** Add:

   ```toml
   [tool.codespell]
   skip = "uv.lock,*.ipynb,htmlcov"
   ```

   Run `uvx codespell` and fix the real typos it finds. Add false positives (for example author initials) to `ignore-words-list`.

10. **Docs for developers.**
    - Replace `CLAUDE.md` "Development Commands" and README "Development" (`README.md:245-290`) with the uv workflow: `uv sync`, `uv run pytest`, `uv run mypy`, `uv run ruff check`, `uv run ruff format`.
    - Add a line stating that `uv run` is the task runner and that no nox/tox file exists by design, with the reason, as in `ripple_detection/CLAUDE.md:172`.
    - Update CLAUDE.md's "100 character line length" to 95.
    - CONTRIBUTING's setup sections (`CONTRIBUTING.md:20-35,90-112`) get the same treatment.
    - Add a CHANGELOG `## [Unreleased]` → `### Changed` entry: "Development tooling follows the Scientific Python guide: strict pytest, mypy and ruff, a `dev` dependency group with a committed lockfile, `py.typed`, `CITATION.cff`."

## Deliberately not in this phase

- `.pre-commit-config.yaml` changes. Phase 2 does these, including `ruff-check` and `--show-fixes`. The existing hooks keep working meanwhile.
- The CI workflow rewrite (phase 2). The only CI edit here is the quality job's Python version (task 7).
- Tutorial deduplication and notebook execution (phase 3).
- Any change to numerical behaviour or public signatures.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest` | All 269 existing tests plus the doctests pass under `filterwarnings=error`; coverage report printed |
| `uv run mypy` | `Success: no issues found` with strict settings |
| `uv run ruff check . && uv run ruff format --check .` | Clean |
| `uv lock --check` | Lockfile current |
| `uvx --from "sp-repo-review[cli]" repo-review .` | PP006, PP302, PP304, PP305, PP306, PP308, PP309, MY101, MY103–MY106 and RF002 now pass; the only remaining failures are GH/PC/SEC (phase 2) and PY007 (by design) |
| `uv build && uvx twine check dist/*` | Wheel contains `statespacecheck/py.typed`; metadata shows `License-Expression: MIT` |
| `tests/test_predictive_pvalue.py::test_sampler_not_callable_error` | Still raises `TypeError` with the interpreter's message |
| CI on the PR | Quality job (Python 3.12) and the full test matrix green |

## Fixtures

None new.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Error-message text is unchanged by the EM rewrite (diff the strings).
