# Phase 3: Documentation (one tutorial source, executed notebooks, hardened docs workflow)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Branch:** `sp-docs`, from `main` after phase 2 merges. One PR.

**Inputs to read first:**

- [mkdocs.yml](../../../../mkdocs.yml):
  - `mkdocs-jupyter` plugin with `execute: false`
  - `gen-files` running `docs/gen_ref_pages.py`
  - the tutorial nav at the end of the file, pointing at `tutorials/0*.ipynb`
- [docs/gen_ref_pages.py](../../../../docs/gen_ref_pages.py): the existing gen-files script, the pattern to extend.
- The duplicated tutorials. `docs/tutorials/0{1..4}_*.ipynb` and `examples/0{1..4}_*.ipynb` are **byte-identical** (checked with `cmp` on 2026-09-25). `examples/*.py` are their jupytext `py:percent` pairs, and they import `examples/utils.py`.
- [.github/workflows/docs.yml](../../../../.github/workflows/docs.yml) (67 lines): unpinned actions, `workflow_dispatch` missing, broad top-level `pages: write` / `id-token: write`.
- [src/statespacecheck/periods.py:25-30](../../../../src/statespacecheck/periods.py) and [:98-106](../../../../src/statespacecheck/periods.py): docstrings claiming "the approach from the paper" and "the paper's weighted average equations". The manuscript (`statespacecheck-paper/manuscript/main.tex`) has no such equations.
- [README.md:304-320](../../../../README.md): the Citation block. It has `version={0.1.0}` and a placeholder DOI `10.5281/zenodo.XXXXXXX`.

**Contracts referenced:** none.

**Designs referenced:** none.

## Tasks

1. **One source for tutorials.** `examples/` is canonical: the `.py` jupytext file plus the paired `.ipynb`, which keeps its outputs.
   - `git rm docs/tutorials/0*.ipynb`.
   - Add `docs/gen_tutorials.py`, a gen-files script that copies each `examples/0*.ipynb` into the virtual `tutorials/` directory:

     ```python
     """Copy the example notebooks into the docs as tutorials."""

     from pathlib import Path

     import mkdocs_gen_files

     examples = Path(__file__).parent.parent / "examples"
     for notebook in sorted(examples.glob("[0-9][0-9]_*.ipynb")):
         with mkdocs_gen_files.open(f"tutorials/{notebook.name}", "wb") as fd:
             fd.write(notebook.read_bytes())
     ```

   - Register it in `mkdocs.yml` under `gen-files: scripts:`, **before** `docs/gen_ref_pages.py`. The nav entries stay as they are.
   - **Verify** with `uv run --extra docs mkdocs build --strict` that the four tutorial pages render with their outputs. If `mkdocs-jupyter` does not pick up gen-files notebooks, use this fallback instead:
     - a copy step in `docs.yml` before the build: `cp examples/0*.ipynb docs/tutorials/`
     - `docs/tutorials/0*.ipynb` added to `.gitignore`
     - a one-line note in CONTRIBUTING on running the copy before `mkdocs serve`

     This is spectral_connectivity's approach (its `docs/conf.py` copies `examples/`).

2. **Execute the notebooks in CI.** In `ci.yml`'s `test` job, on ubuntu/3.12 only (as ripple_detection does), add a step:

   ```bash
   pip install nbconvert ipykernel jupytext
   for nb in examples/0*.ipynb; do
     jupyter nbconvert --to notebook --execute --output-dir "$RUNNER_TEMP" "$nb"
   done
   # the paired .py and .ipynb must agree
   jupytext --sync --warn-only examples/0*.py && git diff --exit-code examples/
   ```

   Add `nbconvert`, `ipykernel` and `jupytext` to the `docs` extra so contributors get them too; `jupytext` is already there.

   The notebooks must run under `filterwarnings`-free defaults. nbconvert doesn't read the pytest config, so no warnings change is needed.

3. **Harden `docs.yml`** to the phase-2 standard:
   - Add `workflow_dispatch`.
   - Top-level `permissions: contents: read`. The deploy job only gets `pages: write` and `id-token: write`.
   - Pin every action to a commit SHA with a version comment, using the same SHAs as `ci.yml` for shared actions (checkout, setup-python, setup-uv).
   - `persist-credentials: false` on checkout.
   - Keep `mkdocs build --strict`.
   - `uvx zizmor .github/workflows/docs.yml` must be clean.

4. **Fix the stale "paper" docstrings in `periods.py`.** Behaviour is unchanged.
   - Line 25-30 (`aggregate_over_period` summary): "an indicator function approach from the paper" → "an indicator (mask) over time points".
   - Lines 98-106 (Notes): drop "from the paper" and "Consistent with paper's weighted average equations". Keep the use-case bullets.

   No other `src/` file mentions the paper (checked with `grep -n -i paper src/statespacecheck/*.py`).

5. **README and docs index.**
   - **Citation** (`README.md:304-320`):
     - Point to `CITATION.cff` ("GitHub's *Cite this repository* button").
     - Fix the bibtex to `version={0.2.0}`, `year={2026}`.
     - Keep the DOI line as a placeholder, with a comment that it arrives with the first Zenodo-archived release (phase 4b).
     - Add a sentence citing the companion paper by title and repository, `https://github.com/edeno/statespacecheck-paper`, with no DOI yet.
   - **Development:** README and `docs/contributing.md` must match `CONTRIBUTING.md` as phase 1 left it. `docs/contributing.md` is a separate copy; make it an include (`--8<-- "CONTRIBUTING.md"`, via `pymdownx.snippets`, which `mkdocs.yml` already enables) so they cannot drift.
   - CHANGELOG `[Unreleased]` → `### Changed`: "Tutorials have a single source in `examples/` and are executed in CI."

## Deliberately not in this phase

- New tutorials for the continuous-mark API (phase 4b).
- Removing or deprecating `periods.py`/`viz.py`. They stay (overview Non-Goals).
- Sphinx/RTD. mkdocs stays.
- Any code-behaviour change.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run --extra docs mkdocs build --strict` | Builds with no warnings; `site/tutorials/01_introduction/index.html` exists and contains rendered output cells |
| CI notebook step | All four notebooks execute without error; `.py`/`.ipynb` pairs in sync |
| `git ls-files docs/tutorials` | Only `index.md` remains |
| `uvx zizmor .github/workflows/` | Clean |
| `uvx --from "sp-repo-review[cli]" repo-review .` | Still exactly PY007, PC140, PC170, PC180 failing |
| `uv run pytest` | Unchanged pass count (docstring edits only) |
| docs.yml on the PR | Build job green; deploy skipped (not main) |

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
