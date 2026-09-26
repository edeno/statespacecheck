# Overview: Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Baseline (measured 2026-09-25)

`uvx --from "sp-repo-review[cli]" repo-review <repo>` results:

| Repo | Fail | Pass | Notes |
| --- | --- | --- | --- |
| statespacecheck | 25 | 30 | 4 skipped |
| ripple_detection (`cf17183`) | 4 | 55 | The failures are deliberate: PY007 (no nox/tox), PC140 (mypy runs as a local hook, which repo-review doesn't detect), PC170, PC180 |
| spectral_connectivity (`c2dbe60`) | 4 | 60 | Same four |

statespacecheck's 25 failures:

- **Packaging:** PY007, PP006.
- **pytest:** PP302, PP304, PP305, PP306, PP308, PP309.
- **GitHub:** GH103, GH200, SEC001.
- **mypy:** MY101, MY103, MY104, MY105, MY106.
- **pre-commit:** PC160, PC170, PC180, PC191, PC192, PC901, PC902, PC903.
- **ruff:** RF002.

**Target:** exactly the references' four failures (PY007, PC140, PC170, PC180) and nothing else. Re-run the command after each phase.

Measured cost of the stricter settings on the current code:

- The references' ruff rule set flags 51 issues in `src/` and `tests/`:
  - EM101/EM102: 39
  - PT018: 2
  - RET504: 2
  - RET505: 2
  - RUF022: 2
  - ICN001: 1
  - RUF043: 1
  - RUF059: 1
  - SIM118: 1
- Strict mypy with the three extra error codes and `warn_unreachable` reports one error: `src/statespacecheck/predictive_checks.py:409-411`, an unreachable `if not callable(...)` check.
- `pytest -W error` passes all 269 tests.

## Current codebase integration points

**Package (`statespacecheck`)**

- `pyproject.toml:1-171`: rewritten section by section in phase 1, following `ripple_detection/pyproject.toml` at `cf17183`.
- `.pre-commit-config.yaml:1-42`: replaced in phase 2.
- `.github/workflows/ci.yml:1-283`: rewritten in phase 2. **Keep the filename `ci.yml` and the environment name `pypi`**, because PyPI's trusted publisher for `statespacecheck` is registered against them.
- `.github/workflows/docs.yml:1-67`: hardened in phase 3.
- `src/statespacecheck/__init__.py:37-67`:
  - Phase 1 sorts `__all__` (RUF022); the `importlib.metadata` fallback for `__version__` stays, as in spectral_connectivity.
  - Phases 4a and 4b add exports.
- `src/statespacecheck/events.py:176-244` (`predictive_mark_probabilities`) and `:319-427` (`event_diagnostics`): numerical output unchanged. The new functions sit beside them, and `event_diagnostics` output must stay bit-identical for the paper. Phase 4b task 1 hardens their error messages and validation without changing any output.
- `src/statespacecheck/predictive_checks.py:409-411`: the dead `callable` check is removed in phase 1.
- `src/statespacecheck/periods.py:25-30,100-106`: docstrings cite "the paper's weighted average equations", which `main.tex` no longer contains. Fixed in phase 3; behaviour unchanged. You chose to keep `periods.py`, `viz.py` and the generic predictive-check functions as general tools.
- `docs/tutorials/0*.ipynb` are git symlinks (mode 120000) to `examples/0*.ipynb`, so the tutorials already have one source; phase 3 keeps them.

**Paper (`statespacecheck-paper`)**

- `pyproject.toml:34` pins `statespacecheck>=0.2.0`.
- Calls into the package (line numbers from branch `statespacecheck-boundary` at `82a0b80`):
  - `src/statespacecheck_paper/diagnostics.py:451`: `event_diagnostics`.
  - `decoding.py:401`: `event_diagnostics`, for Figure 3's rate-override windows.
  - `diagnostics.py:611`: `baseline_threshold`, wrapped in `float()` only because the package shipped no `py.typed` (removed in phase 5).
  - `figure02_panels.py:154-155`: `kl_divergence`, `hpd_overlap`.
  - `figure04_diagnostics.py:387`, `site_export.py:897` and `interactive/data_source.py:493`: `event_likelihood`.
- `figure02_panels.py:170-211` hand-rolls the Monte Carlo predictive p-value. Phase 5 replaces it with `monte_carlo_mark_pvalue`.
- `plotting.py:77` (`compute_hpd_region`) duplicates `highest_density_region`. Its callers are `plotting.py:244-245` and `figure02_panels.py:377,474-475`. Phase 5 replaces it.
- `manuscript/main.tex:375` has the red "archival DOI" TODO; phase 5 fills it.
- The diagnostics cache fingerprint includes the installed package version (`figure04_cache.py:260-261`). Bumping the version recomputes the Figure 4 diagnostics.
- Both figure summaries record the installed package version in `provenance.source.statespacecheck_version` (`scientific_artifacts.py`). The manuscript cites it through `\StatespacecheckVersion`, and `reported_values.py` refuses summaries, or a Figure 4 cache, that record different versions.
- `site/js/metrics.js` mirrors the package's sorted-mark diagnostics and has a parity fixture (`site_export.py:657`).
- **The paper repo has unmerged work** on branches `statespacecheck-boundary` (unpushed, `df887f2..cbc7bfd` as of 2026-09-25) and `shorten-fig-captions`. Both change `manuscript/main.pdf`, a regenerated file: after merging, rebuild it rather than resolving the conflict by hand. Phase 5 must not start until both are merged to `main`.

## Scope and dependency policy

### Goals

- Match the references' pattern:
  - hatchling ≥ 1.27 with hatch-vcs
  - PEP 639 license
  - a `dev` extra and an identical `dev` dependency group
  - a committed `uv.lock` checked in CI
  - strict pytest, strict mypy, and the references' 26-code ruff set at line length 95
  - codespell (configured in `pyproject.toml`, run from pre-commit)
  - `py.typed`
  - `CITATION.cff`
  - the references' pre-commit set with a `ci:` block
  - a single SHA-pinned, least-privilege CI/release workflow: quality + zizmor, lockfile, test matrix 3.10–3.14, dependency floors, build, wheel/sdist smoke test, trusted publishing with attestations, and a GitHub release built from the CHANGELOG
  - Dependabot for Actions
  - Python 3.14 support
- Keep this repo's extras: pydocstyle `D` (numpy convention), mkdocs-material on GitHub Pages, and the three-OS test matrix.
- Expose the manuscript's missing methods as public API (see [shared-contracts.md](shared-contracts.md)):
  - `event_weighted_predictive` (`main.tex:174-178`)
  - `monte_carlo_mark_pvalue` (`main.tex:215-217`, Fig 2b)
  - `clusterless_event_diagnostics` (`main.tex:152,195-201`)
- Release v0.3.0 with a Zenodo DOI; move the paper onto it.

### Non-Goals

- No nox/tox file. `uv run` is the task runner, as in the references (`ripple_detection/CLAUDE.md:172`).
- No Sphinx/RTD migration; mkdocs stays.
- No markdown formatter (PC180) and no pygrep hooks (PC170), matching the references.
- No change to the numerical output of any existing public function. In particular, `event_diagnostics` must stay bit-identical for the paper.
- No removal or deprecation of `periods.py`, `viz.py`, `predictive_density`, `log_predictive_density` or `predictive_pvalue`.
- No time-rescaling / KS diagnostic, f-divergences other than KL, or symmetric KL (the manuscript only mentions these).
- No change to the paper's thresholds, flag rules or decoder. Those are paper-specific by design (`statespacecheck-paper/README.md:165-174`).
- No port of the continuous-mark diagnostics to `site/js/metrics.js`; the site shows sorted data only.

### Deferred

Raised in the review of the paper's move to 0.2.0; not in any phase of this plan:

- **Reusing work across events that share a time bin.** `event_diagnostics` recomputes the predictive HPD region and the predictive mark probabilities for every event, even when several events fall in the same bin. Computing them once per unique `time_ind` would cut that work, but the matrix product's reduction order depends on the number of rows, so the p-values could change in the last bit (statespacecheck `tests/test_events.py::test_batch_size_does_not_change_results` shows this for batch sizes). That breaks the bit-identity goal above. Revisit only if a profile shows the diagnostics dominate a user's runtime.
- **Moving the paper's desktop viewer into this package** ([Open Question 4](#open-questions)).

### Dependency policy

- Runtime dependency floors stay `numpy>=1.26.0` and `matplotlib>=3.8.0`. `scipy>=1.11.0` becomes `>=1.11.1` in phase 2, because 1.11.0 is yanked (as in spectral_connectivity). The new floors job (phase 2) proves they work. Raise any other floor only if that job fails, and record why in the CHANGELOG.
- No new runtime dependencies.
- New dev dependencies:
  - phase 1: `scipy-stubs` (codespell runs from pre-commit and is not a dev dependency, as in the references)
  - phase 3: `jupyter` and `nbconvert`, to execute notebooks
- Pin ruff to a minor version (`ruff>=0.16,<0.17`, as in ripple_detection), and pin the pre-commit ruff rev to the version in `uv.lock`.

## Metrics

- repo-review fails only PY007, PC140, PC170 and PC180 after phase 2 and at the end.
- `uv run pytest`, `uv run mypy`, `uv run ruff check`, `uv run ruff format --check` and `uv lock --check` are clean on every phase branch. CI is green on 3.10–3.14 × {ubuntu, macos, windows} plus the floors job.
- Coverage stays at the current level; phase-4 modules have 100% line coverage.
- On discrete marks, `monte_carlo_mark_pvalue` agrees with the exact `mark_predictive_pvalue` within 4 binomial standard errors at `n_samples=20_000`.
- Paper after phase 5:
  - Figure 3 and 4 numbers (`manuscript/reported_values.tex`) are unchanged apart from `\StatespacecheckVersion`.
  - The Figure 2 p-value changes only by Monte Carlo resampling, within 4 binomial SE of the old value.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The workflow rewrite breaks trusted publishing: PyPI binds the publisher to workflow filename + environment. | Keep `ci.yml` and environment `pypi`. Drop the TestPyPI job; the references don't use one, and the wheel/sdist smoke test covers it. Do a `workflow_dispatch` dry run before tagging. |
| `filterwarnings = ["error"]` fails on warnings that only occur on other OSes or on NumPy at the floor versions. | Phase 2's matrix and floors job surface them. Add narrowly matched `ignore:` entries with a comment naming the platform, as spectral_connectivity does for macOS matmul. Never add a blanket ignore. |
| Enabling EM (exception message) rules churns 39 raises. | Mechanical: `msg = f"..."; raise ValueError(msg)`, the style `predictive_checks.py:395-399` already uses. Existing `pytest.raises(match=...)` tests verify that messages are unchanged. |
| Line length 100 → 95 reformats many lines and muddies blame. | One commit that runs only `ruff format`, listed in `.git-blame-ignore-revs`. |
| The Monte Carlo sampler's memory is `batch × n_samples × n_bins`. | Batch over events (see [designs.md](designs.md#memory-and-batching)); default budget ≈ 130 MB. |
| Monte Carlo results depend on the RNG stream, so the paper's Figure 2 changes. | Figure 2 is a schematic. Accept the change and check the new p-value is within MC error of the old one. `reported_values.tex` must not contain the Figure 2 p-value; verify that in phase 5. |
| The paper has unmerged work on `statespacecheck-boundary` and `shorten-fig-captions`. | Phase 5 has a hard precondition that both are merged. |
| No real clusterless dataset is available in either repo. | Validate on simulated clusterless data: calibration under a correct model, and power under a misspecified one. The real-data smoke test uses the sorted paper data run through the clusterless path, which must match `event_diagnostics`. |

## Rollout Strategy

- **Ordering:** phases 1 → 2 → 3 → 4a → 4b → 5. Each branches from `main` after the previous PR merges; the branch names are in [PLAN.md](PLAN.md).
- **Releases:**
  - Phases 1–3 change tooling, docs and error-message style only, and do not tag a release.
  - Phase 4b tags `v0.3.0`, only after the user explicitly approves the tag push, because it publishes to PyPI.
  - Phase 5 consumes the release.
- **Compatibility:** all public API is additive. Existing functions keep their signatures and outputs, so no deprecation window is needed.

## Open Questions

1. **Name of the clusterless entry point.** Current answer: `clusterless_event_diagnostics`, the term neuroscience users search for. Its docstring notes it applies to any mark space that can be sampled. The alternative is `continuous_mark_event_diagnostics`. Confirm at phase 4b review.
2. **Should `main.tex:350` mention clusterless support?** Current answer: yes, one sentence, drafted in phase 5 for the user to approve. The manuscript is the user's text, so don't commit it without sign-off.
3. **Zenodo DOI scope.** Current answer: enable Zenodo for `statespacecheck` (phase 4b) and cite the v0.3.0 concept DOI in `main.tex:375`. Whether `statespacecheck-paper` also gets its own DOI is the user's call; ask during phase 5.
4. **Should the paper's desktop viewer move into this package?** Current answer: not in this plan. `statespacecheck_paper.interactive` (about 4,200 lines) is tied to the paper: its cache builder imports the paper's Figure 3 simulation and Figure 4 decoder modules, its data source expects the paper's two models (`continuous`, `contfrag`) or the Figure 3 cache, and it needs PySide6, pyqtgraph, zarr, xarray and pandas. The display code (`viewer.py`, `panels.py`, about 2,600 lines) is fairly general. A move would need a documented array input (predictive `(n_time, n_bins)`, likelihood, position bins, event times and marks, intensities `(n_bins, n_marks)`, optional filtered/smoothed distributions), a builder from plain arrays, and an optional extra (`statespacecheck[viewer]`) so the core stays numpy/scipy/matplotlib. If the user wants it, plan it as its own phase after phase 5.

## Estimated Effort

| Phase | Size | What |
| --- | --- | --- |
| 1 | ~250 LOC | mostly config and the EM rewrites |
| 2 | ~350 LOC YAML | |
| 3 | ~100 LOC | |
| 4a | ~350 LOC | source ~150, tests ~200 |
| 4b | ~550 LOC | source ~150, tests ~200, tutorial ~200 (includes task 1's error and validation hardening) |
| 5 | ~150 LOC changed | plus regenerated artifacts |
