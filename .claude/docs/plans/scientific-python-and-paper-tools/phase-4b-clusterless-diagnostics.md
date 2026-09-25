# Phase 4b: Clusterless per-event diagnostics and the v0.3.0 release

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md) · [designs](designs.md)

**Branch:** `clusterless-diagnostics`, from `main` after phase 4a merges. One PR; the release happens after it merges.

**Inputs to read first:**

- [src/statespacecheck/events.py:319-427](../../../../src/statespacecheck/events.py): `event_diagnostics`. Mirror its validation order, batching loop and `EventDiagnostics` construction.
- `src/statespacecheck/continuous_marks.py` (from phase 4a): `_monte_carlo_batch`, `_evaluate_intensity`, and the `rng` handling.
- `tests/conftest.py` (from phase 4a): the `discrete_mark_model` and `clusterless_1d_model` fixtures.
- `statespacecheck-paper/manuscript/main.tex:152` (clusterless Q) and `:195-201` (clusterless f_pred): the definitions.
- `examples/04_predictive_checks.py`: jupytext tutorial style (`py:percent` header; `utils.py` helpers).
- `.github/RELEASE_SETUP.md` and `CITATION.cff` (from phases 1 and 2): the release checklist.

**Contracts referenced (do not weaken):**

- [`clusterless_event_diagnostics`](shared-contracts.md#clusterless_event_diagnostics), including the discrete-mark invariant (bit-identical `hpd_overlap`, `kl_divergence` and `likelihood`)
- [Callable protocols](shared-contracts.md#callable-protocols)
- [Placement and exports](shared-contracts.md#placement-and-exports)

**Designs referenced:**

- [designs.md#clusterless-diagnostics](designs.md#clusterless-diagnostics)
- [#memory-and-batching](designs.md#memory-and-batching)
- [#clusterless-1-d-model](designs.md#clusterless-1-d-model)

## Tasks

1. **`clusterless_event_diagnostics`** in `continuous_marks.py`, as in [designs.md](designs.md#clusterless-diagnostics).
   - Reuse `event_likelihood`, `hpd_overlap`, `kl_divergence` and `_monte_carlo_batch`; don't reimplement them.
   - Docstring:
     - what it needs: the predictive distribution per bin, the joint mark intensity as a callable, the ground intensity, and a mark sampler
     - the clusterless Q and f_pred definitions
     - a note that for finite marks `event_diagnostics` is exact and preferred
     - reproducibility and memory notes (link to `monte_carlo_mark_pvalue`)
     - a doctest on a tiny discrete-encoded model
   - Export it from `__init__.py` and add it to `__all__`, sorted.
   - Confirm the name with the user at PR review ([overview Open Question 1](overview.md#open-questions)).

2. **Simulated clusterless fixture.** Extend `clusterless_1d_model` in `tests/conftest.py` with a session-scoped fixture that simulates:
   - a trajectory
   - spikes and marks
   - the exact grid-filter predictive, for both the true model and the misspecified model (waveform means shifted +0.8)

   Seed it with `np.random.default_rng(20260925)`. Keep it small, about 2,000 bins and 50 position bins, so it builds in under 2 s. Mark the tests that use it `slow` if the build takes longer.

3. **Tests** in `tests/test_continuous_marks.py`; see the validation slice.

4. **Tutorial** `examples/05_clusterless_diagnostics.py` (jupytext `py:percent`) plus the paired `.ipynb` with outputs. It covers:
   1. build the 1-D clusterless model
   2. simulate and decode, reusing the `utils.py` helpers where they fit; add new helpers to `examples/utils.py`, not inline
   3. run `clusterless_event_diagnostics` under the true and the misspecified model
   4. plot the p-value histograms and per-event HPD/KL over time
   5. show on the sorted special case that it agrees with `event_diagnostics`

   Add it to the `mkdocs.yml` nav. The phase-3 gen-files script copies it automatically. The CI notebook step (phase 3) executes it.

5. **User-facing docs.**
   - README "Continuous marks" subsection (added in 4a): a short `clusterless_event_diagnostics` example and a link to tutorial 05.
   - CHANGELOG: rename `[Unreleased]` to `## [0.3.0] - <release date>`, then add `clusterless_event_diagnostics()` under Added. Merge the phase 1–3 "Changed" lines under this version.
   - `CITATION.cff`: `version: 0.3.0` and `date-released`.
   - CLAUDE.md "Core Modules": `continuous_marks.py` entry, now including this function.

6. **Zenodo and release.** Needs user action; **do not push a tag without explicit approval.**
   1. **User:** enable the `edeno/statespacecheck` repository at <https://zenodo.org/account/settings/github/>. Zenodo archives each GitHub *release*, and this repo's `create-release` job makes one.
   2. After the PR merges, run `workflow_dispatch` on `main` and confirm everything up to test-package is green.
   3. **Ask the user** to approve `git tag v0.3.0 && git push origin v0.3.0`.
   4. Watch the publish and create-release jobs. Confirm that:
      - `pip install statespacecheck==0.3.0` works in a fresh `uv venv`
      - PyPI shows the attestations
      - the GitHub release notes are the CHANGELOG section
   5. Once Zenodo has minted the DOI, open a small follow-up PR, `zenodo-doi` from `main`. It:
      - adds `doi:` and `identifiers:` (concept DOI) to `CITATION.cff`
      - replaces the README bibtex placeholder DOI (`README.md` Citation section)
      - adds a Zenodo badge

## Deliberately not in this phase

- A JavaScript port of the continuous-mark diagnostics for the paper's website (overview Non-Goals).
- Real clusterless data. None is available ([overview Risks](overview.md#risks-and-mitigations)). The real-data slice here is the sorted special case of the paper's data, which is exercised in phase 5.
- Any change to the paper repo (phase 5).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_clusterless_matches_event_diagnostics_for_discrete_marks` | Discrete model encoded as callables: `hpd_overlap`, `kl_divergence` and `likelihood` are **`np.array_equal`** to `event_diagnostics`; `predictive_pvalue` within 4 SE (`n_samples=20_000`, **slow**) |
| `test_clusterless_pvalue_equals_monte_carlo_mark_pvalue` | Same seed and batch size: `predictive_pvalue` is bit-identical to `monte_carlo_mark_pvalue(predictive[time_ind], ...).pvalue` |
| `test_clusterless_calibrated_under_true_model` (**slow**) | Simulated session, true model: KS test of p-values against U(0,1) gives p > 0.01; median HPD overlap > 0.5 |
| `test_clusterless_detects_misspecified_marks` (**slow**) | Misspecified waveform means: fraction of p ≤ 0.05 exceeds 0.2, versus ≤ 0.08 under the true model; mean KL higher than the true model's |
| `test_clusterless_repeated_time_bins` | Several events in one bin each get their own diagnostics against the same predictive row |
| `test_clusterless_validation` | `ValueError` for: `event_time_ind` out of range, length mismatch with `event_marks`, bad coverage, `batch_size=0`, an observed-mark intensity that is zero everywhere (from `event_likelihood`) |
| `test_clusterless_return_likelihood` | `likelihood` has shape `(n_events, *spatial_shape)` for a 2-D spatial grid; `None` by default |
| `test_clusterless_empty_events` | Zero events gives empty arrays; callables not called |
| Tutorial 05 executes in CI | Notebook runs; its asserted agreement cell passes |
| `uv run mkdocs build --strict` | Tutorial 05 and the API page render |
| Release checks (task 6) | `pip install statespacecheck==0.3.0` in a clean env; `python -c "import statespacecheck as s; print(s.__version__, s.clusterless_event_diagnostics)"` |

Before choosing the thresholds (0.2, 0.08, 0.5), run the simulation once and record the observed values in the PR description. Set each threshold with margin from what is observed, **then** freeze it. Do not tune the simulation to pass a pre-set threshold.

## Fixtures

- `discrete_mark_model`, `clusterless_1d_model` (from 4a).
- New session-scoped `clusterless_session` (true and misspecified predictive, events, marks), synthesized in `tests/conftest.py`.
- Real data: none here (see above).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also run the `scientific-code-change-audit` skill on the diff before requesting the release tag.
