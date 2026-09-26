# Phase 5: statespacecheck-paper adopts statespacecheck 0.3.0

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

**Repository:** `/Users/edeno/Documents/GitHub/statespacecheck-paper`.

**Branch:** `statespacecheck-0.3`, from the paper's `main`. One PR.

**Precondition (hard):**

- The paper's `statespacecheck-boundary` branch (unpushed, `df887f2..cbc7bfd` on 2026-09-25) and `shorten-fig-captions` branch are merged to `main`. Both change `manuscript/main.pdf`; after merging, rebuild it rather than resolving that conflict by hand.
- `git status` on the paper's `main` is clean.
- statespacecheck 0.3.0 is on PyPI (phase 4b).

If any precondition fails, stop and tell the user.

**Inputs to read first** (all paths relative to the paper repo):

- `CLAUDE.md`:
  - the package boundary (lines 29-34)
  - regeneration commands (lines 81-87): `emit_reported_values.py` → `make -C manuscript` → `export_site_data.py`
  - site parity (lines 67-70)
- `pyproject.toml:34`: `"statespacecheck>=0.2.0"`.
- `src/statespacecheck_paper/figure02_panels.py`:
  - `create_shared_example` (110-243): the hand-rolled Monte Carlo is lines 170-211, the showcase draws 214-228, which consume `rng` after the Monte Carlo
  - `compute_hpd_region` callers at 377 and 474-475
- `src/statespacecheck_paper/plotting.py`:
  - `compute_hpd_region` (77-116)
  - its callers at 244-245
  - the module doctest at lines 9-12
- `tests/test_figures.py:75-122`: Figure 2 data checks, which read `p_value` and `simulated_log_pred`.
- `tests/test_plotting.py:48-102`: `compute_hpd_region` tests (`TestComputeHpdRegion`).
- `src/statespacecheck_paper/figure04_cache.py:260-261`: the version in the diagnostics cache fingerprint.
- `src/statespacecheck_paper/diagnostics.py`:
  - `compute_baseline_diagnostic_thresholds` (542-628)
  - the `float()` around `ssc.baseline_threshold` (610-611), there only because 0.2.0 shipped no `py.typed`
- `src/statespacecheck_paper/reported_values.py`: `_software_versions`, which emits `\StatespacecheckVersion` and checks the recorded versions agree.
- `manuscript/main.tex`:
  - :350 is the package paragraph
  - :375 is the red DOI TODO
- [shared-contracts.md#monte_carlo_mark_pvalue](shared-contracts.md#monte_carlo_mark_pvalue).

**Contracts referenced:**

- [`monte_carlo_mark_pvalue`](shared-contracts.md#monte_carlo_mark_pvalue): the paper calls it with `return_samples=True`.

**Designs referenced:** none.

## Tasks

1. **Baseline capture**, before any code change, on the paper's `main`:
   - Run `uv run python scripts/generate_all_figures.py`. It must be a no-op, or regenerate identical files; confirm with `git status`.
   - Save copies:
     - `manuscript/reported_values.tex`
     - `manuscript/figures/main/figure0{3,4}_summary.json`
     - the Figure 2 `p_value`, from `create_shared_example(np.random.default_rng(42))`, the canonical seed at `figure02_generation.py:109`
   - Keep them all in a scratch directory, outside the repo.

2. **Bump the dependency.**
   - `pyproject.toml:34` → `"statespacecheck>=0.3.0"`.
   - `uv lock --upgrade-package statespacecheck`, then `uv sync --frozen --extra dev --extra interactive`.

3. **Replace the Figure 2 Monte Carlo** (`figure02_panels.py:170-211`) with one call:

   ```python
   check = ssc.monte_carlo_mark_pvalue(
       predictive[np.newaxis, :],
       lambda y: norm.pdf(position_bins[np.newaxis, :], loc=y[:, :1], scale=like_std),
       np.array([[like_mean]]),
       ground_intensity=np.ones(n_bins),
       sample_marks=lambda bins, g: g.normal(position_bins[bins], like_std)[:, np.newaxis],
       n_samples=n_mc_samples,
       rng=rng,
       return_samples=True,
   )
   p_value = float(check.pvalue[0])
   observed_log_pred = float(check.observed_log_density[0])
   simulated_log_pred_values = check.simulated_log_density[0]
   ```

   - **Keep** `cumsum` (`figure02_panels.py:179-180`); the showcase quantiles at :222-223 still use it. It must remain computed from `predictive`.
   - **Delete** the sampling loop and the vectorized density block, along with their comments about preserving the old RNG stream. Those comments no longer apply.
   - `Λ(x) = 1` is correct here: `N(y; x, σ)` integrates to 1 over `y`. Say so in a one-line comment, because the schematic's text (`main.tex:217`) relies on constant total intensity.
   - **Unit check:** the package's `f_pred` is `Σ_x λ(x,y) P(x) / Σ_x Λ P`. With `Λ=1` and `P` normalized, that equals the old `np.sum(predictive * observed_conditional_density)`. The new `observed_log_pred` must equal the old one to rtol 1e-12; assert this once while developing.

4. **Replace `compute_hpd_region` with `ssc.highest_density_region`.**
   - Before switching, compare the masks on the Figure 2 inputs (`predictive`, `likelihood`, coverage 0.95):
     - `compute_hpd_region(x, pdf)`
     - `ssc.highest_density_region(pdf[np.newaxis] / pdf.sum())[0]`
   - If they differ, use the package's anyway. It is the mask `ssc.hpd_overlap` uses for the overlap value printed in the figure, so the drawn bars and the number become consistent. Record the differing bins in the PR description.
   - Callers to update:
     - `figure02_panels.py:377`
     - `figure02_panels.py:474-475`
     - `plotting.py:244-245`
   - Delete `compute_hpd_region` (`plotting.py:77-116`), its module-doctest mention (`plotting.py:9-12`) and its tests (`tests/test_plotting.py:48-102`, `TestComputeHpdRegion`; keep the `gaussian_pdf` fixture if other tests still use it). The package tests `highest_density_region`.

5. **Drop the typing workaround and check `baseline_end_index`.** From the review of the paper's move to 0.2.0.
   - `diagnostics.py:610-611`: remove the `float()` and its comment. statespacecheck 0.3.0 ships `py.typed` (phase 1), so mypy now sees the package's real types. Run `uv run mypy src/` and fix whatever the real types expose, without `# type: ignore`.
   - `compute_baseline_diagnostic_thresholds`: raise `ValueError` unless `0 < baseline_end_index <= n_time` for the arrays passed in. Today an index past the end silently uses the whole recording, and a negative one silently drops the last rows. Add a test for each case. Current callers pass valid indices, so outputs don't change.

6. **Regenerate and compare** against the baseline from task 1:
   - `uv run python scripts/generate_all_figures.py`. The Figure 4 diagnostics recompute because the version in the cache fingerprint changed; the decode does not recompute.
   - `uv run python scripts/emit_reported_values.py`.
   - Diff `reported_values.tex`. The **only** allowed changes are `\StatespacecheckVersion{0.2.0}` → `{0.3.0}` and the source-hash comment line.
   - Diff the Figure 3/4 summary JSONs. They must be identical apart from provenance, because `event_diagnostics` is unchanged. The allowed provenance changes are `source.statespacecheck_version`, `source.source_tree_sha256` and `source.uv_lock_sha256` in both, plus `figure04_decode_cache.statespacecheck_version` and `figure04_decode_cache.diagnostics_fingerprint_sha256`.
   - Compare figures through the summaries, not PNG bytes. `figure04.png` can differ by one pixel (1/255) between a run that recomputes the diagnostics in-process and one that loads them from cache.
   - Figure 2's `p_value` must be within 4 binomial SE of the baseline (`n=1000`; SE = `sqrt(p(1-p)/1000)`). Report both values.
   - `make -C manuscript`. After a `git stash` or `checkout` restores files, it can skip the rebuild and leave a stale `main.pdf` and `main.log`. Before trusting either, force a rebuild with `cd manuscript && latexmk -g -pdf main.tex` and grep the PDF (`pdftotext`) for changed text.
   - `uv run python scripts/export_site_data.py`, then regenerate the JS parity fixture and run `make -C site test`, per `CLAUDE.md:67-70` (needs Node; it isn't installed on the author's machine as of 2026-09-25, so run it in CI or install Node first). The fixture must be unchanged because the sorted diagnostics are unchanged.

7. **Manuscript text.** This is the user's prose: draft it and **ask for approval before committing**.
   - `main.tex:375`: replace the red TODO with the Zenodo concept DOI minted in phase 4b. Whether the paper repo itself also needs a DOI is [overview Open Question 3](overview.md#open-questions); ask.
   - `main.tex:350`: propose one sentence after the spike-sorted description. Draft: "For clusterless decoders, the package evaluates the same three diagnostics from the joint mark intensity, computing the predictive $p$-value by the Monte Carlo construction above."

8. **Paper docs.**
   - `README.md:165-174`, the package boundary: add `monte_carlo_mark_pvalue` and `highest_density_region` to the package-owned computations. The paper `CLAUDE.md` bullet deliberately lists no function names, so leave it.
   - `docs/figure-pipeline.md`: if it describes Figure 2's Monte Carlo, point it at the package.

## Deliberately not in this phase

- Porting continuous-mark diagnostics to `site/js/metrics.js`.
- Changing the paper's thresholds, flag rules, decoder or Figure 3/4 content.
- Using `clusterless_event_diagnostics` in any figure. The paper's analyses are spike-sorted.
- Other local helpers the survey found: `compute_flag_confusion`, `_flag_percentage`, and `get_state_marginalized_posterior`. They are paper-specific by the stated boundary.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest` (paper) | Whole suite green, including `tests/test_figures.py` Figure 2 checks and `tests/test_import_boundaries.py` |
| `reported_values.tex` diff | Only `\StatespacecheckVersion` (and the source-hash comment) changes |
| Figure 3/4 summary JSON diff | Identical apart from the provenance fields listed in task 6 |
| `uv run mypy src/` | Clean with the package's real types and no `float()` workaround |
| `test_baseline_end_index_out_of_range_raises` (parametrized) | An index past the end, zero and negative indices each raise `ValueError` |
| Figure 2 p-value | Within 4 binomial SE of the baseline; both values in the PR description |
| `observed_log_pred` | Equal to the baseline to rtol 1e-12 |
| `make -C manuscript` | PDF builds; no red TODO remains at the Code availability section (after user approval of task 7) |
| `make -C site test` | JS parity passes; fixture unchanged |
| `grep -rn compute_hpd_region src tests` | No matches |
| Figure regeneration (**slow**, ~1 min for Figure 4 diagnostics) | Record wall-clock time in the PR description |

## Fixtures

- Baseline copies from task 1, kept outside the repo.
- Real data: the Figure 4 recording (DANDI 001942) through the existing cache, used by the real-data smoke test (task 6).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also run `scientific-code-change-audit` on the Figure 2 change, which alters a displayed statistic.
