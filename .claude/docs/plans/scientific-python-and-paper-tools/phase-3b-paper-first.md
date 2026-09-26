# Phase 3b: Paper-first user experience

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Branch:** `paper-first-ux`, from `main` after phase 3. One PR, merged before phase 4a.

**Why.** Four audits of `main` @ `9509a00` (API usability, onboarding, paper alignment, Scientific Python practice) found the following:

- **The per-event core matches the paper exactly.** `event_likelihood`, `hpd_overlap`, `kl_divergence`, `mark_predictive_pvalue` and `baseline_threshold` were verified numerically against `main.tex`.
- **Everything a new user sees steers them away from that method:**
  - The README, the docs home and the tutorials teach an older time-bin workflow, framed as "posterior" analysis.
  - Its fixed cutoffs contradict the paper.
  - Its flagging helpers give wrong answers on the paper's outputs.
  - The paper's own per-event workflow has no tutorial.
- **No deprecation cycle** (user decision, 2026-09-26: there are no users yet). Breaking changes go in the CHANGELOG under `[Unreleased]`, to be released as 0.3.0 in phase 4b.

**Decisions (made 2026-09-26):**

- **Keep the dependency floors** at Python ≥ 3.10, NumPy ≥ 1.26, SciPy ≥ 1.11.1, matplotlib ≥ 3.8. The paper repo runs Python 3.11 (`statespacecheck-paper/.python-version`), so SPEC 0's floors (Python ≥ 3.12 as of Q3 2026) would lock it out. Reword the "follows SPEC 0" claim instead.
- **Monte Carlo p-values stay a plain fraction**, with no +1 correction. The paper defines p as the fraction of replicates at or below the observed value (`main.tex:215`). Document that a finite-sample estimate can be exactly 0.
- **Paper terminology is canonical:**
  - one-step predictive distribution
  - single-event likelihood
  - HPD overlap (highest probability-density region)
  - KL divergence D(P‖Q)
  - rank-based predictive p-value
  - event-weighted predictive distribution

  "Posterior" is used only where a smoother or filter output is meant.

## Tasks

Each numbered item is one or more commits. Behaviour changes get a test that fails first and a CHANGELOG line.

### A. Correctness against the paper (breaking; no deprecation)

1. **`flag_extreme_pvalues` becomes one-sided:** it flags `p <= alpha`, and `alpha` is the cutoff itself (default 0.05). Today it flags `p < alpha/2 or p > 1 - alpha/2`, which marks the best-fitting events (p ≈ 1).
   - Update its docstring.
   - Update `plot_diagnostics` to draw one line at `alpha`.
   - Update the `predictive_pvalue` docstring: small p means misfit, and p near 1 does not.
   - Rewrite tutorial 04's text to match.
2. **`flag_low_overlap` and `find_low_overlap_intervals` use `<=`**, matching the paper's "at or below". Today they use `<`, so a `baseline_threshold` of exactly 0 flags nothing.
3. **`flag_extreme_kl` always flags `+inf`.** Its docstring states that the flags are relative to the recording (a robust z-score). It points to `baseline_threshold` plus `flag_events` for the paper's rule.
4. **`baseline_threshold` accepts `+inf`**, so KL is infinite when the supports are disjoint. It returns the quantile, which may itself be `inf`. It still raises on `-inf`, or when no finite values remain.
5. **New `flag_events(diagnostics, *, hpd_overlap_threshold=None, kl_divergence_threshold=None, pvalue_threshold=0.05) -> EventFlags`**, in `events.py`. It implements the paper's rule: HPD at or below, KL at or above, and p at or below. A threshold of `None` skips that metric, and the corresponding field is `None`.
   - `EventFlags` is a NamedTuple of the three optional boolean arrays.
   - Export both from `__init__`.
   - The docstring shows the paper's simulation rule (1st/99th baseline percentiles via `baseline_threshold`, p ≤ 0.05) and its real-data rule (HPD ≤ 0.05, p ≤ 0.05).
6. **The `kl_divergence` NaN handling combines both arrays' NaN masks** before normalizing, as its docstring already claims. Today a NaN only in the likelihood gives `inf`.
7. **Fix docstrings that contradict the code:**
   - `event_diagnostics`: events that share a bin get identical diagnostics only when they come from the same mark.
   - `mark_predictive_pvalue`: the tie tolerance is absolute, scaled by the largest probability in the batch.
   - `hpd_overlap` and `highest_density_region`: the region "size" is a bin count, which equals the paper's volume only on a uniform grid.

### B. Errors a user can act on

8. **`event_diagnostics` validates everything up front**, using the user's argument names:
   - `predictive`: finite, nonnegative, and every referenced row has positive mass.
   - `mark_intensities`: finite and nonnegative, and every mark has positive intensity somewhere.
   - Indices and lengths.

   Errors report **absolute** event, time-bin and mark indices; today they are batch-relative. A shape mismatch where the transposed shape would fit says so ("did you mean `mark_intensities.T`?").
9. **Other error messages:**
   - Index arrays that are floats get a hint to convert times to bin indices (e.g. `np.searchsorted` / `np.digitize`).
   - An empty list of events is accepted.
   - `aggregate_over_period` requires a boolean `time_mask`; today an index array is silently cast.
   - `periods` errors name the user's argument.

### C. Naming consistency (breaking renames, no aliases)

10. **Unify names:**
    - `predictive_density` and `log_predictive_density`: `likelihood` → `observation_likelihood` (unnormalized p(y|x)). This distinguishes it from the normalized `likelihood` of `kl_divergence` and `hpd_overlap`.
    - `plot_diagnostics(tau, z_thresh, alpha)` → `overlap_threshold`, `kl_z_threshold`, `pvalue_threshold`, matching the flag functions.
    - Flag and plot threshold arguments become keyword-only.
    - `baseline_threshold(values, quantile)` keeps its signature.
    - `log_predictive_density`'s `log_likelihood` becomes keyword-only.

### D. Performance, import time and typing

11. **Import time:**
    - `import matplotlib.pyplot` inside `plot_diagnostics`.
    - Replace `scipy.stats.entropy` with `scipy.special.rel_entr(...).sum(axis=-1)` and `median_abs_deviation` with a NumPy MAD.
    - Before and after, record the import time and check that the outputs are identical (or within `rtol=1e-12`) on the property-test fixtures.
12. **Memory in the time-bin functions** (`kl_divergence`, `hpd_overlap`, `highest_density_region`, `log_predictive_density`):
    - Before changing anything, record the time and peak memory at `n_time=100_000`, `n_bins=500`. Baseline measured by the audit: kl 1.67 s / 3.9 GB, hpd 2.39 s / 2.3 GB, hdr 1.04 s / 1.5 GB, log-pred 1.56 s / 4.7 GB.
    - Remove the redundant copies (the double `nan_to_num`, copies from boolean indexing, re-normalizing inside `entropy`).
    - Process time in chunks where needed.
    - Assert that the outputs are unchanged and report the measured deltas.
13. **Types for users:** inputs are `npt.ArrayLike`, outputs are `NDArray[np.float64]`, and `reduction` is `Literal["mean", "sum"]`. `DistributionArray` stays exported, documented as the output type. Check that a consumer snippet type-checks under `mypy --strict`, including list inputs and float64 outputs.

### E. Documentation: the paper's workflow first

14. **README rewrite:**
    - A one-line pitch.
    - The paper (title, authors, repo) and the project website <https://edeno.github.io/statespacecheck-paper/>.
    - `pip install statespacecheck`.
    - A runnable per-event quick start on simulated place fields, with a misfit window: `event_diagnostics` → `baseline_threshold` → `flag_events`, printing the flagged fraction in the baseline and in the misfit window.
    - "What your decoder must provide": shapes, units, and pitfalls (NaN bins, `(n_units, n_bins)` needing `.T`, marginalizing over discrete states).
    - "Reading the results": consistency is not similarity, baseline thresholds, p ≤ 0.05, KL as reference only.
    - "Paper ↔ functions" table.
    - Links to the docs.

    Remove: the fixed-cutoff tables, the hand-written three-function API reference, and the Development section (CONTRIBUTING has it). Keep the shared citation section.
15. **`docs/index.md`:** the same structure (it shares the quick-start and citation sections through snippets where possible). Drop the duplicated "posterior" framing.
16. **New tutorial `examples/05_per_event_diagnostics.py`, with its `.ipynb` pair.** It reproduces the paper's simulation workflow at small scale:
    - a place-field decoder with a grid filter
    - a baseline, then a misfit
    - `event_diagnostics`, `baseline_threshold` (1st/99th percentiles), `flag_events`
    - −log p display
    - comparing two models' flags (the paper's "rescued" spikes)

    Add it first in the tutorials nav and index. Phase 4b's clusterless tutorial becomes 06.
17. **New docs pages:**
    - `docs/interpretation.md`: what each measure means, which direction is bad, consistency vs similarity, and threshold rules.
    - `docs/decoders.md`: using the package with your decoder. It covers a generic grid decoder, `non_local_detector`, and a Gaussian or Kalman predictive discretized onto a grid, with the NaN-interior, transpose and marginalization recipes. It is based on `statespacecheck-paper/src/statespacecheck_paper/figure04_place_fields.py` and `diagnostics.py`.

    Add both to the nav.
18. **Extensions labelled:** the time-bin functions, `periods.py`, `aggregate_over_period`, the generic Monte Carlo `predictive_pvalue`, `predictive_density` and `plot_diagnostics` state in their module/function docstrings and in the docs that they are extensions beyond the paper. They also say how they differ: a whole-bin likelihood includes the exposure term, and time series use `min_len` runs.
19. **Existing tutorials:**
    - 01 and 02: use paper terminology and remove the fixed-cutoff guidance.
    - 03: fix the "KL > 1.0" mislabel and the "lower threshold is stricter" statement.
    - 04: retitle it (one-step predictive check), fix ≥ → ≤, make it one-sided, and fix the broken `flag_extreme_kl(threshold=)` snippet.
    - All: replace the notebook cross-links with site-correct links, and remove the `../README.md` link.
    - `docs/tutorials/index.md`: working run instructions (`uv sync --extra docs` plus a Jupyter front end, or VS Code), and no "Colab badge (if available)".
20. **API reference:**
    - A `docs/reference/index.md` overview grouped by task: per-event diagnostics (the paper), distribution comparison, predictive checks (extension), time-series flagging (extension), plotting.
    - Nav labels in words.
    - `docs/index.md` links to the overview.
    - Remove the stray `reference/SUMMARY` page from search.
    - Document `DistributionArray` and `DEFAULT_COVERAGE`.
21. **Terminology sweep** across README, docs, docstrings and tutorials, using the canonical terms above.

### F. Citation, metadata and project hygiene

22. **`CITATION.cff`:**
    - ORCIDs from `main.tex`: Zeng 0009-0001-4056-552X, Eden 0000-0002-2058-3691, Denovellis 0000-0003-4606-087X.
    - `url` → the repository.
    - A `references` entry for the companion paper (type article, title, the five authors, no DOI yet).
    - Keywords as in `pyproject`.
23. **`pyproject`:**
    - Description mentions per-event diagnostics.
    - Keywords: add neural-decoding, point-process, spike-trains, model-checking, predictive-checks; drop kalman-filter.
    - `project.urls`: add "Paper code" and "Paper website".
24. **Claims:**
    - README and CLAUDE.md: reword "follows SPEC 0" (the floors are wider than SPEC 0, to support the paper's Python 3.11).
    - CLAUDE.md: fix "100 character" and the module list.
    - `RELEASE_SETUP.md`: Zenodo in the future tense until it is enabled.
25. **Community files:**
    - `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1, as in spectral_connectivity)
    - `SECURITY.md` (report privately through GitHub security advisories)
    - `.github/ISSUE_TEMPLATE/` (bug report asking for `statespacecheck.__version__`, numpy/scipy versions and a minimal example; feature request)
    - `.github/pull_request_template.md`
26. **CI:**
    - A weekly scheduled job (SPEC 4) installing nightly numpy, scipy and matplotlib from the scientific-python-nightly-wheels index, running the tests.
    - The floors job constrains `pyparsing<3.3`, so warnings stay errors instead of `-p no:warnings`.
    - Coverage `fail_under = 95`.
    - Dependabot for `uv` (the lockfile), monthly.
27. **Repository settings** (`gh api`, done after the PR merges and reported):
    - Required reviewer `edeno` on the `pypi` environment.
    - Delete the `testpypi` environment.
    - Enable secret scanning and push protection.
    - `main` protection that forbids force-pushes and deletion only (no required reviews, so the local-merge workflow still works).

## Deliberately not in this phase

- New statistical API beyond `flag_events`/`EventFlags`: `event_weighted_predictive`, `monte_carlo_mark_pvalue` and the clusterless path stay in 4a/4b.
- conda-forge feedstock, which is an external submission. Revisit after 0.3.0.
- A dedicated per-event plot function. The tutorial shows the plotting inline; add `plot_event_diagnostics` if users ask.
- Changing any per-event numerical result. `event_diagnostics` output must stay bit-identical; test against a fixture recorded before the change.

## Validation

| Check | Asserts |
| --- | --- |
| New tests for 1–6, 8–10 | Each fails before its fix and passes after |
| `event_diagnostics` bit-identity | Outputs on a recorded fixture (random model, 2-D grid) equal those from `main` exactly |
| Performance before/after table (task 12) | Outputs equal (≤ 1e-12); time and peak memory reported in the PR |
| Import time before/after | Reported in the PR |
| Consumer typing snippet | `mypy --strict` passes |
| README and docs code blocks | Every Python block runs (a test extracts and executes them) |
| Tutorial 05 | Executes in the docs workflow; pair check passes |
| `mkdocs build --strict` | Passes; no 404 tutorial links (checked by grepping the built HTML for `.ipynb` hrefs) |
| pytest, mypy, ruff, pre-commit, zizmor, repo-review | Clean; repo-review still fails only PY007, PC140, PC170 and PC180 |
