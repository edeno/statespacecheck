# Scientific Python Compliance and Paper Tools Implementation Plan

**Status:** Phase 1 implemented on branch `sp-packaging` (not yet merged); phases 2–5 not started.

Bring `statespacecheck` in line with the [Scientific Python development guide](https://learn.scientific-python.org/development/) using the same layout, tooling and CI as `ripple_detection` and `spectral_connectivity`. Then add the diagnostics the manuscript in `statespacecheck-paper` describes but the package does not yet provide:

- the event-weighted predictive state distribution
- a Monte Carlo predictive p-value for continuous or intractable mark spaces
- per-event diagnostics for clusterless (continuous-mark) decoders

Finally, move the paper onto the new release. The Scientific Python phases (1–3) run first. Every phase is its own feature branch and PR off `main`; phase 5 is a branch in the paper repo.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need the new public API signatures?** [shared-contracts.md](shared-contracts.md).
3. **Need the Monte Carlo / clusterless algorithms?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md): integration points, goals and non-goals, the repo-review baseline, risks, open questions.
- [shared-contracts.md](shared-contracts.md): signatures and semantics of the new public functions; phases 4a, 4b and 5 depend on them.
- [designs.md](designs.md): algorithms and code for event weighting, vectorized state sampling, the Monte Carlo p-value, and clusterless diagnostics.
- Phases (each ships as a separable PR on its own feature branch):
  - [phase-1-packaging.md](phase-1-packaging.md): `pyproject.toml`, strict pytest/mypy/ruff, dependency group plus lockfile, `py.typed`, `CITATION.cff`. Branch `sp-packaging`.
  - [phase-2-ci-precommit.md](phase-2-ci-precommit.md): pre-commit hook set; hardened CI/release workflow; Dependabot; zizmor; floors job. Branch `sp-ci`.
  - [phase-3-docs.md](phase-3-docs.md): one source for tutorials; notebooks executed in CI; docs workflow hardened; CONTRIBUTING/README refreshed; stale "paper" docstrings fixed. Branch `sp-docs`.
  - [phase-4a-monte-carlo-pvalue.md](phase-4a-monte-carlo-pvalue.md): `event_weighted_predictive`, `monte_carlo_mark_pvalue`. Branch `mc-mark-pvalue`.
  - [phase-4b-clusterless-diagnostics.md](phase-4b-clusterless-diagnostics.md): `event_diagnostics` error and validation hardening (output unchanged), `clusterless_event_diagnostics`, tutorial, Zenodo, v0.3.0 release. Branch `clusterless-diagnostics`.
  - [phase-5-paper-adoption.md](phase-5-paper-adoption.md): paper repo moves to statespacecheck 0.3.0; replaces its hand-rolled Monte Carlo and HPD mask; drops its `py.typed` workaround and range-checks `baseline_end_index`; regenerates figures, site fixture and DOI. Branch `statespacecheck-0.3` in `statespacecheck-paper`.
