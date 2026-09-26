# Releasing statespacecheck

Pushing a `vX.Y.Z` tag runs `.github/workflows/ci.yml`, which:

1. Runs the quality checks (ruff, mypy, zizmor), the lockfile check, the
   test matrix (Python 3.10–3.14 on Linux, macOS and Windows), and the tests at
   the declared dependency floors
2. Builds the wheel and sdist and smoke-tests an install of each
3. **Publishes to PyPI** through trusted publishing, with build-provenance
   attestations
4. **Creates the GitHub release**, with the version's CHANGELOG section as its
   notes

## One-time setup

### PyPI trusted publishing

PyPI accepts uploads from this workflow without an API token. The publisher is
registered at <https://pypi.org/manage/project/statespacecheck/settings/publishing/>
with:

- **Owner**: `edeno`
- **Repository**: `statespacecheck`
- **Workflow name**: `ci.yml`
- **Environment name**: `pypi`

The workflow filename and the environment name must stay exactly these;
renaming either breaks publishing until the publisher is updated to match.

### Protected `pypi` environment

In the repository's Settings → Environments, give the `pypi` environment a
required reviewer, so each publish waits for a manual approval (Scientific
Python SPEC 8). The `testpypi` environment is no longer used and can be
deleted.

### Codecov

The test job uploads coverage from Python 3.12 on Linux using the
`CODECOV_TOKEN` repository secret; an upload failure does not fail CI.

### Zenodo

Not yet enabled. Once the repository is enabled at
<https://zenodo.org/account/settings/github/>, Zenodo will archive each GitHub
release and mint a DOI for it.

## How to release

1. On `main`, move the CHANGELOG's `[Unreleased]` entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading. The release notes are extracted from
   exactly this heading.
2. Set `version: X.Y.Z` and `date-released` in `CITATION.cff`.
3. Commit, push, and wait for CI on `main` to pass.
4. Tag and push:

   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. Approve the `pypi` deployment in the workflow run.
6. Check the release on [PyPI](https://pypi.org/project/statespacecheck/) and
   [GitHub](https://github.com/edeno/statespacecheck/releases), and in a fresh
   environment:

   ```bash
   pip install statespacecheck==X.Y.Z
   python -c "import statespacecheck; print(statespacecheck.__version__)"
   ```

The version comes from the git tag (hatch-vcs); never edit it in the code.

## Troubleshooting

- **"File already exists" on PyPI**: that version was already published. PyPI
  never accepts the same version twice; release the next patch version.
- **Publishing is rejected as untrusted**: the workflow filename, environment
  name or repository no longer match the trusted publisher above.
- **Empty release notes**: the CHANGELOG heading does not match
  `## [X.Y.Z]` for the tag `vX.Y.Z`.

## Withdrawing a release

PyPI releases cannot be deleted and re-uploaded. To withdraw one, *yank* it in
the project's release settings on PyPI: pip then skips it unless the exact
version is requested. Fix forward with a new patch release.
