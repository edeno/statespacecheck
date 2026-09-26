"""Generate the API reference: an overview page and one page per module."""

import inspect
import re
from pathlib import Path

import mkdocs_gen_files

import statespacecheck

# Every public module, in reading order: (module, page title, what it is for)
MODULES = [
    (
        "events",
        "Per-spike diagnostics",
        (
            "The paper's method: HPD overlap, KL divergence and the rank-based "
            "predictive p-value for every spike, thresholds from a baseline period, "
            "and flagging."
        ),
    ),
    (
        "continuous_marks",
        "Continuous marks",
        (
            "The predictive p-value by Monte Carlo, for marks that cannot be enumerated, "
            "such as the waveform features of clusterless decoding."
        ),
    ),
    (
        "state_consistency",
        "Comparing distributions",
        (
            "HPD overlap and KL divergence between a state distribution and a "
            "likelihood, for each row; used per spike by `event_diagnostics`, or per "
            "time bin with a whole-bin likelihood (an extension beyond the paper)."
        ),
    ),
    (
        "highest_density",
        "Highest-density regions",
        "The regions HPD overlap compares.",
    ),
    (
        "predictive_checks",
        "Predictive densities and Monte Carlo checks",
        (
            "An extension beyond the paper: predictive densities of whole time bins "
            "and a Monte Carlo predictive p-value with a user-supplied sampler."
        ),
    ),
    (
        "periods",
        "Flagging time series",
        (
            "An extension beyond the paper: flagging runs of time bins and aggregating "
            "over periods. For per-spike values use `flag_events`."
        ),
    ),
    (
        "viz",
        "Plotting",
        "An extension beyond the paper: time-series plots of the diagnostics.",
    ),
]

src = Path(__file__).parent.parent / "src" / "statespacecheck"
public_modules = {path.stem for path in src.glob("*.py") if not path.stem.startswith("_")}
missing = public_modules - {module for module, _, _ in MODULES}
if missing:
    msg = f"docs/gen_ref_pages.py does not list the public modules {sorted(missing)}"
    raise RuntimeError(msg)

nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
overview = [
    "# API reference\n\n",
    (
        "Everything below is importable from the top-level package, for example "
        "`statespacecheck.event_diagnostics`.\n\n"
    ),
]
for module, title, description in MODULES:
    ident = f"statespacecheck.{module}"
    page = f"{module}.md"
    nav[title] = page
    with mkdocs_gen_files.open(f"reference/{page}", "w") as fd:
        fd.write(f"# {title}\n\n{description}\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(f"reference/{page}", f"src/statespacecheck/{module}.py")

    names = [
        name
        for name in statespacecheck.__all__
        if getattr(getattr(statespacecheck, name), "__module__", None) == ident
    ]
    overview.append(f"## [{title}]({page})\n\n{description}\n\n")
    for name in names:
        doc = inspect.getdoc(getattr(statespacecheck, name)) or ""
        # Docstrings use reST roles (:func:`name`); show them as code here
        summary = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", doc.split("\n", 1)[0])
        overview.append(f"- [`{name}`]({page}#{ident}.{name}): {summary}\n")
    overview.append("\n")

overview.append(
    "## Types and constants\n\n"
    "- `DistributionArray`: the type of the float64 arrays the functions return, "
    "`numpy.typing.NDArray[numpy.float64]`. Inputs may be any array-like.\n"
    "- `DEFAULT_COVERAGE`: the default coverage of HPD regions, 0.95, as in the paper.\n"
    "- [`LogMarkIntensity`](continuous_marks.md#statespacecheck.continuous_marks.LogMarkIntensity) "
    "and [`MarkSampler`](continuous_marks.md#statespacecheck.continuous_marks.MarkSampler): "
    "the types of the model functions `monte_carlo_mark_pvalue` takes.\n"
)

with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.writelines(overview)
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
