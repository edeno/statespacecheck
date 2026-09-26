"""Check the tutorials in examples/: each jupytext pair agrees, and outputs are current.

Usage::

    python docs/check_tutorials.py [EXECUTED_DIR]

Every tutorial's ``.py`` script and ``.ipynb`` notebook must have the same cells. With
``EXECUTED_DIR``, a directory of freshly executed copies of the notebooks, each
notebook's committed text outputs (printed text and text results; figures are not
compared) must match the fresh run, and the fresh run must not write to stderr, as a
warning does. The site shows the committed outputs, so this keeps them current.
It exits with the failures, or silently if there are none.
"""

import sys
from pathlib import Path

import jupytext
import nbformat

EXAMPLES = Path(__file__).parent.parent / "examples"


def check_pairs() -> list[str]:
    """Return a failure message for each tutorial whose script and notebook differ."""
    scripts = {path.stem for path in EXAMPLES.glob("[0-9][0-9]_*.py")}
    notebooks = {path.stem for path in EXAMPLES.glob("[0-9][0-9]_*.ipynb")}
    if not scripts or scripts != notebooks:
        return [f"unpaired tutorials: {sorted(scripts ^ notebooks)}"]
    failures = []
    for stem in sorted(scripts):
        script, notebook = EXAMPLES / f"{stem}.py", EXAMPLES / f"{stem}.ipynb"
        cells = [(cell.cell_type, cell.source) for cell in jupytext.read(script).cells]
        notebook_cells = [
            (cell.cell_type, cell.source) for cell in jupytext.read(notebook).cells
        ]
        if cells != notebook_cells:
            failures.append(
                f"{script.name} and {notebook.name} differ; run: "
                f"uv run --extra docs jupytext --sync examples/{script.name}"
            )
    return failures


def _text_outputs(path: Path) -> list[list[tuple[str, str]]]:
    """Each code cell's printed text and text results, as (stream name or "result", text)."""
    cells = []
    for cell in nbformat.read(path, as_version=4).cells:
        if cell.cell_type != "code":
            continue
        texts = []
        for output in cell.outputs:
            if output.output_type == "stream":
                texts.append((output.name, output.text))
            elif "text/plain" in output.get("data", {}) and not any(
                key.startswith("image/") for key in output.data
            ):
                texts.append(("result", output.data["text/plain"]))
        cells.append(texts)
    return cells


def check_outputs(executed_dir: Path) -> list[str]:
    """Return a failure message for each stale committed output or warning."""
    failures = []
    for committed in sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb")):
        executed = _text_outputs(executed_dir / committed.name)
        committed_outputs = _text_outputs(committed)
        if len(committed_outputs) != len(executed):
            failures.append(f"{committed.name}: the executed copy has other cells")
            continue
        for index, outputs in enumerate(executed):
            failures += [
                f"{committed.name}, code cell {index}, warns: {text[:300]}"
                for name, text in outputs
                if name == "stderr"
            ]
        for index, (old, new) in enumerate(zip(committed_outputs, executed, strict=True)):
            if old != new:
                failures.append(
                    f"{committed.name}, code cell {index}: the committed output differs "
                    f"from a fresh run; re-execute the notebook\n{old}\n{new}"
                )
    return failures


if __name__ == "__main__":
    failures = check_pairs()
    if len(sys.argv) > 1:
        failures += check_outputs(Path(sys.argv[1]))
    if failures:
        sys.exit("\n".join(failures))
