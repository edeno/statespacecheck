"""Run the Python examples in README.md and the docs pages.

Each ```python block runs in a fresh namespace. When a ```text block follows it, that
block is the expected output. A block preceded by the line ``<!-- not-executed -->``
is skipped (for example, one that needs a decoder package).
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(
    r"(?P<marker><!-- not-executed -->\n)?```python\n(?P<code>.*?)```", re.DOTALL
)
OUTPUT = re.compile(r"\A\s*```text\n(?P<text>.*?)```", re.DOTALL)


def _examples() -> list[tuple[str, str, str | None]]:
    examples = []
    for path in DOCUMENTS:
        content = path.read_text()
        for number, match in enumerate(BLOCK.finditer(content), start=1):
            if match["marker"]:
                continue
            output = OUTPUT.match(content[match.end() :])
            examples.append(
                (f"{path.name}:{number}", match["code"], output["text"] if output else None)
            )
    return examples


@pytest.mark.parametrize(
    ("name", "code", "expected"), _examples(), ids=[example[0] for example in _examples()]
)
def test_example_runs(name, code, expected):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), {"__name__": "__main__"})
    if expected is not None:
        assert stdout.getvalue() == expected
