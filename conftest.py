"""Pytest configuration shared by the test suite and the docstring examples."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

# Non-interactive backend so figures render headless. With warnings as errors,
# examples must close figures (plt.close) rather than call plt.show(), which
# warns under Agg.
mpl.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures after every test and example, pass or fail.

    With warnings as errors, figures left open by a failing test would trip
    matplotlib's too-many-open-figures warning in unrelated tests.
    """
    yield
    plt.close("all")
