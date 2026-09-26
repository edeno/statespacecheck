"""Pytest configuration shared by the test suite and the docstring examples."""

import matplotlib as mpl

# Non-interactive backend so figures render headless. With warnings as errors,
# examples must close figures (plt.close) rather than call plt.show(), which
# warns under Agg.
mpl.use("Agg")
