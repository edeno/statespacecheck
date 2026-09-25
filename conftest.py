"""Pytest configuration shared by the test suite and the docstring examples."""

import matplotlib as mpl

# Non-interactive backend: runs headless, and plt.show() in an example cannot block
mpl.use("Agg")
